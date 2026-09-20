"""Host side of exact token-tree speculative decoding (no draft model, no shipped weights).

Per sequence, candidates for the continuation of the current context come from
  1. a Jacobi (lookahead) window: the model's own predictions at the previous tree's chain nodes,
     shifted past what was accepted - a fixed-point iteration that converges on the real text,
  2. n-gram occurrences in prompt + generated text (n = 3, then 2; several occurrences each),
  3. a pool of the model's predictions at every verified node, keyed by (parent, node) tokens
     (the "free" n-grams lookahead decoding collects).
They are merged into a trie of at most T nodes (node 0 = the last emitted token, whose K/V is not
cached yet), parents before children, so node index >= depth. The device verifies the whole tree
in one forward; `accept` walks it with the model's argmax at each node: every emitted token is the
model's greedy choice on the emitted prefix, i.e. exact.
"""

MAX_OCC = 3


class Tree:
    __slots__ = ("tokens", "depth", "anc", "children", "chain", "n_real")

    def __init__(self, root, T):
        self.tokens = [root]
        self.depth = [0]
        self.anc = [1]
        self.children = [{}]
        self.chain = []
        self.n_real = 1

    def add_path(self, toks, T, max_depth):
        """Insert a candidate continuation; returns the node indices along it."""
        node, nodes = 0, []
        for tok in toks[:max_depth]:
            nxt = self.children[node].get(tok)
            if nxt is None:
                if len(self.tokens) >= T:
                    break
                nxt = len(self.tokens)
                self.tokens.append(tok)
                self.depth.append(self.depth[node] + 1)
                self.anc.append(self.anc[node] | (1 << nxt))
                self.children.append({})
                self.children[node][tok] = nxt
            nodes.append(nxt)
            node = nxt
        return nodes

    def pad(self, T):
        """Fill to T nodes with inert leaves (children of the root, not reachable by `accept`)."""
        root = self.tokens[0]
        while len(self.tokens) < T:
            i = len(self.tokens)
            self.tokens.append(root)
            self.depth.append(1)
            self.anc.append(1 | (1 << i))
            self.children.append({})


class SeqDraft:
    """Candidate sources for one sequence. `seq` is prompt + emitted tokens (shared, appended by caller)."""

    def __init__(self, seq):
        self.seq = seq
        self.idx3, self.idx2 = {}, {}
        self.pool = {}
        self.reg = 1
        self.window = []
        self.register()

    def register(self):
        seq, i3, i2 = self.seq, self.idx3, self.idx2
        for j in range(self.reg, len(seq) - 1):       # only n-grams whose continuation exists
            k2 = (seq[j - 1], seq[j])
            l2 = i2.get(k2)
            if l2 is None:
                i2[k2] = [j + 1]
            else:
                l2.append(j + 1)
                if len(l2) > MAX_OCC:
                    del l2[0]
            if j >= 2:
                k3 = (seq[j - 2], seq[j - 1], seq[j])
                l3 = i3.get(k3)
                if l3 is None:
                    i3[k3] = [j + 1]
                else:
                    l3.append(j + 1)
                    if len(l3) > MAX_OCC:
                        del l3[0]
        self.reg = max(self.reg, len(seq) - 1)

    def candidates(self, depth):
        seq = self.seq
        out = []
        if self.window:
            out.append(self.window[:depth])
        for p in reversed(self.idx3.get((seq[-3], seq[-2], seq[-1]), ())):
            out.append(seq[p:p + depth])
        for p in reversed(self.idx2.get((seq[-2], seq[-1]), ())):
            out.append(seq[p:p + depth])
        chain, a, b = [], seq[-2], seq[-1]
        for _ in range(depth):
            c = self.pool.get((a, b))
            if c is None:
                break
            chain.append(c)
            a, b = b, c
        if chain:
            out.append(chain)
        return out

    def build(self, T, max_depth):
        tree = Tree(self.seq[-1], T)
        cands = self.candidates(max_depth) or [[self.seq[-1]] * max_depth]   # Jacobi seed
        for n, c in enumerate(cands):
            nodes = tree.add_path(c, T, max_depth)
            if n == 0:
                tree.chain = nodes                      # the Jacobi chain (for the window update)
            if len(tree.tokens) >= T:
                break
        tree.n_real = len(tree.tokens)
        tree.pad(T)
        return tree

    def update(self, tree, g):
        """After verification: record the model's predictions and advance the Jacobi window.
        g = model argmax per node. Returns (emitted tokens, accepted node path)."""
        n_real = tree.n_real
        emitted, path, node = [g[0]], [], 0
        while True:
            nxt = tree.children[node].get(emitted[-1])
            if nxt is None:
                break
            path.append(nxt)
            node = nxt
            emitted.append(g[nxt])
        # model predictions at every real node are exact next tokens for hypothetical contexts
        prev = self.seq[-2]
        tok, pool = tree.tokens, self.pool
        parent_tok = [prev] + [0] * (n_real - 1)
        for p in range(n_real):
            for c_tok, c in tree.children[p].items():
                if c < n_real:
                    parent_tok[c] = tok[p]
        for i in range(n_real):
            pool[(parent_tok[i], tok[i])] = g[i]
        # Jacobi: chain node k predicts the token after depth k; keep the part beyond acceptance
        a = len(path)
        chain = tree.chain
        self.window = [g[c] for c in chain[a:]] if len(chain) > a and chain[:a] == path else []
        return emitted, path
