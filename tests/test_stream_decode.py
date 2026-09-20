"""Check streaming ownership and replay count without a local CUDA runtime."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


source = Path(__file__).resolve().parents[1] / "engine" / "engine.py"
tree = ast.parse(source.read_text())
function = next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "stream_decode")
namespace = {}
exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
stream_decode = namespace["stream_decode"]


class Tokens:
    def __init__(self, values):
        self.values = values[:]

    def __getitem__(self, index):
        return Tokens(self.values[index])

    def tolist(self):
        return [row[:] for row in self.values]


class StreamDecodeTest(unittest.TestCase):
    def test_host_snapshots_and_exact_replay_count(self):
        for batch in (1, 4, 16):
            for length in (2, 8, 9, 10, 32, 128):
                first = list(range(batch))
                output = Tokens([])
                state = first[:]
                replays = []
                graphs = {}
                for count in range(1, 9):
                    def replay(count=count):
                        replays.append(count)
                        output.values = []
                        for _ in range(count):
                            state[:] = [value + 7 for value in state]
                            output.values.append(state[:])
                    graphs[count] = SimpleNamespace(replay=replay)
                decoder = SimpleNamespace(output=output, graphs=graphs, chunk_size=8)
                stream = stream_decode(decoder, first, length)
                self.assertEqual(next(stream), first)
                self.assertEqual(len(replays), 1)  # Next GPU step is already queued.
                self.assertEqual(replays[0], min(8, length - 1))
                result = [first] + list(stream)
                self.assertEqual(result, [[i + 7 * step for i in range(batch)]
                                          for step in range(length)])
                self.assertEqual(sum(replays), length - 1)
                self.assertEqual(first, list(range(batch)))


if __name__ == "__main__":
    unittest.main()
