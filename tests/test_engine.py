"""CPU dispatch checks and CUDA-only fused-kernel checks on a tiny Qwen3."""

import copy
from pathlib import Path
import sys
import unittest

import torch
from transformers import DynamicCache, Qwen3Config, Qwen3ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))
from engine import Engine, install_fused_norms, qwen_forward


def tiny_model(device="cpu", dtype=torch.float32):
    torch.manual_seed(42)
    config = Qwen3Config(
        vocab_size=127,
        hidden_size=48,
        intermediate_size=96,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=256,
        rope_theta=5_000_000.0,
        tie_word_embeddings=True,
        attention_dropout=0.0,
        sliding_window=None,
    )
    config._attn_implementation = "sdpa"
    return Qwen3ForCausalLM(config).eval().to(device=device, dtype=dtype)


@torch.inference_mode()
def native_generate(model, prompt, steps):
    current = torch.tensor(prompt, device=model.device)
    cache = DynamicCache()
    result = []
    for _ in range(steps):
        output = model(current, past_key_values=cache, use_cache=True, logits_to_keep=1)
        current = output.logits[:, -1].argmax(-1, keepdim=True)
        result.append(current[:, 0].tolist())
    return result


class ForwardTests(unittest.TestCase):
    def setUp(self):
        self.model = tiny_model()

    @torch.inference_mode()
    def test_prefill_and_cached_decode_match_native_logits(self):
        for batch, prompt_length in ((1, 1), (1, 7), (4, 13)):
            with self.subTest(batch=batch, prompt_length=prompt_length):
                ids = torch.randint(0, 127, (batch, prompt_length))
                candidate_cache, native_cache = DynamicCache(), DynamicCache()
                position = 0
                for _ in range(5):
                    expected = self.model(
                        ids, past_key_values=native_cache, use_cache=True,
                        logits_to_keep=1,
                    ).logits
                    actual = qwen_forward(self.model, ids, candidate_cache, position)
                    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                    position += ids.shape[1]
                    self.assertEqual(candidate_cache.get_seq_length(), position)
                    ids = expected[:, -1].argmax(-1, keepdim=True)

    def test_generation_resets_cache_and_preserves_batch_order(self):
        engine = Engine.__new__(Engine)
        engine.model = self.model
        for prompt in ([[1, 2, 3], [7, 8, 9]], [[11, 12], [23, 24]], [[5]]):
            expected = native_generate(self.model, prompt, 6)
            # Even when the first predicted token is configured as EOS, all
            # six tokens must be emitted under the benchmark contract.
            self.model.config.eos_token_id = expected[0][0]
            self.assertEqual(list(engine.generate(prompt, 6)), expected)
        self.assertEqual(list(engine.generate([[1]], 0)), [])


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for Triton")
class FusedNormTests(unittest.TestCase):
    @torch.inference_mode()
    def test_norm_widths_and_noncontiguous_input(self):
        from kernels.rmsnorm import rms_norm
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

        for width in (128, 2560):
            norm = Qwen3RMSNorm(width).cuda().to(torch.bfloat16)
            norm.weight.copy_(torch.randn_like(norm.weight))
            for x in (
                torch.randn(2, 3, width, device="cuda", dtype=torch.bfloat16),
                torch.randn(2, 3, width * 2, device="cuda", dtype=torch.bfloat16)[..., ::2],
                torch.zeros(2, 3, width, device="cuda", dtype=torch.bfloat16),
            ):
                with self.subTest(width=width, strides=x.stride()):
                    torch.testing.assert_close(
                        rms_norm(x, norm.weight, norm.variance_epsilon), norm(x),
                        rtol=0.02, atol=0.02,
                    )

    @torch.inference_mode()
    def test_fused_model_teacher_forced_on_own_prefix(self):
        native = tiny_model("cuda", torch.bfloat16)
        candidate = copy.deepcopy(native)
        install_fused_norms(candidate)
        engine = Engine.__new__(Engine)
        engine.model = candidate
        for _ in range(2):
            prompt = torch.randint(0, 127, (4, 13), device="cuda")
            tokens = torch.tensor(list(engine.generate(prompt.tolist(), 8)), device="cuda").T
            replay_ids = torch.cat((prompt, tokens[:, :-1]), dim=1)
            logits = native(replay_ids, use_cache=False, logits_to_keep=8).logits.float()
            selected = logits.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
            gap = logits.max(-1).values - selected
            # A tiny random model's logits are small, so use a much tighter
            # bound than the full-checkpoint judge's 2.0-logit allowance.
            self.assertLessEqual(gap.max().item(), 0.03)


if __name__ == "__main__":
    unittest.main()
