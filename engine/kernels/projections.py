"""Pack independent BF16 projections once, sharing storage with prefill."""

import torch


@torch.no_grad()
def pack_projections(model):
    for layer in model.model.layers:
        attn = layer.self_attn
        mlp = layer.mlp
        for owner, name, modules in (
            (attn, "qkv_weight", (attn.q_proj, attn.k_proj, attn.v_proj)),
            (mlp, "gate_up_weight", (mlp.gate_proj, mlp.up_proj)),
        ):
            packed = torch.cat([module.weight for module in modules], dim=0)
            owner.register_buffer(name, packed)
            offset = 0
            for module in modules:
                width = module.weight.shape[0]
                # Native prefill continues to call the original Linear modules.
                # Row slices remain contiguous and share the packed allocation.
                module.weight = torch.nn.Parameter(
                    packed[offset:offset + width], requires_grad=False,
                )
                offset += width
