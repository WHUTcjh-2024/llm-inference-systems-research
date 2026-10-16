"""Cheap checkpoint-free gate for the incremental decode path E1 measures."""

import torch
from dash_attn.models.llama import LlamaConfig, LlamaForCausalLM


def main() -> None:
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        chunk_size=16,
        estimate_diagonal=False,
        sigma=1e6,
        scaling_factor=1.0,
    )
    model = LlamaForCausalLM(config).to("cuda", dtype=torch.bfloat16).eval()
    attention = model.model.layers[0].self_attn.dash_attn
    original_decode = attention.decode
    decode_calls = 0

    def counted_decode(*args, **kwargs):
        nonlocal decode_calls
        decode_calls += 1
        return original_decode(*args, **kwargs)

    attention.decode = counted_decode
    tokens = torch.randint(3, config.vocab_size, (1, 64), device="cuda")
    with torch.inference_mode():
        output = model.generate(
            input_ids=tokens,
            attention_mask=torch.ones_like(tokens),
            max_new_tokens=3,
            min_new_tokens=3,
            do_sample=False,
            use_cache=True,
            pad_token_id=0,
        )
    torch.cuda.synchronize()
    if output.shape != (1, 67) or decode_calls != 2:
        raise RuntimeError(
            f"Incremental decode gate failed: shape={output.shape}, calls={decode_calls}"
        )
    print("Incremental decode smoke passed: 2 decode calls.")


if __name__ == "__main__":
    main()
