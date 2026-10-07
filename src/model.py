"""Builds the ViT for every condition: rank 0 = frozen ViT + linear head, rank > 0 = the same model plus LoRA.

The head reads the CLS token after the final layernorm (no pooler), the same vector as Paddy's cached features,
so the frozen condition is literally rank 0 of the LoRA model. init_head(seed) is shared with Paddy's trainer
so seed s starts from the same head in all three conditions (Plan, decision 3).

Self-check: python -m src.model  (downloads the 346 MB checkpoint once)
"""
import torch
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import ViTForImageClassification

CHECKPOINT = "google/vit-base-patch16-224-in21k"
HEAD_PARAMS = 768 * 10 + 10           # 7,690
LORA_PARAMS_PER_RANK = 2 * 12 * 2 * 768  # (A + B) x 12 blocks x (query, value) x 768 = 36,864 per unit of rank


def init_head(seed):
    """The 768 -> 10 linear head. Same seed, same starting weights, in every condition and in Paddy's trainer."""
    torch.manual_seed(seed)
    return nn.Linear(768, 10)


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(rank, seed, lora_dropout=0.1):
    model = ViTForImageClassification.from_pretrained(CHECKPOINT, num_labels=10)
    model.classifier = init_head(seed)
    model.requires_grad_(False)
    model.classifier.requires_grad_(True)
    if rank > 0:
        # alpha = r keeps LoRA's alpha/r multiplier at 1, so rank doesn't change the effective learning rate (decision 1)
        # q_proj / v_proj are the query and value matrices: transformers 5.x renamed them from "query" / "value"
        config = LoraConfig(r=rank, lora_alpha=rank, lora_dropout=lora_dropout,
                            target_modules=["q_proj", "v_proj"], modules_to_save=["classifier"])
        model = get_peft_model(model, config)
    n = count_trainable(model)
    expected = HEAD_PARAMS + LORA_PARAMS_PER_RANK * rank
    print(f"rank {rank}: {n:,} trainable parameters (head {HEAD_PARAMS:,} + LoRA {n - HEAD_PARAMS:,})")
    assert n == expected, f"rank {rank}: expected {expected:,} trainable, got {n:,}"
    return model


if __name__ == "__main__":
    assert HEAD_PARAMS == 7_690
    assert LORA_PARAMS_PER_RANK * 4 == 147_456 and LORA_PARAMS_PER_RANK * 32 == 1_179_648

    a, b = init_head(0), init_head(0)
    assert torch.equal(a.weight, b.weight) and torch.equal(a.bias, b.bias), "same seed gave different heads"
    assert not torch.equal(a.weight, init_head(1).weight), "different seeds gave the same head"

    models = [build_model(r, seed=0).eval() for r in (0, 4, 32)]

    # LoRA's B starts at zero, so all three conditions must make identical predictions before training
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        logits = [m(pixel_values=x).logits for m in models]
    assert torch.allclose(logits[0], logits[1], atol=1e-5) and torch.allclose(logits[0], logits[2], atol=1e-5), \
        "conditions disagree at initialisation"

    print("model OK")
