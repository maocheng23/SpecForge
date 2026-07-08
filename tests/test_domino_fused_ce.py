"""Equivalence test: dense vs fused/tiled domino CE (loss, metrics, gradients).

Runs on CPU with tiny dims and sdpa attention. `output_hidden` is computed once
before the dense/fused branch, so the attention backend is irrelevant to the
comparison — this isolates the head + cross-entropy math. RNG is seeded
identically before each forward so `_sample_anchor_positions` picks the same
anchors in both paths.

Covers the folded-in optimizations:
  * fused row-tiled two-projection CE (no torch.cat of final_logits)
  * lambda_base==0 base-CE skip (gradient-identical; base_loss log differs)
across lambda_base in {0.0, 0.5, 1.0} and chunk sizes {1, 3, n}.
"""

import copy

import pytest
import torch
import torch.nn as nn
from transformers import Qwen3Config

from specforge.core.domino import OnlineDominoModel
from specforge.modeling.draft.dflash import DFlashDraftModel

HID = 64
VOCAB = 256
BLOCK = 4
NUM_ANCHORS = 8
BSZ = 2
SEQ = 48
MASK_ID = 255


def _build_model(dtype=torch.float32):
    cfg = Qwen3Config(
        hidden_size=HID,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=128,
        vocab_size=VOCAB,
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        attention_bias=False,
        attention_dropout=0.0,
        rope_theta=10000.0,
    )
    cfg._attn_implementation = "sdpa"
    cfg.layer_types = ["full_attention", "full_attention"]
    cfg.num_target_layers = 2
    cfg.block_size = BLOCK
    cfg.dflash_config = {
        "mask_token_id": MASK_ID,
        "target_layer_ids": [0, 1],
        "projector_type": "domino",
        "emb_dim": 48,
        "gru_hidden_dim": 32,
        "pure_draft_prefix_len": 1,
        "shift_label": True,
    }

    torch.manual_seed(1234)
    draft = DFlashDraftModel(cfg).to(dtype)
    lm_head = nn.Linear(HID, VOCAB, bias=False).to(dtype)
    embed_tokens = nn.Embedding(VOCAB, HID).to(dtype)
    for p in lm_head.parameters():
        p.requires_grad_(False)
    for p in embed_tokens.parameters():
        p.requires_grad_(False)

    model = OnlineDominoModel(
        draft_model=draft,
        target_lm_head=lm_head,
        target_embed_tokens=embed_tokens,
        mask_token_id=MASK_ID,
        block_size=BLOCK,
        attention_backend="sdpa",
        num_anchors=NUM_ANCHORS,
        loss_decay_gamma=7.0,
        shift_label=True,
    )
    return model


def _inputs(dtype=torch.float32):
    g = torch.Generator().manual_seed(7)
    input_ids = torch.randint(0, VOCAB, (BSZ, SEQ), generator=g)
    hidden = torch.randn(BSZ, SEQ, 2 * HID, generator=g, dtype=dtype)
    loss_mask = torch.ones(BSZ, SEQ)
    loss_mask[:, : SEQ // 4] = 0.0  # exercise masked-out region
    return input_ids, hidden, loss_mask


def _run(model, inputs, lambda_base, seed=99):
    input_ids, hidden, loss_mask = inputs
    for p in model.draft_model.parameters():
        if p.grad is not None:
            p.grad = None
    torch.manual_seed(seed)
    loss, accuracy, metrics = model(input_ids, hidden, loss_mask, lambda_base)
    loss.backward()
    grads = {
        name: p.grad.detach().clone()
        for name, p in model.draft_model.named_parameters()
        if p.grad is not None
    }
    return loss.detach().clone(), accuracy.detach().clone(), {
        k: v.detach().clone() for k, v in metrics.items()
    }, grads


@pytest.mark.parametrize("lambda_base", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("chunk", [1, 3, NUM_ANCHORS])
def test_fused_matches_dense_exact(lambda_base, chunk):
    """skip_base OFF -> loss, all metrics, and every draft grad match dense."""
    model = _build_model()
    inputs = _inputs()

    model.fused_ce = False
    d_loss, d_acc, d_metrics, d_grads = _run(model, inputs, lambda_base)

    model.fused_ce = True
    model.fused_skip_zero_base_ce = False
    model.ce_chunk = chunk
    f_loss, f_acc, f_metrics, f_grads = _run(model, inputs, lambda_base)

    assert torch.allclose(d_loss, f_loss, atol=1e-4, rtol=1e-4), (
        f"loss mismatch lb={lambda_base} chunk={chunk}: {d_loss} vs {f_loss}"
    )
    assert torch.allclose(d_acc, f_acc, atol=1e-5), "accuracy mismatch"
    for k in ["final_loss", "base_loss", "base_accuracy", "accept_len", "base_accept_len"]:
        assert torch.allclose(d_metrics[k], f_metrics[k], atol=1e-4, rtol=1e-4), (
            f"metric {k} mismatch: {d_metrics[k]} vs {f_metrics[k]}"
        )
    assert d_grads.keys() == f_grads.keys()
    for name in d_grads:
        assert torch.allclose(d_grads[name], f_grads[name], atol=1e-4, rtol=1e-3), (
            f"grad mismatch for {name}: max|Δ|="
            f"{(d_grads[name] - f_grads[name]).abs().max().item()}"
        )


def test_skip_zero_base_ce_grad_identical():
    """At lambda_base==0 the base-CE skip is gradient-identical to dense.

    Only base_loss's *logged* value differs (0 vs the real unweighted base CE);
    every draft gradient, the total loss, final_loss and all argmax metrics match.
    """
    model = _build_model()
    inputs = _inputs()

    model.fused_ce = False
    d_loss, d_acc, d_metrics, d_grads = _run(model, inputs, 0.0)

    model.fused_ce = True
    model.fused_skip_zero_base_ce = True
    model.ce_chunk = 3
    f_loss, f_acc, f_metrics, f_grads = _run(model, inputs, 0.0)

    assert torch.allclose(d_loss, f_loss, atol=1e-4, rtol=1e-4)
    assert torch.allclose(d_metrics["final_loss"], f_metrics["final_loss"], atol=1e-4)
    # base metrics still real (computed from base argmax), only base_loss log is 0
    assert torch.allclose(
        d_metrics["base_accept_len"], f_metrics["base_accept_len"], atol=1e-4
    )
    assert f_metrics["base_loss"].item() == 0.0
    for name in d_grads:
        assert torch.allclose(d_grads[name], f_grads[name], atol=1e-4, rtol=1e-3), (
            f"grad mismatch for {name} under skip"
        )


if __name__ == "__main__":
    for lb in (0.0, 0.5, 1.0):
        for ch in (1, 3, NUM_ANCHORS):
            test_fused_matches_dense_exact(lb, ch)
            print(f"OK exact lb={lb} chunk={ch}")
    test_skip_zero_base_ce_grad_identical()
    print("OK skip-zero-base-ce grad-identical")
    print("ALL EQUIVALENCE TESTS PASSED")
