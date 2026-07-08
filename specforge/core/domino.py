# coding=utf-8
"""Domino Training Wrapper."""

import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as _checkpoint

from specforge.core.dflash import (
    FLEX_ATTENTION_AVAILABLE,
    BlockMask,
    create_dflash_block_mask,
    create_dflash_sdpa_mask,
)
from specforge.modeling.draft.dflash import DFlashDraftModel
from specforge.utils import get_device_type


def compute_accept_len(
    pred_ids_4d: torch.Tensor,
    target_ids_4d: torch.Tensor,
    valid_mask_4d: torch.Tensor,
) -> torch.Tensor:
    """Compute per-block acceptance length.

    For each block, returns the number of consecutive correct predictions
    starting from position 0 (or the first valid position).
    """
    correct = (pred_ids_4d == target_ids_4d) | (~valid_mask_4d)
    accept_prefix = correct.long().cumprod(dim=2) * valid_mask_4d.long()
    return accept_prefix.sum(dim=2).float()


class OnlineDominoModel(nn.Module):
    """Domino online training wrapper with block-wise CE loss."""

    def __init__(
        self,
        draft_model: DFlashDraftModel,
        target_lm_head: nn.Module,
        target_embed_tokens: nn.Module,
        mask_token_id: int,
        block_size: int = 16,
        attention_backend: str = "flex_attention",
        num_anchors: int = 512,
        loss_decay_gamma: Optional[float] = None,
        shift_label: bool = False,
    ):
        super().__init__()
        self.draft_model = draft_model
        self.lm_head = target_lm_head
        self.embed_tokens = target_embed_tokens
        self.block_size = block_size
        self.mask_token_id = mask_token_id
        self.attention_backend = attention_backend
        self.num_anchors = num_anchors
        self.loss_decay_gamma = loss_decay_gamma
        self.shift_label = shift_label

        # Fused / row-tiled loss (opt-in; dense path stays default). Chunks the
        # head+CE over the anchor dim and recomputes each chunk in backward, so
        # the two full-vocab (base + final) logit tensors are never all live at
        # once — bounding the loss-layer peak to one anchor-chunk. Math-exact
        # (fp-close). See examples/disagg/PERF_FINDINGS.md.
        self.fused_ce = os.environ.get("DOMINO_FUSED_CE", "0") == "1"
        self.ce_chunk = max(1, int(os.environ.get("DOMINO_CE_CHUNK", "32")))
        # When the schedule has decayed lambda_base to exactly 0, the base CE
        # contributes 0 to the total-loss gradient; skipping its cross-entropy
        # (not its projection — base_logits still feeds final + base metrics) is
        # gradient-identical and saves that CE's fwd/bwd. Only ever guarded on
        # ==0.0 (the ==1.0 case would zero embed_proj/GRU grads vs the dense
        # materialized-zero and is NOT applied).
        self.fused_skip_zero_base_ce = (
            os.environ.get("DOMINO_SKIP_ZERO_BASE_CE", "1") == "1"
        )

        self._cached_block_mask: Optional[BlockMask] = None
        self._cached_seq_len: Optional[int] = None
        self._cached_bsz: Optional[int] = None

        if self.attention_backend == "flex_attention" and not FLEX_ATTENTION_AVAILABLE:
            raise ValueError(
                "flex_attention is not available on this device; use sdpa/eager."
            )

        # NPU workaround: pre-create a float16 GRU module since Ascend
        # DynamicGRU does not support bfloat16.
        # Use object.__setattr__ to avoid registering it as a submodule,
        # otherwise FSDP will complain about mixed dtypes (bfloat16 + float16).
        gru_fp16 = None
        if get_device_type() == "npu":
            prefix_gru = draft_model.prefix_gru
            gru_fp16 = nn.GRU(
                input_size=prefix_gru.weight_ih_l0.shape[1],
                hidden_size=prefix_gru.hidden_size,
                num_layers=1,
                batch_first=True,
                bias=False,
            )
            gru_fp16.to(
                device=next(prefix_gru.parameters()).device, dtype=torch.float16
            )
            gru_fp16.weight_ih_l0.requires_grad = False
            gru_fp16.weight_hh_l0.requires_grad = False
        object.__setattr__(self, "_gru_fp16", gru_fp16)

    def _sample_anchor_positions(
        self, seq_len: int, loss_mask: torch.Tensor, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Randomly sample anchor positions per sample; returns (anchors, keep_mask)."""
        bs = self.block_size
        bsz = loss_mask.shape[0]
        max_anchor = max(seq_len - bs, 0)

        valid = loss_mask[:, : max_anchor + 1] > 0.5
        valid_counts = valid.sum(dim=1)
        max_n = max(1, min(self.num_anchors, int(valid_counts.max().item()) - 1))

        indices = (
            torch.arange(max_anchor + 1, device=device).unsqueeze(0).expand(bsz, -1)
        )
        masked_indices = torch.where(
            valid, indices, torch.tensor(seq_len + 1, device=device)
        )

        random_vals = torch.rand(bsz, max_anchor + 1, device=device)
        random_vals = torch.where(valid, random_vals, torch.tensor(2.0, device=device))

        _, sorted_idx = random_vals.sort(dim=1)
        gathered = torch.gather(masked_indices, 1, sorted_idx)
        anchors = gathered[:, :max_n].sort(dim=1).values

        keep_mask = torch.arange(max_n, device=device).unsqueeze(
            0
        ) < valid_counts.unsqueeze(1).clamp(max=max_n)
        anchors = torch.where(
            keep_mask, anchors, torch.tensor(0, dtype=torch.long, device=device)
        )

        return anchors, keep_mask

    def prepare_noise_input(
        self, input_ids: torch.Tensor, block_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Prepare noise input: first token of each block is real, rest are MASK."""
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        if block_ids is not None:
            is_block_start = torch.ones(bsz, seq_len, dtype=torch.bool, device=device)
            is_block_start[:, 1:] = block_ids[:, 1:] != block_ids[:, :-1]
        else:
            positions = torch.arange(seq_len, device=device)
            is_block_start = (positions % self.block_size) == 0
            is_block_start = is_block_start.unsqueeze(0).expand(bsz, -1)

        noise_input_ids = torch.full_like(input_ids, self.mask_token_id)
        noise_input_ids[is_block_start] = input_ids[is_block_start]
        return noise_input_ids

    def _create_position_ids(self, anchor_positions: torch.Tensor) -> torch.Tensor:
        """Create absolute position IDs for parallel draft blocks."""
        bsz, n_blocks = anchor_positions.shape
        device = anchor_positions.device
        offsets = torch.arange(self.block_size, device=device).view(1, 1, -1)
        pos_ids = anchor_positions.unsqueeze(-1) + offsets
        return pos_ids.view(bsz, -1)

    def _create_noise_embed(self, input_ids, anchor_positions, block_keep_mask):
        bsz, seq_len = input_ids.shape
        n = anchor_positions.shape[1]
        bs = self.block_size
        device = input_ids.device

        noise_ids = torch.full(
            (bsz, n * bs), self.mask_token_id, dtype=torch.long, device=device
        )

        block_starts = torch.arange(n, device=device) * bs
        block_starts = block_starts.unsqueeze(0).expand(bsz, -1)

        valid_anchor_positions = anchor_positions.clamp(0, seq_len - 1)
        anchor_tokens = torch.gather(input_ids, 1, valid_anchor_positions)

        flat_batch_idx = torch.arange(bsz, device=device).unsqueeze(1).expand(bsz, n)
        noise_ids[flat_batch_idx, block_starts] = torch.where(
            block_keep_mask,
            anchor_tokens,
            torch.tensor(self.mask_token_id, dtype=torch.long, device=device),
        )

        return self.embed_tokens(noise_ids)

    @property
    def _suffix_start(self) -> int:
        """Return suffix_start index based on shift_label and pure_draft_prefix_len."""
        pure_prefix = getattr(self.draft_model, "pure_draft_prefix_len", 0)
        return pure_prefix if self.shift_label else (1 + pure_prefix)

    def _build_domino_head_inputs(
        self,
        input_ids: torch.Tensor,
        anchor_positions: torch.Tensor,
        target_ids: torch.Tensor,
        output_hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, n, bs = target_ids.shape
        hidden4d = output_hidden.reshape(bsz, n, bs, output_hidden.shape[-1])

        prev_ids = target_ids
        if self.shift_label:
            prev_offsets = torch.arange(
                0, self.block_size, device=input_ids.device
            ).view(1, 1, -1)
            prev_indices = (anchor_positions.unsqueeze(-1) + prev_offsets).clamp(
                max=input_ids.size(1) - 1
            )
            prev_ids = torch.gather(
                input_ids.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
                2,
                prev_indices,
            )

        return hidden4d, prev_ids

    def _run_gru(self, gru_inputs: torch.Tensor) -> torch.Tensor:
        """Run GRU with NPU-compatible dtype.

        Ascend NPU DynamicGRU does not support bfloat16; use the pre-created
        float16 GRU module and cast outputs back to the original dtype.
        """
        if self._gru_fp16 is not None and gru_inputs.dtype == torch.bfloat16:
            self._gru_fp16.weight_ih_l0.data.copy_(
                self.draft_model.prefix_gru.weight_ih_l0.data.half()
            )
            self._gru_fp16.weight_hh_l0.data.copy_(
                self.draft_model.prefix_gru.weight_hh_l0.data.half()
            )
            out = self._gru_fp16(gru_inputs.half())[0]
            return out.to(gru_inputs.dtype)
        return self.draft_model.prefix_gru(gru_inputs)[0]

    def _apply_domino_head(
        self,
        base_logits4d: torch.Tensor,
        hidden4d: torch.Tensor,
        prev_ids: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the Domino head: GRU causal state plus logit correction."""
        bsz, n, bs = target_ids.shape
        if self.shift_label:
            block_emb = self.embed_tokens(prev_ids)
            gru_inputs = block_emb.reshape(bsz * n, bs, -1)
            gru_out = self._run_gru(gru_inputs)
            gru_out = gru_out.reshape(bsz, n, bs, -1)
            prefix_states = gru_out[:, :, self._suffix_start :, :]
        else:
            block_emb = self.embed_tokens(target_ids)
            gru_inputs = block_emb[:, :, : bs - 1, :].reshape(bsz * n, bs - 1, -1)
            gru_out = self._run_gru(gru_inputs)
            gru_out = gru_out.reshape(bsz, n, bs - 1, -1)
            prefix_states = gru_out[:, :, self._suffix_start - 1 :, :]
        z_n = hidden4d[:, :, self._suffix_start :, :]
        concat_features = torch.cat([z_n, prefix_states], dim=-1)
        logits_e = self.draft_model.embed_proj(concat_features)

        prefix_logits = base_logits4d[:, :, : self._suffix_start, :]
        suffix_logits = base_logits4d[:, :, self._suffix_start :, :] + logits_e
        return torch.cat([prefix_logits, suffix_logits], dim=2)

    def _fused_head_and_losses(
        self,
        output_hidden: torch.Tensor,
        input_ids: torch.Tensor,
        anchor_positions: torch.Tensor,
        target_ids: torch.Tensor,
        weight_mask: torch.Tensor,
        lambda_base: float,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Row-tiled, activation-checkpointed equivalent of the dense head+CE.

        Chunks the anchor dim; each chunk computes base_logits = lm_head(hidden),
        the domino-head final_logits, and both weighted CE numerators, then is
        recomputed in backward (``checkpoint``) so the full-vocab logit tensors
        for all anchors are never simultaneously live. Math-exact vs the dense
        path up to bf16 cross-chunk reassociation. Returns
        ``(loss, final_loss, base_loss, pred_ids, base_pred_ids)`` where the two
        pred-id tensors are full-vocab argmaxes flattened in the same
        ``(bsz, n, bs)`` row-major order as the dense ``flat_logits``.
        """
        bsz, n, bs = target_ids.shape
        hidden4d, prev_ids = self._build_domino_head_inputs(
            input_ids, anchor_positions, target_ids, output_hidden
        )
        valid_token_count = weight_mask.reshape(-1).sum() + 1e-6
        # base CE contributes lambda_base * base_loss to the total; at exactly 0
        # its gradient is 0, so its cross-entropy is safe to skip (base_logits
        # itself is still built — final = base + correction, and base metrics).
        skip_base = self.fused_skip_zero_base_ce and float(lambda_base) == 0.0

        def _chunk(h4d_c, prev_c, tgt_c, w_c):
            base_c = self.lm_head(h4d_c)
            final_c = self._apply_domino_head(base_c, h4d_c, prev_c, tgt_c)
            vocab = base_c.shape[-1]
            flat_final = final_c.reshape(-1, vocab)
            flat_base = base_c.reshape(-1, vocab)
            t = tgt_c.reshape(-1)
            w = w_c.reshape(-1)
            fnum = (
                F.cross_entropy(flat_final, t, reduction="none") * w
            ).sum()
            if skip_base:
                bnum = fnum.new_zeros(())
            else:
                bnum = (
                    F.cross_entropy(flat_base, t, reduction="none") * w
                ).sum()
            with torch.no_grad():
                p = flat_final.argmax(dim=-1)
                bp = flat_base.argmax(dim=-1)
            return fnum, bnum, p, bp

        final_num = output_hidden.new_zeros(())
        base_num = output_hidden.new_zeros(())
        pred_chunks = []
        base_pred_chunks = []
        for c0 in range(0, n, self.ce_chunk):
            c1 = min(n, c0 + self.ce_chunk)
            args = (
                hidden4d[:, c0:c1],
                prev_ids[:, c0:c1],
                target_ids[:, c0:c1],
                weight_mask[:, c0:c1],
            )
            if torch.is_grad_enabled() and output_hidden.requires_grad:
                fnum, bnum, p, bp = _checkpoint.checkpoint(
                    _chunk, *args, use_reentrant=False
                )
            else:
                fnum, bnum, p, bp = _chunk(*args)
            final_num = final_num + fnum
            base_num = base_num + bnum
            # p/bp are (bsz*nc*bs,) in (b, nc, bs) order -> restore anchor dim.
            pred_chunks.append(p.reshape(bsz, -1, bs))
            base_pred_chunks.append(bp.reshape(bsz, -1, bs))

        final_loss = final_num / valid_token_count
        base_loss = base_num / valid_token_count
        loss = (1.0 - lambda_base) * final_loss + lambda_base * base_loss
        pred_ids = torch.cat(pred_chunks, dim=1).reshape(-1)
        base_pred_ids = torch.cat(base_pred_chunks, dim=1).reshape(-1)
        return loss, final_loss, base_loss, pred_ids, base_pred_ids

    def _compute_extra_metrics(
        self,
        pred_ids: torch.Tensor,
        base_pred_ids: torch.Tensor,
        flat_targets: torch.Tensor,
        binary_eval_mask: torch.Tensor,
        actual_token_count: torch.Tensor,
        target_ids: torch.Tensor,
        eval_weight_mask: torch.Tensor,
        final_loss: torch.Tensor,
        base_loss: torch.Tensor,
        lambda_base: float,
    ) -> Dict[str, torch.Tensor]:
        """Compute auxiliary training metrics that do not affect gradients.

        ``pred_ids`` / ``base_pred_ids`` are the full-vocab argmaxes of the final
        and base logits, precomputed by the caller (both the dense and the fused
        loss paths produce them) so this never needs the full logit tensors.
        """
        bsz, n, bs = target_ids.shape

        base_correct = (base_pred_ids == flat_targets) & (binary_eval_mask > 0.5)
        base_accuracy = base_correct.sum().float() / actual_token_count

        valid_mask_4d = (eval_weight_mask > 0).bool()
        pred_accept_len = compute_accept_len(
            pred_ids.view(bsz, n, bs), target_ids, valid_mask_4d
        )
        base_accept_len = compute_accept_len(
            base_pred_ids.view(bsz, n, bs), target_ids, valid_mask_4d
        )

        valid_block_mask = valid_mask_4d.any(dim=2)
        num_valid_blocks = valid_block_mask.sum().float() + 1e-6
        avg_accept_len = (
            (pred_accept_len + 1.0) * valid_block_mask.float()
        ).sum() / num_valid_blocks
        base_avg_accept_len = (
            (base_accept_len + 1.0) * valid_block_mask.float()
        ).sum() / num_valid_blocks

        return {
            "final_loss": final_loss.detach(),
            "base_loss": base_loss.detach(),
            "base_accuracy": base_accuracy.detach(),
            "accept_len": avg_accept_len.detach(),
            "base_accept_len": base_avg_accept_len.detach(),
            "lambda_base": torch.tensor(lambda_base, device=final_loss.device),
        }

    def _compute_weighted_losses(
        self,
        final_logits: torch.Tensor,
        base_logits: torch.Tensor,
        target_ids: torch.Tensor,
        weight_mask: torch.Tensor,
        lambda_base: float,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        flat_logits = final_logits.reshape(-1, final_logits.size(-1))
        flat_base_logits = base_logits.reshape(-1, base_logits.size(-1))
        flat_targets = target_ids.reshape(-1)
        flat_weights = weight_mask.reshape(-1)

        valid_token_count = flat_weights.sum() + 1e-6

        final_loss_per_token = F.cross_entropy(
            flat_logits, flat_targets, reduction="none"
        )
        final_loss = (final_loss_per_token * flat_weights).sum() / valid_token_count

        base_loss_per_token = F.cross_entropy(
            flat_base_logits, flat_targets, reduction="none"
        )
        base_loss = (base_loss_per_token * flat_weights).sum() / valid_token_count

        loss = (1.0 - lambda_base) * final_loss + lambda_base * base_loss

        return loss, final_loss, base_loss, flat_logits, flat_base_logits, flat_targets

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
        lambda_base: float = 0.0,
    ):
        """Parallel block-wise training forward pass."""
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        anchor_positions, block_keep_mask = self._sample_anchor_positions(
            seq_len, loss_mask, device
        )

        noise_embedding = self._create_noise_embed(
            input_ids, anchor_positions, block_keep_mask
        )

        context_position_ids = (
            torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
        )
        draft_position_ids = self._create_position_ids(anchor_positions)
        full_position_ids = torch.cat([context_position_ids, draft_position_ids], dim=1)

        if self.attention_backend == "flex_attention":
            dflash_attn_mask = create_dflash_block_mask(
                anchor_positions=anchor_positions,
                block_keep_mask=block_keep_mask,
                S=seq_len,
                block_size=self.block_size,
                device=device,
            )
        else:
            dflash_attn_mask = create_dflash_sdpa_mask(
                anchor_positions=anchor_positions,
                block_keep_mask=block_keep_mask,
                S=seq_len,
                block_size=self.block_size,
                device=device,
            )

        output_hidden = self.draft_model(
            position_ids=full_position_ids,
            noise_embedding=noise_embedding,
            target_hidden=hidden_states,
            attention_mask=dflash_attn_mask,
        )

        # --- Labels ---
        label_start = 1 if self.shift_label else 0
        label_offsets = torch.arange(
            label_start, label_start + self.block_size, device=device
        ).view(1, 1, -1)
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets
        valid_label_mask = label_indices < seq_len
        safe_target_indices = label_indices.clamp(max=seq_len - 1)

        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            safe_target_indices,
        )

        bsz, n, bs = target_ids.shape

        # --- Weight mask: block validity * bounds * exclude anchor (pos 0) * loss_mask ---
        # Computed before the logits so it can be fed to either loss path.
        weight_mask = (
            block_keep_mask.unsqueeze(-1).expand(-1, -1, self.block_size).float()
        )
        weight_mask = weight_mask * valid_label_mask.float()

        if not self.shift_label:
            pos_in_block = torch.arange(self.block_size, device=device).view(1, 1, -1)
            weight_mask = weight_mask * (pos_in_block > 0).float()

        original_loss_mask_gathered = torch.gather(
            loss_mask.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            safe_target_indices,
        )
        weight_mask = weight_mask * original_loss_mask_gathered

        # Save eval mask before decay (for accept_len / accuracy stats)
        eval_weight_mask = weight_mask.clone()
        binary_eval_mask = weight_mask.view(-1)

        # --- Loss decay: first valid position gets weight 1.0 ---
        if self.loss_decay_gamma is not None and self.loss_decay_gamma > 0:
            k = torch.arange(self.block_size, device=device).view(1, 1, -1)
            offset = 0 if self.shift_label else 1
            decay_weights = torch.exp(
                -(k - offset).clamp(min=0).float() / self.loss_decay_gamma
            )
            weight_mask = weight_mask * decay_weights

        flat_targets = target_ids.reshape(-1)

        if self.fused_ce:
            # Row-tiled + checkpointed head+CE: never materializes the two
            # full-vocab logit tensors for all anchors at once. Math-exact
            # (fp-close) vs the dense branch below.
            loss, final_loss, base_loss, pred_ids, base_pred_ids = (
                self._fused_head_and_losses(
                    output_hidden=output_hidden,
                    input_ids=input_ids,
                    anchor_positions=anchor_positions,
                    target_ids=target_ids,
                    weight_mask=weight_mask,
                    lambda_base=lambda_base,
                )
            )
        else:
            base_logits = self.lm_head(output_hidden)
            hidden4d, prev_ids = self._build_domino_head_inputs(
                input_ids=input_ids,
                anchor_positions=anchor_positions,
                target_ids=target_ids,
                output_hidden=output_hidden,
            )
            base_logits4d = base_logits.reshape(bsz, n, bs, -1)
            final_logits = self._apply_domino_head(
                base_logits4d=base_logits4d,
                hidden4d=hidden4d,
                prev_ids=prev_ids,
                target_ids=target_ids,
            ).reshape(bsz, n * bs, -1)

            loss, final_loss, base_loss, flat_logits, flat_base_logits, _ = (
                self._compute_weighted_losses(
                    final_logits=final_logits,
                    base_logits=base_logits,
                    target_ids=target_ids,
                    weight_mask=weight_mask,
                    lambda_base=lambda_base,
                )
            )
            with torch.no_grad():
                pred_ids = torch.argmax(flat_logits, dim=-1)
                base_pred_ids = torch.argmax(flat_base_logits, dim=-1)

        # --- Accuracy (shared by both paths) ---
        with torch.no_grad():
            correct = (pred_ids == flat_targets) & (binary_eval_mask > 0.5)
            accuracy_denom = binary_eval_mask.sum()
            actual_token_count = accuracy_denom + 1e-6
            accuracy = correct.sum().float() / actual_token_count

            metrics = self._compute_extra_metrics(
                pred_ids=pred_ids,
                base_pred_ids=base_pred_ids,
                flat_targets=flat_targets,
                binary_eval_mask=binary_eval_mask,
                actual_token_count=actual_token_count,
                target_ids=target_ids,
                eval_weight_mask=eval_weight_mask,
                final_loss=final_loss,
                base_loss=base_loss,
                lambda_base=lambda_base,
            )
            metrics["accuracy_denom"] = accuracy_denom

        return loss, accuracy, metrics
