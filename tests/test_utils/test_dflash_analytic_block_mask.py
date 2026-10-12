import unittest

import torch

from specforge.algorithms.common.dflash_family_model import (
    FLEX_ATTENTION_AVAILABLE,
    create_dflash_block_mask,
    create_dflash_sdpa_mask,
)


def _sorted_anchors(loss_mask, num_anchors, generator):
    """Mirror ``_sample_anchor_positions``: sorted anchors, dropped ones at 0."""

    bsz, seq_len = loss_mask.shape
    num_candidates = seq_len - 1
    valid = (loss_mask[:, :num_candidates] > 0.5) & (
        loss_mask[:, 1 : num_candidates + 1] > 0.5
    )
    valid_counts = valid.sum(dim=1)
    width = min(num_anchors, int(valid_counts.max()))
    random_values = torch.rand(valid.shape, generator=generator)
    random_values.masked_fill_(~valid, 2.0)
    candidates = random_values.argsort(dim=1)[:, :width]
    keep = torch.arange(width).unsqueeze(0) < valid_counts.clamp(max=width).unsqueeze(1)
    anchors = torch.where(keep, candidates, torch.full_like(candidates, num_candidates))
    anchors = anchors.sort(dim=1).values
    keep = anchors < num_candidates
    return torch.where(keep, anchors, 0), keep


def _block_sets(block_mask):
    """(partial, full) as dense ``[B, Q_BLOCKS, KV_BLOCKS]`` booleans."""

    def dense(num, idx):
        B, _, nq = num.shape
        nkv = idx.shape[-1]
        out = torch.zeros(B, nq, nkv, dtype=torch.bool)
        for b in range(B):
            for q in range(nq):
                n = int(num[b, 0, q])
                out[b, q, idx[b, 0, q, :n].long()] = True
        return out

    partial = dense(block_mask.kv_num_blocks, block_mask.kv_indices)
    full = dense(block_mask.full_kv_num_blocks, block_mask.full_kv_indices)
    return partial, full


@unittest.skipUnless(FLEX_ATTENTION_AVAILABLE, "flex_attention unavailable")
class TestAnalyticDFlashBlockMask(unittest.TestCase):
    CASES = [
        # (lengths, supervised_start, num_anchors, block_size, sliding_window, blocks)
        ([700, 523], 40, 64, 8, 256, (128, 128)),
        ([700, 523], 40, 64, 8, None, (128, 128)),
        ([1300], 0, 100, 8, 512, (128, 128)),  # N*block not a multiple of 128
        ([900, 260, 40], 10, 64, 8, 128, (128, 128)),  # dropped anchors
        ([2000, 1999], 300, 128, 8, 2048, (256, 128)),  # FLASH block sizes
        ([333], 5, 16, 4, 64, (64, 64)),
        ([1024, 1024], 0, 128, 16, None, (128, 128)),
    ]

    def test_matches_dense_block_mask(self):
        generator = torch.Generator().manual_seed(0)
        for lengths, start, num_anchors, block_size, window, blocks in self.CASES:
            S = max(lengths)
            loss_mask = torch.zeros(len(lengths), S)
            for row, length in enumerate(lengths):
                loss_mask[row, start:length] = 1.0
                # sparse holes so anchors are not contiguous
                loss_mask[row, start + 7 : length : 13] = 0.0
            for trial in range(3):
                anchors, keep = _sorted_anchors(loss_mask, num_anchors, generator)
                kwargs = dict(
                    anchor_positions=anchors,
                    block_keep_mask=keep,
                    S=S,
                    block_size=block_size,
                    device=torch.device("cpu"),
                    flex_block_size=blocks,
                    sliding_window=window,
                )
                with self.subTest(lengths=lengths, window=window, trial=trial):
                    ref = create_dflash_block_mask(**kwargs, analytic=False)
                    fast = create_dflash_block_mask(**kwargs, analytic=True)
                    self.assertEqual(ref.BLOCK_SIZE, fast.BLOCK_SIZE)
                    self.assertEqual(ref.seq_lengths, fast.seq_lengths)
                    ref_partial, ref_full = _block_sets(ref)
                    fast_partial, fast_full = _block_sets(fast)
                    # Identical full blocks keep the kernel's visit order and
                    # numerics unchanged.
                    self.assertTrue(torch.equal(ref_full, fast_full))
                    # Every block holding a visible key is visited.
                    self.assertTrue(
                        bool(
                            (
                                (ref_partial | ref_full) <= (fast_partial | fast_full)
                            ).all()
                        )
                    )
                    # Extra partial blocks may only hold masked keys.
                    dense = create_dflash_sdpa_mask(
                        anchors, keep, S, block_size, "cpu", sliding_window=window
                    )[:, 0]
                    q_blk, kv_blk = blocks
                    B, Q_LEN, KV_LEN = dense.shape
                    nq, nkv = fast_partial.shape[1:]
                    padded = torch.zeros(B, nq * q_blk, nkv * kv_blk, dtype=torch.bool)
                    padded[:, :Q_LEN, :KV_LEN] = dense
                    any_visible = (
                        padded.view(B, nq, q_blk, nkv, kv_blk).any(dim=4).any(dim=2)
                    )
                    extra = fast_partial & ~(ref_partial | ref_full)
                    self.assertFalse(bool((extra & any_visible).any()))
                    # The backward's transposed (query-block) indices agree too.
                    self.assertTrue(
                        torch.equal(ref.full_q_num_blocks, fast.full_q_num_blocks)
                    )

    def test_typical_layout_is_identical(self):
        """Dense anchors (the training regime) give the exact same BlockMask."""

        generator = torch.Generator().manual_seed(1)
        S = 3000
        loss_mask = torch.zeros(2, S)
        loss_mask[0, 200:2900] = 1.0
        loss_mask[1, 50:2400] = 1.0
        anchors, keep = _sorted_anchors(loss_mask, 512, generator)
        kwargs = dict(
            anchor_positions=anchors,
            block_keep_mask=keep,
            S=S,
            block_size=8,
            device=torch.device("cpu"),
            sliding_window=2048,
        )
        ref = create_dflash_block_mask(**kwargs, analytic=False)
        fast = create_dflash_block_mask(**kwargs, analytic=True)
        for name in (
            "kv_num_blocks",
            "full_kv_num_blocks",
            "q_num_blocks",
            "full_q_num_blocks",
        ):
            self.assertTrue(torch.equal(getattr(ref, name), getattr(fast, name)), name)
        ref_partial, ref_full = _block_sets(ref)
        fast_partial, fast_full = _block_sets(fast)
        self.assertTrue(torch.equal(ref_partial, fast_partial))
        self.assertTrue(torch.equal(ref_full, fast_full))


if __name__ == "__main__":
    unittest.main()
