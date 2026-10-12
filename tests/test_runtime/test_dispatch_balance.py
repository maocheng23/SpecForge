# coding=utf-8
"""Cost-balanced optimizer-window dispatch (CPU only)."""

import os
import random
import tempfile
import unittest
from types import SimpleNamespace

from specforge.runtime.contracts import FeatureSpec, SampleRef
from specforge.runtime.control_plane.controller import DataFlowController
from specforge.runtime.data_plane.dispatch_balance import (
    dflash_dispatch_cost_fn,
    loss_pairs,
    micro_batch_cost,
    plan_balanced_window,
)
from specforge.runtime.data_plane.ref_distributor import RefDistributor
from specforge.runtime.data_plane.streaming_ref_channel import StreamingRefChannel


def _ref(sid, tokens, pairs):
    spec = FeatureSpec(name="hidden_state", shape=(tokens, 8), dtype="bfloat16")
    return SampleRef(
        sample_id=sid,
        run_id="run0",
        source_task_id=None,
        feature_store_uri=f"mooncake://run0/{sid}",
        feature_keys={"hidden_state": f"{sid}/hidden_state"},
        feature_specs={"hidden_state": spec},
        strategy="dflash",
        num_tokens=tokens,
        metadata={"loss_pairs": pairs},
    )


_DRAFT = SimpleNamespace(
    hidden_size=64,
    num_hidden_layers=2,
    intermediate_size=128,
    head_dim=16,
    num_attention_heads=4,
    num_key_value_heads=2,
    vocab_size=1000,
    dflash_config={"block_size": 8, "target_layer_ids": [1, 3, 5]},
)


def _window(n, seed):
    rng = random.Random(seed)
    refs = []
    for i in range(n):
        tokens = rng.choice([rng.randint(50, 400), rng.randint(3000, 4096)])
        refs.append(_ref(f"s{i:03d}", tokens, rng.randint(1, tokens - 1)))
    return refs


class TestPlanBalancedWindow(unittest.TestCase):
    def test_loss_pairs(self):
        self.assertEqual(loss_pairs([0, 1, 1, 1, 0, 1, 1]), 3)
        self.assertEqual(loss_pairs([]), 0)
        self.assertEqual(loss_pairs([1]), 0)

    def test_plan_is_a_partition_with_exact_shapes(self):
        cost = dflash_dispatch_cost_fn(_DRAFT, num_anchors=512)
        for dp, batch, accum in ((3, 2, 6), (4, 2, 4), (2, 1, 5), (1, 2, 3)):
            refs = _window(dp * batch * accum, seed=dp * 10 + accum)
            rounds = plan_balanced_window(refs, dp, batch, accum, cost)
            self.assertEqual(len(rounds), accum)
            seen = []
            for round_batches in rounds:
                self.assertEqual(len(round_batches), dp)
                for mb in round_batches:
                    self.assertEqual(len(mb), batch)
                    seen.extend(ref.sample_id for ref in mb)
            self.assertEqual(sorted(seen), sorted(ref.sample_id for ref in refs))

    def test_plan_is_deterministic(self):
        cost = dflash_dispatch_cost_fn(_DRAFT, num_anchors=512)
        refs = _window(36, seed=7)
        first = plan_balanced_window(refs, 3, 2, 6, cost)
        second = plan_balanced_window(list(reversed(refs)), 3, 2, 6, cost)
        ids = lambda plan: [
            [[r.sample_id for r in mb] for mb in rb] for rb in plan
        ]  # noqa: E731
        self.assertEqual(ids(first), ids(second))

    def test_balances_better_than_round_robin(self):
        cost = dflash_dispatch_cost_fn(_DRAFT, num_anchors=512)
        dp, batch, accum = 4, 2, 4
        worse = 0
        for seed in range(40):
            refs = _window(dp * batch * accum, seed=seed)
            # round-robin baseline, exactly as RefDistributor deals rounds
            rr = [[0.0] for _ in range(dp)]
            round_size = dp * batch
            for start in range(0, len(refs), round_size):
                chunk = refs[start : start + round_size]
                for rank in range(dp):
                    rr[rank][0] += micro_batch_cost(chunk[rank::dp], cost)
            rr_max = max(load[0] for load in rr)
            plan = plan_balanced_window(refs, dp, batch, accum, cost)
            loads = [0.0] * dp
            for round_batches in plan:
                for rank, mb in enumerate(round_batches):
                    loads[rank] += micro_batch_cost(mb, cost)
            worse += max(loads) > rr_max + 1e-6
        self.assertEqual(worse, 0)

    def test_missing_loss_pairs_falls_back_to_tokens(self):
        cost = dflash_dispatch_cost_fn(_DRAFT, num_anchors=16)
        ref = _ref("x", 100, 0)
        ref.metadata.pop("loss_pairs")
        with_pairs = cost(_ref("y", 100, 100))
        self.assertEqual(cost(ref), with_pairs)

    def test_wrong_window_size_raises(self):
        cost = dflash_dispatch_cost_fn(_DRAFT, num_anchors=512)
        with self.assertRaises(ValueError):
            plan_balanced_window(_window(5, 0), 2, 1, 2, cost)


class TestBalancedRefDistributor(unittest.TestCase):
    def test_windows_dispatch_whole_and_balanced(self):
        tmp = tempfile.mkdtemp(prefix="balanced_dispatch_")
        src = os.path.join(tmp, "refs.jsonl")
        inbox_dir = os.path.join(tmp, "inboxes")
        producer = StreamingRefChannel(src)
        cost = dflash_dispatch_cost_fn(_DRAFT, num_anchors=512)
        dist = RefDistributor(
            StreamingRefChannel(src),
            DataFlowController("run0"),
            inbox_dir,
            2,
            feature_store=SimpleNamespace(
                adopt=lambda ref: None, abort=lambda *a: None
            ),
            refs_per_rank_step=4,
            refs_per_rank_batch=2,
            cost_fn=cost,
        )
        refs = _window(19, seed=3)  # two full windows of 8 + 3 held refs
        for ref in refs:
            producer.publish(ref)
        for _ in range(20):
            if not dist.pump():
                break
        inbox = [
            [
                r.sample_id
                for r in StreamingRefChannel(
                    RefDistributor.inbox_path(inbox_dir, k)
                ).poll()
            ]
            for k in range(2)
        ]
        self.assertEqual(len(inbox[0]), 8)
        self.assertEqual(len(inbox[1]), 8)
        # every dispatched ref comes from the first two complete windows
        first16 = {ref.sample_id for ref in refs[:16]}
        self.assertEqual(set(inbox[0]) | set(inbox[1]), first16)
        # each window is split evenly: 4 refs per rank per window
        window_one = {ref.sample_id for ref in refs[:8]}
        self.assertEqual(sum(sid in window_one for sid in inbox[0]), 4)
        self.assertEqual(dist.stats["dispatched"], 16)


if __name__ == "__main__":
    unittest.main()
