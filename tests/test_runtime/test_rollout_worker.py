# coding=utf-8
"""RolloutWorker drives PromptTask -> features -> SampleRef commit (CPU, fake source)."""

import unittest

import torch

from specforge.runtime.control_plane import DataFlowController
from specforge.runtime.data_plane import LocalFeatureStore
from specforge.runtime.inference.capture import CaptureConfig
from specforge.runtime.inference.rollout_worker import RolloutWorker

H = 8


def _capture(layer_ids=(1, 2, 3)):
    return CaptureConfig.from_strategy(
        required_features={"input_ids", "attention_mask", "loss_mask", "hidden_state", "target"},
        aux_hidden_state_layer_ids=layer_ids,
        target_repr="logits",
        target_hidden_size=H,
        target_vocab_size=32,
    )


class FakeSource:
    """Stand-in for SGLangAdapter: returns per-sample feature dicts."""

    def __init__(self, seq=4, target_dim=32, aux_layers=3, bad_layers=False):
        self.seq = seq
        self.target_dim = target_dim
        self.aux_layers = aux_layers
        self.bad_layers = bad_layers

    def generate_features(self, tasks, *, capture):
        out = []
        for _ in tasks:
            feats = {
                "input_ids": torch.zeros(1, self.seq, dtype=torch.long),
                "attention_mask": torch.ones(1, self.seq, dtype=torch.long),
                "loss_mask": torch.ones(1, self.seq, dtype=torch.long),
                "hidden_state": torch.randn(1, self.seq, self.aux_layers * H),
                "target": torch.randn(1, self.seq, self.target_dim),
            }
            ids = (1, 2, 99) if self.bad_layers else (1, 2, 3)
            feats["__aux_layer_ids__"] = ids
            out.append(feats)
        return out


class TestRolloutWorker(unittest.TestCase):
    def test_run_once_commits_refs(self):
        ctrl = DataFlowController("run")
        ctrl.ingest_prompts([{"payload": {"text": "a"}}, {"payload": {"text": "b"}}])
        store = LocalFeatureStore("st")
        w = RolloutWorker(ctrl, store, FakeSource(), _capture(), run_id="run")
        w.start()
        refs = w.run_once(max_tasks=8)
        self.assertEqual(len(refs), 2)
        self.assertEqual(ctrl.status()["samples_committed"], 2)
        self.assertEqual(ctrl.sample_queue.depth(), 2)
        # features landed in the store, retrievable by ref
        out, _ = store.get(refs[0])
        self.assertEqual(set(out), {"input_ids", "attention_mask", "loss_mask", "hidden_state", "target"})
        self.assertEqual(w.health()["state"], "ready")

    def test_capture_mismatch_fails_loudly_and_commits_nothing(self):
        ctrl = DataFlowController("run")
        ctrl.ingest_prompts([{"payload": {"text": "a"}}])
        store = LocalFeatureStore("st")
        w = RolloutWorker(ctrl, store, FakeSource(bad_layers=True), _capture(), run_id="run")
        w.start()
        from specforge.runtime.inference.capture import CaptureMismatchError

        with self.assertRaises(CaptureMismatchError):
            w.run_once(max_tasks=8)
        self.assertEqual(ctrl.status()["samples_committed"], 0)
        self.assertEqual(ctrl.sample_queue.depth(), 0)

    def test_no_tasks_returns_empty(self):
        ctrl = DataFlowController("run")
        store = LocalFeatureStore("st")
        w = RolloutWorker(ctrl, store, FakeSource(), _capture(), run_id="run")
        w.start()
        self.assertEqual(w.run_once(8), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
