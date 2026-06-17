# coding=utf-8
"""FeatureDataLoader: queue + store -> TrainBatch, with injected transform/collate (CPU)."""

import os
import tempfile
import unittest
from dataclasses import replace

import torch

from specforge.runtime.control_plane.controller import DataFlowController
from specforge.runtime.data_plane.feature_dataloader import FeatureDataLoader
from specforge.runtime.data_plane.feature_store import LocalFeatureStore
from specforge.runtime.data_plane.offline_reader import OfflineManifestReader
from specforge.runtime.data_plane.sample_ref_queue import SampleRefQueue


def _offline_eagle3_process_data(raw):
    """Mirror of OfflineEagle3Dataset.process_data (the aux<->target swap)."""
    max_len = 2048
    hidden_state = raw["aux_hidden_state"].squeeze(0)[:max_len][None, :]
    target = raw["hidden_state"].squeeze(0)[:max_len][None, :]
    input_ids = raw["input_ids"][:max_len][None, :]
    loss_mask = raw["loss_mask"][:max_len][None, :].clone()
    loss_mask[0, -1] = 0
    return {
        "attention_mask": torch.ones_like(loss_mask, dtype=torch.long),
        "loss_mask": loss_mask,
        "target": target,
        "hidden_state": hidden_state,
        "input_ids": input_ids,
    }


def _simple_collate(features):
    keys = features[0].keys()
    return {k: torch.cat([f[k] for f in features], dim=0) for k in keys}


class TestFeatureDataLoader(unittest.TestCase):
    def _write_offline_files(self, d, n=4, seq=8, h=4, aux=12):
        for i in range(n):
            torch.save(
                {
                    "input_ids": torch.arange(seq) + i,
                    "loss_mask": torch.ones(seq, dtype=torch.long),
                    "hidden_state": torch.randn(1, seq, h),
                    "aux_hidden_state": torch.randn(1, seq, aux),
                },
                os.path.join(d, f"{i:03d}.ckpt"),
            )

    def test_offline_loader_emits_trainbatch(self):
        with tempfile.TemporaryDirectory() as d:
            self._write_offline_files(d, n=4)
            ctrl = DataFlowController("run")
            ctrl.enqueue_offline_refs(OfflineManifestReader(d, run_id="run").read())
            store = LocalFeatureStore("st")
            loader = FeatureDataLoader(
                store,
                ctrl.sample_queue,
                batch_size=2,
                collate_fn=_simple_collate,
                per_sample_transform=_offline_eagle3_process_data,
            )
            batches = list(loader)
            self.assertEqual(len(batches), 2)  # 4 samples / batch 2
            b = batches[0]
            self.assertEqual(len(b.sample_ids), 2)
            self.assertEqual(b.tensors["input_ids"].shape, (2, 8))
            self.assertEqual(b.tensors["target"].shape, (2, 8, 4))
            self.assertEqual(b.tensors["hidden_state"].shape, (2, 8, 12))
            # aux<->target swap preserved
            self.assertEqual(b.metadata["target_repr"], "hidden_state")
            # all refs acked
            self.assertEqual(ctrl.sample_queue.in_flight(), 0)
            self.assertEqual(ctrl.sample_queue.depth(), 0)

    def test_drop_last(self):
        with tempfile.TemporaryDirectory() as d:
            self._write_offline_files(d, n=3)
            q = SampleRefQueue()
            q.put(OfflineManifestReader(d, run_id="run").read())
            store = LocalFeatureStore("st")
            loader = FeatureDataLoader(
                store, q, batch_size=2, collate_fn=_simple_collate,
                per_sample_transform=_offline_eagle3_process_data, drop_last=True,
            )
            batches = list(loader)
            self.assertEqual(len(batches), 1)  # 3 samples, drop the trailing 1

    def test_mixed_target_repr_fails_and_releases_refs(self):
        with tempfile.TemporaryDirectory() as d:
            self._write_offline_files(d, n=2)
            refs = OfflineManifestReader(d, run_id="run").read()
            refs[1] = replace(
                refs[1],
                metadata={**refs[1].metadata, "target_repr": "logits"},
            )
            q = SampleRefQueue()
            q.put(refs)
            loader = FeatureDataLoader(
                LocalFeatureStore("st"),
                q,
                batch_size=2,
                collate_fn=_simple_collate,
                per_sample_transform=_offline_eagle3_process_data,
            )
            with self.assertRaises(ValueError):
                list(loader)
            self.assertEqual(q.in_flight(), 0)
            self.assertEqual(q.depth(), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
