# coding=utf-8
"""LocalFeatureStore: atomic put, get, idempotent release, abort, file mode (CPU)."""

import os
import tempfile
import unittest

import torch

from specforge.runtime.data_plane.feature_store import LocalFeatureStore
from specforge.runtime.data_plane.offline_reader import OfflineManifestReader


class TestLocalFeatureStore(unittest.TestCase):
    def test_put_returns_ref_with_no_tensors(self):
        store = LocalFeatureStore("st")
        tensors = {
            "input_ids": torch.arange(8).view(1, 8),
            "hidden_state": torch.randn(1, 8, 4),
        }
        ref = store.put(tensors, sample_id="s0", metadata={"run_id": "r", "num_tokens": 8})
        self.assertEqual(ref.sample_id, "s0")
        self.assertTrue(ref.feature_store_uri.startswith("mem://"))
        self.assertEqual(set(ref.feature_specs), {"input_ids", "hidden_state"})
        self.assertEqual(ref.feature_specs["hidden_state"].shape, (1, 8, 4))
        self.assertGreater(ref.estimated_bytes, 0)

    def test_get_returns_tensors_and_handle(self):
        store = LocalFeatureStore("st")
        t = torch.randn(1, 4, 2)
        ref = store.put({"x": t}, sample_id="s0", metadata={})
        out, handle = store.get(ref)
        self.assertTrue(torch.equal(out["x"], t))
        self.assertEqual(handle.sample_id, "s0")

    def test_release_idempotent_and_stale_safe(self):
        store = LocalFeatureStore("st")
        ref = store.put({"x": torch.randn(1, 4)}, sample_id="s0", metadata={})
        _, h = store.get(ref)
        store.release(h)
        store.release(h)  # idempotent: must not raise
        # re-put bumps generation; old handle release is a no-op
        ref2 = store.put({"x": torch.randn(1, 4)}, sample_id="s0", metadata={})
        _, h2 = store.get(ref2)
        store.release(h)  # stale generation -> no-op
        out, _ = store.get(ref2)
        self.assertIn("x", out)
        _ = h2

    def test_abort_evicts(self):
        store = LocalFeatureStore("st")
        ref = store.put({"x": torch.randn(1, 4)}, sample_id="s0", metadata={})
        store.abort("s0", reason="test")
        with self.assertRaises(KeyError):
            store.get(ref)

    def test_health(self):
        store = LocalFeatureStore("st")
        store.put({"x": torch.randn(1, 4)}, sample_id="s0", metadata={})
        h = store.health()
        self.assertEqual(h["resident_samples"], 1)
        self.assertGreater(h["resident_bytes"], 0)

    def test_estimate_bytes(self):
        store = LocalFeatureStore("st")
        ref = store.put({"x": torch.zeros(1, 10, dtype=torch.float32)}, sample_id="s0", metadata={})
        self.assertEqual(store.estimate_bytes(ref.feature_specs), 10 * 4)

    def test_disk_dump_tap(self):
        with tempfile.TemporaryDirectory() as d:
            store = LocalFeatureStore("st", dump_dir=d)
            store.put({"x": torch.randn(1, 4)}, sample_id="s0", metadata={})
            self.assertTrue(os.path.exists(os.path.join(d, "s0.ckpt")))

    def test_file_mode_get_matches_offline_format(self):
        # write an offline-style .ckpt and read it back through the store + reader
        with tempfile.TemporaryDirectory() as d:
            raw = {
                "input_ids": torch.arange(8),
                "loss_mask": torch.ones(8, dtype=torch.long),
                "hidden_state": torch.randn(1, 8, 4),
                "aux_hidden_state": torch.randn(1, 8, 12),
            }
            torch.save(raw, os.path.join(d, "000.ckpt"))
            refs = OfflineManifestReader(d, run_id="off").read()
            self.assertEqual(len(refs), 1)
            self.assertTrue(refs[0].feature_store_uri.startswith("file://"))
            self.assertEqual(set(refs[0].feature_specs), set(raw))
            self.assertEqual(refs[0].feature_specs["hidden_state"].dtype, "float32")
            self.assertEqual(refs[0].num_tokens, 8)
            store = LocalFeatureStore("st")
            out, handle = store.get(refs[0])
            self.assertEqual(set(out), set(raw))
            self.assertTrue(torch.equal(out["aux_hidden_state"], raw["aux_hidden_state"]))
            store.release(handle)

    def test_offline_reader_rejects_missing_required_key(self):
        with tempfile.TemporaryDirectory() as d:
            torch.save({"input_ids": torch.arange(4)}, os.path.join(d, "bad.ckpt"))
            with self.assertRaises(KeyError):
                OfflineManifestReader(d, run_id="off").read()


if __name__ == "__main__":
    unittest.main(verbosity=2)
