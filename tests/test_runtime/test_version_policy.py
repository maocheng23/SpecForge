# coding=utf-8
"""M7: published-weight lifecycle, hot update, two-axis staleness, accept-length.

CPU-only (metadata). The headline gate (test_m7_exit_gate) walks the whole loop:
publish a draft version, hot-update the rollout pool with no restart, see samples
stamp the new version, watch the drift monitor, and roll a regressed draft back.
"""

import os
import tempfile
import unittest

from specforge.runtime.contracts import WeightVersion
from specforge.runtime.control_plane.controller import DataFlowController
from specforge.runtime.control_plane.metadata_store import SQLiteMetadataStore
from specforge.runtime.control_plane.version_policy import (
    DriftMonitor,
    StalenessPolicy,
    WeightPublisher,
    WeightRegistry,
)
from specforge.runtime.inference.hot_update import RolloutDraftServer


def _wv(n, step, parent=None, accept=None):
    metrics = {"accept_length": accept} if accept is not None else {}
    return WeightVersion(
        version_id=f"v{n}",
        draft_weight_version=f"draft-{n}",
        target_model_version="target-A",
        global_step=step,
        parent_version_id=parent,
        metrics=metrics,
    )


class TestWeightRegistry(unittest.TestCase):
    def test_publish_activate_latest(self):
        reg = WeightRegistry()
        reg.publish(_wv(1, 10))
        reg.publish(_wv(2, 20))
        self.assertEqual(reg.latest().version_id, "v2")
        reg.activate("v1")
        self.assertEqual(reg.active().version_id, "v1")
        reg.activate("v2")
        self.assertEqual(reg.active().version_id, "v2")
        self.assertEqual(reg.get("v1").status, "candidate")  # demoted

    def test_publish_idempotent(self):
        reg = WeightRegistry()
        reg.publish(_wv(1, 10))
        reg.publish(_wv(1, 10))
        self.assertEqual(len(reg.history()), 1)

    def test_draft_lag(self):
        reg = WeightRegistry()
        for i in range(1, 4):
            reg.publish(_wv(i, i * 10))
        self.assertEqual(reg.draft_lag("draft-3"), 0)  # newest
        self.assertEqual(reg.draft_lag("draft-2"), 1)
        self.assertEqual(reg.draft_lag("draft-1"), 2)
        self.assertIsNone(reg.draft_lag("draft-unknown"))

    def test_accept_length_and_rollback(self):
        reg = WeightRegistry()
        reg.publish(_wv(1, 10, accept=3.5))
        reg.activate("v1")
        reg.publish(_wv(2, 20, parent="v1"))
        reg.activate("v2")
        reg.record_accept_length("v2", 3.0)  # regressed vs v1's 3.5
        restored = reg.maybe_rollback(regression_tol=0.1)
        self.assertIsNotNone(restored)
        self.assertEqual(reg.active().version_id, "v1")  # rolled back
        self.assertEqual(reg.get("v2").status, "rolled_back")

    def test_no_rollback_when_improved(self):
        reg = WeightRegistry()
        reg.publish(_wv(1, 10, accept=3.0))
        reg.activate("v1")
        reg.publish(_wv(2, 20, parent="v1", accept=3.8))
        reg.activate("v2")
        self.assertIsNone(reg.maybe_rollback())
        self.assertEqual(reg.active().version_id, "v2")


class TestWeightPublisherHotUpdate(unittest.TestCase):
    def test_publish_hot_updates_pool_and_activates(self):
        reg = WeightRegistry()
        pub = WeightPublisher(reg)
        pool = [RolloutDraftServer(f"w{i}") for i in range(3)]
        result = pub.publish(_wv(1, 10), pool)
        self.assertEqual(len(result.acked), 3)
        self.assertEqual(result.failed, [])
        self.assertTrue(result.activated)
        self.assertEqual(reg.active().version_id, "v1")
        # every pool member is now serving the new draft (no restart)
        for s in pool:
            self.assertEqual(s.active_draft_version(), "draft-1")

    def test_failed_member_blocks_activation(self):
        reg = WeightRegistry()
        pub = WeightPublisher(reg)
        good = RolloutDraftServer("w0")
        bad = RolloutDraftServer("w1", apply_fn=lambda v: False)  # fails to apply
        result = pub.publish(_wv(1, 10), [good, bad])
        self.assertEqual(result.acked, ["w0"])
        self.assertEqual(result.failed, ["w1"])
        self.assertFalse(result.activated)
        self.assertIsNone(reg.active())  # not activated with a half-updated pool


class TestStalenessPolicy(unittest.TestCase):
    def setUp(self):
        self.reg = WeightRegistry()
        for i in range(1, 4):
            self.reg.publish(_wv(i, i * 10))

    def test_draft_axis(self):
        pol = StalenessPolicy(max_draft_lag=1, require_target_match=False)
        fresh = pol.assess(
            sample_draft_version="draft-3",
            sample_target_version="target-A",
            registry=self.reg,
            current_target_version="target-A",
        )
        self.assertTrue(fresh.accept)
        stale = pol.assess(
            sample_draft_version="draft-1",
            sample_target_version="target-A",
            registry=self.reg,
            current_target_version="target-A",
        )
        self.assertFalse(stale.accept)  # lag 2 > max 1
        self.assertEqual(stale.draft_lag, 2)

    def test_target_axis(self):
        pol = StalenessPolicy(require_target_match=True)
        a = pol.assess(
            sample_draft_version="draft-3",
            sample_target_version="target-OLD",
            registry=self.reg,
            current_target_version="target-A",
        )
        self.assertFalse(a.accept)
        self.assertTrue(a.target_stale)
        self.assertIn("target_version_mismatch", a.reasons)

    def test_unknown_draft_is_maximally_stale(self):
        pol = StalenessPolicy(max_draft_lag=5)
        a = pol.assess(
            sample_draft_version="draft-ghost",
            sample_target_version="target-A",
            registry=self.reg,
            current_target_version="target-A",
        )
        self.assertFalse(a.accept)
        self.assertIn("unknown_draft_version", a.reasons)


class TestDriftMonitor(unittest.TestCase):
    def test_emits_when_lag_grows(self):
        m = DriftMonitor(window=10)
        for _ in range(10):
            m.observe(0)
        self.assertFalse(m.drifting(mean_lag_threshold=1.0))
        for _ in range(10):
            m.observe(5)  # pool fell behind
        snap = m.snapshot()
        self.assertEqual(snap["mean_lag"], 5.0)
        self.assertTrue(m.drifting(mean_lag_threshold=1.0))

    def test_counts_unknown_versions(self):
        m = DriftMonitor()
        m.observe(0)
        m.observe(None)
        self.assertEqual(m.snapshot()["unknown_version"], 1)
        self.assertEqual(m.snapshot()["samples"], 2)


class TestControllerWeightPublishing(unittest.TestCase):
    def test_controller_publishes_and_durably_recovers(self):
        d = tempfile.mkdtemp()
        path = os.path.join(d, "meta.db")
        store = SQLiteMetadataStore(path)
        ctrl = DataFlowController("run", metadata_store=store)
        pool = [RolloutDraftServer("w0")]
        result = ctrl.publish_weight_version(_wv(1, 10), rollouts=pool)
        self.assertTrue(result.activated)
        self.assertEqual(ctrl.latest_weight_version().version_id, "v1")
        store.close()
        # restart: a fresh controller on the same DB recovers the registry
        store2 = SQLiteMetadataStore(path)
        ctrl2 = DataFlowController("run", metadata_store=store2)
        self.assertEqual(ctrl2.latest_weight_version().version_id, "v1")
        self.assertEqual(ctrl2.active_weight_version().version_id, "v1")
        store2.close()


class TestM7ExitGate(unittest.TestCase):
    def test_m7_exit_gate(self):
        # publish + hot-update with no restart; samples stamp the version; drift
        # monitor emits; accept-length tracked; regressed draft rolls back.
        ctrl = DataFlowController("run")
        pool = [RolloutDraftServer(f"w{i}") for i in range(2)]
        drift = DriftMonitor(window=64)

        # v1 published and hot-updated -> pool serves it without restart
        ctrl.publish_weight_version(_wv(1, 10, accept=3.4), rollouts=pool)
        ctrl.weight_registry.record_accept_length("v1", 3.4)
        sample_v = pool[0].stamp()
        self.assertEqual(sample_v, "draft-1")  # sample records observed draft version
        drift.observe(ctrl.weight_registry.draft_lag(sample_v))

        # v2 published, hot-updated, but accept-length regresses
        ctrl.publish_weight_version(_wv(2, 20, parent="v1"), rollouts=pool)
        self.assertEqual(pool[0].active_draft_version(), "draft-2")  # no restart
        ctrl.weight_registry.record_accept_length("v2", 2.9)  # regression

        # a slow sample still tagged draft-1 is now 1 version behind -> drift
        drift.observe(ctrl.weight_registry.draft_lag("draft-1"))
        self.assertGreater(drift.snapshot()["max_lag"], 0)

        # rollback the regression
        restored = ctrl.weight_registry.maybe_rollback(regression_tol=0.0)
        self.assertEqual(restored.version_id, "v1")
        self.assertEqual(ctrl.active_weight_version().version_id, "v1")


if __name__ == "__main__":
    unittest.main(verbosity=2)
