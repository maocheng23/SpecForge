# coding=utf-8
"""SQLiteSampleRefQueue: durable, cross-process lease/ack contract (online O1.1).

Two queue *instances* over one DB file stand in for two processes (producer pool /
trainer pool), the way the disagg tests use two FeatureStore instances over one
backend. The point of these tests: leasing, ack, fail, partitioning, lease-timeout
reclaim, and durability all hold ACROSS instances, not just across threads.
"""

import tempfile
import unittest

from specforge.runtime.contracts import SampleRef
from specforge.runtime.control_plane.controller import DataFlowController
from specforge.runtime.control_plane.durable_queue import SQLiteSampleRefQueue
from specforge.runtime.control_plane.metadata_store import SQLiteMetadataStore
from specforge.runtime.data_plane.sample_ref_queue import dp_partition


def _ref(sid):
    return SampleRef(
        sample_id=sid,
        run_id="r",
        source_task_id=None,
        feature_store_uri=f"mooncake://st/{sid}",
        feature_keys={"x": f"{sid}/x"},
        feature_specs={},
        strategy="eagle3",
    )


class _FakeClock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


class TestDurableQueueCrossInstance(unittest.TestCase):
    def setUp(self):
        self.path = tempfile.mkdtemp() + "/queue.db"

    def test_commit_in_one_instance_lease_ack_in_another(self):
        producer = SQLiteSampleRefQueue(self.path)
        producer.put([_ref("s0"), _ref("s1")])

        consumer = SQLiteSampleRefQueue(self.path)  # separate instance = process
        leased = consumer.get(10)
        self.assertEqual({r.sample_id for r in leased}, {"s0", "s1"})
        self.assertEqual(consumer.in_flight(), 2)
        consumer.ack(leased)
        self.assertEqual(consumer.depth(), 0)
        self.assertEqual(producer.depth(), 0)  # gone for everyone (shared backend)

    def test_two_consumers_never_double_lease(self):
        producer = SQLiteSampleRefQueue(self.path)
        producer.put([_ref(f"s{i}") for i in range(50)])
        a = SQLiteSampleRefQueue(self.path)
        b = SQLiteSampleRefQueue(self.path)
        ga = {r.sample_id for r in a.get(30)}
        gb = {r.sample_id for r in b.get(30)}
        self.assertEqual(ga & gb, set())  # disjoint
        self.assertEqual(ga | gb, {f"s{i}" for i in range(50)})  # complete
        self.assertEqual(a.in_flight(), 50)
        self.assertEqual(a.depth(), 0)

    def test_partition_shards_are_disjoint_and_complete(self):
        producer = SQLiteSampleRefQueue(self.path)
        producer.put([_ref(f"s{i}") for i in range(40)])
        shard0 = SQLiteSampleRefQueue(self.path).get(100, partition=(0, 2))
        shard1 = SQLiteSampleRefQueue(self.path).get(100, partition=(1, 2))
        ids0 = {r.sample_id for r in shard0}
        ids1 = {r.sample_id for r in shard1}
        self.assertEqual(ids0 & ids1, set())
        self.assertEqual(ids0 | ids1, {f"s{i}" for i in range(40)})
        for sid in ids0:
            self.assertEqual(dp_partition(sid, 2), 0)

    def test_reshard_to_wider_layout_consumes_remainder_once(self):
        q = SQLiteSampleRefQueue(self.path)
        all_ids = {f"s{i}" for i in range(60)}
        q.put([_ref(s) for s in all_ids])
        leased = {r.sample_id for r in q.get(100, partition=(0, 2))}  # width-2 shard 0
        for idx in range(3):  # reshard remainder to width 3
            new = {r.sample_id for r in q.get(100, partition=(idx, 3))}
            self.assertEqual(new & leased, set())  # never re-leased
            leased |= new
        self.assertEqual(leased, all_ids)  # nothing lost
        self.assertEqual(q.depth(), 0)

    def test_partitioned_get_does_not_block_on_empty_shard(self):
        q = SQLiteSampleRefQueue(self.path)
        q.put([_ref("s0")])
        p = dp_partition("s0", 4)
        empty = (p + 1) % 4
        # an empty shard returns immediately even with a timeout (pool non-empty)
        self.assertEqual(q.get(10, timeout_s=0.2, partition=(empty, 4)), [])
        self.assertEqual(len(q.get(10, partition=(p, 4))), 1)

    def test_fail_retryable_requeues_nonretryable_drops(self):
        q = SQLiteSampleRefQueue(self.path)
        q.put([_ref("s0"), _ref("s1")])
        leased = q.get(10)
        q.fail([_ref("s0")], "boom", retryable=True)
        q.fail([_ref("s1")], "boom", retryable=False)
        self.assertEqual(q.depth(), 1)  # s0 back to pending
        self.assertEqual(q.in_flight(), 0)
        again = q.get(10)
        self.assertEqual({r.sample_id for r in again}, {"s0"})  # s1 dropped

    def test_lease_timeout_reclaim(self):
        clock = _FakeClock()
        q = SQLiteSampleRefQueue(self.path, lease_timeout_s=10.0, clock=clock)
        q.put([_ref("s0")])
        q.get(10)  # leased at t=1000
        self.assertEqual(q.in_flight(), 1)
        clock.advance(50.0)  # past the lease timeout
        reclaimed = q.get(10)  # reclaims the expired lease, re-leases it
        self.assertEqual({r.sample_id for r in reclaimed}, {"s0"})

    def test_idempotent_put_on_sample_id(self):
        q = SQLiteSampleRefQueue(self.path)
        q.put([_ref("s0")])
        q.put([_ref("s0")])  # duplicate sample_id -> ignored
        self.assertEqual(q.depth(), 1)
        q2 = SQLiteSampleRefQueue(self.path)
        q2.put([_ref("s0")])  # duplicate across instances too
        self.assertEqual(q2.depth(), 1)

    def test_blocking_get_times_out_when_globally_empty(self):
        q = SQLiteSampleRefQueue(self.path)  # real wall clock
        self.assertEqual(q.get(10, timeout_s=0.1), [])

    def test_durable_across_reopen(self):
        q = SQLiteSampleRefQueue(self.path)
        q.put([_ref("s0"), _ref("s1")])
        q.close()  # process exits
        reopened = SQLiteSampleRefQueue(self.path)  # fresh process attaches
        self.assertEqual(reopened.depth(), 2)
        self.assertEqual({r.sample_id for r in reopened.get(10)}, {"s0", "s1"})


class TestDurableQueueWithController(unittest.TestCase):
    """The O1.1 thesis: a producer controller and a consumer controller in
    separate processes share commit/lease/ack via a shared queue + metadata
    store (both SQLite over the same files), with NO controller code change."""

    def setUp(self):
        d = tempfile.mkdtemp()
        self.qpath = d + "/queue.db"
        self.mpath = d + "/meta.db"

    def _controller(self):
        return DataFlowController(
            "run",
            sample_queue=SQLiteSampleRefQueue(self.qpath),
            metadata_store=SQLiteMetadataStore(self.mpath),
        )

    def test_cross_process_commit_lease_ack_then_reconcile(self):
        producer = self._controller()
        producer.enqueue_offline_refs([_ref(f"s{i}") for i in range(5)])

        consumer = self._controller()  # separate instance = separate process
        lease = consumer.train_lease("t0")
        refs = lease.get(10)
        self.assertEqual({r.sample_id for r in refs}, {f"s{i}" for i in range(5)})
        lease.ack(refs, global_step=1, optimizer_durable=True)

        self.assertEqual(consumer.sample_queue.depth(), 0)
        self.assertEqual(consumer.sample_queue.in_flight(), 0)

        # a fresh controller reconciles release state from the shared durable marker
        report = self._controller().reconcile_on_restart()
        self.assertEqual(set(report["released"]), {f"s{i}" for i in range(5)})
        self.assertEqual(report["requeued"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
