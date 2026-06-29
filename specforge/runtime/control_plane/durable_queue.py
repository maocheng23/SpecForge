# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""SQLiteSampleRefQueue: a durable, cross-process SampleRefQueue (online O1.1).

The in-process ``SampleRefQueue`` couples producer and consumer to one OS process
(a ``threading.Condition`` over in-memory dicts). A disaggregated *run* puts the
rollout pool and the trainer pool in **separate processes**, so the pending/leased
state must be shared across processes — not just threads. This backend keeps the
exact ``put/get/ack/fail/depth/in_flight`` contract of ``SampleRefQueue`` (it is a
drop-in for ``DataFlowController(sample_queue=...)``) but holds the queue in a
SQLite table, so a producer process committing refs and a consumer process leasing
them over a shared DB file see one queue.

Design notes:

* **Metadata-only**, like the in-process queue: only ``SampleRef`` JSON rows are
  stored, never tensors (``assert_no_tensors`` on ``put``). Tensors travel through
  the FeatureStore data plane.
* **Lease race is closed by ``BEGIN IMMEDIATE``**: the read-then-lease in ``get``
  runs in a write transaction, so two consumer processes can never lease the same
  row (the classic SELECT-then-UPDATE race). WAL + ``busy_timeout`` let a second
  process wait out the first's lock instead of erroring.
* **Wall-clock leases**: lease age uses ``clock()`` (default ``time.time``), not
  ``monotonic`` — monotonic clocks are not comparable across processes.
* **Blocking ``get`` polls**: a cross-process condition variable does not exist, so
  a blocking ``get`` polls at ``poll_interval_s`` while its shard is empty. A
  partitioned ``get`` never blocks while the global pool is non-empty (matches the
  in-process queue: don't starve one shard behind another).
"""

from __future__ import annotations

import sqlite3
import threading
import time
from typing import Callable, List, Optional, Tuple

from specforge.runtime.contracts import SampleRef, assert_no_tensors
from specforge.runtime.control_plane.metadata_store import (
    sample_ref_from_json,
    sample_ref_to_json,
)
from specforge.runtime.data_plane.sample_ref_queue import dp_partition

_POLL_INTERVAL_S = 0.05


class SQLiteSampleRefQueue:
    """A durable, cross-process drop-in for :class:`SampleRefQueue`."""

    def __init__(
        self,
        path: str,
        *,
        lease_timeout_s: Optional[float] = None,
        clock: Callable[[], float] = time.time,
        poll_interval_s: float = _POLL_INTERVAL_S,
    ) -> None:
        self.path = path
        self.lease_timeout_s = lease_timeout_s
        self._clock = clock
        self._poll = poll_interval_s
        # autocommit (isolation_level=None) so transactions are driven explicitly
        # with BEGIN IMMEDIATE for the lease race; one connection per instance.
        self._conn = sqlite3.connect(
            path, check_same_thread=False, isolation_level=None
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")  # wait out a writer's lock
        self._lock = threading.RLock()  # serialize this instance's own threads
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS queue ("
                "  sample_id TEXT PRIMARY KEY,"
                "  ref_json  TEXT NOT NULL,"
                "  state     TEXT NOT NULL DEFAULT 'pending',"  # pending | leased
                "  leased_at REAL,"
                "  seq       INTEGER NOT NULL"
                ")"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS queue_state_seq ON queue(state, seq)"
            )
            # durable monotonic FIFO counter (survives restart, shared by all procs)
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS queue_seq (k INTEGER PRIMARY KEY, v INTEGER)"
            )
            self._conn.execute("INSERT OR IGNORE INTO queue_seq (k, v) VALUES (0, 0)")

    # -- internals ---------------------------------------------------------
    def _next_seq_locked(self) -> int:
        """Bump and return the FIFO counter. Caller holds the write transaction."""
        self._conn.execute("UPDATE queue_seq SET v = v + 1 WHERE k = 0")
        return self._conn.execute("SELECT v FROM queue_seq WHERE k = 0").fetchone()[0]

    def _reclaim_expired_locked(self, now: float) -> None:
        """Return timed-out leases to pending. Caller holds the write transaction."""
        if self.lease_timeout_s is None:
            return
        self._conn.execute(
            "UPDATE queue SET state='pending', leased_at=NULL "
            "WHERE state='leased' AND leased_at IS NOT NULL AND leased_at < ?",
            (now - self.lease_timeout_s,),
        )

    # -- write -------------------------------------------------------------
    def put(
        self, refs: List[SampleRef], *, partition_key: Optional[str] = None
    ) -> None:
        # partition_key reserves the producer-side routing seam (accepted+ignored),
        # exactly as in the in-process queue.
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                for ref in refs:
                    assert_no_tensors(ref)  # structural no-tensor guard
                    # Idempotent on sample_id (at-least-once): skip if already
                    # present in either state.
                    if self._conn.execute(
                        "SELECT 1 FROM queue WHERE sample_id = ?", (ref.sample_id,)
                    ).fetchone():
                        continue
                    self._conn.execute(
                        "INSERT INTO queue (sample_id, ref_json, state, leased_at, seq)"
                        " VALUES (?, ?, 'pending', NULL, ?)",
                        (
                            ref.sample_id,
                            sample_ref_to_json(ref),
                            self._next_seq_locked(),
                        ),
                    )
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise

    # -- read --------------------------------------------------------------
    def get(
        self,
        max_refs: int,
        timeout_s: Optional[float] = None,
        *,
        partition_key: Optional[str] = None,
        partition: Optional[Tuple[int, int]] = None,
    ) -> List[SampleRef]:
        """Lease up to ``max_refs`` pending refs; block until some arrive or
        ``timeout_s`` elapses. See class docstring for partition/blocking rules."""
        deadline = None if timeout_s is None else self._clock() + timeout_s
        while True:
            leased = self._lease_once(max_refs, partition)
            if leased:
                return leased
            # Partitioned lease never blocks while the global pool is non-empty —
            # the matching refs simply belong to another shard (don't starve it).
            if partition is not None and self.depth() > 0:
                return leased  # [] — this shard is empty but the pool is not
            if deadline is None or self._clock() >= deadline:
                return leased  # [] — timed out / non-blocking
            time.sleep(self._poll)

    def _lease_once(
        self, max_refs: int, partition: Optional[Tuple[int, int]]
    ) -> List[SampleRef]:
        now = self._clock()
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._reclaim_expired_locked(now)
                rows = self._conn.execute(
                    "SELECT sample_id, ref_json FROM queue "
                    "WHERE state='pending' ORDER BY seq"
                ).fetchall()
                chosen: List[Tuple[str, str]] = []
                for sid, ref_json in rows:
                    if len(chosen) >= max_refs:
                        break
                    if partition is not None:
                        index, num_partitions = partition
                        if dp_partition(sid, num_partitions) != index:
                            continue
                    chosen.append((sid, ref_json))
                for sid, _ in chosen:
                    self._conn.execute(
                        "UPDATE queue SET state='leased', leased_at=? WHERE sample_id=?",
                        (now, sid),
                    )
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
        return [sample_ref_from_json(rj) for _, rj in chosen]

    # -- lease resolution --------------------------------------------------
    def ack(self, refs: List[SampleRef]) -> None:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                for ref in refs:
                    self._conn.execute(
                        "DELETE FROM queue WHERE sample_id = ?", (ref.sample_id,)
                    )  # idempotent
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise

    def fail(self, refs: List[SampleRef], reason: str, retryable: bool) -> None:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                for ref in refs:
                    if retryable:
                        # Back to the pending tail: a fresh seq sends it to the end.
                        self._conn.execute(
                            "UPDATE queue SET state='pending', leased_at=NULL, seq=? "
                            "WHERE sample_id=?",
                            (self._next_seq_locked(), ref.sample_id),
                        )
                    else:
                        self._conn.execute(
                            "DELETE FROM queue WHERE sample_id = ?", (ref.sample_id,)
                        )
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise

    # -- observability -----------------------------------------------------
    def depth(self) -> int:
        with self._lock:
            return self._conn.execute(
                "SELECT COUNT(*) FROM queue WHERE state='pending'"
            ).fetchone()[0]

    def in_flight(self) -> int:
        with self._lock:
            return self._conn.execute(
                "SELECT COUNT(*) FROM queue WHERE state='leased'"
            ).fetchone()[0]

    def close(self) -> None:
        with self._lock:
            self._conn.close()


__all__ = ["SQLiteSampleRefQueue"]
