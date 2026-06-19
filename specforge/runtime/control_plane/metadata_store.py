# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""MetadataStore: the durability seam for recovery-critical control-plane state.

The controller's *recovery-critical* state — committed sample dedup and the
at-least-once durable ack transaction (``{acked sample_ids, global_step,
optimizer-durable marker}``) — lives behind this interface rather than in inline
dicts. The current implementation ships ``InMemoryMetadataStore``; a SQLite
(dev) or Redis/DB (prod) backend is then a *new subclass*, not a
method-by-method rewrite of the controller. The single durable transaction
(``record_train_ack``) is the unit a restart reconciles release state from.

Dependency-light (stdlib only) so it stays importable without torch.
"""

from __future__ import annotations

import abc
import dataclasses
import json
import sqlite3
import threading
from typing import Any, Dict, List, Optional, Set

from specforge.runtime.contracts import FeatureSpec, SampleRef, WeightVersion


class MetadataStore(abc.ABC):
    # -- sample commit / dedup (at-least-once: idempotent on sample_id) ----
    @abc.abstractmethod
    def commit_sample(self, ref: SampleRef) -> bool:
        """Record a committed sample. Returns True if new, False if duplicate."""

    @abc.abstractmethod
    def is_committed(self, sample_id: str) -> bool: ...

    @abc.abstractmethod
    def get_committed(self, sample_id: str) -> Optional[SampleRef]: ...

    @abc.abstractmethod
    def committed_count(self) -> int: ...

    @abc.abstractmethod
    def all_committed_ids(self) -> List[str]:
        """Every committed sample_id. Restart reconciliation iterates this."""

    # -- durable ack transaction -------------------------------------------
    @abc.abstractmethod
    def record_train_ack(
        self,
        sample_ids: List[str],
        *,
        global_step: Optional[int],
        optimizer_durable: bool,
    ) -> None:
        """Commit {acked sample_ids, global_step, optimizer-durable marker} atomically.

        Release state is *derived* from this on restart — it is the single
        transaction recovery reconciles against; never split it.
        """

    @abc.abstractmethod
    def durable_marker(self) -> Dict[str, Any]:
        """{acked: set[str], global_step: int|None, optimizer_durable: bool}."""

    # NOTE: a weight-version registry (put/latest/count) is not yet implemented;
    # it belongs with the rest of the published-weight lifecycle.


class InMemoryMetadataStore(MetadataStore):
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._committed: Dict[str, SampleRef] = {}
        self._acked: Set[str] = set()
        self._global_step: Optional[int] = None
        self._optimizer_durable: bool = False
        self._weights: Dict[str, WeightVersion] = {}
        self._weight_order: List[str] = []

    def commit_sample(self, ref: SampleRef) -> bool:
        with self._lock:
            if ref.sample_id in self._committed:
                return False
            self._committed[ref.sample_id] = ref
            return True

    def is_committed(self, sample_id: str) -> bool:
        with self._lock:
            return sample_id in self._committed

    def get_committed(self, sample_id: str) -> Optional[SampleRef]:
        with self._lock:
            return self._committed.get(sample_id)

    def committed_count(self) -> int:
        with self._lock:
            return len(self._committed)

    def all_committed_ids(self) -> List[str]:
        with self._lock:
            return list(self._committed.keys())

    def record_train_ack(
        self,
        sample_ids: List[str],
        *,
        global_step: Optional[int],
        optimizer_durable: bool,
    ) -> None:
        # one atomic update of {acked ids, global_step, optimizer marker}
        with self._lock:
            self._acked.update(sample_ids)
            if global_step is not None:
                self._global_step = global_step
            self._optimizer_durable = optimizer_durable

    def durable_marker(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "acked": set(self._acked),
                "global_step": self._global_step,
                "optimizer_durable": self._optimizer_durable,
            }

    # -- weight-version registry (M7) --------------------------------------
    def put_weight_version(self, wv: WeightVersion) -> None:
        with self._lock:
            if wv.version_id not in self._weights:
                self._weight_order.append(wv.version_id)
            self._weights[wv.version_id] = wv  # upsert (status/metric updates)

    def get_weight_version(self, version_id: str) -> Optional[WeightVersion]:
        with self._lock:
            return self._weights.get(version_id)

    def all_weight_versions(self) -> List[WeightVersion]:
        with self._lock:
            return [self._weights[v] for v in self._weight_order]

    def latest_weight_version(self) -> Optional[WeightVersion]:
        with self._lock:
            return self._weights[self._weight_order[-1]] if self._weight_order else None


# ---------------------------------------------------------------------------
# Record <-> JSON (the metadata-store payloads that need persisting)
# ---------------------------------------------------------------------------
def sample_ref_to_json(ref: SampleRef) -> str:
    # asdict() recurses into the nested FeatureSpec dataclasses + dicts; tuples
    # (FeatureSpec.shape) degrade to lists, which from_json restores.
    return json.dumps(dataclasses.asdict(ref))


def sample_ref_from_json(blob: str) -> SampleRef:
    d = json.loads(blob)
    specs = {
        name: FeatureSpec(
            name=s["name"],
            shape=tuple(s["shape"]),
            dtype=s["dtype"],
            device_hint=s.get("device_hint"),
            required=s.get("required", True),
            target_repr=s.get("target_repr"),
            target_meta=s.get("target_meta", {}),
        )
        for name, s in d.get("feature_specs", {}).items()
    }
    return SampleRef(
        sample_id=d["sample_id"],
        run_id=d["run_id"],
        source_task_id=d.get("source_task_id"),
        feature_store_uri=d["feature_store_uri"],
        feature_keys=d.get("feature_keys", {}),
        feature_specs=specs,
        strategy=d["strategy"],
        schema_version=d.get("schema_version", 1),
        target_model_version=d.get("target_model_version", "unknown"),
        draft_weight_version=d.get("draft_weight_version"),
        tokenizer_version=d.get("tokenizer_version", "unknown"),
        num_tokens=d.get("num_tokens", 0),
        estimated_bytes=d.get("estimated_bytes", 0),
        metadata=d.get("metadata", {}),
    )


def weight_version_to_json(wv: WeightVersion) -> str:
    return json.dumps(dataclasses.asdict(wv))


def weight_version_from_json(blob: str) -> WeightVersion:
    d = json.loads(blob)
    return WeightVersion(
        version_id=d["version_id"],
        draft_weight_version=d["draft_weight_version"],
        target_model_version=d["target_model_version"],
        global_step=d["global_step"],
        checkpoint_uri=d.get("checkpoint_uri"),
        parent_version_id=d.get("parent_version_id"),
        status=d.get("status", "candidate"),
        metrics=d.get("metrics", {}),
        metadata=d.get("metadata", {}),
    )


class SQLiteMetadataStore(MetadataStore):
    """Durable metadata store: committed refs + the single ack transaction.

    This is the recovery floor B4 requires: after a crash a fresh controller
    reopens the same DB file and reconstructs *exactly* the durable state — the
    committed refs and the ``{acked sample_ids, global_step, optimizer_durable}``
    marker — so it matches today's checkpoint+seek resume rather than regressing
    it. Release state is *derived* from this marker, never stored separately, so
    it can never disagree with the optimizer step (the bug B4 names).

    Same in-process API as ``InMemoryMetadataStore``; the controller does not
    know which backend it holds. SQLite is the dev/single-host durable tier;
    Redis/DB is a later subclass behind the same interface.
    """

    def __init__(self, path: str) -> None:
        # check_same_thread=False + a guard lock: one connection, serialized.
        self.path = path
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")  # durable + concurrent reads
        self._lock = threading.RLock()
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS committed "
                "(sample_id TEXT PRIMARY KEY, ref_json TEXT NOT NULL)"
            )
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS acked (sample_id TEXT PRIMARY KEY)"
            )
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS marker (k TEXT PRIMARY KEY, v TEXT)"
            )
            # weight-version registry: seq preserves publish order for staleness.
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS weight_versions ("
                "version_id TEXT PRIMARY KEY, seq INTEGER, wv_json TEXT NOT NULL)"
            )
            self._conn.commit()

    def commit_sample(self, ref: SampleRef) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "INSERT OR IGNORE INTO committed (sample_id, ref_json) VALUES (?, ?)",
                (ref.sample_id, sample_ref_to_json(ref)),
            )
            self._conn.commit()
            return cur.rowcount == 1  # 0 == duplicate (idempotent, at-least-once)

    def is_committed(self, sample_id: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM committed WHERE sample_id = ?", (sample_id,)
            ).fetchone()
            return row is not None

    def get_committed(self, sample_id: str) -> Optional[SampleRef]:
        with self._lock:
            row = self._conn.execute(
                "SELECT ref_json FROM committed WHERE sample_id = ?", (sample_id,)
            ).fetchone()
        return sample_ref_from_json(row[0]) if row else None

    def committed_count(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM committed").fetchone()[0]

    def all_committed_ids(self) -> List[str]:
        with self._lock:
            rows = self._conn.execute("SELECT sample_id FROM committed").fetchall()
        return [r[0] for r in rows]

    def record_train_ack(
        self,
        sample_ids: List[str],
        *,
        global_step: Optional[int],
        optimizer_durable: bool,
    ) -> None:
        # ONE transaction commits {acked ids, global_step, optimizer marker}
        # together — never split, so a restart reconciles against a single fact.
        with self._lock:
            self._conn.executemany(
                "INSERT OR IGNORE INTO acked (sample_id) VALUES (?)",
                [(s,) for s in sample_ids],
            )
            if global_step is not None:
                self._conn.execute(
                    "INSERT OR REPLACE INTO marker (k, v) VALUES ('global_step', ?)",
                    (json.dumps(global_step),),
                )
            self._conn.execute(
                "INSERT OR REPLACE INTO marker (k, v) VALUES ('optimizer_durable', ?)",
                (json.dumps(bool(optimizer_durable)),),
            )
            self._conn.commit()

    def durable_marker(self) -> Dict[str, Any]:
        with self._lock:
            acked = {
                r[0]
                for r in self._conn.execute("SELECT sample_id FROM acked").fetchall()
            }
            rows = dict(self._conn.execute("SELECT k, v FROM marker").fetchall())
        gs = json.loads(rows["global_step"]) if "global_step" in rows else None
        od = (
            json.loads(rows["optimizer_durable"])
            if "optimizer_durable" in rows
            else False
        )
        return {"acked": acked, "global_step": gs, "optimizer_durable": od}

    # -- weight-version registry (M7) --------------------------------------
    def put_weight_version(self, wv: WeightVersion) -> None:
        with self._lock:
            row = self._conn.execute(
                "SELECT seq FROM weight_versions WHERE version_id = ?", (wv.version_id,)
            ).fetchone()
            if row is None:  # new version -> append at next seq
                seq = (
                    self._conn.execute(
                        "SELECT COALESCE(MAX(seq), -1) + 1 FROM weight_versions"
                    ).fetchone()[0]
                )
            else:  # upsert keeps publish order (status/metric updates)
                seq = row[0]
            self._conn.execute(
                "INSERT OR REPLACE INTO weight_versions (version_id, seq, wv_json) "
                "VALUES (?, ?, ?)",
                (wv.version_id, seq, weight_version_to_json(wv)),
            )
            self._conn.commit()

    def get_weight_version(self, version_id: str) -> Optional[WeightVersion]:
        with self._lock:
            row = self._conn.execute(
                "SELECT wv_json FROM weight_versions WHERE version_id = ?", (version_id,)
            ).fetchone()
        return weight_version_from_json(row[0]) if row else None

    def all_weight_versions(self) -> List[WeightVersion]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT wv_json FROM weight_versions ORDER BY seq ASC"
            ).fetchall()
        return [weight_version_from_json(r[0]) for r in rows]

    def latest_weight_version(self) -> Optional[WeightVersion]:
        with self._lock:
            row = self._conn.execute(
                "SELECT wv_json FROM weight_versions ORDER BY seq DESC LIMIT 1"
            ).fetchone()
        return weight_version_from_json(row[0]) if row else None

    def close(self) -> None:
        with self._lock:
            self._conn.close()


__all__ = [
    "MetadataStore",
    "InMemoryMetadataStore",
    "SQLiteMetadataStore",
    "sample_ref_to_json",
    "sample_ref_from_json",
    "weight_version_to_json",
    "weight_version_from_json",
]
