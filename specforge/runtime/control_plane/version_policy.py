# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Published-weight lifecycle + two-axis staleness (M7).

The trainer produces new draft weights; the rollout pool must pick them up
*without a restart* and every sample must record which draft version produced it,
so the loop can reason about staleness and roll back a regression. This module is
the control-plane half of that (metadata only — never the weights):

* ``WeightRegistry`` — the ordered history of published ``WeightVersion``s, the
  current ``active`` one, per-version accept-length, and rollback. Optionally
  durable through a ``MetadataStore`` so versions survive a restart.
* ``WeightPublisher`` — publish a new version and hot-update a rollout pool,
  collecting acks; activate only once the pool confirms (no half-updated pool).
* ``StalenessPolicy`` — the **two axes** a sample can be stale on: the *draft*
  axis (how many published versions behind the sample's draft weights are) and
  the *target* axis (whether the sample's target model matches the current one).
* ``DriftMonitor`` — the rollout-distribution drift signal: the spread of draft
  lags across recent samples, with an emit threshold.

The actual SGLang weight swap lives behind the rollout's ``hot_update_draft_weights``
seam (``inference/sglang_adapter.py``); this module never touches a GPU.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Protocol

from specforge.runtime.contracts import WeightVersion


class HotUpdatableRollout(Protocol):
    """A rollout worker/engine whose draft weights can be swapped live."""

    def hot_update_draft_weights(self, version: WeightVersion) -> bool:
        """Load ``version``'s draft weights with no restart. True == applied."""

    def active_draft_version(self) -> Optional[str]: ...


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
class WeightRegistry:
    """Ordered history of published draft-weight versions + active pointer.

    Publish order is the staleness axis: a sample's draft lag is its distance
    from the newest published version. ``metadata_store`` (optional) makes the
    registry durable; without it, history is in-process.
    """

    def __init__(self, metadata_store: Optional[Any] = None) -> None:
        self._order: List[str] = []  # version_ids, oldest -> newest
        self._versions: Dict[str, WeightVersion] = {}
        self._active_id: Optional[str] = None
        self._store = metadata_store
        self._lock = threading.RLock()
        if self._store is not None and hasattr(self._store, "all_weight_versions"):
            for wv in self._store.all_weight_versions():
                self._order.append(wv.version_id)
                self._versions[wv.version_id] = wv
                if wv.status == "active":
                    self._active_id = wv.version_id

    def _persist(self, wv: WeightVersion) -> None:
        if self._store is not None and hasattr(self._store, "put_weight_version"):
            self._store.put_weight_version(wv)

    def publish(self, wv: WeightVersion) -> WeightVersion:
        """Append a new version (idempotent on version_id)."""
        with self._lock:
            if wv.version_id in self._versions:
                return self._versions[wv.version_id]
            self._versions[wv.version_id] = wv
            self._order.append(wv.version_id)
            self._persist(wv)
            return wv

    def _replace(self, version_id: str, **changes: Any) -> WeightVersion:
        import dataclasses

        wv = dataclasses.replace(self._versions[version_id], **changes)
        self._versions[version_id] = wv
        self._persist(wv)
        return wv

    def activate(self, version_id: str) -> WeightVersion:
        """Make ``version_id`` the active version; demote the prior active."""
        with self._lock:
            if version_id not in self._versions:
                raise KeyError(f"unknown weight version {version_id}")
            if self._active_id is not None and self._active_id != version_id:
                prev = self._versions[self._active_id]
                if prev.status == "active":
                    self._replace(self._active_id, status="candidate")
            wv = self._replace(version_id, status="active")
            self._active_id = version_id
            return wv

    def active(self) -> Optional[WeightVersion]:
        with self._lock:
            return self._versions.get(self._active_id) if self._active_id else None

    def latest(self) -> Optional[WeightVersion]:
        with self._lock:
            return self._versions[self._order[-1]] if self._order else None

    def get(self, version_id: str) -> Optional[WeightVersion]:
        with self._lock:
            return self._versions.get(version_id)

    def history(self) -> List[WeightVersion]:
        with self._lock:
            return [self._versions[v] for v in self._order]

    def draft_lag(self, draft_weight_version: Optional[str]) -> Optional[int]:
        """How many published versions newer than this sample's draft weights.

        0 == the sample used the newest published draft; None == unknown version
        (treated as maximally stale by the policy).
        """
        with self._lock:
            for i in range(len(self._order) - 1, -1, -1):
                if self._versions[self._order[i]].draft_weight_version == draft_weight_version:
                    return (len(self._order) - 1) - i
            return None

    def record_accept_length(self, version_id: str, accept_length: float) -> WeightVersion:
        with self._lock:
            wv = self._versions[version_id]
            metrics = {**wv.metrics, "accept_length": float(accept_length)}
            return self._replace(version_id, metrics=metrics)

    def maybe_rollback(self, *, regression_tol: float = 0.0) -> Optional[WeightVersion]:
        """Roll back the active version if its accept-length regressed vs parent.

        Returns the version rolled back *to* (the parent) if a rollback happened,
        else None. The rolled-back version is marked ``rolled_back`` so it is
        never re-activated by accident.
        """
        with self._lock:
            if self._active_id is None:
                return None
            active = self._versions[self._active_id]
            parent_id = active.parent_version_id
            if parent_id is None or parent_id not in self._versions:
                return None
            parent = self._versions[parent_id]
            a_acc = active.metrics.get("accept_length")
            p_acc = parent.metrics.get("accept_length")
            if a_acc is None or p_acc is None:
                return None
            if a_acc < p_acc - regression_tol:
                self._replace(self._active_id, status="rolled_back")
                restored = self._replace(parent_id, status="active")
                self._active_id = parent_id
                return restored
            return None


# ---------------------------------------------------------------------------
# Publisher (publish + hot-update + ack)
# ---------------------------------------------------------------------------
@dataclass
class PublishResult:
    version: WeightVersion
    acked: List[str]
    failed: List[str]
    activated: bool


class WeightPublisher:
    """Publish a new draft version and hot-update the rollout pool atomically.

    A version is only ``activate``d once the whole pool acks the hot update, so
    rollout never runs half on new and half on old weights. A pool member that
    fails to apply is reported; the version stays a candidate.
    """

    def __init__(self, registry: WeightRegistry) -> None:
        self.registry = registry

    def publish(
        self, version: WeightVersion, rollouts: List[HotUpdatableRollout]
    ) -> PublishResult:
        self.registry.publish(version)
        acked: List[str] = []
        failed: List[str] = []
        for i, r in enumerate(rollouts):
            ident = str(getattr(r, "worker_id", i))
            try:
                ok = r.hot_update_draft_weights(version)
            except Exception:
                ok = False
            (acked if ok else failed).append(ident)
        activated = bool(rollouts) and not failed
        if activated or not rollouts:
            self.registry.activate(version.version_id)
            activated = True
        return PublishResult(
            version=self.registry.get(version.version_id),
            acked=acked,
            failed=failed,
            activated=activated,
        )


# ---------------------------------------------------------------------------
# Two-axis staleness
# ---------------------------------------------------------------------------
@dataclass
class StalenessAssessment:
    draft_lag: Optional[int]
    target_stale: bool
    accept: bool
    reasons: List[str] = field(default_factory=list)


@dataclass
class StalenessPolicy:
    """Two independent staleness axes for a rollout sample.

    * **draft axis** — ``max_draft_lag``: reject a sample whose draft weights are
      more than N published versions behind (None = no draft bound).
    * **target axis** — ``require_target_match``: reject a sample produced against
      a target model version other than the current one (the target changed under
      the rollout). Unknown draft version => maximally stale.
    """

    max_draft_lag: Optional[int] = None
    require_target_match: bool = True

    def assess(
        self,
        *,
        sample_draft_version: Optional[str],
        sample_target_version: str,
        registry: WeightRegistry,
        current_target_version: str,
    ) -> StalenessAssessment:
        reasons: List[str] = []
        lag = registry.draft_lag(sample_draft_version)
        accept = True
        if self.max_draft_lag is not None:
            if lag is None:
                accept = False
                reasons.append("unknown_draft_version")
            elif lag > self.max_draft_lag:
                accept = False
                reasons.append(f"draft_lag>{self.max_draft_lag}")
        target_stale = (
            self.require_target_match
            and sample_target_version != current_target_version
        )
        if target_stale:
            accept = False
            reasons.append("target_version_mismatch")
        return StalenessAssessment(
            draft_lag=lag, target_stale=target_stale, accept=accept, reasons=reasons
        )


# ---------------------------------------------------------------------------
# Drift monitor
# ---------------------------------------------------------------------------
class DriftMonitor:
    """Rollout-distribution drift: the spread of draft lags over recent samples.

    A healthy colocated loop keeps lag ~0; a lagging or partially-updated pool
    shows a rising mean/max lag. ``drifting`` fires when the mean lag crosses a
    threshold so the orchestrator can react (pause, force a sync, alarm).
    """

    def __init__(self, window: int = 256) -> None:
        self._lags: Deque[int] = deque(maxlen=window)
        self._lock = threading.Lock()
        self._unknown = 0
        self._total = 0

    def observe(self, draft_lag: Optional[int]) -> None:
        with self._lock:
            self._total += 1
            if draft_lag is None:
                self._unknown += 1
            else:
                self._lags.append(int(draft_lag))

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            lags = list(self._lags)
            n = len(lags)
            return {
                "samples": self._total,
                "unknown_version": self._unknown,
                "window": n,
                "mean_lag": (sum(lags) / n) if n else 0.0,
                "max_lag": max(lags) if lags else 0,
            }

    def drifting(self, *, mean_lag_threshold: float) -> bool:
        return self.snapshot()["mean_lag"] > mean_lag_threshold


__all__ = [
    "HotUpdatableRollout",
    "WeightRegistry",
    "WeightPublisher",
    "PublishResult",
    "StalenessPolicy",
    "StalenessAssessment",
    "DriftMonitor",
]
