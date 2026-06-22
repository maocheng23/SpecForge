# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""RolloutWorker: PromptTask -> features -> FeatureStore -> SampleRef commit.

The worker is deliberately small and strategy-agnostic: it leases prompt tasks,
asks a ``feature_source`` (e.g. a wrapper over the target model's
``generate_eagle3_data``, or ``SGLangAdapter``) for per-sample features,
verifies them against the typed ``CaptureConfig`` *before* writing, writes them
to the ``FeatureStore``, and commits the resulting ``SampleRef`` metadata to the
controller. It never hands a tensor to the controller. Strategy-specific capture
requirements live in ``CaptureConfig`` + the feature schema, not here.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol

from specforge.runtime.contracts import PromptTask, SampleRef
from specforge.runtime.inference.capture import (
    CaptureConfig,
    CaptureMismatchError,
    verify_capture,
)

# health states: a worker REPORTS health; the controller decides scheduling.
HEALTH_STATES = ("starting", "ready", "paused", "draining", "unhealthy", "stopped")


class FeatureSource(Protocol):
    def generate_features(
        self, tasks: List[PromptTask], *, capture: CaptureConfig
    ) -> List[Dict[str, Any]]:
        ...


class RolloutWorker:
    def __init__(
        self,
        controller,
        feature_store,
        feature_source: FeatureSource,
        capture: CaptureConfig,
        *,
        run_id: str,
        worker_id: Optional[str] = None,
        strategy: str = "eagle3",
        target_model_version: str = "unknown",
        tokenizer_version: str = "unknown",
        draft_weight_version: Optional[str] = None,
    ) -> None:
        self.controller = controller
        self.feature_store = feature_store
        self.feature_source = feature_source
        self.capture = capture
        self.run_id = run_id
        self.strategy = strategy
        self.target_model_version = target_model_version
        self.tokenizer_version = tokenizer_version
        self.draft_weight_version = draft_weight_version
        self._state = "starting"
        self._inflight = 0
        self._recent_failures: List[str] = []
        self._last_commit_count = 0
        self.worker_id = controller.register_rollout_worker(
            {"worker_id": worker_id, "strategy": strategy, "role": "rollout"}
        )

    def start(self) -> None:
        self._state = "ready"

    def stop(self, reason: str = "stopped") -> None:
        # graceful: finish in-flight, then mark stopped (drain happens in run_once)
        self._state = "stopped"

    def _sample_id(self, task: PromptTask) -> str:
        return f"{self.run_id}:{task.task_id}"

    def run_once(self, max_tasks: int) -> List[SampleRef]:
        """Lease up to ``max_tasks``, extract + persist features, commit refs.

        Every leased task ends in exactly one terminal controller action:
        ``commit_samples`` (success) or ``fail_prompt_tasks`` (any failure). A
        single capture mismatch still commits the batch's good samples, then
        re-raises so the driver sees the worker went unhealthy.
        """
        if self._state in ("stopped", "draining"):
            return []
        tasks = self.controller.lease_prompt_tasks(self.worker_id, max_tasks)
        if not tasks:
            return []
        self._inflight = len(tasks)
        self._state = "ready"
        try:
            feats_list = self._generate_or_fail(tasks)
            refs, capture_error = self._persist(tasks, feats_list)
            if refs:
                self.controller.commit_samples(self.worker_id, refs)
                self._last_commit_count += len(refs)
            if capture_error is not None:
                self._state = "unhealthy"
                raise capture_error
            return refs
        finally:
            self._inflight = 0

    def _fail(
        self,
        task_ids: List[str],
        reason: str,
        *,
        retryable: bool,
        fatal: bool = False,
    ) -> None:
        """Single terminal failure path: record it, optionally go unhealthy,
        release the prompt leases (requeued iff ``retryable``)."""
        self._recent_failures.append(reason)
        if fatal:
            self._state = "unhealthy"
        self.controller.fail_prompt_tasks(
            self.worker_id, task_ids, reason=reason, retryable=retryable
        )

    def _generate_or_fail(self, tasks: List[PromptTask]) -> List[Dict[str, Any]]:
        """Run the (batched) feature source and enforce the 1-record-per-task
        contract. Both failures are batch-fatal: fail every lease, then raise."""
        task_ids = [t.task_id for t in tasks]
        try:
            feats_list = self.feature_source.generate_features(
                tasks, capture=self.capture
            )
        except Exception as exc:  # rollout failure before any feature write
            self._fail(task_ids, f"generate_features: {exc}", retryable=True, fatal=True)
            raise
        if len(feats_list) != len(tasks):
            reason = (
                f"generate_features returned {len(feats_list)} feature records "
                f"for {len(tasks)} tasks"
            )
            self._fail(task_ids, reason, retryable=False, fatal=True)
            raise ValueError(reason)
        return feats_list

    def _persist(
        self, tasks: List[PromptTask], feats_list: List[Dict[str, Any]]
    ) -> "tuple[List[SampleRef], Optional[CaptureMismatchError]]":
        """Per-sample: verify against the capture contract, then write to the
        feature store. Each task self-resolves (commit-ready ref or a fail), so
        no lease is stranded; the first capture mismatch is returned to the
        caller to re-raise after the good samples are committed."""
        refs: List[SampleRef] = []
        capture_error: Optional[CaptureMismatchError] = None
        for task, feats in zip(tasks, feats_list):
            sample_id = self._sample_id(task)
            recorded = feats.pop("__aux_layer_ids__", None)
            try:
                verify_capture(
                    feats,
                    self.capture,
                    sample_id=sample_id,
                    recorded_aux_layer_ids=recorded,
                )
            except CaptureMismatchError as exc:
                # Loud failure: do not persist a corrupt sample, but keep this
                # batch's other prompt leases moving so no lease is stranded.
                self._fail([task.task_id], str(exc), retryable=False)
                if capture_error is None:
                    capture_error = exc
                continue
            try:
                ref = self.feature_store.put(
                    feats,
                    sample_id=sample_id,
                    metadata=self._put_metadata(task),
                )
            except Exception as exc:  # partial write -> abort, report
                self.feature_store.abort(sample_id, reason=f"put_failed:{exc}")
                self._fail([task.task_id], str(exc), retryable=True)
                continue
            refs.append(ref)
        return refs, capture_error

    def _put_metadata(self, task: PromptTask) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "source_task_id": task.task_id,
            "strategy": self.strategy,
            "target_repr": self.capture.target_repr,
            "vocab_map_version": self.capture.vocab_map_version,
            "ttt_length": self.capture.extra.get("ttt_length"),
            "target_model_version": self.target_model_version,
            "tokenizer_version": self.tokenizer_version,
            "draft_weight_version": self.draft_weight_version,
            "num_tokens": int(task.metadata.get("num_tokens", 0)),
        }

    def drain(self) -> None:
        """Stop leasing new work; in-flight is finished by the active run_once."""
        self._state = "draining"

    # Draft-weight hot update (update_weights -> adapter) is not yet supported.
    # draft_weight_version is still recorded as rollout provenance on each sample.

    def health(self) -> Dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "state": self._state,
            "strategy": self.strategy,
            "draft_weight_version": self.draft_weight_version,
            "in_flight": self._inflight,
            "recent_failures": self._recent_failures[-5:],
            "committed": self._last_commit_count,
        }


__all__ = ["RolloutWorker", "FeatureSource", "HEALTH_STATES"]
