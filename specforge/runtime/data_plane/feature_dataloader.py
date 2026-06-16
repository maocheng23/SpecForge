# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""FeatureDataLoader: ``SampleRef`` + ``FeatureStore`` -> ``TrainBatch``.

The loader is the one place the online/offline difference is erased: it leases
refs from a queue, fetches their tensors from the store, normalizes each sample
(an injectable ``per_sample_transform``), collates a batch (an injectable
``collate_fn``), and emits a ``TrainBatch``. Because both transform and collate
are injected, the loader carries no model knowledge and is unit-testable on CPU;
the offline-EAGLE3 run injects the existing ``OfflineEagle3Dataset.process_data``
and ``DataCollatorWithPadding`` so the result is bit-identical to today's path.

clone-on-fetch is the default: the loader clones tensors out of the store and
releases the store handle immediately, so prefetch can never race a release.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterator, List, Optional

import torch

from specforge.runtime.contracts import SampleRef, TrainBatch
from specforge.runtime.data_plane.feature_store import FeatureStore
from specforge.runtime.data_plane.sample_ref_queue import SampleRefQueue

PerSampleTransform = Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]
CollateFn = Callable[[List[Dict[str, torch.Tensor]]], Dict[str, Any]]


def _default_collate(features: List[Dict[str, torch.Tensor]]) -> Dict[str, Any]:
    """Trivial stack collate (used only when no collate_fn is injected)."""
    keys = features[0].keys()
    return {k: torch.stack([f[k] for f in features], dim=0) for k in keys}


class FeatureDataLoader:
    def __init__(
        self,
        store: FeatureStore,
        queue: SampleRefQueue,
        *,
        batch_size: int = 1,
        collate_fn: Optional[CollateFn] = None,
        per_sample_transform: Optional[PerSampleTransform] = None,
        device: "torch.device | str" = "cpu",
        clone_on_fetch: bool = True,
        drop_last: bool = True,
        strategy: str = "eagle3",
        ack: bool = True,
    ) -> None:
        self.store = store
        self.queue = queue
        self.batch_size = batch_size
        self.collate_fn = collate_fn or _default_collate
        self.per_sample_transform = per_sample_transform
        self.device = device
        self.clone_on_fetch = clone_on_fetch
        self.drop_last = drop_last
        self.strategy = strategy
        self.ack = ack

    def _materialize(self, ref: SampleRef) -> Dict[str, torch.Tensor]:
        tensors, handle = self.store.get(ref, device=self.device)
        if self.clone_on_fetch:
            tensors = {k: v.clone() for k, v in tensors.items()}
        self.store.release(handle, reason="loaded")
        if self.per_sample_transform is not None:
            tensors = self.per_sample_transform(tensors)
        return tensors

    def __iter__(self) -> Iterator[TrainBatch]:
        while True:
            refs = self.queue.get(self.batch_size, timeout_s=0.0)
            if not refs:
                return
            if self.drop_last and len(refs) < self.batch_size:
                # Incomplete trailing batch: fail-retryable so it is not lost,
                # then stop (mirrors DataLoader(drop_last=True) per epoch pass).
                self.queue.fail(refs, reason="drop_last", retryable=True)
                return
            per_sample = [self._materialize(r) for r in refs]
            batch_tensors = self.collate_fn(per_sample)
            batch = TrainBatch(
                sample_ids=[r.sample_id for r in refs],
                strategy=self.strategy,
                tensors=batch_tensors,
                metadata={
                    "target_repr": refs[0].metadata.get("target_repr"),
                    "ttt_length": refs[0].metadata.get("ttt_length"),
                },
            )
            yield batch
            if self.ack:
                self.queue.ack(refs)

    def close(self) -> None:
        pass


__all__ = ["FeatureDataLoader", "PerSampleTransform", "CollateFn"]
