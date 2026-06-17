# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""FeatureStore: the data plane's large-tensor storage and transfer boundary.

``FeatureStore`` is the abstract contract; ``LocalFeatureStore`` is the phase-1
implementation. Per ADR-0003 the local backend keeps features in memory on the
hot path, with an *optional* disk/mmap debug dump that doubles as the
capture/replay tap. It also supports a read-only "existing file" mode so the
``OfflineManifestReader`` can reference precomputed ``.ckpt`` files without
copying them.

Backends later (shared memory, Mooncake/RDMA) slot in behind the same API; the
lease/generation/clone-on-fetch primitives are carried here so phase 1 pays
nothing for them but the contract is already exercised.
"""

from __future__ import annotations

import abc
import dataclasses
import gzip
import io
import itertools
import os
import threading
import uuid
from typing import Any, Dict, List, Optional, Tuple

import torch

from specforge.runtime.contracts import (
    SCHEMA_VERSION,
    FeatureHandle,
    FeatureSpec,
    SampleRef,
)

_DTYPE_BYTES = {  # best-effort; falls back to element_size() for real tensors
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "int64": 8,
    "int32": 4,
    "int16": 2,
    "uint8": 1,
    "bool": 1,
}


def _dtype_str(t: torch.Tensor) -> str:
    return str(t.dtype).replace("torch.", "")


def spec_from_tensor(name: str, t: torch.Tensor, **kw: Any) -> FeatureSpec:
    return FeatureSpec(
        name=name, shape=tuple(t.shape), dtype=_dtype_str(t), **kw
    )


class FeatureStore(abc.ABC):
    """Stores and serves large feature tensors. Carries no scheduling state."""

    @abc.abstractmethod
    def put(
        self,
        tensors: Dict[str, torch.Tensor],
        *,
        sample_id: str,
        metadata: Dict[str, Any],
    ) -> SampleRef:
        ...

    @abc.abstractmethod
    def get(
        self,
        sample_ref: SampleRef,
        *,
        device: "torch.device | str" = "cpu",
        names: Optional[List[str]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], FeatureHandle]:
        ...

    @abc.abstractmethod
    def release(self, handle: FeatureHandle, *, reason: str = "consumed") -> None:
        ...

    @abc.abstractmethod
    def abort(self, sample_id: str, *, reason: str) -> None:
        ...

    def estimate_bytes(self, specs: Dict[str, FeatureSpec]) -> int:
        total = 0
        for spec in specs.values():
            n = 1
            for d in spec.shape:
                n *= int(d)
            total += n * _DTYPE_BYTES.get(spec.dtype, 4)
        return total

    @abc.abstractmethod
    def health(self) -> Dict[str, Any]:
        ...


def load_feature_file(path: str) -> Dict[str, torch.Tensor]:
    """Load a SpecForge offline feature file (mirrors OfflineEagle3Dataset)."""
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as f:
            return torch.load(io.BytesIO(f.read()), weights_only=False)
    return torch.load(path, weights_only=False, mmap=True)


class LocalFeatureStore(FeatureStore):
    """In-memory feature store with optional disk dump and read-only file mode.

    Two ref flavours are served transparently so the loader/trainer path is
    identical online vs offline:

    * ``mem://<store_id>/<sample_id>`` — produced by :meth:`put` (online rollout).
    * ``file://<abs_path>``           — produced by ``OfflineManifestReader``;
      :meth:`get` lazily loads the named keys out of the existing file.
    """

    def __init__(
        self,
        store_id: Optional[str] = None,
        *,
        dump_dir: Optional[str] = None,
        clone_on_get: bool = False,
    ) -> None:
        self.store_id = store_id or uuid.uuid4().hex[:8]
        self.dump_dir = dump_dir
        # When True the store itself clones on get(); normally the *loader* owns
        # the clone policy (clone-on-fetch default lives there), so this is off.
        self.clone_on_get = clone_on_get
        self._mem: Dict[str, Dict[str, torch.Tensor]] = {}
        self._generation: Dict[str, int] = {}
        self._active_leases: Dict[str, FeatureHandle] = {}
        self._lock = threading.RLock()
        self._counter = itertools.count()
        if dump_dir:
            os.makedirs(dump_dir, exist_ok=True)

    # -- write -------------------------------------------------------------
    def put(
        self,
        tensors: Dict[str, torch.Tensor],
        *,
        sample_id: str,
        metadata: Dict[str, Any],
    ) -> SampleRef:
        if not tensors:
            raise ValueError("put requires at least one tensor")
        # Atomic from the controller's view: materialize fully, *then* return a
        # ref. A failure before the final assignment leaves no committed ref.
        staged = {k: v for k, v in tensors.items()}
        specs = {k: spec_from_tensor(k, v) for k, v in staged.items()}
        # Stamp the target feature's representation + vocab-map version onto its
        # spec so the trainer-side mapping is version-gated (ADR-0001 / pruned_logits).
        target_repr = metadata.get("target_repr")
        target_name = metadata.get("target_feature_name", "target")
        if target_repr and target_name in specs:
            vmv = metadata.get("vocab_map_version")
            specs[target_name] = dataclasses.replace(
                specs[target_name],
                target_repr=target_repr,
                target_meta={"vocab_map_version": vmv} if vmv else {},
            )
        num_tokens = int(metadata.get("num_tokens", 0))
        with self._lock:
            gen = self._generation.get(sample_id, 0) + 1
            self._generation[sample_id] = gen
            self._mem[sample_id] = staged
        if self.dump_dir:  # opt-in capture/replay tap (ADR-0003)
            self._dump(sample_id, staged)
        ref = SampleRef(
            sample_id=sample_id,
            run_id=str(metadata.get("run_id", "unknown")),
            source_task_id=metadata.get("source_task_id"),
            feature_store_uri=f"mem://{self.store_id}/{sample_id}",
            feature_keys={k: f"{sample_id}/{k}" for k in staged},
            feature_specs=specs,
            strategy=metadata.get("strategy", "eagle3"),
            schema_version=int(metadata.get("schema_version", SCHEMA_VERSION)),
            target_model_version=str(metadata.get("target_model_version", "unknown")),
            draft_weight_version=metadata.get("draft_weight_version"),
            tokenizer_version=str(metadata.get("tokenizer_version", "unknown")),
            num_tokens=num_tokens,
            estimated_bytes=sum(t.numel() * t.element_size() for t in staged.values()),
            metadata={
                k: v for k, v in metadata.items() if k not in ("num_tokens",)
            },
        )
        return ref

    def _dump(self, sample_id: str, tensors: Dict[str, torch.Tensor]) -> None:
        path = os.path.join(self.dump_dir, f"{sample_id}.ckpt")
        tmp = path + ".tmp"
        torch.save({k: v.detach().cpu() for k, v in tensors.items()}, tmp)
        os.replace(tmp, path)  # atomic publish

    # -- read --------------------------------------------------------------
    def get(
        self,
        sample_ref: SampleRef,
        *,
        device: "torch.device | str" = "cpu",
        names: Optional[List[str]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], FeatureHandle]:
        uri = sample_ref.feature_store_uri
        wanted = names or list(sample_ref.feature_keys.keys())
        if uri.startswith("file://"):
            tensors = self._get_from_file(uri[len("file://") :], sample_ref, wanted)
            generation = 0
        else:
            tensors, generation = self._get_from_mem(sample_ref, wanted)
        if str(device) != "cpu":
            tensors = {k: v.to(device) for k, v in tensors.items()}
        if self.clone_on_get:
            tensors = {k: v.clone() for k, v in tensors.items()}
        handle = FeatureHandle(
            sample_id=sample_ref.sample_id,
            generation=generation,
            lease_token=f"{sample_ref.sample_id}:{next(self._counter)}",
        )
        with self._lock:
            self._active_leases[handle.lease_token] = handle
        return tensors, handle

    def _get_from_mem(
        self, ref: SampleRef, wanted: List[str]
    ) -> Tuple[Dict[str, torch.Tensor], int]:
        with self._lock:
            if ref.sample_id not in self._mem:
                raise KeyError(f"sample {ref.sample_id} not in store {self.store_id}")
            stored = self._mem[ref.sample_id]
            gen = self._generation.get(ref.sample_id, 0)
        missing = [n for n in wanted if n not in stored]
        if missing:
            raise KeyError(f"sample {ref.sample_id} missing features {missing}")
        return {n: stored[n] for n in wanted}, gen

    def _get_from_file(
        self, path: str, ref: SampleRef, wanted: List[str]
    ) -> Dict[str, torch.Tensor]:
        raw = load_feature_file(path)
        out = {}
        for n in wanted:
            # feature_keys may remap a logical name -> a raw file key.
            raw_key = ref.feature_keys.get(n, n)
            raw_key = raw_key.split("/")[-1] if "/" in raw_key else raw_key
            if raw_key not in raw:
                raise KeyError(f"{path} missing key {raw_key!r} for feature {n!r}")
            out[n] = raw[raw_key]
        return out

    # -- lifetime ----------------------------------------------------------
    def release(self, handle: FeatureHandle, *, reason: str = "consumed") -> None:
        # Idempotent + safe against re-leased samples: a stale generation no-ops.
        with self._lock:
            self._active_leases.pop(handle.lease_token, None)
            cur = self._generation.get(handle.sample_id)
            if cur is not None and handle.generation != cur:
                return  # stale handle -> no-op
            # In-memory backend keeps owning tensors; physical free happens on
            # abort or GC (M5). Releasing a lease is enough here.

    def abort(self, sample_id: str, *, reason: str = "aborted") -> None:
        with self._lock:
            self._mem.pop(sample_id, None)
            self._generation.pop(sample_id, None)

    def health(self) -> Dict[str, Any]:
        with self._lock:
            resident_bytes = sum(
                t.numel() * t.element_size()
                for feats in self._mem.values()
                for t in feats.values()
            )
            return {
                "store_id": self.store_id,
                "resident_samples": len(self._mem),
                "active_leases": len(self._active_leases),
                "resident_bytes": resident_bytes,
            }


__all__ = ["FeatureStore", "LocalFeatureStore", "load_feature_file", "spec_from_tensor"]
