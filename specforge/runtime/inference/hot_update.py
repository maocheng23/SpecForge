# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Draft-weight hot update for the rollout/serving engine (M7).

The trainer publishes a new ``WeightVersion`` (``control_plane/version_policy.py``);
the rollout pool must swap to it *with no restart* so the loop keeps producing
samples — and every sample produced afterward must stamp the new
``draft_weight_version`` so its provenance and staleness are knowable.

``RolloutDraftServer`` is the dependency-light reference implementation of the
``HotUpdatableRollout`` seam: it records the active draft version and, on update,
loads the weights via an injected ``apply_fn``. The real SGLang server subclasses
this and points ``apply_fn`` at ``engine.update_weights_from_disk(uri)`` — the
control-plane orchestration (publish → ack → activate → rollback) is identical
either way and is fully testable here without a GPU.
"""

from __future__ import annotations

import threading
from typing import Callable, List, Optional

from specforge.runtime.contracts import WeightVersion


class RolloutDraftServer:
    """A rollout engine whose draft weights can be hot-swapped.

    ``apply_fn(version) -> bool`` performs the actual load (default: a no-op that
    succeeds, for the local/mock path). Subclass or inject ``apply_fn`` to call
    the real SGLang weight update. ``active_draft_version()`` is what a sample
    stamps as its provenance.
    """

    def __init__(
        self,
        worker_id: str,
        *,
        apply_fn: Optional[Callable[[WeightVersion], bool]] = None,
    ) -> None:
        self.worker_id = worker_id
        self._apply_fn = apply_fn
        self._active: Optional[WeightVersion] = None
        self._history: List[str] = []
        self._lock = threading.Lock()

    def hot_update_draft_weights(self, version: WeightVersion) -> bool:
        """Swap to ``version`` with no restart. True == applied + now serving it."""
        ok = True if self._apply_fn is None else bool(self._apply_fn(version))
        if ok:
            with self._lock:
                self._active = version
                self._history.append(version.draft_weight_version)
        return ok

    def active_draft_version(self) -> Optional[str]:
        with self._lock:
            return self._active.draft_weight_version if self._active else None

    def update_history(self) -> List[str]:
        with self._lock:
            return list(self._history)

    def stamp(self) -> Optional[str]:
        """The draft_weight_version a sample produced *now* should record."""
        return self.active_draft_version()


__all__ = ["RolloutDraftServer"]
