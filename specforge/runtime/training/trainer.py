# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""TrainerCore + TrainerController: the trainer-boundary split (M3).

* ``TrainerCore`` owns exactly one train/eval step plus the grad-accumulation and
  optimizer boundary. It is **branch-free**: it never inspects online/offline or
  ``target_repr`` and never applies a projection — that is the strategy's job. It
  consumes a normalized ``TrainBatch`` and delegates the forward/loss to the
  strategy and the backward/step to the backend.
* ``TrainerController`` owns the lifecycle: ``fit`` / ``evaluate`` /
  ``save_checkpoint`` / weight publication. The training *script* becomes a thin
  launcher that builds these and calls ``fit``.

EAGLE3 and DFlash share this lifecycle unchanged — only the strategy differs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

import torch

from specforge.runtime.contracts import TrainBatch
from specforge.runtime.training.backend import TrainingBackend
from specforge.runtime.training.strategy import DraftTrainStrategy, StepOutput


@dataclass(frozen=True)
class Checkpoint:
    """A saved training checkpoint location (resume target).

    Deliberately NOT a published "weight version" — the published-weight
    lifecycle (versioning, publisher, serving accept-length gate, hot update) is
    deferred to M7. This record only says where a checkpoint is and at what step.
    """

    checkpoint_uri: str
    global_step: int
    epoch: int
    strategy: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def _scalar(x: Any) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.detach().float().mean().item())
    if isinstance(x, (list, tuple)) and x:
        return float(torch.stack([t.detach().float() for t in x]).mean().item())
    return float(x)


class TrainerCore:
    """One step: forward/loss (strategy) -> backward (backend) -> optimizer boundary."""

    def __init__(
        self,
        strategy: DraftTrainStrategy,
        backend: TrainingBackend,
        *,
        accumulation_steps: int = 1,
    ) -> None:
        self.strategy = strategy
        self.backend = backend
        self.accumulation_steps = max(1, accumulation_steps)
        self._micro = 0

    def train_step(self, batch: TrainBatch) -> Dict[str, Any]:
        out: StepOutput = self.strategy.forward_loss(batch)
        loss = out.loss / self.accumulation_steps
        self.backend.backward(loss)
        self._micro += 1
        grad_norm = None
        if self._micro % self.accumulation_steps == 0:
            grad_norm = self.backend.step()
        return self._report(out, grad_norm, mode="train")

    @torch.no_grad()
    def eval_step(self, batch: TrainBatch) -> Dict[str, Any]:
        out: StepOutput = self.strategy.forward_loss(batch)
        return self._report(out, None, mode="eval")

    def _report(self, out: StepOutput, grad_norm, mode: str) -> Dict[str, Any]:
        rep: Dict[str, Any] = {"loss": _scalar(out.loss)}
        for key in ("acces", "acceptance_rates", "plosses"):
            if key in out.metrics:
                rep[key.rstrip("es") if key == "acces" else key] = _scalar(
                    out.metrics[key]
                )
        if "accuracy" in out.metrics:
            rep["acc"] = _scalar(out.metrics["accuracy"])
        if grad_norm is not None:
            rep["grad_norm"] = _scalar(grad_norm)
        rep["mode"] = mode
        return rep


class TrainerController:
    """Lifecycle: fit / evaluate / checkpoint. Script becomes a launcher.

    Weight publishing + the serving accept-length gate are deferred to M7;
    save_checkpoint just persists training state and returns a Checkpoint.
    """

    def __init__(
        self,
        core: TrainerCore,
        *,
        run_id: str,
        output_dir: str = "./output",
        save_interval: int = 0,
        eval_interval: int = 0,
        log_interval: int = 50,
        max_steps: Optional[int] = None,
        num_epochs: int = 1,
        logger: Optional[Callable[[Dict[str, Any], int], None]] = None,
        ack_fn: Optional[Callable[[List[str], int], None]] = None,
        start_step: int = 0,
        start_epoch: int = 0,
    ) -> None:
        self.core = core
        self.run_id = run_id
        self.output_dir = output_dir
        self.save_interval = save_interval
        self.eval_interval = eval_interval
        self.log_interval = log_interval
        self.max_steps = max_steps
        self.num_epochs = num_epochs
        self.logger = logger
        # ack_fn(sample_ids, global_step): acks consumed refs at the optimizer-step
        # boundary with the step number, so the controller records the durable
        # {acked, global_step, optimizer marker} transaction. If None, the loader
        # is assumed to ack (e.g. simple/equivalence runs).
        self.ack_fn = ack_fn
        self.global_step = start_step
        self.epoch = start_epoch
        self.last_metrics: Dict[str, Any] = {}

    def fit(self, data: Iterable[TrainBatch], eval_data: Optional[Iterable] = None) -> int:
        module = self.core.strategy.trainable_module()
        module.train()
        pending_ack: List[str] = []
        for epoch in range(self.epoch, self.num_epochs):
            self.epoch = epoch
            if hasattr(data, "set_epoch"):
                data.set_epoch(epoch)
            for batch in data:
                self.global_step += 1
                pending_ack.extend(batch.sample_ids)
                metrics = self.core.train_step(batch)
                self.last_metrics = metrics
                # "grad_norm" present == optimizer stepped (grad-accum boundary).
                if self.ack_fn is not None and "grad_norm" in metrics:
                    self.ack_fn(pending_ack, self.global_step)
                    pending_ack = []
                if self.logger and self.global_step % max(1, self.log_interval) == 0:
                    self.logger(metrics, self.global_step)
                if (
                    self.eval_interval
                    and eval_data is not None
                    and self.global_step % self.eval_interval == 0
                ):
                    self.evaluate(eval_data)
                    module.train()
                if self.save_interval and self.global_step % self.save_interval == 0:
                    self.save_checkpoint(self.global_step)
                if self.max_steps is not None and self.global_step >= self.max_steps:
                    return self.global_step
        return self.global_step

    @torch.no_grad()
    def evaluate(self, data: Iterable[TrainBatch]) -> Dict[str, float]:
        module = self.core.strategy.trainable_module()
        module.eval()
        agg: Dict[str, list] = {}
        n = 0
        for batch in data:
            rep = self.core.eval_step(batch)
            n += 1
            for k, v in rep.items():
                if isinstance(v, (int, float)):
                    agg.setdefault(k, []).append(v)
        return {k: sum(vs) / len(vs) for k, vs in agg.items() if vs}

    def save_checkpoint(self, step: int) -> Checkpoint:
        ckpt_dir = os.path.join(self.output_dir, f"{self.run_id}-step{step}")
        is_rank0 = (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0
        full_state = self.core.backend.state_dict()
        draft_state = self.core.strategy.checkpoint_state_filter(full_state)
        if is_rank0:
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(
                {
                    "draft_state_dict": draft_state,
                    "global_step": step,
                    "epoch": self.epoch,
                    "strategy": self.core.strategy.name,
                    "run_id": self.run_id,
                },
                os.path.join(ckpt_dir, "training_state.pt"),
            )
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        return Checkpoint(
            checkpoint_uri=f"file://{os.path.abspath(ckpt_dir)}",
            global_step=step,
            epoch=self.epoch,
            strategy=self.core.strategy.name,
        )


__all__ = ["TrainerCore", "TrainerController", "Checkpoint"]
