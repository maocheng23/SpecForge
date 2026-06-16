# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Launch helpers that wire the DataFlow runtime from a RunConfig.

The training *script* becomes a thin launcher (M3): it parses args, calls one of
these builders, and runs ``TrainerController.fit``. All training logic lives in
the runtime components, not the script. This module wires the **offline EAGLE3**
path end to end:

    OfflineManifestReader -> DataFlowController -> SampleRefQueue
        -> FeatureDataLoader(process_data, DataCollatorWithPadding)
        -> TrainBatch -> Eagle3TrainStrategy -> TrainerCore/Controller -> FSDP

Online wiring (RolloutWorker + SGLangAdapter) composes the same control/data
plane; see ``inference/`` and the M2 test for the equivalent assembly.
"""

from __future__ import annotations

from typing import Optional

from specforge.runtime.control_plane import DataFlowController
from specforge.runtime.data_plane import (
    FeatureDataLoader,
    LocalFeatureStore,
    OfflineManifestReader,
)
from specforge.runtime.training.backend import FSDPTrainingBackend, ParallelConfig
from specforge.runtime.training.strategy import Eagle3TrainStrategy
from specforge.runtime.training.trainer import TrainerController, TrainerCore


def build_offline_eagle3_controller(
    *,
    hidden_states_path: str,
    eagle3_model,
    target_head,
    optimizer,
    run_id: str,
    output_dir: str,
    ttt_length: int = 7,
    max_len: int = 2048,
    batch_size: int = 1,
    accumulation_steps: int = 1,
    num_epochs: int = 1,
    max_steps: Optional[int] = None,
    save_interval: int = 0,
    eval_interval: int = 0,
    tp_size: int = 1,
    sp_ulysses_size: int = 1,
    sp_ring_size: int = 1,
    logger=None,
):
    """Assemble the offline-EAGLE3 dataflow and return (controller, loader)."""
    from specforge.data.preprocessing import OfflineEagle3Dataset
    from specforge.data.utils import DataCollatorWithPadding

    controller = DataFlowController(run_id)
    controller.enqueue_offline_refs(
        OfflineManifestReader(
            hidden_states_path,
            run_id=run_id,
            ttt_length=ttt_length,
            max_len=max_len,
            target_repr="hidden_state",
        ).read()
    )
    store = LocalFeatureStore(run_id)
    # Lease + ack go through the controller (durable ack transaction; cross-node
    # ready) rather than the raw queue. The trainer owns the ack at the
    # optimizer-step boundary, so the loader does not ack.
    trainer_id = controller.register_trainer({"role": "trainer", "run_id": run_id})
    lease = controller.train_lease(trainer_id)
    loader = FeatureDataLoader(
        store,
        lease,
        batch_size=batch_size,
        collate_fn=DataCollatorWithPadding(),
        per_sample_transform=lambda raw: OfflineEagle3Dataset.process_data(raw, max_len),
        drop_last=True,
        strategy="eagle3",
        ack=False,
    )

    parallel = ParallelConfig.from_distributed(
        tp_size=tp_size, sp_ulysses_size=sp_ulysses_size, sp_ring_size=sp_ring_size
    )
    backend = FSDPTrainingBackend(parallel)
    backend.prepare_model(eagle3_model)
    backend.set_optimizer(optimizer)
    strategy = Eagle3TrainStrategy(eagle3_model, target_head=target_head)
    core = TrainerCore(strategy, backend, accumulation_steps=accumulation_steps)
    trainer = TrainerController(
        core,
        run_id=run_id,
        output_dir=output_dir,
        num_epochs=num_epochs,
        max_steps=max_steps,
        save_interval=save_interval,
        eval_interval=eval_interval,
        logger=logger,
        ack_fn=lambda ids, step: controller.ack_train_refs(
            trainer_id, ids, global_step=step, optimizer_durable=True
        ),
    )
    return trainer, loader


__all__ = ["build_offline_eagle3_controller"]
