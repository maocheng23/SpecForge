# coding=utf-8
"""TrainerCore grad-accum + TrainerController fit/checkpoint + serving gate (CPU)."""

import tempfile
import unittest

import torch
import torch.nn as nn

from specforge.runtime.contracts import TrainBatch, WeightVersion
from specforge.runtime.training.backend import TrainingBackend
from specforge.runtime.training.evaluation import (
    AcceptLengthResult,
    ServingAcceptLengthGate,
)
from specforge.runtime.training.strategy import DraftTrainStrategy, StepOutput
from specforge.runtime.training.trainer import TrainerController, TrainerCore


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.ones(1))


class FakeStrategy(DraftTrainStrategy):
    name = "fake"
    required_features = {"x"}

    def __init__(self):
        self.model = TinyModel()

    def trainable_module(self):
        return self.model

    def forward_loss(self, batch: TrainBatch) -> StepOutput:
        self.validate_batch(batch)
        loss = (self.model.w * batch.tensors["x"].sum()).abs()
        return StepOutput(loss=loss, metrics={"accuracy": torch.tensor(0.5)})


class FakeBackend(TrainingBackend):
    name = "fake"

    def __init__(self, model):
        self.model = model
        self.steps = 0
        self.backwards = 0

    def prepare_model(self, model):
        return model

    def backward(self, loss):
        self.backwards += 1
        loss.backward()

    def step(self):
        self.steps += 1
        return torch.tensor(1.0)

    def state_dict(self):
        return {"draft_model.w": self.model.w.detach().clone()}

    def load_state_dict(self, state):
        pass


def _batch():
    return TrainBatch(sample_ids=["s"], strategy="fake", tensors={"x": torch.ones(2)}, metadata={})


class TestTrainerCore(unittest.TestCase):
    def test_accumulation_boundary(self):
        strat = FakeStrategy()
        backend = FakeBackend(strat.model)
        core = TrainerCore(strat, backend, accumulation_steps=2)
        m0 = core.train_step(_batch())
        self.assertNotIn("grad_norm", m0)  # no optimizer step yet
        self.assertEqual(backend.steps, 0)
        m1 = core.train_step(_batch())
        self.assertIn("grad_norm", m1)  # step on the 2nd micro-batch
        self.assertEqual(backend.steps, 1)
        self.assertEqual(backend.backwards, 2)

    def test_validate_batch_missing_feature(self):
        strat = FakeStrategy()
        bad = TrainBatch(sample_ids=["s"], strategy="fake", tensors={}, metadata={})
        with self.assertRaises(ValueError):
            strat.forward_loss(bad)


class TestTrainerController(unittest.TestCase):
    def test_fit_and_checkpoint(self):
        strat = FakeStrategy()
        backend = FakeBackend(strat.model)
        core = TrainerCore(strat, backend, accumulation_steps=1)
        published = []
        with tempfile.TemporaryDirectory() as d:
            ctrl = TrainerController(
                core, run_id="r", output_dir=d, max_steps=3, num_epochs=5,
                publisher=published.append,
            )
            data = [_batch() for _ in range(10)]
            step = ctrl.fit(data)
            self.assertEqual(step, 3)  # max_steps honored
            self.assertEqual(backend.steps, 3)
            wv = ctrl.save_checkpoint(step)
            self.assertIsInstance(wv, WeightVersion)
            self.assertTrue(wv.checkpoint_uri.startswith("file://"))
            self.assertEqual(published[-1].version_id, wv.version_id)


class TestServingGate(unittest.TestCase):
    def test_gate_populates_accept_length(self):
        def bench(version, cfg):
            return AcceptLengthResult(accept_length=3.4, speedup=1.8, serving_config=cfg)

        gate = ServingAcceptLengthGate(
            bench, baseline_accept_length=3.0, serving_config={"topk": 8}
        )
        wv = WeightVersion("v1", "r", 10, "file://ckpt")
        out = gate.evaluate(wv)
        self.assertEqual(out.metadata["accept_length"], 3.4)
        self.assertTrue(out.metadata["promotable"])
        self.assertEqual(out.metadata["serving_config"], {"topk": 8})

    def test_gate_rejects_regression(self):
        def bench(version, cfg):
            return AcceptLengthResult(accept_length=2.5, speedup=1.1, serving_config=cfg)

        gate = ServingAcceptLengthGate(bench, baseline_accept_length=3.0)
        out = gate.evaluate(WeightVersion("v1", "r", 10, "file://ckpt"))
        self.assertFalse(out.metadata["promotable"])  # below baseline

    def test_gate_rejects_no_speedup(self):
        def bench(version, cfg):
            return AcceptLengthResult(accept_length=3.5, speedup=0.9, serving_config=cfg)

        gate = ServingAcceptLengthGate(bench, baseline_accept_length=3.0)
        out = gate.evaluate(WeightVersion("v1", "r", 10, "file://ckpt"))
        self.assertFalse(out.metadata["promotable"])  # speedup < 1


if __name__ == "__main__":
    unittest.main(verbosity=2)
