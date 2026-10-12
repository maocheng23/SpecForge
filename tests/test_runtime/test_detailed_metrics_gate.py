# coding=utf-8
"""SPECFORGE_DETAILED_METRICS gates the detailed-metrics micro-batches (CPU)."""

import os
import tempfile
import unittest
from unittest import mock

import torch

from specforge.runtime.contracts import TrainBatch
from specforge.training.controller import TrainerController, TrainerCore
from specforge.training.strategies.base import StepOutput
from tests.test_runtime.test_trainer import FakeBackend, FakeStrategy

_ENV = "SPECFORGE_DETAILED_METRICS"


class DetailedStrategy(FakeStrategy):
    """Emit a detail-only ratio and count, like DFlash's log-step diagnostics."""

    def forward_loss(self, batch: TrainBatch, ctx=None) -> StepOutput:
        out = super().forward_loss(batch, ctx)
        if ctx is None or not ctx.collect_detailed_metrics:
            return out
        hits = batch.tensors["hits"].reshape(())
        total = batch.tensors["total"].reshape(())
        return StepOutput(
            loss=out.loss,
            metrics=out.metrics,
            ratio_metrics={"detail/hit_rate": (hits, total)},
            sum_metrics={"detail/count": total},
        )


def _batch(hits, total):
    return TrainBatch(
        sample_ids=[f"s{hits}-{total}"],
        strategy="fake",
        tensors={
            "x": torch.ones(2),
            "hits": torch.tensor(float(hits)),
            "total": torch.tensor(float(total)),
        },
        metadata={},
    )


def _fit(env_value, *, accumulation_steps=3, max_steps=2, log_interval=2):
    environ = {} if env_value is None else {_ENV: env_value}
    strat = DetailedStrategy()
    core = TrainerCore(
        strat, FakeBackend(strat.model), accumulation_steps=accumulation_steps
    )
    # Distinct ratios per micro-batch expose which ones were accumulated.
    data = [
        _batch(hits, 10 * (index + 1))
        for index, hits in enumerate(range(1, accumulation_steps * max_steps + 1))
    ]
    logged = []
    with (
        mock.patch.dict(os.environ, environ, clear=False),
        tempfile.TemporaryDirectory() as d,
    ):
        if env_value is None:
            os.environ.pop(_ENV, None)
        ctrl = TrainerController(
            core,
            run_id="r",
            output_dir=d,
            max_steps=max_steps,
            num_epochs=1,
            log_interval=log_interval,
            logger=lambda metrics, step: logged.append((dict(metrics), step)),
        )
        ctrl.fit(data)
    flags = [ctx.collect_detailed_metrics for ctx in strat.contexts]
    return flags, logged, data


class TestDetailedMetricsGate(unittest.TestCase):
    def test_default_collects_every_microbatch_of_the_log_step(self):
        for value in (None, "", "all", " ALL "):
            with self.subTest(value=value):
                flags, logged, _ = _fit(value)
                self.assertEqual(flags, [False] * 3 + [True] * 3)
                ((metrics, step),) = logged
                self.assertEqual(step, 2)
                self.assertNotIn("detailed_metrics/microbatch_fraction", metrics)
                # Ratio over all three micro-batches of step 2: (4+5+6)/(40+50+60).
                self.assertAlmostEqual(metrics["detail/hit_rate"], 15 / 150)
                self.assertAlmostEqual(metrics["detail/count"], 150.0)

    def test_last_microbatch_collects_only_the_boundary_microbatch(self):
        flags, logged, _ = _fit("last_microbatch")
        self.assertEqual(flags, [False] * 5 + [True])
        ((metrics, step),) = logged
        self.assertEqual(step, 2)
        self.assertAlmostEqual(metrics["detailed_metrics/microbatch_fraction"], 1 / 3)
        # Only micro-batch 6 contributes: 6 / 60, and its count alone.
        self.assertAlmostEqual(metrics["detail/hit_rate"], 6 / 60)
        self.assertAlmostEqual(metrics["detail/count"], 60.0)
        # Always-on metrics still cover the whole window.
        self.assertIn("loss", metrics)
        self.assertIn("perf/train_compute_time_s", metrics)

    def test_last_microbatch_matches_all_without_accumulation(self):
        all_flags, all_logged, _ = _fit("all", accumulation_steps=1, max_steps=4)
        last_flags, last_logged, _ = _fit(
            "last_microbatch", accumulation_steps=1, max_steps=4
        )
        self.assertEqual(all_flags, [False, True, False, True])
        self.assertEqual(last_flags, all_flags)
        for (all_metrics, _), (last_metrics, _) in zip(all_logged, last_logged):
            self.assertEqual(
                all_metrics["detail/hit_rate"], last_metrics["detail/hit_rate"]
            )

    def test_skipping_the_pass_keeps_the_anchor_rng_stream(self):
        """Gated micro-batches must not shift later micro-batches' numerics."""
        from tests.test_utils.test_dflash2_fused_head import _online_model

        model = _online_model("cpu", torch.float32)
        generator = torch.Generator().manual_seed(3)

        def batch(length):
            loss_mask = torch.ones(2, length)
            loss_mask[:, :5] = 0
            loss_mask[1, length - 9 :] = 0
            return {
                "input_ids": torch.randint(0, 500, (2, length), generator=generator),
                "hidden_states": torch.randn(2, length, 128, generator=generator),
                "loss_mask": loss_mask,
            }

        batches = (batch(40), batch(33))

        def run(first_detailed):
            torch.manual_seed(7)
            return [
                model(**inputs, collect_detailed_metrics=detailed)
                for inputs, detailed in zip(batches, (first_detailed, True))
            ]

        every, gated = run(True), run(False)
        for (every_loss, _, every_metrics), (gated_loss, _, gated_metrics) in zip(
            every, gated
        ):
            self.assertTrue(torch.equal(every_loss, gated_loss))
        every_ratios = every[1][2]["ratio_metrics"]
        gated_ratios = gated[1][2]["ratio_metrics"]
        self.assertEqual(set(every_ratios), set(gated_ratios))
        self.assertIn("dflash/hard_label/walk_accepted_length", gated_ratios)
        for name, (numerator, denominator) in every_ratios.items():
            with self.subTest(metric=name):
                self.assertTrue(torch.equal(numerator, gated_ratios[name][0]))
                self.assertTrue(torch.equal(denominator, gated_ratios[name][1]))

    def test_unknown_mode_is_rejected(self):
        strat = DetailedStrategy()
        core = TrainerCore(strat, FakeBackend(strat.model))
        with (
            mock.patch.dict(os.environ, {_ENV: "first_microbatch"}),
            tempfile.TemporaryDirectory() as d,
        ):
            with self.assertRaisesRegex(ValueError, _ENV):
                TrainerController(core, run_id="r", output_dir=d)


if __name__ == "__main__":
    unittest.main()
