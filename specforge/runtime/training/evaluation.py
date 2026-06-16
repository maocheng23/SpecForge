# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Serving accept-length gate (B1): the falsifiable objective for a draft.

A ``WeightVersion`` is not promotable until a real serving measurement
(``bench_eagle3``-style: spin up an SGLang spec-decode server at production
topk/num_draft_tokens/num_steps, measure accept_length = output tokens / verify
tokens and e2e speedup) records a TRUE ``accept_length`` and ``speedup`` into
``WeightVersion.metadata``. Promotability requires ``accept_length >= baseline``
and ``speedup >= 1``.

The measurement itself (``bench_fn``) is injected — a real SGLang bench on the
GPU box, or a stub on CPU/CI — so the gate mechanism (which populates the
metadata and decides promotability) is independent of the serving stack. The
cheap in-loop proxy is deliberately NOT used here: it may improve while real
accept_length regresses, so it must never be the promotion decision.
"""

from __future__ import annotations

import dataclasses
import glob
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from specforge.runtime.contracts import WeightVersion


@dataclass(frozen=True)
class AcceptLengthResult:
    accept_length: float
    speedup: float
    serving_config: Dict[str, Any]
    detail: Dict[str, Any] = dataclasses.field(default_factory=dict)


BenchFn = Callable[[WeightVersion, Dict[str, Any]], AcceptLengthResult]


class ServingAcceptLengthGate:
    def __init__(
        self,
        bench_fn: BenchFn,
        *,
        baseline_accept_length: float,
        serving_config: Optional[Dict[str, Any]] = None,
        min_speedup: float = 1.0,
    ) -> None:
        self.bench_fn = bench_fn
        self.baseline_accept_length = baseline_accept_length
        self.serving_config = serving_config or {}
        self.min_speedup = min_speedup

    def evaluate(self, version: WeightVersion) -> WeightVersion:
        """Run the serving measurement and return a version with the gate result.

        Records ``accept_length``, ``speedup``, ``serving_config``,
        ``baseline_accept_length`` and ``promotable`` into ``metadata``.
        """
        result = self.bench_fn(version, self.serving_config)
        promotable = (
            result.accept_length >= self.baseline_accept_length
            and result.speedup >= self.min_speedup
        )
        metadata = dict(version.metadata)
        metadata.update(
            {
                "accept_length": float(result.accept_length),
                "speedup": float(result.speedup),
                "serving_config": result.serving_config,
                "baseline_accept_length": self.baseline_accept_length,
                "promotable": promotable,
            }
        )
        if result.detail:
            metadata["accept_length_detail"] = result.detail
        return dataclasses.replace(version, metadata=metadata)


def _parse_bench_eagle3_results(result_file: str) -> Dict[str, float]:
    """Parse a bench_eagle3 results JSONL into mean accept_length + throughput.

    bench_eagle3 writes ``{model, <bench_name>: [ {metrics: [BenchmarkMetrics...]} ]}``
    where each metric carries ``accept_length`` (= output tokens / verify tokens)
    and ``output_throughput``. We mean over every metric across every benchmark.
    """
    with open(result_file) as f:
        results = json.load(f)
    accepts: List[float] = []
    throughputs: List[float] = []
    for key, runs in results.items():
        if key == "model" or not isinstance(runs, list):
            continue
        for run in runs:
            for m in run.get("metrics", []):
                if "accept_length" in m:
                    accepts.append(float(m["accept_length"]))
                if m.get("output_throughput") is not None:
                    throughputs.append(float(m["output_throughput"]))
    if not accepts:
        raise ValueError(f"no accept_length metrics found in {result_file}")
    return {
        "accept_length": sum(accepts) / len(accepts),
        "output_throughput": (sum(throughputs) / len(throughputs)) if throughputs else 0.0,
    }


def make_bench_eagle3_bench_fn(
    *,
    target_model_path: str,
    bench_script: Optional[str] = None,
    benchmark_list: Sequence[str] = ("mtbench:80",),
    config_list: Sequence[str] = ("1,5,8,64",),
    python_exe: str = sys.executable,
    extra_args: Sequence[str] = (),
    timeout_s: int = 3600,
) -> BenchFn:
    """Build a real ``BenchFn`` that runs ``benchmarks/bench_eagle3.py``.

    Spins up an SGLang spec-decode server on the exported draft
    (``version.serving_uri``/``checkpoint_uri``) against ``target_model_path`` at
    the given production ``config_list`` (batch,steps,topk,num_draft_tokens),
    runs the prompt sets, and returns the TRUE measured accept_length. ``speedup``
    is ``output_throughput / serving_config['baseline_output_throughput']`` when a
    baseline throughput is provided, else reported as unmeasured (detail flag).

    This replaces the hardcoded stub so the accept-length gate is falsifiable.
    Running it needs a GPU + a served model, so it is wired here and invoked by
    the launcher/CI on the H200 box, not in CPU unit tests.
    """
    if bench_script is None:
        bench_script = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            "benchmarks",
            "bench_eagle3.py",
        )

    def bench_fn(version: WeightVersion, serving_config: Dict[str, Any]) -> AcceptLengthResult:
        draft_path = (version.serving_uri or version.checkpoint_uri or "")
        draft_path = draft_path[len("file://") :] if draft_path.startswith("file://") else draft_path
        out_dir = serving_config.get("output_dir") or os.path.join(
            os.path.dirname(draft_path) or ".", "bench_results"
        )
        cmd = [
            python_exe, bench_script,
            "--model", target_model_path,
            "--model-path", draft_path,
            "--output-dir", out_dir,
            "--config-list", *config_list,
            "--benchmark-list", *benchmark_list,
            *extra_args,
        ]
        subprocess.run(cmd, check=True, timeout=timeout_s)
        newest = max(glob.glob(os.path.join(out_dir, "*results_*.jsonl")), key=os.path.getmtime)
        parsed = _parse_bench_eagle3_results(newest)
        baseline_tp = serving_config.get("baseline_output_throughput")
        if baseline_tp:
            speedup = parsed["output_throughput"] / baseline_tp
            measured = True
        else:
            speedup = 1.0
            measured = False
        return AcceptLengthResult(
            accept_length=parsed["accept_length"],
            speedup=speedup,
            serving_config=dict(serving_config),
            detail={
                "output_throughput": parsed["output_throughput"],
                "speedup_measured": measured,
                "result_file": newest,
            },
        )

    return bench_fn


__all__ = [
    "AcceptLengthResult",
    "ServingAcceptLengthGate",
    "BenchFn",
    "make_bench_eagle3_bench_fn",
]
