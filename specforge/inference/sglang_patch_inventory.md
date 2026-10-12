# SGLang patch inventory and supported version

SpecForge pins `sglang==0.5.18` by default. A separately versioned patch supports
the Kimi K3 SGLang fork at the current validated `kimi-k3` branch tip `9acd9cb`
(and its original `f8493a4` integration point). There are deliberately separate
SGLang integration surfaces.

## Online: external spec-capture server

Online training uses one of these source-specific patches:

| Target | Patch | Capture methods |
|---|---|---|
| SGLang v0.5.18 | [`patches/sglang/v0.5.18/spec-capture.patch`](../../patches/sglang/v0.5.18/spec-capture.patch) | EAGLE3, DFlash, DSpark |
| Kimi K3 SGLang `9acd9cb` (`f8493a4` compatible) | [`patches/sglang/kimi-k3-f8493a4/spec-capture.patch`](../../patches/sglang/kimi-k3-f8493a4/spec-capture.patch) | EAGLE3, DFlash, DSpark |

The patch adds `--enable-spec-capture` and a server-side sink that:

1. captures requested auxiliary and final hidden states during prefill;
2. writes tensors into Mooncake using `MooncakeFeatureStore`'s key layout
   from a background writer thread (one `batch_put_from` RPC per scheduler
   batch), off the scheduler's critical path; and
3. returns only key, shape, and dtype metadata in
   `meta_info["spec_capture"]`, and only after every feature object of the
   request has been durably published — a response therefore guarantees the
   refs it names are readable.

Capture transfers overlap the next target prefill instead of blocking it:
the aux/last-hidden D2H rides the overlap scheduler's `copy_to_cpu` copy
stream, per-request tensors stay zero-copy views into the batch-level host
buffer, and the scheduler retains at most
`SGLANG_SPEC_CAPTURE_MAX_PENDING_BATCHES` (default 2) in-flight batches
before it blocks on the oldest — bounding pinned-host memory while keeping
backpressure. The idle loop never enters the sleeper while a transfer is
outstanding, so a lone in-flight response cannot deadlock a waiting
producer. Set `SGLANG_SPEC_CAPTURE_TIMING=1` to log per-stage
materialize/register/put timings and queue-to-stream latency.

The v0.5.18 patch also carries experimental capture-server fast paths. Each is
off unless its variable is `1` in the server environment (for managed-local
launches, `deployment.disaggregated.managed_local.capture_servers[].env`).
Each path logs once when it engages, or says why it was ignored:

| Variable | Effect | Engages only when |
|---|---|---|
| `SGLANG_SPEC_CAPTURE_SKIP_TAIL` | Stops the Qwen3.5 dense text model right after the last aux tap is captured (the input of layer `max(aux)+1`). Skips the remaining layers, the final norm, the logits, and the last-hidden clone/D2H. | Every request in the batch is a prefill-only capture request that asks for `aux` and not `last_hidden` (teacher metrics off). The batch has no cached prefix and no logprobs, the radix cache is off, and the server runs a single pipeline stage with no DP attention, prefill CP or TBO. Warmup, `/health_generate` and mixed batches run the full model. |
| `SGLANG_CAPTURE_FA_SKIP_KV` | FA3 prefill of capture batches runs on the varlen kernel over raw K/V and skips the paged KV-cache write. This mirrors `fa_skip_kv_cache` on embedding servers. | Spec capture is on, chunked prefill is `-1`, radix cache is off, the model is non-MLA, FA version is 3, the KV cache is model-dtype (no FP8 descale), and the batch is capture-only and prefill-only with no prefix. |
| `SGLANG_GDN_NOCOPY` | Qwen3.5-layout GDN layers whose value/key head ratio is outside the fused split kernel's set (Qwen3.8-27B: 48/16 = 3) stop copying in prefill: `[q|k|v]` is passed to the causal conv as a strided column slice instead of a `torch.cat` (the Triton conv then writes a dense channel-last output), a chunked fused split writes contiguous q/k/v for any `qkv_dim` (no `.contiguous()` per tensor), and the gated RMSNorm runs on `[T, H*D]` rows with `group_size=D` and a head-tiled weight, so `z` stays a strided view. Bitwise identical taps. | CUDA, Qwen3.5 contiguous `[q|k|v|z]` projection layout, 2-D projection output; otherwise the stock path runs. |

`SGLANG_SPEC_CAPTURE_DEBUG_DUMP=<dir>` is for parity checks only. It
`torch.save`s each staged sample to `<dir>/<store>_<sample>_g<gen>.pt` and
never contacts Mooncake, so never set it on a training server.

The client boundary is
[`adapters/server_capture.py`](adapters/server_capture.py). Algorithm-owned
providers map generic server artifacts (`aux`, `last_hidden`, passthrough
inputs) to training feature names. No trainer or producer process imports
SGLang model-runner internals or loads a target model.

The default patch is dry-run validated against the v0.5.18 tag. Capture requests
carry a unique `extra_key`, so every training sample executes a full prefill even
when radix cache support is present. Managed-local launch preserves the
historical disabled-cache default; hybrid targets that require the unified radix
tree set `model.sglang_disable_radix_cache: false`.

For targets that declare `logits_mup_width_multiplier`, the SGLang model passes
an LM-head-scaled hidden state into the logits processor. The capture patch
restores the pre-head-scale post-norm representation because SpecForge folds
the same multiplier into the frozen target head used during training.

Apply the default patch with `scripts/apply_sglang_spec_capture_patch.sh`, or
the K3 patch with
`scripts/apply_sglang_spec_capture_patch.sh --target kimi-k3-9acd9cb`.
On the default patch, `--spec-capture-method dspark` rides the DFlash aux
plumbing (`set_dflash_layers_to_capture`), which stock v0.5.18 models, including
Inkling, implement; DSpark and DFlash capture the same aux/last-hidden artifacts,
so managed-local DSpark launches work unchanged. The K3 patch instead routes
`--spec-capture-method dspark` to the model's dedicated
`set_dspark_layers_to_capture` hook. It also keeps 64K capture correct by using
64-bit Triton pointer arithmetic, scale-stable residual scoring, and a generic
Marlin reduction fallback when the token dimension exceeds CUDA grid.y's
65,535 limit. The server-capture unit and GPU gates must pass before updating
either supported source revision.

## Offline: dedicated local capture

[`../offline_capture`](../offline_capture) is used exclusively by
`scripts/prepare_hidden_states.py`. Its `sglang_backend` owns the local,
version-pinned APIs required for offline EAGLE3 preprocessing:

| Dependency | Upgrade risk |
|---|---|
| `CaptureHiddenMode.FULL` and logits-processor replacement | hidden-state output fields or pruning behavior may change |
| `set_eagle3_layers_to_capture` / `set_dflash_layers_to_capture` / `set_dspark_layers_to_capture` | strategy-specific layer-selection APIs may move |
| `ScheduleBatch`, `ForwardBatch`, and `ModelRunner` construction | constructor and memory-pool setup may change |
| splitting captured states by request input length | token packing conventions may change |
| DP-attention/model-parallel initialization patches | distributed group signatures may change |

This package computes no logits and supports text EAGLE3, DFlash, Domino, and
K3 DSpark state capture needed by the preprocessing script. It does not provide
HF/custom backends, VLM capture, online rollout, or a general target-engine
factory.

`tests/test_runtime/test_sglang_0518_compat.py` guards the patched 0.5.18 API
seams, and
`tests/test_offline_capture/test_sglang_backend.py`
provides the GPU smoke coverage for dense and MoE offline capture.
