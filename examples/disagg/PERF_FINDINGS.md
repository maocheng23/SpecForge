# Domino Disagg — Performance Findings & Tuning Guide

Perf investigation of online domino-disagg training on a single 8×H200 node
(1 patched SGLang capture server + 7 FSDP draft trainers, Qwen3-8B target +
~1.1 B DFlash draft). All throughput numbers are **total-samples ÷ total-time
integrated over ≥390 s** — single ~20 s windows are unreliable here (bursty
supply, ~1.8× spread).

## Headline

Config/env tuning alone took the 1srv+DP7 setup from **28.8 → 50.1 samples/s
(+74%)** with util/SM/mem healthy on all 8 GPUs (server util 98% / SM 76%;
trainers util 77–100% / SM 76–85%). No model or default-behavior change — every
lever below is an env knob that is off by default.

## Best config (settings only)

```bash
U=http://127.0.0.1:30000
export DISAGG_SERVER_URLS="$U,$U,$U,$U,$U,$U,$U,$U"   # 8 concurrent producer workers
export CLONE_ON_FETCH=0        # skip the redundant clone on the mooncake zero-copy path
export LOADER_PREFETCH=2       # background-thread batch prefetch (hide fetch latency)
export SERVER_MEM_FRACTION=0.5 # right-size the server KV reservation (126 GB -> ~78 GB)
ACCUM=8 BATCH_SIZE=2 MOONCAKE_PROTOCOL=rdma MOONCAKE_RDMA_DEVICES=mlx5_0,...,mlx5_7
# then the usual run_qwen3_8b_domino_disagg_1srv_dp7.sh invocation
```

## What each lever does (and why)

| Lever | Effect | Mechanism |
|---|---|---|
| `DISAGG_SERVER_URLS` = URL repeated N× | breaks the single-producer ceiling | N concurrent rollout workers with disjoint leases drive one server (the blocking HTTP prefill call releases the GIL) |
| `ACCUM=8` | 460 → ~280 ms/microstep | `no_sync` grad accumulation amortizes the FSDP reduce-scatter across 8 microsteps (comm drops to ~1 ms/microstep) |
| `CLONE_ON_FETCH=0` | −15 ms/batch | the mooncake zero-copy `get()` already allocates a fresh tensor; the extra defensive clone is redundant |
| `LOADER_PREFETCH=2` | removes fetch from the step | a background thread materializes batches ahead so the training step never pays get/collate latency inline |
| `SERVER_MEM_FRACTION=0.5` | server 126 → 78 GB | the default 0.85 hoards KV cache the capture-only server never uses; no perf cost |

Also added (server-side, in `patches/sglang/.../spec-capture.patch`): the capture
sink keeps its hidden-state slices on GPU and does one `torch.cat` + a single D2H
per request instead of a per-prefill-batch unpinned copy (`d2h` 5–8 → ~3.8 ms/sample).

## The pipeline is a supply/demand seesaw on 8 GPUs

At 50 samples/s the system is **supply-bound**: the single server can just barely
feed 7 trainers (loaders wait ~40 ms/batch; producer `in_flight` stays low). The
GPU split is a genuine trade-off, and 1:7 wins:

| Split | Throughput | Regime |
|---|---|---|
| **1 srv + 7 trn** | **50.1/s** | supply-bound (best) |
| 2 srv + 6 trn (b2) | 39.0/s | demand-bound (only 6 trainers) |
| 2 srv + 6 trn (b8) | 39.5/s | demand-bound (bigger batch changes nothing) |

Each trainer GPU ≈ 7 samples/s; one server ≈ 52–57 samples/s. Seven trainers
(~50/s demand) are almost exactly balanced by one server, so trading a trainer
for a second server loses more than it gains.

## Bigger batch does NOT help (and ~100% memory is an anti-goal)

| batch | accum | throughput | trainer mem | trainer util |
|---|---|---|---|---|
| **2** | **8** | **50.1/s** | 45 GB (~31%) | 77–100% |
| 4 | 4 | 47.5/s | 65 GB (~45%) | 64–99% |
| 8 | 2 | 47.2/s | 113 GB (~79%) | 100% |

Memory does climb toward full with batch (b8 → 113 GB, would OOM ~b12–16) but
throughput *falls* — bigger batch just makes each fetch 4× heavier (`get_ms`
20→90) while per-sample compute stays flat. Low trainer memory (~31%) is the
**efficient** state, not a symptom. The trainer is compute-latency-bound
(MFU ~14%), not memory-bound; "full util/memory" ≠ "fast".

## Why is per-sample trainer demand so low? (num_anchors)

Per-microstep compute (batch 2, `PROFILE_STEPS`, cuda-synchronized — trust the
composition, not the absolute):

| num_anchors | fwd | bwd | opt | data_wait |
|---|---|---|---|---|
| 256 (default) | ~88 ms | ~150 ms | 1.3 ms | ~40 ms (14%) |
| 64 | ~43 ms | ~58 ms | 1.3 ms | ~145 ms (38%) — now supply-starved |

The domino/DFlash draft training forward **expands every sample to
`num_anchors × block_size` = 256 × 16 = 4096 draft positions** (independent of
sequence length), each pushed through the 5-layer draft plus a full
151,936-vocab head. Fitting `compute ≈ fixed + k·anchors`: **~53 ms fixed + ~0.73
ms/anchor**, so **~78% of the ~240 ms/microstep compute at 256 anchors is the
anchor expansion.** Cutting anchors to 64 more than halves compute — and the
trainer immediately becomes supply-starved (data_wait 40 → 145 ms), which both
confirms the anchor expansion is the demand driver *and* re-exposes the server as
the next wall.

So the low demand is **by design, not inefficiency**: domino deliberately does
256× the per-token prediction work to densify the draft's training signal. The
optimizer/comm is a non-factor under `ACCUM=8` (1.3 ms/microstep). `num_anchors`
is a **quality↔throughput knob**, not a free win — fewer anchors = less training
signal per sample, so it trades acceptance-length/convergence for speed and must
be validated on the acceptance-length curve, not adopted blindly.

## Remaining ceilings (need code, not config)

1. **Supply ≈ 52–57/s** — the capture sink still runs on the SGLang scheduler
   thread (~17 ms/sample serialized; `put_ms≈7` dominated by per-put
   register/unregister at only ~3.3 GB/s). Fix: async writer queue, preserving
   the "committed ⇒ readable" contract (delay the HTTP response until the put
   lands, or make the consumer `get()` retry-on-missing briefly).
2. **Trainer demand ≈ 50/s** — the num_anchors expansion above; the compute
   lever is fewer anchors / CUDA-graphing the many small anchor-block kernels /
   a reduced-vocab draft head.

## Profiling knobs (all env-gated, default-off)

`PROFILE_PRODUCER=N` (`[prod]`/`[prod-http]`), `PROFILE_LOADER=N` (`[loader rK]`),
`PROFILE_STORE=N` (`[store rK]`), `PROFILE_DISTRIB=secs` (`[dist]`),
`PROFILE_STEPS=N` (`[profile]`/`[profile2]` data-wait vs fwd/bwd/opt),
`PROFILE_TORCH=N` (rank-0 torch.profiler), `FSDP_SHARDING=NO_SHARD`.
