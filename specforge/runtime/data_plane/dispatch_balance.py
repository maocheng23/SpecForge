"""Cost-balanced dispatch of one optimizer window across DP ranks.

The :class:`RefDistributor` releases refs one optimizer window at a time. By
default it deals them round-robin, so each rank's share of a window carries a
random amount of work, and every rank then waits at the gradient reduction for
the slowest one. Within a micro-batch, samples are also padded to the longest
sequence and the widest anchor set.

With a cost function the whole window is planned at once:

1. Sort the window's refs by estimated cost and chunk neighbours into
   micro-batches, so samples of similar shape share padding.
2. Assign micro-batches to ranks longest-processing-time first, each rank
   receiving exactly ``accumulation_steps`` micro-batches.

The objective is unchanged: the trainer normalizes gradients by the global
loss denominator of the optimizer step, so moving a sample between ranks or
micro-batches of the same window only changes floating-point reduction order.
"""

from __future__ import annotations

import os
from typing import Callable, List, Optional, Sequence, Tuple

from specforge.runtime.contracts import SampleRef

#: Per-ref padded widths: a micro-batch of ``n`` refs costs
#: ``n * sum_d max_i width_d(ref_i)``, every dimension being padded to the
#: widest ref of the micro-batch.
CostFn = Callable[[SampleRef], Tuple[float, ...]]

BALANCED_DISPATCH_ENV = "SPECFORGE_BALANCED_DISPATCH"


def balanced_dispatch_enabled() -> bool:
    return os.environ.get(BALANCED_DISPATCH_ENV, "0") == "1"


def micro_batch_cost(refs: Sequence[SampleRef], cost_fn: CostFn) -> float:
    widths = [cost_fn(ref) for ref in refs]
    return len(refs) * sum(max(column) for column in zip(*widths))


def plan_balanced_window(
    refs: Sequence[SampleRef],
    dp_size: int,
    batch_size: int,
    accumulation_steps: int,
    cost_fn: CostFn,
) -> List[List[List[SampleRef]]]:
    """Return ``rounds[accumulation_step][dp_rank] -> refs`` for one window."""

    expected = dp_size * batch_size * accumulation_steps
    if len(refs) != expected:
        raise ValueError(f"window holds {len(refs)} refs, expected {expected}")
    # Sort heaviest first; sample_id breaks ties so the plan is deterministic.
    order = sorted(refs, key=lambda ref: (-sum(cost_fn(ref)), ref.sample_id))
    micro_batches = [
        order[start : start + batch_size] for start in range(0, len(order), batch_size)
    ]
    costed = sorted(
        (
            (micro_batch_cost(mb, cost_fn), index, mb)
            for index, mb in enumerate(micro_batches)
        ),
        key=lambda item: (-item[0], item[1]),
    )
    loads = [0.0] * dp_size
    assigned: List[List[List[SampleRef]]] = [[] for _ in range(dp_size)]
    for cost, _, mb in costed:
        rank = min(
            (r for r in range(dp_size) if len(assigned[r]) < accumulation_steps),
            key=lambda r: (loads[r], r),
        )
        assigned[rank].append(mb)
        loads[rank] += cost
    return [
        [assigned[rank][step] for rank in range(dp_size)]
        for step in range(accumulation_steps)
    ]


def dflash_dispatch_cost_fn(draft_config, num_anchors: int) -> Optional[CostFn]:
    """Estimate a DFlash-family sample's trainer work from its ref.

    Work per kept anchor (``block_size`` query rows through every draft layer
    plus the frozen full-vocabulary head) dominates; the context tokens add
    the fused target-feature projection and per-layer context K/V. Anchors
    are capped at ``num_anchors`` and counted from ``metadata["loss_pairs"]``
    (positions whose token and successor are both supervised), falling back
    to the token count for refs that do not carry it.
    """

    try:
        hidden = int(draft_config.hidden_size)
        layers = int(draft_config.num_hidden_layers)
        inter = int(draft_config.intermediate_size)
        head_dim = int(
            getattr(draft_config, "head_dim", None)
            or hidden // int(draft_config.num_attention_heads)
        )
        q_dim = int(draft_config.num_attention_heads) * head_dim
        kv_dim = int(draft_config.num_key_value_heads) * head_dim
        vocab = int(draft_config.vocab_size)
        dflash = dict(getattr(draft_config, "dflash_config", None) or {})
        block = int(dflash.get("block_size") or getattr(draft_config, "block_size"))
        taps = len(dflash.get("target_layer_ids") or ()) or 1
    except (AttributeError, TypeError, ValueError):
        return None
    per_row = (
        6 * layers * (2 * hidden * q_dim + 2 * hidden * kv_dim + 3 * hidden * inter)
    )
    per_anchor = block * per_row + (block - 1) * 4 * hidden * vocab
    per_token = 4 * taps * hidden * hidden + 6 * layers * 2 * hidden * kv_dim
    scale = float(per_token)

    def cost(ref: SampleRef) -> Tuple[float, float]:
        tokens = int(ref.num_tokens or 0)
        pairs = ref.metadata.get("loss_pairs") if ref.metadata else None
        anchors = min(num_anchors, int(pairs) if pairs is not None else tokens)
        return (anchors * per_anchor / scale, float(tokens))

    return cost


def loss_pairs(loss_mask: Sequence[int]) -> int:
    """Positions ``p`` whose token and successor are both supervised."""

    count = 0
    previous = False
    for value in loss_mask:
        current = bool(value)
        count += previous and current
        previous = current
    return count


__all__ = [
    "BALANCED_DISPATCH_ENV",
    "CostFn",
    "balanced_dispatch_enabled",
    "dflash_dispatch_cost_fn",
    "loss_pairs",
    "micro_batch_cost",
    "plan_balanced_window",
]
