#!/bin/bash
# EAGLE3 draft training for Kimi K2.7 Code (4x GB200, WORLD=16). Sibling of
# run_kimi_k2.7_code_dflash.sh — SAME target / data / tokenize-cache / env, but
# the EAGLE3 algorithm (train_eagle3.py + LlamaForCausalLMEagle3 draft).
#
# Deltas vs the dflash run script (verified against train_eagle3.py + args.py):
#   --draft-config-path        -> --draft-model-config   (arg renamed)
#   DROP --mask-token-id --block-size --num-anchors --loss-decay-gamma (dflash-only)
#   LR 6e-4 -> 1e-4, warmup 0.04 -> 0.015, max-grad-norm 1.0 -> 0.5 (eagle3 defaults)
# Tokenize-cache reuse across runs requires an identical --target-model-path string
# (the cache_key md5 includes it), so keep TARGET_MODEL stable across launches.

if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -euo pipefail

SCRIPT_PATH=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/$(basename "${BASH_SOURCE[0]}")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
ROOT_DIR=$(dirname "$SCRIPT_DIR")

# ---- knobs ----------------------------------------------------------------
NNODES=${NNODES:-4}
NUM_GPUS=${NUM_GPUS:-4}
WORLD=$((NNODES * NUM_GPUS))               # tp = ep = dp = WORLD (attention dp)
MASTER_ADDR=${MASTER_ADDR:-}
MASTER_PORT=${MASTER_PORT:-29500}
BATCH_SIZE=${BATCH_SIZE:-1}
LEARNING_RATE=${LEARNING_RATE:-1e-4}       # eagle3 default (NOT dflash's block-loss-tuned 6e-4)
NUM_EPOCHS=${NUM_EPOCHS:-6}
LOG_INTERVAL=${LOG_INTERVAL:-50}
SAVE_INTERVAL=${SAVE_INTERVAL:-2000}
EVAL_INTERVAL=${EVAL_INTERVAL:-5000}
MEM_FRACTION=${MEM_FRACTION:-0.4}
MAX_RESTARTS=${MAX_RESTARTS:-0}

REPORT_TO=${REPORT_TO:-tensorboard}
TARGET_MODEL=${TARGET_MODEL:-moonshotai/Kimi-K2.7-Code}
DATA_DIR=$ROOT_DIR/cache/dataset/nemotron-post-training-v2
OUTPUT_DIR=$ROOT_DIR/outputs/kimi-k2.7-code-eagle3-nemotron
DRAFT_CONFIG=$ROOT_DIR/configs/kimi-k2.7-code-eagle3.json
# 2048 (not dflash's 4096): eagle3-online runs target_batch_size = tp_size*batch_size
# = 16 samples PER RANK through the target forward (vs dflash's 1 sample/rank sharded
# over WORLD), so 16*4096 tokens of activations OOM the 1T target on 184GB GPUs.
# 2048 halves it to fit. Minor A/B confound vs dflash@4096.
MAX_LENGTH=${MAX_LENGTH:-2048}
CHAT_TEMPLATE=${CHAT_TEMPLATE:-kimi-k2.5-instruct}
ATTENTION_BACKEND=${ATTENTION_BACKEND:-flex_attention}
RDZV_ID=${RDZV_ID:-kimi-k2.7-code-eagle3}

export HF_HOME=${HF_HOME:-/cluster-storage/models}
log() { echo "[$(date -u +%FT%TZ)] $*"; }

cmd_setup() {
  cd "$ROOT_DIR"
  pip install --no-deps -e .
  pip install accelerate tensorboard yunchang qwen-vl-utils
  python3 - <<'PY'
import importlib, sys
miss=[m for m in ["torch","sglang","transformers","datasets","accelerate",
                  "yunchang","flash_attn","deep_ep","tensorboard","specforge"]
      if (importlib.util.find_spec(m) is None)]
if miss: print("PREFLIGHT FAILED, missing:", miss); sys.exit(1)
import torch, sglang
print(f"PREFLIGHT OK  torch={torch.__version__}  sglang={sglang.__version__}")
PY
}

cmd_train() {
  : "${NODE_RANK:?set NODE_RANK=0 (master) or 1..3}"
  [ -n "$MASTER_ADDR" ] || { echo "ERROR: MASTER_ADDR unset"; exit 1; }

  [ "$REPORT_TO" = "wandb" ] && export WANDB_API_KEY=${WANDB_API_KEY:?set WANDB_API_KEY for wandb reporting}
  export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$ROOT_DIR/cache/compiled_kernels}
  export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
  export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
  export SPECFORGE_DATA_NUM_PROC=${SPECFORGE_DATA_NUM_PROC:-64}
  export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-bond0}
  export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-bond0}
  # Do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments (breaks pynccl NVLS alloc).

  if ! python3 -c "import specforge, yunchang, accelerate, sglang, deep_ep" 2>/dev/null; then
    echo "ERROR: training deps missing on this node. Run: $0 setup" >&2; exit 1
  fi

  local tracker=(--report-to "$REPORT_TO")
  [ "$REPORT_TO" = "wandb" ] && tracker+=(--wandb-project specforge-eagle3 --wandb-name kimi-k2.7-code-eagle3)

  mkdir -p "$OUTPUT_DIR"
  log "train(eagle3): node_rank=$NODE_RANK/$NNODES world=$WORLD master=$MASTER_ADDR:$MASTER_PORT \
batch=$BATCH_SIZE lr=$LEARNING_RATE (eff. global batch=$((WORLD * BATCH_SIZE)))"

  torchrun \
    --nnodes "$NNODES" --nproc-per-node "$NUM_GPUS" --node-rank "$NODE_RANK" \
    --rdzv-backend c10d --rdzv-endpoint "$MASTER_ADDR:$MASTER_PORT" \
    --rdzv-id "$RDZV_ID" --max-restarts "$MAX_RESTARTS" \
    "$ROOT_DIR/scripts/train_eagle3.py" \
    --target-model-path "$TARGET_MODEL" \
    --target-model-backend sglang --trust-remote-code \
    --tp-size "$WORLD" \
    `# Pure TP (no dp-attention/EP): eagle3's capture replicates the full batch per` \
    `# rank anyway, so dp-attention buys nothing here but triggers sglang's newer` \
    `# post_forward_mlp_sync_batch on the custom eagle3 logits output (which lacks` \
    `# next_token_logits). require_mlp_sync=False -> that path is skipped. Captured` \
    `# hidden states are identical regardless of target parallelism, so the draft` \
    `# (and the A/B vs dflash) is unaffected.` \
    --sglang-attention-backend flashinfer \
    --sglang-mem-fraction-static "$MEM_FRACTION" \
    --sglang-max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS:-1024}" \
    --draft-model-config "$DRAFT_CONFIG" \
    --embedding-key language_model.model.embed_tokens.weight \
    --lm-head-key language_model.lm_head.weight \
    --train-data-path "$DATA_DIR/nemotron_v2_train.jsonl" \
    --eval-data-path "$DATA_DIR/nemotron_v2_eval_2k.jsonl" \
    --output-dir "$OUTPUT_DIR" --cache-dir "$ROOT_DIR/cache" \
    --num-epochs "$NUM_EPOCHS" --batch-size "$BATCH_SIZE" --learning-rate "$LEARNING_RATE" \
    --warmup-ratio 0.015 --max-grad-norm 0.5 --max-length "$MAX_LENGTH" \
    --chat-template "$CHAT_TEMPLATE" --attention-backend "$ATTENTION_BACKEND" \
    --dataloader-num-workers 0 --log-interval "$LOG_INTERVAL" --save-interval "$SAVE_INTERVAL" --eval-interval "$EVAL_INTERVAL" \
    --dist-timeout 1800 "${tracker[@]}" --resume
}

case "${1:-}" in
  setup)  cmd_setup ;;
  train)  cmd_train ;;
  *) echo "usage: $0 {setup|train}  (launch via launch_eagle3_node.sh watchdog)"; exit 1 ;;
esac
