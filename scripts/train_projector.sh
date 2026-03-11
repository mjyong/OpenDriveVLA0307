#!/bin/bash
# ============================================================
# DriveVLA Projector-Only Training Script
# ============================================================
# 只训练 3 个 multimodal projector:
#   - mm_projector_scene  (img_feat_2D → LLM hidden)
#   - mm_projector_track  (track_query → LLM hidden)
#   - mm_projector_map    (map_query   → LLM hidden)
#
# 冻结:
#   - LLM (Qwen2.5-0.5B)
#   - Vision Tower (UniAD: ResNet101 + BEV + TrackHead + SegHead)
#
# 用法:
#   bash scripts/train_projector.sh [CKPT_PATH] [NUM_GPU]
#   bash scripts/train_projector.sh checkpoints/DriveVLA-Qwen2.5-0.5B-Instruct 4
# ============================================================

set -e

CKPT_PATH=${1:-"checkpoints/DriveVLA-Qwen2.5-0.5B-Instruct"}
NUM_GPU=${2:-1}
DATA_PATH=${3:-""}  # 留空则使用在线生成

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="output/train_projector/${TIMESTAMP}"

mkdir -p ${OUTPUT_DIR}/logs

echo "============================================================"
echo "  DriveVLA Projector-Only Training"
echo "============================================================"
echo "  CKPT_PATH  : ${CKPT_PATH}"
echo "  NUM_GPU    : ${NUM_GPU}"
echo "  DATA_PATH  : ${DATA_PATH:-'(online generation)'}"
echo "  OUTPUT_DIR : ${OUTPUT_DIR}"
echo "============================================================"

# ---- 训练参数 ----
BATCH_SIZE=1
GRAD_ACCUM=16
EPOCHS=3
LR=1e-4
WARMUP_RATIO=0.05

# 有效 batch size = BATCH_SIZE * GRAD_ACCUM * NUM_GPU
EFFECTIVE_BS=$((BATCH_SIZE * GRAD_ACCUM * NUM_GPU))
echo "  Effective batch size: ${EFFECTIVE_BS}"
echo "============================================================"

# ---- 构建数据参数 ----
DATA_ARGS=""
if [ -n "${DATA_PATH}" ]; then
    DATA_ARGS="--data_path ${DATA_PATH}"
fi

# ---- 启动训练 ----
PYTHONPATH="$(pwd)":$PYTHONPATH \
torchrun --nproc_per_node=${NUM_GPU} \
    drivevla/train_projector.py \
    --model_path ${CKPT_PATH} \
    --attn_implementation sdpa \
    ${DATA_ARGS} \
    --use_uniad_pth True \
    --in_nuscenes_order True \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs ${EPOCHS} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --learning_rate ${LR} \
    --warmup_ratio ${WARMUP_RATIO} \
    --lr_scheduler_type cosine \
    --weight_decay 0.01 \
    --bf16 True \
    --logging_steps 10 \
    --save_steps 500 \
    --save_total_limit 3 \
    --dataloader_num_workers 4 \
    --remove_unused_columns False \
    --gradient_checkpointing False \
    --report_to tensorboard \
    --ddp_find_unused_parameters False \
    2>&1 | tee ${OUTPUT_DIR}/logs/train.log

echo ""
echo "============================================================"
echo "  Training complete!"
echo "  Output: ${OUTPUT_DIR}"
echo "  Log:    ${OUTPUT_DIR}/logs/train.log"
echo "============================================================"
