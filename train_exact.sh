#!/bin/bash
# ============ 精确训练命令（⚡ 力大砖飞放大配方：640px / bs128 / 30ep）============
# 产出：放大配方训练 run（基座配方 448/12ep 的已验证命令见文件末尾注释）
# 注意：GPU 非确定性 → 训练结果统计上同分布但非逐位相同；推理结果以
#       reproduce.sh（权重推理）为准，本脚本用于训练过程复现审计。
# 外部依赖：code/cache/{dup_exclude.txt, head_loss_sce.npz, noise_scores.npz}（均由训练集生成）
set -e
CODE="$(cd "$(dirname "$0")" && pwd)/code"
export PYTHONPATH="$CODE:$CODE/scripts"
export AIC_DATA_ROOT="${AIC_DATA_ROOT:?请先 export AIC_DATA_ROOT 指向复赛数据集（含 train/ 与 test/）}"
cd "$CODE/scripts"

python 28_finetune_v2.py \
  --resolution 640 --epochs 30 --batch-size 128 \
  --lr-backbone 3e-05 --llrd 0.85 --lr-head 0.002 --weight-decay 0.05 \
  --warmup-frac 0.08 --mixup 0.8 --cutmix 1.0 --mix-prob 0.7 --label-smoothing 0.1 \
  --ema-decay 0.9996 --drop-path 0.3 --random-erasing 0.35 --degrade-p 0.875 \
  --cosine-head --margin 0.2 \
  --co-teaching --ct-forget-rate 0.05 --ct-warmup-epochs 1.0 --ct-head-noise 0.02 \
  --ct-protect-small --small-oversample 4.0 --small-frac 0.5 \
  --exclude-list "$CODE/cache/dup_exclude.txt" \
  --head-init "$CODE/cache/head_loss_sce.npz" \
  --no-foreach --no-test-ruler --no-pin-memory --grad-ckpt \
  --aug-scale 0.3 --num-workers 2 --eval-workers 2 --full-data --seed 0

echo "TRAIN_DONE: runs/*_v2_640*/ckpt_ep30.pt（放大配方：约为基座 5–6× 训练量；A100 约 15–20h / 4090D 约 1 天+）"
echo "训练完成后推理出包：bash reproduce.sh（自动读最新 run 的 ckpt_ep30）"

# ---------------------------------------------------------------------------
# 【基座配方（历史，已验证 78.1% 本地一致率）】把上方参数替换为下面这组即可复现：
#   --resolution 448 --epochs 12 --batch-size 32 \
#   --lr-backbone 2e-05 --lr-head 0.001 --warmup-frac 0.05 \
#   --ema-decay 0.9995 --drop-path 0.2 --random-erasing 0.25 （其余参数与上方相同）
