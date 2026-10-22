#!/bin/bash
# ============ 一键复现：权重 → 提交包（77.655% 形态） ============
# 前置：DATA_ROOT 指向复赛数据集（configs/paths.py 自动探测，或 export AIC_DATA_ROOT=...）
# 运行位置：本脚本可在任意目录执行；内部自动切到 代码/scripts。
set -e

HERE="$(cd "$(dirname "$0")" && pwd)"
CODE="$HERE/code"
CKPT="${AIC_CKPT:?请设置 AIC_CKPT 指向你的训练检查点（本仓库不含模型权重）}"
VIEWS="orig,mild0,mild1,mild2,mild3,mild4,cc085,cc070,cc055,cc040"

[ -f "$CKPT" ] || { echo "⛔ 找不到权重: $CKPT"; exit 1; }
[ -f "$CODE/cache/dup_exclude.txt" ] || { echo "⛔ 缺少 代码/cache/dup_exclude.txt"; exit 1; }

export PYTHONPATH="$CODE:$CODE/scripts"
cd "$CODE/scripts"

echo "[1/3] 544 池（10 视角，~40 分钟）"
python 58_test_viewpool.py --ckpt "$CKPT" --res 544 --views "$VIEWS" --num-workers 4

echo "[2/3] 448 池（~25 分钟）"
python 58_test_viewpool.py --ckpt "$CKPT" --res 448 --views "$VIEWS" --num-workers 4

echo "[3/3] 等权聚合 → Sinkhorn 软配额校准"
P544=$(ls -td ../runs/*_pool544_ep* | head -1)/pool.npz
P448=$(ls -td ../runs/*_pool448_ep* | head -1)/pool.npz
LOGITS="$CODE/runs/_reproduce_eq_t1_logits.npy"

python 59_recipe_pack.py --pool "$P544" --pool-lo "$P448" \
  --recipe custom --views "$VIEWS" --recipe-lo custom --views-lo "$VIEWS" \
  --save-logits "$LOGITS" --tag reproduce_eq_t1

PKG=$(ls -td ../runs/*_rcp_reproduce_eq_t1* | head -1)
cp "$LOGITS" "$PKG/test_logits.npy"
# 74 号的 balance.json 记录源包 zip 的 md5；59 号写的是 pred_results.zip（下划线），
# 这里补一份连字符名，保证 74 号可直接消费。
cp "$PKG/pred_results.zip" "$PKG/pred-results.zip"

python 74_balance_pack.py --pkg "$PKG"
BAL=$(ls -td ../runs/*_bal | head -1)

echo "DONE: 提交包 = $BAL/pred-results.zip（37,444 行；本地一致率用 score_local.py 复核）"
