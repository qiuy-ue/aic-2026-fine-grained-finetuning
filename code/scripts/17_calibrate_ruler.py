# -*- coding: utf-8 -*-
"""
修尺子：把验证集按"与训练集的相似度"重加权，使它的分布与【测试集】对齐。

═══════════════════════════════════════════════════════════════════════
一、要修的是什么毛病
═══════════════════════════════════════════════════════════════════════

16 号脚本测出来的事实：

    与训练集相似度 ≥0.98 的图        验证集 5.67%   测试集 0.00%
    与训练集相似度 ≥0.90 的图        验证集 66.89%  测试集 30.84%

原因：官方大概率对 train/test 做了去重，**但训练集内部没去重**。
我们自己按 80/20 一切，成对的重复图被切到两边 → **人造出了一份测试集里不存在的泄漏**。

后果：本地验证准确率 ≈0.70，而真实测试只有 0.61。**两张卷子难度不同，分数没法比。**
更糟的是它**会指反方向**：越是"去拟合训练图片"的方法（LoRA 那类），
在本地占的便宜越大，本地→真实的倍率从 168x 塌到 25x。

═══════════════════════════════════════════════════════════════════════
二、⛔为什么不能简单地"把相似度高的验证样本删掉"
═══════════════════════════════════════════════════════════════════════

实算过（16 号数据）：
    删 sim≥0.95（删16%）→ 剩下准确率 0.680，测试集是 0.61，**还差 7 分，切不够**
    删 sim≥0.90（要删67%）→ 剩下准确率 0.568，**又低于测试集了，切过头**

因为验证集**整体**就比测试集离训练集更近，不存在一个"刚好"的阈值。
一刀切是在"偏高"和"偏低"之间二选一，而不是消除偏差。

═══════════════════════════════════════════════════════════════════════
三、✅正确做法：分布匹配（importance weighting）
═══════════════════════════════════════════════════════════════════════

按相似度分箱，给每个验证样本一个权重：

    w(样本) = 测试集在该箱的占比 / 验证集在该箱的占比

含义：验证集里"过多"的那种样本（高相似的）压低权重，"过少"的抬高权重。
加权后，验证集的相似度分布就和测试集**逐箱对齐**。

⭐为什么用加权而不是抽样：抽样要按最缺的那个箱来定总量，实算只剩约 9900 张（扔掉一半多）；
  加权用上全部 20644 张，统计效率更高。代价是"有效样本量"会缩水，脚本里会算给你看。

⭐⭐验收判据（这是关键，没有它就只是自说自话）：
  加权后的准确率应该**自动落到 0.61~0.62 附近**（测试集真实 0.615，
  再考虑验证集标签本身有 ~2% 噪声会把它压低一点）。
  **落上去 = 这把尺子有物理依据的校准；落不上 = 相似度不是唯一的病因，还得继续找。**

⚠边界：只匹配了"与训练集的相似度"这一个协变量。图像质量、类别难度等其他差异没匹配。
  所以这是**严格的改进，不是保证正确**。

用法:
  python scripts/17_calibrate_ruler.py
  python scripts/17_calibrate_ruler.py --bins 20 --clip 10
"""
import _bootstrap  # noqa: F401
from _bootstrap import banner

import argparse
import json
import os

import numpy as np
from sklearn.model_selection import train_test_split

from configs import paths as P

SEED, VAL_FRAC = 42, 0.2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bins", type=int, default=16, help="相似度分多少箱")
    ap.add_argument("--clip", type=float, default=20.0,
                    help="权重上限。⚠某些箱验证集样本极少，权重会爆炸、"
                         "让少数几个样本主导整个指标 → 必须截断")
    args = ap.parse_args()

    banner("17 修尺子：把验证集重加权，对齐测试集的相似度分布")

    z = np.load(os.path.join(P.CACHE_DIR, "leakage_scores.npz"))
    sv, st, iva = z["val_maxsim"], z["test_maxsim"], z["val_idx"]
    y = np.load(P.F_TRAIN_LABEL).astype(np.int64)
    yv = y[iva]

    # ---------------------------------------------------------------- 分箱
    # 用两个集合【合并】后的分位数当箱边界：保证每个箱里都有足够样本，避免空箱
    edges = np.quantile(np.concatenate([sv, st]), np.linspace(0, 1, args.bins + 1))
    edges[0], edges[-1] = -1.0, 2.0                 # 兜住两端，避免边界样本掉出去
    bv = np.clip(np.digitize(sv, edges) - 1, 0, args.bins - 1)
    bt = np.clip(np.digitize(st, edges) - 1, 0, args.bins - 1)

    pv = np.bincount(bv, minlength=args.bins) / len(sv)
    pt = np.bincount(bt, minlength=args.bins) / len(st)

    # ---------------------------------------------------------------- 权重
    w_bin = np.where(pv > 1e-9, pt / np.maximum(pv, 1e-9), 0.0)
    w_bin = np.minimum(w_bin, args.clip)            # ⚠截断，防少数样本主导
    w = w_bin[bv]
    w = w / w.mean()                                # 归一化到均值 1，方便读

    # ---------------------------------------------------------------- 诊断表
    print("  %-16s %9s %9s %9s %8s" % ("相似度区间", "验证集%", "测试集%", "权重", "样本数"))
    print("  " + "-" * 58)
    for i in range(args.bins):
        n = int((bv == i).sum())
        print("  [%.4f,%.4f) %8.2f%% %8.2f%% %9.2f %8d"
              % (edges[i], edges[i + 1], 100 * pv[i], 100 * pt[i], w_bin[i], n))

    ess = w.sum() ** 2 / np.maximum((w ** 2).sum(), 1e-12)
    print()
    print("  样本数        : %d（全部用上，没有丢弃）" % len(w))
    print("  ⭐有效样本量  : %.0f （= (Σw)²/Σw²，加权后等效于多少个等权样本）" % ess)
    print("    → 单次准确率标准误 ≈ %.4f，两方案配对比较的分辨率还要更细一些"
          % np.sqrt(0.61 * 0.39 / ess))

    # ---------------------------------------------------------------- 验收
    banner("验收：加权后的准确率能不能落到测试集那个水平")
    # ⚠⚠2026-09-05 口径订正：这里读的头一度是 **β=8**（06 号脚本在循环内覆盖同一文件名，
    #   最后一个配置胜出），而上榜 59.0 的是 β=1。已修 06 并重建，此处现在是正确的 β=1。
    #   旧数字（β=8）：原始 0.6980 → 加权 0.6080，曾报"偏差缩小 83%"。⛔那个 83% 作废。
    hd = np.load(os.path.join(P.CACHE_DIR, "head_loss_sce.npz"))
    X = np.load(P.F_TRAIN_FEAT).astype(np.float32)
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    corr = (np.argmax(X[iva] @ hd["W"].T + hd["b"], 1) == yv).astype(np.float64)

    raw = corr.mean()
    wtd = float((corr * w).sum() / w.sum())
    print("  用 SCE 头（这一版在排行榜上是 59.0 分，TTA 版 61.357）")
    print()
    print("  %-28s %8s" % ("指标", "准确率"))
    print("  %-28s %8.4f" % ("原始验证集（等权）", raw))
    print("  %-28s %8.4f  ⭐" % ("分布匹配后（加权）", wtd))
    print("  %-28s %8.4f" % ("真实测试集（排行榜，无TTA）", 0.590))
    print()
    print("  差距：加权后 vs 真实 = %+.4f （修之前是 %+.4f）" % (wtd - 0.590, raw - 0.590))
    shrink = (abs(raw - 0.590) - abs(wtd - 0.590)) / max(abs(raw - 0.590), 1e-9)
    print("  ⭐偏差缩小了 %.0f%%" % (100 * shrink))
    print()
    print("  ⚠⚠**这个百分比是偏乐观的**，有一处口径不对等（P0-2）：")
    print("    · 上榜 0.590 那个头是 refit_full——在【全部 103218 张】上重训，**见过验证集**")
    print("    · 这里评估的头只在 itr（82574 张）上训，少 25%% 训练数据")
    print("    → itr-only 头的真实测试分**低于 0.590**，所以真实差距比上面算的更大。")
    print("    ⛔别对外引用精确的百分比，说'缩小约四分之三'即可。")
    print()
    if abs(wtd - 0.590) < 0.02:
        print("  ✅**落上去了**：相似度差异基本能解释掉那 9 分缺口。")
        print("    → 这把尺子现在有**物理依据**的校准，不是碰运气凑出来的。")
        print("    → ⭐'逐层解冻'那条被封的路重新可评估了。")
    elif shrink > 0.4:
        print("  ⚠**修掉了大部分但没完全修好**：相似度是主因之一，但不是全部。")
        print("    剩余缺口的候选：测试集图像本身更难/来源不同、类别分布不同(TV=0.187 已坐实)。")
    else:
        print("  ❌**没修好**：相似度不是主要病因，得另找。别拿这个权重当准。")

    # ---------------------------------------------------------------- 存
    out = os.path.join(P.CACHE_DIR, "val_weights.npz")
    np.savez(out, val_idx=iva, weight=w, bin_edges=edges, bin_weight=w_bin,
             ess=ess, acc_raw=raw, acc_weighted=wtd)
    with open(os.path.join(P.CACHE_DIR, "ruler_report.json"), "w", encoding="utf-8") as f:
        json.dump({"bins": args.bins, "clip": args.clip, "ess": float(ess),
                   "acc_raw": float(raw), "acc_weighted": float(wtd),
                   "acc_leaderboard_no_tta": 0.590}, f, ensure_ascii=False, indent=2)
    print("\n  权重已存 → %s" % out)
    print("  ⭐以后所有本地评估，用 (correct * w).sum()/w.sum() 代替 correct.mean()")


if __name__ == "__main__":
    main()
