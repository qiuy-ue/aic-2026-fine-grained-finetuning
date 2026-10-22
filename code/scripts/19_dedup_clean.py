# -*- coding: utf-8 -*-
"""
用【重复图 + 标签冲突】清洗训练集 —— 一个不依赖模型判断的机械式标签错误检测器。

═══════════════════════════════════════════════════════════════════════
一、和之前那次失败的清洗有什么本质不同
═══════════════════════════════════════════════════════════════════════

上一次清洗（08号，kNN 可疑度，soft@20%）在排行榜上 **−1.3 分**，是失败的。
根因已查明：**可疑度分不清"难"和"错"**。它的判据是"邻居的标签跟我不一样"，
但邻居不一样有两种原因：①标签真错了 ②标签完全正确、只是这两个物种照片上就分不开。
实测证明②大量存在（0467↔0156 对称双向混淆），把它们一起洗掉 = **亲手注入新噪声**。

这次的判据完全不同：

    ⭐同一张图出现两次，却挂着两个不同的标签 → 至少有一个是错的。

**这是逻辑事实，不需要模型判断，也不存在"难 vs 错"的混淆。**
16号脚本已实证：相似度≈1.0 的那批里 74% 是跨类别的，而紧邻的下一档只有 6.3%——
是断崖不是渐变；并已排除"退化灰图"假设（组内两两相似度 0.6085，与随机组 0.6017 一样）。

═══════════════════════════════════════════════════════════════════════
二、找到冲突之后，删哪一张？
═══════════════════════════════════════════════════════════════════════

我们只知道"至少一个错"，**不知道是哪一个**。三种处理，脚本都实现了：

  both   两张都删。⭐保守但干净：宁可损失一张好样本，也不留一张坏的。
  none   都不删，只统计。用作对照组（证明收益确实来自删除，不是别的）。
  ⚠不做"猜哪张对然后改标签"——那就退化回上次失败的老路（拿模型判断当真相）。

⚠代价要说清楚：删 both 会连带删掉正确的那一张。若冲突对里恰好各有一半是对的，
  删 2 张换掉 1 个错标 —— **这个交易划不划算，正是本脚本要用尺子量出来的**。

═══════════════════════════════════════════════════════════════════════
三、用哪把尺子量
═══════════════════════════════════════════════════════════════════════

18号回考的结论，两把一起用：
  · 判方向：干净60% 与 分布匹配 **一致**才可信；打架 → 本地无效，必须交榜
  · 判幅度：分布匹配 **×100 直接读成分数**（冻结特征类会系统低估约 0.5 分）
⚠⚠**验证集里也有冲突样本**。如果只清训练集、不清验证集，验证集里那些
  "标签本来就错"的样本会一直判模型错，压低所有方案的分数——但**对所有方案一视同仁**，
  所以横向比较仍然成立。脚本会额外报一个"验证集也去冲突"的版本作参考。

用法:
  python scripts/19_dedup_clean.py                 # 阈值 0.99，删 both
  python scripts/19_dedup_clean.py --thr 0.995
  python scripts/19_dedup_clean.py --mode none     # 对照组：只统计不删
"""
import _bootstrap  # noqa: F401
from _bootstrap import banner

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as Fn
from sklearn.model_selection import train_test_split

from configs import paths as P

SEED, VAL_FRAC = 42, 0.2


def find_dup_pairs(X, device, thr, chunk=2048):
    """
    对每一行找它在全库里最像的另一行（排除自己）。返回 (nn_idx, nn_sim)。
    ⚠只取 top-1：三份以上的重复只能捕获部分，作为下界够用了。
    ⚠分块 chunk×N×4B；N=10.3万时 2048 行 ≈ 845MB，8GB 显存安全。
    """
    Xt = torch.from_numpy(X).to(device)
    N = len(X)
    nn_i = np.zeros(N, dtype=np.int64)
    nn_s = np.zeros(N, dtype=np.float32)
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        sim = Xt[s:e] @ Xt.T
        sim[torch.arange(e - s), torch.arange(s, e)] = -1.0   # 排除自己
        v, i = sim.max(1)
        nn_s[s:e] = v.cpu().numpy()
        nn_i[s:e] = i.cpu().numpy()
        del sim
    del Xt
    torch.cuda.empty_cache()
    return nn_i, nn_s


def train_sce(Xtr, ytr, K, C=10.0, n_iter=300, lr=0.05, device="cuda", beta=1.0):
    """与 06/18 同一套 SCE + Adam + 1/(2Cn)。换任何一样都没法跟历史比。"""
    n = len(ytr)
    Xt = torch.from_numpy(Xtr).to(device)
    yt = torch.from_numpy(ytr).to(device)
    W = torch.zeros(K, Xtr.shape[1], device=device, requires_grad=True)
    b = torch.zeros(K, device=device, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr)
    reg = 1.0 / (2.0 * C * n)
    for _ in range(n_iter):
        opt.zero_grad(set_to_none=True)
        lg = Xt @ W.T + b
        p = Fn.softmax(lg, 1).clamp(1e-7, 1.0)
        oh = Fn.one_hot(yt, K).float()
        L = Fn.cross_entropy(lg, yt) - beta * (p * ((1 - oh) * -4.0)).sum(1).mean()
        (L + reg * (W * W).sum()).backward()
        opt.step()
    return W.detach().cpu().numpy(), b.detach().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thr", type=float, default=0.99, help="判定为'同一张图'的余弦相似度阈值")
    ap.add_argument("--mode", choices=["both", "none"], default="both")
    ap.add_argument("--chunk", type=int, default=2048)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    banner("19 重复图冲突清洗（阈值 %.3f，模式 %s）" % (args.thr, args.mode))

    X = np.load(P.F_TRAIN_FEAT).astype(np.float32)
    y = np.load(P.F_TRAIN_LABEL).astype(np.int64)
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    K = int(y.max()) + 1
    N = len(y)

    # ---------------------------------------------------------------- 找重复
    print("  全库扫描 %d 张两两相似度（分块 %d 行）..." % (N, args.chunk))
    nn_i, nn_s = find_dup_pairs(X, device, args.thr, args.chunk)

    is_dup = nn_s >= args.thr
    cross = is_dup & (y != y[nn_i])            # 重复 且 标签不同 = 冲突
    same = is_dup & (y == y[nn_i])
    banner("扫描结果")
    print("  %-34s %8d  (%.2f%%)" % ("有近重复副本的图", is_dup.sum(), 100 * is_dup.mean()))
    print("  %-34s %8d  (%.2f%%)  ← 无害，只是冗余"
          % ("├ 副本同类", same.sum(), 100 * same.mean()))
    print("  %-34s %8d  (%.2f%%)  ⭐至少一个标签是错的"
          % ("└ 副本【异类】=冲突", cross.sum(), 100 * cross.mean()))
    print()
    print("  → 冲突涉及约 %d 对，即**至少 %d 个错标**，占全库 %.2f%%"
          % (cross.sum() // 2, cross.sum() // 2, 50.0 * cross.mean()))
    print("  ⚠这是下界：只能抓到'恰好有重复副本'的那部分错标，没副本的抓不到。")
    print("  ⭐参照：HX 人工数第0000类得 3/206 = 1.46%%，两种独立方法量级一致。")

    # ---------------------------------------------------------------- 划分
    idx = np.arange(N)
    itr, iva = train_test_split(idx, test_size=VAL_FRAC, random_state=SEED, stratify=y)
    Xv, yv = X[iva], y[iva]

    wz = np.load(os.path.join(P.CACHE_DIR, "val_weights.npz"))
    w = wz["weight"]
    nz = np.load(os.path.join(P.CACHE_DIR, "noise_scores.npz"), allow_pickle=True)
    m60 = nz["suspicion"][iva] <= np.quantile(nz["suspicion"][iva], 0.60)
    vcross = cross[iva]                         # 验证集里的冲突样本

    def measure(W_, b_):
        c = (np.argmax(Xv @ W_.T + b_, 1) == yv).astype(np.float64)
        return dict(clean60=float(c[m60].mean()),
                    matched=float((c * w).sum() / w.sum()),
                    matched_nodup=float((c[~vcross] * w[~vcross]).sum()
                                        / w[~vcross].sum()),
                    raw=float(c.mean()))

    # ---------------------------------------------------------------- 对照
    banner("对照实验：清洗前 vs 清洗后（其余一切不变）")
    base = measure(*train_sce(X[itr], y[itr], K, device=device))
    print("  基线（不清洗）      训练 %d 张" % len(itr))

    keep = itr if args.mode == "none" else itr[~cross[itr]]
    print("  清洗后              训练 %d 张（删了 %d 张，%.2f%%）"
          % (len(keep), len(itr) - len(keep), 100 * (1 - len(keep) / len(itr))))
    new = measure(*train_sce(X[keep], y[keep], K, device=device))

    banner("结果")
    hdr = "  %-26s %10s %10s %10s" % ("尺子", "基线", "清洗后", "差值")
    print(hdr)
    print("  " + "-" * (len(hdr) + 4))
    rows = {}
    for key, nm in (("clean60", "干净60%（老尺子）"),
                    ("matched", "⭐分布匹配（新尺子）"),
                    ("matched_nodup", "分布匹配+验证集也去冲突"),
                    ("raw", "原始等权（参考）")):
        d = new[key] - base[key]
        rows[key] = d
        print("  %-26s %10.4f %10.4f %+10.4f" % (nm, base[key], new[key], d))

    banner("判读")
    agree = np.sign(rows["clean60"]) == np.sign(rows["matched"])
    print("  方向：干净60%% %+.4f   分布匹配 %+.4f  → %s"
          % (rows["clean60"], rows["matched"],
             "✅一致，可信" if agree else "⚠打架 → 本地无效，必须交榜裁决"))
    print("  幅度：分布匹配 ×100 = **%+.2f 分**（冻结特征类系统低估约 0.5 分，可上修）"
          % (rows["matched"] * 100))
    print()
    print("  ⚠有效样本量 11730 → 单次标准误约 0.0045；")
    print("    差值小于约 0.004 时本地分辨不出来，别当真。")

    out = os.path.join(P.CACHE_DIR, "dedup_clean.npz")
    np.savez(out, nn_idx=nn_i, nn_sim=nn_s, cross=cross, same=same, thr=args.thr)
    with open(os.path.join(P.CACHE_DIR, "dedup_report.json"), "w", encoding="utf-8") as f:
        json.dump({"thr": args.thr, "mode": args.mode,
                   "n_dup": int(is_dup.sum()), "n_cross": int(cross.sum()),
                   "n_same": int(same.sum()), "base": base, "cleaned": new,
                   "delta": rows}, f, ensure_ascii=False, indent=2)
    print("\n  已存 → %s" % out)


if __name__ == "__main__":
    main()
