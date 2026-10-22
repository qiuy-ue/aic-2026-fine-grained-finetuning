# -*- coding: utf-8 -*-
"""
均衡分配（直推）：测试集每类约 50 张这件事，本身就是免费的信息。纯 CPU，不碰 GPU。

═══════════════════════════════════════════════════════════════════════
一、第一性原理：为什么"知道答案的分布"能提高准确率
═══════════════════════════════════════════════════════════════════════

**生活类比**：50 道选择题，你知道 A/B/C/D 各是 25 道。做完发现自己选了 40 个 A、
只有 10 个 D。**即使不知道具体哪题错，你也已经确定"A 里必有 15 个是错的"**——
把你选 A 时最没把握的 15 道改判成 D，期望上一定不亏。
⭐这不是猜，是**约束**：答案分布是硬事实，你的答卷违反了它，违反的地方必然有错。

**本任务的实证**（这两条是全部依据，不是"业界都这么干"）：
    训练集每类 201~213 张，标准差 3.4 → **主办方刻意做成均衡的**
    测试集 24967 / 500 = **49.93** → 同一套构造方式，几乎必然是每类 50

**模型违反了这个约束，而且违反得很厉害**：
    预测分布 min=9  中位=49  max=108  标准差 12.94
    纯随机误差下应有的标准差 = √(50×(1−1/500)) = 7.06
    → ⭐**离散度是纯噪声的 1.83 倍，多出来的全是系统性偏见**

偏见从哪来？细粒度分类里有**"吸引子类"**——长着"通用长相"的类，模型拿不准时
就往那儿倒，于是它吃掉了邻居的样本。这不是随机噪声，**是有方向的，能掰回来**。

═══════════════════════════════════════════════════════════════════════
二、⚠这和"先验校正"不是一回事
═══════════════════════════════════════════════════════════════════════

    先验校正：logit − τ·log(先验)     依赖你猜的先验准不准（软的、逐类平移）
    ⭐均衡分配：在"每类配额 n/K"的**硬约束**下解最优分配（Sinkhorn）
本脚本**两个都跑**，就是为了把它们分开看。

═══════════════════════════════════════════════════════════════════════
三、⛔判据提前写死（含一个专门用来证伪自己的对照）
═══════════════════════════════════════════════════════════════════════

**⭐必做的平凡解释排除**：如果"往任何目标分布上掰都能涨"，那涨的就不是"均衡"这个信息。
所以除了均衡配额，还跑一个**随机偏斜配额**做对照：
    均衡配额 涨、随机配额 跌  → ✅**是"均衡"这条信息在起作用**，可采信
    两个都涨                  → ⛔平凡效应（Sinkhorn 只是在做平滑），⛔不可采信
    均衡配额 不涨             → ⛔假设不成立或模型偏见不是这种形态

**采信规则**：配对检验，b=基准对/新错，c=基准错/新对，Δ=(c−b)/n，SE=√(b+c)/n。
⛔**|Δ| < 2×SE 一律判为零。**

**⭐还有一个上界诊断**：用验证集的**真实**每类张数当配额（作弊，现实拿不到）。
它给出"完美知道分布能拿多少分"的天花板，用来判断均衡假设离最优有多远。

═══════════════════════════════════════════════════════════════════════
四、⚠这把尺子在这个改动上偏保守（好事）
═══════════════════════════════════════════════════════════════════════

  病①（夸大死记硬背）：本改动不碰权重，拿不到"背训练图"的便宜 → 不适用
  病②（对域偏移瞎）：收益大小取决于**模型偏见有多大**。测试集有域偏移，
     模型更弱、偏见**更大** → ⭐本地读数很可能**低估**真实收益。方向是保守的。

⚠但本脚本用的是**线性探针**特征（榜上 59.0），不是微调模型。
⛔**只能用来判断"值不值得做"，不能用来预测涨几分。**

用法（纯 CPU，训练在跑也能安全执行）：
  python scripts/35_balanced_assign.py
"""
import _bootstrap  # noqa: F401
from _bootstrap import banner

import argparse
import json
import math
import os

import numpy as np
from sklearn.model_selection import train_test_split

from configs import paths as P


def sinkhorn(logits, quota, T=1.0, iters=60):
    """把 softmax 概率反复行/列归一化，逼它同时满足『每张图归一到 1』和『每类总量=配额』。

    ⭐这是 Sinkhorn-Knopp。它求的是"在配额约束下总似然最大"的软分配，
    收敛后取 argmax 就是每张图的新预测。iters 够大即收敛，不是玄学超参。
    """
    q = np.asarray(quota, np.float64)
    q = q / q.sum()                              # 目标列分布（和为1）
    Q = np.exp((logits - logits.max(1, keepdims=True)) / T).astype(np.float64)
    Q /= Q.sum()
    n = Q.shape[0]
    for _ in range(iters):
        Q *= (q / np.maximum(Q.sum(0), 1e-300))[None, :]    # 列 → 配额
        Q *= ((1.0 / n) / np.maximum(Q.sum(1), 1e-300))[:, None]   # 行 → 每图 1
    return Q.argmax(1)


def prior_correct(logits, prior, tau):
    """先验校正：把 logit 减去 τ·log(先验)。⚠这是软的逐类平移，不是硬配额。"""
    p = np.asarray(prior, np.float64)
    p = p / p.sum()
    return (logits - tau * np.log(np.maximum(p, 1e-12))[None, :]).argmax(1)


def paired(base, new, y):
    """配对检验：只数两者判断不一致的题，比两个独立准确率相减灵敏得多。"""
    ok0, ok1 = (base == y), (new == y)
    b = int((ok0 & ~ok1).sum())
    c = int((~ok0 & ok1).sum())
    n = len(y)
    d = (c - b) / n
    se = math.sqrt(b + c) / n if (b + c) else 0.0
    return b, c, d, se


def row(name, pred, base, y, note=""):
    b, c, d, se = paired(base, pred, y)
    acc = float((pred == y).mean())
    sig = abs(d) > 2 * se
    tag = ("✅显著更好" if d > 0 else "❌显著更差") if sig else "⚠判为零"
    if base is pred or (pred is base):
        tag = "基准"
    chg = int((pred != base).sum())
    print("  %-30s %8.4f %+9.4f %8.4f %7d %7d %7d  %-10s %s"
          % (name[:30], acc, d, 2 * se, chg, b, c, tag, note))
    return dict(name=name, acc=acc, delta=d, se=se, changed=chg, b=b, c=c, sig=bool(sig))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    banner("35 均衡分配：测试集每类约 50 张 = 免费信息（纯 CPU）")

    feat = np.load(os.path.join(P.CACHE_DIR, "train_features.npy"), mmap_mode="r")
    y_all = np.load(os.path.join(P.CACHE_DIR, "train_labels.npy")).astype(np.int64)
    hd = np.load(os.path.join(P.CACHE_DIR, "head_loss_sce.npz"))
    W, bb = hd["W"].astype(np.float32), hd["b"].astype(np.float32)
    K = W.shape[0]

    _, iva = train_test_split(np.arange(len(y_all)), test_size=0.2,
                              random_state=42, stratify=y_all)
    X = np.array(feat[iva], np.float32)
    if bool(hd["normalize"]):
        X /= np.linalg.norm(X, axis=1, keepdims=True).clip(1e-12)
    y = y_all[iva]
    logits = (X @ W.T + bb).astype(np.float32)
    n = len(y)

    true_q = np.bincount(y, minlength=K).astype(np.float64)
    print("  验证集 %d 张 / %d 类   真实每类：min=%d 中位=%.0f max=%d 标准差=%.2f"
          % (n, K, true_q.min(), np.median(true_q), true_q.max(), true_q.std()))
    base = logits.argmax(1)
    pq = np.bincount(base, minlength=K).astype(np.float64)
    exp_sd = math.sqrt((n / K) * (1 - 1.0 / K))
    print("  模型预测每类：min=%d 中位=%.0f max=%d 标准差=%.2f"
          % (pq.min(), np.median(pq), pq.max(), pq.std()))
    print("  ⭐纯随机误差下应有的标准差 = %.2f  →  实测是它的 **%.2f 倍**（多出来的是系统偏见）"
          % (exp_sd, pq.std() / exp_sd))
    print("  ⛔判据已写死：|Δ| < 2×配对标准误 → 判为零\n")

    uni = np.ones(K)                                   # 均衡配额（现实可用）
    rs = np.random.RandomState(args.seed)
    skew = rs.dirichlet(np.ones(K) * 3.0)              # 随机偏斜配额（对照，必须让它跌）
    skew = skew / skew.sum()

    banner("主表（配对检验；⚠随机偏斜那行是【对照】，它必须变差才说明结论成立）")
    print("  %-30s %8s %9s %8s %7s %7s %7s  %-10s %s"
          % ("方案", "准确率", "配对Δ", "2×SE", "改判数", "改错", "改对", "判定", "备注"))
    print("  " + "-" * 116)
    out = [row("基准 argmax", base, base, y)]

    best_T, best_d = None, -1e9
    for T in (0.02, 0.05, 0.1, 0.2, 0.5, 1.0):
        p = sinkhorn(logits, uni, T=T, iters=args.iters)
        r = row("⭐均衡配额 Sinkhorn T=%.2f" % T, p, base, y)
        out.append(r)
        if r["delta"] > best_d:
            best_T, best_d = T, r["delta"]
    print("  " + "-" * 116)
    p_ctl = sinkhorn(logits, skew, T=best_T, iters=args.iters)
    ctl = row("⛔对照:随机偏斜配额 T=%.2f" % best_T, p_ctl, base, y, "必须变差")
    out.append(ctl)
    p_orc = sinkhorn(logits, true_q, T=best_T, iters=args.iters)
    orc = row("🔮上界:真实配额(作弊) T=%.2f" % best_T, p_orc, base, y, "现实拿不到")
    out.append(orc)
    print("  " + "-" * 116)
    for tau in (0.25, 0.5, 1.0):
        p = prior_correct(logits, pq + 1.0, tau)
        out.append(row("先验校正 τ=%.2f (软,非硬配额)" % tau, p, base, y))

    banner("判读（判据跑之前就写死了）")
    bal = [r for r in out if r["name"].startswith("⭐")]
    bestbal = max(bal, key=lambda r: r["delta"])
    okbal = bestbal["sig"] and bestbal["delta"] > 0
    ctl_bad = (ctl["delta"] < 0 and ctl["sig"]) or (not ctl["sig"] and ctl["delta"] <= 0)

    print("  ① 均衡配额最好一档：Δ=%+.4f  (2×SE=%.4f)  →  %s"
          % (bestbal["delta"], 2 * bestbal["se"], "✅显著" if okbal else "⚠不显著"))
    print("  ② ⛔平凡解释对照（随机偏斜配额）：Δ=%+.4f  →  %s"
          % (ctl["delta"], "✅没涨，很好" if ctl_bad else "❌它也涨了"))
    print("  ③ 🔮完美知道分布的上界：Δ=%+.4f （均衡假设已吃到其中 %.0f%%）"
          % (orc["delta"], 100 * bestbal["delta"] / orc["delta"] if orc["delta"] > 0 else 0))
    print()
    if okbal and ctl_bad:
        print("  ✅⭐**结论成立**：涨分来自『测试集类别均衡』这条真信息，不是平滑的平凡效应。")
        print("     推荐 T=%.2f。⭐纯推理期改动，不碰权重、不违反单模型约束。" % bestbal["name"].split("T=")[1])
        print("     ⚠本地用的是线性探针（榜上59.0），**只信方向不信幅度**；")
        print("     ⚠微调模型偏见更小，收益大概率**小于**这里的读数 → 必须交榜裁决。")
    elif okbal and not ctl_bad:
        print("  ⛔**平凡效应**：往随机分布掰也一样涨 → Sinkhorn 只是在做平滑，")
        print("     ⛔『均衡』这条信息并没有被真正用上，不可采信。")
    else:
        print("  ⛔**没有显著收益** → 要么均衡假设不成立，要么模型偏见不是这种形态。")
        print("     ⛔别在这条路上继续投入。")

    p = os.path.join(P.CACHE_DIR, "balanced_assign.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(dict(n=n, K=K, pred_sd=float(pq.std()), rand_sd=exp_sd, rows=out),
                  fh, ensure_ascii=False, indent=2)
    print("\n  已存 → %s" % p)


if __name__ == "__main__":
    main()
