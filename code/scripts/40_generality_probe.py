# -*- coding: utf-8 -*-
"""
40：**通用性探针** —— 这套配方换到「更多类别 / 长尾分布」还成立吗？用数据回答，不靠讲道理。

═══════════════════════════════════════════════════════════════════════════
为什么可以用线性探针当替身（⭐这次的替身是合格的，理由要说清）
═══════════════════════════════════════════════════════════════════════════
上一次（近邻投票）我用冻结 CLIP 当替身失败了，因为替身在**最关键的中间量**上
就不像真身。这次不同：

  **长尾主要打击的就是分类头**（交叉熵让决策边界偏向头部类），
  而线性探针**本身就是一个分类头**。

→ 替身在被测属性上与真身同构。⚠但**幅度仍不可信**，只看方向和相对排序。

═══════════════════════════════════════════════════════════════════════════
实验设计（⛔跑之前写死，事后不许改）
═══════════════════════════════════════════════════════════════════════════
**实验A 类别数缩放**：K ∈ {100,200,300,400,500}，**每类样本数固定**。
  ⚠不能直接比准确率——K 越大随机基线 1/K 越低，准确率天然下降。
  ⭐要看的是**下降速度**：接近对数级缓降 = 能扩展；断崖 = 撑不住。

**实验B 长尾**：K=500 全用，样本数按指数衰减，不平衡比 imb ∈ {1,10,50,100}。
  ⭐⭐**关键对照**：每个 imb 都配一个「平衡但总样本数相同」的对照组。
     ⛔否则长尾组的性能下降会混进「总样本变少」这个平凡解释，无法归因到不平衡。
  ⭐⭐**测试集始终保持平衡**（模拟真实场景：本赛测试集是 500×50 的均衡集）。
     ⛔若测试集也长尾，「只猜头部类」会得高分，整个测量失去意义。

**实验C 补救措施**（只在最严重的 imb 上跑）：
  ① 基线普通 CE  ② 类别加权 CE  ③ logit adjustment（训练时）  ④ post-hoc 先验修正
  ⭐logit adjustment 是**从贝叶斯最优决策推出来的**，不是启发式：
     训练集先验 π_c 偏斜时，最优决策边界要按 log π_c 平移回去。
  ⚠这与 HX 已否决的「测试集先验校正」**性质完全不同**：
     那个是猜测试集分布（未知、且本赛均衡），这个是修正训练集分布（已知、可精确计算）。
"""
import argparse
import io
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as Fn

# ⚠line_buffering=True 必须加：TextIOWrapper 默认带块缓冲，重定向到文件时
#   输出会全憋在内存里直到进程退出——跑 15 分钟的实验将全程看不到任何进度，
#   中途崩了更是一无所获。
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
DEV = "cpu"                       # ⚠固定 CPU：GPU 正被训练/流水线占着，不去抢


def train_head(Xtr, ytr, K, epochs=40, lr=3e-3, bs=2048, mode="ce",
               prior=None, tau=1.0, seed=0):
    """训练一个线性分类头。mode 决定用哪种长尾补救。"""
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    W = torch.zeros(K, Xtr.shape[1], requires_grad=True)
    torch.nn.init.normal_(W, std=0.01)
    b = torch.zeros(K, requires_grad=True)
    opt = torch.optim.AdamW([W, b], lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    cw = None
    if mode == "cw":                       # 类别加权：按 1/n_c 给权重（再归一化）
        cnt = torch.bincount(ytr, minlength=K).float().clamp(min=1)
        cw = (1.0 / cnt)
        cw = cw / cw.mean()
    logpi = torch.log(prior.clamp(min=1e-12)) if prior is not None else None

    n = len(ytr)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for s in range(0, n, bs):
            i = perm[s:s + bs]
            lg = Xtr[i] @ W.T + b
            if mode == "la":               # logit adjustment：训练时把先验加进 logit
                lg = lg + tau * logpi
            loss = Fn.cross_entropy(lg, ytr[i], weight=cw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()
    return W.detach(), b.detach()


def evaluate(W, b, Xte, yte, K, posthoc_prior=None, tau=1.0):
    with torch.no_grad():
        lg = Xte @ W.T + b
        if posthoc_prior is not None:      # post-hoc：推理时把训练先验减掉
            lg = lg - tau * torch.log(posthoc_prior.clamp(min=1e-12))
        pr = lg.argmax(1)
    acc = (pr == yte).float().mean().item()
    per = torch.zeros(K)
    for c in range(K):
        m = (yte == c)
        per[c] = (pr[m] == c).float().mean().item() if m.any() else float("nan")
    return acc, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-test-per-class", type=int, default=20,
                    help="⭐测试集每类固定张数，保证测试集平衡")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=2, help="每个配置重复几个种子")
    ap.add_argument("--skip-a", action="store_true")
    args = ap.parse_args()

    X = np.load(os.path.join(CACHE, "train_features.npy")).astype(np.float32)
    y = np.load(os.path.join(CACHE, "train_labels.npy")).astype(np.int64)
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    X, y = torch.from_numpy(X), torch.from_numpy(y)
    K_ALL = int(y.max()) + 1
    by = [torch.where(y == c)[0] for c in range(K_ALL)]
    cnt = np.array([len(v) for v in by])
    print("=" * 78)
    print("特征 %s  类别 %d  每类张数 min=%d 中位=%d max=%d"
          % (tuple(X.shape), K_ALL, cnt.min(), int(np.median(cnt)), cnt.max()))
    print("⚠冻结 CLIP 特征 + 线性头。**只看方向与相对排序，幅度不可信**")
    print("=" * 78)

    # 每类先切出固定张数当测试集（⭐测试集恒平衡）
    rng = np.random.default_rng(42)
    te_idx, pool = [], []
    nt = args.n_test_per_class
    for c in range(K_ALL):
        p = by[c][torch.from_numpy(rng.permutation(len(by[c])))]
        te_idx.append(p[:nt])
        pool.append(p[nt:])
    te_idx = torch.cat(te_idx)
    Xte, yte = X[te_idx], y[te_idx]
    print("测试集 %d 张（每类 %d，恒平衡）  训练池每类 min=%d"
          % (len(te_idx), nt, min(len(p) for p in pool)))

    # ───────────────────────── 实验 A：类别数缩放 ─────────────────────────
    if not args.skip_a:
        print("\n" + "=" * 78)
        print("【实验A】类别数 K 变大，性能怎么掉？（每类样本数固定 %d）" % 60)
        print("⭐看的不是准确率绝对值（K越大随机基线越低），是**下降速度**")
        print("=" * 78)
        print("%-8s %10s %12s %14s %12s" % ("类别数K", "准确率", "随机基线", "相对随机倍数", "耗时秒"))
        print("-" * 78)
        NPC = 60
        rowsA = []
        for K in [100, 200, 300, 400, 500]:
            accs = []
            t0 = time.time()
            for sd in range(args.seeds):
                r2 = np.random.default_rng(100 + sd)
                cls = np.sort(r2.choice(K_ALL, K, replace=False))
                remap = {int(c): i for i, c in enumerate(cls)}
                tr = torch.cat([pool[c][:NPC] for c in cls])
                ytr = torch.tensor([remap[int(v)] for v in y[tr]])
                m = torch.isin(yte, torch.from_numpy(cls))
                yt2 = torch.tensor([remap[int(v)] for v in yte[m]])
                W, b = train_head(X[tr], ytr, K, epochs=args.epochs, seed=sd)
                a, _ = evaluate(W, b, Xte[m], yt2, K)
                accs.append(a)
            a = float(np.mean(accs))
            rowsA.append((K, a))
            print("%-8d %10.4f %12.4f %14.1f %12.1f"
                  % (K, a, 1.0 / K, a * K, time.time() - t0))
        print("-" * 78)
        r0, r1 = rowsA[0], rowsA[-1]
        print("⭐K 从 %d→%d（5倍），准确率 %.4f→%.4f，**只掉了 %.1f%%**"
              % (r0[0], r1[0], r0[1], r1[1], 100 * (1 - r1[1] / r0[1])))
        print("   若接近对数级缓降 → 类别数不是瓶颈，扩到 1000/1500 可预期同样缓降。")

    # ───────────────────────── 实验 B：长尾 ─────────────────────────
    print("\n" + "=" * 78)
    print("【实验B】长尾 vs 平衡（⭐每档都配「平衡但总样本数相同」的对照）")
    print("=" * 78)
    print("%-6s %8s %9s %9s %10s %10s %10s" %
          ("不平衡比", "总样本", "长尾acc", "平衡对照", "Δ(归因不平衡)", "头部100类", "尾部100类"))
    print("-" * 78)
    K = K_ALL
    NMAX = min(len(p) for p in pool)      # 头部类张数上限
    rowsB = {}
    for imb in [1, 10, 50, 100]:
        ns = [max(2, int(round(NMAX * imb ** (-c / (K - 1))))) for c in range(K)]
        tr = torch.cat([pool[c][:ns[c]] for c in range(K)])
        ytr = y[tr]
        tot = len(tr)
        nbal = max(2, tot // K)
        trb = torch.cat([pool[c][:nbal] for c in range(K)])
        a_lt = a_bal = 0.0
        per_lt = None
        for sd in range(args.seeds):
            W, b = train_head(X[tr], ytr, K, epochs=args.epochs, seed=sd)
            a, per = evaluate(W, b, Xte, yte, K)
            a_lt += a / args.seeds
            per_lt = per if per_lt is None else per_lt + per
            W2, b2 = train_head(X[trb], y[trb], K, epochs=args.epochs, seed=sd)
            a2, _ = evaluate(W2, b2, Xte, yte, K)
            a_bal += a2 / args.seeds
        per_lt /= args.seeds
        head = float(np.nanmean(per_lt[:100].numpy()))
        tail = float(np.nanmean(per_lt[-100:].numpy()))
        rowsB[imb] = (tr, ytr, tot, a_lt, a_bal, head, tail)
        print("%-6d %8d %9.4f %9.4f %10.4f %10.4f %10.4f"
              % (imb, tot, a_lt, a_bal, a_lt - a_bal, head, tail))
    print("-" * 78)
    print("⭐Δ 才是**不平衡本身**的代价（已扣掉「样本变少」这个平凡解释）")
    print("⭐头部/尾部的差距是长尾的**特征伤口**——总体准确率会掩盖它")

    # ───────────────────────── 实验 C：补救措施 ─────────────────────────
    imb = 100
    tr, ytr, tot, a_lt, _, _, _ = rowsB[imb]
    prior = torch.bincount(ytr, minlength=K).float()
    prior = prior / prior.sum()
    print("\n" + "=" * 78)
    print("【实验C】最严重档 imb=%d 上，四种做法谁能救回来" % imb)
    print("=" * 78)
    print("%-30s %10s %10s %10s %10s" % ("做法", "总准确率", "Δ vs 基线", "头部100类", "尾部100类"))
    print("-" * 78)
    base = None
    for name, mode, ph in [
        ("① 基线：普通交叉熵", "ce", None),
        ("② 类别加权 CE (1/n_c)", "cw", None),
        ("③ logit adjustment(训练时)", "la", None),
        ("④ post-hoc 先验修正(推理时)", "ce", prior),
    ]:
        acc = 0.0
        per_s = None
        for sd in range(args.seeds):
            W, b = train_head(X[tr], ytr, K, epochs=args.epochs, mode=mode,
                              prior=prior, seed=sd)
            a, per = evaluate(W, b, Xte, yte, K, posthoc_prior=ph)
            acc += a / args.seeds
            per_s = per if per_s is None else per_s + per
        per_s /= args.seeds
        if base is None:
            base = acc
        print("%-30s %10.4f %10s %10.4f %10.4f"
              % (name, acc, "—" if acc == base else "%+.4f" % (acc - base),
                 float(np.nanmean(per_s[:100].numpy())),
                 float(np.nanmean(per_s[-100:].numpy()))))
    print("=" * 78)
    print("⭐读法：若 ③/④ 能把尾部类拉回来且总准确率不掉 → 长尾**有便宜的解**，")
    print("   配方可以直接复用，只需在分类头上加一行先验修正。")
    print("⛔若四种都救不回来 → 长尾需要更深的改造（重采样/两阶段训练/专门的损失）。")


if __name__ == "__main__":
    main()
