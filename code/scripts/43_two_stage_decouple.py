# -*- coding: utf-8 -*-
"""
43：**两阶段解耦**补测——40 号留下的缺口，也是 HX 那条一等约束的直接检验。

═══════════════════════════════════════════════════════════════════════════
要回答的问题
═══════════════════════════════════════════════════════════════════════════
40 号已证：长尾下这套配方会崩（imb=100 时尾部 100 类准确率 **0.0000**），
四种单阶段补丁最多把总 acc 从 0.1662 救到 0.3187，但**尾部只到 0.0172**。

40 号同时给了一条关键线索：长尾训练下**头部 100 类 0.6120，比平衡对照的整体
0.4948 还高** → 特征骨干在长尾数据上**学得好好的，坏掉的是分类头**。

→ 若这条线索为真，那么修法就是 HX 要的字面形态：**只重训分类头**。
   本脚本就是去验证它。

═══════════════════════════════════════════════════════════════════════════
⛔替身合格性：40 号那套替身在这个问题上**不合格**，必须改造
═══════════════════════════════════════════════════════════════════════════
40 号的替身 = 冻结 CLIP 特征 + 线性头。但两阶段解耦的定义是
「stage1 训特征 → **冻结特征** → stage2 重训头」。
⛔在 40 号的替身里，特征是冻结 CLIP 抽好的，**stage1 根本改不动它**
   → "冻结特征重训头" ≡ "直接用平衡数据训头"，整个实验退化成一句废话。

**修法：加一个会被 stage1 改变、被 stage2 冻结的表示层。**

⚠⚠ 而且这个表示层**必须带非线性**，这不是调参而是正确性问题：
   若 rep 是纯线性 W_r，则 head(rep(x)) = (W_h W_r) x 仍是**一个线性映射**。
   于是「冻结 W_r 重训 W_h」的可达函数集合 ⊂ 「直接训一个线性头」的集合，
   两阶段永远不可能赢——⛔实验会给出一个由代数结构决定的假阴性。
   → 用 Linear(512→512) + GELU。

═══════════════════════════════════════════════════════════════════════════
⭐预注册对照（跑之前写死）
═══════════════════════════════════════════════════════════════════════════
⑤ 两阶段（平衡重采样重训头）—— 真实配置
⑥ 两阶段（**长尾**重训头）    —— ⭐必须有：隔离「平衡」这一个变量。
   否则无法区分「平衡起了作用」和「多训一轮头就能涨」。
⑦ oracle 特征探针 —— 用 stage1 学到的 rep + **平衡数据训头**，
   直接读出「stage1 的特征到底坏没坏」。这是本实验真正的诊断量。
"""
import io
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
DEV = "cpu"                      # ⚠GPU 正在出 384+TTA 包，不去抢


class Net(nn.Module):
    """rep（会被 stage1 改变、被 stage2 冻结）+ head（stage2 重训的对象）。"""

    def __init__(self, d, K):
        super().__init__()
        self.rep = nn.Sequential(nn.Linear(d, d), nn.GELU())   # ⚠非线性是必需的，见文件头
        self.head = nn.Linear(d, K)

    def forward(self, x):
        return self.head(self.rep(x))


def fit(model, X, y, K, epochs, lr, bs=1024, balanced=False, params=None, seed=0):
    """balanced=True → **类平衡重采样**（每个 epoch 每类抽同样多张，少的类重复抽）。"""
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    ps = params if params is not None else list(model.parameters())
    opt = torch.optim.AdamW(ps, lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n = len(y)
    if balanced:
        by = [torch.where(y == c)[0] for c in range(K)]
        per = max(1, n // K)
    for _ in range(epochs):
        if balanced:
            # ⭐平衡重采样：尾部类被**重复**抽到，直到与头部类等量。
            #   第一性原理：交叉熵按样本求和，头部 2000 张 vs 尾部 20 张
            #   → 头部话语权 100 倍。重采样是直接把话语权掰回 1:1。
            idx = torch.cat([b[torch.randint(len(b), (per,), generator=g)] for b in by])
            idx = idx[torch.randperm(len(idx), generator=g)]
        else:
            idx = torch.randperm(n, generator=g)
        for s in range(0, len(idx), bs):
            i = idx[s:s + bs]
            loss = Fn.cross_entropy(model(X[i]), y[i])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sch.step()
    return model


@torch.no_grad()
def ev(model, X, y, K):
    pr = model(X).argmax(1)
    acc = (pr == y).float().mean().item()
    per = torch.tensor([(pr[y == c] == c).float().mean().item() for c in range(K)])
    return acc, float(per[:100].mean()), float(per[-100:].mean())


def main():
    t0 = time.time()
    X = np.load(os.path.join(CACHE, "train_features.npy")).astype(np.float32)
    y = np.load(os.path.join(CACHE, "train_labels.npy")).astype(np.int64)
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    X, y = torch.from_numpy(X), torch.from_numpy(y)
    K = int(y.max()) + 1
    d = X.shape[1]

    # 与 40 号完全同一套构造（seed=42），⭐测试集恒平衡
    rng = np.random.default_rng(42)
    by = [torch.where(y == c)[0] for c in range(K)]
    te, pool = [], []
    for c in range(K):
        p = by[c][torch.from_numpy(rng.permutation(len(by[c])))]
        te.append(p[:20])
        pool.append(p[20:])
    te = torch.cat(te)
    Xte, yte = X[te], y[te]
    NMAX = min(len(p) for p in pool)

    IMB = 100
    ns = [max(2, int(round(NMAX * IMB ** (-c / (K - 1))))) for c in range(K)]
    tr = torch.cat([pool[c][:ns[c]] for c in range(K)])
    Xtr, ytr = X[tr], y[tr]
    # 平衡对照：**总样本数相同**，⛔否则性能差会混进"样本变少"这个平凡解释
    nb = max(2, len(tr) // K)
    trb = torch.cat([pool[c][:nb] for c in range(K)])

    print("=" * 92)
    print("43 两阶段解耦   imb=%d   长尾训练 %d 张（头部%d/尾部%d）  平衡对照 %d 张"
          % (IMB, len(tr), ns[0], ns[-1], len(trb)))
    print("测试集 %d 张（每类 20，恒平衡）  ⭐替身已加非线性 rep 层，见文件头" % len(te))
    print("=" * 92)
    print("%-34s %9s %10s %11s %s" % ("做法", "总acc", "头部100类", "尾部100类", "说明"))
    print("-" * 92)

    E1, E2, LR = 30, 30, 3e-3

    # ── ① 单阶段基线：长尾数据直接训（rep+head 一起）
    m1 = fit(Net(d, K), Xtr, ytr, K, E1, LR)
    a, h, t = ev(m1, Xte, yte, K)
    print("%-34s %9.4f %10.4f %11.4f %s" % ("① 单阶段 CE（长尾）", a, h, t, "= stage1"))
    base = a

    # ── ② 上限参照：平衡数据（同总量）训
    m2 = fit(Net(d, K), X[trb], y[trb], K, E1, LR)
    ab, hb, tb = ev(m2, Xte, yte, K)
    print("%-34s %9.4f %10.4f %11.4f %s" % ("② 平衡对照（同总量）", ab, hb, tb, "⭐上限参照"))

    # ── ③ 单阶段 + 平衡重采样（⛔不是两阶段，是同一次训练里就平衡）
    m3 = fit(Net(d, K), Xtr, ytr, K, E1, LR, balanced=True)
    a3, h3, t3 = ev(m3, Xte, yte, K)
    print("%-34s %9.4f %10.4f %11.4f %s"
          % ("③ 单阶段+平衡重采样", a3, h3, t3, "%+.4f" % (a3 - base)))

    # ── ⑤ ⭐两阶段：冻结 ① 的 rep，用平衡重采样**只重训 head**
    def stage2(src, bal, tag, note):
        m = Net(d, K)
        m.rep.load_state_dict(src.rep.state_dict())
        for p in m.rep.parameters():
            p.requires_grad_(False)
        m.rep.eval()
        fit(m, Xtr, ytr, K, E2, LR, balanced=bal, params=list(m.head.parameters()))
        a_, h_, t_ = ev(m, Xte, yte, K)
        print("%-34s %9.4f %10.4f %11.4f %s"
              % (tag, a_, h_, t_, "%+.4f  %s" % (a_ - base, note)))
        return a_, t_

    a5, t5 = stage2(m1, True, "⑤ ⭐两阶段（平衡重训头）", "真实配置")
    a6, t6 = stage2(m1, False, "⑥ 两阶段（长尾重训头）", "⭐对照：隔离'平衡'")

    # ── ⑦ oracle 特征探针：stage1 的 rep + **平衡数据**训头
    #    ⭐这才是本实验真正的诊断量：直接读出「stage1 学到的特征坏没坏」
    m7 = Net(d, K)
    m7.rep.load_state_dict(m1.rep.state_dict())
    for p in m7.rep.parameters():
        p.requires_grad_(False)
    m7.rep.eval()
    fit(m7, X[trb], y[trb], K, E2, LR, params=list(m7.head.parameters()))
    a7, h7, t7 = ev(m7, Xte, yte, K)
    print("%-34s %9.4f %10.4f %11.4f %s"
          % ("⑦ ⭐oracle：长尾特征+平衡训头", a7, h7, t7, "诊断量"))

    # ══════════════════════════════ 判读
    print("=" * 92)
    print("⭐① vs ⑦ ——「长尾把特征学坏了多少」")
    print("   ⑦ 用的**就是 ① 学出来的特征**，只是换了平衡数据训头。")
    print("   ⑦ %.4f  vs  ② 平衡从头训 %.4f  → 差距 %+.4f" % (a7, ab, a7 - ab))
    if a7 >= ab - 0.02:
        print("   ✅**特征没坏**（追平甚至超过从头平衡训）→ ⭐⭐坏的确实只有分类头，")
        print("     HX 那条「只改分类头就能复用」的约束**成立**。")
    else:
        print("   ⛔**特征也坏了**（比平衡从头训低 %.4f）→ 光改分类头不够，" % (ab - a7))
        print("     长尾场景下骨干训练本身也要动（重采样/重加权要进 stage1）。")

    print("\n⭐⑤ vs ⑥ ——「涨的是'平衡'还是'多训一轮头'」")
    print("   ⑤ 平衡重训 %.4f  vs  ⑥ 长尾重训 %.4f  → %+.4f" % (a5, a6, a5 - a6))
    print("   尾部100类：⑤ %.4f  vs  ⑥ %.4f" % (t5, t6))
    if a5 - a6 > 0.02:
        print("   ✅**是'平衡'在起作用**，不是多训一轮的平凡效应。")
    else:
        print("   ⛔**拉不开** → ⑤ 的增益不能归因到平衡重采样。")

    print("\n⭐⑤ vs ③ ——「两阶段 比 单阶段内平衡 强吗」")
    print("   ⑤ %.4f  vs  ③ %.4f  → %+.4f" % (a5, a3, a5 - a3))
    print("   ⭐第一性原理：特征学习要「数据越多越好」（含头部海量样本），")
    print("     决策边界要「各类一样多」——**两个需求在数学上打架**，")
    print("     同一次训练里没法同时满足。两阶段就是拆开各取所需。")
    print("     → 若 ⑤ > ③，说明这个「打架」是真实的，拆开确有收益。")
    print("\n⚠边界：替身是「冻结CLIP特征 + 可训练 rep 层」，rep 只有 512×512 一层，")
    print("  而真身骨干是 8800 万参数的 ViT。**只信方向与相对排序，幅度不可外推。**")
    print("\n总耗时 %.1f 分钟" % ((time.time() - t0) / 60))


if __name__ == "__main__":
    main()
