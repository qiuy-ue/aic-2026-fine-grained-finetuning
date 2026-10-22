# -*- coding: utf-8 -*-
"""
41：**训练集标签噪声诊断**——赛题本名就叫「噪声标签细粒度」，而我们至今没直接碰过训练标签。
    先花 30 分钟 CPU 决定要不要投 8 小时 GPU。

═══════════════════════════════════════════════════════════════════════════
为什么现在做这件事（不是"该试试了"，是有具体缺口）
═══════════════════════════════════════════════════════════════════════════
榜上 73.64，前十 85~92。伪标签甜点已探明是倒 U 形、峰在 ep02、全程只赚 +0.88
（ep02=73.64 → ep04=72.89，实测在掉）。测试几何也已探明：裁剪判为零、
分辨率 384 之后收益消失。→ **零碎优化的量级补不上 16 分的洞，必须找主路。**

⚠**唯一试过的相关动作是"删可疑样本"，结果 −1.3 分。但删 ≠ 改**：
   删掉是把信息连同噪声一起扔了；软化是把权重降下来但信息保留。
   ⛔不能用那次失败否掉这次——两者在信息论上是不同的操作。

═══════════════════════════════════════════════════════════════════════════
⭐⭐实验设计的命门：为什么必须有「打乱对照」
═══════════════════════════════════════════════════════════════════════════
"把硬标签软化"这个动作**本身就带正则化效果**——它就是 label smoothing 的一般形式。
所以**哪怕软化的方向纯粹是噪声，准确率也可能涨**。

→ 要区分下面这两件事，只能靠对照：
     (a) 模型的第二意见**携带关于真实标签的信息**   → 值得投 B2
     (b) 只是"把 one-hot 削平了一点"的正则化效应    → ⛔没肉，转 448

⭐**对照 = 把 OOF 概率在样本之间随机打乱**。
   打乱后：边缘分布不变、每行的熵不变、平滑强度不变，
   **唯独丢掉了"这一行对应哪一张图"的对应关系**。
   → 若对照也涨同样多 → 是 (b)，一票否决。

═══════════════════════════════════════════════════════════════════════════
⛔循环固化：为什么必须 out-of-fold
═══════════════════════════════════════════════════════════════════════════
模型的"第二意见"来自它自己在这批标签上训出来的结果。若拿同一批样本上的
自身预测去软化自身标签，模型只会**确认它已经相信的东西**，指标必然变好，
但那是自我确认，不是纠错。
→ 必须 out-of-fold：**用 A 折训出来的模型，去给 B 折打分**。

═══════════════════════════════════════════════════════════════════════════
⭐预注册判据（⛔跑之前写死，事后不许改）
═══════════════════════════════════════════════════════════════════════════
- 共检验 6 个真实配置（软化 α×4 + 硬重标注阈值×2）→ 按判据7 做 Bonferroni 校正，
  z 临界值 = inv_cdf(1 − 0.05/(2×6)) = **2.64**，不再是 2.00。
- 判为"有肉"必须**三条同时成立**：
    ① 最优配置的配对 Δ > 2.64 × 配对SE
    ② **同 α 的打乱对照 Δ 不显著**（否则是平凡的平滑效应）
    ③ 改动样本数 ≥ 500（判据9：只写阈值方向不够，必须写死最小样本量）
- 两把尺子（原始带噪 / 低可疑度子集）**打架则本地一律无效**，交外部裁决（判据2）。
  ⚠低可疑度子集与软化信号共用 CLIP 特征 → 有循环风险，**以原始带噪那把为主**。
"""
import io
import os
import sys
import time
from statistics import NormalDist

import numpy as np
import torch
import torch.nn.functional as Fn
from sklearn.model_selection import train_test_split

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")

N_TESTS = 6                                  # ⭐真实配置数，Bonferroni 用
ZC = NormalDist().inv_cdf(1 - 0.05 / (2 * N_TESTS))
MIN_CHANGED = 500                            # ⭐最小改动样本量


def bar(t):
    print("\n" + "=" * 82)
    print(t)
    print("=" * 82)


# ══════════════════════════════════════════════════ 线性头（支持软标签）
def train_head(X, T_soft, K, epochs=30, lr=3e-3, bs=2048, seed=0):
    """T_soft 是 (N,K) 的软目标；硬标签场景传 one-hot 即可，两者走同一条码。"""
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    W = torch.zeros(K, X.shape[1])
    torch.nn.init.normal_(W, std=0.01)
    W.requires_grad_(True)
    b = torch.zeros(K, requires_grad=True)
    opt = torch.optim.AdamW([W, b], lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n = len(X)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for s in range(0, n, bs):
            i = perm[s:s + bs]
            lg = X[i] @ W.T + b
            loss = -(T_soft[i] * Fn.log_softmax(lg, 1)).sum(1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sch.step()
    return W.detach(), b.detach()


def onehot(y, K):
    T = torch.zeros(len(y), K)
    T[torch.arange(len(y)), y] = 1.0
    return T


def paired_delta(ok_a, ok_b):
    """配对比较：只有结论不同的样本才携带信息（McNemar 的思路）。
    ⭐比独立两样本的 SE 灵敏得多——共同答对/共同答错的部分被消掉了。"""
    n01 = int((~ok_a & ok_b).sum())     # a错b对 = 改好了
    n10 = int((ok_a & ~ok_b).sum())     # a对b错 = 改坏了
    n = len(ok_a)
    d = (n01 - n10) / n
    se = np.sqrt(max(n01 + n10, 1)) / n
    return d, se, n01, n10


def main():
    t_all = time.time()
    X = np.load(os.path.join(CACHE, "train_features.npy")).astype(np.float32)
    y = np.load(os.path.join(CACHE, "train_labels.npy")).astype(np.int64)
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    K = int(y.max()) + 1
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)

    bar("41 训练集标签噪声诊断   特征 %s   类别 %d" % (tuple(X.shape), K))
    print("⚠冻结 CLIP 特征 + 线性头当替身。**只看方向与相对排序，幅度不可信**（判据3）")
    print("⭐替身合格性：被测的是「标签质量」，而线性头对标签质量的敏感度与真身同源——")
    print("   两者都是拿交叉熵去拟合同一批标签。⚠但正则化强度不同，幅度不可外推。")

    # ══════════════════════════════ 步骤1：机械式噪声率下界（不依赖任何模型）
    bar("步骤1／3  机械式噪声率下界：近似重复图，标签却不同")
    print("⭐第一性原理：两张几乎相同的图被标了**不同的类**，必有一个是错的。")
    print("   这个推理**不依赖任何模型判断谁对谁错**，所以给出的是硬下界。")
    ns = np.load(os.path.join(CACHE, "noise_scores.npz"))
    nbr = ns["nbr_idx"]                      # (N,20) 训练集内部的最近邻
    assert len(nbr) == len(y), "noise_scores 与特征行数对不上，缓存不同步"
    # 逐样本算它与自己 20 个近邻的余弦相似度
    sims = np.einsum("ij,ikj->ik", X, X[nbr])
    lab_ne = (y[nbr] != y[:, None])
    print("\n%-12s %12s %12s %14s" % ("相似度阈值", "近似重复对数", "其中标签不同", "→噪声率下界"))
    print("-" * 82)
    for th in [0.95, 0.97, 0.98, 0.99, 0.995]:
        m = sims >= th
        npair = int(m.sum())
        nconf = int((m & lab_ne).sum())
        # 每个冲突对至少有 1 个样本标错；除以 2 是因为一对被两端各数了一次
        lo = nconf / 2 / len(y)
        print("%-12.3f %12d %12d %13.2f%%" % (th, npair, nconf, 100 * lo))
    print("-" * 82)
    print("⚠**这是下界不是估计值**：只抓得到「有近似重复搭档」的错标；")
    print("   一张孤零零的错标图，这个方法完全看不见。真实噪声率只会更高。")
    print("⚠另一头的偏差：CLIP 余弦 0.98 不等于同一张图——同一次拍摄的连拍也能到 0.98，")
    print("   而连拍拍到两只不同的鸟是可能的。所以低阈值那几行要打折看。")

    # ══════════════════════════════ 步骤2：out-of-fold 第二意见
    bar("步骤2／3  生成 out-of-fold 概率（⛔防循环固化的关键一步）")
    itr, iva = train_test_split(np.arange(len(y)), test_size=0.2,
                                random_state=42, stratify=y)
    print("训练 %d / 验证 %d（seed=42，与全项目其它实验同一套切分，可横向比）"
          % (len(itr), len(iva)))
    Xtr, ytr = Xt[itr], yt[itr]
    Xva, yva = Xt[iva], yt[iva]

    f0, f1 = train_test_split(np.arange(len(itr)), test_size=0.5,
                              random_state=7, stratify=ytr.numpy())
    P_oof = torch.zeros(len(itr), K)
    for a, bfold in [(f0, f1), (f1, f0)]:
        W, b = train_head(Xtr[a], onehot(ytr[a], K), K, seed=0)
        P_oof[bfold] = torch.softmax(Xtr[bfold] @ W.T + b, 1)
    print("✅ 每个训练样本都拿到了「**不是它自己训出来的模型**」给的概率")
    agree = (P_oof.argmax(1) == ytr).float().mean().item()
    print("   OOF 预测与给定标签一致率 = %.4f  → 不一致的 %.1f%% 是候选噪声"
          % (agree, 100 * (1 - agree)))
    print("   ⚠这 %.1f%% 里既有真错标，也有单纯的模型能力不足，**两者混在一起**——" % (100 * (1 - agree)))
    print("   这正是下一步要靠对照去分离的东西。")

    # ══════════════════════════════ 步骤3：软化 / 重标注 + 打乱对照
    bar("步骤3／3  真正的检验：软化标签有没有肉？（含打乱对照）")
    T_hard = onehot(ytr, K)
    Wb, bb = train_head(Xtr, T_hard, K, seed=0)
    ok_base = ((Xva @ Wb.T + bb).argmax(1) == yva).numpy()
    acc_base = ok_base.mean()

    # 低可疑度子集：第二把尺子。⚠与软化信号共用 CLIP 特征，有循环风险，只作参考
    susp = ns["suspicion"][iva]
    clean_m = susp <= np.quantile(susp, 0.6)
    print("基准（原始硬标签）：验证 %d 张 acc=%.4f ；低可疑60%%子集 acc=%.4f"
          % (len(yva), acc_base, ok_base[clean_m].mean()))
    print("⭐Bonferroni：检验 %d 个配置 → z 临界值 %.2f（不是 2.00）" % (N_TESTS, ZC))
    print("⭐最小改动样本量 %d（判据9：只写阈值方向，小样本上必有一档偶然转正）" % MIN_CHANGED)

    rng = np.random.default_rng(0)
    shuf = torch.from_numpy(rng.permutation(len(itr)))     # ⭐打乱对照用的置换

    rows = []
    print("\n%-24s %8s %9s %9s %9s %8s %s"
          % ("配置", "改动数", "验证acc", "配对Δ", "阈值(2.64SE)", "低可疑60%", "判定"))
    print("-" * 100)

    def run(name, T_new, n_changed, is_ctrl=False):
        W, b = train_head(Xtr, T_new, K, seed=0)
        ok = ((Xva @ W.T + b).argmax(1) == yva).numpy()
        d, se, n01, n10 = paired_delta(ok_base, ok)
        thr = ZC * se
        if n_changed < MIN_CHANGED:
            verd = "⛔改动太少(<%d)" % MIN_CHANGED
        elif d > thr:
            verd = "✅显著更好"
        elif d < -thr:
            verd = "❌显著更差"
        else:
            verd = "⚠判为零"
        print("%-24s %8d %9.4f %+9.4f %9.4f %8.4f %s"
              % (name, n_changed, ok.mean(), d, thr, ok[clean_m].mean(), verd))
        rows.append((name, is_ctrl, d, thr, n_changed, n01, n10))
        return d, thr

    # ── 软化：target = (1-α)·onehot + α·p_oof
    for a in [0.2, 0.4, 0.6, 0.8]:
        T = (1 - a) * T_hard + a * P_oof
        run("软化 α=%.1f" % a, T, len(itr))
        run("  └打乱对照 α=%.1f" % a, (1 - a) * T_hard + a * P_oof[shuf],
            len(itr), is_ctrl=True)

    # ── 硬重标注：OOF 很自信且与给定标签冲突时，直接换标签
    oof_p, oof_c = P_oof.max(1).values, P_oof.argmax(1)
    for c in [0.5, 0.8]:
        m = (oof_p >= c) & (oof_c != ytr)
        y2 = ytr.clone()
        y2[m] = oof_c[m]
        run("硬重标注 conf≥%.1f" % c, onehot(y2, K), int(m.sum()))
        # 对照：随机挑同样多的样本，换成随机类别
        idx = torch.from_numpy(rng.choice(len(itr), int(m.sum()), replace=False))
        y3 = ytr.clone()
        y3[idx] = torch.from_numpy(rng.integers(0, K, int(m.sum())))
        run("  └随机改同样多", onehot(y3, K), int(m.sum()), is_ctrl=True)

    # ══════════════════════════════ 结论
    bar("判定（按跑之前写死的三条判据）")
    real = [r for r in rows if not r[1]]
    ctrl = {r[0].replace("  └打乱对照", "软化").replace("  └随机改同样多", "随机"): r
            for r in rows if r[1]}
    best = max(real, key=lambda r: r[2])
    name, _, d, thr, nch, n01, n10 = best
    print("最佳真实配置：%s   配对Δ=%+.4f  阈值=%.4f  改动 %d 张（改好 %d／改坏 %d）"
          % (name, d, thr, nch, n01, n10))
    c1 = d > thr
    c2 = all(r[2] <= r[3] for r in rows if r[1])       # 所有对照都不显著
    c3 = nch >= MIN_CHANGED
    print("  ① Δ 超过 Bonferroni 阈值        : %s" % ("✅" if c1 else "⛔"))
    print("  ② 所有打乱/随机对照均不显著      : %s" % ("✅" if c2 else "⛔"))
    print("  ③ 改动样本数 ≥ %d              : %s" % (MIN_CHANGED, "✅" if c3 else "⛔"))
    print("-" * 82)
    if c1 and c2 and c3:
        print("⭐⭐**有肉 → 投 B2（训练标签软化／自蒸馏）。**")
        print("   接入点：28_finetune_v2.py:468 的 `ys` 已经是软标签张量，")
        print("   把 one-hot+label_smoothing 换成读外部概率矩阵即可，soft_ce 不用动。")
        print("   ⚠真身上必须重做 OOF（当前 OOF 来自冻结 CLIP 线性头，不是微调后的骨干）。")
    elif not c2:
        print("⛔**对照也显著 → 这是平凡的平滑／正则效应，不是标签信息。**")
        print("   → 别投 B2。转 B3（448px 训练）。")
    else:
        print("⛔**没肉 → 别投 B2，转 B3（448px 训练）。**")
        print("   ⚠但这个结论只对**冻结 CLIP 线性头**成立。真身正则化更强、容量更大，")
        print("   对噪声的敏感度不同 → 属于判据3 的'只信方向'范畴，方向是负，就够了。")
    print("\n总耗时 %.1f 分钟" % ((time.time() - t_all) / 60))


if __name__ == "__main__":
    main()
