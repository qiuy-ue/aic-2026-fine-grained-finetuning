# -*- coding: utf-8 -*-
"""
42：标签噪声诊断 **v2**——41 号的结论我自己不采信，这里修三处硬伤后重做。

═══════════════════════════════════════════════════════════════════════════
⛔41 号错在哪（三条，前两条足以让结论作废）
═══════════════════════════════════════════════════════════════════════════
**硬伤① 替身太弱，弱到没有意见可言。**
   41 号用小批量 AdamW(lr=3e-3, wd=1e-4, 30轮) 训头，基准只有 **0.5874**；
   而项目里现成的参考头 `head_loss_sce.npz` 在**同一套 seed=42 切分**上是 **0.6943**。
   ⭐更致命的证据：「硬重标注 conf≥0.5」那两行**改动数 = 0**——
   说明这个头的最大概率**从来没超过 0.5**，它的"第二意见"是一团糊。
   而本实验测的恰恰是「模型的意见比标签更可信吗」。
   ⛔**替身在被测的那个属性上就不像真身** → 与近邻投票那次栽的是同一个坑。
   **修法**：照抄 06_loss_experiments.py 的配方（全批量 Adam lr=0.05 + L2 1/(2Cn), C=10），
   跑 GPU，把基准拉回 0.69 一线。

**硬伤② 对照没对齐平滑强度，比较不公平。**
   41 号的对照是「把 P_oof **跨样本**打乱」。但真实的 P_oof 有约 53% 的概率
   把质量压回给定标签本身 → **它施加的平滑比打乱版更弱**。
   于是"打乱版赢"可能只是因为它平滑得更狠，⛔不能归因到"真实意见没信息"。
   **修法：行内打乱**——保留 `P[i, y_i]` 分毫不动，只把**剩下那部分质量**
   在其他 K−1 个类之间随机重排。
   → 逐样本精确对齐：给定标签上的质量一样、离标签质量的熵一样、平滑强度一样，
     **唯一被破坏的是「模型知不知道它可能是哪个别的类」**——那才是被测对象。

**硬伤③ 尺子选错了。**
   41 号主看「原始带噪 acc」和「低可疑60%」。但后者与软化信号共用 CLIP 特征（循环风险），
   而前者含**训练↔验证近似重复**造成的泄漏虚高。
   **修法**：主尺改用项目标准的**无泄漏子集**（`leakage_scores.npz` 的 val_maxsim<0.90，
   6835 张），这也是唯一被证明与榜分同向的那把。

═══════════════════════════════════════════════════════════════════════════
⭐新增第二条臂：机械臂（完全不依赖模型强弱）
═══════════════════════════════════════════════════════════════════════════
第一条臂（模型软化）的结论**天然受限于替身强度**：弱老师给不出好意见，
负结果说明不了"标签噪声这条路没肉"。
→ 所以补一条**不需要老师**的臂：
   步骤1 已用纯几何找出 10496 对「近似重复但标签不同」的样本对。
   **两张几乎相同的图被标了不同的类，必有一个错**——这个推理零模型参与。
   处理：把这些样本的目标改成 `0.5·onehot(自己的标签) + 0.5·onehot(邻居的标签)`，
   即「我不知道是哪个，但一定是这两个之一」。
   ⭐**对照必做**：对**随机挑的同样多个非冲突样本**做同样的处理。
     否则无法区分"抓对了噪声"和"随便糊 1 万个样本都能涨"。
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
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# ⭐预注册（跑之前写死）：真实配置 = 软化α×4 + 机械臂×2 = 6 个
N_TESTS = 6
ZC = NormalDist().inv_cdf(1 - 0.05 / (2 * N_TESTS))
MIN_CHANGED = 500
SIM_TH = 0.95            # 机械臂的近似重复阈值（步骤1 实测此档 10496 对冲突）


def bar(t):
    print("\n" + "=" * 96)
    print(t)
    print("=" * 96)


def train_head(X, T, K, C=10.0, n_iter=500, lr=0.05, seed=0):
    """⭐配方照抄 06_loss_experiments.py：全批量 Adam + L2 = 1/(2Cn)·|W|²。
    ⚠必须用同一套配方，否则基准就不是 0.69，整个替身失去代表性。
    T 是 (N,K) 软目标；硬标签场景传 one-hot，两者走同一条码。"""
    torch.manual_seed(seed)
    n, d = X.shape
    W = torch.zeros(K, d, device=DEV, requires_grad=True)
    b = torch.zeros(K, device=DEV, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr)
    reg = 1.0 / (2.0 * C * n)
    for _ in range(n_iter):
        opt.zero_grad(set_to_none=True)
        lg = X @ W.T + b
        loss = -(T * Fn.log_softmax(lg, 1)).sum(1).mean() + reg * (W * W).sum()
        loss.backward()
        opt.step()
    return W.detach(), b.detach()


def onehot(y, K):
    T = torch.zeros(len(y), K, device=DEV)
    T[torch.arange(len(y), device=DEV), y] = 1.0
    return T


def paired(ok_a, ok_b):
    """配对比较：只有结论不同的样本携带信息。共同对/共同错的部分被消掉，比独立两样本灵敏。"""
    n01 = int((~ok_a & ok_b).sum())
    n10 = int((ok_a & ~ok_b).sum())
    n = len(ok_a)
    return (n01 - n10) / n, np.sqrt(max(n01 + n10, 1)) / n, n01, n10


def row_shuffle_offlabel(P, y, rng):
    """⭐⭐行内打乱：`P[i, y_i]` 分毫不动，其余 K−1 个位置上的质量随机重排。

    为什么这么设计：它让对照组与真实组在**平滑强度上逐样本完全一致**
    （给定标签上留多少质量一样、剩余质量的熵一样），
    唯一被破坏的是「哪个别的类拿到多少」这个**对应关系**。
    → 于是两者的差值可以干净地归因到"模型知不知道它可能是哪个别的类"。
    """
    N, K = P.shape
    ar = np.arange(N)
    P2 = P.copy()
    py = P2[ar, y].copy()
    P2[ar, y] = 0.0                                  # 先把标签列摘出来
    perm = np.argsort(rng.random((N, K)), axis=1)    # 每行一个独立置换
    P2 = np.take_along_axis(P2, perm, axis=1)
    v = P2[ar, y].copy()                             # 现在落在标签列上的是别人的质量
    zpos = np.argmax(perm == y[:, None], axis=1)     # 那个 0 被搬到哪去了
    # ⚠⚠两行的先后顺序是**正确性问题**，不是风格问题（v2 首跑被自检抓到了）：
    #   有约 1/K 的行恰好 perm[i,y]==y，此时 zpos==y，两次写落在同一列。
    #   先写 v 再写 py → 最终留下 py ✅；反过来 → py 被 v(=0) 覆盖，
    #   那一行的给定标签质量被清空，整行变成"纯别的类"。
    #   103218×0.8 行里期望 165 行中招，正是自检报的 9.74e-01 偏差。
    P2[ar, zpos] = v                                 # 把 v 塞进 0 的位置，总质量守恒
    P2[ar, y] = py                                   # 再把原质量放回标签列（后写者胜）
    return P2


def main():
    t0 = time.time()
    X = np.load(os.path.join(CACHE, "train_features.npy")).astype(np.float32)
    y = np.load(os.path.join(CACHE, "train_labels.npy")).astype(np.int64)
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    K = int(y.max()) + 1
    ns = np.load(os.path.join(CACHE, "noise_scores.npz"))
    lk = np.load(os.path.join(CACHE, "leakage_scores.npz"))

    itr, iva = train_test_split(np.arange(len(y)), test_size=0.2,
                                random_state=42, stratify=y)
    assert np.array_equal(lk["val_idx"], iva), "泄漏缓存与本切分行序不符"
    noleak = lk["val_maxsim"] < 0.90                 # ⭐主尺：无泄漏子集

    Xt = torch.from_numpy(X).to(DEV)
    yt = torch.from_numpy(y).to(DEV)
    Xtr, ytr = Xt[itr], yt[itr]
    Xva, yva = Xt[iva], yt[iva]
    ytr_np = y[itr]

    bar("42 标签噪声诊断 v2   设备 %s   特征 %s   类别 %d" % (DEV, tuple(X.shape), K))
    print("⭐主尺 = 无泄漏子集 %d 张（占验证集 %.0f%%）——项目里唯一被证明与榜分同向的尺子"
          % (noleak.sum(), 100 * noleak.mean()))
    print("⚠副尺 = 原始带噪全集（含训练↔验证近似重复造成的虚高）。**两把打架则本地无效**（判据2）")
    print("⭐Bonferroni：检验 %d 个真实配置 → z 临界 %.2f（不是 2.00）" % (N_TESTS, ZC))

    # ══════════════════════════════ 基准 + 替身合格性自检
    T_hard = onehot(ytr, K)
    Wb, bb = train_head(Xtr, T_hard, K)
    ok_b_all = ((Xva @ Wb.T + bb).argmax(1) == yva).cpu().numpy()
    ok_base = ok_b_all[noleak]
    print("\n基准头：无泄漏 %.4f   带噪全集 %.4f" % (ok_base.mean(), ok_b_all.mean()))
    ref = os.path.join(CACHE, "head_loss_sce.npz")
    if os.path.exists(ref):
        h = np.load(ref)
        Wr = torch.from_numpy(h["W"].astype(np.float32)).to(DEV)
        br = torch.from_numpy(h["b"].astype(np.float32)).to(DEV)
        okr = ((Xva @ Wr.T + br).argmax(1) == yva).cpu().numpy()
        print("参考头(head_loss_sce)：无泄漏 %.4f   带噪全集 %.4f"
              % (okr[noleak].mean(), okr.mean()))
        print("⭐替身合格性自检：两者带噪全集相差 %.4f —— 差得大就说明配方还没对齐，"
              "结论要打折" % abs(okr.mean() - ok_b_all.mean()))

    # ══════════════════════════════ OOF 第二意见
    bar("生成 out-of-fold 第二意见（⛔防循环固化）")
    f0, f1 = train_test_split(np.arange(len(itr)), test_size=0.5,
                              random_state=7, stratify=ytr_np)
    P_oof = torch.zeros(len(itr), K, device=DEV)
    for a, bf in [(f0, f1), (f1, f0)]:
        W, b = train_head(Xtr[a], onehot(ytr[a], K), K)
        P_oof[bf] = torch.softmax(Xtr[bf] @ W.T + b, 1)
    pmax = P_oof.max(1).values
    agree = (P_oof.argmax(1) == ytr).float().mean().item()
    print("OOF 与给定标签一致率 %.4f ；OOF 最大概率 中位=%.3f  >0.5占%.1f%%  >0.8占%.1f%%"
          % (agree, pmax.median().item(), 100 * (pmax > .5).float().mean(),
             100 * (pmax > .8).float().mean()))
    if (pmax > 0.5).float().mean() < 0.05:
        print("⛔⛔**替身依然没有意见**（>0.5 的不足 5%%）→ 这条臂的结论不可采信，只看机械臂。")

    P_np = P_oof.cpu().numpy()
    rng = np.random.default_rng(0)
    P_row = torch.from_numpy(row_shuffle_offlabel(P_np, ytr_np, rng)).to(DEV)
    P_col = P_oof[torch.from_numpy(rng.permutation(len(itr))).to(DEV)]
    # 自检：行内打乱必须精确保住标签列的质量
    d_chk = (P_row[torch.arange(len(itr), device=DEV), ytr]
             - P_oof[torch.arange(len(itr), device=DEV), ytr]).abs().max().item()
    print("✅行内打乱自检：标签列质量最大偏差 %.2e（应为 0），行和偏差 %.2e"
          % (d_chk, (P_row.sum(1) - 1).abs().max().item()))

    # ══════════════════════════════ 机械臂的冲突对
    bar("机械臂：纯几何找出的「近似重复但标签不同」")
    nbr = ns["nbr_idx"]
    sims = np.einsum("ij,ikj->ik", X, X[nbr])
    conf = (sims >= SIM_TH) & (y[nbr] != y[:, None])
    has = conf.any(1)
    first = np.argmax(conf, 1)
    partner = np.where(has, y[nbr[np.arange(len(y)), first]], -1)
    m_tr = has[itr]
    print("全集 %d 张里 %d 张有冲突搭档（%.2f%%）；其中落在训练侧 %d 张"
          % (len(y), has.sum(), 100 * has.mean(), m_tr.sum()))

    # ══════════════════════════════ 逐配置检验
    bar("检验（⭐主尺=无泄漏子集；对照不显著才算真信息）")
    print("%-28s %8s %9s %9s %10s %9s %s"
          % ("配置", "改动数", "无泄漏", "配对Δ", "阈值2.64SE", "带噪全集", "判定"))
    print("-" * 110)
    rows = []

    def run(name, T, n_ch, ctrl=False, arm=1):
        W, b = train_head(Xtr, T, K)
        ok_all = ((Xva @ W.T + b).argmax(1) == yva).cpu().numpy()
        ok = ok_all[noleak]
        d, se, n01, n10 = paired(ok_base, ok)
        thr = ZC * se
        v = ("⛔改动<%d" % MIN_CHANGED if n_ch < MIN_CHANGED else
             "✅显著更好" if d > thr else "❌显著更差" if d < -thr else "⚠判为零")
        print("%-28s %8d %9.4f %+9.4f %10.4f %9.4f %s"
              % (name, n_ch, ok.mean(), d, thr, ok_all.mean(), v))
        rows.append((name, ctrl, d, thr, n_ch, n01, n10, arm))
        return ok

    def head_to_head(tag, ok_real, ok_ctrl):
        """⭐⭐真实组 **直接配对** 对照组——这才是本实验真正的检验统计量。

        ⛔为什么不能拿「真实Δ」减「对照Δ」了事：两个 Δ 都是相对同一个基准算的，
          它们的误差**高度正相关**（共用基准的随机性），简单相减后的 SE
          既不是两者之和也不是平方和——那个数是错的，会把边缘结论判成显著。
        ⭐直接配对则把基准彻底消掉：只数「真实对/对照错」与「真实错/对照对」两格。
        """
        d, se, n01, n10 = paired(ok_ctrl, ok_real)      # 正号 = 真实赢对照
        thr = ZC * se
        v = ("✅真实赢对照" if d > thr else
             "❌**对照赢真实**" if d < -thr else "⚠拉不开")
        print("      ⇒ %s：真实−对照 = %+.4f  阈值 %.4f  (真实独赢%d／对照独赢%d)  %s"
              % (tag, d, thr, n01, n10, v))
        return d, thr

    print("【臂1】模型软化：target = (1−α)·onehot + α·P")
    arm1 = []
    for a in [0.2, 0.4, 0.6, 0.8]:
        okr = run("软化 α=%.1f" % a, (1 - a) * T_hard + a * P_oof, len(itr))
        okc = run("  └行内打乱 α=%.1f ⭐" % a, (1 - a) * T_hard + a * P_row, len(itr), True)
        run("  └跨样本打乱 α=%.1f" % a, (1 - a) * T_hard + a * P_col, len(itr), True)
        arm1.append(head_to_head("α=%.1f 真实 vs 行内打乱" % a, okr, okc))

    print("\n【臂2】机械臂：冲突对 → 0.5·自己 + 0.5·邻居（零模型参与）")
    arm2 = []
    idx_tr = np.where(m_tr)[0]
    for w in [0.5, 0.3]:
        T = T_hard.clone()
        pl = torch.from_numpy(partner[itr][m_tr]).to(DEV)
        ii = torch.from_numpy(idx_tr).to(DEV)
        T[ii] = 0.0
        T[ii, ytr[ii]] = 1.0 - w
        T[ii, pl] = w
        okr = run("冲突对软化 w=%.1f" % w, T, len(idx_tr), arm=2)
        # ⭐对照：随机挑同样多的**非冲突**样本，糊成"自己(1-w) + 随机类(w)"
        pool = np.where(~m_tr)[0]
        rsel = rng.choice(pool, len(idx_tr), replace=False)
        T2 = T_hard.clone()
        jj = torch.from_numpy(rsel).to(DEV)
        rc = torch.from_numpy(rng.integers(0, K, len(idx_tr))).to(DEV)
        T2[jj] = 0.0
        T2[jj, ytr[jj]] = 1.0 - w
        T2[jj, rc] = w
        okc = run("  └随机同量对照 w=%.1f" % w, T2, len(idx_tr), True, arm=2)
        arm2.append(head_to_head("w=%.1f 真实 vs 随机同量" % w, okr, okc))

    # ══════════════════════════════ 判定（⭐分臂，⛔不许一刀切）
    bar("判定")
    print("⛔为什么必须**分臂判**：两条臂问的是两个不同的问题，用的是两套不同的信号，")
    print("   一条臂的对照显著与否，跟另一条臂的结论**没有任何逻辑关系**。")
    print("   （v2 首跑我把它们混在一个 all() 里判了，臂2 被臂1 的对照连坐——那是判定 bug。）\n")

    def verdict(arm_no, name, hh):
        """一条臂过关的充要条件：**真实组直接配对赢自己的对照组**。"""
        real = [r for r in rows if not r[1] and r[7] == arm_no]
        best = max(real, key=lambda r: r[2])
        nm, _, d, thr, nch, n01, n10, _ = best
        win = [x for x in hh if x[0] > x[1]]
        print("【臂%d %s】最佳=%s  相对基准Δ=%+.4f(阈值%.4f)  改动%d(好%d/坏%d)"
              % (arm_no, name, nm, d, thr, nch, n01, n10))
        c1, c3 = d > thr, nch >= MIN_CHANGED
        c2 = len(win) > 0
        print("   ① 相对基准显著 %s   ② ⭐直接配对赢自己的对照 %s(%d/%d档)   ③ 改动≥%d %s"
              % ("✅" if c1 else "⛔", "✅" if c2 else "⛔", len(win), len(hh),
                 MIN_CHANGED, "✅" if c3 else "⛔"))
        return c1 and c2 and c3

    p1 = verdict(1, "模型软化", arm1)
    p2 = verdict(2, "机械冲突对", arm2)
    print("-" * 96)
    if p1:
        print("⭐⭐**臂1 有肉 → 投 B2**。接入点 28_finetune_v2.py:468 的 `ys` 已是软标签张量。")
        print("   ⚠真身上必须重做 OOF（当前 OOF 来自冻结 CLIP 头，不是微调后的骨干）。")
    else:
        print("⛔**臂1 判负：模型的第二意见不值得信。**")
        print("   ⭐机理（四档方向一致，不是噪声）：模型的错是**系统性**的——细粒度近缘物种")
        print("     固定混淆。把质量压到「它觉得最像的那几个类」= 反复加固同一个错误方向；")
        print("     而随机撒开的质量在梯度里互相抵消，只剩纯平滑效应，所以对照反而赢。")
        print("   ⚠这与「测试集近邻投票」被毙是**同一个根因**，我在同一个洞里踩了第二次。")
    if p2:
        print("⭐**臂2 有肉（且成本近乎为零：只读 noise_scores.npz，不需训练/不需OOF）**")
        print("   → ⛔但幅度只有零点几分级别，**不值得单独开一次训练**；")
        print("     作为**免费搭车项**挂进下一次 B3/B4 训练即可。")
    else:
        print("⛔**臂2 也判负** → 训练标签这条路整体没肉，转 B3（448px 训练）。")
    print("\n⚠共同边界：替身是**冻结 CLIP 线性头**（带噪全集 %.4f），真身是微调骨干（≈0.79）。"
          % ok_b_all.mean())
    print("  真身的第二意见质量更高 → 臂1 的负结论**在真身上可能没这么绝对**。")
    print("  ⭐但有独立同向证据：伪标签实验已在**真身**上证实「拿模型自己的预测当目标」")
    print("    是倒U形且峰值只赚 +0.88 分（ep02=73.64→ep04=72.89）。两条证据同向。")
    print("\n总耗时 %.1f 分钟" % ((time.time() - t0) / 60))


if __name__ == "__main__":
    main()
