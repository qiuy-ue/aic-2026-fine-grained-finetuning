# -*- coding: utf-8 -*-
"""
零 GPU / 纯 numpy：判定训练集标签噪声是【人为注入的对称噪声】还是【自然网络噪声/类定义重叠】。

判别原理（预注册）：
  H_inject : 主办方人为注入 p% 对称噪声 → 冲突的 (真类, 错类) 对在 500×499 上**近似均匀**，
             任何单个类对占全部冲突的份额应 ~ 1/(500*499) = 4.0e-6，top-50 类对合计 < 0.1%。
  H_natural: 自然噪声/类定义重叠 → 冲突**高度集中**在少数近缘类对上（幂律），
             top-50 类对能吃掉冲突的很大一块。
两者在这个统计量上相差 3~4 个数量级，⛔不需要任何模型判断，纯几何。
"""
import io
import os
import sys
import time

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")


def bar(t):
    print("\n" + "=" * 90)
    print(t)
    print("=" * 90)


def main():
    t0 = time.time()
    X = np.load(os.path.join(CACHE, "train_features.npy")).astype(np.float32)
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    ns = np.load(os.path.join(CACHE, "noise_scores.npz"))
    nbr = ns["nbr_idx"]                          # (N,20)
    y = ns["labels"].astype(np.int64)
    N, K = len(y), int(y.max()) + 1
    bar("训练集 %d 张 / %d 类 / 近邻 k=%d" % (N, K, nbr.shape[1]))

    # ---- 分块算 20-NN 余弦（避免 (N,20,512) 一次性 4.2GB）
    sims = np.empty(nbr.shape, np.float32)
    CH = 4096
    for s in range(0, N, CH):
        e = min(s + CH, N)
        sims[s:e] = np.einsum("ij,ikj->ik", X[s:e], X[nbr[s:e]])
    del X

    lab_ne = (y[nbr] != y[:, None])

    bar("A. 复核 41 号的「机械下界」，并给出【条件冲突率】——两者差 20 倍")
    print("%-8s %10s %10s %12s %12s %12s %12s"
          % ("阈值", "有向近邻对", "标签不同", "41的下界%", "条件冲突率%", "涉及图片数", "占全集%"))
    print("-" * 90)
    cond = {}
    for th in [0.95, 0.97, 0.98, 0.99, 0.995, 0.999]:
        m = sims >= th
        npair = int(m.sum())
        nconf = int((m & lab_ne).sum())
        involved = int(m.any(1).sum())
        lo = nconf / 2 / N
        c = nconf / max(npair, 1)
        cond[th] = (npair, nconf, involved, c)
        print("%-8.3f %10d %10d %11.2f%% %11.2f%% %12d %11.2f%%"
              % (th, npair, nconf, 100 * lo, 100 * c, involved, 100 * involved / N))
    print("-" * 90)
    print("⭐ 41 号的下界 = nconf/2/N（分母是【全集】）；条件冲突率 = nconf/npair（分母是【真有重复搭档的那些对】）。")
    print("   若「有重复搭档」的图与其它图噪声率相同，则条件冲突率≈2ε(1-ε) ⇒ ε≈条件率/2。")

    bar("B. ⭐⭐ 决定性检验：冲突落在哪些【类对】上——均匀 还是 集中？")
    for th in [0.98, 0.995]:
        m = (sims >= th) & lab_ne
        ii, jj = np.nonzero(m)
        a, b = y[ii], y[nbr[ii, jj]]
        lo_, hi_ = np.minimum(a, b), np.maximum(a, b)
        key = lo_.astype(np.int64) * K + hi_        # 无序类对
        uk, cnt = np.unique(key, return_counts=True)
        order = np.argsort(-cnt)
        tot = cnt.sum()
        npairs_possible = K * (K - 1) // 2
        print("\n--- 阈值 %.3f：冲突有向对 %d，涉及 %d 个不同类对（可能的类对共 %d）"
              % (th, tot, len(uk), npairs_possible))
        for topn in [1, 5, 10, 20, 50, 100]:
            if topn <= len(uk):
                sh = cnt[order[:topn]].sum() / tot
                unif = topn / npairs_possible
                print("    top-%-4d 类对吃掉冲突的 %6.2f%%   （均匀注入噪声下应为 %.4f%%，倍数 %.0f×）"
                      % (topn, 100 * sh, 100 * unif, sh / unif))
        print("    最大的 12 个类对：")
        for t_ in order[:12]:
            ca, cb = divmod(int(uk[t_]), K)
            print("      %04d ↔ %04d : %d 次" % (ca, cb, cnt[t_]))

    bar("C. 冲突对的【对称性】：A→B 和 B→A 都出现 = 类定义重叠；单向 = 个别错标")
    th = 0.98
    m = (sims >= th) & lab_ne
    ii, jj = np.nonzero(m)
    a, b = y[ii], y[nbr[ii, jj]]
    dir_key = a.astype(np.int64) * K + b
    ud, dc = np.unique(dir_key, return_counts=True)
    dset = set(ud.tolist())
    both = sum(1 for k_ in ud.tolist() if (k_ % K) * K + (k_ // K) in dset)
    print("  有向类对 %d 个，其中 %d 个（%.1f%%）反向也存在 → 双向/对称"
          % (len(ud), both, 100 * both / max(len(ud), 1)))

    bar("D. kNN 一致率分布：人为注入噪声通常出现【双峰】，自然噪声是单峰长尾")
    ag = ns["agree"].astype(np.float64)
    h, edges = np.histogram(ag, bins=20, range=(0, 1))
    for i in range(20):
        print("  [%.2f,%.2f) %7d %s" % (edges[i], edges[i + 1], h[i],
                                        "#" * int(60 * h[i] / max(h.max(), 1))))
    print("  一致率均值 %.4f 中位 %.4f ；=0 的样本 %d (%.2f%%)"
          % (ag.mean(), np.median(ag), int((ag == 0).sum()), 100 * (ag == 0).mean()))

    bar("E. 每类有多少张图的 20-NN 多数票指向【另一个固定的类】（= 类级别的系统性重叠）")
    vote = ns["vote"].astype(np.int64)            # kNN 多数票类
    disagree = vote != y
    print("  多数票 != 给定标签 的样本：%d (%.2f%%)" % (int(disagree.sum()), 100 * disagree.mean()))
    dk = y[disagree].astype(np.int64) * K + vote[disagree]
    ud2, dc2 = np.unique(dk, return_counts=True)
    o2 = np.argsort(-dc2)
    tot2 = dc2.sum()
    for topn in [10, 50, 100, 500]:
        if topn <= len(ud2):
            print("    top-%-4d 有向类对吃掉 %.2f%%（均匀应 %.4f%%）"
                  % (topn, 100 * dc2[o2[:topn]].sum() / tot2, 100 * topn / (K * (K - 1))))
    print("    最大的 10 个有向类对（真类→被投向的类）：")
    for t_ in o2[:10]:
        ca, cb = divmod(int(ud2[t_]), K)
        n_ca = int((y == ca).sum())
        print("      %04d → %04d : %d 次（占 %04d 类 %d 张的 %.1f%%）"
              % (ca, cb, dc2[t_], ca, n_ca, 100 * dc2[t_] / n_ca))

    print("\n总耗时 %.1f 分钟" % ((time.time() - t0) / 60))


if __name__ == "__main__":
    main()
