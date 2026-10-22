# -*- coding: utf-8 -*-
"""_probe_posthoc_head · 纯 CPU / 零 GPU：在 c3 的 logits 上量化所有【推理期后处理】的改动量。

⛔ 只读、不写任何成绩包。只 print 数字。
⚠ 本脚本**没有标签**，所以它 ⛔不给准确率、只给「这个后处理会改多少张预测」，
   以及能在无标签下算出的硬约束（配额超额量 = 保证错的下界）。

用法（主机）：
  D:\\Anaconda\\envs\\dl\\python.exe scripts\\_probe_posthoc_head.py
"""
import _bootstrap  # noqa: F401

import csv
import json
import os

import numpy as np

from configs import paths as P

R = P.RUNS_DIR
POOL_HI = os.path.join(R, "20260916_210154_pool544_ep12", "pool.npz")
POOL_LO = os.path.join(R, "20260916_175916_pool448_ep12", "pool.npz")
C3_CSV = os.path.join(R, "20260916_210948_rcp_c3_ms448_544_ep12", "pred_results.csv")

CC = ["cc085", "cc070", "cc055", "cc040"]
MILD = ["mild%d" % i for i in range(5)]
BASE = ["orig"] + MILD
C3 = BASE + CC


def l2(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(1e-12)


def form(pool, names):
    z = np.load(pool)
    meta = json.load(open(os.path.join(os.path.dirname(pool), "pool_meta.json"),
                          encoding="utf-8"))
    vn = meta["view_names"]
    idx = [vn.index(n) for n in names]
    F = z["F"]
    acc = np.zeros((F.shape[1], F.shape[2]), np.float32)
    for i in idx:                       # 逐视角累加，峰值内存少一份 (V,N,D)
        acc += l2(F[i].astype(np.float32))
    acc /= len(idx)
    head = dict(W=z["W"].astype(np.float32), b=z["b"].astype(np.float32),
                s=float(z["s"]) if "s" in z.files else None)
    del F, z
    return l2(acc), head, meta


def stats(pred, K=500, tag=""):
    c = np.bincount(pred, minlength=K).astype(np.float64)
    n = len(pred)
    rnd = np.sqrt((n / K) * (1 - 1.0 / K))
    over = float(np.maximum(c - n / K, 0).sum())
    print("    %-34s min=%3d max=%3d sd=%6.2f (%.2fx随机)  超配额=%5.0f (%.2f%%)"
          % (tag, c.min(), c.max(), c.std(), c.std() / rnd, over, 100 * over / n))
    return c


def sinkhorn(lg, quota, T, iters=200):
    q = np.asarray(quota, np.float64)
    q = q / q.sum()
    Q = np.exp((lg - lg.max(1, keepdims=True)) / T).astype(np.float64)
    Q /= Q.sum()
    n = Q.shape[0]
    for _ in range(iters):
        Q *= (q / np.maximum(Q.sum(0), 1e-300))[None, :]
        Q *= ((1.0 / n) / np.maximum(Q.sum(1), 1e-300))[:, None]
    return Q.argmax(1)


def main():
    print("=" * 100)
    Xhi, head, mhi = form(POOL_HI, C3)
    Xlo, _, mlo = form(POOL_LO, BASE)
    X = l2((Xlo + Xhi) / 2.0)
    del Xhi, Xlo
    W, b, s = head["W"], head["b"], head["s"]
    Wn = W / np.linalg.norm(W, axis=1, keepdims=True).clip(1e-12)
    cos = (X @ Wn.T).astype(np.float32)          # (N,500) 余弦
    lg = s * cos
    pred = lg.argmax(1)
    n, K = lg.shape
    print("  重建 c3：N=%d K=%d  s=%.4f  (hi=%s@%d, lo=%s@%d, ep%s)"
          % (n, K, s, os.path.basename(os.path.dirname(POOL_HI)), mhi["res"],
             os.path.basename(os.path.dirname(POOL_LO)), mlo["res"], mhi["epoch"]))

    # ── 0. 与已出的 c3 包逐行核对（证明 59 号可零 GPU 精确复现）
    cn = json.load(open(P.F_MANIFEST, encoding="utf-8"))["class_names"]
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location(
        "pred26", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "26_predict_finetuned.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["pred26"] = m
    spec.loader.exec_module(m)
    paths = m._ex.list_test_files()
    ref = {}
    with open(C3_CSV, newline="", encoding="utf-8-sig") as f:
        for r in csv.reader(f):
            if len(r) >= 2:
                ref[r[0].strip()] = r[-1].strip()
    dif = sum(1 for p, c in zip(paths, pred) if ref[os.path.basename(p)] != cn[c])
    print("  ⭐ 与榜上 c3 包的逐行分歧 = %d  → %s" %
          (dif, "✅逐位复现" if dif == 0 else "⛔对不上，后面的数别信"))

    # ── 1. 置信度 / 边界结构
    srt = np.sort(lg, 1)
    marg = srt[:, -1] - srt[:, -2]
    pm = np.exp(lg - lg.max(1, keepdims=True))
    pm /= pm.sum(1, keepdims=True)
    top1p = pm.max(1)
    print("\n  【置信结构】top1 余弦 mean=%.4f  top1-top2 logit 差：中位=%.3f p10=%.3f p25=%.3f"
          % (cos.max(1).mean(), np.median(marg), np.percentile(marg, 10),
             np.percentile(marg, 25)))
    print("    softmax top1 概率：中位=%.4f  <0.5 的有 %d 张(%.1f%%)  <0.3 的有 %d 张(%.1f%%)"
          % (np.median(top1p), (top1p < .5).sum(), 100 * (top1p < .5).mean(),
             (top1p < .3).sum(), 100 * (top1p < .3).mean()))
    print("    ⚠ 温度：s=%.2f 是唯一缩放，argmax 与 s 无关；但 softmax 概率强烈依赖 s" % s)

    # ── 2. 类分布
    print("\n  【类分布】(配额 %.2f/类)" % (n / K))
    c0 = stats(pred, K, "c3 原样")

    # ── 3. Sinkhorn 均衡配额（含 35 号没扫过的 T>1）
    print("\n  【Sinkhorn 均衡配额】T 越小越接近硬分配；35 号只扫到 T=1.0（最优落在边界）")
    uni = np.ones(K)
    for T in (0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
        p = sinkhorn(lg, uni, T)
        ch = int((p != pred).sum())
        c = np.bincount(p, minlength=K)
        print("    T=%-5.2f 改判 %5d 张 (%5.2f%%)   新分布 sd=%6.2f  min=%3d max=%3d"
              % (T, ch, 100 * ch / n, c.std(), c.min(), c.max()))

    # ── 4. 先验校正（软，逐类平移）
    print("\n  【先验校正 lg − τ·log(自身预测先验)】")
    pq = np.bincount(pred, minlength=K).astype(np.float64) + 1.0
    lp = np.log(pq / pq.sum())
    for tau in (0.25, 0.5, 1.0, 2.0, 4.0):
        p = (lg - tau * lp[None, :]).argmax(1)
        ch = int((p != pred).sum())
        c = np.bincount(p, minlength=K)
        print("    τ=%-5.2f 改判 %5d 张 (%5.2f%%)   新分布 sd=%6.2f  min=%3d max=%3d"
              % (tau, ch, 100 * ch / n, c.std(), c.min(), c.max()))

    # ── 5. 特征后处理（全部在 L2 之前）
    print("\n  【特征后处理】（⚠ 头 W 是在未变换特征上训的，全是分布失配，只给改动量）")
    def chg(Xn, tag):
        p = (l2(Xn) @ Wn.T).argmax(1)
        ch = int((p != pred).sum())
        c = np.bincount(p, minlength=K)
        print("    %-36s 改判 %5d 张 (%5.2f%%)  新分布 sd=%6.2f" % (tag, ch, 100 * ch / n, c.std()))

    mu = X.mean(0, keepdims=True)
    print("    测试集特征均值向量范数 ||mu|| = %.4f（X 已 L2，故 1.0 是上限）"
          % float(np.linalg.norm(mu)))
    chg(X - mu, "去测试集均值方向 (centering)")
    chg(X - 0.5 * mu, "去 0.5×均值方向")
    for beta in (0.5, 0.7, 1.3, 2.0):
        chg(np.sign(X) * np.abs(X) ** beta, "Tukey 幂变换 beta=%.1f" % beta)
    # 简易白化：只去掉 top-k 主方向（用 W 的协方差近似，不碰测试集统计）
    U, S, Vt = np.linalg.svd(Wn - Wn.mean(0, keepdims=True), full_matrices=False)
    for k in (1, 3, 8):
        Pk = Vt[:k]
        chg(X - (X @ Pk.T) @ Pk, "去 W 的前 %d 个主方向" % k)

    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
