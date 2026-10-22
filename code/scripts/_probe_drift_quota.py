# -*- coding: utf-8 -*-
"""_probe_drift_quota · 零 GPU 两个小测量（只 print，不写任何东西）

① 微调后的视觉特征相对**原版 CLIP** 漂了多远
   → 决定「CLIP 文本塔的类名嵌入还能不能索引我们的特征空间」。
② 线性探针（35 号那张表的基座）在验证集上的**超配额量**，
   用来和 c3 在真测试集上的 7.65% 做同口径比较。
"""
import _bootstrap  # noqa: F401

import os

import numpy as np
from sklearn.model_selection import train_test_split

from configs import paths as P

R = P.RUNS_DIR


def l2(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(1e-12)


def main():
    # ── ① 漂移：原版 CLIP@224 特征  vs  微调 ep12 的 orig@544 特征（同一批图、同序）
    z = np.load(os.path.join(R, "20260916_210154_pool544_ep12", "pool.npz"))
    Fft = l2(z["F"][0].astype(np.float32))          # view 0 = orig
    W = z["W"].astype(np.float32)
    del z
    Fcl = np.load(P.F_TEST_FEAT, mmap_mode="r")
    Fcl = l2(np.asarray(Fcl, np.float32))
    print("  原版CLIP特征 %s   微调ep12 orig@544 特征 %s" % (Fcl.shape, Fft.shape))
    c = (Fcl * Fft).sum(1)
    print("  ⭐ 逐图 cos(原版CLIP, 微调后) : 中位=%.4f  mean=%.4f  p5=%.4f  p95=%.4f"
          % (np.median(c), c.mean(), np.percentile(c, 5), np.percentile(c, 95)))
    # 各向异性（cone effect）
    for nm, F in (("原版CLIP", Fcl), ("微调ep12", Fft)):
        mu = F.mean(0)
        print("    %s：||mean(L2特征)|| = %.4f  → 平均两两余弦 ≈ %.4f"
              % (nm, np.linalg.norm(mu), float(np.linalg.norm(mu) ** 2)))
    # 微调头 W 在原版 CLIP 空间里还有多少判别力（同一批图，换特征算 argmax 的一致性）
    Wn = W / np.linalg.norm(W, axis=1, keepdims=True).clip(1e-12)
    a_ft = (Fft @ Wn.T).argmax(1)
    a_cl = (Fcl @ Wn.T).argmax(1)
    print("  ⭐ 用同一个头 W：微调特征 vs 原版CLIP特征 的 argmax 一致率 = %.4f (随机=0.002)"
          % float((a_ft == a_cl).mean()))
    del Fcl, Fft

    # ── ② 线性探针在验证集上的超配额量（35 号同口径）
    feat = np.load(P.F_TRAIN_FEAT, mmap_mode="r")
    y_all = np.load(P.F_TRAIN_LABEL).astype(np.int64)
    hd = np.load(os.path.join(P.CACHE_DIR, "head_loss_sce.npz"))
    Wp, bp = hd["W"].astype(np.float32), hd["b"].astype(np.float32)
    _, iva = train_test_split(np.arange(len(y_all)), test_size=0.2,
                              random_state=42, stratify=y_all)
    X = np.array(feat[iva], np.float32)
    if bool(hd["normalize"]):
        X = l2(X)
    y = y_all[iva]
    lgp = X @ Wp.T + bp
    pr = lgp.argmax(1)
    K, n = Wp.shape[0], len(y)
    cnt = np.bincount(pr, minlength=K).astype(np.float64)
    tq = np.bincount(y, minlength=K).astype(np.float64)
    acc = float((pr == y).mean())
    over_u = float(np.maximum(cnt - n / K, 0).sum())          # 相对均衡配额
    over_t = float(np.maximum(cnt - tq, 0).sum())             # 相对真实配额
    err = int((pr != y).sum())
    print("\n  线性探针验证集 n=%d 准确率=%.4f 错 %d 张" % (n, acc, err))
    print("  预测分布 sd=%.2f (随机 %.2f = %.2fx)  真实分布 sd=%.2f"
          % (cnt.std(), np.sqrt((n / K) * (1 - 1.0 / K)),
             cnt.std() / np.sqrt((n / K) * (1 - 1.0 / K)), tq.std()))
    print("  ⭐ 超【均衡】配额 = %.0f 张 (%.2f%% of n，占错题的 %.1f%%)"
          % (over_u, 100 * over_u / n, 100 * over_u / err))
    print("  ⭐ 超【真实】配额 = %.0f 张 (%.2f%% of n，占错题的 %.1f%%)"
          % (over_t, 100 * over_t / n, 100 * over_t / err))
    print("  ⚠ 对照：c3@真测试集 超均衡配额 1910 张 (7.65%% of n)；"
          "按榜分 80.4542 推错题 4880 张 → 占错题 39.1%%")


if __name__ == "__main__":
    main()
