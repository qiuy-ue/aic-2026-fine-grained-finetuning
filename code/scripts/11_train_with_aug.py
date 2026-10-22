# -*- coding: utf-8 -*-
"""
B6 配套：用 K 个增强视角的特征训练线性头，量出【数据增强单独值多少分】。

═══ 实验设计：一次只改一个变量 ═══

  对照组：SCE + 原始单视角特征   ← 这就是已上榜 59 分的配置
  实验组：SCE + K 个增强视角
  差值   = 数据增强的净贡献

⛔不叠加样本清洗——它还没上过榜、本身是个问号，拿问号当地基就说不清是谁的功劳。

═══ 「5 个视角怎么用」有三种做法，各自假设不同 ═══

  epoch_sample  每轮为每张图随机抽一个视角
                → 最接近真实增强的行为(每轮看到的版本不同)，但全批量训练里
                  "轮"的概念弱，随机性可能不够
  all_samples   把 K 个视角全当独立样本，训练集直接变 K 倍大
                → 数据量最大，但同一张图的 K 个视角高度相关，
                  等价样本量远小于 K 倍，且会让"每张图的话语权"翻 K 倍
  mean_feature  K 个视角的特征取平均，仍是每张图一个向量
                → 不是增强而是【去噪】：多次观测取平均能抵消单次裁剪的偶然偏差
                  (就是之前讲的 TTA 思路用在训练侧)。最省内存，训练成本不变

⚠三者是不同的假设，不能靠想选，必须都跑。

═══ 评估 ═══
沿用 05 校准过的干净验证子集(40/60/80% 三档同向才采信)。
⚠验证集一律用【原始单视角】特征——否则改的是考卷不是学生，比较就失去意义。

用法:
  python scripts/11_train_with_aug.py
  python scripts/11_train_with_aug.py --modes mean_feature --seeds 42,1,2
"""
import _bootstrap  # noqa: F401
from _bootstrap import banner

import argparse
import importlib.util
import json
import os
import time

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from configs import paths as P

_spec = importlib.util.spec_from_file_location(
    "loss_exp", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "06_loss_experiments.py"))
_le = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_le)

VAL_FRAC = 0.2
KEEPS = (0.40, 0.60, 0.80)


def l2(a, axis=-1):
    return a / np.maximum(np.linalg.norm(a, axis=axis, keepdims=True), 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", type=str,
                    default="baseline,mean_feature,all_samples,epoch_sample")
    ap.add_argument("--loss", type=str, default="sce")
    ap.add_argument("--C", type=float, default=10.0)
    ap.add_argument("--n-iter", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--seeds", type=str, default="42,1,2")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = [int(s) for s in args.seeds.split(",")]

    banner("11 数据增强的净贡献 (底座 = %s C=%g，已上榜59分的配置)"
           % (args.loss.upper(), args.C))

    faug = os.path.join(P.CACHE_DIR, "train_features_aug.npy")
    if not os.path.exists(faug):
        raise SystemExit("找不到 %s\n请先跑: python scripts/10_extract_augmented.py" % faug)

    X1 = l2(np.load(P.F_TRAIN_FEAT).astype(np.float32))          # (N,512) 原始单视角
    XA = np.load(faug).astype(np.float32)                         # (K,N,512) 增强视角
    y = np.load(P.F_TRAIN_LABEL).astype(np.int64)
    susp = np.load(os.path.join(P.CACHE_DIR, "noise_scores.npz"),
                   allow_pickle=True)["suspicion"]
    K_views, N, D = XA.shape
    n_cls = int(y.max()) + 1
    print("  原始特征 %s   增强特征 %s (%d 个视角)" % (X1.shape, XA.shape, K_views))

    # 视角间一致性：太高=增强等于没做；太低=裁太狠可能破坏语义
    smp = l2(XA[:, :3000], axis=2)
    cons = float(np.einsum("ijk,jk->ij", smp, l2(smp.mean(0), axis=1)).mean())
    print("  视角间一致性 %.4f  %s" % (
        cons, "✅强度合适" if 0.85 <= cons <= 0.95 else
        ("⚠增强偏弱，可能白做" if cons > 0.95 else "⚠裁得偏狠，注意是否破坏了判别细节")))

    XAn = l2(XA, axis=2)
    Xmean = l2(XA.mean(0))          # 先平均再归一化：多观测取平均抵消单次裁剪的偶然偏差

    rows = []
    for sd in seeds:
        itr, iva = train_test_split(np.arange(N), test_size=VAL_FRAC,
                                    random_state=sd, stratify=y)
        # ⚠验证集永远用原始单视角，否则改的是考卷不是学生
        Xv, yv, sv = X1[iva], y[iva], susp[iva]
        masks = {k: (sv <= np.quantile(sv, k)) for k in KEEPS}
        rng = np.random.default_rng(sd)

        for mode in args.modes.split(","):
            mode = mode.strip()
            if mode == "baseline":
                Xt, yt = X1[itr], y[itr]
            elif mode == "mean_feature":
                Xt, yt = Xmean[itr], y[itr]
            elif mode == "all_samples":
                Xt = XAn[:, itr].reshape(-1, D)
                yt = np.tile(y[itr], K_views)
            elif mode == "epoch_sample":
                pick = rng.integers(0, K_views, size=len(itr))
                Xt, yt = XAn[pick, itr], y[itr]
            else:
                print("  跳过未知模式:", mode)
                continue

            t0 = time.time()
            W, b = _le.train(Xt, yt, args.C, args.loss, args.n_iter, device, args.lr)
            pred = np.argmax(Xv @ W.T + b, 1)
            accs = {k: float((pred[m] == yv[m]).mean()) for k, m in masks.items()}
            rows.append(dict(mode=mode, seed=sd, n_train=len(yt),
                             acc={str(k): v for k, v in accs.items()},
                             seconds=time.time() - t0))
        print("  seed=%-4d 完成" % sd)

    # ---------------------------------------------------------------- 汇总
    banner("结果（均值±标准差；三档同向才采信）")
    modes = []
    for r in rows:
        if r["mode"] not in modes:
            modes.append(r["mode"])
    agg = {m: {k: np.array([r["acc"][str(k)] for r in rows if r["mode"] == m])
               for k in KEEPS} for m in modes}
    base = agg.get("baseline")

    hdr = "  %-14s %9s " % ("用法", "训练样本") + \
          " ".join("%17s" % ("干净%.0f%%" % (k * 100)) for k in KEEPS) + "   vs 无增强"
    print(hdr)
    print("  " + "-" * (len(hdr) + 4))
    for m in sorted(modes, key=lambda m: -agg[m][0.60].mean()):
        nt = next(r["n_train"] for r in rows if r["mode"] == m)
        line = "  %-14s %9d " % (m, nt)
        for k in KEEPS:
            v = agg[m][k]
            line += "%10.4f±%.4f " % (v.mean(), v.std())
        if base is not None and m != "baseline":
            d = [agg[m][k].mean() - base[k].mean() for k in KEEPS]
            allpos, allneg = all(x > 0 for x in d), all(x < 0 for x in d)
            line += "  %s%+.4f" % ("✅" if allpos else ("❌" if allneg else "⚠混 "),
                                   float(np.mean(d)))
        print(line)

    print("\n  ✅=三档全优于无增强  ❌=三档全劣  ⚠=方向不一致(不可采信)")
    print("  ⚠本地只信方向不信幅度：实测倍率 117x~370x 且不稳定(天花板效应)，")
    print("     看到 +0.005 也值得交榜验证，别因为『才 0.005』就放弃。")

    out = os.path.join(P.CACHE_DIR, "aug_experiments.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"loss": args.loss, "C": args.C, "seeds": seeds,
                   "k_views": int(K_views), "view_consistency": cons, "rows": rows,
                   "created": time.strftime("%Y-%m-%d %H:%M:%S")}, f,
                  ensure_ascii=False, indent=2)
    print("\n  结果已存 → %s" % out)


if __name__ == "__main__":
    main()
