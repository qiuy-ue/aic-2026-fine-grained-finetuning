# -*- coding: utf-8 -*-
"""
四格对照：头到底该在【单视角】还是【TTA平均】特征上训练。

═══ 为什么这条路其实从没被测过 ═══

当前线上配置有一个**训练/预测分布不一致**：
  头在【单视角】特征上训练（06_loss_experiments.py:160-162）
  却应用到【TTA平均】特征上出提交（61.357 那版）

11_train_with_aug.py 的 mean_feature（−0.0019）曾被当成"训练侧增强无效"的证据，
但它测的**根本不是同一个运算**：

  | | 视角数 | 含原图锚点 | 平均前逐视角L2 | 评估用什么 |
  | 线上TTA(+2.357) |  6  | ✅ | ✅ | — |
  | mean_feature    |  5  | ❌ | ❌ 平均裸特征 | ⚠单视角(11:109) |

三处都不同，且它训练用平均、评估用单视角，**自己内部就不一致**。
→ "训练侧 TTA"是一条被误判为死路、实际从未走过的路。

═══ 四格缺一不可 ═══

  训练\评估   单视角        TTA
  单视角      ①纯基线      ②当前线上配置
  TTA         ③反向错配    ④⭐待测的一致配置

只跑④和②比不出所以然——①③是用来确认"收益来自一致性，不是来自别的"。

═══ 自检（不过就说明特征构造写错了）═══
第②格必须**复现 18 号脚本第③题**的 TTA 数字（干净60% +0.0059、分布匹配 +0.0187）。

用法:
  python scripts/22_train_side_tta.py
  python scripts/22_train_side_tta.py --submit   # 若④胜出，直接出提交文件
"""
import _bootstrap  # noqa: F401
from _bootstrap import banner

import argparse
import json
import os
import zipfile

import numpy as np
import torch
import torch.nn.functional as Fn
from sklearn.model_selection import train_test_split

from configs import paths as P

SEED, VAL_FRAC = 42, 0.2


def l2(a, axis=-1):
    return a / np.maximum(np.linalg.norm(a, axis=axis, keepdims=True), 1e-12)


def train_sce(Xtr, ytr, K, C=10.0, n_iter=300, lr=0.05, device="cuda", beta=1.0):
    """与 06/18/19 逐字同款：SCE(α=1,β=1,A=-4) + 全批量 Adam + reg=1/(2Cn)、截距不正则。
    ⛔别换优化器或损失，换了就没法跟历史数字比。"""
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
    del Xt, yt
    torch.cuda.empty_cache()
    return W.detach().cpu().numpy(), b.detach().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", action="store_true", help="④胜出时直接出提交文件")
    ap.add_argument("--force-submit", action="store_true", help="不管结果都出提交文件")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    banner("22 四格对照：头该在单视角还是 TTA 平均特征上训练")

    y = np.load(P.F_TRAIN_LABEL).astype(np.int64)
    K = int(y.max()) + 1
    X1 = l2(np.load(P.F_TRAIN_FEAT).astype(np.float32))          # 原图单视角

    # ⭐严格照抄线上配方：逐视角先 L2 → 连同原图一起平均 → 再 L2 一次
    A = np.load(os.path.join(P.CACHE_DIR, "train_features_aug.npy"), mmap_mode="r")
    print("  增强缓存 %s  →  构造 l2(mean(l2(原图), l2(视角1..%d)))" % (str(A.shape), A.shape[0]))
    acc = X1.copy()
    for v in range(A.shape[0]):
        acc += l2(np.asarray(A[v], dtype=np.float32))
    Xt_ = l2(acc / (A.shape[0] + 1))
    del acc, A

    idx = np.arange(len(y))
    itr, iva = train_test_split(idx, test_size=VAL_FRAC, random_state=SEED, stratify=y)
    yv = y[iva]

    wz = np.load(os.path.join(P.CACHE_DIR, "val_weights.npz"))
    assert np.array_equal(wz["val_idx"], iva), "权重文件的划分与此处不一致"
    w = wz["weight"]
    nz = np.load(os.path.join(P.CACHE_DIR, "noise_scores.npz"), allow_pickle=True)
    sv = nz["suspicion"][iva]
    m60 = sv <= np.quantile(sv, 0.60)

    # ---------------------------------------------------------------- 四格
    banner("四格结果（尺子：干净60% + 分布匹配）")
    heads = {"单视角": train_sce(X1[itr], y[itr], K, device=device),
             "TTA": train_sce(Xt_[itr], y[itr], K, device=device)}
    evals = {"单视角": X1[iva], "TTA": Xt_[iva]}

    print("  %-10s %-10s %10s %10s %10s" % ("训练特征", "评估特征", "干净60%", "分布匹配", "原始等权"))
    print("  " + "-" * 56)
    R = {}
    for tn, (W_, b_) in heads.items():
        for en, Xe in evals.items():
            c = (np.argmax(Xe @ W_.T + b_, 1) == yv).astype(np.float64)
            r = dict(clean60=float(c[m60].mean()),
                     matched=float((c * w).sum() / w.sum()), raw=float(c.mean()))
            R[(tn, en)] = r
            star = " ⭐" if (tn, en) == ("TTA", "TTA") else (
                " ←线上" if (tn, en) == ("单视角", "TTA") else "")
            print("  %-10s %-10s %10.4f %10.4f %10.4f%s"
                  % (tn, en, r["clean60"], r["matched"], r["raw"], star))

    # ---------------------------------------------------------------- 自检
    banner("自检：第②格必须复现 18 号脚本第③题的 TTA 数字")
    d60 = R[("单视角", "TTA")]["clean60"] - R[("单视角", "单视角")]["clean60"]
    dmt = R[("单视角", "TTA")]["matched"] - R[("单视角", "单视角")]["matched"]
    print("  本脚本  干净60%% %+.4f   分布匹配 %+.4f" % (d60, dmt))
    print("  18号    干净60%% %+.4f   分布匹配 %+.4f" % (0.0059, 0.0187))
    ok = abs(d60 - 0.0059) < 0.002 and abs(dmt - 0.0187) < 0.003
    print("  → %s" % ("✅ 对得上，特征构造正确" if ok
                      else "❌ 对不上！特征构造可能写错了，下面的结论别信"))

    # ---------------------------------------------------------------- 判读
    banner("判读：④(一致) vs ②(当前线上)")
    a, bq = R[("TTA", "TTA")], R[("单视角", "TTA")]
    g60, gmt = a["clean60"] - bq["clean60"], a["matched"] - bq["matched"]
    agree = np.sign(g60) == np.sign(gmt)
    print("  干净60%% %+.4f   分布匹配 %+.4f  -> %s"
          % (g60, gmt, "✅一致，可信" if agree else "⚠打架 → 本地无效，必须交榜"))
    print("  幅度：分布匹配 ×100 = **%+.2f 分**（冻结特征类系统低估约 0.5 分，可上修）" % (gmt * 100))
    print("  ⚠ ESS=11730、标准误≈0.0045；|差值|<0.004 判为零，不许解读。")
    if abs(gmt) < 0.004:
        print("  → 幅度低于分辨率下限，**判定为零**。")
    win = agree and gmt > 0.004

    with open(os.path.join(P.CACHE_DIR, "train_side_tta.json"), "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in R.items()}, f, ensure_ascii=False, indent=2)

    # ---------------------------------------------------------------- 提交
    if not (args.force_submit or (args.submit and win)):
        print("\n  （未出提交文件；加 --submit 在④胜出时出，或 --force-submit 强制出）")
        return

    banner("生成提交文件：TTA训练 + TTA预测")
    # ⭐提交用 refit_full：在全部 103218 张上重训，与 59.0/61.357 那两版的口径一致
    print("  头：在**全部 %d 张**上重训（refit_full，与历史提交口径一致）" % len(y))
    Wf, bf = train_sce(Xt_, y, K, device=device)
    ft = os.path.join(P.RUNS_DIR, "20260904_232902_sce_tta", "test_feat_tta.npy")
    Xte = np.load(ft).astype(np.float32)
    print("  测试特征：复用 %s（已是 l2(mean(l2(原图),l2(5视角)))，模长实测=1）" % os.path.basename(ft))
    paths = open(P.F_TEST_PATH, encoding="utf-8").read().splitlines()
    cn = json.load(open(P.F_MANIFEST, encoding="utf-8"))["class_names"]
    pred = np.argmax(Xte @ Wf.T + bf, 1)
    u, c = np.unique(pred, return_counts=True)
    print("  预测覆盖 %d/%d 类，每类 min=%d max=%d" % (len(u), K, c.min(), c.max()))

    od = os.path.join(P.RUNS_DIR, "20260905_trainside_tta")
    os.makedirs(od, exist_ok=True)
    csv = os.path.join(od, P.SUBMISSION_FILENAME)
    with open(csv, "w", encoding="utf-8", newline="\n") as f:
        f.write("".join("%s,%s\n" % (os.path.basename(p), cn[i]) for p, i in zip(paths, pred)))
    zp = csv.replace(".csv", ".zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(csv, arcname=P.SUBMISSION_FILENAME)
    np.savez(os.path.join(od, "head.npz"), W=Wf, b=bf, normalize=True, C=10.0, n_classes=K)

    # 与真正的 61.357 那版比改了多少
    prev = os.path.join(P.RUNS_DIR, "20260904_232902_sce_tta", P.SUBMISSION_FILENAME)
    if os.path.exists(prev):
        old = [l.split(",")[1].strip() for l in open(prev, encoding="utf-8")]
        d = sum(1 for x, i in zip(old, pred) if x != cn[i])
        print("  与 61.357 那版相比：改变 %d/%d (%.1f%%)" % (d, len(pred), 100 * d / len(pred)))

    print("\n  CSV : %s (%d 行)" % (csv, len(pred)))
    print("  ZIP : %s (%.1f MB)  ← 提交这个" % (zp, os.path.getsize(zp) / 1024 ** 2))


if __name__ == "__main__":
    main()
