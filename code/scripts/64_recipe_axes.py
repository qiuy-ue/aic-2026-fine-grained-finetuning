# -*- coding: utf-8 -*-
"""61 · TTA 配方的三条【零 GPU 轴】扫描 + 单视角体检。⛔只读不写包。

轴 A  cc 族权重 w_cc（0 = base，∞ = 纯 cc）
轴 B  多尺度高分支权重 w_hi（0 = 纯 448，∞ = 纯 544）
轴 C  ⭐聚合空间：特征平均（现行） vs 概率平均 vs 对数概率平均
      ⚠ 余弦头下「特征平均 → L2 → 打分」与「logit 平均」argmax **严格等价**
        （L2 只乘一个逐图正标量，不改 argmax）——本脚本会实测验证这一条。
        但 **softmax 概率平均不等价**，是一条从没测过的轴。

外加：每个视角**单独**出预测，与合奏比分歧（无标签下唯一能做的视角体检）。
      ⛔ 这不是准确率，只是「离合奏多远」。

用法（主机）：
  D:\\Anaconda\\envs\\dl\\python.exe scripts\\61_recipe_axes.py
"""
import _bootstrap  # noqa: F401
from _bootstrap import banner

import argparse
import csv
import importlib.util
import json
import os
import sys

import numpy as np
from PIL import Image

from configs import paths as P


def _load_sibling(fname, modname):
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(modname, os.path.join(here, fname))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


pred26 = _load_sibling("26_predict_finetuned.py", "pred26")
_ex = pred26._ex

R = P.RUNS_DIR
POOL_HI = os.path.join(R, "20260916_210154_pool544_ep12", "pool.npz")
POOL_LO = os.path.join(R, "20260916_175916_pool448_ep12", "pool.npz")
LB = os.path.join(R, "20260916_131347_ms448_544_tta", "pred_results.csv")

CC = ["cc085", "cc070", "cc055", "cc040"]
CORNER = ["corner_tl", "corner_tr", "corner_bl", "corner_br"]
MILD = ["mild%d" % i for i in range(5)]
STRONG = ["strong%d" % i for i in range(5)]
BASE = ["orig"] + MILD
ALL20 = ["orig", "orig_flip"] + CC + CORNER + MILD + STRONG


def l2(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(1e-12)


def softmax(x):
    x = x - x.max(1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(1, keepdims=True)


def load_pool(path):
    z = np.load(path)
    meta = json.load(open(os.path.join(os.path.dirname(path), "pool_meta.json"),
                          encoding="utf-8"))
    F16 = z["F"]
    V, N, D = F16.shape
    Fn = np.empty((V, N, D), dtype=np.float32)
    for v in range(V):
        a = F16[v].astype(np.float32)
        Fn[v] = a / np.linalg.norm(a, axis=1, keepdims=True).clip(1e-12)
    head = dict(W=z["W"].astype(np.float32), b=z["b"].astype(np.float32))
    if "s" in z.files:
        head["s"] = float(z["s"])
    return Fn, meta, head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, default=3000, help="抽样几张测试图量尺寸")
    args = ap.parse_args()

    banner("61 配方轴扫描（零 GPU）")
    Fhi, mhi, head = load_pool(POOL_HI)
    vhi = mhi["view_names"]
    Flo, mlo, _ = load_pool(POOL_LO)
    vlo = mlo["view_names"]
    N = mhi["n"]
    s = head["s"]
    Wn = head["W"] / np.linalg.norm(head["W"], axis=1, keepdims=True).clip(1e-12)
    Xlo = l2(Flo[[vlo.index(v) for v in BASE]].mean(0))
    del Flo

    paths = _ex.list_test_files()
    cn = np.array(json.load(open(P.F_MANIFEST, encoding="utf-8"))["class_names"])
    bn = [os.path.basename(p) for p in paths]
    lbm = {}
    with open(LB, newline="", encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                lbm[row[0].strip()] = row[-1].strip()
    lb = np.array([lbm[b] for b in bn])

    def pred_feat(views, wmap=None, whi=1.0):
        idx = [vhi.index(v) for v in views]
        w = np.array([(wmap or {}).get(v, 1.0) for v in views], dtype=np.float32)
        Xh = l2(np.tensordot(w, Fhi[idx], axes=(0, 0)) / w.sum())
        X = l2((1.0 * Xlo + whi * Xh) / (1.0 + whi))
        return cn[np.argmax(s * (X @ Wn.T), 1)]

    c3 = pred_feat(BASE + CC)
    base = pred_feat(BASE)

    def rep(tag, p, ref=c3, refname="c3"):
        d = int((p != ref).sum())
        dl = int((p != lb).sum())
        print("  %-24s vs %-4s %5d (%.2f%%)  vs LB %5d  ±2SE(vs%s) %.3f"
              % (tag, refname, d, 100.0 * d / N, dl, refname,
                 2 * np.sqrt(d) / N * 100))
        return d

    # ───────────────────────────────────────────────── 轴 A：cc 权重
    banner("轴 A · cc 族权重 w_cc（base 里 orig/mild 权重固定 1；w_cc=1 即 c3）")
    for wcc in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 10.0]:
        if wcc == 0.0:
            p = base
        else:
            p = pred_feat(BASE + CC, {c: wcc for c in CC})
        rep("w_cc=%.2f (cc份额%.0f%%)" % (wcc, 100 * 4 * wcc / (6 + 4 * wcc)), p)
    rep("w_cc=inf (=ccpure,无mild)", pred_feat(["orig"] + CC))

    # ───────────────────────────────────────────────── 轴 B：高分支权重
    banner("轴 B · 多尺度高分支权重 w_hi（w_hi=1 即现行 45 号口径）")
    for whi in [0.0, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]:
        if whi == 0.0:
            idxb = [vlo.index(v) for v in BASE]  # 纯 448 base
            p = cn[np.argmax(s * (Xlo @ Wn.T), 1)]
        else:
            p = pred_feat(BASE + CC, None, whi)
        rep("c3 @ w_hi=%.2f" % whi, p)
    print("  ⚠ w_hi=0 那行 = 纯 448 base 单尺度（无 544 分支），仅作端点参照")

    # ───────────────────────────────────────────────── 轴 C：聚合空间
    banner("轴 C · 聚合空间：特征平均 / 概率平均 / 对数概率(几何)平均")

    def logits_of(views, wmap=None):
        idx = [vhi.index(v) for v in views]
        w = np.array([(wmap or {}).get(v, 1.0) for v in views], dtype=np.float32)
        return [(wi, s * (Fhi[j] @ Wn.T)) for wi, j in zip(w, idx)]

    lo_lg = s * (Xlo @ Wn.T)          # 低分支（已按特征平均聚合成一个分支）

    for nm, views in (("base", BASE), ("c3", BASE + CC), ("all20", ALL20)):
        lgs = logits_of(views)
        wsum = sum(wi for wi, _ in lgs)
        # ① 特征平均（现行）
        p_feat = pred_feat(views)
        # ② logit 平均（应与①严格等价——除了低分支的 L2 缩放，故单独验高分支）
        hi_mean_lg = sum(wi * lg for wi, lg in lgs) / wsum
        p_lgt_hi = cn[np.argmax(hi_mean_lg, 1)]
        p_feat_hi = cn[np.argmax(s * (l2(np.tensordot(
            np.ones(len(views), np.float32), Fhi[[vhi.index(v) for v in views]],
            axes=(0, 0)) / len(views)) @ Wn.T), 1)]
        eq = int((p_lgt_hi != p_feat_hi).sum())
        # ③ 概率平均（视角级 softmax 再平均；低分支按 1 份计入）
        Pacc = np.zeros((N, Wn.shape[0]), dtype=np.float32)
        for wi, lg in lgs:
            Pacc += wi * softmax(lg)
        Pacc /= wsum
        Pms = 0.5 * Pacc + 0.5 * softmax(lo_lg)
        p_prob = cn[np.argmax(Pms, 1)]
        # ④ 对数概率平均（= 几何平均概率）
        Lacc = np.zeros((N, Wn.shape[0]), dtype=np.float32)
        for wi, lg in lgs:
            z = lg - lg.max(1, keepdims=True)
            Lacc += wi * (z - np.log(np.exp(z).sum(1, keepdims=True)))
        Lacc /= wsum
        z = lo_lg - lo_lg.max(1, keepdims=True)
        Lms = 0.5 * Lacc + 0.5 * (z - np.log(np.exp(z).sum(1, keepdims=True)))
        p_geo = cn[np.argmax(Lms, 1)]
        print("\n  【%s】高分支内部 logit平均 vs 特征平均 分歧 = %d  （0 = 等价性验证通过）"
              % (nm, eq))
        rep("%s 概率平均(算术)" % nm, p_prob, p_feat, "特征")
        rep("%s 概率平均(几何)" % nm, p_geo, p_feat, "特征")

    del Fhi

    # ───────────────────────────────────────────────── 测试图尺寸 + cc 几何
    banner("测试图尺寸 & cc 族的物理几何")
    ws, hs = [], []
    step = max(1, len(paths) // args.sizes)
    for p in paths[::step]:
        try:
            with Image.open(p) as im:
                w, h = im.size
            ws.append(w)
            hs.append(h)
        except Exception:
            pass
    ws, hs = np.array(ws), np.array(hs)
    print("  抽样 %d 张：宽 中位%d min%d max%d ；高 中位%d min%d max%d"
          % (len(ws), np.median(ws), ws.min(), ws.max(),
             np.median(hs), hs.min(), hs.max()))
    uniq, cnt = np.unique(np.stack([ws, hs], 1), axis=0, return_counts=True)
    top = np.argsort(-cnt)[:6]
    for t in top:
        print("    %dx%d : %d 张 (%.1f%%)" % (uniq[t][0], uniq[t][1], cnt[t],
                                             100.0 * cnt[t] / len(ws)))
    Rr = 544
    print("\n  在 %dx%d 的图上，Resize(%d 短边)+CenterCrop(%d) 的实际视野与 token 密度："
          % (int(np.median(ws)), int(np.median(hs)), Rr, Rr))
    W0, H0 = float(np.median(ws)), float(np.median(hs))
    print("  %-8s %-14s %-12s %-12s %-14s %s"
          % ("scale", "视野(原图px)", "占全图面积", "px/patch", "JPEG亮度块/patch", "色度块/patch"))
    for sc in [1.0, 0.85, 0.70, 0.55, 0.40, 0.30, 0.25]:
        fov = sc * H0                       # 最终方形视野边长（原图像素）
        area = fov * fov / (W0 * H0)
        ppp = fov / (Rr / 32.0)
        print("  %-8.2f %-14s %-12.3f %-12.2f %-14.2f %.2f"
              % (sc, "%.0fx%.0f" % (fov, fov), area, ppp, ppp / 8.0, ppp / 16.0))
    print("\n  ⭐读法：4:2:0 ⇒ 色度块 16x16 原图px、亮度块 8x8。")
    print("  一个 ViT patch 若覆盖 <1 个亮度块，放大只是在插值 DCT 伪影，⛔买不到新信息。")


if __name__ == "__main__":
    main()
