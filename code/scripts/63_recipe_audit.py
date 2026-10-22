# -*- coding: utf-8 -*-
"""60 · TTA 配方审计：零 GPU，把 c3 的 +0.5808 拆成「换池子」和「加 cc」两部分，
并枚举现有池子就能做出来的全部候选配方，实算两两配对分歧矩阵。

⛔ 只读，不写任何成绩包（除非 --emit 指定要出的配方名）。
聚合口径逐字对齐 59/45/26：
    单分辨率 X = L2( Σwᵢ·L2(Fᵢ) / Σwᵢ )
    多尺度   X = L2( (w_lo·X_lo + w_hi·X_hi) / (w_lo+w_hi) )     w_lo=w_hi=1 时 = 45 号
    分类     余弦头 lg = s·(f̂ @ Ŵ.T)

用法（主机）：
  D:\\Anaconda\\envs\\dl\\python.exe scripts\\60_recipe_audit.py
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

CC = ["cc085", "cc070", "cc055", "cc040"]
CORNER = ["corner_tl", "corner_tr", "corner_bl", "corner_br"]
MILD = ["mild%d" % i for i in range(5)]
STRONG = ["strong%d" % i for i in range(5)]
BASE = ["orig"] + MILD
ALL20 = (["orig", "orig_flip"] + CC + CORNER + MILD + STRONG)

# (名字, 视角列表, 权重字典, 高分支相对权重 w_hi)
CANDS = [
    ("base",        BASE,                              None,                1.0),
    ("c3",          BASE + CC,                         None,                1.0),
    ("c2",          BASE + STRONG[:4],                 None,                1.0),
    ("c5",          BASE + CORNER,                     None,                1.0),
    ("all",         ALL20,                             None,                1.0),
    ("allw3",       ALL20,                             {"orig": 3.0},       1.0),
    # ── 纯放大（= 21 号表的 04 / B4，那里实测 −0.0100，⚠ 预注册的不利证据）
    ("ccpure",      ["orig"] + CC,                     None,                1.0),
    ("cc2only",     ["orig", "cc085", "cc070"],        None,                1.0),   # 21 号 05
    # ── cc 加权（把 c3 沿 cc 方向推）
    ("c3_ccw2",     BASE + CC,     {c: 2.0 for c in CC},                     1.0),
    ("c3_ccw3",     BASE + CC,     {c: 3.0 for c in CC},                     1.0),
    ("c3_ccw5",     BASE + CC,     {c: 5.0 for c in CC},                     1.0),
    ("c3_ccw05",    BASE + CC,     {c: 0.5 for c in CC},                     1.0),
    ("c3_origw3",   BASE + CC,     {"orig": 3.0},                            1.0),
    ("c3_no040",    BASE + CC[:3],                     None,                1.0),
    ("c3_040w2",    BASE + CC,     {"cc040": 2.0},                           1.0),
    # ── c3 + 别的族
    ("c3_flip",     BASE + CC + ["orig_flip"],         None,                1.0),
    ("c3_corner",   BASE + CC + CORNER,                None,                1.0),
    ("c3_strong",   BASE + CC + STRONG,                None,                1.0),
    ("c3_cs",       BASE + CC + CORNER + STRONG,       None,                1.0),   # all 去 orig_flip
    # ── 去掉 mild
    ("all_nomild",  ["orig", "orig_flip"] + CC + CORNER + STRONG, None,      1.0),
    ("div9",        ["orig", "orig_flip"] + CC + CORNER,          None,      1.0),
    ("all_ccw2",    ALL20,         {c: 2.0 for c in CC},                     1.0),
    ("all_ccw3",    ALL20,         {c: 3.0 for c in CC},                     1.0),
    # ── 多尺度分支权重（高分支加权）
    ("c3_hi2",      BASE + CC,                         None,                2.0),
    ("c3_hi3",      BASE + CC,                         None,                3.0),
    ("base_hi2",    BASE,                              None,                2.0),
    ("c3_ccw2_hi2", BASE + CC,     {c: 2.0 for c in CC},                     2.0),
    ("all_hi2",     ALL20,                             None,                2.0),
]

# 已存在的成绩包（CSV）
PACKS = {
    "LB_Q544":  os.path.join(R, "20260916_131347_ms448_544_tta", "pred_results.csv"),
    "pk_c3":    os.path.join(R, "20260916_210948_rcp_c3_ms448_544_ep12", "pred_results.csv"),
    "pk_all":   os.path.join(R, "20260916_210739_rcp_all_ms448_544_ep12", "pred_results.csv"),
    "pk_allw3": os.path.join(R, "20260916_211058_rcp_allw3_ms448_544_ep12", "pred_results.csv"),
    "pk_c5":    os.path.join(R, "20260916_211034_rcp_c5_ms448_544_ep12", "pred_results.csv"),
    "pk_c2":    os.path.join(R, "20260916_211012_rcp_c2_ms448_544_ep12", "pred_results.csv"),
    "LB_512":   os.path.join(R, "20260915_015302_ms448_512_tta", "pred_results.csv"),
}


def l2(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(1e-12)


def load_pool(path):
    z = np.load(path)
    meta = json.load(open(os.path.join(os.path.dirname(path), "pool_meta.json"),
                          encoding="utf-8"))
    F16 = z["F"]                                   # (V,N,D) fp16
    V, N, D = F16.shape
    Fn = np.empty((V, N, D), dtype=np.float32)
    for v in range(V):                             # 逐视角转换+归一，控制峰值内存
        a = F16[v].astype(np.float32)
        Fn[v] = a / np.linalg.norm(a, axis=1, keepdims=True).clip(1e-12)
    head = dict(W=z["W"].astype(np.float32), b=z["b"].astype(np.float32))
    if "s" in z.files:
        head["s"] = float(z["s"])
    del F16
    return Fn, meta, head


def agg(Fn, vn, names, wmap):
    idx = [vn.index(n) for n in names]
    w = np.array([(wmap or {}).get(n, 1.0) for n in names], dtype=np.float32)
    return l2(np.tensordot(w, Fn[idx], axes=(0, 0)) / w.sum())


def read_csv_pred(p):
    m = {}
    with open(p, newline="", encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                m[row[0].strip()] = row[-1].strip()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=POOL_HI)
    ap.add_argument("--pool-lo", default=POOL_LO)
    ap.add_argument("--out", default=os.path.join(P.CACHE_DIR, "recipe_audit.json"))
    args = ap.parse_args()

    banner("60 配方审计（零 GPU）")
    print("  hi = %s" % args.pool)
    print("  lo = %s" % args.pool_lo)

    Fhi, mhi, head = load_pool(args.pool)
    vhi = mhi["view_names"]
    Flo, mlo, _ = load_pool(args.pool_lo)
    vlo = mlo["view_names"]
    print("  hi %d 视角 @%d  lo %d 视角 @%d  n=%d" %
          (len(vhi), mhi["res"], len(vlo), mlo["res"], mhi["n"]))

    Xlo = agg(Flo, vlo, BASE, None)                # 低分支固定 base
    del Flo

    paths = _ex.list_test_files()
    cn = json.load(open(P.F_MANIFEST, encoding="utf-8"))["class_names"]
    base_names = [os.path.basename(p) for p in paths]
    assert len(paths) == mhi["n"]

    s = head.get("s")
    cosine = bool(mhi.get("cosine", False))
    sub = int(mhi.get("subcenters", 0))

    # ── 逐候选出预测（类名字符串数组，方便与 CSV 直接比）
    preds = {}
    meta_rows = []
    for name, views, wmap, whi in CANDS:
        miss = [v for v in views if v not in vhi]
        if miss:
            print("  ⛔ 跳过 %s：池子缺 %s" % (name, miss))
            continue
        Xh = agg(Fhi, vhi, views, wmap)
        X = l2((1.0 * Xlo + whi * Xh) / (1.0 + whi))
        lg = pred26.apply_head(X, head["W"], head["b"], s, cosine, sub)
        p = np.argmax(lg, 1)
        # 置信裕度（top1-top2），用于看改动落在哪
        part = np.partition(lg, -2, axis=1)
        marg = part[:, -1] - part[:, -2]
        preds[name] = np.array([cn[i] for i in p])
        cov = len(set(p.tolist()))
        meta_rows.append(dict(name=name, nviews=len(views), whi=whi,
                              cov=cov, margin_med=float(np.median(marg)),
                              weights={k: v for k, v in (wmap or {}).items()}))
        print("  %-13s %2d 视角 w_hi=%.0f  覆盖 %d/500  中位裕度 %.4f"
              % (name, len(views), whi, cov, float(np.median(marg))))
    del Fhi

    # ── 已有包
    for k, p in PACKS.items():
        if not os.path.exists(p):
            print("  ⚠ 缺 %s" % p)
            continue
        m = read_csv_pred(p)
        preds[k] = np.array([m[b] for b in base_names])

    keys = [k for k, _, _, _ in CANDS if k in preds] + [k for k in PACKS if k in preds]
    N = len(base_names)

    banner("配对分歧矩阵（张数；%d 张总量）" % N)
    hdr = "%-13s" % "" + "".join("%9s" % k[:8] for k in keys)
    print(hdr)
    M = {}
    for a in keys:
        row = "%-13s" % a
        for b in keys:
            d = int((preds[a] != preds[b]).sum())
            M[(a, b)] = d
            row += "%9d" % d
        print(row)

    banner("⭐ 关键三角：914 张分歧的分解")
    tri = [("base(池) vs LB_Q544  = 【换池子/mild 重抽样】", "base", "LB_Q544"),
           ("c3(池)   vs base(池) = 【加 cc 这一族】", "c3", "base"),
           ("c3(池)   vs LB_Q544  = 【两者合计，台账记的 914】", "c3", "LB_Q544"),
           ("c3(池)   vs pk_c3    = 【自检，应当 = 0】", "c3", "pk_c3")]
    for lab, a, b in tri:
        if a in preds and b in preds:
            d = M[(a, b)]
            se = np.sqrt(d) / N * 100
            print("  %-46s %5d 张 (%.2f%%)  误差棒 ±%.3f" % (lab, d, 100.0 * d / N, 2 * se))

    if "base" in preds and "c3" in preds and "LB_Q544" in preds:
        a = preds["base"] != preds["LB_Q544"]
        b = preds["c3"] != preds["base"]
        c = preds["c3"] != preds["LB_Q544"]
        print("\n  维恩：|换池|=%d  |加cc|=%d  |合计|=%d  |换池∩加cc|=%d"
              % (a.sum(), b.sum(), c.sum(), (a & b).sum()))
        print("  若两部分独立叠加，合计应≈ %d；实测 %d" % (a.sum() + b.sum() - 2 * (a & b).sum(), c.sum()))

    banner("每个候选 vs 【榜上 80.4542 的 c3 包】的分歧（= 下一发的信息量与误差棒）")
    print("  %-13s %7s %7s %8s %8s" % ("候选", "vs c3", "vs LB", "±2SE(c3)", "P(噪声≥+386张)"))
    for k in keys:
        if k in ("pk_c3", "c3"):
            pass
        d3 = M[(k, "pk_c3")] if ("pk_c3", k) in M or (k, "pk_c3") in M else None
        dl = M[(k, "LB_Q544")]
        if d3 is None:
            continue
        se = np.sqrt(d3) / N * 100 if d3 else 0.0
        # 纯噪声（E[Δ]=0）下净变化 SD = sqrt(d3)/2；要净 +386 张
        p = 0.0
        if d3 > 0:
            sd = np.sqrt(d3) / 2.0
            from math import erfc, sqrt
            p = 0.5 * erfc((386.0 / sd) / sqrt(2))
        print("  %-13s %7d %7d %8.3f %12.4f%%" % (k, d3, dl, 2 * se, 100 * p))

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(dict(keys=keys, n=N,
                       matrix={"%s|%s" % k: v for k, v in M.items()},
                       cands=meta_rows), f, ensure_ascii=False, indent=1)
    print("\n  已写 %s" % args.out)


if __name__ == "__main__":
    main()
