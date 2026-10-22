# -*- coding: utf-8 -*-
"""60 · 一次载入池子，把所有候选配方一次算完，输出两两分歧矩阵。⭐零 GPU、纯 numpy。

存在的理由：59 号每跑一次都要重新 load 1GB 的池子，而我们要问的问题是**关系型**的
（A 和 B 差多少、C 站在 A 和 B 中间的哪里），必须把所有候选放在同一张表里才看得出来。

⭐ 本脚本要回答的**第一个**问题（其余都次要）：
   c3 相对榜上包的 914 张分歧里，多少来自「**换了池子**」、多少来自「**加了 cc**」？
   58 号 docstring 亲口说：本池的 mild0..4 与现有成绩包**不是同一次抽样**
   ⇒ 就算配方一模一样，用池子重组的 base 也会和榜上包差几百行。
   三角形一比就知道：
       同池base ↔ 榜上包   = 纯「换池子」的噪声（配方相同）
       同池base ↔ c3       = 纯「加 cc」的效应（池子相同）  ⭐这才是 c3 的真实改动量
       c3      ↔ 榜上包    = 914（两者叠加，已知）

⛔ 聚合口径逐字对齐 59/26/45，一个字都不许改：
   单分辨率  X = L2( Σ wᵢ·L2(Fᵢ) / Σ wᵢ )
   多尺度    X = L2( (X_lo + X_hi) / 2 )
   分类      lg = apply_head(X)

用法：
  python scripts\\60_recipe_matrix.py                    # 全套
  python scripts\\60_recipe_matrix.py --triangle-only    # 只跑上面那个三角（快）
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
POOL544 = os.path.join(R, "20260916_210154_pool544_ep12", "pool.npz")
POOL544B = os.path.join(R, "20260916_212040_pool544_ep12", "pool.npz")
POOL448 = os.path.join(R, "20260916_175916_pool448_ep12", "pool.npz")

# 已经出过包、其中一个已知榜分的 CSV
PACKS = {
    "BOARD_Q544": os.path.join(R, "20260916_131347_ms448_544_tta", "pred_results.csv"),
    "PACK_c3": os.path.join(R, "20260916_210948_rcp_c3_ms448_544_ep12", "pred_results.csv"),
    "PACK_all": os.path.join(R, "20260916_210739_rcp_all_ms448_544_ep12", "pred_results.csv"),
    "PACK_allw3": os.path.join(R, "20260916_211058_rcp_allw3_ms448_544_ep12", "pred_results.csv"),
    "PACK_c5": os.path.join(R, "20260916_211034_rcp_c5_ms448_544_ep12", "pred_results.csv"),
    "PACK_c2": os.path.join(R, "20260916_211012_rcp_c2_ms448_544_ep12", "pred_results.csv"),
}

CC = ["cc085", "cc070", "cc055", "cc040"]
CORNER = ["corner_tl", "corner_tr", "corner_bl", "corner_br"]
MILD = ["mild%d" % i for i in range(5)]
STRONG = ["strong%d" % i for i in range(5)]
BASE = ["orig"] + MILD


def l2(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(1e-12)


class Pool(object):
    """载入一次，之后组任意配方都是零 IO。⭐逐视角 L2 只做一次（59 号每次都重做）。"""

    def __init__(self, path):
        z = np.load(path)
        F = z["F"].astype(np.float32)                 # (V,N,D)
        self.Fn = l2(F)                               # 逐视角先 L2（口径同 26:19）
        del F
        self.meta = json.load(open(os.path.join(os.path.dirname(path), "pool_meta.json"),
                                   encoding="utf-8"))
        self.vn = self.meta["view_names"]
        self.head = dict(W=z["W"], b=z["b"])
        if "s" in z.files:
            self.head["s"] = float(z["s"])
        self.path = path

    def has(self, names):
        return all(n in self.vn for n in names)

    def form(self, names, wmap=None):
        if names is None:
            names = list(self.vn)
        idx = [self.vn.index(n) for n in names]
        w = np.array([(wmap or {}).get(n, 1.0) for n in names], dtype=np.float32)
        acc = np.tensordot(w, self.Fn[idx], axes=(0, 0)) / w.sum()
        return l2(acc)


def predict(X, pool):
    h = pool.head
    lg = pred26.apply_head(X, h["W"], h["b"], h.get("s"),
                           pool.meta.get("cosine", False), pool.meta.get("subcenters", 0))
    return np.argmax(lg, 1).astype(np.int32)


def read_csv_pred(p, cn_index):
    """CSV -> 与 list_test_files() 同序的类索引数组。"""
    m = {}
    with open(p, newline="", encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                m[row[0].strip()] = row[-1].strip()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--triangle-only", action="store_true")
    ap.add_argument("--check-dup-pool", action="store_true",
                    help="核实两份 544 池子是不是逐元素相同（要多载一次 1GB）")
    args = ap.parse_args()

    banner("60 配方分歧矩阵（零 GPU）")
    paths = _ex.list_test_files()
    names_on_disk = [os.path.basename(p) for p in paths]
    cn = json.load(open(P.F_MANIFEST, encoding="utf-8"))["class_names"]
    cn_idx = {c: i for i, c in enumerate(cn)}
    N = len(paths)
    print("  测试集 %d 张，%d 类" % (N, len(cn)))

    print("  载入 544 池 ...", flush=True)
    p544 = Pool(POOL544)
    print("    %d 视角：%s" % (len(p544.vn), ",".join(p544.vn)))
    print("  载入 448 池 ...", flush=True)
    p448 = Pool(POOL448)
    print("    %d 视角：%s" % (len(p448.vn), ",".join(p448.vn)))

    # ── 已出包的预测（从 CSV 读，与磁盘顺序对齐）
    preds = {}
    for k, path in PACKS.items():
        if not os.path.exists(path):
            print("  ⚠ 缺 %s" % path)
            continue
        m = read_csv_pred(path, cn_idx)
        miss = [n for n in names_on_disk if n not in m]
        if miss:
            print("  ⚠ %s 缺 %d 行，跳过" % (k, len(miss)))
            continue
        preds[k] = np.array([cn_idx[m[n]] for n in names_on_disk], dtype=np.int32)
    print("  读入已出包 %d 个：%s" % (len(preds), ",".join(preds)))

    # ── ⭐ 核心三角：同一个池子组出的 base
    def ms(names_hi, names_lo=BASE, wmap=None):
        Xh = p544.form(names_hi, wmap)
        Xl = p448.form(names_lo)
        return l2((Xl + Xh) / 2.0)

    print("\n  组【同池 base】（544 侧 base + 448 侧 base，配方与榜上包完全相同）...", flush=True)
    preds["SAMEPOOL_base"] = predict(ms(BASE), p544)
    print("  组【同池 c3】（复算，核对与 PACK_c3 是否 0 分歧）...", flush=True)
    preds["SAMEPOOL_c3"] = predict(ms(BASE + CC), p544)

    def dis(a, b):
        return int((preds[a] != preds[b]).sum())

    def fmt(a, b):
        d = dis(a, b)
        se = np.sqrt(d) / N * 100
        return d, 100.0 * d / N, 2 * se

    banner("⭐ 核心三角：914 张分歧里，多少是【换池子】、多少是【加 cc】")
    for a, b, why in [
        ("SAMEPOOL_c3", "PACK_c3", "自检：复算应当 = 0"),
        ("SAMEPOOL_base", "BOARD_Q544", "纯【换池子】：配方相同，只有 mild 重抽样"),
        ("SAMEPOOL_base", "SAMEPOOL_c3", "⭐纯【加 cc】：池子相同，只多了 cc×4"),
        ("PACK_c3", "BOARD_Q544", "已知 = 914（两者叠加）"),
    ]:
        if a in preds and b in preds:
            d, pct, eb = fmt(a, b)
            print("  %-16s ↔ %-14s  %5d 张 (%.2f%%)  误差棒 ±%.3f   %s"
                  % (a, b, d, pct, eb, why))

    if args.triangle_only:
        return

    # ── 新候选配方：全部零 GPU
    CAND = [
        ("pure_zoom", ["orig"] + CC, BASE, None, "扔掉 mild，只留原图+cc×4（放大假说的极端版）"),
        ("c3_w2cc", BASE + CC, BASE, {c: 2.0 for c in CC}, "c3 但 cc 权重×2"),
        ("c3_w3cc", BASE + CC, BASE, {c: 3.0 for c in CC}, "c3 但 cc 权重×3"),
        ("c3_flipo", BASE + CC + ["orig_flip"], BASE, None, "c3 + 原图翻转"),
        ("c3_corner", BASE + CC + CORNER, BASE, None, "c3 + 四角（两族叠加）"),
        ("c3_strong", BASE + CC + STRONG[:4], BASE, None, "c3 + 强随机×4"),
        ("all_nomild", ["orig", "orig_flip"] + CC + CORNER + STRONG, BASE, None,
         "全部但扔掉 mild（mild 是最不多样的一族）"),
        ("zoom_deep", ["orig", "cc070", "cc055", "cc040"], BASE, None, "只留深放大（扔 cc085）"),
    ]
    banner("新候选配方（零 GPU，均为 ms448_544）")
    for tag, hi, lo, wmap, why in CAND:
        if not p544.has(hi):
            print("  ⛔ %-12s 池子缺视角，跳过" % tag)
            continue
        preds[tag] = predict(ms(hi, lo, wmap), p544)
        d3, p3, e3 = fmt(tag, "SAMEPOOL_c3")
        db, pb, eb = fmt(tag, "BOARD_Q544")
        print("  %-12s %2d视角  vs c3: %5d (%.2f%%, ±%.3f)   vs 榜上: %5d (%.2f%%)   %s"
              % (tag, len(hi), d3, p3, e3, db, pb, why))

    # ── 全矩阵
    banner("两两分歧矩阵（张数）")
    keys = list(preds)
    w = max(len(k) for k in keys) + 1
    print(" " * w + "".join("%8s" % k[:7] for k in keys))
    for a in keys:
        print("%-*s" % (w, a) + "".join("%8d" % dis(a, b) for b in keys))

    # ── 覆盖率自检
    banner("各配方的类覆盖 / 预测分布（看有没有长尾坍塌）")
    for k in keys:
        cnt = np.bincount(preds[k], minlength=len(cn))
        nz = int((cnt > 0).sum())
        print("  %-16s 覆盖 %3d/%d  最大类 %3d  最小非零 %2d  空类 %3d  基尼 %.4f"
              % (k, nz, len(cn), cnt.max(), cnt[cnt > 0].min(), (cnt == 0).sum(),
                 gini(cnt)))

    if args.check_dup_pool and os.path.exists(POOL544B):
        banner("核实两份 544 池子是不是重复劳动")
        z2 = np.load(POOL544B)
        F2 = z2["F"]
        z1 = np.load(POOL544)
        F1 = z1["F"]
        same = bool(np.array_equal(F1[:, :2000], F2[:, :2000]))
        print("  前 2000 张 × 全部视角逐元素相同：%s" % ("是 ⇒ ⛔重复劳动" if same else "否"))


def gini(c):
    c = np.sort(c.astype(np.float64))
    n = len(c)
    if c.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * c).sum()) / (n * c.sum()) - (n + 1) / n)


if __name__ == "__main__":
    main()
