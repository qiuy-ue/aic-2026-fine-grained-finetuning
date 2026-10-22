# -*- coding: utf-8 -*-
"""61 · 把两个【同 ckpt、同分辨率、同 seed】的视角池沿视角轴拼起来。⭐零 GPU、纯 numpy。

存在的理由：58 号一次只能建一批视角，而 59 号只吃**一个**池子文件。
想在已有池子上补几个视角（比如给 448 底座补 cc 族），就需要把两份拼成一份。
⛔ 重建整个池子是几十上百分钟 GPU；拼接是几十秒 IO。

⭐ 三道闸（任何一道不过就拒绝写盘）：
  ① ckpt / res / res_train / seed / n 必须逐项相同 —— 否则特征不可比
  ② 两边**共有的视角**必须逐元素完全相同 —— 这是"两次运行确实对齐"的硬证据
     （58 号的种子是按【每张图】固定的 seed*1000003+i，与 worker/batch 无关，
      所以同一个视角跑两次必须逐字节一样；如果不一样，说明有东西被改过，⛔不能拼）
  ③ 分类头 W/b/s 必须相同 —— 它们来自同一个 ckpt，不同就是拼错了对象

用法：
  python scripts\\61_merge_pools.py --a runs\\A_pool448_ep12 --b runs\\B_pool448_ep12
  python scripts\\61_merge_pools.py --a ... --b ... --out runs\\my_pool448_ep12_merged
"""
import _bootstrap  # noqa: F401
from _bootstrap import banner

import argparse
import json
import os
import shutil
import time

import numpy as np

from configs import paths as P

KEYS_MUST_MATCH = ["res", "res_train", "seed", "n", "ckpt", "epoch", "cosine", "subcenters"]


def load(d):
    z = np.load(os.path.join(d, "pool.npz"))
    m = json.load(open(os.path.join(d, "pool_meta.json"), encoding="utf-8"))
    return z, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="池子目录 A（视角排在前面）")
    ap.add_argument("--b", required=True, help="池子目录 B")
    ap.add_argument("--out", default=None, help="输出目录，默认自动命名")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    A = args.a if os.path.isabs(args.a) else os.path.join(P.PROJECT_ROOT, args.a)
    B = args.b if os.path.isabs(args.b) else os.path.join(P.PROJECT_ROOT, args.b)

    banner("61 视角池拼接")
    za, ma = load(A)
    zb, mb = load(B)
    print("  A  %-42s %2d 视角  @%dpx  ep%s" %
          (os.path.basename(A), len(ma["view_names"]), ma["res"], ma["epoch"]))
    print("     %s" % ",".join(ma["view_names"]))
    print("  B  %-42s %2d 视角  @%dpx  ep%s" %
          (os.path.basename(B), len(mb["view_names"]), mb["res"], mb["epoch"]))
    print("     %s" % ",".join(mb["view_names"]))

    # ── 闸①：元信息
    bad = [k for k in KEYS_MUST_MATCH if ma.get(k) != mb.get(k)]
    if bad:
        for k in bad:
            print("  ⛔ %s 不同：A=%r  B=%r" % (k, ma.get(k), mb.get(k)))
        raise SystemExit("⛔ 闸① 未通过：两个池子不是同一次实验的产物，拒绝拼接。")
    print("  ✅ 闸①  ckpt/res/seed/n 全部一致")

    Fa, Fb = za["F"], zb["F"]
    if Fa.shape[1:] != Fb.shape[1:]:
        raise SystemExit("⛔ 形状不符：%s vs %s" % (Fa.shape, Fb.shape))

    # ── 闸②：共有视角必须逐元素相同
    va, vb = list(ma["view_names"]), list(mb["view_names"])
    shared = [v for v in va if v in vb]
    if not shared:
        print("  ⚠ 闸②  两池没有共有视角 —— **无法证明对齐**。")
        print("     ⛔ 58 号的 orig 是确定性视角，正常情况下两次运行必然都含它。")
        raise SystemExit("⛔ 拒绝盲拼。请重建其中一个池子时带上 orig。")
    # ⚠ 2026-09-16 实测修正：**不能用 array_equal**。
    #   orig 是确定性视角（Resize+CenterCrop，无随机），但两次运行实测 91.6% 的行不逐字节相同。
    #   原因是 GPU matmul 的归约顺序非确定 + 特征以 float16 存盘，末位抖动被放大。
    #   ⛔ "不逐字节相同" ≠ "错位"。真正要判的是**同一行是不是同一张图**，那要看余弦：
    #      fp16 抖动 → cos ≈ 1 - 1e-4；错位 → cos 掉到类间水平（本模型约 0.5~0.9）。
    #   判据：最小余弦 ≥ 0.99 才放行（留足余量，⛔别卡在 0.9999 上被 fp16 尾巴绊倒）。
    MIN_COS = 0.99
    for v in shared:
        ea = Fa[va.index(v)].astype(np.float32)
        eb = Fb[vb.index(v)].astype(np.float32)
        na = ea / np.linalg.norm(ea, axis=1, keepdims=True).clip(1e-12)
        nb = eb / np.linalg.norm(eb, axis=1, keepdims=True).clip(1e-12)
        cos = (na * nb).sum(1)
        nbyte = int((Fa[va.index(v)] != Fb[vb.index(v)]).any(axis=1).sum())
        lo, med = float(cos.min()), float(np.median(cos))
        bad = int((cos < MIN_COS).sum())
        print("  闸②  共有视角 %-10s  逐字节不同 %d/%d 行 | 余弦 min=%.6f 中位=%.6f | "
              "低于 %.2f 的有 %d 行" % (v, nbyte, len(cos), lo, med, MIN_COS, bad))
        if bad:
            worst = np.argsort(cos)[:5]
            raise SystemExit("⛔ 闸② 未通过：%s 有 %d 行余弦 < %.2f（最差几行：%s，cos=%s）\n"
                             "   → 这是**真的错位**（两池的第 i 行不是同一张图），拒绝拼接。"
                             % (v, bad, MIN_COS, worst.tolist(),
                                np.round(cos[worst], 4).tolist()))
        print("       ✅ 全部 ≥ %.2f ⇒ 对齐，差异只是 fp16 末位抖动" % MIN_COS)

    # ── 闸③：分类头
    for k in ["W", "b"]:
        if not np.array_equal(za[k], zb[k]):
            raise SystemExit("⛔ 闸③ 未通过：分类头 %s 不同。" % k)
    if ("s" in za.files) != ("s" in zb.files):
        raise SystemExit("⛔ 闸③ 未通过：一个有 scale s 一个没有。")
    if "s" in za.files and float(za["s"]) != float(zb["s"]):
        raise SystemExit("⛔ 闸③ 未通过：scale s 不同（%g vs %g）" % (za["s"], zb["s"]))
    print("  ✅ 闸③  分类头 W/b%s 完全相同" % ("/s" if "s" in za.files else ""))

    # ── 拼接：A 全部 + B 里 A 没有的
    add = [v for v in vb if v not in va]
    if not add:
        raise SystemExit("⛔ B 没有任何 A 之外的视角，拼了也没用。")
    names = va + add
    print("\n  拼接结果：%d + %d = **%d 视角**" % (len(va), len(add), len(names)))
    print("     新增：%s" % ",".join(add))

    if args.dry_run:
        print("\n  DRY-RUN：⛔未写盘。")
        return

    F = np.concatenate([Fa, Fb[[vb.index(v) for v in add]]], axis=0)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = args.out or os.path.join(P.RUNS_DIR, "%s_pool%d_ep%s_m%d"
                                   % (stamp, ma["res"], ma["epoch"], len(names)))
    if not os.path.isabs(out):
        out = os.path.join(P.PROJECT_ROOT, out)
    os.makedirs(out, exist_ok=True)
    payload = dict(F=F, W=za["W"], b=za["b"])
    if "s" in za.files:
        payload["s"] = za["s"]
    np.savez(os.path.join(out, "pool.npz"), **payload)

    meta = dict(ma)
    meta["view_names"] = names
    meta["merged_from"] = [os.path.basename(A), os.path.basename(B)]
    meta["elapsed_min"] = float(ma.get("elapsed_min", 0)) + float(mb.get("elapsed_min", 0))
    meta["created"] = stamp
    with open(os.path.join(out, "pool_meta.json"), "w", encoding="utf-8") as fj:
        json.dump(meta, fj, ensure_ascii=False, indent=2)
    print("  ⭐输出 -> %s  (%.0f MB)"
          % (out, os.path.getsize(os.path.join(out, "pool.npz")) / 1e6))


if __name__ == "__main__":
    main()
