# -*- coding: utf-8 -*-
"""74 · 均衡分配出包：拿一个已出好的包的 test_logits.npy，按「测试集每类张数相同」重新分配类别，另出一个包。

⭐纯 CPU、不碰 GPU、不 import torch（训练在跑时也能安全执行，内存 < 1GB）。
⭐不改权重、不重训：只改「最后一步怎么从分数里挑类别」。原包一个字节都不动。

为什么（大白话）：
  赛题说测试集「类别分布均衡」（37444 张 / 750 类 ≈ 每类 50 张）。A1 包却把 48 个类
  各猜了 80 张以上、51 个类不到 25 张 —— 多出来的那部分**必然是错的**（≥12.9% 的测试图）。
  原因是模型拿不准时总往某些「长相通用」的类里倒。
  做法：给每个类配一个「加减分」，被猜多的类整体扣分、被猜少的类整体加分，
  反复调到每类被猜的张数都接近 50；每张图再按调整后的分数挑最高的那个类。
  已经很有把握的图（最高分远高于第二名）基本不会被改；被改的是那些本来就在两三个类之间摇摆的。

算法：Sinkhorn-Knopp（35 号同款）。把模型自己的 softmax 概率反复做「行归一 → 列拉到配额」，
  收敛后等价于每类加一个常数偏置 log v_k，预测 = argmax(logits/T + log v)。
  T=1.0 = 直接用模型自己的概率，⛔没有在测试集上调任何超参（测试集没有标签，也不该调）。

⚠合规：推理时用到了「整个测试集的预测分布」这个统计量（直推式）。
  它不参与训练、不改权重、单模型单次前向；但是否符合十一(二)1，由用户判断。

用法：
  python scripts/74_balance_pack.py --pkg runs\\20260925_130132_rcp_A1_w3cc_ms448_544_ep12
"""
import _bootstrap  # noqa: F401
from _bootstrap import banner

import argparse
import hashlib
import json
import os
import time
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(os.path.dirname(HERE), "runs")


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest().upper()


def sinkhorn_bias(P, iters):
    """P: (N,K) 每行和为 1 的概率。返回列缩放 v（K,），使 P·diag(v) 行归一后每列总量 = N/K。"""
    n, k = P.shape
    target = n / k
    v = np.ones(k, np.float64)
    for it in range(iters):
        R = P * v[None, :].astype(np.float32)
        R /= np.maximum(R.sum(1, keepdims=True), 1e-30)
        col = R.sum(0).astype(np.float64)
        v *= target / np.maximum(col, 1e-12)
        v /= np.exp(np.log(v).mean())            # 只是数值归一，不改 argmax
    return v, col


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", required=True, help="已出好的包目录（含 pred_results.csv + test_logits.npy）")
    ap.add_argument("--T", type=float, default=1.0, help="温度；1.0=用模型自己的概率（默认，不调）")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--dry-run", action="store_true", help="只打印统计，不写包")
    args = ap.parse_args()

    pkg = os.path.abspath(args.pkg)
    csv0 = os.path.join(pkg, "pred_results.csv")
    lg = np.load(os.path.join(pkg, "test_logits.npy")).astype(np.float32)
    rows = [ln.rstrip("\n").split(",") for ln in open(csv0, encoding="utf-8") if ln.strip()]
    names = [r[0] for r in rows]
    lab0 = [r[1] for r in rows]
    N, K = lg.shape
    banner("74 均衡分配出包  源包 %s" % os.path.basename(pkg))
    print("  logits (%d, %d)   CSV %d 行" % (N, K, len(rows)))
    if len(rows) != N:
        raise SystemExit("⛔ 行数不符：CSV %d vs logits %d" % (len(rows), N))
    base = lg.argmax(1)
    # ⛔对齐自检：logits 的第 i 行必须就是 CSV 的第 i 行（类别号 = 四位补零的下标）
    same = sum(1 for i in range(N) if "%04d" % base[i] == lab0[i])
    print("  ⭐对齐自检：argmax(logits) 与原 CSV 一致 %d/%d" % (same, N))
    if same != N:
        raise SystemExit("⛔ logits 与 CSV 对不上，拒绝出包")

    z = lg / args.T
    z -= z.max(1, keepdims=True)
    P = np.exp(z)
    P /= P.sum(1, keepdims=True)
    v, col = sinkhorn_bias(P, args.iters)
    new = (P * v[None, :].astype(np.float32)).argmax(1)

    c0 = np.bincount(base, minlength=K)
    c1 = np.bincount(new, minlength=K)
    tgt = N / K
    ex0 = np.maximum(c0 - tgt, 0).sum()
    ex1 = np.maximum(c1 - tgt, 0).sum()
    chg = int((new != base).sum())
    conf = P.max(1)
    print("  软分配每类总量：min=%.1f max=%.1f（目标 %.1f）" % (col.min(), col.max(), tgt))
    print("  每类预测张数  原包：min=%d 中位=%d max=%d 覆盖 %d 类 | 超出配额必错 %.0f 张 (%.2f%%)"
          % (c0.min(), np.median(c0), c0.max(), (c0 > 0).sum(), ex0, 100 * ex0 / N))
    print("                新包：min=%d 中位=%d max=%d 覆盖 %d 类 | 超出配额必错 %.0f 张 (%.2f%%)"
          % (c1.min(), np.median(c1), c1.max(), (c1 > 0).sum(), ex1, 100 * ex1 / N))
    print("  ⭐改判 %d 张 (%.2f%%)   ⇒ 榜分变化必在 ±%.2f 分以内" % (chg, 100 * chg / N, 100 * chg / N))
    m = new != base
    if chg:
        print("  被改判的图，原最高概率：中位 %.3f  p90 %.3f（没被改的：中位 %.3f）"
              % (np.median(conf[m]), np.percentile(conf[m], 90), np.median(conf[~m])))
    bias = np.log(v)
    top_dn = np.argsort(bias)[:5]
    top_up = np.argsort(-bias)[:5]
    print("  扣分最多的类：" + "  ".join("%04d(原%d→%d)" % (k, c0[k], c1[k]) for k in top_dn))
    print("  加分最多的类：" + "  ".join("%04d(原%d→%d)" % (k, c0[k], c1[k]) for k in top_up))

    if args.dry_run:
        print("\n  DRY-RUN：⛔未写包")
        return
    stamp = time.strftime("%Y%m%d_%H%M%S")
    suf = "_bal" if args.T == 1.0 else "_balT%03d" % round(args.T * 100)
    out = os.path.join(RUNS, "%s_%s%s" % (stamp, os.path.basename(pkg).split("_", 2)[2], suf))
    os.makedirs(out)
    csv1 = os.path.join(out, "pred_results.csv")
    with open(csv1, "w", encoding="utf-8", newline="\n") as f:     # 与原包同格式：无表头、无 BOM、LF
        for nm, c in zip(names, new):
            f.write("%s,%04d\n" % (nm, c))
    zp = os.path.join(out, "pred_results.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv1, "pred_results.csv")
    with open(os.path.join(out, "balance.json"), "w", encoding="utf-8") as fj:
        json.dump(dict(source_pkg=pkg, source_zip_md5=md5(os.path.join(pkg, "pred_results.zip")),
                       method="sinkhorn uniform quota", T=args.T, iters=args.iters,
                       changed=chg, excess_before=float(ex0), excess_after=float(ex1),
                       created=stamp), fj, ensure_ascii=False, indent=2)
    print("\n  ⭐输出 -> %s" % zp)
    print("  MD5 %s" % md5(zp))


if __name__ == "__main__":
    main()
