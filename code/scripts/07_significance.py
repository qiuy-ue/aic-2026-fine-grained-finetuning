# -*- coding: utf-8 -*-
"""
给本地实验的结论量误差棒：一个方案比另一个高 0.005，到底是真差异还是随机波动？

═══ 为什么必须做这一步 ═══

06 的第一轮结果里，GCE 最优配置比 CE 高 0.005。差点就当成"GCE 更好"的结论了。
但干净60%那把尺子只有约 12000 个样本，准确率 0.93 附近的二项标准误约 0.0023，
两模型比较的差值标准误约 0.0033 —— **0.005 只有 1.5 个标准误，远达不到显著**。

更要命的是：那把尺子只在【两个真值点】上校准过（C=30 vs C=10，真实差 1.1 分）。
它被证明能分辨 1.1 分的差距，但**没有任何证据表明它能分辨 0.1 分的差距**。
拿它裁决 0.005 的差异，是超出适用范围使用。

═══ 本脚本做三件事 ═══
1. 多 seed 重复切分 → 看排名换个切分还稳不稳（消除"只跑了一个 split"的运气成分）
2. 配对 bootstrap → 直接估计"A 比 B 好"这件事的置信度，不靠正态近似
3. 报告"这把尺子的分辨率下限"→ 小于这个量级的差异一律不采信，必须交给排行榜裁决

用法:
  python scripts/07_significance.py --configs "ce:C=10, gce:C=10:q=0.3, sce:C=10"
  python scripts/07_significance.py --seeds 42,1,2,3,4
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

# 复用 06 里已经写好并验证过的损失函数与训练循环，避免两处实现漂移
_spec = importlib.util.spec_from_file_location(
    "loss_exp", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "06_loss_experiments.py"))
_le = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_le)

VAL_FRAC = 0.2
KEEPS = (0.40, 0.60, 0.80)


def parse_configs(s):
    """'ce:C=10, gce:C=10:q=0.3' → [{'loss':'ce','C':10.0,'extra':None}, ...]"""
    out = []
    for item in s.split(","):
        parts = [p.strip() for p in item.strip().split(":") if p.strip()]
        if not parts:
            continue
        cfg = {"loss": parts[0], "C": 10.0, "extra": None}
        for kv in parts[1:]:
            k, v = kv.split("=")
            if k.strip() == "C":
                cfg["C"] = float(v)
            else:
                cfg["extra"] = (k.strip(), float(v))
        cfg["tag"] = parts[0] + "".join(" %s" % p for p in parts[1:])
        out.append(cfg)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", type=str,
                    default="ce:C=10, gce:C=10:q=0.3, gce:C=100:q=0.7, sce:C=10")
    ap.add_argument("--seeds", type=str, default="42,1,2,3,4")
    ap.add_argument("--n-iter", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap 重采样次数")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = [int(s) for s in args.seeds.split(",")]
    cfgs = parse_configs(args.configs)

    banner("07 显著性检验：这些差异是真的吗？")
    X = np.load(P.F_TRAIN_FEAT).astype(np.float32)
    y = np.load(P.F_TRAIN_LABEL).astype(np.int64)
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    susp = np.load(os.path.join(P.CACHE_DIR, "noise_scores.npz"),
                   allow_pickle=True)["suspicion"]
    print("  配置 %d 个，切分种子 %d 个 → 共 %d 次训练"
          % (len(cfgs), len(seeds), len(cfgs) * len(seeds)))

    # ---------------------------------------------------------------- 尺子的分辨率
    banner("这把尺子的分辨率下限（先搞清楚它能分辨多小的差异）")
    n0 = int(len(y) * VAL_FRAC)
    print("  %-12s %10s %12s %14s" % ("干净度档位", "样本数", "单模型标准误", "两模型差值标准误"))
    for k in KEEPS:
        n = int(n0 * k)
        se = np.sqrt(0.93 * 0.07 / n)          # 按观测到的 acc≈0.93 估
        print("  %-12s %10d %12.4f %14.4f" % ("保留%.0f%%" % (k * 100), n, se, se * np.sqrt(2)))
    print("\n  → 小于『两模型差值标准误 × 2』的差异，本地无法可靠区分，必须交排行榜裁决。")
    print("  ⚠更根本的局限：这把尺子只在【2 个真值点】上校准过(C=30 vs C=10，真实差 1.1 分)，")
    print("     只证明了能分辨 1.1 分量级的差距，未证明能分辨 0.1 分量级。")

    # ---------------------------------------------------------------- 多 seed 训练
    banner("多种子重复实验")
    res = {c["tag"]: {k: [] for k in KEEPS} for c in cfgs}
    preds_by_seed = {}     # 存预测，供 bootstrap 用

    for sd in seeds:
        itr, iva = train_test_split(np.arange(len(y)), test_size=VAL_FRAC,
                                    random_state=sd, stratify=y)
        Xtr, ytr, Xv, yv, sv = X[itr], y[itr], X[iva], y[iva], susp[iva]
        masks = {k: (sv <= np.quantile(sv, k)) for k in KEEPS}
        preds_by_seed[sd] = {"yv": yv, "masks": masks}
        for c in cfgs:
            if c["extra"]:
                _le.LOSSES[c["loss"]][1][c["extra"][0]] = c["extra"][1]
            W, b = _le.train(Xtr, ytr, c["C"], c["loss"], args.n_iter, device, args.lr)
            pred = np.argmax(Xv @ W.T + b, 1)
            preds_by_seed[sd][c["tag"]] = pred
            for k in KEEPS:
                m = masks[k]
                res[c["tag"]][k].append(float((pred[m] == yv[m]).mean()))
        print("  seed=%-4d 完成" % sd)

    # ---------------------------------------------------------------- 汇总
    banner("跨种子结果：均值 ± 标准差")
    hdr = "  %-18s " % "配置" + " ".join("%18s" % ("干净%.0f%%" % (k * 100)) for k in KEEPS)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for c in cfgs:
        line = "  %-18s " % c["tag"]
        for k in KEEPS:
            v = np.array(res[c["tag"]][k])
            line += "%11.4f±%.4f " % (v.mean(), v.std())
        print(line)

    # 排名稳定性：每个 seed 下谁是第一
    banner("排名稳定性（换个切分，第一名会变吗）")
    for k in KEEPS:
        winners = []
        for i, sd in enumerate(seeds):
            best = max(cfgs, key=lambda c: res[c["tag"]][k][i])
            winners.append(best["tag"])
        uniq = sorted(set(winners))
        flag = "✅ 稳定" if len(uniq) == 1 else "⚠ 不稳定(%d 种)" % len(uniq)
        print("  干净%.0f%%: %s   各 seed 冠军: %s" % (k * 100, flag, winners))

    # ---------------------------------------------------------------- 配对 bootstrap
    base = cfgs[0]
    banner("配对 bootstrap vs 基准 %s（同一批样本上直接比，不靠正态近似）" % base["tag"])
    print("  %-18s %8s %10s %20s %s" % ("配置", "档位", "平均差值", "95%置信区间", "结论"))
    rng = np.random.default_rng(0)
    boot_rows = []
    for c in cfgs[1:]:
        for k in KEEPS:
            diffs = []
            for sd in seeds:
                d = preds_by_seed[sd]
                m = d["masks"][k]
                ok_a = (d[c["tag"]][m] == d["yv"][m]).astype(np.float32)
                ok_b = (d[base["tag"]][m] == d["yv"][m]).astype(np.float32)
                delta = ok_a - ok_b                       # 逐样本配对差
                idx = rng.integers(0, len(delta), size=(args.boot, len(delta)))
                diffs.append(delta[idx].mean(1))
            allb = np.concatenate(diffs)
            lo, hi = np.percentile(allb, [2.5, 97.5])
            mean = allb.mean()
            sig = "✅ 显著优于基准" if lo > 0 else ("❌ 显著劣于基准" if hi < 0
                                              else "⚠ 无法区分(区间跨0)")
            print("  %-18s %8s %+10.4f  [%+.4f, %+.4f]  %s"
                  % (c["tag"], "%.0f%%" % (k * 100), mean, lo, hi, sig))
            boot_rows.append(dict(config=c["tag"], keep=k, mean=float(mean),
                                  lo=float(lo), hi=float(hi), significant=bool(lo > 0 or hi < 0)))

    # ---------------------------------------------------------------- 结论
    banner("结论")
    winners = {}
    for c in cfgs[1:]:
        rs = [r for r in boot_rows if r["config"] == c["tag"]]
        allpos = all(r["lo"] > 0 for r in rs)
        anyneg = any(r["hi"] < 0 for r in rs)
        winners[c["tag"]] = "✅ 三档均显著优于基准" if allpos else (
            "❌ 存在显著劣于基准的档位" if anyneg else "⚠ 与基准无法区分")
        print("  %-20s %s" % (c["tag"], winners[c["tag"]]))
    if all("⚠" in v for v in winners.values()):
        print("\n  → 所有候选与基准都无法在本地区分。")
        print("     这不是坏消息，是【诚实的结论】：本地尺子的分辨率不够，")
        print("     应挑 2-3 个候选直接交排行榜，用真实分数裁决，别在本地过度调参。")

    out = os.path.join(P.CACHE_DIR, "significance.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"seeds": seeds, "configs": [c["tag"] for c in cfgs],
                   "acc": {t: {str(k): v for k, v in d.items()} for t, d in res.items()},
                   "bootstrap": boot_rows, "verdict": winners,
                   "created": time.strftime("%Y-%m-%d %H:%M:%S")}, f,
                  ensure_ascii=False, indent=2)
    print("\n  结果已存 → %s" % out)


if __name__ == "__main__":
    main()
