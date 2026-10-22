# -*- coding: utf-8 -*-
"""
长尾预演：用初赛数据人工造长尾，测「kNN 可疑度会不会冤枉尾部类」。

═══ 为什么现在就要做 ═══

初赛是均衡的（每类 201~213 张，比值 1.06），所以我们至今一行长尾代码都没写。
但官方原文明确：**复赛与半决赛都是长尾**，而复赛占线上综合成绩 40%、初赛不计分。
→ 在均衡数据上再抠 0.5 分对最终成绩零贡献；提前把长尾的坑趟出来才是高价值的事。

═══ 要验证的具体风险 ═══

推理：尾部类若只剩 5 张图，那么一张【标签完全正确】的尾类样本，
它的 20 个最近邻里最多只可能有 4 个同类——**邻居一致率天然就低**。
而头部类有几百张同类，一致率天然就高。
→ **可疑度会系统性冤枉尾类**，而尾类样本本就稀少，再被当噪声剔除/降权 = 直接抹掉。

这正是赛题最核心的难点：**区分「错标」与「正确但稀有」**。

═══ 实验设计：因果对照（关键）═══

不是简单地"在长尾数据上跑一遍看看"，而是：
  **取同一批样本、同样的标签，唯一改变的是「它所属类别有多少样本」，看可疑度怎么变。**
因为是同一个样本自己跟自己比，可疑度若升高，只可能由类别频率变化造成 —— 这是干净的因果。
均衡版本一直留在手上当对照组，这是真实复赛数据永远给不了的条件。

⚠合规：只用初赛数据做重采样，不引入任何外部数据，不碰测试集。

用法:
  python scripts/09_longtail_probe.py                    # 默认不平衡因子 10/50/100
  python scripts/09_longtail_probe.py --imb 50 --k 20
"""
import _bootstrap  # noqa: F401
from _bootstrap import banner

import argparse
import json
import os
import time

import numpy as np
import torch

from configs import paths as P


# ====================================================================== 造长尾
def make_longtail_index(y, n_classes, imb_factor, rng):
    """
    按指数分布重采样成长尾（CIFAR-LT / ImageNet-LT 的标准做法）。
      第 0 类保留 n_max 张，第 (C-1) 类保留 n_max/imb_factor 张，中间按指数插值。
    返回被保留样本的下标。
    """
    counts = np.bincount(y, minlength=n_classes)
    n_max = counts.min()                       # 用最小类打底，保证每类都取得到
    keep = []
    per_class = []
    for c in range(n_classes):
        n_c = max(1, int(round(n_max * (1.0 / imb_factor) ** (c / (n_classes - 1)))))
        idx_c = np.where(y == c)[0]
        sel = rng.choice(idx_c, size=min(n_c, len(idx_c)), replace=False)
        keep.append(sel)
        per_class.append(len(sel))
    return np.sort(np.concatenate(keep)), np.array(per_class)


# ====================================================================== 信号
def knn_agree(F, y, k, block, device):
    """返回每个样本的邻居一致率 + 加权得票差（与 04 同一套口径）。"""
    N = F.shape[0]
    Ft = torch.from_numpy(F).to(device)
    yt = torch.from_numpy(y).to(device)
    K = int(y.max()) + 1
    agree = np.empty(N, dtype=np.float32)
    gap = np.empty(N, dtype=np.float32)
    for s in range(0, N, block):
        e = min(s + block, N)
        sims = Ft[s:e] @ Ft.T
        kk = min(k + 1, N)
        v, i = torch.topk(sims, kk, dim=1)
        v, i = v[:, 1:], i[:, 1:]                       # 去掉自己
        lab = yt[i]
        agree[s:e] = (lab == yt[s:e, None]).float().mean(1).cpu().numpy()
        w = v.clamp_min(0)
        w = w / w.sum(1, keepdim=True).clamp_min(1e-12)
        acc = torch.zeros(e - s, K, device=device)
        acc.scatter_add_(1, lab, w)
        top = acc.max(1).values
        mine = acc.gather(1, yt[s:e, None]).squeeze(1)
        gap[s:e] = (top - mine).cpu().numpy()
        del sims, v, i, lab, w, acc
    del Ft
    if device == "cuda":
        torch.cuda.empty_cache()
    return agree, gap


def proto_signals(F, y, n_classes):
    """到最近类原型的相似度、以及与自己类原型的差值。"""
    proto = np.zeros((n_classes, F.shape[1]), dtype=np.float32)
    for c in range(n_classes):
        m = (y == c)
        if m.any():
            proto[c] = F[m].mean(0)
    proto /= np.maximum(np.linalg.norm(proto, axis=1, keepdims=True), 1e-12)
    sims = F @ proto.T
    return sims.max(1), sims[np.arange(len(y)), y]


def build_corrections(gap, y, per_cls, n_classes):
    """
    三种把「类别规模偏见」从可疑度里剔掉的做法，返回 {名称: 可疑度分数}。

    raw        原样（对照组，已知会冤枉尾类）
    within     **类内分位**：只和同类样本比排名。尾类只与尾类竞争，天然公平。
               代价=隐含假设"每个类的噪声率相同"，若某类整体质量差会清洗不足。
    residual   **回归校正**：先拟合"得票差随类别规模怎么变"这条趋势线，
               再取每个样本相对趋势线的【残差】。残差为正=比同规模的同伴更可疑。
               比类内分位更宽松——允许不同类噪声率不同，只扣掉规模带来的系统性部分。
    """
    out = {"raw (原样)": np.argsort(np.argsort(gap)) / (len(gap) - 1)}

    # ---- 类内分位 ----
    within = np.zeros(len(gap), dtype=np.float64)
    for c in range(n_classes):
        m = np.where(y == c)[0]
        if len(m) == 0:
            continue
        within[m] = (np.argsort(np.argsort(gap[m])) / max(len(m) - 1, 1)) if len(m) > 1 else 0.5
    out["within (类内分位)"] = within

    # ---- 回归校正：扣掉 log(类别样本数) 能解释的那部分 ----
    freq = per_cls[y].astype(np.float64)
    lf = np.log(freq)
    coef = np.polyfit(lf, gap, deg=3)              # 3 次多项式足够刻画这条单调趋势
    resid = gap - np.polyval(coef, lf)
    out["residual (回归校正)"] = np.argsort(np.argsort(resid)) / (len(resid) - 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imb", type=str, default="10,50,100", help="不平衡因子，逗号分隔")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--block", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)

    banner("09 长尾预演：可疑度会不会冤枉尾部类？")
    X = np.load(P.F_TRAIN_FEAT).astype(np.float32)
    y = np.load(P.F_TRAIN_LABEL).astype(np.int64)
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    K = int(y.max()) + 1
    cnt0 = np.bincount(y, minlength=K)
    print("  原始(均衡): %d 样本，%d 类，每类 %d~%d 张，比值 %.2f"
          % (len(y), K, cnt0.min(), cnt0.max(), cnt0.max() / cnt0.min()))

    # ---------------------------------------------------------------- 均衡基准
    banner("对照组：均衡数据上的信号（作为基准）")
    t0 = time.time()
    ag_bal, gap_bal = knn_agree(X, y, args.k, args.block, device)
    ms_bal, own_bal = proto_signals(X, y, K)
    print("  邻居一致率 均值 %.4f  得票差 均值 %.4f  (%.1fs)"
          % (ag_bal.mean(), gap_bal.mean(), time.time() - t0))

    results = {}
    for imb in [float(v) for v in args.imb.split(",")]:
        banner("长尾版本：不平衡因子 = %g" % imb)
        idx, per_cls = make_longtail_index(y, K, imb, rng)
        Xl, yl = X[idx], y[idx]
        print("  重采样后: %d 样本  每类 %d~%d 张  实际比值 %.1f"
              % (len(idx), per_cls.min(), per_cls.max(), per_cls.max() / per_cls.min()))

        ag_lt, gap_lt = knn_agree(Xl, yl, args.k, args.block, device)
        ms_lt, own_lt = proto_signals(Xl, yl, K)

        # ---- 因果对照：同一批样本，唯一变化是所属类别的样本数 ----
        pos = {g: i for i, g in enumerate(idx)}          # 全局下标 → 长尾版下标
        gi = idx                                          # 这些样本在两个版本里都在
        li = np.array([pos[g] for g in gi])
        freq = per_cls[y[gi]]                             # 每个样本所属类在长尾版的样本数

        d_ag = ag_lt[li] - ag_bal[gi]                     # 一致率的变化
        d_gap = gap_lt[li] - gap_bal[gi]                  # 得票差的变化（越大越可疑）

        print("\n  ⭐因果对照：同一样本、同一标签，只改其类别的样本数")
        print("  %-14s %8s %12s %12s %12s" %
              ("类别规模分组", "样本数", "一致率变化", "得票差变化", "结论"))
        # 按类别在长尾版中的样本数分组
        edges = [0, 5, 10, 25, 60, 10 ** 9]
        names = ["极尾(≤5)", "尾(6-10)", "中(11-25)", "中上(26-60)", "头(>60)"]
        rows = []
        for lo, hi, nm in zip(edges[:-1], edges[1:], names):
            m = (freq > lo) & (freq <= hi)
            if not m.any():
                continue
            a, g = float(d_ag[m].mean()), float(d_gap[m].mean())
            verdict = "⚠被冤枉" if (a < -0.05 and g > 0.02) else ("正常" if abs(a) < 0.05 else "偏移")
            print("  %-14s %8d %+12.4f %+12.4f %12s" % (nm, m.sum(), a, g, verdict))
            rows.append(dict(group=nm, n=int(m.sum()), d_agree=a, d_gap=g))

        # 相关性：类别越小，可疑度是否越高
        r_ag = float(np.corrcoef(np.log(freq), ag_lt[li])[0, 1])
        r_gap = float(np.corrcoef(np.log(freq), gap_lt[li])[0, 1])
        print("\n  corr(log 类别样本数, 邻居一致率) = %+.3f   (正值=类越小一致率越低)" % r_ag)
        print("  corr(log 类别样本数, 得票差)     = %+.3f   (负值=类越小越可疑)" % r_gap)

        # 若按可疑度剔除 top-20%，各规模组会被剔掉多少（这是最要命的数字）
        susp_lt = (np.argsort(np.argsort(gap_lt)) / (len(yl) - 1))
        thr = np.quantile(susp_lt, 0.8)
        print("\n  若按可疑度剔除最可疑的 20%%，各组的【被剔除率】：")
        for lo, hi, nm in zip(edges[:-1], edges[1:], names):
            m = (per_cls[yl] > lo) & (per_cls[yl] <= hi)
            if not m.any():
                continue
            dr = float((susp_lt[m] >= thr).mean())
            flag = "  ⚠⚠远高于平均" if dr > 0.35 else ("  ⚠偏高" if dr > 0.26 else "")
            print("    %-14s 被剔除 %5.1f%%%s" % (nm, dr * 100, flag))

        # ------------------------------------------------------------ 修正方案
        banner("修正方案对比（不平衡因子 %g）" % imb)
        print("  目标：让各规模组的被剔除率接近，不再系统性冤枉尾类。\n")
        fixes = build_corrections(gap_lt, yl, per_cls, K)
        print("  %-22s " % "可疑度算法" + " ".join("%11s" % n for n in names) + "   最大/最小")
        print("  " + "-" * 92)
        fix_stat = {}
        for fname, score in fixes.items():
            thr_f = np.quantile(score, 0.8)
            drops = []
            for lo, hi, nm in zip(edges[:-1], edges[1:], names):
                m = (per_cls[yl] > lo) & (per_cls[yl] <= hi)
                drops.append(float((score[m] >= thr_f).mean()) if m.any() else np.nan)
            dv = np.array([d for d in drops if not np.isnan(d)])
            ratio = dv.max() / max(dv.min(), 1e-9)
            flag = "✅ 已消除" if ratio < 1.6 else ("⚠ 仍偏" if ratio < 3 else "❌ 严重")
            print("  %-22s " % fname + " ".join("%10.1f%%" % (d * 100) for d in drops)
                  + "   %5.1fx %s" % (ratio, flag))
            fix_stat[fname] = dict(drops=[float(d) for d in drops], ratio=float(ratio))
        print("\n  『最大/最小』= 被剔除率最高组 ÷ 最低组。=1 表示完全公平，越大越偏。")

        results[str(imb)] = dict(n=int(len(idx)), per_class_min=int(per_cls.min()),
                                 per_class_max=int(per_cls.max()), groups=rows,
                                 corr_agree=r_ag, corr_gap=r_gap, fixes=fix_stat)

    # ---------------------------------------------------------------- 结论
    banner("结论")
    worst = max(results.values(), key=lambda r: abs(r["corr_gap"]))
    if abs(worst["corr_gap"]) > 0.15 or abs(worst["corr_agree"]) > 0.15:
        print("  ⚠⚠可疑度确实与类别规模强相关 —— 尾类会被系统性冤枉。")
        print("  → 复赛【不能】直接沿用当前的可疑度做清洗，必须先做类别内归一化：")
        print("     把「跟全体比」改成「只跟同类比」，即每个类内部单独算可疑度分位，")
        print("     这样尾类只与尾类竞争，不会因为同类邻居少就整体背锅。")
    else:
        print("  ✅未发现明显的类别规模偏见，当前可疑度在长尾下仍可用（但仍建议复赛复测）。")
    print("\n  ⚠局限：这里的『长尾』是人工重采样造的，真实复赛数据的长尾成因、")
    print("     噪声与类别频率的耦合方式都可能不同，结论只能作为预警，不能当定论。")

    out = os.path.join(P.CACHE_DIR, "longtail_probe.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"k": args.k, "seed": args.seed, "results": results,
                   "created": time.strftime("%Y-%m-%d %H:%M:%S")}, f,
                  ensure_ascii=False, indent=2)
    print("\n  结果已存 → %s" % out)


if __name__ == "__main__":
    main()
