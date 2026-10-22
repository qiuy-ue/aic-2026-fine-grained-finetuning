# -*- coding: utf-8 -*-
"""
检测官方点名的第三类噪声：【弱相关标注】——标签和图片有语义联系，
但这张图不足以支撑这个分类（鸟巢照片标成鸟种、栖息地风景、标本、插画、远景……）。

═══ 为什么现有的 kNN 检测抓不住它（结构性盲区）═══

假设某类 200 张图里有 30 张是鸟巢照片。这 30 张【彼此长得很像】，
在特征空间里聚成一团、互相当邻居 → 每张的"邻居一致率"都极高
→ **它们互相作证，把彼此洗白了**。

"看邻居"只能抓【孤立】的错误，抓不住【成建制】的错误。
而网上爬来的数据，噪声恰恰倾向于成建制（搜某物种搜到一批鸟巢图，不是随机事件）。

═══ 本脚本的两步判据 ═══

第1步「类内分裂」：每个类内部是不是分成了两坨互不相像的？
  正常类应是一坨；被污染的类会分成两坨（一坨鸟、一坨鸟巢）。
  ⚠单独用会误伤——雄/雌、幼/成、花/整株 这些合法的类内差异也会分成两坨，那是该学的。

第2步「跨类呼应」← 决定性判据：
  把每个类里"小的那坨"拿出来跨类比较。
  · A 的小坨和 B 的小坨【互相很像】 → 它们是同一种东西(都是鸟巢/风景/标本)
    却被贴了不同物种标签 → **弱相关污染的铁证**
  · A 的小坨只跟 A 类内部像 → 是该物种合法的另一种形态 → **该留着**

  精髓：污染是【跨类共享】的（所有鸟巢都像鸟巢），
        合法的类内变异是【各类独有】的（A 的雌鸟不像 B 的雌鸟）。

⚠这是诊断工具，不是定论。任何"剔除"决定都必须交排行榜验证——
  本地评估对这类基于自身特征的清洗存在循环优化，测不准（见 08 的实证）。

用法:
  python scripts/12_detect_weakrelated.py            # CPU 运行，不与 GPU 任务抢资源
"""
import _bootstrap  # noqa: F401
from _bootstrap import banner

import argparse
import json
import os
import time

import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

from configs import paths as P


def l2(a, axis=-1):
    return a / np.maximum(np.linalg.norm(a, axis=axis, keepdims=True), 1e-12)


def split_classes(X, y, n_classes, min_size, seed):
    """
    第1步：每个类内部做 2-聚类，返回每个类的（小坨中心、大坨中心、小坨占比、分裂程度）。
    分裂程度 = 1 - cos(小坨中心, 大坨中心)，越大说明两坨越不像。
    """
    minor_c = np.zeros((n_classes, X.shape[1]), dtype=np.float32)
    major_c = np.zeros((n_classes, X.shape[1]), dtype=np.float32)
    minor_frac = np.zeros(n_classes, dtype=np.float32)
    split_deg = np.zeros(n_classes, dtype=np.float32)
    minor_mask = np.zeros(len(y), dtype=bool)

    for c in tqdm(range(n_classes), desc="类内分裂", unit="cls", dynamic_ncols=True):
        idx = np.where(y == c)[0]
        if len(idx) < min_size:
            continue
        km = KMeans(n_clusters=2, n_init=4, random_state=seed).fit(X[idx])
        lab = km.labels_
        n0, n1 = (lab == 0).sum(), (lab == 1).sum()
        small = 0 if n0 <= n1 else 1
        cs = l2(X[idx][lab == small].mean(0))
        cb = l2(X[idx][lab != small].mean(0))
        minor_c[c], major_c[c] = cs, cb
        minor_frac[c] = min(n0, n1) / len(idx)
        split_deg[c] = 1.0 - float(cs @ cb)
        minor_mask[idx[lab == small]] = True
    return minor_c, major_c, minor_frac, split_deg, minor_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-size", type=int, default=30, help="类样本数少于此值不做分裂")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--topk", type=int, default=15, help="展示最可疑的前几个类")
    args = ap.parse_args()

    banner("12 弱相关标注检测（类内分裂 + 跨类呼应）")
    X = l2(np.load(P.F_TRAIN_FEAT).astype(np.float32))
    y = np.load(P.F_TRAIN_LABEL).astype(np.int64)
    n_cls = int(y.max()) + 1
    print("  样本 %d，类别 %d  （CPU 运行，不占 GPU）" % (len(y), n_cls))

    t0 = time.time()
    minor_c, major_c, minor_frac, split_deg, minor_mask = split_classes(
        X, y, n_cls, args.min_size, args.seed)
    print("  完成，耗时 %.1fs" % (time.time() - t0))

    banner("第1步：类内分裂程度")
    print("  分裂程度 = 1 - cos(小坨中心, 大坨中心)，越大说明类内有两种互不相像的东西")
    print("  小坨占比: 均值 %.3f   分裂程度: 均值 %.4f  中位 %.4f  最大 %.4f"
          % (minor_frac.mean(), split_deg.mean(), np.median(split_deg), split_deg.max()))

    # ---------------------------------------------------------------- 第2步
    banner("第2步：跨类呼应（决定性判据）")
    print("  比较两件事：")
    print("    (a) 各类【小坨】之间的相似度  —— 若污染是跨类共享的，这个值会偏高")
    print("    (b) 各类【大坨】之间的相似度  —— 正常物种之间应当互不相像，作为参照\n")

    valid = split_deg > 0
    mc, Mc = minor_c[valid], major_c[valid]
    n_v = valid.sum()

    def offdiag_mean(A):
        S = A @ A.T
        iu = np.triu_indices(len(A), k=1)
        return S[iu]

    s_minor = offdiag_mean(mc)
    s_major = offdiag_mean(Mc)
    print("  (a) 小坨互相似度: 均值 %.4f  95%%分位 %.4f  最大 %.4f"
          % (s_minor.mean(), np.percentile(s_minor, 95), s_minor.max()))
    print("  (b) 大坨互相似度: 均值 %.4f  95%%分位 %.4f  最大 %.4f"
          % (s_major.mean(), np.percentile(s_major, 95), s_major.max()))
    delta = s_minor.mean() - s_major.mean()
    print("\n  差值 (a)-(b) = %+.4f" % delta)
    if delta > 0.03:
        print("  ⚠⚠各类的『小坨』确实彼此更像 → 支持【存在跨类共享的弱相关污染】")
    elif delta > 0.01:
        print("  ⚠ 有弱信号，但不强，不足以据此大规模剔除")
    else:
        print("  ✅ 未发现跨类共享的模式 → 类内分裂更可能是合法的类内差异")
        print("     (雄/雌、幼/成、花/整株等)，【不应】按这个信号剔除样本")

    # 找出"小坨与其他类小坨特别像"的类，这些最可疑
    S = mc @ mc.T
    np.fill_diagonal(S, -1)
    peer = S.max(1)                      # 每个类的小坨，与别类小坨的最高相似度
    cls_ids = np.where(valid)[0]
    order = np.argsort(-peer)[:args.topk]
    print("\n  最可疑的 %d 个类（其『小坨』与别类『小坨』高度雷同）:" % args.topk)
    print("  %-8s %10s %10s %14s %14s" %
          ("类别", "小坨占比", "分裂程度", "与别类小坨最像", "对应类别"))
    for o in order:
        c = cls_ids[o]
        partner = cls_ids[int(S[o].argmax())]
        print("  %-8s %10.3f %10.4f %14.4f %14s"
              % ("%04d" % c, minor_frac[c], split_deg[c], peer[o], "%04d" % partner))

    # ---------------------------------------------------------------- 与已有信号对比
    banner("与现有 kNN 可疑度的关系（验证『结构性盲区』这个说法）")
    nz = np.load(os.path.join(P.CACHE_DIR, "noise_scores.npz"), allow_pickle=True)
    susp, ood = nz["suspicion"], nz["ood_score"]
    print("  小坨样本 %d 个 (%.1f%%)" % (minor_mask.sum(), minor_mask.mean() * 100))
    print("  %-22s %12s %12s" % ("", "小坨样本", "其余样本"))
    print("  %-22s %12.4f %12.4f" % ("平均 kNN 可疑度", susp[minor_mask].mean(),
                                     susp[~minor_mask].mean()))
    print("  %-22s %12.4f %12.4f" % ("平均 OOD 分数", ood[minor_mask].mean(),
                                     ood[~minor_mask].mean()))
    hi = susp >= np.quantile(susp, 0.8)
    print("  %-22s %11.1f%% %11.1f%%" % ("被判为最可疑20%的比例",
                                         hi[minor_mask].mean() * 100,
                                         hi[~minor_mask].mean() * 100))
    if abs(susp[minor_mask].mean() - susp[~minor_mask].mean()) < 0.05:
        print("\n  ⭐现有 kNN 可疑度对小坨样本【几乎无感】——证实了结构性盲区：")
        print("     成建制的噪声互相当邻居、互相作证，一致率反而很高，逃过了检测。")
    else:
        print("\n  现有可疑度对小坨样本有一定区分力，盲区没有想象中严重。")

    out = os.path.join(P.CACHE_DIR, "weakrelated_scores.npz")
    np.savez_compressed(out, minor_mask=minor_mask, minor_frac=minor_frac,
                        split_deg=split_deg, peer_sim=peer, cls_ids=cls_ids)
    with open(os.path.join(P.CACHE_DIR, "weakrelated_meta.json"), "w",
              encoding="utf-8") as f:
        json.dump({"min_size": args.min_size, "seed": args.seed,
                   "minor_sim_mean": float(s_minor.mean()),
                   "major_sim_mean": float(s_major.mean()),
                   "delta": float(delta),
                   "minor_sample_frac": float(minor_mask.mean()),
                   "verdict": ("存在跨类共享的弱相关污染" if delta > 0.03 else
                               "弱信号" if delta > 0.01 else "未发现跨类共享模式"),
                   "created": time.strftime("%Y-%m-%d %H:%M:%S")}, f,
                  ensure_ascii=False, indent=2)
    print("\n  结果已存 → %s" % out)
    print("  ⚠这是诊断，不是定论。任何剔除决定都必须交排行榜验证——")
    print("     本地评估对这类基于自身特征的清洗存在循环优化，测不准。")


if __name__ == "__main__":
    main()
