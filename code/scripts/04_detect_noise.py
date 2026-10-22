# -*- coding: utf-8 -*-
"""
在已缓存的 512 维特征上做【免训练】的噪声标签检测。不训练任何模型、不碰 GPU 之外的东西，
纯粹是几何计算：看每张图周围最像它的邻居，标签跟它一致吗？

═══ 原理：两根正交的轴 ═══

轴1「邻居一致率」——CLIP 已经把长得像的照片摆在 512 维空间里相邻的位置。
  若一张图标"麻雀"，而离它最近的 20 张全标"燕子"，大概率是它自己标错了。
  （类比：核对住户信息，某人填"7号楼"，周围 20 户都填"3号楼"，多半是他填错了。）
  白送的好处：邻居的多数票，就是它该有的标签。

轴2「最大原型相似度」——上面那招有个漏洞：万一样本"住在荒郊野外"、跟 500 类里
  任何一类都不像呢？（风景照混入、图片损坏、拍的是不在这 500 类里的物种。）
  官方原文管这叫"弱相关标注"。这类改标签也救不了（正确答案压根不在选项里），只能扔。

═══ 四分解与对应动作 ═══
                   邻居一致(合群)          邻居不一致(不合群)
  像某一类      │  clean 干净→留着      │  mislabeled 错标→改多数票 / hard 难样本→降权
  跟谁都不像    │  rare  罕见视角→别误伤 │  ood 开放集噪声→扔掉

⚠为什么必须分这四类：处理动作完全不同。一刀切全扔会连难样本一起扔掉，
  而难样本卡在类别边界上、信息量最大，扔了反而伤害模型判断边界的能力。

═══ 方法归属（都已双源核实存在，2026-09-04）═══
  本脚本轴1 的做法是 SimiFeat 的简化版：
    Zhu et al. "Detecting Corrupted Labels Without Training a Model to Predict"
    arXiv 2110.06283（免训练、纯特征空间）
  同族：FINE (2102.11628)、Confident Learning/cleanlab (1911.00068)
  写论文/技术文档时必须引 SimiFeat，否则会被质疑未引最相关工作。

用法:
  python scripts/04_detect_noise.py                  # 默认 k=20
  python scripts/04_detect_noise.py --k 10,20,50     # 敏感性对比
  python scripts/04_detect_noise.py --limit 20000    # 小规模先验证流程
"""
import _bootstrap  # noqa: F401
from _bootstrap import banner

import argparse
import json
import os
import time

import numpy as np
import torch
from tqdm import tqdm

from configs import paths as P


# ====================================================================== kNN
def knn_search(F, k, block=4096, device="cuda"):
    """
    分块求每个样本的 k 近邻（余弦相似度，F 必须已 L2 归一化）。

    为什么分块：完整的 103218×103218 相似度矩阵是 42GB，显存和内存都放不下。
    每次只算 block 行与全体的相似度（block×103218），算完立刻取 topk 丢掉，
    显存占用降到 GB 以内。总算力约 2*N^2*512 ≈ 10.9 TFLOP，4060 上分钟级。
    """
    N = F.shape[0]
    Ft = torch.from_numpy(F).to(device)
    idx_out = np.empty((N, k), dtype=np.int32)
    sim_out = np.empty((N, k), dtype=np.float32)

    t0 = time.time()
    bar = tqdm(range(0, N, block), desc="kNN k=%d" % k, unit="blk", dynamic_ncols=True)
    for s in bar:
        e = min(s + block, N)
        sims = Ft[s:e] @ Ft.T                       # (b, N) 余弦相似度
        # 排除自己：自身相似度恒为 1，必然排第一，取 k+1 个再砍掉第一列
        v, i = torch.topk(sims, k + 1, dim=1)
        idx_out[s:e] = i[:, 1:].cpu().numpy()
        sim_out[s:e] = v[:, 1:].cpu().numpy()
        del sims, v, i
        bar.set_postfix_str("%.0f 样本/s" % (e / max(time.time() - t0, 1e-6)))
    bar.close()
    del Ft
    if device == "cuda":
        torch.cuda.empty_cache()
    print("  kNN 完成，耗时 %.1fs" % (time.time() - t0))
    return idx_out, sim_out


def neighbor_stats(nbr_idx, nbr_sim, y, n_classes):
    """
    轴1：邻居投票。

    ⚠这里做过一次重要修正。第一版用「邻居一致率 < 0.5」当判据，是错的：
      · 一致率分布实测是单调下降的，**没有双峰**，不存在天然分界点，0.5 纯属拍脑袋
      · 500 类下随机基线仅 1/500=0.002，实测均值 0.40 已是随机的 201 倍，信号本就很强
      · 按 0.5 切会把 52.6% 的样本判成"难样本"，显然荒谬
    改用**得票差** vote_gap = 最高票类得票 - 自己标签得票（都按相似度加权）：
      gap ≈ 0  → 自己的标签得到邻居支持（哪怕不是最高票，也只是平手）→ 干净
      gap 大   → 邻居明确指向另一个类 → 真正可疑
    实测支撑：「最像的类≠标注类」占 45.2%，但差值 >0.05 的仅 9.5%，
    说明绝大多数"不一致"只是几乎平手的难分样本，不是错标。

    返回 (agree 一致率, vote 多数票标签, vote_gap 得票差, own_vote 自己标签的得票)
    """
    N, k = nbr_idx.shape
    nbr_lab = y[nbr_idx]                                     # (N,k) 每个邻居的标签
    agree = (nbr_lab == y[:, None]).mean(1).astype(np.float32)

    # 相似度加权：越像的邻居话语权越大，比单纯计数更稳
    w = np.clip(nbr_sim, 0, None).astype(np.float32)
    w = w / np.maximum(w.sum(1, keepdims=True), 1e-12)

    vote = np.empty(N, dtype=np.int64)
    vote_gap = np.empty(N, dtype=np.float32)
    own_vote = np.empty(N, dtype=np.float32)
    for s in tqdm(range(0, N, 20000), desc="加权投票", unit="blk", dynamic_ncols=True):
        e = min(s + 20000, N)
        acc = np.zeros((e - s, n_classes), dtype=np.float32)
        rows = np.repeat(np.arange(e - s), k)
        np.add.at(acc, (rows, nbr_lab[s:e].ravel()), w[s:e].ravel())
        vote[s:e] = acc.argmax(1)
        top = acc.max(1)
        mine = acc[np.arange(e - s), y[s:e]]
        own_vote[s:e] = mine
        vote_gap[s:e] = top - mine                            # ≥0，越大越可疑
    return agree, vote, vote_gap, own_vote


def prototype_sim(F, y, n_classes):
    """
    轴2：到最近类原型的相似度。
    原型 = 该类所有样本特征的平均方向（会被噪声轻微污染，但平均对少数离群点不敏感，够用）。
    ⚠这里用【全部】样本算原型，包含噪声；更严谨可迭代精炼，但第一版保持简单。
    """
    proto = np.zeros((n_classes, F.shape[1]), dtype=np.float32)
    for c in range(n_classes):
        m = (y == c)
        if m.any():
            proto[c] = F[m].mean(0)
    proto /= np.maximum(np.linalg.norm(proto, axis=1, keepdims=True), 1e-12)
    sims = F @ proto.T                                        # (N, C)
    return sims.max(1), sims.argmax(1), sims[np.arange(len(y)), y], proto


# ====================================================================== 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=str, default="20", help="逗号分隔，如 10,20,50 做敏感性对比")
    ap.add_argument("--agree-thr", type=float, default=0.5,
                    help="邻居一致率低于此值判为『不合群』")
    ap.add_argument("--ood-quantile", type=float, default=0.05,
                    help="最大原型相似度低于全体这个分位数的判为『跟谁都不像』")
    ap.add_argument("--block", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=0, help=">0 时只取前 N 个样本，验证流程用")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ks = [int(x) for x in args.k.split(",")]

    banner("04 免训练噪声检测（kNN + 原型，不训练任何模型）")
    X = np.load(P.F_TRAIN_FEAT).astype(np.float32)
    y = np.load(P.F_TRAIN_LABEL).astype(np.int64)
    if args.limit:
        X, y = X[:args.limit], y[:args.limit]
    n_classes = int(y.max()) + 1
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    print("  样本 %d，类别 %d，设备 %s" % (len(y), n_classes, device))
    print("  显存预算: 每块 %d 行 × %d 列 × 4B ≈ %.2f GB"
          % (args.block, len(y), args.block * len(y) * 4 / 1024 ** 3))

    # ---------------------------------------------------------------- 轴2
    banner("轴2：到最近类原型的相似度（『是不是跟谁都不像』）")
    max_sim, argmax_cls, own_sim, _ = prototype_sim(X, y, n_classes)
    ood_thr = np.quantile(max_sim, args.ood_quantile)
    print("  最大原型相似度: 均值 %.4f  中位 %.4f  %.0f%%分位 %.4f(=OOD阈值)"
          % (max_sim.mean(), np.median(max_sim), args.ood_quantile * 100, ood_thr))
    print("  与【自己标注类】的相似度: 均值 %.4f" % own_sim.mean())
    print("  最像的类 ≠ 标注类 的样本占比: %.1f%%" % ((argmax_cls != y).mean() * 100))

    results = {}
    for k in ks:
        # ------------------------------------------------------------ 轴1
        banner("轴1：k=%d 的加权邻居投票" % k)
        nbr_idx, nbr_sim = knn_search(X, k, args.block, device)
        agree, vote, vote_gap, own_vote = neighbor_stats(nbr_idx, nbr_sim, y, n_classes)
        print("  邻居一致率(仅供参考): 均值 %.4f  中位 %.4f  随机基线 %.4f(=1/%d)"
              % (agree.mean(), np.median(agree), 1.0 / n_classes, n_classes))
        print("  得票差 vote_gap    : 均值 %.4f  中位 %.4f" % (vote_gap.mean(), np.median(vote_gap)))
        for t in (0.0, 0.1, 0.3, 0.5):
            print("    gap > %.1f 的样本: %6d (%.1f%%)"
                  % (t, (vote_gap > t).sum(), (vote_gap > t).mean() * 100))

        # ------------------------------------------------------------ 可疑度排序
        # ⚠不做硬分类。分布无双峰=没有天然分界点，任何绝对阈值都是武断的。
        # 改为输出连续可疑度，下游用不同截断比例去试，让最终分数来选阈值。
        banner("可疑度排序 (k=%d)" % k)
        # 两个信号各自转成 [0,1] 的分位排名，避免量纲不同没法比
        rank_gap = np.argsort(np.argsort(vote_gap)) / (len(y) - 1)      # 越大越像错标
        proto_gap = max_sim - own_sim                                    # 原型层面的同类信号
        rank_proto = np.argsort(np.argsort(proto_gap)) / (len(y) - 1)
        suspicion = (rank_gap + rank_proto) / 2                          # 错标可疑度
        ood_score = 1.0 - np.argsort(np.argsort(max_sim)) / (len(y) - 1) # 越大越像OOD

        print("  错标可疑度 suspicion: 两个独立信号(邻居得票差 + 原型相似度差)的分位排名均值")
        print("  两信号相关性 corr = %+.3f  (越低说明越互补，都高才更可信)"
              % np.corrcoef(rank_gap, rank_proto)[0, 1])
        print("\n  按不同截断比例，会剔除/改标多少样本：")
        print("  %-10s %10s %12s" % ("截断比例", "样本数", "suspicion阈值"))
        for q in (0.05, 0.10, 0.20, 0.30):
            thr = np.quantile(suspicion, 1 - q)
            print("  %-10s %10d %12.4f" % ("top %.0f%%" % (q * 100), int(len(y) * q), thr))
        print("\n  参考: 由基线准确率反推的噪声率上界约 30.7%%（真实值更低）")
        print("        『最像的类≠标注类』占 %.1f%%，但原型差>0.05 的仅 %.1f%%"
              % ((argmax_cls != y).mean() * 100, (proto_gap > 0.05).mean() * 100))
        print("        → 多数『不一致』只是几乎平手的难分样本，不是错标")

        # 高可疑样本里，邻居多数票给出的候选新标签
        top20 = suspicion >= np.quantile(suspicion, 0.8)
        print("\n  最可疑的 20%% 样本中，邻居多数票 ≠ 原标签 的占比: %.1f%%"
              % ((vote[top20] != y[top20]).mean() * 100))
        print("  (这批的多数票标签可作为『改标』方案的候选)")

        results[k] = dict(agree=agree, vote=vote, vote_gap=vote_gap, own_vote=own_vote,
                          suspicion=suspicion.astype(np.float32),
                          ood_score=ood_score.astype(np.float32),
                          nbr_idx=nbr_idx, nbr_sim=nbr_sim)

    # ---------------------------------------------------------------- 落盘
    k0 = ks[0]
    r = results[k0]
    out = os.path.join(P.CACHE_DIR, "noise_scores.npz")
    np.savez_compressed(
        out, k=k0, agree=r["agree"], vote=r["vote"], vote_gap=r["vote_gap"],
        own_vote=r["own_vote"], suspicion=r["suspicion"], ood_score=r["ood_score"],
        max_sim=max_sim, argmax_cls=argmax_cls, own_sim=own_sim, labels=y,
        nbr_idx=r["nbr_idx"][:, :min(k0, 20)])
    meta = {"k_list": ks, "k_saved": k0,
            "n_samples": int(len(y)), "n_classes": n_classes,
            "suspicion_quantiles": {("top%d%%" % int(q * 100)):
                                    float(np.quantile(r["suspicion"], 1 - q))
                                    for q in (0.05, 0.10, 0.20, 0.30)},
            "argmax_ne_label_frac": float((argmax_cls != y).mean()),
            "proto_gap_gt005_frac": float(((max_sim - own_sim) > 0.05).mean()),
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "design_note": "不做硬分类：一致率分布单调下降无双峰，绝对阈值无依据。"
                           "输出连续可疑度，由下游截断比例+最终分数选阈值。",
            "method_ref": "SimiFeat arXiv 2110.06283 的简化版；同族 FINE 2102.11628 / cleanlab 1911.00068"}
    with open(os.path.join(P.CACHE_DIR, "noise_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    banner("完成")
    print("  逐样本分数 → %s  (%.1f MB)" % (out, os.path.getsize(out) / 1024 ** 2))
    print("    suspicion  错标可疑度[0,1]，越大越可能标错")
    print("    ood_score  开放集可疑度[0,1]，越大越『跟谁都不像』")
    print("    vote       邻居多数票标签，可作为『改标』的候选")
    print("  汇总信息   → %s" % os.path.join(P.CACHE_DIR, "noise_meta.json"))
    print("\n  ⚠下一步必须先校准，不要直接拿去清洗：")
    print("     用已知真实分数(C=30→56.0, C=10→57.1)反过来考这把新尺子，")
    print("     若它也判 C=10 优于 C=30 才算可信；判反了说明还没调好。")


if __name__ == "__main__":
    main()
