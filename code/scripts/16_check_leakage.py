# -*- coding: utf-8 -*-
"""
诊断：训练集与验证集之间是否存在【近重复图片】，导致本地评估系统性虚高。

═══════════════════════════════════════════════════════════════════════
一、为什么怀疑它
═══════════════════════════════════════════════════════════════════════

两条对不上的数字：

  ① 由 0.693 = (1-e)·m 反推（e≈2%，HX 人工核实的噪声率）
     → 模型在【验证集】上的真实准确率 m ≈ 0.70
     而【测试集】真实只有 0.61 ~ 0.615。**差 9 分**。
     两个集合都来自同一批数据，凭什么差这么多？

  ② 本地→真实的换算倍率，冻结特征类方法稳定在 135~170x，
     但 LoRA（唯一一个真去改骨干的）**塌到 25x**。

一个假设能同时解释这两件事：**训练集和验证集里有近重复的图片**
（同一张图的不同尺寸/水印版本，或同一次拍摄的连拍）。网爬细粒度数据里极常见。

  · SCE / TTA **不改特征提取器** → 占不到这个便宜 → 倍率正常
  · LoRA **真的在按训练图片调骨干** → 近重复的验证图跟着受益 → 本地虚高
  · 而**测试集里没有一张是模型训练时见过的** → 榜上归零

═══════════════════════════════════════════════════════════════════════
二、⭐光证明"存在相似图片"是不够的
═══════════════════════════════════════════════════════════════════════

"验证集里有 5% 的图和训练集很像" —— 这句话本身**证明不了它抬高了分数**。
细粒度数据集里同类图片本来就长得像，相似度高是正常的。

所以本脚本做两个**对照**，缺一不可：

  ⭐对照1（有没有偏袒）：val 和 test **都**拿去和训练集比最大相似度。
    如果 val 的分布明显更靠右、test 更靠左 → 那 9 分缺口就有了直接解释。
    如果两边分布一样 → 假设不成立，得另找原因。

  ⭐对照2（因果检验，最关键）：把 val 按"最大相似度"切成两半，**分别算准确率**。
    如果泄漏真在虚高，**高相似组的准确率应该显著高于低相似组**。
    如果两组准确率差不多 → 说明相似归相似，并没有转化成分数优势。

**只有对照2成立，才算坐实。** 否则只是"发现了一个相关，没证明因果"。

═══════════════════════════════════════════════════════════════════════
三、用哪份特征
═══════════════════════════════════════════════════════════════════════

用【原版冻结 CLIP】抽的 cache/train_features.npy + test_features.npy。
⚠这里要检测的是"两张图是不是同一张图"——这是**数据本身的性质**，
  跟用哪个模型无关，用原版特征最干净（也不用重抽）。
⚠余弦相似度只是近重复的**代理指标**，不是像素级判定：
  同类不同个体也可能到 0.95。所以阈值扫一排，并且必须配合对照2看。

用法:
  python scripts/16_check_leakage.py
  python scripts/16_check_leakage.py --topk 3     # 顺便打印几对最像的，肉眼验一验
"""
import _bootstrap  # noqa: F401
from _bootstrap import banner

import argparse
import json
import os

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from configs import paths as P

SEED, VAL_FRAC = 42, 0.2        # ⚠必须与 02/06/13 一致，否则查的不是同一个划分
THRESHOLDS = (0.90, 0.95, 0.98, 0.99)


def max_sim_to_train(Q, T, device, chunk=2048, want_idx=False):
    """
    对 Q 的每一行，求它与 T 中所有行的最大余弦相似度（及对应行号）。
    Q、T 必须已 L2 归一化，这样点积就是余弦相似度。
    ⚠分块是为了显存：chunk 行 × len(T) 列 × 4B。
      len(T)≈8.3万时，2048 行 ≈ 0.68GB，8GB 显存安全。
    """
    Tt = torch.from_numpy(T).to(device)
    best = np.zeros(len(Q), dtype=np.float32)
    bidx = np.zeros(len(Q), dtype=np.int64)
    for s in range(0, len(Q), chunk):
        e = min(s + chunk, len(Q))
        q = torch.from_numpy(Q[s:e]).to(device)
        sim = q @ Tt.T                        # (chunk, N_train)
        v, i = sim.max(dim=1)
        best[s:e] = v.cpu().numpy()
        bidx[s:e] = i.cpu().numpy()
        del sim
    del Tt
    torch.cuda.empty_cache()
    return (best, bidx) if want_idx else best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--topk", type=int, default=0, help=">0 时打印最像的几对文件路径，肉眼复核")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    banner("16 近重复图诊断：本地评估是不是被泄漏抬高了")

    # ---------------------------------------------------------------- 数据
    X = np.load(P.F_TRAIN_FEAT).astype(np.float32)
    y = np.load(P.F_TRAIN_LABEL).astype(np.int64)
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    tr_paths = open(P.F_TRAIN_PATH, encoding="utf-8").read().splitlines()

    idx = np.arange(len(y))
    itr, iva = train_test_split(idx, test_size=VAL_FRAC, random_state=SEED, stratify=y)
    Xtr, Xva, yva = X[itr], X[iva], y[iva]

    Xte = np.load(P.F_TEST_FEAT).astype(np.float32)
    Xte /= np.maximum(np.linalg.norm(Xte, axis=1, keepdims=True), 1e-12)

    print("  训练 %d / 验证 %d / 测试 %d   (划分 seed=%d，与所有实验一致)"
          % (len(itr), len(iva), len(Xte), SEED))
    print("  特征来源: 原版冻结 CLIP（检测'是不是同一张图'是数据本身的性质，与模型无关）")

    # ---------------------------------------------------------------- 对照1
    banner("对照1：val 和 test 分别与【同一个训练集】比最大相似度")
    sv, svi = max_sim_to_train(Xva, Xtr, device, args.chunk, want_idx=True)
    st = max_sim_to_train(Xte, Xtr, device, args.chunk)

    print("  %-8s %8s %8s %8s %8s %8s" % ("集合", "中位", "均值", "p90", "p99", "最大"))
    for nm, s in (("验证集", sv), ("测试集", st)):
        print("  %-8s %8.4f %8.4f %8.4f %8.4f %8.4f"
              % (nm, np.median(s), s.mean(), np.percentile(s, 90),
                 np.percentile(s, 99), s.max()))
    print()
    print("  %-10s %12s %12s %10s" % ("相似度阈值", "验证集占比", "测试集占比", "倍数"))
    rows = []
    for t in THRESHOLDS:
        pv, pt = float((sv >= t).mean()), float((st >= t).mean())
        ratio = pv / pt if pt > 1e-9 else float("inf")
        print("  >= %-7.2f %11.2f%% %11.2f%% %9.1fx"
              % (t, 100 * pv, 100 * pt, ratio))
        rows.append(dict(thr=t, val_frac=pv, test_frac=pt, ratio=ratio))
    print()
    print("  ⭐怎么读：验证集占比【明显高于】测试集 → 支持'泄漏'假设；")
    print("            两边差不多 → 假设不成立，那 9 分缺口得另找原因（比如测试集本身更难）。")

    # ---------------------------------------------------------------- 对照2
    banner("对照2（因果检验，最关键）：高相似组 vs 低相似组，准确率差多少")
    hp = os.path.join(P.CACHE_DIR, "head_loss_sce.npz")
    if not os.path.exists(hp):
        print("  ⚠找不到 %s，跳过因果检验" % hp)
        rows2 = None
    else:
        hd = np.load(hp)
        W, b = hd["W"].astype(np.float32), hd["b"].astype(np.float32)
        pred = np.argmax(Xva @ W.T + b, 1)
        correct = (pred == yva)
        print("  用 SCE 头（榜上 59.0 那版）在验证集上逐样本判对错，再按相似度分组\n")
        print("  %-22s %8s %10s" % ("分组", "样本数", "准确率"))
        rows2 = []
        for t in THRESHOLDS:
            m = sv >= t
            if m.sum() < 30 or (~m).sum() < 30:
                continue
            ah, al = float(correct[m].mean()), float(correct[~m].mean())
            print("  相似度 >= %-11.2f %8d %9.4f" % (t, m.sum(), ah))
            print("  相似度 <  %-11.2f %8d %9.4f      差值 %+.4f"
                  % (t, (~m).sum(), al, ah - al))
            rows2.append(dict(thr=t, n_high=int(m.sum()), acc_high=ah,
                              n_low=int((~m).sum()), acc_low=al, gap=ah - al))
        print()
        print("  ⭐怎么读：高相似组准确率【明显更高】→ 泄漏确实在抬分，坐实；")
        print("            两组差不多 → 相似归相似，没转化成分数优势，尺子没坏在这。")
        print("  ⚠混淆因素：'与训练集很像'的图可能本来就是**典型样本**（更好认），")
        print("    这本身就会让准确率更高，与泄漏无关。所以差值大只是**支持**，不是铁证。")

    # ---------------------------------------------------------------- 去重验证集
    banner("产物：去重验证集掩码")
    out = os.path.join(P.CACHE_DIR, "leakage_scores.npz")
    np.savez(out, val_idx=iva, val_maxsim=sv, val_nn_train_row=itr[svi], test_maxsim=st)
    print("  已存 → %s" % out)
    print("  里面有：验证集每个样本与训练集的最大相似度 val_maxsim，")
    print("          以及最像的那张训练图在原始清单里的行号 val_nn_train_row。")
    print("  ⭐用法：后续评估时用 val_maxsim < 阈值 做掩码，就得到一个'去重验证集'。")

    if args.topk:
        banner("最像的 %d 对（肉眼复核用）" % args.topk)
        order = np.argsort(-sv)[:args.topk]
        for r in order:
            print("  相似度 %.4f" % sv[r])
            print("    验证: %s" % tr_paths[iva[r]])
            print("    训练: %s" % tr_paths[itr[svi[r]]])

    with open(os.path.join(P.CACHE_DIR, "leakage_report.json"), "w", encoding="utf-8") as f:
        json.dump({"thresholds": list(THRESHOLDS), "contrast1": rows,
                   "contrast2": rows2, "seed": SEED}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
