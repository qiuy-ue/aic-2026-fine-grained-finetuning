# -*- coding: utf-8 -*-
"""
校准本地验证集：用【已知的真实分数】反过来考验"剔除可疑样本"这个做法是否有效。

═══ 为什么必须做这一步 ═══

我们已经知道两件事实：
  · 真实分数：C=30 → 56.0，C=10 → 57.1          （C=10 更好）
  · 带噪验证集：C=30 → 0.6926，C=10 → 0.6841    （说反了！）

带噪验证集把顺序判反了，因为它和训练集共享同一套错标签，会奖励"死记硬背"。
现在的问题是：04 检测出的可疑样本，剔掉之后能不能把这个顺序**纠正**过来？

  能纠正 → 说明检测确实抓到了噪声，这把新尺子可以用来做快速迭代
  纠不正 → 说明检测没抓准，别拿它去清洗训练集，先回去调检测

⚠这是本工程唯一一处能用"外部真值"验证内部方法的地方，做完才谈得上后面的实验。

⚠局限（必须写进文档）：只有两个真实分数点，样本量极小，只能验证"排序方向"，
  不能验证"数值准确度"。而且这仍是用模型自己的特征判断谁干净、再用它评估模型，
  有循环论证成分。结论只能是"比带噪验证集更好的相对指南针"，不是无偏真值。

用法:
  python scripts/05_calibrate_valset.py
"""
import _bootstrap  # noqa: F401
from _bootstrap import banner

import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from configs import paths as P

# 已知的外部真值（官方排行榜），本脚本的判据全靠它
TRUTH = {30.0: 56.0, 10.0: 57.1}
SEED, VAL_FRAC = 42, 0.2          # 必须与 02_train_head.py 完全一致，否则 split 对不上


def train_head(Xtr, ytr, C, n_iter=500, device="cuda"):
    """与 02_train_head.py 的 torch 后端同一套目标函数，保证可比。"""
    n, d = Xtr.shape
    K = int(ytr.max()) + 1
    Xt = torch.from_numpy(Xtr).to(device)
    yt = torch.from_numpy(ytr).to(device)
    W = torch.zeros(K, d, device=device, requires_grad=True)
    b = torch.zeros(K, device=device, requires_grad=True)
    reg = 1.0 / (2.0 * C * n)
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=1, history_size=10,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(Xt @ W.T + b, yt) + reg * (W * W).sum()
        loss.backward()
        return loss

    prev, stall = None, 0
    for _ in range(n_iter):
        lv = float(opt.step(closure).detach())
        if prev is not None and abs(prev - lv) < 1e-7 * max(abs(prev), 1.0):
            stall += 1
            if stall >= 5:
                break
        else:
            stall = 0
        prev = lv
    return W.detach().cpu().numpy(), b.detach().cpu().numpy()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    banner("05 校准本地验证集（用已知真实分数当裁判）")

    X = np.load(P.F_TRAIN_FEAT).astype(np.float32)
    y = np.load(P.F_TRAIN_LABEL).astype(np.int64)
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)

    nz = np.load(os.path.join(P.CACHE_DIR, "noise_scores.npz"), allow_pickle=True)
    susp, ood = nz["suspicion"], nz["ood_score"]
    print("  样本 %d，可疑度已加载（suspicion / ood_score）" % len(y))
    print("  已知真实分数: " + "  ".join("C=%g→%.1f" % (c, s) for c, s in TRUTH.items()))

    idx = np.arange(len(y))
    itr, iva = train_test_split(idx, test_size=VAL_FRAC, random_state=SEED, stratify=y)
    print("  切分: 训练 %d / 验证 %d (seed=%d，与 02 脚本一致)" % (len(itr), len(iva), SEED))

    # ---------------------------------------------------------------- 训练各 C
    banner("训练各 C 的模型（训练集部分保持原样，只动验证集）")
    Cs = sorted(TRUTH.keys(), reverse=True)
    models = {}
    for C in Cs:
        t0 = time.time()
        models[C] = train_head(X[itr], y[itr], C, device=device)
        print("  C=%-6g 训练完成 (%.1fs)" % (C, time.time() - t0))

    # ---------------------------------------------------------------- 逐截断评估
    banner("在不同『干净度截断』的验证集上评估")
    print("  做法: 按 suspicion 从低到高保留前 X%% 的验证样本，越靠前标签越可信\n")

    sv, ov = susp[iva], ood[iva]
    Xv, yv = X[iva], y[iva]
    keeps = [1.00, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30]

    hdr = "  %-9s %8s " % ("保留比例", "样本数") + " ".join("%9s" % ("C=%g" % c) for c in Cs)
    print(hdr + "   %9s  %s" % ("差值", "与真值一致?"))
    print("  " + "-" * (len(hdr) + 22))

    rows = []
    truth_better = max(TRUTH, key=TRUTH.get)          # 真实更好的那个 C（=10）
    truth_worse = min(TRUTH, key=TRUTH.get)
    truth_delta = TRUTH[truth_better] - TRUTH[truth_worse]

    for kp in keeps:
        m = sv <= np.quantile(sv, kp) if kp < 1.0 else np.ones(len(sv), bool)
        accs = {}
        for C in Cs:
            W, b = models[C]
            accs[C] = float((np.argmax(Xv[m] @ W.T + b, 1) == yv[m]).mean())
        delta = accs[truth_better] - accs[truth_worse]
        ok = "✅ 一致" if delta > 0 else "❌ 判反"
        print("  %-9s %8d " % ("%.0f%%" % (kp * 100), m.sum())
              + " ".join("%9.4f" % accs[c] for c in Cs)
              + "   %+9.4f  %s" % (delta, ok))
        rows.append({"keep": kp, "n": int(m.sum()),
                     "acc": {str(c): accs[c] for c in Cs}, "delta": delta})

    print("\n  真值方向: C=%g 比 C=%g 高 %.1f 分 → delta 应为【正】"
          % (truth_better, truth_worse, truth_delta))

    # ---------------------------------------------------------------- 同时试 OOD 截断
    banner("对照：改用 ood_score 截断（剔除『跟谁都不像』的样本）")
    print("  %-9s %8s " % ("保留比例", "样本数")
          + " ".join("%9s" % ("C=%g" % c) for c in Cs) + "   %9s  %s" % ("差值", "与真值一致?"))
    for kp in (1.00, 0.90, 0.80, 0.70):
        m = ov <= np.quantile(ov, kp) if kp < 1.0 else np.ones(len(ov), bool)
        accs = {}
        for C in Cs:
            W, b = models[C]
            accs[C] = float((np.argmax(Xv[m] @ W.T + b, 1) == yv[m]).mean())
        delta = accs[truth_better] - accs[truth_worse]
        print("  %-9s %8d " % ("%.0f%%" % (kp * 100), m.sum())
              + " ".join("%9.4f" % accs[c] for c in Cs)
              + "   %+9.4f  %s" % (delta, "✅ 一致" if delta > 0 else "❌ 判反"))

    # ---------------------------------------------------------------- 结论
    banner("结论")
    good = [r for r in rows if r["delta"] > 0]
    if good:
        best = max(good, key=lambda r: r["delta"])
        print("  ✅ 存在能纠正排序的截断比例，最强的是 保留 %.0f%%（delta=%+.4f，%d 个样本）"
              % (best["keep"] * 100, best["delta"], best["n"]))
        print("  → 说明 04 的可疑度检测确实抓到了噪声，可以用它做本地快速迭代。")
        print("  → 建议后续实验一律用『保留 %.0f%%』这把尺子，并在文档里注明它的循环论证局限。"
              % (best["keep"] * 100))
    else:
        print("  ❌ 没有任何截断比例能把排序纠正过来。")
        print("  → 说明可疑度检测没抓准噪声，【不要】拿它去清洗训练集。")
        print("  → 下一步应回去调 04：换 k、换信号组合，或改用别的判据。")
    print("\n  ⚠局限: 仅两个真值点，只能验证排序方向、不能验证数值；")
    print("     且用模型自身特征判断谁干净再评估模型，有循环论证成分。")

    with open(os.path.join(P.CACHE_DIR, "valset_calibration.json"), "w", encoding="utf-8") as f:
        json.dump({"truth": {str(k): v for k, v in TRUTH.items()}, "rows": rows,
                   "seed": SEED, "val_frac": VAL_FRAC,
                   "created": time.strftime("%Y-%m-%d %H:%M:%S")}, f,
                  ensure_ascii=False, indent=2)
    print("\n  结果已存 → %s" % os.path.join(P.CACHE_DIR, "valset_calibration.json"))


if __name__ == "__main__":
    main()
