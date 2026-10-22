# -*- coding: utf-8 -*-
"""
在缓存好的 512 维特征上训练【线性头】(512 → 500)，产出基线分数。

模型结构就一层：  logits = W @ x + b     W:(500,512)  b:(500,)  共 256,500 个参数
骨干那 87,849,216 个参数全程冻结、不参与训练，也不出现在这个脚本里——
它的全部贡献已经固化在 cache/*.npy 里了。

两个后端（数学上等价，C 的语义已对齐，可互相验证）：
  torch  (默认) GPU 全批量 L-BFGS。拟牛顿法 + strong_wolfe 线搜索，不需要调学习率；
                每次迭代都能刷新训练/验证准确率，有真正的实时进度条。快约两个数量级。
  sklearn       LogisticRegression(lbfgs)，CLIP 原论文做 linear probe 的协议。
                慢（实测约 2.4 s/迭代），留作正确性对照。

⚠关于准确率怎么读：训练集和验证集的标签【本身就带噪】(这正是赛题要解决的问题)。
  所以这里的 val acc 衡量的是"对带噪标签的拟合程度"，不等于官方榜上的真实分数。
  如果训练标签里有 e 的比例是错的，一个完美模型在带噪验证集上的准确率上限约为 1-e。

用法:
  python scripts/02_train_head.py                       # 默认：搜 C + 全量重训
  python scripts/02_train_head.py --c-grid 1.0          # 只跑一个 C，最快出基线
  python scripts/02_train_head.py --backend sklearn     # 换对照后端
  python scripts/02_train_head.py --dump-proba          # 额外导出软标签矩阵(自蒸馏用)
  python scripts/02_train_head.py --smoke               # 用 _smoke 小缓存冒烟
"""
import _bootstrap  # noqa: F401
from _bootstrap import banner

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from configs import paths as P


# ====================================================================== 数据
def load_features(smoke=False):
    fx, fy = P.F_TRAIN_FEAT, P.F_TRAIN_LABEL
    if smoke:
        fx = fx.replace(".npy", "_smoke.npy")
        fy = fy.replace(".npy", "_smoke.npy")
    if not os.path.exists(fx):
        raise SystemExit("找不到特征缓存 %s\n请先跑: python scripts/01_extract_features.py" % fx)
    X = np.load(fx).astype(np.float32)
    y = np.load(fy).astype(np.int64)
    return X, y


def l2_normalize(X):
    """
    把每个 512 维向量的长度归一化成 1，只保留方向。
    CLIP 的语义相似度本来就定义在方向上(余弦相似度)，归一化后 L-BFGS 收敛也明显更快。
    """
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-12)


# ====================================================================== torch 后端
def train_head_torch(Xtr, ytr, Xva, yva, C, n_iter, device, log_every=1, desc="训练"):
    """
    GPU 全批量 L-BFGS 训练线性头。

    目标函数与 sklearn 的 LogisticRegression 对齐：
      sklearn 最小化   0.5*||W||^2 + C * Σ_i loss_i
      整体除以 (C*n) 得等价形式：  mean_loss + 1/(2*C*n) * ||W||^2
    所以这里 weight_decay 系数取 1/(2*C*n)。截距 b 不参与正则化(与 sklearn 一致)。
    """
    n, d = Xtr.shape
    K = int(max(ytr.max(), yva.max())) + 1

    Xtr_t = torch.from_numpy(Xtr).to(device)
    ytr_t = torch.from_numpy(ytr).to(device)
    Xva_t = torch.from_numpy(Xva).to(device)
    yva_t = torch.from_numpy(yva).to(device)

    W = torch.zeros(K, d, device=device, requires_grad=True)   # 零初始化，凸问题不需要随机
    b = torch.zeros(K, device=device, requires_grad=True)
    reg = 1.0 / (2.0 * C * n)

    # max_iter=1：每次 step() 只走一次 L-BFGS 迭代，这样外层才能逐次刷新进度条。
    # L-BFGS 的历史(old_dirs/old_stps)保存在 optimizer.state 里，跨 step 调用不会丢。
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=1, history_size=10,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(Xtr_t @ W.T + b, ytr_t) + reg * (W * W).sum()
        loss.backward()
        return loss

    hist = []
    prev, stall = None, 0
    bar = tqdm(range(n_iter), desc=desc, unit="iter", dynamic_ncols=True)
    for it in bar:
        loss = opt.step(closure)
        # LBFGS 交还的 loss 还挂着计算图；先 detach 再取值，否则 PyTorch 会告警
        lv = float(loss.detach())

        if (it + 1) % log_every == 0 or it == n_iter - 1:
            with torch.no_grad():
                tr_acc = ((Xtr_t @ W.T + b).argmax(1) == ytr_t).float().mean().item()
                va_acc = ((Xva_t @ W.T + b).argmax(1) == yva_t).float().mean().item()
            # 实时把两个准确率显示在进度条上，训练过程可见
            bar.set_postfix_str("loss=%.4f  训练acc=%.4f  验证acc=%.4f" % (lv, tr_acc, va_acc))
            hist.append({"iter": it + 1, "loss": lv, "train_acc": tr_acc, "val_acc": va_acc})

        # 收敛判据：损失连续 5 次几乎不动就停，别空转
        if prev is not None and abs(prev - lv) < 1e-7 * max(abs(prev), 1.0):
            stall += 1
            if stall >= 5:
                bar.set_postfix_str((bar.postfix or "") + "  [已收敛,提前停止]")
                break
        else:
            stall = 0
        prev = lv
    bar.close()

    with torch.no_grad():
        tr_acc = ((Xtr_t @ W.T + b).argmax(1) == ytr_t).float().mean().item()
        va_pred = (Xva_t @ W.T + b).argmax(1)
        va_acc = (va_pred == yva_t).float().mean().item()

    return {"W": W.detach().cpu().numpy(), "b": b.detach().cpu().numpy(),
            "train_acc": tr_acc, "val_acc": va_acc,
            "val_pred": va_pred.cpu().numpy(), "history": hist, "n_iter_run": len(hist)}


# ====================================================================== sklearn 后端
def train_head_sklearn(Xtr, ytr, Xva, yva, C, n_iter, desc="训练"):
    """CLIP 原论文的 linear probe 协议。慢，但作为独立实现可以交叉验证 torch 后端。"""
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression

    print("  [sklearn] C=%g max_iter=%d  实测约 2.4s/迭代，请耐心等待…" % (C, n_iter))
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        clf = LogisticRegression(C=C, max_iter=n_iter, n_jobs=-1, verbose=1)
        clf.fit(Xtr, ytr)
    print("  [sklearn] 实际迭代 %s 次，耗时 %.1fs" % (clf.n_iter_, time.time() - t0))

    va_pred = clf.predict(Xva)
    # 统一成与 torch 后端相同的产物格式，下游脚本只认一种
    return {"W": clf.coef_.astype(np.float32), "b": clf.intercept_.astype(np.float32),
            "train_acc": float(clf.score(Xtr, ytr)),
            "val_acc": float((va_pred == yva).mean()),
            "val_pred": va_pred, "history": [], "n_iter_run": int(np.max(clf.n_iter_))}


# ====================================================================== 诊断
def per_class_report(y_true, y_pred, n_classes, class_names=None, topk=8):
    """每类准确率分布。当前初赛类别均衡，这份数据主要用于后续定位'难类/噪声重的类'。"""
    accs = np.full(n_classes, np.nan, dtype=np.float64)
    for c in range(n_classes):
        m = (y_true == c)
        if m.any():
            accs[c] = (y_pred[m] == c).mean()
    valid = ~np.isnan(accs)
    print("  每类准确率: 均值 %.4f  中位数 %.4f  最低 %.4f  最高 %.4f"
          % (np.nanmean(accs), np.nanmedian(accs), np.nanmin(accs), np.nanmax(accs)))
    order = np.argsort(np.where(valid, accs, np.inf))
    worst = [(int(c), float(accs[c])) for c in order[:topk]]
    name = (lambda c: class_names[c]) if class_names else (lambda c: "%04d" % c)
    print("  最差 %d 个类: %s" % (topk, ", ".join("%s(%.2f)" % (name(c), a) for c, a in worst)))
    return {"mean": float(np.nanmean(accs)), "median": float(np.nanmedian(accs)),
            "min": float(np.nanmin(accs)), "max": float(np.nanmax(accs)),
            "worst": worst, "per_class": accs.tolist()}


# ====================================================================== 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["torch", "sklearn"], default="torch")
    ap.add_argument("--c-grid", type=str, default="0.1,1.0,10.0",
                    help="逗号分隔的正则化强度候选；C 越大正则越弱")
    ap.add_argument("--n-iter", type=int, default=500, help="L-BFGS 最大迭代次数")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-normalize", action="store_true", help="不做 L2 归一化(默认做)")
    ap.add_argument("--no-refit-full", action="store_true",
                    help="不用最佳 C 在全量数据上重训(默认会重训，提交模型用它)")
    ap.add_argument("--dump-proba", action="store_true",
                    help="额外导出全训练集的 (N,500) 完整概率矩阵，供后续自蒸馏当软标签用")
    ap.add_argument("--smoke", action="store_true", help="用 _smoke 小缓存快速验证流程")
    args = ap.parse_args()

    P.ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    c_grid = [float(c) for c in args.c_grid.split(",")]

    banner("02 训练线性头：512 → 500")
    X, y = load_features(smoke=args.smoke)
    n_classes = int(y.max()) + 1
    print("  特征      : %s  %s" % (X.shape, X.dtype))
    print("  类别数    : %d" % n_classes)
    print("  头参数量  : {:,}  (骨干 {:,} 个参数全程冻结、不在此训练)"
          .format(n_classes * X.shape[1] + n_classes, P.CLIP_VISUAL_PARAMS))

    if not args.no_normalize:
        X = l2_normalize(X)
        print("  预处理    : L2 归一化(只保留方向，与 CLIP 的余弦相似度语义一致)")

    Xtr, Xva, ytr, yva = train_test_split(
        X, y, test_size=args.val_frac, random_state=args.seed, stratify=y)
    print("  训练/验证 : %d / %d  (分层切分, seed=%d)" % (len(ytr), len(yva), args.seed))
    print("  ⚠ 验证集标签本身也带噪，val acc 不等于官方榜真实分数，只能横向比较不同方案。")

    class_names = None
    mf = P.F_MANIFEST.replace(".json", "_smoke.json") if args.smoke else P.F_MANIFEST
    if os.path.exists(mf):
        class_names = json.load(open(mf, encoding="utf-8")).get("class_names")

    # ------------------------------------------------------------ 搜索 C
    banner("超参数搜索：L2 正则化强度 C  (共 %d 个候选)" % len(c_grid))
    results = []
    for i, C in enumerate(c_grid):
        print("\n--- [%d/%d] C = %g ---" % (i + 1, len(c_grid), C))
        t0 = time.time()
        if args.backend == "torch":
            r = train_head_torch(Xtr, ytr, Xva, yva, C, args.n_iter, device,
                                 desc="C=%g" % C)
        else:
            r = train_head_sklearn(Xtr, ytr, Xva, yva, C, args.n_iter, desc="C=%g" % C)
        r["C"] = C
        r["seconds"] = time.time() - t0
        results.append(r)
        print("  → 训练准确率 %.4f   验证准确率 %.4f   (%.1fs, %d 迭代)"
              % (r["train_acc"], r["val_acc"], r["seconds"], r["n_iter_run"]))
        gap = r["train_acc"] - r["val_acc"]
        print("     过拟合间隙 = %.4f %s" % (gap, "(偏大，可考虑减小 C 加强正则)" if gap > 0.10 else ""))

    best = max(results, key=lambda r: r["val_acc"])
    banner("搜索结果")
    print("  %-10s %-12s %-12s %-10s" % ("C", "训练acc", "验证acc", "耗时"))
    for r in results:
        star = " ★最佳" if r is best else ""
        print("  %-10g %-12.4f %-12.4f %-10.1fs%s"
              % (r["C"], r["train_acc"], r["val_acc"], r["seconds"], star))

    banner("最佳配置的详细诊断 (C=%g)" % best["C"])
    pc = per_class_report(yva, best["val_pred"], n_classes, class_names)

    # ------------------------------------------------------------ 全量重训
    W, b = best["W"], best["b"]
    if not args.no_refit_full:
        banner("用最佳 C 在【全部训练数据】上重训（这一版用于生成提交）")
        print("  说明: 上面的验证准确率来自 80/20 切分，是诚实的估计；")
        print("        这里把验证集也加回去重训，只为让提交模型多用 20%% 的数据。")
        if args.backend == "torch":
            rf = train_head_torch(X, y, Xva, yva, best["C"], args.n_iter, device,
                                  desc="全量重训")
        else:
            rf = train_head_sklearn(X, y, Xva, yva, best["C"], args.n_iter)
        W, b = rf["W"], rf["b"]
        print("  → 全量训练准确率 %.4f  (注: 此处验证集已被纳入训练，其 acc 不再有意义)"
              % rf["train_acc"])

    # ------------------------------------------------------------ 落盘
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(P.RUNS_DIR, ts + ("_smoke" if args.smoke else ""))
    os.makedirs(run_dir, exist_ok=True)

    np.savez(os.path.join(run_dir, "head.npz"), W=W, b=b,
             normalize=(not args.no_normalize), C=best["C"], n_classes=n_classes)

    metrics = {
        "timestamp": ts, "backend": args.backend, "device": device,
        "normalize": not args.no_normalize, "val_frac": args.val_frac, "seed": args.seed,
        "n_iter_max": args.n_iter, "n_train": int(len(ytr)), "n_val": int(len(yva)),
        "n_classes": n_classes, "refit_full": not args.no_refit_full,
        "best_C": best["C"], "best_train_acc": best["train_acc"],
        "best_val_acc": best["val_acc"],
        "grid": [{"C": r["C"], "train_acc": r["train_acc"], "val_acc": r["val_acc"],
                  "seconds": r["seconds"], "n_iter_run": r["n_iter_run"]} for r in results],
        "per_class": {k: v for k, v in pc.items() if k != "per_class"},
        "history_best_C": best["history"],
        "note": "训练/验证标签均含噪，val_acc 不等于官方榜真实分数",
    }
    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    np.save(os.path.join(run_dir, "per_class_acc.npy"), np.asarray(pc["per_class"]))

    # ------------------------------------------------------------ 软标签(自蒸馏用)
    if args.dump_proba:
        banner("导出全训练集软标签矩阵（自蒸馏用）")
        with torch.no_grad():
            Xt = torch.from_numpy(X).to(device)
            Wt = torch.from_numpy(W).to(device)
            bt = torch.from_numpy(b).to(device)
            proba = torch.softmax(Xt @ Wt.T + bt, dim=1).cpu().numpy().astype(np.float32)
        out = os.path.join(run_dir, "train_proba.npy")
        np.save(out, proba)
        mx = proba.max(1)
        print("  形状 %s  大小 %.1f MB" % (proba.shape, os.path.getsize(out) / 1024 ** 2))
        print("  行顺序与 cache/train_features.npy 严格一致（同一个 X，未打乱）")
        print("  最大概率分布: 均值 %.3f  中位数 %.3f  ≥0.95 的占比 %.1f%%"
              % (mx.mean(), np.median(mx), (mx >= 0.95).mean() * 100))
        print("  → 若 ≥0.95 占比很高，说明输出很'尖'，自蒸馏温度 T 建议从 4 或 8 起步；否则 T=2。")

    banner("完成")
    print("  ★ 基线验证准确率 : %.4f  (C=%g, 后端=%s)"
          % (best["val_acc"], best["C"], args.backend))
    print("  产物目录         :", run_dir)
    print("  下一步           : python scripts/03_predict_test.py --run %s" % os.path.basename(run_dir))


if __name__ == "__main__":
    main()
