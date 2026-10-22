# -*- coding: utf-8 -*-
"""
第1组对照实验：换损失函数抗噪。固定其他一切，只动"用什么损失去训这个线性头"。

═══ 为什么损失函数能抗噪 ═══

标准交叉熵有个致命弱点：**对错得越离谱的样本，惩罚越重**。
一个标错的样本，模型越是"正确地"认为它不属于那个类，损失就越大、梯度越大，
反而**逼着模型去死记这个错误**。下面三个损失都是在修这个毛病。

  GCE  广义交叉熵：L=(1-p_y^q)/q。q→0 退化成 CE，q=1 变成 MAE。
       q 越大，单个样本能贡献的梯度上限越低，噪声样本再离谱也拽不动模型。
       Zhang & Sabuncu, arXiv 1805.07836（引用 1951，已双源核实）
  SCE  对称交叉熵：L=α·CE + β·RCE。RCE 是把预测和标签调换位置算的"反向"交叉熵，
       它对错标签天然不敏感，用来抵消 CE 的过拟合倾向。
       Wang et al., arXiv 1908.06112（引用 1082，已双源核实）
  ELR  早期学习正则化：利用"模型先学对的、后背错的"这个规律——
       维护每个样本预测的移动平均 t，加一项 log(1-<p,t>) 把后期的预测拽回早期判断。
       Liu et al., arXiv 2007.00151（引用 318，已双源核实）

═══ 评估用哪把尺子 ═══
用 05 校准过的干净验证子集（保留 40%/60%/80% 三档同时报告，三档同向才采信）。
⚠绝对数值不可信（保留40%时acc≈0.95，真实只有57%），只能横向比大小。

用法:
  python scripts/06_loss_experiments.py                    # 跑全部四组
  python scripts/06_loss_experiments.py --losses ce,gce    # 只跑指定的
  python scripts/06_loss_experiments.py --C 10             # 指定正则强度
"""
import _bootstrap  # noqa: F401
from _bootstrap import banner

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as Fn
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from configs import paths as P

SEED, VAL_FRAC = 42, 0.2
KEEPS = (0.40, 0.60, 0.80)      # 三档干净度截断，三档同向才采信


# ====================================================================== 损失
def loss_ce(logits, y, state):
    return Fn.cross_entropy(logits, y)


def loss_gce(logits, y, state):
    """L = (1 - p_y^q)/q，q∈(0,1]。q 越大越抗噪但越难拟合。"""
    q = state["q"]
    p = Fn.softmax(logits, 1).gather(1, y[:, None]).squeeze(1).clamp_min(1e-7)
    return ((1.0 - p.pow(q)) / q).mean()


def loss_sce(logits, y, state):
    """L = α·CE + β·RCE。RCE 用 A 代替 log(0)，A 是个负常数(原论文取 -4 或 -6)。"""
    a, bta, A = state["alpha"], state["beta"], state["A"]
    ce = Fn.cross_entropy(logits, y)
    p = Fn.softmax(logits, 1).clamp(1e-7, 1.0)
    oh = Fn.one_hot(y, logits.shape[1]).float()
    rce = -(p * (oh * 0.0 + (1 - oh) * A)).sum(1).mean()   # log(one-hot) → 0 或 A
    return a * ce + bta * rce


def loss_elr(logits, y, state):
    """
    L = CE + λ·log(1 - <p, t>)，t = 预测的移动平均(EMA)。
    t 记录"模型早期认为这个样本是什么"，正则项惩罚后期预测偏离 t，
    从而阻止模型在训练后期去死记噪声标签。
    ⚠t 必须在 LBFGS 的 line search 之外更新，否则目标函数在一次 step 内会变，
      违反优化器假设。这里用 Adam 逐迭代更新，天然满足。
    """
    lam, beta = state["lam"], state["beta"]
    p = Fn.softmax(logits, 1)
    t = state["target"]
    with torch.no_grad():                        # t 只做记录，不参与反传
        pd = (p / p.sum(1, keepdim=True)).detach()
        t.mul_(beta).add_(pd, alpha=1 - beta)
    ce = Fn.cross_entropy(logits, y)
    reg = torch.log((1.0 - (p * t).sum(1)).clamp_min(1e-7)).mean()
    return ce + lam * reg


LOSSES = {
    "ce":  (loss_ce,  {}, "标准交叉熵（对照组）"),
    "gce": (loss_gce, {"q": 0.7}, "广义交叉熵 q=0.7"),
    "sce": (loss_sce, {"alpha": 1.0, "beta": 1.0, "A": -4.0}, "对称交叉熵 α=β=1, A=-4"),
    "elr": (loss_elr, {"lam": 3.0, "beta": 0.7}, "早期学习正则 λ=3, β=0.7"),
}


# ====================================================================== 训练
def train(Xtr, ytr, C, loss_name, n_iter, device, lr=0.05):
    """
    全批量 Adam 训练。

    ⚠为什么这里不用 02 脚本的 LBFGS：LBFGS 依赖目标函数在 line search 期间保持不变，
      而 GCE/SCE/ELR 都是非凸的、ELR 还要逐步更新内部状态 t，两者不兼容。
      改用全批量 Adam：梯度无采样噪声，线性模型 + 已归一化特征下 lr 很稳。
      **所有损失（含 CE 对照组）都走同一套优化器**，否则比较不公平。
    """
    fn, params, _ = LOSSES[loss_name]
    n, d = Xtr.shape
    K = int(ytr.max()) + 1
    Xt = torch.from_numpy(Xtr).to(device)
    yt = torch.from_numpy(ytr).to(device)

    W = torch.zeros(K, d, device=device, requires_grad=True)
    b = torch.zeros(K, device=device, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr)

    state = dict(params)
    if loss_name == "elr":                       # ELR 需要每个样本一份 EMA 记录
        state["target"] = torch.full((n, K), 1.0 / K, device=device)

    reg_coef = 1.0 / (2.0 * C * n)               # 与 02 脚本同一套 L2 口径，可比
    bar = tqdm(range(n_iter), desc="%-4s" % loss_name, unit="it", dynamic_ncols=True)
    for it in bar:
        opt.zero_grad(set_to_none=True)
        logits = Xt @ W.T + b
        loss = fn(logits, yt, state) + reg_coef * (W * W).sum()
        loss.backward()
        opt.step()
        if (it + 1) % 25 == 0:
            with torch.no_grad():
                acc = ((Xt @ W.T + b).argmax(1) == yt).float().mean().item()
            bar.set_postfix_str("loss=%.4f 训练acc=%.4f" % (float(loss.detach()), acc))
    bar.close()
    return W.detach().cpu().numpy(), b.detach().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--losses", type=str, default="ce,gce,sce,elr")
    ap.add_argument("--C", type=float, default=10.0, help="L2 强度，用已知较优的 C=10")
    ap.add_argument("--n-iter", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--c-grid", type=str, default=None,
                    help="逗号分隔。⚠固定 C=10 比较损失函数是【不公平】的——"
                         "C=10 是为交叉熵调出来的，而 GCE/ELR 自己就在抑制噪声拟合，"
                         "叠加强 L2 会过度压制导致欠拟合。给每个损失配自己的最优 C 才公平。")
    ap.add_argument("--q-grid", type=str, default=None, help="GCE 的 q 候选，逗号分隔")
    ap.add_argument("--lam-grid", type=str, default=None, help="ELR 的 λ 候选，逗号分隔")
    ap.add_argument("--beta-grid", type=str, default=None,
                    help="SCE 的 β 候选。β 控制抗噪项强度：SCE 展开后 ≈ CE + β·4·(1-p_y)，"
                         "β→0 退化回普通交叉熵，β 越大越抗噪但越难拟合(难样本也拽不动模型)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    banner("06 第1组实验：损失函数对照 (C=%g, %d 迭代, Adam lr=%g)"
           % (args.C, args.n_iter, args.lr))

    X = np.load(P.F_TRAIN_FEAT).astype(np.float32)
    y = np.load(P.F_TRAIN_LABEL).astype(np.int64)
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    nz = np.load(os.path.join(P.CACHE_DIR, "noise_scores.npz"), allow_pickle=True)
    susp = nz["suspicion"]

    idx = np.arange(len(y))
    itr, iva = train_test_split(idx, test_size=VAL_FRAC, random_state=SEED, stratify=y)
    Xtr, ytr, Xv, yv, sv = X[itr], y[itr], X[iva], y[iva], susp[iva]
    print("  训练 %d / 验证 %d" % (len(ytr), len(yv)))
    print("  评估尺子: 干净验证子集 保留 %s（05 已校准，只能比大小不能读数）"
          % "/".join("%.0f%%" % (k * 100) for k in KEEPS))

    masks = {k: (sv <= np.quantile(sv, k)) for k in KEEPS}
    masks[1.0] = np.ones(len(yv), bool)

    # 每个损失要试哪些 (C, 关键超参) 组合
    c_list = [float(c) for c in args.c_grid.split(",")] if args.c_grid else [args.C]
    q_list = [float(q) for q in args.q_grid.split(",")] if args.q_grid else [None]
    lam_list = [float(l) for l in args.lam_grid.split(",")] if args.lam_grid else [None]
    beta_list = [float(b) for b in args.beta_grid.split(",")] if args.beta_grid else [None]
    EXTRA_KEY = {"gce": "q", "elr": "lam", "sce": "beta"}

    def combos(name):
        extra = {"gce": q_list, "elr": lam_list, "sce": beta_list}.get(name, [None])
        return [(c, e) for c in c_list for e in extra]

    rows = []
    for name in args.losses.split(","):
        name = name.strip()
        if name not in LOSSES:
            print("  跳过未知损失:", name)
            continue
        banner("%s — %s" % (name.upper(), LOSSES[name][2]))
        for C, extra in combos(name):
            if extra is not None:                      # 临时覆盖该损失的关键超参
                LOSSES[name][1][EXTRA_KEY[name]] = extra
            sym = {"gce": "q", "elr": "λ", "sce": "β"}.get(name, "?")
            tag = "C=%g" % C + ("" if extra is None else "  %s=%g" % (sym, extra))
            t0 = time.time()
            W, b = train(Xtr, ytr, C, name, args.n_iter, device, args.lr)
            pred = np.argmax(Xv @ W.T + b, 1)
            tr_acc = float((np.argmax(Xtr @ W.T + b, 1) == ytr).mean())
            accs = {k: float((pred[m] == yv[m]).mean()) for k, m in masks.items()}
            print("  %-18s 训练acc %.4f  带噪 %.4f  " % (tag, tr_acc, accs[1.0])
                  + " ".join("干净%.0f%% %.4f" % (k * 100, accs[k]) for k in KEEPS)
                  + "  (%.0fs)" % (time.time() - t0))
            rows.append(dict(loss=name, C=C, extra=extra, tag=tag, train_acc=tr_acc,
                             acc={str(k): v for k, v in accs.items()},
                             seconds=time.time() - t0,
                             params=dict(LOSSES[name][1])))
            # ⚠⚠2026-09-05 修的 bug：原来这里只写 "head_loss_<loss>.npz"，
            #   而它在 (C × 超参) 的循环【内部】—— 每个配置都覆盖同一个文件名，
            #   **最后一个配置胜出**。β 网格跑到 8 时，cache/head_loss_sce.npz 其实是 β=8 的头，
            #   但 13/16/17 三个脚本都在读它、且注释里称它是"榜上 59.0 那版(β=1)"。
            #   → LoRA 的 LP-FT 起点头用错了、尺子校准的口径也错了。
            # 修法：文件名带上关键超参，各存各的，谁都不覆盖谁。
            hp = dict(W=W, b=b, normalize=True, C=C, n_classes=int(y.max()) + 1,
                      loss=name, params=json.dumps(dict(LOSSES[name][1])))
            sym = EXTRA_KEY.get(name)
            tagf = "head_loss_%s_C%g%s.npz" % (
                name, C, "" if (sym is None or extra is None) else "_%s%g" % (sym, extra))
            np.savez(os.path.join(P.CACHE_DIR, tagf), **hp)
            # 兼容旧路径：仍写一份不带超参的，但**只在单配置运行时**才写，
            # 避免网格扫描时又出现"最后一个悄悄覆盖"的老毛病。
            if len(c_list) == 1 and len(combos(name)) == 1:
                np.savez(os.path.join(P.CACHE_DIR, "head_loss_%s.npz" % name), **hp)
            else:
                print("      ⚠网格模式：只写 %s，不覆盖 head_loss_%s.npz" % (tagf, name))

    # ---------------------------------------------------------------- 汇总
    banner("汇总（按干净60%%排序；三档同向才采信）")
    # 对照基准 = 交叉熵里最好的那个配置（而不是随便一个），否则对 CE 不公平
    ce_rows = [r for r in rows if r["loss"] == "ce"]
    base = max(ce_rows, key=lambda r: r["acc"][str(0.60)]) if ce_rows else None
    if base:
        print("  对照基准 = CE 的最优配置 (%s)\n" % base["tag"])
    hdr = "  %-5s %-16s %9s %9s " % ("损失", "配置", "训练acc", "带噪") + \
          " ".join("%9s" % ("干净%.0f%%" % (k * 100)) for k in KEEPS)
    print(hdr)
    print("  " + "-" * (len(hdr) + 12))
    for r in sorted(rows, key=lambda r: -r["acc"][str(0.60)]):
        line = "  %-5s %-16s %9.4f %9.4f " % (r["loss"], r["tag"], r["train_acc"],
                                              r["acc"]["1.0"]) + \
               " ".join("%9.4f" % r["acc"][str(k)] for k in KEEPS)
        if base and r is not base:
            d = [r["acc"][str(k)] - base["acc"][str(k)] for k in KEEPS]
            allpos, allneg = all(x > 0 for x in d), all(x < 0 for x in d)
            line += "  %s%+.4f" % ("✅" if allpos else ("❌" if allneg else "⚠混 "),
                                   float(np.mean(d)))
        print(line)
    if base:
        print("\n  ✅=三档全优于CE最优  ❌=三档全劣  ⚠=三档方向不一致(不可采信)")

    out = os.path.join(P.CACHE_DIR, "loss_experiments.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"C": args.C, "n_iter": args.n_iter, "lr": args.lr,
                   "keeps": list(KEEPS), "rows": rows,
                   "created": time.strftime("%Y-%m-%d %H:%M:%S")}, f,
                  ensure_ascii=False, indent=2)
    print("\n  结果已存 → %s" % out)


if __name__ == "__main__":
    main()
