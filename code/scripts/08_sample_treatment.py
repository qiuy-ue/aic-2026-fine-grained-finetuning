# -*- coding: utf-8 -*-
"""
第2组实验：拿 04 算出的可疑度去【处理训练样本】。B1 筛选/降权 + B2 重新打标。

═══ 四种处理方式（对同一批可疑样本做不同的事）═══

  drop     直接扔掉可疑样本                  —— 最简单，但会连难样本一起扔
  weight   可疑样本降权(不扔)                —— 保留信息，只是少信一点
  relabel  把标签换成邻居多数票(硬替换)      —— 有救的话不如救回来
  soft     换成"原标签与多数票的混合"(软标签) —— 不赌全对，按可疑度插值

⚠为什么要同时试这四种、而不是直接选一个：
  它们对应的假设完全不同。drop 假设可疑样本全是废的；relabel 假设邻居多数票一定对；
  weight/soft 假设"可疑"是程度问题而非是非问题。哪个假设更贴近真实噪声结构，
  不能靠想，只能靠对照实验。

⚠合规性：赛题明确"人工数据清洗仅可作辅助参考，最终方案须能由提交代码直接复现"。
  这里全部是代码里的自动流程（可疑度由 kNN 算出、阈值是参数），没有任何手工名单，符合要求。
  另：全程只动【训练集】，测试集一张都没碰，不触犯"测试数据不得参与训练"。

═══ 评估 ═══
沿用 05 校准过的干净验证子集(40%/60%/80% 三档)，三档同向才采信。
⚠验证集【不做任何处理】——只处理训练集。否则就是在改考卷答案，等于作弊自己。

用法:
  python scripts/08_sample_treatment.py                      # 默认全套
  python scripts/08_sample_treatment.py --methods drop,soft
  python scripts/08_sample_treatment.py --fracs 0.05,0.1,0.2
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
import torch.nn.functional as Fn
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from configs import paths as P

_spec = importlib.util.spec_from_file_location(
    "loss_exp", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "06_loss_experiments.py"))
_le = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_le)

VAL_FRAC = 0.2
# ⚠⚠这里必须包含 1.00(原始未筛选)，因为本脚本存在一个特有的循环优化陷阱：
#   验证集的筛选用的是 suspicion，训练集的处理用的也是 suspicion —— 同一个分数！
#   剔掉高可疑的训练样本，模型当然更擅长低可疑的验证样本(它们本就是同一批分布)，
#   这种"涨分"是自己跟自己较劲的假象，不是真本事。
#   判据：若假象成立，则【验证集筛得越松，收益越小】，到 1.00 时应归零甚至转负。
#   所以 1.00 这一档虽然带噪，却是本组实验里唯一不与处理方式共享偏置的参照。
KEEPS = (0.40, 0.60, 0.80, 1.00)


def train_weighted(Xtr, ytr_soft, w, C, loss_name, n_iter, device, lr=0.05,
                   desc="训练"):
    """
    带样本权重的软标签训练。
    ytr_soft: (n,K) 每行是该样本的目标分布（硬标签就是 one-hot）
    w:        (n,)  样本权重，全 1 就是不加权
    """
    n, d = Xtr.shape
    K = ytr_soft.shape[1]
    Xt = torch.from_numpy(Xtr).to(device)
    Yt = torch.from_numpy(ytr_soft).to(device)
    wt = torch.from_numpy(w).to(device)
    wt = wt / wt.mean()                    # 归一化，让不同权重方案的有效学习率可比

    W = torch.zeros(K, d, device=device, requires_grad=True)
    b = torch.zeros(K, device=device, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr)
    reg_coef = 1.0 / (2.0 * C * n)
    sce = (loss_name == "sce")             # 沿用第1组胜出的 SCE

    bar = tqdm(range(n_iter), desc=desc, unit="it", dynamic_ncols=True, leave=False)
    for _ in bar:
        opt.zero_grad(set_to_none=True)
        logits = Xt @ W.T + b
        logp = Fn.log_softmax(logits, 1)
        ce = -(Yt * logp).sum(1)                        # 软标签交叉熵
        if sce:                                          # SCE 的反向项
            p = Fn.softmax(logits, 1).clamp(1e-7, 1.0)
            rce = -(p * (Yt > 0).float().mul(0.0) + p * (Yt <= 0).float().mul(-4.0)).sum(1)
            ce = ce + rce
        loss = (ce * wt).mean() + reg_coef * (W * W).sum()
        loss.backward()
        opt.step()
    bar.close()
    return W.detach().cpu().numpy(), b.detach().cpu().numpy()


def build_targets(y, vote, susp, K, method, frac):
    """
    根据处理方式，生成 (软标签矩阵, 样本权重, 保留掩码)。
    frac = 认定为"可疑"的比例（可疑度最高的这部分）
    """
    thr = np.quantile(susp, 1 - frac)
    sus = susp >= thr                                    # 被认定可疑的样本
    Y = np.zeros((len(y), K), dtype=np.float32)
    Y[np.arange(len(y)), y] = 1.0                        # 先全部按原标签
    w = np.ones(len(y), dtype=np.float32)
    keep = np.ones(len(y), dtype=bool)

    if method == "drop":
        keep = ~sus
    elif method == "weight":
        w[sus] = 0.2                                     # 可疑样本只给 0.2 的话语权
    elif method == "relabel":
        Y[sus] = 0.0
        Y[sus, vote[sus]] = 1.0                          # 硬替换成邻居多数票
    elif method == "soft":
        # 按可疑度在"原标签"和"多数票"之间插值：越可疑越信邻居
        a = np.clip((susp[sus] - thr) / max(1e-9, susp.max() - thr), 0, 1)[:, None]
        Ys = np.zeros((sus.sum(), K), dtype=np.float32)
        Ys[np.arange(sus.sum()), y[sus]] = 1.0
        Yv = np.zeros((sus.sum(), K), dtype=np.float32)
        Yv[np.arange(sus.sum()), vote[sus]] = 1.0
        Y[sus] = (1 - a) * Ys + a * Yv
    elif method != "none":
        raise ValueError(method)
    return Y, w, keep, int(sus.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", type=str, default="none,drop,weight,relabel,soft")
    ap.add_argument("--fracs", type=str, default="0.10,0.20,0.30")
    ap.add_argument("--loss", type=str, default="sce", choices=["ce", "sce"],
                    help="底座损失，默认用第1组胜出的 SCE")
    ap.add_argument("--C", type=float, default=10.0)
    ap.add_argument("--n-iter", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--seeds", type=str, default="42,1,2")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = [int(s) for s in args.seeds.split(",")]
    fracs = [float(f) for f in args.fracs.split(",")]

    banner("08 第2组实验：样本处理 (底座损失=%s, C=%g)" % (args.loss.upper(), args.C))
    X = np.load(P.F_TRAIN_FEAT).astype(np.float32)
    y = np.load(P.F_TRAIN_LABEL).astype(np.int64)
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    nz = np.load(os.path.join(P.CACHE_DIR, "noise_scores.npz"), allow_pickle=True)
    susp, vote = nz["suspicion"], nz["vote"].astype(np.int64)
    K = int(y.max()) + 1
    print("  样本 %d，类别 %d，种子 %s" % (len(y), K, seeds))
    print("  ⚠只处理训练集；验证集保持原样(否则等于改考卷答案)")

    rows = []
    for sd in seeds:
        itr, iva = train_test_split(np.arange(len(y)), test_size=VAL_FRAC,
                                    random_state=sd, stratify=y)
        Xv, yv, sv = X[iva], y[iva], susp[iva]
        masks = {k: (sv <= np.quantile(sv, k)) for k in KEEPS}
        ytr, susp_tr, vote_tr = y[itr], susp[itr], vote[itr]

        for method in args.methods.split(","):
            method = method.strip()
            flist = [0.0] if method == "none" else fracs
            for fr in flist:
                Y, w, keep, n_sus = build_targets(ytr, vote_tr, susp_tr, K, method, fr)
                tag = method if method == "none" else "%s@%.0f%%" % (method, fr * 100)
                t0 = time.time()
                W, b = train_weighted(X[itr][keep], Y[keep], w[keep], args.C,
                                      args.loss, args.n_iter, device, args.lr, tag)
                pred = np.argmax(Xv @ W.T + b, 1)
                accs = {k: float((pred[m] == yv[m]).mean()) for k, m in masks.items()}
                rows.append(dict(method=method, frac=fr, tag=tag, seed=sd,
                                 n_used=int(keep.sum()), n_suspect=n_sus,
                                 acc={str(k): v for k, v in accs.items()},
                                 seconds=time.time() - t0))
        print("  seed=%-4d 完成 (%d 个配置)" % (sd, len(rows) // len(seeds)))

    # ---------------------------------------------------------------- 汇总
    banner("跨种子汇总（均值±标准差；三档同向才采信）")
    tags = []
    for r in rows:
        if r["tag"] not in tags:
            tags.append(r["tag"])
    agg = {t: {k: np.array([r["acc"][str(k)] for r in rows if r["tag"] == t])
               for k in KEEPS} for t in tags}

    base = agg["none"]
    print("  基准 = none（不做任何处理，纯 %s）\n" % args.loss.upper())
    hdr = "  %-16s %8s " % ("处理方式", "用了样本") + \
          " ".join("%17s" % ("干净%.0f%%" % (k * 100)) for k in KEEPS) + "   判定"
    print(hdr)
    print("  " + "-" * (len(hdr) + 6))
    for t in sorted(tags, key=lambda t: -agg[t][0.60].mean()):
        nu = next(r["n_used"] for r in rows if r["tag"] == t)
        line = "  %-16s %8d " % (t, nu)
        for k in KEEPS:
            v = agg[t][k]
            line += "%10.4f±%.4f " % (v.mean(), v.std())
        if t != "none":
            d = [agg[t][k].mean() - base[k].mean() for k in KEEPS]
            allpos, allneg = all(x > 0 for x in d), all(x < 0 for x in d)
            line += "  %s%+.4f" % ("✅" if allpos else ("❌" if allneg else "⚠混 "),
                                   float(np.mean(d)))
        print(line)
    print("\n  ✅=三档全优于基准  ❌=三档全劣  ⚠=方向不一致(不可采信)")
    print("  ⚠标准差是跨 %d 个种子的；差异小于约 0.006 时本地无法可靠区分(见 07 的分辨率分析)"
          % len(seeds))

    out = os.path.join(P.CACHE_DIR, "sample_treatment.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"loss": args.loss, "C": args.C, "seeds": seeds, "fracs": fracs,
                   "rows": rows, "created": time.strftime("%Y-%m-%d %H:%M:%S")},
                  f, ensure_ascii=False, indent=2)
    print("\n  结果已存 → %s" % out)


if __name__ == "__main__":
    main()
