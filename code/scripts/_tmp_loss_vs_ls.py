# -*- coding: utf-8 -*-
"""
零 GPU（强制 CPU）：⭐决定性对照 —— 「噪声鲁棒损失」的收益，在**已经有 label smoothing**
之后还剩多少？

背景：02_实验日志 3.1 记「SCE +1.9 分」，是全项目唯一一个被认为有效的抗噪招数。
     但那次对照组是**裸 CE（ls=0）**。而 28_finetune_v2.py:697 现在固定 ls=0.1。
     SCE 的 RCE 项展开后 ≈ β·|A|·(1−p_y) —— 一个**线性、梯度有上界**的项；
     label smoothing 同样是在压"对 p_y→1 的无上界追求"。两者机制高度重叠。
⇒ 若 CE+ls0.1 已经吃掉了 SCE 的大部分收益，则「换损失」这条线的余量就不是 +1.9，而是残差。

⛔ 边界：这是**冻结 CLIP 线性头替身**（41/42 已证它与真身结论可能不一致）。
   只用来回答「残差还剩多少」这一个问题，⛔不外推成榜分。
"""
import io
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as Fn
from sklearn.model_selection import train_test_split

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
DEV = "cpu"                                    # ⛔本轮禁 GPU，强制 CPU
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
SEED, VAL_FRAC = 42, 0.2


def bar(t):
    print("\n" + "=" * 96)
    print(t)
    print("=" * 96)


# ─────────────────────────────────────────────── 损失（全部吃**软目标** T，与 28 号同口径）
def L_ce(lg, T, st):
    return -(T * Fn.log_softmax(lg, 1)).sum(1).mean()


def L_gce(lg, T, st):
    """软目标版 GCE：L = Σ_k T_k (1 − p_k^q)/q。T 为 one-hot 时逐字退化成原版 GCE。
    ⭐这是唯一能和 mixup/cutmix/label-smoothing 共存的 GCE 写法。"""
    q = st["q"]
    p = Fn.softmax(lg, 1).clamp_min(1e-7)
    return (T * (1.0 - p.pow(q)) / q).sum(1).mean()


def L_sce(lg, T, st):
    """软目标版 SCE：CE + β·RCE，RCE = −Σ_k p_k log(T_k)，T 被 clamp 在 exp(A)。"""
    a, b, A = st["alpha"], st["beta"], st["A"]
    p = Fn.softmax(lg, 1).clamp(1e-7, 1.0)
    ce = -(T * Fn.log_softmax(lg, 1)).sum(1).mean()
    rce = -(p * torch.log(T.clamp_min(float(np.exp(A))))).sum(1).mean()
    return a * ce + b * rce


def L_elr(lg, T, st):
    """ELR：CE + λ·log(1 − <p, t>)，t = 逐样本预测 EMA。⚠需要 (N,K) 的额外状态。"""
    lam, beta = st["lam"], st["beta"]
    p = Fn.softmax(lg, 1)
    idx = st["_idx"]
    with torch.no_grad():
        st["target"][idx] = beta * st["target"][idx] + (1 - beta) * p.detach()
    t = st["target"][idx]
    ce = -(T * Fn.log_softmax(lg, 1)).sum(1).mean()
    reg = torch.log((1.0 - (p * t).sum(1)).clamp_min(1e-7)).mean()
    return ce + lam * reg


def L_taylor(lg, T, st):
    """Taylor-CE 阶数 t：−log p ≈ Σ_{i=1..t} (1−p)^i / i。t=1 → MAE（最抗噪），t→∞ → CE。"""
    order = int(st["order"])
    p = Fn.softmax(lg, 1).clamp_min(1e-7)
    acc = 0.0
    for i in range(1, order + 1):
        acc = acc + (1.0 - p).pow(i) / i
    return (T * acc).sum(1).mean()


def L_logitclip(lg, T, st):
    """Logit Clipping：把 logit 向量的 L2 范数截断到 δ，再算 CE。零额外状态。"""
    d = st["delta"]
    n = lg.norm(dim=1, keepdim=True).clamp_min(1e-12)
    lgc = lg * torch.clamp(d / n, max=1.0)
    return -(T * Fn.log_softmax(lgc, 1)).sum(1).mean()


LOSSES = {"ce": L_ce, "gce": L_gce, "sce": L_sce, "elr": L_elr,
          "taylor": L_taylor, "logitclip": L_logitclip}


def train_head(X, T, K, fn, st, epochs=30, lr=3e-3, bs=2048, seed=0, wd=1e-4):
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    W = torch.zeros(K, X.shape[1])
    torch.nn.init.normal_(W, std=0.01)
    W.requires_grad_(True)
    b = torch.zeros(K, requires_grad=True)
    opt = torch.optim.AdamW([W, b], lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n = len(X)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for s in range(0, n, bs):
            i = perm[s:s + bs]
            st["_idx"] = i
            lg = X[i] @ W.T + b
            loss = fn(lg, T[i], st)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sch.step()
    return W.detach(), b.detach()


def soft_target(y, K, ls):
    T = torch.full((len(y), K), ls / K)
    T[torch.arange(len(y)), y] = 1.0 - ls + ls / K
    return T


def paired(ok_a, ok_b):
    n01 = int((~ok_a & ok_b).sum())
    n10 = int((ok_a & ~ok_b).sum())
    n = len(ok_a)
    return (n01 - n10) / n, np.sqrt(max(n01 + n10, 1)) / n, n01, n10


def main():
    t0 = time.time()
    X = np.load(os.path.join(CACHE, "train_features.npy")).astype(np.float32)
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    y = np.load(os.path.join(CACHE, "train_labels.npy")).astype(np.int64)
    K = int(y.max()) + 1
    itr, iva = train_test_split(np.arange(len(y)), test_size=VAL_FRAC,
                                random_state=SEED, stratify=y)
    Xt = torch.from_numpy(X[itr])
    Xv = torch.from_numpy(X[iva])
    ytr = torch.from_numpy(y[itr])
    yv = torch.from_numpy(y[iva])

    lk = np.load(os.path.join(CACHE, "leakage_scores.npz"))
    mnl = lk["val_maxsim"] < 0.90                      # ⭐主尺：无泄漏子集
    ns = np.load(os.path.join(CACHE, "noise_scores.npz"), allow_pickle=True)
    sv = ns["suspicion"][iva]
    m60 = sv <= np.quantile(sv, 0.60)

    bar("损失 × label-smoothing 交叉对照   训练 %d / 验证 %d / 类 %d   设备 %s"
        % (len(itr), len(iva), K, DEV))
    print("主尺=无泄漏子集 %d 张；副尺=干净60%%子集 %d 张；参考=带噪全集"
          % (int(mnl.sum()), int(m60.sum())))

    CFG = [
        ("CE  ls=0.0   ⭐06号当年的对照组", "ce", {}, 0.0),
        ("CE  ls=0.1   ⭐28号现在跑的",     "ce", {}, 0.1),
        ("CE  ls=0.2",                      "ce", {}, 0.2),
        ("SCE β=1 A=-4  ls=0.0 ⭐复现+1.9", "sce", {"alpha": 1.0, "beta": 1.0, "A": -4.0}, 0.0),
        ("SCE β=1 A=-4  ls=0.1",            "sce", {"alpha": 1.0, "beta": 1.0, "A": -4.0}, 0.1),
        ("SCE β=0.3 A=-4 ls=0.1",           "sce", {"alpha": 1.0, "beta": 0.3, "A": -4.0}, 0.1),
        ("GCE q=0.7   ls=0.0",              "gce", {"q": 0.7}, 0.0),
        ("GCE q=0.7   ls=0.1",              "gce", {"q": 0.7}, 0.1),
        ("GCE q=0.4   ls=0.1",              "gce", {"q": 0.4}, 0.1),
        ("ELR λ=3 β=0.7 ls=0.1",            "elr", {"lam": 3.0, "beta": 0.7}, 0.1),
        ("ELR λ=1 β=0.9 ls=0.1",            "elr", {"lam": 1.0, "beta": 0.9}, 0.1),
        ("Taylor t=2  ls=0.1",              "taylor", {"order": 2}, 0.1),
        ("Taylor t=4  ls=0.1",              "taylor", {"order": 4}, 0.1),
        ("LogitClip δ=8 ls=0.1",            "logitclip", {"delta": 8.0}, 0.1),
    ]

    rows, ok_ref = [], None
    print("\n%-34s %9s %9s %9s %9s" % ("配置", "带噪全集", "⭐无泄漏", "干净60%", "训练acc"))
    print("-" * 96)
    for name, ln, prm, ls in CFG:
        st = dict(prm)
        if ln == "elr":
            st["target"] = torch.full((len(itr), K), 1.0 / K)
        T = soft_target(ytr, K, ls)
        W, b = train_head(Xt, T, K, LOSSES[ln], st, seed=0)
        pv = (Xv @ W.T + b).argmax(1)
        ok = (pv == yv).numpy()
        tr = float(((Xt @ W.T + b).argmax(1) == ytr).float().mean())
        print("%-34s %9.4f %9.4f %9.4f %9.4f"
              % (name, ok.mean(), ok[mnl].mean(), ok[m60].mean(), tr))
        rows.append((name, ok))
        if name.startswith("CE  ls=0.1"):
            ok_ref = ok

    bar("⭐配对比较：全部 vs 「CE ls=0.1」（= 28 号现在真正在跑的那个）")
    print("%-34s %10s %10s %10s %s" % ("配置", "配对Δ(全集)", "2SE", "Δ(无泄漏)", "判定"))
    print("-" * 96)
    for name, ok in rows:
        d, se, n01, n10 = paired(ok_ref, ok)
        dn, sen, _, _ = paired(ok_ref[mnl], ok[mnl])
        v = "✅显著更好" if d > 2 * se else ("❌显著更差" if d < -2 * se else "⚠判为零")
        print("%-34s %+10.4f %10.4f %+10.4f %s  (好%d/坏%d)" % (name, d, 2 * se, dn, v, n01, n10))

    json.dump([r[0] for r in rows], open(os.path.join(CACHE, "_tmp_loss_vs_ls_cfg.json"), "w"))
    print("\n总耗时 %.1f 分钟" % ((time.time() - t0) / 60))


if __name__ == "__main__":
    main()
