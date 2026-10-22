# -*- coding: utf-8 -*-
"""
拿 5 个真实排行榜分数，回头考 17 号造出来的新尺子。

═══ 为什么必须做这一步 ═══

17 号把加权后的验证准确率从 0.698 拉到 0.608（真实 0.590），偏差缩小 83%。
但**"整体水平对上了"不等于"能分辨方案好坏"**——这是两件事：

  · 对上整体水平 = 尺子的**零点**校准了
  · 能分辨好坏   = 尺子的**刻度**是对的

一把尺子可能零点很准，但刻度是歪的（比如高相似样本压得太狠，
连带把某类方法的优势也压没了）。**只有拿已知答案的题去考它，才知道刻度对不对。**

═══ 考题（都有真实分数当标准答案）═══

  ①CE C=30 vs C=10      真实 C=10 好 +1.1     ⛔老尺子(原始验证集)**判反了**
  ②CE vs SCE            真实 SCE 好 +1.9
  ③无TTA vs 有TTA       真实 TTA 好 +2.357
  ④LoRA α=0 vs α=1      真实 ≈0 (+0.148)      ⛔老尺子说 +1 分，**高估 7 倍**

⭐及格线：①和④是老尺子做错的两道，**新尺子必须做对**，否则就是换了个错法。

用法:
  python scripts/18_revalidate_ruler.py            # 跑①②③（纯缓存，快）
  python scripts/18_revalidate_ruler.py --with-lora  # 加考④（要重抽验证集特征，约2分钟）
"""
import _bootstrap  # noqa: F401
from _bootstrap import banner

import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as Fn
from sklearn.model_selection import train_test_split

from configs import paths as P

SEED, VAL_FRAC = 42, 0.2


def _load_sibling(fname, modname):
    """
    ⚠⚠这两行必须放在【文件顶层】执行，不能塞进 main() 里。

    Windows 的 DataLoader 走 spawn：子进程会重跑主脚本的**顶层代码**，
    然后才反序列化 Dataset 类。而 main() 被 `if __name__=="__main__"` 挡着，
    子进程的 __name__ 是 "__mp_main__"，**根本不会执行 main()**
    → sys.modules['lora13'] 没注册 → worker 反序列化失败 → ⚠**父进程静默挂起**
    （不是报错退出，是死等，比报错难发现得多。2026-09-05 实测卡了 44 分钟）。
    """
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(modname, os.path.join(here, fname))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


_ex = _load_sibling("01_extract_features.py", "extract01")
_lo = _load_sibling("13_train_lora.py", "lora13")


def train_head(Xtr, ytr, C, loss, n_iter=300, lr=0.05, device="cuda", beta=1.0):
    """与 06 号同一套：全批量 Adam，L2 口径 1/(2Cn)。换了优化器就没法跟历史比。"""
    n, d = Xtr.shape
    K = int(ytr.max()) + 1
    Xt = torch.from_numpy(Xtr).to(device)
    yt = torch.from_numpy(ytr).to(device)
    W = torch.zeros(K, d, device=device, requires_grad=True)
    b = torch.zeros(K, device=device, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr)
    reg = 1.0 / (2.0 * C * n)
    for _ in range(n_iter):
        opt.zero_grad(set_to_none=True)
        lg = Xt @ W.T + b
        if loss == "ce":
            L = Fn.cross_entropy(lg, yt)
        else:                                    # SCE：α·CE + β·RCE，A=-4
            p = Fn.softmax(lg, 1).clamp(1e-7, 1.0)
            oh = Fn.one_hot(yt, K).float()
            L = Fn.cross_entropy(lg, yt) - beta * (p * ((1 - oh) * -4.0)).sum(1).mean()
        (L + reg * (W * W).sum()).backward()
        opt.step()
    return W.detach().cpu().numpy(), b.detach().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-lora", action="store_true", help="加考第④题（要重抽特征，约2分钟）")
    ap.add_argument("--lora-run", type=str, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    banner("18 拿真实分数回头考新尺子")

    X = np.load(P.F_TRAIN_FEAT).astype(np.float32)
    y = np.load(P.F_TRAIN_LABEL).astype(np.int64)
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    idx = np.arange(len(y))
    itr, iva = train_test_split(idx, test_size=VAL_FRAC, random_state=SEED, stratify=y)
    Xtr, ytr, Xv, yv = X[itr], y[itr], X[iva], y[iva]

    wz = np.load(os.path.join(P.CACHE_DIR, "val_weights.npz"))
    assert np.array_equal(wz["val_idx"], iva), "权重文件的划分和这里不一致"
    w = wz["weight"]

    # 老尺子：可疑度切出来的"干净60%"（04/05 那一套）
    nz = np.load(os.path.join(P.CACHE_DIR, "noise_scores.npz"), allow_pickle=True)
    sv = nz["suspicion"][iva]
    m60 = sv <= np.quantile(sv, 0.60)

    def three(corr):
        """同一批逐样本对错，用三把尺子各量一遍。"""
        return (float(corr.mean()),                                   # 老：原始等权
                float(corr[m60].mean()),                              # 老：干净60%
                float((corr * w).sum() / w.sum()))                    # ⭐新：分布匹配

    def corr_of(W_, b_, Xval=None):
        Xu = Xv if Xval is None else Xval
        return (np.argmax(Xu @ W_.T + b_, 1) == yv).astype(np.float64)

    cases = []

    # ---------------------------------------------------------------- ① C
    banner("① CE：C=30 vs C=10   （真实：C=10 好 +1.1 分）")
    c30 = corr_of(*train_head(Xtr, ytr, 30.0, "ce", device=device))
    c10 = corr_of(*train_head(Xtr, ytr, 10.0, "ce", device=device))
    cases.append(("① C=10 优于 C=30", +1.1, three(c10), three(c30)))

    # ---------------------------------------------------------------- ② SCE
    banner("② C=10：SCE vs CE     （真实：SCE 好 +1.9 分）")
    sce = corr_of(*train_head(Xtr, ytr, 10.0, "sce", device=device))
    cases.append(("② SCE 优于 CE", +1.9, three(sce), three(c10)))

    # ---------------------------------------------------------------- ③ TTA
    banner("③ SCE：有TTA vs 无TTA （真实：TTA 好 +2.357 分）")
    fa = os.path.join(P.CACHE_DIR, "train_features_aug.npy")
    if os.path.exists(fa):
        A = np.load(fa, mmap_mode="r")                 # (K, N, 512)，只取验证行
        Av = np.asarray(A[:, iva, :], dtype=np.float32)
        Av /= np.maximum(np.linalg.norm(Av, axis=2, keepdims=True), 1e-12)
        Xtta = np.concatenate([Xv[None], Av], 0).mean(0)   # ⚠必须带原图，它是锚点
        Xtta /= np.maximum(np.linalg.norm(Xtta, axis=1, keepdims=True), 1e-12)
        # ⚠必须用【同一个头】比，只让特征不同——否则就同时改了两个变量
        Ws, bs = train_head(Xtr, ytr, 10.0, "sce", device=device)
        cases.append(("③ TTA 优于 无TTA", +2.357,
                      three(corr_of(Ws, bs, Xtta)), three(corr_of(Ws, bs))))
    else:
        print("  跳过：找不到 %s" % fa)

    # ---------------------------------------------------------------- ④ LoRA
    if args.with_lora:
        banner("④ LoRA：α=1 vs α=0   （真实：+0.148，基本为零）")
        rd = args.lora_run or sorted(d for d in os.listdir(P.RUNS_DIR) if d.endswith("_lora"))[-1]
        rd = os.path.join(P.RUNS_DIR, rd)
        ckf = sorted(f for f in os.listdir(rd) if f.startswith("ckpt_"))[-1]
        ck = torch.load(os.path.join(rd, ckf), map_location="cpu", weights_only=False)
        tr = ck["args"]
        model, prep = _lo.load_clip_fp32(device)
        _lo.inject_lora(model, tr["rank"], tr["lora_alpha"],
                        tuple(t.strip() for t in tr["targets"].split(",")),
                        tr.get("lora_mlp", False), None)
        model.load_state_dict(ck["lora"], strict=False)
        model.to(device)
        paths, _, _ = _ex.list_train_files()
        vp = [paths[i] for i in iva]
        Fv = _lo.encode(model, prep, vp, device, 128, 2, "LoRA验证集")
        Wl = ck["W"].numpy().astype(np.float32)
        bl = ck["b"].numpy().astype(np.float32)
        lora = (np.argmax(Fv @ Wl.T + bl, 1) == yv).astype(np.float64)
        cases.append(("④ LoRA 优于 无LoRA", +0.148, three(lora), three(sce)))

    # ---------------------------------------------------------------- 判卷
    banner("判卷：三把尺子谁做对了几道")
    print("  %-22s %8s | %-22s | %-22s" % ("题目", "真实分差", "老尺子", "新尺子"))
    print("  %-22s %8s | %10s %11s | %10s %10s"
          % ("", "", "原始等权", "干净60%", "分布匹配", "推算真实"))
    print("  " + "-" * 96)
    score = {"raw": 0, "clean60": 0, "new": 0}
    rows = []
    for name, truth, a, bb in cases:
        d = [a[i] - bb[i] for i in range(3)]
        ok = [np.sign(x) == np.sign(truth) for x in d]
        for k, i in (("raw", 0), ("clean60", 1), ("new", 2)):
            score[k] += int(ok[i])
        # 新尺子换算：加权验证集与真实测试集是同一个难度尺度，1 个百分点 ≈ 1 分
        print("  %-22s %+8.2f | %+9.4f%s %+9.4f%s | %+9.4f%s %+9.2f"
              % (name, truth, d[0], "✅" if ok[0] else "❌",
                 d[1], "✅" if ok[1] else "❌",
                 d[2], "✅" if ok[2] else "❌", d[2] * 100))
        rows.append(dict(case=name, truth=truth, raw=d[0], clean60=d[1], new=d[2],
                         ok_raw=bool(ok[0]), ok_clean60=bool(ok[1]), ok_new=bool(ok[2])))

    n = len(cases)
    print("\n  方向正确率：原始等权 %d/%d   干净60%% %d/%d   ⭐分布匹配 %d/%d"
          % (score["raw"], n, score["clean60"], n, score["new"], n))
    print("\n  ⭐⭐关键看'推算真实'那一列：新尺子和真实测试集是**同一个难度尺度**，")
    print("     所以本地差 1 个百分点就该 ≈ 真实差 1 分，**不再需要 168x 那种玄学倍率**。")
    print("     如果这一列的数字和'真实分差'对得上，说明刻度也校准了，不只是零点。")

    with open(os.path.join(P.CACHE_DIR, "ruler_revalidation.json"), "w", encoding="utf-8") as f:
        json.dump({"cases": rows, "score": score}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
