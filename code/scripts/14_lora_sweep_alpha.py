# -*- coding: utf-8 -*-
"""
WiSE-FT 系数 α 扫描：不重训，只把 LoRA 补丁乘一个系数，看准确率怎么变。

═══════════════════════════════════════════════════════════════════════
一、为什么这是免费的
═══════════════════════════════════════════════════════════════════════

WiSE-FT(arXiv 2109.01903) 原做法：微调后权重与原始权重加权平均，当"微调深度"的保险丝
    θ_α = (1-α)·θ0 + α·θ_ft
普通微调下这很贵：要留两份 8785 万参数的权重，逐元素插值，再造第三个模型。

但 LoRA **从来没修改过 θ0**（它是冻结的），微调后的模型在内存里就字面地存成
θ_ft = θ0 + BA。代进去：

    θ_α = (1-α)θ0 + α(θ0 + BA) = θ0 - αθ0 + αθ0 + αBA = θ0 + α·BA
                                    ↑ 这两项恒等抵消（不是近似）

==> **WiSE-FT 退化成"把补丁乘一个系数"**。代码上就是改一个 float：m.scaling *= α。
    不重训、不占显存、不需要第二份权重。

⚠ 但那一次训练本身省不掉：训练前 B=0，α 乘上去还是 0。α 是缩放旋钮，
  得先有东西给它缩放。省下的是"重复训练 N 次"，不是"训练"。

═══════════════════════════════════════════════════════════════════════
二、⭐为什么要扫到 α > 1（原论文没这么用）
═══════════════════════════════════════════════════════════════════════

原论文 α∈[0,1]，因为它是"在原版和微调版之间插值"。
但公式 θ0 + α·BA 对 α=1.5 同样合法 —— 含义是**把补丁加倍施加**，
等于**外推出一个"训得更狠"的模型**。

这给了一个 2 分钟的诊断，去回答一个本来要花 55 分钟重训才能回答的问题：

    α>1 还在涨   -> ⭐训欠了，该加大 rank / 提学习率，值得再花 55 分钟
    α=1 见顶走平 -> 当前配置已到位，直接交榜
    峰值在 α<1   -> 训过头了，用那个 α，别再加火力
    一条平线     -> 补丁几乎没起作用，回头查配置

⚠外推没有理论保证（插值有"线性模式连通性"的实验支撑，外推没有）。
  它只当**方向诊断**用，不作为提交配置的首选。

═══════════════════════════════════════════════════════════════════════
三、头（分类器）怎么处理
═══════════════════════════════════════════════════════════════════════

上面的抵消只对骨干成立 —— θ0 那边没有天然对应的"原版头"。
本脚本的处理：**用 LP-FT 第一阶段那个头当锚点**，头也按同一个 α 插值：
    W_α = (1-α)·W_LP + α·W_微调后
这样两个端点才自洽：
    α=0 -> 原版骨干 + LP 头 = 就是训练前那个基线
    α=1 -> 完整补丁 + 微调后的头 = 就是 13 号脚本汇总表最后一行
⚠**这是设计选择，不是论文原文规定的。**

用法:
  python scripts/14_lora_sweep_alpha.py --run 20260905_102803_lora
  python scripts/14_lora_sweep_alpha.py --run xxx --tag ep1.50        # 换个存档点
  python scripts/14_lora_sweep_alpha.py --run xxx --alphas 0,0.5,1,2  # 自定义
"""
import _bootstrap  # noqa: F401
from _bootstrap import banner

import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from configs import paths as P


def _load_sibling(fname, modname):
    """⚠必须注册 sys.modules，否则 Windows spawn 的子进程反序列化 Dataset 时
       会报 `Can't pickle ...: import of module '<modname>' failed`。"""
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(modname, os.path.join(here, fname))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod          # ⚠必须在 exec_module 之前
    spec.loader.exec_module(mod)
    return mod


_ex = _load_sibling("01_extract_features.py", "extract01")
_lora = _load_sibling("13_train_lora.py", "lora13")     # 复用同一套注入代码，保证一致


def set_alpha(model, alpha):
    """把所有 LoRA 模块的 scaling 乘上 α。这就是 WiSE-FT 的全部实现。"""
    n = 0
    for m in model.modules():
        if isinstance(m, (_lora.LoRAAttention, _lora.LoRALinear)):
            m.scaling = m.base_scaling * alpha
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default=None,
                    help="runs/ 下的目录名。留空则自动取最新一次 *_lora 训练"
                         "（方便在 PyCharm 里直接点运行，不用记时间戳）")
    ap.add_argument("--tag", type=str, default=None,
                    help="用哪个存档点，默认取最后一个（如 ep3.00）")
    ap.add_argument("--alphas", type=str, default="0,0.25,0.5,0.75,1.0,1.25,1.5,2.0",
                    help="⭐含 >1 的外推点，用于诊断'是不是训欠了'")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=4,
                    help="本脚本没有并行的训练 loader，可以比 13 号开大些")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.run:
        run_dir = args.run if os.path.isdir(args.run) else os.path.join(P.RUNS_DIR, args.run)
    else:
        # 自动找最新一次 LoRA 训练：目录名带 _lora 且里面真的有 ckpt
        cands = sorted(d for d in os.listdir(P.RUNS_DIR)
                       if d.endswith("_lora")
                       and os.path.isdir(os.path.join(P.RUNS_DIR, d))
                       and any(f.startswith("ckpt_")
                               for f in os.listdir(os.path.join(P.RUNS_DIR, d))))
        assert cands, ("runs/ 下没有可用的 *_lora 训练产物，"
                       "先跑 python scripts/13_train_lora.py")
        run_dir = os.path.join(P.RUNS_DIR, cands[-1])   # 目录名是时间戳，排序即时序
        print("  (未指定 --run，自动选用最新一次训练: %s)" % cands[-1])
    assert os.path.isdir(run_dir), "找不到 %s" % run_dir

    # ---------------------------------------------------------- 选存档点
    cks = sorted(f for f in os.listdir(run_dir) if f.startswith("ckpt_") and f.endswith(".pt"))
    assert cks, "该目录下没有 ckpt_*.pt"
    fname = "ckpt_%s.pt" % args.tag if args.tag else cks[-1]
    ck = torch.load(os.path.join(run_dir, fname), map_location="cpu", weights_only=False)
    tr_args = ck["args"]

    banner("14 WiSE-FT α 扫描  run=%s  存档=%s" % (os.path.basename(run_dir), fname))
    print("  该存档的训练配置: r=%d %s%s  lr_lora=%g  epochs=%s"
          % (tr_args["rank"], tr_args["targets"],
             " +mlp" if tr_args.get("lora_mlp") else "",
             tr_args["lr_lora"], tr_args["epochs"]))
    print("  可选存档点: %s" % ", ".join(c[5:-3] for c in cks))

    # ---------------------------------------------------------- 数据（与训练同一套划分）
    paths, labels, class_names = _ex.list_train_files()
    y = labels.astype(np.int64)
    idx = np.arange(len(y))
    _, iva = train_test_split(idx, test_size=_lora.VAL_FRAC,
                              random_state=_lora.SEED, stratify=y)
    va_paths = [paths[i] for i in iva]
    yv = y[iva]

    nz = np.load(os.path.join(P.CACHE_DIR, "noise_scores.npz"), allow_pickle=True)
    sv = nz["suspicion"][iva]
    masks = {k: (sv <= np.quantile(sv, k)) for k in _lora.KEEPS}
    masks[1.0] = np.ones(len(yv), bool)
    print("  验证集 %d 张（与 13 号同一套划分 seed=%d）" % (len(yv), _lora.SEED))

    # ---------------------------------------------------------- 模型 + 补丁
    model, preprocess = _lora.load_clip_fp32(device)
    layers = None
    if tr_args.get("last_n_layers", 0) > 0:
        n_blk = len(model.visual.transformer.resblocks)
        layers = list(range(n_blk - tr_args["last_n_layers"], n_blk))
    targets = tuple(t.strip() for t in tr_args["targets"].split(","))
    _lora.inject_lora(model, tr_args["rank"], tr_args["lora_alpha"],
                      targets, tr_args.get("lora_mlp", False), layers)
    missing, unexpected = model.load_state_dict(ck["lora"], strict=False)
    assert not unexpected, "存档里有模型不认识的键: %s" % unexpected[:3]
    n_loaded = len(ck["lora"])
    model.to(device)
    for m in model.modules():                    # 记下"原始 scaling"，α 在它基础上缩放
        if isinstance(m, (_lora.LoRAAttention, _lora.LoRALinear)):
            m.base_scaling = m.scaling
    print("  已载入 %d 个 LoRA 张量" % n_loaded)

    # ---------------------------------------------------------- 两个头（插值的两端）
    hd0 = np.load(tr_args["head_init"])                    # LP 头 = α=0 的锚点
    W0, b0 = hd0["W"].astype(np.float32), hd0["b"].astype(np.float32)
    W1, b1 = ck["W"].numpy().astype(np.float32), ck["b"].numpy().astype(np.float32)
    print("  头插值: α=0 用 %s，α=1 用存档里微调后的头（⚠这是设计选择，非论文原文）"
          % os.path.basename(tr_args["head_init"]))

    # ---------------------------------------------------------- 扫
    alphas = [float(a) for a in args.alphas.split(",")]
    print("  要扫的 α: %s   （>1 是外推，仅作方向诊断，不作提交首选）"
          % ", ".join("%g" % a for a in alphas))
    print("  预计每个 α 约 1 分钟（重抽 %d 张验证图的特征）\n" % len(yv))

    rows = []
    for a in alphas:
        set_alpha(model, a)
        W = (1 - a) * W0 + a * W1                # 头按同一个 α 插值，两端才自洽
        b = (1 - a) * b0 + a * b1
        t0 = time.time()
        Fv = _lora.encode(model, preprocess, va_paths, device,
                          args.batch_size, args.num_workers, "α=%.2f" % a)
        pred = np.argmax(Fv @ W.T + b, 1)
        accs = {str(k): float((pred[m] == yv[m]).mean()) for k, m in masks.items()}
        rows.append(dict(alpha=a, acc=accs, seconds=time.time() - t0))
        print("  α=%.2f  " % a + "  ".join(
            "干净%.0f%% %.4f" % (k * 100, accs[str(k)]) for k in _lora.KEEPS)
            + "  原始100%% %.4f  (%.0fs)" % (accs["1.0"], time.time() - t0))

    # ---------------------------------------------------------- 汇总
    base = rows[0] if rows[0]["alpha"] == 0 else None
    banner("汇总：相对 α=0（原版骨干 + LP 头）的变化")
    hdr = "  %-8s " % "α" + " ".join("%10s" % ("干净%.0f%%" % (k * 100)) for k in _lora.KEEPS) \
          + "%11s %8s" % ("原始100%", "判定")
    print(hdr)
    print("  " + "-" * (len(hdr) + 4))
    for r in rows:
        if base:
            d = [r["acc"][str(k)] - base["acc"][str(k)] for k in _lora.KEEPS]
            draw = r["acc"]["1.0"] - base["acc"]["1.0"]
            v = "基准" if r is base else (
                "✅一致" if all(x > 0 for x in d) and draw > 0 else
                "❌一致劣" if all(x < 0 for x in d) and draw < 0 else "⚠打架")
            print("  %-8.2f " % r["alpha"] + " ".join("%+10.4f" % x for x in d)
                  + "%+11.4f %8s" % (draw, v))
        else:
            print("  %-8.2f " % r["alpha"] + " ".join(
                "%10.4f" % r["acc"][str(k)] for k in _lora.KEEPS)
                + "%11.4f" % r["acc"]["1.0"])

    # ---------------------------------------------------------- 形状诊断
    key = [r["acc"]["1.0"] for r in rows]         # 用最灵敏的那一列（原始100%）判形状
    imax = int(np.argmax(key))
    span = max(key) - min(key)
    banner("形状诊断（⭐读形状，不读极值）")
    print("  最高点在 α=%.2f，全程波动幅度 %.4f" % (rows[imax]["alpha"], span))
    if span < 0.002:
        verdict = ("一条平线 —— 补丁几乎没起作用。回头查学习率/注入位置，"
                   "别在这个配置上继续投入。")
    elif rows[imax]["alpha"] > 1.0:
        verdict = ("⭐**外推区还在涨** —— 说明这次**训欠了**，补丁劲不够。\n"
                   "     值得再花 ~55 分钟，用更大的 rank(如 16) 或更高的 lr-lora(如 3e-4) 重训。\n"
                   "     ⚠但**别直接拿 α>1 去交榜**：外推没有理论保证，只作方向诊断。")
    elif rows[imax]["alpha"] >= 0.9:
        verdict = ("α=1 附近见顶 —— 当前配置已到位，加火力的边际收益可能不大。\n"
                   "     建议：直接用 α=1 出提交，把算力留给别的方向。")
    else:
        verdict = ("⭐峰值在 α=%.2f < 1 —— **训过头了**，确实需要这根保险丝。\n"
                   "     用这个 α 出提交；⛔别再加大 rank 或学习率。" % rows[imax]["alpha"])
    print("  " + verdict)
    print("\n  ⚠所有判断都受尺子限制：小于 0.006 的差异本地分辨不出来；"
          "候选越多越容易挑到运气（多重比较）。")
    print("  ⚠换算参考：原始100%% 上 +0.006 ≈ 真实 +1 分（n=3 实测，仅当量级参考）。")

    out = os.path.join(run_dir, "alpha_sweep_%s.json" % fname[5:-3])
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"run": os.path.basename(run_dir), "ckpt": fname,
                   "train_args": tr_args, "rows": rows,
                   "created": time.strftime("%Y-%m-%d %H:%M:%S")}, f,
                  ensure_ascii=False, indent=2)
    print("\n  结果已存 → %s" % out)


if __name__ == "__main__":
    main()
