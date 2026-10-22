# -*- coding: utf-8 -*-
"""62 · WiSE-FT：权重空间插值 θ_α = (1−α)·θ_原版CLIP + α·θ_微调。**零 GPU、零训练。**

⭐为什么做它（2026-09-16）
───────────────────────────────────────────────────────────────────────
本工程在权重空间上只动过**一个**点：SWA（55 号，L 的 ep09~12 平均）。
实测它离 ep12 只有 `‖Δθ‖/‖θ‖ = 0.000564`，而**整条微调相对原版 CLIP 的位移是 0.07098**
→ SWA 只是微调总位移的 **0.79%**，功能上只翻转了 **292/24967 (1.17%)** 张预测。
⛔ 一个只翻 292 张的改动，**天花板**就是 +1.17 分（全翻对），实际期望 ±0.14。

WiSE-FT 给的是**同一条直线上的一个可调幅度旋钮**：α=0.9 就是往回走 10%，
= SWA 扰动的 **12.6 倍**；α 越小幅度越大。⭐这是权重侧唯一"能把幅度调到够大"的现成手段。

机制（arXiv 2109.01903，Wortsman et al. 2022 CVPR）：微调会破坏预训练模型的鲁棒性，
把两端权重线性插值可以同时拿到"微调的拟合"和"预训练的鲁棒"。⚠ 论文的主战场是
**分布偏移**；本赛题训练/测试集之间**确实存在已实测的偏移**（训练集 1198 种亮度量化表、
3.5% 非 JPEG、长边=500 只占 3.25%；测试集 1 种表、100% 4:2:0、100% 长边 500）。
⛔ 但"同分布 Top-1 也会涨"在本赛题**未经验证**，这正是要上榜测的。

⛔ 不是集成：产出是**单一一份权重**，推理跑一次前向，骨干仍是 OpenAI 基线 CLIP ViT-B/32。
   （合规自洽：如果权重平均违规，EMA 从第一天起就违规——见台账实验 O。）

═══════════════════════════════════════════════════════════════════════
用法
═══════════════════════════════════════════════════════════════════════
  # ① 只看幅度，不写文件（⭐先跑这个）
  python scripts\\62_wise_ft.py --ckpt runs\\20260914_125846_v2_448_cos_dp020_re25_ep12\\ckpt_ep12.pt --scan

  # ② 生成 α=0.9 的权重
  python scripts\\62_wise_ft.py --ckpt runs\\20260914_125846_v2_448_cos_dp020_re25_ep12\\ckpt_ep12.pt --alpha 0.9

  下一步（GPU）：58 建池 → 59 组 c3 配方；或 56_pack_ckpt.py 直接出 Q 口径包。
"""
import _bootstrap  # noqa: F401
from _bootstrap import banner

import argparse
import importlib.util
import os
import sys

import numpy as np
import torch

from configs import paths as P


def _load_sibling(fname, modname):
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(modname, os.path.join(here, fname))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


pred26 = _load_sibling("26_predict_finetuned.py", "pred26")
_ft = pred26._ft


def log(m):
    print(m, flush=True)


def rel(a, b):
    """‖a−b‖ / ‖b‖，只统计浮点张量。"""
    d = sum(float(((a[k].float() - b[k].float()) ** 2).sum())
            for k in b if b[k].is_floating_point())
    n = sum(float((b[k].float() ** 2).sum()) for k in b if b[k].is_floating_point())
    return (d / max(n, 1e-12)) ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="微调端 θ_ft（28 号存的任意 ckpt）")
    ap.add_argument("--alpha", type=float, default=0.9,
                    help="θ_α = (1−α)·θ_CLIP + α·θ_ft。α=1 就是原 ckpt，α 越小越靠近 CLIP")
    ap.add_argument("--head-mode", type=str, default="keep", choices=["keep", "lerp"],
                    help="keep=分类头原样不动（幅度全部来自骨干，最保守）；"
                         "lerp=头也插值回 cache/head_loss_sce.npz（= 冻结CLIP特征上的线性探针，"
                         "它才是 θ_CLIP 天然配对的分类器）。⚠余弦头只看方向，故按行归一后插值再归一")
    ap.add_argument("--head-init", type=str,
                    default=os.path.join(P.CACHE_DIR, "head_loss_sce.npz"),
                    help="--head-mode lerp 时的头 θ0，默认就是 28 号 --head-init 用的那一份")
    ap.add_argument("--out", type=str, default=None,
                    help="默认写到 ckpt 同目录 ckpt_wise<α三位>_<headmode>.pt")
    ap.add_argument("--scan", action="store_true",
                    help="只打印若干 α 的位移幅度表，⛔不写任何文件")
    args = ap.parse_args()

    ck_path = args.ckpt if os.path.isabs(args.ckpt) else os.path.join(P.PROJECT_ROOT, args.ckpt)
    if not os.path.exists(ck_path):
        sys.exit("⛔ 找不到 %s" % ck_path)

    banner("60 WiSE-FT  α=%.3f  head=%s" % (args.alpha, args.head_mode))
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    tr = ck.get("args", {})
    res_train = int(tr.get("resolution", 224))
    log("  ckpt   %s  ep%s  训练@%dpx  cosine=%s"
        % (os.path.basename(os.path.dirname(ck_path)), ck.get("epoch"),
           res_train, tr.get("cosine_head")))

    # ── θ0：原版 CLIP，先把位置编码插到 ckpt 的网格（⛔顺序同 26/28/58）
    import clip
    model, _ = clip.load(P.CLIP_MODEL_NAME, device="cpu", download_root=P.CLIP_DOWNLOAD_ROOT)
    model.float()
    g0, g1 = _ft.interpolate_pos_embed(model, res_train)
    log("  θ0     原版 CLIP ViT-B/32，位置编码 %d×%d → %d×%d" % (g0, g0, g1, g1))
    sd0 = {k: v.detach().clone() for k, v in model.visual.state_dict().items()}
    sd1 = ck["visual"]

    # ── 三道闸（照 55 号的规矩：错了不会报错、只会悄悄变差的地方必须拦）
    k0, k1 = set(sd0), set(sd1)
    if k0 != k1:
        sys.exit("⛔ 键集不一致：只在CLIP=%s 只在ckpt=%s" % (sorted(k0 - k1), sorted(k1 - k0)))
    bad = [k for k in k0 if tuple(sd0[k].shape) != tuple(sd1[k].shape)]
    if bad:
        sys.exit("⛔ 形状不一致：%s" % bad[:5])
    log("  ✅ 键集/形状一致（%d 个张量）" % len(k0))
    d_full = rel(sd1, sd0)
    log("  ⭐ 微调总位移 ‖θ_ft − θ_CLIP‖/‖θ_CLIP‖ = %.5f" % d_full)

    if args.scan:
        log("\n  α      ‖θ_α−θ_ft‖/‖θ_ft‖    相对 SWA(0.000564) 的倍数")
        nft = sum(float((sd1[k].float() ** 2).sum()) for k in sd1 if sd1[k].is_floating_point())
        n0 = sum(float(((sd1[k].float() - sd0[k].float()) ** 2).sum())
                 for k in sd1 if sd1[k].is_floating_point())
        base = (n0 / max(nft, 1e-12)) ** 0.5     # α=0 时的位移
        for a in (0.98, 0.95, 0.90, 0.80, 0.70, 0.50, 0.0):
            dd = (1 - a) * base
            log("  %.2f   %.6f              %6.1f×" % (a, dd, dd / 0.000564))
        log("\n  ⛔ SCAN 模式，未写任何文件。")
        return

    # ── 逐元素插值
    a = float(args.alpha)
    out = {}
    for k in sd1:
        if sd1[k].is_floating_point():
            out[k] = ((1 - a) * sd0[k].float() + a * sd1[k].float()).to(sd1[k].dtype)
        else:
            out[k] = sd1[k].clone()
    log("  骨干插值完成，位移 ‖θ_α−θ_ft‖/‖θ_ft‖ = %.6f" % rel(out, sd1))

    W = ck["W"].clone()
    b = ck["b"].clone()
    if args.head_mode == "lerp":
        hz = np.load(args.head_init)
        W0 = torch.tensor(hz["W"], dtype=torch.float32)
        if tuple(W0.shape) != tuple(W.shape):
            sys.exit("⛔ 头形状不一致：%s vs %s（子中心头请用 --head-mode keep）"
                     % (tuple(W0.shape), tuple(W.shape)))
        if tr.get("cosine_head") or int(tr.get("subcenters", 0) or 0) > 1:
            # ⭐余弦头只看方向：先把两端逐行归一，插值后再归一，⛔别让量级差主导插值
            n0 = W0 / W0.norm(dim=1, keepdim=True).clamp_min(1e-12)
            n1 = W.float() / W.float().norm(dim=1, keepdim=True).clamp_min(1e-12)
            Wn = (1 - a) * n0 + a * n1
            Wn = Wn / Wn.norm(dim=1, keepdim=True).clamp_min(1e-12)
            scale = W.float().norm(dim=1, keepdim=True)      # 沿用微调端的行范数
            W = (Wn * scale).to(ck["W"].dtype)
            cs = float((n1 * (W.float() / W.float().norm(dim=1, keepdim=True))).sum(1).mean())
            log("  头插值（余弦口径）：与微调头的平均余弦 %.5f" % cs)
        else:
            W = ((1 - a) * W0 + a * W.float()).to(ck["W"].dtype)
            b = ((1 - a) * torch.tensor(hz["b"], dtype=torch.float32)
                 + a * b.float()).to(ck["b"].dtype)
            log("  头插值（线性口径）：W 与 b 都插")
    else:
        log("  头**原样保留**（--head-mode keep）")

    blob = {"visual": out, "W": W, "b": b,
            "epoch": "wise%03d%s" % (round(a * 100), "L" if args.head_mode == "lerp" else ""),
            "args": ck["args"],                 # ⛔必须原样带走：26/45/58 靠它读 resolution/cosine_head
            "metrics": ck.get("metrics", {}),   # ⚠沿用源 ckpt 的，⛔这不是 WiSE 自己的指标
            "used_ema": ck.get("used_ema", False),
            "wise_alpha": a, "wise_head_mode": args.head_mode,
            "wise_from": os.path.basename(ck_path)}
    if "s" in ck:
        blob["s"] = float(ck["s"])

    out_p = args.out or os.path.join(os.path.dirname(ck_path),
                                     "ckpt_wise%03d%s.pt"
                                     % (round(a * 100),
                                        "_lerp" if args.head_mode == "lerp" else ""))
    if not os.path.isabs(out_p):
        out_p = os.path.join(P.PROJECT_ROOT, out_p)
    torch.save(blob, out_p)
    log("\n  ⭐已写出 %s  (%.0f MB)" % (out_p, os.path.getsize(out_p) / 1e6))
    log("  下一步（GPU）：")
    log("    A 快读（Q 口径，对照 79.8734）：scripts\\56_pack_ckpt.py --ckpt %s --eval-res 544" % out_p)
    log("    B 满配（c3 口径，对照 80.4542）：58 建 448/544 两个池 → 59 --recipe c3")


if __name__ == "__main__":
    main()
