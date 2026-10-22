# -*- coding: utf-8 -*-
"""
LR Range Test（Smith 2017, arXiv 1506.01186）：2 分钟画出"损失 vs 学习率"曲线，
自动定出学习率上界。⭐取代"手测三个 lr 各跑一轮"那种 24 分钟的笨办法。

═══════════════════════════════════════════════════════════════════════
一、为什么 sklearn/L-BFGS 的自动步长在这里用不了（HX 的问题）
═══════════════════════════════════════════════════════════════════════

L-BFGS 的 line search 确实能自动定步长，但它有两个隐含前提，微调骨干时两个都不成立：

  ①**问题是凸的** —— 逻辑回归在固定特征上凸、全局唯一最优。
    微调神经网络非凸，无数局部解。

  ⛔②**目标就是最小化训练损失** —— 这条才是要害。
    line search 只看训练损失。而微调预训练模型时我们**根本不想最小化训练损失**，
    我们要的是"学会这个任务，同时别把 4 亿图文对练出的特征毁掉"。这两个目标冲突。
    **实测证据**：LoRA 那次 lr=0.02，训练损失确实降了，验证准确率从 0.9467 崩到 0.6667。
    line search 会兴高采烈地选那个大步长——它没有任何机制能看见"我把预训练知识毁了"。

  附带：06 号脚本里非凸损失（GCE/SCE/ELR）也不能用 L-BFGS，
  因为 line search 要求目标函数在试探期间不变，而增强/dropout/小批量都会让它变。

═══════════════════════════════════════════════════════════════════════
二、那 Range Test 又凭什么能用
═══════════════════════════════════════════════════════════════════════

它不做 line search，也不声称找到"最优 lr"。它只做一件很朴素的事：

  **在一次前进中，让 lr 从极小指数增长到极大，记录每一步的训练损失。**

会看到三段：
  ① lr 太小 → 损失几乎不动（模型没在学）
  ② lr 合适 → 损失快速下降       ← 这一段的右端附近是"最大可用 lr"
  ③ lr 过大 → 损失发散/爆炸       ← 发散点就是硬上界

⚠**它给的是上界，不是答案。** 它同样只看训练损失，所以同样看不见"毁没毁预训练特征"。
⭐**微调的正确用法：取发散点的 1/5 ~ 1/10。**
  从头训练可以贴着最陡点用；微调必须留大余量，因为我们要保护的东西 Range Test 看不见。
  定完范围后，仍要用**验证准确率**验证一个候选（那才是唯一能看见"毁没毁"的量）。

用法:
  python scripts/25_lr_range_test.py                       # @224，约 2 分钟
  python scripts/25_lr_range_test.py --resolution 336      # 换分辨率后重测
  python scripts/25_lr_range_test.py --min-lr 1e-8 --max-lr 1e-2 --steps 150
"""
import _bootstrap  # noqa: F401
from _bootstrap import banner

import argparse
import importlib.util
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as Fn
from PIL import ImageFile
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs import paths as P

ImageFile.LOAD_TRUNCATED_IMAGES = True
SEED, VAL_FRAC = 42, 0.2


def _load_sibling(fname, modname):
    """⚠sys.modules 注册必须在文件顶层，否则 Windows spawn 的 worker 静默挂起。"""
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(modname, os.path.join(here, fname))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


_ex = _load_sibling("01_extract_features.py", "extract01")
_ft = _load_sibling("24_finetune_full.py", "finetune24")   # 复用模型/数据/变换，保证同口径


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", type=int, default=224)
    ap.add_argument("--min-lr", type=float, default=1e-7)
    ap.add_argument("--max-lr", type=float, default=1e-2)
    ap.add_argument("--steps", type=int, default=120, help="120 步 × batch64 ≈ 7680 张 ≈ 2 分钟")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--smooth", type=float, default=0.9, help="损失的指数滑动平均系数")
    ap.add_argument("--aug-scale", type=float, default=0.5)
    ap.add_argument("--head-init", type=str,
                    default=os.path.join(P.CACHE_DIR, "head_loss_sce.npz"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    banner("25 LR Range Test @%dpx  %g → %g  %d 步"
           % (args.resolution, args.min_lr, args.max_lr, args.steps))

    paths, labels, class_names = _ex.list_train_files()
    y = labels.astype(np.int64)
    K = len(class_names)
    itr, _ = train_test_split(np.arange(len(y)), test_size=VAL_FRAC,
                              random_state=SEED, stratify=y)
    tr_paths = [paths[i] for i in itr]

    import clip
    model, _ = clip.load(P.CLIP_MODEL_NAME, device="cpu",
                         download_root=P.CLIP_DOWNLOAD_ROOT)
    model.float()
    g_old, g_new = _ft.interpolate_pos_embed(model, args.resolution)
    visual = model.visual.to(device)
    for p_ in visual.parameters():
        p_.requires_grad = True
    print("  位置编码 %d×%d → %d×%d   可训练参数 %s"
          % (g_old, g_old, g_new, g_new,
             format(sum(p.numel() for p in visual.parameters()), ",")))

    hd = np.load(args.head_init)
    W = torch.tensor(hd["W"], dtype=torch.float32, device=device, requires_grad=True)
    b = torch.tensor(hd["b"], dtype=torch.float32, device=device, requires_grad=True)

    train_tf, _ = _ft.build_transforms(args.resolution, args.aug_scale)
    loader = DataLoader(_ft.ImgDS(tr_paths, y[itr], train_tf), batch_size=args.batch_size,
                        shuffle=True, num_workers=args.num_workers, drop_last=True,
                        pin_memory=(device == "cuda"))

    # 骨干扫 lr；头固定在 1e-3（我们要测的是"骨干能承受多大步长"，别让头掺进来）
    opt = torch.optim.AdamW([
        {"params": list(visual.parameters()), "lr": args.min_lr, "weight_decay": 0.01},
        {"params": [W, b], "lr": 1e-3, "weight_decay": 0.0},
    ])
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    mult = (args.max_lr / args.min_lr) ** (1.0 / max(args.steps - 1, 1))

    lrs, raw, sm = [], [], []
    avg, best = 0.0, float("inf")
    lr = args.min_lr
    visual.train()
    bar = tqdm(total=args.steps, desc="range", unit="step", dynamic_ncols=True)
    it = iter(loader)
    for i in range(args.steps):
        try:
            im, yy = next(it)
        except StopIteration:
            it = iter(loader)
            im, yy = next(it)
        opt.param_groups[0]["lr"] = lr
        im, yy = im.to(device, non_blocking=True), yy.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
            f = visual(im)
        f = f.float()
        f = f / f.norm(dim=1, keepdim=True).clamp_min(1e-12)
        loss = Fn.cross_entropy(f @ W.T + b, yy, label_smoothing=0.1)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(list(visual.parameters()) + [W, b], 1.0)
        scaler.step(opt)
        scaler.update()

        v = float(loss.detach())
        avg = args.smooth * avg + (1 - args.smooth) * v          # EMA 去噪
        s = avg / (1 - args.smooth ** (i + 1))                   # 偏差修正
        lrs.append(lr)
        raw.append(v)
        sm.append(s)
        best = min(best, s)
        bar.update(1)
        bar.set_postfix_str("lr=%.2e loss=%.3f" % (lr, s))
        # 发散判据：平滑损失冲到最低点的 4 倍，或出 NaN → 已经炸了，没必要再往上试
        if i > 10 and (s > 4 * best or not math.isfinite(s)):
            print("\n  在 lr=%.2e 处发散，提前停止" % lr)
            break
        lr *= mult
    bar.close()

    # ---------------------------------------------------------------- 判读
    lrs, sm = np.array(lrs), np.array(sm)
    i_min = int(np.argmin(sm))
    grad = np.gradient(sm, np.log10(lrs))          # 对 log10(lr) 求导，找下降最陡处
    valid = slice(5, max(i_min, 6))
    i_steep = int(np.argmin(grad[valid]) + valid.start) if i_min > 6 else i_min
    lr_div = float(lrs[-1])
    lr_steep = float(lrs[i_steep])

    banner("结果")
    print("  %-24s %.2e" % ("损失最低点的 lr", lrs[i_min]))
    print("  %-24s %.2e   ← 下降最陡" % ("最陡下降处的 lr", lr_steep))
    print("  %-24s %.2e   ← 硬上界" % ("发散处的 lr", lr_div))
    print()
    print("  ⭐**微调建议区间：%.1e ~ %.1e**（最陡点的 1/10 ~ 1/5）"
          % (lr_steep / 10, lr_steep / 5))
    print("  ⚠为什么要留这么大余量：Range Test 只看训练损失，")
    print("    **它看不见'预训练特征被毁了'**——那只能用验证准确率看。")
    print("    从头训练可以贴着最陡点用；微调必须保守。")
    print("  ⭐定完范围后，仍要用 24 号脚本跑 1 轮，看验证指标确认没崩。")

    # 打一张 ASCII 曲线，不用装 matplotlib
    print("\n  损失 vs lr（对数横轴）:")
    lo, hi = float(sm.min()), float(np.percentile(sm, 98))
    for i in range(0, len(sm), max(len(sm) // 24, 1)):
        n = int(48 * (sm[i] - lo) / max(hi - lo, 1e-9))
        print("   %8.1e |%s" % (lrs[i], "#" * max(min(n, 48), 0)))

    out = os.path.join(P.CACHE_DIR, "lr_range_%d.json" % args.resolution)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"resolution": args.resolution, "lrs": lrs.tolist(),
                   "loss_smooth": sm.tolist(), "loss_raw": raw,
                   "lr_steepest": lr_steep, "lr_divergence": lr_div,
                   "suggest_lo": lr_steep / 10, "suggest_hi": lr_steep / 5},
                  f, ensure_ascii=False, indent=2)
    print("\n  已存 → %s" % out)


if __name__ == "__main__":
    main()
