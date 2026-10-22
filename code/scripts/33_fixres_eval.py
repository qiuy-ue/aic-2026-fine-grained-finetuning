# -*- coding: utf-8 -*-
"""
FixRes：不重训，只把【评估分辨率】调高，看能不能白拿几分。

═══════════════════════════════════════════════════════════════════════
一、第一性原理：训练和测试时，物体在画面里的大小是不一样的
═══════════════════════════════════════════════════════════════════════

训练用 `RandomResizedCrop(320, scale=(a,1.0))`：
  先从原图抠出**面积占比 s** 的一块（s 在 [a,1] 均匀），再拉伸到 320×320。
  → 物体被**放大**了 1/√s 倍。

评估用 `Resize(320短边) + CenterCrop(320)`：
  整个短边压到 320，**一点没放大**（相当于 s=1）。

→ ⭐**模型见惯了"放大版"的鸟，考试时给它看"原始大小"的鸟 → 尺度不匹配。**

放大倍数的期望（s~U[a,1]）：
    E[1/√s] = ∫_a^1 s^(-1/2) ds / (1-a) = 2(1-√a)/(1-a) = **2/(1+√a)**
    a=0.5 → 1.172      a=0.4 → 1.225      a=0.3 → 1.292

**补偿办法**：评估时把分辨率提到 320 × E[1/√s]，物体在 token 网格里的占比就对上了。
⚠ViT-B/32 只接受 32 的倍数，所以取最近的合法值。

⭐**这不是玄学调参，是 FixRes(arXiv 1906.06423) 的核心论点，有闭式推导。**

═══════════════════════════════════════════════════════════════════════
二、为什么这条读数比"改骨干"那类可信
═══════════════════════════════════════════════════════════════════════

尺子的病①（夸大死记硬背）针对的是**改变模型权重**的改动。
本脚本**一个权重都不动**，只换评估分辨率 → 拿不到"背训练图"的便宜。
它更像 TTA 那一族（本地读数与真实基本 1:1）。

⚠但仍要防病②，所以验证图**一律先套上测试集同款退化**（长边→500，JPEG q75）。

═══════════════════════════════════════════════════════════════════════
三、⛔判据提前写死（跑之前就钉住，不许事后找理由）
═══════════════════════════════════════════════════════════════════════

**预测**：最优点应落在 `320 × 2/(1+√a)` 向 32 取整的那一档（a=0.5 时 = **384**）。

**采信规则**（用配对检验，不用两个独立准确率相减）：
  - 同一批图、同一个模型，只换分辨率 → **必须做配对比较**：
    b = 320对/新错，c = 320错/新对，Δ=(c−b)/n，标准误 = √(b+c)/n
  - **|Δ| < 2×SE → 判为零**，不许解读
  - 曲线**先升后降且峰值落在预测档** → ✅理论成立，采信
  - 曲线**单调降** → ⛔分辨率没头寸，别投 448 训练
  - 曲线**单调升到边界** → ⚠说明补偿量估小了，但也说明 448 训练值得投

用法（⚠训练同时在跑，务必用小 batch + 单 worker，防 [WinError 1455]）：
  python scripts/33_fixres_eval.py --limit 5000 --batch-size 16 --num-workers 1
"""
import _bootstrap  # noqa: F401
from _bootstrap import banner

import argparse
import importlib.util
import json
import math
import os
import sys
import time

import numpy as np
import torch
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from configs import paths as P

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _ld(fname, modname):
    """⚠sys.modules 注册必须在文件顶层，否则 Windows spawn 的 worker 静默挂起。"""
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(modname, os.path.join(here, fname))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


_ex = _ld("01_extract_features.py", "extract01")
_ft = _ld("24_finetune_full.py", "finetune24")
_dg = _ld("27_domain_gap_jpeg.py", "domaingap27")

# ⭐必须用【未污染】的 ckpt：全量数据那版把验证集也训了，本地读数无意义。
DEFAULT_CKPT = "runs/20260905_203636_v2_320/ckpt_best.pt"   # 榜上 71.12


class DS(Dataset):
    """⚠不能用 lambda（Windows spawn 无法序列化）。"""

    def __init__(self, paths, tf, res):
        self.p, self.tf, self.res = paths, tf, res

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        try:
            with Image.open(self.p[i]) as im:
                return self.tf(_dg.degrade_like_test(im)), i    # ⭐一律先退化
        except Exception:
            return torch.zeros(3, self.res, self.res), i


@torch.no_grad()
def run(ck, res_eval, paths, yv, dev, bs, nw):
    """每档都从原版 CLIP 重建 → 插值到训练分辨率 → 装权重 → 再插值到评估分辨率。"""
    import clip
    res_train = ck["args"].get("resolution", 224)
    model, _ = clip.load(P.CLIP_MODEL_NAME, device="cpu", download_root=P.CLIP_DOWNLOAD_ROOT)
    model.float()
    _ft.interpolate_pos_embed(model, res_train)          # 7×7 → 训练时的网格
    model.visual.load_state_dict(ck["visual"], strict=True)
    g0, g1 = _ft.interpolate_pos_embed(model, res_eval)   # ⭐再插到评估网格
    vis = model.visual.to(dev).eval()

    W = torch.tensor(ck["W"].numpy().astype(np.float32), device=dev)
    b = torch.tensor(ck["b"].numpy().astype(np.float32), device=dev)
    _, tf = _ft.build_transforms(res_eval, 0.5)

    ld = DataLoader(DS(paths, tf, res_eval), batch_size=bs, shuffle=False,
                    num_workers=nw, pin_memory=False)
    c = np.zeros(len(paths), bool)
    bar = tqdm(total=len(paths), desc="@%d(%dx%d格)" % (res_eval, g1, g1),
               unit="img", unit_scale=True, leave=False)
    for im, ix in ld:
        im = im.to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=(dev == "cuda")):
            f = vis(im)
        f = f.float()
        f = f / f.norm(dim=1, keepdim=True).clamp_min(1e-12)
        ixn = ix.numpy()
        c[ixn] = ((f @ W.T + b).argmax(1).cpu().numpy() == yv[ixn])
        bar.update(im.shape[0])
    bar.close()
    del vis, model
    torch.cuda.empty_cache()
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=DEFAULT_CKPT)
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=1,
                    help="⚠训练在跑时保持 1：总进程数必须 <=8，否则 [WinError 1455]")
    ap.add_argument("--res-list", type=str, default="320,352,384,416,448")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    banner("33 FixRes：只调评估分辨率，不重训")

    f = args.ckpt if os.path.exists(args.ckpt) else os.path.join(P.RUNS_DIR, "..", args.ckpt)
    ck = torch.load(f, map_location="cpu", weights_only=False)
    res_train = ck["args"].get("resolution", 224)
    a = ck["args"].get("aug_scale", 0.5)
    pred_res = 32 * round(res_train * (2.0 / (1.0 + math.sqrt(a))) / 32)

    print("  ckpt        %s (ep%s)" % (os.path.basename(os.path.dirname(f)), ck.get("epoch")))
    print("  训练分辨率  %d,  aug_scale a=%.2f" % (res_train, a))
    print("  ⭐理论最优  %d × 2/(1+√%.2f) = %.0f  →  取 32 的倍数 = **%d**"
          % (res_train, a, res_train * 2.0 / (1.0 + math.sqrt(a)), pred_res))
    print("  ⛔判据已写死：|Δ| < 2×配对标准误 一律判为零\n")

    paths, labels, _ = _ex.list_train_files()
    y = labels.astype(np.int64)
    _, iva = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42, stratify=y)
    ms = np.load(os.path.join(P.CACHE_DIR, "leakage_scores.npz"))["val_maxsim"]
    wz = np.load(os.path.join(P.CACHE_DIR, "val_weights.npz"))["weight"]

    sub = np.arange(len(iva))[:args.limit] if args.limit else np.arange(len(iva))
    iva, ms, w = iva[sub], ms[sub], wz[sub]
    va = [paths[i] for i in iva]
    yv = y[iva]
    mnl = ms < 0.90
    print("  验证子集 %d 张（已套测试集同款退化），其中无泄漏 %d 张\n" % (len(va), mnl.sum()))

    reslist = [int(x) for x in args.res_list.split(",")]
    out, base = [], None
    for r in reslist:
        t0 = time.time()
        c = run(ck, r, va, yv, dev, args.batch_size, args.num_workers)
        if base is None:
            base = c
        bb = int((base & ~c).sum())          # 基准对、这档错
        cc = int((~base & c).sum())          # 基准错、这档对
        d = (cc - bb) / len(va)
        se = math.sqrt(bb + cc) / len(va) if (bb + cc) else 0.0
        out.append(dict(res=r, noleak=float(c[mnl].mean()),
                        matched=float((c * w).sum() / w.sum()), raw=float(c.mean()),
                        b=bb, c=cc, delta=d, se=se))
        print("  @%-4d 退化无泄漏 %.4f  退化匹配 %.4f  |  配对Δ %+.4f ± %.4f  (%.1f分钟)"
              % (r, out[-1]["noleak"], out[-1]["matched"], d, se, (time.time() - t0) / 60))

    banner("判读（判据在跑之前就写死了）")
    print("  %-6s %10s %10s %12s %10s %s" % ("分辨率", "无泄漏", "分布匹配", "配对Δ", "2×SE", "判定"))
    print("  " + "-" * 66)
    best, bestd = reslist[0], 0.0
    for r in out:
        if r["res"] == reslist[0]:
            print("  %-6d %10.4f %10.4f %12s %10s %s" % (r["res"], r["noleak"],
                                                         r["matched"], "—", "—", "基准"))
            continue
        sig = abs(r["delta"]) > 2 * r["se"]
        tag = ("✅显著" if r["delta"] > 0 else "❌显著变差") if sig else "⚠判为零"
        print("  %-6d %10.4f %10.4f %+12.4f %10.4f %s"
              % (r["res"], r["noleak"], r["matched"], r["delta"], 2 * r["se"], tag))
        if sig and r["delta"] > bestd:
            best, bestd = r["res"], r["delta"]

    print()
    ds = [r["delta"] for r in out[1:]]
    if bestd == 0.0:
        print("  ⚠**全部落在噪声里** → 分辨率错配这条路没肉，⛔别投 448 训练。")
    elif best == pred_res:
        print("  ✅⭐**峰值 %d 正好落在理论预测档** → FixRes 理论在本任务成立，" % best)
        print("     预测测试集也能拿到约 %+.2f 分（纯测试期改动，倍率接近 1:1）。" % (bestd * 100))
        print("     ⭐**顺带说明分辨率还有头寸 → 448 训练值得投。**")
    elif len(ds) >= 2 and all(ds[i] <= ds[i + 1] for i in range(len(ds) - 1)):
        print("  ⚠**单调升到边界 %d**，说明补偿量估小了。可再往上试；" % reslist[-1])
        print("     ⭐无论如何都说明 **448 训练值得投**。")
    else:
        print("  ⚠峰值 %d ≠ 理论预测 %d。理论只对了一半：**有肉但机理没吃透**，" % (best, pred_res))
        print("     ⛔别把这个 %d 当规律外推到别的训练分辨率，得重测。" % best)

    p = os.path.join(P.CACHE_DIR, "fixres.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(dict(ckpt=f, res_train=res_train, aug_scale=a,
                       pred_res=pred_res, rows=out), fh, ensure_ascii=False, indent=2)
    print("\n  已存 → %s" % p)


if __name__ == "__main__":
    main()
