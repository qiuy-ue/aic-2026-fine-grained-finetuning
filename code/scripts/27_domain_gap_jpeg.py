# -*- coding: utf-8 -*-
"""
决定性检验：测试集与训练集的【图片规格差异】是不是那 9~16 分缺口的真凶。

═══════════════════════════════════════════════════════════════════════
一、查到的事实（2026-09-05，全量扫描，不是抽样）
═══════════════════════════════════════════════════════════════════════

    测试集 24967 张：**长边全部精确等于 500**（p1=p50=p100=500，一张例外都没有）
                     短边中位 375，JPEG 量化表均值 29.0，**100% 是 JPEG**
    训练集         ：长边中位 640，**67.3% 超过 512**
                     短边中位 488，JPEG 量化表均值 17.3，含 PNG/WEBP/GIF/MPO

标定实验：把训练图按「长边缩到500 + JPEG quality=q」处理，q=75 时量化表均值
**恰好 = 29.0**，与测试集分毫不差。（75 正是 PIL 存 JPEG 的默认值。）

→ **测试集的处理流程被完整复原：长边→500，JPEG quality=75。**
→ 我们一直在拿**高清原图**训练，让模型去考**压缩糊图**。模型从没见过 JPEG 块效应
  和降采样模糊。这是实打实的域偏移。

═══════════════════════════════════════════════════════════════════════
二、这个脚本验什么
═══════════════════════════════════════════════════════════════════════

把**同一套退化**施加到验证集上，用**同一个模型**评估。

  · 若准确率从 ~0.78 明显掉向测试集的 ~0.62 → ⭐假设成立，缺口找到了，
    而且我们顺手得到一把**真正对齐测试条件的尺子**
  · 若几乎不掉 → 假设不成立，缺口另有原因，别浪费时间去做退化增强

⚠这是"只改一个变量"的干净对照：同一批图、同一个模型、同一个头，只有图片规格不同。

═══════════════════════════════════════════════════════════════════════
三、⛔顺带炸掉一个我原本要做的决定
═══════════════════════════════════════════════════════════════════════

测试图**短边中位只有 375**。`Resize(448)` 会把它**放大 1.2 倍**——
在插值凭空造像素，不是获取信息。**测试图的实际分辨率就是物理天花板。**
320 已经贴着 375，⛔**别再往 448 冲**。幸好在跑之前查了。

用法:
  python scripts/27_domain_gap_jpeg.py                 # 用最新微调 ckpt
  python scripts/27_domain_gap_jpeg.py --ckpt runs/xxx/ckpt_best.pt
"""
import _bootstrap  # noqa: F401
from _bootstrap import banner

import argparse
import importlib.util
import io
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm import tqdm

from configs import paths as P

ImageFile.LOAD_TRUNCATED_IMAGES = True
SEED, VAL_FRAC = 42, 0.2

# ⭐复原出来的测试集处理参数（长边像素 / JPEG 质量），由全量扫描 + 量化表标定得出
TEST_LONG_SIDE = 500
TEST_JPEG_Q = 75


def _load_sibling(fname, modname):
    """⚠sys.modules 注册必须在文件顶层，否则 Windows spawn 的 worker 静默挂起。"""
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(modname, os.path.join(here, fname))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


_ex = _load_sibling("01_extract_features.py", "extract01")
_ft = _load_sibling("24_finetune_full.py", "finetune24")


def degrade_like_test(im, long_side=TEST_LONG_SIDE, quality=TEST_JPEG_Q):
    """
    把一张 PIL 图按测试集那套流程处理：长边缩到 long_side，再用 JPEG quality 重编码。

    ⚠只缩不放：测试集里长边全是 500，说明主办方是把大图缩下来；
      本来就小于 500 的图不会被放大（放大只会凭空造像素）。
    ⚠必须真的走一遍 JPEG 编解码，不能只是"加点模糊"——
      我们要复现的是 **JPEG 的 8×8 块效应和色度下采样**，那是有特定结构的失真。
    """
    im = im.convert("RGB")
    w, h = im.size
    s = long_side / max(w, h)
    if s < 1.0:
        im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


class ValDS(Dataset):
    """⚠不能用 lambda（Windows spawn 无法序列化），退化开关做成实例属性。"""

    def __init__(self, paths, tf, res, degrade):
        self.p, self.tf, self.res, self.degrade = paths, tf, res, degrade

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        try:
            with Image.open(self.p[i]) as im:
                if self.degrade:
                    im = degrade_like_test(im)
                else:
                    im = im.convert("RGB")
                return self.tf(im), i
        except Exception:
            return torch.zeros(3, self.res, self.res), i


@torch.no_grad()
def run(visual, W, b, tf, paths, yv, device, bs, nw, res, degrade, desc):
    ld = DataLoader(ValDS(paths, tf, res, degrade), batch_size=bs, shuffle=False,
                    num_workers=nw, pin_memory=False)
    visual.eval()
    cs = np.zeros(len(paths), dtype=bool)
    bar = tqdm(total=len(paths), desc=desc, unit="img", unit_scale=True, dynamic_ncols=True)
    for im, ix in ld:
        im = im.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
            f = visual(im)
        f = f.float()
        f = f / f.norm(dim=1, keepdim=True).clamp_min(1e-12)
        ix = ix.numpy()
        cs[ix] = ((f @ torch.tensor(W.T, device=device)
                   + torch.tensor(b, device=device)).argmax(1).cpu().numpy() == yv[ix])
        bar.update(im.shape[0])
    bar.close()
    return cs.astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help=">0 时只测前 N 张（快速冒烟）")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt:
        ckp = args.ckpt
    else:
        cands = sorted(d for d in os.listdir(P.RUNS_DIR)
                       if "_ft" in d and os.path.exists(
                           os.path.join(P.RUNS_DIR, d, "ckpt_best.pt")))
        assert cands, "先跑 24_finetune_full.py"
        ckp = os.path.join(P.RUNS_DIR, cands[-1], "ckpt_best.pt")
    ck = torch.load(ckp, map_location="cpu", weights_only=False)
    res = ck["args"].get("resolution", 224)

    banner("27 域偏移检验：测试集的【长边500 + JPEG q75】是不是缺口的真凶")
    print("  模型 : %s  第%s轮  @%dpx" % (os.path.basename(os.path.dirname(ckp)),
                                          ck.get("epoch"), res))
    print("  退化 : 长边→%d，JPEG quality=%d（由全量扫描+量化表标定复原）"
          % (TEST_LONG_SIDE, TEST_JPEG_Q))

    paths, labels, class_names = _ex.list_train_files()
    y = labels.astype(np.int64)
    _, iva = train_test_split(np.arange(len(y)), test_size=VAL_FRAC,
                              random_state=SEED, stratify=y)
    if args.limit:
        iva = iva[:args.limit]
    va = [paths[i] for i in iva]
    yv = y[iva]

    wz = np.load(os.path.join(P.CACHE_DIR, "val_weights.npz"))
    w = wz["weight"][:len(iva)] if args.limit else wz["weight"]
    lk = np.load(os.path.join(P.CACHE_DIR, "leakage_scores.npz"))
    mnl = (lk["val_maxsim"][:len(iva)] if args.limit else lk["val_maxsim"]) < 0.90

    import clip
    model, _ = clip.load(P.CLIP_MODEL_NAME, device="cpu", download_root=P.CLIP_DOWNLOAD_ROOT)
    model.float()
    _ft.interpolate_pos_embed(model, res)
    visual = model.visual
    visual.load_state_dict(ck["visual"], strict=True)
    visual = visual.to(device)
    W, b = ck["W"].numpy().astype(np.float32), ck["b"].numpy().astype(np.float32)
    _, eval_tf = _ft.build_transforms(res, 0.5)

    print("  验证集 %d 张\n" % len(iva))
    t0 = time.time()
    c_hi = run(visual, W, b, eval_tf, va, yv, device, args.batch_size,
               args.num_workers, res, False, "原始高清")
    c_lo = run(visual, W, b, eval_tf, va, yv, device, args.batch_size,
               args.num_workers, res, True, "测试集规格")

    def rep(c):
        return (float(c.mean()), float((c * w).sum() / w.sum()), float(c[mnl].mean()))

    a1, m1, n1 = rep(c_hi)
    a2, m2, n2 = rep(c_lo)

    banner("结果")
    print("  %-22s %10s %12s %10s" % ("验证集图片规格", "原始等权", "分布匹配", "无泄漏"))
    print("  " + "-" * 58)
    print("  %-22s %10.4f %12.4f %10.4f" % ("原始高清（现状）", a1, m1, n1))
    print("  %-22s %10.4f %12.4f %10.4f" % ("⭐压成测试集规格", a2, m2, n2))
    print("  %-22s %+10.4f %+12.4f %+10.4f" % ("差值", a2 - a1, m2 - m1, n2 - n1))
    print()
    print("  参照：该模型在【真实测试集】上的榜分 ≈ 0.62（@224版实测 62.1）")
    print("  耗时 %.1f 分钟" % ((time.time() - t0) / 60))

    banner("判读")
    drop = a1 - a2
    if drop > 0.03:
        print("  ⭐⭐**假设成立**：光是把验证图压成测试集规格，准确率就掉了 %.4f。" % drop)
        print("     → 那 9~16 分缺口里，**有 %.1f 分是图片规格差异造成的**。" % (100 * drop))
        print("     → ✅**该做的事**：训练时把同一套退化当增强加进去，让模型见过糊图。")
        print("     → ✅**顺带白捡**：把退化后的验证集当新尺子，它比现在这把更贴近测试条件。")
    elif drop > 0.01:
        print("  ⚠**部分成立**：掉了 %.4f，是缺口的一部分但不是全部。值得做退化增强，" % drop)
        print("     但别指望它补上全部差距。")
    else:
        print("  ❌**假设不成立**：只掉了 %.4f。图片规格不是主因，" % drop)
        print("     ⛔别去做退化增强，省下算力找别的原因。")

    out = os.path.join(P.CACHE_DIR, "domain_gap_jpeg.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"ckpt": ckp, "resolution": res,
                   "long_side": TEST_LONG_SIDE, "jpeg_q": TEST_JPEG_Q,
                   "hi": {"raw": a1, "matched": m1, "noleak": n1},
                   "lo": {"raw": a2, "matched": m2, "noleak": n2}},
                  f, ensure_ascii=False, indent=2)
    print("\n  已存 → %s" % out)


if __name__ == "__main__":
    main()
