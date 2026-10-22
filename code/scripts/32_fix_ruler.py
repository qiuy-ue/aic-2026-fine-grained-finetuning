# -*- coding: utf-8 -*-
"""
修尺子 + ⭐用 4 个已知榜分的模型【回测】新尺子到底修好没有。

═══════════════════════════════════════════════════════════════════════
一、旧尺子坏在哪（4 个真值点摆出来，问题一眼可见）
═══════════════════════════════════════════════════════════════════════

  模型            本地(分布匹配)  榜上   Δ本地    Δ真实    倍率
  线性探针         0.6129        59.0    —       —       —
  完整微调@224     0.6747        61.30  +0.0618  +2.30   0.37   ← 夸大3倍
  完整微调@320     0.7074        62.10  +0.0327  +0.80   0.25   ← 夸大4倍
  v2配方@320       0.7302        71.12  +0.0228  +9.02   ⭐3.96 ← 缩小10倍

**同一把尺子，一会儿放大一会儿缩小，方向还相反。** 这是它坏掉的确证。
⛔我曾拿前两段拟合出"0.42 折扣"再外推到第三段——两点定不出曲线，代价是一整天。

═══════════════════════════════════════════════════════════════════════
二、两个独立的病，方向相反
═══════════════════════════════════════════════════════════════════════

**病① 夸大"死记硬背"换来的进步**
  验证集是从训练集切的 → 同图源/同风格/甚至近重复图。
  模型学到的"数据集癖好"在 训练↔验证 之间通用，在 训练↔测试 之间不通用。
  → 越是"加大拟合"的改动（解冻骨干/提分辨率），本地涨幅里套路分越多 → 倍率 < 1

**病② 对"缩小域偏移"的进步完全瞎**
  本地验证 = 高清原图（长边中位640，量化表17.3）
  真实测试 = 压糊的图（长边全500，量化表29.0，q75）
  JPEG退化增强教的是"糊图上也能认对"——在高清卷子上⭐一分都用不上。
  → 尺子上根本没有这一格刻度 → 倍率 >> 1

⭐**倍率不是"方法族"的属性，而是"这次进步里死记硬背 vs 真本事+对齐域 的配比"。**

═══════════════════════════════════════════════════════════════════════
三、修法：一刀治两病
═══════════════════════════════════════════════════════════════════════

  治病② → 验证图套上**与训练增强同一套**退化（长边→500，JPEG q75）
  治病① → 只用**无泄漏子集**（与训练集最高相似度 < 0.90 的那批），
           这批图没有"孪生兄弟"，套路分拿不到

⚠**不能只是宣称它更好。** 用 4 个已知榜分的 ckpt 回测：
  倍率稳定接近 1.0 → ✅修好了      倍率仍忽大忽小 → ❌没修好，继续只信榜单

用法:
  python scripts/32_fix_ruler.py --limit 6000
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

# ⭐4 个已知榜分的锚点。都是 TTA 版分数——TTA 对所有模型是同一个常数偏移，不影响倍率。
ANCHORS = [
    ("runs/20260905_160606_ft224/ckpt_best.pt",  61.30, "完整微调@224"),
    ("runs/20260905_181918_ft320/ckpt_best.pt",  62.10, "完整微调@320"),
    ("runs/20260905_203636_v2_320/ckpt_best.pt", 71.12, "v2配方@320"),
    ("runs/20260906_005936_v2_320/ckpt_best.pt", 72.76, "v2全量 [污染]"),
]


class DS(Dataset):
    """⚠不能用 lambda（Windows spawn 无法序列化），退化开关做成实例属性。"""

    def __init__(self, paths, tf, res, deg):
        self.p, self.tf, self.res, self.deg = paths, tf, res, deg

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        try:
            with Image.open(self.p[i]) as im:
                im = _dg.degrade_like_test(im) if self.deg else im.convert("RGB")
                return self.tf(im), i
        except Exception:
            return torch.zeros(3, self.res, self.res), i


@torch.no_grad()
def ev(vis, W, b, tf, paths, yv, dev, res, deg, nm):
    ld = DataLoader(DS(paths, tf, res, deg), batch_size=32, shuffle=False,
                    num_workers=3, pin_memory=False)
    vis.eval()
    c = np.zeros(len(paths), bool)
    Wt = torch.tensor(W, device=dev)
    bt = torch.tensor(b, device=dev)
    bar = tqdm(total=len(paths), desc=nm, unit="img", unit_scale=True, leave=False)
    for im, ix in ld:
        im = im.to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=(dev == "cuda")):
            f = vis(im)
        f = f.float()
        f = f / f.norm(dim=1, keepdim=True).clamp_min(1e-12)
        ixn = ix.numpy()
        c[ixn] = ((f @ Wt.T + bt).argmax(1).cpu().numpy() == yv[ixn])
        bar.update(im.shape[0])
    bar.close()
    return c.astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=6000,
                    help="验证子集大小（GPU 还在训练，别抢满）")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    banner("32 修尺子：验证集套测试规格 + 只用无泄漏子集，再用 4 个已知榜分回测")

    paths, labels, _ = _ex.list_train_files()
    y = labels.astype(np.int64)
    _, iva = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42, stratify=y)
    wz = np.load(os.path.join(P.CACHE_DIR, "val_weights.npz"))["weight"]
    ms = np.load(os.path.join(P.CACHE_DIR, "leakage_scores.npz"))["val_maxsim"]

    sub = np.arange(len(iva))[:args.limit] if args.limit else np.arange(len(iva))
    iva = iva[sub]
    w = wz[sub]
    ms = ms[sub]
    va = [paths[i] for i in iva]
    yv = y[iva]
    mnl = ms < 0.90
    print("  验证子集 %d 张，其中无泄漏 %d 张（%.0f%%）\n"
          % (len(va), mnl.sum(), 100.0 * mnl.mean()))

    import clip
    rows = []
    for rel, real, tag in ANCHORS:
        f = rel if os.path.exists(rel) else os.path.join(P.RUNS_DIR, "..", rel)
        if not os.path.exists(f):
            print("  ⛔缺 %s，跳过" % rel)
            continue
        ck = torch.load(f, map_location="cpu", weights_only=False)
        res = ck["args"].get("resolution", 224)
        model, _ = clip.load(P.CLIP_MODEL_NAME, device="cpu",
                             download_root=P.CLIP_DOWNLOAD_ROOT)
        model.float()
        _ft.interpolate_pos_embed(model, res)
        vis = model.visual
        vis.load_state_dict(ck["visual"], strict=True)
        vis = vis.to(dev)
        W = ck["W"].numpy().astype(np.float32)
        b = ck["b"].numpy().astype(np.float32)
        _, tf = _ft.build_transforms(res, 0.5)

        t0 = time.time()
        c_hi = ev(vis, W, b, tf, va, yv, dev, res, False, "清晰")
        c_lo = ev(vis, W, b, tf, va, yv, dev, res, True, "退化")
        rows.append(dict(tag=tag, real=real, res=res,
                         old=float((c_hi * w).sum() / w.sum()),   # 旧尺子=清晰+分布匹配
                         new=float(c_lo[mnl].mean()),             # ⭐新尺子=退化+无泄漏
                         deg_matched=float((c_lo * w).sum() / w.sum()),
                         hi_noleak=float(c_hi[mnl].mean())))
        print("  %-16s @%d  旧尺子 %.4f  ⭐新尺子 %.4f  榜上 %.2f  (%.1f分钟)"
              % (tag, res, rows[-1]["old"], rows[-1]["new"], real, (time.time() - t0) / 60))
        del vis, model
        torch.cuda.empty_cache()

    if len(rows) < 3:
        print("\n  ⛔可用锚点不足 3 个，无法回测")
        return

    banner("回测：哪把尺子能追上榜单")
    print("  %-16s %9s %9s %9s %9s" % ("模型", "旧尺子", "⭐新尺子", "退化+匹配", "榜上"))
    print("  " + "-" * 60)
    for r in rows:
        print("  %-16s %9.4f %9.4f %9.4f %9.2f"
              % (r["tag"], r["old"], r["new"], r["deg_matched"], r["real"]))

    clean = [r for r in rows if "污染" not in r["tag"]]
    print("\n  逐段倍率 = Δ真实 / (Δ本地 × 100)  —— ⭐越接近 1.0 且越稳定越好")
    print("  ⚠只用未污染的 %d 个点（全量数据那版验证集已进训练，读数无意义）" % len(clean))
    print("  %-26s %11s %11s" % ("区间", "旧尺子", "⭐新尺子"))
    print("  " + "-" * 50)
    ro, rn = [], []
    for i in range(1, len(clean)):
        dr = clean[i]["real"] - clean[i - 1]["real"]
        do = (clean[i]["old"] - clean[i - 1]["old"]) * 100
        dn = (clean[i]["new"] - clean[i - 1]["new"]) * 100
        fo = dr / do if abs(do) > 1e-9 else float("nan")
        fn = dr / dn if abs(dn) > 1e-9 else float("nan")
        ro.append(fo)
        rn.append(fn)
        print("  %-26s %11.2f %11.2f"
              % ("%s→%s" % (clean[i - 1]["tag"][:7], clean[i]["tag"][:9]), fo, fn))

    banner("判读（判据提前写死，不事后找理由）")

    def spread(v):
        v = [x for x in v if x == x and x > 0]
        return (max(v) / min(v)) if len(v) >= 2 else float("inf")

    so, sn = spread(ro), spread(rn)
    print("  倍率离散度 max/min（⭐越接近 1 说明尺子越线性）：旧 %.1f   ⭐新 %.1f" % (so, sn))
    good = [x for x in rn if x == x and x > 0]
    if sn < so / 2.0 and sn < 3.0:
        print("  ✅**新尺子明显更线性** → 以后用它做决策，倍率取 %.2f" % np.median(good))
    elif sn < so:
        print("  ⚠新尺子有改善但仍不够线性 → 只能参考方向，幅度依旧不可信")
    else:
        print("  ❌**没修好**：域偏移不是倍率乱跳的主因。")
        print("     ⛔继续只信榜单，别用本地挑方案。")

    out = os.path.join(P.CACHE_DIR, "ruler_fix.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(dict(rows=rows, ratio_old=ro, ratio_new=rn,
                       spread_old=so, spread_new=sn), fh, ensure_ascii=False, indent=2)
    print("\n  已存 → %s" % out)


if __name__ == "__main__":
    main()
