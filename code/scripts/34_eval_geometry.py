# -*- coding: utf-8 -*-
"""
评估几何两阶段搜索：【看到多少画面】× 【看得多细】。一个权重都不改，纯测试期白捡分。

═══════════════════════════════════════════════════════════════════════
一、第一性原理：测试集的形状是死的，我们的评估变换却在扔画面
═══════════════════════════════════════════════════════════════════════

实测（24967 张全体，四个置信度分组一模一样）：
    长边中位 500   短边中位 375   面积中位 18.8万   → 主办方统一压过，**没有小图**

现在的评估变换：
    T.Resize(320)      → 500×375 变成 427×320
    T.CenterCrop(320)  → 从 427 宽里只取中间 320

  **左右各切掉 12.5%，一共丢掉 25% 的画幅。**
  ⚠这个比例和分辨率无关：375/500=0.75 是死的，升到 448 照样丢 25%。

训练时用的是 RandomResizedCrop —— 随机位置裁。模型见过画面的各个角落；
考试却只给它看正中间。**鸟不站在正中央的那些图，翅膀尖/尾羽可能已经被裁没了。**

═══════════════════════════════════════════════════════════════════════
二、第二个维度：FixRes 尺度错配（arXiv 1906.06423，有闭式解）
═══════════════════════════════════════════════════════════════════════

RandomResizedCrop(320, scale=(a,1.0)) 抠出面积占比 s∈[a,1] 再拉到 320
  → 物体被放大 1/√s 倍，期望 E[1/√s] = 2(1-√a)/(1-a) = **2/(1+√a)**

本工程 a=0.30 → 倍数 1.292 → 320×1.292 = 413.5 → 取 32 的倍数 = ⭐**416**

评估用 Resize+CenterCrop 等价于 s=1（一点没放大）→ **训练放大、考试不放大 = 尺度错配**。
补偿办法：把评估分辨率提到 416，物体在 token 网格里的占比就对上了。

═══════════════════════════════════════════════════════════════════════
三、为什么分两阶段，而不是 4×5=20 格全扫
═══════════════════════════════════════════════════════════════════════

两个维度近似正交：「裁剪方式」管**看到多少内容**，「分辨率」管**看得多细**。
  阶段1：固定 320，比 4 种裁剪方式        (4 次)
  阶段2：用阶段1 冠军，扫 5 档分辨率      (5 次)
  阶段3：⭐**正交性复核** —— 用阶段1 亚军在阶段2 最优分辨率再跑一次，
         看排序有没有翻转。翻转 = 正交假设不成立 = 阶段2 的结论不能外推。
9+1 次而不是 20 次，且第 3 阶段专门用来**证伪自己的假设**。

⚠没有第 3 阶段的话，"两维正交"就只是个未经检验的方便假设——
   记忆里踩过的坑：《一次只改一个变量，且警惕隐藏的第二个变量》。

═══════════════════════════════════════════════════════════════════════
四、⛔判据提前写死（跑之前就钉住，不许跑完再找理由）
═══════════════════════════════════════════════════════════════════════

**尺子按记忆里的修法配置**（两个方向相反的病都要治）：
  治病② 域偏移 → 验证图**一律套上测试集同款退化**（长边→500，JPEG q75）
  治病① 死记硬背 → 主读数只看**无泄漏子集**（与训练集最高相似度 < 0.90）

**采信规则（配对检验，不用两个独立准确率相减）**：
  同一批图、同一个模型，只换几何 → 逐样本配对：
    b = 基准对/新错，  c = 基准错/新对
    Δ = (c−b)/n,  SE = √(b+c)/n
  ⛔**|Δ| < 2×SE 一律判为零，不许解读。**

**阶段1 预注册判读**：
  B/C/E 显著 > A  → ✅确证 CenterCrop 在丢分，改评估变换（纯白捡）
  全落噪声        → ⚠画面边缘没有判别信息，保持 A，别折腾
  A 显著最好      → ⛔"摄影师把主体放中央"的先验是真的，⛔别动裁剪

**阶段2 预注册判读**（预测靶心 = 416）：
  峰值正好落在 416      → ✅FixRes 理论在本任务成立，且**448 训练值得投**
  单调升到边界 448      → ⚠补偿量估小了，但结论同样是"448 训练值得投"
  单调降 / 全落噪声     → ⛔分辨率这条路没肉，**别投 448 训练**
  峰值 ≠ 416 且非单调   → ⚠有肉但机理没吃透，⛔别把这个数外推到别的训练分辨率

用法（⚠必须等训练结束再跑，否则 [WinError 1455]）：
  python scripts/34_eval_geometry.py --limit 6000
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
from statistics import NormalDist      # ⛔多重比较校正用，标准库，无新依赖

import numpy as np
import torch
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
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

# ⭐必须用【未污染】的 ckpt：全量数据那版把验证集也训进去了，本地读数无意义。
DEFAULT_CKPT = "runs/20260905_203636_v2_320/ckpt_best.pt"   # 榜上 71.12

NORM = T.Normalize((0.48145466, 0.4578275, 0.40821073),
                   (0.26862954, 0.26130258, 0.27577711))
BIC = T.InterpolationMode.BICUBIC


# ── 几何变体：⚠全部写成类，不能用 lambda（Windows spawn 序列化不了）────────
class PadSquare:
    """补边成正方形：保住 100% 画面且不形变，代价是主体在画布里变小。"""

    def __call__(self, im):
        w, h = im.size
        s = max(w, h)
        c = Image.new("RGB", (s, s), (124, 116, 104))   # CLIP 训练集均值色
        c.paste(im, ((s - w) // 2, (s - h) // 2))
        return c


class ThreeCrop:
    """沿长边取 左/中/右 三块方形，堆成 (3,3,H,W)。丢 0% 画面且不形变，代价 3 倍算力。"""

    def __init__(self, res):
        self.res = res
        self.post = T.Compose([T.ToTensor(), NORM])

    def __call__(self, im):
        w, h = im.size
        s = min(w, h)
        outs = []
        for k in (0, 1, 2):
            if w >= h:
                x = int(k * (w - s) / 2)
                box = (x, 0, x + s, s)
            else:
                y = int(k * (h - s) / 2)
                box = (0, y, s, y + s)
            outs.append(self.post(im.crop(box).resize((self.res, self.res), Image.BICUBIC)))
        return torch.stack(outs, 0)


def geometries(res):
    """返回 {名字: (transform, 每张图产生几个视角)}。"""
    return {
        "A 现状:短边缩放+中央裁(丢25%画面)": (
            T.Compose([T.Resize(res, interpolation=BIC), T.CenterCrop(res),
                       T.ToTensor(), NORM]), 1),
        "B 直接压方形(全画面,横向挤25%)": (
            T.Compose([T.Resize((res, res), interpolation=BIC), T.ToTensor(), NORM]), 1),
        "C 补边成方形(全画面,主体缩小)": (
            T.Compose([PadSquare(), T.Resize((res, res), interpolation=BIC),
                       T.ToTensor(), NORM]), 1),
        "E 三裁剪左中右平均(全画面,3x算力)": (ThreeCrop(res), 3),
    }


class DS(Dataset):
    """⭐所有验证图一律先套测试集同款退化——这是修尺子病②的关键，不能省。"""

    def __init__(self, paths, tf, res, ncrop):
        self.p, self.tf, self.res, self.nc = paths, tf, res, ncrop

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        try:
            with Image.open(self.p[i]) as im:
                return self.tf(_dg.degrade_like_test(im)), i
        except Exception:
            z = torch.zeros(3, self.res, self.res)
            return (z.unsqueeze(0).repeat(self.nc, 1, 1, 1) if self.nc > 1 else z), i


@torch.no_grad()
def evaluate(vis, W, b, tf, ncrop, paths, yv, dev, res, bs, nw, desc, store=None):
    """返回逐样本的对/错布尔数组。多视角时对 logits 取平均（线性层，平均 logit 才是对的）。

    ⭐`store` 传一个 dict 进来时，顺便把**逐样本 logits**（float16）存进去。
    多花 6 MB/配置，换来三件事后免 GPU 就能做的分析：
      ① 多尺度/多几何**组合**（同一个模型的多视角平均，仍是单模型，合规）
      ② 预测分布离散度 TV（用来裁决"测试集到底均不均衡"那桩悬案）
      ③ 阈值/温度重扫，不用再跑一遍推理
    ⛔上一轮就是因为"算完就扔"，任何要 logits 的分析都得重跑 GPU。别再犯。
    """
    ld = DataLoader(DS(paths, tf, res, ncrop), batch_size=bs, shuffle=False,
                    num_workers=nw, pin_memory=False)
    correct = np.zeros(len(paths), bool)
    lg_all = np.zeros((len(paths), W.shape[0]), np.float16) if store is not None else None
    bar = tqdm(total=len(paths), desc=desc[:20], unit="img", unit_scale=True, leave=False)
    for im, ix in ld:
        n = im.shape[0]
        if ncrop > 1:
            im = im.view(n * ncrop, *im.shape[2:])
        im = im.to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=(dev == "cuda")):
            f = vis(im)
        f = f.float()
        f = f / f.norm(dim=1, keepdim=True).clamp_min(1e-12)
        logit = f @ W.T + b
        if ncrop > 1:
            logit = logit.view(n, ncrop, -1).mean(1)
        ixn = ix.numpy()
        lc = logit.cpu().numpy()
        correct[ixn] = (lc.argmax(1) == yv[ixn])
        if lg_all is not None:
            lg_all[ixn] = lc.astype(np.float16)
        bar.update(n)
    bar.close()
    if store is not None:
        store[desc] = lg_all
    return correct


def dispersion(logit, K):
    """预测类别分布的离散度。⭐用来裁决『测试集到底均不均衡』那桩悬案。

    ⛔悬案背景：旧的置换检验说测试集不均衡（偏离零分布 94 个标准差），
    但那个检验**明写的前提**是"假定测试图片成像特性与训练集一致"——而实测成像特性差很多
    （训练 640×480/85KB，测试 500×375/38KB），且它的零分布**没把模型自身的吸引子类偏见算进去**。

    ⭐**决定性测量**：本脚本的验证图**已知均衡**且**套了测试集同款退化**，
    所以这里量出来的 TV，就是"均衡 + 同款域偏移"下**应该**有的 TV。
      量出的 TV ≈ 测试集的 0.0959  → 域偏移足以解释，**测试集大概率是均衡的**
      量出的 TV 仍远低于 0.0959    → 域偏移解释不掉，**测试集确实不均衡**
    ⚠尺寸修正：验证子集每类张数少于测试集的 49.93，TV 天然偏高，须按 √(49.93/m) 折算。
    """
    pred = logit.astype(np.float32).argmax(1)
    c = np.bincount(pred, minlength=K).astype(float)
    n = len(pred)
    tv = 0.5 * np.abs(c / n - 1.0 / K).sum()
    corr = math.sqrt(49.93 / (n / K))
    return tv, tv / corr, float(c.std()), int(c.max())


def paired(base, new, n):
    """配对检验：只看两者判断不一致的那些题，比两个独立准确率相减灵敏得多。"""
    bb = int((base & ~new).sum())      # 基准对、新方案错
    cc = int((~base & new).sum())      # 基准错、新方案对
    d = (cc - bb) / n
    se = math.sqrt(bb + cc) / n if (bb + cc) else 0.0
    return bb, cc, d, se


def show(rows, base_key, title):
    """统一的判读表。⛔|Δ| < 2×SE 一律判为零。"""
    banner(title)
    print("  %-34s %9s %9s %11s %9s  %s"
          % ("方案", "⭐无泄漏", "分布匹配", "配对Δ", "2×SE", "判定"))
    print("  " + "-" * 92)
    best, bestd = base_key, 0.0
    for k, r in rows.items():
        if k == base_key:
            print("  %-34s %9.4f %9.4f %11s %9s  基准" % (k[:34], r["noleak"], r["matched"], "—", "—"))
            continue
        sig = abs(r["delta"]) > 2 * r["se"]
        tag = ("✅显著更好" if r["delta"] > 0 else "❌显著更差") if sig else "⚠判为零"
        print("  %-34s %9.4f %9.4f %+11.4f %9.4f  %s"
              % (k[:34], r["noleak"], r["matched"], r["delta"], 2 * r["se"], tag))
        if sig and r["delta"] > bestd:
            best, bestd = k, r["delta"]
    return best, bestd


def load_model(ck, res_eval, dev):
    """每次都从原版 CLIP 重建 → 插值到训练网格 → 装权重 → 再插值到评估网格。
    ⭐interpolate_pos_embed 是从模型**当前**网格读起点的，所以可以连调两次。"""
    import clip
    res_train = ck["args"].get("resolution", 224)
    model, _ = clip.load(P.CLIP_MODEL_NAME, device="cpu",
                         download_root=P.CLIP_DOWNLOAD_ROOT)
    model.float()
    _ft.interpolate_pos_embed(model, res_train)          # 7×7 → 训练网格
    model.visual.load_state_dict(ck["visual"], strict=True)
    _, g = _ft.interpolate_pos_embed(model, res_eval)     # ⭐再插到评估网格
    return model, model.visual.to(dev).eval(), g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=DEFAULT_CKPT)
    ap.add_argument("--limit", type=int, default=6000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=3)
    ap.add_argument("--res-list", type=str, default="320,352,384,416,448")
    ap.add_argument("--skip-stage2", action="store_true")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    banner("34 评估几何：看到多少画面 × 看得多细（一个权重都不改）")

    f = args.ckpt if os.path.exists(args.ckpt) else os.path.join(P.RUNS_DIR, "..", args.ckpt)
    ck = torch.load(f, map_location="cpu", weights_only=False)
    res_train = ck["args"].get("resolution", 224)
    a = ck["args"].get("aug_scale", 0.5)
    pred_res = 32 * round(res_train * (2.0 / (1.0 + math.sqrt(a))) / 32)
    if ck["args"].get("full_data"):
        print("  ⛔⛔ 这个 ckpt 是 full_data 训的，验证集已进训练，本地读数无意义！中止。")
        return

    print("  ckpt          %s (ep%s)" % (os.path.basename(os.path.dirname(f)), ck.get("epoch")))
    print("  训练分辨率    %d,  aug_scale a=%.2f" % (res_train, a))
    print("  ⭐FixRes靶心  %d × 2/(1+√%.2f) = %.1f → 取32倍数 = **%d**"
          % (res_train, a, res_train * 2.0 / (1.0 + math.sqrt(a)), pred_res))
    print("  ⛔判据已写死  |Δ| < 2×配对标准误 → 判为零，不许解读\n")

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
    n = len(va)
    print("  验证子集 %d 张（⭐全部套测试集同款退化），其中无泄漏 %d 张（%.0f%%）\n"
          % (n, mnl.sum(), 100.0 * mnl.mean()))

    def pack(c, base):
        bb, cc, d, se = paired(base, c, n) if base is not None else (0, 0, 0.0, 0.0)
        return dict(noleak=float(c[mnl].mean()), matched=float((c * w).sum() / w.sum()),
                    raw=float(c.mean()), b=bb, c=cc, delta=d, se=se)

    result = {"ckpt": f, "res_train": res_train, "aug_scale": a, "pred_res": pred_res}

    # ── 阶段 1：固定 320，比 4 种裁剪方式 ────────────────────────────────
    banner("阶段1／3：固定 %d px，只改【看到多少画面】" % res_train)
    model, vis, g = load_model(ck, res_train, dev)
    W = torch.tensor(ck["W"].numpy().astype(np.float32), device=dev)
    b = torch.tensor(ck["b"].numpy().astype(np.float32), device=dev)

    LG = {}                     # ⭐所有配置的逐样本 logits 都存这里，供阶段4 免 GPU 复用
    s1, base_c, s1keys = {}, None, []
    for k, (tf, nc) in geometries(res_train).items():
        t0 = time.time()
        c = evaluate(vis, W, b, tf, nc, va, yv, dev, res_train,
                     max(4, args.batch_size // nc), args.num_workers, k, store=LG)
        if base_c is None:
            base_c = c
        s1[k] = pack(c, base_c)
        s1[k]["_c"] = c
        s1keys.append(k)
        print("  %-34s 无泄漏 %.4f  (%.1f 分钟)" % (k[:34], s1[k]["noleak"], (time.time() - t0) / 60))
    del vis, model
    torch.cuda.empty_cache()

    g1best, g1d = show(s1, s1keys[0], "阶段1 判读（判据跑之前就写死了）")
    if g1d == 0.0:
        print("\n  ⚠**全部落在噪声里** → 画面边缘没有判别信息，保持现状 A，⛔别在这上面花时间。")
    else:
        print("\n  ✅**%s** 显著更好，配对 Δ=%+.4f → 换评估变换即可白捡，一个权重都不用改。"
              % (g1best[:34], g1d))
    # 亚军（用于阶段3 正交性复核）
    runner = sorted([k for k in s1keys if k != g1best],
                    key=lambda k: -s1[k]["delta"])[0]
    result["stage1"] = {k: {kk: vv for kk, vv in v.items() if kk != "_c"} for k, v in s1.items()}
    result["stage1_best"] = g1best

    if args.skip_stage2:
        return

    # ── 阶段 2：用阶段1 冠军，扫分辨率 ──────────────────────────────────
    banner("阶段2／3：用【%s】，只改【看得多细】(靶心 %d)" % (g1best[:28], pred_res))
    tf_best_maker, nc_best = None, None
    s2, base2, s2keys = {}, None, []
    reslist = [int(x) for x in args.res_list.split(",")]
    for r in reslist:
        t0 = time.time()
        model, vis, g = load_model(ck, r, dev)
        W = torch.tensor(ck["W"].numpy().astype(np.float32), device=dev)
        b = torch.tensor(ck["b"].numpy().astype(np.float32), device=dev)
        tf, nc = geometries(r)[g1best]
        c = evaluate(vis, W, b, tf, nc, va, yv, dev, r,
                     max(4, args.batch_size // nc), args.num_workers, "@%d" % r, store=LG)
        if base2 is None:
            base2 = c
        key = "%d px (%d×%d 格)" % (r, g, g)
        s2[key] = pack(c, base2)
        s2keys.append(key)
        print("  %-34s 无泄漏 %.4f  (%.1f 分钟)" % (key, s2[key]["noleak"], (time.time() - t0) / 60))
        del vis, model
        torch.cuda.empty_cache()

    r2best, r2d = show(s2, s2keys[0], "阶段2 判读")
    best_res = int(r2best.split()[0])
    ds = [s2[k]["delta"] for k in s2keys[1:]]
    print()
    if r2d == 0.0:
        print("  ⛔**全部落在噪声里** → 分辨率错配这条路没肉，**别投 448 训练**。")
        verdict = "no_res_headroom"
    elif best_res == pred_res:
        print("  ✅⭐**峰值 %d 正好落在 FixRes 理论靶心** → 理论在本任务成立。" % best_res)
        print("     纯测试期改动（不碰权重），倍率接近 1:1，预计榜上约 %+.2f 分。" % (r2d * 100))
        print("     ⭐并且说明分辨率仍有头寸 → **448 训练值得投**。")
        verdict = "fixres_confirmed"
    elif len(ds) >= 2 and all(ds[i] <= ds[i + 1] for i in range(len(ds) - 1)):
        print("  ⚠**单调升到边界 %d**，说明补偿量估小了，可再往上试。" % reslist[-1])
        print("     ⭐无论如何都说明 **448 训练值得投**。")
        verdict = "monotone_up"
    else:
        print("  ⚠峰值 %d ≠ 理论靶心 %d：**有肉但机理没吃透**。" % (best_res, pred_res))
        print("     ⛔别把这个 %d 外推到别的训练分辨率，换了配置必须重测。" % best_res)
        verdict = "peak_mismatch"
    result["stage2"] = s2
    result["stage2_best_res"] = best_res
    result["verdict"] = verdict

    # ── 阶段 3：正交性复核 ──────────────────────────────────────────────
    banner("阶段3／3：正交性复核 —— 用亚军【%s】在 %d px 再跑一次" % (runner[:24], best_res))
    print("  ⭐目的：如果冠亚军在 %d px 上排序翻转，说明两个维度**不正交**，" % best_res)
    print("     阶段2 的结论就不能外推。这一步是专门用来**证伪我自己的假设**的。\n")
    model, vis, g = load_model(ck, best_res, dev)
    W = torch.tensor(ck["W"].numpy().astype(np.float32), device=dev)
    b = torch.tensor(ck["b"].numpy().astype(np.float32), device=dev)
    tf, nc = geometries(best_res)[runner]
    c_run = evaluate(vis, W, b, tf, nc, va, yv, dev, best_res,
                     max(4, args.batch_size // nc), args.num_workers, "复核")
    del vis, model
    torch.cuda.empty_cache()

    c_win = None   # 冠军在 best_res 上的结果：阶段2 已经算过
    win_noleak = s2["%d px (%d×%d 格)" % (best_res, best_res // 32, best_res // 32)]["noleak"]
    run_noleak = float(c_run[mnl].mean())
    d1 = s1[g1best]["noleak"] - s1[runner]["noleak"]     # 320 px 上的差距
    d2 = win_noleak - run_noleak                          # best_res 上的差距
    print("  %-34s @%d %.4f    @%d %.4f" % ("冠军 " + g1best[:28], res_train,
                                            s1[g1best]["noleak"], best_res, win_noleak))
    print("  %-34s @%d %.4f    @%d %.4f" % ("亚军 " + runner[:28], res_train,
                                            s1[runner]["noleak"], best_res, run_noleak))
    print("  冠亚差距：@%d %+.4f   →   @%d %+.4f" % (res_train, d1, best_res, d2))
    ortho = (d1 * d2 > 0) or (abs(d2) < 0.004)
    print()
    if ortho:
        print("  ✅**排序没翻转** → 两维近似正交，阶段1+2 的结论可以叠加使用。")
    else:
        print("  ⛔⛔**排序翻转了** → 两维**不正交**，必须在目标分辨率上重新比裁剪方式，")
        print("     ⛔阶段2 的最优分辨率也只对冠军裁剪成立，不能外推。")
    result["stage3"] = dict(runner=runner, d_at_train=d1, d_at_best=d2, orthogonal=bool(ortho))

    # ── 阶段 4：白捡的两件事（logits 已经存下来了，纯 CPU，零额外推理）────
    K = W.shape[0]
    banner("阶段4／4之一：多尺度组合（同一个模型的多视角平均，⭐仍是单模型，合规）")
    print("  ⭐为什么可能有肉：不同分辨率下 patch 网格切在**不同的位置**，")
    print("     同一只鸟的同一条纹路，320 下可能正好被切在 patch 边界上、448 下就落在中间。")
    print("     → 各档犯的错**不完全重叠**，平均 logits 可以互相补位。")
    print("  ⚠这**不是多模型集成**：权重只有一套，只是同一个模型看了几眼。\n")
    res_keys = [k for k in LG if k and k.startswith("@")]
    combos = []
    for i in range(len(res_keys)):
        for j in range(i + 1, len(res_keys) + 1):
            grp = res_keys[i:j]
            if len(grp) < 2:
                continue
            lg = np.mean([LG[k].astype(np.float32) for k in grp], 0)
            c = (lg.argmax(1) == yv)
            combos.append((float(c[mnl].mean()), grp, c))

    # ⛔多重比较校正（Bonferroni）——这里要连查 %d 个组合，还要**挑最大的那个**。
    #   2×SE 对应单次约 5%% 误判率；跑 m 次，全是噪声也有 1-0.95^m 的概率蹦出"显著"
    #   （m=10 时约 40%%），"挑最大"本身又偏向选中运气最好的那个。
    #   → 把 5%% 的总容错摊到 m 次上：每次只许 0.05/m，阈值 2×SE → z_c×SE。
    m_cmp = max(1, len(combos))
    zc = NormalDist().inv_cdf(1.0 - 0.05 / (2.0 * m_cmp))
    print("  ⛔判据（跑之前写死）：%d 个组合做 Bonferroni 校正 → 阈值 **%.2f×SE**（不是 2×SE）"
          % (m_cmp, zc))
    print("  ⭐预期：各档是**同一个模型**看同一张图，犯的错高度相关 →")
    print("     真实收益应当**很小**（自测里用互相独立的假数据能冲到 1.0000，那是假象）。")
    print("     ⚠若真跑出很大的 Δ，先怀疑代码，别急着高兴。\n")
    single_best = max(((s2[k]["noleak"], k) for k in s2keys))
    print("  %-40s %9s %11s %9s  %s"
          % ("组合", "⭐无泄漏", "配对Δ vs单档", "%.2f×SE" % zc, "判定"))
    print("  " + "-" * 88)
    print("  %-40s %9.4f %11s %9s  单档最好" % (single_best[1][:40], single_best[0], "—", "—"))
    base4 = LG["@%d" % best_res].astype(np.float32).argmax(1) == yv
    best4 = (single_best[0], None, base4)
    for acc, grp, c in sorted(combos, key=lambda x: -x[0])[:6]:
        bb, cc, d, se = paired(base4, c, n)
        sig = abs(d) > zc * se
        tag = ("✅显著更好" if d > 0 else "❌显著更差") if sig else "⚠判为零"
        print("  %-40s %9.4f %+11.4f %9.4f  %s"
              % ("+".join(x.lstrip("@") for x in grp)[:40], acc, d, zc * se, tag))
        if sig and d > 0 and acc > best4[0]:
            best4 = (acc, grp, c)
    print()
    if best4[1] is None:
        print("  ⚠多尺度组合**没有显著收益** → 各档犯的错高度重叠，⛔别为它多花 N 倍推理时间。")
    else:
        print("  ✅⭐**%s 组合显著更好**（无泄漏 %.4f）→ 纯测试期白捡，代价只是多跑几遍推理。"
              % ("+".join(x.lstrip("@") for x in best4[1]), best4[0]))
    result["stage4_multiscale"] = dict(
        best=("+".join(best4[1]) if best4[1] else None), acc=best4[0], single=single_best[0],
        n_compared=m_cmp, z_bonferroni=zc)

    banner("阶段4／4之二：⭐裁决悬案——测试集到底均不均衡")
    print("  逻辑：这里的验证图**已知均衡**，且**已套测试集同款退化**。")
    print("     所以量出的 TV = 『均衡 + 同款域偏移』下**应该**有的 TV。\n")
    print("  %-24s %9s %11s %9s" % ("配置", "TV(原始)", "TV(尺寸修正)", "max类"))
    print("  " + "-" * 60)
    tvs = {}
    for k in ["@%d" % res_train, "@%d" % best_res]:
        if k in LG and LG[k] is not None:
            tv, tvc, sd, mx = dispersion(LG[k], K)
            tvs[k] = tvc
            print("  %-24s %9.4f %11.4f %9d" % (k, tv, tvc, mx))
    ref_tv = 0.0959          # 当前最好模型(榜72.76)在测试集上的实测 TV
    got = tvs.get("@%d" % best_res, tvs.get("@%d" % res_train, 0.0))
    print("\n  参照：当前最好模型在**测试集**上的实测 TV = **%.4f**（榜 72.76）" % ref_tv)
    print("  ⛔判据（跑之前写死）：")
    print("     退化后均衡验证集 TV ≥ 0.8×测试集 → 域偏移足以解释 → **测试集大概率均衡**")
    print("     退化后均衡验证集 TV < 0.5×测试集 → 解释不掉        → **测试集确实不均衡**")
    print("     介于两者之间                      → ⚠仍判不了，别再花时间")
    r = got / ref_tv if ref_tv else 0.0
    print("\n  实测比值 %.4f / %.4f = **%.2f**  →  %s"
          % (got, ref_tv, r,
             "✅**测试集大概率是均衡的**（旧的94σ确认作废）" if r >= 0.8 else
             ("⛔**测试集确实不均衡**（旧结论方向对、但那个94σ的算法仍是错的）" if r < 0.5
              else "⚠**仍判不了**，⛔到此为止别再投入")))
    print("  ⭐⭐但无论哪边：Sinkhorn 作弊用真实配额的 oracle 上界只有 +0.0024（35号脚本），")
    print("     **两边都指向『别做均衡分配』**，所以这个结论不改变任何行动。")
    result["stage4_balance"] = dict(tv_degraded_val=got, tv_test_ref=ref_tv, ratio=r)

    lp = os.path.join(P.CACHE_DIR, "eval_geometry_logits.npz")
    np.savez_compressed(lp, y=yv, noleak=mnl,
                        **{k.replace("/", "_"): v for k, v in LG.items() if v is not None})
    print("\n  ⭐逐样本 logits → %s（下次做组合/阈值分析不用再跑 GPU）" % lp)

    banner("最终结论")
    print("  ⭐推荐评估几何：**%s @ %d px**" % (g1best[:40], best_res))
    print("  ⭐本地无泄漏读数：%.4f（基准 A@%d = %.4f，Δ=%+.4f）"
          % (win_noleak, res_train, s1[s1keys[0]]["noleak"], win_noleak - s1[s1keys[0]]["noleak"]))
    print("  ⚠这是纯测试期改动，倍率接近 1:1，但**仍必须交榜裁决**——本地尺子只信方向。")
    print("  ⛔别和 Stage3 训练的效果混在一次提交里，否则涨了也不知道是谁的功劳。")

    p = os.path.join(P.CACHE_DIR, "eval_geometry.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print("\n  已存 → %s" % p)


if __name__ == "__main__":
    main()
