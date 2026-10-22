# -*- coding: utf-8 -*-
"""
20：在【验证集】上一次性抽一个"多样化视角池"，把 TTA 的配方搜索从 GPU 任务变成离线任务。

═══════════════════════════════════════════════════════════════════════
一、为什么要做视角池，而不是"想试一个配方就抽一次"
═══════════════════════════════════════════════════════════════════════

TTA 的配方 = "用哪几个视角求平均"。可选视角有几十个，组合是指数级的。
如果每换一个配方就重抽一次特征，每次都要过一遍 GPU（分钟级），根本搜不动。

但注意一个关键事实：**求平均是在特征空间里做的，跟骨干无关**。
骨干冻结 ⇒ 每个视角的 512 维特征是确定的、可缓存的。
所以只要把"所有候选视角"各抽一次存下来，之后任何子集的平均值都是
**纯 numpy 的加法**，毫秒级。==> 抽一次(约十几分钟)，换来无限次离线搜索。

⭐这就是本脚本唯一的设计要点：把一次性的 GPU 成本，换成无限次的组合自由度。

═══════════════════════════════════════════════════════════════════════
二、⭐要检验的核心假设："瓶颈是看不清"
═══════════════════════════════════════════════════════════════════════

CLIP ViT-B/32 把图缩到 224x224，再切成 32x32 的格子 —— 一共只有 7x7=49 个 patch。
细粒度物种的判别线索（喙的形状、羽毛纹理、叶脉走向）在这个粒度下很可能**被压没了**。

现有配方是 RandomResizedCrop(scale=0.7~1.0)，实测**视角间一致性 0.987**
（10 号脚本自己打印的数）——5 个视角几乎是同一张图，平均它们等于什么都没做，
却仍然在榜上拿到了 +2.357 分。这说明：**这条路上还有钱没捡**。

于是本脚本引入一族**确定性的中心放大**视角：先把原图中心裁掉 s 比例，再送进 224。
s 越小 = 放得越大 = 每个 patch 覆盖的真实像素越少 = "看得越清"。
如果"看不清"真是瓶颈，这族视角应该明显优于温和随机裁剪；如果不是，就该被否定。
⚠这是一个**可以被数据否定的假设**，不是既定结论。本脚本负责把两边的料都备齐。

═══════════════════════════════════════════════════════════════════════
三、视角池的构成（24 个视角，分 6 组）
═══════════════════════════════════════════════════════════════════════

  G0 原图      orig            CLIP 官方 preprocess：Resize(224,bicubic)+CenterCrop(224)
                               ⚠必须在池子里：实测只用增强视角会变差，原图是"标准视角"锚点
  G1 翻转      orig_flip       原图的确定性水平翻转（不是随机翻转）
  G2 多尺度    cc085/070/055/040   ⭐中心放大，检验核心假设
  G3 尺度翻转  cc*_flip        G2 各自的水平翻转
  G4 四角      corner_tl/tr/bl/br  空间位移视角（从中心 256 方块里切 4 个 224 角）
  G5 温和随机  mild0..4        现配方 scale=(0.7,1.0)，重抽一份用于自检
  G6 强随机    strong0..4      ⭐scale=(0.4,0.75)，更狠的随机裁剪

⚠关于 s 的口径（报告里必须讲清，否则会误读）：
  · 本脚本的 s 是**边长比例**（线性），s=0.4 ⇒ 面积只剩 0.16
  · RandomResizedCrop 的 scale 是**面积比例**，scale=0.4 ⇒ 边长约 0.63
  两者不是一个尺度，对比时要换算。

═══════════════════════════════════════════════════════════════════════
四、两个省算力的小手艺
═══════════════════════════════════════════════════════════════════════

1) **一次解码，多次变换**：Dataset 每个 __getitem__ 只 open/decode 一次 JPEG，
   然后就地产出全部 24 个视角。比"跑 24 遍数据集"省掉 23 次 JPEG 解码。
2) **翻转在张量上做，不重做 resize**：bicubic 重采样对水平镜像是对称的，
   ⇒ 先翻转再 resize 与先 resize 再翻转结果相同（只差浮点舍入）。
   Normalize 是逐通道的，也和空间翻转可交换。
   ⇒ 5 个翻转视角是 torch.flip 白送的，省下 5 次昂贵的 bicubic 缩放。

═══════════════════════════════════════════════════════════════════════
五、⚠Windows 踩坑（都是真踩过的，别改）
═══════════════════════════════════════════════════════════════════════

  · 动态加载 01 号脚本必须在**文件顶层**且先注册 sys.modules —— 见 _load_sibling 注释
  · transform 里**不能出现 lambda**（spawn 序列化不了），RGB 转换放进 Dataset
  · worker 总数 <= 8，否则 [WinError 1455] 页面文件太小
  · 数据里有截断图，必须开 LOAD_TRUNCATED_IMAGES

存 float16 是**无损**的：CLIP 在 cuda 上本来就跑 fp16，encode_image 的输出本身
就是 fp16 可表示的数，01 号存 fp32 只是上转，没有额外信息。存 fp16 省一半磁盘。

用法:
  python scripts/20_build_viewpool.py --limit 200      # 冒烟(不覆盖正式缓存)
  python scripts/20_build_viewpool.py                  # 正式跑，产出 cache/viewpool_val.npy
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
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from configs import paths as P

ImageFile.LOAD_TRUNCATED_IMAGES = True     # 官方说明数据里有截断图，必开

# ⚠必须与 18 号脚本逐字一致，否则本地结果与历史实验不可比
SEED, VAL_FRAC = 42, 0.2


def _load_sibling(fname, modname):
    """
    ⚠⚠这几行必须放在【文件顶层】执行，不能塞进 main() 里。

    Windows 的 DataLoader 走 spawn：子进程会重跑主脚本的**顶层代码**，然后才反序列化
    Dataset 类。而 main() 被 `if __name__=="__main__"` 挡着，子进程的 __name__ 是
    "__mp_main__"，**根本不会执行 main()** → sys.modules 里没这个模块 → worker
    反序列化失败 → ⚠**父进程静默挂起**（不是报错退出，是死等）。2026-09-05 实测卡过 44 分钟。
    """
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(modname, os.path.join(here, fname))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod                 # ⚠必须在 exec_module 之前注册
    spec.loader.exec_module(mod)
    return mod


_ex = _load_sibling("01_extract_features.py", "extract01")

# ---------------------------------------------------------------- 视角定义
CC_SCALES = (0.85, 0.70, 0.55, 0.40)          # 中心放大的**边长**比例
MILD_SCALE = (0.70, 1.00)                     # 现配方（面积比例）
STRONG_SCALE = (0.40, 0.75)                   # 更强随机（面积比例）
N_MILD = N_STRONG = 5
RATIO = (0.85, 1.18)                          # 与 10 号一致，别改

VIEW_NAMES = (
    ["orig", "orig_flip"]
    + ["cc%03d" % int(s * 100) for s in CC_SCALES]
    + ["cc%03d_flip" % int(s * 100) for s in CC_SCALES]
    + ["corner_tl", "corner_tr", "corner_bl", "corner_br"]
    + ["mild%d" % i for i in range(N_MILD)]
    + ["strong%d" % i for i in range(N_STRONG)]
)
NUM_VIEWS = len(VIEW_NAMES)                   # 24

# 分组，只用于报告"这一组和原图有多像"
GROUPS = {
    "G0 原图":        ["orig"],
    "G1 原图翻转":    ["orig_flip"],
    "G2 多尺度放大":  ["cc085", "cc070", "cc055", "cc040"],
    "G3 多尺度+翻转": ["cc085_flip", "cc070_flip", "cc055_flip", "cc040_flip"],
    "G4 四角位移":    ["corner_tl", "corner_tr", "corner_bl", "corner_br"],
    "G5 温和随机(现配方)": ["mild%d" % i for i in range(N_MILD)],
    "G6 强随机":      ["strong%d" % i for i in range(N_STRONG)],
}


def center_crop_ratio(im, s):
    """按**边长比例** s 裁出中心区域。保持宽高比不变 ⇒ 等价于对中心放大 1/s 倍。
    s=1.0 时退化成原图，所以这一族和 G0 是同一个连续家族，可比性最好。"""
    w, h = im.size
    cw, ch = max(1, int(round(w * s))), max(1, int(round(h * s)))
    left, top = (w - cw) // 2, (h - ch) // 2
    return im.crop((left, top, left + cw, top + ch))


class ViewPoolDataset(Dataset):
    """
    一次解码，产出全部 24 个视角，返回 (24,3,224,224)。
    ⚠所有 transform 都是模块级对象/内置类，**没有 lambda** —— spawn 才能序列化。
    """

    def __init__(self, paths, normalize, seed):
        self.paths, self.seed = paths, seed
        bic = T.InterpolationMode.BICUBIC
        # CLIP 官方 preprocess 的等价重建（RGB 转换挪到 __getitem__ 里做）
        self.tf_center = T.Compose([T.Resize(224, interpolation=bic),
                                    T.CenterCrop(224), T.ToTensor(), normalize])
        # 四角视角：先归一到中心 256 方块，再切 4 个 224 的角
        # ⚠不直接在 Resize(256) 后的长条上切角：宽图的左右角会跑到画面之外的内容上，
        #   那已经不是"同一个物体的另一个视角"，而是另一张图了
        self.tf_sq256 = T.Compose([T.Resize(256, interpolation=bic), T.CenterCrop(256)])
        self.tf_norm = T.Compose([T.ToTensor(), normalize])
        self.tf_mild = T.Compose([
            T.RandomResizedCrop(224, scale=MILD_SCALE, ratio=RATIO, interpolation=bic),
            T.RandomHorizontalFlip(), T.ToTensor(), normalize])
        self.tf_strong = T.Compose([
            T.RandomResizedCrop(224, scale=STRONG_SCALE, ratio=RATIO, interpolation=bic),
            T.RandomHorizontalFlip(), T.ToTensor(), normalize])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            with Image.open(self.paths[i]) as im0:
                im = im0.convert("RGB")        # ⚠convert 必须在 with 内，否则句柄已关
        except Exception:
            # 极端损坏图：全零占位，保证行对齐；失败数会汇总打印，不静默吞掉
            return torch.zeros(NUM_VIEWS, 3, 224, 224), i, 0

        # 每张图固定种子 ⇒ 随机视角可复现（worker 之间互不干扰，重跑结果一致）
        torch.manual_seed(self.seed * 1000003 + i)

        v_orig = self.tf_center(im)
        v_cc = [self.tf_center(center_crop_ratio(im, s)) for s in CC_SCALES]
        # ⭐翻转白送：bicubic 对镜像对称、Normalize 逐通道 ⇒ 张量上翻等于图像上翻
        views = [v_orig, torch.flip(v_orig, dims=[2])]
        views += v_cc
        views += [torch.flip(t, dims=[2]) for t in v_cc]

        sq = self.tf_sq256(im)                 # 256x256 PIL
        for top, left in ((0, 0), (0, 32), (32, 0), (32, 32)):
            views.append(self.tf_norm(sq.crop((left, top, left + 224, top + 224))))

        for _ in range(N_MILD):
            views.append(self.tf_mild(im))
        for _ in range(N_STRONG):
            views.append(self.tf_strong(im))

        return torch.stack(views, 0), i, 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=8,
                    help="⚠单位是【图片】不是视角：一批 Dataset 产出 batch*24 个视角。"
                         "这个数只影响 CPU 端解码批量和常驻内存(8 → 约 1.5GB)，"
                         "**不等于送进 GPU 的张数**，后者由 --gpu-chunk 单独控制")
    ap.add_argument("--gpu-chunk", type=int, default=128,
                    help="⚠⚠一次送进 GPU 的【实际张数】上限。batch_size*24 可能有一两百张，"
                         "整批 forward 会顶显存；这里切成不超过 gpu_chunk 张的小块喂。"
                         "8GB 卡上要给排在后面的 LoRA 任务留余量，所以默认 128 而不是 256。")
    ap.add_argument("--num-workers", type=int, default=6,
                    help="⚠总数 <=8，再多会 [WinError 1455] 页面文件太小")
    ap.add_argument("--limit", type=int, default=0, help=">0 时只跑前 N 张验证图（冒烟，不覆盖正式缓存）")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    P.ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    banner("20 构建验证集多样化视角池（%d 个视角）" % NUM_VIEWS)
    print("  设备:", device)

    # ------------------------------------------------------------ 划分（必须与 18 号一致）
    y = np.load(P.F_TRAIN_LABEL).astype(np.int64)
    idx = np.arange(len(y))
    itr, iva = train_test_split(idx, test_size=VAL_FRAC, random_state=SEED, stratify=y)
    paths_all, _, _ = _ex.list_train_files()
    assert len(paths_all) == len(y), "路径数与标签数不一致，特征缓存和数据目录对不上"
    val_paths = [paths_all[i] for i in iva]
    if args.limit:
        val_paths, iva = val_paths[:args.limit], iva[:args.limit]
    N = len(val_paths)
    print("  验证集 %d 张（SEED=%d VAL_FRAC=%.2f，与 18 号同一划分）" % (N, SEED, VAL_FRAC))
    print("  总前向次数 %d = %d 张 x %d 视角" % (N * NUM_VIEWS, N, NUM_VIEWS))

    # ------------------------------------------------------------ 骨干（冻结，不动）
    model, preprocess = _ex.load_frozen_clip(device)
    normalize = [t for t in preprocess.transforms if isinstance(t, T.Normalize)][0]
    print("  归一化参数取自 CLIP 官方 preprocess:", normalize.mean, normalize.std)
    print("  ⛔全程不做颜色抖动——颜色本身是区分近缘物种的关键特征，改了等于制造错标")

    ds = ViewPoolDataset(val_paths, normalize, args.seed)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=False, drop_last=False)

    out = np.zeros((NUM_VIEWS, N, P.FEATURE_DIM), dtype=np.float16)
    model_dtype = next(model.visual.parameters()).dtype
    nfail, t0 = 0, time.time()
    bar = tqdm(total=N, desc="视角池", unit="img", unit_scale=True, dynamic_ncols=True)
    with torch.no_grad():
        for imgs, idxs, ok in loader:
            b = imgs.shape[0]
            flat = imgs.view(b * NUM_VIEWS, 3, 224, 224)
            # ⚠分块喂 GPU：b*24 张一次性 forward 会顶显存，且这台机器上还有别的任务排队。
            #   拆成 <=gpu_chunk 张的小块，峰值显存只由 gpu_chunk 决定，与 batch_size 解耦。
            parts = []
            for st in range(0, flat.shape[0], args.gpu_chunk):
                chunk = flat[st:st + args.gpu_chunk].to(device, non_blocking=True).to(model_dtype)
                parts.append(model.encode_image(chunk).float().cpu().numpy())
            f = np.concatenate(parts, 0)                           # (b*24,512)
            f = f.reshape(b, NUM_VIEWS, P.FEATURE_DIM).transpose(1, 0, 2)
            out[:, idxs.numpy(), :] = f.astype(np.float16)
            nfail += int((ok == 0).sum())
            bar.update(b)
            if bar.n > args.batch_size:
                bar.set_postfix_str("%.0f img/s 失败%d" % (bar.n / (time.time() - t0), nfail))
    bar.close()
    print("  完成 %.1f 分钟，读取失败 %d 张" % ((time.time() - t0) / 60, nfail))

    # ⭐后面的自检和相似度统计全是 numpy，不需要 GPU。这台 8GB 卡后面还有 LoRA 任务排队，
    #   所以这里立刻把骨干和缓存块还回去，别白占着显存等脚本跑完。
    if device == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1024 ** 3
        del model, loader
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print("  显存峰值 %.2f GB，已释放（当前占用 %.3f GB）"
              % (peak, torch.cuda.memory_allocated() / 1024 ** 3))

    # ------------------------------------------------------------ 自检 + 相似度体检
    banner("自检：重建的 orig 视角是否等于 01 号存下来的原图特征")
    X0 = np.load(P.F_TRAIN_FEAT).astype(np.float32)[iva]
    X0 /= np.maximum(np.linalg.norm(X0, axis=1, keepdims=True), 1e-12)
    V = out.astype(np.float32)
    V /= np.maximum(np.linalg.norm(V, axis=2, keepdims=True), 1e-12)
    cos0 = float((V[0] * X0).sum(1).mean())
    print("  orig vs cache/train_features.npy : 平均余弦 %.6f" % cos0)
    print("  → 应当 >0.9999。低了说明我重建的 preprocess 和 CLIP 官方那套对不上，"
          "后面所有结论都不可信")

    banner("视角强度体检：每组视角与【原图】的平均余弦相似度")
    print("  判读：>0.99 = 增强等于没做；<0.85 = 可能裁太狠把语义裁没了")
    name2i = {n: i for i, n in enumerate(VIEW_NAMES)}
    sims = {}
    for gname, members in GROUPS.items():
        per = [float((V[name2i[m]] * V[0]).sum(1).mean()) for m in members]
        gm = V[[name2i[m] for m in members]].mean(0)
        gm /= np.maximum(np.linalg.norm(gm, axis=1, keepdims=True), 1e-12)
        sims[gname] = dict(per_view=dict(zip(members, per)),
                           mean=float(np.mean(per)),
                           group_mean_vs_orig=float((gm * V[0]).sum(1).mean()))
        print("  %-22s 组均 %.4f   组内平均后 vs 原图 %.4f   逐视角 %s"
              % (gname, sims[gname]["mean"], sims[gname]["group_mean_vs_orig"],
                 " ".join("%s=%.4f" % (m, p) for m, p in zip(members, per))))

    # ------------------------------------------------------------ 落盘
    suffix = "_smoke" if args.limit else ""
    fout = os.path.join(P.CACHE_DIR, "viewpool_val%s.npy" % suffix)
    np.save(fout, out)
    meta = dict(view_names=VIEW_NAMES, groups=GROUPS, n=N, seed=SEED, val_frac=VAL_FRAC,
                aug_seed=args.seed, cc_scales=list(CC_SCALES),
                cc_scale_meaning="边长比例(线性)，面积比例 = s^2",
                mild_scale=list(MILD_SCALE), strong_scale=list(STRONG_SCALE),
                ratio=list(RATIO), dtype="float16(与CLIP fp16输出等价，无损)",
                failed=nfail, orig_vs_cache_cos=cos0, group_sims=sims,
                elapsed_min=(time.time() - t0) / 60,
                created=time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(os.path.join(P.CACHE_DIR, "viewpool_val%s_meta.json" % suffix),
              "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    banner("完成")
    print("  %s  形状 %s  (%.0f MB)" % (fout, out.shape, os.path.getsize(fout) / 1024 ** 2))
    print("  下一步: python scripts/21_search_tta.py")


if __name__ == "__main__":     # ⚠Windows 下 DataLoader 多进程必须有这层保护
    main()
