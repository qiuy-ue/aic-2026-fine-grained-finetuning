# -*- coding: utf-8 -*-
"""
B6：为每张训练图缓存 K 个【随机增强视角】的特征，把当初为了速度放弃的数据增强找回来。

═══ 为什么原来没有增强 ═══

数据增强 = 每轮训练时对同一张图做不同的随机裁剪/翻转，让模型每次看到略有差异的版本，
变相扩充数据量。但我们把特征缓存成了【每张图一个 512 维向量】——
一个固定的向量没法表达"这张图的多个随机版本"，所以缓存和增强天生互斥。
这是当初拿"训练快 100 倍"换来的代价，不是白赚的。

═══ 折中办法 ═══

不存 1 个视角，存 K 个。训练时每轮为每张图随机抽一个视角用。
  K=5 时：103218×5×512×4B ≈ 1.0 GB，内存装得下，抽取耗时约 5×6=30 分钟(一次性)。
这样既保住"训练时不用碰 GPU"的速度优势，又拿回了增强。
⚠不完全等价于真增强——真增强每轮都是全新的随机版本，这里只是在 K 个固定版本里轮换。
  K 越大越接近，但存储和抽取成本线性上升。

═══ ⭐为什么必须在 LoRA 之前做 ═══

一旦上 LoRA，骨干每轮都要重推原图，"顺手做个随机裁剪"就不额外花钱了——
增强会自然而然地被加进去。后果是：**LoRA 涨的分里混着增强的功劳，你分不清是谁的**。
所以要先单独把增强的收益量出来，后面才能干净归因。这是实验设计问题，不是效率问题。

═══ 增强强度的取舍 ═══

CLIP 官方预处理是 Resize(224)+CenterCrop(224)，很"老实"。我们加的增强必须温和：
  · RandomResizedCrop scale=(0.7,1.0) —— 不敢用常见的 0.08 下限，
    因为细粒度物种识别靠的往往是局部细节(喙的形状、羽毛纹理)，裁太狠会把判别区域裁没
  · RandomHorizontalFlip —— 动植物左右翻转后仍是同一物种，安全
  · ⛔不做颜色抖动 —— 颜色本身就是区分近缘物种的关键特征，改了等于制造错标

用法:
  python scripts/10_extract_augmented.py            # K=5
  python scripts/10_extract_augmented.py --k 3      # 更省空间
"""
import _bootstrap  # noqa: F401
from _bootstrap import banner

import argparse
import importlib.util
import json
import os
import time

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm import tqdm

from configs import paths as P

ImageFile.LOAD_TRUNCATED_IMAGES = True      # 官方说明数据里有截断图，必开

_spec = importlib.util.spec_from_file_location(
    "extract", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "01_extract_features.py"))
_ex = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ex)


def build_aug(normalize):
    """
    温和增强。复用 CLIP 官方的归一化参数，保证数值分布与预训练一致。
    ⚠不要在这里放 lambda：Windows 的 DataLoader 多进程走 spawn，会把整个 transform
      序列化传给子进程，而 lambda 不可序列化 → AttributeError。RGB 转换挪进 Dataset。
    """
    return T.Compose([
        T.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.85, 1.18),
                            interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize,
    ])


class AugDataset(Dataset):
    def __init__(self, paths, tf):
        self.paths, self.tf = paths, tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            with Image.open(self.paths[i]) as im:
                return self.tf(im.convert("RGB")), i, 1   # 先转 RGB 再增强
        except Exception:
            return torch.zeros(3, 224, 224), i, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5, help="每张图缓存几个增强视角")
    ap.add_argument("--split", choices=["train", "test"], default="train",
                    help="⭐实测：训练时用增强【无效】(骨干冻结、学不了不变性)，"
                         "但预测时把多视角特征平均【有效】(纯统计的降噪效应，四档全优)。"
                         "所以 test 这一份才是真正要用的。")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    P.ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    banner("10 缓存 K=%d 个增强视角的特征" % args.k)

    model, preprocess = _ex.load_frozen_clip(device)
    # 从官方 preprocess 里取出 Normalize，保证与预训练时的数值分布完全一致
    normalize = [t for t in preprocess.transforms if isinstance(t, T.Normalize)][0]
    print("  归一化参数直接取自 CLIP 官方 preprocess:", normalize.mean, normalize.std)
    aug = build_aug(normalize)
    print("  增强: RandomResizedCrop(224, scale=0.7~1.0) + 水平翻转")
    print("  ⛔不做颜色抖动——颜色是区分近缘物种的关键特征，改了等于制造错标")

    if args.split == "train":
        paths, _, _ = _ex.list_train_files()
    else:
        paths = _ex.list_test_files()      # 与 01 用同一套排序，保证行顺序严格对齐
    N = len(paths)
    print("  数据集: %s" % args.split)
    size_gb = N * args.k * P.FEATURE_DIM * 4 / 1024 ** 3
    print("  图片 %d 张 × %d 视角 = %d 次前向，预计缓存 %.2f GB"
          % (N, args.k, N * args.k, size_gb))

    out = np.zeros((args.k, N, P.FEATURE_DIM), dtype=np.float32)
    model_dtype = next(model.visual.parameters()).dtype
    total_failed = 0
    t_all = time.time()

    for v in range(args.k):
        torch.manual_seed(args.seed + v)      # 每个视角用不同随机种子，保证可复现
        ds = AugDataset(paths, aug)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=(device == "cuda"))
        t0, nfail = time.time(), 0
        bar = tqdm(total=N, desc="视角 %d/%d" % (v + 1, args.k), unit="img",
                   unit_scale=True, dynamic_ncols=True)
        with torch.no_grad():
            for imgs, idxs, ok in loader:
                imgs = imgs.to(device, non_blocking=True).to(model_dtype)
                f = model.encode_image(imgs)
                out[v][idxs.numpy()] = f.float().cpu().numpy()
                nfail += int((ok == 0).sum())
                bar.update(imgs.shape[0])
        bar.close()
        total_failed += nfail
        print("  视角 %d 完成 %.1fs，读取失败 %d" % (v + 1, time.time() - t0, nfail))

    fout = os.path.join(P.CACHE_DIR, "%s_features_aug.npy" % args.split)
    np.save(fout, out)
    with open(os.path.join(P.CACHE_DIR, "aug_meta_%s.json" % args.split),
              "w", encoding="utf-8") as f:
        json.dump({"k": args.k, "n": N, "seed": args.seed, "split": args.split,
                   "aug": "RandomResizedCrop(224,scale=0.7-1.0,ratio=0.85-1.18)+HFlip",
                   "no_color_jitter_reason": "颜色是区分近缘物种的关键特征",
                   "failed": total_failed, "shape": list(out.shape),
                   "elapsed_sec": time.time() - t_all,
                   "created": time.strftime("%Y-%m-%d %H:%M:%S")}, f,
                  ensure_ascii=False, indent=2)

    banner("完成")
    print("  形状 %s → %s (%.2f GB)" % (out.shape, fout, os.path.getsize(fout) / 1024 ** 3))
    print("  总耗时 %.1f 分钟，读取失败 %d 张" % ((time.time() - t_all) / 60, total_failed))
    # 视角间差异有多大：太小说明增强等于没做，太大说明裁得过狠把语义裁没了
    s = out[:, :2000] / np.maximum(np.linalg.norm(out[:, :2000], axis=2, keepdims=True), 1e-12)
    cos = np.einsum("ijk,jk->ij", s, s.mean(0) / np.maximum(
        np.linalg.norm(s.mean(0), axis=1, keepdims=True), 1e-12))
    print("  视角间一致性: 各视角与其均值的余弦相似度 %.4f" % cos.mean())
    print("    → 接近 1.00 说明增强太弱等于没做；低于 0.85 说明裁太狠可能破坏语义")
    print("\n  下一步: python scripts/11_train_with_aug.py")


if __name__ == "__main__":
    main()
