# -*- coding: utf-8 -*-
"""
把 train/ 和 test/ 的全部图片，用【冻结的】CLIP ViT-B/32 图像编码器编码成 512 维特征，
一次性存成 .npy 缓存。之后训练线性头只读这份缓存，再也不用碰 GPU、不用重复解码 JPEG。

为什么这样做（原理）：
  骨干冻结 + eval() 之后，它就是一个确定性函数——同一张图任何时候输入，输出永远是同一组
  512 个数字。而训练线性头要反复过几十上百个 epoch，如果不缓存，每个 epoch 都要把同样的
  图重新推一遍骨干，算出完全一样的结果。缓存把这份重复劳动砍成只做一次。

⚠官方原文提示：部分图片文件存在截断，在文件系统里可能显示异常，但保证 pillow 能读。
  所以这里必须开 LOAD_TRUNCATED_IMAGES，否则 PIL 会抛异常打断整个提取流程。

用法:
  python scripts/01_extract_features.py                # 抽 train + test
  python scripts/01_extract_features.py --split train  # 只抽 train
  python scripts/01_extract_features.py --limit 2000   # 小样本冒烟测试
  python scripts/01_extract_features.py --selftest     # 只验证"冻结=确定性函数"
"""
import _bootstrap  # noqa: F401
from _bootstrap import banner

import argparse
import json
import os
import time

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from configs import paths as P

# ⚠官方明说数据里有截断图片；不开这个开关 PIL 会在读到截断处直接抛异常
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# ====================================================================== 文件清单
def list_train_files():
    """
    返回 (paths, labels, class_names)。
    顺序严格确定：类别文件夹名排序 → 类别内文件名排序。
    这个顺序就是特征矩阵的行顺序，后面所有脚本都依赖它，不能改。
    """
    class_names = sorted(d for d in os.listdir(P.TRAIN_DIR)
                         if os.path.isdir(os.path.join(P.TRAIN_DIR, d)))
    paths, labels = [], []
    for idx, cname in enumerate(class_names):
        cdir = os.path.join(P.TRAIN_DIR, cname)
        for fn in sorted(os.listdir(cdir)):
            if fn.lower().endswith(IMG_EXT):
                paths.append(os.path.join(cdir, fn))
                labels.append(idx)
    return paths, np.asarray(labels, dtype=np.int16), class_names


def list_test_files():
    """测试集是扁平目录，只有文件名没有标签。同样按文件名排序保证可复现。"""
    return [os.path.join(P.TEST_DIR, fn)
            for fn in sorted(os.listdir(P.TEST_DIR))
            if fn.lower().endswith(IMG_EXT)]


# ====================================================================== Dataset
class ImageListDataset(Dataset):
    """按路径列表读图 + 走 CLIP 官方 preprocess。读失败时返回全零图并标记，保持行对齐。"""

    def __init__(self, file_paths, preprocess):
        self.file_paths = file_paths
        self.preprocess = preprocess

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, i):
        try:
            with Image.open(self.file_paths[i]) as im:
                # convert 必须在 with 内做，否则句柄关了拿不到像素
                tensor = self.preprocess(im.convert("RGB"))
            return tensor, i, 1  # 1 = 读取成功
        except Exception:
            # 极端损坏的图：给一张全零图占位，保证特征矩阵行数与文件列表严格一致。
            # 失败数会在最后汇总打印出来，不静默吞掉。
            return torch.zeros(3, 224, 224), i, 0


# ====================================================================== 骨干
def load_frozen_clip(device):
    """加载 CLIP ViT-B/32，冻结全部视觉参数并切 eval()。返回 (model, preprocess)。"""
    import clip
    model, preprocess = clip.load(P.CLIP_MODEL_NAME, device=device,
                                  download_root=P.CLIP_DOWNLOAD_ROOT)
    model.eval()
    for prm in model.parameters():
        prm.requires_grad = False          # 显式声明意图：骨干一个数字都不训练

    n_visual = sum(p.numel() for p in model.visual.parameters())
    print("  骨干            : CLIP %s (OpenAI 官方权重)" % P.CLIP_MODEL_NAME)
    print("  图像编码器参数量: {:,}".format(n_visual))
    if n_visual != P.CLIP_VISUAL_PARAMS:
        print("  ⚠ 与记录值 {:,} 不符，权重可能不是预期版本！".format(P.CLIP_VISUAL_PARAMS))
    print("  预处理          : Resize(224,bicubic) → CenterCrop(224) → RGB → Norm")
    return model, preprocess


@torch.no_grad()
def encode_all(model, preprocess, file_paths, device, batch_size, num_workers, desc):
    """把一批图片编码成 (N,512) float32 特征矩阵。带实时进度条。"""
    ds = ImageListDataset(file_paths, preprocess)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=(device == "cuda"),
                        drop_last=False)

    feats = np.zeros((len(file_paths), P.FEATURE_DIM), dtype=np.float32)
    failed_idx = []
    model_dtype = next(model.visual.parameters()).dtype
    t0 = time.time()
    t_warm = None       # 首个 batch 之后才开始计真实吞吐(排除 Windows worker 启动开销)
    n_warm = 0

    # 进度条按【图片数】走，而不是 batch 数，剩余时间估计更直观
    bar = tqdm(total=len(file_paths), desc=desc, unit="img",
               unit_scale=True, dynamic_ncols=True)
    for bi, (imgs, idxs, ok_flags) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True).to(model_dtype)  # CLIP 在 cuda 上是 fp16
        out = model.encode_image(imgs)                   # (B,512) fp16
        feats[idxs.numpy()] = out.float().cpu().numpy()  # 存 fp32，避免后续精度纠缠
        bad = idxs[ok_flags == 0]
        if bad.numel():
            failed_idx.extend(bad.tolist())

        bar.update(imgs.shape[0])
        if bi == 0:
            t_warm, n_warm = time.time(), bar.n
        elif t_warm is not None:
            rate = (bar.n - n_warm) / max(time.time() - t_warm, 1e-6)
            bar.set_postfix_str("%.0f img/s  失败%d" % (rate, len(failed_idx)))
    bar.close()

    dt = time.time() - t0
    steady = (len(file_paths) - n_warm) / max(time.time() - t_warm, 1e-6) if t_warm else 0
    print("  完成 %d 张，总耗时 %.1fs（含启动），稳态吞吐 %.0f img/s，读取失败 %d 张"
          % (len(file_paths), dt, steady, len(failed_idx)))
    if failed_idx:
        print("  ⚠ 失败样本索引(前10): %s" % failed_idx[:10])
    return feats, failed_idx


# ====================================================================== 自检
@torch.no_grad()
def selftest(model, preprocess, device):
    """
    验证前面讲的核心前提：冻结 + eval 之后，骨干是确定性函数。
    同一张图连推两次，两次特征必须【逐位完全相同】。这条不成立，缓存策略就不成立。
    """
    banner("自检：冻结骨干是否为确定性函数")
    paths, _, _ = list_train_files()
    sample = paths[:8]
    outs = []
    for _ in range(2):
        batch = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in sample])
        batch = batch.to(device).to(next(model.visual.parameters()).dtype)
        outs.append(model.encode_image(batch).float().cpu().numpy())
    same = np.array_equal(outs[0], outs[1])
    maxdiff = float(np.abs(outs[0] - outs[1]).max())
    print("  两次编码逐位完全相同 : %s   (最大差值 %.3e)" % (same, maxdiff))
    print("  特征形状             :", outs[0].shape)
    print("  结论                 : %s" %
          ("✅ 确定性成立，可以放心缓存" if same else "❌ 不确定，缓存策略需要复查"))
    return same


# ====================================================================== 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test", "both"], default="both")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=2,
                    help="⚠本机实测：8 会炸 WinError 1455（页面文件太小）")
    # ⚠2026-09-22 实测踩坑：Windows 的 DataLoader 是 spawn 模式，每个 worker 都要重新
    #   完整 import 一遍 torch，连带加载 cuFFT/cuDNN 那几百 MB 的 DLL。默认 8 个 worker
    #   同时申请虚拟内存，会把系统提交限制顶爆，报
    #       OSError: [WinError 1455] 页面文件太小，无法完成操作
    #   而且 train 阶段可能侥幸过关、切到 test 重开 worker 时才炸，很有迷惑性。
    #   这台机器上 2 个 worker 已经能跑到 ~850 img/s（瓶颈在 GPU 不在解码），够用。
    #   要提到 8，得先把 Windows 虚拟内存改成"系统托管"并重启。--num-workers 0 是最保险的兜底。
    ap.add_argument("--limit", type=int, default=0,
                    help=">0 时每个 split 只取前 N 张，用于冒烟测试(不会覆盖正式缓存)")
    ap.add_argument("--selftest", action="store_true", help="只跑确定性自检然后退出")
    args = ap.parse_args()

    P.ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    banner("01 特征提取：冻结 CLIP ViT-B/32 → 512 维缓存")
    print("  设备            :", device)
    model, preprocess = load_frozen_clip(device)

    if args.selftest:
        selftest(model, preprocess, device)
        return

    # 冒烟测试模式下不覆盖正式缓存，产物加 _smoke 后缀
    suffix = "_smoke" if args.limit else ""

    def outp(path):
        root, ext = os.path.splitext(path)
        return root + suffix + ext

    manifest = {
        "clip_model": P.CLIP_MODEL_NAME,
        "feature_dim": P.FEATURE_DIM,
        "visual_params": sum(p.numel() for p in model.visual.parameters()),
        "device": device,
        "batch_size": args.batch_size,
        "smoke_limit": args.limit,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # ------------------------------------------------------------ train
    if args.split in ("train", "both"):
        banner("抽取 train 特征")
        paths, labels, class_names = list_train_files()
        if args.limit:
            paths, labels = paths[:args.limit], labels[:args.limit]
        print("  图片 %d 张，类别 %d 个" % (len(paths), len(class_names)))

        feats, failed = encode_all(model, preprocess, paths, device,
                                   args.batch_size, args.num_workers, "train")
        np.save(outp(P.F_TRAIN_FEAT), feats)
        np.save(outp(P.F_TRAIN_LABEL), labels)
        with open(outp(P.F_TRAIN_PATH), "w", encoding="utf-8") as f:
            f.write("\n".join(paths))
        manifest.update(train_count=len(paths), train_failed=len(failed),
                        num_classes=len(class_names), class_names=class_names)
        print("  已存 → %s  (%.1f MB)"
              % (outp(P.F_TRAIN_FEAT), os.path.getsize(outp(P.F_TRAIN_FEAT)) / 1024 ** 2))

    # ------------------------------------------------------------ test
    if args.split in ("test", "both"):
        banner("抽取 test 特征")
        paths = list_test_files()
        if args.limit:
            paths = paths[:args.limit]
        print("  图片 %d 张（无标签）" % len(paths))

        feats, failed = encode_all(model, preprocess, paths, device,
                                   args.batch_size, args.num_workers, "test ")
        np.save(outp(P.F_TEST_FEAT), feats)
        with open(outp(P.F_TEST_PATH), "w", encoding="utf-8") as f:
            f.write("\n".join(paths))
        manifest.update(test_count=len(paths), test_failed=len(failed))
        print("  已存 → %s  (%.1f MB)"
              % (outp(P.F_TEST_FEAT), os.path.getsize(outp(P.F_TEST_FEAT)) / 1024 ** 2))

    with open(outp(P.F_MANIFEST), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    banner("完成")
    print("  缓存目录 :", P.CACHE_DIR)
    print("  下一步   : python scripts/02_train_head.py")


if __name__ == "__main__":   # Windows 下 DataLoader 多进程必须有这个保护
    main()
