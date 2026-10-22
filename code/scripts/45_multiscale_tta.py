# -*- coding: utf-8 -*-
"""② 多尺度 TTA（交接文档 6.1 候选②，预注册实验 H）

同一份权重分别在 448 与 512 两个分辨率各做一次完整 TTA（原图 + 5 视角，
逐视角 L2 → 平均 → 再 L2），然后把两组 TTA 特征**再平均** → argmax 出包。

为什么合规：
- 从头到尾只有一个模型、一份权重（E，degrade 0.875 @448），只是让它换距离看了两次；
- 512 分支直接复用 carrier 已出包 test_feat_head.npz 里的 F(=X512)，只新编码 448 分支；
- 不是多模型集成（1.2 节不违规）。

唯一变量 = 推理侧"加一个 448 分支 + 特征平均"，其余（权重/头/TTA 配方/类别顺序）一字不改。
用法:
  python scripts/45_multiscale_tta.py
  python scripts/45_multiscale_tta.py --ckpt runs/xxx/ckpt_ep02.pt \
      --x512 runs/yyy_ft512_ep2_tta/test_feat_head.npz --res-lo 448 --res-hi 512
"""
import _bootstrap  # noqa: F401
from _bootstrap import banner

import argparse
import importlib.util
import json
import os
import shutil
import sys
import time

import numpy as np
import torch
from torchvision import transforms as T

from configs import paths as P

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _load_sibling(fname, modname):
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(modname, os.path.join(here, fname))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


pred26 = _load_sibling("26_predict_finetuned.py", "pred26")


def _l2(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def encode_tta(visual, paths, device, bs, nw, res, k, eval_tf):
    """在单个分辨率 res 上跑完整 TTA（原图 + k 个增强视角），
    返回 L2 归一化的最终 TTA 特征。配方与 26 号逐字一致。"""
    F0 = pred26.encode(visual, eval_tf, paths, device, bs, nw, res, "原图@%d  " % res)
    norm = [t for t in eval_tf.transforms if isinstance(t, T.Normalize)][0]
    aug = pred26._aug.build_aug(norm)
    if res != 224:  # build_aug 写死 224，换分辨率时跟着改
        aug.transforms[0] = T.RandomResizedCrop(
            res, scale=(0.7, 1.0), ratio=(0.85, 1.18),
            interpolation=T.InterpolationMode.BICUBIC)
    print("  TTA@%d：RandomResizedCrop(%d, scale=0.7~1.0)+HFlip，⛔无颜色抖动" % (res, res))
    views = [F0]
    for v in range(k):
        torch.manual_seed(v)  # 与 10/26 同一套种子，可复现
        views.append(pred26.encode(visual, aug, paths, device, bs, nw, res,
                                   "视角%d@%d" % (v + 1, res)))
    return _l2(np.mean(views, 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str,
                    default=r"D:\aic_data\AIC\AIC2026_9_4\runs\20260910_175802_v2_448\ckpt_ep02.pt",
                    help="carrier 权重（E, degrade 0.875 @448）")
    ap.add_argument("--x512", type=str,
                    default=r"D:\aic_data\AIC\AIC2026_9_4\runs\20260910_190506_ft512_ep2_tta\test_feat_head.npz",
                    help="carrier 已出包的 512-TTA 特征文件（F=X512）")
    ap.add_argument("--res-lo", type=int, default=448)
    ap.add_argument("--res-hi", type=int, default=512)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------------------------------------------------------------- 载入 carrier
    tmp = args.ckpt + ".readcopy"
    shutil.copy2(args.ckpt, tmp)
    ck = torch.load(tmp, map_location="cpu", weights_only=False)
    os.remove(tmp)
    res_train = ck["args"].get("resolution", 224)

    banner("45 多尺度 TTA  ckpt=%s  epoch=%s"
           % (os.path.basename(os.path.dirname(args.ckpt)), ck.get("epoch")))
    print("  carrier 训练 @%dpx（预计 448）" % res_train)

    import clip
    model, _ = clip.load(P.CLIP_MODEL_NAME, device="cpu",
                         download_root=P.CLIP_DOWNLOAD_ROOT)
    model.float()
    pred26._ft.interpolate_pos_embed(model, res_train)  # 先插到 ckpt 网格再 load
    visual = model.visual
    missing, unexpected = visual.load_state_dict(ck["visual"], strict=True)
    print("  骨干权重已载入（strict=True）")
    assert res_train == args.res_lo, ("carrier 训练分辨率 %d != --res-lo %d，"
                                      "确认 ckpt 网格与低分辨率一致" % (res_train, args.res_lo))
    visual = visual.to(device)
    W = ck["W"].numpy().astype(np.float32)
    b = ck["b"].numpy().astype(np.float32)
    cosine = bool(ck["args"].get("cosine_head", False))
    subcenters = int(ck["args"].get("subcenters", 0))
    s = float(ck["s"]) if (cosine or subcenters > 1) else None

    paths = pred26._ex.list_test_files()
    cn = json.load(open(P.F_MANIFEST, encoding="utf-8"))["class_names"]
    K = W.shape[0]
    print("  测试集 %d 张，类别 %d" % (len(paths), K))

    _, eval_tf = pred26._ft.build_transforms(args.res_lo, 0.5)
    t0 = time.time()

    # ---------------------------------------------------------------- 448 分支（新编码）
    banner("① 448 分支（原图 + %d 视角 TTA）" % args.k)
    X448 = encode_tta(visual, paths, device, args.batch_size, args.num_workers,
                      args.res_lo, args.k, eval_tf)
    print("  448 TTA 特征 %s" % (X448.shape,))

    # ---------------------------------------------------------------- 512 分支（复用已出包）
    banner("② 512 分支（复用 carrier 已出包 F=X512）")
    z = np.load(args.x512)
    X512 = z["F"].astype(np.float32)
    assert len(X512) == len(paths), "512 特征行数 %d != 测试集 %d" % (len(X512), len(paths))
    print("  512 TTA 特征 %s（来自 %s）" % (X512.shape, os.path.basename(args.x512)))

    # ---------------------------------------------------------------- 多尺度平均
    banner("③ 多尺度平均 → argmax")
    X_ms = _l2((X448 + X512) / 2.0)
    pred_ms = np.argmax(pred26.apply_head(X_ms, W, b, s, cosine, subcenters), 1)

    pred_lo = np.argmax(pred26.apply_head(X448, W, b, s, cosine, subcenters), 1)
    pred_hi = np.argmax(pred26.apply_head(X512, W, b, s, cosine, subcenters), 1)
    print("  MS vs 448-only 分歧: %d/%d (%.1f%%)"
          % ((pred_ms != pred_lo).sum(), len(paths), 100 * (pred_ms != pred_lo).mean()))
    print("  MS vs 512-only 分歧: %d/%d (%.1f%%)"
          % ((pred_ms != pred_hi).sum(), len(paths), 100 * (pred_ms != pred_hi).mean()))
    print("  448-only vs 512-only 分歧: %d/%d (%.1f%%)"
          % ((pred_lo != pred_hi).sum(), len(paths), 100 * (pred_lo != pred_hi).mean()))

    stamp = time.strftime("%Y%m%d_%H%M%S")
    d = os.path.join(P.RUNS_DIR, "%s_ms%d_%d_tta" % (stamp, args.res_lo, args.res_hi))
    pred26.write_submission(d, "多尺度TTA", paths, pred_ms, cn, K)
    np.savez(os.path.join(d, "test_feat_head.npz"), F=X_ms, W=W, b=b,
             **({"s": s, "cosine": cosine} if cosine else {}))

    banner("完成，总耗时 %.1f 分钟" % ((time.time() - t0) / 60))
    print("  输出 ZIP -> %s\\pred_results.zip" % d)


if __name__ == "__main__":
    main()