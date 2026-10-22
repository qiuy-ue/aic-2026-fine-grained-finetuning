# -*- coding: utf-8 -*-
"""69 · 训练集自蒸馏的教师软标签：用一个 ckpt 给【全部训练图】出 softmax 分布。

═══════════════════════════════════════════════════════════════════════
⭐ 合规边界（2026-09-18，用户裁定 + 规则原文）
═══════════════════════════════════════════════════════════════════════
  十(二)：「…或结合教师模型、自训练等方法增强鲁棒性」—— 教师模型被点名鼓励。
  本脚本**只读训练图**（`_ex.list_train_files()`），⛔一张测试图都不碰；
  产出只在训练时当软目标用，提交的仍是单一学生模型（五、4 单一模型/单一推理流程）。

═══════════════════════════════════════════════════════════════════════
⚠ 为什么默认建议 T=2（在 28 号的 --distill-temp 里给）
═══════════════════════════════════════════════════════════════════════
  教师（L@ep12）是 --full-data 训的，这 103218 张它**全部见过**，in-sample 分布匹配 0.97 ⇒
  它对训练图极自信，T=1 的软标签 ≈ one-hot，蒸馏会退化成 label smoothing（我们已经有 0.1 了）。
  本脚本把 T=1 / T=2 / T=4 下的「最大概率均值」「熵」都打印出来，⭐让温度的选择有数可依。
  存盘的是 T=1 的原始概率（float16）；温度在 28 号加载时再施加，这样一份文件配任意 T。

产物（与 28 号 --distill <prefix> 的约定一致）：
  <prefix>_prob.npy    (N, K) float16   softmax 概率，行序 = <prefix>_paths.txt
  <prefix>_paths.txt   N 行训练图路径
  <prefix>_meta.json   ckpt / 视角 / 分辩率 / 与真标签的一致率 / 各温度统计

用法：
  python scripts\\69_teacher_trainset.py --ckpt runs\\20260914_125846_v2_448_cos_dp020_re25_ep12\\ckpt_ep12.pt `
      --res 448 --out cache\\kd_L
  --limit 500 冒烟。视角：orig + orig_flip（2 视角，特征平均后过头）。@448 约 30 分钟。
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
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from configs import paths as P


def _load_sibling(fname, modname):
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(modname, os.path.join(here, fname))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


pred26 = _load_sibling("26_predict_finetuned.py", "pred26")
_ex = pred26._ex
_ft = pred26._ft


class TrainViewDS(Dataset):
    """每张训练图产出 orig 与 orig_flip 两个视角（确定性，无随机）。⚠无 lambda（Windows spawn）。"""

    def __init__(self, paths, tf, res):
        self.p, self.tf, self.res = paths, tf, res

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        try:
            with Image.open(self.p[i]) as im:
                x = self.tf(im.convert("RGB"))
            return torch.stack([x, torch.flip(x, dims=[2])], 0), i, 1
        except Exception:
            z = torch.zeros(3, self.res, self.res)
            return torch.stack([z, z], 0), i, 0


def stats_at(prob, T):
    p = prob.astype(np.float32) ** (1.0 / T)
    p /= p.sum(1, keepdims=True).clip(1e-8)
    ent = float(-(p * np.log(p.clip(1e-12))).sum(1).mean())
    return dict(T=T, mean_maxprob=float(p.max(1).mean()), mean_entropy=ent,
                mean_second=float(np.sort(p, 1)[:, -2].mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--res", type=int, default=448)
    ap.add_argument("--out", type=str, required=True, help="输出前缀，如 cache\\kd_L")
    ap.add_argument("--batch-size", type=int, default=32, help="单位=图片；实际前向张数 ×2")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help=">0 只跑前 N 张（冒烟）")
    args = ap.parse_args()
    if args.res % 32:
        ap.error("--res 必须是 32 的倍数")

    banner("69 训练集教师软标签  @%dpx  视角 orig+flip" % args.res)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ck_path = args.ckpt if os.path.isabs(args.ckpt) else os.path.join(P.PROJECT_ROOT, args.ckpt)
    tmp = ck_path + ".readcopy"
    shutil.copy2(ck_path, tmp)
    ck = torch.load(tmp, map_location="cpu", weights_only=False)
    os.remove(tmp)
    tr = ck.get("args", {})
    res_train = tr.get("resolution", 224)
    cosine, subc = bool(tr.get("cosine_head", False)), int(tr.get("subcenters", 0))
    print("  教师 ckpt  %s  ep%s  训练@%dpx  余弦头=%s" %
          (os.path.basename(os.path.dirname(ck_path)), ck.get("epoch"), res_train, cosine))

    import clip
    model, _ = clip.load(P.CLIP_MODEL_NAME, device="cpu", download_root=P.CLIP_DOWNLOAD_ROOT)
    model.float()
    _ft.interpolate_pos_embed(model, res_train)
    model.visual.load_state_dict(ck["visual"], strict=True)
    if args.res != res_train:
        _ft.interpolate_pos_embed(model, args.res)
        print("  ⭐推理分辨率 %d ≠ 训练 %d → 两步插值，口径同 26/58" % (args.res, res_train))
    visual = model.visual.to(device).eval()

    W = ck["W"].numpy().astype(np.float32)
    b = ck["b"].numpy().astype(np.float32)
    s = float(ck["s"]) if "s" in ck else None

    paths, labels, _ = _ex.list_train_files()
    labels = np.asarray(labels).astype(np.int64)
    if args.limit:
        paths, labels = paths[:args.limit], labels[:args.limit]
    N, K = len(paths), W.shape[0]
    print("  训练图 %d 张 × 2 视角 = %d 次前向   ⛔不读任何测试图" % (N, 2 * N))

    _, eval_tf = _ft.build_transforms(args.res, 0.5)
    dl = DataLoader(TrainViewDS(paths, eval_tf, args.res), batch_size=args.batch_size,
                    shuffle=False, num_workers=args.num_workers, pin_memory=True)

    prob = np.zeros((N, K), dtype=np.float16)
    # ⭐顺手存教师（L）在训练图上的 512 维特征（两视角平均、L2 归一）：
    #   用途 = 零 GPU 量「文本原型 vs 微调后特征空间」的对齐度，决定文本锚点的 λ（实验 K）。
    feat = np.zeros((N, W.shape[1]), dtype=np.float16)
    feats_done, failed, t0 = 0, 0, time.time()
    with torch.no_grad():
        for imgs, idx, ok in dl:
            B = imgs.shape[0]
            flat = imgs.view(B * 2, 3, args.res, args.res).to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
                f = visual(flat).float()
            f = f / f.norm(dim=1, keepdim=True).clamp_min(1e-12)
            f = f.view(B, 2, -1).mean(1)                              # 两视角特征平均
            f = (f / f.norm(dim=1, keepdim=True).clamp_min(1e-12)).cpu().numpy()
            feat[idx.numpy()] = f.astype(np.float16)
            lg = pred26.apply_head(f, W, b, s, cosine, subc)          # 与出包完全同一口径
            lg = lg - lg.max(1, keepdims=True)
            p = np.exp(lg)
            p /= p.sum(1, keepdims=True)
            prob[idx.numpy()] = p.astype(np.float16)
            failed += int((ok == 0).sum())
            feats_done += B
            if feats_done % (args.batch_size * 100) < args.batch_size:
                el = (time.time() - t0) / 60
                print("  %6d/%d  %.1f 分钟  预计总计 %.0f 分钟" % (feats_done, N, el, el / max(feats_done, 1) * N),
                      flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    out = args.out + ("_smoke" if args.limit else "")
    np.save(out + "_prob.npy", prob)
    np.save(out + "_feat.npy", feat)
    with open(out + "_paths.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(paths) + "\n")

    agree = float((prob.astype(np.float32).argmax(1) == labels).mean())
    st = [stats_at(prob, T) for T in (1.0, 2.0, 4.0)]
    meta = dict(ckpt=ck_path, epoch=str(ck.get("epoch")), res=args.res, views=["orig", "orig_flip"],
                n=N, K=K, failed=failed, agree_with_label=agree, temps=st,
                elapsed_min=(time.time() - t0) / 60, created=time.strftime("%Y%m%d_%H%M%S"))
    with open(out + "_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    banner("教师体检")
    print("  argmax 与真标签一致：%.4f   ⚠这是 in-sample（教师见过这些图），只读方向" % agree)
    print("  %-4s %-14s %-12s %-14s" % ("T", "max概率均值", "熵均值", "第二候选均值"))
    for x in st:
        print("  %-4g %-14.4f %-12.4f %-14.4f" % (x["T"], x["mean_maxprob"], x["mean_entropy"], x["mean_second"]))
    print("  ⭐读法：T=1 若 max概率均值 >0.9 ⇒ 软标签≈one-hot，蒸馏会退化成 label smoothing；"
          "选一个让第二候选有分量（~0.05 以上）的 T 给 28 号的 --distill-temp")
    print("  失败 %d 张，耗时 %.1f 分钟" % (failed, (time.time() - t0) / 60))
    print("  ⭐已写出 %s_prob.npy (%.0f MB) / _paths.txt / _meta.json"
          % (out, os.path.getsize(out + "_prob.npy") / 1e6))
    print("  下一步：65_chain_exp.py --tag K --resume-from <L@ep12> --epochs 8 --aug-geo --distill %s --distill-temp 2" % out)


if __name__ == "__main__":
    main()
