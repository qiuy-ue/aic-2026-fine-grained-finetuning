# -*- coding: utf-8 -*-
"""
用**完整微调后的骨干**出提交文件。先出无 TTA 版（快），再出 TTA 版。

═══ 为什么要出两版 ═══

无 TTA 版**隔离了微调这一个变量**——它的榜分直接回答"完整微调值多少分"，
是校准本地尺子在这个全新 regime 下的唯一真值点。
TTA 版是在它之上叠加已确认有效的 +2.357 那套配方，通常分更高。
⭐两版都出，成本只差 5 个视角的编码时间，但归因干净得多。

═══ ⚠骨干一变，所有缓存全部作废 ═══

cache/test_features*.npy 都是**原版冻结 CLIP** 抽的，与微调后的骨干无关，
必须用新骨干重抽。这就是"碰骨干贵一百倍"的具体体现。

═══ TTA 配方严格照抄上榜那版（61.357）═══

    最终特征 = L2( mean( L2(原图), L2(视角1..5) ) )
⚠必须**带原图**（实测只用增强视角反而变差，原图是锚点）；
⚠必须**逐视角先 L2、平均后再 L2**；
⛔增强不做颜色抖动（颜色是区分近缘物种的关键特征）。
增强用 `10_extract_augmented.build_aug`，与线上那版逐字一致——
**只让骨干这一个变量变**，别顺手换配方。

用法:
  python scripts/26_predict_finetuned.py                 # 自动取最新 ckpt_best.pt
  python scripts/26_predict_finetuned.py --no-tta        # 只出无 TTA 版
  python scripts/26_predict_finetuned.py --ckpt runs/xxx/ckpt_best.pt
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
import zipfile

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm import tqdm

from configs import paths as P

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _load_sibling(fname, modname):
    """⚠sys.modules 注册必须在文件顶层，否则 Windows spawn 的 worker 静默挂起。"""
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(modname, os.path.join(here, fname))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


_ex = _load_sibling("01_extract_features.py", "extract01")
_aug = _load_sibling("10_extract_augmented.py", "augment10")
_ft = _load_sibling("24_finetune_full.py", "finetune24")


class TestDS(Dataset):
    def __init__(self, paths, tf, res):
        self.p, self.tf, self.res = paths, tf, res

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        try:
            with Image.open(self.p[i]) as im:
                return self.tf(im.convert("RGB")), i
        except Exception:
            return torch.zeros(3, self.res, self.res), i


@torch.no_grad()
def encode(visual, tf, paths, device, bs, nw, res, desc):
    """用当前骨干编码，返回逐行 L2 归一化后的特征。"""
    ld = DataLoader(TestDS(paths, tf, res), batch_size=bs, shuffle=False,
                    num_workers=nw, pin_memory=False)
    out = np.zeros((len(paths), P.FEATURE_DIM), dtype=np.float32)
    visual.eval()
    bar = tqdm(total=len(paths), desc=desc, unit="img", unit_scale=True, dynamic_ncols=True)
    for im, ix in ld:
        im = im.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
            f = visual(im)
        out[ix.numpy()] = f.float().cpu().numpy()
        bar.update(im.shape[0])
    bar.close()
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


def apply_head(F, W, b, s=None, cosine=False, subcenters=0):
    """从头的参数算 logits。cosine 头（③a）：W 逐行 L2 归一 × 学习标量 s；否则线性 + b。
    子中心头（③b）：W (K,N,d)，每类 N 个子中心取 max cosine。
    F 需已 L2 归一（encode 返回的就是归一化特征）。"""
    if subcenters > 1:
        Kc, Nc, d = W.shape
        Wn = W / np.maximum(np.linalg.norm(W, axis=-1, keepdims=True), 1e-12)  # (K,N,d)
        sims = F @ Wn.reshape(Kc * Nc, d).T                                  # (B, K*N)
        return float(s) * sims.reshape(-1, Kc, Nc).max(axis=2)               # (B, K)
    if cosine:
        Wn = W / np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-12)
        return float(s) * (F @ Wn.T)
    return F @ W.T + b


def write_submission(out_dir, tag, paths, pred, class_names, K):
    os.makedirs(out_dir, exist_ok=True)
    csv = os.path.join(out_dir, P.SUBMISSION_FILENAME)
    with open(csv, "w", encoding="utf-8", newline="\n") as f:
        f.write("".join("%s,%s\n" % (os.path.basename(p), class_names[i])
                        for p, i in zip(paths, pred)))
    zp = csv.replace(".csv", ".zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(csv, arcname=P.SUBMISSION_FILENAME)
    u, c = np.unique(pred, return_counts=True)
    print("  [%s] 覆盖 %d/%d 类  每类 min=%d max=%d 均值=%.1f"
          % (tag, len(u), K, c.min(), c.max(), c.mean()))
    print("  [%s] ZIP → %s  (%.1f MB)" % (tag, zp, os.path.getsize(zp) / 1024 ** 2))
    return csv, zp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=None, help="留空自动取最新 *_ft* 的 ckpt_best.pt")
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--k", type=int, default=5, help="TTA 增强视角数（配方同上榜那版）")
    ap.add_argument("--batch-size", type=int, default=48,
                    help="⚠训练可能还在跑、显存共享，别开太大")
    ap.add_argument("--num-workers", type=int, default=2,
                    help="⚠训练在跑时它有 4+2 个 worker，总数超 8 会报 WinError 1455")
    ap.add_argument("--eval-res", type=int, default=0,
                    help="⭐推理分辨率，留 0 = 用 ckpt 的训练分辨率。"
                         "设成别的值走两步插值（先插到训练网格 strict 载入，再插到评估网格），"
                         "⚠口径与 34_eval_geometry 完全一致，否则榜分对不上本地读数。"
                         "必须是 32 的倍数（ViT-B/32 的 patch 边长）。")
    args = ap.parse_args()
    if args.eval_res and args.eval_res % 32:
        ap.error("--eval-res 必须是 32 的倍数（ViT-B/32 的 patch 是 32×32 像素），"
                 "给了 %d" % args.eval_res)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------------------------------------------------------------- 找 ckpt
    if args.ckpt:
        ckp = args.ckpt
    else:
        cands = sorted(d for d in os.listdir(P.RUNS_DIR)
                       if "_ft" in d and os.path.exists(
                           os.path.join(P.RUNS_DIR, d, "ckpt_best.pt")))
        assert cands, "runs/ 下没有微调产物，先跑 24_finetune_full.py"
        ckp = os.path.join(P.RUNS_DIR, cands[-1], "ckpt_best.pt")

    # ⚠训练进程可能正在覆写这个文件 → 先复制一份再读，避免读到写了一半的
    tmp = ckp + ".readcopy"
    shutil.copy2(ckp, tmp)
    ck = torch.load(tmp, map_location="cpu", weights_only=False)
    os.remove(tmp)
    tr = ck["args"]
    res_train = tr.get("resolution", 224)
    res = args.eval_res or res_train

    banner("26 用微调后的骨干出提交  ckpt=%s  epoch=%s"
           % (os.path.basename(os.path.dirname(ckp)), ck.get("epoch")))
    m = ck.get("metrics", {})
    print("  该 ckpt 的本地指标：分布匹配 %.4f  无泄漏 %.4f  干净60%% %.4f"
          % (m.get("matched", float("nan")), m.get("noleak", float("nan")),
             m.get("clean60", float("nan"))))
    print("  训练配置：@%dpx  lr骨干=%g  loss=%s  第%s轮"
          % (res_train, tr.get("lr_backbone"), tr.get("loss"), ck.get("epoch")))
    if res != res_train:
        print("  ⭐推理分辨率被改成 %dpx（训练是 %dpx）→ 走两步插值，"
              "口径同 34_eval_geometry" % (res, res_train))

    # ---------------------------------------------------------------- 模型
    import clip
    model, _ = clip.load(P.CLIP_MODEL_NAME, device="cpu",
                         download_root=P.CLIP_DOWNLOAD_ROOT)
    model.float()
    _ft.interpolate_pos_embed(model, res_train)    # ⚠必须先插值到 ckpt 的分辨率再 load
    visual = model.visual
    missing, unexpected = visual.load_state_dict(ck["visual"], strict=True)
    if res != res_train:
        # ⭐第二步插值：载入之后再插到评估网格。
        #   ⛔不能反过来（先插到评估网格再 load）——那样 load 的 strict 检查会因
        #   token 数不符而失败，只能 strict=False 混过去，等于悄悄丢掉位置编码。
        #   interpolate_pos_embed 是从模型**当前**网格读起点的，所以可以连调两次。
        _ft.interpolate_pos_embed(model, res)
    visual = model.visual
    visual = visual.to(device)
    W = ck["W"].numpy().astype(np.float32)
    b = ck["b"].numpy().astype(np.float32)
    cosine = bool(tr.get("cosine_head", False))
    subcenters = int(tr.get("subcenters", 0))
    s = float(ck["s"]) if (cosine or subcenters > 1) else None
    head_tag = (("（子中心 ③b ×%d）" % subcenters) if subcenters > 1
                else ("（余弦 ③a）" if cosine else ""))
    print("  骨干权重已载入（strict=True 通过），头 W%s%s"
          % (tuple(W.shape), head_tag))

    paths = _ex.list_test_files()          # 与 01 同一套排序，行序严格对齐
    cn = json.load(open(P.F_MANIFEST, encoding="utf-8"))["class_names"]
    K = W.shape[0]
    print("  测试集 %d 张，类别编号取自 manifest (%s ~ %s)" % (len(paths), cn[0], cn[-1]))

    _, eval_tf = _ft.build_transforms(res, 0.5)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    t0 = time.time()

    # ---------------------------------------------------------------- ① 无 TTA
    banner("① 无 TTA 版（隔离'完整微调'这一个变量，是校准尺子的真值点）")
    F0 = encode(visual, eval_tf, paths, device, args.batch_size,
                args.num_workers, res, "原图  ")
    pred0 = np.argmax(apply_head(F0, W, b, s, cosine, subcenters), 1)
    d0 = os.path.join(P.RUNS_DIR, "%s_ft%d_ep%s_notta" % (stamp, res, ck.get("epoch")))
    write_submission(d0, "无TTA", paths, pred0, cn, K)
    # ⭐把特征和头存下来：抽一次测试集要好几分钟，而"先验校正"这类事后调整
    #   只是在 logits 上做加减，不需要再碰 GPU。存下来就能反复试不同后处理。
    np.savez(os.path.join(d0, "test_feat_head.npz"), F=F0, W=W, b=b,
             **({"s": s, "cosine": cosine, "subcenters": subcenters}
                if (cosine or subcenters > 1) else {}))
    print("  耗时 %.1f 分钟" % ((time.time() - t0) / 60))

    if args.no_tta:
        return

    # ---------------------------------------------------------------- ② TTA
    banner("② TTA 版（配方严格照抄 61.357 那版，只让骨干这一个变量变）")
    norm = [t for t in eval_tf.transforms if isinstance(t, T.Normalize)][0]
    aug = _aug.build_aug(norm)
    if res != 224:      # build_aug 写死 224，换分辨率时要跟着改
        aug.transforms[0] = T.RandomResizedCrop(
            res, scale=(0.7, 1.0), ratio=(0.85, 1.18),
            interpolation=T.InterpolationMode.BICUBIC)
    print("  RandomResizedCrop(%d, scale=0.7~1.0)+HFlip，⛔无颜色抖动" % res)
    views = [F0]
    for v in range(args.k):
        torch.manual_seed(v)               # 与 10 号同一套种子，可复现
        views.append(encode(visual, aug, paths, device, args.batch_size,
                            args.num_workers, res, "视角%d" % (v + 1)))
    X = np.mean(views, 0)
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    pred1 = np.argmax(apply_head(X, W, b, s, cosine, subcenters), 1)
    d1 = os.path.join(P.RUNS_DIR, "%s_ft%d_ep%s_tta" % (stamp, res, ck.get("epoch")))
    write_submission(d1, "TTA", paths, pred1, cn, K)
    np.savez(os.path.join(d1, "test_feat_head.npz"), F=X, W=W, b=b,
             **({"s": s, "cosine": cosine, "subcenters": subcenters}
                if (cosine or subcenters > 1) else {}))   # 同上，供事后后处理
    print("  TTA 改变了 %d/%d 张预测 (%.1f%%)"
          % ((pred0 != pred1).sum(), len(pred0), 100 * (pred0 != pred1).mean()))

    # 与历史最优比
    prev = os.path.join(P.RUNS_DIR, "20260904_232902_sce_tta", P.SUBMISSION_FILENAME)
    if os.path.exists(prev):
        old = [l.split(",")[1].strip() for l in open(prev, encoding="utf-8")]
        for nm, pr in (("无TTA", pred0), ("TTA", pred1)):
            d = sum(1 for x, i in zip(old, pr) if x != cn[i])
            print("  %s 版 vs 旧的 61.357：改变 %d/%d (%.1f%%)"
                  % (nm, d, len(pr), 100 * d / len(pr)))

    banner("完成，总耗时 %.1f 分钟" % ((time.time() - t0) / 60))
    print("  ⭐建议先交【TTA 版】拿分；若还有提交名额，把【无TTA 版】也交了——")
    print("     那能把'微调'和'TTA'两部分的贡献拆干净，对复赛很值钱。")


if __name__ == "__main__":
    main()
