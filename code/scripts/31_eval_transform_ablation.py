# -*- coding: utf-8 -*-
"""
评估变换消融：现在的 Resize+CenterCrop 到底丢了多少分。

⚠发现的不一致：
  训练 RandomResizedCrop(320, ratio=0.75~1.33) —— 随机裁一块，压成方形，**有形变**
  评估 Resize(短边320)+CenterCrop(320)        —— 保长宽比，**只取中央**
  测试图 500×375 → Resize后 427×320 → 中央裁320 → **丢掉 25% 的画面宽度**
物种不在正中央就被裁没了。本脚本对同一个模型试几种评估变换，只改这一个变量。
"""
import _bootstrap  # noqa: F401
from _bootstrap import banner
import argparse, importlib.util, os, sys
import numpy as np, torch
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm import tqdm
from configs import paths as P
ImageFile.LOAD_TRUNCATED_IMAGES = True

def _ld(f, m):
    here = os.path.dirname(os.path.abspath(__file__))
    sp = importlib.util.spec_from_file_location(m, os.path.join(here, f))
    mo = importlib.util.module_from_spec(sp); sys.modules[m] = mo; sp.loader.exec_module(mo)
    return mo
_ex = _ld("01_extract_features.py", "extract01"); _ft = _ld("24_finetune_full.py", "finetune24")

NORM = T.Normalize((0.48145466,0.4578275,0.40821073),(0.26862954,0.26130258,0.27577711))
BIC = T.InterpolationMode.BICUBIC

class PadSquare:
    """把图补成正方形（用边缘反射填充），保住整张画面又不形变。"""
    def __call__(self, im):
        w, h = im.size
        s = max(w, h)
        c = Image.new("RGB", (s, s), (124, 116, 104))   # 用 CLIP 均值色填充
        c.paste(im, ((s - w) // 2, (s - h) // 2))
        return c

def variants(res):
    return {
        "A 现状 Resize短边+中央裁": T.Compose([T.Resize(res, interpolation=BIC), T.CenterCrop(res), T.ToTensor(), NORM]),
        "B 直接压成方形(全画面)":   T.Compose([T.Resize((res, res), interpolation=BIC), T.ToTensor(), NORM]),
        "C 补边成方形(全画面)":     T.Compose([PadSquare(), T.Resize((res, res), interpolation=BIC), T.ToTensor(), NORM]),
        "D Resize1.14倍+中央裁":    T.Compose([T.Resize(int(res*1.14), interpolation=BIC), T.CenterCrop(res), T.ToTensor(), NORM]),
    }

class DS(Dataset):
    def __init__(s, p, tf, r): s.p, s.tf, s.r = p, tf, r
    def __len__(s): return len(s.p)
    def __getitem__(s, i):
        try:
            with Image.open(s.p[i]) as im: return s.tf(im.convert("RGB")), i
        except Exception: return torch.zeros(3, s.r, s.r), i

@torch.no_grad()
def ev(vis, W, b, tf, paths, yv, dev, res, nm):
    ld = DataLoader(DS(paths, tf, res), batch_size=32, shuffle=False, num_workers=4)
    vis.eval(); c = np.zeros(len(paths), bool)
    bar = tqdm(total=len(paths), desc=nm[:12], unit="img", unit_scale=True, leave=False)
    Wt = torch.tensor(W, device=dev); bt = torch.tensor(b, device=dev)
    for im, ix in ld:
        im = im.to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=(dev == "cuda")): f = vis(im)
        f = f.float(); f = f / f.norm(dim=1, keepdim=True).clamp_min(1e-12)
        ixn = ix.numpy(); c[ixn] = ((f @ Wt.T + bt).argmax(1).cpu().numpy() == yv[ixn])
        bar.update(im.shape[0])
    bar.close(); return c.astype(np.float64)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--limit", type=int, default=6000); a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False); res = ck["args"].get("resolution", 320)
    banner("31 评估变换消融 @%dpx  ckpt=%s" % (res, os.path.basename(os.path.dirname(a.ckpt))))
    paths, labels, _ = _ex.list_train_files(); y = labels.astype(np.int64)
    _, iva = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42, stratify=y)
    wz = np.load(os.path.join(P.CACHE_DIR, "val_weights.npz")); wfull = wz["weight"]
    lk = np.load(os.path.join(P.CACHE_DIR, "leakage_scores.npz")); msfull = lk["val_maxsim"]
    if a.limit: sub = np.arange(len(iva))[:a.limit]; iva2 = iva[sub]; w = wfull[sub]; ms = msfull[sub]
    else: iva2 = iva; w = wfull; ms = msfull
    va = [paths[i] for i in iva2]; yv = y[iva2]; mnl = ms < 0.90
    import clip
    model, _ = clip.load(P.CLIP_MODEL_NAME, device="cpu", download_root=P.CLIP_DOWNLOAD_ROOT)
    model.float(); _ft.interpolate_pos_embed(model, res); vis = model.visual
    vis.load_state_dict(ck["visual"], strict=True); vis = vis.to(dev)
    W = ck["W"].numpy().astype(np.float32); b = ck["b"].numpy().astype(np.float32)
    print("  验证子集 %d 张\n" % len(iva2))
    print("  %-28s %10s %10s %10s" % ("评估变换", "原始等权", "分布匹配", "无泄漏"))
    print("  " + "-" * 62)
    base = None
    for nm, tf in variants(res).items():
        c = ev(vis, W, b, tf, va, yv, dev, res, nm)
        r = (c.mean(), (c*w).sum()/w.sum(), c[mnl].mean())
        if base is None: base = r
        print("  %-28s %10.4f %10.4f %10.4f  %s" % (nm, r[0], r[1], r[2],
              "" if nm.startswith("A") else "%+.4f" % (r[1]-base[1])))
    print("\n  ⭐若 B/C 明显优于 A → 中央裁剪确实在丢画面，评估变换要换，且训练也该跟着对齐。")


if __name__ == "__main__":
    main()
