# -*- coding: utf-8 -*-
"""零 GPU、秒级：量「文本原型 T」与「微调后模型 L 的训练图特征」的对齐度，决定实验 K 的 λ。
用法：python scripts\\_align_check.py  [cache\\text_protos.npz] [cache\\kd_L_feat.npy]
"""
import os
import sys

import numpy as np

protos = sys.argv[1] if len(sys.argv) > 1 else r"cache\text_protos.npz"
lfeat = sys.argv[2] if len(sys.argv) > 2 else r"cache\kd_L_feat.npy"

z = np.load(protos, allow_pickle=True)
T, M, source, classes = z["T"].astype(np.float32), z["M"].astype(np.float32), z["source"], list(z["classes"])
K = T.shape[0]
Fl = np.load(lfeat).astype(np.float32)
with open(lfeat.replace("_feat.npy", "_paths.txt"), encoding="utf-8") as f:
    lp = [ln.strip() for ln in f if ln.strip()]
yl = np.array([classes.index(os.path.basename(os.path.dirname(p))) for p in lp], dtype=np.int64)
Fl /= np.linalg.norm(Fl, axis=1, keepdims=True).clip(1e-12)

rng = np.random.RandomState(0)
sub = rng.choice(len(yl), min(20000, len(yl)), replace=False)
has_txt = source == "text"

def top1(P, mask=None):
    pr = (Fl[sub] @ P.T).argmax(1)
    ok = pr == yl[sub]
    if mask is None:
        return float(ok.mean()), len(sub)
    keep = mask[yl[sub]]
    return float(ok[keep].mean()), int(keep.sum())

Ml = np.zeros_like(M)
for i in range(K):
    Ml[i] = Fl[yl == i].mean(0)
Ml /= np.linalg.norm(Ml, axis=1, keepdims=True).clip(1e-12)

a_T, _ = top1(T)
a_txt, n_t = top1(np.where(has_txt[:, None], T, 0), has_txt)
a_M, _ = top1(M)
own = float(np.mean([Ml[i] @ T[i] for i in range(K)]))
off = float(np.mean((Ml @ T.T)[~np.eye(K, dtype=bool)]))
own_txt = float(np.mean([Ml[i] @ T[i] for i in range(K) if has_txt[i]])) if has_txt.any() else float("nan")
# 原版 CLIP 空间里的同一组数（参照）
own0 = float(np.mean([M[i] @ T[i] for i in range(K)]))
off0 = float(np.mean((M @ T.T)[~np.eye(K, dtype=bool)]))

print("文本原型 %d/%d 类来自文本，其余为原版 CLIP 图像类均值回退" % (int(has_txt.sum()), K))
print("在【微调后 L】的训练图特征上（抽 %d 张）：" % len(sub))
print("  零样本 top-1：最终原型 T %.4f | 纯文本类 %.4f (%d 张) | 原版图像均值 M %.4f" % (a_T, a_txt, n_t, a_M))
print("  L 类均值 ↔ T：本类余弦 %.4f（纯文本类 %.4f） vs 他类 %.4f" % (own, own_txt, off))
print("参照（原版 CLIP 空间）：M ↔ T 本类 %.4f vs 他类 %.4f" % (own0, off0))
gap = own - off
print("⭐读法：本类-他类余弦差 %.4f；零样本 top-1 %.4f" % (gap, a_T))
if a_T > 0.30:
    print("   ⇒ 空间没转远，λ=0.3 是温和约束")
elif a_T > 0.05:
    print("   ⇒ 部分对齐，λ 取 0.1~0.2")
else:
    print("   ⇒ 空间已转离文本，λ 降到 0.05~0.1，否则是大扰动")
