# -*- coding: utf-8 -*-
"""零 GPU 只读体检：为"训练侧现成余量"这条线取硬数据。
⛔不写任何成绩包、不碰 runs/ 下已有文件、不用 CUDA。
"""
import _bootstrap  # noqa: F401
import hashlib
import importlib.util
import json
import os
import sys

import numpy as np
import torch

from configs import paths as P

ROOT = P.PROJECT_ROOT
L12 = os.path.join(ROOT, "runs", "20260914_125846_v2_448_cos_dp020_re25_ep12")


def _load_sibling(fname, modname):
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(modname, os.path.join(here, fname))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


pred26 = _load_sibling("26_predict_finetuned.py", "pred26")
_ft = pred26._ft


def line(t):
    print("\n" + "=" * 72 + "\n  " + t + "\n" + "=" * 72, flush=True)


# ─────────────────────────────────────────── ① ckpt 元信息（含 SWA）
line("(1) ckpt 元信息：used_ema / args / s")
for name in ("ckpt_ep09.pt", "ckpt_ep10.pt", "ckpt_ep11.pt", "ckpt_ep12.pt", "ckpt_swa0912.pt"):
    p = os.path.join(L12, name)
    if not os.path.exists(p):
        print("  缺 %s" % name)
        continue
    ck = torch.load(p, map_location="cpu", weights_only=False)
    print("  %-18s epoch=%-14s used_ema=%-6s s=%.4f  nkeys=%d  swa_from=%s"
          % (name, ck.get("epoch"), ck.get("used_ema"), float(ck.get("s", 0)),
             len(ck["visual"]), ck.get("swa_from", "-")))
    del ck

# ─────────────────────────────────────────── ② 原版 CLIP vs L@ep12 的权重漂移
line("(2) WiSE-FT 可行性：θ_CLIP(插到448) vs θ_L@ep12 的漂移")
import clip  # noqa: E402
model, _ = clip.load(P.CLIP_MODEL_NAME, device="cpu", download_root=P.CLIP_DOWNLOAD_ROOT)
model.float()
g0, g1 = _ft.interpolate_pos_embed(model, 448)
print("  位置编码 %d×%d → %d×%d" % (g0, g0, g1, g1))
sd0 = {k: v.detach().clone() for k, v in model.visual.state_dict().items()}

ck = torch.load(os.path.join(L12, "ckpt_ep12.pt"), map_location="cpu", weights_only=False)
sd1 = ck["visual"]
k0, k1 = set(sd0), set(sd1)
print("  键数 CLIP=%d  ckpt=%d   只在CLIP=%s   只在ckpt=%s"
      % (len(k0), len(k1), sorted(k0 - k1)[:5], sorted(k1 - k0)[:5]))
shape_bad = [k for k in (k0 & k1) if tuple(sd0[k].shape) != tuple(sd1[k].shape)]
print("  形状不一致的键：%s" % (shape_bad[:5] if shape_bad else "无 → ⭐可直接做逐元素插值"))

num = den = 0.0
per = {}
for k in sorted(k0 & k1):
    a, b = sd0[k].float(), sd1[k].float()
    if not a.is_floating_point():
        continue
    d = float(((b - a) ** 2).sum())
    n = float((a ** 2).sum())
    num += d
    den += n
    tag = k.split(".")[0]
    if k.startswith("transformer.resblocks."):
        tag = "block%02d" % int(k.split(".")[2])
    p_ = per.setdefault(tag, [0.0, 0.0])
    p_[0] += d
    p_[1] += n
print("  ⭐全局 ‖θ_ft − θ_CLIP‖ / ‖θ_CLIP‖ = %.5f" % (num / max(den, 1e-12)) ** 0.5)
for tag in sorted(per):
    d, n = per[tag]
    print("     %-14s %.5f" % (tag, (d / max(n, 1e-12)) ** 0.5))

# 对照：SWA 离 ep12 有多远（台账记 0.00056，复核）
sw = os.path.join(L12, "ckpt_swa0912.pt")
if os.path.exists(sw):
    cks = torch.load(sw, map_location="cpu", weights_only=False)
    d2 = sum(float(((cks["visual"][k].float() - sd1[k].float()) ** 2).sum())
             for k in sd1 if sd1[k].is_floating_point())
    n2 = sum(float((sd1[k].float() ** 2).sum()) for k in sd1 if sd1[k].is_floating_point())
    print("  对照 ‖θ_SWA − θ_ep12‖ / ‖θ_ep12‖ = %.6f" % (d2 / max(n2, 1e-12)) ** 0.5)
    # SWA 相对于"从 CLIP 到 ep12 的总位移"占多大比例
    print("  ⭐SWA 的扰动 = 微调总位移的 %.3f%%"
          % (100.0 * ((d2 / max(n2, 1e-12)) ** 0.5) / ((num / max(den, 1e-12)) ** 0.5)))
    del cks
del ck, sd1, sd0, model

# ─────────────────────────────────────────── ③ 头的 θ0 候选
line("(3) WiSE-FT 的头 θ0 候选：cache/head_loss_sce.npz")
hz = np.load(os.path.join(P.CACHE_DIR, "head_loss_sce.npz"))
for k in hz.files:
    print("  %-10s shape=%s dtype=%s" % (k, hz[k].shape, hz[k].dtype))
print("  ⚠ ckpt 用的是余弦头（W 逐行归一 + 标量 s、b 弃用）")

# ─────────────────────────────────────────── ④ 两份 544 池子是否重复劳动
line("(4) 两份 pool544 是不是重复劳动（流式 md5，内存 ~0）")


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest().upper()


for d in ("20260916_210154_pool544_ep12", "20260916_212040_pool544_ep12"):
    p = os.path.join(ROOT, "runs", d, "pool.npz")
    print("  %-30s %d bytes  md5=%s" % (d, os.path.getsize(p), md5(p)))

# ─────────────────────────────────────────── ⑤ 步数/EMA 窗口
line("(5) EMA 有效窗口 vs 一轮的步数")
h = json.load(open(os.path.join(L12, "history.json"), encoding="utf-8"))
a = h["args"]
print("  batch=%d  epochs=%d  ema_decay=%s" % (a["batch_size"], a["epochs"], a["ema_decay"]))
print("  ⚠ 步数从日志读更准；这里只给公式：窗口 = 1/(1-decay) = %.0f 步" % (1 / (1 - a["ema_decay"])))
print("\n  DONE")
