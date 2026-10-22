# -*- coding: utf-8 -*-
"""53：定位-放大的**前置检查**——模型自己到底能不能找到主体？

═══════════════════════════════════════════════════════════════════════════
为什么要先做这一步（⛔不先做就是在赌）
═══════════════════════════════════════════════════════════════════════════
「定位到主体再放大喂进去」这个方案，全部价值建立在一个**没验证过的前提**上：
                  模型能把主体框出来。
框不准的话，裁剪会把鸟裁掉一半，比不裁还差。所以先花半小时量这件事，
⛔别直接写完整流程去烧 GPU 出包。（交接文档 4.6：先做 oracle/前提诊断，
再决定要不要投入。）

═══════════════════════════════════════════════════════════════════════════
用什么当热力图：patch token × 分类头的类别方向
═══════════════════════════════════════════════════════════════════════════
CLIP 的 visual 只把 **CLS** 那一路送进 ln_post/proj，patch token 被丢掉了。
但 patch token 走**同一个** ln_post 和 proj，就得到每个格子自己的 512 维嵌入。
拿它和**我们训出来的分类头**里预测类的那一行做余弦，就是一张类别相关的热力图。

⭐ 这么选有三个好处：
  1. 不需要梯度、不需要给 MultiheadAttention 挂 need_weights 钩子
  2. 用的是**我们自己训的头**，不是通用显著性——它找的是"哪块像第 k 类"
  3. 零额外参数、零额外数据 → 合规性与 TTA 同级（一个模型一份权重）

═══════════════════════════════════════════════════════════════════════════
判据（跑之前写死，交接文档 4.8）
═══════════════════════════════════════════════════════════════════════════
主判据 = **隐含放大倍率的中位数**（= 1/sqrt(框面积占比)）：
   < 1.20  → ⛔ 主体本来就占满画面，裁剪买不到 token 密度，**放弃这条路**
   1.20~1.40 → ⚠ 边缘，要看分歧率再定
   ≥ 1.40  → ✅ 有肉。作为参照：448→512 只有 1.15 倍就换到了 +0.49 分
副判据 = 全图预测 vs 放大后预测的**分歧率**：
   < 2%   → 改动太小，就算方向对也测不出
   5~20%  → 合理（TTA 内部分歧同量级）
   > 40%  → ⛔ 多半是裁坏了，不是"看得更清"
"""
import _bootstrap  # noqa: F401
from _bootstrap import banner

import argparse
import importlib.util
import os

import numpy as np
import torch
from PIL import Image, ImageFile

from configs import paths as P

ImageFile.LOAD_TRUNCATED_IMAGES = True
HERE = os.path.dirname(os.path.abspath(__file__))


def load_src(fname, modname):
    spec = importlib.util.spec_from_file_location(modname, os.path.join(HERE, fname))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_ex = load_src("01_extract_features.py", "extract01")
_ft = load_src("24_finetune_full.py", "finetune24")
_pred = load_src("26_predict_finetuned.py", "pred26")


# ────────────────────────────────────────────────────────── 前向到 patch 级
@torch.no_grad()
def forward_tokens(visual, x):
    """照抄 CLIP VisionTransformer.forward，只把最后一步改成"CLS 和 patch 都过 ln_post/proj"。
    返回 (cls_embed (N,512), patch_embed (N,L,512))。"""
    x = visual.conv1(x)                              # (N, width, g, g)
    g = x.shape[-1]
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)     # (N, g*g, width)
    cls = visual.class_embedding.to(x.dtype) + torch.zeros(
        x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
    x = torch.cat([cls, x], dim=1)
    x = x + visual.positional_embedding.to(x.dtype)
    x = visual.ln_pre(x)
    x = x.permute(1, 0, 2)                           # NLD -> LND
    x = visual.transformer(x)
    x = x.permute(1, 0, 2)                           # LND -> NLD
    x = visual.ln_post(x)                            # ⭐对**所有** token 做，不只 CLS
    if visual.proj is not None:
        x = x @ visual.proj
    return x[:, 0, :], x[:, 1:, :], g


@torch.no_grad()
def forward_tokens_maskclip(visual, x):
    """MaskCLIP 式取 patch 特征：**最后一层只走 value 投影，跳过注意力混合**。

    ⭐为什么要这么改（这是 09-13 对照组逼出来的）：
      CLIP 预训练只监督 CLS 那一路，最后一层的注意力把所有位置的信息**混进 CLS**，
      于是 patch token 自己反而不再对齐分类空间 —— 实测拿它做热力图，
      **随机方向的集中度(0.367) 比真类方向(0.329) 还高**，等于零信息。
      绕开最后那次注意力混合、直接取 value 投影，每个位置的语义就保住了。
    """
    x = visual.conv1(x)
    g = x.shape[-1]
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
    cls = visual.class_embedding.to(x.dtype) + torch.zeros(
        x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
    x = torch.cat([cls, x], dim=1)
    x = x + visual.positional_embedding.to(x.dtype)
    x = visual.ln_pre(x)
    x = x.permute(1, 0, 2)                           # NLD -> LND
    blocks = visual.transformer.resblocks
    for blk in blocks[:-1]:
        x = blk(x)
    last = blocks[-1]
    y = last.ln_1(x)
    d = y.shape[-1]
    Wi, bi = last.attn.in_proj_weight, last.attn.in_proj_bias
    v = y @ Wi[2 * d:, :].T + bi[2 * d:]             # ⭐只要 value，不算注意力权重
    x = x + last.attn.out_proj(v)
    x = x + last.mlp(last.ln_2(x))
    x = x.permute(1, 0, 2)                           # LND -> NLD
    x = visual.ln_post(x)
    if visual.proj is not None:
        x = x @ visual.proj
    return x[:, 1:, :], g


def l2(a, axis=-1):
    return a / np.clip(np.linalg.norm(a, axis=axis, keepdims=True), 1e-12, None)


# ────────────────────────────────────────────────────────── 坐标还原
def crop_geometry(w0, h0, res):
    """Resize(短边→res) + CenterCrop(res) 的正向几何，用于把格子坐标映回原图。"""
    scale = res / min(w0, h0)
    w1, h1 = round(w0 * scale), round(h0 * scale)
    left, top = (w1 - res) // 2, (h1 - res) // 2
    return scale, left, top


def heat_to_box(heat, res, keep):
    """热力图 → res 坐标系里的框。keep = 保留最高的这部分热量质量。"""
    g = heat.shape[0]
    flat = heat.flatten()
    order = np.argsort(flat)[::-1]
    csum = np.cumsum(flat[order] - flat.min())
    total = max(csum[-1], 1e-12)
    n_keep = max(int(np.searchsorted(csum, keep * total)) + 1, 1)
    mask = np.zeros(g * g, bool)
    mask[order[:n_keep]] = True
    mask = mask.reshape(g, g)
    ys, xs = np.where(mask)
    cell = res / g
    return (xs.min() * cell, ys.min() * cell, (xs.max() + 1) * cell, (ys.max() + 1) * cell)


def box_in_original(box, w0, h0, res, margin):
    """res 坐标系的框 → 原图坐标系，加边距并夹到图内。"""
    scale, left, top = crop_geometry(w0, h0, res)
    x0 = (box[0] + left) / scale
    y0 = (box[1] + top) / scale
    x1 = (box[2] + left) / scale
    y1 = (box[3] + top) / scale
    mw, mh = (x1 - x0) * margin, (y1 - y0) * margin
    x0, y0 = max(x0 - mw, 0), max(y0 - mh, 0)
    x1, y1 = min(x1 + mw, w0), min(y1 + mh, h0)
    # 防退化：至少留 20% 边长
    if x1 - x0 < 0.2 * w0:
        c = (x0 + x1) / 2
        x0, x1 = max(c - 0.1 * w0, 0), min(c + 0.1 * w0, w0)
    if y1 - y0 < 0.2 * h0:
        c = (y0 + y1) / 2
        y0, y1 = max(c - 0.1 * h0, 0), min(c + 0.1 * h0, h0)
    return (x0, y0, x1, y1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str,
                    default=os.path.join(P.RUNS_DIR, "20260911_080944_v2_448_cos", "ckpt_ep02.pt"))
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--keep", type=float, default=0.60, help="框住多少比例的热量质量")
    ap.add_argument("--margin", type=float, default=0.12, help="框外再放这么多边距")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--control", type=str, default="none",
                    choices=["none", "random", "otherclass"],
                    help="⭐平凡解释对照组（4.6 节强制）。热力图改用随机方向/别的类的方向，"
                         "若集中度与真类一样高 → 说明这张图根本不含类别信息，仪器是瞎的")
    ap.add_argument("--heat", type=str, default="clsproj", choices=["clsproj", "maskclip"],
                    help="clsproj=末层 patch token 直接过 ln_post/proj（09-13 实测：零信息，随机对照更高）；"
                         "maskclip=末层只走 value 投影，跳过注意力混合")
    args = ap.parse_args()

    banner("53 定位前置检查：模型能不能自己找到主体")
    dev = args.device if torch.cuda.is_available() else "cpu"

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    tr = ck.get("args", {})
    res_train = int(tr.get("resolution", 448))
    import clip
    model, _ = clip.load(P.CLIP_MODEL_NAME, device="cpu", download_root=P.CLIP_DOWNLOAD_ROOT)
    model.float()
    _ft.interpolate_pos_embed(model, res_train)
    visual = model.visual
    visual.load_state_dict(ck["visual"], strict=True)
    if args.res != res_train:
        _ft.interpolate_pos_embed(model, args.res)
    visual = model.visual.to(dev).eval()

    W = ck["W"].numpy().astype(np.float32)
    b = ck["b"].numpy().astype(np.float32)
    cosine = bool(tr.get("cosine_head", False))
    sub = int(tr.get("subcenters", 0))
    s = float(ck["s"]) if (cosine or sub > 1) else None
    print("  ckpt=%s  训练@%dpx  推理@%dpx  头 W%s"
          % (os.path.basename(os.path.dirname(args.ckpt)), res_train, args.res, tuple(W.shape)))

    # 类别方向（用于把 patch 打成"像不像第 k 类"）
    Wd = W if W.ndim == 2 else W.reshape(W.shape[0], -1, W.shape[-1]).mean(1)
    Wn = l2(Wd.astype(np.float32))

    _, eval_tf = _ft.build_transforms(args.res, 0.5)
    paths = _ex.list_test_files()
    rng = np.random.RandomState(0)
    sel = rng.choice(len(paths), size=min(args.n, len(paths)), replace=False)
    sel_paths = [paths[i] for i in sel]
    print("  测试集抽样 %d 张（seed=0）" % len(sel_paths))

    zooms, areas, conc, boxes, preds_full, preds_zoom = [], [], [], [], [], []
    conf_full, conf_zoom = [], []

    for i0 in range(0, len(sel_paths), args.batch_size):
        chunk = sel_paths[i0:i0 + args.batch_size]
        ims = [Image.open(p).convert("RGB") for p in chunk]
        xb = torch.stack([eval_tf(im) for im in ims]).to(dev)
        cls_e, pat_e, g = forward_tokens(visual, xb)
        cls_np = l2(cls_e.cpu().numpy().astype(np.float32))
        if args.heat == "maskclip":
            # ⚠预测仍然走**正常**前向（cls_np），只有热力图换成 MaskCLIP 路径——
            #   绕开注意力的那条路算出来的 CLS 不是真 CLS，⛔不能拿去做分类。
            pat_e, g = forward_tokens_maskclip(visual, xb)
        pat_np = l2(pat_e.cpu().numpy().astype(np.float32))

        lg = _pred.apply_head(cls_np, W, b, s, cosine, sub)
        k = np.argmax(lg, 1)
        p = np.exp(lg - lg.max(1, keepdims=True))
        conf_full += list((p / p.sum(1, keepdims=True)).max(1))
        preds_full += list(k)

        # ── 每张图的类别热力图 → 框 → 原图裁剪
        crops = []
        for j, im in enumerate(ims):
            # ⭐对照组：把"预测类的方向"换掉。若集中度不变 → 热力图不含类别信息
            if args.control == "random":
                v = rng.randn(Wn.shape[1]).astype(np.float32)
                d = v / np.linalg.norm(v)
            elif args.control == "otherclass":
                kk = (k[j] + 1 + rng.randint(Wn.shape[0] - 1)) % Wn.shape[0]
                d = Wn[kk]
            else:
                d = Wn[k[j]]
            heat = (pat_np[j] @ d).reshape(g, g)
            hm = heat - heat.min()
            conc.append(float(np.sort(hm.flatten())[::-1][:max(int(0.25 * g * g), 1)].sum()
                              / max(hm.sum(), 1e-12)))
            bx = heat_to_box(heat, args.res, args.keep)
            w0, h0 = im.size
            ob = box_in_original(bx, w0, h0, args.res, args.margin)
            frac = ((ob[2] - ob[0]) * (ob[3] - ob[1])) / float(w0 * h0)
            areas.append(frac)
            zooms.append(1.0 / max(np.sqrt(frac), 1e-6))
            boxes.append(ob)
            crops.append(im.crop((int(ob[0]), int(ob[1]), int(ob[2]), int(ob[3]))))

        xz = torch.stack([eval_tf(c) for c in crops]).to(dev)
        cz, _, _ = forward_tokens(visual, xz)
        cz = l2(cz.cpu().numpy().astype(np.float32))
        lgz = _pred.apply_head(cz, W, b, s, cosine, sub)
        kz = np.argmax(lgz, 1)
        pz = np.exp(lgz - lgz.max(1, keepdims=True))
        conf_zoom += list((pz / pz.sum(1, keepdims=True)).max(1))
        preds_zoom += list(kz)

    zooms = np.array(zooms)
    areas = np.array(areas)
    conc = np.array(conc)
    dis = float(np.mean(np.array(preds_full) != np.array(preds_zoom)))

    banner("结果")
    print("  热量集中度（最高 25%% 格子占多少热量；均匀分布=0.25）")
    print("     中位数 %.3f   均值 %.3f   ⭐远高于 0.25 才说明模型确实在'看某个地方'"
          % (np.median(conc), conc.mean()))
    print()
    print("  框住的面积占原图比例   中位数 %.3f  (25%%分位 %.3f, 75%%分位 %.3f)"
          % (np.median(areas), np.percentile(areas, 25), np.percentile(areas, 75)))
    print("  ⭐隐含放大倍率         中位数 %.3f  (25%%分位 %.3f, 75%%分位 %.3f)"
          % (np.median(zooms), np.percentile(zooms, 25), np.percentile(zooms, 75)))
    print("     参照：448→512 只有 1.15 倍，换到了 +0.49 分")
    print()
    print("  全图预测 vs 放大预测 分歧率  %.1f%%  (%d/%d)"
          % (100 * dis, int(dis * len(zooms)), len(zooms)))
    print("  平均最大 softmax：全图 %.4f → 放大 %.4f  (%+.4f)"
          % (np.mean(conf_full), np.mean(conf_zoom), np.mean(conf_zoom) - np.mean(conf_full)))
    print("     ⚠置信度不是正确率，只能当很弱的旁证")

    banner("判定（跑前写死的判据）")
    z = float(np.median(zooms))
    if z < 1.20:
        v = "⛔ 放弃：主体本来就占满画面，裁剪买不到 token 密度"
    elif z < 1.40:
        v = "⚠ 边缘：放大倍率不够大，要靠分歧率决定值不值得"
    else:
        v = "✅ 有肉：放大倍率 %.2f 远超 448→512 的 1.15 倍" % z
    print("  主判据 隐含放大倍率中位数 = %.3f → %s" % (z, v))
    if dis < 0.02:
        print("  副判据 分歧率 %.1f%% < 2%% → ⚠改动太小，就算方向对也测不出" % (100 * dis))
    elif dis > 0.40:
        print("  副判据 分歧率 %.1f%% > 40%% → ⛔多半是裁坏了，不是看得更清" % (100 * dis))
    else:
        print("  副判据 分歧率 %.1f%% 落在合理区间（5~20%% 与 TTA 内部分歧同量级）" % (100 * dis))


if __name__ == "__main__":
    main()
