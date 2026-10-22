# -*- coding: utf-8 -*-
"""
完整微调 CLIP ViT-B/32 骨干（可选提高输入分辨率）—— 离开 60 分区间的主线。

═══════════════════════════════════════════════════════════════════════
一、为什么之前的路走错了（把账算清楚，免得再犯）
═══════════════════════════════════════════════════════════════════════

之前整套策略建立在一句话上：「瓶颈是噪声不是容量，放开骨干=给容量去死记噪声」。
**这句话的两个支柱后来都塌了**：

  ①「噪声 30%」—— 那是由 val acc 0.693 推的**上界**，不是点估计。
     HX 人工数第 0000 类：206 张里错标 **3 张 = 1.46%**；
     重复图冲突检测独立估出 **1.06%**。两种方法收敛 → **根本没有噪声要怕**。
  ②「干净子集 acc 0.93，说明 CLIP 特征够用」—— 后来查明"干净子集"其实是**简单子集**
     （suspicion 量的是"难"不是"错"）。**简单题做对 0.93，说明不了难题。**

两个支柱塌了，正确结论是反的：**应该往死里加容量。**

而 13 号那次 LoRA 是 **r=4、只挂 q/v、lr=1e-4、3 轮 = 147K 参数 = 骨干的 0.168%**，
训练日志里批准确率 0.85，跟线性头的 0.81 几乎一样 —— **模型压根没动**。
那不是"微调无效"的证据，是"根本没微调"的证据。⛔别再拿它当关闭轴 A 的理由。

═══════════════════════════════════════════════════════════════════════
二、这个脚本做什么
═══════════════════════════════════════════════════════════════════════

**完整微调整个视觉编码器的 8785 万参数**（不是 LoRA 的 14 万），配 LP-FT 起手。

⭐**可选：把位置编码插值到更高分辨率**。这直接解决"看不清"这个诊断：

    224×224 ÷ 32×32 的块 = 7×7  =  49 个格子   ← 现在
    336×336 ÷ 32        = 10×10 = 100 个格子   (2.0x)
    448×448 ÷ 32        = 14×14 = 196 个格子   (4.0x)

细粒度识别靠喙形/羽纹/叶脉，这些在原图占几十像素，缩到 224 之后连一个格子的
四分之一都不到。TTA 那 +2.357 分本质上就是在用随机裁剪**笨拙地模拟放大**——
直接提分辨率是从根上解决。
⚠位置编码插值不改变权重来源，仍是 OpenAI 官方 CLIP ViT-B/32，不违反赛题约束
（赛题禁的是"换模型/多模型集成/融合/投票"，不是微调）。

═══════════════════════════════════════════════════════════════════════
三、⭐最重要的监控量：训练准确率
═══════════════════════════════════════════════════════════════════════

线性头在 C=10 时训练 acc 只有 **0.81**。完整微调应该能把它推到 **0.95+**。

  · 训练 acc 上不去 → **欠拟合**，加大 lr / 加轮数 / 减正则
  · 训练 acc 上去了但验证不涨 → 过拟合，这时才需要正则和抗噪
  ⭐**在噪声只有 1.5% 的情况下，先追训练 acc，别过早怕过拟合。**

用法:
  python scripts/24_finetune_full.py --smoke                 # 3分钟冒烟，先确认能跑
  python scripts/24_finetune_full.py                         # 阶段A：224，15轮，约2小时
  python scripts/24_finetune_full.py --resolution 336 --epochs 10 --batch-size 32
  python scripts/24_finetune_full.py --resume runs/xxx/ckpt_best.pt --resolution 448
"""
import _bootstrap  # noqa: F401
from _bootstrap import banner

import argparse
import importlib.util
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm import tqdm

from configs import paths as P


def train_test_split(*a, **k):
    """⭐延迟导入 sklearn（09-24 实踩）：Windows spawn 的每个 DataLoader worker 都会把本脚本
    （以及用 _load_sibling 加载的兄弟脚本）从头 import 一遍。顶层 import sklearn 会连带加载
    scipy 那一串 DLL，worker 根本用不上，却要多提交几百 MB 虚拟内存 —— 本机内存墙下评估
    worker 就是死在 `scipy ... _rbfinterp_pythran: 页面文件太小`。调用方式与原函数完全相同。"""
    from sklearn.model_selection import train_test_split as _tts
    return _tts(*a, **k)

ImageFile.LOAD_TRUNCATED_IMAGES = True
SEED, VAL_FRAC = 42, 0.2


def _load_sibling(fname, modname):
    """⚠sys.modules 注册必须在【文件顶层】执行，否则 Windows spawn 的 worker 静默挂起。"""
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(modname, os.path.join(here, fname))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


_ex = _load_sibling("01_extract_features.py", "extract01")


# ══════════════════════════════════════════════════════════ 分辨率
def _resize_pos_embed(old, g_new):
    """
    把一份位置编码张量 (1+g_old², width) 插值成 (1+g_new², width)。
    ⚠cls token 那一行不参与插值（它没有空间位置），只插值 grid 部分。
    """
    width = old.shape[1]
    g_old = int(math.sqrt(old.shape[0] - 1))
    if g_old == g_new:
        return old
    cls_pe, grid = old[:1].float(), old[1:].float()
    grid = grid.reshape(g_old, g_old, width).permute(2, 0, 1).unsqueeze(0)
    grid = Fn.interpolate(grid, size=(g_new, g_new), mode="bicubic", align_corners=False)
    grid = grid.squeeze(0).permute(1, 2, 0).reshape(g_new * g_new, width)
    return torch.cat([cls_pe, grid], 0).to(old.dtype)


def interpolate_pos_embed(model, new_res):
    """
    把 CLIP 视觉塔的位置编码从 7×7 插值到 (new_res/32)×(new_res/32)。

    CLIP 的 VisionTransformer.forward 里，位置编码是直接加在 token 序列上的：
        x = cat([cls_token, patches], dim=1) + self.positional_embedding
    所以只要把这个参数换成正确尺寸的，其余代码一行都不用改。
    conv1 是 stride=32 的卷积，天然支持任意输入尺寸。

    ⚠cls token 那一行不参与插值（它没有空间位置），只插值 grid 部分。
    """
    v = model.visual
    old = v.positional_embedding.data                     # (1+g², width)
    width = old.shape[1]
    g_old = int(math.sqrt(old.shape[0] - 1))
    g_new = new_res // 32
    if g_new == g_old:
        return g_old, g_new
    v.positional_embedding = nn.Parameter(_resize_pos_embed(old, g_new))
    v.input_resolution = new_res
    return g_old, g_new


def build_transforms(res, aug_scale):
    """CLIP 官方的归一化参数，别改；只换尺寸和增强强度。"""
    norm = T.Normalize((0.48145466, 0.4578275, 0.40821073),
                       (0.26862954, 0.26130258, 0.27577711))
    train_tf = T.Compose([
        T.RandomResizedCrop(res, scale=(aug_scale, 1.0), ratio=(0.8, 1.25),
                            interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(),
        T.ToTensor(), norm,
    ])
    eval_tf = T.Compose([
        T.Resize(res, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(res), T.ToTensor(), norm,
    ])
    return train_tf, eval_tf


# ══════════════════════════════════════════════════════════ 数据
class ImgDS(Dataset):
    """⚠transform 里不能有 lambda（Windows spawn 无法序列化），RGB 转换放这里。"""

    def __init__(self, paths, labels, tf):
        self.p, self.y, self.tf = paths, labels, tf

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        try:
            with Image.open(self.p[i]) as im:
                return self.tf(im.convert("RGB")), int(self.y[i])
        except Exception:
            return torch.zeros(3, self.tf.transforms[0].size[0]
                               if hasattr(self.tf.transforms[0], "size") else 224,
                               224), int(self.y[i])


# ══════════════════════════════════════════════════════════ 损失
def make_loss(kind, K, ls=0.1, beta=1.0):
    """
    ⭐默认用 CE + label smoothing，不用 SCE —— 这是**刻意的选择**，理由：
      SCE 的 RCE 项**有界**，抗噪正源于此；但有界 = 难样本也拽不动模型。
      它在"线性头 + 以为有 30% 噪声"的旧设定下是对的（榜上确认 +1.9）。
      现在噪声只有 1.5%、且我们**正需要模型去啃难样本**，有界损失反而挡路。
      label smoothing 0.1 足以应付 1.5% 的噪声，又不封顶梯度。
    ⚠这是假设，不是定论。--loss sce 可以切回去做对照。
    """
    if kind == "ce_ls":
        return lambda lg, y: Fn.cross_entropy(lg, y, label_smoothing=ls)
    if kind == "ce":
        return lambda lg, y: Fn.cross_entropy(lg, y)

    def sce(lg, y):
        p = Fn.softmax(lg, 1).clamp(1e-7, 1.0)
        oh = Fn.one_hot(y, K).float()
        return Fn.cross_entropy(lg, y) - beta * (p * ((1 - oh) * -4.0)).sum(1).mean()
    return sce


# ══════════════════════════════════════════════════════════ 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", type=int, default=224,
                    help="输入分辨率，必须是 32 的倍数。224/288/336/384/448")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=64, help="8GB 显存下 224px 的安全值")
    ap.add_argument("--lr-backbone", type=float, default=1e-5,
                    help="⚠完整微调必须用很小的 lr，1e-5 量级；大了会把预训练特征冲垮")
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--aug-scale", type=float, default=0.5,
                    help="RandomResizedCrop 的面积下限。⭐骨干在训练了，增强机制才真正生效")
    ap.add_argument("--loss", choices=["ce_ls", "ce", "sce"], default="ce_ls")
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--freeze-first-n", type=int, default=0,
                    help=">0 时冻结最靠前的 N 层（学的是边缘/纹理等通用特征）")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--eval-workers", type=int, default=2,
                    help="⚠必须小于训练 worker：评估在训练循环内触发，两边进程数相加，"
                         "总数 >8 会报 [WinError 1455] 页面文件太小")
    ap.add_argument("--head-init", type=str,
                    default=os.path.join(P.CACHE_DIR, "head_loss_sce.npz"),
                    help="LP-FT 起手用的头。⛔别随机初始化")
    ap.add_argument("--resume", type=str, default=None, help="从某个 ckpt 继续（换分辨率时用）")
    ap.add_argument("--smoke", action="store_true", help="小样本冒烟，3 分钟验证全流程")
    args = ap.parse_args()

    assert args.resolution % 32 == 0, "分辨率必须是 32 的倍数"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    P.ensure_dirs()

    banner("24 完整微调 CLIP ViT-B/32  @%dpx  %d轮  loss=%s"
           % (args.resolution, args.epochs, args.loss))

    # ---------------------------------------------------------------- 数据划分
    paths, labels, class_names = _ex.list_train_files()
    y = labels.astype(np.int64)
    K = len(class_names)
    idx = np.arange(len(y))
    itr, iva = train_test_split(idx, test_size=VAL_FRAC, random_state=SEED, stratify=y)
    if args.smoke:
        itr, iva = itr[:2000], iva[:1000]
    tr_paths = [paths[i] for i in itr]
    va_paths = [paths[i] for i in iva]
    ytr, yv = y[itr], y[iva]
    print("  训练 %d / 验证 %d / 类别 %d  (seed=%d，与所有历史实验一致)"
          % (len(itr), len(iva), K, SEED))

    # ---------------------------------------------------------------- 尺子
    wz = np.load(os.path.join(P.CACHE_DIR, "val_weights.npz"))
    w_all = wz["weight"]
    if args.smoke:
        w = w_all[:len(iva)]
    else:
        assert np.array_equal(wz["val_idx"], iva), "权重文件划分不一致"
        w = w_all
    nz = np.load(os.path.join(P.CACHE_DIR, "noise_scores.npz"), allow_pickle=True)
    sv = nz["suspicion"][iva]
    m60 = sv <= np.quantile(sv, 0.60)

    # ⭐⭐无泄漏子集：这批验证图在训练集里【没有近重复副本】。
    #   为什么必须单看它：LoRA 那次实测，改骨干的方法在本地被高估 2.5 倍
    #   （本地 +0.0037 → 真实 +0.15）。**完整微调的记忆能力远强于 r=4 的 LoRA，
    #   高估只会更严重**——模型会去背训练图，而验证集里 5.67% 的图在训练集有孪生兄弟，
    #   测试集里则是 0.00%。分布匹配权重压低了它们，但压不干净。
    #   判据：**这个子集和分布匹配同向上升才可信；只有主指标涨 = 在背训练图。**
    lk = np.load(os.path.join(P.CACHE_DIR, "leakage_scores.npz"))
    assert np.array_equal(lk["val_idx"], iva) or args.smoke, "泄漏文件划分不一致"
    ms = lk["val_maxsim"][:len(iva)] if args.smoke else lk["val_maxsim"]
    mnl = ms < 0.90                       # 6835 张 (33.1%)
    print("  尺子  ①分布匹配(主) ②⭐无泄漏子集 val_maxsim<0.90 (%d张) ③干净60%%(副)"
          % mnl.sum())
    print("  ⭐判读：|分布匹配Δ|>0.010 才可信、<0.005 是噪声（5次真值实证）；"
          "①②必须同向")

    # ---------------------------------------------------------------- 模型
    import clip
    model, _ = clip.load(P.CLIP_MODEL_NAME, device="cpu",
                         download_root=P.CLIP_DOWNLOAD_ROOT)
    model.float()                                    # ⚠fp16 主权重训不稳，必须 fp32
    visual = model.visual
    g_old, g_new = interpolate_pos_embed(model, args.resolution)
    print("  位置编码  %d×%d = %d 格  →  %d×%d = %d 格  (%.1fx)"
          % (g_old, g_old, g_old ** 2, g_new, g_new, g_new ** 2,
             (g_new / g_old) ** 2))
    visual = visual.to(device)
    for p_ in visual.parameters():
        p_.requires_grad = True
    if args.freeze_first_n > 0:
        for i in range(args.freeze_first_n):
            for p_ in visual.transformer.resblocks[i].parameters():
                p_.requires_grad = False
        print("  已冻结最靠前的 %d 层（通用特征，动它风险大）" % args.freeze_first_n)
    n_train_p = sum(p.numel() for p in visual.parameters() if p.requires_grad)
    print("  可训练骨干参数 %s / %s  (%.1f%%)  ⭐对比 13号LoRA 只有 147,456 (0.17%%)"
          % (format(n_train_p, ","), format(P.CLIP_VISUAL_PARAMS, ","),
             100.0 * n_train_p / P.CLIP_VISUAL_PARAMS))

    # ---------------------------------------------------------------- 头
    hd = np.load(args.head_init)
    W = torch.tensor(hd["W"], dtype=torch.float32, device=device, requires_grad=True)
    b = torch.tensor(hd["b"], dtype=torch.float32, device=device, requires_grad=True)
    print("  头  LP-FT 起手 <- %s  W%s" % (os.path.basename(args.head_init), tuple(W.shape)))
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        sd = dict(ck["visual"])
        # ⚠⚠换分辨率时必须先把 ckpt 里的位置编码插值到新尺寸再 load。
        #   `strict=False` 只放过"键缺失/多余"，**尺寸不符照样抛 RuntimeError**。
        #   224 的 ckpt 是 (1+7², 768)=(50,768)，336 要 (1+10², 768)=(101,768)。
        k = "positional_embedding"
        if k in sd and sd[k].shape[0] != g_new ** 2 + 1:
            print("  位置编码：ckpt 是 %s，本次要 %d → 插值迁移"
                  % (tuple(sd[k].shape), g_new ** 2 + 1))
            sd[k] = _resize_pos_embed(sd[k], g_new)
        miss, unexp = visual.load_state_dict(sd, strict=False)
        assert not unexp, "ckpt 里有模型不认识的键: %s" % unexp[:3]
        if miss:
            print("  ⚠ckpt 里缺这些键（将保留预训练值）: %s" % miss[:5])
        W.data = ck["W"].to(device)
        b.data = ck["b"].to(device)
        print("  已从 %s 恢复（第 %s 轮，该 ckpt 分辨率 %s）"
              % (os.path.basename(os.path.dirname(args.resume)), ck.get("epoch"),
                 ck.get("args", {}).get("resolution")))

    # ---------------------------------------------------------------- 训练配置
    train_tf, eval_tf = build_transforms(args.resolution, args.aug_scale)
    loader = DataLoader(ImgDS(tr_paths, ytr, train_tf), batch_size=args.batch_size,
                        shuffle=True, num_workers=args.num_workers, drop_last=True,
                        pin_memory=(device == "cuda"), persistent_workers=True)
    vloader = DataLoader(ImgDS(va_paths, yv, eval_tf), batch_size=args.batch_size,
                         shuffle=False, num_workers=args.eval_workers, pin_memory=False)

    # 骨干和头用差两个数量级的学习率：骨干只是微调，头要真的学
    opt = torch.optim.AdamW([
        {"params": [p_ for p_ in visual.parameters() if p_.requires_grad],
         "lr": args.lr_backbone, "weight_decay": args.weight_decay},
        {"params": [W, b], "lr": args.lr_head, "weight_decay": 0.0},
    ])
    spe = len(loader)
    total = spe * args.epochs
    warm = max(int(total * args.warmup_frac), 1)
    base_lrs = [g["lr"] for g in opt.param_groups]
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    crit = make_loss(args.loss, K, args.label_smoothing)
    print("  优化  AdamW  骨干lr=%g  头lr=%g  wd=%g  warmup=%d/%d步  余弦退火"
          % (args.lr_backbone, args.lr_head, args.weight_decay, warm, total))
    print("  增强  RandomResizedCrop(%d, scale=%.2f~1.0)+HFlip  ⛔无颜色抖动"
          % (args.resolution, args.aug_scale))
    print("  进程  训练%d + 评估%d = 峰值%d" % (args.num_workers, args.eval_workers,
                                               args.num_workers + args.eval_workers))

    run = os.path.join(P.RUNS_DIR, time.strftime("%Y%m%d_%H%M%S")
                       + "_ft%d" % args.resolution)
    os.makedirs(run, exist_ok=True)
    print("  产物  ->", run)

    @torch.no_grad()
    def evaluate():
        visual.eval()
        cs = []
        bar = tqdm(total=len(yv), desc="eval", unit="img", unit_scale=True,
                   dynamic_ncols=True, leave=False)
        for im, yy in vloader:
            im = im.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
                f = visual(im)
            f = f.float()
            f = f / f.norm(dim=1, keepdim=True).clamp_min(1e-12)
            cs.append(((f @ W.T + b).argmax(1).cpu().numpy() == yy.numpy()))
            bar.update(im.shape[0])
        bar.close()
        visual.train()
        c = np.concatenate(cs).astype(np.float64)
        return dict(matched=float((c * w).sum() / w.sum()),
                    noleak=float(c[mnl].mean()),      # ⭐无泄漏下界
                    clean60=float(c[m60].mean()) if not args.smoke else float(c.mean()),
                    raw=float(c.mean()))

    banner("起点评估（原版 CLIP + LP 头，等价于当前 61.5 那条线的骨架）")
    hist = [dict(epoch=0.0, train_acc=None, **evaluate())]
    print("  分布匹配 %.4f   ⭐无泄漏 %.4f   干净60%% %.4f   原始等权 %.4f"
          % (hist[0]["matched"], hist[0]["noleak"], hist[0]["clean60"], hist[0]["raw"]))
    if not args.smoke and args.resolution == 224:
        print("  自检：@224 起点的分布匹配应 ≈ 0.6129  →  %s"
              % ("✅ 对得上" if abs(hist[0]["matched"] - 0.6129) < 0.004
                 else "❌ 对不上！数据管线或头加载有问题，后面全废"))
    ref = hist[0]

    banner("开训")
    visual.train()
    step, t0, best = 0, time.time(), -1.0
    for ep in range(1, args.epochs + 1):
        cor = tot = 0
        bar = tqdm(total=spe, desc="ep%d/%d" % (ep, args.epochs), unit="it",
                   dynamic_ncols=True)
        for im, yy in loader:
            sc = ((step + 1) / warm if step < warm else
                  0.5 * (1 + math.cos(math.pi * (step - warm) / max(total - warm, 1))))
            for g, bl in zip(opt.param_groups, base_lrs):
                g["lr"] = bl * sc
            im = im.to(device, non_blocking=True)
            yy = yy.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
                f = visual(im)
            f = f.float()
            f = f / f.norm(dim=1, keepdim=True).clamp_min(1e-12)
            lg = f @ W.T + b
            loss = crit(lg, yy)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                [p_ for p_ in visual.parameters() if p_.requires_grad] + [W, b], 1.0)
            scaler.step(opt)
            scaler.update()
            step += 1
            cor += (lg.argmax(1) == yy).sum().item()
            tot += yy.numel()
            bar.update(1)
            if step % 20 == 0:
                bar.set_postfix_str("loss=%.3f 训练acc=%.4f lr=%.1e"
                                    % (float(loss.detach()), cor / tot,
                                       opt.param_groups[0]["lr"]))
        bar.close()
        tr_acc = cor / max(tot, 1)
        m = evaluate()
        d = m["matched"] - ref["matched"]
        dn = m["noleak"] - ref["noleak"]
        hist.append(dict(epoch=float(ep), train_acc=tr_acc, **m))
        flag = "✅同向" if (d > 0) == (dn > 0) else "⚠背图?"
        print("  [ep%02d] ⭐训练acc %.4f | 分布匹配 %.4f(%+.4f=%+.2f分) | "
              "⭐无泄漏 %.4f(%+.4f) %s | %.1f分钟"
              % (ep, tr_acc, m["matched"], d, d * 100,
                 m["noleak"], dn, flag, (time.time() - t0) / 60))
        if m["matched"] > best:
            best = m["matched"]
            torch.save({"visual": visual.state_dict(), "W": W.detach().cpu(),
                        "b": b.detach().cpu(), "epoch": ep, "args": vars(args),
                        "metrics": m}, os.path.join(run, "ckpt_best.pt"))
        with open(os.path.join(run, "history.json"), "w", encoding="utf-8") as fjs:
            json.dump({"args": vars(args), "history": hist}, fjs,
                      ensure_ascii=False, indent=2)

    # ---------------------------------------------------------------- 汇总
    banner("汇总")
    print("  %-6s %10s %12s %10s %10s" % ("epoch", "训练acc", "分布匹配", "Δ", "推算分数"))
    for h in hist:
        d = h["matched"] - ref["matched"]
        print("  %-6.0f %10s %12.4f %+10.4f %+9.2f"
              % (h["epoch"], "-" if h["train_acc"] is None else "%.4f" % h["train_acc"],
                 h["matched"], d, d * 100))
    print("\n  ⭐诊断：训练acc 若没到 0.95+ → **还在欠拟合**，加轮数/加 lr/减增强强度。")
    print("     线性头基线的训练acc 只有 0.81，完整微调应该远超它。")
    print("  ⭐下一步：用 ckpt_best.pt 接 --resolution 336 或 448 继续（位置编码会自动插值）")


if __name__ == "__main__":
    main()
