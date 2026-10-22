# -*- coding: utf-8 -*-
"""
微调 v2：把现代细粒度微调的整套正则化加上。24 号那版配方太朴素了。

═══════════════════════════════════════════════════════════════════════
一、为什么要有 v2（一个不正常的数字）
═══════════════════════════════════════════════════════════════════════

    线性探针（冻结骨干）  59.0
    完整微调（全部参数）  61.296   ← **只涨了 2.3 分**

细粒度任务上完整微调通常比线性探针高 10~15 分。只高 2.3 说明配方有问题。
而 24 号那版只有「随机裁剪 + 水平翻转 + label smoothing」——
现代细粒度微调的东西一个都没上。同时训练 acc 0.9887 / 验证 0.79，
**20 个点的鸿沟 = 典型过拟合**，说明瓶颈早就从"容量"转成"泛化"了。

═══════════════════════════════════════════════════════════════════════
二、v2 加了什么，每一条的理由
═══════════════════════════════════════════════════════════════════════

**① Mixup / CutMix** —— 直接打击"背样本"
   把两张图按比例混合，标签也按同比例混合。模型没法靠记住某一张图蒙对，
   必须学到真正的类别特征。细粒度分类的标配。
   ⚠会让训练 acc 看起来变低，**那是正常的、也是目的**——别再拿训练 acc 当进度条。

**② 层级学习率衰减 LLRD** —— 靠前的层少动，靠后的层多动
   ViT 靠前的层学的是边缘/纹理这类通用特征（CLIP 用 4 亿图训出来的，比我们
   10 万张能训出来的好得多），靠后的层才是任务相关的。
   给第 i 层的学习率 = base_lr × decay^(顶层号 − i)。微调 Transformer 的标准做法。

**③ 权重 EMA** —— 几乎免费的稳定收益
   维护一份权重的指数滑动平均。取的是"训练轨迹的平均"而不是"最后一步的落点"，
   等于免费的模型平均。⚠评估要用 EMA 那份权重，不是当前权重。

**④ JPEG 退化增强** —— 对齐测试集的图片规格（27 号脚本实测出来的）
   ⭐全量扫描发现：测试集 24967 张**长边全部精确等于 500**，JPEG 量化表均值 29.0；
   标定出主办方的流程是「长边→500，JPEG quality=75」。而训练集是原始高清爬取物
   （长边中位 640、67% 超过 512、还混着 PNG/WEBP/GIF）。
   实测把验证图压成测试规格，准确率掉 0.6~1.7 个点（难样本掉得更多）。
   ⚠**这不是主因（掉得不够多），但它是免费的增强**，顺手加上。

**⑤ 更强的随机裁剪** scale 0.5→0.3，配合 Mixup 一起提高难度

⛔**没加颜色抖动**：颜色是区分近缘物种的关键特征，改了等于制造错标。

═══════════════════════════════════════════════════════════════════════
三、⛔一条被否定的决定：别再往 448 冲
═══════════════════════════════════════════════════════════════════════

测试图**短边中位只有 375**。Resize(448) 会把它**放大 1.2 倍**，
在插值凭空造像素，不是获取信息。**测试图的实际分辨率就是物理天花板。**
320 已经贴着 375，够了。

用法:
  python scripts/28_finetune_v2.py --smoke                      # 3分钟冒烟
  python scripts/28_finetune_v2.py --resume runs/xxx/ckpt_best.pt --epochs 20
"""
import _bootstrap  # noqa: F401
from _bootstrap import banner

import argparse
import copy
import importlib.util
import io
import json
import math
import os
import random
import re
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
import torch.utils.checkpoint
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
    """⚠sys.modules 注册必须在文件顶层，否则 Windows spawn 的 worker 静默挂起。"""
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(modname, os.path.join(here, fname))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


_ex = _load_sibling("01_extract_features.py", "extract01")
_ft = _load_sibling("24_finetune_full.py", "finetune24")


# ══════════════════════════════════════════════════════════ 数据
class TrainDS(Dataset):
    """
    ⚠不能用 lambda（Windows spawn 无法序列化），所有开关做成实例属性。
    退化增强放在 transform **之前**：要先在原图尺度上做 JPEG 退化，
    才能复现测试集那条"先缩到500再压"的流水线；缩到 320 之后再压就不是一回事了。

    ⭐sw = 每个样本的损失权重。真标签为 1.0，伪标签为 --pseudo-weight（默认 0.5）。
      ⚠伪标签必然含错（教师本身只有 62 分），必须让真标签主导。
    """

    def __init__(self, paths, labels, tf, degrade_p, res, sw=None, degrade_fixed=False,
                 soft=None, n_real=None, K=0, kd=None):
        self.p, self.y, self.tf = paths, labels, tf
        self.degrade_p, self.res = degrade_p, res
        self.degrade_fixed = degrade_fixed
        self.sw = np.ones(len(paths), np.float32) if sw is None else sw
        self.soft = soft                      # 软伪标签：shape (n_pseudo, K) 或 None
        self.n_real = len(paths) if n_real is None else n_real
        self.K = K
        # ⭐实验 C2：训练集自蒸馏的教师分布，shape (n_real, K)，mmap 的 np.memmap 或 None。
        #   与 soft（伪标签软目标）是两条独立通道：soft 只作用于 i ≥ n_real，kd 只作用于 i < n_real。
        self.kd = kd

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        try:
            with Image.open(self.p[i]) as im:
                im = im.convert("RGB")
                if self.degrade_p > 0 and random.random() < self.degrade_p:
                    if self.degrade_fixed:
                        # 钉死在测试集精确规格（长边500, JPEG q75）—— 对齐 100%
                        ls, q = 500, 75
                    else:
                        # 随机化参数而不是钉死在 (500,75)：我们只标定出中位数，
                        # 让模型对一个范围鲁棒比对一个点鲁棒更安全
                        ls = random.choice([440, 470, 500, 500, 530])
                        q = random.choice([65, 70, 75, 75, 80, 85])
                    w, h = im.size
                    s = ls / max(w, h)
                    if s < 1.0:
                        im = im.resize((max(1, round(w * s)), max(1, round(h * s))),
                                       Image.BICUBIC)
                    buf = io.BytesIO()
                    im.save(buf, "JPEG", quality=q)
                    buf.seek(0)
                    im = Image.open(buf).convert("RGB")
                return (self.tf(im), int(self.y[i]), float(self.sw[i]), self._soft(i), self._kd(i))
        except Exception:
            return (torch.zeros(3, self.res, self.res), int(self.y[i]),
                    float(self.sw[i]), self._soft(i), self._kd(i))

    def _soft(self, i):
        """软伪标签：伪标签样本返回教师概率（已归一），真标签样本返回全零。"""
        if self.soft is not None and i >= self.n_real:
            return torch.as_tensor(self.soft[i - self.n_real])
        return torch.zeros(self.K)

    def _kd(self, i):
        """训练集自蒸馏：真标签样本返回教师分布（float32、已归一），其余返回全零。"""
        if self.kd is not None and i < self.n_real:
            return torch.as_tensor(np.asarray(self.kd[i], dtype=np.float32))
        return torch.zeros(self.K)


class EvalDS(Dataset):
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


# ══════════════════════════════════════════════════════════ Mixup / CutMix
def rand_bbox(h, w, lam):
    """CutMix 的随机矩形：面积占比 = 1-lam，位置随机（可能出界，clip 掉）。"""
    r = math.sqrt(1.0 - lam)
    cw, ch = int(w * r), int(h * r)
    cx, cy = np.random.randint(w), np.random.randint(h)
    x1, y1 = np.clip(cx - cw // 2, 0, w), np.clip(cy - ch // 2, 0, h)
    x2, y2 = np.clip(cx + cw // 2, 0, w), np.clip(cy + ch // 2, 0, h)
    return x1, y1, x2, y2


def mix_batch(x, y_soft, sw, mixup_a, cutmix_a, prob):
    """
    以 prob 的概率对这一批做 Mixup 或 CutMix（各半）。
    ⭐标签也要按同样比例混合 —— 这才是它起作用的地方：
      模型必须输出"70% 是 A、30% 是 B"，没法靠死记某一张图蒙对。
    ⚠样本权重也要跟着混：混合样本的权重 = 两个来源权重的同比例加权，
      否则"真标签 + 伪标签"混出来的样本会被当成全真的。
    """
    if np.random.rand() > prob:
        return x, y_soft, sw
    idx = torch.randperm(x.size(0), device=x.device)
    if np.random.rand() < 0.5 and mixup_a > 0:                    # Mixup
        lam = float(np.random.beta(mixup_a, mixup_a))
        x = lam * x + (1 - lam) * x[idx]
    elif cutmix_a > 0:                                            # CutMix
        lam = float(np.random.beta(cutmix_a, cutmix_a))
        x1, y1, x2, y2 = rand_bbox(x.size(2), x.size(3), lam)
        x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
        lam = 1 - ((x2 - x1) * (y2 - y1) / (x.size(-1) * x.size(-2)))  # 用实际面积修正
    else:
        return x, y_soft, sw
    return (x, lam * y_soft + (1 - lam) * y_soft[idx],
            lam * sw + (1 - lam) * sw[idx])


def soft_ce(logits, target, sw=None):
    """软标签交叉熵。⚠混合之后目标不再是 one-hot，不能用 F.cross_entropy(int 标签)。"""
    per = -(target * Fn.log_softmax(logits, dim=1)).sum(1)
    if sw is None:
        return per.mean()
    # ⚠按权重归一化而不是简单 mean：否则一批里伪标签多的时候整体梯度会被压小
    return (per * sw).sum() / sw.sum().clamp_min(1e-8)


# ══════════════════════════════════════════════════════════ LLRD
def layer_id_of(name, n_blocks):
    """
    把参数名映射到"第几层"，用于层级学习率衰减。
    0 = 输入端（conv1/位置编码/ln_pre），1..n = 各 transformer block，n+1 = 输出端。
    """
    if name.startswith(("conv1", "class_embedding", "positional_embedding", "ln_pre")):
        return 0
    if name.startswith("transformer.resblocks."):
        return int(name.split(".")[2]) + 1
    return n_blocks + 1          # ln_post / proj


def build_param_groups(visual, base_lr, decay, wd):
    """⭐靠前的层学通用特征（CLIP 用 4 亿图训的），少动；靠后的层学任务特征，多动。"""
    n = len(visual.transformer.resblocks)
    top = n + 1
    groups, info = {}, {}
    for nm, p_ in visual.named_parameters():
        if not p_.requires_grad:
            continue
        lid = layer_id_of(nm, n)
        lr = base_lr * (decay ** (top - lid))
        # 1 维参数（LayerNorm 的 weight/bias、各种 bias）不做权重衰减，这是通行做法
        key = (lid, p_.ndim > 1)
        groups.setdefault(key, []).append(p_)
        info[lid] = lr
    out = [{"params": ps, "lr": base_lr * (decay ** (top - lid)),
            "weight_decay": wd if is_w else 0.0}
           for (lid, is_w), ps in sorted(groups.items())]
    return out, info


# ══════════════════════════════════════════════════════════ DropPath（随机深度）
# ⭐为什么加这个：3.2 节的诊断规则是「train >> test = 在死背 → 加正则」。
#   实测泛化落差自 v2@320 起就卡在 ~20 分不动（当时 train 0.9337 / 留出 0.7276 / 榜 71.12；
#   现在 in-sample 0.9739 / 榜 ~78.3，还是 19 分）。而 09-09 以来七个实验全是**决策边界**
#   旋钮（degrade/头形态/伪标签软硬），⛔没有一个动过正则化——落差自然不动。
#   grep 全文：模型级正则（drop-path / dropout）一个都没有，增强只有 crop+flip。
#   → 这一族的药只开了一半，DropPath 是 ViT 微调最标准的那半。
#
# ⚠实现上有两个坑，都绕开了：
#   ① ⛔不能用「包一层 wrapper 换掉 resblocks[i]」——那会给 state_dict 的键加前缀，
#      --resume 和 26/45 的 strict=True 全部会炸。
#      → 改成把 DropPath 挂成 block 的子模块：它没有任何 parameter/buffer，
#        **state_dict 逐键不变**，老 ckpt 照样能读，新 ckpt 也照样能被 26/45 读。
#   ② CLIP 的 transformer 是 **LND 布局**（seq, batch, dim），batch 在第 1 维不是第 0 维。
#      掩码必须是 (1, N, 1) 才是"按样本丢",写成 (N,1,1) 会变成"按 token 位置丢"，
#      那是另一种完全不同的正则，且会把 CLS 位一起丢掉。
class DropPath(nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.p = float(p)

    def forward(self, x):
        if self.p <= 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        shape = [1] * x.ndim
        shape[1] = x.shape[1]                      # ⭐LND：按 batch 维丢
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep                     # 除以 keep：保持期望不变


def _dp_block_forward(self, x):
    """替换 CLIP ResidualAttentionBlock.forward，在两条残差支路上各加一个 DropPath。
    ⚠写成模块级函数（不是闭包）——EMA 里有 copy.deepcopy(visual)，
      闭包捕获的自由变量会跟着被复制，模块级函数 + MethodType 才干净。"""
    x = x + self._dp_attn(self.attention(self.ln_1(x)))
    x = x + self._dp_mlp(self.mlp(self.ln_2(x)))
    return x


def apply_drop_path(visual, p):
    """线性递增（DeiT/timm 惯例）：第 0 层 0，最后一层 p。底层是通用特征，别多扰动。"""
    import types
    blocks = visual.transformer.resblocks
    n = len(blocks)
    rates = []
    for i, blk in enumerate(blocks):
        dp = p * i / max(n - 1, 1)
        blk._dp_attn = DropPath(dp)
        blk._dp_mlp = DropPath(dp)
        blk.forward = types.MethodType(_dp_block_forward, blk)
        rates.append(dp)
    return rates


def _ckpt_transformer_forward(self, x):
    """替换 CLIP Transformer.forward：训练且需要梯度时逐 block 做梯度检查点，否则原样。
    ⭐前向只存每个 block 的入口，反向时现场重算该 block —— 激活显存从 O(层数) 降到 O(1)，
      代价是多算一遍前向（约 +30% 计算）。
    ⭐数学上逐位等价：checkpoint 默认 preserve_rng_state=True，重算时 DropPath 的伯努利掩码
      与第一次前向**同一个随机数**；autocast 状态也会被非重入实现一并还原。⛔不是超参。
    ⚠只在 self.training and grad 开启时生效：co-teaching 的 ct_probe（no_grad）、评估、EMA 影子
      （eval 模式）都走原路径，不付重算的钱。
    ⚠同样必须是**模块级**函数：EMA 与 co-teaching B 网都要 deepcopy(visual)。"""
    if self.training and torch.is_grad_enabled():
        for blk in self.resblocks:
            x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
        return x
    return self.resblocks(x)


def apply_grad_ckpt(visual):
    import types
    visual.transformer.forward = types.MethodType(_ckpt_transformer_forward, visual.transformer)


# ══════════════════════════════════════════════════════════ EMA
class EMA:
    """权重的指数滑动平均。取训练轨迹的平均而非终点落点，等于免费的模型平均。"""

    def __init__(self, module, decay):
        self.decay = decay
        self.shadow = copy.deepcopy(module).eval()
        for p_ in self.shadow.parameters():
            p_.requires_grad_(False)

    @torch.no_grad()
    def update(self, module):
        d = self.decay
        for s, m in zip(self.shadow.state_dict().values(), module.state_dict().values()):
            if s.dtype.is_floating_point:
                s.mul_(d).add_(m.detach(), alpha=1 - d)
            else:
                s.copy_(m)


# ══════════════════════════════════════════════════════════ 几何 TrivialAugment（实验 K）
class GeoTrivialAugment(T.TrivialAugmentWide):
    """torchvision 的 TrivialAugmentWide，算子空间砍成几何六件（⛔无颜色项）。
    ⚠必须是**模块级**类：Windows spawn 的 DataLoader worker 要 pickle 变换对象，
      写在 main() 里的局部类会炸（09-18 冒烟实炸：Can't pickle local object 'main.<locals>._GeoTA'）。
      这和 TrainDS 那条"不能用 lambda"是同一个坑。"""

    def _augmentation_space(self, num_bins):
        full = super()._augmentation_space(num_bins)
        keep = ("Identity", "ShearX", "ShearY", "TranslateX", "TranslateY", "Rotate")
        return {k: full[k] for k in keep}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", type=int, default=320)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr-backbone", type=float, default=1.5e-5, help="最顶层的 lr，往下逐层衰减")
    ap.add_argument("--llrd", type=float, default=0.75, help="层级衰减系数，0.65~0.85 常用")
    ap.add_argument("--freeze-first-n", type=int, default=0,
                    help="⭐硬冻结 layer_id < N 的层（0=输入端conv1/posemb/ln_pre，"
                         "1..12=block00..11）。0 表示不冻，全量微调。"
                         "⛔LLRD 替代不了它：LLRD 只管 lr，管不住梯度——44号实测底层"
                         "实际漂移是 lr 该给的 3.47 倍。硬冻结 lr=0 → 漂移严格为 0。")
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--mixup", type=float, default=0.8)
    ap.add_argument("--cutmix", type=float, default=1.0)
    ap.add_argument("--mix-prob", type=float, default=0.7)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--ema-decay", type=float, default=0.9995)
    # ── 实验 J：模型级正则 + 增强加码（默认全关，保住 D/E/F/③a/③b 的逐字复现）
    ap.add_argument("--drop-path", type=float, default=0.0,
                    help="随机深度，最后一层的丢弃率（逐层线性递增）。0=关。ViT 微调常用 0.1~0.3")
    ap.add_argument("--random-erasing", type=float, default=0.0,
                    help="RandomErasing 概率。0=关。挡住一块逼模型别只盯一个局部特征")
    ap.add_argument("--degrade-p", type=float, default=0.5,
                    help="以多大概率把训练图压成测试集规格（长边500+JPEG q75 附近随机）")
    ap.add_argument("--degrade-fixed", action="store_true",
                    help="⭕JPEG 退化参数钉死为（长边500, q75）= 测试集精确规格，不再随机化量级")
    ap.add_argument("--cosine-head", action="store_true",
                    help="③a 余弦分类头：lg = s·(f̂@Ŵ.T)，f̂/Ŵ 均 L2 归一、s 学习标量、去偏置 b")
    ap.add_argument("--subcenters", type=int, default=0,
                    help="③b 子中心分类头：每类 N 个子中心（>1 启用，W 变 (K,N,d)），"
                         "取最像那个（max cosine）当该类得分。⭐叠加在余弦头上，"
                         "唯一变量=单中心→N 子中心；起步=复制原中心+小扰动防塌缩")
    # ── 实验 U（2026-09-17）：角度间隔。⭐③a 改的是头的**参数化**、③b 改的是头的**容量**，
    #   两者都没动过训练目标的**几何**——"什么才算判对"是第三根轴，这是它没被"头形态族封口"覆盖的原因。
    ap.add_argument("--margin", type=float, default=0.0,
                    help="⭐CosFace 加性余弦间隔 m：训练损失用 s·(cosθ − m·y)，**推理⛔不减**。"
                         "0=关（逐位复现 L/J/M）。0.1/0.2/0.35 是常用档。"
                         "⚠必须配 --cosine-head（或 --subcenters）：线性头的 logit 不是余弦，"
                         "减间隔没有几何含义。"
                         "⭐选 CosFace 而不是 ArcFace 的理由：我们的目标是 mixup 之后的**软标签**，"
                         "加性间隔能按软权重线性摊开（cosθ−m·y），角度间隔 cos(θ+m) 摊不开")
    # ── 实验 C（2026-09-18，合规体制 v3）：训练增强补上"几何 TrivialAugment"。
    #   现有增强 = RRC + HFlip + JPEG 退化 + RandomErasing，⛔一个 AutoAugment 族的项都没有。
    #   只取几何算子（Identity/ShearX/ShearY/TranslateX/TranslateY/Rotate），⛔不动颜色：
    #   鸟类的颜色本身是判别特征，颜色抖动等于制造错标（这是 v2 起就写死的判断）。
    ap.add_argument("--aug-geo", action="store_true",
                    help="⭐几何 TrivialAugment（每张图随机挑 1 个几何算子、随机幅度），排在 RRC 之后、HFlip 之前。"
                         "⛔无颜色项。默认关，保住所有历史配方的逐位复现")
    # ── 实验 C2（2026-09-18）：**训练集**自蒸馏。教师 = 上一发训出的模型在【训练图】上的软预测，
    #   学生目标 = (1−α)·(one-hot+平滑) + α·教师分布。⭐十(二)原文"结合教师模型、自训练"，
    #   ⛔只用训练图（十一(二)1 禁止的是测试数据进训练），提交的仍是单一学生模型。
    #   文件：<prefix>_prob.npy (N,K) float16 + <prefix>_paths.txt（N 行，与 prob 同序），
    #   用 mmap 读，⛔别整块进内存（4 个 worker 各拷一份会撞内存墙）。
    ap.add_argument("--distill", type=str, default=None,
                    help="训练集自蒸馏教师软标签前缀，如 cache\\kd_C1（需 kd_C1_prob.npy + kd_C1_paths.txt）")
    ap.add_argument("--distill-alpha", type=float, default=0.5, help="教师分布在目标里的权重 α")
    ap.add_argument("--distill-temp", type=float, default=1.0, help="教师分布温度：prob^(1/T) 后归一，1=不动")
    # ── 实验 K（2026-09-18）：⭐CLIP 文本塔当【固定锚点】—— 规则五、2/十(二)点名的「表征约束 / 约束表征漂移」。
    #   500 个类的文本原型 T (K,512)（68 号从训练图元数据里的拉丁学名编码，缺名类用原版 CLIP 图像类均值），
    #   训练时多一项 λ·soft_ce(s·cos(f, T), y)：把微调后的视觉特征往 CLIP 原生的图文对齐空间拉，
    #   = 五、3「最大程度保留和利用 CLIP 预训练先验」。⛔只在训练损失里；推理仍走原 head_logits，单模型。
    ap.add_argument("--text-anchor", type=str, default=None,
                    help="文本原型 npz（键 T (K,512)），如 cache\\text_protos.npz。需 --cosine-head（共用 s）")
    ap.add_argument("--text-lambda", type=float, default=0.3, help="文本锚点损失的权重 λ")
    # ══════════════════════════════════════════════ 实验 CT · Co-teaching（2026-09-24 新增）
    ap.add_argument("--co-teaching", action="store_true",
                    help="⭐同时训两份网络，每一步各自挑出『损失最小的那部分样本』喂给**对方**更新。"
                         "机制：错标样本在训练早期损失偏高，小损失子集里错标比例更低；两份网络"
                         "的错误不同步，交叉筛选能避免单网络自我确认（这正是它比 self-paced 强的地方）。"
                         "⛔产出仍是**单一模型**（只交 A 网），推理一次前向 —— 不是多模型集成。")
    ap.add_argument("--ct-forget-rate", type=float, default=0.15,
                    help="最终丢弃比例 τ：每批丢掉损失最大的 τ 比例样本。"
                         "⚠⚠**0.15 不是按本数据集校准的**，是 co-teaching 原论文的默认值，"
                         "那边的实验设定是 20%%~45%% 噪声。本赛题实测错标率只有 1%%上下"
                         "（初赛人工数 1~1.5%%；复赛阶段0 跨类冲突 2.61%% 减去同物种团 1.60%% ≈ 1.0%%）"
                         "⇒ τ=0.15 是靶子的十几倍，丢掉的绝大部分是干净数据。\n"
                         "默认仍是 0.15，是为了 A1/A2 能逐位复现。09-24 选 0.15 的理由是『单变量对照要读得出结果』："
                         "τ 贴合噪声时每批只丢 1~2 张，单独测大概率落在误差棒内。\n"
                         "⭐A3（09-29）改用 0.05：A3 不是单变量对照，读不读得出单项效果不再是约束；"
                         "而加上第二轮伪标签后，要防的错标 = 训练集 1~1.5%% + 伪标签里判错的那部分"
                         "（2.5 万张，按八五成准算约 2%%），合起来 3~4%%，τ≈噪声率正是 co-teaching 原论文的设定。"
                         "丢多了的代价是细粒度最值钱的近亲难样本（A2 在自己训过的图上只有 81%% 与标签一致）。\n"
                         "⚠还有一种本赛题从没量过的噪声：规则提的『弱相关标注』（栖息地照片标成鸟）。"
                         "上面 1%% 的口径只覆盖『重复图跨类冲突』，如果弱相关那一类很多，整个判断要重审。")
    ap.add_argument("--ct-warmup-epochs", type=float, default=1.0,
                    help="前多少轮不筛（τ 从 0 线性爬到 --ct-forget-rate）。"
                         "⭐必须要有：训练初期两份网络都还没学会，此时的『小损失』只反映"
                         "样本简单不简单，与标签对不对无关，提前筛等于按难度删数据。")
    ap.add_argument("--ct-head-noise", type=float, default=0.02,
                    help="B 网分类头初始化的扰动幅度（相对 W 的元素尺度）。"
                         "⚠经典 co-teaching 靠**两个随机初始化**制造分歧，但本赛题规定骨干必须是"
                         "OpenAI 的同一份 CLIP 权重 ⇒ 两网起点相同，分歧只能来自 DropPath/Mixup 的"
                         "随机性和这里的头扰动。**这是本实现相对原论文最弱的一环**，"
                         "看 history.json 里的 ct_divergence（两网预测分歧率）判断它有没有真的分开。")
    # ══════════════════════════════════════════════ A3（2026-09-29）：去重名单 + 小类保护，默认全关（A1/A2 逐位复现）
    ap.add_argument("--exclude-list", type=str, default=None,
                    help="76 号产出的排除名单（每行 '类文件夹/文件名'）。只从训练索引里去掉，"
                         "验证子集不动（本地尺子的权重数组按验证子集对齐）")
    ap.add_argument("--small-oversample", type=float, default=0.0,
                    help="小类过采样的倍数上限，0=关。小类每张真标签图每轮重复 min(上限, √(中位数/该类图数)) 次"
                         "（平方根采样，综述 3.1.1；小数部分按固定种子随机取整）。"
                         "⚠上限别太大：同几张图看太多遍会连背景一起背下来（综述 [55]）")
    ap.add_argument("--small-frac", type=float, default=0.5,
                    help="图数 < 中位数 × 这个比例的类算小类（与 76 号同一口径，按相对量定，换阶段不用改）")
    ap.add_argument("--ct-protect-small", action="store_true",
                    help="co-teaching 筛选时小类样本一律保留，不参与丢弃。"
                         "小类见得少、损失天然偏高，全局小损失准则会先丢它们（主文档十四）")
    ap.add_argument("--no-foreach", action="store_true",
                    help="AdamW 逐个参数更新（foreach=False），⛔不改公式、只省显存。09-29 A3 冒烟实踩："
                         "默认的批量实现会把全部参数的二阶矩一次性复制一份做开方，两网时峰值多占几百 MB，"
                         "桌面上开着视频/浏览器时第一步就爆显存")
    ap.add_argument("--no-test-ruler", action="store_true",
                    help="不读由测试集特征算出的本地尺子文件（17 号 val_weights、16 号 leakage_scores），"
                         "本地指标只用验证子集的原始准确率。⭐完全不碰测试集的训练必须开（规则十一(二)1）")
    ap.add_argument("--no-pin-memory", action="store_true",
                    help="训练读图器不用锁页内存，⛔不改任何超参。09-29 A3 续跑测试实踩：物理内存只剩几 GB 时 "
                         "CUDA 申请锁页内存失败（fatal: Memory allocation failure → CUDA unknown error）")
    ap.add_argument("--grad-ckpt", action="store_true",
                    help="⭐梯度检查点：用约 30%% 的前向重算换掉大部分激活显存，⛔不改任何超参、结果逐位等价。"
                         "09-24 冒烟实踩：co-teaching 两网 @448 bs32 在本机 OOM（崩在 A 网带梯度的前向）")
    ap.add_argument("--aug-scale", type=float, default=0.3, help="随机裁剪面积下限")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--eval-workers", type=int, default=2)
    ap.add_argument("--loader-timeout", type=int, default=600,
                    help="训练读图器取一批图的超时（秒），0=不限。⭐09-24 冒烟实踩：本机内存墙下 4 个 worker "
                         "会在第一批图上卡死——不报错、GPU 空转，日志停在「开训」。有了超时，卡住会直接抛错退出，"
                         "⛔不会再静默挂一整夜。正常一批 32 张只要几秒，600 秒只在真卡住时才会触发")
    ap.add_argument("--head-init", type=str,
                    default=os.path.join(P.CACHE_DIR, "head_loss_sce.npz"))
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--pseudo", type=str, default=None,
                    help="30 号产出的 pseudo_labels.npz。⭐把测试集伪标签图并进训练集——"
                         "属于『对齐测试域』这一族（本地尺子对这一族天生是瞎的，因为本地验证集"
                         "自己就是训练集分布，尺子上根本没有这一格刻度）。⛔必须交榜裁决。")
    ap.add_argument("--pseudo-weight", type=float, default=0.5,
                    help="伪标签样本的损失权重。⚠别设 1.0，教师本身只有 62 分")
    ap.add_argument("--soft-pseudo", action="store_true",
                    help="④ 软伪标签：伪标签样本的监督目标用教师的完整概率分布（teacher_soft.npz）"
                         "替代 one-hot argmax。⭐单变量对照：其余（cosine/448/degrade/多尺度）不变")
    ap.add_argument("--pseudo-soft-temp", type=float, default=1.0,
                    help="软伪标签温度锐化：prob^(1/T) 后重新归一。默认 1.0=直接用教师原始分布")
    ap.add_argument("--full-data", action="store_true",
                    help="⭐用全部 103218 张训练（平时只用 80%%=82574 张）。"
                         "+25%% 数据是最可靠的一档收益，但**验证集就变成 in-sample 了**，"
                         "指标只能看训练是否正常，不能当泛化估计。"
                         "⚠只在配方已验证过之后用它收尾。")
    ap.add_argument("--seed", type=int, default=0,
                    help="训练随机种子（数据 shuffle/裁剪翻转/mixup/jpeg 退化）。默认 0 与历史一致；"
                         "重测训练方差时改它")
    ap.add_argument("--continue-run", type=str, default=None,
                    help="⭐断点续跑：给一个训练目录，接着往下训（写回同一目录）。"
                         "有 train_state.pt → **精确续跑**（权重/优化器/EMA/随机数/步数全恢复）；"
                         "没有 → **近似续跑**：从最后一个 ckpt_epNN.pt（co-teaching 另加 ckpt_b_epNN.pt）恢复两网权重，"
                         "学习率接着第 NN×每轮步数 走，AdamW 动量从零重建（约一千步恢复）。"
                         "⛔关键超参必须与原训练一致，否则拒绝续跑（换了超参就不叫续跑）。"
                         "09-24 实踩：A1 在第 2 轮中途整机卡死重启，只剩 ckpt_ep01")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    assert args.resolution % 32 == 0, "ViT-B/32 的分辨率必须是 32 的倍数（336 不是！）"
    if args.margin > 0 and not (args.cosine_head or args.subcenters > 1):
        ap.error("--margin 必须配 --cosine-head 或 --subcenters：线性头的 logit 是 f@W.T+b，"
                 "不是 cosθ，从它上面减一个常数没有几何含义（也不等于 CosFace）")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    P.ensure_dirs()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    banner("28 微调 v2 @%dpx  %d轮  Mixup%.1f/CutMix%.1f  LLRD%.2f  EMA%.4f  退化%.0f%%"
           % (args.resolution, args.epochs, args.mixup, args.cutmix,
              args.llrd, args.ema_decay, 100 * args.degrade_p))

    # ---------------------------------------------------------------- 数据
    paths, labels, class_names = _ex.list_train_files()
    y = labels.astype(np.int64)
    K = len(class_names)
    itr, iva = train_test_split(np.arange(len(y)), test_size=VAL_FRAC,
                               random_state=SEED, stratify=y)
    if args.smoke:
        itr, iva = itr[:1500], iva[:800]
    if args.full_data and not args.smoke:
        itr = np.arange(len(y))          # ⚠验证集也进训练集了，下面的指标全部 in-sample
    # ── A3：去重名单（76 号）。只从训练索引里去掉，验证子集 iva 不动。
    all_keep = np.ones(len(y), bool)
    if args.exclude_list:
        with open(args.exclude_list, encoding="utf-8") as fx:
            ex_keys = {ln.strip().replace("\\", "/") for ln in fx if ln.strip()}
        rk = [os.path.basename(os.path.dirname(p)) + "/" + os.path.basename(p) for p in paths]
        all_keep = np.array([k not in ex_keys for k in rk], bool)
        n_hit = int((~all_keep).sum())
        if n_hit != len(ex_keys):
            raise SystemExit("⛔ 排除名单 %d 行，只对上训练集 %d 张：名单和当前数据不是同一份"
                             % (len(ex_keys), n_hit))
        n0 = len(itr)
        itr = itr[all_keep[itr]]
        print("  ⭐去重名单：排除 %d 张（训练索引 %d → %d），验证子集不动" % (n_hit, n0, len(itr)))
    # ── A3：小类按全量训练集（去重之后）的图数定，冒烟也用同一口径
    cls_cnt = np.bincount(y[all_keep], minlength=K)
    small_cls = cls_cnt < args.small_frac * np.median(cls_cnt)
    if args.small_oversample > 0 or args.ct_protect_small:
        print("  ⭐小类（< 中位数 %.0f × %.2f）%d 个：%s"
              % (np.median(cls_cnt), args.small_frac, int(small_cls.sum()),
                 " ".join("%s:%d" % (class_names[c], cls_cnt[c]) for c in np.where(small_cls)[0])))
    if args.small_oversample > 0:
        rs = np.random.RandomState(args.seed + 7)       # 独立的固定种子，不扰动主训练的随机数序列
        med = float(np.median(cls_cnt))
        extra = []
        for c in np.where(small_cls)[0]:
            r = min(args.small_oversample, math.sqrt(med / max(cls_cnt[c], 1)))
            for i in itr[y[itr] == c]:
                extra.extend([i] * (int(math.floor(r - 1.0)) + int(rs.rand() < (r - 1.0) % 1.0)))
        if extra:
            itr = np.concatenate([itr, np.asarray(extra, dtype=itr.dtype)])
        print("  ⭐小类过采样（上限 ×%.1f，平方根采样）：每轮多出 %d 张重复图"
              % (args.small_oversample, len(extra)))
    tr_p = [paths[i] for i in itr]
    va_p = [paths[i] for i in iva]
    ytr, yv = y[itr], y[iva]
    sw = np.ones(len(itr), np.float32)
    n_real = len(tr_p)          # ⭐真标签数量（伪标签会拼到 tr_p 末尾，用这个切分软标签）
    pseudo_soft = None
    print("  训练 %d / 验证 %d / 类别 %d  (seed=%d，与所有历史实验一致)"
          % (len(itr), len(iva), K, SEED))
    if args.full_data and not args.smoke:
        print("  ⚠⚠**全量数据模式**：验证集已并入训练，下面所有指标都是 in-sample，")
        print("     只能看'训练是否正常'，**不能当泛化估计**。真值只能靠交榜。")

    # ---------------------------------------------------------------- 伪标签
    if args.pseudo:
        pz = np.load(args.pseudo, allow_pickle=True)
        pp = [str(x) for x in pz["paths"]]
        py = pz["labels"].astype(np.int64)
        tr_p = tr_p + pp
        ytr = np.concatenate([ytr, py])
        # ⭐C1：若 30 号存了逐样本权重（类别内排名加权），用它；否则沿用统一权重。
        use_rank = ("sample_weight" in pz.files and "rank_weight" in pz.files
                    and bool(pz["rank_weight"]))
        if use_rank:
            pw = pz["sample_weight"].astype(np.float32)
            # ⭐⭐把**均值锁死**到 args.pseudo_weight：npz 只提供权重的「形状」
            #   （这一类内部谁更可信），「总影响力」始终由命令行这一个旋钮控制。
            #   ⛔不锁均值的话，改 w_min/w_max 会同时动"形状"和"总量"两个变量，
            #     A/B 出来的差值就无法归因——那正是我们反复栽跟头的那种设计。
            pw = pw * (args.pseudo_weight / max(float(pw.mean()), 1e-8))
            print("  ⭐伪标签权重：**类别内排名加权**，范围 [%.3f, %.3f]，均值锁死 %.2f"
                  % (pw.min(), pw.max(), pw.mean()))
        else:
            pw = np.full(len(py), args.pseudo_weight, np.float32)
            print("  伪标签权重：统一 %.2f" % args.pseudo_weight)
        sw = np.concatenate([sw, pw])
        print("  ⭐伪标签：并入 %d 张测试图（真标签权重为 1.0），训练集 → %d 张"
              % (len(pp), len(tr_p)))
        print("  ⛔提醒：本地尺子对伪标签**无效**（验证集仍是训练集分布，"
              "伪标签的作用是往测试集分布靠）→ 必须交榜裁决")
        # ------------------------------------------------------------ ④软伪标签
        if args.soft_pseudo:
            sz = np.load(os.path.join(P.CACHE_DIR, "teacher_soft.npz"), allow_pickle=True)
            prob_all = sz["prob"].astype(np.float32)          # (24967, K) 全测试图概率
            # teacher_soft 行序 = 全测试图序（list_test_files），与 pz["all_paths"] 对齐
            idx_of = {str(p): i for i, p in enumerate(pz["all_paths"])}
            soft_rows = np.stack([prob_all[idx_of[p]] for p in pp], 0).astype(np.float32)
            if abs(args.pseudo_soft_temp - 1.0) > 1e-8:
                soft_rows = soft_rows ** (1.0 / max(args.pseudo_soft_temp, 1e-3))
                soft_rows = soft_rows / soft_rows.sum(1, keepdims=True)
            agree = float((soft_rows.argmax(1) == py).mean())
            print("  ④软伪标签：teacher_soft 对齐 %d/%d 张，argmax×硬标签一致 %.4f"
                  % (len(pp), len(pp), agree))
            pseudo_soft = soft_rows

    if args.no_test_ruler:
        # 09-30：val_weights（17 号）与 leakage_scores（16 号）都是拿测试图的原版 CLIP 特征算的。
        #   它们不进损失，但每轮末按『分布匹配』分数决定存 EMA 还是当前权重 ⇒ 测试集间接参与了模型选择。
        #   完全合规的版本一个由测试集算出的文件都不读：两把尺子退化为验证子集的原始准确率。
        w = np.ones(len(iva), np.float32)
        mnl = np.ones(len(iva), bool)
        print("  ⭐不读任何由测试集算出的文件（val_weights / leakage_scores）：分布匹配 = 无泄漏 = 原始准确率，"
              "EMA/当前权重也按它选")
    else:
        wz = np.load(os.path.join(P.CACHE_DIR, "val_weights.npz"))
        w = wz["weight"][:len(iva)] if args.smoke else wz["weight"]
        lk = np.load(os.path.join(P.CACHE_DIR, "leakage_scores.npz"))
        mnl = (lk["val_maxsim"][:len(iva)] if args.smoke else lk["val_maxsim"]) < 0.90
    nz = np.load(os.path.join(P.CACHE_DIR, "noise_scores.npz"), allow_pickle=True)
    sv = nz["suspicion"][iva]
    m60 = sv <= np.quantile(sv, 0.60)
    print("  尺子  分布匹配(主) + 无泄漏子集 %d张 + 干净60%%" % mnl.sum())
    # ⛔2026-09-08 拆除：这里原来印的是「真实增益 ≈ 分布匹配Δ × 100 × 0.42（n=2 实测重现）」。
    #   该系数已被实测推翻，偏差 9.4 倍：据它预测 v2 配方 +0.95 分，真实榜分 +9.02 分。
    #   根因：n=2 定不出一条曲线是直的还是弯的，两个点的"重现"只是巧合的近邻。
    #   ⭐更深的根因：换算倍率根本不是常数，它取决于「这次进步里，死记硬背 vs 对齐测试域
    #     各占多大比例」——同一把本地尺子对前者夸大 3~4 倍，对后者缩小 10 倍（方向相反的两个病）。
    #   → 本地读数只许读**方向**，⛔一律不许换算成分数。判定必须交排行榜。
    print("  ⚠本地Δ只读方向，⛔不可换算成榜分（旧的×0.42系数已实测作废，偏差9.4倍）")

    # ---------------------------------------------------------------- 模型
    import clip
    model, _ = clip.load(P.CLIP_MODEL_NAME, device="cpu", download_root=P.CLIP_DOWNLOAD_ROOT)
    model.float()
    g_old, g_new = _ft.interpolate_pos_embed(model, args.resolution)
    visual = model.visual
    print("  位置编码 %d×%d=%d格 → %d×%d=%d格" % (g_old, g_old, g_old ** 2,
                                                  g_new, g_new, g_new ** 2))

    hd = np.load(args.head_init)
    W = torch.tensor(hd["W"], dtype=torch.float32, requires_grad=True, device=device)
    b = torch.tensor(hd["b"], dtype=torch.float32, requires_grad=True, device=device)

    # ------------------------------------------------------------ ③b 子中心分类头
    # 每类 N 个子中心：W (K,d) → (K,N,d)。起步=复制原中心 + 小扰动，防止 N 个中心
    # 塌缩成同一个（那样就退化回 ③a 单中心）。扰动量 0.3 ≈ W 元素尺度(2.3)的 13%，
    # 使子中心初始余弦差异 ~1% 足够分化。detach.clone 后才是可训练的 leaf。
    if args.subcenters > 1:
        W = (W[:, None, :].repeat(1, args.subcenters, 1)
             + torch.randn(W.shape[0], args.subcenters, W.shape[1], device=device) * 0.3)
        W = W.contiguous().clone().detach().requires_grad_(True)
        print("  ⭐子中心头 ③b：W (K,d)→(K,N,d)=(%d,%d,%d)，起步=复制原中心+扰动0.3"
              % (W.shape[0], W.shape[1], W.shape[2]))

    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        sd = dict(ck["visual"])
        k = "positional_embedding"
        if k in sd and sd[k].shape[0] != g_new ** 2 + 1:
            sd[k] = _ft._resize_pos_embed(sd[k], g_new)
            print("  位置编码：ckpt %s → 插值迁移" % (tuple(ck["visual"][k].shape),))
        visual.load_state_dict(sd, strict=False)
        if args.subcenters > 1 and ck["W"].ndim == 2:
            # 从单中心 ckpt 恢复 → 扩成 N 子中心 + 扰动（形状必须匹配 W.data）
            Wc = ck["W"].to(device)[:, None, :].repeat(1, args.subcenters, 1)
            W.data = Wc + torch.randn_like(Wc) * 0.3
        else:
            W.data = ck["W"].to(device)
        b.data = ck["b"].to(device)
        print("  已从 %s 第%s轮恢复" % (os.path.basename(os.path.dirname(args.resume)),
                                        ck.get("epoch")))
    else:
        print("  从原版 CLIP + LP 头起手（LP-FT）")

    # -------------------------------- ③a 余弦 / ③b 子中心 共用可学习标量 s
    # lg = s·(f̂ @ Ŵ.T)。f 在前向里本来就会 L2 归一，这里把 W 逐行/逐子中心归一 + 学习标量 s，
    # 去掉 |W| 的类频率偏置，判决只看方向。s 初值取 W 范数均值，让余弦头起步量级 ≈ 旧线性头。
    # ③b 是余弦头的多中心推广：唯一变量=单中心→N 子中心，故同样需要 s（叠加在 ③a 上）。
    if args.cosine_head or args.subcenters > 1:
        s = torch.tensor(W.detach().float().norm(dim=-1).mean(), device=device,
                         requires_grad=True)
        tag = "子中心 ③b" if args.subcenters > 1 else "余弦头 ③a"
        print("  ⭐%s：s 初值=%.3f（W 范数均值），偏置 b 弃用" % (tag, float(s)))
        if args.margin > 0:
            print("  ⭐实验 U · CosFace 间隔 m=%.3f：训练损失用 s·(cosθ − m·y)"
                  "  ⇒ 起步时间隔在 logit 上值 %.2f（s=%.3f × m）"
                  % (args.margin, float(s) * args.margin, float(s)))
            print("     ⛔只作用于**训练损失**；评估/出包走原 head_logits（不减间隔），"
                  "口径与 26/45/58/59 逐字一致")
    else:
        s = None

    # ⭐实验 K：文本锚点原型。固定、不训练、不进 optimizer；与余弦头共用可学习标量 s（量级对齐）。
    Tt = None
    if args.text_anchor:
        if s is None:
            raise SystemExit("⛔ --text-anchor 需要 --cosine-head：锚点损失用 s·cos(f,T)，线性头没有 s")
        tz = np.load(args.text_anchor, allow_pickle=True)
        Tt = torch.tensor(np.asarray(tz["T"], dtype=np.float32), device=device)
        Tt = Tt / Tt.norm(dim=1, keepdim=True).clamp_min(1e-12)
        if Tt.shape != (K, W.shape[-1]):
            raise SystemExit("⛔ 文本原型形状 %s ≠ (K=%d, d=%d)" % (tuple(Tt.shape), K, W.shape[-1]))
        src = tz["source"].tolist() if "source" in tz.files else []
        n_txt = sum(1 for x in src if x == "text")
        with torch.no_grad():
            Wn0 = W.detach().float()
            Wn0 = Wn0 / Wn0.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            align = float((Wn0 * Tt).sum(-1).mean()) if Wn0.ndim == 2 else float("nan")
        print("  ⭐实验 K · 文本锚点：%d 类原型（文本 %d / 图像均值回退 %d），λ=%.2f，"
              "起步时 头原型↔文本原型 逐类余弦均值 %.4f（⚠只读方向：越低说明微调后特征空间离文本越远）"
              % (K, n_txt, K - n_txt, args.text_lambda, align))
        print("     ⛔只作用于训练损失；评估/出包走原 head_logits，单模型单推理流程")

    def head_logits(f, Wt, bt, st):
        """统一算 logits。子中心时 Wt 为 (K,N,d)、取 max cosine；cosine 时单中心；否则线性 + bt。"""
        if args.subcenters > 1:
            Kc, Nc = Wt.shape[0], Wt.shape[1]
            Wn = Wt / Wt.norm(dim=-1, keepdim=True).clamp_min(1e-12)   # (K,N,d)
            sims = f @ Wn.reshape(Kc * Nc, -1).T                        # (B, K*N)
            return st * sims.view(-1, Kc, Nc).max(dim=2).values         # (B, K)
        if args.cosine_head:
            Wn = Wt / Wt.norm(dim=1, keepdim=True).clamp_min(1e-12)
            return st * (f @ Wn.T)
        return f @ Wt.T + bt

    for p_ in visual.parameters():
        p_.requires_grad = True

    # ⭐硬冻结底部若干层（44 号诊断的直接落地）
    # ⛔LLRD 替代不了硬冻结，这不是"两种写法"而是**两种能力**：
    #   权重更新量 ≈ lr × 梯度 × 步数。LLRD 只按深度缩 lr，**管不住梯度**。
    #   44 号实测（ckpt=20260906_110505_v2_320 ep02，LLRD=0.85）：
    #     LLRD 给底层的 lr 比 = 0.85^13 = 0.121×（自以为压到 1/8.3）
    #     实测底层/顶层权重漂移比 = 0.419×  → **是 lr 该给的 3.47 倍**
    #   → 底层梯度幅值本身就比顶层大 3.5 倍，把 lr 调小只是缓解，不是关闭。
    #   硬冻结 lr=0 → 漂移**严格为 0**，与梯度多大无关。这是 LLRD 够不着的解。
    # ⭐另一层价值：冻住的层不进反传图 → 省显存省时间，是上 448px 的前置条件。
    if args.freeze_first_n > 0:
        nb = len(visual.transformer.resblocks)
        frozen = 0
        for nm, p_ in visual.named_parameters():
            if layer_id_of(nm, nb) < args.freeze_first_n:   # ⭐与 LLRD 共用同一个口径函数
                p_.requires_grad_(False)
                frozen += p_.numel()
        tot = sum(p_.numel() for p_ in visual.parameters())
        print("  ⭐硬冻结 layer_id < %d（0=输入端, 1..%d=block00..%02d）："
              "冻住 %.1fM/%.1fM 参数 (%.0f%%)"
              % (args.freeze_first_n, nb, nb - 1,
                 frozen / 1e6, tot / 1e6, 100.0 * frozen / max(tot, 1)))
        # build_param_groups 里有 `if not p_.requires_grad: continue`，
        # 所以冻住的参数不会进 optimizer，LLRD 会自动跳过它们。

    # ⭐实验 J：随机深度。放在 resume 之后——DropPath 不带任何参数，
    #   state_dict 逐键不变，所以放前放后都能读老 ckpt，这里放后面只是为了读代码顺。
    if args.drop_path > 0:
        rates = apply_drop_path(visual, args.drop_path)
        print("  ⭐DropPath 随机深度：%d 层 %.3f → %.3f（逐层线性递增）"
              % (len(rates), rates[0], rates[-1]))
    if args.grad_ckpt:
        apply_grad_ckpt(visual)       # ⚠必须在下面 B 网 deepcopy 之前，两网才同构
        print("  ⭐梯度检查点：12 个 block 逐个重算，激活显存 O(层数)→O(1)；⛔超参不变、结果逐位等价")
    visual = visual.to(device)

    # ══════════════════════════════════════════════ 实验 CT · 建 B 网（co-teaching 的第二份）
    # ⭐放在这里的理由：必须在 freeze / drop_path / to(device) **全部生效之后**深拷贝，
    #   两网才逐位同构（否则 B 网可能没挂 DropPath，分歧来源就不干净了）。
    # ⚠deepcopy 安全性：_dp_block_forward 是**模块级**函数 + MethodType 绑定（见 :293 的注释），
    #   EMA 从第一天起就在 deepcopy 同一个对象，这条路径已被全项目验证过。
    visual_b = W_b = b_b = s_b = None
    if args.co_teaching:
        if args.subcenters > 1:
            raise SystemExit("⛔ --co-teaching 未适配子中心头（③b 已判死，不该再组合）")
        visual_b = copy.deepcopy(visual)
        Wb0 = W.detach().clone()
        if args.ct_head_noise > 0:
            # 扰动幅度按 W 自身的元素尺度算，换分辨率/换类别数都不用重调这个旋钮
            Wb0 = Wb0 + torch.randn_like(Wb0) * (args.ct_head_noise * Wb0.abs().mean())
        W_b = Wb0.contiguous().requires_grad_(True)
        b_b = b.detach().clone().requires_grad_(True)
        s_b = s.detach().clone().requires_grad_(True) if s is not None else None
        n_par = sum(p_.numel() for p_ in visual_b.parameters())
        print("  ⭐实验 CT · Co-teaching：已建 B 网（骨干 %.1fM 参数，与 A 网同起点）" % (n_par / 1e6))
        print("     τ=%.3f（最终丢弃比例），前 %.1f 轮线性爬升，头扰动 %.3f"
              % (args.ct_forget_rate, args.ct_warmup_epochs, args.ct_head_noise))
        print("     ⭐筛选在**未混合**的干净批上做：Mixup 混完的样本损失反映的是"
              "『这一对混得好不好』，⛔不是『这张图标签对不对』，拿它筛等于筛了个寂寞")
        print("     ⛔出包只用 A 网（ckpt_ep*.pt），B 网另存 ckpt_b_ep*.pt 仅供诊断 —— "
              "单一模型、单一推理流程，合规于规则五、4")

    # ---------------------------------------------------------------- 变换
    norm = T.Normalize((0.48145466, 0.4578275, 0.40821073),
                       (0.26862954, 0.26130258, 0.27577711))
    _train_ops = [
        T.RandomResizedCrop(args.resolution, scale=(args.aug_scale, 1.0),
                            ratio=(0.75, 1.333),
                            interpolation=T.InterpolationMode.BICUBIC)]
    # ⭐实验 C：几何 TrivialAugment。复用 torchvision 的算子实现，只把算子空间砍成几何六件：
    #   Identity / ShearX / ShearY / TranslateX / TranslateY / Rotate（幅度上限照 torchvision 默认：
    #   shear 0.99、平移 32px、旋转 135°——TrivialAugment 的定义就是"随机算子 × 随机幅度"，⛔不调）。
    #   fill = CLIP 均值色（0~255），空出来的边角填数据集平均色而不是黑色。
    if args.aug_geo:
        _train_ops.append(GeoTrivialAugment(interpolation=T.InterpolationMode.BILINEAR,
                                            fill=[int(round(255 * m)) for m in (0.48145466, 0.4578275, 0.40821073)]))
    _train_ops += [T.RandomHorizontalFlip(), T.ToTensor(), norm]
    # ⭐实验 J：RandomErasing 必须排在 norm **之后**（它擦的是张量不是 PIL 图）。
    #   填充值取 0 = 归一化之后的数据集均值色，不引入新的颜色分布。
    #   ⛔仍然不加颜色抖动：细粒度鸟类的颜色本身就是判别信号，那是另一个变量，要单独测。
    if args.random_erasing > 0:
        _train_ops.append(T.RandomErasing(p=args.random_erasing,
                                          scale=(0.02, 0.20), ratio=(0.3, 3.3), value=0))
    train_tf = T.Compose(_train_ops)
    eval_tf = T.Compose([
        T.Resize(args.resolution, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(args.resolution), T.ToTensor(), norm])
    print("  增强  RandomResizedCrop(%d, scale=%.2f~1.0)%s+HFlip+JPEG退化%.0f%%%s  ⛔无颜色抖动"
          % (args.resolution, args.aug_scale,
             "+⭐几何TrivialAugment(shear/translate/rotate)" if args.aug_geo else "",
             100 * args.degrade_p,
             ("+RandomErasing%.0f%%" % (100 * args.random_erasing)) if args.random_erasing > 0 else ""))
    if args.degrade_fixed:
        print("  ⭐JPEG 退化参数钉死：(长边500, q75) —— 测试集精确规格 100% 对齐")

    # ── 实验 C2：载入训练集自蒸馏的教师分布（mmap），按【文件名】对齐到 tr_p 的前 n_real 个真样本
    kd_arr = None
    if args.distill:
        kd_prob = np.load(args.distill + "_prob.npy", mmap_mode="r")          # (N_t, K) float16
        with open(args.distill + "_paths.txt", encoding="utf-8") as fk:
            kd_paths = [ln.strip() for ln in fk if ln.strip()]
        if kd_prob.shape != (len(kd_paths), K):
            raise SystemExit("⛔ 教师文件形状 %s 与 paths %d / K %d 不符" % (kd_prob.shape, len(kd_paths), K))
        row_of = {os.path.basename(p): j for j, p in enumerate(kd_paths)}
        idx = np.array([row_of.get(os.path.basename(p), -1) for p in tr_p[:n_real]], dtype=np.int64)
        miss = int((idx < 0).sum())
        if miss:
            raise SystemExit("⛔ 教师分布缺 %d/%d 张训练图，⛔不许带着缺口训" % (miss, n_real))
        kd_arr = np.asarray(kd_prob[idx], dtype=np.float32)                   # (n_real, K)，~200MB
        if abs(args.distill_temp - 1.0) > 1e-8:
            kd_arr = kd_arr ** (1.0 / max(args.distill_temp, 1e-3))
        kd_arr = kd_arr / kd_arr.sum(1, keepdims=True).clip(1e-8)
        agree = float((kd_arr.argmax(1) == ytr[:n_real]).mean())
        print("  ⭐实验 C2 · 训练集自蒸馏：教师分布对齐 %d/%d 张，α=%.2f T=%.2f，"
              "教师 argmax 与真标签一致 %.4f（⚠这是 in-sample，只读方向）"
              % (n_real, n_real, args.distill_alpha, args.distill_temp, agree))
        print("     ⛔只作用于训练图；测试图从未进过训练（十一(二)1）")

    loader = DataLoader(TrainDS(tr_p, ytr, train_tf, args.degrade_p, args.resolution, sw,
                                degrade_fixed=args.degrade_fixed,
                                soft=pseudo_soft, n_real=n_real, K=K, kd=kd_arr),
                        batch_size=args.batch_size, shuffle=True, drop_last=True,
                        num_workers=args.num_workers,
                        pin_memory=(device == "cuda" and not args.no_pin_memory),
                        # ⛔不常驻（09-24 实踩）：常驻的话轮末评估再开 2 个 = 同时 4 个 worker，
                        #   本机内存墙下直接 `页面文件太小`。每轮开头重启 worker 只多花十几秒；
                        #   只影响读图并行度，⛔不是超参
                        persistent_workers=False,
                        timeout=(args.loader_timeout if args.num_workers > 0 else 0))
    vloader = DataLoader(EvalDS(va_p, eval_tf, args.resolution),
                         batch_size=args.batch_size, shuffle=False,
                         num_workers=args.eval_workers, pin_memory=False,
                         # ⭐评估读图器同样要超时：09-24 评估 worker 启动时死于「页面文件太小」，
                         #   主进程没超时就一直干等，日志停住、不报错也不退出
                         timeout=(args.loader_timeout if args.eval_workers > 0 else 0))

    # ---------------------------------------------------------------- 优化器
    pg, info = build_param_groups(visual, args.lr_backbone, args.llrd, args.weight_decay)
    head_params = [W, s] if (args.cosine_head or args.subcenters > 1) else [W, b]
    pg.append({"params": head_params, "lr": args.lr_head, "weight_decay": 0.0})
    opt = torch.optim.AdamW(pg, foreach=(False if args.no_foreach else None))
    base_lrs = [g["lr"] for g in opt.param_groups]
    print("  LLRD  顶层 %.2e → 底层 %.2e （衰减 %.2f，共 %d 层）"
          % (info[max(info)], info[min(info)], args.llrd, len(info)))

    spe = len(loader)
    total = spe * args.epochs
    warm = max(int(total * args.warmup_frac), 1)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    ema_v = EMA(visual, args.ema_decay)
    ema_W, ema_b = W.detach().clone(), b.detach().clone()
    ema_s = s.detach().clone() if (args.cosine_head or args.subcenters > 1) else None

    # ── 实验 CT：B 网的优化器/调度/EMA，与 A 网逐项同构（⛔别给 B 网另设超参，
    #    那会多出一个说不清的变量；co-teaching 的分歧只该来自随机性与交叉筛选）
    opt_b = base_lrs_b = scaler_b = ema_vb = ema_Wb = ema_bb = ema_sb = None
    if args.co_teaching:
        pg_b, _ = build_param_groups(visual_b, args.lr_backbone, args.llrd, args.weight_decay)
        head_params_b = [W_b, s_b] if args.cosine_head else [W_b, b_b]
        pg_b.append({"params": head_params_b, "lr": args.lr_head, "weight_decay": 0.0})
        opt_b = torch.optim.AdamW(pg_b, foreach=(False if args.no_foreach else None))
        base_lrs_b = [g["lr"] for g in opt_b.param_groups]
        scaler_b = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
        ema_vb = EMA(visual_b, args.ema_decay)
        ema_Wb, ema_bb = W_b.detach().clone(), b_b.detach().clone()
        ema_sb = s_b.detach().clone() if args.cosine_head else None
        if device == "cuda":
            free_b, tot_b = torch.cuda.mem_get_info()
            print("  ⚠显存：两网 + 两份 AdamW 状态 + 两份 EMA 影子 ≈ 3.5 GB 常驻，"
                  "当前空闲 %.1f/%.1f GB" % (free_b / 1e9, tot_b / 1e9))
            print("     ⭐A/B 的前向反向是**串行**的（不是同时），峰值只多一份激活；"
                  "若仍 OOM，先降 --batch-size 到 16，⛔别动 lr")
    print("  计划  %d步/轮 × %d轮 = %d步，warmup %d步 + 余弦退火" % (spe, args.epochs, total, warm))

    # ⭐把被扫描的变量写进目录名：三条臂只差 freeze 深度，光看时间戳分不清谁是谁。
    #   （9-06 吃过亏：三个同名的 ft384_ep2_notta 目录，只能靠 md5 事后辨认。）
    # ══════════════════════════════════════════════ 断点续跑（09-24 新增）
    #   这些字段决定了训练轨迹，续跑时⛔必须与原训练逐项相同；worker 数、超时、梯度检查点只影响速度/显存，不查。
    _SAME = ("resolution", "batch_size", "lr_backbone", "llrd", "freeze_first_n", "lr_head",
             "weight_decay", "warmup_frac", "mixup", "cutmix", "mix_prob", "label_smoothing",
             "ema_decay", "drop_path", "random_erasing", "degrade_p", "degrade_fixed", "cosine_head",
             "subcenters", "margin", "aug_geo", "distill", "distill_alpha", "distill_temp",
             "text_anchor", "text_lambda", "aug_scale", "head_init", "resume", "pseudo",
             "pseudo_weight", "soft_pseudo", "pseudo_soft_temp", "full_data", "seed",
             "co_teaching", "ct_forget_rate", "ct_warmup_epochs", "ct_head_noise", "epochs",
             "exclude_list", "small_oversample", "small_frac", "ct_protect_small")
    resume = None
    if args.continue_run:
        rdir = args.continue_run if os.path.isdir(args.continue_run) else os.path.dirname(args.continue_run)
        rdir = rdir if os.path.isabs(rdir) else os.path.join(P.PROJECT_ROOT, rdir)
        with open(os.path.join(rdir, "history.json"), encoding="utf-8") as fh:
            hj = json.load(fh)
        old = hj["args"]
        diff = [k for k in _SAME if k in old and old[k] != getattr(args, k)
                and not (k == "epochs" and args.smoke)]          # 冒烟的续跑测试要多跑一轮，只在冒烟下放行
        if diff:
            raise SystemExit("⛔ 续跑拒绝：这些超参与原训练不同 %s\n   原：%s\n   现：%s"
                             % (diff, [old[k] for k in diff], [getattr(args, k) for k in diff]))
        stp = os.path.join(rdir, "train_state.pt")
        if os.path.exists(stp):
            # ── 精确续跑：mmap 读，⛔别一次性把 2.8GB 全读进内存（本机内存墙）
            st = torch.load(stp, map_location="cpu", weights_only=False, mmap=True)
            nets = [("A", visual, W, b, s, opt, scaler, ema_v, ema_W, ema_b, ema_s)]
            if args.co_teaching:
                nets.append(("B", visual_b, W_b, b_b, s_b, opt_b, scaler_b, ema_vb, ema_Wb, ema_bb, ema_sb))
            for key, vis_, W_, b_, s_, opt_, sc_, ev_, eW_, eb_, es_ in nets:
                d = st[key]
                vis_.load_state_dict(d["visual"])
                with torch.no_grad():
                    W_.copy_(d["W"].to(device)); b_.copy_(d["b"].to(device))
                    if s_ is not None:
                        s_.copy_(d["s"].to(device))
                    ev_.shadow.load_state_dict(d["ema_visual"])
                    eW_.copy_(d["ema_W"].to(device)); eb_.copy_(d["ema_b"].to(device))
                    if es_ is not None:
                        es_.copy_(d["ema_s"].to(device))
                opt_.load_state_dict(d["opt"])        # ⚠放在 cpu 读：AdamW 的 step 计数需留在 CPU，其余状态自动搬上 GPU
                sc_.load_state_dict(d["scaler"])
            rg = st["rng"]
            torch.set_rng_state(rg["torch"].cpu().clone())
            if device == "cuda" and rg.get("cuda") is not None:
                torch.cuda.set_rng_state_all([t.cpu().clone() for t in rg["cuda"]])
            np.random.set_state(rg["numpy"])
            random.setstate(rg["py"])
            resume = dict(mode="exact", ep=int(st["ep"]), step=int(st["step"]), hist=list(st["hist"]),
                          best=float(st["best"]), ref=dict(st["ref"]), log=list(st.get("resumed", [])))
            del st
        else:
            # ── 近似续跑：只有每轮存的权重（EMA 或当前，哪份好存哪份），优化器从零建
            eps = sorted(int(f[7:9]) for f in os.listdir(rdir) if re.fullmatch(r"ckpt_ep\d\d\.pt", f))
            if not eps:
                raise SystemExit("⛔ %s 里既没有 train_state.pt 也没有 ckpt_epNN.pt，无从续跑" % rdir)
            n_ep = eps[-1]
            pairs = [("ckpt_ep%02d.pt" % n_ep, visual, W, b, s, ema_v, ema_W, ema_b, ema_s)]
            if args.co_teaching:
                if not os.path.exists(os.path.join(rdir, "ckpt_b_ep%02d.pt" % n_ep)):
                    raise SystemExit("⛔ co-teaching 近似续跑需要 ckpt_b_ep%02d.pt（B 网），目录里没有" % n_ep)
                pairs.append(("ckpt_b_ep%02d.pt" % n_ep, visual_b, W_b, b_b, s_b, ema_vb, ema_Wb, ema_bb, ema_sb))
            for fn, vis_, W_, b_, s_, ev_, eW_, eb_, es_ in pairs:
                ck = torch.load(os.path.join(rdir, fn), map_location="cpu", weights_only=False, mmap=True)
                vis_.load_state_dict(ck["visual"])
                ev_.shadow.load_state_dict(ck["visual"])            # EMA 影子从同一份权重重新起步
                with torch.no_grad():
                    W_.copy_(ck["W"].to(device)); b_.copy_(ck["b"].to(device))
                    eW_.copy_(ck["W"].to(device)); eb_.copy_(ck["b"].to(device))
                    if s_ is not None and "s" in ck:
                        s_.fill_(float(ck["s"]))
                        es_.fill_(float(ck["s"]))
                del ck
            torch.manual_seed(args.seed + 1000 * n_ep)               # 续跑段换一个确定的种子，⛔别复用第 1 段的数据顺序
            np.random.seed(args.seed + 1000 * n_ep)
            random.seed(args.seed + 1000 * n_ep)
            hist0 = hj["history"]
            best0 = max((max(h.get("ema_matched", -1), h.get("raw_matched", -1)) for h in hist0), default=-1.0)
            resume = dict(mode="approx", ep=n_ep, step=n_ep * spe, hist=list(hist0), best=best0,
                          ref=dict(hj["ref"]), log=list(hj.get("resumed", [])))
        resume["log"].append(dict(mode=resume["mode"], after_epoch=resume["ep"], step=resume["step"],
                                  time=time.strftime("%Y-%m-%d %H:%M:%S")))
        print("  ⭐断点续跑【%s】：从第 %d 轮末接着训（第 %d 步），写回 %s"
              % ("精确" if resume["mode"] == "exact" else "近似", resume["ep"], resume["step"], rdir))
        if resume["mode"] == "approx":
            print("     ⚠近似续跑：两网权重与学习率进度已恢复；AdamW 动量从零重建（约一千步恢复），"
                  "EMA 影子从恢复的权重重新起步。已记入 history.json 的 resumed 字段")
    run = rdir if resume else os.path.join(P.RUNS_DIR, time.strftime("%Y%m%d_%H%M%S")
                       + "_v2_%d" % args.resolution
                       + ("_fz%d" % args.freeze_first_n if args.freeze_first_n else "")
                       + ("_cos" if args.cosine_head else "")
                       + ("_sc%d" % args.subcenters if args.subcenters > 1 else "")
                       + ("_m%03d" % round(args.margin * 100) if args.margin > 0 else "")
                       + ("_txt" if "text" in os.path.basename(args.head_init).lower() else "")
                       + ("_geo" if args.aug_geo else "")
                       + ("_kd%02d" % round(args.distill_alpha * 100) if args.distill else "")
                       + ("_ta%02d" % round(args.text_lambda * 100) if args.text_anchor else "")
                       + ("_ct%02d" % round(args.ct_forget_rate * 100) if args.co_teaching else "")
                       + ("_ps" if (args.co_teaching and args.ct_protect_small) else "")
                       + ("_dx" if args.exclude_list else "")
                       + ("_so%02d" % round(args.small_oversample * 10) if args.small_oversample > 0 else "")
                       + ("_clip0" if not args.resume else "")
                       + ("_dp%03d" % round(args.drop_path * 100) if args.drop_path > 0 else "")
                       + ("_re%02d" % round(args.random_erasing * 100) if args.random_erasing > 0 else "")
                       + ("_ep%d" % args.epochs if args.epochs != 2 else ""))
    os.makedirs(run, exist_ok=True)
    print("  产物  ->", run)

    @torch.no_grad()
    def evaluate(vis, Wm, bm, sm, tag):
        vis.eval()
        cs = []
        Wt = torch.tensor(Wm, device=device) if isinstance(Wm, np.ndarray) else Wm
        bt = torch.tensor(bm, device=device) if isinstance(bm, np.ndarray) else bm
        st = sm
        bar = tqdm(total=len(yv), desc=tag, unit="img", unit_scale=True,
                   dynamic_ncols=True, leave=False)
        for im, ix in vloader:
            im = im.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
                f = vis(im)
            f = f.float()
            f = f / f.norm(dim=1, keepdim=True).clamp_min(1e-12)
            cs.append(head_logits(f, Wt, bt, st).argmax(1).cpu().numpy() == yv[ix.numpy()])
            bar.update(im.shape[0])
        bar.close()
        c = np.concatenate(cs).astype(np.float64)
        return dict(matched=float((c * w).sum() / w.sum()),
                    noleak=float(c[mnl].mean()),
                    clean60=float(c[m60].mean()) if not args.smoke else float(c.mean()),
                    raw=float(c.mean()))

    banner("起点评估")
    if resume:
        ref = resume["ref"]
        print("  （续跑：沿用原训练的起点评估记录）")
    else:
        ref = evaluate(visual, W, b, s, "起点")
    print("  分布匹配 %.4f  无泄漏 %.4f  原始 %.4f"
          % (ref["matched"], ref["noleak"], ref["raw"]))

    # ══════════════════════════════════════════════ 实验 CT · co-teaching 的两个零件
    def ct_keep_n(bs, ep_f):
        """R(t)：丢弃比例 τ 在前 --ct-warmup-epochs 轮里线性爬到设定值，之后恒定。
        ⭐爬升而不是一上来就满档，是 co-teaching 原论文的设计：训练初期模型还没学会，
          此时的『小损失』只说明样本简单，与标签对不对无关 —— 提前筛 = 按难度删数据，
          那正是 19 号『用模型打分清洗』掉 1.3 分的死法。"""
        if not args.co_teaching:
            return bs
        tau = args.ct_forget_rate * min(ep_f / max(args.ct_warmup_epochs, 1e-9), 1.0)
        return max(1, int(round(bs * (1.0 - tau))))

    @torch.no_grad()
    def ct_probe(vis, Wt, bt, st, x, ysb):
        """在**未混合**的干净批上算逐样本损失 + 预测，作为筛选依据。
        ⚠用 eval 模式跑：DropPath 会让同一张图每次损失都不同，那是噪声不是信号，
          拿带随机性的损失排序等于掷骰子筛数据。"""
        was = vis.training
        vis.eval()
        with torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
            f = vis(x)
        vis.train(was)
        f = f.float()
        f = f / f.norm(dim=1, keepdim=True).clamp_min(1e-12)
        lg = head_logits(f, Wt, bt, st)
        return -(ysb * Fn.log_softmax(lg, dim=1)).sum(1), lg.argmax(1)

    def train_step(vis, Wt, bt, st, optm, sclr, hp, blrs, sel, xraw, ysraw, wtraw, scale_now):
        """对一份网络走一步。sel=None 表示整批都学（=关掉 co-teaching 时的原行为）。"""
        for g, bl in zip(optm.param_groups, blrs):
            g["lr"] = bl * scale_now
        if sel is not None:
            x, ysx, wtx = xraw[sel], ysraw[sel], wtraw[sel]    # 高级索引 = 复制，下面的 CutMix 原地改也安全
        elif args.co_teaching:
            x, ysx, wtx = xraw.clone(), ysraw, wtraw           # ⛔必须 clone：CutMix 原地改 x，否则 B 网拿到 A 网混过的图
        else:
            x, ysx, wtx = xraw, ysraw, wtraw
        x, ysx, wtx = mix_batch(x, ysx, wtx, args.mixup, args.cutmix, args.mix_prob)
        optm.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
            f = vis(x)
        f = f.float()
        f = f / f.norm(dim=1, keepdim=True).clamp_min(1e-12)
        lg = head_logits(f, Wt, bt, st)
        # ⭐实验 U · CosFace 加性余弦间隔：head_logits 出来已经是 s·cosθ，
        #   所以 s·(cosθ − m·y) ≡ lg − s·m·y，**一步减法**就够，⛔不必改 head_logits
        #   （那个函数同时是推理路径，改它会污染 26/45/58/59 的口径）。
        #   ⚠ysx 是 label smoothing + mixup 之后的软标签 (B,K)、行和=1 ⇒ 间隔按软权重
        #     线性摊到两个来源类上，**这正是选 CosFace 而不选 ArcFace 的理由**。
        lg_loss = lg - st * args.margin * ysx if args.margin > 0 else lg
        loss = soft_ce(lg_loss, ysx, wtx)
        # ⭐实验 K：文本锚点项。f 已 L2 归一、Tt 已 L2 归一 ⇒ s·cos；目标用同一个 ysx。
        if Tt is not None:
            loss = loss + args.text_lambda * soft_ce(st * (f @ Tt.T), ysx, wtx)
        sclr.scale(loss).backward()
        sclr.unscale_(optm)
        torch.nn.utils.clip_grad_norm_(list(vis.parameters()) + hp, 1.0)
        sclr.step(optm)
        sclr.update()
        # ⭐更新完立刻释放梯度：co-teaching 时 A 网的梯度（~350MB）不必占着显存等 B 网跑完。
        #   下一步开头本来就 zero_grad(set_to_none=True)，提前到这里数学上完全等价。
        optm.zero_grad(set_to_none=True)
        # ⭐混合后acc 仍用**未减间隔**的 lg 统计，否则这条过程判据与 L/J/M 不可比。
        return float(loss.detach()), lg

    small_t = (torch.as_tensor(small_cls, device=device)
               if (args.co_teaching and args.ct_protect_small) else None)
    banner("开训")
    hist, step, t0, best = [], 0, time.time(), -1.0
    resumed_log = []
    if resume:
        hist, step, best, resumed_log = resume["hist"], resume["step"], resume["best"], resume["log"]
    oh_off = args.label_smoothing / K
    visual.train()
    if args.co_teaching:
        visual_b.train()
    for ep in range((resume["ep"] + 1) if resume else 1, args.epochs + 1):
        cor = tot = 0
        ct_kept = ct_seen = ct_overlap = ct_div = ct_seln = ct_prot = 0   # 实验 CT 的过程读数
        bar = tqdm(total=spe, desc="ep%d/%d" % (ep, args.epochs), unit="it",
                   dynamic_ncols=True)
        for im, yy, wt, ysf, ykd in loader:
            sc = ((step + 1) / warm if step < warm else
                  0.5 * (1 + math.cos(math.pi * (step - warm) / max(total - warm, 1))))
            im = im.to(device, non_blocking=True)
            yy = yy.to(device, non_blocking=True)
            wt = wt.to(device, non_blocking=True).float()
            # one-hot + label smoothing，然后才做 mixup（顺序不能反）
            ys = torch.full((yy.size(0), K), oh_off, device=device)
            ys.scatter_(1, yy[:, None], 1.0 - args.label_smoothing + oh_off)
            if pseudo_soft is not None:
                is_soft = ysf.abs().sum(1) > 0      # 伪标签样本（软目标非零）
                if is_soft.any():
                    ys[is_soft] = ysf[is_soft].to(device)   # 用教师分布替代 one-hot
            # ⭐实验 C2：训练集自蒸馏——真样本的目标 = (1−α)·(one-hot+平滑) + α·教师分布。
            #   放在 mixup **之前**：mixup 会把两张图的目标按 λ 混，蒸馏目标也一并被混，顺序才对。
            if kd_arr is not None:
                has = ykd.abs().sum(1) > 0
                if has.any():
                    ys[has] = (1.0 - args.distill_alpha) * ys[has] + args.distill_alpha * ykd[has].to(device)
            # ══════════════════════ 实验 CT · 交叉筛选（⛔在 mix_batch **之前**）
            sel_a = sel_b = None
            if args.co_teaching:
                nk = ct_keep_n(im.size(0), step / max(spe, 1))
                la, pa = ct_probe(visual, W, b, s, im, ys)
                lb, pb = ct_probe(visual_b, W_b, b_b, s_b, im, ys)
                ct_div += int((pa != pb).sum())          # 两网分歧率：分歧塌到 0 = co-teaching 退化成自训练
                ct_seen += im.size(0)
                if nk < im.size(0):
                    # A3：小类样本的损失记成 −∞，排序时永远排在最前 ⇒ 一律保留
                    if small_t is not None:
                        prot = small_t[yy]
                        ct_prot += int(prot.sum())
                        la = la.masked_fill(prot, float("-inf"))
                        lb = lb.masked_fill(prot, float("-inf"))
                    # ⭐⭐交叉：**B 挑的样本给 A 学，A 挑的给 B 学**。
                    #   这一"交叉"是 co-teaching 的全部要害：若各自学自己挑的（self-paced），
                    #   网络会不断确认自己已有的偏见，错标样本一旦被判为"简单"就永远留在集合里。
                    sel_a = torch.argsort(lb)[:nk]
                    sel_b = torch.argsort(la)[:nk]
                    ct_kept += nk
                    # ⚠重合率的分母只能是**真正筛过的**样本数：warmup 期整批都留，
                    #   把它算进去会把重合率稀释成 100%，那条体检读数就废了
                    ct_seln += nk
                    ct_overlap += int(torch.isin(sel_a, sel_b).sum())
                else:
                    ct_kept += im.size(0)

            loss_v, lg = train_step(visual, W, b, s, opt, scaler, head_params,
                                    base_lrs, sel_a, im, ys, wt, sc)
            if args.co_teaching:
                train_step(visual_b, W_b, b_b, s_b, opt_b, scaler_b, head_params_b,
                           base_lrs_b, sel_b, im, ys, wt, sc)

            ema_v.update(visual)
            with torch.no_grad():
                ema_W.mul_(args.ema_decay).add_(W.detach(), alpha=1 - args.ema_decay)
                ema_b.mul_(args.ema_decay).add_(b.detach(), alpha=1 - args.ema_decay)
                if args.cosine_head or args.subcenters > 1:
                    ema_s.mul_(args.ema_decay).add_(s.detach(), alpha=1 - args.ema_decay)
            if args.co_teaching:
                ema_vb.update(visual_b)
                with torch.no_grad():
                    ema_Wb.mul_(args.ema_decay).add_(W_b.detach(), alpha=1 - args.ema_decay)
                    ema_bb.mul_(args.ema_decay).add_(b_b.detach(), alpha=1 - args.ema_decay)
                    if args.cosine_head:
                        ema_sb.mul_(args.ema_decay).add_(s_b.detach(), alpha=1 - args.ema_decay)

            step += 1
            yy_used = yy[sel_a] if sel_a is not None else yy
            cor += (lg.argmax(1) == yy_used).sum().item()
            tot += yy_used.numel()
            bar.update(1)
            if step % 20 == 0:
                bar.set_postfix_str("loss=%.3f 混合后acc=%.3f lr=%.1e%s"
                                    % (loss_v, cor / tot, opt.param_groups[-1]["lr"],
                                       ("" if not args.co_teaching else
                                        " 留%.0f%% 分歧%.1f%%"
                                        % (100.0 * ct_kept / max(ct_seen, 1),
                                           100.0 * ct_div / max(ct_seen, 1)))))
        bar.close()

        m_raw = evaluate(visual, W, b, s, "eval")
        m_ema = evaluate(ema_v.shadow, ema_W, ema_b, ema_s, "ema ")
        visual.train()
        use_ema = m_ema["matched"] > m_raw["matched"]
        m = m_ema if use_ema else m_raw
        d = m["matched"] - ref["matched"]
        dn = m["noleak"] - ref["noleak"]
        ct_stat = {}
        if args.co_teaching:
            visual_b.train()
            ct_stat = dict(ct_keep_rate=ct_kept / max(ct_seen, 1),
                           ct_divergence=ct_div / max(ct_seen, 1),
                           ct_sel_overlap=(ct_overlap / ct_seln) if ct_seln else float("nan"))
            if small_t is not None:
                ct_stat["ct_protected"] = ct_prot / max(ct_seen, 1)
                print("  [ep%02d·CT] 小类免筛样本占 %.2f%%" % (ep, 100 * ct_stat["ct_protected"]))
            # ⭐这三个数是 co-teaching 的体检表，⛔不是成绩：
            #   ct_divergence 两网预测分歧率 —— **塌到 0 就等于单网自训练**，交叉筛选失去意义
            #     （本实现两网同起点于 OpenAI CLIP，这是最可能出问题的地方，逐轮盯它）
            #   ct_sel_overlap 两网选中样本的重合率 —— 同理，接近 1 说明两网在挑同一批
            #   ct_keep_rate   实际保留比例 —— 对账 τ 爬升是否按计划生效
            print("  [ep%02d·CT] 保留 %.1f%% | 两网分歧 %.2f%% | 选样重合 %.1f%% %s"
                  % (ep, 100 * ct_stat["ct_keep_rate"], 100 * ct_stat["ct_divergence"],
                     100 * ct_stat["ct_sel_overlap"],
                     "⚠分歧过低=已退化成自训练" if ct_stat["ct_divergence"] < 0.02 else "✅"))
        hist.append(dict(epoch=ep, train_acc=cor / max(tot, 1), ema=bool(use_ema),
                         **ct_stat,
                         **{("ema_" + k): v for k, v in m_ema.items()},
                         **{("raw_" + k): v for k, v in m_raw.items()}))
        # ⛔括号里原本还印一个「→真实%+.2f分」（=d×100×0.42），已于 2026-09-08 拆除，理由见上文。
        print("  [ep%02d] 混合后acc %.4f | 当前 %.4f | EMA %.4f | ⭐取%s %.4f "
              "(Δ%+.4f 只读方向) | 无泄漏 %+.4f %s | %.0f分钟"
              % (ep, cor / max(tot, 1), m_raw["matched"], m_ema["matched"],
                 "EMA" if use_ema else "当前", m["matched"], d, dn,
                 "✅" if (d > 0) == (dn > 0) else "⚠", (time.time() - t0) / 60))

        src = ema_v.shadow if use_ema else visual
        blob = {"visual": src.state_dict(),
                "W": (ema_W if use_ema else W).detach().cpu(),
                "b": (ema_b if use_ema else b).detach().cpu(),
                "epoch": ep, "args": vars(args), "metrics": m,
                "used_ema": bool(use_ema)}
        if args.cosine_head or args.subcenters > 1:
            blob["s"] = float((ema_s if use_ema else s).detach().cpu())
        # ⛔⛔ ckpt_best.pt 在 --full-data 下是**陷阱**，不是保险：
        #   判据 m["matched"] 此时是 in-sample（验证集已并入训练），
        #   它随轮数**单调上升**（模型在背答案）→ "best" 恒等于"最后一轮"，
        #   而榜分实测**多训是负的**（320: ep02=73.64 → ep04=72.89）。
        #   → 它系统性地保存最差的那个，还顺手覆盖掉好的那个。
        #   9-07 的 448 训练差点因此丢掉 ep01（当时靠手动 cp 抢救）。
        # ⭐修法：每轮都单独落盘，选哪一轮交给榜分裁决，本地不做这个决定。
        torch.save(blob, os.path.join(run, "ckpt_ep%02d.pt" % ep))
        # ── 实验 CT：B 网单独落盘。⛔**只作诊断**（比如查两网是否真的分开了）。
        #   出包、提交、技术报告的数字一律只用 A 网 —— 规则五、4 要求单一模型/单一推理流程，
        #   把 A、B 的 logits 平均一下就是模型集成，⛔那是直接违规，别为了 0.2 分赌资格。
        if args.co_teaching and ep == args.epochs:
            blob_b = {"visual": (ema_vb.shadow if use_ema else visual_b).state_dict(),
                      "W": (ema_Wb if use_ema else W_b).detach().cpu(),
                      "b": (ema_bb if use_ema else b_b).detach().cpu(),
                      "epoch": ep, "args": vars(args), "metrics": m,
                      "used_ema": bool(use_ema), "co_teaching_net": "B",
                      "ct_stat": ct_stat}
            if args.cosine_head:
                blob_b["s"] = float((ema_sb if use_ema else s_b).detach().cpu())
            torch.save(blob_b, os.path.join(run, "ckpt_b_ep%02d.pt" % ep))
        if m["matched"] > best:
            best = m["matched"]
            torch.save(blob, os.path.join(run, "ckpt_best.pt"))
        with open(os.path.join(run, "history.json"), "w", encoding="utf-8") as fj:
            json.dump({"args": vars(args), "ref": ref, "history": hist, "resumed": resumed_log},
                      fj, ensure_ascii=False, indent=2)
        # ⭐完整训练状态，供精确续跑（~2.8GB，每轮覆盖）。⚠先写 .tmp 再 os.replace：
        #   写到一半断电的话，上一轮的状态文件还完整留着，⛔不会两头落空。
        st = {"ep": ep, "step": step, "hist": hist, "best": best, "ref": ref, "args": vars(args),
              "resumed": resumed_log,
              "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
                      "py": random.getstate(),
                      "cuda": torch.cuda.get_rng_state_all() if device == "cuda" else None}}
        nets = [("A", visual, W, b, s, opt, scaler, ema_v, ema_W, ema_b, ema_s)]
        if args.co_teaching:
            nets.append(("B", visual_b, W_b, b_b, s_b, opt_b, scaler_b, ema_vb, ema_Wb, ema_bb, ema_sb))
        for key, vis_, W_, b_, s_, opt_, sc_, ev_, eW_, eb_, es_ in nets:
            st[key] = {"visual": vis_.state_dict(), "W": W_.detach(), "b": b_.detach(),
                       "s": None if s_ is None else s_.detach(),
                       "opt": opt_.state_dict(), "scaler": sc_.state_dict(),
                       "ema_visual": ev_.shadow.state_dict(), "ema_W": eW_, "ema_b": eb_, "ema_s": es_}
        tmp_st = os.path.join(run, "train_state.pt.tmp")
        torch.save(st, tmp_st)
        os.replace(tmp_st, os.path.join(run, "train_state.pt"))
        del st

    banner("汇总")
    print("  %-6s %10s %10s %10s %10s" % ("epoch", "混合acc", "当前", "EMA", "Δ分布匹配"))
    for h in hist:
        print("  %-6d %10.4f %10.4f %10.4f %+10.4f"
              % (h["epoch"], h["train_acc"], h["raw_matched"], h["ema_matched"],
                 max(h["raw_matched"], h["ema_matched"]) - ref["matched"]))
    print("\n  ⚠混合后acc 会明显低于 24 号那版的 0.98——**那是 Mixup 的目的，不是退步**。")
    print("  ⭐下一步：python scripts/26_predict_finetuned.py --ckpt %s/ckpt_best.pt" % run)


if __name__ == "__main__":
    main()
