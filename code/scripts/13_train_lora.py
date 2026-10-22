# -*- coding: utf-8 -*-
"""
轴A 第一步：给冻结的 CLIP 骨干挂 LoRA 低秩补丁，和线性头一起微调（LP-FT 配方）。

═══════════════════════════════════════════════════════════════════════
一、为什么到这一步才动骨干
═══════════════════════════════════════════════════════════════════════

前面所有实验（损失函数/清洗/增强）都只动"怎么用 512 维特征"，骨干一直冻着，
所以能吃缓存，10~60 秒一次。一旦碰骨干，**缓存全部作废**，每轮要重推 10 万张图。
成本从 1 分钟涨到 1 小时，是 100 倍的差距 —— 所以不能撒网试，必须一次设计对。

那为什么还要动它？因为线性头有个天花板：它只能在 CLIP 排好的"座位表"上画刀。
如果 CLIP 把两个近缘物种放到几乎同一个座位（它是拿网上图文对训的，从没被要求
区分这两种鸟），那**任何刀都切不开**。实测到的 0467<->0156 对称双向混淆
（A->B .333, B->A .350）就是这个病：标签是对的，只是坐一块了。
LoRA 干的事就是**去改排座位的规则本身**，把挤在一起的两类拉开。

═══════════════════════════════════════════════════════════════════════
二、LoRA 是什么（数学只有一行）
═══════════════════════════════════════════════════════════════════════

ViT 每层都有 768x768 的权重矩阵 W。全参微调 = 直接改里面 59 万个数。
LoRA 说：别动 W，在旁边挂一个又瘦又长的补丁：

    W_新 = W(冻住)  +  B @ A        A: (r,768)   B: (768,r)   r=4

B@A 乘出来还是 768x768，能直接加上去；但自由参数只有 2*r*768 = 6144 个。
"低秩"的含义 = **这个补丁能表达的变化方向最多只有 r 个**。

⭐B 初始化为全 0 ==> 训练第 0 步时 B@A = 0，模型与原版 CLIP **逐位相同**。
  这正是 --selftest 的判据：第 0 步不相同就说明注入写错了。

⛔别把"参数少"当成"记不住噪声"的保证。数一下就知道：
    LoRA r=4 挂 12 层 q,v = 12*2*(4*768 + 768*4) = 147,456 个参数
    而现在的线性头       = 500*512 + 500        = 256,500 个参数
  **LoRA 比线性头还小**，可你已经实测过 C=1000 时线性头把 98.4% 的带噪标签
  全背下来了。==> 真正的旋钮从来不是参数量，是**正则化强度 + 训练时长**。
  这次实验的主角其实是"在它开始背噪声之前，我在哪一刻叫停"。

═══════════════════════════════════════════════════════════════════════
三、为什么头必须用现成的、不能随机初始化（LP-FT，arXiv 2202.10054）
═══════════════════════════════════════════════════════════════════════

把 CLIP 特征想成一位视力极好但不懂鸟类学的**观察员**，线性头是下判断的**裁判**。
随机初始化的裁判一开始纯属瞎猜 -> 损失巨大 -> 梯度一路往回"怪罪"观察员 ->
把 4 亿图文对练出来的好眼力搅坏（论文标题里 distort 的意思）。
先把裁判训好，他的抱怨才是可信信号。**我们那个 59 分的头正好就是第一阶段。**

═══════════════════════════════════════════════════════════════════════
四、⭐WiSE-FT 在 LoRA 下是免费的（本次设计最值钱的一条）
═══════════════════════════════════════════════════════════════════════

WiSE-FT(arXiv 2109.01903) 原做法：微调后权重与原始权重加权平均当保险丝
    θ_α = (1-α)·θ0 + α·θ_ft
对 LoRA 代入 θ_ft = θ0 + BA：
    θ_α = (1-α)θ0 + α(θ0 + BA) = θ0 + α·BA
==> **WiSE-FT 退化成"把补丁乘一个系数 α"**，训练只跑一次，α 事后随便扫。
配合本脚本每半轮存一次档，**一次训练换来一张 (存档时刻 x α) 的二维表**。
在"一次 1.5 小时、不能撒网"的约束下，这是能榨出的最大信息量。

═══════════════════════════════════════════════════════════════════════
五、哪些变量被刻意固定住了（不这样做就没法和 59.0 比）
═══════════════════════════════════════════════════════════════════════
  · 数据划分  seed=42 / 20% 验证，与 02/06 完全一致
  · 损失      SCE(α=1,β=1,A=-4) —— 唯一上榜确认过的组件(+1.9)，此刻不能换
  · L2 口径   1/(2Cn), C=10 —— 与前面所有实验同一把尺
  · 评估尺子  05 校准过的干净验证子集 40%/60%/80%，三档同向才采信
⚠归因陷阱：本版同时开了 LoRA + 增强。涨分了**不能说"LoRA 值 X 分"**。
  要拆干净得再跑一版 --no-aug。第一版先看整体有没有戏。

用法:
  python scripts/13_train_lora.py --selftest        # 30秒：验证注入正确(必做)
  python scripts/13_train_lora.py --limit 3000      # 2分钟：冒烟，验证全流程能跑通
  python scripts/13_train_lora.py                   # 正式：约 40~60 分钟
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
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm import tqdm

from configs import paths as P

ImageFile.LOAD_TRUNCATED_IMAGES = True      # 官方说明数据里有截断图，必开

SEED, VAL_FRAC = 42, 0.2                    # ⚠必须与 02/06 一致，否则不可比
KEEPS = (0.40, 0.60, 0.80)                  # 干净验证子集三档，三档同向才采信


def _load_sibling(fname, modname):
    """
    加载同目录下以数字开头的脚本(不能用 import，因为模块名不能以数字开头)。

    ⚠必须注册进 sys.modules：Windows 的 DataLoader 走 spawn，子进程要反序列化
      Dataset 类时会执行 `import <modname>`。动态加载的模块不在 sys.modules 里，
      子进程就会报 `Can't pickle <class 'xxx.YYY'>: import of module 'xxx' failed`。
      注册之后，子进程重跑主脚本顶层代码时会把它填回去，反序列化才能找到。
    """
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(modname, os.path.join(here, fname))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod          # ⚠必须在 exec_module 之前
    spec.loader.exec_module(mod)
    return mod


_ex = _load_sibling("01_extract_features.py", "extract01")     # 文件清单 + 编码工具
_aug = _load_sibling("10_extract_augmented.py", "augment10")   # 复用同一套增强，保证可比


# ══════════════════════════════════════════════════════════ LoRA 注入
class LoRAAttention(nn.Module):
    """
    包住 CLIP 里的 nn.MultiheadAttention，给 q / v 投影加低秩补丁。

    ⚠为什么要包一层、不能直接换成两个 nn.Linear：
      CLIP 用的是 PyTorch 原生 MultiheadAttention，它把 q,k,v 三个投影**融合**成
      一个 (3*768, 768) 的大矩阵 in_proj_weight，前 768 行是 q、中间是 k、后面是 v。
      所以我们按行切片，只给 q 段和 v 段加补丁，再拼回去，然后仍然调用官方的
      F.multi_head_attention_forward。
      ⭐这样做的好处：补丁为 0 时走的是**和原版完全相同的代码路径**，
        数值可以逐位对齐，--selftest 才有意义。
    """

    def __init__(self, mha: nn.MultiheadAttention, r: int, alpha: float, targets=("q", "v")):
        super().__init__()
        self.base = mha                      # 原模块整个留着，权重一个都不改
        self.embed_dim = mha.embed_dim
        self.num_heads = mha.num_heads
        self.targets = tuple(targets)
        self.scaling = alpha / r             # LoRA 惯例：用 alpha/r 解耦"秩"和"步长"

        E = self.embed_dim
        self.lora_A = nn.ParameterDict()
        self.lora_B = nn.ParameterDict()
        for t in self.targets:
            A = nn.Parameter(torch.empty(r, E))
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))   # A 随机(官方实现同款)
            self.lora_A[t] = A
            self.lora_B[t] = nn.Parameter(torch.zeros(E, r))  # ⭐B 全 0 => 起步等于原模型

    def merged_in_proj(self):
        """把 (3E,E) 的融合投影矩阵按 q/k/v 切开，给目标段加补丁，再拼回去。"""
        E = self.embed_dim
        W = self.base.in_proj_weight
        parts = []
        for i, name in enumerate(("q", "k", "v")):
            seg = W[i * E:(i + 1) * E]
            if name in self.targets:
                seg = seg + (self.lora_B[name] @ self.lora_A[name]) * self.scaling
            parts.append(seg)
        return torch.cat(parts, 0)

    def forward(self, query, key, value, need_weights=False, attn_mask=None, **kw):
        w = self.merged_in_proj()
        return Fn.multi_head_attention_forward(
            query, key, value, self.embed_dim, self.num_heads,
            w, self.base.in_proj_bias,
            self.base.bias_k, self.base.bias_v, self.base.add_zero_attn,
            self.base.dropout, self.base.out_proj.weight, self.base.out_proj.bias,
            training=self.training, key_padding_mask=None,
            need_weights=need_weights, attn_mask=attn_mask)


class LoRALinear(nn.Module):
    """给普通 nn.Linear 挂补丁（用于 --lora-mlp 时的 mlp.c_fc / mlp.c_proj）。"""

    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base = base
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))

    def forward(self, x):
        return self.base(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling


def inject_lora(model, r, alpha, targets, do_mlp, layers=None):
    """
    往 model.visual.transformer.resblocks 里注入 LoRA。返回 (lora参数列表, 注入报告)。
    layers=None 表示全部 12 层；也可以只挂靠近输出的几层
    （靠前的层学的是边缘/纹理等通用特征，动它风险更大）。
    """
    blocks = model.visual.transformer.resblocks
    idxs = list(range(len(blocks))) if layers is None else layers
    params, report = [], []
    for i in idxs:
        blk = blocks[i]
        wrapped = LoRAAttention(blk.attn, r, alpha, targets)
        blk.attn = wrapped
        params += list(wrapped.lora_A.parameters()) + list(wrapped.lora_B.parameters())
        report.append("blk%02d.attn.%s" % (i, "+".join(targets)))
        if do_mlp:
            for nm in ("c_fc", "c_proj"):
                sub = LoRALinear(getattr(blk.mlp, nm), r, alpha)
                setattr(blk.mlp, nm, sub)
                params += [sub.lora_A, sub.lora_B]
                report.append("blk%02d.mlp.%s" % (i, nm))
    return params, report


def load_clip_fp32(device):
    """
    加载 CLIP 并转成 fp32。
    ⚠clip.load 在 cuda 上会把权重转成 fp16（推理够用），但**训练时 fp16 主权重不稳**：
      LoRA 的更新量很小，fp16 的精度不够表示，加上去可能直接被舍入吃掉。
      正确做法 = 主权重 fp32 + 前向用 autocast 临时降到 fp16 提速。
    """
    import clip
    model, preprocess = clip.load(P.CLIP_MODEL_NAME, device=device,
                                  download_root=P.CLIP_DOWNLOAD_ROOT)
    model.float()                                   # ⭐fp16 -> fp32 主权重
    for prm in model.parameters():
        prm.requires_grad = False                   # 先全冻，注入后只放开 LoRA
    n_visual = sum(p.numel() for p in model.visual.parameters())
    if n_visual != P.CLIP_VISUAL_PARAMS:
        print("  ⚠ 图像编码器参数量 {:,} 与记录值 {:,} 不符".format(n_visual, P.CLIP_VISUAL_PARAMS))
    return model, preprocess


# ══════════════════════════════════════════════════════════ 自检
@torch.no_grad()
def selftest(device, r, alpha, targets, do_mlp):
    """
    验证两件事，任一不过后面全部结论作废：
      1) 补丁 B=0 时，注入后的模型输出必须与原版**逐位相同**（否则说明我注入写错了）
      2) 随便给 B 填个非零值后，输出必须**变化**（否则说明补丁根本没接进计算图）
    """
    banner("自检：LoRA 注入是否正确")
    model, preprocess = load_clip_fp32(device)
    paths, _, _ = _ex.list_train_files()
    batch = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in paths[:8]]).to(device)

    model.eval()
    ref = model.encode_image(batch).float().cpu().numpy()

    lora_params, report = inject_lora(model, r, alpha, targets, do_mlp)
    model.to(device)          # ⚠新建的 LoRA 参数默认在 CPU 上，必须搬到和骨干同一个设备
    model.eval()
    out0 = model.encode_image(batch).float().cpu().numpy()
    same = np.array_equal(ref, out0)
    d0 = float(np.abs(ref - out0).max())

    for prm in lora_params:                      # 只有 B 是全 0，随便扰动一下
        if prm.dim() == 2 and prm.abs().sum() == 0:
            prm.add_(torch.randn_like(prm) * 0.01)
    out1 = model.encode_image(batch).float().cpu().numpy()
    d1 = float(np.abs(ref - out1).max())

    n_lora = sum(p.numel() for p in lora_params)
    n_head = 500 * P.FEATURE_DIM + 500
    print("  注入位置        : %d 处 (%s%s)" % (len(report), report[0],
                                                " ... " + report[-1] if len(report) > 1 else ""))
    print("  LoRA 可训练参数 : {:,}".format(n_lora))
    print("  对比：线性头     : {:,}   骨干全参: {:,}".format(n_head, P.CLIP_VISUAL_PARAMS))
    print("  占骨干比例      : %.3f%%" % (100.0 * n_lora / P.CLIP_VISUAL_PARAMS))
    print()
    print("  [1] B=0 时与原版逐位相同 : %s   (最大差值 %.3e)" % (same, d0))
    print("  [2] 扰动 B 后输出改变    : %s   (最大差值 %.3e)" % (d1 > 1e-4, d1))
    ok = same and d1 > 1e-4
    print("\n  结论: %s" % ("✅ 注入正确，可以开训" if ok
                            else "❌ 注入有问题，先别跑正式训练"))
    if n_lora < n_head:
        print("\n  ⚠注意看上面的数字：LoRA 参数比线性头还少，"
              "但你实测过线性头能背下 98.4%% 的带噪标签。\n"
              "    ==> 别指望'参数少'能抗噪，真正的旋钮是正则强度和**什么时候叫停**。")
    return ok


# ══════════════════════════════════════════════════════════ 数据
class TrainDataset(Dataset):
    """训练集：走增强。⚠transform 里不能有 lambda（Windows spawn 无法序列化）。"""

    def __init__(self, paths, labels, tf):
        self.paths, self.labels, self.tf = paths, labels, tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            with Image.open(self.paths[i]) as im:
                return self.tf(im.convert("RGB")), int(self.labels[i]), 1
        except Exception:
            return torch.zeros(3, 224, 224), int(self.labels[i]), 0


class EvalDataset(Dataset):
    """验证集：走 CLIP 官方 preprocess（center crop），不加任何增强。"""

    def __init__(self, paths, tf):
        self.paths, self.tf = paths, tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            with Image.open(self.paths[i]) as im:
                return self.tf(im.convert("RGB")), i, 1
        except Exception:
            return torch.zeros(3, 224, 224), i, 0


# ══════════════════════════════════════════════════════════ 损失
def loss_sce(logits, y, alpha=1.0, beta=1.0, A=-4.0):
    """
    对称交叉熵 L = α·CE + β·RCE（与 06 号脚本逐字一致，换了就没法比）。
    RCE 是把预测和标签调换位置算的"反向"交叉熵，对错标签天然不敏感，
    用来抵消 CE"错得越离谱惩罚越重、反而逼模型死记错标"的毛病。
    """
    ce = Fn.cross_entropy(logits, y)
    p = Fn.softmax(logits, 1).clamp(1e-7, 1.0)
    oh = Fn.one_hot(y, logits.shape[1]).float()
    rce = -(p * (oh * 0.0 + (1 - oh) * A)).sum(1).mean()
    return alpha * ce + beta * rce


# ══════════════════════════════════════════════════════════ 评估
@torch.no_grad()
def encode(model, tf, paths, device, bs, nw, desc, amp=True):
    """用当前（含 LoRA 的）骨干把一批图编码成 L2 归一化后的特征。"""
    # ⚠pin_memory=False：评估不是瓶颈，而锁页内存不能被换出、会加重提交压力。
    #   评估 loader 是在训练循环内部起的，此时训练的 worker 还活着，两边资源是叠加的。
    loader = DataLoader(EvalDataset(paths, tf), batch_size=bs, shuffle=False,
                        num_workers=nw, pin_memory=False)
    feats = np.zeros((len(paths), P.FEATURE_DIM), dtype=np.float32)
    model.eval()
    bar = tqdm(total=len(paths), desc=desc, unit="img", unit_scale=True, dynamic_ncols=True)
    for imgs, idxs, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=(amp and device == "cuda")):
            f = model.encode_image(imgs)
        feats[idxs.numpy()] = f.float().cpu().numpy()
        bar.update(imgs.shape[0])
    bar.close()
    return feats / np.maximum(np.linalg.norm(feats, axis=1, keepdims=True), 1e-12)


def evaluate(model, tf, val_paths, yv, masks, W, b, device, bs, nw, tag):
    """在验证集上出四档准确率（干净40/60/80 + 原始100）。"""
    Fv = encode(model, tf, val_paths, device, bs, nw, "评估 " + tag)
    pred = np.argmax(Fv @ W.T + b, 1)
    return {str(k): float((pred[m] == yv[m]).mean()) for k, m in masks.items()}


# ══════════════════════════════════════════════════════════ 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="只验证注入正确性然后退出")
    ap.add_argument("--limit", type=int, default=0, help=">0 时只用前 N 张训练图冒烟")
    # --- LoRA 结构
    ap.add_argument("--rank", type=int, default=4, help="秩 r。⚠r 大不等于更能抗噪，见文件头")
    ap.add_argument("--lora-alpha", type=float, default=4.0, help="缩放，惯例取 = rank")
    ap.add_argument("--targets", type=str, default="q,v", help="挂在哪些投影上: q,k,v 任选")
    ap.add_argument("--lora-mlp", action="store_true", help="额外挂到 mlp 的两个 Linear 上")
    ap.add_argument("--last-n-layers", type=int, default=0,
                    help=">0 时只挂最靠近输出的 N 层（靠前的层学通用特征，动它风险更大）")
    # --- 训练
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=128, help="8GB 显存下的安全值")
    ap.add_argument("--num-workers", type=int, default=4,
                    help="⚠训练 loader 的 worker 数。别调大：见下面 --eval-workers 的说明")
    ap.add_argument("--eval-workers", type=int, default=2,
                    help="⚠⚠评估 loader 单独一个更小的值，不能和训练共用。"
                         "原因：Windows 的 DataLoader 走 spawn，每个 worker 都是新进程、"
                         "都要 import torch + 加载 CUDA DLL，各自预留几个 GB 的 commit。"
                         "而评估发生在训练循环【内部】——此时训练的 worker 还全都活着，"
                         "两边进程数直接相加。8+8=16 会打爆系统提交上限，报 "
                         "'[WinError 1455] 页面文件太小'（2026-09-05 实测踩过）。"
                         "评估是纯前向、GPU 吃得快，2 个 worker 够喂饱。")
    ap.add_argument("--lr-lora", type=float, default=1e-4, help="比训头(0.05)小两个量级，LP-FT要求")
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--C", type=float, default=10.0, help="头的 L2 口径，与前面实验一致")
    ap.add_argument("--warmup", type=int, default=100, help="前多少步线性升温")
    ap.add_argument("--evals-per-epoch", type=int, default=2,
                    help="每轮评估+存档几次。⭐30%%噪声下'何时叫停'是主要旋钮，多存几个点")
    ap.add_argument("--no-aug", action="store_true",
                    help="关掉增强。⭐要把'LoRA 的功劳'和'增强的功劳'拆开时跑这个")
    ap.add_argument("--head-init", type=str,
                    default=os.path.join(P.CACHE_DIR, "head_loss_sce.npz"),
                    help="LP-FT 第一阶段的头。⛔别用随机初始化，会把骨干搅坏")
    ap.add_argument("--no-amp", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    targets = tuple(t.strip() for t in args.targets.split(",") if t.strip())

    if args.selftest:
        selftest(device, args.rank, args.lora_alpha, targets, args.lora_mlp)
        return

    P.ensure_dirs()
    banner("13 LoRA 微调（LP-FT 配方） r=%d %s%s" %
           (args.rank, "+".join(targets), " +mlp" if args.lora_mlp else ""))

    # ---------------------------------------------------------- 数据划分
    paths, labels, class_names = _ex.list_train_files()
    y = labels.astype(np.int64)
    idx = np.arange(len(y))
    itr, iva = train_test_split(idx, test_size=VAL_FRAC, random_state=SEED, stratify=y)
    if args.limit:                                  # 冒烟模式：两边都砍小
        itr, iva = itr[:args.limit], iva[:max(args.limit // 4, 200)]

    tr_paths = [paths[i] for i in itr]
    va_paths = [paths[i] for i in iva]
    ytr, yv = y[itr], y[iva]
    K = len(class_names)
    print("  训练 %d / 验证 %d / 类别 %d" % (len(itr), len(iva), K))
    print("  划分  seed=%d test_size=%.2f stratify=y  (⚠与 02/06 完全一致，才可比)"
          % (SEED, VAL_FRAC))

    # ---------------------------------------------------------- 评估尺子
    nz = np.load(os.path.join(P.CACHE_DIR, "noise_scores.npz"), allow_pickle=True)
    sv = nz["suspicion"][iva]
    masks = {k: (sv <= np.quantile(sv, k)) for k in KEEPS}
    masks[1.0] = np.ones(len(yv), bool)
    print("  尺子  干净验证子集 %s + 原始100%%" % "/".join("%.0f%%" % (k * 100) for k in KEEPS))
    print("  ⚠只能比大小不能读数；三档同向才采信；两个指标打架就必须交排行榜裁决")

    # ---------------------------------------------------------- 模型
    model, preprocess = load_clip_fp32(device)
    layers = None
    if args.last_n_layers > 0:
        n_blk = len(model.visual.transformer.resblocks)
        layers = list(range(n_blk - args.last_n_layers, n_blk))
    lora_params, report = inject_lora(model, args.rank, args.lora_alpha,
                                      targets, args.lora_mlp, layers)
    for p_ in lora_params:
        p_.requires_grad = True
    model.to(device)
    n_lora = sum(p.numel() for p in lora_params)
    print("  LoRA  注入 %d 处，可训练参数 %s（占骨干 %.3f%%）"
          % (len(report), format(n_lora, ","), 100.0 * n_lora / P.CLIP_VISUAL_PARAMS))

    # ---------------------------------------------------------- 头（LP-FT 第一阶段产物）
    hd = np.load(args.head_init)
    W = torch.tensor(hd["W"], dtype=torch.float32, device=device, requires_grad=True)
    b = torch.tensor(hd["b"], dtype=torch.float32, device=device, requires_grad=True)
    print("  头初始化 <- %s  (W%s)  ⭐LP-FT：绝不随机初始化"
          % (os.path.basename(args.head_init), tuple(W.shape)))
    assert W.shape == (K, P.FEATURE_DIM), "头的形状和类别数对不上"

    # ---------------------------------------------------------- 增强
    # 归一化参数直接从 CLIP 官方 preprocess 里取，保证与预训练时的数值分布完全一致
    normalize = [t for t in preprocess.transforms if isinstance(t, T.Normalize)][0]
    if args.no_aug:
        train_tf = preprocess
        print("  增强  ⛔关闭（用于把 LoRA 和增强的功劳拆开）")
    else:
        train_tf = _aug.build_aug(normalize)
        print("  增强  RandomResizedCrop(224,scale=0.7~1.0)+HFlip，⛔无颜色抖动")
        print("        ⭐冻结骨干时增强是**结构性无效**（骨干学不了任何东西，机制没启动）；"
              "现在骨干在训练，机制才第一次生效")

    # ---------------------------------------------------------- 优化器
    opt = torch.optim.AdamW([
        {"params": lora_params, "lr": args.lr_lora, "weight_decay": 0.0},
        {"params": [W, b], "lr": args.lr_head, "weight_decay": 0.0},
    ])
    loader = DataLoader(TrainDataset(tr_paths, ytr, train_tf), batch_size=args.batch_size,
                        shuffle=True, num_workers=args.num_workers,
                        pin_memory=(device == "cuda"), drop_last=True)
    steps_per_epoch = len(loader)
    total_steps = max(int(steps_per_epoch * args.epochs), 1)
    eval_every = max(steps_per_epoch // max(args.evals_per_epoch, 1), 1)
    reg_coef = 1.0 / (2.0 * args.C * len(itr))      # ⚠与 02/06 同一套 L2 口径
    amp_on = (not args.no_amp) and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_on)
    print("  优化  AdamW  lora_lr=%g  head_lr=%g  L2(头)=1/(2Cn)=%.3e  AMP=%s"
          % (args.lr_lora, args.lr_head, reg_coef, amp_on))
    print("  进程  训练 %d worker + 评估 %d worker = 峰值 %d 个（评估发生在训练循环内部，"
          "两边同时活着）" % (args.num_workers, args.eval_workers,
                             args.num_workers + args.eval_workers))
    if args.num_workers + args.eval_workers > 8:
        print("  ⚠ 峰值超过 8 个进程，Windows 下有 '[WinError 1455] 页面文件太小' 的风险")
    print("  计划  %d 步/轮 x %.1f 轮 = %d 步，每 %d 步评估存档一次"
          % (steps_per_epoch, args.epochs, total_steps, eval_every))

    run_dir = os.path.join(P.RUNS_DIR, time.strftime("%Y%m%d_%H%M%S") + "_lora")
    os.makedirs(run_dir, exist_ok=True)
    print("  产物  ->", run_dir)

    # ---------------------------------------------------------- 训练循环
    def lr_at(step):
        """线性升温 + 余弦退火。升温是为了别在头几步就把预训练权重冲歪。"""
        if step < args.warmup:
            return (step + 1) / args.warmup
        t = (step - args.warmup) / max(total_steps - args.warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))

    base_lrs = [g["lr"] for g in opt.param_groups]
    history, step, t0 = [], 0, time.time()

    def do_eval(tag):
        accs = evaluate(model, preprocess, va_paths, yv, masks,
                        W.detach().cpu().numpy(), b.detach().cpu().numpy(),
                        device, args.batch_size, args.eval_workers, tag)
        ck = os.path.join(run_dir, "ckpt_%s.pt" % tag)
        torch.save({"lora": {k: v.detach().cpu() for k, v in model.state_dict().items()
                             if "lora_" in k},
                    "W": W.detach().cpu(), "b": b.detach().cpu(),
                    "args": vars(args), "step": step, "tag": tag, "acc": accs}, ck)
        line = "  [%s] " % tag + "  ".join(
            "干净%.0f%% %.4f" % (k * 100, accs[str(k)]) for k in KEEPS) + \
            "  原始100%% %.4f" % accs["1.0"]
        print(line)
        history.append({"tag": tag, "step": step, "acc": accs,
                        "minutes": (time.time() - t0) / 60})
        with open(os.path.join(run_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump({"args": vars(args), "history": history,
                       "keeps": list(KEEPS)}, f, ensure_ascii=False, indent=2)
        model.train()

    banner("起点评估（LoRA 全 0，等价于原版 CLIP + 59 分的头）")
    do_eval("ep0.00")
    ref0 = history[0]["acc"]

    banner("开训")
    model.train()
    bar = tqdm(total=total_steps, desc="train", unit="step", dynamic_ncols=True)
    done = False
    while not done:
        for imgs, ys, _ in loader:
            if step >= total_steps:
                done = True
                break
            scale = lr_at(step)
            for g, bl in zip(opt.param_groups, base_lrs):
                g["lr"] = bl * scale

            imgs = imgs.to(device, non_blocking=True)
            ys = ys.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            # ⚠只有骨干走 fp16 自动混合精度；头和损失留在 fp32。
            #   原因：softmax/log 在 fp16 下容易溢出或下溢，而头的计算量极小，省这点没意义。
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp_on):
                f = model.encode_image(imgs)
            f = f.float()
            f = f / f.norm(dim=1, keepdim=True).clamp_min(1e-12)  # ⚠头是在归一化特征上训的
            logits = f @ W.T + b
            loss = loss_sce(logits, ys) + reg_coef * (W * W).sum()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            step += 1
            bar.update(1)
            if step % 20 == 0:
                with torch.no_grad():
                    acc = (logits.argmax(1) == ys).float().mean().item()
                bar.set_postfix_str("loss=%.4f 批acc=%.3f lr=%.1e"
                                    % (float(loss.detach()), acc, opt.param_groups[0]["lr"]))
            if step % eval_every == 0 or step == total_steps:
                bar.close()
                do_eval("ep%.2f" % (step / steps_per_epoch))
                bar = tqdm(total=total_steps, initial=step, desc="train",
                           unit="step", dynamic_ncols=True)
    bar.close()

    # ---------------------------------------------------------- 汇总
    banner("汇总：每个存档点相对起点(α=0)的变化")
    hdr = "  %-10s %9s " % ("存档", "分钟") + " ".join(
        "%10s" % ("干净%.0f%%" % (k * 100)) for k in KEEPS) + "%10s %8s" % ("原始100%", "判定")
    print(hdr)
    print("  " + "-" * (len(hdr) + 4))
    for h in history:
        d = [h["acc"][str(k)] - ref0[str(k)] for k in KEEPS]
        draw = h["acc"]["1.0"] - ref0["1.0"]
        if h["tag"] == "ep0.00":
            verdict = "基准"
        elif all(x > 0 for x in d) and draw > 0:
            verdict = "✅一致"
        elif all(x < 0 for x in d) and draw < 0:
            verdict = "❌一致劣"
        else:
            verdict = "⚠打架"
        print("  %-10s %9.1f " % (h["tag"], h["minutes"]) +
              " ".join("%+10.4f" % x for x in d) + "%+10.4f %8s" % (draw, verdict))

    print("\n  ⚠判定口径（来自 5 次真实提交的实证）：")
    print("    · 干净档与原始档**两个指标一致** -> 可信；**打架** -> 本地一律无效，必须交榜")
    print("    · 只信方向不信幅度：实测本地->真实的倍率在 117x~370x 之间飘")
    print("    · 小于 0.006 的差异本地分辨不出来（尺子的分辨率下限）")
    print("\n  下一步: python scripts/14_lora_sweep_alpha.py --run %s" % os.path.basename(run_dir))
    print("    ⭐WiSE-FT 在 LoRA 下就是把补丁乘 α，**不用重训**，"
          "一次训练能换来 (存档点 x α) 一整张二维表")


if __name__ == "__main__":       # Windows 下 DataLoader 多进程必须有这个保护
    main()
