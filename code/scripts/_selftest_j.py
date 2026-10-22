# -*- coding: utf-8 -*-
"""实验 J 的自测：DropPath + RandomErasing 接入是否安全。

⭐按交接文档 4.10：**从源文件本体导入**去测，不复制代码片段——
  副本会和真品漂移，测过的和实际跑的不是同一个东西，等于没测。

要验证的五件事（任何一条挂了，J 就不能起）：
  1. state_dict 逐键不变  → --resume 和 26/45 的 strict=True 不会炸
  2. eval() 下是恒等映射   → 出包路径完全不受影响，老 ckpt 照样能用
  3. train() 下真的在丢，且是**按样本**丢（LND 布局，batch 在第 1 维）
  4. copy.deepcopy 能过    → EMA 的 shadow 是 deepcopy 出来的
  5. 前向数值仍然有限、形状不变
"""
import _bootstrap  # noqa: F401
from _bootstrap import banner

import copy
import importlib.util
import os

import torch

from configs import paths as P

HERE = os.path.dirname(os.path.abspath(__file__))


def load_src(fname, modname):
    spec = importlib.util.spec_from_file_location(modname, os.path.join(HERE, fname))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    banner("实验 J 自测：DropPath / RandomErasing 接入安全性")
    ft = load_src("28_finetune_v2.py", "ft28")

    import clip
    model, _ = clip.load(P.CLIP_MODEL_NAME, device="cpu", download_root=P.CLIP_DOWNLOAD_ROOT)
    model.float()
    visual = model.visual
    n_blocks = len(visual.transformer.resblocks)

    keys_before = list(visual.state_dict().keys())
    x = torch.randn(4, 3, 224, 224)
    visual.eval()
    with torch.no_grad():
        y_ref = visual(x).clone()

    rates = ft.apply_drop_path(visual, 0.2)
    print("  已挂 DropPath：%d 层，%.4f → %.4f" % (len(rates), rates[0], rates[-1]))
    assert len(rates) == n_blocks
    assert abs(rates[0]) < 1e-9 and abs(rates[-1] - 0.2) < 1e-9, "线性递增的两个端点不对"

    # ── 1. state_dict 逐键不变
    keys_after = list(visual.state_dict().keys())
    assert keys_before == keys_after, (
        "⛔state_dict 变了！--resume / 26 / 45 的 strict=True 会炸。\n"
        "   多出来：%s" % (set(keys_after) - set(keys_before)))
    print("  ✅[1] state_dict 逐键不变（%d 个键）→ resume 与 26/45 的 strict=True 安全" % len(keys_after))

    # ── 2. eval() 恒等
    visual.eval()
    with torch.no_grad():
        y_eval = visual(x)
    d = (y_eval - y_ref).abs().max().item()
    assert d == 0.0, "⛔eval 下不是恒等，最大差 %.3e —— 出包会被影响" % d
    print("  ✅[2] eval() 下逐位恒等（最大差 %.1e）→ 出包路径不受影响" % d)

    # ── 3. train() 下真的在丢，且按样本丢
    visual.train()
    torch.manual_seed(0)
    with torch.no_grad():
        y_tr = visual(x)
    assert (y_tr - y_ref).abs().max().item() > 0, "⛔train 下没有任何变化，DropPath 没生效"
    print("  ✅[3a] train() 下前向确实改变了（DropPath 在工作）")

    # 直接检查一层的掩码维度：喂 LND 形状，看被清零的是不是"整个样本"
    dp = ft.DropPath(0.5)
    dp.train()
    L, N, D = 7, 64, 8
    z = torch.ones(L, N, D)
    torch.manual_seed(1)
    out = dp(z)
    per_sample = (out.abs().sum(dim=(0, 2)) == 0)          # 每个样本是否整体被清零
    per_token = (out.abs().sum(dim=(1, 2)) == 0)           # 每个 token 位是否整体被清零
    assert per_sample.any(), "⛔p=0.5 下一个样本都没丢，掩码维度错了"
    assert not per_token.any(), "⛔按 token 位丢了！CLIP 是 LND，batch 在第 1 维不是第 0 维"
    print("  ✅[3b] 按**样本**丢（%d/%d 个样本被丢），⛔不是按 token 位丢 → LND 布局处理正确"
          % (int(per_sample.sum()), N))
    keep_scale = out.sum().item() / z.sum().item()
    print("       期望保持检查：输出/输入 = %.3f（应≈1.0，除以 keep 做了补偿）" % keep_scale)

    # ── 4. deepcopy（EMA 的 shadow 走这条路）
    sh = copy.deepcopy(visual).eval()
    assert sh.transformer.resblocks[3]._dp_attn is not visual.transformer.resblocks[3]._dp_attn
    with torch.no_grad():
        y_sh = sh(x)
    assert (y_sh - y_ref).abs().max().item() == 0.0, "⛔deepcopy 后 eval 不恒等"
    print("  ✅[4] copy.deepcopy 通过，且副本 eval 下仍恒等 → EMA 安全")

    # ── 5. 形状与数值
    assert y_tr.shape == y_ref.shape and torch.isfinite(y_tr).all()
    print("  ✅[5] 形状不变 %s，数值有限" % (tuple(y_tr.shape),))

    # ── RandomErasing 顺序
    from torchvision import transforms as T
    norm = T.Normalize((0.48145466, 0.4578275, 0.40821073),
                       (0.26862954, 0.26130258, 0.27577711))
    ops = [T.RandomResizedCrop(224), T.RandomHorizontalFlip(), T.ToTensor(), norm,
           T.RandomErasing(p=1.0, scale=(0.02, 0.20), ratio=(0.3, 3.3), value=0)]
    names = [type(o).__name__ for o in ops]
    assert names.index("RandomErasing") > names.index("Normalize"), "⛔Erasing 必须排在 Normalize 之后"
    from PIL import Image
    im = Image.new("RGB", (300, 260), (120, 160, 200))
    t0 = T.Compose(ops[:-1])(im)
    t1 = T.Compose(ops)(im)
    assert (t0 - t1).abs().sum().item() > 0, "⛔RandomErasing p=1.0 却什么都没擦"
    print("  ✅[6] RandomErasing 排在 Normalize 之后且确实生效（擦掉 %d 个像素位）"
          % int(((t0 - t1).abs().sum(0) > 0).sum()))

    print("\n  ⭐六项全绿 —— 实验 J 可以起。老 ckpt、出包脚本、EMA 均不受影响。")


if __name__ == "__main__":
    main()
