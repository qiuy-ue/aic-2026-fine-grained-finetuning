# -*- coding: utf-8 -*-
"""
44：**逐层漂移诊断**——微调到底动了骨干的哪几层？先花 1 分钟，再决定投不投 4.5 小时。

═══════════════════════════════════════════════════════════════════════════
为什么写这个脚本（HX 提的问题，戳中了原方案的一个真缺陷）
═══════════════════════════════════════════════════════════════════════════
HX 问：「要做解冻和微调不应该从最底层一层一层试吗？」

⭐他抓到的真问题：**LLRD 只有一个旋钮（decay），它强制假设 lr 随深度是
  「指数衰减」这个形状。** 若真实最优是阶梯形（前 6 层完全不动、后 6 层全速），
  LLRD 无论 decay 取多少都表达不出来。→ 逐层硬冻结能找到 LLRD 够不着的解。

⛔但逐层试 = 12+ 次完整训练 ≈ 18 小时，剩余时间做不完。
⭐所以先问一个便宜得多的问题：**底层现在到底动了没有？**

  - 若底层几乎没漂移 → LLRD=0.85 **已经实质冻住了它**，
    硬冻结买不到新的表达能力（只买到算力），⛔别投那 4.5 小时。
  - 若底层漂移很大 → 说明"该不该动底层"是个悬而未决的问题，值得投。

这是[[判据6：oracle上界诊断]]的同一个套路：用便宜的测量决定昂贵的投入。

═══════════════════════════════════════════════════════════════════════════
⭐两个量，问的是两件不同的事（⛔别混为一谈）
═══════════════════════════════════════════════════════════════════════════
**① 权重相对漂移** ‖W_ft − W_0‖_F ⁄ ‖W_0‖_F
   问的是「**参数**动了多少」。便宜（不用跑图片），但⚠**它不等于功能变了多少**：
   神经网络有大量冗余方向，权重动了不代表输出跟着动。

**② 输出特征漂移**（逐层 hook 真实图片的激活，算余弦相似度）
   问的是「**功能**变了多少」——这才是我真正关心的。
   ⭐**必须两个都看**：若权重动了但特征没动 → 动的是冗余方向，等于没动。

═══════════════════════════════════════════════════════════════════════════
⭐关键对照：漂移量 vs LLRD 分配的 lr 比例
═══════════════════════════════════════════════════════════════════════════
LLRD 给底层的 lr 是顶层的 0.85¹³ = **1/8.3**。若实测「底层漂移/顶层漂移」也≈1/8.3，
说明**漂移量就是 lr 定的**，调 decay 能直接控制它。
若底层漂移**远小于** 1/8.3 → 说明底层梯度本身就小，它是**自然冻结**的，
⛔那么调 LLRD 或硬冻结底层**都不会有什么效果**，这条路整个关闭。
"""
import io
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bootstrap  # noqa: F401,E402
from configs import paths as P  # noqa: E402

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(P.RUNS_DIR, "20260906_110505_v2_320", "ckpt_ep02.pt")
LLRD, N_BLK = 0.85, 12


def layer_id(name):
    """与 28_finetune_v2.layer_id_of 逐字同一套口径，否则两张表对不上。"""
    if name.startswith(("conv1", "class_embedding", "positional_embedding", "ln_pre")):
        return 0
    if name.startswith("transformer.resblocks."):
        return int(name.split(".")[2]) + 1
    return N_BLK + 1


def main():
    t0 = time.time()
    import clip
    model, _ = clip.load(P.CLIP_MODEL_NAME, device="cpu",
                         download_root=P.CLIP_DOWNLOAD_ROOT)
    model.float()
    base = {k: v.clone() for k, v in model.visual.state_dict().items()}

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    ft = ck["visual"]
    tr = ck["args"]
    print("=" * 88)
    print("44 逐层漂移诊断   ckpt=%s ep%s  训练@%dpx  LLRD=%.2f  lr_backbone=%g"
          % (os.path.basename(os.path.dirname(CKPT)), ck.get("epoch"),
             tr.get("resolution", 224), tr.get("llrd", LLRD), tr.get("lr_backbone")))
    print("=" * 88)

    # ── ① 权重相对漂移（逐层聚合）
    num, den, npar = {}, {}, {}
    for k, w0 in base.items():
        if k not in ft or w0.shape != ft[k].shape:
            continue                       # positional_embedding 因插值换了形状，跳过
        lid = layer_id(k)
        d = (ft[k].float() - w0.float())
        num[lid] = num.get(lid, 0.0) + float((d * d).sum())
        den[lid] = den.get(lid, 0.0) + float((w0.float() ** 2).sum())
        npar[lid] = npar.get(lid, 0) + w0.numel()

    lids = sorted(num)
    rel = {l: np.sqrt(num[l] / max(den[l], 1e-12)) for l in lids}
    top = rel[max(lids)]
    print("%-6s %-30s %12s %12s %12s" %
          ("层id", "位置", "相对漂移", "占顶层比例", "LLRD给的lr比"))
    print("-" * 88)
    for l in lids:
        pos = ("输入端 conv1/posemb/ln_pre" if l == 0 else
               "输出端 ln_post/proj" if l == max(lids) else "block %02d" % (l - 1))
        lr_ratio = LLRD ** (max(lids) - l)
        print("%-6d %-30s %12.5f %11.3f× %11.3f×"
              % (l, pos, rel[l], rel[l] / max(top, 1e-12), lr_ratio))
    print("-" * 88)

    bot = rel[0]
    print("⭐底层/顶层 实测漂移比 = %.4f   ；LLRD 分配的 lr 比 = %.4f (0.85^%d)"
          % (bot / max(top, 1e-12), LLRD ** max(lids), max(lids)))
    r = (bot / max(top, 1e-12)) / (LLRD ** max(lids))
    if r < 0.5:
        print("   → 实测漂移比**远小于** lr 比（只有 %.2f 倍）：底层梯度本身就小，" % r)
        print("     它是**自然冻结**的。⛔调 LLRD / 硬冻结底层都不会有什么效果。")
    elif r > 2.0:
        print("   → 实测漂移比**远大于** lr 比（%.2f 倍）：底层动得比 lr 该给的还多，" % r)
        print("     ⭐说明底层梯度很大——这正是「小数据破坏通用特征」的形态，值得压。")
    else:
        print("   → 实测漂移比与 lr 比同量级（%.2f 倍）：漂移量**就是 lr 定的**，" % r)
        print("     ⭐调 decay 能直接控制它 → LLRD 扫描是有效的旋钮。")

    # ── ② 输出特征漂移（⭐权重动了≠功能变了，必须验这一步）
    print("\n" + "=" * 88)
    print("② 逐层**输出特征**漂移（真实图片，⭐这才是「功能变了多少」）")
    print("=" * 88)
    res = tr.get("resolution", 224)
    _ft = _bootstrap.load_module("24_finetune_full.py", "finetune24") \
        if hasattr(_bootstrap, "load_module") else None
    if _ft is None:                        # _bootstrap 没这个 helper 就手动加载
        import importlib.util
        sp = importlib.util.spec_from_file_location(
            "finetune24", os.path.join(ROOT, "scripts", "24_finetune_full.py"))
        _ft = importlib.util.module_from_spec(sp)
        sys.modules["finetune24"] = _ft
        sp.loader.exec_module(_ft)

    from PIL import Image
    import glob
    # ⚠manifest 只存元数据，没有文件列表 → 直接扫盘。
    # ⭐每个类只取 1 张、横跨多个类：漂移是逐样本余弦再平均，
    #   取样要覆盖类别空间，⛔别全从一个类里取（那测的是某一类的漂移）。
    files = []
    for cdir in sorted(glob.glob(os.path.join(P.TRAIN_DIR, "*")))[:128]:
        g = sorted(glob.glob(os.path.join(cdir, "*")))
        if g:
            files.append(g[0])
    if not files:
        print("⚠TRAIN_DIR 下没扫到图片，跳过特征漂移（权重那张表已足够定性）")
        return
    _, tf = _ft.build_transforms(res, 0.5)
    ims = []
    for p in files[:128]:
        try:
            with Image.open(p) as im:
                ims.append(tf(im.convert("RGB")))
        except Exception:
            pass
    if len(ims) < 16:
        print("⚠只读到 %d 张图，样本太少，跳过（权重那张表已足够定性）" % len(ims))
        return
    x = torch.stack(ims)
    print("用 %d 张真实训练图 @%dpx" % (len(x), res))

    def acts(sd):
        m2, _ = clip.load(P.CLIP_MODEL_NAME, device="cpu",
                          download_root=P.CLIP_DOWNLOAD_ROOT)
        m2.float()
        _ft.interpolate_pos_embed(m2, res)
        m2.visual.load_state_dict(sd, strict=True)
        v = m2.visual.eval()
        out = {}
        hs = [v.transformer.resblocks[i].register_forward_hook(
            lambda mo, i_, o, k=i: out.__setitem__(k, o.detach().float()))
            for i in range(N_BLK)]
        with torch.no_grad():
            out["final"] = v(x).detach().float()
        for h in hs:
            h.remove()
        return out

    _ft.interpolate_pos_embed(model, res)
    a0 = acts(model.visual.state_dict())
    a1 = acts(ft)
    print("\n%-14s %14s %16s" % ("位置", "余弦相似度", "1−cos（漂移）"))
    print("-" * 88)
    drift = {}
    for k in list(range(N_BLK)) + ["final"]:
        u, v_ = a0[k].flatten(0, -2), a1[k].flatten(0, -2)
        c = torch.nn.functional.cosine_similarity(u, v_, dim=-1).mean().item()
        drift[k] = 1 - c
        print("%-14s %14.6f %16.6f"
              % ("block %02d" % k if k != "final" else "⭐最终特征", c, 1 - c))
    print("-" * 88)
    lo = np.mean([drift[i] for i in range(4)])
    hi = np.mean([drift[i] for i in range(N_BLK - 4, N_BLK)])
    print("⭐前4块平均漂移 %.6f   后4块平均漂移 %.6f   比值 %.4f" % (lo, hi, lo / max(hi, 1e-12)))
    print("\n判读：")
    if lo < 0.01:
        print("  ✅**底层功能几乎没变**（1−cos < 0.01）→ LLRD=%.2f 已实质冻住底层。" % LLRD)
        print("  ⛔**别投 4.5 小时做硬冻结扫描**——它买到的只有算力，没有表达能力上的差别。")
        print("  ⭐但「买到算力」本身有价值：冻住前 N 层→反传不穿过它们→显存和时间都省，")
        print("    这正好是上 448px 的前置条件。→ 硬冻结应作为 **448 的加速手段**，")
        print("    而不是当成一个独立的调参实验。")
    else:
        print("  ⚠**底层功能确实变了**（1−cos = %.4f）→ 「该不该动底层」是悬而未决的，" % lo)
        print("  ⭐值得投一次硬冻结 A/B。⛔但仍别做 12 层逐层扫——做 3 个断点 {0,4,8} 即可，")
        print("    3 次训练就能看出曲线形状，若某个断点明显好再在它附近细化。")
    print("\n总耗时 %.1f 分钟" % ((time.time() - t0) / 60))


if __name__ == "__main__":
    main()
