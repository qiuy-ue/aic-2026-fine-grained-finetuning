# -*- coding: utf-8 -*-
"""58 · 在**测试集**上建多样化视角池：一次编码，之后任意 TTA 配方都是零 GPU。

═══════════════════════════════════════════════════════════════════════
一、为什么做这个（2026-09-15）
═══════════════════════════════════════════════════════════════════════
`cache/tta_search.json`（21 号，34 种组合 × 20644 张验证图，配对比较）里，
**我们现在用的配方就是那张表的「01 ★基线」**，而有 5 个组合以 4~5σ 超过它：

    greedy 前向选择(13视角)  0.6469  Δ=+0.0153      ← 最好
    18 原图×3+全部增强(29)   0.6432  Δ=+0.0116  SE 0.0022
    11 现配方+多尺度(10)     0.6415  Δ=+0.0099  SE 0.0022
    C3 现配方+多尺度放大×4    0.6415  Δ=+0.0095  SE 0.0022
    01 ★基线 = 现配方(6)     0.6316    —          ← 26 号至今用的就是它
    00 无TTA(1)             0.6129  Δ=-0.0188

⭐⭐ **这张表的灵魂是对照组 C1**：在现配方上再加 **4 个同族随机**视角 → Δ = **+0.0000**。
   → **不是"视角越多越好"，是"视角要多样"。** 而各族与原图的余弦相似度排序：

       cc 多尺度放大 0.906  <  strong 强随机 0.947  <  corner 四角 0.957  <  mild 现配方 0.966
       增益          +0.0095   >        +0.0073     >        +0.0052     >     +0.0000

   **完全同序。** 我们现在的 6 视角 = 原图 + 5 个 mild，而 5 个 mild 彼此余弦 0.966
   —— 等于同一个视角看了 5 遍。

═══════════════════════════════════════════════════════════════════════
二、⚠ 为什么不能直接照搬 21 号的排名（这一条必须读）
═══════════════════════════════════════════════════════════════════════
20/21 号的视角池是在 **224px + 冻结 CLIP + 线性探针**（准确率 0.63）上建的；
我们现在是 **448 训练 / 512 推理 / 全微调**（0.79）。而且我们的模型是用
`RandomResizedCrop(scale 0.3~1.0)` 训的，本来就见惯缩放 —— cc 族对它未必同样"新鲜"。

⛔ **而且没法重测**：21 号能排序是因为验证集有标签；今天的 ckpt 是 `--full-data` 训的，
   验证集已并入训练（34 号 :311 对 full_data 直接拒绝运行，同一个理由）。

👉 **所以本脚本只负责"把池子建出来"，⛔不负责排序。**
   排序交给唯一的裁判：池子建好后用 59 号离线组出全部候选配方（零 GPU），
   按新规则（每小时无限次、只打最后一次）**每小时让排行榜排一个**。

⭐ 唯一几乎不依赖迁移假设的一条：**C1 那个 +0.0000 的对照组**证明"多样性付钱、同族凑数不付钱"，
   而我们恰好站在最不多样的那一点上。往哪个方向走不确定，但**该走**是确定的。

═══════════════════════════════════════════════════════════════════════
三、视角定义：逐字照抄 20 号，只把 224 按比例放大到 --res
═══════════════════════════════════════════════════════════════════════
    orig / orig_flip          Resize(R)+CenterCrop(R)（= 26 号的 eval_tf），翻转白送
    cc085/070/055/040(+flip)  按**边长比例**裁中心再拉回 R ⇒ 中心放大 1/s 倍
    corner_tl/tr/bl/br        先归一到 SQ=round(R×256/224) 方块，再切 4 个 R 的角
                              ⚠不在长条上切角：宽图的左右角会切到画面外的内容
    mild0..4                  RandomResizedCrop(R, scale=(0.70,1.00), ratio=(0.85,1.18))
    strong0..4                RandomResizedCrop(R, scale=(0.40,0.75), ratio=(0.85,1.18))

⭐ **种子按【每张图】固定**（`seed*1000003 + i`，抄 20 号）——与 worker 数、batch 大小**无关**。
   ⛔ 这比 26 号的「每视角 `manual_seed(v)` 再起一趟 DataLoader」稳：后者的随机视角会随
   `--batch-size/--num-workers` 在 worker 间重新分配（CLAUDE.md 记过这个坑）。
   代价：本池子的 mild0..4 与现有成绩包**不是同一次抽样** ⇒ 用池子重组的"基线配方"
   与现有包会差几百行，**这是正常的**，不是 bug。真正干净的对照是
   **基线配方 vs 增强配方，两者都从同一个池子里组**。

用法：
  python scripts\58_test_viewpool.py --ckpt <ckpt> --res 512 --limit 200   # 冒烟
  python scripts\58_test_viewpool.py --ckpt <ckpt> --res 512               # 正式（~110分钟）
  python scripts\58_test_viewpool.py --ckpt <ckpt> --res 448               # 多尺度的低分支
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

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from configs import paths as P


def _load_sibling(fname, modname):
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(modname, os.path.join(here, fname))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


pred26 = _load_sibling("26_predict_finetuned.py", "pred26")
_ex = pred26._ex
_ft = pred26._ft

# ── 视角定义常数：⛔逐字照抄 20 号，一个数都别改（改了 21 号那张表就对不上）
CC_SCALES = (0.85, 0.70, 0.55, 0.40)
MILD_SCALE = (0.70, 1.00)
STRONG_SCALE = (0.40, 0.75)
N_MILD = N_STRONG = 5
RATIO = (0.85, 1.18)
SQ_RATIO = 256.0 / 224.0          # 四角视角的外扩比例，同 20 号

VIEW_NAMES = (
    ["orig", "orig_flip"]
    + ["cc%03d" % int(s * 100) for s in CC_SCALES]
    + ["cc%03d_flip" % int(s * 100) for s in CC_SCALES]
    + ["corner_tl", "corner_tr", "corner_bl", "corner_br"]
    + ["mild%d" % i for i in range(N_MILD)]
    + ["strong%d" % i for i in range(N_STRONG)]
)
NUM_VIEWS = len(VIEW_NAMES)       # 24
GROUPS = {
    "G0 原图": ["orig"],
    "G1 原图翻转": ["orig_flip"],
    "G2 多尺度放大": ["cc%03d" % int(s * 100) for s in CC_SCALES],
    "G3 多尺度+翻转": ["cc%03d_flip" % int(s * 100) for s in CC_SCALES],
    "G4 四角位移": ["corner_tl", "corner_tr", "corner_bl", "corner_br"],
    "G5 温和随机(现配方)": ["mild%d" % i for i in range(N_MILD)],
    "G6 强随机": ["strong%d" % i for i in range(N_STRONG)],
}


def center_crop_ratio(im, s):
    """按**边长比例** s 裁出中心区域，保持宽高比 ⇒ 等价于中心放大 1/s 倍。⛔抄 20 号，别改。"""
    w, h = im.size
    cw, ch = max(1, int(round(w * s))), max(1, int(round(h * s)))
    left, top = (w - cw) // 2, (h - ch) // 2
    return im.crop((left, top, left + cw, top + ch))


# ── 预设视角子集。⭐GPU 成本与视角数**线性**，所以要能挑。
PRESETS = {
    # 全部 24 个（~8.3 分钟/视角 @544 ⇒ 约 3.3 小时）
    "all": VIEW_NAMES,
    # 19 个：砍掉 cc*_flip（它同时与 G2 和 orig_flip 重叠，是冗余度最高的一族）
    "std19": [v for v in VIEW_NAMES if not v.startswith("cc") or not v.endswith("_flip")],
    # 11 个：base + 21 号实测单族最佳的 cc×4 + orig_flip
    "lean11": ["orig", "orig_flip"] + ["cc%03d" % int(s * 100) for s in CC_SCALES]
              + ["mild%d" % i for i in range(N_MILD)],
    # 6 个：= 26 号现配方，多尺度的低分支只要它
    "base": ["orig"] + ["mild%d" % i for i in range(N_MILD)],
}


class ViewPoolDS(Dataset):
    """一次解码产出所选视角，返回 (V,3,R,R)。⚠无 lambda（Windows spawn 要能序列化）。
    ⭐只构造 keep 里要的视角——⛔别全做完再丢，CPU 侧会成为瓶颈。"""

    def __init__(self, paths, normalize, res, seed, keep=None):
        self.paths, self.res, self.seed = paths, res, seed
        self.keep = list(keep) if keep else list(VIEW_NAMES)
        self.ks = set(self.keep)
        bic = T.InterpolationMode.BICUBIC
        self.sq = int(round(res * SQ_RATIO))
        self.off = self.sq - res                       # 224→256 时是 32，按比例放大
        self.tf_center = T.Compose([T.Resize(res, interpolation=bic), T.CenterCrop(res),
                                    T.ToTensor(), normalize])
        self.tf_sq = T.Compose([T.Resize(self.sq, interpolation=bic), T.CenterCrop(self.sq)])
        self.tf_norm = T.Compose([T.ToTensor(), normalize])
        self.tf_mild = T.Compose([
            T.RandomResizedCrop(res, scale=MILD_SCALE, ratio=RATIO, interpolation=bic),
            T.RandomHorizontalFlip(), T.ToTensor(), normalize])
        self.tf_strong = T.Compose([
            T.RandomResizedCrop(res, scale=STRONG_SCALE, ratio=RATIO, interpolation=bic),
            T.RandomHorizontalFlip(), T.ToTensor(), normalize])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        R, ks = self.res, self.ks
        V = len(self.keep)
        try:
            with Image.open(self.paths[i]) as im0:
                im = im0.convert("RGB")        # ⚠convert 必须在 with 内，否则句柄已关
        except Exception:
            return torch.zeros(V, 3, R, R), i, 0

        # ⭐每张图固定种子 ⇒ 与 worker 数/batch 大小无关，重跑逐位一致
        # ⚠⚠ 随机视角必须**按固定顺序**消费随机数，否则换 --views 会改变 mild/strong 的抽样。
        #    所以 mild/strong 一律全抽 N 个，只是不保留不要的那些。
        torch.manual_seed(self.seed * 1000003 + i)

        got = {}
        if "orig" in ks or "orig_flip" in ks:
            v_orig = self.tf_center(im)
            if "orig" in ks:
                got["orig"] = v_orig
            if "orig_flip" in ks:     # ⭐翻转在张量上做：bicubic 镜像对称、Normalize 逐通道
                got["orig_flip"] = torch.flip(v_orig, dims=[2])
        for s in CC_SCALES:
            n, nf = "cc%03d" % int(s * 100), "cc%03d_flip" % int(s * 100)
            if n in ks or nf in ks:
                t = self.tf_center(center_crop_ratio(im, s))
                if n in ks:
                    got[n] = t
                if nf in ks:
                    got[nf] = torch.flip(t, dims=[2])
        cn = ["corner_tl", "corner_tr", "corner_bl", "corner_br"]
        if any(c in ks for c in cn):
            sq, d = self.tf_sq(im), self.off
            for nm, (top, left) in zip(cn, ((0, 0), (0, d), (d, 0), (d, d))):
                if nm in ks:
                    got[nm] = self.tf_norm(sq.crop((left, top, left + R, top + R)))
        for j in range(N_MILD):                    # ⛔全抽，保住随机数序列
            t = self.tf_mild(im)
            if ("mild%d" % j) in ks:
                got["mild%d" % j] = t
        for j in range(N_STRONG):
            t = self.tf_strong(im)
            if ("strong%d" % j) in ks:
                got["strong%d" % j] = t

        return torch.stack([got[n] for n in self.keep], 0), i, 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--res", type=int, default=512, help="必须是 32 的倍数")
    ap.add_argument("--batch-size", type=int, default=2,
                    help="⚠单位是【图片】不是视角：一批 Dataset 产出 batch×24 个视角")
    ap.add_argument("--gpu-chunk", type=int, default=24,
                    help="⚠一次送进 GPU 的【实际张数】上限。8GB 卡 @512 要保守")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help=">0 时只跑前 N 张（冒烟，写 _smoke 目录）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--views", type=str, default="std19",
                    help="预设 %s，或逗号分隔的视角名。⭐GPU 成本与视角数线性"
                         % "/".join(PRESETS))
    args = ap.parse_args()

    if args.res % 32:
        ap.error("--res 必须是 32 的倍数（ViT-B/32 的 patch 是 32×32），给了 %d" % args.res)
    if args.views in PRESETS:
        keep = list(PRESETS[args.views])
    else:
        keep = [v.strip() for v in args.views.split(",")]
        bad = [v for v in keep if v not in VIEW_NAMES]
        if bad:
            ap.error("未知视角 %s。合法：%s" % (bad, VIEW_NAMES))
    V = len(keep)

    banner("58 测试集视角池  %d 个视角 @%dpx  (--views %s)" % (V, args.res, args.views))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── 载入 ckpt（两步插值，口径与 26/45 逐字一致）
    ck_path = args.ckpt if os.path.isabs(args.ckpt) else os.path.join(P.PROJECT_ROOT, args.ckpt)
    tmp = ck_path + ".readcopy"
    shutil.copy2(ck_path, tmp)                      # ⚠训练在跑时直接读会撞上半写状态
    ck = torch.load(tmp, map_location="cpu", weights_only=False)
    os.remove(tmp)
    tr = ck.get("args", {})
    res_train = tr.get("resolution", 224)
    print("  ckpt   %s  ep%s  训练@%dpx" %
          (os.path.basename(os.path.dirname(ck_path)), ck.get("epoch"), res_train))

    import clip
    model, _ = clip.load(P.CLIP_MODEL_NAME, device="cpu", download_root=P.CLIP_DOWNLOAD_ROOT)
    model.float()
    _ft.interpolate_pos_embed(model, res_train)     # ⚠必须先插到 ckpt 网格再 load
    visual = model.visual
    visual.load_state_dict(ck["visual"], strict=True)
    if args.res != res_train:
        _ft.interpolate_pos_embed(model, args.res)  # 再插到评估网格（⛔顺序不能反）
        print("  ⭐推理分辨率 %d ≠ 训练 %d → 两步插值，口径同 26/34" % (args.res, res_train))
    visual = model.visual.to(device).eval()
    print("  骨干权重已载入（strict=True 通过）")

    # ── 数据
    paths = _ex.list_test_files()
    if args.limit:
        paths = paths[:args.limit]
    N = len(paths)
    _, eval_tf = _ft.build_transforms(args.res, 0.5)
    normalize = [t for t in eval_tf.transforms if isinstance(t, T.Normalize)][0]

    ds = ViewPoolDS(paths, normalize, args.res, args.seed, keep)
    print("  测试集 %d 张 × %d 视角 = %d 次前向" % (N, V, N * V))
    print("  视角：%s" % ",".join(keep))
    print("  四角：先归一到 %d 方块，再切 4 个 %d 的角（偏移 %d）" % (ds.sq, args.res, ds.off))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)

    out = np.zeros((V, N, P.FEATURE_DIM), dtype=np.float16)
    failed, done, t0 = 0, 0, time.time()
    with torch.no_grad():
        for imgs, idx, ok in dl:
            b = imgs.shape[0]
            flat = imgs.view(b * V, 3, args.res, args.res).to(device, non_blocking=True)
            feats = []
            for j in range(0, flat.shape[0], args.gpu_chunk):   # ⚠切块喂，别顶爆 8G
                feats.append(visual(flat[j:j + args.gpu_chunk]).float().cpu().numpy())
            f = np.concatenate(feats, 0).reshape(b, V, P.FEATURE_DIM)
            out[:, idx.numpy(), :] = f.transpose(1, 0, 2).astype(np.float16)
            failed += int((ok == 0).sum())
            done += b
            if done % (args.batch_size * 200) < args.batch_size:
                el = (time.time() - t0) / 60
                print("  %6d/%d  %.1f 分钟  预计总计 %.0f 分钟"
                      % (done, N, el, el / max(done, 1) * N), flush=True)

    # ── 落盘
    stamp = time.strftime("%Y%m%d_%H%M%S")
    tag = "_smoke" if args.limit else ""
    d = os.path.join(P.RUNS_DIR, "%s_pool%d_ep%s%s" % (stamp, args.res, ck.get("epoch"), tag))
    os.makedirs(d, exist_ok=True)
    np.savez(os.path.join(d, "pool.npz"), F=out,
             W=ck["W"].numpy().astype(np.float32), b=ck["b"].numpy().astype(np.float32),
             **({"s": np.float32(ck["s"])} if "s" in ck else {}))
    meta = dict(view_names=keep, groups=GROUPS, n=N, res=args.res, res_train=res_train,
                ckpt=ck_path, epoch=str(ck.get("epoch")), seed=args.seed, failed=failed,
                cosine=bool(tr.get("cosine_head", False)), subcenters=int(tr.get("subcenters", 0)),
                cc_scales=list(CC_SCALES), mild_scale=list(MILD_SCALE),
                strong_scale=list(STRONG_SCALE), ratio=list(RATIO),
                elapsed_min=(time.time() - t0) / 60, created=stamp)
    with open(os.path.join(d, "pool_meta.json"), "w", encoding="utf-8") as fj:
        json.dump(meta, fj, ensure_ascii=False, indent=2)

    # ── 体检：各族与原图的余弦（应当复现 20 号的排序 mild > corner > strong > cc）
    banner("体检：各视角族与原图的余弦相似度")
    Fn = out.astype(np.float32)
    Fn /= np.linalg.norm(Fn, axis=2, keepdims=True).clip(1e-12)
    o = Fn[keep.index("orig")]
    print("  %-22s %-10s %s" % ("族", "本次@%d" % args.res, "20号@224(冻结CLIP)"))
    ref = {"G0 原图": 1.000, "G1 原图翻转": 0.981, "G2 多尺度放大": 0.906,
           "G3 多尺度+翻转": 0.903, "G4 四角位移": 0.957,
           "G5 温和随机(现配方)": 0.966, "G6 强随机": 0.947}
    for g, vs in GROUPS.items():
        vs = [v for v in vs if v in keep]
        if not vs:
            continue
        c = float(np.mean([float((Fn[keep.index(v)] * o).sum(1).mean()) for v in vs]))
        print("  %-22s %-10.4f %.3f" % (g, c, ref.get(g, float("nan"))))
    print("\n  ⭐读法：本次的**排序**若与 20 号一致 → 视角构造正确；")
    print("  ⛔ 但绝对值会不同（分辨率与模型都变了），⛔不许拿绝对值下结论。")

    print("\n  失败 %d 张，耗时 %.1f 分钟" % (failed, (time.time() - t0) / 60))
    print("  ⭐已写出 %s\\pool.npz  (%.0f MB)"
          % (d, os.path.getsize(os.path.join(d, "pool.npz")) / 1e6))
    print("  下一步：59_recipe_pack.py --pool %s\\pool.npz --recipe <配方名>" % d)


if __name__ == "__main__":
    main()
