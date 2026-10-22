# -*- coding: utf-8 -*-
"""
教师-学生第一步：用当前最好的模型给【测试集】打伪标签。

═══════════════════════════════════════════════════════════════════════
一、⛔【已作废】下面这段推理在 2026-09-06 被实测证伪，原文保留当反面教材
═══════════════════════════════════════════════════════════════════════

⛔⛔**以下方框内是错的，别信，别再拿它做决策。**
┌────────────────────────────────────────────────────────────────────┐
│ 实测建立的换算关系（n=2 重现）：改骨干类方法，本地涨幅 × 0.42 才是真实涨幅。 │
│     要榜上 0.92 → 需要本地 0.6129 + (0.92−0.59)/0.42 = 1.399，不可能。   │
│     即使本地做到完美 1.0，榜上也只有 0.753。                             │
│ 所以"把模型训得更好"这条路有天花板，而天花板不是模型能力，是那个折扣。       │
└────────────────────────────────────────────────────────────────────┘

**次日实测**：v2 配方本地只涨 0.0228，按上式预测榜上 +0.95，**实际 +9.02，差 9.4 倍**。
榜分从 62.10 一把冲到 71.12，**直接穿过我宣布的"天花板 75.3"下方那条推理**。

⭐**错在哪（两条，都要记住）**：
  ① **两个点定不出一条曲线是直的还是弯的。** n=2 的"重现"只是巧合的近邻，不是规律。
  ② **倍率不是"方法属于哪一族"的属性**，而是"这次进步里，
     死记硬背 vs 真本事+对齐域 各占多大比例"的属性。
     v2 配方的涨幅主要来自**对齐测试域**（JPEG 退化增强），
     而本地验证集是高清原图 —— **尺子上根本没有这一格刻度**，所以它读成了 1/10。

⛔**更要命的错**：我拿三条真实的硬约束（骨干锁死/类别细粒度/每类样本少）
论证"到不了 90"。三条约束全是真的，**结论却是错的**——因为当时真正的瓶颈
是①正则化太弱 ②域没对齐，**这两条都不在我列的约束里**。
⭐**报天花板前必须区分「物理不可能」和「我还没试出来」。**

═══════════════════════════════════════════════════════════════════════
一之二、伪标签为什么仍然值得做（这段是对的，与上面无关）
═══════════════════════════════════════════════════════════════════════

已实测的域偏移（不是推测，是量出来的）：
    训练集：长边中位 640，短边中位 480，文件 85.4 KB，JPEG 量化表 17.3
    测试集：长边**全部 500**，短边中位 375，文件 38.4 KB，量化表 29.0（q≈75）

→ ⭐**模型只在"训练集分布"上学过，从没在"测试集分布"上学过。**
→ **最直接的解法：让它直接在测试图上学。** 这就是伪标签自训练。
  （另一条互补解法 = 把训练图退化成考卷规格，即 v2 里的 degrade 增强，已实测 +9 分那批的主力。）

═══════════════════════════════════════════════════════════════════════
二、⚠三个不做就会翻车的设计点
═══════════════════════════════════════════════════════════════════════

**① 用 TTA 版预测当教师** —— 教师质量决定学生上限。TTA 已实测在榜上有效，白拿。

**② ⭐必须做类别均衡选取，不能只按置信度卡阈值**
   只卡阈值 → 模型本来就预测得多的类，高置信样本也多 → 选出来的伪标签更偏
   → 学生学完偏得更狠 → **自我强化的死循环**（这正是我们在"清洗"上栽过的那种坑）。
   做法：**每个类各取置信度最高的前 k 个**，让伪标签集接近均衡。

   ⚠**别和"均衡分配"搞混，名字像，是两回事，一个有用一个是死路**：
     ✅**均衡选取**（本脚本做的）：挑训练用的伪标签时逐类限额 → 防自我强化 → 有用
     ⛔**均衡分配**（Sinkhorn 直推，35 号脚本实测）：把最终预测硬掰成每类 50 张
        → **oracle 上界只有 +0.0024**，即使作弊用真实配额也就这么多 → **死路**
     原因：细粒度分类的错误是**成对互换型**（A 判成 B 的同时 B 判成 A），
     总量完全均衡，**总量约束对它免疫**。⭐分布偏斜是模型弱的**症状**，不是独立的病
     —— 榜分 61.30→72.76 期间，预测分布标准差自己从 21.42 掉到 12.94，没人去掰它。

**③ 伪标签样本的权重要低于真标签**
   伪标签里必然有错（教师 2026-09-06 已到 72.76）。给它们 0.5 的权重，
   让真标签仍然主导，伪标签只负责"把模型往测试集分布上拉"。

**④ ⚠模型是 under-confident 的，阈值别照搬直觉**
   实测：测试集平均置信度 0.539，而真实准确率 0.7276 —— **模型比自己以为的更准**。
   这是 label smoothing 0.1 + Mixup 的必然签名（训练目标本来就不让它输出 1.0）。
   → ⭐`--min-conf 0.5` 定保守了，下一轮可放宽阈值 / 提高 --per-class 上限，多喂点数据。

═══════════════════════════════════════════════════════════════════════
三、⚠合规与评估
═══════════════════════════════════════════════════════════════════════

**只用官方给的测试集【图片】，不碰任何标签**，属于 transductive（直推）推断，
是半监督学习的标准做法，不违反赛题约束（禁的是换模型/多模型集成）。

⛔**但本地尺子对它无效**：我们的验证集仍是训练集分布，
而伪标签的作用恰恰是"往测试集分布靠"——**本地可能不涨甚至微跌，榜上却涨**。
**这一步必须交榜裁决，别信本地。**

用法:
  python scripts/30_pseudo_label.py                       # 自动取最新最好 ckpt
  python scripts/30_pseudo_label.py --per-class 40 --min-conf 0.5
"""
import _bootstrap  # noqa: F401
from _bootstrap import banner

import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm import tqdm

from configs import paths as P

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _load_sibling(fname, modname):
    """⚠sys.modules 注册必须在文件顶层，否则 Windows spawn 的 worker 静默挂起。"""
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(modname, os.path.join(here, fname))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


_ex = _load_sibling("01_extract_features.py", "extract01")
_aug = _load_sibling("10_extract_augmented.py", "augment10")
_ft = _load_sibling("24_finetune_full.py", "finetune24")


class DS(Dataset):
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


@torch.no_grad()
def encode(visual, tf, paths, device, bs, nw, res, desc):
    ld = DataLoader(DS(paths, tf, res), batch_size=bs, shuffle=False,
                    num_workers=nw, pin_memory=False)
    out = np.zeros((len(paths), P.FEATURE_DIM), np.float32)
    visual.eval()
    bar = tqdm(total=len(paths), desc=desc, unit="img", unit_scale=True, dynamic_ncols=True)
    for im, ix in ld:
        im = im.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
            f = visual(im)
        out[ix.numpy()] = f.float().cpu().numpy()
        bar.update(im.shape[0])
    bar.close()
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--tta", type=int, default=5, help="教师用几个增强视角（0=不用TTA）")
    ap.add_argument("--per-class", type=int, default=40,
                    help="⭐每类最多选多少个伪标签（类别均衡的关键旋钮）。"
                         "测试集 24967/500≈50，取 40 意味着大约覆盖 80%%")
    ap.add_argument("--min-conf", type=float, default=0.5,
                    help="置信度下限。低于它的一律不要，哪怕该类没选够")
    ap.add_argument("--weight", type=float, default=0.5,
                    help="伪标签样本相对真标签的权重。⚠别设 1.0，教师只有 62 分")
    # ⭐⭐零 GPU 通道：26_predict_finetuned.py 每次出包都会把 F/W/b 存进
    #   test_feat_head.npz。TTA 版存的就是**TTA 平均后**的特征——正是教师该用的那份。
    #   → 重做伪标签不需要再跑 6 遍推理（省 35 分钟 GPU）。
    ap.add_argument("--from-npz", type=str, default=None,
                    help="直接读某个出包目录的 test_feat_head.npz 当教师，⭐完全不占 GPU")
    # ⭐⭐2026-09-24 新增：直接吃 59 号出包时 --save-logits 存下的 logits（复赛主通道）
    #   为什么必须加这条，上面两条都不能用：
    #   ① --ckpt 与 --from-npz 都按 `X @ W.T + b` 算分数 —— 那是**线性头**的公式。
    #      本脚本写于换余弦头之前，从没适配过：余弦头的真实打分是 s·(x̂ @ Ŵ.T)、不用 b，
    #      而 ckpt 里的 b 是 LP 头初值、训练中没更新过却非零 ⇒ 两个公式的 argmax 对不上，
    #      拿到的伪标签**不是**模型自己的预测。
    #   ② --ckpt 走的是本脚本自己的 TTA（原图+5 个随机裁剪），⛔不是我们实际提交的
    #      ms448_544 + c3both_w3cc。教师质量决定学生上限，老师就该是**交上去的那个包**。
    #   59 号用 26 号的 apply_head（余弦头感知）算 logits，行序已断言 == list_test_files()，
    #   与它写出的 pred_results.csv 是同一份数据 ⇒ 这里直接用，零 GPU、口径逐位一致。
    ap.add_argument("--from-logits", type=str, default=None,
                    help="⭐读 59 号 --save-logits 存的 (N,K) logits 当教师（=实际提交的 TTA 包）。"
                         "同目录若有 pred_results.csv 会逐张自检，对不上直接停")
    ap.add_argument("--out", type=str, default=None,
                    help="伪标签输出路径（默认 cache/pseudo_labels.npz）。"
                         "⭐建议按教师命名，如 cache/pseudo_A1.npz，免得下一发把上一发盖掉")
    # ⭐C1：类别内相对可信度软加权（替代"统一 0.5"）
    # ⛔为什么不能直接拿全局 conf 当权重：实测各类置信度尺度差很多
    #   （conf>0.5 的样本每类 min=1 / 中位 29 / max=61）。用全局 conf →
    #   "简单类"整体权重高、"难类"整体权重低 = 告诉模型「你已经会的多练、不会的少练」，
    #   **方向正好反了**。→ 必须先按类归一化，只保留"这一类内部谁更可信"。
    # ⭐区间默认 [0.25, 0.75]，**中点恰为 0.5 = 旧的统一权重**：
    #   这样 A/B 只改「权重的分布形状」，不改「伪标签的总影响力」，是干净的单变量。
    ap.add_argument("--rank-weight", action="store_true",
                    help="⭐开启类别内排名加权（关闭则沿用统一 --weight）")
    ap.add_argument("--w-min", type=float, default=0.25)
    ap.add_argument("--w-max", type=float, default=0.75)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=3)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    lg_teacher = None

    if args.from_logits:
        # ═══ ⭐复赛主通道：教师 = 实际提交的那个 TTA 包 ═══════════════════════
        banner("30 教师打伪标签  ⭐零GPU：读 59 号 logits  %s" % args.from_logits)
        lg_teacher = np.load(args.from_logits).astype(np.float32)
        paths = _ex.list_test_files()
        if lg_teacher.ndim != 2 or len(lg_teacher) != len(paths):
            raise SystemExit("⛔ logits 形状 %s 与测试集 %d 张对不上" % (lg_teacher.shape, len(paths)))
        K = lg_teacher.shape[1]
        ckp = args.from_logits
        n_views = "recipe"
        X = None
        # ⭐自检：argmax 必须与同目录 pred_results.csv 逐张一致 —— 一次验完「行序 / 头公式 / 文件没拿错」
        sib = os.path.join(os.path.dirname(os.path.abspath(args.from_logits)), "pred_results.csv")
        if os.path.exists(sib):
            import csv
            with open(sib, newline="", encoding="utf-8-sig") as fh:
                ref = {r[0].strip(): int(r[-1]) for r in csv.reader(fh) if len(r) >= 2}
            mine = lg_teacher.argmax(1)
            same = sum(1 for i, p_ in enumerate(paths)
                       if ref.get(os.path.basename(p_)) == int(mine[i]))
            print("  ⭐自检：与同目录 pred_results.csv 一致 %d/%d (%.4f%%)"
                  % (same, len(paths), 100.0 * same / len(paths)))
            if same != len(paths):
                raise SystemExit("⛔ 与已出包的预测对不上，⛔不许拿没验过的教师去喂训练")
        else:
            print("  ⚠同目录没有 pred_results.csv，跳过自检")
    elif args.from_npz:
        # ═══ ⭐零 GPU 通道 ═══════════════════════════════════════════════
        # 26_predict_finetuned.py 出包时已经把 (F, W, b) 落盘了。
        # TTA 目录里的 F = **6 个视角 L2 平均后再 L2** 的最终特征，
        # 与它自己写出来的 pred_results.csv 是同一份数据 → 直接复用即可。
        banner("30 教师打伪标签  ⭐零GPU：直接读 %s" % os.path.basename(
            os.path.dirname(args.from_npz)))
        z = np.load(args.from_npz)
        # ⚠2026-09-07 修：下面存 teacher_soft.npz 时要用 len(views)，而 views 只在
        #   GPU 分支里存在 → 零GPU 分支必须先把视角数补上，否则主产物存完才崩。
        n_views = 6 if args.from_npz.rstrip("/\\").replace("\\", "/").split("/")[-2]\
            .endswith("_tta") else 1
        X = z["F"].astype(np.float32)
        W = z["W"].astype(np.float32)
        b = z["b"].astype(np.float32)
        K = W.shape[0]
        ckp = args.from_npz
        paths = _ex.list_test_files()
        assert len(paths) == len(X), \
            "行数对不上：list_test_files=%d 而 npz=%d" % (len(paths), len(X))
        X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        # ⭐⭐自检：拿 F@Wᵀ+b 的 argmax 去比对同目录那份已提交的 pred_results.csv。
        #   它一次性验三件事：①文件顺序一致 ②W/b 配对正确 ③特征没存错。
        #   ⛔不做这一步，"零GPU"就是在拿一个没验过的捷径去喂训练。
        sib = os.path.join(os.path.dirname(args.from_npz), "pred_results.csv")
        if os.path.exists(sib):
            import csv
            with open(sib, newline="", encoding="utf-8") as fh:
                ref = {r[0]: int(r[1]) for r in csv.reader(fh) if r}
            mine = (X @ W.T + b).argmax(1)
            same = sum(1 for i, p_ in enumerate(paths)
                       if ref.get(os.path.basename(p_)) == int(mine[i]))
            print("  ⭐自检：与同目录 pred_results.csv 一致 %d/%d (%.4f%%)"
                  % (same, len(paths), 100.0 * same / len(paths)))
            assert same == len(paths), "⛔与已提交的预测对不上，零GPU通道不可信，停手"
        else:
            print("  ⚠同目录没有 pred_results.csv，跳过自检")
    else:
        # ---------------------------------------------------------------- 找最好的 ckpt
        if args.ckpt:
            ckp = args.ckpt
        else:
            best, ckp = -1, None
            for d in os.listdir(P.RUNS_DIR):
                f = os.path.join(P.RUNS_DIR, d, "ckpt_best.pt")
                if not os.path.exists(f) or ("_ft" not in d and "_v2" not in d):
                    continue
                try:
                    m = torch.load(f, map_location="cpu",
                                   weights_only=False).get("metrics", {})
                    if m.get("matched", -1) > best:
                        best, ckp = m["matched"], f
                except Exception:
                    pass
            assert ckp, "找不到任何微调 ckpt"
            print("  自动选用本地分布匹配最高的 ckpt（%.4f）" % best)

        ck = torch.load(ckp, map_location="cpu", weights_only=False)
        res = ck["args"].get("resolution", 224)
        banner("30 教师打伪标签  ckpt=%s ep%s @%dpx"
               % (os.path.basename(os.path.dirname(ckp)), ck.get("epoch"), res))
        print("  该教师本地：分布匹配 %.4f  无泄漏 %.4f"
              % (ck.get("metrics", {}).get("matched", float("nan")),
                 ck.get("metrics", {}).get("noleak", float("nan"))))

        import clip
        model, _ = clip.load(P.CLIP_MODEL_NAME, device="cpu",
                             download_root=P.CLIP_DOWNLOAD_ROOT)
        model.float()
        _ft.interpolate_pos_embed(model, res)
        visual = model.visual
        visual.load_state_dict(ck["visual"], strict=True)
        visual = visual.to(device)
        W = ck["W"].numpy().astype(np.float32)
        b = ck["b"].numpy().astype(np.float32)
        K = W.shape[0]

        paths = _ex.list_test_files()
        _, eval_tf = _ft.build_transforms(res, 0.5)
        print("  测试集 %d 张\n" % len(paths))

        t0 = time.time()
        views = [encode(visual, eval_tf, paths, device, args.batch_size,
                        args.num_workers, res, "原图  ")]
        if args.tta:
            norm = [t for t in eval_tf.transforms if isinstance(t, T.Normalize)][0]
            aug = _aug.build_aug(norm)
            if res != 224:
                aug.transforms[0] = T.RandomResizedCrop(
                    res, scale=(0.7, 1.0), ratio=(0.85, 1.18),
                    interpolation=T.InterpolationMode.BICUBIC)
            for v in range(args.tta):
                torch.manual_seed(v)
                views.append(encode(visual, aug, paths, device, args.batch_size,
                                    args.num_workers, res, "视角%d" % (v + 1)))
        X = np.mean(views, 0)
        n_views = len(views)
        X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        print("  教师编码完成（%d 视角），%.1f 分钟"
              % (len(views), (time.time() - t0) / 60))

    # ---------------------------------------------------------------- 概率与诊断
    # ⚠只有 --from-logits 分支是余弦头安全的；另两条仍是线性头公式，余弦头 ckpt ⛔别走那两条
    lg = lg_teacher if lg_teacher is not None else X @ W.T + b
    lg = lg - lg.max(1, keepdims=True)
    p = np.exp(lg)
    p /= p.sum(1, keepdims=True)
    conf = p.max(1)
    pred = p.argmax(1)

    banner("诊断：教师的置信度长什么样")
    print("  置信度  中位 %.3f  p10 %.3f  p25 %.3f  p75 %.3f  p90 %.3f"
          % (np.median(conf), *np.percentile(conf, [10, 25, 75, 90])))
    for t in (0.3, 0.5, 0.7, 0.9):
        print("    >=%.1f 的样本占比 %.1f%%  (n=%d)" % (t, 100 * (conf >= t).mean(),
                                                        (conf >= t).sum()))
    cnt = np.bincount(pred, minlength=K)
    print("  预测类别分布：min=%d 中位=%d max=%d（均匀应为 %.0f）"
          % (cnt.min(), np.median(cnt), cnt.max(), len(paths) / K))

    # ---------------------------------------------------------------- 类别均衡选取
    banner("选取伪标签（⭐类别均衡，不是单纯卡阈值）")
    print("  ⚠为什么不能只卡阈值：模型本来预测得多的类，高置信样本也多 →")
    print("    选出来的伪标签更偏 → 学生学完偏得更狠 → 自我强化死循环。")
    print("  做法：每类各取置信度最高的前 %d 个，且置信度必须 >= %.2f\n"
          % (args.per_class, args.min_conf))
    sel, swt = [], []
    for c in range(K):
        idx = np.where((pred == c) & (conf >= args.min_conf))[0]
        if len(idx) == 0:
            continue
        idx = idx[np.argsort(-conf[idx])][:args.per_class]
        sel.append(idx)
        # ⭐C1：类别内**排名** → 权重。第 1 名拿 w_max，最后一名拿 w_min，线性。
        # ⚠为什么用排名而不是 conf 本身：排名对"这一类置信度整体偏高/偏低"天然免疫，
        #   而 conf 会把**类间尺度差**混进权重——那正是要除掉的东西（见 --rank-weight 注释）。
        n_c = len(idx)
        r = np.arange(n_c, dtype=np.float32) / max(n_c - 1, 1)
        swt.append(args.w_max - (args.w_max - args.w_min) * r)
    sel = np.concatenate(sel) if sel else np.array([], int)
    swt = (np.concatenate(swt) if swt else np.array([], np.float32)).astype(np.float32)
    if not args.rank_weight:
        swt = np.full(len(sel), args.weight, np.float32)
    scnt = np.bincount(pred[sel], minlength=K)
    print("  %-26s %8d 张 (%.1f%% 的测试集)" % ("选中的伪标签", len(sel),
                                                100 * len(sel) / len(paths)))
    print("  %-26s %d 个类没选到样本" % ("覆盖", (scnt == 0).sum()))
    print("  %-26s min=%d 中位=%d max=%d" % ("每类张数", scnt.min(), np.median(scnt),
                                              scnt.max()))
    print("  %-26s %.3f" % ("选中样本的平均置信度", conf[sel].mean()))
    print()
    print("  ⭐对比：若只卡阈值 %.2f 不做均衡 → %d 张，每类 max=%d（偏得多）"
          % (args.min_conf, (conf >= args.min_conf).sum(),
             np.bincount(pred[conf >= args.min_conf], minlength=K).max()))

    print("  %-26s %s  [%.2f, %.2f] 均值 %.3f"
          % ("逐样本权重", "⭐类别内排名加权" if args.rank_weight else "统一",
             swt.min() if len(swt) else 0, swt.max() if len(swt) else 0,
             swt.mean() if len(swt) else 0))
    if args.rank_weight:
        print("     ⭐均值应≈%.2f（=旧的统一权重）→ A/B 只改**权重分布形状**，"
              "不改伪标签总影响力，是干净的单变量" % ((args.w_min + args.w_max) / 2))

    out = args.out or os.path.join(P.CACHE_DIR, "pseudo_labels.npz")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez(out, paths=np.array([paths[i] for i in sel]), labels=pred[sel].astype(np.int64),
             conf=conf[sel].astype(np.float32), weight=np.float32(args.weight),
             sample_weight=swt, rank_weight=bool(args.rank_weight),
             teacher_ckpt=ckp, per_class=args.per_class, min_conf=args.min_conf,
             all_pred=pred.astype(np.int64), all_conf=conf.astype(np.float32),
             all_paths=np.array(paths))
    print("\n  已存 → %s" % out)

    # ⭐软标签概率矩阵单独存一份（2026-09-06 补：上一轮只存了 argmax 和 max-prob，
    #   结果任何要 logits 的分析都得重跑一次 GPU 推理）。float16 = 25MB，够用。
    sp = os.path.join(os.path.dirname(os.path.abspath(out)),
                      "teacher_soft.npz" if not args.out else
                      os.path.splitext(os.path.basename(out))[0] + "_soft.npz")
    feat = X.astype(np.float32) if X is not None else np.zeros((0,), np.float32)   # logits 通道没有特征
    np.savez_compressed(sp, prob=p.astype(np.float16), feat=feat,
                        teacher_ckpt=ckp, n_views=n_views)
    print("  ⭐软标签+特征 → %s  (prob %s float16, feat %s)"
          % (sp, p.shape, feat.shape))
    print("    用途：自蒸馏(软标签当训练目标) / 免GPU的分布诊断 / 阈值重扫，不用再跑推理。")
    print("  ⭐下一步：python scripts/28_finetune_v2.py --pseudo %s ..." % out)
    print("  ⛔提醒：本地尺子对伪标签**无效**（验证集仍是训练集分布，"
          "而伪标签的作用是往测试集分布靠）。**必须交榜裁决。**")


if __name__ == "__main__":
    main()
