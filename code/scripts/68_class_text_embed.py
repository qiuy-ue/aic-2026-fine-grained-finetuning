# -*- coding: utf-8 -*-
"""68 · 每个类的 CLIP 文本原型：从训练图元数据里**自动**提取拉丁学名 → CLIP 文本塔编码 → 体检。

⚠2026-09-22 复赛适配：类别数不再写死。闸③的排名上限由 RANK_FRAC(5%) × 实际类别数 算出
  （初赛 500 类→25 名，复赛 750 类→38 名）；并加了一道「特征缓存是否属于本阶段」的硬校验。

═══════════════════════════════════════════════════════════════════════
⭐ 为什么能做、为什么合规（2026-09-18）
═══════════════════════════════════════════════════════════════════════
  · 类名官方只给数字 ID，但 37.48% 训练图的 IPTC/XMP/EXIF 里带真实拉丁学名（09-16 普查）。
    学名在**官方训练集文件自身**里，不是额外数据集（十一(二)2）；提取完全由本脚本自动完成（五、6 可复现）。
  · 十一(二)3：可用 CLIP ViT-B/32 预训练权重 = 完整双塔，文本塔合法；五、2 点名 Prompt Tuning。
  · 产物只当训练时的固定锚点（28 号 --text-anchor）或冷启动的头初值（--head-init），推理仍是单模型。

═══════════════════════════════════════════════════════════════════════
⭐ 自动筛名的三道闸（⛔不许手工挑名字，否则复赛审核过不了五、6）
═══════════════════════════════════════════════════════════════════════
  闸① 形态：正则 `[A-Z][a-z]{2,} [a-z]{3,}`（属名 + 种名），只在 SOS 之前的头部字节里找
  闸② 跨类扩散：同一短语出现在 ≥ SPREAD_MAX 个类里 = 通用词（"Description rdf"/"All rights"…），⛔剔除
      （真物种名只会出现在 1~2 个类，重复物种对如 0266/0424 正好是 2）
  闸③ 检索自检：文本原型 vs **原版 CLIP 图像类均值**（同一图文空间）的余弦，本类排名 ≤ RANK_MAX 才留；
      否则名字大概率错/CLIP 不认识 → 回退到图像类均值当原型
  缺名类一律回退图像类均值（它和文本原型在同一空间，是"视觉版类名"）。

用法：
  python scripts\\68_class_text_embed.py                       # 全流程，~5 分钟 CPU
  python scripts\\68_class_text_embed.py --l-feat cache\\kd_L_feat.npy   # 附加：量微调后(L)特征与文本的对齐度
产物：cache\\class_latin_names.json   cache\\text_protos.npz（T/T_text/M/source/names/rank）
"""
import _bootstrap  # noqa: F401
from _bootstrap import banner

import argparse
import collections
import json
import os
import re
import time

import numpy as np
import torch

from configs import paths as P

BIN = re.compile(rb"\b([A-Z][a-z]{2,}) ([a-z]{3,})\b")
HEAD_BYTES = 65536
SPREAD_MAX = 8          # 闸②：出现在 ≥8 个类的短语视为通用词
MIN_COUNT = 2           # 一个类里至少 2 张图带同一个名字
RANK_FRAC = 0.05        # 闸③：文本原型检索本类图像均值，排名要进前 5%
# ⚠原来这里硬写 RANK_MAX = 25，那是按初赛 500 类算的（500×5%）。复赛 750 类还用 25，
#   等于把闸③从「前 5%」悄悄收紧成「前 3.3%」，会多刷掉一批本来合格的学名。
#   改成按实际类别数算，换阶段不用再记得改这一行。
TEMPLATES = ("a photo of a {}.",
             "a close-up photo of a {}, a species of animal or plant.",
             "a photo of the {}, in the wild.")


def scan_names(train_dir):
    classes = sorted(d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d)))
    per_cls = {}
    spread = collections.Counter()
    t0 = time.time()
    for ci, c in enumerate(classes):
        cnt = collections.Counter()
        d = os.path.join(train_dir, c)
        for fn in os.listdir(d):
            try:
                with open(os.path.join(d, fn), "rb") as f:
                    b = f.read(HEAD_BYTES)
            except OSError:
                continue
            sos = b.find(b"\xff\xda")
            head = b[:sos] if sos > 0 else b
            for name in set(m.group(0).decode("latin-1") for m in BIN.finditer(head)):
                cnt[name] += 1
        per_cls[c] = cnt
        for name in cnt:
            spread[name] += 1
        if (ci + 1) % 50 == 0:
            print("  扫到 %d/%d 类  %.1f 分钟" % (ci + 1, len(classes), (time.time() - t0) / 60), flush=True)
    return classes, per_cls, spread


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--l-feat", type=str, default=None,
                    help="69 号存的微调模型(L)训练图特征 *_feat.npy（同目录需有 *_paths.txt），量对齐度用")
    ap.add_argument("--sample", type=int, default=20000, help="零样本体检抽多少张训练图")
    args = ap.parse_args()

    banner("68 类文本原型  学名提取 → CLIP 文本编码 → 体检")
    classes, per_cls, spread = scan_names(P.TRAIN_DIR)
    K = len(classes)
    RANK_MAX = max(5, int(round(K * RANK_FRAC)))
    print("  阶段 %s：%d 个类，闸③排名上限 = 前 %.0f%% = %d 名"
          % (P.STAGE, K, RANK_FRAC * 100, RANK_MAX))

    # ── 闸①② 选名
    names, cands = [None] * K, {}
    for i, c in enumerate(classes):
        ok = [(n, k) for n, k in per_cls[c].most_common() if spread[n] < SPREAD_MAX and k >= MIN_COUNT]
        cands[c] = [{"name": n, "count": k, "spread": spread[n]} for n, k in ok[:5]]
        if ok:
            names[i] = ok[0][0]
    n_named = sum(1 for x in names if x)
    generic = [n for n, s in spread.most_common(15) if s >= SPREAD_MAX]
    print("  闸①②：%d/%d 类有候选学名；被判为通用词的短语（前 15）：%s" % (n_named, K, generic))

    # ── 原版 CLIP 图像类均值（与文本同一空间）
    F = np.load(P.F_TRAIN_FEAT).astype(np.float32)
    with open(P.F_TRAIN_PATH, encoding="utf-8") as f:
        fp = [ln.strip() for ln in f if ln.strip()]
    # ⛔特征缓存必须是本阶段的。初赛那份是 103218×512/500 类，行数和类别含义都对不上，
    #   放着不管会静默错位跑完全程。这里宁可硬崩，也不要错得很安静。
    if len(fp) != F.shape[0] or len(fp) != P.EXPECT_NUM_TRAIN:
        raise SystemExit(
            "特征缓存对不上：%s 有 %d 行，paths 有 %d 行，本阶段(%s)应为 %d 行。\n"
            "多半是 cache/ 里还留着上一阶段的缓存 —— 先跑 01_extract_features.py 重建。"
            % (P.F_TRAIN_FEAT, F.shape[0], len(fp), P.STAGE, P.EXPECT_NUM_TRAIN))
    cls2i = {c: i for i, c in enumerate(classes)}   # 原来用 classes.index()，750×148695 次字符串比较
    y = np.array([cls2i[os.path.basename(os.path.dirname(p))] for p in fp], dtype=np.int64)
    F /= np.linalg.norm(F, axis=1, keepdims=True).clip(1e-12)
    M = np.zeros((K, F.shape[1]), np.float32)
    for i in range(K):
        M[i] = F[y == i].mean(0)
    M /= np.linalg.norm(M, axis=1, keepdims=True).clip(1e-12)

    # ── CLIP 文本编码（CPU 即可，1500 条提示词）
    import clip
    model, _ = clip.load(P.CLIP_MODEL_NAME, device="cpu", download_root=P.CLIP_DOWNLOAD_ROOT)
    model.float().eval()
    T_text = np.full((K, F.shape[1]), np.nan, np.float32)
    with torch.no_grad():
        idx = [i for i in range(K) if names[i]]
        for j in range(0, len(idx), 64):
            chunk = idx[j:j + 64]
            embs = []
            for tpl in TEMPLATES:
                tok = clip.tokenize([tpl.format(names[i]) for i in chunk])
                e = model.encode_text(tok).float()
                embs.append(e / e.norm(dim=1, keepdim=True))
            e = torch.stack(embs, 0).mean(0)
            e = e / e.norm(dim=1, keepdim=True)
            T_text[chunk] = e.numpy()

    # ── 闸③ 检索自检：文本原型能不能在 K 个图像类均值里找回自己的类
    rank = np.full(K, -1, np.int64)
    source = np.array(["imgmean"] * K, dtype=object)
    T = M.copy()
    for i in range(K):
        if not names[i]:
            continue
        sims = M @ T_text[i]
        r = int((sims > sims[i]).sum()) + 1
        rank[i] = r
        if r <= RANK_MAX:
            T[i] = T_text[i]
            source[i] = "text"
    n_text = int((source == "text").sum())
    print("  闸③：%d/%d 个有名类通过检索自检（本类排名≤%d），%d 个回退图像均值；无名 %d 类"
          % (n_text, n_named, RANK_MAX, n_named - n_text, K - n_named))
    print("  排名分布（有名类）：中位 %s，≤1 的 %d，≤5 的 %d，>100 的 %d"
          % (int(np.median(rank[rank > 0])) if n_named else "-",
             int((rank == 1).sum()), int(((rank > 0) & (rank <= 5)).sum()), int((rank > 100).sum())))

    # ── 零样本体检：在原版 CLIP 特征上，文本原型 / 图像均值 / 混合原型 的 top-1
    rng = np.random.RandomState(0)
    sub = rng.choice(len(y), min(args.sample, len(y)), replace=False)
    def acc(protos, mask=None):
        pr = (F[sub] @ protos.T).argmax(1)
        ok = pr == y[sub]
        if mask is not None:
            keep = mask[y[sub]]
            return float(ok[keep].mean()), int(keep.sum())
        return float(ok.mean()), len(sub)
    has_txt = source == "text"
    Tt_only = np.where(np.isnan(T_text), 0, T_text)
    a_img, _ = acc(M)
    a_mix, _ = acc(T)
    a_txt, n_t = acc(Tt_only, has_txt)
    print("  零样本 top-1（原版 CLIP 特征，抽 %d 张）：图像类均值 %.4f | 最终原型(文本+回退) %.4f | "
          "纯文本(只算有文本的类, %d 张) %.4f" % (len(sub), a_img, a_mix, n_t, a_txt))

    # ── 可选：微调后特征（L）与文本原型的对齐度
    if args.l_feat:
        Fl = np.load(args.l_feat).astype(np.float32)
        with open(args.l_feat.replace("_feat.npy", "_paths.txt"), encoding="utf-8") as f:
            lp = [ln.strip() for ln in f if ln.strip()]
        yl = np.array([cls2i[os.path.basename(os.path.dirname(p))] for p in lp], dtype=np.int64)
        Fl /= np.linalg.norm(Fl, axis=1, keepdims=True).clip(1e-12)
        subl = rng.choice(len(yl), min(args.sample, len(yl)), replace=False)
        Ml = np.zeros_like(M)
        for i in range(K):
            Ml[i] = Fl[yl == i].mean(0)
        Ml /= np.linalg.norm(Ml, axis=1, keepdims=True).clip(1e-12)
        zs_l = float(((Fl[subl] @ T.T).argmax(1) == yl[subl]).mean())
        own = float(np.mean([Ml[i] @ T[i] for i in range(K)]))
        off = float(np.mean((Ml @ T.T)[~np.eye(K, dtype=bool)]))
        print("  ⭐微调后(L)特征 vs 文本原型：零样本 top-1 %.4f | 本类余弦均值 %.4f vs 他类 %.4f"
              % (zs_l, own, off))
        print("     读法：top-1 若仍 >0.3 ⇒ 空间没转远，λ=0.3 的锚点是温和约束；若 <0.05 ⇒ 空间已转离，"
              "锚点会是大扰动，λ 要降到 0.05~0.1 或先做 warmup")

    # ── 落盘
    np.savez(os.path.join(P.CACHE_DIR, "text_protos.npz"), T=T, T_text=T_text, M=M,
             source=np.array(source.tolist()), names=np.array([x or "" for x in names]), rank=rank,
             templates=np.array(TEMPLATES), classes=np.array(classes))
    with open(os.path.join(P.CACHE_DIR, "class_latin_names.json"), "w", encoding="utf-8") as f:
        json.dump({"classes": {c: {"chosen": names[i], "source": str(source[i]), "rank": int(rank[i]),
                                   "candidates": cands[c]} for i, c in enumerate(classes)},
                   "generic_phrases": generic, "params": dict(spread_max=SPREAD_MAX, min_count=MIN_COUNT,
                                                             rank_max=RANK_MAX, templates=TEMPLATES)},
                  f, ensure_ascii=False, indent=1)
    print("  ⭐已写出 cache\\text_protos.npz（T %s）与 cache\\class_latin_names.json" % (T.shape,))
    print("  例：0000=%s  0050=%s  0155=%s" % (names[0], names[50] if K > 50 else "-", names[155] if K > 155 else "-"))


if __name__ == "__main__":
    main()
