# -*- coding: utf-8 -*-
"""75 · 只重训分类头（解耦学习 cRT）：骨干不动，用训练集特征按类均衡采样重训余弦头，再换到测试特征池上出包。

为什么（大白话）：
  A2 的第一轮伪标签里有 16 个类一张都没有（7 个训练集最小的类 + 9 个 A1 几乎不判的类），
  A2 在这 16 类上一共只判了 29 张（测试集里约 800 张）。骨干学到的特征未必认不出它们，
  更可能是分类头被「训练集 + 伪标签」的类别比例带偏了。
  做法：骨干冻住，只把最后的分类头重训一遍，训练时每个类每轮抽到的张数一样多（小类有放回重复）。

⭐只用训练集训练头，⛔一张测试图都不参与训练；测试特征池只在最后出包时用。
⭐单模型：A2 的骨干 + 新的分类头，单次前向。

流程：
  1. 69 号用 A2 权重抽训练集特征（orig+flip 平均、L2 归一，@448）→ cache\\trfeat_A2_feat.npy / _prob.npy / _paths.txt
  2. 去噪：每个类内部按 A2 给它自己标签的概率排序，去掉最低的 15%（比例同 co-teaching 的 τ=0.15，
     但按类排名，不会因为小类损失整体偏高就把小类整批丢掉）
  3. 从 A2 的头初始化（行归一），按类均衡采样重训 W；尺度 s 固定为 A2 的值，logits 尺度不变
  4. 按参考包 recipe.json 的同一口径（59 号 form）聚合测试池特征 → 新头 logits → 出包
     出的包带 test_logits.npy，直接交给 74 号做均衡分配

自检：先用 A2 原来的头走一遍本脚本的测试口径，argmax 必须与参考包 CSV 逐张一致，否则拒绝出包。

用法：
  python scripts\\75_rehead_crt.py --feat cache\\trfeat_A2 --ref-pkg runs\\20260926_222804_rcp_A2_w3cc_ms448_544_ep12
  python scripts\\74_balance_pack.py --pkg runs\\<上一步生成的 rcp_A2crt 包>
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
import torch.nn.functional as TF

from configs import paths as P


def _load_sibling(fname, modname):
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(modname, os.path.join(here, fname))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


rcp59 = _load_sibling("59_recipe_pack.py", "rcp59")
pred26 = rcp59.pred26
_ex = rcp59._ex


def parse_w(s):
    if not s:
        return None
    return {k.strip(): float(v) for k, _, v in (kv.partition(":") for kv in s.split(","))}


def test_features(rc):
    """按参考包 recipe.json 聚合测试特征，口径与 59 号逐字一致。返回 (X, head, meta)。"""
    wh = dict(zip(rc["views"], rc["weights"]))
    X, meta, head, _, _ = rcp59.form(rc["pool"], rc["views"], wh)
    if rc.get("pool_lo"):
        Xlo, mlo, _, _, _ = rcp59.form(rc["pool_lo"], rc["views_lo"], parse_w(rc.get("weights_lo_arg")))
        if mlo["n"] != meta["n"]:
            raise SystemExit("⛔ 两个池子行数不同")
        w_hi = float(rc.get("w_hi", 1.0))
        X = rcp59.l2((Xlo + w_hi * X) / (1.0 + w_hi))
    return X.astype(np.float32), head, meta


def cos_logits(X, W, s):
    Wn = W / np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-12)
    return float(s) * (X @ Wn.T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", required=True, help="69 号的输出前缀，如 cache\\trfeat_A2")
    ap.add_argument("--ref-pkg", required=True, help="参考包（A2 原始包），读它的 recipe.json 和 CSV 做自检")
    ap.add_argument("--drop-frac", type=float, default=0.15, help="每类内部去掉 A2 置信度最低的比例")
    ap.add_argument("--per-class", type=int, default=200, help="每轮每类抽几张（小类有放回）")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="A2crt")
    ap.add_argument("--pseudo-ref", default=os.path.join(P.PROJECT_ROOT, "cache", "pseudo_A1.npz"),
                    help="只用于诊断：统计哪些类在第一轮伪标签里是 0 张（不参与训练）")
    ap.add_argument("--dry-run", action="store_true", help="只打印统计，不写包")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    feat_path = args.feat if os.path.isabs(args.feat) else os.path.join(P.PROJECT_ROOT, args.feat)
    ref = args.ref_pkg if os.path.isabs(args.ref_pkg) else os.path.join(P.PROJECT_ROOT, args.ref_pkg)
    banner("75 只重训分类头（cRT）  特征 %s" % os.path.basename(feat_path))

    # ── 训练集特征与标签（标签 = 路径里的类文件夹名，与 01 号 list_train_files 同一规则）
    F = np.load(feat_path + "_feat.npy").astype(np.float32)
    prob = np.load(feat_path + "_prob.npy")
    paths = [l.rstrip("\n") for l in open(feat_path + "_paths.txt", encoding="utf-8") if l.strip()]
    cn = json.load(open(P.F_MANIFEST, encoding="utf-8"))["class_names"]
    cidx = {c: i for i, c in enumerate(cn)}
    y = np.array([cidx[os.path.basename(os.path.dirname(p))] for p in paths], dtype=np.int64)
    N, D = F.shape
    K = len(cn)
    if len(paths) != N or prob.shape != (N, K):
        raise SystemExit("⛔ 特征/概率/路径行数对不上")
    F /= np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-12)
    cnt = np.bincount(y, minlength=K)
    print("  训练特征 %d × %d   类 %d   每类 min=%d 中位=%d max=%d" % (N, D, K, cnt.min(), np.median(cnt), cnt.max()))

    # ── 按类去噪
    py = prob[np.arange(N), y].astype(np.float32)
    keep = np.ones(N, bool)
    for c in range(K):
        ii = np.where(y == c)[0]
        k = int(np.floor(args.drop_frac * len(ii)))
        if k:
            keep[ii[np.argsort(py[ii])[:k]]] = False
    print("  按类去掉 A2 置信度最低的 %.0f%%：保留 %d / %d 张   A2 头在训练图上与标签一致 %.4f（in-sample，只看方向）"
          % (100 * args.drop_frac, keep.sum(), N, (prob.argmax(1) == y).mean()))
    by_c = [np.where(keep & (y == c))[0] for c in range(K)]

    # ── 参考口径与自检
    rc = json.load(open(os.path.join(ref, "recipe.json"), encoding="utf-8"))
    X, head, meta = test_features(rc)
    if not meta.get("cosine", False):
        raise SystemExit("⛔ 本脚本只支持余弦头")
    W0, s0 = head["W"].astype(np.float32), float(head["s"])
    lg0 = cos_logits(X, W0, s0)
    ref_rows = [l.strip().split(",") for l in open(os.path.join(ref, "pred_results.csv"), encoding="utf-8") if l.strip()]
    test_paths = _ex.list_test_files()
    if [os.path.basename(p) for p in test_paths] != [r[0] for r in ref_rows]:
        raise SystemExit("⛔ 测试集顺序与参考包不一致")
    ref_pred = np.array([cidx[r[1]] for r in ref_rows])
    same = int((lg0.argmax(1) == ref_pred).sum())
    print("  ⭐自检：A2 原头 + 同一口径 与参考包 argmax 一致 %d/%d" % (same, len(ref_pred)))
    if same != len(ref_pred):
        raise SystemExit("⛔ 口径对不上，拒绝出包")
    print("  A2 头尺度 s = %.3f（固定不训）" % s0)

    # ── 重训头
    W = torch.tensor(W0 / np.linalg.norm(W0, axis=1, keepdims=True), requires_grad=True)
    Ft = torch.from_numpy(F)
    yt = torch.from_numpy(y)
    opt = torch.optim.AdamW([W], lr=args.lr, weight_decay=0.0)
    steps_per_ep = int(np.ceil(K * args.per_class / args.batch_size))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * steps_per_ep)
    # 诊断用：第一轮伪标签里一张都没有的类（没有伪标签文件就退回训练集最小的 16 类）
    small, small_name = np.argsort(cnt)[:16], "训练集最小的 16 类"
    if args.pseudo_ref and os.path.exists(args.pseudo_ref):
        pc = np.bincount(np.load(args.pseudo_ref)["labels"].astype(np.int64), minlength=K)
        small, small_name = np.where(pc == 0)[0], "伪标签为 0 的 %d 类" % int((pc == 0).sum())
    print("  诊断类（%s）：%s" % (small_name, " ".join(cn[c] for c in small)))
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        idx = np.concatenate([rng.choice(ii, args.per_class, replace=len(ii) < args.per_class) for ii in by_c])
        rng.shuffle(idx)
        tot, n = 0.0, 0
        for j in range(0, len(idx), args.batch_size):
            b = torch.from_numpy(idx[j:j + args.batch_size])
            Wn = TF.normalize(W, dim=1)
            lg = s0 * Ft[b] @ Wn.T
            loss = TF.cross_entropy(lg, yt[b], label_smoothing=args.label_smoothing)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss) * len(b)
            n += len(b)
        with torch.no_grad():
            Wc = W.detach().numpy()
        pt = cos_logits(X, Wc, s0).argmax(1)
        tc = np.bincount(pt, minlength=K)
        print("  ep%02d  loss %.4f  | 测试集原始预测：与 A2 不同 %5d 张  覆盖 %d 类  每类 min=%d max=%d  诊断类合计 %d 张"
              % (ep, tot / n, (pt != ref_pred).sum(), (tc > 0).sum(), tc.min(), tc.max(), tc[small].sum()))

    Wf = W.detach().numpy().astype(np.float32)
    lg = cos_logits(X, Wf, s0)
    pred = lg.argmax(1)
    ptr = cos_logits(F, Wf, s0).argmax(1)
    print("\n  新头在训练图上与标签一致 %.4f（A2 头 %.4f，in-sample）   用时 %.1f 秒"
          % ((ptr == y).mean(), (prob.argmax(1) == y).mean(), time.time() - t0))
    c0, c1 = np.bincount(ref_pred, minlength=K), np.bincount(pred, minlength=K)
    print("  %s在测试集上被判张数（A2 原头 → 新头）：" % small_name)
    print("   " + "  ".join("%s:%d→%d" % (cn[c], c0[c], c1[c]) for c in small))

    if args.dry_run:
        print("\n  DRY-RUN：⛔未写包")
        return
    stamp = time.strftime("%Y%m%d_%H%M%S")
    tag = rc["tag"].replace("A2", args.tag, 1) if rc["tag"].startswith("A2") else args.tag + "_" + rc["tag"]
    out = os.path.join(P.RUNS_DIR, "%s_rcp_%s_ms448_544_ep%s" % (stamp, tag, meta["epoch"]))
    pred26.write_submission(out, "cRT 新头", test_paths, pred, cn, K)
    np.save(os.path.join(out, "test_logits.npy"), lg.astype(np.float32))
    np.savez(os.path.join(out, "head_crt.npz"), W=Wf, s=np.float32(s0))
    with open(os.path.join(out, "recipe.json"), "w", encoding="utf-8") as fj:
        json.dump(dict(rc, crt=dict(feat=feat_path, ref_pkg=ref, drop_frac=args.drop_frac, per_class=args.per_class,
                                    epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
                                    label_smoothing=args.label_smoothing, seed=args.seed,
                                    changed_vs_ref=int((pred != ref_pred).sum())),
                       created=stamp), fj, ensure_ascii=False, indent=2)
    print("\n  ⭐输出 -> %s" % out)
    print("  下一步：python scripts\\74_balance_pack.py --pkg %s" % out)


if __name__ == "__main__":
    main()
