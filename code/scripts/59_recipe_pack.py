# -*- coding: utf-8 -*-
"""59 · 从 58 号建好的视角池里**离线**组出任意 TTA 配方并出包。⭐零 GPU、秒级。

这是 21 号（在验证集上离线搜 TTA 配方）在**测试集**上的对应物。区别只有一个：
⛔ 测试集没有标签，所以本脚本**不排序、不挑选**，只负责把配方变成成绩包。
   排序交给唯一的裁判——排行榜，按新规则（每小时无限次、只打最后一次）一小时排一个。

═══════════════════════════════════════════════════════════════════════
聚合口径：⛔逐字对齐 26/45，不许自创
═══════════════════════════════════════════════════════════════════════
    单分辨率  X = L2( Σ wᵢ·L2(Fᵢ) / Σ wᵢ )        26:19「逐视角先 L2、平均后再 L2」
    多尺度    X = L2( (X_lo + X_hi) / 2 )          45:「X_ms = _l2((X448 + X512)/2)」
    分类      余弦头 lg = s·(f̂ @ Ŵ.T)              26:apply_head

═══════════════════════════════════════════════════════════════════════
预设配方（名字与 cache/tta_search.json 里 21 号那张表对齐）
═══════════════════════════════════════════════════════════════════════
    base    原图 + mild×5                    = 「01 ★基线」，26 号至今用的配方（6 视角）
    c3      base + cc×4                      21 号实测 Δ=+0.0095 (4.3σ) ← 单族最佳
    c6      base + cc_flip×4                 Δ=+0.0079
    c2      base + strong×4                  Δ=+0.0073
    c5      base + corner×4                  Δ=+0.0052
    c1      base + mild 再抽 4（⛔本池无）    Δ=+0.0000 ← 对照组，证明"同族凑数不付钱"
    greedy  21 号 greedy 前向选择的 13 个视角  Δ=+0.0153 ← 表上最高
    all24   全部 24 个视角等权
    all24w3 全部 24 个，原图权重 ×3           ≈「18 原图×3+全部增强」Δ=+0.0116

⚠⚠ **这些 Δ 全部测于 224px + 冻结 CLIP + 线性探针（准确率 0.63）**，
   而我们是 448训/512推/全微调（0.79）。**幅度与排序都未经验证可迁移。**
   唯一几乎不依赖迁移假设的是对照组 c1 那个 +0.0000：它证明「多样性付钱、同族凑数不付钱」，
   而现配方 base 恰好是最不多样的一个（mild 之间余弦 0.966）。
   ⭐ 所以"该加多样性"是确定的，"加哪一族最好"不确定 → 让榜单排。

用法：
  python scripts\\59_recipe_pack.py --pool runs\\xxx_pool512_ep12\\pool.npz --recipe base --dry-run
  python scripts\\59_recipe_pack.py --pool runs\\xxx_pool512_ep12\\pool.npz \\
         --pool-lo runs\\yyy_pool448_ep12\\pool.npz --recipe greedy
  # ⚡ 三池（力大砖飞放大配方：640/544/448 等权）
  python scripts\\59_recipe_pack.py --pool runs\\..\\xx_pool640_ep30\\pool.npz \\
         --pool-mid runs\\..\\xx_pool544_ep30\\pool.npz --pool-lo runs\\..\\xx_pool448_ep30\\pool.npz \\
         --recipe custom --views <views> --recipe-mid custom --views-mid <views> --recipe-lo custom --views-lo <views>
  python scripts\\59_recipe_pack.py --pool ... --recipe custom --views orig,cc085,mild0,mild1
"""
import _bootstrap  # noqa: F401
from _bootstrap import banner

import argparse
import csv
import importlib.util
import json
import os
import sys
import time

import numpy as np

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

CC = ["cc085", "cc070", "cc055", "cc040"]
CCF = [c + "_flip" for c in CC]
CORNER = ["corner_tl", "corner_tr", "corner_bl", "corner_br"]
MILD = ["mild%d" % i for i in range(5)]
STRONG = ["strong%d" % i for i in range(5)]
BASE = ["orig"] + MILD

# ⚠ greedy 那 12 步里的 "mildC0/C1/C3" 是 21 号自己的标签，本池叫 mild0/1/3。
#   这个映射是**推断**，不是查证过的——若要严格复现 21 号的 greedy，需回查 21 号源码。
GREEDY = ["orig", "corner_tl", "cc085_flip", "mild1", "cc055", "mild0",
          "corner_bl", "strong0", "mild3", "strong1", "corner_br", "cc085", "strong4"]

RECIPES = {
    "base":    (BASE, None),
    "c3":      (BASE + CC, None),
    "c6":      (BASE + CCF, None),
    "c2":      (BASE + STRONG[:4], None),
    "c5":      (BASE + CORNER, None),
    "greedy":  (GREEDY, None),
    # ⭐ None = 用池子里**全部**视角（池子建了几个就用几个）。std19 池是 20 个。
    "all":     (None, None),
    "allw3":   (None, {"orig": 3.0}),        # 原图权重 ×3，≈21号「18 原图×3+全部增强」
    "all24":   (None, None),                 # 旧名，保留兼容
    "all24w3": (None, {"orig": 3.0}),
    "cc_only": (["orig"] + CC + CCF, None),
    "div10":   (["orig"] + MILD + CC, None),  # = c3 的别名，留着好读
}


def l2(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(1e-12)


def form(pool_path, names, wmap):
    """把一个池子按配方聚合成 (N,D) 的 L2 特征。"""
    z = np.load(pool_path)
    meta = json.load(open(os.path.join(os.path.dirname(pool_path), "pool_meta.json"),
                          encoding="utf-8"))
    vn = meta["view_names"]
    if names is None:
        names = list(vn)
    miss = [n for n in names if n not in vn]
    if miss:
        raise SystemExit("⛔ 池子里没有这些视角：%s\n   池子有：%s" % (miss, vn))
    F = z["F"].astype(np.float32)                       # (V,N,D)
    w = np.array([(wmap or {}).get(n, 1.0) for n in names], dtype=np.float32)
    idx = [vn.index(n) for n in names]
    acc = np.tensordot(w, l2(F[idx]), axes=(0, 0)) / w.sum()   # Σwᵢ·L2(Fᵢ)/Σw
    head = dict(W=z["W"], b=z["b"])
    if "s" in z.files:
        head["s"] = float(z["s"])
    return l2(acc), meta, head, names, w


def read_csv_pred(p):
    m = {}
    with open(p, newline="", encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                m[row[0].strip()] = row[-1].strip()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=str, required=True, help="高分辨率池子（或唯一的池子）")
    ap.add_argument("--pool-lo", type=str, default=None,
                    help="低分辨率池子；给了就做多尺度 L2((X_lo+X_hi)/2)，口径同 45 号")
    # ⚡ 力大砖飞三池扩展：额外接受一个中间分辨率池子，三池等权 X = L2((X_lo + X_mid + X_hi)/3)。
    #    ⛔ 不传 --pool-mid 时行为与旧版逐位一致（向后兼容）。
    ap.add_argument("--pool-mid", type=str, default=None,
                    help="中间分辨率池子（三池扩展）；与 --pool-lo 一起构成 640/544/448 三池")
    ap.add_argument("--recipe-mid", type=str, default="base",
                    help="【中分支】配方，默认 base（语义同 --recipe-lo）")
    ap.add_argument("--views-mid", type=str, default=None,
                    help="--recipe-mid custom 时的逗号分隔视角名")
    ap.add_argument("--weights-mid", type=str, default=None,
                    help="【中分支】逐视角权重覆盖，格式同 --weights")
    ap.add_argument("--w-mid", type=float, default=1.0,
                    help="三池时【中分支】相对权重（默认 1.0 = 三池等权）")
    ap.add_argument("--recipe", type=str, default="base",
                    help="【高分支】配方。预设名 或 custom（配 --views）。可用：" + "/".join(RECIPES))
    # ⭐低分支单独指定：实验 S 的设计就是「448 侧固定 base、只换 544 侧的配方」——
    #   这样对照 Q@544=79.8734 才是**单变量**。而且 448 底座池只建了 base 那 6 个视角，
    #   把高分支配方原样传给它会直接报「池子里没有这些视角」。
    ap.add_argument("--recipe-lo", type=str, default="base",
                    help="【低分支】配方，默认 base（= 26 号现配方，也是 448 底座池唯一有的）")
    ap.add_argument("--views", type=str, default=None, help="--recipe custom 时的逗号分隔视角名")
    # ⭐2026-09-16 补：低分支也要能自定义。理由——c3 上榜 +0.5808(4.8σ)，但 cc 族在总聚合里
    #   的实际占比只有 4/10 × 1/2 = **20%**，因为**低分支一个 cc 都没有**。
    #   沿"放大"这个已验证方向再推一档，最自然的一步就是让低分支也吃 cc。
    ap.add_argument("--views-lo", type=str, default=None,
                    help="--recipe-lo custom 时的逗号分隔视角名（⚠低分支池里得真有这些视角）")
    ap.add_argument("--weights-lo", type=str, default=None,
                    help="【低分支】逐视角权重覆盖，格式同 --weights")
    # ⭐ 2026-09-16 新增（纯 CPU，向后兼容：不传就与旧版逐位一致）
    ap.add_argument("--weights", type=str, default=None,
                    help="【高分支】逐视角权重覆盖，形如 cc085:2,cc070:2,cc055:2,cc040:2。"
                         "未列出的视角权重 = 1（或预设里的值）。⭐可与预设 --recipe 叠加。")
    ap.add_argument("--w-hi", type=float, default=1.0,
                    help="多尺度时【高分支】的相对权重：X = L2((X_lo + w_hi·X_hi)/(1+w_hi))。"
                         "默认 1.0 = 45 号原口径（等权），⛔改了就不再是 Q@544 的对照口径。")
    ap.add_argument("--tag", type=str, default=None,
                    help="输出目录里用的配方标签（默认 = --recipe；custom 配方强烈建议给一个）")
    ap.add_argument("--compare", type=str, default=None,
                    help="拿一个已有的 pred_results.csv 比配对分歧（实算误差棒）")
    ap.add_argument("--dry-run", action="store_true", help="只算不写包")
    # ⭐ 推理期后处理（Sinkhorn/先验校正/融合）需要的是 logits，不是 argmax。
    #   本脚本纯 numpy、秒级，重跑一次就能拿到，所以**不默认存**（25 MB/次，会堆满盘）。
    #   ⚠ 可与 --dry-run 并用：只导出 logits、⛔不落成绩包。
    ap.add_argument("--save-logits", type=str, default=None,
                    help="把 (N,K) float32 logits 另存为这个 .npy 路径（⭐可配 --dry-run 只导数不出包）")
    args = ap.parse_args()

    if args.recipe == "custom":
        if not args.views:
            ap.error("--recipe custom 必须配 --views")
        names, wmap = [v.strip() for v in args.views.split(",")], None
    elif args.recipe in RECIPES:
        names, wmap = RECIPES[args.recipe]
    else:
        ap.error("未知配方 %s，可用：%s" % (args.recipe, "/".join(RECIPES)))

    # ⭐ --weights 覆盖（叠加在预设 wmap 之上）。names 为 None 时（all/allw3）留到 form 里解析，
    #    所以这里只做语法检查，合法性交给 form 的 miss 检查。
    if args.weights:
        over = {}
        for kv in args.weights.split(","):
            k, sep, v = kv.strip().partition(":")
            if not sep:
                ap.error("--weights 格式应为 名字:权重[,名字:权重...]，看到 %r" % kv)
            try:
                over[k.strip()] = float(v)
            except ValueError:
                ap.error("--weights 里 %r 的权重不是数字" % k)
        if names is not None:
            bad = [k for k in over if k not in names]
            if bad:
                ap.error("--weights 里的视角不在本配方内：%s\n   本配方视角：%s" % (bad, names))
        wmap = dict(wmap or {})
        wmap.update(over)
    if args.w_hi <= 0:
        ap.error("--w-hi 必须 > 0，给了 %g" % args.w_hi)

    banner("59 离线组配方  recipe=%s" % args.recipe)
    X, meta, head, names, w = form(args.pool, names, wmap)
    print("  高分支池 @%dpx  ep%s  %d 视角：%s"
          % (meta["res"], meta["epoch"], len(names), ",".join(names)))
    if wmap:
        print("  权重：%s" % {n: float(wi) for n, wi in zip(names, w) if wi != 1.0})

    rname = args.tag or args.recipe
    tag = "%s_%d" % (rname, meta["res"])
    if args.pool_lo:
        if args.recipe_lo == "custom":
            if not args.views_lo:
                raise SystemExit("⛔ --recipe-lo custom 必须配 --views-lo")
            nl, wl = [v.strip() for v in args.views_lo.split(",")], None
        elif args.recipe_lo not in RECIPES:
            raise SystemExit("⛔ 未知 --recipe-lo %s，可用：%s" % (args.recipe_lo, "/".join(RECIPES)))
        else:
            nl, wl = RECIPES[args.recipe_lo]
        if args.weights_lo:
            ov = {}
            for kv in args.weights_lo.split(","):
                k, sep, v = kv.strip().partition(":")
                if not sep:
                    raise SystemExit("⛔ --weights-lo 格式应为 名字:权重[,...]，看到 %r" % kv)
                ov[k.strip()] = float(v)
            if nl is not None:
                bad = [k for k in ov if k not in nl]
                if bad:
                    raise SystemExit("⛔ --weights-lo 里的视角不在低分支配方内：%s" % bad)
            wl = dict(wl or {})
            wl.update(ov)
        Xlo, mlo, _, nlo, _ = form(args.pool_lo, nl, wl)
        if mlo["epoch"] != meta["epoch"]:
            raise SystemExit("⛔ 两个池子不是同一个 ckpt：ep%s vs ep%s" % (mlo["epoch"], meta["epoch"]))
        if mlo["n"] != meta["n"]:
            raise SystemExit("⛔ 两个池子行数不同：%d vs %d" % (mlo["n"], meta["n"]))
        print("  低分支池 @%dpx  recipe=%s  %d 视角：%s"
              % (mlo["res"], args.recipe_lo, len(nlo), ",".join(nlo)))
        # ⛔ w_hi=1 时逐位等于 45 号的「_l2((X448+X512)/2)」；⚠ 改了 w_hi 就不再是 Q@544 的对照口径
        X = l2((Xlo + args.w_hi * X) / (1.0 + args.w_hi))
        tag = "%s_ms%d_%d" % (rname, mlo["res"], meta["res"])
        if args.w_hi != 1.0:
            print("  ⚠ 高分支权重 w_hi=%g（默认 1.0）→ X = L2((X_lo + %g·X_hi)/%g)"
                  % (args.w_hi, args.w_hi, 1.0 + args.w_hi))
            tag += "_whi%g" % args.w_hi

    # ⚡ 三池扩展：中分支 + 三池等权（力大砖飞放大配方）
    nmid = None
    if args.pool_mid:
        if args.recipe_mid == "custom":
            if not args.views_mid:
                raise SystemExit("⛔ --recipe-mid custom 必须配 --views-mid")
            nm, wm = [v.strip() for v in args.views_mid.split(",")], None
        elif args.recipe_mid not in RECIPES:
            raise SystemExit("⛔ 未知 --recipe-mid %s，可用：%s" % (args.recipe_mid, "/".join(RECIPES)))
        else:
            nm, wm = RECIPES[args.recipe_mid]
        if args.weights_mid:
            ovm = {}
            for kv in args.weights_mid.split(","):
                k, sep, v = kv.strip().partition(":")
                if not sep:
                    raise SystemExit("⛔ --weights-mid 格式应为 名字:权重[,...]，看到 %r" % kv)
                ovm[k.strip()] = float(v)
            if nm is not None:
                bad = [k for k in ovm if k not in nm]
                if bad:
                    raise SystemExit("⛔ --weights-mid 里的视角不在中分支配方内：%s" % bad)
            wm = dict(wm or {})
            wm.update(ovm)
        if args.w_mid <= 0:
            raise SystemExit("⛔ --w-mid 必须 > 0，给了 %g" % args.w_mid)
        Xmid, mmid, _, nmid, _ = form(args.pool_mid, nm, wm)
        if mmid["epoch"] != meta["epoch"]:
            raise SystemExit("⛔ 中分支池与高分支不是同一个 ckpt：ep%s vs ep%s"
                             % (mmid["epoch"], meta["epoch"]))
        if mmid["n"] != meta["n"]:
            raise SystemExit("⛔ 中分支池行数不同：%d vs %d" % (mmid["n"], meta["n"]))
        print("  中分支池 @%dpx  recipe=%s  %d 视角：%s"
              % (mmid["res"], args.recipe_mid, len(nmid), ",".join(nmid)))
        if args.pool_lo:
            # 三池：X = L2((X_lo + w_mid·X_mid + w_hi·X_hi)/(1 + w_mid + w_hi))；默认全 1 = 等权
            X = l2((Xlo + args.w_mid * Xmid + args.w_hi * X) / (1.0 + args.w_mid + args.w_hi))
            tag = "%s_ms3_%d_%d_%d" % (rname, mlo["res"], mmid["res"], meta["res"])
            print("  ⚡ 三池聚合：@%d + @%d + @%d（w_hi=%.3g / w_mid=%.3g）"
                  % (mlo["res"], mmid["res"], meta["res"], args.w_hi, args.w_mid))
        else:
            X = l2((Xmid + args.w_hi * X) / (1.0 + args.w_hi))
            tag = "%s_ms%d_%d" % (rname, mmid["res"], meta["res"])

    lg = pred26.apply_head(X, head["W"], head["b"], head.get("s"),
                           meta.get("cosine", False), meta.get("subcenters", 0))
    pred = np.argmax(lg, 1)

    if args.save_logits:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_logits)) or ".", exist_ok=True)
        np.save(args.save_logits, lg.astype(np.float32))
        print("  ⭐ logits (%d,%d) float32 → %s  (%.1f MB)"
              % (lg.shape[0], lg.shape[1], args.save_logits,
                 os.path.getsize(args.save_logits) / 1024 ** 2))

    paths = _ex.list_test_files()
    if len(paths) != len(pred):
        raise SystemExit("⛔ 行数不符：池子 %d vs 测试集 %d" % (len(pred), len(paths)))
    cn = json.load(open(P.F_MANIFEST, encoding="utf-8"))["class_names"]
    K = head["W"].shape[0]
    cov = len(set(pred.tolist()))
    print("  覆盖 %d/%d 类" % (cov, K))
    if cov < K:
        print("  ⚠ 有 %d 个类一张都没预测到（历史包都是 500/500，值得看一眼）" % (K - cov))

    if args.compare:
        ref = read_csv_pred(args.compare)
        mine = {os.path.basename(p): cn[c] for p, c in zip(paths, pred)}
        ks = set(ref) & set(mine)
        dif = sum(1 for k in ks if ref[k] != mine[k])
        se = np.sqrt(dif) / len(ks) * 100
        print("  vs %s：分歧 %d/%d (%.2f%%)  SE=%.3f  ⭐误差棒 ±%.3f"
              % (os.path.basename(os.path.dirname(args.compare)), dif, len(ks),
                 100.0 * dif / len(ks), se, 2 * se))

    if args.dry_run:
        print("\n  DRY-RUN：⛔未写包。去掉 --dry-run 才落盘。")
        return

    stamp = time.strftime("%Y%m%d_%H%M%S")
    d = os.path.join(P.RUNS_DIR, "%s_rcp_%s_ep%s" % (stamp, tag, meta["epoch"]))
    pred26.write_submission(d, "配方 " + args.recipe, paths, pred, cn, K)
    with open(os.path.join(d, "recipe.json"), "w", encoding="utf-8") as fj:
        json.dump(dict(recipe=args.recipe, tag=rname, views=names, weights=w.tolist(),
                       weights_arg=args.weights, w_hi=args.w_hi,
                       recipe_lo=(args.recipe_lo if args.pool_lo else None),
                       views_lo=(nlo if args.pool_lo else None),
                       weights_lo_arg=args.weights_lo,
                       recipe_mid=(args.recipe_mid if args.pool_mid else None),
                       views_mid=nmid,
                       weights_mid_arg=args.weights_mid, w_mid=args.w_mid,
                       pool=args.pool, pool_lo=args.pool_lo, pool_mid=args.pool_mid,
                       epoch=meta["epoch"],
                       res=meta["res"], created=stamp), fj, ensure_ascii=False, indent=2)
    print("  ⭐输出 -> %s\\pred_results.zip" % d)


if __name__ == "__main__":
    main()
