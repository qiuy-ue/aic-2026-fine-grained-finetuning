# -*- coding: utf-8 -*-
"""76 · 重复图按类型处理 → 训练排除名单（只读训练集，纯规则，不用任何模型打分）

为什么要做：
  复赛训练集里有一批逐字节相同的图（MD5 相同）。同一张图挂在两个不同的类下，
  至少有一个标签是错的；同一张图在同一类里存了好几份，等于这张图的权重被放大了。
  A1/A2 都没处理过。阶段 0 体检（docs/阶段0_复赛数据体检结论.md 第 5 节）已经说明：
  全删会把某些类删掉一半，必须按类型分开处理。

规则（阈值都是相对量，换一个阶段的数据不用改）：
  1. 类内重复：同一类里 MD5 相同的图只留一份（按路径排序留第一份）。
  2. 跨类重复：先算每一对类的重叠率 = 两类共享的图数 / 较小那个类的图数。
     - 这一组涉及的类对里，只要有一对重叠率 ≥ --sys-overlap（默认 5%），
       说明这两个类大面积共享图片，多半是同一物种或近缘种编成了两个号，
       删了只会白扔信号 ⇒ 整组保留（每个类里各留一份）。
     - 全部 < 5%：零星错标。分不出哪个标签对 ⇒ 所有副本都删；
       但小类（图数 < 中位数 × --small-frac）里的那份保留，小类丢不起图。
  ⛔不猜"哪个标签是对的"，也不改标签：那就又回到用模型判断当真相的老路（初赛 19 号前身 −1.3）。

⭐可复现：MD5 与规则都是确定的，同样的训练集永远得到同样的名单。
  名单只是本脚本的输出，训练链每次起飞前都会重新生成（70 号 --dedup），不依赖事先存好的文件。

输出：
  cache/dup_exclude.txt   每行一个 "类文件夹/文件名"，28 号 --exclude-list 读它
  cache/dup_exclude.json  统计
  cache/train_md5.tsv     MD5 缓存（"类文件夹/文件名<TAB>md5"），下次同一文件跳过重算

用法：
  python scripts\\76_dup_exclude.py
"""
import _bootstrap  # noqa: F401
from _bootstrap import banner

import argparse
import hashlib
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations

import numpy as np

from configs import paths as P

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def list_train():
    """与 01 号 list_train_files 同一顺序（类文件夹排序 → 类内文件名排序）。
    ⛔不 import 01 号：那边顶层会拉起 torch，本脚本只需要文件列表。"""
    cn = sorted(d for d in os.listdir(P.TRAIN_DIR) if os.path.isdir(os.path.join(P.TRAIN_DIR, d)))
    paths, labels = [], []
    for i, c in enumerate(cn):
        for fn in sorted(os.listdir(os.path.join(P.TRAIN_DIR, c))):
            if fn.lower().endswith(IMG_EXT):
                paths.append(os.path.join(P.TRAIN_DIR, c, fn))
                labels.append(i)
    return paths, np.array(labels, np.int64), cn


def relkey(p):
    return os.path.basename(os.path.dirname(p)) + "/" + os.path.basename(p)


def md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sys-overlap", type=float, default=0.05,
                    help="类对重叠率达到这个比例就算『成片共享』（同物种/近缘种），整组不删")
    ap.add_argument("--small-frac", type=float, default=0.5,
                    help="图数 < 中位数 × 这个比例的类算小类，零星冲突里小类那份不删（与 28 号同一口径）")
    ap.add_argument("--workers", type=int, default=8, help="算 MD5 的线程数（纯读盘，不占 GPU）")
    ap.add_argument("--out", default=os.path.join(P.CACHE_DIR, "dup_exclude.txt"))
    args = ap.parse_args()
    banner("76 重复图按类型处理 → 训练排除名单")

    paths, y, cn = list_train()
    N, K = len(paths), len(cn)
    keys = [relkey(p) for p in paths]
    cnt = np.bincount(y, minlength=K)
    med = float(np.median(cnt))
    small = cnt < args.small_frac * med
    print("  训练图 %d 张 / %d 类，每类中位 %.0f 张；小类（< %.0f 张）%d 个"
          % (N, K, med, args.small_frac * med, int(small.sum())))

    # ── MD5（带缓存：文件大小没变就沿用上次的结果）
    cache_p = os.path.join(P.CACHE_DIR, "train_md5.tsv")
    old = {}
    if os.path.exists(cache_p):
        with open(cache_p, encoding="utf-8") as f:
            for ln in f:
                k, sz, h = ln.rstrip("\n").split("\t")
                old[k] = (int(sz), h)
    sizes = [os.path.getsize(p) for p in paths]
    todo = [i for i in range(N) if old.get(keys[i], (-1, ""))[0] != sizes[i]]
    md5 = [old[keys[i]][1] if keys[i] in old else "" for i in range(N)]
    t0 = time.time()
    if todo:
        print("  算 MD5：%d 张（缓存命中 %d 张），%d 线程" % (len(todo), N - len(todo), args.workers))
        with ThreadPoolExecutor(args.workers) as ex:
            for j, (i, h) in enumerate(zip(todo, ex.map(lambda i: md5_file(paths[i]), todo)), 1):
                md5[i] = h
                if j % 20000 == 0:
                    print("    %d/%d  %.1f 分钟" % (j, len(todo), (time.time() - t0) / 60), flush=True)
        with open(cache_p, "w", encoding="utf-8", newline="\n") as f:
            f.write("".join("%s\t%d\t%s\n" % (keys[i], sizes[i], md5[i]) for i in range(N)))
        print("  MD5 用时 %.1f 分钟，已缓存 → %s" % ((time.time() - t0) / 60, cache_p))
    else:
        print("  MD5 全部命中缓存")

    # ── 分组
    groups = defaultdict(list)
    for i, h in enumerate(md5):
        groups[h].append(i)
    dup_groups = [g for g in groups.values() if len(g) > 1]
    cross = [g for g in dup_groups if len(set(y[g].tolist())) > 1]
    shared = defaultdict(int)                       # (a,b) → 共享的图数（按组计）
    for g in cross:
        for a, b in combinations(sorted(set(y[g].tolist())), 2):
            shared[(a, b)] += 1
    ovl = {pr: s / min(cnt[pr[0]], cnt[pr[1]]) for pr, s in shared.items()}

    excl = set()
    n_intra = n_sys_g = n_spo_g = n_spo_img = n_keep_small = 0
    for g in dup_groups:
        by_c = defaultdict(list)
        for i in sorted(g, key=lambda i: keys[i]):
            by_c[int(y[i])].append(i)
        cls = sorted(by_c)
        if len(cls) == 1:                           # 规则 1：类内重复
            for i in by_c[cls[0]][1:]:
                excl.add(i)
                n_intra += 1
            continue
        systematic = any(ovl[(a, b)] >= args.sys_overlap for a, b in combinations(cls, 2))
        if systematic:                              # 规则 2a：成片共享 → 每类留一份
            n_sys_g += 1
            for c in cls:
                for i in by_c[c][1:]:
                    excl.add(i)
                    n_intra += 1
            continue
        n_spo_g += 1                                # 规则 2b：零星错标 → 删，小类那份留
        for c in cls:
            for j, i in enumerate(by_c[c]):
                if small[c] and j == 0:
                    n_keep_small += 1
                    continue
                excl.add(i)
                n_spo_img += 1

    lost = np.bincount(y[sorted(excl)], minlength=K) if excl else np.zeros(K, np.int64)
    frac = lost / np.maximum(cnt, 1)
    worst = np.argsort(-frac)[:8]
    print("\n  重复组 %d 个（跨类 %d 个）" % (len(dup_groups), len(cross)))
    print("  规则 1  类内多余副本            删 %5d 张（含成片共享组里的类内副本）" % n_intra)
    print("  规则 2a 成片共享（重叠≥%.0f%%）    %5d 组，保留" % (100 * args.sys_overlap, n_sys_g))
    print("  规则 2b 零星冲突（重叠<%.0f%%）    %5d 组，删 %d 张，小类里留了 %d 张"
          % (100 * args.sys_overlap, n_spo_g, n_spo_img, n_keep_small))
    print("  合计排除 %d 张（%.2f%%），涉及 %d 个类" % (len(excl), 100 * len(excl) / N, int((lost > 0).sum())))
    print("  删得最多的类：" + "  ".join("%s %d/%d(%.0f%%)" % (cn[c], lost[c], cnt[c], 100 * frac[c])
                                      for c in worst if lost[c]))
    print("  小类被删：%d 张" % int(lost[small].sum()))
    if frac.max() > 0.2:
        print("  ⚠有类被删超过 20%%：%s —— 检查一下规则是不是误伤" % cn[int(frac.argmax())])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write("".join(keys[i] + "\n" for i in sorted(excl)))
    stat = dict(n_train=N, n_exclude=len(excl), n_dup_groups=len(dup_groups), n_cross_groups=len(cross),
                intra_removed=n_intra, systematic_groups=n_sys_g, sporadic_groups=n_spo_g,
                sporadic_removed=n_spo_img, kept_in_small=n_keep_small,
                sys_overlap=args.sys_overlap, small_frac=args.small_frac, median_class_size=med,
                n_small_classes=int(small.sum()), classes_touched=int((lost > 0).sum()),
                max_class_loss=float(frac.max()), created=time.strftime("%Y%m%d_%H%M%S"))
    with open(os.path.splitext(args.out)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(stat, f, ensure_ascii=False, indent=2)
    print("\n  ⭐名单 → %s（%d 行）" % (args.out, len(excl)))


if __name__ == "__main__":
    main()
