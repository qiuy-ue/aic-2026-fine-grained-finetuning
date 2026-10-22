# -*- coding: utf-8 -*-
"""
_tmp_pairsig.py —— 纯 CPU：同物种类对之间，**测试集侧还能不能观察到的**源印记有多大可判别性。

⭐ 逻辑前提（必须先说清楚，否则整条路是空的）：
   主办方把测试集统一重编码成 [长边<=500 + JPEG q75 + 4:2:0 + 无任何元数据]。
   因此训练集里那些源印记（格式 WEBP/PNG、1198 种量化表、EXIF、文件名）
   **在测试集上全部被抹掉**，⛔不可用。
   只有两样东西穿过了这道工序：
     (a) 宽高比 w/h        —— 精确保留
     (b) 长边是否 < 500     —— 原图长边 <500 的不会被放大，所以"小图"这个事实保留
   本脚本只量这两样，量的是**上界**：用 AUC 判两个类的这两个量能不能分开。

用法:
  python _tmp_pairsig.py 0266 0424 0156 0467 0155 0195      # 成对给
  python _tmp_pairsig.py --scan pairs.json                  # 批量
"""
import os, sys, json, collections
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

DATA_ROOT = r"D:\aic_data\dataset"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")
CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")


def cls_feats(c):
    """返回 [(ar, longside_after_org_pipeline, fmt, w, h, name)]"""
    d = os.path.join(TRAIN_DIR, c)
    out = []
    for n in sorted(os.listdir(d)):
        p = os.path.join(d, n)
        try:
            with Image.open(p) as im:
                w, h = im.size
                fmt = im.format
        except Exception:
            continue
        ls = max(w, h)
        out.append((w / max(1, h), min(ls, 500), fmt, w, h, n))
    return out


def auc(a, b):
    """a,b 是两组标量，返回 Mann-Whitney AUC（0.5=不可分）"""
    xs = sorted([(v, 0) for v in a] + [(v, 1) for v in b])
    r = 0
    n0 = 0
    i = 0
    # 处理并列：给同值组平均秩
    while i < len(xs):
        j = i
        while j < len(xs) and xs[j][0] == xs[i][0]:
            j += 1
        grp = xs[i:j]
        n0g = sum(1 for _, t in grp if t == 0)
        n1g = len(grp) - n0g
        r += n0 * n1g + n0g * n1g * 0.5
        n0 += n0g
        i = j
    return r / max(1, len(a) * len(b))


def show(c, f):
    fmts = collections.Counter(x[2] for x in f)
    small = sum(1 for x in f if x[1] < 500)
    ars = sorted(x[0] for x in f)
    print("  类 %s: n=%d  格式=%s  长边<500 占比=%.1f%%  宽高比 中位=%.3f  [p10=%.3f p90=%.3f]"
          % (c, len(f), dict(fmts), 100.0 * small / max(1, len(f)),
             ars[len(ars) // 2], ars[int(.1 * len(ars))], ars[int(.9 * len(ars))]))


def main():
    args = [a for a in sys.argv[1:]]
    # 先把测试集的宽高比分布量一遍（这是"能不能用"的锚）
    tfs = sorted(os.listdir(TEST_DIR))
    import random
    random.seed(0)
    sub = random.sample(tfs, min(4000, len(tfs)))
    tar, tsmall = [], 0
    for n in sub:
        try:
            with Image.open(os.path.join(TEST_DIR, n)) as im:
                w, h = im.size
        except Exception:
            continue
        tar.append(w / max(1, h))
        if max(w, h) < 500:
            tsmall += 1
    ts = sorted(tar)
    print("=== 测试集(抽 %d 张) 宽高比 中位=%.3f [p10=%.3f p90=%.3f]  长边<500 占比=%.2f%% ==="
          % (len(tar), ts[len(ts) // 2], ts[int(.1 * len(ts))], ts[int(.9 * len(ts))],
             100.0 * tsmall / max(1, len(tar))))
    print()
    for i in range(0, len(args) - 1, 2):
        a, b = args[i], args[i + 1]
        fa, fb = cls_feats(a), cls_feats(b)
        print("### 对 %s <-> %s" % (a, b))
        show(a, fa)
        show(b, fb)
        A = auc([x[0] for x in fa], [x[0] for x in fb])
        print("  ⭐ 宽高比 AUC = %.4f  (0.5=完全不可分; |AUC-0.5|*2 = 可判别度)" % A)
        sa = [1.0 if x[1] < 500 else 0.0 for x in fa]
        sb = [1.0 if x[1] < 500 else 0.0 for x in fb]
        print("  ⭐ '长边<500' AUC = %.4f   (%.1f%% vs %.1f%%)"
              % (auc(sa, sb), 100 * sum(sa) / max(1, len(sa)), 100 * sum(sb) / max(1, len(sb))))
        # 逐字节重复图在两类里的分布
        import hashlib
        ha = {}
        for x in fa:
            with open(os.path.join(TRAIN_DIR, a, x[5]), "rb") as f:
                ha.setdefault(hashlib.md5(f.read()).hexdigest(), []).append(x[5])
        nd = 0
        for x in fb:
            with open(os.path.join(TRAIN_DIR, b, x[5]), "rb") as f:
                if hashlib.md5(f.read()).hexdigest() in ha:
                    nd += 1
        print("  ⭐ 两类间【逐字节完全相同】的图 = %d 张 (占 %s 的 %.1f%%, 占 %s 的 %.1f%%)"
              % (nd, a, 100.0 * nd / max(1, len(fa)), b, 100.0 * nd / max(1, len(fb))))
        print()


if __name__ == "__main__":
    main()
