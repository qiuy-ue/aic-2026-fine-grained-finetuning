# -*- coding: utf-8 -*-
"""
JPEG 取证：把主办方的编码器参数【逐字节反推出来】，让退化增强从"像"变成"一模一样"。

═══════════════════════════════════════════════════════════════════════
一、第一性原理：JPEG 文件头里明明白白写着编码参数
═══════════════════════════════════════════════════════════════════════

JPEG 不是"糊一点"，它是一套**确定性**的有损流程：
    RGB → YCbCr → 色度下采样 → 8×8 分块 → DCT → **除以量化表** → 熵编码

⭐**量化表（64 个整数）会原样写进文件头**（DQT 段），因为解码器必须知道它。
⭐**色度下采样方式也写在 SOF 段**里。
→ 所以主办方用什么参数压的图，**不用猜，可以直接读出来**。这是取证，不是估计。

**为什么这值得较真**：v2 那 +9 分里，JPEG 退化增强是主力。
它的作用机理是让模型见到"和考卷同款的失真"。
如果我们的量化表和主办方的差一截，模型练的就是**另一种失真**——像，但不是。

⚠色度下采样尤其要紧：4:2:0 会把**颜色分辨率砍掉一半**（亮度保留全分辨率）。
细粒度物种识别高度依赖细小的颜色斑块（眼纹、翼斑）。
**练的时候颜色是全分辨率、考的时候被砍半 → 正好砸在最吃劲的特征上。**

═══════════════════════════════════════════════════════════════════════
二、⛔判据提前写死
═══════════════════════════════════════════════════════════════════════

  量化表**逐字节完全相同**  → ✅质量因子找准了，这一项不用改
  量化表差 <2%              → ⚠够接近，收益有限，先不动
  量化表差 >5% 或下采样不同  → ⭐**有免费的分要捡，改 degrade_like_test**

⚠**注意**：找到"最匹配的 quality"不等于"主办方就是用这个数"——
不同编码器（PIL/libjpeg/ImageMagick/OpenCV）同一个 quality 的表可能不同。
⭐**我们要的不是猜中它的参数，是让我们产出的表和它的表一致**，这才是目标。

用法（纯 CPU，训练在跑也能安全执行）：
  python scripts/36_jpeg_forensics.py
"""
import _bootstrap  # noqa: F401
from _bootstrap import banner

import argparse
import importlib.util
import io
import json
import os
import sys

import numpy as np
from PIL import Image, ImageFile, JpegImagePlugin

from configs import paths as P

ImageFile.LOAD_TRUNCATED_IMAGES = True

SUBS = {0: "4:4:4 (不下采样)", 1: "4:2:2 (横向砍半)", 2: "4:2:0 (横竖都砍半)", -1: "未知"}


def _ld(fname, modname):
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(modname, os.path.join(here, fname))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


_ex = _ld("01_extract_features.py", "extract01")


def probe(im):
    """读出一张 JPEG 的量化表和色度下采样方式。⭐这些是文件里写死的，不是估的。"""
    q = im.quantization                      # {0: 亮度表, 1: 色度表}, 每张 64 个整数
    try:
        s = JpegImagePlugin.get_sampling(im)
    except Exception:
        s = -1
    return {k: np.array(v, np.int64) for k, v in q.items()}, s


def qsig(q):
    """量化表的标量摘要：亮度表均值。数越大 = 压得越狠。"""
    return float(np.mean(q[0])) if 0 in q else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-test", type=int, default=400)
    ap.add_argument("--n-train", type=int, default=200)
    args = ap.parse_args()

    banner("36 JPEG 取证：把主办方的编码参数逐字节反推出来")

    test_paths = sorted(os.listdir(P.TEST_DIR))
    rs = np.random.RandomState(0)
    tsel = rs.choice(len(test_paths), min(args.n_test, len(test_paths)), replace=False)

    # ── 一、读出测试集的真实编码参数 ─────────────────────────────────
    banner("一／四：测试集的真实编码参数（直接从文件头读，不是估计）")
    qs, ss, tabs = [], [], []
    for i in tsel:
        try:
            with Image.open(os.path.join(P.TEST_DIR, test_paths[i])) as im:
                q, s = probe(im)
                if 0 not in q:
                    continue
                qs.append(qsig(q))
                ss.append(s)
                tabs.append(q[0])
        except Exception:
            pass
    qs = np.array(qs)
    tab_test = np.median(np.stack(tabs), 0).astype(np.int64)
    uniq, cnts = np.unique(np.stack(tabs), axis=0, return_counts=True)
    print("  抽样 %d 张" % len(qs))
    print("  亮度量化表均值：中位 %.2f   p10 %.2f   p90 %.2f   标准差 %.3f"
          % (np.median(qs), np.percentile(qs, 10), np.percentile(qs, 90), qs.std()))
    print("  ⭐不同的量化表共 %d 种（若=1，说明**全部用同一套参数一次性批处理**）" % len(uniq))
    if len(uniq) <= 3:
        for u, c in sorted(zip(uniq, cnts), key=lambda x: -x[1]):
            print("      占比 %5.1f%%  表均值 %.2f" % (100 * c / len(qs), u.mean()))
    sm = {}
    for s in ss:
        sm[s] = sm.get(s, 0) + 1
    print("  色度下采样：" + "  ".join("%s×%d" % (SUBS.get(k, k), v) for k, v in sm.items()))
    test_sub = max(sm, key=sm.get)

    # ── 二、暴力搜出最匹配的 PIL quality + subsampling ────────────────
    banner("二／四：暴力搜 PIL 参数，找出能复现这张表的 (quality, subsampling)")
    src = Image.new("RGB", (64, 64), (128, 100, 90))
    best = []
    for sub in (0, 1, 2):
        for qlt in range(40, 100):
            buf = io.BytesIO()
            src.save(buf, "JPEG", quality=qlt, subsampling=sub)
            buf.seek(0)
            with Image.open(buf) as im2:
                q2, s2 = probe(im2)
            if 0 not in q2:
                continue
            err = np.abs(q2[0].astype(float) - tab_test).mean()
            rel = err / max(tab_test.mean(), 1e-9)
            exact = bool((q2[0] == tab_test).all())
            best.append((rel, qlt, sub, exact, s2))
    best.sort()
    print("  %-8s %-24s %10s %8s" % ("quality", "subsampling", "表相对误差", "逐字节相同"))
    print("  " + "-" * 56)
    for rel, qlt, sub, exact, s2 in best[:5]:
        print("  %-8d %-24s %9.2f%% %8s" % (qlt, SUBS.get(sub, sub), 100 * rel,
                                            "✅是" if exact else "否"))
    brel, bq, bsub, bexact, _ = best[0]

    # ── 三、看现在的 degrade_like_test 产出什么 ───────────────────────
    banner("三／四：现在的 degrade_like_test（quality=75, PIL默认下采样）产出什么")
    paths, _, _ = _ex.list_train_files()
    rsel = rs.choice(len(paths), min(args.n_train, len(paths)), replace=False)
    cur_tabs, cur_subs = [], []
    for i in rsel:
        try:
            with Image.open(paths[i]) as im:
                im = im.convert("RGB")
                w, h = im.size
                s = 500.0 / max(w, h)
                if s < 1.0:
                    im = im.resize((max(1, round(w * s)), max(1, round(h * s))),
                                   Image.BICUBIC)
                buf = io.BytesIO()
                im.save(buf, "JPEG", quality=75)      # ← 现状
                buf.seek(0)
                with Image.open(buf) as im2:
                    q2, s2 = probe(im2)
                if 0 in q2:
                    cur_tabs.append(q2[0])
                    cur_subs.append(s2)
        except Exception:
            pass
    tab_cur = np.median(np.stack(cur_tabs), 0).astype(np.int64)
    cur_rel = np.abs(tab_cur.astype(float) - tab_test).mean() / max(tab_test.mean(), 1e-9)
    csm = {}
    for s in cur_subs:
        csm[s] = csm.get(s, 0) + 1
    cur_sub = max(csm, key=csm.get)
    print("  现状表均值 %.2f   vs 测试集 %.2f   →  相对误差 **%.2f%%**"
          % (tab_cur.mean(), tab_test.mean(), 100 * cur_rel))
    print("  现状下采样 %s   vs 测试集 %s   →  %s"
          % (SUBS.get(cur_sub, cur_sub), SUBS.get(test_sub, test_sub),
             "✅一致" if cur_sub == test_sub else "⛔**不一致**"))
    print("  逐字节完全相同：%s" % ("✅是" if (tab_cur == tab_test).all() else "否"))

    # ── 四、判读 ──────────────────────────────────────────────────────
    banner("四／四：判读（判据跑之前就写死了）")
    # ⚠量化表与 subsampling 无关，所以三种 subsampling 会并列第一。
    #   下采样方式必须用【直接从文件头读出的 test_sub】，不能用搜索排序的第一名。
    print("  测试集真实参数  ≈ PIL quality=%d（表相对误差 %.2f%%%s）, 下采样 %s〔直接读出〕"
          % (bq, 100 * brel, "，逐字节相同" if bexact else "", SUBS.get(test_sub, test_sub)))
    print("  当前退化增强    = PIL quality=75, %s（表相对误差 %.2f%%）"
          % (SUBS.get(cur_sub, cur_sub), 100 * cur_rel))
    print()
    sub_bad = (cur_sub != test_sub)
    tab_bad = cur_rel > 0.05
    if not sub_bad and cur_rel < 0.02:
        print("  ✅**已经对齐**（表误差 <2% 且下采样一致）→ 这一项没有免费的分，别动它。")
        verdict = "aligned"
    else:
        print("  ⭐**没对齐，有免费的分**：")
        if sub_bad:
            print("     ⛔色度下采样不同！%s vs %s —— 这直接决定**颜色分辨率是否被砍半**，"
                  % (SUBS.get(cur_sub, cur_sub), SUBS.get(test_sub, test_sub)))
            print("        细粒度物种识别高度依赖细小色斑（眼纹/翼斑），这一条影响不小。")
        if tab_bad:
            print("     ⚠量化表差 %.1f%%（判据是 5%%）→ 压缩强度不匹配。" % (100 * cur_rel))
        print()
        print("  ⭐改法（改 scripts/27_domain_gap_jpeg.py 的 degrade_like_test）：")
        print("       im.save(buf, \"JPEG\", quality=%d, subsampling=%d)" % (bq, bsub))
        print("  ⚠⚠但**别写死成一个值**：主办方是一次性批处理（量化表种类=%d 可佐证），"
              % len(uniq))
        print("     而增强需要**随机性**才不会让模型只学会一种失真。")
        print("     建议 quality 在 [%d, %d] 随机、subsampling 固定为 %d。"
              % (max(40, bq - 6), min(98, bq + 6), bsub))
        verdict = "misaligned"

    print()
    print("  ⛔**这仍是猜测层面的收益，必须交榜裁决**——本地验证集是高清原图，")
    print("     记忆里那条『病②：尺子对缩小域偏移的进步完全瞎』正好卡在这里。")

    out = os.path.join(P.CACHE_DIR, "jpeg_forensics.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(dict(test_table=tab_test.tolist(), test_sub=int(test_sub),
                       n_distinct_tables=int(len(uniq)),
                       best_quality=int(bq), best_sub=int(bsub),
                       best_rel_err=float(brel), exact=bool(bexact),
                       cur_table=tab_cur.tolist(), cur_sub=int(cur_sub),
                       cur_rel_err=float(cur_rel), verdict=verdict), fh,
                  ensure_ascii=False, indent=2)
    print("  已存 → %s" % out)


if __name__ == "__main__":
    main()
