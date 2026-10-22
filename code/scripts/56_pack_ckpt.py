# -*- coding: utf-8 -*-
"""56 · 给**任意一个** ckpt 出一套标准包（26 单512 → 45 多尺度 448+512）。

52 号的 package() 只能跟在它自己的训练链后面跑。SWA / 补出历史端点这类
"权重已经在盘上、只差出包"的场景需要一个独立入口，这就是本脚本。

⛔ **出包只传 --ckpt 和 --eval-res 512，其余全默认。**
   改 --batch-size / --num-workers 会改变 TTA 随机视角在 worker 间的分配 →
   包就不是同口径了，没法和历史分数比（CLAUDE.md 踩过的坑）。

⚠ 必须用 Start-Process 独立启动（agent 的后台 job 会被回收，已证实）：
  $env:PYTHONIOENCODING='utf-8'
  Start-Process -FilePath 'D:\Anaconda\envs\dl\python.exe' `
    -ArgumentList 'scripts\56_pack_ckpt.py --ckpt <路径>' `
    -WorkingDirectory 'D:\aic_data\AIC\AIC2026_9_4' `
    -RedirectStandardOutput 'runs\pack_swa.log' -RedirectStandardError 'runs\pack_swa.err'

节奏：26 单512 ≈ 40 分钟 + 45 多尺度 ≈ 30 分钟 = **约 70 分钟**。
"""
import argparse
import glob
import hashlib
import os
import subprocess
import sys
import time

PY = r"D:\Anaconda\envs\dl\python.exe"
ROOT = r"D:\aic_data\AIC\AIC2026_9_4"
RUNS = os.path.join(ROOT, "runs")


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def run(cmd, what):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)           # 记忆坑①：沙箱 shell 会污染，必须清
    env.pop("PYTHONHOME", None)
    env["PYTHONIOENCODING"] = "utf-8"
    log("%s：%s" % (what, " ".join(cmd[1:])))
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT, env=env)
    log("%s 结束，退出码 %d，耗时 %.1f 分钟" % (what, r.returncode, (time.time() - t0) / 60))
    return r.returncode


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest().upper()


def linecount(p):
    with open(p, encoding="utf-8") as f:
        return sum(1 for _ in f)


def newest_new(pattern, before):
    new = sorted(set(glob.glob(os.path.join(RUNS, pattern))) - before)
    return new[-1] if new else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--tag", type=str, default=None, help="日志里的标签，默认取 ckpt 文件名")
    # ⭐2026-09-15 新增：高分支分辨率可变。动机见 runs\实验台账.md 实验 Q——
    #   34 号自己的 FixRes 闭式解代入 L 的真实参数（res_train=448, aug_scale=0.30）
    #   靶心 = 448 × 2/(1+√0.30) = 578.9 → 576，而我们一直评在 512（=448×1.143）。
    #   ⛔ 26/45 的目录名本来就是参数化的（_ft%d_ / _ms%d_%d_），换分辨率不会撞名。
    ap.add_argument("--eval-res", type=int, default=512,
                    help="26 的评估分辨率，同时作为 45 的高分支。必须是 32 的倍数")
    ap.add_argument("--res-lo", type=int, default=448,
                    help="45 的低分支。⚠45:108 硬断言它必须等于 ckpt 的训练分辨率，⛔别乱改")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ck = args.ckpt if os.path.isabs(args.ckpt) else os.path.join(ROOT, args.ckpt)
    if not os.path.exists(ck):
        log("⛔ 找不到 %s" % ck)
        sys.exit(1)
    if args.eval_res % 32:
        log("⛔ --eval-res 必须是 32 的倍数（ViT-B/32 的 patch 是 32×32），给了 %d" % args.eval_res)
        sys.exit(1)
    tag = args.tag or os.path.basename(ck).replace("ckpt_", "").replace(".pt", "")
    R, LO = args.eval_res, args.res_lo
    g_tta, g_ms = "*_ft%d_ep*_tta" % R, "*_ms%d_%d_tta" % (LO, R)

    s26 = [PY, os.path.join(ROOT, "scripts", "26_predict_finetuned.py"),
           "--ckpt", ck, "--eval-res", str(R)]
    s45 = [PY, os.path.join(ROOT, "scripts", "45_multiscale_tta.py"),
           "--ckpt", ck, "--res-lo", str(LO), "--res-hi", str(R)]
    if args.dry_run:
        log("DRY-RUN step1：" + " ".join(s26[1:]))
        log("DRY-RUN step2：" + " ".join(s45[1:]) + " --x512 <step1 产出的 npz>")
        log("DRY-RUN 目录名将是 %s 和 %s" % (g_tta, g_ms))
        return

    # ── step1：单 R（一次跑出 notta + tta 两个目录）
    before_tta = set(glob.glob(os.path.join(RUNS, g_tta)))
    if run(s26, "step1 单%d %s" % (R, tag)):
        sys.exit(1)
    tta_dir = newest_new(g_tta, before_tta)
    if not tta_dir:
        log("⛔ 未找到新增的 %s" % g_tta)
        sys.exit(1)

    # ── step2：多尺度 LO+R（高分支直接读 step1 的 npz，45 不再重算）
    before_ms = set(glob.glob(os.path.join(RUNS, g_ms)))
    if run(s45 + ["--x512", os.path.join(tta_dir, "test_feat_head.npz")],
           "step2 多尺度 %d+%d %s" % (LO, R, tag)):
        sys.exit(1)
    ms_dir = newest_new(g_ms, before_ms)
    if not ms_dir:
        log("⛔ 未找到新增的 %s" % g_ms)
        sys.exit(1)

    # ── 汇总
    log("")
    for name, d in [("%s 多尺度%d+%d TTA(主交)" % (tag, LO, R), ms_dir),
                    ("%s 单%d TTA" % (tag, R), tta_dir),
                    ("%s 单%d notta" % (tag, R), tta_dir.replace("_tta", "_notta"))]:
        z, c = os.path.join(d, "pred_results.zip"), os.path.join(d, "pred_results.csv")
        log("%s\n  %s\n  MD5=%s  行=%d"
            % (name, z, md5(z) if os.path.exists(z) else "MISSING",
               linecount(c) if os.path.exists(c) else -1))
    log("⭐ 出包完成")


if __name__ == "__main__":
    main()
