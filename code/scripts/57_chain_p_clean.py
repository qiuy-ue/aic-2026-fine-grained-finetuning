# -*- coding: utf-8 -*-
"""实验 P · 合规重建：从 **CLIP 原始权重** 一口气长训，**⛔不带任何测试集伪标签**。

═══════════════════════════════════════════════════════════════════════════
⛔ 为什么必须重建（2026-09-15，读官方 PDF 原文后确认）
═══════════════════════════════════════════════════════════════════════════
赛题 PDF 十一、(二)数据使用约束 第 1 条**逐字**：

    「参赛者每个赛事阶段仅可使用赛事官方提供的当前阶段的数据集进行训练、验证和测试。
      需注意：**测试数据仅可用于测试，不得以自监督、无监督等方式参与到模型训练过程中。**」

而 30_pseudo_label.py 干的正是「用当前最好的模型给【测试集】打伪标签」，
chain_l12.log:50 自报「并入 16341 张测试图」（占测试集 65.5%）。

⚠ 边界要说清：「伪标签优化」「结合教师模型、自训练」官方在五、2 和 十、(二) 里**明确允许**，
  但那是在**训练数据**上。对**测试数据**另有上面这条单独的、更严的禁令。我们踩的是后者。

⭐⭐ 而且污染**烤进了初始权重**，去掉 --pseudo 重跑一发是不够的：

    L@ep12 (448,12轮) 🔴 ← 20260906_200353_v2_384 (384,2轮) 🔴
                         ← 20260906_110505_v2_320 (320,8轮) 🔴
                         ← 20260906_005936_v2_320  ⚠本地已无（老机器 F: 盘）

  整条 resume 链每一段都带伪标签 → **只能从 CLIP 原始权重重新起手。**

⭐ 干净的部件（已核实可复用，⛔别顺手也换掉）：
    cache/head_loss_sce.npz   06_loss_experiments.py:160-161 只读 F_TRAIN_FEAT / F_TRAIN_LABEL
    cache/noise_scores.npz    04_detect_noise.py:157-158 同上

═══════════════════════════════════════════════════════════════════════════
⭐ 顺带：这一发同时就是 6.1 节候选⑤（从没做过的那条）
═══════════════════════════════════════════════════════════════════════════
候选⑤原文：「现在是三个人各跑 100 米接力，从来没试过一个人跑 800 米。
            长跑的配速策略和冲刺完全不是一回事——末期学习率极小时，
            模型才有机会走进更精细的极小点，而接力式每次都在重新起步。」
→ 合规要求 和 唯一没试过的训练侧结构实验，是**同一个动作**。

═══════════════════════════════════════════════════════════════════════════
⚠ 轮数怎么定 / 为什么敢一次跑这么长
═══════════════════════════════════════════════════════════════════════════
- 去掉伪标签后训练集 119559 → **103218**（−13.7%）→ 约 **45 分钟/轮**（原 52）
- 从原版 CLIP 起手比"接着 384 的 ckpt 训"要走的路远得多，而 dp=0.2 又会让收敛更慢
  （「正则越强收敛越慢」是 J 那次确立的耦合），所以 ⛔**不能照搬 12 轮**
- **20 轮 ≈ 15 小时**。⭐ 风险≈0：28 号每轮都单独落盘 ckpt_epNN.pt，
  **选哪一轮交给榜分裁决，本地不做这个决定**（这正是 ckpt_best 那个坑的既定修法）

⚠ 必须用 Start-Process 独立启动（agent 的后台 job 会被回收，已证实）：
  $env:PYTHONIOENCODING='utf-8'
  Start-Process -FilePath 'D:\Anaconda\envs\dl\python.exe' `
    -ArgumentList 'scripts\57_chain_p_clean.py --epochs 20 --endpoints 20' `
    -WorkingDirectory 'D:\aic_data\AIC\AIC2026_9_4' `
    -RedirectStandardOutput 'runs\chain_p.log' -RedirectStandardError 'runs\chain_p.err'

⛔ 跑之前一定先 --dry-run 看命令行。
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

# ── P 的训练参数。⭐与 L（79.8454）的差别**只有三处**，其余 25 个参数逐字照抄 history.json：
#      ① ⛔删 --pseudo / --pseudo-weight   （合规：测试集不得参与训练）
#      ② ⛔删 --resume                      （污染在初始权重里 → 从原版 CLIP 起）
#      ③ --epochs 12 → 20                   （路更远 + 强正则收敛更慢；全部 ckpt 落盘兜底）
TRAIN_ARGS = [
    "--resolution", "448",
    "--epochs", "20",
    "--batch-size", "32",
    "--lr-backbone", "2e-05",
    "--llrd", "0.85",
    "--degrade-p", "0.875",
    "--cosine-head",
    "--aug-scale", "0.3",
    "--num-workers", "4",
    "--eval-workers", "2",
    "--full-data",
    "--seed", "0",
    "--drop-path", "0.2",
    "--random-erasing", "0.25",
    # ⛔ 故意没有 --resume ：28_finetune_v2.py:523 走 else 分支「从原版 CLIP + LP 头起手（LP-FT）」
    # ⛔ 故意没有 --pseudo ：这正是本次重建的全部理由
    # --head-init 用默认 cache/head_loss_sce.npz（已核实只用训练集特征，干净）
]
ENDPOINTS = ["ckpt_ep20.pt"]

BANNED = {"--pseudo", "--pseudo-weight", "--resume", "--soft-pseudo", "--pseudo-soft-temp"}


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def guard(cmd):
    """⭐合规闸：这一发的全部意义就是不带伪标签，⛔宁可不起飞也不能带着跑 15 小时。"""
    bad = [a for a in cmd if a in BANNED]
    if bad:
        log("⛔⛔ 合规闸拦截：命令行里出现了 %s —— 这一发的全部意义就是不带它们。" % bad)
        sys.exit(2)
    log("✅ 合规闸通过：无 %s" % "/".join(sorted(BANNED)))


def run(cmd, what):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)          # 记忆坑①：沙箱 shell 会污染，必须清
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


def package(ckpt, tag):
    """一个端点的两步出包：26 单512（notta+tta）→ 45 多尺度 448+512。
    ⛔ 只传 --ckpt 和 --eval-res 512，其余全默认（改 batch/workers 会换 TTA 口径）。"""
    before_tta = set(glob.glob(os.path.join(RUNS, "*_ft512_ep*_tta")))
    if run([PY, os.path.join(ROOT, "scripts", "26_predict_finetuned.py"),
            "--ckpt", ckpt, "--eval-res", "512"], "step1 单512 %s" % tag):
        return None
    tta_dir = newest_new("*_ft512_ep*_tta", before_tta)
    if not tta_dir:
        log("!! %s 未找到新增的 *_ft512_ep*_tta" % tag)
        return None

    before_ms = set(glob.glob(os.path.join(RUNS, "*_ms448_512_tta")))
    if run([PY, os.path.join(ROOT, "scripts", "45_multiscale_tta.py"),
            "--ckpt", ckpt, "--x512", os.path.join(tta_dir, "test_feat_head.npz")],
           "step2 多尺度 %s" % tag):
        return None
    ms_dir = newest_new("*_ms448_512_tta", before_ms)
    if not ms_dir:
        log("!! %s 未找到新增的 *_ms448_512_tta" % tag)
        return None

    return [("%s 多尺度448+512 TTA(主交)" % tag, ms_dir),
            ("%s 单512 TTA" % tag, tta_dir),
            ("%s 单512 notta" % tag, tta_dir.replace("_tta", "_notta"))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait-for", type=str, default=None,
                    help="先等这个文件出现，避免和上一个任务抢 GPU（单卡并行会双双掉速 80%%）")
    ap.add_argument("--wait-gpu-idle", action="store_true",
                    help="轮询到没有别的 python 进程占卡再起飞")
    ap.add_argument("--wait-timeout", type=int, default=14400)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-smoke", action="store_true",
                    help="⛔默认不跳——15 小时后才发现挂在第 3 分钟太贵")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--endpoints", type=str, default=None,
                    help="逗号分隔轮号，如 '16,20'。⛔端点法：别出中间点")
    args = ap.parse_args()

    global ENDPOINTS
    if args.endpoints:
        ENDPOINTS = ["ckpt_ep%02d.pt" % int(e) for e in args.endpoints.split(",")]
    if args.epochs:
        TRAIN_ARGS[TRAIN_ARGS.index("--epochs") + 1] = str(args.epochs)

    train_cmd = [PY, os.path.join(ROOT, "scripts", "28_finetune_v2.py")] + TRAIN_ARGS
    guard(train_cmd)

    ep = int(TRAIN_ARGS[TRAIN_ARGS.index("--epochs") + 1])
    if args.dry_run:
        log("DRY-RUN 训练命令：")
        print("    " + " ".join(train_cmd[1:]))
        log("DRY-RUN 出包端点：%s" % ENDPOINTS)
        log("DRY-RUN 预计：%d 轮 × 约 45 分钟 ≈ %.1f 小时，+ 每端点出包 70 分钟"
            % (ep, ep * 45 / 60.0))
        log("DRY-RUN ⭐核对：命令行里应当**看不到** --resume 和 --pseudo")
        return

    # ── 0. 等 GPU 空出来
    if args.wait_for:
        log("等待 %s ..." % args.wait_for)
        t0 = time.time()
        while not os.path.exists(args.wait_for):
            if time.time() - t0 > args.wait_timeout:
                log("!! 等待超时，放弃")
                sys.exit(1)
            time.sleep(30)
    if args.wait_gpu_idle:
        log("轮询 GPU 空闲（等别的 python 进程退出）...")
        t0 = time.time()
        while True:
            r = subprocess.run(["powershell", "-NoProfile", "-Command",
                                "(Get-Process python -ErrorAction SilentlyContinue).Count"],
                               capture_output=True, text=True)
            n = (r.stdout or "0").strip() or "0"
            try:
                n = int(n)
            except ValueError:
                n = 0
            if n <= 1:           # 只剩自己这条链
                log("GPU 空出（python 进程数 =%d），起飞" % n)
                break
            if time.time() - t0 > args.wait_timeout:
                log("!! 等 GPU 超时，放弃")
                sys.exit(1)
            time.sleep(60)
        time.sleep(15)

    # ── 0.5 起飞前冒烟。⭐这一发真正要验的是「不给 --resume 时 LP-FT 分支能不能跑起来」——
    #     这条分支全项目**从来没有被执行过**，是本次最大的未知。
    if not args.no_smoke:
        smoke = list(train_cmd)
        smoke[smoke.index("--epochs") + 1] = "2"
        smoke.append("--smoke")
        guard(smoke)
        if run(smoke, "起飞前冒烟（重点验 LP-FT 分支）"):
            log("!! 冒烟失败，⛔不起飞（省下 15 小时）")
            sys.exit(1)
        log("✅冒烟通过，起飞")

    # ── 1. 训练
    before = set(glob.glob(os.path.join(RUNS, "*_v2_448_cos_dp*")))
    if run(train_cmd, "训练 P（%d 轮，无伪标签，从原版 CLIP 起）" % ep):
        log("!! 训练失败，链中止")
        sys.exit(1)
    train_dir = newest_new("*_v2_448_cos_dp*", before)
    if not train_dir:
        log("!! 找不到 P 的训练目录")
        sys.exit(1)
    log("训练目录 = %s" % train_dir)

    # ── 2. 端点出包
    pkgs = []
    for name in ENDPOINTS:
        ck = os.path.join(train_dir, name)
        if not os.path.exists(ck):
            log("!! 缺 %s，跳过" % name)
            continue
        got = package(ck, name.replace("ckpt_", "").replace(".pt", ""))
        if got:
            pkgs += got

    # ── 3. 汇总
    lines = ["实验 P（合规重建：无伪标签 + 从原版 CLIP 起 + %d 轮）  %s"
             % (ep, time.strftime("%Y-%m-%d %H:%M:%S")),
             "训练目录: %s" % train_dir,
             "⭐ 本次训练命令行不含 --pseudo / --resume（合规闸已在起飞前校验）", ""]
    for tag, d in pkgs:
        z, c = os.path.join(d, "pred_results.zip"), os.path.join(d, "pred_results.csv")
        item = "%s\n  %s\n  MD5=%s  行=%d" % (
            tag, z, md5(z) if os.path.exists(z) else "MISSING",
            linecount(c) if os.path.exists(c) else -1)
        lines.append(item)
        log(item)
    with open(os.path.join(train_dir, "DONE.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log("全部完成")


if __name__ == "__main__":
    main()
