# -*- coding: utf-8 -*-
"""实验 J 的执行链：等 GPU 空 → 训练 6 轮 → 端点出包（ep02 / ep06）→ 写 DONE。

⭐为什么不沿用 51 号的"轮询 ckpt 判完成"：那条路今天刚出过事——`ckpt_best.pt`
  每轮都写，被当成完成信号，出了 ep1 废包。这里改成 **subprocess.run 同步串联**：
  训练进程退出码就是完成信号，⛔没有"猜它跑完没有"这个环节，整类 bug 消失。

⚠ 必须用 Start-Process 独立启动（trae/agent 的后台 job 会被回收，已证实）：
  Start-Process -FilePath "D:\Anaconda\envs\dl\python.exe" `
    -ArgumentList "scripts\52_chain_j.py" `
    -WorkingDirectory "D:\aic_data\AIC\AIC2026_9_4" `
    -RedirectStandardOutput "runs\chain_j.log" -RedirectStandardError "runs\chain_j.err"

端点法（交接文档 4.4）：只出 ep02 和 ep06 两个端点，⛔不出 ep04。
  ep02 → 与 ③a@ep02 比，读**正则化包**的净效应（同 2 轮，唯一变量=正则）
  ep06 → 与 J@ep02 比，读**正则加强后甜点是否后移**
  中间点只提供精度，形状没定之前精度没有意义；ep04 等形状要了再补。
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

# ── J 的训练参数。⭐与 ③a 的差别**只有最后两行 + epochs**，其余逐字照抄 history.json
TRAIN_ARGS = [
    "--resolution", "448",
    "--epochs", "6",                 # ③a 是 2；余弦退火按 6 轮重新规划
    "--batch-size", "32",
    "--lr-backbone", "2e-05",
    "--llrd", "0.85",
    "--degrade-p", "0.875",
    "--cosine-head",                 # ③a 单中心余弦头（⛔不带 --subcenters）
    "--aug-scale", "0.3",
    "--num-workers", "4",
    "--eval-workers", "2",
    "--resume", "runs/20260906_200353_v2_384/ckpt_best.pt",
    "--pseudo", "cache/pseudo_labels.npz",
    "--pseudo-weight", "0.5",
    "--full-data",
    "--seed", "0",
    "--drop-path", "0.2",            # ⭐实验 J 的变量①：模型级正则，全项目首次
    "--random-erasing", "0.25",      # ⭐实验 J 的变量②：增强加码
]
ENDPOINTS = ["ckpt_ep02.pt", "ckpt_ep06.pt"]

# ── 2026-09-14：J@ep06 = 79.1284（vs 真实水平线 78.2767 = +0.85，≈5σ，新历史最高）
#   → 方向证实。按 6.5 节"同方向再推一档、端点法别小步挪"，轮数 6 → 12。
#   ⭐推轮数而不是推正则剂量的依据：J 跑满 6 轮 in-sample 才 0.9476 且**仍在单调上升**，
#     比老配方 2 轮的 0.9739 还低一大截 —— 模型**仍然欠拟合**，此时加大正则只会更欠。


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


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
    after = set(glob.glob(os.path.join(RUNS, pattern)))
    new = sorted(after - before)
    return new[-1] if new else None


def package(ckpt, tag):
    """一个端点的两步出包：26 单512（notta+tta）→ 45 多尺度 448+512。"""
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
            ("%s 单512 TTA(次交)" % tag, tta_dir),
            ("%s 单512 notta" % tag, tta_dir.replace("_tta", "_notta"))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait-for", type=str, default=None,
                    help="先等这个文件出现（上一轮的 DONE.txt），避免两个任务抢 GPU")
    ap.add_argument("--wait-timeout", type=int, default=7200)
    ap.add_argument("--dry-run", action="store_true", help="只打印将要执行的命令，不动 GPU")
    ap.add_argument("--no-smoke", action="store_true",
                    help="跳过起飞前冒烟。⛔默认不跳——6 小时后才发现挂在第 3 分钟太贵")
    ap.add_argument("--epochs", type=int, default=None, help="覆盖 TRAIN_ARGS 里的轮数")
    ap.add_argument("--endpoints", type=str, default=None,
                    help="逗号分隔的端点轮号，如 '9,12'。⛔端点法：只出两个端点，不出中间点")
    ap.add_argument("--drop-path", type=float, default=None,
                    help="覆盖 TRAIN_ARGS 里的 drop-path 剂量（实验 M：0.2 → 0.35）")
    ap.add_argument("--random-erasing", type=float, default=None,
                    help="覆盖 TRAIN_ARGS 里的 RandomErasing 概率")
    # ⭐实验 R（2026-09-15）：正则族三条腿里只剩【步长】没动过。
    #   轮数 → L（6→12，+0.717 ✅）；剂量 → M（dp 0.2→0.35，Δ=−0.004 判平）；步长 → 本参数。
    #   M 的数据直接指着这条：dp 加码把 in-sample 平台压低 5.4 个点（.9702→.9161）而榜分不动
    #   = **正则腾出了拟合余量，优化器没去填**。
    #   ⚠ 而 2e-5 这个值【无任何选型记录】：唯一一次 lr 实测（runs\_lrprobe_*）选的是 1.5e-5，
    #     且是在 224px / 1轮 / 无LLRD / 无Mixup / 无EMA / 无drop-path 上做的
    #     —— 和「甜点=2轮」同一天、同一把尺子、同一个**已作废**的配方。同构的条件结论。
    ap.add_argument("--lr-backbone", type=str, default=None,
                    help="覆盖骨干顶层 lr（实验 R：2e-05 → 4e-05）。⛔端点法，别小步挪")
    ap.add_argument("--lr-head", type=str, default=None, help="覆盖分类头 lr（默认 1e-3，一般别动）")
    ap.add_argument("--wait-gpu-idle", action="store_true",
                    help="轮询到没有别的 python 进程占卡再起飞（⛔单卡并行出包双双掉速 80%%）")
    args = ap.parse_args()

    global ENDPOINTS
    if args.endpoints:
        ENDPOINTS = ["ckpt_ep%02d.pt" % int(e) for e in args.endpoints.split(",")]
    if args.epochs:
        TRAIN_ARGS[TRAIN_ARGS.index("--epochs") + 1] = str(args.epochs)
    # ⭐实验 M（2026-09-15）：L 证明 dp=0.2 下 12 轮已跑到 in-sample 平台（ep11→ep12 仅 +0.0002）。
    #   平台到了就只能把平台本身压低 → 加正则剂量。端点法：0.2 一步到 0.35，⛔不小步挪。
    if args.drop_path is not None:
        TRAIN_ARGS[TRAIN_ARGS.index("--drop-path") + 1] = str(args.drop_path)
    if args.random_erasing is not None:
        TRAIN_ARGS[TRAIN_ARGS.index("--random-erasing") + 1] = str(args.random_erasing)
    if args.lr_backbone is not None:
        TRAIN_ARGS[TRAIN_ARGS.index("--lr-backbone") + 1] = args.lr_backbone
    if args.lr_head is not None:
        if "--lr-head" in TRAIN_ARGS:
            TRAIN_ARGS[TRAIN_ARGS.index("--lr-head") + 1] = args.lr_head
        else:
            TRAIN_ARGS.extend(["--lr-head", args.lr_head])

    train_cmd = [PY, os.path.join(ROOT, "scripts", "28_finetune_v2.py")] + TRAIN_ARGS
    if args.dry_run:
        log("DRY-RUN 训练命令：")
        print("    " + " ".join(train_cmd[1:]))
        log("DRY-RUN 出包端点：%s" % ENDPOINTS)
        log("DRY-RUN ⭐逐项核对：lr-backbone=%s  drop-path=%s  erasing=%s  epochs=%s"
            % (TRAIN_ARGS[TRAIN_ARGS.index("--lr-backbone") + 1],
               TRAIN_ARGS[TRAIN_ARGS.index("--drop-path") + 1],
               TRAIN_ARGS[TRAIN_ARGS.index("--random-erasing") + 1],
               TRAIN_ARGS[TRAIN_ARGS.index("--epochs") + 1]))
        return

    # ── 0'. 等 GPU 空出来（⛔单卡并行出包会让两边掉速约 80%）
    if args.wait_gpu_idle:
        log("轮询 GPU 空闲（等别的 python 进程退出）...")
        t0 = time.time()
        while True:
            r = subprocess.run(["powershell", "-NoProfile", "-Command",
                                "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" "
                                "| Where-Object { $_.CommandLine -notlike '*52_chain_j*' }).Count"],
                               capture_output=True, text=True)
            try:
                n = int((r.stdout or "0").strip() or "0")
            except ValueError:
                n = 0
            if n == 0:
                log("GPU 空出（占卡的 python 进程 =0），起飞")
                break
            if time.time() - t0 > args.wait_timeout:
                log("!! 等 GPU 超时 %d 秒，放弃" % args.wait_timeout)
                sys.exit(1)
            time.sleep(60)
        time.sleep(15)

    # ── 0. 等上一轮出包结束（单卡并行=互相拖慢，老机器实测推理反降 72%）
    if args.wait_for:
        log("等待 %s ..." % args.wait_for)
        t0 = time.time()
        while not os.path.exists(args.wait_for):
            if time.time() - t0 > args.wait_timeout:
                log("!! 等待超时 %d 秒，放弃" % args.wait_timeout)
                sys.exit(1)
            time.sleep(30)
        log("上一轮已结束，GPU 空出，开始 J")
        time.sleep(10)

    # ── 0.5 起飞前冒烟：1500 训 / 800 验 × 2 轮，几分钟。
    #     真正要验的是「drop-path 进了 AMP 训练循环 / RandomErasing 进了 dataloader /
    #     EMA 的 deepcopy 在被 patch 过的 block 上不炸 / ckpt 能落盘」这四件事。
    #     自测脚本只测了前向，⛔测不到训练循环里 autocast + GradScaler 的交互。
    if not args.no_smoke:
        smoke = [x for x in train_cmd]
        smoke[smoke.index("--epochs") + 1] = "2"
        # ⚠--smoke 只截训练集（1500 张），**不截伪标签**——带着那 16341 张跑，
        #   冒烟要 15 分钟而不是 3 分钟。伪标签这条路 J 没动过，不需要冒烟覆盖。
        i = smoke.index("--pseudo")
        del smoke[i:i + 2]
        i = smoke.index("--pseudo-weight")
        del smoke[i:i + 2]
        smoke.append("--smoke")
        if run(smoke, "起飞前冒烟"):
            log("!! 冒烟失败，⛔不起飞（省下 6 小时）")
            sys.exit(1)
        log("✅冒烟通过，起飞")

    # ── 1. 训练
    before_train = set(glob.glob(os.path.join(RUNS, "*_v2_448_cos_dp*")))
    if run(train_cmd, "训练 J（6 轮）"):
        log("!! 训练失败，链中止")
        sys.exit(1)
    train_dir = newest_new("*_v2_448_cos_dp*", before_train)
    if not train_dir:
        log("!! 找不到 J 的训练目录")
        sys.exit(1)
    log("训练目录 = %s" % train_dir)

    # ── 2. 两个端点各出一套包
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
    lines = ["实验 J（drop-path 0.2 + RandomErasing 0.25 + 6 轮）出包完成  %s"
             % time.strftime("%Y-%m-%d %H:%M:%S"),
             "训练目录: %s" % train_dir, ""]
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
