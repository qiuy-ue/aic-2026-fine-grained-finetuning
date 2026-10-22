# -*- coding: utf-8 -*-
"""接管 52_chain_j.py 的出包阶段，把端点顺序换成「ep12 优先」。

⭐为什么需要这个脚本
  chain_l12 是用 `--endpoints 9,12` 起的，ENDPOINTS 在**进程启动那一刻**就解析进内存了
  （交接文档 3.9 节记过这个坑：改源文件不影响已在跑的进程）。
  它会先给 ep09 出包（70 分钟）再给 ep12 出包。而 ep09 是 12 轮余弦 schedule 走到 3/4 的
  **中途快照**（LR 没退到底），是诊断点不是竞争者；ep12 才是那个决定性的数。
  → 把顺序倒过来，决定性的榜分早 70 分钟到手，同时省下 70 GPU 分钟给下一个实验。

⛔它会 kill 进程，所以默认 --dry-run。确认无误再加 --go。

三种用法
  1) 等训练自然结束后接管（推荐）：
       python scripts/54_takeover_pack.py --wait-train --endpoints ep12 --go
  2) 止损：立刻掐掉训练，拿已落盘的某一轮出包：
       python scripts/54_takeover_pack.py --kill-now --endpoints ep09 --go
  3) 只看它打算干什么：
       python scripts/54_takeover_pack.py --wait-train --endpoints ep12,ep09

⚠ 前置事实（2026-09-14 记）
  - 训练目录 = runs/20260914_125846_v2_448_cos_dp020_re25_ep12
  - chain 的日志 = runs/chain_l12.log，训练完成那一行长这样：
      [HH:MM:SS] 训练 J（6 轮） 结束，退出码 0，耗时 xxx.x 分钟
    （"6 轮"是 chain 里写死的文案，实际是 12 轮，别被它骗了）
  - 每个端点出包 = 26 号单512（约 39 分钟）+ 45 号多尺度 448+512（约 30 分钟）
"""
import _bootstrap  # noqa: F401

import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
import time

PY = r"D:\Anaconda\envs\dl\python.exe"
ROOT = r"D:\aic_data\AIC\AIC2026_9_4"
RUNS = os.path.join(ROOT, "runs")

TRAIN_DIR = os.path.join(RUNS, "20260914_125846_v2_448_cos_dp020_re25_ep12")
CHAIN_LOG = os.path.join(RUNS, "chain_l12.log")
DONE_MARK = "训练 J（6 轮） 结束"

# ⭐只 kill 命令行里含这些脚本名的 python 进程，⛔不碰别的 python
KILL_PATTERNS = ("52_chain_j.py", "26_predict_finetuned.py", "45_multiscale_tta.py")


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


# ────────────────────────────────────────────────────────────── 进程
def list_python_procs():
    """返回 [(pid, cmdline), ...]。wmic 在本机不存在，用 PowerShell + CIM。"""
    ps = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
          "Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress")
    r = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = (r.stdout or "").strip()
    if not out:
        return []
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        log("!! 解析 PowerShell 输出失败，原样贴出前 500 字：")
        print(out[:500])
        return []
    if isinstance(data, dict):
        data = [data]
    return [(int(d["ProcessId"]), d.get("CommandLine") or "") for d in data]


def kill_chain(dry):
    """掐掉 chain 及其出包子进程。⛔训练子进程若还在，会一并被 /T 带走——
    所以只在『训练已结束』或『明确要止损』时调用。"""
    hits = [(pid, cl) for pid, cl in list_python_procs()
            if any(p in cl for p in KILL_PATTERNS)]
    if not hits:
        log("没有找到 chain / 出包进程（可能已自己退出），继续")
        return
    for pid, cl in hits:
        short = cl if len(cl) < 160 else cl[:157] + "..."
        if dry:
            log("DRY-RUN 将 kill PID=%d  %s" % (pid, short))
        else:
            log("kill PID=%d  %s" % (pid, short))
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                           capture_output=True)
    if not dry:
        time.sleep(15)          # 等显存真正释放
        log("已等 15 秒让显存释放")


def train_state():
    """读 chain 日志，返回 (是否已结束, 退出码或 None)。"""
    if not os.path.exists(CHAIN_LOG):
        return False, None
    with open(CHAIN_LOG, encoding="utf-8", errors="replace") as f:
        for line in f:
            if DONE_MARK in line:
                code = None
                if "退出码" in line:
                    try:
                        code = int(line.split("退出码")[1].split("，")[0].strip())
                    except (ValueError, IndexError):
                        pass
                return True, code
    return False, None


# ────────────────────────────────────────────────────────────── 出包
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


def package(ep, dry):
    """一个端点的两步出包：26 单512（notta+tta）→ 45 多尺度 448+512。
    与 52_chain_j.py 的 package() 逐字同构，⛔别改口径。"""
    ckpt = os.path.join(TRAIN_DIR, "ckpt_%s.pt" % ep)
    if not os.path.exists(ckpt):
        log("!! 缺 %s，跳过" % ckpt)
        return []
    if dry:
        log("DRY-RUN %s step1: 26_predict_finetuned.py --ckpt %s --eval-res 512" % (ep, ckpt))
        log("DRY-RUN %s step2: 45_multiscale_tta.py --ckpt %s --x512 <上一步的 tta 目录>" % (ep, ckpt))
        return []

    before_tta = set(glob.glob(os.path.join(RUNS, "*_ft512_ep*_tta")))
    if run([PY, os.path.join(ROOT, "scripts", "26_predict_finetuned.py"),
            "--ckpt", ckpt, "--eval-res", "512"], "step1 单512 %s" % ep):
        return []
    tta_dir = newest_new("*_ft512_ep*_tta", before_tta)
    if not tta_dir:
        log("!! %s 未找到新增的 *_ft512_ep*_tta" % ep)
        return []

    before_ms = set(glob.glob(os.path.join(RUNS, "*_ms448_512_tta")))
    if run([PY, os.path.join(ROOT, "scripts", "45_multiscale_tta.py"),
            "--ckpt", ckpt, "--x512", os.path.join(tta_dir, "test_feat_head.npz")],
           "step2 多尺度 %s" % ep):
        return []
    ms_dir = newest_new("*_ms448_512_tta", before_ms)
    if not ms_dir:
        log("!! %s 未找到新增的 *_ms448_512_tta" % ep)
        return []

    return [("%s 多尺度448+512 TTA ⭐主交" % ep, ms_dir),
            ("%s 单512 TTA（内部尺子，⛔不占名额）" % ep, tta_dir),
            ("%s 单512 notta（内部尺子）" % ep, tta_dir.replace("_tta", "_notta"))]


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--wait-train", action="store_true",
                   help="等 chain 日志出现『训练结束』再接管（正常路径）")
    g.add_argument("--kill-now", action="store_true",
                   help="⛔立刻掐掉训练（止损用）。已落盘的 ckpt 不会丢")
    g.add_argument("--pack-now", action="store_true",
                   help="⭐并行出包：训练继续跑，只给已落盘的 ckpt 出包，"
                        "⛔不 kill 任何进程。代价=单卡争用，两边都变慢")
    ap.add_argument("--endpoints", type=str, default="ep12",
                   help="逗号分隔、**按出包先后顺序**，如 'ep12' 或 'ep12,ep09'")
    ap.add_argument("--go", action="store_true",
                   help="⛔真的执行。不加这个只打印计划（dry-run）")
    ap.add_argument("--poll", type=int, default=60, help="轮询间隔秒")
    ap.add_argument("--timeout", type=int, default=4 * 3600, help="等待训练的上限秒")
    args = ap.parse_args()
    dry = not args.go
    eps = [e.strip() for e in args.endpoints.split(",") if e.strip()]

    mode = ("等训练结束" if args.wait_train else
            "⛔立刻止损" if args.kill_now else "⭐并行出包（不动训练）")
    log("模式=%s  端点顺序=%s  %s"
        % (mode, eps, "DRY-RUN（不会动任何东西）" if dry else "⭐真跑"))
    log("训练目录 = %s" % TRAIN_DIR)

    if args.wait_train:
        t0 = time.time()
        while True:
            done, code = train_state()
            if done:
                if code not in (0, None):
                    log("!! 训练退出码 %s ≠ 0，⛔不出包，人工看 runs/chain_l12.err" % code)
                    sys.exit(1)
                log("✅训练已结束（退出码 %s）" % code)
                break
            if time.time() - t0 > args.timeout:
                log("!! 等待超时 %d 秒，放弃" % args.timeout)
                sys.exit(1)
            have = sorted(os.path.basename(p) for p in
                          glob.glob(os.path.join(TRAIN_DIR, "ckpt_ep*.pt")))
            log("训练还在跑，已落盘 %d 个 ckpt（最新 %s），%d 秒后再看"
                % (len(have), have[-1] if have else "无", args.poll))
            if dry:
                log("DRY-RUN 不真等，直接退出。真跑时请加 --go")
                break
            time.sleep(args.poll)

    # ⭐--pack-now 是唯一不掐进程的模式：训练继续跑，我们并行占一份显存出包。
    #   其余两种模式都必须先把 chain 掐掉，否则它会自己去给 ep09 出包抢 GPU。
    if args.pack_now:
        log("⭐并行模式：⛔不 kill 任何进程，训练继续。单卡争用，两边都会变慢")
    else:
        kill_chain(dry)

    pkgs = []
    for ep in eps:
        pkgs += package(ep, dry)

    if dry:
        log("DRY-RUN 结束。确认无误后加 --go 真跑。")
        return

    lines = ["接管出包完成  %s" % time.strftime("%Y-%m-%d %H:%M:%S"),
             "训练目录: %s" % TRAIN_DIR, "端点顺序: %s" % eps, ""]
    for tag, d in pkgs:
        z, c = os.path.join(d, "pred_results.zip"), os.path.join(d, "pred_results.csv")
        item = "%s\n  %s\n  MD5=%s  行=%d" % (
            tag, z, md5(z) if os.path.exists(z) else "MISSING",
            linecount(c) if os.path.exists(c) else -1)
        lines.append(item)
        log(item)
    with open(os.path.join(TRAIN_DIR, "TAKEOVER_DONE.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log("全部完成")


if __name__ == "__main__":
    main()
