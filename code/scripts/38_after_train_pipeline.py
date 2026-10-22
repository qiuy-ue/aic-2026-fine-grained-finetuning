# -*- coding: utf-8 -*-
"""
训练结束后的 GPU 接力流水线：自动等 GPU 空出来 → 依次出多档提交 → 跑评估几何。

═══════════════════════════════════════════════════════════════════════
为什么要有它
═══════════════════════════════════════════════════════════════════════

GPU 只有一块、8 GB，所有要用它的活只能**排队**。训练预计 15:10 左右结束，
后面还有 5 个活要跑（4 档提交 + 1 个评估几何），加起来约 86 分钟。
如果靠人盯着"训完了没"再手动敲下一条命令，中间的空窗全是白白流失的机时。

⭐**等待的对象不是"训练进程结束"这个事件，而是"显存够我用了"这个状态。**
直接测状态，比猜进程更本质：进程可能还没退但已经释放了显存，也可能退了但
句柄没回收。`nvidia-smi` 报的已用显存是唯一直接的判据。

⚠连续采样 3 次都低于阈值才算数——单次读数可能落在两个 batch 之间的低谷。

═══════════════════════════════════════════════════════════════════════
为什么档位这么选（ep02/03/04/06/08）
═══════════════════════════════════════════════════════════════════════

本次训练是**从榜上 72.76 分的 ckpt 继续训**，并且首次引入
**伪标签**（13783 张测试图用模型自己的预测当标签，权重 0.50）。

伪标签的收益曲线是**先升后降**的：
  - 升：测试图的**像素分布**是真的，模型能学到"测试集这批照片长什么样"，
        这是训练集里根本没有的信息 → 直接缩小域偏移
  - 降：标签是**模型自己写的**，训久了就是反复背诵自己的答案，
        错的地方被越背越牢（自我确认偏差），且不可能被纠正

⭐所以甜点大概率**靠前**，而 `ckpt_best.pt` 在 `--full-data` 下恰好等于
**最后一轮**（in-sample 指标必然单调升）——正好指反方向。
→ 多档全出，让排行榜选，⛔别信本地指标。

⚠GPU 时间便宜（11.6 分钟/档），提交次数贵。多生成不花提交名额，
   所以宁可多出几档备着。

用法（后台）：
  python scripts/38_after_train_pipeline.py --run runs/20260906_110505_v2_320
"""
import argparse
import json
import os
import subprocess
import sys
import time

PY = r"C:/Users/HX/Anaconda3/envs/pytorch/python.exe"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def gpu_used_mib():
    """已用显存(MiB)。读不到就返回一个大数，宁可多等也不要撞车。"""
    try:
        o = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30)
        return int(o.stdout.strip().splitlines()[0])
    except Exception as e:
        print("[pipe] ⚠nvidia-smi 读取失败(%s)，当作忙" % e, flush=True)
        return 99999


def wait_gpu(free_mib, need_hits, poll, timeout_min):
    """⭐等的是"显存够用"这个状态，不是"训练进程消失"这个事件。"""
    t0, hits = time.time(), 0
    while time.time() - t0 < timeout_min * 60:
        u = gpu_used_mib()
        hits = hits + 1 if u < free_mib else 0
        print("[pipe] 显存已用 %d MiB (阈值 %d)  连续空闲 %d/%d  已等 %.0f 分"
              % (u, free_mib, hits, need_hits, (time.time() - t0) / 60), flush=True)
        if hits >= need_hits:
            return True
        time.sleep(poll)
    return False


def run(tag, cmd, log):
    """一步失败不许影响后面的步骤——每步独立记账。"""
    print("\n[pipe] ▶ %s\n[pipe]   %s\n[pipe]   日志 → %s"
          % (tag, " ".join(cmd[1:]), log), flush=True)
    t0 = time.time()
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    try:
        with open(log, "w", encoding="utf-8", errors="replace") as f:
            rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=f,
                                stderr=subprocess.STDOUT).returncode
    except Exception as e:
        rc = -1
        print("[pipe] ⛔ %s 启动失败: %s" % (tag, e), flush=True)
    dt = (time.time() - t0) / 60
    print("[pipe] %s %s  退出码 %s  耗时 %.1f 分"
          % ("✅" if rc == 0 else "⛔", tag, rc, dt), flush=True)
    return rc, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--epochs", type=int, nargs="+", default=[2, 3, 4, 6, 8],
                    help="要出提交的轮次（⭐靠前的更可能是甜点，见文件头说明）")
    ap.add_argument("--free-mib", type=int, default=1500)
    ap.add_argument("--hits", type=int, default=3)
    ap.add_argument("--poll", type=float, default=60.0)
    ap.add_argument("--timeout-min", type=float, default=300.0)
    ap.add_argument("--geo-limit", type=int, default=6000)
    ap.add_argument("--skip-geo", action="store_true")
    args = ap.parse_args()

    rd = args.run if os.path.isabs(args.run) else os.path.join(ROOT, args.run)
    print("=" * 66, flush=True)
    print("38 训练后接力流水线   run=%s" % os.path.basename(rd), flush=True)
    print("=" * 66, flush=True)

    if not wait_gpu(args.free_mib, args.hits, args.poll, args.timeout_min):
        print("[pipe] ⛔ 等待超时，GPU 一直没空出来，收工。", flush=True)
        return 1
    print("[pipe] ⭐GPU 已空闲，开始接力。", flush=True)

    # 训练到底跑了几轮 —— 快照可能少于计划轮数（中途崩了/指标没创新高）
    hp = os.path.join(rd, "history.json")
    try:
        nh = len(json.load(open(hp, encoding="utf-8"))["history"])
    except Exception:
        nh = -1
    print("[pipe] history 共 %d 轮" % nh, flush=True)

    # ⚠竞态保护：GPU 释放（进程退出）晚于 history 落盘，理论上 37 号守护
    #   早就拷完了最后一轮。但"理论上"不等于"一定"——多等 2 分钟，
    #   代价是 2 分钟机时，收益是不会白丢掉最贵的那一档快照。
    if nh > 0:
        lastck = os.path.join(rd, "ckpt_ep%02d.pt" % nh)
        for _ in range(8):
            if os.path.exists(lastck):
                break
            print("[pipe] 等最后一轮快照 %s 落盘…" % os.path.basename(lastck), flush=True)
            time.sleep(15)

    jobs, missing = [], []
    for ep in args.epochs:
        ck = os.path.join(rd, "ckpt_ep%02d.pt" % ep)
        (jobs if os.path.exists(ck) else missing).append((ep, ck))
    if missing:
        print("[pipe] ⚠缺快照，跳过: %s"
              % ", ".join("ep%02d" % e for e, _ in missing), flush=True)
    print("[pipe] 计划出 %d 档提交: %s"
          % (len(jobs), ", ".join("ep%02d" % e for e, _ in jobs)), flush=True)

    report = []
    for ep, ck in jobs:
        rc, dt = run("提交 ep%02d" % ep,
                     [PY, "-u", "scripts/26_predict_finetuned.py", "--ckpt", ck],
                     os.path.join(rd, "_pred_ep%02d.log" % ep))
        report.append(("提交 ep%02d" % ep, rc, dt))

    if not args.skip_geo:
        rc, dt = run("评估几何",
                     [PY, "-u", "scripts/34_eval_geometry.py",
                      "--limit", str(args.geo_limit)],
                     os.path.join(ROOT, "runs", "_geometry.log"))
        report.append(("评估几何", rc, dt))

    print("\n" + "=" * 66, flush=True)
    print("流水线结束", flush=True)
    for t, rc, dt in report:
        print("  %s %-14s 退出码=%-4s %5.1f 分" % ("✅" if rc == 0 else "⛔", t, rc, dt),
              flush=True)
    print("=" * 66, flush=True)

    # 把生成的 zip 全列出来，方便直接交
    outs = []
    rr = os.path.join(ROOT, "runs")
    for d in sorted(os.listdir(rr)):
        z = os.path.join(rr, d, "pred_results.zip")
        if os.path.exists(z) and os.path.getmtime(z) > time.time() - 6 * 3600:
            outs.append((d, os.path.getmtime(z)))
    print("最近 6 小时生成的提交包：", flush=True)
    for d, mt in outs:
        print("  %s  %s" % (time.strftime("%H:%M", time.localtime(mt)), d), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
