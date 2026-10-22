# -*- coding: utf-8 -*-
"""73 · 训练状态速览（5 分钟巡检用）。⭐只用标准库、只读日志，WSL 与 Windows 都能跑，不碰 GPU。

读 runs/ 下的执行链日志，给出：当前是哪一发、第几轮第几步、速度、损失、co-teaching 读数、
各轮末读数、链事件、报错、训练目录里的权重/续跑状态，以及按实测速度推算的剩余时间。
"""
import glob
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(os.path.dirname(HERE), "runs")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def tail(path, n=60000):
    with open(path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - n))
        return f.read().decode("utf-8", "replace")


def main():
    # 当前最活跃的链：最近写入的 chain_*.err
    errs = sorted(glob.glob(os.path.join(RUNS, "chain_*.err")), key=os.path.getmtime)
    if not errs:
        print("没有找到任何 chain_*.err")
        return
    err = errs[-1]
    log = err[:-4] + ".log"
    # 训练进度条写 .err，建池/出包的进度写 .log —— 两个取较新的，免得出包阶段误报"没动"
    age = time.time() - max(os.path.getmtime(p) for p in (err, log) if os.path.exists(p))
    tag = os.path.basename(err)[6:-4]
    print("链 %s  日志 %.0f 秒前更新%s" % (tag, age, "  ⚠超过 10 分钟没动" if age > 600 else ""))

    t = tail(err).replace("\r", "\n")
    bars = re.findall(r"(ep\d+/\d+):\s+(\d+)%\|[^|]*\|\s*(\d+)/(\d+) \[([^<\]]+)<([^,\]]+),\s*([\d.]+)(it/s|s/it)(?:, ([^\]]*))?\]", t)
    if bars:
        ep, pct, cur, tot, el, rem, spd, unit, post = bars[-1]
        its = float(spd) if unit == "it/s" else 1.0 / float(spd)
        print("进度 %s  %s/%s (%s%%)  %.2f 步/秒  本轮还剩 %s  | %s" % (ep, cur, tot, pct, its, rem, post))
    else:
        other = [ln for ln in t.split("\n") if ln.strip()][-1:] or [""]
        print("进度（非训练阶段）: %s" % other[0][:160])
    if os.path.exists(log) and os.path.getmtime(log) > os.path.getmtime(err):
        # 58/59/30 的进度打在 stdout：取 .log 最后一行非空
        last = [ln for ln in tail(log, 4000).replace("\r", "\n").split("\n") if ln.strip()][-1:] or [""]
        print("出包进度: %s" % last[0].strip()[:160])
    tb = [ln for ln in t.split("\n") if re.search(r"Traceback|Error|out of memory", ln)]
    if tb:
        print("⛔报错: %s" % tb[-1][:200])

    if os.path.exists(log):
        L = open(log, encoding="utf-8", errors="replace").read()
        eplines = re.findall(r"^\s+(\[ep\d+(?:·CT)?\].*)$", L, re.M)
        real = [x for x in eplines]                 # 冒烟的轮次也在里面，取正式训练段之后的
        cut = L.rfind("▶ 续跑训练") if "▶ 续跑训练" in L else L.rfind("▶ 训练 ")
        if cut >= 0:
            real = re.findall(r"^\s+(\[ep\d+(?:·CT)?\].*)$", L[cut:], re.M)
        for x in real[-2:]:
            print("轮末 " + x.strip()[:170])
        ev = [ln for ln in L.split("\n") if ln.startswith("[09-") and "跳过" not in ln]
        if ev:
            print("链事件 " + ev[-1][:170])
        m = re.search(r"产物  -> (\S+)", L[cut:] if cut >= 0 else L) or re.search(r"写回 (\S+)", L)
        tdir = None
        if m:
            tdir = os.path.join(RUNS, os.path.basename(m.group(1).replace("\\", "/").rstrip("/")))
        if tdir and os.path.isdir(tdir):
            ck = sorted(f for f in os.listdir(tdir) if re.fullmatch(r"ckpt_ep\d\d\.pt", f))
            st = os.path.join(tdir, "train_state.pt")
            print("权重 %s  续跑状态 %s" % (
                ",".join(c[7:9] for c in ck) or "无",
                ("有(%.1fGB, %.0f 分钟前)" % (os.path.getsize(st) / 1e9, (time.time() - os.path.getmtime(st)) / 60))
                if os.path.exists(st) else "无"))
    dones = sorted(glob.glob(os.path.join(RUNS, "_chain_*_DONE.txt")), key=os.path.getmtime)
    if dones and time.time() - os.path.getmtime(dones[-1]) < 86400 * 2:
        print("完成标记 " + ", ".join(os.path.basename(d) for d in dones[-2:]))


if __name__ == "__main__":
    main()
