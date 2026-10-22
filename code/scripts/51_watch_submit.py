"""独立 wrapper：监控训练进程 → 完成/死亡判定 → step1 单512出包 → step2 多尺度 → 写 DONE/FAILED。

⚠ 用 Start-Process 独立启动，全程不依赖 trae 后台 job（那条路已被证明会被回收）。
完成信号 = 训练目录下出现 DONE.txt（成功）或 FAILED.txt（训练死了）。
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


def proc_alive(pid):
    r = subprocess.run(["tasklist", "/fi", "PID eq %d" % pid, "/fo", "csv", "/nh"],
                       capture_output=True, text=True)
    return str(pid) in r.stdout


def find_ckpt(train_dir):
    # ⛔只认最后一轮的 ckpt（2 轮训练 → ckpt_ep02）。ckpt_best.pt 每轮结束都会写，
    #   曾在 2026-09-13 把 ep1 的 ckpt_best 误当完成信号，出了 ep1 废包。
    p = os.path.join(train_dir, "ckpt_ep02.pt")
    return p if os.path.exists(p) else None


def run(cmd):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env["PYTHONIOENCODING"] = "utf-8"
    log("执行: " + " ".join(cmd))
    r = subprocess.run(cmd, cwd=ROOT, env=env)
    if r.returncode != 0:
        log("!! 命令失败，退出码 %d" % r.returncode)
        sys.exit(r.returncode)


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def linecount(p):
    with open(p, encoding="utf-8") as f:
        return sum(1 for _ in f)


def write_flag(train_dir, name, lines):
    p = os.path.join(train_dir, name)
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log("已写 %s -> %s" % (name, p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", type=str, default=None)
    ap.add_argument("--watch-pid", type=int, default=0, help="训练进程 PID，死了且无 ckpt 判失败")
    ap.add_argument("--poll", type=int, default=30)
    args = ap.parse_args()

    train_dir = args.train_dir
    if not train_dir:
        cands = sorted(d for d in os.listdir(RUNS) if d.endswith("_cos_sc3"))
        assert cands, "runs/ 下没有 *_cos_sc3 训练目录"
        train_dir = os.path.join(RUNS, cands[-1])
    log("wrapper 接管 训练目录=%s  watch_pid=%d" % (train_dir, args.watch_pid))

    # ---------------------------------------------------------------- 1. 等训练（含死亡检测）
    ckpt = find_ckpt(train_dir)
    while ckpt is None:
        if args.watch_pid and not proc_alive(args.watch_pid):
            write_flag(train_dir, "FAILED.txt",
                       ["训练进程 PID=%d 已退出，但未生成 ckpt_ep02.pt" % args.watch_pid,
                        "时间: %s" % time.strftime("%Y-%m-%d %H:%M:%S")])
            log("!! 训练进程已死，FAILED.txt 已写，wrapper 退出")
            sys.exit(1)
        time.sleep(args.poll)
        ckpt = find_ckpt(train_dir)
    time.sleep(3)
    log("训练完成 ckpt=%s" % ckpt)

    # ---------------------------------------------------------------- 2. step1 单512
    before_tta = set(glob.glob(os.path.join(RUNS, "*_ft512_ep*_tta")))
    run([PY, os.path.join(ROOT, "scripts", "26_predict_finetuned.py"),
         "--ckpt", ckpt, "--eval-res", "512"])
    after_tta = set(glob.glob(os.path.join(RUNS, "*_ft512_ep*_tta")))
    new_tta = sorted(after_tta - before_tta)
    assert new_tta, "!! 未找到 step1 新增 *_ft512_ep*_tta"
    tta_dir = new_tta[-1]
    x512 = os.path.join(tta_dir, "test_feat_head.npz")
    notta_dir = tta_dir.replace("_tta", "_notta")

    # ---------------------------------------------------------------- 3. step2 多尺度
    before_ms = set(glob.glob(os.path.join(RUNS, "*_ms448_512_tta")))
    run([PY, os.path.join(ROOT, "scripts", "45_multiscale_tta.py"),
         "--ckpt", ckpt, "--x512", x512])
    after_ms = set(glob.glob(os.path.join(RUNS, "*_ms448_512_tta")))
    new_ms = sorted(after_ms - before_ms)
    assert new_ms, "!! 未找到 step2 新增 *_ms448_512_tta"
    ms_dir = new_ms[-1]

    # ---------------------------------------------------------------- 4. 汇总 DONE
    pkgs = [("多尺度448+512 TTA(主交)", ms_dir),
            ("单512 TTA(次交)", tta_dir),
            ("单512 notta(可选)", notta_dir)]
    lines = ["③b 子中心头 出包完成  时间=%s" % time.strftime("%Y-%m-%d %H:%M:%S"),
             "训练目录: %s" % train_dir, ""]
    for tag, d in pkgs:
        zipf = os.path.join(d, "pred_results.zip")
        csvf = os.path.join(d, "pred_results.csv")
        m = md5(zipf) if os.path.exists(zipf) else "MISSING"
        n = linecount(csvf) if os.path.exists(csvf) else -1
        item = "%s\n  %s\n  MD5=%s  行=%d" % (tag, zipf, m, n)
        lines.append(item)
        log(item)
    write_flag(train_dir, "DONE.txt", lines)
    log("全部完成")


if __name__ == "__main__":
    main()