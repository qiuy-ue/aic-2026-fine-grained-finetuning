"""③b 子中心头 自动化编排：等训练完成 → step1 单512出包 → step2 多尺度 → 汇总 MD5/行数。

用法：
    python scripts/50_orchestrate_submit.py [--train-dir runs/xxx_v2_448_cos_sc3]

默认不加参数时，自动找 runs/ 下最新的 *_cos_sc3 训练目录。
全程跑完会在训练目录写出 DONE.txt，并打印三份 zip 的路径 + MD5 + 行数。
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


def run(cmd):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env["PYTHONIOENCODING"] = "utf-8"
    print(">>> " + " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=ROOT, env=env)
    if r.returncode != 0:
        print("!! 命令失败，退出码 %d" % r.returncode, flush=True)
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


def find_ckpt(train_dir):
    ep02 = os.path.join(train_dir, "ckpt_ep02.pt")
    if os.path.exists(ep02):
        return ep02
    # 万一训练是 1 轮或 ckpt_best，回退
    for name in ("ckpt_ep02.pt", "ckpt_best.pt", "ckpt_ep01.pt"):
        p = os.path.join(train_dir, name)
        if os.path.exists(p):
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", type=str, default=None)
    ap.add_argument("--poll", type=int, default=30, help="轮询间隔秒")
    args = ap.parse_args()

    train_dir = args.train_dir
    if not train_dir:
        cands = sorted(d for d in os.listdir(RUNS) if d.endswith("_cos_sc3"))
        assert cands, "runs/ 下没有 *_cos_sc3 训练目录（③b 还没起？）"
        train_dir = os.path.join(RUNS, cands[-1])
    print("监控训练目录: %s" % train_dir, flush=True)

    # ---------------------------------------------------------------- 1. 等训练
    ckpt = find_ckpt(train_dir)
    while ckpt is None:
        time.sleep(args.poll)
        ckpt = find_ckpt(train_dir)
    time.sleep(3)   # 让 ckpt 落盘稳定
    print("训练完成，ckpt = %s" % ckpt, flush=True)

    # ---------------------------------------------------------------- 2. step1 单512
    before_tta = set(glob.glob(os.path.join(RUNS, "*_ft512_ep2_tta")))
    run([PY, os.path.join(ROOT, "scripts", "26_predict_finetuned.py"),
         "--ckpt", ckpt, "--eval-res", "512"])
    after_tta = set(glob.glob(os.path.join(RUNS, "*_ft512_ep2_tta")))
    new_tta = sorted(after_tta - before_tta)
    assert new_tta, "!! 未找到 step1 新增的 *_ft512_ep2_tta 目录"
    tta_dir = new_tta[-1]
    x512 = os.path.join(tta_dir, "test_feat_head.npz")
    notta_dir = tta_dir.replace("_tta", "_notta")
    print("step1 tta 特征: %s" % x512, flush=True)

    # ---------------------------------------------------------------- 3. step2 多尺度
    before_ms = set(glob.glob(os.path.join(RUNS, "*_ms448_512_tta")))
    run([PY, os.path.join(ROOT, "scripts", "45_multiscale_tta.py"),
         "--ckpt", ckpt, "--x512", x512])
    after_ms = set(glob.glob(os.path.join(RUNS, "*_ms448_512_tta")))
    new_ms = sorted(after_ms - before_ms)
    assert new_ms, "!! 未找到 step2 新增的 *_ms448_512_tta 目录"
    ms_dir = new_ms[-1]

    # ---------------------------------------------------------------- 4. 汇总
    pkgs = [
        ("多尺度448+512 TTA (主交)", ms_dir),
        ("单512 TTA (次交)", tta_dir),
        ("单512 notta (可选)", notta_dir),
    ]
    lines = ["③b 子中心头 出包完成", "训练目录: %s" % train_dir, ""]
    for tag, d in pkgs:
        zipf = os.path.join(d, "pred_results.zip")
        csvf = os.path.join(d, "pred_results.csv")
        m = md5(zipf) if os.path.exists(zipf) else "MISSING"
        n = linecount(csvf) if os.path.exists(csvf) else -1
        item = "%s\n  %s\n  MD5=%s  行=%d" % (tag, zipf, m, n)
        lines.append(item)
        print(item, flush=True)
    done = os.path.join(train_dir, "DONE.txt")
    with open(done, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n全部完成，汇总已写: %s" % done, flush=True)


if __name__ == "__main__":
    main()