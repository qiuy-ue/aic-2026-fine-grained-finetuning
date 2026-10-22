# -*- coding: utf-8 -*-
"""
用训好的线性头对官方测试集出预测，生成可直接提交的 pred_results.csv + zip。

提交格式（来自赛题原文「十三、提交要求」，不是猜的）：
  · CSV，每行两个字段：图片文件名, 类别编号
  · 类别编号统一四位数字，不足四位前补 0        例: xxxxxxxxxxxx.jpg,0001
  · 图片文件名须与测试集完全一致（含大小写和扩展名）
  · 文件必须命名为 pred_results.csv，再压成一个 zip 提交
  · 原文示例逗号后带一个空格，判断为排版所致；默认按标准 CSV 不加空格，
    如需完全照抄示例可加 --space-after-comma

用法:
  python scripts/03_predict_test.py                    # 用最新一次训练的头
  python scripts/03_predict_test.py --run 20260904_1633 # 指定某次训练
  python scripts/03_predict_test.py --list             # 列出所有可用的训练结果
"""
import _bootstrap  # noqa: F401
from _bootstrap import banner

import argparse
import json
import os
import zipfile

import numpy as np

from configs import paths as P


def list_runs():
    if not os.path.isdir(P.RUNS_DIR):
        return []
    return sorted(d for d in os.listdir(P.RUNS_DIR)
                  if os.path.exists(os.path.join(P.RUNS_DIR, d, "head.npz")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default=None,
                    help="runs/ 下的目录名；默认用时间戳最新的一个")
    ap.add_argument("--list", action="store_true", help="列出所有可用训练结果后退出")
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--space-after-comma", action="store_true",
                    help="逗号后加空格，完全照抄赛题原文示例的排版")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    runs = list_runs()
    if args.list or not runs:
        banner("可用的训练结果")
        if not runs:
            print("  (空) 请先跑: python scripts/02_train_head.py")
            return
        for d in runs:
            mp = os.path.join(P.RUNS_DIR, d, "metrics.json")
            acc = "?"
            if os.path.exists(mp):
                m = json.load(open(mp, encoding="utf-8"))
                acc = "验证acc=%.4f  C=%g  后端=%s" % (
                    m.get("best_val_acc", float("nan")), m.get("best_C", 0), m.get("backend"))
            print("  %-28s %s" % (d, acc))
        if args.list:
            return

    run = args.run or runs[-1]
    run_dir = os.path.join(P.RUNS_DIR, run)
    head_path = os.path.join(run_dir, "head.npz")
    if not os.path.exists(head_path):
        raise SystemExit("找不到 %s\n可用: %s" % (head_path, runs))

    banner("03 生成提交文件")
    print("  使用训练结果 :", run)

    # ---------------------------------------------------------------- 头
    hd = np.load(head_path)
    W, b = hd["W"], hd["b"]
    do_norm = bool(hd["normalize"])
    print("  头形状       : W%s  b%s   (%d 个参数)"
          % (W.shape, b.shape, W.size + b.size))
    print("  L2 归一化    : %s  (必须与训练时一致，否则预测会全错)" % do_norm)

    mp = os.path.join(run_dir, "metrics.json")
    if os.path.exists(mp):
        m = json.load(open(mp, encoding="utf-8"))
        print("  该模型验证acc: %.4f  (C=%g, 带噪验证集上的值，非官方榜分数)"
              % (m.get("best_val_acc", float("nan")), m.get("best_C", 0)))

    # ---------------------------------------------------------------- 测试特征
    fx = P.F_TEST_FEAT.replace(".npy", "_smoke.npy") if args.smoke else P.F_TEST_FEAT
    fp = P.F_TEST_PATH.replace(".txt", "_smoke.txt") if args.smoke else P.F_TEST_PATH
    if not os.path.exists(fx):
        raise SystemExit("找不到测试特征 %s\n请先跑 01_extract_features.py" % fx)

    X = np.load(fx).astype(np.float32)
    paths = open(fp, encoding="utf-8").read().splitlines()
    if len(paths) != len(X):
        raise SystemExit("对齐检查失败：特征 %d 行，路径 %d 行" % (len(X), len(paths)))
    print("  测试样本     : %d" % len(X))

    if do_norm:
        X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)

    # ---------------------------------------------------------------- 类别编号映射
    # 类别索引 → 官方四位编号。优先用 manifest 里记录的真实文件夹名，
    # 而不是想当然地按 "%04d" % idx 生成——万一编号不连续就会全错位。
    mf = P.F_MANIFEST.replace(".json", "_smoke.json") if args.smoke else P.F_MANIFEST
    class_names = None
    if os.path.exists(mf):
        class_names = json.load(open(mf, encoding="utf-8")).get("class_names")
    if class_names and len(class_names) == W.shape[0]:
        print("  类别编号来源 : manifest 记录的真实文件夹名 (%s ~ %s)"
              % (class_names[0], class_names[-1]))
    else:
        class_names = ["%04d" % i for i in range(W.shape[0])]
        print("  ⚠类别编号来源: manifest 不可用，退化为按索引生成 %s ~ %s"
              % (class_names[0], class_names[-1]))

    # ---------------------------------------------------------------- 预测
    preds = np.empty(len(X), dtype=np.int64)
    for s in range(0, len(X), args.batch_size):
        e = min(s + args.batch_size, len(X))
        preds[s:e] = (X[s:e] @ W.T + b).argmax(1)

    uniq, cnt = np.unique(preds, return_counts=True)
    print("  预测覆盖类别 : %d / %d" % (len(uniq), W.shape[0]))
    print("  每类预测数   : min=%d  max=%d  均值=%.1f" % (cnt.min(), cnt.max(), cnt.mean()))
    if len(uniq) < W.shape[0]:
        print("  ⚠ 有 %d 个类一次都没被预测到" % (W.shape[0] - len(uniq)))

    # ---------------------------------------------------------------- 写文件
    sep = ", " if args.space_after_comma else ","
    csv_path = os.path.join(run_dir, P.SUBMISSION_FILENAME)
    with open(csv_path, "w", encoding="utf-8", newline="\n") as f:
        for p, c in zip(paths, preds):
            f.write("%s%s%s\n" % (os.path.basename(p), sep, class_names[c]))

    zip_path = os.path.join(run_dir, P.SUBMISSION_FILENAME.replace(".csv", ".zip"))
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(csv_path, arcname=P.SUBMISSION_FILENAME)  # zip 内必须是这个名字

    banner("完成")
    print("  CSV : %s  (%.1f MB, %d 行, 无表头)"
          % (csv_path, os.path.getsize(csv_path) / 1024 ** 2, len(preds)))
    print("  ZIP : %s  (%.1f MB)  ← 提交这个"
          % (zip_path, os.path.getsize(zip_path) / 1024 ** 2))
    print("\n  前 3 行预览:")
    with open(csv_path, encoding="utf-8") as f:
        for _ in range(3):
            print("   ", f.readline().rstrip())


if __name__ == "__main__":
    main()
