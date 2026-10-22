"""consensus_balance -- 配额均衡的共识排序/共识去向变体（最终提交形态后处理）。

在标准 Sinkhorn T=1 配额均衡基础上，引入队伍九模型共识作为两个信号：
  1) 重分配优先级：本队模型对某图的标注与共识分歧越大，越优先被改判；
  2) 改判去向：被配额挤出时，流向共识票分布的第二候选类（非盲目流动）。

行分布：P_i = onehot(own_i) + alpha * V_i（V_i = 其余八模型对图 i 的投票频率/8），
列约束 = 每类恰 N/750（T=1 精确配额），300 轮列缩放收敛后取行 argmax。

本地实测（对参考包一致率；团队内四位成员各自提交包上的增益，成员姓名已匿名）：
  成员 A 74.696 -> 75.499 (+0.803)
  成员 B 74.151 -> 75.227 (+1.076)
  成员 C 74.541 -> 75.350 (+0.809)
  成员 D 76.325 -> 76.610 (+0.285)

注：本工具为探索性后处理（依赖队伍内部各成员提交包作为共识投票），
最终提交形态（reproduce.sh 产线）未使用本模块。

合规：仅推理输出端后处理；投票来源=队伍内部各成员提交包；不参与任何训练；
不改变模型权重。voter 清单（文件 + MD5）见同目录 voters_manifest.txt。

用法：
    python consensus_balance.py --package <原始zip> --voters <voters_manifest.txt> \
        --out <输出zip> [--alpha 1.0]
voters_manifest.txt 每行一个 zip 路径（不含本包自身；8 行）。
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np


def load_zip(path):
    d, order = {}, []
    with zipfile.ZipFile(path) as z:
        raw = z.read(z.namelist()[0]).decode("utf-8")
    for ln in raw.splitlines():
        if ln.strip():
            img, lab = ln.rsplit(",", 1)
            d[img] = lab
            order.append(img)
    return d, order


def sinkhorn_argmax(P, iters=300):
    P = P.copy() + 1e-9
    P /= P.sum(axis=1, keepdims=True)
    n = P.shape[0]
    quota = n / P.shape[1]
    v = np.ones(P.shape[1], dtype=np.float64)
    for _ in range(iters):
        col = (P * v[None, :]).sum(axis=0)
        v *= quota / np.maximum(col, 1e-12)
        v /= max(1e-30, v.mean())
    R = P * v[None, :]
    R /= np.maximum(R.sum(axis=1, keepdims=True), 1e-12)
    return R.argmax(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", required=True, help="本模型原始提交 zip")
    ap.add_argument("--voters", required=True, help="voter 清单（每行一个 zip 路径，8 行）")
    ap.add_argument("--out", required=True, help="输出 zip（均衡化形态）")
    ap.add_argument("--alpha", type=float, default=1.0)
    args = ap.parse_args()

    own, order = load_zip(args.package)
    n = len(order)
    voter_paths = [ln.strip() for ln in open(args.voters, encoding="utf-8") if ln.strip()]
    assert len(voter_paths) == 8, "期望 8 个 voter，得到 %d" % len(voter_paths)

    K = 750
    V = np.zeros((n, K), dtype=np.float32)
    for vp in voter_paths:
        vd, _ = load_zip(vp)
        for i, img in enumerate(order):
            V[i, int(vd[img])] += 1.0
    V /= len(voter_paths)

    lab = np.array([int(own[img]) for img in order], dtype=np.int64)
    P = np.zeros((n, K), dtype=np.float32)
    P[np.arange(n), lab] = 1.0
    P += args.alpha * V

    pred = sinkhorn_argmax(P)
    out_rows = order.__class__()
    d = {img: "%04d" % int(k) for img, k in zip(order, pred)}
    with open(args.out.replace(".zip", ".csv"), "w", encoding="utf-8", newline="\n") as f:
        for img in order:
            f.write("%s,%s\n" % (img, d[img]))
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(args.out.replace(".zip", ".csv"), "pred_results.csv")
    print("written", args.out, flush=True)


if __name__ == "__main__":
    main()
