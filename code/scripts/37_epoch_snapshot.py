# -*- coding: utf-8 -*-
"""
逐轮快照守护：把每轮的 ckpt_best.pt 另存一份，攒出一条【epoch 阶梯】。

═══════════════════════════════════════════════════════════════════════
为什么需要它（第一性原理）
═══════════════════════════════════════════════════════════════════════

`28_finetune_v2.py` 的存盘逻辑是：

    if m["matched"] > best:        # 验证集指标创新高才写
        torch.save(..., "ckpt_best.pt")   # ⚠只有这一个文件，反复覆盖

⛔**在 `--full-data` 模式下这条逻辑是坏的**：验证集已经并入训练集，
`m["matched"]` 是 **in-sample** 的——它衡量的是"模型把训练图背得多熟"，
必然随轮次单调上升。于是"挑指标最高的一轮" = **挑最过拟合的一轮**，
`ckpt_best.pt` 实际上就等于**最后一轮**，我们对轮次没有任何选择权。

**这为什么要命**：上一轮的 72.76 分是在 **ep4** 拿到的。
如果这次跑满 8 轮反而过了头，4 小时 GPU 换来一个更差的模型，**且无法回退**
（中间轮次的权重已经被覆盖掉，只能重训）。

⭐**解法不是去改判据（我们没有可信的判据——真值只能靠交榜），
   而是把选择权保留下来**：每轮存一份，之后拿 ep4/ep6/ep8 分别交榜，
   让**排行榜**来选，而不是让一把已知有病的尺子来选。

⚠成本：每份 353 MB。8 轮 = 2.8 GB。F 盘剩 524 GB，可忽略。

═══════════════════════════════════════════════════════════════════════
实现要点
═══════════════════════════════════════════════════════════════════════

⚠**必须等 history.json 写完再拷贝**。训练脚本的顺序是
    torch.save(ckpt_best.pt)  →  写 history.json
所以"history.json 变了"就是"ckpt_best.pt 已经写完了"的可靠信号，
不会拷到一个写了一半的文件。

⚠指标没创新高的那一轮不会写盘 → 这轮的 ckpt_best.pt 还是上一轮的内容。
   靠 mtime 判断：只在 ckpt_best.pt 确实被重写过时才快照。

用法（后台跑，纯文件拷贝，不碰 GPU）：
  python scripts/37_epoch_snapshot.py --run runs/20260906_110505_v2_320 --epochs 8
"""
import argparse
import json
import os
import shutil
import time


def read_hist(p):
    """⚠训练脚本可能正在写，读失败就当没变。"""
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f).get("history", [])
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--poll", type=float, default=20.0)
    ap.add_argument("--timeout-min", type=float, default=420.0)
    args = ap.parse_args()

    hp = os.path.join(args.run, "history.json")
    cp = os.path.join(args.run, "ckpt_best.pt")
    seen, last_ck_mtime, t0 = 0, 0.0, time.time()

    print("[snapshot] 守护 %s，目标 %d 轮" % (args.run, args.epochs), flush=True)

    while time.time() - t0 < args.timeout_min * 60:
        h = read_hist(hp)
        if h is not None and len(h) > seen:
            seen = len(h)
            ep = h[-1]["epoch"]
            time.sleep(3)                      # 让文件句柄彻底释放
            if os.path.exists(cp):
                mt = os.path.getmtime(cp)
                dst = os.path.join(args.run, "ckpt_ep%02d.pt" % ep)
                if mt > last_ck_mtime and not os.path.exists(dst):
                    shutil.copy2(cp, dst)
                    last_ck_mtime = mt
                    print("[snapshot] ep%02d → %s (%.0f MB)  train_acc=%.4f"
                          % (ep, os.path.basename(dst),
                             os.path.getsize(dst) / 1e6,
                             h[-1].get("train_acc", float("nan"))), flush=True)
                else:
                    print("[snapshot] ep%02d 指标未创新高，ckpt 没被重写，跳过" % ep,
                          flush=True)
            if seen >= args.epochs:
                print("[snapshot] 已到 %d 轮，收工。" % args.epochs, flush=True)
                break
        time.sleep(args.poll)

    snaps = sorted(f for f in os.listdir(args.run) if f.startswith("ckpt_ep"))
    print("[snapshot] 共留下 %d 份：%s" % (len(snaps), ", ".join(snaps)), flush=True)


if __name__ == "__main__":
    main()
