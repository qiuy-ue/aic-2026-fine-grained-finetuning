# -*- coding: utf-8 -*-
"""
39：448px 训练的**预算探测**——显存能吃多大 batch、一轮要多久。等流水线跑完自动执行。

⛔为什么必须实测而不是估：
   我估过一次「448/bs16 约 3.8 GB」，但那个估算依赖一个我不确定的前提——
   attention 显存是否走 memory-efficient 实现。若不走，它按 token² 增长：
   448 是 196 token、320 是 100 token → **3.84 倍**，估算会整个塌掉。
   ⭐显存这种东西没有"大概"，撞了就是 OOM 崩在第 3 小时。**测。**

⭐同时测吞吐（img/s），因为「值不值得投」取决于一轮要多久：
   若 448 一轮 2 小时，4 轮 = 8 小时，这已经是要占掉一整晚的决策，
   必须和「多训几轮 320」放在同一张表上比，而不是拍脑袋。

⚠等待逻辑：等的是**「38 号流水线这个进程消失」**，不是「GPU 空闲」。
   因为流水线在两个子任务之间会**短暂释放显存**，光看显存会误判为"空了"而插队撞车。
   ⭐等进程是等事件本身，等显存是等它的一个不可靠代理。
"""
import argparse
import io
import os
import subprocess
import sys
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))


def alive(pid):
    """进程还在不在。⚠用 tasklist 而不是 os.kill(pid,0)：Windows 上后者对已退出
       但句柄未回收的进程仍返回成功，会一直等下去。"""
    try:
        o = subprocess.run(["tasklist", "/FI", "PID eq %d" % pid, "/NH"],
                           capture_output=True, text=True, timeout=30)
        return str(pid) in o.stdout
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait-pid", type=int, default=0, help="等这个进程退出（38号流水线）")
    ap.add_argument("--res", type=int, default=448)
    ap.add_argument("--batches", type=int, nargs="+", default=[8, 12, 16, 24, 32])
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--n-train", type=int, default=117000,
                    help="一轮的样本数，用于把 img/s 换算成小时（全量约 11.7 万）")
    args = ap.parse_args()

    if args.wait_pid:
        t0 = time.time()
        while alive(args.wait_pid):
            print("[probe] 等 PID %d（38号流水线）结束… 已等 %.0f 分"
                  % (args.wait_pid, (time.time() - t0) / 60), flush=True)
            time.sleep(120)
        print("[probe] ⭐流水线已结束，等 90 秒让显存回收干净。", flush=True)
        time.sleep(90)

    import torch
    import clip
    import _bootstrap  # noqa: F401
    from configs import paths as P
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "finetune24", os.path.join(ROOT, "scripts", "24_finetune_full.py"))
    _ft = importlib.util.module_from_spec(spec)
    sys.modules["finetune24"] = _ft
    spec.loader.exec_module(_ft)

    dev = "cuda"
    total = torch.cuda.get_device_properties(0).total_memory / 2 ** 20
    print("=" * 70)
    print("GPU 总显存 %.0f MiB   探测分辨率 %d px" % (total, args.res))
    print("=" * 70)

    def probe_one(bs):
        """探一个 batch size，返回 (峰值显存MiB, img/s)。

        ⭐**做成独立函数是必需的，不是为了好看**：函数返回后，模型/优化器等局部变量
           随作用域自动回收，下一轮才是从干净状态开始测。
        ⛔别想着用 `locals().pop(...)` 手动清 —— CPython 里 `locals()` 返回的是**快照字典**，
           改它根本不影响真正的局部变量，上一轮的显存会一直挂着，
           导致读数偏大、更早误报 OOM，把 batch 决策带偏。
        """
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model, _ = clip.load(P.CLIP_MODEL_NAME, device="cpu",
                             download_root=P.CLIP_DOWNLOAD_ROOT)
        model.float()
        _ft.interpolate_pos_embed(model, args.res)
        visual = model.visual.to(dev)
        if args.ckpt:
            ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
            sd = dict(ck["visual"])
            k = "positional_embedding"
            if sd[k].shape != visual.state_dict()[k].shape:
                sd[k] = _ft._resize_pos_embed(sd[k], args.res // 32)
            visual.load_state_dict(sd, strict=False)
        head = torch.nn.Linear(512, 500).to(dev)
        # ⚠必须建真的 AdamW 并**真的 step 一次**：优化器状态(exp_avg/exp_avg_sq)
        #   占的显存是权重的 2 倍，只前向反向会漏算这一大块。
        opt = torch.optim.AdamW(list(visual.parameters()) + list(head.parameters()),
                                lr=1e-5, weight_decay=0.05)
        scaler = torch.amp.GradScaler("cuda")
        n_it, t0 = 4, None
        for it in range(n_it + 1):
            x = torch.randn(bs, 3, args.res, args.res, device=dev)
            yb = torch.randint(0, 500, (bs,), device=dev)
            with torch.amp.autocast("cuda"):
                loss = torch.nn.functional.cross_entropy(head(visual(x)), yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            if it == 0:               # ⚠第 0 次含 cudnn 选算法的开销，不计时
                t0 = time.time()
        return (torch.cuda.max_memory_allocated() / 2 ** 20,
                bs * n_it / (time.time() - t0))

    print("\n%-6s %12s %12s %10s %12s" % ("batch", "峰值显存MiB", "占总显存", "img/s", "一轮(小时)"))
    print("-" * 70)
    best = 0
    for bs in args.batches:
        try:
            peak, ips = probe_one(bs)
            print("%-6d %12.0f %11.1f%% %10.1f %12.2f"
                  % (bs, peak, 100 * peak / total, ips, args.n_train / ips / 3600))
            best = bs
        except RuntimeError as e:
            # ⚠OutOfMemoryError 继承自 RuntimeError，捕父类更稳（老版本 torch 没有那个子类）
            oom = "out of memory" in str(e).lower()
            print("%-6d %12s %s" % (bs, "⛔ OOM" if oom else "⛔ 报错", "" if oom else str(e)[:70]))
            break
        except Exception as e:
            print("%-6d %12s %s" % (bs, "⛔ 报错", str(e)[:70]))
            break
        finally:
            torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("⭐最大可用 batch = %d" % best)
    print("⚠实际训练还要加 EMA 影子权重（≈权重大小的 1 份）和 dataloader 的锁页内存，")
    print("  留 15%% 余量更稳 → 建议用 **%d**" % max(4, int(best * 0.85)))
    print("=" * 70)


if __name__ == "__main__":
    main()
