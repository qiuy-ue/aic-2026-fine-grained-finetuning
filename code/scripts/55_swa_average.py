# -*- coding: utf-8 -*-
"""55 · 权重平均（SWA）：把同一次训练的若干个 ckpt 逐元素平均成**一份**权重。

⭐为什么做这个（2026-09-15）：28 号第 737 行有个一直没被发现的 bug——

    use_ema = m_ema["matched"] > m_raw["matched"]
    src = ema_v.shadow if use_ema else visual      # ⛔只存赢的那份，输的那份直接丢掉

  判据 `matched` 是 **in-sample 分布匹配**（--full-data 下验证集已并入训练）。
  而这段代码往下三行的注释亲口写着这个判据是陷阱、"选哪一轮交给榜分裁决，本地不做这个决定"。
  ⭐这条纪律给「选轮数」打了补丁（12 个 ckpt 全存），给「选 EMA/raw」**没打**。

  后期 raw 必赢是**机制必然**：raw 刚在这批数据上走完梯度，EMA 是它的滞后平均，
  比"谁把训练集背得更熟" raw 按定义赢。实测赢的幅度：L@ep12 = 0.0002、M@ep12 = 0.0000
  （严格 `>`，平局判 raw）。而 in-sample 与榜分实测**反向过两次**
  （③b .9739→78.27，L .9702→79.85）→ 这一票零信息，却销毁了输的那份权重。

  → L/M 的 EMA 权重已经没了。**checkpoint 平均是 EMA 的离线版**，可以零训练把它捡回来。

⭐前提正好成立（⛔不成立就别做）：
  L 的 ep11→ep12 in-sample 只动 +0.0002 → 已经在同一个盆地里晃，不是还在赶路。
  余弦退火的尾段 LR 极小，几个端点彼此很近 —— 这正是 SWA 成立的标准场景。

⛔ **不是集成**：产出是单一一份权重，推理只跑一次前向。骨干仍是 OpenAI 基线 CLIP ViT-B/32。
   （2026-09-15 用户裁定：只要还是 openai 基线的 clip 就可以。）

⚠⚠ **最要命的坑，脚本会自己拦**：同一次训练里，`used_ema` 早期是 True、后期是 False
   （L: ep01~08 存的是 EMA 权重，ep09~12 存的是 raw）。**两种权重混着平均是错的**，
   必须全部同源。见 check_homogeneous()。

用法：
  python scripts\55_swa_average.py --train-dir runs\20260914_125846_v2_448_cos_dp020_re25_ep12 \
      --epochs 9,10,11,12 --dry-run          # 先看一眼再动手
"""
import argparse
import os
import sys

import torch

ROOT = r"D:\aic_data\AIC\AIC2026_9_4"


def log(msg):
    print(msg, flush=True)


def banner(t):
    log("\n" + "=" * 68 + "\n  " + t + "\n" + "=" * 68)


def load_meta(path):
    """只读元信息，⛔不把 337MB 的权重留在内存里。"""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    meta = dict(epoch=ck.get("epoch"), used_ema=bool(ck.get("used_ema", False)),
                args=ck.get("args", {}), metrics=ck.get("metrics", {}),
                has_s="s" in ck, nkeys=len(ck["visual"]))
    del ck
    return meta


def check_homogeneous(paths, metas):
    """⭐三道闸，任何一道不过就退出。这几道都是"错了也不会报错、只会悄悄变差"的那种。"""
    ok = True

    # 闸①：used_ema 必须全同 —— 混着平均 = 把两个不同的东西加起来
    flags = {m["used_ema"] for m in metas}
    if len(flags) > 1:
        log("  ⛔ used_ema 不一致，⛔不能平均：")
        for p, m in zip(paths, metas):
            log("     ep%02d used_ema=%s  %s" % (m["epoch"], m["used_ema"], os.path.basename(p)))
        log("     （L 的 ep01~08 存的是 EMA 权重、ep09~12 存的是 raw，只能在同一段里选）")
        ok = False
    else:
        log("  ✅ used_ema 全部 = %s（同源）" % list(flags)[0])

    # 闸②：影响推理链路的训练参数必须全同（26/45 靠 args 决定头形态和位置编码网格）
    KEYS = ["resolution", "cosine_head", "subcenters", "drop_path", "llrd", "lr_backbone"]
    for k in KEYS:
        vals = {repr(m["args"].get(k)) for m in metas}
        if len(vals) > 1:
            log("  ⛔ args[%s] 不一致：%s" % (k, vals))
            ok = False
    if ok:
        log("  ✅ args 关键字段一致：%s"
            % ", ".join("%s=%s" % (k, metas[0]["args"].get(k)) for k in KEYS))

    # 闸③：state_dict 键数必须全同
    if len({m["nkeys"] for m in metas}) > 1:
        log("  ⛔ visual 的键数不一致：%s" % {m["nkeys"] for m in metas})
        ok = False
    else:
        log("  ✅ visual 键数一致：%d" % metas[0]["nkeys"])

    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", type=str, required=True)
    ap.add_argument("--epochs", type=str, default="9,10,11,12",
                    help="逗号分隔的轮号。⛔只能取 used_ema 相同的那一段")
    ap.add_argument("--out", type=str, default=None, help="默认 ckpt_swa<首><尾>.pt")
    ap.add_argument("--dry-run", action="store_true", help="只做检查和统计，不写文件")
    args = ap.parse_args()

    tdir = args.train_dir if os.path.isabs(args.train_dir) else os.path.join(ROOT, args.train_dir)
    eps = [int(e) for e in args.epochs.split(",")]
    paths = [os.path.join(tdir, "ckpt_ep%02d.pt" % e) for e in eps]

    banner("55 权重平均（SWA）  %d 个端点：ep%s" % (len(eps), " / ep".join(str(e) for e in eps)))
    log("  训练目录 %s" % os.path.basename(tdir))

    for p in paths:
        if not os.path.exists(p):
            log("  ⛔ 缺 %s" % p)
            sys.exit(1)

    # ── 1. 一致性检查（⛔先查再平均）
    banner("① 一致性检查")
    metas = [load_meta(p) for p in paths]
    for m in metas:
        log("  ep%02d  in-sample matched=%.4f  used_ema=%s"
            % (m["epoch"], m["metrics"].get("matched", float("nan")), m["used_ema"]))
    if not check_homogeneous(paths, metas):
        log("\n  ⛔ 检查未通过，不生成任何文件。")
        sys.exit(1)

    # ── 2. 流式累加（⛔别一次性载入 4×337MB）
    banner("② 逐元素平均")
    n = len(paths)
    acc, ref_blob = None, None
    head = {"W": None, "b": None, "s": 0.0}
    for i, p in enumerate(paths):
        ck = torch.load(p, map_location="cpu", weights_only=False)
        sd = ck["visual"]
        if acc is None:
            acc = {k: (v.double().clone() if v.is_floating_point() else v.clone())
                   for k, v in sd.items()}
            head["W"] = ck["W"].double().clone()
            head["b"] = ck["b"].double().clone()
            ref_blob = {k: ck[k] for k in ("args", "epoch", "metrics") if k in ck}
        else:
            for k, v in sd.items():
                if v.is_floating_point():
                    acc[k] += v.double()
                # ⚠非浮点（若有）不平均，沿用第一份 —— 平均整数张量没有意义
            head["W"] += ck["W"].double()
            head["b"] += ck["b"].double()
        head["s"] += float(ck.get("s", 0.0))
        log("  [%d/%d] ep%02d 已累加" % (i + 1, n, ck.get("epoch")))
        del ck, sd

    n_float = 0
    for k in acc:
        if acc[k].is_floating_point():
            acc[k] /= n
            n_float += 1
    head["W"] /= n
    head["b"] /= n
    head["s"] /= n
    log("  平均完成：%d/%d 个浮点张量参与" % (n_float, len(acc)))

    # ── 3. 落盘（dtype 必须还原成 float32，否则 26 的 strict=True 会通过但显存翻倍）
    last = torch.load(paths[-1], map_location="cpu", weights_only=False)
    out_sd = {k: (v.to(last["visual"][k].dtype) if v.is_floating_point() else v)
              for k, v in acc.items()}

    blob = {"visual": out_sd,
            "W": head["W"].to(last["W"].dtype),
            "b": head["b"].to(last["b"].dtype),
            "epoch": "swa%s" % "_".join(str(e) for e in eps),
            "args": last["args"],
            "metrics": last["metrics"],          # ⚠沿用末轮的，⛔这不是 SWA 自己的指标
            "used_ema": last.get("used_ema", False),
            "swa_from": [os.path.basename(p) for p in paths]}
    if "s" in last:
        blob["s"] = head["s"]
        log("  余弦头 s：%.4f（末轮 %.4f）" % (head["s"], float(last["s"])))

    # 位移体检：SWA 权重离末轮有多远？太远说明这几个端点还没进同一个盆地。
    d2 = sum(float(((out_sd[k].float() - last["visual"][k].float()) ** 2).sum())
             for k in out_sd if out_sd[k].is_floating_point())
    n2 = sum(float((last["visual"][k].float() ** 2).sum())
             for k in out_sd if out_sd[k].is_floating_point())
    log("  ‖SWA − ep%s‖ / ‖ep%s‖ = %.5f   （很小=几个端点确在同一盆地，SWA 前提成立）"
        % (eps[-1], eps[-1], (d2 / max(n2, 1e-12)) ** 0.5))

    if args.dry_run:
        log("\n  DRY-RUN：检查全过、平均算完，⛔未写文件。去掉 --dry-run 才落盘。")
        return

    out = args.out or os.path.join(tdir, "ckpt_swa%02d%02d.pt" % (eps[0], eps[-1]))
    if not os.path.isabs(out):
        out = os.path.join(tdir, out)
    torch.save(blob, out)
    log("\n  ⭐已写出 %s  (%.0f MB)" % (out, os.path.getsize(out) / 1e6))
    log("  下一步：26_predict_finetuned.py --ckpt %s --eval-res 512" % out)
    log("  ⛔出包只传 --ckpt 和 --eval-res 512，其余全默认（改 batch/workers 会换 TTA 口径）")


if __name__ == "__main__":
    main()
