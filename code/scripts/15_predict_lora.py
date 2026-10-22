# -*- coding: utf-8 -*-
"""
用 LoRA 微调后的骨干 + TTA 出提交文件。

═══ ⚠为什么不能复用 cache/ 里现成的测试特征 ═══

cache/test_features.npy 和 test_features_aug.npy 都是用【原版冻结 CLIP】抽的。
LoRA 一旦改了骨干，同一张图编码出来的 512 维向量就变了 —— 那些缓存**全部作废**。
所以这个脚本必须拿新骨干把测试集重抽一遍。这就是"碰骨干贵一百倍"的具体体现。

═══ TTA 的做法（沿用 09-04 已上榜验证过的配方，+2.357 分）═══

  最终特征 = L2归一化( mean(原图, 增强视角1..K) )
⚠必须**带上原图**。实测只用 5 个增强视角反而变差——原图是"标准视角"，是锚点，丢了就散了。
⛔增强不做颜色抖动：颜色是区分近缘物种的关键特征，改了等于制造错标。

═══ 口径对齐（保证和 61.357 那版只差一个变量）═══

  · 头：直接用 13 号训练时那个头（在同样的 80% 训练集上练的），**不重新在全量上训**
    —— 否则就同时改了"骨干"和"训练数据量"两个变量，涨跌都归不了因
  · α：WiSE-FT 系数，θ_α = θ0 + α·BA。默认 1.0（14 号扫描出来的峰值）
  · 类别编号：从 manifest 读真实文件夹名，不按索引硬生成（编号万一不连续就全错位）

用法:
  python scripts/15_predict_lora.py                      # 最新一次 LoRA + TTA + α=1
  python scripts/15_predict_lora.py --alpha 0.75         # 换 α
  python scripts/15_predict_lora.py --no-tta             # 只用原图，用来单独看 LoRA 的贡献
"""
import _bootstrap  # noqa: F401
from _bootstrap import banner

import argparse
import importlib.util
import json
import os
import sys
import time
import zipfile

import numpy as np
import torch

from configs import paths as P


def _load_sibling(fname, modname):
    """⚠必须注册 sys.modules，否则 Windows spawn 子进程反序列化 Dataset 会失败。"""
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(modname, os.path.join(here, fname))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


_ex = _load_sibling("01_extract_features.py", "extract01")
_aug = _load_sibling("10_extract_augmented.py", "augment10")
_lora = _load_sibling("13_train_lora.py", "lora13")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default=None, help="留空自动取最新一次 *_lora 训练")
    ap.add_argument("--tag", type=str, default=None, help="用哪个存档点，默认最后一个")
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="WiSE-FT 系数。14 号扫描的峰值是 1.0；⛔别填 >1（外推无理论保证）")
    ap.add_argument("--no-tta", action="store_true", help="关掉 TTA，只用原图")
    ap.add_argument("--k", type=int, default=5, help="TTA 用几个增强视角")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--space-after-comma", action="store_true")
    args = ap.parse_args()

    assert args.alpha <= 1.0 or True, ""
    if args.alpha > 1.0:
        print("  ⚠⚠ α>1 是外推，没有理论保证，只该用于方向诊断，不建议拿去提交")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------------------------------------------------------------- 找存档
    if args.run:
        run_dir = args.run if os.path.isdir(args.run) else os.path.join(P.RUNS_DIR, args.run)
    else:
        cands = sorted(d for d in os.listdir(P.RUNS_DIR)
                       if d.endswith("_lora")
                       and os.path.isdir(os.path.join(P.RUNS_DIR, d))
                       and any(f.startswith("ckpt_")
                               for f in os.listdir(os.path.join(P.RUNS_DIR, d))))
        assert cands, "runs/ 下没有 LoRA 训练产物，先跑 13_train_lora.py"
        run_dir = os.path.join(P.RUNS_DIR, cands[-1])
    cks = sorted(f for f in os.listdir(run_dir) if f.startswith("ckpt_") and f.endswith(".pt"))
    fname = "ckpt_%s.pt" % args.tag if args.tag else cks[-1]
    ck = torch.load(os.path.join(run_dir, fname), map_location="cpu", weights_only=False)
    tr = ck["args"]

    banner("15 生成提交：LoRA%s  α=%g  存档=%s"
           % ("" if args.no_tta else " + TTA", args.alpha, fname))
    print("  训练配置 : r=%d %s  lr_lora=%g  epochs=%s"
          % (tr["rank"], tr["targets"], tr["lr_lora"], tr["epochs"]))

    # ---------------------------------------------------------------- 骨干 + 补丁
    model, preprocess = _lora.load_clip_fp32(device)
    layers = None
    if tr.get("last_n_layers", 0) > 0:
        nb = len(model.visual.transformer.resblocks)
        layers = list(range(nb - tr["last_n_layers"], nb))
    targets = tuple(t.strip() for t in tr["targets"].split(","))
    _lora.inject_lora(model, tr["rank"], tr["lora_alpha"], targets,
                      tr.get("lora_mlp", False), layers)
    _, unexpected = model.load_state_dict(ck["lora"], strict=False)
    assert not unexpected, "存档里有模型不认识的键"
    model.to(device)
    n = 0
    for m in model.modules():                       # 施加 WiSE-FT 系数 α
        if isinstance(m, (_lora.LoRAAttention, _lora.LoRALinear)):
            m.scaling = m.scaling * args.alpha
            n += 1
    print("  补丁     : 已载入并按 α=%g 缩放（%d 个模块）" % (args.alpha, n))

    # ---------------------------------------------------------------- 头（同样按 α 插值）
    hd0 = np.load(tr["head_init"])
    W = (1 - args.alpha) * hd0["W"].astype(np.float32) + args.alpha * ck["W"].numpy().astype(np.float32)
    b = (1 - args.alpha) * hd0["b"].astype(np.float32) + args.alpha * ck["b"].numpy().astype(np.float32)
    print("  头       : W%s（α=0 端用 %s，α=1 端用存档里微调后的头）"
          % (tuple(W.shape), os.path.basename(tr["head_init"])))

    # ---------------------------------------------------------------- 抽测试特征
    paths = _ex.list_test_files()               # 与 01 同一套排序，保证行对齐
    print("  测试集   : %d 张" % len(paths))
    normalize = None
    from torchvision import transforms as T
    normalize = [t for t in preprocess.transforms if isinstance(t, T.Normalize)][0]

    t0 = time.time()
    views = []
    F0 = _lora.encode(model, preprocess, paths, device,
                      args.batch_size, args.num_workers, "原图  ")
    views.append(F0)

    if not args.no_tta:
        aug = _aug.build_aug(normalize)
        print("  TTA      : 原图 + %d 个增强视角取平均（⚠必须带原图，它是锚点）" % args.k)
        for v in range(args.k):
            torch.manual_seed(v)                # 与 10 号脚本同一套种子，可复现
            views.append(_lora.encode(model, aug, paths, device,
                                      args.batch_size, args.num_workers,
                                      "视角%d " % (v + 1)))

    # ⚠encode() 内部已逐视角 L2 归一化；这里取平均后必须【再归一化一次】
    X = np.mean(views, axis=0)
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    print("  特征     : %d 个视角平均 → %s，耗时 %.1f 分钟"
          % (len(views), X.shape, (time.time() - t0) / 60))

    # ---------------------------------------------------------------- 类别编号
    class_names = None
    if os.path.exists(P.F_MANIFEST):
        class_names = json.load(open(P.F_MANIFEST, encoding="utf-8")).get("class_names")
    if class_names and len(class_names) == W.shape[0]:
        print("  类别编号 : 取自 manifest 的真实文件夹名 (%s ~ %s)"
              % (class_names[0], class_names[-1]))
    else:
        class_names = ["%04d" % i for i in range(W.shape[0])]
        print("  ⚠类别编号: manifest 不可用，退化为按索引生成")

    # ---------------------------------------------------------------- 预测 + 写文件
    preds = (X @ W.T + b).argmax(1)
    uniq, cnt = np.unique(preds, return_counts=True)
    print("  预测覆盖 : %d / %d 类   每类 min=%d max=%d 均值=%.1f"
          % (len(uniq), W.shape[0], cnt.min(), cnt.max(), cnt.mean()))

    tag = "lora_a%g%s" % (args.alpha, "" if args.no_tta else "_tta")
    out_dir = os.path.join(run_dir, "submit_" + tag)
    os.makedirs(out_dir, exist_ok=True)
    sep = ", " if args.space_after_comma else ","
    csv_path = os.path.join(out_dir, P.SUBMISSION_FILENAME)
    with open(csv_path, "w", encoding="utf-8", newline="\n") as f:
        for p, c in zip(paths, preds):
            f.write("%s%s%s\n" % (os.path.basename(p), sep, class_names[c]))
    zip_path = os.path.join(out_dir, P.SUBMISSION_FILENAME.replace(".csv", ".zip"))
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(csv_path, arcname=P.SUBMISSION_FILENAME)

    # 和上一版提交比一比改了多少张，用来判断"这次改动到底动了多少东西"
    prev = os.path.join(P.RUNS_DIR, "20260904_232902_sce_tta", P.SUBMISSION_FILENAME)
    if os.path.exists(prev):
        old = {l.split(",")[0]: l.split(",")[1].strip()
               for l in open(prev, encoding="utf-8") if "," in l}
        new = {os.path.basename(p): class_names[c] for p, c in zip(paths, preds)}
        both = set(old) & set(new)
        diff = sum(1 for k in both if old[k] != new[k])
        print("  与上一版(SCE+TTA 61.357) 相比：改变了 %d/%d 张预测 (%.1f%%)"
              % (diff, len(both), 100.0 * diff / max(len(both), 1)))

    banner("完成")
    print("  CSV : %s  (%d 行, 无表头)" % (csv_path, len(preds)))
    print("  ZIP : %s  (%.1f MB)  ← 提交这个"
          % (zip_path, os.path.getsize(zip_path) / 1024 ** 2))
    print("\n  前 3 行:")
    for l in open(csv_path, encoding="utf-8").readlines()[:3]:
        print("   ", l.rstrip())


if __name__ == "__main__":
    main()
