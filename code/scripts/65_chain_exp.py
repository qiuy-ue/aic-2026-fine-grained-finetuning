# -*- coding: utf-8 -*-
"""65 · 训练侧实验的一条龙执行链（09-17 起的标准口径：**ms448_544 + w3cc**）

冒烟 → 12 轮训练 → 建 544 池 → 建 448 池 → 59 组 w3cc 出包 → 算配对分歧 → 写 DONE。

═══════════════════════════════════════════════════════════════════════
⭐ 为什么不接着用 52_chain_j.py（两条硬伤，都会让 10 小时白跑）
═══════════════════════════════════════════════════════════════════════
① **出包口径是旧的**：52 走 26+45 = `ms448_512` + base 6 视角，而 09-17 起全项目
   对照的是 **81.0419**（`c3both_w3cc` / `ms448_544` / 10 视角）。用旧口径出的包
   ⛔没法跟 81.0419 比 —— 上一轮花了三发才买回来的教训（09 章 9.9「口径全程钉死」）。
② **训练目录 glob 写死** `*_v2_448_cos_dp*`：一旦加 `--freeze-first-n 4`，28 号生成的
   目录名是 `..._v2_448_fz4_cos_dp020_re25_ep12`，**匹配不上**，链会在"找不到训练目录"处中止。

═══════════════════════════════════════════════════════════════════════
⭐ 内建的单变量闸（本链存在的第二个理由）
═══════════════════════════════════════════════════════════════════════
纪律「⛔一次只改一个旋钮」以前靠人肉核对。本链把它变成**开跑前的断言**：
把本次的有效参数与 **L@ep12 的 history.json**（81.0419 那条线的骨干）逐字段比，
diff 必须**恰好等于**命令行声明的那几个旋钮，否则⛔直接拒绝起飞。
（J 那次合并两个变量的代价，台账里记着；M/R 的判据作废也是同一类病。）

═══════════════════════════════════════════════════════════════════════
用法（⛔必须 Start-Process 独立启动：agent 的后台 job 会被回收，已证实）
═══════════════════════════════════════════════════════════════════════
  $env:PYTHONIOENCODING='utf-8'
  Start-Process -FilePath 'D:\Anaconda\envs\dl\python.exe' `
    -ArgumentList 'scripts\65_chain_exp.py --tag V --freeze-first-n 4' `
    -WorkingDirectory 'D:\aic_data\AIC\AIC2026_9_4' `
    -RedirectStandardOutput 'D:\aic_data\AIC\AIC2026_9_4\runs\chain_V.log' `
    -RedirectStandardError  'D:\aic_data\AIC\AIC2026_9_4\runs\chain_V.err'

  排队（⛔单卡串行，别并行）：下一发加 `--wait-for runs\_chain_V_DONE.txt`

  自检：`--dry-run` 只打印命令与单变量 diff，不动 GPU。
"""
import argparse
import csv
import glob
import hashlib
import json
import math
import os
import subprocess
import sys
import time

PY = r"D:\Anaconda\envs\dl\python.exe"
ROOT = r"D:\aic_data\AIC\AIC2026_9_4"
RUNS = os.path.join(ROOT, "runs")

# ── 对照基准：榜面 81.0419 的那个包（c3both_w3cc / ms448_544 / L@ep12）
REF_DIR = os.path.join(RUNS, "20260917_090512_rcp_c3both_w3cc_ms448_544_ep12")
REF_CSV = os.path.join(REF_DIR, "pred_results.csv")
REF_SCORE = 81.0419
# ── L@ep12 的训练参数真值（单变量闸拿它当尺子）
L_HISTORY = os.path.join(RUNS, "20260914_125846_v2_448_cos_dp020_re25_ep12", "history.json")

# ── 口径常数：⛔一个字都别改，改了就不再是 81.0419 的对照
C3_VIEWS = "orig,mild0,mild1,mild2,mild3,mild4,cc085,cc070,cc055,cc040"
W3CC = "cc085:3,cc070:3,cc055:3,cc040:3"

# ── L 的训练配方，逐字照抄 history.json（⛔别"顺手"改任何一项）
TRAIN_ARGS = [
    "--resolution", "448",
    "--epochs", "12",
    "--batch-size", "32",
    "--lr-backbone", "2e-05",
    "--llrd", "0.85",
    "--degrade-p", "0.875",
    "--cosine-head",
    "--aug-scale", "0.3",
    "--num-workers", "4",
    "--eval-workers", "2",
    "--resume", "runs/20260906_200353_v2_384/ckpt_best.pt",
    "--pseudo", "cache/pseudo_labels.npz",
    "--pseudo-weight", "0.5",
    "--full-data",
    "--seed", "0",
    "--drop-path", "0.2",
    "--random-erasing", "0.25",
]

# 命令行旋钮 → 28 号 history.json 里的字段名（单变量闸用）
KNOB2KEY = {
    "margin": "margin",
    "freeze_first_n": "freeze_first_n",
    "epochs": "epochs",
    "drop_path": "drop_path",
    "random_erasing": "random_erasing",
    "lr_backbone": "lr_backbone",
    "head_init": "head_init",          # ⭐实验 C：换头初值（文本塔原型 npz），格式同 head_loss_sce.npz
    "distill": "distill",              # ⭐实验 C2：训练集自蒸馏的教师软标签 npz
    "text_anchor": "text_anchor",      # ⭐实验 K：CLIP 文本原型当固定锚点（表征约束）
    "text_lambda": "text_lambda",
}
# 开关型旋钮（store_true）→ 字段名
SWITCH2KEY = {
    "aug_geo": "aug_geo",              # ⭐实验 C：几何 TrivialAugment
}
# ⭐"删掉一个 flag"也是旋钮：实验 C 从原版 CLIP 起（删 --resume）、不带测试集伪标签（删 --pseudo）。
#   闸门里这类 key 若在命令行缺席，有效值记为 None，与 L 的路径比就会出现在 diff 里，⛔不会被漏掉。
REMOVABLE = {"resume": "--resume", "pseudo": "--pseudo"}
# TRAIN_ARGS 里的 flag → 字段名（用来"模拟"出本次的有效参数字典）
FLAG2KEY = {
    "--resolution": "resolution", "--epochs": "epochs", "--batch-size": "batch_size",
    "--lr-backbone": "lr_backbone", "--llrd": "llrd", "--degrade-p": "degrade_p",
    "--aug-scale": "aug_scale", "--num-workers": "num_workers",
    "--eval-workers": "eval_workers", "--resume": "resume", "--pseudo": "pseudo",
    "--pseudo-weight": "pseudo_weight", "--seed": "seed", "--drop-path": "drop_path",
    "--random-erasing": "random_erasing", "--freeze-first-n": "freeze_first_n",
    "--margin": "margin", "--lr-head": "lr_head", "--weight-decay": "weight_decay",
    "--head-init": "head_init", "--distill": "distill",
    "--distill-alpha": "distill_alpha", "--distill-temp": "distill_temp",
    "--text-anchor": "text_anchor", "--text-lambda": "text_lambda",
}
FLAG2KEY_STORE_TRUE = {"--cosine-head": "cosine_head", "--full-data": "full_data",
                       "--degrade-fixed": "degrade_fixed", "--soft-pseudo": "soft_pseudo",
                       "--aug-geo": "aug_geo"}


def log(msg):
    print("[%s] %s" % (time.strftime("%m-%d %H:%M:%S"), msg), flush=True)


def run(cmd, what):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)          # 记忆坑①：沙箱 shell 会污染，必须清
    env.pop("PYTHONHOME", None)
    env["PYTHONIOENCODING"] = "utf-8"
    log("▶ %s：%s" % (what, " ".join(cmd[1:])))
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT, env=env)
    log("■ %s 结束，退出码 %d，耗时 %.1f 分钟" % (what, r.returncode, (time.time() - t0) / 60))
    return r.returncode


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest().upper()


def newest_new(pattern, before):
    """返回本步骤**新增**的目录（⛔别用"最新目录"，并行/残留会串味）。"""
    new = sorted(set(glob.glob(os.path.join(RUNS, pattern))) - before)
    return new[-1] if new else None


def read_pred(p):
    m = {}
    with open(p, newline="", encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                m[row[0].strip()] = row[-1].strip()
    return m


def disagree(a, b):
    """配对分歧 + 实算误差棒（判读纪律 1：误差棒要实算，⛔别拍脑袋）。"""
    A, B = read_pred(a), read_pred(b)
    keys = set(A) & set(B)
    d = sum(1 for k in keys if A[k] != B[k])
    n = max(len(keys), 1)
    se = math.sqrt(max(d, 1)) / n * 100.0
    return d, n, se, 2 * se


# ══════════════════════════════════════════════════════════ 单变量闸
def effective_args(train_args):
    """把 TRAIN_ARGS 解析成 {字段: 值}，缺省项留给 L 的值（比较时只看被显式给出的）。"""
    out, i = {}, 0
    while i < len(train_args):
        a = train_args[i]
        if a in FLAG2KEY_STORE_TRUE:
            out[FLAG2KEY_STORE_TRUE[a]] = True
            i += 1
        elif a in FLAG2KEY:
            out[FLAG2KEY[a]] = train_args[i + 1]
            i += 2
        else:                                   # 未登记的 flag → 闸门必须能看见它
            out["⚠未登记:" + a] = "?"
            i += 1
    for key in REMOVABLE:                       # ⭐缺席即 None（"删旋钮"也要被闸门看见）
        out.setdefault(key, None)
    return out


def same(x, y):
    """L 的 history.json 里是 python 值，TRAIN_ARGS 里是字符串，按数值比。"""
    if x is None or y is None:
        return x is None and y is None
    if isinstance(y, bool) or isinstance(x, bool):
        return bool(x) == bool(y)
    try:
        return abs(float(x) - float(y)) < 1e-12
    except (TypeError, ValueError):
        # 路径：统一斜杠后比**文件名**（L 的 head_init 记的是绝对路径，TRAIN_ARGS 里可能是相对路径）
        sx, sy = str(x).replace("\\", "/"), str(y).replace("\\", "/")
        return sx == sy or (("/" in sx or "/" in sy) and sx.split("/")[-1] == sy.split("/")[-1])


def single_knob_gate(train_args, declared):
    """⭐与 L@ep12 逐字段比，diff 必须恰好 = 声明的旋钮，否则⛔拒绝起飞。"""
    with open(L_HISTORY, encoding="utf-8") as f:
        ref = json.load(f)["args"]
    eff = effective_args(train_args)
    diff = []
    for k, v in sorted(eff.items()):
        if k.startswith("⚠未登记:"):
            diff.append((k, "—", v))
            continue
        if k not in ref:                        # L 那会儿还没有这个旋钮（如 margin）
            diff.append((k, "(L 里没有)", v))
            continue
        if not same(v, ref[k]):
            diff.append((k, ref[k], v))
    print("  ┌─ 单变量闸：本次 vs L@ep12（81.0419 那条线的骨干）")
    if not diff:
        print("  │  逐字段全等（⚠ 声明了旋钮却没有 diff = 旋钮没生效，见下）")
    for k, a, b in diff:
        print("  │  %-18s  L=%-28s 本次=%s" % (k, a, b))
    got = {k for k, _, _ in diff}
    want = set(declared)
    extra, missing = got - want, want - got
    ok = not extra and not missing
    print("  │  声明的旋钮：%s" % (sorted(want) or "（无 → 这是一次复现跑）"))
    print("  └─ %s" % ("✅ diff 恰好等于声明，单变量成立"
                       if ok else
                       "⛔ 不成立！多出 %s，缺少 %s" % (sorted(extra), sorted(missing))))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", type=str, required=True,
                    help="实验代号（U/V/W/...），用于日志、DONE 文件名与包目录标签")
    # ── 可扫的旋钮：每个都会自动进"声明列表"，⛔一次只声明一个
    ap.add_argument("--margin", type=float, default=None,
                    help="CosFace 加性余弦间隔（实验 U）。⚠需 28 号已打 --margin 补丁")
    ap.add_argument("--freeze-first-n", type=int, default=None,
                    help="硬冻结底部若干层（实验 V），layer_id<n；0=输入端，1..12=block")
    ap.add_argument("--epochs", type=int, default=None, help="覆盖轮数（实验 W：12→24）")
    ap.add_argument("--drop-path", type=float, default=None, help="覆盖 drop-path 剂量")
    ap.add_argument("--random-erasing", type=float, default=None)
    ap.add_argument("--lr-backbone", type=str, default=None)
    # ── 实验 C（合规体制 v3，2026-09-18）的四个旋钮。⚠它们**一起动**，预注册里必须明写"分不开"
    ap.add_argument("--from-clip", action="store_true",
                    help="删 --resume：从原版 CLIP 权重起手（⛔现有 resume 链每段都带测试集伪标签）")
    ap.add_argument("--no-pseudo", action="store_true",
                    help="删 --pseudo/--pseudo-weight：不把测试图并入训练（十一(二)1）")
    ap.add_argument("--head-init", type=str, default=None,
                    help="换分类头初值 npz（键 W,b，同 head_loss_sce.npz）。实验 C 给 cache\\head_init_text.npz")
    ap.add_argument("--aug-geo", action="store_true",
                    help="训练增强加几何 TrivialAugment（⛔无颜色项），需 28 号支持 --aug-geo")
    ap.add_argument("--distill", type=str, default=None,
                    help="训练集自蒸馏：教师软标签前缀（需 <prefix>_prob.npy + <prefix>_paths.txt），需 28 号支持 --distill")
    ap.add_argument("--distill-alpha", type=float, default=None, help="教师分布权重 α（默认 0.5）")
    ap.add_argument("--distill-temp", type=float, default=None, help="教师分布温度 T（默认 1.0）")
    ap.add_argument("--resume-from", type=str, default=None,
                    help="换 resume 起点（默认 L 的 384 站）。实验 K：接着 L@ep12 继续训")
    ap.add_argument("--text-anchor", type=str, default=None,
                    help="CLIP 文本原型 npz 当固定锚点（实验 K），需 28 号支持 --text-anchor")
    ap.add_argument("--text-lambda", type=float, default=None, help="文本锚点损失权重 λ（默认 0.3）")
    ap.add_argument("--endpoint", type=int, default=None,
                    help="出包用第几轮的 ckpt（默认=最后一轮）。⛔端点法：别出中间点")
    ap.add_argument("--wait-for", type=str, default=None,
                    help="先等这个文件出现（上一发的 _chain_*_DONE.txt）")
    ap.add_argument("--wait-timeout", type=int, default=172800)
    ap.add_argument("--no-smoke", action="store_true",
                    help="跳过起飞前冒烟。⛔默认不跳——10 小时后才发现挂在第 3 分钟太贵")
    ap.add_argument("--skip-train", type=str, default=None,
                    help="跳过训练，直接拿这个【训练目录】出包（断点续链用）")
    ap.add_argument("--dry-run", action="store_true", help="只打印命令与单变量 diff，不动 GPU")
    args = ap.parse_args()

    # ── 组装训练参数 + 声明旋钮
    declared = []
    for knob, key in KNOB2KEY.items():
        v = getattr(args, knob)
        if v is None:
            continue
        declared.append(key)
        flag = "--" + knob.replace("_", "-")
        if flag in TRAIN_ARGS:
            TRAIN_ARGS[TRAIN_ARGS.index(flag) + 1] = str(v)
        else:
            TRAIN_ARGS.extend([flag, str(v)])
    for knob, key in SWITCH2KEY.items():
        if getattr(args, knob):
            declared.append(key)
            TRAIN_ARGS.append("--" + knob.replace("_", "-"))
    if args.resume_from:                     # 换 resume 起点（⭐实验 K：接着 L@ep12）
        TRAIN_ARGS[TRAIN_ARGS.index("--resume") + 1] = args.resume_from
        declared.append("resume")
    for knob, flag in (("distill_alpha", "--distill-alpha"), ("distill_temp", "--distill-temp")):
        v = getattr(args, knob)
        if v is not None:                    # 与 --distill 同属一个旋钮，闸门按字段名单独登记
            TRAIN_ARGS.extend([flag, str(v)])
            declared.append(knob)
    if args.from_clip:                       # 删 --resume <path>
        i = TRAIN_ARGS.index("--resume")
        del TRAIN_ARGS[i:i + 2]
        if "resume" not in declared:
            declared.append("resume")
    if args.no_pseudo:                       # 删 --pseudo <path> 与 --pseudo-weight <v>
        for f in ("--pseudo", "--pseudo-weight"):
            if f in TRAIN_ARGS:
                i = TRAIN_ARGS.index(f)
                del TRAIN_ARGS[i:i + 2]
        declared.append("pseudo")

    print("=" * 70)
    print("65 执行链  实验 %s   口径 ms448_544 + w3cc   对照 %.4f" % (args.tag, REF_SCORE))
    print("=" * 70)
    train_cmd = [PY, os.path.join(ROOT, "scripts", "28_finetune_v2.py")] + TRAIN_ARGS
    print("  训练命令：\n    " + " ".join(train_cmd[1:]) + "\n")
    gate_ok = single_knob_gate(TRAIN_ARGS, declared)
    if not gate_ok and not args.dry_run:
        log("⛔ 单变量闸不通过，拒绝起飞。要么改命令，要么在 TRAIN_ARGS 里改基线（并写台账）")
        sys.exit(2)

    ep_last = int(TRAIN_ARGS[TRAIN_ARGS.index("--epochs") + 1])
    ep_pack = args.endpoint or ep_last
    if args.dry_run:
        log("DRY-RUN 出包端点 = ckpt_ep%02d.pt" % ep_pack)
        log("DRY-RUN 池子视角 = %s" % C3_VIEWS)
        log("DRY-RUN 配方 = c3 + weights %s（双分支）" % W3CC)
        log("DRY-RUN 对照 = %s" % REF_CSV)
        return

    for p in (REF_CSV, L_HISTORY):
        if not os.path.exists(p):
            log("⛔ 缺少基准文件 %s" % p)
            sys.exit(2)

    # ── 0. 排队（⛔单卡串行：并行两边掉速 80%，还可能把 10 小时的训练 OOM 掉）
    if args.wait_for:
        wf = args.wait_for if os.path.isabs(args.wait_for) else os.path.join(ROOT, args.wait_for)
        log("排队等待 %s ..." % wf)
        t0 = time.time()
        while not os.path.exists(wf):
            if time.time() - t0 > args.wait_timeout:
                log("!! 等待超时 %d 秒，放弃" % args.wait_timeout)
                sys.exit(1)
            time.sleep(60)
        log("上一发已完成，GPU 空出")
        time.sleep(20)

    t_chain = time.time()
    train_dir = args.skip_train

    if not train_dir:
        # ── 0.5 起飞前冒烟：⛔真正要验的是"新旋钮进没进训练循环"，自测脚本测不到 AMP 交互
        if not args.no_smoke:
            smoke = list(train_cmd)
            smoke[smoke.index("--epochs") + 1] = "2"
            for f in ("--pseudo", "--pseudo-weight"):      # ⚠不截伪标签的话冒烟要 15 分钟
                if f in smoke:                             # （--no-pseudo 时本来就没有）
                    i = smoke.index(f)
                    del smoke[i:i + 2]
            smoke.append("--smoke")
            if run(smoke, "起飞前冒烟"):
                log("!! 冒烟失败，⛔不起飞（省下 10 小时）")
                sys.exit(1)
            log("✅ 冒烟通过")

        # ── 1. 训练
        before = set(glob.glob(os.path.join(RUNS, "*_v2_448*")))
        if run(train_cmd, "训练 %s（%d 轮）" % (args.tag, ep_last)):
            log("!! 训练失败，链中止")
            sys.exit(1)
        train_dir = newest_new("*_v2_448*", before)
        if not train_dir:
            log("!! 找不到训练目录")
            sys.exit(1)
    log("训练目录 = %s" % train_dir)

    ckpt = os.path.join(train_dir, "ckpt_ep%02d.pt" % ep_pack)
    if not os.path.exists(ckpt):
        log("!! 缺 %s" % ckpt)
        sys.exit(1)

    # ── 2. 建池子：先 544（高分支）后 448（低分支）。⚠换 ckpt 必须重建，池子不能跨 ckpt 复用
    pools = {}
    for res in (544, 448):
        before = set(glob.glob(os.path.join(RUNS, "*_pool%d_ep*" % res)))
        if run([PY, os.path.join(ROOT, "scripts", "58_test_viewpool.py"),
                "--ckpt", ckpt, "--res", str(res), "--views", C3_VIEWS],
               "建 %d 池（10 视角）" % res):
            log("!! 建 %d 池失败" % res)
            sys.exit(1)
        d = newest_new("*_pool%d_ep*" % res, before)
        if not d:
            log("!! 找不到新建的 %d 池" % res)
            sys.exit(1)
        pools[res] = os.path.join(d, "pool.npz")
        log("池子 @%d = %s" % (res, d))

    # ── 3. 组配方出包（纯 numpy，秒级）：c3both_w3cc，与 81.0419 逐项同口径
    tag = "%s_w3cc" % args.tag
    before = set(glob.glob(os.path.join(RUNS, "*_rcp_*")))
    if run([PY, os.path.join(ROOT, "scripts", "59_recipe_pack.py"),
            "--pool", pools[544], "--pool-lo", pools[448],
            "--recipe", "c3", "--weights", W3CC,
            "--recipe-lo", "custom", "--views-lo", C3_VIEWS, "--weights-lo", W3CC,
            "--tag", tag, "--compare", REF_CSV], "组配方出包 %s" % tag):
        log("!! 出包失败")
        sys.exit(1)
    pkg = newest_new("*_rcp_*", before)
    if not pkg:
        log("!! 找不到新出的包")
        sys.exit(1)

    # ── 4. 自己再算一遍分歧（⛔别只信别人打印的数）
    zipf, csvf = os.path.join(pkg, "pred_results.zip"), os.path.join(pkg, "pred_results.csv")
    d, n, se, bar = disagree(csvf, REF_CSV)
    lines = [
        "实验 %s 出包完成  %s" % (args.tag, time.strftime("%Y-%m-%d %H:%M:%S")),
        "旋钮      : %s" % (" ".join(
            ["%s=%s" % (k, getattr(args, k)) for k in KNOB2KEY if getattr(args, k) is not None]
            + [k for k in SWITCH2KEY if getattr(args, k)]
            + (["from_clip"] if args.from_clip else []) + (["no_pseudo"] if args.no_pseudo else [])
        ) or "（无）"),
        "训练目录  : %s" % train_dir,
        "出包端点  : ckpt_ep%02d.pt" % ep_pack,
        "池子 @544 : %s" % os.path.dirname(pools[544]),
        "池子 @448 : %s" % os.path.dirname(pools[448]),
        "成绩包    : %s" % zipf,
        "MD5       : %s" % (md5(zipf) if os.path.exists(zipf) else "MISSING"),
        "",
        "⭐ vs 81.0419 那个包：分歧 %d/%d (%.2f%%)  SE=%.3f  实算误差棒 ±%.3f"
        % (d, n, 100.0 * d / n, se, bar),
        "   判读：|Δ| < %.3f 读『测不出差别』；分歧 < ~300 张就⛔别交（省名额）" % bar,
        "链总耗时  : %.1f 小时" % ((time.time() - t_chain) / 3600),
    ]
    txt = "\n".join(lines) + "\n"
    print("\n" + txt)
    for p in (os.path.join(train_dir, "DONE.txt"),
              os.path.join(RUNS, "_chain_%s_DONE.txt" % args.tag)):
        with open(p, "w", encoding="utf-8") as f:
            f.write(txt)
    log("全部完成 → runs\\_chain_%s_DONE.txt" % args.tag)


if __name__ == "__main__":
    main()
