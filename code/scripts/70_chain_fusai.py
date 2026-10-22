# -*- coding: utf-8 -*-
"""70 · 复赛一条龙执行链（工作区 = 本仓库 code 目录 / python）

准备缓存 → 冒烟 → 训练 → 建 544 池 → 建 448 池 → 组 c3both_w3cc 出包 → 写 DONE。

═══════════════════════════════════════════════════════════════════════
⭐ 为什么另起一个链，不改 65_chain_exp.py
═══════════════════════════════════════════════════════════════════════
65 号身上有三样东西**整个绑死在初赛**，改起来比重写还费事：

① 路径：`D:\\Anaconda\\envs\\dl\\python.exe` + `D:\\aic_data\\AIC\\AIC2026_9_4`
   ——（旧机器环境）。当前环境： `python` + `code 目录`。
② 单变量闸拿**初赛 L@ep12 的 history.json**当尺子，要求本次参数与它只差声明的那几个。
   复赛是**从原版 CLIP 重新起跑**，跟 L 的 diff 会有一大片，闸门必然拒绝起飞；
   而且就算放行，"与初赛 L 的差异"在复赛也没有任何解释意义。
③ 对照基准写死 `REF_SCORE = 81.0419` 和初赛那个 csv —— 复赛第一发**没有对照**，
   它本身就是复赛的起跑线。

⛔ 还有一条**规则问题**：65 的 TRAIN_ARGS 默认带 `--resume runs\\2026...384\\ckpt_best.pt`
   和 `--pseudo cache\\pseudo_labels.npz`，两样都是**初赛产物**。
   规则十一(二)2「复赛期间不得使用初赛的数据」⇒ 本链从原版 CLIP 起、不带任何伪标签，
   这不是可选项，是硬约束。

═══════════════════════════════════════════════════════════════════════
本链的配方（哪些是初赛验过直接搬的、哪些是新的）
═══════════════════════════════════════════════════════════════════════
直接搬（交接文档第 24 节，初赛累计贡献最大的几件）：
    JPEG 退化增强对齐测试域 · Mixup/CutMix · LLRD 0.85 · EMA · 余弦头
    DropPath 0.2 + RandomErasing 0.25 · 448 训练 · ms448_544 + w3cc 的放大 TTA
新的：
    --co-teaching   两网交叉筛小损失样本（28 号新增，见那边的 CT 段注释）
    测试集伪标签     两段式：教师这一发 --make-pseudo → 学生那一发 --pseudo（见下面用法）
                    ⭐教师 = 实际提交的那个 TTA 包的 logits（30 号新增 --from-logits，余弦头口径）
                    ⚠规则十一(二)1 风险项，用户 09-24 拍板要做，复现审核时会被看到
可选（默认不开）：
    WiSE-FT 插值    训练后零 GPU 的权重回拉。⛔先 `62_wise_ft.py --scan` 看位移再决定
删掉的：
    ⛔ 文本塔 / 学名原型（--head-init text / --text-anchor）：用户 09-24 决定不走这条路，
       只用视觉侧。68 号脚本保留在盘上但本链不调用。
    ⛔ 初赛的伪标签文件：那是初赛教师打的，撞规则十一(二)2。复赛伪标签必须由复赛教师重新打。

═══════════════════════════════════════════════════════════════════════
用法（⛔必须 Start-Process 独立启动：终端一关，子进程会被回收）
═══════════════════════════════════════════════════════════════════════
  # ① 先看一眼要跑什么，不动 GPU
  "python" scripts\\70_chain_fusai.py --tag A1 --dry-run

  # ② 准备缓存（约 20~40 分钟，一次性；缓存已在就自动跳过）
  "python" scripts\\70_chain_fusai.py --tag A1 --prep-only

  # ③ 教师 A1：训练 → 出包（第一个可提交的包）→ 用这个包打伪标签
  Start-Process ... -ArgumentList 'scripts\\70_chain_fusai.py --tag A1 --co-teaching --make-pseudo'

  # ④ 学生 A2：吃 A1 的伪标签重新从原版 CLIP 训（可以和 ③ 同时启动，它会排队等 A1 完成）
  Start-Process ... -ArgumentList 'scripts\\70_chain_fusai.py --tag A2 --co-teaching
                                   --pseudo cache\\pseudo_A1.npz --wait-for runs\\_chain_A1_DONE.txt'
  （--compare 不用填：排队结束后自动从 A1 的 DONE 文件里读 A1 主包当对照）

  完整的 Start-Process 写法见 docs\\复赛_起跑清单.md 第 4 节。
"""
import _bootstrap  # noqa: F401  ⭐只为把 stdout 切成 UTF-8：Windows 控制台是 cp936，
                   #              本文件满屏 ⭐⛔，不切会 UnicodeEncodeError 直接中断

import argparse
import csv
import glob
import hashlib
import math
import os
import subprocess
import sys
import time

# ── 本机常量（⚠换机器只改这三行）
PY = sys.executable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs")
CACHE = os.path.join(ROOT, "cache")

# ── 口径常数：⛔一个字都别改，改了出的包就不再与本链历史可比
#   （初赛花三发才买回来的教训：口径一换，之前所有分歧数与误差棒全部作废）
C3_VIEWS = "orig,mild0,mild1,mild2,mild3,mild4,cc085,cc070,cc055,cc040"
W3CC = "cc085:3,cc070:3,cc055:3,cc040:3"

# ── 复赛基线配方。⛔从原版 CLIP 起（无 --resume）、无 --pseudo，见文件头的规则说明。
TRAIN_ARGS = [
    "--resolution", "448",
    "--epochs", "12",          # 用户 09-24 定：与初赛 L 配方一致，⛔不多改一个变量
    "--batch-size", "32",
    "--lr-backbone", "2e-05",
    "--llrd", "0.85",
    "--degrade-p", "0.875",
    "--cosine-head",
    "--aug-scale", "0.3",
    "--num-workers", "2",      # ⛔本机内存墙：4 个会在第一批图上卡死（09-24 冒烟实踩，无报错、GPU 空转）。
                               #   只改读图并行度，⛔不是模型超参；01 号抽特征、28 号评估一直用 2 个都正常
    "--eval-workers", "2",
    "--full-data",
    "--seed", "0",
    "--drop-path", "0.2",
    "--random-erasing", "0.25",
]

# ── 准备阶段：28 号在 :523~:529 无条件读这四个缓存，缺一个就直接崩。
#   前三个是**本地尺子**的零件，第四个是分类头的初值（LP-FT 的 LP）。
#   全部只吃 512 维特征缓存，都是分钟级，⛔别跳。
PREP = [
    ("01_extract_features.py", [], [os.path.join(CACHE, "train_features.npy"),
                                    os.path.join(CACHE, "test_features.npy")],
     "抽 512 维特征（train+test，约 6~10 分钟）"),
    ("04_detect_noise.py", [], [os.path.join(CACHE, "noise_scores.npz")],
     "噪声可疑度打分 → noise_scores.npz"),
    # ⚠05 号只写校准报告 valset_calibration.json（06 号读它），⛔不产 val_weights.npz。
    #   09-24 第一次 --prep-only 时这里登记错了产物，05 跑完 val_weights 仍缺，链却照样报「完成」。
    ("05_calibrate_valset.py", [], [os.path.join(CACHE, "valset_calibration.json")],
     "干净验证子集校准 → valset_calibration.json"),
    ("06_loss_experiments.py", [], [os.path.join(CACHE, "head_loss_sce.npz")],
     "线性探针头（微调的起点头）→ head_loss_sce.npz"),
    ("16_check_leakage.py", [], [os.path.join(CACHE, "leakage_scores.npz")],
     "训练/验证泄漏打分 → leakage_scores.npz"),
    # ⭐val_weights.npz 的真正产出者。吃 16 号的 leakage_scores + 06 号的 head_loss_sce ⇒ 必须排最后。
    #   ⚠它打印的「验收」一节拿的是**初赛**榜分 0.590 当参照，复赛下那段 ✅/⚠/❌ 判语无意义，别读；
    #   存下来的权重本身只依赖「验证集 vs 测试集」的相似度分箱，是数据驱动的，复赛照样成立。
    ("17_calibrate_ruler.py", [], [os.path.join(CACHE, "val_weights.npz")],
     "验证集分布匹配权重 → val_weights.npz"),
]


def log(msg):
    print("[%s] %s" % (time.strftime("%m-%d %H:%M:%S"), msg), flush=True)


def run(cmd, what):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)          # 沙箱/IDE 会污染，必须清
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
    """返回本步骤**新增**的目录（⛔别用"最新目录"，残留会串味）。"""
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
    """配对分歧 + 实算误差棒（判读纪律：误差棒要实算，⛔别拍脑袋）。"""
    A, B = read_pred(a), read_pred(b)
    keys = set(A) & set(B)
    d = sum(1 for k in keys if A[k] != B[k])
    n = max(len(keys), 1)
    se = math.sqrt(max(d, 1)) / n * 100.0
    return d, n, se, 2 * se


def preflight():
    """起飞前的硬检查。⛔每一条都是"崩了才发现要重来 N 小时"的那种。"""
    bad = []
    if not os.path.exists(PY):
        bad.append("解释器不在：%s" % PY)
    if not os.path.isdir(RUNS):
        os.makedirs(RUNS, exist_ok=True)
    clipw = os.path.join(ROOT, "models", "clip", "ViT-B-32.pt")
    if not os.path.exists(clipw):
        bad.append("CLIP 权重不在：%s（跑 00_setup_check.py 会自动下）" % clipw)
    # 磁盘：co-teaching 每轮存两份 337MB 的 ckpt，8 轮 = 5.4GB，再加两个池子
    try:
        import shutil
        free_gb = shutil.disk_usage(ROOT).free / 1e9
        if free_gb < 12:
            bad.append("E 盘只剩 %.1f GB —— 8 轮 co-teaching 要 ~6 GB 存 ckpt + ~1 GB 池子，"
                       "先清 runs\\ 里的初赛中间 ckpt" % free_gb)
        else:
            log("磁盘剩余 %.1f GB ✅" % free_gb)
    except Exception as e:                                  # noqa: BLE001
        log("⚠磁盘检查跳过：%s" % e)
    return bad


TEST_RULER = ("16_check_leakage.py", "17_calibrate_ruler.py")   # 这两个要读测试图特征


def do_prep(dry, no_test_ruler=False):
    """按顺序补齐 28 号要的缓存。已存在的自动跳过（幂等）。
    no_test_ruler：训练不读测试集算出的尺子文件，16/17 号既不补跑也不检查（09-30，完全不碰测试集的训练）。"""
    log("── 准备阶段：检查 28 号依赖的缓存")
    for script, extra, outs, what in PREP:
        if no_test_ruler and script in TEST_RULER:
            log("  ⏭ 不用测试集尺子，跳过：%s" % what)
            continue
        have = all(os.path.exists(o) for o in outs)
        if have:
            log("  ✅ 已有，跳过：%s" % what)
            continue
        if dry:
            log("  DRY-RUN 会跑：%s  (%s)" % (script, what))
            continue
        if run([PY, os.path.join(ROOT, "scripts", script)] + extra, what):
            log("!! %s 失败，链中止" % script)
            log("   ⚠若报 [WinError 1455] 页面文件太小 —— 那是 Windows 虚拟内存不够，"
                "不是代码问题。改法见 docs\\复赛_起跑清单.md 第 1 节。")
            sys.exit(1)
        # ⭐退出码 0 ≠ 产物落盘。⛔必须逐个核对，否则登记错产物时链会假装完成（09-24 实踩）
        miss = [o for o in outs if not os.path.exists(o)]
        if miss:
            log("!! %s 退出码为 0，但没产出 %s —— 链中止" % (script, ", ".join(os.path.basename(m) for m in miss)))
            sys.exit(1)
    # 最后再对一遍 28 号真正要读的清单（与 28 号 :548~:556 的 np.load 逐项对应）
    need = ["head_loss_sce.npz", "val_weights.npz", "leakage_scores.npz", "noise_scores.npz"]
    if no_test_ruler:
        need = [n for n in need if n not in ("val_weights.npz", "leakage_scores.npz")]
    lack = [n for n in need if not os.path.exists(os.path.join(CACHE, n))]
    if lack and not dry:
        log("!! 28 号起飞必读的缓存仍缺：%s" % ", ".join(lack))
        sys.exit(1)
    log("── 准备阶段完成（28 号必读的 %d 个缓存已逐个核对）" % len(need))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", type=str, required=True, help="实验代号，进目录名与 DONE 文件名")
    ap.add_argument("--epochs", type=int, default=None, help="训练轮数（默认 12，与初赛 L 一致）")
    ap.add_argument("--batch-size", type=int, default=None, help="显存不够就降到 16，⛔别动 lr")
    ap.add_argument("--co-teaching", action="store_true",
                    help="⭐开 co-teaching（28 号新增）。两网交叉筛小损失样本，出包仍只用 A 网")
    ap.add_argument("--ct-forget-rate", type=float, default=None, help="丢弃比例 τ（默认 0.15）")
    ap.add_argument("--ct-warmup-epochs", type=float, default=None, help="τ 爬升轮数（默认 1）")
    # ── A3（09-29）：以下默认全关，不带这些参数时与 A1/A2 的训练命令逐字相同
    ap.add_argument("--margin", type=float, default=None, help="CosFace 间隔 m（28 号 --margin），默认不传=0")
    ap.add_argument("--dedup", action="store_true",
                    help="开训前现场跑 76 号生成去重名单 cache\\dup_exclude.txt，并传给 28 号 --exclude-list。"
                         "⭐每次都重新生成，不依赖事先存好的名单（规则五、6：清洗要能由代码直接复现）")
    ap.add_argument("--small-oversample", type=float, default=None, help="小类过采样倍数上限（28 号同名参数）")
    ap.add_argument("--small-frac", type=float, default=None, help="小类口径：图数 < 中位数 × 这个比例（默认 0.5）")
    ap.add_argument("--ct-protect-small", action="store_true", help="co-teaching 筛选时小类一律保留（28 号同名参数）")
    ap.add_argument("--no-foreach", action="store_true",
                    help="AdamW 逐个参数更新，只省显存不改公式（28 号同名参数，09-29 A3 冒烟爆显存后加）")
    ap.add_argument("--no-pin-memory", action="store_true",
                    help="训练读图器不用锁页内存，只影响传输方式（28 号同名参数，09-29 A3 续跑测试内存不足后加）")
    ap.add_argument("--no-test-ruler", action="store_true",
                    help="训练时不读由测试集算出的尺子文件（28 号同名参数），准备阶段也不补跑、不检查 16/17 号。"
                         "⭐完全不碰测试集的训练必须开")
    ap.add_argument("--wise", type=float, default=None,
                    help="⭐训练后顺手做 WiSE-FT 权重插值 θ_α=(1−α)·原版CLIP + α·微调，"
                         "再出一个包。零 GPU 训练成本，但要多建一次池子（~2 小时）。"
                         "⛔先用 `62_wise_ft.py --scan` 看位移再决定值不值，别盲开")
    ap.add_argument("--wise-only", action="store_true",
                    help="配合 --skip-train 用：**只**给插值权重建池出包，跳过主 ckpt。"
                         "⛔没有这个开关的话，--skip-train --wise 会把主包的池子白建一遍（多烧 2 小时）")
    # ── 伪标签（两段式：教师这一发 --make-pseudo 产出，学生那一发 --pseudo 吃进去）
    ap.add_argument("--make-pseudo", action="store_true",
                    help="⭐出包后用**这个包自己的 logits** 当教师，给测试集打伪标签 → "
                         "cache\\pseudo_<tag>.npz。零 GPU，秒级。教师=实际提交的 TTA 包（余弦头口径）")
    ap.add_argument("--pseudo", type=str, default=None,
                    help="学生这一发：把哪个伪标签 npz 并进训练集（如 cache\\pseudo_A1.npz）。"
                         "⛔必须是**复赛教师**产出的；初赛那份撞规则十一(二)2")
    ap.add_argument("--pseudo-weight", type=float, default=0.5,
                    help="伪标签样本权重（初赛 L 用 0.5，⛔别设 1.0：教师自己也有错）")
    ap.add_argument("--pseudo-per-class", type=int, default=40,
                    help="--make-pseudo 时每类最多取几张（初赛 40；复赛 37444/750≈50 张/类，同样≈80%%）")
    ap.add_argument("--endpoint", type=int, default=None, help="出包用第几轮（默认最后一轮）")
    ap.add_argument("--compare", type=str, default=None,
                    help="与哪个 pred_results.csv 算配对分歧（第一发没有对照，留空）")
    ap.add_argument("--wait-for", type=str, default=None, help="先等这个 DONE 文件出现再起飞")
    ap.add_argument("--wait-timeout", type=int, default=172800)
    ap.add_argument("--prep-only", action="store_true", help="只补缓存，不训练")
    ap.add_argument("--no-smoke", action="store_true",
                    help="跳过起飞前冒烟。⛔默认不跳——20 小时后才发现挂在第 3 分钟太贵")
    ap.add_argument("--skip-train", type=str, default=None, help="跳过训练，拿这个训练目录出包")
    ap.add_argument("--resume-train", type=str, default=None,
                    help="⭐断点续跑：训练中途断了（09-24 整机卡死实踩），给出原训练目录，从最后存下的那一轮接着训，"
                         "之后照常建池出包。⛔其余参数必须与原来那一发一模一样（28 号会逐项核对，不一致直接拒绝）")
    ap.add_argument("--dry-run", action="store_true", help="只打印，不动 GPU")
    args = ap.parse_args()

    # ── 组装训练参数
    for knob, flag in (("epochs", "--epochs"), ("batch_size", "--batch-size")):
        v = getattr(args, knob)
        if v is not None:
            TRAIN_ARGS[TRAIN_ARGS.index(flag) + 1] = str(v)
    if args.co_teaching:
        TRAIN_ARGS.append("--co-teaching")
        # ⭐两网 @448 bs32 在本机 8GB 上 OOM（09-24 冒烟实踩）。检查点只换显存、不改超参
        TRAIN_ARGS.append("--grad-ckpt")
        for knob, flag in (("ct_forget_rate", "--ct-forget-rate"),
                           ("ct_warmup_epochs", "--ct-warmup-epochs")):
            v = getattr(args, knob)
            if v is not None:
                TRAIN_ARGS.extend([flag, str(v)])
    if args.margin:
        TRAIN_ARGS.extend(["--margin", str(args.margin)])
    dedup_list = os.path.join(CACHE, "dup_exclude.txt")
    if args.dedup:
        TRAIN_ARGS.extend(["--exclude-list", dedup_list])
    if args.small_oversample:
        TRAIN_ARGS.extend(["--small-oversample", str(args.small_oversample)])
    if args.small_frac is not None:
        TRAIN_ARGS.extend(["--small-frac", str(args.small_frac)])
    if args.ct_protect_small:
        TRAIN_ARGS.append("--ct-protect-small")
    if args.no_foreach:
        TRAIN_ARGS.append("--no-foreach")
    if args.no_pin_memory:
        TRAIN_ARGS.append("--no-pin-memory")
    if args.no_test_ruler:
        TRAIN_ARGS.append("--no-test-ruler")
    pseudo_abs = None
    if args.pseudo:
        pseudo_abs = args.pseudo if os.path.isabs(args.pseudo) else os.path.join(ROOT, args.pseudo)
        if "初赛" in pseudo_abs:
            log("⛔ %s 是初赛产物，复赛不得使用（规则十一(二)2）" % pseudo_abs)
            sys.exit(2)
        TRAIN_ARGS.extend(["--pseudo", pseudo_abs, "--pseudo-weight", str(args.pseudo_weight)])

    ep_last = int(TRAIN_ARGS[TRAIN_ARGS.index("--epochs") + 1])
    ep_pack = args.endpoint or ep_last
    bs = int(TRAIN_ARGS[TRAIN_ARGS.index("--batch-size") + 1])

    print("=" * 70)
    print("70 复赛执行链  实验 %s   口径 ms448_544 + w3cc" % args.tag)
    print("  750 类 / 148,695 训练 / 37,444 测试   ⛔从原版 CLIP 起、无文本塔")
    print("  伪标签：%s" % (("⭐学生 · 并入 %s（权重 %.2f）" % (args.pseudo, args.pseudo_weight))
                          if args.pseudo else
                          ("⭐教师 · 出包后打伪标签 → cache\\pseudo_%s.npz" % args.tag)
                          if args.make_pseudo else "无"))
    if args.pseudo or args.make_pseudo:
        print("  ⚠规则十一(二)1「测试数据仅可用于测试，不得以自监督、无监督等方式参与训练」——"
              "测试集伪标签是用户 09-24 拍板的风险项，复现审核时会被看到")
    print("=" * 70)
    train_cmd = [PY, os.path.join(ROOT, "scripts", "28_finetune_v2.py")] + TRAIN_ARGS
    print("  训练命令：\n    " + " ".join(train_cmd[1:]) + "\n")

    if args.co_teaching:
        print("  ⭐co-teaching 已开：τ=%s，出包端点 ckpt_ep%02d.pt（⛔A 网，不碰 B 网）"
              % (args.ct_forget_rate if args.ct_forget_rate is not None else 0.15, ep_pack))
        print("  ⏱ 成本：GPU 算力 ≈ 2.3×，但**数据加载只做一次**（两网共用同一批解码好的图），")
        print("     而初赛实测瓶颈正是数据加载（GPU 利用率均值 ~69%）⇒ 实际墙钟预计 1.8~2.0×，")
        print("     ⚠这是推算不是实测，第一轮跑完看 [ep01] 那行的分钟数就知道真值了。")

    # ── 0. 排队（⛔单卡串行：并行两边掉速 80%，还可能把长训练 OOM 掉）
    #   ⭐必须排在 preflight / 准备缓存**之前**：两发同时启动时，排队的那一发要是先跑 do_prep，
    #     会和第一发同时抽特征、抢同一块 GPU —— 2026-09-24 从 65 号搬过来时发现的顺序坑。
    if args.wait_for and not args.dry_run:
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
        # ⭐没给 --compare 就自动拿上一发的主包当对照（起飞时上一发的包还不存在，没法手填路径）
        if not args.compare:
            with open(wf, encoding="utf-8") as fd:
                for ln in fd:
                    if ln.startswith("成绩包 ["):
                        z = ln.split(":", 1)[1].strip()
                        c = os.path.join(os.path.dirname(z), "pred_results.csv")
                        if os.path.exists(c):
                            args.compare = c
                            log("自动对照 = %s" % c)
                        break
    if pseudo_abs and not args.dry_run and not os.path.exists(pseudo_abs):
        log("⛔ 伪标签文件不在：%s（教师那一发要带 --make-pseudo）" % pseudo_abs)
        sys.exit(2)

    bad = preflight()
    if bad:
        for x in bad:
            log("⛔ " + x)
        if not args.dry_run:
            sys.exit(2)

    do_prep(args.dry_run, args.no_test_ruler)
    if args.prep_only:
        log("--prep-only：缓存已就绪，停在这里")
        return

    if args.dry_run:
        log("DRY-RUN 出包端点 = ckpt_ep%02d.pt" % ep_pack)
        log("DRY-RUN 池子视角 = %s" % C3_VIEWS)
        log("DRY-RUN 配方     = c3 双分支 + 权重 %s" % W3CC)
        log("DRY-RUN batch    = %d（co-teaching=%s）" % (bs, args.co_teaching))
        if args.wise:
            log("DRY-RUN WiSE-FT  = α %.2f，训练后多出一个包" % args.wise)
        if args.make_pseudo:
            log("DRY-RUN 伪标签   = 出包后 → cache\\pseudo_%s.npz（每类 ≤%d 张）"
                % (args.tag, args.pseudo_per_class))
        return

    t_chain = time.time()
    train_dir = args.skip_train

    if args.dedup and not train_dir:
        cmd76 = [PY, os.path.join(ROOT, "scripts", "76_dup_exclude.py"), "--out", dedup_list]
        if args.small_frac is not None:
            cmd76 += ["--small-frac", str(args.small_frac)]
        if os.path.exists(dedup_list):
            os.remove(dedup_list)                       # ⛔先删旧名单：76 号若半路失败，不能拿上一次的名单糊过去
        if run(cmd76, "重复图按类型处理 → 去重名单") or not os.path.exists(dedup_list):
            log("!! 去重名单没生成，链中止")
            sys.exit(1)

    if not train_dir:
        # ── 0.5 起飞前冒烟：⛔真正要验的是"新旋钮进没进训练循环"，
        #     自测脚本测不到 AMP / DataLoader worker / 显存这些交互
        if not args.no_smoke:
            smoke = list(train_cmd)
            smoke[smoke.index("--epochs") + 1] = "2"
            for f in ("--pseudo", "--pseudo-weight"):      # ⚠不截的话 3 万张伪标签图全进冒烟
                if f in smoke:                             #   （伪标签分支是初赛验过的老代码）
                    i = smoke.index(f)
                    del smoke[i:i + 2]
            smoke.append("--smoke")
            before_s = set(glob.glob(os.path.join(RUNS, "*_v2_448*")))
            if run(smoke, "起飞前冒烟（含 co-teaching 路径）" if args.co_teaching else "起飞前冒烟"):
                log("!! 冒烟失败，⛔不起飞（省下 20 小时）")
                sys.exit(1)
            # ── 续跑测试：从冒烟第 2 轮末的完整状态精确续跑第 3 轮。
            #    ⛔存状态和读状态这两段代码只有在真断电时才会被用到，那时再发现它是坏的就晚了。
            sdir = newest_new("*_v2_448*", before_s)
            if sdir:
                sm2 = list(smoke)
                sm2[sm2.index("--epochs") + 1] = "3"
                sm2 += ["--continue-run", sdir]
                if run(sm2, "续跑测试（冒烟第 2 轮 → 精确续跑第 3 轮）"):
                    log("!! 续跑测试失败，⛔不起飞：真断了会接不上")
                    sys.exit(1)
                for f in glob.glob(os.path.join(sdir, "*.pt")):     # 冒烟权重没有用处，每份 353MB~2.8GB
                    os.remove(f)
            log("✅ 冒烟通过（含续跑测试）")

        # ── 1. 训练
        if args.resume_train:
            rdir = args.resume_train if os.path.isabs(args.resume_train) else os.path.join(ROOT, args.resume_train)
            if not os.path.isdir(rdir):
                log("!! 续跑目录不存在：%s" % rdir)
                sys.exit(1)
            if run(train_cmd + ["--continue-run", rdir], "续跑训练 %s（接着 %s）" % (args.tag, os.path.basename(rdir))):
                log("!! 续跑训练失败，链中止。⭐再次续跑用同一条命令即可（每轮末都存了完整状态）")
                sys.exit(1)
            train_dir = rdir
        else:
            before = set(glob.glob(os.path.join(RUNS, "*_v2_448*")))
            if run(train_cmd, "训练 %s（%d 轮）" % (args.tag, ep_last)):
                log("!! 训练失败，链中止。⭐若是中途断电/卡死，加 --resume-train <训练目录> 续跑")
                sys.exit(1)
            train_dir = newest_new("*_v2_448*", before)
            if not train_dir:
                log("!! 找不到训练目录")
                sys.exit(1)
        # 训完了，续跑状态（~2.8GB）就没用了，删掉给后面的建池和 A2 腾盘
        stf = os.path.join(train_dir, "train_state.pt")
        if os.path.exists(stf):
            os.remove(stf)
    log("训练目录 = %s" % train_dir)

    ckpt = os.path.join(train_dir, "ckpt_ep%02d.pt" % ep_pack)
    if not os.path.exists(ckpt):
        log("!! 缺 %s" % ckpt)
        sys.exit(1)

    # ── 2~3. 建池 + 出包。WiSE-FT 时同一套流程跑两遍（两个 ckpt 各自一套池子）。
    #     ⚠池子**不能跨 ckpt 复用**：池子里存的是该 ckpt 编码出来的特征。
    if args.wise_only and not args.wise:
        log("⛔ --wise-only 必须配 --wise <α>")
        sys.exit(2)
    targets = [] if args.wise_only else [(ckpt, args.tag)]
    if args.wise:
        wout = os.path.join(train_dir, "ckpt_wise%03d.pt" % round(args.wise * 100))
        if run([PY, os.path.join(ROOT, "scripts", "62_wise_ft.py"),
                "--ckpt", ckpt, "--alpha", str(args.wise), "--out", wout],
               "WiSE-FT 插值 α=%.2f（零 GPU）" % args.wise):
            log("!! WiSE-FT 失败，跳过它，继续主线")
        elif os.path.exists(wout):
            targets.append((wout, "%s_wise%03d" % (args.tag, round(args.wise * 100))))

    made = []
    for ck, tg in targets:
        pools = {}
        ok = True
        for res in (544, 448):
            before = set(glob.glob(os.path.join(RUNS, "*_pool%d_ep*" % res)))
            if run([PY, os.path.join(ROOT, "scripts", "58_test_viewpool.py"),
                    "--ckpt", ck, "--res", str(res), "--views", C3_VIEWS,
                    "--num-workers", "2"],        # ⛔58 号默认 4，本机内存墙下会炸（同训练那次）
                   "[%s] 建 %d 池（10 视角）" % (tg, res)):
                log("!! 建 %d 池失败" % res)
                ok = False
                break
            d = newest_new("*_pool%d_ep*" % res, before)
            if not d:
                log("!! 找不到新建的 %d 池" % res)
                ok = False
                break
            pools[res] = os.path.join(d, "pool.npz")
            log("[%s] 池子 @%d = %s" % (tg, res, d))
        if not ok:
            continue

        before = set(glob.glob(os.path.join(RUNS, "*_rcp_*")))
        tmp_logits = os.path.join(RUNS, "_logits_%s.npy" % tg)
        cmd = [PY, os.path.join(ROOT, "scripts", "59_recipe_pack.py"),
               "--pool", pools[544], "--pool-lo", pools[448],
               "--recipe", "c3", "--weights", W3CC,
               "--recipe-lo", "custom", "--views-lo", C3_VIEWS, "--weights-lo", W3CC,
               "--tag", "%s_w3cc" % tg, "--save-logits", tmp_logits]
        if args.compare:
            cmd += ["--compare", args.compare]
        if run(cmd, "[%s] 组配方出包" % tg):
            log("!! 出包失败")
            continue
        pkg = newest_new("*_rcp_*", before)
        if pkg:
            if os.path.exists(tmp_logits):
                os.replace(tmp_logits, os.path.join(pkg, "test_logits.npy"))
            made.append((tg, pkg))

    if not made:
        log("!! 一个包都没出来")
        sys.exit(1)

    pseudo_out = None
    if args.make_pseudo:
        main_pkg = next((p for t, p in made if t == args.tag), None)
        lg_path = os.path.join(main_pkg, "test_logits.npy") if main_pkg else None
        if not lg_path or not os.path.exists(lg_path):
            log("!! 找不到主包的 test_logits.npy，伪标签没打")
        else:
            pseudo_out = os.path.join(CACHE, "pseudo_%s.npz" % args.tag)
            if run([PY, os.path.join(ROOT, "scripts", "30_pseudo_label.py"),
                    "--from-logits", lg_path, "--out", pseudo_out,
                    "--per-class", str(args.pseudo_per_class),
                    "--weight", str(args.pseudo_weight)],
                   "打伪标签（教师 = %s 主包）" % args.tag):
                log("!! 打伪标签失败")
                pseudo_out = None

    # ── 4. 自己再算一遍（⛔别只信别人打印的数）
    lines = ["复赛实验 %s 出包完成  %s" % (args.tag, time.strftime("%Y-%m-%d %H:%M:%S")),
             "配方      : 448训 / ms448_544 + w3cc / 余弦头 / dp0.20 / re0.25 / 退化0.875",
             "旋钮      : %s" % (("co-teaching τ=%s 爬升%s轮"
                                  % (args.ct_forget_rate if args.ct_forget_rate is not None else 0.15,
                                     args.ct_warmup_epochs if args.ct_warmup_epochs is not None else 1.0))
                                 if args.co_teaching else "（基线，无 co-teaching）")
                               + ("  CosFace m=%s" % args.margin if args.margin else "")
                               + ("  去重名单" if args.dedup else "")
                               + ("  小类过采样≤×%s" % args.small_oversample if args.small_oversample else "")
                               + ("  小类免筛" if args.ct_protect_small else "")
                               + ("  不读测试集尺子" if args.no_test_ruler else ""),
             "伪标签    : %s" % (("学生 · %s 权重 %.2f" % (args.pseudo, args.pseudo_weight))
                                  if args.pseudo else "无（本发不吃伪标签）"),
             "训练目录  : %s" % train_dir,
             "出包端点  : ckpt_ep%02d.pt" % ep_pack,
             ""]
    if args.make_pseudo:
        lines += ["⭐教师产出伪标签 : %s" % (pseudo_out or "!! 失败，见日志"),
                  "   学生那一发：--pseudo %s" % (os.path.relpath(pseudo_out, ROOT) if pseudo_out else "?"),
                  ""]
    for tg, pkg in made:
        zipf = os.path.join(pkg, "pred_results.zip")
        csvf = os.path.join(pkg, "pred_results.csv")
        lines.append("成绩包 [%s] : %s" % (tg, zipf))
        lines.append("MD5         : %s" % (md5(zipf) if os.path.exists(zipf) else "MISSING"))
        if args.compare and os.path.exists(csvf):
            d, n, se, bar = disagree(csvf, args.compare)
            lines.append("  vs 对照：分歧 %d/%d (%.2f%%)  SE=%.3f  实算误差棒 ±%.3f"
                         % (d, n, 100.0 * d / n, se, bar))
            lines.append("  判读：|Δ榜分| < %.3f 读『测不出差别』；分歧 < ~300 张就⛔别交（省名额）" % bar)
    if len(made) > 1:
        c0, c1 = [os.path.join(p, "pred_results.csv") for _, p in made[:2]]
        d, n, se, bar = disagree(c0, c1)
        lines += ["", "⭐ 主包 vs WiSE 包：分歧 %d/%d (%.2f%%)  误差棒 ±%.3f"
                      % (d, n, 100.0 * d / n, bar),
                  "   ⛔分歧太小就别占提交名额 —— 两个包实际是同一份答案"]
    lines += ["", "⚠提交前自查：文件名必须是 pred_results.csv、类别编号四位补零、"
                  "图片文件名与测试集逐字一致（大小写+扩展名）",
              "链总耗时  : %.1f 小时" % ((time.time() - t_chain) / 3600)]

    txt = "\n".join(lines) + "\n"
    print("\n" + txt)
    for p in (os.path.join(train_dir, "DONE.txt"),
              os.path.join(RUNS, "_chain_%s_DONE.txt" % args.tag)):
        with open(p, "w", encoding="utf-8") as f:
            f.write(txt)
    log("全部完成 → runs\\_chain_%s_DONE.txt" % args.tag)


if __name__ == "__main__":
    main()
