# -*- coding: utf-8 -*-
"""
只读体检脚本：确认环境和数据都对得上，再动手跑后面的流程。
本脚本不写任何文件、不改任何配置，随时可以重复跑。

用法:  python scripts/00_setup_check.py
"""
import _bootstrap  # noqa: F401  (必须第一个 import)
from _bootstrap import banner

import importlib
import os
import platform
import sys

from configs import paths as P

FAILURES = []
WARNINGS = []


def check(ok, label, detail="", warn_only=False):
    """统一的检查项打印。ok=True 打勾，否则记进失败/警告清单。"""
    mark = "[ OK ]" if ok else ("[WARN]" if warn_only else "[FAIL]")
    print("%s %-38s %s" % (mark, label, detail))
    if not ok:
        (WARNINGS if warn_only else FAILURES).append("%s  %s" % (label, detail))
    return ok


# ====================================================================== 解释器
banner("1. Python 解释器")
print("  exe      :", sys.executable)
print("  version  :", sys.version.replace("\n", " "))
print("  platform :", platform.platform())
check(sys.version_info >= (3, 8), "Python >= 3.8", "当前 %d.%d" % sys.version_info[:2])
# 记录一个已知限制，不算失败：3.9 会让部分新库静默降级
if sys.version_info[:2] == (3, 9):
    print("  注: Python 3.9 下 transformers/peft/accelerate 等会静默装成老版本；")
    print("      当前基线只依赖 torch/clip/sklearn，不受影响。")


# ====================================================================== 依赖包
banner("2. 依赖包")
WANT = ["torch", "torchvision", "numpy", "PIL", "sklearn", "tqdm",
        "clip", "ftfy", "regex", "joblib"]
for name in WANT:
    try:
        m = importlib.import_module(name)
        v = getattr(m, "__version__", None) or getattr(m, "VERSION", "(无版本号)")
        check(True, name, str(v))
    except Exception as e:
        check(False, name, "缺失 (%s)" % type(e).__name__)


# ====================================================================== GPU
banner("3. GPU / CUDA")
try:
    import torch
    print("  torch      :", torch.__version__, " built-cuda:", torch.version.cuda)
    avail = torch.cuda.is_available()
    check(avail, "CUDA 可用", "" if avail else "只能跑 CPU，特征提取会非常慢")
    if avail:
        p = torch.cuda.get_device_properties(0)
        vram = p.total_memory / 1024 ** 3
        print("  device     : %s  (算力 %d.%d)" % (p.name, p.major, p.minor))
        # ⚠任务管理器里的"共享GPU内存"是从系统内存借的，不是真显存，别按那个数字规划
        check(True, "专用显存(真 VRAM)", "%.1f GB" % vram)
        if vram < 6:
            check(False, "显存充裕度", "%.1f GB 偏小，batch 要调低" % vram, warn_only=True)
except Exception as e:
    check(False, "torch 导入", repr(e))


# ====================================================================== 数据集
banner("4. 赛题数据（%s）" % P.STAGE)
ok_train = check(os.path.isdir(P.TRAIN_DIR), "train 目录存在", P.TRAIN_DIR)
ok_test = check(os.path.isdir(P.TEST_DIR), "test 目录存在", P.TEST_DIR)

if ok_train:
    class_dirs = sorted(d for d in os.listdir(P.TRAIN_DIR)
                        if os.path.isdir(os.path.join(P.TRAIN_DIR, d)))
    check(len(class_dirs) == P.EXPECT_NUM_CLASSES, "类别数",
          "实测 %d / 官方 %d" % (len(class_dirs), P.EXPECT_NUM_CLASSES))
    # 类别文件夹名应当是四位数字，后面 03 出提交文件时要原样用作类别编号
    bad = [d for d in class_dirs if not (len(d) == 4 and d.isdigit())]
    check(not bad, "类别名均为四位数字", "异常项: %s" % bad[:5] if bad else "0000 ~ %s" % class_dirs[-1])

    n_train, per_class = 0, []
    for d in class_dirs:
        n = len(os.listdir(os.path.join(P.TRAIN_DIR, d)))
        per_class.append(n)
        n_train += n
    check(n_train == P.EXPECT_NUM_TRAIN, "训练图片总数",
          "实测 %d / 官方 %d" % (n_train, P.EXPECT_NUM_TRAIN))
    if per_class:
        s = sorted(per_class)
        lo, hi, K = s[0], s[-1], len(s)
        p05, p50, p95 = s[K // 20], s[K // 2], s[int(K * 0.95)]
        print("  每类样本数 : min=%d  p05=%d  中位=%d  p95=%d  max=%d  均值=%.1f" %
              (lo, p05, p50, p95, hi, n_train / K))
        # ⚠max/min 会被单个极小类带偏（复赛有一个 5 张的类，比值一下变成 49.6），
        #   真正反映整体是不是长尾的是 p95/p05 —— 复赛实测 1.61，基本均衡。
        print("  长尾程度   : p95/p05=%.2f  max/min=%.2f  (<50张的类 %d 个，<100张 %d 个)" %
              (p95 / max(p05, 1), hi / max(lo, 1),
               sum(1 for n in s if n < 50), sum(1 for n in s if n < 100)))
        check(p95 / max(p05, 1) < 3.0, "类别分布接近均衡",
              "p95/p05=%.2f；若 >3 说明真长尾，logit adjustment 那一发才值得排前面"
              % (p95 / max(p05, 1)), warn_only=True)

if ok_test:
    n_test = len(os.listdir(P.TEST_DIR))
    check(n_test == P.EXPECT_NUM_TEST, "测试图片总数",
          "实测 %d / 官方 %d" % (n_test, P.EXPECT_NUM_TEST))


# ====================================================================== CLIP 权重
banner("5. CLIP ViT-B/32 权重")
print("  下载目录 :", P.CLIP_DOWNLOAD_ROOT, "(按 F 盘铁律，不放 C 盘缓存)")
if os.path.isdir(P.CLIP_DOWNLOAD_ROOT):
    files = os.listdir(P.CLIP_DOWNLOAD_ROOT)
    for f in files:
        size = os.path.getsize(os.path.join(P.CLIP_DOWNLOAD_ROOT, f)) / 1024 ** 2
        print("    %-50s %.1f MB" % (f, size))
    check(bool(files), "权重已下载", "%d 个文件" % len(files),
          warn_only=True)
else:
    check(False, "权重已下载", "目录不存在，首次跑 01 会自动下载(约338MB)", warn_only=True)
print("  ⚠赛题硬约束：骨干必须是 CLIP ViT-B/32，且仅限 OpenAI 官方公开权重。")


# ====================================================================== 产物目录
banner("6. 产物目录（本脚本只读，不创建）")
for label, d in [("cache 特征缓存", P.CACHE_DIR), ("runs 训练产物", P.RUNS_DIR)]:
    exists = os.path.isdir(d)
    print("  %-14s %-55s %s" % (label, d, "已存在" if exists else "尚未创建(01/02 会自动建)"))


# ====================================================================== 汇总
banner("体检结论")
if FAILURES:
    print("❌ %d 项未通过，先解决再往下跑：" % len(FAILURES))
    for x in FAILURES:
        print("   -", x)
else:
    print("✅ 关键项全部通过。")
if WARNINGS:
    print("\n⚠ %d 项提醒（不阻塞）：" % len(WARNINGS))
    for x in WARNINGS:
        print("   -", x)
print("\n下一步: python scripts/01_extract_features.py")
sys.exit(1 if FAILURES else 0)
