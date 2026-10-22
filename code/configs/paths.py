# -*- coding: utf-8 -*-
"""
全工程共用的路径与常量。其他脚本一律 from configs.paths import ... ，
不要在各自脚本里硬写路径，避免改数据位置时漏改。
"""
import os

# ---------------------------------------------------------------- 工程根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------- 赛题数据
# ⛔赛题十一(二)2：每个阶段只能用本阶段的数据，复赛期间不得使用初赛的数据。
#   换阶段时只改下面这一段；初赛路径已彻底移除，不留注释，免得被顺手改回去。
STAGE = "复赛"

# 官方复赛数据集：train/ 下 750 个类别文件夹(0000~0749)，test/ 为扁平的无标签图片。
# ⚠解压后是双层同名目录，DATA_ROOT 要指到里面那层才有 train/ 和 test/。
# ---- auto-adapt data root (复现友好：按序探测，找不到时用环境变量覆盖) ----
import os as _os
_candidates = [
    _os.environ.get("AIC_DATA_ROOT", ""),
    _os.path.join(PROJECT_ROOT, "复赛数据集"),
    _os.path.join(_os.path.dirname(PROJECT_ROOT), "复赛数据集"),
]
DATA_ROOT = next((c for c in _candidates if c and _os.path.isdir(_os.path.join(c, "train"))), _candidates[3])
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")

# 2026-09-22 在现场逐目录点数得到，用于体检脚本核对数据是否完整/解压是否漏文件。
# （交接文档写的 1500 类 / 297282 / 74896 是错的，正好差一倍，别照抄。）
EXPECT_NUM_CLASSES = 750
EXPECT_NUM_TRAIN = 148695
EXPECT_NUM_TEST = 37444

# ---------------------------------------------------------------- 产物目录
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")     # 512 维特征缓存
RUNS_DIR = os.path.join(PROJECT_ROOT, "runs")       # 每次训练的模型与指标
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")   # CLIP 权重下载位置(不放C盘)
CLIP_DOWNLOAD_ROOT = os.path.join(MODELS_DIR, "clip")

# 特征缓存文件名(01_extract_features.py 产出，02/03 消费)
F_TRAIN_FEAT = os.path.join(CACHE_DIR, "train_features.npy")   # (N,512) float32
F_TRAIN_LABEL = os.path.join(CACHE_DIR, "train_labels.npy")    # (N,)    int16
F_TRAIN_PATH = os.path.join(CACHE_DIR, "train_paths.txt")      # N 行，与上面同序
F_TEST_FEAT = os.path.join(CACHE_DIR, "test_features.npy")     # (M,512) float32
F_TEST_PATH = os.path.join(CACHE_DIR, "test_paths.txt")        # M 行，与上面同序
F_MANIFEST = os.path.join(CACHE_DIR, "manifest.json")          # 元信息+一致性校验

# ---------------------------------------------------------------- 模型约束
# ⚠赛题硬约束：必须严格使用 CLIP ViT-B/32，且仅限 OpenAI 官方公开的预训练权重。
# 不得替换为其他视觉基础模型或更大模型，不得多模型集成。
CLIP_MODEL_NAME = "ViT-B/32"
FEATURE_DIM = 512
# CLIP ViT-B/32 图像编码器参数量(2026-09-04 从真实权重 sum(p.numel()) 实测)
CLIP_VISUAL_PARAMS = 87_849_216

# ---------------------------------------------------------------- 提交格式
# 官方原文(十三、提交要求)：CSV 每行两字段 = 图片文件名, 类别编号(四位数字前补0)
# 文件必须命名为 pred_results.csv，再压成 zip 提交。示例：xxxxxxxx.jpg, 0001
SUBMISSION_FILENAME = "pred_results.csv"


def ensure_dirs():
    """建好所有产物目录(幂等)。"""
    for d in (CACHE_DIR, RUNS_DIR, MODELS_DIR, CLIP_DOWNLOAD_ROOT):
        os.makedirs(d, exist_ok=True)
