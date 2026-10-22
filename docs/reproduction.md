# 复现指南

## 0. 本仓库包含 / 不包含

| | 内容 |
|---|---|
| ✅ 包含 | 训练与推理代码（`code/`）、精确训练命令、推理复现脚本、提交自检脚本、训练侧缓存（`code/cache/`） |
| ❌ 不包含 | 模型权重（`ckpt_ep12`，约 353MB）、赛题数据（train/test 图片）、提交预测文件、组委会参考包、内部服务器与账号信息 |

## 1. 环境

- Python 3.12 / PyTorch 2.x（CUDA）/ timm / albumentations / pandas / ftfy / regex
- `pip install -r code/requirements.txt`
- CLIP ViT-B/32 官方权重（timm 名：`vit_base_patch32_clip_224.openai`）首次运行需联网下载或自行放入 `code/models/clip/`

## 2. 数据准备

```bash
export AIC_DATA_ROOT=/path/to/复赛数据集    # 该目录下需有 train/ 与 test/
```

`code/configs/paths.py` 会自动探测（环境变量 → 仓库内 `./复赛数据集` → 上级目录），不硬编码任何机器路径。

## 3. 推理复现（提交包形态）

```bash
export AIC_CKPT=/path/to/ckpt_ep12.pt        # 训练产出的检查点（本仓库不含权重）
bash reproduce.sh
```

内部流程：`58_test_viewpool.py --res 640` → `--res 544` → `--res 448`（各 10 视角池）→ `59_recipe_pack.py --pool-mid`（三池等权聚合）→ `74_balance_pack.py`（Sinkhorn T=1 软配额）→ 提交包形态（37,444 行 / 无表头 / 四位类号）。

## 4. 训练精确复现（审计用）

```bash
export AIC_DATA_ROOT=/path/to/复赛数据集
bash train_exact.sh
```

- 放大配方参数（640px / 30ep / bs128）全部固化；基座配方（448 / 12ep）以注释保存在脚本末尾；
- 训练侧依赖 `code/cache/{dup_exclude.txt, head_loss_sce.npz, noise_scores.npz}`（均由训练集生成，已内置）；
- **GPU 非确定性**：训练结果统计上同分布、但非逐位相同；推理结果的复现以第 3 节的权重推理为准。

## 5. 提交包自检

```bash
python verify_submission.py pred-results.zip
```

7 项结构检查：zip 必须为**裸单成员 `pred_results.csv`**（不能有文件夹包裹）、37,444 行、四位数标签、范围 0000–0749、无表头无空行、按文件名排序、文件名与测试集一致。

## 6. 硬件参考

放大配方的训练量约为基座的 5–6×（640px×30ep vs 448px×12ep）：A100 约 15–20 小时、RTX 4090 D 约 1 天+；三池特征提取合计约 2 小时量级（@4090D）。A100 40GB / 4090D 24GB 均可完成。

## 7. 已知坑

见 [known-pitfalls.md](known-pitfalls.md) —— 13 条踩坑记录（几何错配、`/dev/shm`、伪装 JPG、mixup 实现、fold-all 权重、分辨率迁移、manifest 冻结等）。

## 8. 基座配方（历史）

本仓库主配方为放大版；448 / 12ep 的基座配方（已验证 78.1% 本地一致率）命令以注释形式保留在 `train_exact.sh` 末尾，详细研究过程见 `docs/research-log.md`。
