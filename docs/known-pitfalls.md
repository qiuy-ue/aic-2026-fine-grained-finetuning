# 已知坑与规避（踩过的坑，逐条归档）

> 全部来自比赛实操。多数坑都有对应的门控或检查单，写下来是为了不再踩第二次。

1. **几何错配（−8pp 级，最大的一次事故）**
   训练用 RRC（无填充），验证 / 提交却用 letterbox 灰边画布 —— 模型从未见过灰边。线上 61.89% vs 离线 73.43%，缺口 −11.5pp。
   规避：**训练增强 ↔ 验证几何 ↔ 提交管线三端几何一致性**写进出包前检查单；`valid_geometry` 开关保留，用于复现历史数字。

2. **`/dev/shm` 撑爆**
   未设上限的等比缩放 + 多 worker × prefetch 会瞬时打满共享内存。
   规避：`torch.multiprocessing.set_sharing_strategy("file_system")` + 缩放长边封顶（`LONG_CAP=1280`）+ 降 batch/worker。

3. **伪装 JPG 杀死训练**
   扩展名是 .jpg、实际内嵌 TIFF 头（EXIF 段非法），header-only 探测拦不住。
   规避：manifest 阶段做**完整解码演练**（与训练完全相同的 EXIF → convert → thumbnail 路径）；Dataset 内加有界容错回退，单坏文件不得杀死 worker。

4. **mixup `scatter_add_` 类型错误**
   第三参不接受 Python float。规避：用 `torch.full((n,1), 1.0-lam)` 张量形式；soft target 下 train-top1 按 dominant class 计数。

5. **fold-all 模式 checkpoint 存错权重**
   曾保存原始权重而非 EMA。规避：`ema.module if ema is not None else model`，重启训练前 grep 确认。

6. **448 分辨率训练严重过拟合**
   train 89% / valid 59%。规避：提分辨率必须同步加强正则，且要在修正后的 CenterCrop 口径下重新门控。

7. **跨分辨率推理崩掉（pos_embed 插值）**
   320 权重直接 448 推理 −4pp。规避：直接否决跨分辨率推理路线，训练/推理分辨率保持一致。

8. **manifest 冻结原则**
   不因单个坏文件重建 manifest —— 折划分漂移的代价远大于一张图的收益；148,694 行为冻结版本。

9. **验证指标方差陷阱**
   tail 桶只有 36 个验证样本，任何 "tail +x pp" 不作门控依据；看 middle（479 样本）与 macro。

10. **后台训练进程脱离不可靠**
    SSH 会话结束后内联 nohup 偶发未真正脱离。规避：改用「独立启动脚本 + 独立执行」模式，链上每步可单独重跑。

11. **打包脚本的分辨率断言门会拦新配置**
    512 分辨率曾被拦。规避：放宽门前先核对训练/评测/打包三端分辨率一致。

12. **补丁插入位置不当引用未初始化变量**
    曾触发 `UnboundLocalError`。规避：补丁后先 `python -m py_compile` / 干跑一次再上服务器。

13. **大文件传输超时**
    改流式管道写入替代大文件直传；传输后一律 SHA256 比对本地与远端。
