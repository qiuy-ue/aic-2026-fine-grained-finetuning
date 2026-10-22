# -*- coding: utf-8 -*-
"""G 实验：D+E+F 三模型 softmax 概率平均软集成（纯后处理，复用已出包特征）。
输出：runs/G_ens_deft/pred_results.{csv,zip} + 分歧统计。用法：python 28_ens_softmax.py
提交格式与 26_predict_finetuned 完全一致：image_id=测试文件名(含扩展名), class_name=类名。"""
import os, json, sys, importlib.util, numpy as np, zipfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs import paths as P


def _load_sibling(fname, modname):
    """与 26_predict_finetuned 相同的同级脚本加载方式（保持测试集排序一致）。"""
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(modname, os.path.join(here, fname))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


_ex = _load_sibling("01_extract_features.py", "extract01")

RUNS = r"D:\aic_data\AIC\AIC2026_9_4\runs"
MODELS = {
    "D": os.path.join(RUNS, "20260909_135924_ft512_ep2_tta", "test_feat_head.npz"),
    "E": os.path.join(RUNS, "20260910_190506_ft512_ep2_tta", "test_feat_head.npz"),
    "F": os.path.join(RUNS, "20260910_213030_ft512_ep2_tta", "test_feat_head.npz"),
}
OUT_DIR = os.path.join(RUNS, "G_ens_deft")
os.makedirs(OUT_DIR, exist_ok=True)

cn = json.load(open(P.F_MANIFEST, encoding="utf-8"))["class_names"]
paths = _ex.list_test_files()
assert len(paths) == 24967, f"测试集 {len(paths)} != 24967"


def softmax(x):
    x = x - x.max(1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(1, keepdims=True)


feats = {}
for k, p in MODELS.items():
    z = np.load(p)
    F, W, b = z["F"], z["W"], z["b"]
    assert len(F) == 24967, f"{k}: 特征行数 {len(F)} != 24967"
    feats[k] = softmax(F @ W.T + b)
    print(f"{k}: 特征 {F.shape} -> 概率 {feats[k].shape}")

probs = np.mean(list(feats.values()), axis=0)
ens = probs.argmax(1)

for k, Pk in feats.items():
    single = Pk.argmax(1)
    diff = int((ens != single).sum())
    print(f"ENS vs {k} 分歧: {diff}/24967 ({diff/24967*100:.1f}%)")

# 提交 CSV/ZIP（格式与 26 write_submission 一致）
csv_path = os.path.join(OUT_DIR, P.SUBMISSION_FILENAME)
with open(csv_path, "w", encoding="utf-8", newline="\n") as f:
    f.write("".join("%s,%s\n" % (os.path.basename(p), cn[i])
                    for p, i in zip(paths, ens)))
zip_path = os.path.join(OUT_DIR, P.SUBMISSION_FILENAME.replace(".csv", ".zip"))
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(csv_path, arcname=P.SUBMISSION_FILENAME)
u, c = np.unique(ens, return_counts=True)
print(f"\n输出 ZIP -> {zip_path}")
print(f"覆盖 {len(u)}/500 类  每类 min={c.min()} max={c.max()} 均值={c.mean():.1f}")
print(f"CSV 行数 = {sum(1 for _ in open(csv_path))}")