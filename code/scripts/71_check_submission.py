# -*- coding: utf-8 -*-
"""71 · 提交前体检：pred_results.zip 交上去之前，逐条核对官方格式。⭐只用标准库，零 GPU、秒级。

═══════════════════════════════════════════════════════════════════════
为什么要单独一个脚本（⛔别只信出包脚本自己打印的数）
═══════════════════════════════════════════════════════════════════════
格式错一次，这次提交就作废，而复赛的提交名额与截止日期都不宽裕。59 号出包时会打印覆盖率，
但它是**写包的那一方**，自己验自己有盲区：它不会去数 zip 里有没有多余文件、首行有没有 BOM、
文件名大小写与测试集是否逐字一致。本脚本站在**评测方**的角度，只看交上去的那个 zip。

官方要求（赛题十三(一)、(二)）：
  · CSV 每行两个字段：图片文件名, 类别编号；类别编号四位数字，不足补 0
  · 文件名与测试集**完全一致**（大小写、扩展名）
  · 文件名必须是 pred_results.csv，压成一个 zip 提交

⭐已被评测系统接受过的实际写法（初赛 81.0419 那个包，runs\\20260917_090512_...）：
    00012f3f8db94b50bdd15611c1bb699b.jpg,0128
  逗号后**无空格**、无表头、LF 换行。本脚本对「逗号后有空格」只警告不判错（官方示例本身带空格）。

用法：
  python scripts\\71_check_submission.py runs\\<出包目录>                 # 目录或 zip 都行
  python scripts\\71_check_submission.py runs\\<A2包> --compare runs\\<A1包>\\pred_results.csv
退出码：0 = 可以交；1 = ⛔有硬错误，别交
"""
import argparse
import csv
import hashlib
import io
import math
import os
import re
import sys
import zipfile
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

try:                                   # Windows 控制台是 cp936，满屏 ⭐⛔ 会 UnicodeEncodeError
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def _local(p):
    """configs.paths 里写的是 Windows 路径；在 WSL 里跑时自动换成 /mnt/<盘符>/...，方便两边都能用。"""
    if os.path.exists(p):
        return p
    m = re.match(r"^([A-Za-z]):[\\/](.*)$", p)
    if m:
        q = "/mnt/%s/%s" % (m.group(1).lower(), m.group(2).replace("\\", "/"))
        if os.path.exists(q):
            return q
    return p


def load_rows_from_zip(zpath, errs, warns):
    with zipfile.ZipFile(zpath) as z:
        names = [i.filename for i in z.infolist() if not i.is_dir()]
        if names != ["pred_results.csv"]:
            if "pred_results.csv" in names:
                warns.append("zip 里除了 pred_results.csv 还有别的文件：%s" % [n for n in names if n != "pred_results.csv"])
            else:
                errs.append("zip 里没有位于根目录的 pred_results.csv（实际内容：%s）。"
                            "⚠常见错法：把整个文件夹压进去了" % names)
                return None
        raw = z.read("pred_results.csv")
    return raw


def parse(raw, errs, warns):
    if raw.startswith(b"\xef\xbb\xbf"):
        errs.append("文件以 UTF-8 BOM 开头 —— 第一行的文件名会变成 '\\ufeffxxx.jpg'，评测方匹配不上")
        raw = raw[3:]
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError:
        errs.append("文件里有非 ASCII 字符（测试集文件名都是十六进制 + .jpg，不该出现）")
        text = raw.decode("utf-8", "replace")
    if "\r\n" in text:
        warns.append("换行是 CRLF（Windows 风格）。初赛被接受的包是 LF；一般无害，但与已验证写法不同")
    rows = []
    spaced = 0
    for ln, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split(",")
        if len(parts) != 2:
            errs.append("第 %d 行不是恰好 2 个字段：%r" % (ln, line[:80]))
            if len(errs) > 20:
                break
            continue
        name, lab = parts[0], parts[1]
        if lab.startswith(" "):
            spaced += 1
        rows.append((ln, name.strip(), lab.strip(), name != name.strip()))
    if spaced:
        warns.append("%d 行的逗号后带空格（官方示例就带空格，一般无害；初赛被接受的包不带）" % spaced)
    return rows


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest().upper()


def read_pred(path):
    m = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.reader(f):
            if len(r) >= 2:
                m[r[0].strip()] = r[-1].strip()
    return m


def main():
    try:
        from configs import paths as P
        test_dir_default, k_default = P.TEST_DIR, P.EXPECT_NUM_CLASSES
    except Exception:                                        # noqa: BLE001
        test_dir_default, k_default = None, 750
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="出包目录，或 pred_results.zip 本身")
    ap.add_argument("--test-dir", default=test_dir_default, help="测试集目录（默认读 configs/paths.py）")
    ap.add_argument("--num-classes", type=int, default=k_default)
    ap.add_argument("--compare", default=None, help="另一个 pred_results.csv，算配对分歧与误差棒")
    args = ap.parse_args()

    tgt = _local(args.target)
    zpath = os.path.join(tgt, "pred_results.zip") if os.path.isdir(tgt) else tgt
    print("=" * 70)
    print("71 提交前体检  %s" % zpath)
    print("=" * 70)
    errs, warns = [], []
    if not os.path.exists(zpath):
        print("⛔ 找不到 %s" % zpath)
        sys.exit(1)
    if os.path.basename(zpath).lower() != "pred_results.zip":
        warns.append("zip 文件名是 %s（官方只规定 csv 名，zip 名不限；照惯例叫 pred_results.zip）"
                     % os.path.basename(zpath))

    raw = load_rows_from_zip(zpath, errs, warns)
    rows = parse(raw, errs, warns) if raw is not None else []

    # ── 对测试集逐字核对
    test_dir = _local(args.test_dir) if args.test_dir else None
    if not test_dir or not os.path.isdir(test_dir):
        errs.append("找不到测试集目录 %s，无法核对文件名（用 --test-dir 指定）" % args.test_dir)
        truth = None
    else:
        truth = set(os.listdir(test_dir))
    K = args.num_classes
    names = [r[1] for r in rows]
    cnt = Counter(names)
    dup = [n for n, c in cnt.items() if c > 1]
    if dup:
        errs.append("有 %d 个文件名重复出现，例：%s" % (len(dup), dup[:3]))
    bad_lab = [(ln, lab) for ln, _, lab, _ in rows if not re.fullmatch(r"\d{4}", lab)]
    if bad_lab:
        errs.append("%d 行的类别编号不是四位数字，例：第 %d 行 %r" % (len(bad_lab), bad_lab[0][0], bad_lab[0][1]))
    out_rng = [(ln, lab) for ln, _, lab, _ in rows if re.fullmatch(r"\d{4}", lab) and int(lab) >= K]
    if out_rng:
        errs.append("%d 行的类别编号超出 0000~%04d，例：第 %d 行 %s" % (len(out_rng), K - 1, out_rng[0][0], out_rng[0][1]))
    padded = sum(1 for r in rows if r[3])
    if padded:
        warns.append("%d 行的文件名前后带空白" % padded)
    if truth is not None:
        s = set(names)
        miss = truth - s
        extra = s - truth
        if extra:
            low = {t.lower(): t for t in truth}
            case = [e for e in extra if e.lower() in low]
            if case:
                errs.append("%d 个文件名只差大小写，例：%s ≠ 测试集的 %s" % (len(case), case[0], low[case[0].lower()]))
            errs.append("%d 个文件名不在测试集里，例：%s" % (len(extra), sorted(extra)[:3]))
        if miss:
            errs.append("测试集有 %d 张图没有预测，例：%s" % (len(miss), sorted(miss)[:3]))
        if len(rows) != len(truth):
            errs.append("行数 %d ≠ 测试集 %d 张" % (len(rows), len(truth)))

    # ── 分布体检（不判错，只提示）
    labs = Counter(r[2] for r in rows if re.fullmatch(r"\d{4}", r[2]))
    if labs:
        per = sorted(labs.values())
        n = len(rows)
        print("  行数          : %d%s" % (n, ("  （测试集 %d）" % len(truth)) if truth else ""))
        print("  覆盖类别      : %d / %d" % (len(labs), K))
        print("  每类预测张数  : min=%d  中位=%d  max=%d  （测试集官方说类别均衡，≈%.0f 张/类）"
              % (per[0], per[len(per) // 2], per[-1], n / K))
        if len(labs) < K:
            warns.append("有 %d 个类一张都没预测到" % (K - len(labs)))
        if per[-1] > 4 * n / K:
            warns.append("某个类被预测了 %d 张，是均匀值的 %.1f 倍，看一眼是不是模型塌到某类了"
                         % (per[-1], per[-1] / (n / K)))
        print("  样例          : %s" % ", ".join("%s,%s" % (r[1], r[2]) for r in rows[:2]))
    print("  zip MD5       : %s" % md5(zpath))

    if args.compare:
        A = {r[1]: r[2] for r in rows}
        B = read_pred(_local(args.compare))
        ks = set(A) & set(B)
        d = sum(1 for k in ks if A[k] != B[k])
        se = math.sqrt(max(d, 1)) / max(len(ks), 1) * 100
        print("  vs 对照       : 分歧 %d/%d (%.2f%%)  实算误差棒 ±%.3f 分" % (d, len(ks), 100.0 * d / max(len(ks), 1), 2 * se))
        if d < 300:
            warns.append("与对照只差 %d 张 —— 榜分不可能有可测的变化，⛔别浪费提交名额" % d)

    print()
    for w in warns:
        print("  ⚠ " + w)
    for e in errs:
        print("  ⛔ " + e)
    print()
    if errs:
        print("⛔⛔ 有 %d 条硬错误 —— 别交。" % len(errs))
        sys.exit(1)
    print("✅ 格式全部通过，可以交。上传入口：排行榜「赛题结果文档」（只收 zip）。")


if __name__ == "__main__":
    main()
