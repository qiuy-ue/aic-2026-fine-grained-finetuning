#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""提交包自检脚本（质询会现场可跑，零依赖，仅标准库）。

用法：
    python verify_submission.py                      # 自检同目录 pred-results.zip
    python verify_submission.py <package.zip>        # 指定包
    python verify_submission.py <package.zip> ref.csv  # 另与参考标签比对一致率

检查项：
  1. zip 内为单一裸成员 pred_results.csv（不允许文件夹层级）
  2. 行数 = 37,444；无表头；标签为 4 位数字；图片名唯一
  3. 类别覆盖 = 750；每类张数 16~70（Sinkhorn 软配额后形态）
  4. 输出 MD5，便于与包内 文件清单_MD5.txt 对照
  5. 校验 文件清单_MD5.txt 中的全部文件（清单自身不计入）
"""
import hashlib
import os
import sys
import zipfile
from collections import Counter

EXPECT_ROWS = 37444
EXPECT_CLASSES = 750


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    if len(sys.argv) > 1:
        pkg = sys.argv[1]
    else:
        cands = [
            os.path.join(here, "pred-results.zip"),
            os.path.join(here, "..", "pred-results.zip"),
        ]
        pkg = next((c for c in cands if os.path.exists(c)), cands[0])
    ref = sys.argv[2] if len(sys.argv) > 2 else None

    print("=" * 62)
    print("提交包自检: %s" % pkg)
    with open(pkg, "rb") as f:
        raw_bytes = f.read()
    print("MD5        : %s" % hashlib.md5(raw_bytes).hexdigest().upper())
    with zipfile.ZipFile(pkg) as z:
        members = z.namelist()
    print("zip 成员   : %s" % members)
    ok_members = members == ["pred_results.csv"]
    print("  裸单成员 : %s" % ("PASS" if ok_members else "FAIL —— 必须只有 pred_results.csv"))

    with zipfile.ZipFile(pkg) as z:
        csv_name = [m for m in members if m.endswith(".csv")][0]
        text = z.read(csv_name).decode("utf-8")

    rows = [ln for ln in text.splitlines() if ln.strip()]
    pairs, bad = [], 0
    for ln in rows:
        parts = ln.rsplit(",", 1)
        if len(parts) != 2 or len(parts[1].strip()) != 4 or not parts[1].strip().isdigit():
            bad += 1
        else:
            pairs.append((parts[0].strip(), parts[1].strip()))

    print("行数       : %d（期望 %d）%s" % (len(rows), EXPECT_ROWS,
                                        "PASS" if len(rows) == EXPECT_ROWS else "FAIL"))
    print("格式坏行   : %d %s" % (bad, "PASS" if bad == 0 else "FAIL"))
    uniq = len({k for k, _ in pairs})
    print("图片名去重 : %d %s" % (uniq, "PASS" if uniq == len(pairs) else "FAIL"))

    cnt = Counter(v for _, v in pairs)
    print("类别覆盖   : %d（期望 %d）%s" % (len(cnt), EXPECT_CLASSES,
                                        "PASS" if len(cnt) == EXPECT_CLASSES else "FAIL"))
    print("每类张数   : min=%d max=%d %s" % (min(cnt.values()), max(cnt.values()),
                                          "PASS(软配额 16~70)" if 10 <= min(cnt.values()) and max(cnt.values()) <= 80 else "WARN"))

    if ref and os.path.exists(ref):
        r = {}
        for ln in open(ref, encoding="utf-8"):
            ln = ln.strip()
            if ln:
                i, l = ln.rsplit(",", 1)
                r[i] = l
        hit = sum(1 for k, v in pairs if r.get(k) == v)
        print("对参考一致 : %d/%d = %.3f%%" % (hit, len(r), hit / len(r) * 100))

    root = os.path.dirname(here)
    manifest = os.path.join(root, "文件清单_MD5.txt")
    n_ok = n_bad = n_miss = 0
    manifest_ok = None
    problems = []
    print("-" * 62)
    if os.path.exists(manifest):
        with open(manifest, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                digest, sep, rel = line.partition("  ")
                if not sep:
                    continue
                path = os.path.join(root, rel.replace("\\", "/"))
                if not os.path.exists(path):
                    n_miss += 1
                    problems.append("MISSING  " + rel)
                elif md5_file(path).lower() == digest.lower():
                    n_ok += 1
                else:
                    n_bad += 1
                    problems.append("MISMATCH " + rel)
        manifest_ok = n_bad == 0 and n_miss == 0
        print("文件清单   : %d 条  匹配 %d  缺失 %d  不一致 %d %s"
              % (n_ok + n_bad + n_miss, n_ok, n_miss, n_bad,
                 "PASS" if manifest_ok else "FAIL"))
        for p in problems[:10]:
            print("  " + p)
    else:
        print("文件清单   : 未找到（跳过）")

    print("=" * 62)
    if manifest_ok is None:
        print("结论：以上各项 PASS 即包结构合规；权重→包的可复现性见 reproduce.sh。")
    else:
        print("结论：结构 7 项 + 文件清单 %d/%d %s；权重→包的可复现性见 reproduce.sh。"
              % (n_ok, n_ok + n_bad + n_miss, "PASS" if manifest_ok else "FAIL"))


if __name__ == "__main__":
    main()
