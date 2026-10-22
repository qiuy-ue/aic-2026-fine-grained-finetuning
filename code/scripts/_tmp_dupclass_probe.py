# -*- coding: utf-8 -*-
"""
_tmp_dupclass_probe.py —— 纯 CPU / 纯 stdlib：全 500 类【逐字节完全相同】重复图普查
策略：先 stat 取文件大小做预筛（同一字节内容必然同大小），只对"大小碰撞组"算 md5。
输出：
  A) 全库逐字节重复组统计
  B) 跨类重复对矩阵 top-K（哪些类对共享同一批图 = 同物种/同一份数据被切两半的铁证）
  C) 每个类的文件级指纹（格式/尺寸/DQT），用于判断"两个同物种类之间有没有可判别印记"
用法:
  python _tmp_dupclass_probe.py sizes      # 只做 stat，落盘 cache/_tmp_sizes.json
  python _tmp_dupclass_probe.py md5        # 读上一步，对碰撞组算 md5，落盘 _tmp_dupgroups.json
  python _tmp_dupclass_probe.py pairs      # 出跨类重复对矩阵
"""
import os, sys, json, hashlib, collections, time

DATA_ROOT = r"D:\aic_data\dataset"
if not os.path.isdir(DATA_ROOT):
    DATA_ROOT = "/workspace/D:/aic_data/dataset"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "cache")
F_SIZES = os.path.join(OUT, "_tmp_sizes.json")
F_GROUPS = os.path.join(OUT, "_tmp_dupgroups.json")


def step_sizes():
    t = time.time()
    rec = []   # [cls, name, size]
    for c in sorted(os.listdir(TRAIN_DIR)):
        d = os.path.join(TRAIN_DIR, c)
        if not os.path.isdir(d):
            continue
        for e in os.scandir(d):
            try:
                rec.append([c, e.name, e.stat().st_size])
            except Exception:
                pass
        if len(rec) % 20000 < 250:
            print("  %6d  %.0fs" % (len(rec), time.time() - t), flush=True)
    json.dump(rec, open(F_SIZES, "w"))
    print("共 %d 张, %.0fs -> %s" % (len(rec), time.time() - t, F_SIZES))
    cnt = collections.Counter(r[2] for r in rec)
    coll = [s for s, v in cnt.items() if v > 1]
    ncoll = sum(cnt[s] for s in coll)
    print("大小碰撞组 %d 个, 覆盖 %d 张 (%.2f%%)" % (len(coll), ncoll, 100.0 * ncoll / len(rec)))


def _hash_one(cn):
    c, n = cn
    try:
        with open(os.path.join(TRAIN_DIR, c, n), "rb") as f:
            return (hashlib.md5(f.read()).hexdigest(), c, n)
    except Exception:
        return None


def step_md5():
    from concurrent.futures import ProcessPoolExecutor
    rec = json.load(open(F_SIZES))
    cnt = collections.Counter(r[2] for r in rec)
    todo = [(r[0], r[1]) for r in rec if cnt[r[2]] > 1]
    print("需算 md5 的 %d 张" % len(todo), flush=True)
    t = time.time()
    groups = collections.defaultdict(list)
    with ProcessPoolExecutor(max_workers=12) as ex:
        for i, out in enumerate(ex.map(_hash_one, todo, chunksize=64)):
            if out is None:
                continue
            groups[out[0]].append([out[1], out[2]])
            if i % 8000 == 0:
                print("  %6d/%d  %.0fs" % (i, len(todo), time.time() - t), flush=True)
    g = {h: v for h, v in groups.items() if len(v) > 1}
    json.dump(g, open(F_GROUPS, "w"))
    tot = sum(len(v) for v in g.values())
    print("逐字节重复组 %d 个, 覆盖 %d 张 (%.2f%% of 103218), %.0fs"
          % (len(g), tot, 100.0 * tot / 103218, time.time() - t))


def step_pairs():
    g = json.load(open(F_GROUPS))
    pair = collections.Counter()       # (clsA,clsB) -> 共享的重复组数
    same = 0
    cross = 0
    cls_dupimgs = collections.Counter()
    for h, v in g.items():
        cs = sorted(set(x[0] for x in v))
        if len(cs) == 1:
            same += 1
            continue
        cross += 1
        for a in cs:
            cls_dupimgs[a] += 1
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                pair[(cs[i], cs[j])] += 1
    print("重复组总数 %d ; 组内同类 %d ; ⭐跨类 %d" % (len(g), same, cross))
    print("\n=== 跨类共享【逐字节相同】图的类对 (top 60) ===")
    print("  共享组数  类A    类B    |A|   |B|   共享/min(|A|,|B|)")
    sizes = {}
    for c in sorted(os.listdir(TRAIN_DIR)):
        d = os.path.join(TRAIN_DIR, c)
        if os.path.isdir(d):
            sizes[c] = len(os.listdir(d))
    for (a, b), v in pair.most_common(60):
        na, nb = sizes.get(a, 0), sizes.get(b, 0)
        print("  %6d   %s  %s  %4d  %4d   %.3f" % (v, a, b, na, nb, v / max(1, min(na, nb))))
    json.dump({"%s|%s" % k: v for k, v in pair.items()},
              open(os.path.join(OUT, "_tmp_crosspairs.json"), "w"))


if __name__ == "__main__":
    m = sys.argv[1] if len(sys.argv) > 1 else "sizes"
    {"sizes": step_sizes, "md5": step_md5, "pairs": step_pairs}[m]()
