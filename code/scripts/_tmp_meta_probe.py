# -*- coding: utf-8 -*-
"""
_tmp_meta_probe.py  —— 纯 CPU / 纯 stdlib 的图像元数据探针（⛔不碰 GPU，⛔不改任何已有文件）
用法:
  python _tmp_meta_probe.py test   [N]      # 扫测试集前 N 张(N=0 表示全部)
  python _tmp_meta_probe.py train  [N]      # 扫训练集随机 N 张
  python _tmp_meta_probe.py pair   0266 0424   # 对比两个类的文件级指纹
"""
import os, sys, io, struct, collections, hashlib, random, json

DATA_ROOT = r"D:\aic_data\dataset"
if not os.path.isdir(DATA_ROOT):
    DATA_ROOT = "/workspace/D:/aic_data/dataset"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")

MARKER_NAME = {
    0xE0: "APP0", 0xE1: "APP1", 0xE2: "APP2", 0xE3: "APP3", 0xE4: "APP4",
    0xE5: "APP5", 0xE6: "APP6", 0xE7: "APP7", 0xE8: "APP8", 0xE9: "APP9",
    0xEA: "APP10", 0xEB: "APP11", 0xEC: "APP12", 0xED: "APP13", 0xEE: "APP14",
    0xEF: "APP15", 0xFE: "COM", 0xDB: "DQT", 0xC4: "DHT", 0xDD: "DRI",
    0xC0: "SOF0", 0xC1: "SOF1", 0xC2: "SOF2", 0xC3: "SOF3", 0xDA: "SOS",
}


def parse_jpeg(buf):
    """返回 (segments, sof_info)。segments = [(name, payload_bytes)]; 只到 SOS 为止。"""
    segs = []
    sof = None
    n = len(buf)
    if n < 4 or buf[0] != 0xFF or buf[1] != 0xD8:
        return None, None
    i = 2
    while i < n - 1:
        if buf[i] != 0xFF:
            i += 1
            continue
        m = buf[i + 1]
        if m == 0xFF:
            i += 1
            continue
        if m in (0xD8, 0x01) or 0xD0 <= m <= 0xD7:
            i += 2
            continue
        if m == 0xD9:
            break
        if i + 4 > n:
            break
        ln = struct.unpack(">H", buf[i + 2:i + 4])[0]
        payload = buf[i + 4:i + 2 + ln]
        name = MARKER_NAME.get(m, "M%02X" % m)
        if m in (0xC0, 0xC1, 0xC2, 0xC3) and len(payload) >= 6:
            prec = payload[0]
            h = struct.unpack(">H", payload[1:3])[0]
            w = struct.unpack(">H", payload[3:5])[0]
            nc = payload[5]
            comps = []
            for c in range(nc):
                off = 6 + c * 3
                if off + 3 <= len(payload):
                    comps.append((payload[off], payload[off + 1] >> 4, payload[off + 1] & 0xF, payload[off + 2]))
            sof = dict(marker=name, prec=prec, h=h, w=w, ncomp=nc, comps=comps)
        segs.append((name, payload))
        if m == 0xDA:
            break
        i = i + 2 + ln
    return segs, sof


def printable(b, minlen=4):
    """抽取 payload 里的可打印 ASCII 串"""
    out, cur = [], []
    for ch in b:
        if 32 <= ch < 127:
            cur.append(chr(ch))
        else:
            if len(cur) >= minlen:
                out.append("".join(cur))
            cur = []
    if len(cur) >= minlen:
        out.append("".join(cur))
    return out


def scan_dir(files, label=""):
    seg_hist = collections.Counter()
    sig_hist = collections.Counter()        # 文件魔数
    sof_hist = collections.Counter()        # (w,h,subsample,prec,marker)
    dqt_hash = collections.Counter()        # 量化表指纹
    app_examples = collections.defaultdict(list)
    str_hist = collections.Counter()
    nbytes_hist = []
    has_any_meta = 0
    ntot = 0
    trail_hist = collections.Counter()   # EOI 之后还有多少字节（隐藏数据的经典位置）
    app0_len = collections.Counter()
    for p in files:
        try:
            with open(p, "rb") as f:
                buf = f.read()
        except Exception:
            continue
        ntot += 1
        nbytes_hist.append(len(buf))
        sig = buf[:4].hex()
        if buf[:3] == b"\xff\xd8\xff":
            sig_hist["JPEG"] += 1
        elif buf[:8] == b"\x89PNG\r\n\x1a\n":
            sig_hist["PNG"] += 1
        elif buf[:4] == b"RIFF" and buf[8:12] == b"WEBP":
            sig_hist["WEBP"] += 1
        elif buf[:3] == b"GIF":
            sig_hist["GIF"] += 1
        elif buf[:2] == b"BM":
            sig_hist["BMP"] += 1
        else:
            sig_hist["OTHER:" + sig] += 1
        # EOI 之后的尾巴
        e = buf.rfind(b"\xff\xd9")
        trail_hist[len(buf) - (e + 2) if e >= 0 else -1] += 1
        segs, sof = parse_jpeg(buf)
        if segs is None:
            continue
        names = [s[0] for s in segs]
        seg_hist[tuple(sorted(set(names)))] += 1
        meta_hit = False
        for name, payload in segs:
            if name in ("DQT", "DHT", "SOS", "DRI") or name.startswith("SOF"):
                if name == "DQT":
                    dqt_hash[hashlib.md5(payload).hexdigest()[:10]] += 1
                continue
            # APPn / COM
            key = name + ":" + payload[:12].split(b"\x00")[0].decode("latin-1", "replace")
            if name == "APP0":
                app0_len[len(payload)] += 1
            if len(app_examples[key]) < 3:
                app_examples[key].append((os.path.basename(p), len(payload)))
            else:
                app_examples[key].append(None)
            if name not in ("APP0",):   # APP0=JFIF 是无信息的
                meta_hit = True
            for s in printable(payload, 5):
                str_hist[s[:120]] += 1
        if meta_hit:
            has_any_meta += 1
        if sof:
            ss = ",".join("%d:%dx%d" % (c[0], c[1], c[2]) for c in sof["comps"])
            sof_hist[(sof["marker"], sof["w"], sof["h"], sof["ncomp"], ss)] += 1
    return dict(ntot=ntot, sig=sig_hist, seg=seg_hist, sof=sof_hist, dqt=dqt_hash,
                app=app_examples, strs=str_hist, meta=has_any_meta, sizes=nbytes_hist,
                trail=trail_hist, app0len=app0_len)


def report(r, topn=25, name=""):
    print("=" * 78)
    print("### %s   共 %d 文件" % (name, r["ntot"]))
    print("-- 文件魔数:", dict(r["sig"].most_common(10)))
    print("-- 含非 APP0 元数据段的文件数: %d / %d = %.2f%%" %
          (r["meta"], r["ntot"], 100.0 * r["meta"] / max(1, r["ntot"])))
    print("-- 段组合 (top %d):" % topn)
    for k, v in r["seg"].most_common(topn):
        print("     %6d  %s" % (v, "+".join(k)))
    print("-- SOF/尺寸/子采样 (top %d, 共 %d 种):" % (topn, len(r["sof"])))
    for k, v in r["sof"].most_common(topn):
        print("     %6d  %s %dx%d ncomp=%d  %s" % (v, k[0], k[1], k[2], k[3], k[4]))
    print("-- DQT 指纹种类数: %d ; top:" % len(r["dqt"]))
    for k, v in r["dqt"].most_common(8):
        print("     %6d  %s" % (v, k))
    print("-- APPn 头部标识 (top %d, 共 %d 种):" % (topn, len(r["app"])))
    for k in sorted(r["app"], key=lambda x: -len(r["app"][x]))[:topn]:
        ex = r["app"][k][:2]
        print("     %6d  %-28s  例: %s" % (len(r["app"][k]), k, ex))
    print("-- 元数据里的可打印串 (top %d, 共 %d 种):" % (topn * 2, len(r["strs"])))
    for k, v in r["strs"].most_common(topn * 2):
        print("     %6d  %r" % (v, k))
    print("-- EOI 之后的尾巴字节数分布:", dict(r["trail"].most_common(10)))
    print("-- APP0 payload 长度分布:", dict(r["app0len"].most_common(10)))
    if r["sizes"]:
        s = sorted(r["sizes"])
        print("-- 字节数 min/中位/max = %d / %d / %d" % (s[0], s[len(s) // 2], s[-1]))


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "test"
    if mode == "test":
        N = int(sys.argv[2]) if len(sys.argv) > 2 else 500
        fs = sorted(os.listdir(TEST_DIR))
        if N:
            random.seed(0)
            fs = random.sample(fs, min(N, len(fs)))
        files = [os.path.join(TEST_DIR, x) for x in fs]
        report(scan_dir(files), name="TEST (%d)" % len(files))
    elif mode == "train":
        N = int(sys.argv[2]) if len(sys.argv) > 2 else 500
        allf = []
        for c in sorted(os.listdir(TRAIN_DIR)):
            d = os.path.join(TRAIN_DIR, c)
            if os.path.isdir(d):
                for x in os.listdir(d):
                    allf.append(os.path.join(d, x))
        random.seed(0)
        files = random.sample(allf, min(N, len(allf))) if N else allf
        report(scan_dir(files), name="TRAIN (%d of %d)" % (len(files), len(allf)))
    elif mode == "pair":
        for c in sys.argv[2:]:
            d = os.path.join(TRAIN_DIR, c)
            files = [os.path.join(d, x) for x in sorted(os.listdir(d))]
            report(scan_dir(files), topn=15, name="TRAIN/%s (%d)" % (c, len(files)))
    else:
        print("unknown mode")


if __name__ == "__main__":
    main()
