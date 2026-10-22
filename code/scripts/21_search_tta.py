# -*- coding: utf-8 -*-
"""
21：在 20 号抽好的视角池上离线搜索 TTA 配方。纯 numpy，不碰 GPU，秒级出结果。

═══════════════════════════════════════════════════════════════════════
一、这一步在做什么
═══════════════════════════════════════════════════════════════════════

TTA 的全部内容就一句话：**把同一张图的多个视角的特征平均起来再分类**。
20 号已经把 24 个候选视角的特征存死了，所以"换配方"在这里只是换几个数组求平均——
毫秒级。于是可以把几十种组合全部量一遍，而不是凭直觉挑一个去打榜。

⚠**只训一个头，所有组合共用**。这是实验设计的硬要求：
   如果每个组合都重训一次头，那两个组合之间就同时差了"特征"和"头"两个变量，
   分不清是谁的功劳。18 号脚本比 TTA 那一题时也是这么做的（同一个头，只换特征）。

═══════════════════════════════════════════════════════════════════════
二、两把尺子 + 判读规则（照 18 号的结论用，不自创）
═══════════════════════════════════════════════════════════════════════

  ①干净60%   ：suspicion <= 60% 分位的子集上算准确率（老尺子里唯一还算能用的）
  ②⭐分布匹配：val_weights.npz 的权重加权准确率，**已与测试集难度对齐**
               ⇒ 差 1 个百分点 ≈ 真实差 1 分，不需要玄学倍率

  · 两把尺子**方向一致**才可信；**打架**就判本地无效，别硬解释
  · 幅度只读②，换算 = 分布匹配 x 100
  · ⚠ESS=11730，单次标准误 0.0045 ⇒ |Δ| < 0.004 一律判"低于分辨率"

  另外打印"原始等权"仅供参考。⛔不用它下结论——18 号已经实测它把 C 那道题判反了。

═══════════════════════════════════════════════════════════════════════
三、⭐配对标准误：为什么这里能比 0.0045 分得更细
═══════════════════════════════════════════════════════════════════════

0.0045 是"一次独立测量"的标准误，适用于比较两个**各自独立**的实验。
但本脚本里所有组合都在**同一批样本、同一个头**上跑，只有特征不同
⇒ 绝大多数样本在两个配方下**答案完全一样**，那部分方差被完全抵消了。
真正贡献不确定性的只有"两边答案不同"的那少数样本。

所以额外报一个**配对标准误**（比值估计量的 delta 法）：
    Δ = Σwᵢdᵢ / Σwᵢ ,   dᵢ = 对错之差 ∈ {-1,0,+1}
    SE_pair = sqrt(Σwᵢ²(dᵢ-Δ)²) / Σwᵢ
同时打印"两边答案不同的样本数"，那才是这个比较真正的信息量。

⚠但**头条结论仍以 ±0.004 为准**。原因：配对标准误只回答"这个差在**这批验证样本上**
是不是真的"，它管不了**尺子本身的模型误差**（分布匹配权重是估出来的，
17/18 号实测它把整体水平从 0.694 拉到 0.613 而真实是 0.590，仍有 0.023 的残差）。
把配对 SE 当成"能测出 0.001 的分差"是自欺欺人。

═══════════════════════════════════════════════════════════════════════
四、⚠一个必须写清楚的口径陷阱
═══════════════════════════════════════════════════════════════════════

  · 多尺度视角 cc0XX 的 s 是**边长**比例   s=0.40 ⇒ 面积只剩 **0.16**
  · RandomResizedCrop 的 scale 是**面积**比例  scale=0.40 ⇒ 边长约 **0.63**

⇒ 光看数字会误以为"strong(0.40~0.75) 比 cc040 更狠"，**恰恰相反**：
   cc040 是全池子里最激进的视角，strong 只相当于边长 0.63~0.87。
   报告里必须显式换算，否则结论会被读反。

用法:
  python scripts/21_search_tta.py
  python scripts/21_search_tta.py --smoke      # 用 20 号的 _smoke 池子跑通流程
"""
import _bootstrap  # noqa: F401
from _bootstrap import banner

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as Fn
from sklearn.model_selection import train_test_split

from configs import paths as P

# ⚠必须与 18/20 号逐字一致，否则本地结果与历史实验不可比
SEED, VAL_FRAC = 42, 0.2
RESOLUTION = 0.004          # ESS=11730 ⇒ 单次 SE≈0.0045，小于这个数的差判为零


# ══════════════════════════════════════════════════════════════ 头
def train_head(Xtr, ytr, C, loss, n_iter=300, lr=0.05, device="cuda", beta=1.0):
    """
    ⚠与 18_revalidate_ruler.py 里那份**逐字相同**（它又与 06 号相同）。
      全批量 Adam、L2 口径 1/(2Cn)、SCE 的 A=-4、beta=1。
      换优化器或换正则口径就没法跟历史实验比，别动。
    ⚠这里坚持**自己现训**而不是加载 cache/head_loss_sce.npz：
      那个缓存文件历史上被 06 号的超参循环覆盖过（最后胜出的是 beta=8），
      自己训一遍才能保证拿到的确实是 beta=1 的头。
    """
    n, d = Xtr.shape
    K = int(ytr.max()) + 1
    Xt = torch.from_numpy(Xtr).to(device)
    yt = torch.from_numpy(ytr).to(device)
    W = torch.zeros(K, d, device=device, requires_grad=True)
    b = torch.zeros(K, device=device, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr)
    reg = 1.0 / (2.0 * C * n)
    for _ in range(n_iter):
        opt.zero_grad(set_to_none=True)
        lg = Xt @ W.T + b
        if loss == "ce":
            L = Fn.cross_entropy(lg, yt)
        else:                                    # SCE：α·CE + β·RCE，A=-4
            p = Fn.softmax(lg, 1).clamp(1e-7, 1.0)
            oh = Fn.one_hot(yt, K).float()
            L = Fn.cross_entropy(lg, yt) - beta * (p * ((1 - oh) * -4.0)).sum(1).mean()
        (L + reg * (W * W).sum()).backward()
        opt.step()
    return W.detach().cpu().numpy(), b.detach().cpu().numpy()


# ══════════════════════════════════════════════════════════════ 组合
def combine(pool, spec):
    """
    spec = [(视角名, 权重), ...] → 合成一个 (N,512) 的 TTA 特征。
    规则：每个视角**先各自 L2 归一化**（pool 里存的已经是归一化后的），
          加权平均，**再归一化一次**。
    ⚠第二次归一化不能省：平均之后模长会变小（视角越不一致缩得越多），
      而线性头的 logit 与模长成正比 ⇒ 不重新归一化就等于给"视角分歧大的样本"
      整体调低了置信度，那是另一种效应，会污染对比。
    """
    n, d = pool[spec[0][0]].shape
    acc = np.zeros((n, d), dtype=np.float32)
    tot = 0.0
    for name, wt in spec:
        acc += np.float32(wt) * pool[name]
        tot += wt
    acc /= np.float32(tot)
    acc /= np.maximum(np.linalg.norm(acc, axis=1, keepdims=True), 1e-12)
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="用 20 号的 _smoke 池子跑通流程")
    ap.add_argument("--greedy-steps", type=int, default=12, help="贪心前向选择最多加几个视角")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    banner("21 TTA 配方离线搜索")

    # ------------------------------------------------------------ 划分（与 18/20 一致）
    X = np.load(P.F_TRAIN_FEAT).astype(np.float32)
    y = np.load(P.F_TRAIN_LABEL).astype(np.int64)
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    idx = np.arange(len(y))
    itr, iva = train_test_split(idx, test_size=VAL_FRAC, random_state=SEED, stratify=y)

    suffix = "_smoke" if args.smoke else ""
    pool_f = os.path.join(P.CACHE_DIR, "viewpool_val%s.npy" % suffix)
    meta = json.load(open(os.path.join(P.CACHE_DIR, "viewpool_val%s_meta.json" % suffix),
                          encoding="utf-8"))
    Vraw = np.load(pool_f)
    if args.smoke:                      # 冒烟池只有前 N 行，把验证索引也截齐
        iva = iva[:Vraw.shape[1]]
    Xv, yv = X[iva], y[iva]
    N = len(iva)
    print("  验证集 %d 张，视角池 %s ← %s" % (N, Vraw.shape, pool_f))

    # ------------------------------------------------------------ 池子（全部 L2 归一化）
    pool = {}
    Vf = Vraw.astype(np.float32)
    Vf /= np.maximum(np.linalg.norm(Vf, axis=2, keepdims=True), 1e-12)
    for i, nm in enumerate(meta["view_names"]):
        pool[nm] = Vf[i]
    del Vf, Vraw

    # ⭐把 10 号缓存的 5 个视角也并进来：它们才是**榜上那个 61.357 配方的原件**，
    #   用它当基线，"相对现配方的差值"才是逐位可比的，而不是"近似复现"。
    cached_names = []
    fa = os.path.join(P.CACHE_DIR, "train_features_aug.npy")
    if os.path.exists(fa):
        A = np.load(fa, mmap_mode="r")                       # (5,103218,512)
        Av = np.asarray(A[:, iva, :], dtype=np.float32)
        Av /= np.maximum(np.linalg.norm(Av, axis=2, keepdims=True), 1e-12)
        for i in range(Av.shape[0]):
            nm = "mildC%d" % i
            pool[nm] = Av[i]
            cached_names.append(nm)
        del Av
        print("  并入 10 号缓存的 %d 个现配方视角（基线用它，保证与榜上配方逐位一致）"
              % len(cached_names))
    print("  池子共 %d 个视角" % len(pool))

    # ------------------------------------------------------------ 尺子
    wz = np.load(os.path.join(P.CACHE_DIR, "val_weights.npz"))
    if args.smoke:
        assert np.array_equal(wz["val_idx"][:N], iva), "权重文件的划分和这里不一致"
        w = wz["weight"][:N]
    else:
        assert np.array_equal(wz["val_idx"], iva), "权重文件的划分和这里不一致"
        w = wz["weight"]
    ess = w.sum() ** 2 / (w ** 2).sum()
    nz = np.load(os.path.join(P.CACHE_DIR, "noise_scores.npz"), allow_pickle=True)
    sv = nz["suspicion"][iva]
    m60 = sv <= np.quantile(sv, 0.60)
    print("  尺子①干净60%%: %d 样本   尺子②分布匹配: ESS %.0f (单次SE %.4f)"
          % (m60.sum(), ess, 0.5 / np.sqrt(ess)))

    # ------------------------------------------------------------ 头（只训一次）
    banner("训练线性头（SCE, C=10, beta=1, A=-4, Adam lr=0.05, 300 iter）")
    print("  ⚠只训这一次，所有组合共用 —— 只让特征变，不让头变")
    W, b = train_head(X[itr], y[itr], 10.0, "sce", device=device)
    if device == "cuda":
        torch.cuda.empty_cache()        # 头训完就把显存还回去，后面全是 numpy

    def correct_of(F):
        return (np.argmax(F @ W.T + b, 1) == yv).astype(np.float64)

    def rulers(c):
        return dict(raw=float(c.mean()),
                    clean60=float(c[m60].mean()),
                    matched=float((c * w).sum() / w.sum()))

    def paired(ca, cb, ww=None):
        """配对比较：只有'两边答案不同'的样本才携带信息。"""
        ww = w if ww is None else ww
        d = ca - cb
        delta = float((ww * d).sum() / ww.sum())
        se = float(np.sqrt((ww ** 2 * (d - delta) ** 2).sum()) / ww.sum())
        return delta, se, int((d != 0).sum())

    # ------------------------------------------------------------ 预注册组合
    CC = ["cc085", "cc070", "cc055", "cc040"]
    CCF = [c + "_flip" for c in CC]
    CORNER = ["corner_tl", "corner_tr", "corner_bl", "corner_br"]
    MILD = ["mild%d" % i for i in range(5)]
    STRONG = ["strong%d" % i for i in range(5)]
    MC = cached_names if cached_names else MILD

    def eq(names):                       # 等权
        return [(n, 1.0) for n in names]

    combos = [
        ("00 无TTA(原图)",              eq(["orig"])),
        ("01 ★基线=现配方(缓存)",        eq(["orig"] + MC)),
        ("02 现配方(20号新抽,自检)",      eq(["orig"] + MILD)),
        ("03 原图+翻转(最便宜)",         eq(["orig", "orig_flip"])),
        ("04 原图+多尺度x4",            eq(["orig"] + CC)),
        ("05 原图+温和尺度(085,070)",    eq(["orig", "cc085", "cc070"])),
        ("06 原图+强尺度(055,040)",      eq(["orig", "cc055", "cc040"])),
        ("07 原图+翻转+多尺度+尺度翻转",   eq(["orig", "orig_flip"] + CC + CCF)),
        ("08 原图+强随机x5",            eq(["orig"] + STRONG)),
        ("09 原图+四角x4",              eq(["orig"] + CORNER)),
        ("10 现配方+强随机",            eq(["orig"] + MC + STRONG)),
        ("11 现配方+多尺度",            eq(["orig"] + MC + CC)),
        ("12 全部视角(29)",             eq(list(pool.keys()))),
        ("13 原图+翻转+多尺度+强随机",    eq(["orig", "orig_flip"] + CC + STRONG)),
        ("14 原图+多尺度(去掉最狠的040)",  eq(["orig", "cc085", "cc070", "cc055"])),
        ("15 原图+多尺度+四角",          eq(["orig"] + CC + CORNER)),
        ("16 现配方+翻转",              eq(["orig", "orig_flip"] + MC)),
        # --- 加权变体：原图是"标准视角"锚点，试试给它更大话语权
        ("17 原图x2+多尺度",            [("orig", 2.0)] + eq(CC)),
        ("18 原图x3+全部增强",          [("orig", 3.0)] + eq(
            [n for n in pool if n != "orig"])),
        ("19 原图x2+翻转+多尺度+尺度翻转", [("orig", 2.0)] + eq(["orig_flip"] + CC + CCF)),
    ]

    banner("组合评测（★=基线；Δ 相对基线；|Δ|<%.3f 判为低于分辨率）" % RESOLUTION)
    print("  %-28s %4s | %8s %8s | %9s %9s %8s %7s"
          % ("组合", "视角", "干净60%", "⭐分布匹配", "Δ干净60", "⭐Δ匹配", "配对SE", "分歧数"))
    print("  " + "-" * 108)

    ones = np.ones_like(w)

    def run_table(title, table, ref_spec, ref_tag, note=None):
        """
        跑一张对照表。ref_spec = 这张表的参照系（Δ 都相对它算）。
        ⚠每张表用**自己的参照系**：跨表比 Δ 会把不同的对照条件混在一起。
        """
        banner(title)
        if note:
            print(note)
        print("  参照系: %s" % ref_tag)
        print("  %-30s %4s | %8s %8s | %9s %9s %8s %7s"
              % ("组合", "视角", "干净60%", "⭐分布匹配", "Δ干净60", "⭐Δ匹配", "配对SE", "分歧数"))
        print("  " + "-" * 110)
        ref_c = correct_of(combine(pool, ref_spec))
        rows = []
        for name, spec in table:
            c = correct_of(combine(pool, spec))
            r = rulers(c)
            # 尺①也用配对口径算，这样"它到底有没有分辨力"是可查的，而不是看一眼差值拍脑袋
            d1, se1, _ = paired(c[m60], ref_c[m60], ones[m60])
            d2, se, ndis = paired(c, ref_c)
            # 判读：先看幅度够不够格，再看两把尺子方向一致不一致
            if ndis == 0:
                verdict = "★参照系本身"
            elif abs(d2) < RESOLUTION:
                verdict = "≈参照(低于分辨率)"
            elif d1 * d2 < 0:
                verdict = "⚠两尺打架,本地无效"        # 严格反向才算打架
            elif abs(d1) < se1:
                verdict = ("✅优于参照" if d2 > 0 else "❌劣于参照") + "(尺①未表态)"
            else:
                verdict = "✅优于参照" if d2 > 0 else "❌劣于参照"
            print("  %-30s %4d | %8.4f %8.4f | %+9.4f %+9.4f %8.4f %7d  %s"
                  % (name, len(spec), r["clean60"], r["matched"], d1, d2, se, ndis, verdict))
            rows.append(dict(name=name, n_views=len(spec), views=[s[0] for s in spec],
                             weights=[s[1] for s in spec], **r,
                             d_clean60=d1, se_clean60=se1,
                             d_matched=d2, paired_se=se, n_disagree=ndis,
                             est_score=r["matched"] * 100, verdict=verdict))
        return rows

    results = run_table(
        "表A 预注册组合（★=榜上现配方；|Δ|<%.3f 判为低于分辨率）" % RESOLUTION,
        combos, eq(["orig"] + MC), "01 ★现配方(缓存) = 榜上 61.357 那一版")
    print("\n  ⭐幅度读法：'分布匹配 x 100' 直接当预计分数（冻结特征类方法系统低估约 0.5 分）")

    # ---------------------------------------------------------- 表B：等视角数对照
    # ⚠⚠这张表才是核心假设的**干净判据**。
    #   表A 里各组合视角数不同（3~29 个），而 TTA 的收益本身就随视角数单调增
    #   （平均的样本越多、方差越小）。⇒ "多尺度组合分数高"完全可能只是"视角多"，
    #   跟"放大"这件事一点关系都没有。**必须把视角数钉死才能归因到增强类型上。**
    #   下面每一行都是「原图 + 4 个增强视角」，只有那 4 个视角的**族别**不同。
    tableB = [
        ("B1 现配方随机x4(缓存)",      eq(["orig"] + MC[:4])),
        ("B2 现配方随机x4(新抽)",      eq(["orig"] + MILD[:4])),
        ("B3 强随机x4",               eq(["orig"] + STRONG[:4])),
        ("B4 多尺度放大x4",           eq(["orig"] + CC)),
        ("B5 温和放大x4(085/070+翻转)", eq(["orig", "cc085", "cc070",
                                          "cc085_flip", "cc070_flip"])),
        ("B6 四角位移x4",             eq(["orig"] + CORNER)),
        ("B7 翻转类x4",               eq(["orig", "orig_flip", "cc085_flip",
                                        "cc070_flip", "cc055_flip"])),
    ]
    results += run_table(
        "表B ⭐等视角数对照：原图+4 个增强视角，只换族别",
        tableB, eq(["orig"] + MC[:4]), "B1 现配方随机x4（同族、同视角数）",
        note="  ⚠TTA 收益随视角数单调增，表A 的组合视角数从 3 到 29 不等 ⇒ 不能直接归因。\n"
             "    这张表把视角数**钉死在 4**，差异只剩'增强族别'一个变量。")

    # ---------------------------------------------------------- 表C：增量归因
    # ⚠核心假设的第二个干净判据：在现配方之上**再加 4 个视角**，
    #   加"多尺度"和加"同族的另外 4 个随机视角"哪个更值？
    #   C1 是**关键对照组**：它什么新花样都没有，纯粹把视角数从 6 加到 10。
    #   如果 C3(多尺度) 打不过 C1，那"多尺度"就毫无特殊性，收益全来自视角数。
    BASE6 = ["orig"] + MC
    tableC = [
        ("C0 现配方(6视角,不加)",       eq(BASE6)),
        ("C1 ★+同族随机x4(纯凑数对照)",  eq(BASE6 + MILD[:4])),
        ("C2 +强随机x4",              eq(BASE6 + STRONG[:4])),
        ("C3 +多尺度放大x4",           eq(BASE6 + CC)),
        ("C4 +温和放大x4",            eq(BASE6 + ["cc085", "cc070",
                                                "cc085_flip", "cc070_flip"])),
        ("C5 +四角位移x4",            eq(BASE6 + CORNER)),
        ("C6 +翻转类x4",              eq(BASE6 + ["orig_flip", "cc085_flip",
                                                 "cc070_flip", "cc055_flip"])),
    ]
    results += run_table(
        "表C ⭐增量归因：现配方 6 视角之上再加 4 个，加什么最值",
        tableC, eq(BASE6 + MILD[:4]), "C1 +同族随机x4（纯凑数，什么新花样都没有）",
        note="  ⚠参照系故意选'再加 4 个同族随机视角'。\n"
             "    只有**打赢 C1** 才能说某个族别有特殊价值；打平就只是'视角数变多'而已。")

    # ------------------------------------------------------------ 贪心（探索性）
    banner("贪心前向选择（⚠探索性，必然过拟合这把尺子，只看'哪一族被反复选中'）")
    print("  在 %d 个候选视角上逐个加入使分布匹配最高的那个。" % len(pool))
    print("  ⚠这是在**同一批 ESS=11730 的样本上反复挑最大值**，")
    print("    挑 %d 步就等于做了上百次比较，最优值必然被高估。" % args.greedy_steps)
    print("    ⛔不要拿这条轨迹的终点当推荐配置，它只用来看趋势。")
    chosen, rest = ["orig"], [n for n in pool if n != "orig"]
    traj = []
    cur = rulers(correct_of(combine(pool, eq(chosen))))["matched"]
    print("  step 0  %-14s 分布匹配 %.4f" % ("orig", cur))
    for step in range(1, args.greedy_steps + 1):
        best, bestv = None, -1.0
        for cand in rest:
            v = rulers(correct_of(combine(pool, eq(chosen + [cand]))))["matched"]
            if v > bestv:
                best, bestv = cand, v
        chosen.append(best)
        rest.remove(best)
        traj.append(dict(step=step, added=best, matched=bestv, gain=bestv - cur))
        print("  step %-2d +%-14s 分布匹配 %.4f  (相对上一步 %+.4f)"
              % (step, best, bestv, bestv - cur))
        cur = bestv

    # ------------------------------------------------------------ 落盘
    out = dict(n_val=N, ess=float(ess), resolution=RESOLUTION,
               baseline="01 ★基线=现配方(缓存)",
               head="SCE C=10 beta=1 A=-4 Adam lr=0.05 300iter (现训,不读缓存)",
               view_group_sims=meta.get("group_sims"),
               orig_vs_cache_cos=meta.get("orig_vs_cache_cos"),
               combos=results, greedy=traj)
    fo = os.path.join(P.CACHE_DIR, "tta_search%s.json" % suffix)
    with open(fo, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    banner("完成")
    print("  结果 →", fo)
    print("  ⛔本脚本不生成任何提交文件；测试集特征等配方定下来再由主线处理。")


if __name__ == "__main__":
    main()
