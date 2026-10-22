# -*- coding: utf-8 -*-
"""66 · `--margin`（CosFace 加性余弦间隔）补丁的自检 —— ⭐零 GPU、几秒钟。

⛔ **没跑过这个脚本不许开实验 U**（09 章 9.9「第0天」那一行写死的条件）。

═══════════════════════════════════════════════════════════════════════
⭐ 设计原则：**从源码里把真代码抠出来测**，⛔别在测试里重打一遍
═══════════════════════════════════════════════════════════════════════
测试里重写一遍被测表达式 = 只测了"我打字有没有打错两遍"，是自我实现的预言。
本脚本用 `ast` 把 28 号里**真正在跑的那几行**取出来 exec/eval，所以它测的是
**盘上那个文件**，不是我的记忆。

五道闸：
  闸① 源码闸：`margin` ⛔不出现在**推理路径**（head_logits / evaluate 的函数体）里
  闸② 恒等闸：margin=0 时 `lg_loss is lg`（**同一个对象**，不是"数值相等"）
               ⇒ 下游计算图逐位不变 ⇒ L/J/M 的复现不受影响，这是最强的那种等价
  闸③ 数学闸：margin>0 时 ≡ 教科书 CosFace `s·(cosθ − m)`（one-hot 情形对参考实现）
  闸④ 软标签闸：mixup 软标签下间隔按 λ **线性摊开**、总质量 = m；
               且梯度真的流到 s 和 W；loss(m>0) > loss(m=0)（间隔确实在加压）
  闸⑤ 参数闸：`--margin` 不配 `--cosine-head` 时脚本**拒绝启动**（线性头上减常数没有几何含义）

用法：D:\\Anaconda\\envs\\dl\\python.exe scripts\\66_selfcheck_margin.py
"""
import ast
import os
import re
import subprocess
import sys

import torch
import torch.nn.functional as Fn

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(HERE, "28_finetune_v2.py")
PY = sys.executable

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print("  %s %-44s %s" % ("✅" if ok else "⛔", name, detail))
    return ok


def func_node(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise SystemExit("⛔ 源码里找不到函数 %s —— 补丁改了结构？" % name)


def main():
    with open(SRC_PATH, encoding="utf-8") as f:
        src = f.read()
    lines = src.splitlines()
    tree = ast.parse(src)

    print("=" * 70)
    print("66 自检 · CosFace `--margin` 补丁   目标文件 %s" % os.path.basename(SRC_PATH))
    print("=" * 70)

    # ── 闸① 源码闸：推理路径必须干净 ─────────────────────────────────────
    print("\n[闸①] 源码闸：margin ⛔不许出现在推理路径里")
    for fname in ("head_logits", "evaluate"):
        nd = func_node(tree, fname)
        body = "\n".join(lines[nd.lineno - 1:nd.end_lineno])
        check("%s() 函数体不含 margin" % fname, "margin" not in body,
              "第 %d~%d 行" % (nd.lineno, nd.end_lineno))
    hits = [i + 1 for i, l in enumerate(lines) if "margin" in l and "label_smoothing" not in l
            and "label-smoothing" not in l]
    print("     margin 出现在第 %s 行（应当只有：argparse/校验/打印/目录名/损失那一行）" % hits)

    # ── 闸② 恒等闸：margin=0 → 同一个对象 ────────────────────────────────
    print("\n[闸②] 恒等闸：margin=0 时 lg_loss **就是** lg（同一个对象）")
    m = re.search(r"^\s*lg_loss\s*=\s*(.+)$", src, re.M)
    if not m:
        raise SystemExit("⛔ 源码里找不到 `lg_loss = ...` 那一行 —— 补丁没打上？")
    expr = m.group(1).strip()
    print("     抠出来的真表达式：lg_loss = %s" % expr)

    class A:            # 冒充 argparse 的 Namespace
        pass

    torch.manual_seed(0)
    B, K = 8, 6
    lg = torch.randn(B, K)
    s = torch.tensor(20.0, requires_grad=True)
    ys = Fn.softmax(torch.randn(B, K), dim=1)

    a = A(); a.margin = 0.0
    out0 = eval(expr, {}, {"lg": lg, "s": s, "ys": ys, "args": a})
    check("margin=0 → lg_loss is lg", out0 is lg,
          "⇒ 下游逐位不变，L/J/M 的复现不受影响")

    a.margin = 0.2
    out1 = eval(expr, {}, {"lg": lg, "s": s, "ys": ys, "args": a})
    check("margin>0 → 返回新张量且确实变小", (out1 is not lg) and bool((out1 <= lg).all()),
          "最大降幅 %.3f（= s·m·max(y) = %.3f）"
          % (float((lg - out1).max()), 20.0 * 0.2 * float(ys.max())))

    # ── 闸③ 数学闸：one-hot 情形 ≡ 教科书 CosFace ───────────────────────
    print("\n[闸③] 数学闸：one-hot 目标下 ≡ s·(cosθ − m)")
    # 用真·soft_ce（同样从源码抠）算，⛔不在测试里重写一遍
    sce_node = func_node(tree, "soft_ce")
    ns = {"Fn": Fn, "torch": torch}
    exec("\n".join(lines[sce_node.lineno - 1:sce_node.end_lineno]), ns)
    soft_ce = ns["soft_ce"]

    torch.manual_seed(1)
    d = 16
    f = Fn.normalize(torch.randn(B, d), dim=1)
    W = Fn.normalize(torch.randn(K, d), dim=1)
    s_val, m_val = 30.0, 0.35
    s3 = torch.tensor(s_val)
    cos = f @ W.T
    lg3 = s3 * cos                                   # = head_logits 的余弦分支
    y_idx = torch.randint(0, K, (B,))
    ys3 = Fn.one_hot(y_idx, K).float()

    a3 = A(); a3.margin = m_val
    ours = eval(expr, {}, {"lg": lg3, "s": s3, "ys": ys3, "args": a3})
    ref = s_val * (cos - m_val * ys3)                # 教科书 CosFace（另写一遍，故意的）
    check("logits 与参考实现一致", torch.allclose(ours, ref, atol=1e-5),
          "最大偏差 %.2e" % float((ours - ref).abs().max()))
    check("损失与 F.cross_entropy(参考 logits) 一致",
          torch.allclose(soft_ce(ours, ys3), Fn.cross_entropy(ref, y_idx), atol=1e-5),
          "%.6f vs %.6f" % (float(soft_ce(ours, ys3)), float(Fn.cross_entropy(ref, y_idx))))

    # ── 闸④ 软标签闸：mixup 下按 λ 线性摊开 + 梯度确实在流 ──────────────
    print("\n[闸④] 软标签闸：mixup 目标下间隔按 λ 线性摊开（⭐选 CosFace 不选 ArcFace 的理由）")
    lam, ia, ib = 0.7, 2, 5
    ys4 = torch.zeros(1, K)
    ys4[0, ia], ys4[0, ib] = lam, 1 - lam
    lg4 = torch.zeros(1, K)
    a4 = A(); a4.margin = m_val
    d4 = (lg4 - eval(expr, {}, {"lg": lg4, "s": s3, "ys": ys4, "args": a4}))[0]
    check("间隔只落在两个来源类上", float(d4.sum()) > 0 and
          bool((d4[[i for i in range(K) if i not in (ia, ib)]] == 0).all()),
          "A 类 %.3f : B 类 %.3f = %.2f : %.2f" % (d4[ia], d4[ib], lam, 1 - lam))
    check("摊开比例 = λ:(1−λ)，总质量 = s·m",
          abs(float(d4[ia]) - s_val * m_val * lam) < 1e-4
          and abs(float(d4.sum()) - s_val * m_val) < 1e-4,
          "总质量 %.4f（应 = %.1f×%.2f = %.3f）" % (float(d4.sum()), s_val, m_val, s_val * m_val))

    # 真实训练里 s 是可学习的：梯度必须同时流到 s 和 W（CosFace 的 s 乘在整个 (cos−m) 上）
    Wg = W.clone().requires_grad_(True)
    sg = torch.tensor(s_val, requires_grad=True)
    lg5 = sg * (Fn.normalize(f, dim=1) @ Fn.normalize(Wg, dim=1).T)
    a5 = A(); a5.margin = m_val
    loss_m = soft_ce(eval(expr, {}, {"lg": lg5, "s": sg, "ys": ys3, "args": a5}), ys3)
    loss_m.backward()
    check("梯度同时流到 s 与 W", sg.grad is not None and Wg.grad is not None
          and float(sg.grad.abs()) > 0 and float(Wg.grad.abs().max()) > 0,
          "|∂L/∂s|=%.4f  max|∂L/∂W|=%.4f" % (float(sg.grad.abs()), float(Wg.grad.abs().max())))
    a0 = A(); a0.margin = 0.0
    loss_0 = soft_ce(eval(expr, {}, {"lg": lg3, "s": s3, "ys": ys3, "args": a0}), ys3)
    check("loss(m>0) > loss(m=0)（间隔确实在加压）",
          float(soft_ce(ours, ys3)) > float(loss_0),
          "%.4f > %.4f（差 %.4f）" % (float(soft_ce(ours, ys3)), float(loss_0),
                                      float(soft_ce(ours, ys3)) - float(loss_0)))

    # ── 闸⑤ 参数闸：线性头上不许用 margin ────────────────────────────────
    print("\n[闸⑤] 参数闸：--margin 不配 --cosine-head 必须**拒绝启动**")
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    r = subprocess.run([PY, SRC_PATH, "--margin", "0.2", "--smoke"],
                       cwd=os.path.dirname(HERE), env=env,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    msg = ((r.stderr or "") + (r.stdout or "")).strip().splitlines()
    check("退出码 ≠ 0 且给出理由", r.returncode != 0 and any("margin" in x for x in msg),
          (msg[-1][:70] if msg else "（无输出）"))

    # ── 汇总 ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  通过 %d 道，失败 %d 道" % (len(PASS), len(FAIL)))
    if FAIL:
        print("  ⛔ 失败：%s  → **不许开 U**" % FAIL)
        sys.exit(1)
    print("  ✅ 全部通过 → 可以开实验 U（`65_chain_exp.py --tag U --margin 0.2`）")
    print("  ⚠ 但这只证明**数学与恒等性**；『新旋钮真进了 AMP 训练循环』由 65 的起飞前冒烟兜底。")
    print("=" * 70)


if __name__ == "__main__":
    main()
