# -*- coding: utf-8 -*-
"""
每个脚本开头 import 一下这个模块，解决三件杂事：
1) 把工程根目录加进 sys.path，这样 scripts/ 下的脚本能 import configs.paths
2) 把随工程附带的 OpenAI 官方 clip 包接进 sys.path，这样 import clip 能用
3) 把 stdout/stderr 切成 UTF-8，否则 Windows 控制台(cp936)遇到部分字符会报
   UnicodeEncodeError 直接中断脚本
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# OpenAI 官方 clip 包(赛题十一(二)3 明确允许的 "openai/clip 库")随工程附带在
# third_party/CLIP/ 下，不装进 conda 环境，免得污染别的项目，也让评审拿到代码就能跑。
# ⚠追加到 sys.path 末尾而不是开头：那个目录里还有 tests/ data/ setup.py 这些通用名字，
#   放开头会把同名模块盖掉。末尾则只有在别处找不到时才会命中，唯一新增的就是 clip。
_CLIP_SRC = os.path.join(_ROOT, "third_party", "CLIP")
if os.path.isdir(_CLIP_SRC) and _CLIP_SRC not in sys.path:
    sys.path.append(_CLIP_SRC)

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass


def banner(title):
    """统一的分节标题，让长脚本的输出好读。"""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
