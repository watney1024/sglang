#!/usr/bin/env python3
"""校验课程文档里的源码锚点是否还对得上当前仓库。

用法:
    python lesson/tools/check_anchors.py
    python lesson/tools/check_anchors.py --quiet

锚点约定
--------
每篇文档的 frontmatter 里写:

    anchors:
      - python/sglang/srt/managers/scheduler.py:4374 # get_next_batch_to_run

正文里可以写简写 `scheduler.py:4374`。解析规则:

1. 先按完整路径找; 找不到再按 basename 找。
2. 多个候选时, 用"行号是否越界 + 符号是否出现"来消歧;
   仍然并列时优先 python/sglang/srt/ 下的文件。
3. frontmatter 带符号名时, 符号必须出现在该行 ±6 行内。
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LESSON_ROOT = REPO_ROOT / "lesson"

FRONTMATTER_ANCHOR = re.compile(r"^\s*-\s+([^\s:#]+):(\d+)\s*(?:#\s*(\S+))?\s*$")
INLINE_REF = re.compile(
    r"`([A-Za-z0-9_./-]+\.(?:py|rs|cc|cu|cuh|h|hpp|proto|toml|md)):(\d+)`"
)
WINDOW = 6

# 同名前缀很多（loader.py / model_runner.py / engine.py 都有多份），按子树给权重，
# 让"课程实际讲的实现"优先于硬件后端、rust 扩展、测试脚本。
CORE_DIRS = (
    "python/sglang/srt/managers/",
    "python/sglang/srt/model_executor/",
    "python/sglang/srt/mem_cache/",
    "python/sglang/srt/model_loader/",
    "python/sglang/srt/configs/",
    "python/sglang/srt/layers/",
    "python/sglang/srt/entrypoints/",
    "python/sglang/srt/distributed/",
    "python/sglang/srt/speculative/",
    "python/sglang/srt/compilation/",
    "python/sglang/srt/eplb/",
    "python/sglang/srt/lora/",
)
LOW_PRIORITY_DIRS = ("benchmark/", "tools/", "test/", "docs/", "examples/")


def path_score(rel: str) -> int:
    if rel.startswith(LOW_PRIORITY_DIRS):
        return -3
    if rel.startswith(CORE_DIRS):
        return 5
    if rel.startswith("python/sglang/srt/"):
        return 4
    if rel.startswith("python/sglang/"):
        return 3
    if rel.startswith("python/"):
        return 1
    return 0

_PATH_INDEX: dict[str, list[Path]] | None = None
_LINE_CACHE: dict[Path, list[str] | None] = {}


def read_lines(path: Path) -> list[str] | None:
    if path in _LINE_CACHE:
        return _LINE_CACHE[path]
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        lines = None
    _LINE_CACHE[path] = lines
    return lines


def build_index() -> dict[str, list[Path]]:
    global _PATH_INDEX
    if _PATH_INDEX is not None:
        return _PATH_INDEX

    index: dict[str, list[Path]] = {}
    for p in REPO_ROOT.rglob("*.py"):
        if any(part in {".git", "build", "dist", "__pycache__"} for part in p.parts):
            continue
        index.setdefault(p.name, []).append(p)
    _PATH_INDEX = index
    return index


def candidates(path_str: str) -> list[Path]:
    direct = REPO_ROOT / path_str
    if direct.is_file():
        return [direct]

    index = build_index()
    name = Path(path_str).name
    all_matches = index.get(name, [])
    suffix_matches = [p for p in all_matches if p.as_posix().endswith("/" + path_str)]
    return suffix_matches or all_matches


def symbol_near(lines: list[str], lineno: int, symbol: str) -> bool:
    lo = max(0, lineno - 1 - WINDOW)
    hi = min(len(lines), lineno + WINDOW)
    return any(symbol in line for line in lines[lo:hi])


def check_anchor(path_str: str, lineno: int, symbol: str | None) -> str | None:
    cands = candidates(path_str)
    if not cands:
        return f"文件不存在: {path_str}"

    diagnostics: list[str] = []
    viable: list[tuple[int, Path]] = []

    for p in cands:
        lines = read_lines(p)
        rel = p.relative_to(REPO_ROOT)
        if lines is None:
            diagnostics.append(f"{rel}: 无法读取")
            continue
        if not (1 <= lineno <= len(lines)):
            diagnostics.append(f"{rel}: 行号越界 (共 {len(lines)} 行)")
            continue
        if symbol and not symbol_near(lines, lineno, symbol):
            diagnostics.append(f"{rel}: 附近找不到 {symbol!r}")
            continue
        viable.append((path_score(rel.as_posix()), p))

    if len(viable) == 1:
        return None

    if not viable:
        detail = "; ".join(diagnostics[:3])
        return f"{path_str}:{lineno} 无法定位 ({detail})"

    # 多个候选都成立: 按子树权重取最高, 仍并列说明引用真的有歧义
    viable.sort(key=lambda x: -x[0])
    if len(viable) >= 2 and viable[0][0] > viable[1][0]:
        return None
    shown = ", ".join(p.relative_to(REPO_ROOT).as_posix() for _, p in viable[:4])
    return f"引用有歧义: {path_str}:{lineno} 匹配 {len(viable)} 个文件 ({shown})"


def parse_frontmatter(text: str) -> tuple[str, list[tuple[str, int, str | None]]]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return text, []

    end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
    if end is None:
        return text, []

    anchors: list[tuple[str, int, str | None]] = []
    in_anchors = False
    for line in lines[1:end]:
        if re.match(r"^anchors:\s*$", line):
            in_anchors = True
            continue
        if not in_anchors:
            continue
        m = FRONTMATTER_ANCHOR.match(line)
        if m:
            anchors.append((m.group(1), int(m.group(2)), m.group(3)))
        elif line.strip() and not line.startswith((" ", "\t")):
            in_anchors = False

    return "\n".join(lines[end + 1 :]), anchors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quiet", action="store_true", help="只输出失败项")
    args = parser.parse_args()

    docs = sorted(LESSON_ROOT.rglob("*.md"))
    total_anchors = 0
    total_inline = 0
    failures: list[tuple[Path, str]] = []

    for doc in docs:
        text = doc.read_text(encoding="utf-8", errors="replace")
        body, anchors = parse_frontmatter(text)
        checked: set[tuple[str, int]] = set()

        for path_str, lineno, symbol in anchors:
            total_anchors += 1
            checked.add((path_str, lineno))
            err = check_anchor(path_str, lineno, symbol)
            if err:
                failures.append((doc, err))

        inline_count = 0
        for m in INLINE_REF.finditer(body):
            path_str, lineno = m.group(1), int(m.group(2))
            if (path_str, lineno) in checked:
                continue
            inline_count += 1
            total_inline += 1
            err = check_anchor(path_str, lineno, None)
            if err:
                failures.append((doc, err))

        if not args.quiet:
            rel = doc.relative_to(REPO_ROOT).as_posix()
            print(f"{rel}: frontmatter {len(anchors)} 条, 内联 {inline_count} 条")

    print()
    print(f"锚点合计: frontmatter {total_anchors}, 内联 {total_inline}")

    if failures:
        print(f"失败 {len(failures)} 项:")
        for doc, err in failures:
            print(f"  - {doc.relative_to(REPO_ROOT).as_posix()}: {err}")
        return 1

    print("全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
