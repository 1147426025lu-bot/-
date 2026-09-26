# -*- coding: utf-8 -*-
"""排版门禁总检查（只读，不产生任何文件）。

为什么要有这个脚本
------------------
《交付与复现说明》第二节第 4 条把「四条排版门禁各为 0，且摘要恰一页」写成硬门禁，
但在此之前**没有任何程序在查它**——`tests_regression.py` 与 `build_submission.py`
里都搜不到相关检查，全靠人眼。2026-09-25 的摘要改写因此静默地把「关键词」两行挤到
了第 3 页（PDF 里摘要独占一页的约束被破坏），四条 grep 门禁却全部是 0，谁也没发现。
本脚本把这五条判据做成一次调用、一个退出码，失败时打印是哪一条、差多少。

判据
----
1.  `Overfull \\hbox` / `Overfull \\vbox` / `Reference .* undefined` /
    `Citation .* undefined` 在 `main.log` 上的出现次数各为 0（用 `grep -c` 口径，
    即按行计数，与《交付与复现说明》一致）。
2.  摘要恰一页：`摘要` 所在页必须同时含 `关键词`，且**下一页的首个非空行是 `目录`**。
    第 2 条需要 `pdftotext`（poppler）。**没有它时如实报「未判定」并按失败退出**——
    「工具缺失」不等于「判据通过」，这正是本项目不许把未做的复核当成已做的规矩。

用法
----
    python code/check_layout_gates.py            # 在工程根目录下运行
退出码 0 = 全部通过；1 = 有判据不通过（含无法判定）。
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(ROOT, 'main.log')
PDF = os.path.join(ROOT, 'main.pdf')

# grep -c 口径：数「含有该模式的行数」，不是匹配次数
GREP_GATES = [
    ('Overfull \\hbox', r'Overfull \\hbox'),
    ('Overfull \\vbox', r'Overfull \\vbox'),
    ('Reference undefined', r'Reference .* undefined'),
    ('Citation undefined', r'Citation .* undefined'),
]


def count_lines(pattern: str, text: str) -> int:
    rx = re.compile(pattern)
    return sum(1 for line in text.splitlines() if rx.search(line))


def pdf_page(pdftotext: str, path: str, page: int) -> list[str]:
    out = subprocess.run(
        [pdftotext, '-enc', 'UTF-8', '-f', str(page), '-l', str(page), path, '-'],
        capture_output=True, check=False)
    text = out.stdout.decode('utf-8', errors='replace')
    return [ln.strip() for ln in text.replace('\r', '').split('\n') if ln.strip()]


def main() -> int:
    fails: list[str] = []

    if not os.path.exists(LOG):
        print('[FAIL] 找不到 main.log —— 先编译：latexmk -g -xelatex main.tex')
        return 1
    log = open(LOG, encoding='utf-8', errors='replace').read()

    # ---- 判据 1：四条排版门禁 ----
    for name, pattern in GREP_GATES:
        n = count_lines(pattern, log)
        print('[%s] %-20s %d' % ('OK  ' if n == 0 else 'FAIL', name, n))
        if n:
            fails.append('%s = %d（应为 0）' % (name, n))

    # ---- 页数（记录用，不作判据）----
    m = re.search(r'Output written on \S+ \((\d+) pages', log)
    pages = int(m.group(1)) if m else None
    print('[INFO] 页数 %s' % (pages if pages is not None else '未在 main.log 中找到'))

    # ---- 判据 2：摘要恰一页 ----
    pdftotext = shutil.which('pdftotext')
    if not pdftotext:
        print('[FAIL] 摘要恰一页：本机无 pdftotext，**未判定**（不等于通过）')
        print('       装法：poppler（Windows 可用 scoop/choco 安装 poppler，'
              '或把 pdftotext.exe 放进 PATH）')
        fails.append('摘要恰一页未判定（缺 pdftotext）')
    elif not os.path.exists(PDF):
        print('[FAIL] 摘要恰一页：找不到 main.pdf')
        fails.append('摘要恰一页未判定（缺 main.pdf）')
    else:
        if pages is None:
            fails.append('摘要恰一页未判定（无法从 main.log 取页数）')
            pages = 0
        abs_page = None
        for p in range(1, pages + 1):
            if any('摘' in ln and '要' in ln and '：' in ln for ln in pdf_page(pdftotext, PDF, p)):
                abs_page = p
                break
        if abs_page is None:
            print('[FAIL] 摘要恰一页：在 %d 页中没找到「摘 要：」标头' % pages)
            fails.append('摘要恰一页未判定（找不到摘要页）')
        else:
            lines = pdf_page(pdftotext, PDF, abs_page)
            has_kw = any('关键词' in ln for ln in lines)
            nxt = pdf_page(pdftotext, PDF, abs_page + 1) if abs_page < pages else []
            nxt_first = nxt[0] if nxt else '(无下一页)'
            ok = has_kw and nxt_first.startswith('目录')
            print('[%s] 摘要恰一页：摘要=P.%d，关键词%s，下一页首行=%r'
                  % ('OK  ' if ok else 'FAIL', abs_page,
                     '同页' if has_kw else '**不在同页**', nxt_first))
            if not ok:
                fails.append('摘要非一页（关键词%s，下一页首行 %r）'
                             % ('不在摘要页' if not has_kw else '在摘要页', nxt_first))

    print()
    if fails:
        print('门禁未通过，共 %d 条：' % len(fails))
        for f in fails:
            print('  - %s' % f)
        return 1
    print('排版门禁全部通过。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
