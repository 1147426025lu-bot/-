# -*- coding: utf-8 -*-
r"""打包提交材料。

分两个包：
  提交包/论文-<队号>.pdf        论文正文，单独一个文件
  提交包/支撑材料.zip           源码、图表、程序、结果、AI 使用详情

刻意排除三类东西：
  1) 智能体与编辑器的工作文件（CLAUDE.md / AGENTS.md / .mathmodel/）——
     它们不是参赛材料，混进附件既无意义又容易被误读；
  2) 编译中间产物与临时截图；
  3) 未被任何 .tex/.py 引用的字体与大体积原始栅格。

字体这里要分清两类，不能一概而论"字库是系统依赖、不该打包"：
  · KaiTi.ttf / SimHei.ttf / simsun.ttf（合计约 30MB）确实是纯冗余——宏包走的是
    ctex 的 Windows 字体集，读系统字库，本地这三份没有任何文件引用。
  · LiSu.ttf 相反，gmcmthesis.cls 第 207 行用 \setCJKfamilyfont{gmcm-cover-label}
    [Path=./]{LiSu.ttf} 显式按相对路径加载，是本工程唯一不能删的字体文件。
    实测：把它移走后 xelatex 直接以 exit=1 失败，报
    `! Package fontspec Error: The font "LiSu" cannot be found`。
    注意 latexmk 会掩盖这个失败——字体文件不在它的依赖数据库里，源文件没变时
    它会直接跳过编译并返回 0，让人误以为通过。要验证必须删掉 main.log/main.xdv
    后单独跑 xelatex。

用法：python build_submission.py
"""
import io
import os
import re
import shutil
import sys
import zipfile

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, '提交包')
TEAM = '20260095597'

# 论文正文单独放
PAPER_PDF = ['main.pdf']
# 支撑材料：源码 + 图表 + 程序 + 结果 + AI 使用详情
SUPPORT = [
    'main.tex', 'gmcmthesis.cls', 'logo.pdf', 'title.pdf', 'LiSu.ttf',
    'sections', 'figures', 'code', 'results',
    'AI工具使用详情.tex', 'AI工具使用详情.pdf',
    '结果提交表-已填写.xlsx',
    # 空模板是 fill_results.py 的输入：只给填好的成品、不给模板，评委照着
    # 代码清单跑这一脚本会直接 FileNotFoundError。模板本身就是赛题下发的
    # 文件，随包带回没有额外风险，却能让整条流水线在解压后真正跑通。
    '结果提交模板.xlsx',
]
# 只收代码真正读得到的数据：5 个 xlsx + DEM 的 .mat。
# 同目录的 .tif 与其余 GIS 图层（水系/道路/水体/点位）没有任何脚本引用，
# 22MB 里绝大部分是它们，不进包。
DATA = [
    '数据/无人机应急物资运输基础数据',
    '数据/镇龙乡地理空间数据/镇龙乡及周边地理数据/数字高程模型数据（DEM）/镇龙乡及周边30米DEM.mat',
]

SKIP_DIR = {'__pycache__', '.ipynb_checkpoints'}
SKIP_EXT = {'.pyc', '.aux', '.log', '.out', '.fls', '.fdb_latexmk', '.xdv',
            '.synctex.gz', '.bak', '.tmp'}

# render_drawio.py 每次都会顺带导出一张 PNG 供人眼检查版式，但正文插图自
# 改用矢量 PDF 后就不再引用它。这类"工作用预览图"不该进提交包，可它天天
# 被重新生成，手工删一次下次又冒出来。改成按引用关系过滤：只保留 .tex 里
# 真的 \includegraphics 到的图，.drawio 源文件属于交付物，一律保留。
FIG_EXT = {'.png', '.pdf', '.jpg', '.jpeg', '.eps'}


def referenced_figures():
    """扫出全部 .tex 里 \\includegraphics 引用到的图文件名。

    必须带上扩展名比对：正文写的是 figures/fig_roadmap.pdf，同目录下的
    fig_roadmap.png 只是渲染脚本顺带导出的预览图，按主干名匹配会把两者
    一起放行，等于没过滤。引用时不带扩展名的（LaTeX 会自行试扩展名）
    则退回按主干名匹配。
    """
    used = set()
    texs = [os.path.join(ROOT, 'main.tex')]
    for f in os.listdir(os.path.join(ROOT, 'sections')):
        if f.endswith('.tex'):
            texs.append(os.path.join(ROOT, 'sections', f))
    for t in texs:
        with open(t, encoding='utf-8') as fh:
            for m in re.finditer(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', fh.read()):
                used.add(os.path.basename(m.group(1)))
    return used


def copy_tree(src, dst, fig_ok=None):
    n = 0
    for dirpath, dirnames, filenames in os.walk(src):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIR]
        for f in filenames:
            ext = os.path.splitext(f)[1].lower()
            if ext in SKIP_EXT:
                continue
            if fig_ok is not None and ext in FIG_EXT:
                if f not in fig_ok and os.path.splitext(f)[0] not in fig_ok:
                    continue
            s = os.path.join(dirpath, f)
            rel = os.path.relpath(s, src)
            d = os.path.join(dst, rel)
            os.makedirs(os.path.dirname(d), exist_ok=True)
            shutil.copy2(s, d)
            n += 1
    return n


def main():
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    paper_dir = os.path.join(OUT, '论文')
    sup_dir = os.path.join(OUT, '支撑材料')
    os.makedirs(paper_dir)
    os.makedirs(sup_dir)

    # 1) 论文正文
    for f in PAPER_PDF:
        shutil.copy2(os.path.join(ROOT, f), os.path.join(paper_dir, '论文-%s.pdf' % TEAM))
        print('论文  %s' % os.path.join('论文', '论文-%s.pdf' % TEAM))

    # 2) 支撑材料
    total = 0
    for item in SUPPORT + DATA:
        s = os.path.join(ROOT, item.replace('/', os.sep))
        if not os.path.exists(s):
            print('  !! 缺失，跳过：%s' % item)
            continue
        # 数据类的目录必须在包里保持原层级（数据/…），不能压平成顶层目录：
        # core.py 是按 '数据/无人机应急物资运输基础数据' 找数据的，压平后
        # 评委解压再跑代码会直接 FileNotFoundError。根目录下的单文件条目
        # （main.tex、.cls、字体等）仍平铺，保持包顶层整洁。
        keep_tree = item.startswith('数据/')
        if os.path.isdir(s):
            fig_ok = referenced_figures() if os.path.basename(item) == 'figures' else None
            dst = os.path.join(sup_dir, item.replace('/', os.sep)) if keep_tree \
                else os.path.join(sup_dir, os.path.basename(item))
            total += copy_tree(s, dst, fig_ok)
        else:
            d = os.path.join(sup_dir, item.replace('/', os.sep)) if keep_tree \
                else os.path.join(sup_dir, os.path.basename(item))
            os.makedirs(os.path.dirname(d), exist_ok=True)
            shutil.copy2(s, d)
            total += 1
    print('支撑材料  共 %d 个文件' % total)

    # 3) 支撑材料打成 zip——多数提交系统只收单个压缩包
    zpath = os.path.join(OUT, '支撑材料.zip')
    with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for dirpath, dirnames, filenames in os.walk(sup_dir):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIR]
            for f in filenames:
                p = os.path.join(dirpath, f)
                z.write(p, os.path.relpath(p, sup_dir))
    print('已打包  %s  (%.1f MB)' % (os.path.basename(zpath), os.path.getsize(zpath) / 1e6))

    # 3.5) 确认排除了不该出现的东西（必须在删暂存目录之前做）
    print('\n排除项自检：')
    bad = []
    for dirpath, dirnames, filenames in os.walk(sup_dir):
        for f in filenames:
            low = f.lower()
            if low in ('claude.md', 'agents.md') or f.endswith(('.aux', '.log', '.fls')):
                bad.append(os.path.relpath(os.path.join(dirpath, f), sup_dir))
    if os.path.exists(os.path.join(sup_dir, '.mathmodel')):
        bad.append('.mathmodel/')
    print('  未发现智能体工作文件或编译中间产物' if not bad else '  !! 残留：%s' % bad)

    # 4.5) 收掉暂存目录。支撑材料.zip 已经独立完整，暂存的 提交包/支撑材料/
    # 只是打包过程中的中间态，留着会让 提交包/ 的体积凭空翻倍、也让"包里到底
    # 有什么"出现两个互相矛盾的事实来源（目录 vs 压缩包）。想看内容直接开 zip。
    shutil.rmtree(sup_dir)
    print('  已清理暂存目录 提交包/支撑材料/')

    # 5) 项目根目录不得残留 Word/WPS 文档。
    # 这类文件的 docProps/core.xml 里会带作者名与 Microsoft 账户邮箱（实测某草稿
    # 的 lastModifiedBy 就是一个 QQ 邮箱），解压扫描即构成身份泄露，属附件2明令
    # 禁止的"可能显示答题人身份的标志"。这里只报警不自动删——删文件不是打包脚本
    # 该干的事，交给人确认后再移出。
    print('\nWord/WPS 残留检查：')
    stray = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIR and d != '提交包']
        for f in filenames:
            if f.lower().endswith(('.doc', '.docx', '.wps', '.dot', '.dotx')):
                stray.append(os.path.relpath(os.path.join(dirpath, f), ROOT))
    if stray:
        print('  !! 项目内仍有 Word/WPS 文档，打包前必须移出：')
        for p in stray:
            print('     %s' % p)
    else:
        print('  无 .doc/.docx/.wps 残留')


if __name__ == '__main__':
    main()
