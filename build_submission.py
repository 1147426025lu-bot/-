# -*- coding: utf-8 -*-
# 本程序及代码是在人工智能工具辅助下完成的。
# 工具名称：DeepSeek；版本/型号：DeepSeek V4.1 Flash（API 模型名 deepseek-flash）；
# 开发机构：杭州深度求索人工智能基础技术研究有限公司；
# 版本发布日期：2026 年 9 月 10 日。
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
  · LiSu.ttf 是第三方商业字体（方正隶书），.gitignore 里写明"授权不允许再分发"，
    **本包同样不转发它**——不转发给仓库、却转发给评委，是同一个法律问题的两种
    写法。上一版把它列进 SUPPORT，等于一边拒绝入库一边随包分发。
    不转发的前提是"没有它也能编"：gmcmthesis.cls 已改为三级回退（工程内
    LiSu.ttf → 系统隶书 LiSu/STLiti → 楷体）。实测把 LiSu.ttf 移走后 xelatex
    仍以 exit=0 编出全文，无 fontspec 报错；本机装有系统隶书，回退后拿到的是
    同一个字形。注意 latexmk 会掩盖这类失败——字体不在它的依赖数据库里，源文件
    没变时会直接跳过编译并返回 0，要验证必须删掉 main.log/main.xdv 后单独跑。

数据清单（F16 续）：DATA 不再手写，改为**直接向 solution_io.input_hashes() 要**。
上一版这里手写 6 个"代码真正读得到"的文件，而 input_hashes() 对整棵 数据/ 树
做哈希（17 个），两份清单一分开就必然走味：评委解压后 input_hashes() 比方案记录
少 11 项，verify.py 与 q3/q4 的加载器会判"输入数据与方案记录不一致"并拒绝加载——
包看着完整，却一条都跑不通。现在打包集合与哈希集合是同一个函数的结果，只能同时变。

清单预检（F16）：旧版对缺失项只打印一行「跳过」就继续，于是少一个结果表、少一张
图，照样产出一个「看起来完整」的提交包——缺的恰恰是最该被发现的东西。现在打包前
先逐项核对：任一必需文件缺失、或正文 \\includegraphics 引用的图不存在，就列出**全部**
缺失项并以非零码中止，且**不创建输出目录**、不动上一版包。`--check` 只预检不打包。

用法：python build_submission.py [--check]
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

# 数据清单的唯一真相源在 code/solution_io.py（方案记录里的 input_hashes 也由它
# 生成）。打包器不另写一份，否则两份清单迟早走味。
sys.path.insert(0, os.path.join(ROOT, 'code'))
import solution_io as sio                                          # noqa: E402

# 论文正文单独放
PAPER_PDF = ['main.pdf']
# 支撑材料：源码 + 图表 + 程序 + 结果 + AI 使用详情
SUPPORT = [
    'main.tex', 'gmcmthesis.cls', 'logo.pdf', 'title.pdf',
    'sections', 'figures', 'code', 'results',
    'AI工具使用详情.tex', 'AI工具使用详情.pdf',
    '结果提交表-已填写.xlsx',
    # 空模板是 fill_results.py 的输入：只给填好的成品、不给模板，评委照着
    # 代码清单跑这一脚本会直接 FileNotFoundError。模板本身就是赛题下发的
    # 文件，随包带回没有额外风险，却能让整条流水线在解压后真正跑通。
    '结果提交模板.xlsx',
    # 打包器本身也要进包：code/tests_regression.py 的 T21 直接 import 它来核对
    # 清单一致性与预检行为，评委按《交付与复现说明》跑回归测试时缺了它就会得到
    # "30 通过 1 失败"，看着像工程有问题。它不是交付物的一部分，但不进包会让
    # 交付的测试套件在自己声称的环境里跑不过——这个代价比多带一个脚本大。
    'build_submission.py',
]
# 要打进包的数据 = input_hashes() 覆盖的全部文件。刻意如此：非计算图层（DEM 的
# .tif、水系/道路/水体/点位、说明 PDF、详情地图 HTML）确实没有脚本读，但它们
# **计入输入身份**——input_hashes() 对整棵树做哈希，方案记录里存的就是这 17 项。
# 少带 11 项，评委解压后每一条方案都会被判"输入数据不一致"。要么两边都收，要么
# 两边都不收；后者要先改哈希口径并重出方案记录，不是打包器该自作主张的事。
DATA = sorted(sio.input_hashes())
# 本地才有的资源：不进包（授权不允许再分发），但打包前要确认它在位，且确认
# 缺了它也算得出论文——见 gmcmthesis.cls 的字体回退。
LOCAL_ONLY = ['LiSu.ttf']

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


def missing_referenced_figures():
    """正文引用但文件不存在的图。引用带路径时按相对 ROOT 的路径找，找不到再按
    figures/ 目录下的主干名找（LaTeX 允许省略扩展名）。"""
    miss = []
    for name in sorted(referenced_figures()):
        stem = os.path.splitext(name)[0]
        cands = [os.path.join(ROOT, name), os.path.join(ROOT, 'figures', name),
                 os.path.join(ROOT, 'figures', stem + '.pdf'),
                 os.path.join(ROOT, 'figures', stem + '.png')]
        if not any(os.path.exists(c) for c in cands):
            miss.append(name)
    return miss


def _local_note():
    """本地专有资源一句话状态。缺 LiSu.ttf 不影响出论文（cls 有字体回退），
    但缺了要说清楚，免得有人以为是打包漏了。"""
    miss = [f for f in LOCAL_ONLY if not os.path.exists(os.path.join(ROOT, f))]
    if not miss:
        return '%s 齐备（%s）' % ('、'.join(LOCAL_ONLY), '不随包分发')
    return ('%s 本机不存在——本包不含它（授权不允许再分发），'
            '编译时 cls 自动回退到系统隶书/楷体' % '、'.join(miss))


def preflight():
    """打包前的清单核对：返回缺失项列表（空表示可以打包）。"""
    bad = []
    for item in PAPER_PDF:
        if not os.path.exists(os.path.join(ROOT, item)):
            bad.append('论文正文 %s（缺它就没有可提交的 PDF）' % item)
    # DATA 空 = 数据目录整个不在（纯仓库检出就是这样）。这不能算"没有数据要打"，
    # 而是最该拦下的情形：包里没有原始数据，评审跑不了任何脚本。
    if not DATA:
        bad.append('原始数据 数据/（打包清单取自 solution_io.input_hashes()，'
                   '当前一个文件都没有：请确认赛题数据已就位）')
    for item in SUPPORT + DATA:
        if not os.path.exists(os.path.join(ROOT, item.replace('/', os.sep))):
            bad.append('支撑材料 %s' % item)
    bad += ['正文引用的图 figures/%s' % f for f in missing_referenced_figures()]
    return bad


def main():
    if '--check' in sys.argv[1:]:
        bad = preflight()
        if bad:
            print('预检不通过，共 %d 项缺失：' % len(bad))
            for b in bad:
                print('  !! %s' % b)
            return 1
        print('预检通过：清单 %d 项齐全（含数据 %d 项），正文引用的图全部存在'
              % (len(PAPER_PDF) + len(SUPPORT) + len(DATA), len(DATA)))
        print('  本地资源（不入包）：%s' % _local_note())
        return 0

    bad = preflight()
    if bad:
        # 中止时**不创建输出目录**：上一版 提交包/ 原样保留，人不会拿到一个
        # 少了结果表或图、却看起来完整的包。
        print('清单预检未通过，已中止打包（未改动 %s）：' % os.path.basename(OUT))
        for b in bad:
            print('  !! 缺失：%s' % b)
        return 1

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
    print('支撑材料  共 %d 个文件（其中原始数据 %d 个，与 input_hashes() 同一来源）'
          % (total, len(DATA)))
    print('本地资源（不入包）：%s' % _local_note())

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
            # 第三方商业字体不得随包转发。它曾经就在 SUPPORT 里，所以这条要真检。
            if f in LOCAL_ONLY:
                bad.append('%s（本地资源，不得随包分发）' % f)
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
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
