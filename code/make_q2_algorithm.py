# -*- coding: utf-8 -*-
# 本程序及代码是在人工智能工具辅助下完成的。
# 工具名称：DeepSeek；版本/型号：DeepSeek V4.1 Flash（API 模型名 deepseek-flash）；
# 开发机构：杭州深度求索人工智能基础技术研究有限公司；
# 版本发布日期：2026 年 9 月 10 日。

"""生成论文《问题二求解框架》图的 draw.io 源文件。

与《总技术路线图》(make_roadmap.py) 共用同一套图形语汇——纯黑白、空心块箭头、
竖排栏标签——故两图一眼同族；区别只在版式：本图是**横版三栏**，三栏依次对应
6.2 节的 (1) 构造基线、(3) ALNS 搜索、(4) ε-约束扫描，栏间用空心块箭头连接，
底部一条通栏横条给出推荐方案的落点。

之所以做成横版：原图是 724×1062 的纵向长条（宽高比 1.467），按 0.9\\textwidth
插入后图高与整页的《总技术路线图》几乎等大，却没有路线图那一整套落版处理
（\\clearpage + 独立小节标题 + 不加 \\caption），挂在正文中间两头不靠。三栏横版
把宽高比提到 2 以上，图高降到约三分之一版心，字号同时升到与路线图同口径。

所有版式坐标由下面的参数推出，改内容不用动坐标。

用法：python code/make_q2_algorithm.py
输出：figures/fig_q2_algorithm.drawio（再由 render_drawio.py 渲染为 PNG/PDF）
"""
import pathlib

# 字宽模型、盒高规则、竖排工具与样式串一律从 make_roadmap 复用，不另抄一份，
# 否则两份必然漂移。注意**不要** import 它的 BOX_W / BOX_AVAIL——那两个量是按
# 四栏反解的，本图是三栏，必须自己算。
from make_roadmap import Fig, box_h, vert, S_BOX, S_VLAB, S_ARROW

# ---------------------------------------------------------------- 版式参数
W = 816                 # 画布宽，与图 2.1《总技术路线图》同宽 ⇒ 同字号口径
ML = MR = 20            # 画布左右外边距
VLAB_W = 20             # 竖排栏标签宽（竖排单字宽 = 字号 17px）
VLAB_GAP = 4
ARROW_W = 26            # 栏之间的空心块箭头
ARROW_GAP = 14
BOX_GAP = 8
BAR_GAP = 24            # 三栏与底部通栏横条之间
BAR_H = 34              # 与 BOX_H1 同高，保证横条内单行字有同样的行高
TOP = BOTTOM = 14

NCOL = 3

# ---------------------------------------------------------------- 图的内容
# 三栏逐栏对应 6.2 节的 (1)(3)(4)；每个盒内文字都必须与 code/q2.py 的实测算子
# 与 results/q2_pareto.csv 的推荐方案对得上。
COLUMNS = [
    {
        'label': '构造基线',
        'boxes': [
            '紧急货箱\n单独组批',
            '非紧急货箱\nFFD 装箱',
            '地理相近架次\n贪心合并',
            '紧迫度排序\n枚举 A/B/C 机型',
            '最早空闲无人机\n与就绪电池',
        ],
    },
    {
        'label': '邻域搜索',
        'boxes': [
            'ALNS 解空间\n(划分, 派工顺序 π)',
            '7 个破坏算子\n随机·Shaw·整区',
            '4 个修复算子\n含成组修复',
            '轮盘赌自适应\n算子权重',
            '模拟退火\n接受准则',
        ],
    },
    {
        'label': '约束扫描',
        'boxes': [
            # K 的六档取值必须逐字列出（扫描的是离散档位，不是连续区间），
            # 而 `K∈{20,22,25,30,35,40}` 一行要 196px > 可用 184px，故拆三行。
            'ε-约束扫描\n固定 K∈\n{20,22,25,30,35,40}',
            '逐档最优化\n时效目标',
            '四维非支配前沿\n架次·能耗·时间·时延',
            '时效优先取\nK ≤ 20 档',
            '架次数是手段\n而非目的',
        ],
    },
]

# 落点：取自 results/q2_pareto.csv 首行（K≤20 档）与 6.3 节正文。
BAR = '输出推荐方案：20 架次 · 68.713 kWh · 8732.6 s · 硬违约 0'

# 盒宽由画布宽反解：三栏（竖排标签 + 内容盒）等分剩余的横向空间。
BOX_W = (W - ML - MR - NCOL * (VLAB_W + VLAB_GAP)
         - (NCOL - 1) * (ARROW_W + 2 * ARROW_GAP)) // NCOL
BOX_AVAIL = BOX_W - 14


# ---------------------------------------------------------------- 主流程
def build():
    f = Fig()

    col_w = VLAB_W + VLAB_GAP + BOX_W
    step = col_w + ARROW_W + 2 * ARROW_GAP
    col_h = max(sum(box_h(t) for t in c['boxes']) + BOX_GAP * (len(c['boxes']) - 1)
                for c in COLUMNS)
    H = TOP + col_h + BAR_GAP + BAR_H + BOTTOM

    for j, c in enumerate(COLUMNS):
        x = ML + j * step
        f.check('栏%d 竖标签' % (j + 1), vert(c['label']), VLAB_W)
        f.add('vlab_%d' % j, vert(c['label']), S_VLAB, x, TOP + 4, VLAB_W, col_h)
        bx = x + VLAB_W + VLAB_GAP
        by = TOP
        for k, t in enumerate(c['boxes']):
            f.check('栏%d 盒%d' % (j + 1, k + 1), t, BOX_AVAIL)
            f.add('box_%d_%d' % (j, k), t, S_BOX, bx, by, BOX_W, box_h(t))
            by += box_h(t) + BOX_GAP

    # 栏之间的空心块箭头：竖直居中于三栏的内容区
    ay = TOP + col_h / 2.0 - 13
    for j in range(NCOL - 1):
        f.add('arr_%d' % j, '', S_ARROW,
              ML + j * step + col_w + ARROW_GAP, ay, ARROW_W, 26)

    # 底部通栏横条：右缘与第三栏右缘对齐
    bar_w = (NCOL - 1) * step + col_w
    f.check('底部横条', BAR, bar_w - 14)
    f.add('bar', BAR, S_BOX, ML, TOP + col_h + BAR_GAP, bar_w, BAR_H)

    return f, H, col_h, bar_w


def main():
    f, H, col_h, bar_w = build()
    out = pathlib.Path('figures/fig_q2_algorithm.drawio')
    xml = ('<mxfile host="app.diagrams.net" agent="mathmodel" version="24.7.17" pages="1">\n'
           '  <diagram id="q2algorithm" name="问题二求解框架">\n'
           '    <mxGraphModel dx="%d" dy="%d" grid="0" gridSize="10" guides="1" '
           'tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" '
           'pageWidth="%d" pageHeight="%d" math="0" shadow="0">\n'
           '      <root>\n        <mxCell id="0" />\n        <mxCell id="1" parent="0" />\n'
           '        %s\n      </root>\n    </mxGraphModel>\n  </diagram>\n</mxfile>\n'
           % (W, H, W, H, '\n        '.join(f.cells)))
    out.write_text(xml, encoding='utf-8')
    print('OK %s  画布 %dx%d  宽高比 %.3f  栏高 %d  盒宽 %d  可用 %d  横条宽 %d'
          % (out, W, H, W / float(H), col_h, BOX_W, BOX_AVAIL, bar_w))
    for w in f.warn:
        print('WARN', w)


if __name__ == '__main__':
    main()
