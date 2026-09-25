# -*- coding: utf-8 -*-
# 本程序及代码是在人工智能工具辅助下完成的。
# 工具名称：DeepSeek；版本/型号：DeepSeek V4.1 Flash（API 模型名 deepseek-flash）；
# 开发机构：杭州深度求索人工智能基础技术研究有限公司；
# 版本发布日期：2026 年 9 月 10 日。

"""生成论文《总技术路线图》的 draw.io 源文件。

版式参照竞赛优秀论文的通行画法：竖版整页、纯黑白、每条带对应一个子问题，
带内按"阶段"分栏（竖排阶段名 + 纵列方法盒），阶段之间用空心块箭头连接，
带与带之间用空心向下块箭头衔接。图形不含标题与图注——节标题即图题。

坐标全部由下面的版式参数推出，改内容不用动坐标；改版式只动 LAYOUT。

用法：python code/make_roadmap.py
输出：figures/fig_roadmap.drawio（再由 render_drawio.py 渲染为 PNG/PDF）
"""
import html
import json
import pathlib

# ---------------------------------------------------------------- 图上数字的来源
# 图里的数字一律从 results/paper_metrics.json 取，与正文同源。手写数字必然走味：
# 本图曾长期写着问题一的 "63.18 kWh"，而正文与结果表早已改成 59.236——图重新
# 生成时不会报错，只是把旧数原样再画一遍。现在缺指标或文件缺失一律直接停。
_HERE = pathlib.Path(__file__).resolve().parent
_METRICS = _HERE.parent / 'results' / 'paper_metrics.json'
_M = None


def _mv(name):
    """取一个指标的数值（原样，不格式化）。"""
    global _M
    if _M is None:
        if not _METRICS.exists():
            raise SystemExit('缺 %s：请先运行各问求解脚本与 fill_results.py 生成指标；'
                             '本脚本不接受手写数字。' % _METRICS)
        _M = json.loads(_METRICS.read_text(encoding='utf-8'))
    if name not in _M:
        raise SystemExit('paper_metrics.json 缺指标 %s（图上数字必须与正文同源）' % name)
    return _M[name]['value']


def _num(name, nd=2):
    """取数值并保留 nd 位小数。"""
    return '%.*f' % (nd, float(_mv(name)))


def _pct(name):
    """取百分比：paper_metrics 里存的是 '66.18\\%' 这种带 LaTeX 转义的字符串。"""
    return str(_mv(name)).replace('\\%', '').strip()

# ---------------------------------------------------------------- 版式参数
W = 816                 # 画布宽（压到 \textwidth=455pt 后，17px 字约 9.5pt）
ML = MR = 20            # 画布左右外边距
PAD = 8                 # 带容器内边距
OUTER_W = 28            # 带左侧竖排标签（"问题一"）宽度
OUTER_GAP = 6
ARROW_W = 26            # 阶段之间的空心块箭头
ARROW_GAP = 7
VLAB_W = 20             # 阶段竖排标签宽度（竖排单字宽 = 字号 17px）
VLAB_GAP = 4
BOX_GAP = 8
BOX_H1 = 34             # 单行盒高
BOX_H2 = 50             # 双行盒高
BAND_GAP = 40           # 带间距（容纳向下块箭头）
TOP = BOTTOM = 16
FS = 17                 # 全图统一字号

NSTAGE = 4              # 统一四栏，保证五条带逐列对齐

# ---------------------------------------------------------------- 图形语汇
S_BOX = ('rounded=0;html=1;whiteSpace=wrap;fillColor=#ffffff;strokeColor=#000000;'
         'strokeWidth=1;fontSize=%d;fontColor=#000000;align=center;verticalAlign=middle;' % FS)
S_VLAB = ('text;html=1;strokeColor=none;fillColor=none;whiteSpace=wrap;'
          'fontSize=%d;fontColor=#000000;align=center;verticalAlign=top;' % FS)
S_OLAB = ('text;html=1;strokeColor=none;fillColor=none;whiteSpace=wrap;'
          'fontSize=%d;fontColor=#000000;align=center;verticalAlign=middle;' % FS)
S_BAND = ('rounded=0;html=1;fillColor=none;strokeColor=#000000;strokeWidth=1;'
          'dashed=1;dashPattern=7 5;')
S_STAGE = ('rounded=0;html=1;fillColor=none;strokeColor=#000000;strokeWidth=1;'
           'dashed=1;dashPattern=4 4;')
S_ARROW = ('shape=singleArrow;direction=east;html=1;fillColor=#ffffff;'
           'strokeColor=#000000;strokeWidth=1;arrowWidth=0.42;arrowSize=0.30;')
S_DOWN = ('shape=singleArrow;direction=south;html=1;fillColor=#ffffff;'
          'strokeColor=#000000;strokeWidth=1;arrowWidth=0.55;arrowSize=0.40;')


# ---------------------------------------------------------------- 图的内容
# 每条带：outer = 左侧竖排标签；stages = 四个阶段（label 竖排阶段名，boxes 纵列方法盒）
BANDS = [
    {
        'outer': '公共模型',
        'stages': [
            {'label': '地形净空', 'boxes': [
                'DEM 像元遍历',
                '最高地形\n+ 50 m',
                '爬升 / 下降\n高度 h+, h-',
            ]},
            {'label': '航程能耗', 'boxes': [
                '3/2 次幂\n航程律',
                '三段航时累加',
                '巡航 + 爬升\n能耗分解',
            ]},
            {'label': '电池周转', 'boxes': [
                '两阶段充电',
                'SOC 回充\n周转',
            ]},
            {'label': '通信链路', 'boxes': [
                '自由空间损耗\nITU-R P.525',
                '视线遮挡判定',
                '双向链路门限',
            ]},
        ],
    },
    {
        'outer': '问题一',
        'stages': [
            {'label': '最大载荷', 'boxes': [
                '二分法求\n最大安全载荷',
                'C 型 S008',
                '降至\n58.57 kg',
            ]},
            {'label': '货箱组批', 'boxes': [
                '同质类多重集\n动态规划',
                '双路径 ILP\n交叉复核',
            ]},
            {'label': '方案权衡', 'boxes': [
                '大区用 C\n小区用 B',
                '%s 架次\n%s kWh' % (_mv('QoneTrips'), _num('QoneEnergy')),
            ]},
            # 数值取自 results/q1_sensitivity.csv：C 型均值 78.69 kg(ρ=0.10) → 52.71 kg(ρ=0.40)，
            # 即 ρ 每提升 10%，平均最大安全载荷下降 (78.69-52.71)/3 ≈ 8.7 kg。
            {'label': '灵敏度', 'boxes': [
                '返航余量扫描',
                '余量每增 10%',
                'C 型降\n8.7 kg',
            ]},
        ],
    },
    {
        'outer': '问题二',
        'stages': [
            {'label': '架次构造', 'boxes': [
                '硬时限\n入硬约束',
                '构造基线\n离散事件调度',
                'ALNS 破坏\n修复搜索',
            ]},
            {'label': '机型调度', 'boxes': [
                '枚举三种机型',
                '最早空闲机\n就绪电池',
                '离散事件调度',
            ]},
            {'label': '求解结果', 'boxes': [
                '%s 架次\n零违约' % _mv('QtwoTrips'),
                '%s kWh\n%d s' % (_num('QtwoEnergy'), round(float(_mv('QtwoMakespan')))),
            ]},
            {'label': '方案对比', 'boxes': [
                'ε-约束扫描\nK≤20…40',
                '两层精确验证\n间隙全为 0',
            ]},
        ],
    },
    {
        'outer': '问题三',
        'stages': [
            {'label': '直连判定', 'boxes': [
                '轨迹分步采样',
                '视线遮挡\n与链路预算',
                # 口径与 q3_coverage.csv 一致（直连 / 中继 / 中断三项占比）。不写「盲区」：
                # 这 33.82% 由中继覆盖，中断为 0（同带末栏），写成「盲区」会与之冲突。
                '%s%% 直连\n%s%% 中继' % (_pct('QthreeDirect'), _pct('QthreeRelayFrac')),
            ]},
            {'label': '中继选址', 'boxes': [
                '候选悬停点\n网格生成',
                '集合覆盖\n整数规划',
                '三站离地高度\n150/230/190m',
            ]},
            {'label': '中继调度', 'boxes': [
                '中继 %s 架次' % _mv('QthreeRelays'),
                '%s kWh' % _num('QthreeRelayEnergy'),
            ]},
            {'label': '零中断', 'boxes': [
                '运输开始时刻\n入决策变量',
                '中继链\n级联修复',
                '中断\n%s%%' % _pct('QthreeGap'),
            ]},
        ],
    },
    {
        'outer': '问题四',
        'stages': [
            {'label': '分区约束', 'boxes': [
                '并查集绑定',
                '同架次服务区',
                '不可分割单元',
            ]},
            {'label': '任务分组', 'boxes': [
                '受限增长串\n完全枚举',
                '分区数\n2047 / 86526',
            ]},
            {'label': '峰值并发', 'boxes': [
                '分组独立执行',
                '峰值资源需求',
            ]},
            # 两种分组的不可行成因不同，这条带必须把它们分开写：K=3 的中继无人机是
            # 结构性短缺（每个含失效区间的组至少占 1 架，3 组的下界 3 已超过库存 2），
            # K=2 则不是任何单类短缺，纯粹来自「没有分区能同时取到八类最小值」的联合约束。
            {'label': '资源缺口', 'boxes': [
                'K=3 中继\n结构性缺1架',
                '其余各类\n非结构性缺',
                '库存可行\n分区数为 0',
            ]},
        ],
    },
]


# ---------------------------------------------------------------- 工具函数
def esc(t):
    """转义为 XML 属性值：先转义原文，再把换行换成 draw.io 的换行实体。"""
    return html.escape(t, quote=True).replace('\n', '&lt;br&gt;')


def text_w(line, fs=FS):
    """估算一行文字的像素宽。

    系数按本图渲染器的实测结果标定：SimHei 的汉字与拉丁字形都比 draw.io 的
    名义宽度略宽（汉字约 1.07 字身、半角约 0.55 字身），用 1.0/0.5 会漏报贴边。
    """
    return sum(fs * 1.07 if ord(c) > 0x2E7F else fs * 0.55 for c in line)


def vert(t):
    """竖排标签：逐字堆叠。不能改用 draw.io 的 horizontal=0——中文字形会躺倒。"""
    return '\n'.join(t)


def box_h(t):
    """盒高随行数增长：单行 34px，每多一行加 16px。"""
    return BOX_H1 + (t.count('\n')) * 16


class Fig:
    def __init__(self):
        self.cells = []
        self.warn = []

    def add(self, cid, value, style, x, y, w, h, vertex=True, edge=False):
        self.cells.append(
            '<mxCell id="%s" value="%s" style="%s" %s="1" parent="1">\n'
            '  <mxGeometry x="%.1f" y="%.1f" width="%.1f" height="%.1f" as="geometry" />\n'
            '</mxCell>' % (cid, esc(value), style, 'edge' if edge else 'vertex',
                           x, y, w, h))

    def check(self, tag, t, avail):
        for ln in t.split('\n'):
            if text_w(ln) > avail:
                self.warn.append('%s 文字超框：%r 需要 %.0fpx，可用 %.0fpx'
                                 % (tag, ln, text_w(ln), avail))


# ---------------------------------------------------------------- 主流程
def build():
    f = Fig()

    stage_w = VLAB_W + VLAB_GAP + BOX_W
    stage_x0 = ML + PAD + OUTER_W + OUTER_GAP
    step = stage_w + 2 * ARROW_GAP + ARROW_W

    # 各带高度取所有带的最大值，保证五条带等高、逐列对齐
    band_h = 0
    for b in BANDS:
        hmax = 0
        for st in b['stages']:
            h = sum(box_h(t) for t in st['boxes']) + BOX_GAP * (len(st['boxes']) - 1)
            hmax = max(hmax, h)
        band_h = max(band_h, hmax + 2 * PAD)

    H = TOP + len(BANDS) * band_h + (len(BANDS) - 1) * BAND_GAP + BOTTOM

    # 先画所有容器，再画内容，最后画箭头——渲染器按文档顺序叠放
    for i, b in enumerate(BANDS):
        y = TOP + i * (band_h + BAND_GAP)
        f.add('band_%d' % i, '', S_BAND, ML, y, W - ML - MR, band_h)
    for i, b in enumerate(BANDS):
        y = TOP + i * (band_h + BAND_GAP)
        for j in range(NSTAGE):
            f.add('stage_%d_%d' % (i, j), '', S_STAGE, stage_x0 + j * step,
                  y + PAD, stage_w, band_h - 2 * PAD)

    for i, b in enumerate(BANDS):
        y = TOP + i * (band_h + BAND_GAP)
        f.check('带%d 左侧' % (i + 1), vert(b['outer']), OUTER_W)
        f.add('olab_%d' % i, vert(b['outer']), S_OLAB, ML + PAD, y + PAD,
              OUTER_W, band_h - 2 * PAD)
        for j, st in enumerate(b['stages']):
            sx = stage_x0 + j * step
            f.check('带%d 阶段%d' % (i + 1, j + 1), vert(st['label']), VLAB_W)
            f.add('vlab_%d_%d' % (i, j), vert(st['label']), S_VLAB, sx, y + PAD + 4,
                  VLAB_W, band_h - 2 * PAD)
            by = y + PAD
            for k, t in enumerate(st['boxes']):
                f.check('带%d 阶段%d' % (i + 1, j + 1), t, BOX_AVAIL)
                f.add('box_%d_%d_%d' % (i, j, k), t, S_BOX,
                      sx + VLAB_W + VLAB_GAP, by, BOX_W, box_h(t))
                by += box_h(t) + BOX_GAP

    # 阶段之间的空心块箭头：竖直居中于该带的盒区
    ay = TOP + band_h / 2.0 - 13
    for i in range(len(BANDS)):
        for j in range(NSTAGE - 1):
            x = stage_x0 + j * step + stage_w + ARROW_GAP
            f.add('arr_%d_%d' % (i, j), '', S_ARROW, x, ay, ARROW_W, 26)

    # 带与带之间的空心向下块箭头
    for i in range(len(BANDS) - 1):
        y = TOP + (i + 1) * band_h + i * BAND_GAP + 5
        f.add('down_%d' % i, '', S_DOWN, W / 2.0 - 15, y, 30, BAND_GAP - 10)

    return f, H


# 盒宽由画布宽反解：四栏（竖排标签 + 方法盒）等分剩余的横向空间。
BOX_W = (W - MR - PAD - ML - PAD - OUTER_W - OUTER_GAP
         - NSTAGE * (VLAB_W + VLAB_GAP)
         - (NSTAGE - 1) * (ARROW_W + 2 * ARROW_GAP)) // NSTAGE
# 盒内可用宽度。SimHei 实排比下面的字宽模型略宽，故再留 6px 安全余量，
# 否则中文行会被盒子右边线切掉。
BOX_AVAIL = BOX_W - 14


def main():
    f, H = build()
    out = pathlib.Path('figures/fig_roadmap.drawio')
    xml = ('<mxfile host="app.diagrams.net" agent="mathmodel" version="24.7.17" pages="1">\n'
           '  <diagram id="roadmap" name="总技术路线图">\n'
           '    <mxGraphModel dx="%d" dy="%d" grid="0" gridSize="10" guides="1" '
           'tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" '
           'pageWidth="%d" pageHeight="%d" math="0" shadow="0">\n'
           '      <root>\n        <mxCell id="0" />\n        <mxCell id="1" parent="0" />\n'
           '        %s\n      </root>\n    </mxGraphModel>\n  </diagram>\n</mxfile>\n'
           % (W, H, W, H, '\n        '.join(f.cells)))
    out.write_text(xml, encoding='utf-8')
    print('OK %s  画布 %dx%d  盒宽 %d  带高 %d' % (out, W, H, BOX_W, (H - TOP - BOTTOM
          - (len(BANDS) - 1) * BAND_GAP) // len(BANDS)))
    for w in f.warn:
        print('WARN', w)


if __name__ == '__main__':
    main()
