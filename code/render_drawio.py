# -*- coding: utf-8 -*-
# 本程序及代码是在人工智能工具辅助下完成的。
# 工具名称：DeepSeek；版本/型号：DeepSeek V4.1 Flash（API 模型名 deepseek-flash）；
# 开发机构：杭州深度求索人工智能基础技术研究有限公司；
# 版本发布日期：2026 年 9 月 10 日。

"""把 paper-diagram 技能生成的 .drawio 离线渲染成 PNG/PDF。

本机无 draw.io 桌面版且 GitHub 不可达，故直接解析 mxGraph XML 的几何与样式，
用 matplotlib 按 1 px = 1 数据单位重绘。仅覆盖 roadmap/taskflow 模板用到的图元：
矩形（实/虚线框）、纯文本、singleArrow 块箭头、背景板。

用法：python code/render_drawio.py figures/fig_roadmap.drawio --dpi 250
"""
import argparse
import html
import pathlib
import re
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle

rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

PT_PER_PX = 72.0 / 100.0          # 画布按 100 px = 1 inch 建轴
PX = lambda v: float(v) * PT_PER_PX


def style_dict(s):
    out = {}
    for part in (s or '').split(';'):
        if not part:
            continue
        if '=' in part:
            k, v = part.split('=', 1)
            out[k.strip()] = v.strip()
        else:
            # draw.io 约定：首个不带等号的 token 就是形状名（rhombus、text、ellipse…）
            out.setdefault('shape', part.strip())
    return out


def unescape(v):
    if v is None:
        return ''
    v = v.replace('<br>', '\n').replace('&lt;br&gt;', '\n')
    v = re.sub(r'&#x0*a;', '\n', v)
    return html.unescape(v)


def arrow_poly(x, y, w, h, direction, arrow_width, arrow_size):
    """draw.io singleArrow 的多边形顶点。"""
    if direction == 'south':
        x, y, w, h = y, x, h, w            # 转置处理
        pts = arrow_poly(x, y, w, h, 'east', arrow_width, arrow_size)
        return [(py, px) for px, py in pts]
    shaft = h * arrow_width
    head_len = arrow_size * w
    head_len = min(head_len, w * 0.9)
    y0, y1 = y + (h - shaft) / 2.0, y + (h + shaft) / 2.0
    yc = y + h / 2.0
    return [(x, y0), (x + w - head_len, y0), (x + w - head_len, y),
            (x + w, yc), (x + w - head_len, y + h),
            (x + w - head_len, y1), (x, y1)]


def parse_edge(geo):
    """返回 (source, [拐点...], target)。"""
    src = tgt = None
    pts = []
    for child in geo:
        if child.tag == 'mxPoint':
            p = (float(child.get('x', 0)), float(child.get('y', 0)))
            if child.get('as') == 'sourcePoint':
                src = p
            elif child.get('as') == 'targetPoint':
                tgt = p
        elif child.tag == 'Array':
            for pt in child.findall('mxPoint'):
                pts.append((float(pt.get('x', 0)), float(pt.get('y', 0))))
    return src, pts, tgt


def draw_edge(ax, route, st):
    """按 draw.io 的 block 箭头画折线 + 实心箭头。"""
    color = st.get('strokeColor', '#000000')
    lw = PX(st.get('strokeWidth', 1))
    dashed = st.get('dashed') == '1'
    head = st.get('endArrow', 'block') != 'none'
    hl = max(PX(st.get('endSize', 6)) * 2.0, 5.0)      # 箭头长
    hw = hl * 0.62                                      # 箭头半宽

    pts = list(route)
    if head and len(pts) >= 2:
        (x0, y0), (x1, y1) = pts[-2], pts[-1]
        dx, dy = x1 - x0, y1 - y0
        n = (dx * dx + dy * dy) ** 0.5 or 1.0
        ux, uy = dx / n, dy / n
        pts[-1] = (x1 - ux * hl, y1 - uy * hl)          # 线段提前收住，避免箭头处双线
        ax.add_patch(Polygon([(x1, y1),
                              (x1 - ux * hl - uy * hw, y1 - uy * hl + ux * hw),
                              (x1 - ux * hl + uy * hw, y1 - uy * hl - ux * hw)],
                             closed=True, facecolor=color, edgecolor='none', zorder=2))
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs, ys, color=color, linewidth=lw, solid_capstyle='butt',
            linestyle=(0, (8, 5)) if dashed else 'solid', zorder=2)


def render(src, dpi=250, scale=1.0):
    root = ET.parse(src).getroot()
    cells, edges = [], []
    for cell in root.iter('mxCell'):
        geo = cell.find('mxGeometry')
        if geo is None:
            continue
        if cell.get('edge') == '1':
            s, pts, t = parse_edge(geo)
            if s and t:
                edges.append({'style': style_dict(cell.get('style', '')),
                              'route': [s] + pts + [t]})
            continue
        if geo.get('as') != 'geometry':
            continue
        style = style_dict(cell.get('style', ''))
        if style.get('shape') == 'singleArrow' or 'fillColor' in style or 'strokeColor' in style:
            cells.append({
                'id': cell.get('id'), 'text': unescape(cell.get('value')),
                'style': style,
                'x': float(geo.get('x', 0)), 'y': float(geo.get('y', 0)),
                'w': float(geo.get('width', 0)), 'h': float(geo.get('height', 0)),
            })

    W = max([c['x'] + c['w'] for c in cells] +
            [p[0] for e in edges for p in e['route']])
    H = max([c['y'] + c['h'] for c in cells] +
            [p[1] for e in edges for p in e['route']])

    fig = plt.figure(figsize=(W / 100.0, H / 100.0), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis('off')

    for e in edges:
        draw_edge(ax, e['route'], e['style'])

    for c in cells:
        st, x, y, w, h = c['style'], c['x'], c['y'], c['w'], c['h']
        fill, stroke = st.get('fillColor', 'none'), st.get('strokeColor', 'none')
        fill = None if fill in ('none', '') else fill
        edge = None if stroke in ('none', '') else stroke
        dashed = st.get('dashed') == '1'
        lw = PX(st.get('strokeWidth', 1))

        if st.get('shape') == 'singleArrow':
            pts = arrow_poly(x, y, w, h, st.get('direction', 'east'),
                             float(st.get('arrowWidth', 0.6)),
                             float(st.get('arrowSize', 0.42)))
            # 空心块箭头（fillColor=#ffffff + strokeColor）是论文总技术路线图的通用语汇，
            # 这里必须同时描边，否则白箭头在白底上什么都看不见。
            ax.add_patch(Polygon(pts, closed=True, facecolor=fill or '#222222',
                                 edgecolor=edge or 'none',
                                 linewidth=lw if edge else 0, zorder=2))
            continue

        ls = (0, (8, 5)) if dashed else 'solid'
        if st.get('shape') == 'rhombus':
            pts = [(x + w / 2, y), (x + w, y + h / 2), (x + w / 2, y + h), (x, y + h / 2)]
            ax.add_patch(Polygon(pts, closed=True, facecolor=fill or 'none',
                                 edgecolor=edge or 'none', linewidth=lw,
                                 linestyle=ls, zorder=1))
        elif fill or edge:
            radius = 0.0
            if st.get('rounded') == '1':
                radius = min(float(st.get('arcSize', 15)) / 100.0 * min(w, h), min(w, h) / 2.0)
            if radius > 0.5:
                ax.add_patch(FancyBboxPatch(
                    (x + radius, y + radius), w - 2 * radius, h - 2 * radius,
                    boxstyle=f'round,pad={radius}', facecolor=fill or 'none',
                    edgecolor=edge or 'none', linewidth=lw, linestyle=ls, zorder=1))
            else:
                ax.add_patch(Rectangle((x, y), w, h, facecolor=fill or 'none',
                                       edgecolor=edge or 'none', linewidth=lw,
                                       linestyle=ls, zorder=1))

        if not c['text']:
            continue

        fs = float(st.get('fontSize', 12))
        body = st.get('fontStyle', '0')
        weight = 'bold' if ('1' in body.split('=')[0]) else 'normal'
        color = st.get('fontColor', '#222222')
        align = st.get('align', 'center')
        va = st.get('verticalAlign', 'middle')
        pad = float(st.get('spacingLeft', 4)) if align == 'left' else 0
        tx = x + pad if align == 'left' else x + w / 2.0
        if va == 'middle':
            ty, tva = y + h / 2.0, 'center'
        else:
            ty, tva = y + 3, 'top'

        ax.text(tx, ty, c['text'], fontsize=fs * PT_PER_PX, fontweight=weight,
                color=color, ha=align if align in ('left', 'center') else 'left',
                va=tva, linespacing=(fs + 3) / fs, zorder=3)

    out_png = pathlib.Path(src).with_suffix('.png')
    fig.savefig(out_png, dpi=dpi, facecolor='white')
    fig.savefig(pathlib.Path(src).with_suffix('.pdf'), facecolor='white')
    plt.close(fig)
    print(f'OK {out_png.name}  canvas={W:.0f}x{H:.0f}  dpi={dpi}  '
          f'pixels={fig.get_size_inches()[0]*dpi:.0f}x{fig.get_size_inches()[1]*dpi:.0f}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('drawio')
    ap.add_argument('--dpi', type=int, default=250)
    a = ap.parse_args()
    render(a.drawio, a.dpi)
