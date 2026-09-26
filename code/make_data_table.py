# -*- coding: utf-8 -*-
# 本程序及代码是在人工智能工具辅助下完成的。
# 工具名称：DeepSeek；版本/型号：DeepSeek V4.1 Flash（API 模型名 deepseek-flash）；
# 开发机构：杭州深度求索人工智能基础技术研究有限公司；
# 版本发布日期：2026 年 9 月 10 日。

"""生成论文 3.x 节的「数据来源与结构」表（只读派生）。

为什么要有这个脚本
------------------
评审看数据表时最先问的三件事是：这份数据是哪来的、字段是什么、规模多大。
手写这张表有两个风险：一是规模数字会随数据更新而过期，二是容易把「数据集里
给了但本文没用」的图层也写成「用到」。故本脚本**运行时实地打开每个文件量 shape**，
并在 `code/` 里逐项指出读取位置；读不到的、`code/` 里指不出的，一律不进表
（数据集里另有村镇点位/水体/水系/道路四个图层，全文代码没有读取，因此单列一行
如实标注「未进入求解链路」，而不是从表里悄悄省掉）。

产物：`sections/generated_datatable.tex`（被 `sections/3_assumptions.tex` \\input）。
用法：python code/make_data_table.py
"""
from __future__ import annotations

import os
import sys

sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

import pandas as pd
import scipy.io as sio

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
sys.path.insert(0, HERE)

import core   # noqa: E402  只借用它解析出来的路径常量，不重算任何模型量

OUT_TEX = os.path.join(ROOT, 'sections', 'generated_datatable.tex')


def _xlsx_shape(name):
    f = os.path.join(core.DATA_BASE, name)
    return pd.read_excel(f, header=None).shape


def _sheet_shape(name, sheet):
    f = os.path.join(core.DATA_BASE, name)
    return pd.read_excel(f, sheet_name=sheet).shape


def _mat_shape(path):
    m = sio.loadmat(path)
    return m['dem'].shape


def _fmt(shape):
    """规模列写成 行×列。TeX 里 × 用 $\\times$，免得在窄列里被当成普通字符排得很宽。"""
    return '%d$\\times$%d' % (shape[0], shape[1])


def build_rows():
    dem = _mat_shape(core.DEM_MAT)
    rows = [
        dict(f='调度中心与服务区.xlsx',
             src='题给附件一。\\texttt{core.py:107} 以 \\texttt{header=None} 读入，'
                 '第 2 行为 O01、第 6--20 行为 S001--S015',
             key='编号、名称、经度、纬度、海拔、常住人口',
             scale=_fmt(_xlsx_shape('调度中心与服务区.xlsx')),
             use='四问'),
        dict(f='运输无人机数据.xlsx',
             src='题给附件一。\\texttt{core.py:126}（机型参数）、'
                 '\\texttt{:155}（电池）、\\texttt{:167}（两阶段充电）分三段解析',
             key='空载质量、最大载货质量、可用装载体积、航程、速度、'
                 '电池可用能量、返航余量 $\\rho$',
             scale=_fmt(_xlsx_shape('运输无人机数据.xlsx')),
             use='四问'),
        dict(f='中继无人机数据.xlsx',
             src='题给附件一。\\texttt{core.py:179}（机身）、\\texttt{:205}（能源组件）、'
                 '\\texttt{:212}（悬停与通信功耗）分三段解析',
             key='起飞质量、悬停功率、通信功率、周转时间、能源组件容量',
             scale=_fmt(_xlsx_shape('中继无人机数据.xlsx')),
             # 问题二的落盘门禁要在等价最优解类内按第三问口径择序，故它也读这一份。
             use='二（门禁）、三、四问'),
        dict(f='物资需求与配送时限.xlsx',
             src='题给附件一。\\texttt{core.py:219} 只读 sheet「逐箱货箱清单」'
                 '（同一文件的其余 sheet 未被代码使用）',
             key='货箱编号、服务区、物资类型、单箱质量与体积、是否首批、'
                 '首批截止时刻、期望送达时刻、应急优先系数',
             scale=_fmt(_sheet_shape('物资需求与配送时限.xlsx', '逐箱货箱清单')),
             use='二、三问'),
        dict(f='通信链路参数.xlsx',
             src='题给附件一。\\texttt{core.py:239} 逐行取值',
             key='载波频率、系统损耗、地形遮挡附加损耗、接收灵敏度、衰落裕量、'
                 '各端发射功率与天线增益',
             scale=_fmt(_xlsx_shape('通信链路参数.xlsx')),
             use='三问'),
        dict(f='镇龙乡及周边 30\\,m DEM.mat',
             src='题给附件二。\\texttt{core.py:259} 经 \\texttt{scipy.io.loadmat} 读入，'
                 '含地理变换与 EPSG 码，用于把经纬度换算为栅格行列',
             key='dem（高程栅格）、longitude、latitude、nodata、transform',
             scale=_fmt(dem),
             use='四问'),
        dict(f='村镇点位／水体／水系／道路（.csv 与 .mat）',
             src='随题给地理数据集提供；\\texttt{code/} 全文无任何读取语句，'
                 '我们的求解与绘图均未使用',
             key='---（未读取）',
             scale='---',
             use='未进入求解链路'),
    ]
    return rows


TEMPLATE = r"""% 本文件由 code/make_data_table.py 生成，请勿手改。
% 规模列的数字是脚本运行时实地打开每个文件量出来的 shape；改动数据后重跑即可。
% 「未进入求解链路」一行是如实标注：数据集里确实给了村镇点位/水体/水系/道路四个图层，
% 但全文代码没有读取，故不从表里省掉，而是明写未用——省掉会让读者以为数据集只有六份。
\begin{table}[htbp]
	\centering
	\small
	\caption{题给数据的来源、结构与用途}
	\label{tab:data-source}
	% 五列都取 X：文件名与「来源与口径」都含不可断的长串（如 .xlsx 前缀），
	% 若给自然宽度的 l/c 列，窄列的 \hsize 会被挤到几个 pt，一个汉字一行。
	% 系数和必须等于 X 列数 5。规模列最窄，但其中最长的一条是 1309×1486。
	\begin{tabularx}{\textwidth}{@{}>{\hsize=1.05\hsize}X >{\hsize=1.20\hsize}X >{\hsize=1.30\hsize}X >{\hsize=0.70\hsize}X >{\hsize=0.75\hsize}X@{}}
		\toprule
		数据文件 & 来源与口径 & 关键字段 & 规模（行$\times$列） & 用途 \\
		\midrule
%(ROWS)s		\bottomrule
	\end{tabularx}
\end{table}
"""


def _row_tex(r):
    return '\t\t%s & %s & %s & %s & %s \\\\\n' % (
        r['f'], r['src'], r['key'], r['scale'], r['use'])


def main():
    rows = build_rows()
    # 行与行之间用 \midrule 分隔，末行之后不补——多一条 \midrule 会紧贴 \bottomrule 成双线。
    body = '\t\t\\midrule\n'.join(_row_tex(r) for r in rows)
    tex = TEMPLATE.replace('%(ROWS)s', body)
    with open(OUT_TEX, 'w', encoding='utf-8') as f:
        f.write(tex)
    print('已写出 %s（%d 行数据）' % (os.path.relpath(OUT_TEX, ROOT), len(rows)))
    for r in rows:
        print('  %-34s %s' % (r['f'], r['scale']))


if __name__ == '__main__':
    main()
