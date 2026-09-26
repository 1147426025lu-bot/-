# -*- coding: utf-8 -*-
# 本程序及代码是在人工智能工具辅助下完成的。
# 工具名称：DeepSeek；版本/型号：DeepSeek V4.1 Flash（API 模型名 deepseek-flash）；
# 开发机构：杭州深度求索人工智能基础技术研究有限公司；
# 版本发布日期：2026 年 9 月 10 日。

"""生成论文 7.7 节旁「逐箱交付对照」表（只读派生）。

为什么要有这个脚本
------------------
第三问改了运输侧的时刻表（错峰推迟 3 个架次），于是必须回答一个会被追问的问题：
\textbf{零迟到是不是靠把急着要的箱子挪到后面换来的？} 这个问题只能逐箱回答——
看被推迟的那几个架次上装的是什么箱、它们离硬时限还有多远，同时看最紧的那个箱子
所在的架次到底动没动。把两张逐箱表（问题二基线、问题三交付）按货箱编号对起来，
一眼就能看出来。

三项硬门禁（不通过即拒绝落盘，而不是照写一张好看的表）
  1) 两表各 80 行、货箱编号集合逐位相同（无孤儿、无重复）；
  2) 同一货箱在两表里的\textbf{架次编号}必须完全相同——第三问只后移开始时刻、
     不改组批与路线，若架次编号对不上，说明「固定问题二组批与路线」这句话不成立，
     本表连同正文的相应表述都要重写；
  3) 全表 0 个迟到箱（与 q3_metrics.csv 的「迟到箱数」一致）。

产物
  1) `sections/generated_deliverytable.tex`（被 `sections/7_problem3.tex` \\input）；
  2) `results/delivery_stats.csv`（一行一指标），供 `code/paper_metrics.py` 取数。
为什么还要第 2 份：正文要讲「被推迟的 6 个箱里只有 2 个带硬时限」这类\textbf{跨两张表
统计出来}的事实。若把 6、2、7632.8 直接写在 LaTeX 里，一旦重跑换了排班，生成的表会
跟着变而正文不会——这正是本项目反复出过的「正文 63.180、代码 59.236」那类事故。
故把统计量也落盘，正文一律走宏。

用法：python code/make_delivery_table.py
"""
from __future__ import annotations

import os
import sys

sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
RES = os.path.join(ROOT, 'results')
OUT_TEX = os.path.join(ROOT, 'sections', 'generated_deliverytable.tex')
OUT_STATS = os.path.join(RES, 'delivery_stats.csv')


def _csv(name):
    return pd.read_csv(os.path.join(RES, name), encoding='utf-8-sig')


def build():
    q2 = _csv('q2_box_delivery.csv')
    q3 = _csv('q3_box_delivery.csv')
    st = _csv('q3_stagger.csv')

    for nm, df in (('q2_box_delivery.csv', q2), ('q3_box_delivery.csv', q3)):
        if len(df) != 80 or df['货箱编号'].duplicated().any():
            raise AssertionError('%s 应为 80 行且货箱编号不重复，实测 %d 行、重复 %d 个'
                                 % (nm, len(df), int(df['货箱编号'].duplicated().sum())))
    a, b = set(q2['货箱编号']), set(q3['货箱编号'])
    if a != b:
        raise AssertionError('两表货箱集合不一致：仅问题二有 %s，仅问题三有 %s'
                             % (sorted(a - b), sorted(b - a)))

    m = q2[['货箱编号', '架次编号', '交付完成时刻s']].merge(
        q3[['货箱编号', '架次编号', '类别', '是否首批', '硬时限时刻s',
            '交付完成时刻s', '硬时限余量s', '是否迟到']],
        on='货箱编号', suffixes=('_2', '_3'))
    # 门禁 2：架次编号必须逐行相同（第三问固定组批与路线、只后移时刻）
    bad = m[m['架次编号_2'] != m['架次编号_3']]
    if len(bad):
        raise AssertionError('有 %d 个货箱在两表里的架次编号不同（如 %s），'
                             '「第三问固定问题二组批与路线」不成立，本表与正文须重写'
                             % (len(bad), bad['货箱编号'].tolist()[:5]))
    # 门禁 3：零迟到
    late = int((m['是否迟到'].astype(str).str.strip() == '是').sum())
    if late:
        raise AssertionError('逐箱表里有 %d 个迟到箱，与「零迟到」的交付结论冲突' % late)

    m['推迟s'] = m['交付完成时刻s_3'] - m['交付完成时刻s_2']
    # 错峰架次：只列真的被推迟了的（推迟 s > 0），不用「错峰方案存在」代替
    stag = st[st['推迟s'] > 1e-6].set_index('架次编号')['推迟s'].to_dict()

    hard = m[m['硬时限时刻s'].notna()].sort_values('硬时限余量s')
    return m, hard, stag


def stats(m, hard, stag):
    """正文要引用的跨表统计量。逐项都带一句口径，供 results/delivery_stats.csv 备查。"""
    delayed = m[m['推迟s'] > 1e-6]
    delayed_hard = delayed[delayed['硬时限时刻s'].notna()]
    # 「被推迟的架次里有几个一个带硬时限的箱子都没载」——这是「推迟量落在不急的箱子上」
    # 的直接度量，比单看箱数更能说明问题：整个架次都没有时限压力。
    trips_with_hard = set(hard['架次编号_3'])
    stag_loose = sorted(set(stag) - trips_with_hard)
    tight = hard.iloc[0]
    rows = [
        ('有硬时限货箱数', len(hard), '个', 'q3_box_delivery.csv 中 硬时限时刻s 非空的行数'),
        ('首批货箱数', int((hard['是否首批'] == '是').sum()), '个',
         '上者中 是否首批=是 的行数'),
        ('被推迟货箱数', len(delayed), '个',
         'q2/q3 逐箱表按货箱编号相减后 推迟s > 0 的行数'),
        ('被推迟且带硬时限货箱数', len(delayed_hard), '个',
         '上者中 硬时限时刻s 非空的行数'),
        ('被推迟架次数', len(stag), '个', 'q3_stagger.csv 中 推迟s > 0 的行数'),
        ('无非时限箱的被推迟架次数', len(stag_loose), '个',
         '被推迟架次中，一个带硬时限的货箱都不载者'),
        ('被推迟架次的最紧硬时限余量', float(delayed_hard['硬时限余量s'].min()), 's',
         '被推迟货箱中带硬时限者的硬时限余量最小值'),
        ('最紧箱所在架次的推迟量', float(stag.get(tight['架次编号_3'], 0.0)), 's',
         '硬时限余量最小的货箱，其架次在 q3_stagger.csv 中的推迟量'),
        ('最紧箱硬时限余量', float(tight['硬时限余量s']), 's',
         'q3_box_delivery.csv 硬时限余量s 的最小值（与 q3_metrics.csv 同口径）'),
    ]
    return rows


def _fmt(v, nd):
    return '---' if pd.isna(v) else ('%.*f' % (nd, float(v)))


def _row_tex(r, stag):
    trip = r['架次编号_3']
    mark = '$^{\\dagger}$' if trip in stag else ''
    return '\t\t%s & %s%s & %s & %s & %s & %s & %s & %s & %s \\\\\n' % (
        r['货箱编号'], trip, mark, r['类别'], r['是否首批'],
        _fmt(r['交付完成时刻s_2'], 1), _fmt(r['交付完成时刻s_3'], 1),
        _fmt(r['推迟s'], 1), _fmt(r['硬时限时刻s'], 1),
        _fmt(r['硬时限余量s'], 1))


TEMPLATE = r"""% 本文件由 code/make_delivery_table.py 生成，请勿手改。
% 行是「有硬时限的货箱」全体（31 个），按硬时限余量升序；每行的两个交付时刻
% 分别取自问题二基线与问题三交付，两表按货箱编号对齐（80/80、0 孤儿）。
\begin{table}[htbp]
	\centering
	\small
	\caption{有硬时限货箱的逐箱交付对照}
	\label{tab:box-delivery}
	% 九列全取按比例 X 列，系数和必须等于 X 列数 9.00（tabularx 的要求）。
	% 列内既有等宽的长编号（S002-MED-01，连字符处可断），也有纯数字列；
	% 不留自然宽度列，避免某一列的 \hsize 被挤成几个 pt、一行只放一个汉字。
	\begin{tabularx}{\textwidth}{@{}>{\hsize=1.45\hsize}X >{\hsize=0.60\hsize}X >{\hsize=0.90\hsize}X >{\hsize=0.60\hsize}X >{\hsize=1.10\hsize}X >{\hsize=1.10\hsize}X >{\hsize=0.90\hsize}X >{\hsize=0.90\hsize}X >{\hsize=1.45\hsize}X@{}}
		\toprule
		货箱编号 & 架次 & 类别 & 首批 & \shortstack{问题二\\交付 s} & \shortstack{问题三\\交付 s} & 推迟 s & \shortstack{硬时限\\时刻 s} & \shortstack{硬时限\\余量 s} \\
		\midrule
%(ROWS)s		\bottomrule
	\end{tabularx}
	\vspace{2pt}
	{\footnotesize $^{\dagger}$ 该架次在第三问中被错峰推迟。}
\end{table}
"""


def main():
    m, hard, stag = build()
    body = ''.join(_row_tex(r, stag) for _, r in hard.iterrows())
    with open(OUT_TEX, 'w', encoding='utf-8') as f:
        f.write(TEMPLATE.replace('%(ROWS)s', body))
    st = stats(m, hard, stag)
    pd.DataFrame(st, columns=['指标', '数值', '单位', '口径']).to_csv(
        OUT_STATS, index=False, encoding='utf-8')

    print('已写出 %s（%d 行）与 %s（%d 项）'
          % (os.path.relpath(OUT_TEX, ROOT), len(hard),
             os.path.relpath(OUT_STATS, ROOT), len(st)))
    for k, v, u, _ in st:
        print('  %-24s %s %s' % (k, v, u))
    print('  被错峰的架次：%s（推迟合计 %.1f s）'
          % ('、'.join('%s %.1f s' % (k, v) for k, v in sorted(stag.items())),
             sum(stag.values())))
    print('  有硬时限的箱：%d 个（首批 %d、医疗物资 %d）'
          % (len(hard), int((hard['是否首批'] == '是').sum()),
             int((hard['类别'] == '医疗物资').sum())))
    tight = hard.iloc[0]
    print('  最紧的一箱：%s（%s）余量 %.3f s，所在架次 %s，该架次推迟 %.1f s'
          % (tight['货箱编号'], tight['类别'], tight['硬时限余量s'],
             tight['架次编号_3'], stag.get(tight['架次编号_3'], 0.0)))
    for k, v in sorted(stag.items()):
        sub = hard[hard['架次编号_3'] == k]
        print('  错峰架次 %s：载有硬时限箱 %d 个，其中最小余量 %s s'
              % (k, len(sub), ('%.1f' % sub['硬时限余量s'].min()) if len(sub) else '（无）'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
