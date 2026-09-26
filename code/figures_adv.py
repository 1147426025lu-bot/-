# -*- coding: utf-8 -*-
# 本程序及代码是在人工智能工具辅助下完成的。
# 工具名称：DeepSeek；版本/型号：DeepSeek V4.1 Flash（API 模型名 deepseek-flash）；
# 开发机构：杭州深度求索人工智能基础技术研究有限公司；
# 版本发布日期：2026 年 9 月 10 日。

"""论文配图（升级版）：统一主题的多面板综合图。

与 figures.py 的分工：figures.py 出地形图、单因素敏感性曲线与中继覆盖地图；
本脚本出问题一~四的多面板综合图。所有数字一律**读** results/*.csv，不在此处
重算模型——图与解同源，才不会出现「图画的是另一套方案」。
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import json
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D
from matplotlib.transforms import offset_copy
import seaborn as sns

# 只复用 core 的两阶段充电公式与机型/电池参数表——是「读参数、套论文 §4 的公式」，
# 不是在此重算模型：架次时刻、机型与电池指派一律来自 results/*.csv。
from core import (charge_time, load_transport_uav_types, load_batteries,
                  load_relay_type, load_relay_batteries)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
RES = os.path.join(ROOT, 'results')
FIG = os.path.join(ROOT, 'figures')
os.makedirs(FIG, exist_ok=True)

sns.set_theme(style='whitegrid', context='paper')
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['SimSun', 'SimHei', 'Microsoft YaHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False
rcParams['savefig.dpi'] = 300
rcParams['figure.dpi'] = 120
# 版面口径（实测，与 figures.py 的同一套算法）：A4 正文宽 = 210-22.5-22.5 = 165 mm
# = 6.50 in；图按 0.95\textwidth = 6.17 in 插入，故**缩放比 = 6.17 / 源图宽(in)**。
# 若源图宽取 15~16 in，缩放比只有 0.39~0.41，12 pt 刻度印出来是 4.9 pt——比正文
# 小一半还多，等于看不清。故所有图的源宽一律压到 9.0~10.8 in（缩放比 0.57~0.69），
# 使 12 pt 刻度落在 7~8 pt、15 pt 标题落在 8~10 pt，与现有 fig_q1_sens2d、
# fig_q2_gantt 的实际印刷字号持平。**新增图时先按这条算一遍源宽，不要凭手感定。**
rcParams['axes.titlesize'] = 15
rcParams['axes.labelsize'] = 13
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12
rcParams['legend.fontsize'] = 11
rcParams['axes.grid'] = True
rcParams['grid.alpha'] = 0.3
# 子图编号统一摆在坐标区**上外方、与左边线对齐**：放进坐标区会压住热力图的首格
# 数值或与图例打架，压住的恰好是要给评委看的数据；摆在左边线外侧又会撞上最上面
# 那个 y 刻度标签。标题一律居中，编号在左、标题在中，留 20 pt 间距互不干扰。
rcParams['axes.titlepad'] = 20

C_A, C_B, C_C = '#4C72B0', '#55A868', '#C44E52'
TYPE_COLOR = {'A': C_A, 'B': C_B, 'C': C_C}
C_DIRECT, C_RELAY, C_OUT = '#4C72B0', '#55A868', '#D1495B'
PANEL = dict(fontsize=17, fontweight='bold')


def rd(name):
    return pd.read_csv(os.path.join(RES, name), encoding='utf-8-sig')


def panel_tag(ax, tag, dx=0.0, dy=1.012, ha='left'):
    """子图编号 (a)(b)… 统一放在坐标区上外方、与左边线对齐。

    3D 轴没有带 transform 的 text()，须走 text2D；且 3D 的 transAxes 是整个投影
    包围盒，左上角正好压着 z 轴名，故 3D 面板由调用处传 dx/dy 收进框内。
    """
    fn = ax.text2D if hasattr(ax, 'text2D') else ax.text
    fn(dx, dy, tag, transform=ax.transAxes, ha=ha, va='bottom', **PANEL)


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), bbox_inches='tight')
    plt.close(fig)
    print('saved', name)


# ---------------------------------------------------------------- 问题一

def fig_q1_pareto():
    """(a) 逐 K 扫描与非支配前沿 (b) 各区「架次-能耗」曲线族 (c) 临界 ρ 阶梯跳变。"""
    p = rd('q1_pareto.csv').sort_values('架次数')
    scan = rd('q1_scan.csv').sort_values('架次数')
    area = rd('q1_pareto_area.csv')
    crit = rd('q1_critical_rho.csv')

    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.6))

    # (a) 逐 K 扫描画在**目标空间**里，而不是「架次数 vs 两个纵轴」。
    # 双纵轴的画法在这里是失败的：能耗的降幅只有 0.17%，作业时间的涨幅却有
    # 269%，同一张坐标轴无论怎么定范围都装不下这两个量级——画出来要么两条
    # 曲线重合、要么前沿那点降耗被压成一条直线。改成 (该档作业时间, 该档能耗)
    # 的散点后，「被支配」就是字面意义上的「点在右上」：除两个前沿点外，每档
    # 的能耗与作业时间都同时更大，读者一眼能看出第 20 档起为何不再划算。
    ax = axes[0]
    t_all = scan['对应作业时间h'].to_numpy(float)
    e_all = scan['最小能耗kWh'].to_numpy(float)
    k_all = scan['架次数'].to_numpy(int)
    ndm = (scan['非支配'] == '是').to_numpy()
    ax.scatter(t_all[~ndm], e_all[~ndm], color='0.72', s=34, zorder=3,
               label='被支配档')
    ax.scatter(t_all[ndm], e_all[ndm], color=C_C, s=150, marker='*', zorder=5,
               edgecolor='black', linewidth=0.8, label='非支配档')
    # 两档前沿点连成虚线，前沿就是这条线的左下端。
    ax.plot(t_all[ndm], e_all[ndm], '--', color=C_C, lw=1.4, zorder=4)
    # 前沿那两档只差 0.46 h、0.10 kWh，逐点标数字会糊成一团，改为一处带箭头的
    # 文字说明；其余档位彼此分得开，照旧标数字。
    fr = np.where(ndm)[0]
    if len(fr):
        ax.annotate('非支配前沿仅两档\n（%s 架次）'
                    % '、'.join(str(k_all[i]) for i in fr),
                    xy=(t_all[fr[-1]], e_all[fr[-1]]),
                    xytext=(t_all[fr[-1]] + 3.2, e_all[fr[-1]] + 0.6),
                    fontsize=10.5, va='center',
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.1))
    for kk in (30, 40, 60, 80):
        i = np.where(k_all == kk)[0]
        if len(i):
            ax.annotate(str(kk), (t_all[i[0]], e_all[i[0]]), textcoords='offset points',
                        xytext=(7, 3), fontsize=10.5)
    ax.set_xlabel('该档能耗最小解的作业时间 (h)')
    ax.set_ylabel('该档最小总能耗 (kWh)')
    ax.legend(loc='upper left', fontsize=9.5)
    ax.set_title('逐架次扫描与非支配前沿')
    panel_tag(ax, '(a)')

    # (b) 各区曲线族：按**机型组合**画（允许混用后组合串形如「B1+C1」，
    # 不再是单一机型）。纯单机型组合按型着色，混用组合画成灰细线。
    ax = axes[1]
    pure = area[~area['机型组合'].str.contains(r'\+')]
    mixed = area[area['机型组合'].str.contains(r'\+')]
    for _, g in mixed.groupby(['服务区', '机型组合']):
        g = g.sort_values('架次数')
        ax.plot(g['架次数'], g['能耗kWh'], '-', color='0.75', lw=0.9, zorder=1)
    for tid in ['A', 'B', 'C']:
        sub = pure[pure['机型组合'].str.startswith(tid)]
        for _, g in sub.groupby('服务区'):
            g = g.sort_values('架次数')
            ax.plot(g['架次数'], g['能耗kWh'], '-', color=TYPE_COLOR[tid],
                    alpha=0.5, lw=1.1, zorder=2)
        ax.plot(sub['架次数'], sub['能耗kWh'], 'o', color=TYPE_COLOR[tid], ms=2.5,
                alpha=0.55, zorder=3)
    ax.set_yscale('log')
    ax.set_xlabel('架次数')
    ax.set_ylabel('架次能耗 (kWh，对数轴)')
    ax.set_title('各服务区「架次–能耗」曲线族')
    hs = [Patch(color=TYPE_COLOR[t], label='纯 %s 型' % t) for t in 'ABC']
    if len(mixed):
        hs.append(Patch(color='0.75', label='混用机型组合'))
    ax.legend(handles=hs, loc='lower right', fontsize=9.5)
    panel_tag(ax, '(b)')

    # (c) 临界 ρ：架次数随返航余量的阶梯跳变（只取全局那几行，机型级的是失效点）
    ax = axes[2]
    gl = crit[crit['服务区'] == '全局'].sort_values('临界ρ')
    xs, ys = [0.10], [int(gl.iloc[0]['跳变前架次'])]
    for _, row in gl.iterrows():
        xs += [row['临界ρ'], row['临界ρ']]
        ys += [int(row['跳变前架次']), int(row['跳变后架次'])]
    xs.append(0.40); ys.append(ys[-1])
    ax.step(xs, ys, where='post', color=C_C, lw=2.4)
    ax.scatter(gl['临界ρ'], gl['跳变后架次'], color='black', zorder=5, s=45)
    # 跳变点的数值一律挂在点的**左上方**：跳变后架次最高到 25，往上写会被轴顶切掉，
    # 而每个跳变点右侧正好是抬升后的水平段，写右边就压在线上。
    ax.set_ylim(17.4, 26.8)
    for _, row in gl.iterrows():
        ax.annotate('%.4f' % row['临界ρ'], xy=(row['临界ρ'], row['跳变后架次']),
                    xytext=(-7, 5), textcoords='offset points', ha='right',
                    fontsize=10.5)
    ax.axvline(0.20, color='gray', ls=':', lw=1.6)
    ax.text(0.203, 23.6, '题目给定 ρ=0.20', rotation=90, fontsize=11, color='gray',
            va='center')
    ax.set_xlabel('返航安全余量 ρ')
    ax.set_ylabel('推荐方案架次数')
    ax.set_title('临界 ρ 与架次跳变')
    panel_tag(ax, '(c)')

    fig.tight_layout(w_pad=3.0)
    save(fig, 'fig_q1_pareto.png')


def fig_q1_payload():
    """(a) 服务区×机型最大安全载荷 (b) q_max 随 ρ (c) ρ–κ 协同扰动。"""
    mp = rd('q1_max_payload.csv')
    sens = rd('q1_sensitivity.csv')
    s2 = rd('q1_sensitivity2d.csv')

    # 高度取到 4.6 in（其余三栏图都只到 3.6~3.9）：(c) 的 ρ 网格有 16 行，行距必须
    # 容得下格内 9.5 pt 的数字——按 3.9 in 出图时行距只有 ~10 pt，实测格内数字上下
    # 粘连；4.6 in 给到 ~15 pt 行距，留出 5 pt 净空。
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 4.6))

    # (a) 热力图：A/B/C 载荷量级差 3 倍，按列归一化才看得出区内差异；格内仍写原值
    ax = axes[0]
    M = mp[['A', 'B', 'C']].values
    im = ax.imshow(M, cmap='YlGnBu', aspect='auto')
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            r, g, b, _ = im.cmap(im.norm(M[i, j]))
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            ax.text(j, i, '%.1f' % M[i, j], ha='center', va='center', fontsize=10.5,
                    color='white' if lum < 0.55 else 'black')
    ax.set_xticks(range(3)); ax.set_xticklabels(['A', 'B', 'C'])
    ax.set_yticks(range(len(mp))); ax.set_yticklabels(mp['服务区'], fontsize=9.5)
    ax.set_xlabel('机型'); ax.set_ylabel('服务区')
    ax.set_title('最大安全载荷 q_max (kg)')
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).set_label('kg', fontsize=11)
    panel_tag(ax, '(a)')

    # (b) 三机型平均 q_max 随 ρ 变化（对 15 个服务区取均值）
    ax = axes[1]
    for tid in 'ABC':
        g = sens[sens['机型'] == tid].groupby('ρ')['q_max'].mean()
        ax.plot(g.index * 100, g.values, '-o', color=TYPE_COLOR[tid], ms=6, lw=2.0,
                label='机型 %s' % tid)
    ax.axvline(20, color='gray', ls=':', lw=1.6)
    ax.set_xlabel('返航安全余量 ρ (%)')
    ax.set_ylabel('平均最大安全载荷 (kg)')
    ax.set_title('ρ 对载荷能力的影响')
    ax.legend()
    panel_tag(ax, '(b)')

    # (c) 机型 C 的 (ρ, κ) 协同：单因素曲线看不出参数耦合
    ax = axes[2]
    c = s2[s2['机型'] == 'C']
    piv = c.pivot_table(index='ρ', columns='κ', values='平均q_max')
    sns.heatmap(piv, ax=ax, cmap='rocket_r', annot=True, fmt='.1f',
                annot_kws={'fontsize': 9.5}, cbar_kws={'label': '平均 q_max (kg)'})
    ax.set_xlabel('电池可用能量 κ'); ax.set_ylabel('返航安全余量 ρ')
    ax.set_title('机型 C 的 (ρ, κ) 协同扰动')
    panel_tag(ax, '(c)')

    fig.tight_layout(w_pad=3.0)
    save(fig, 'fig_q1_payload.png')


# ---------------------------------------------------------------- 问题二

def fig_q2_alns():
    """(a) ALNS 收敛 (b) 消融对照 (c) ε-约束前沿。"""
    tr = rd('q2_alns_trace.csv')
    abl = rd('q2_ablation.csv')
    par = rd('q2_pareto.csv')

    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.7))

    # (a) 收敛曲线：这是**推荐档那一次真实运行**的轨迹（F12：原先这里另跑一次
    # 1500 轮，轨迹终值与任何报告过的数字都对不上）。故轨迹的终值就是推荐解的
    # 目标值，参考线取 Pareto 前沿里的最小目标值，两者在末端重合——重合本身
    # 就是「曲线收敛到推荐解」的证据。
    ax = axes[0]
    ax.plot(tr['迭代'], tr['最优目标值'], color=C_A, lw=2.2, label='推荐档实跑')
    # 图例文字必须短：三栏并排后本栏坐标区只有 1.85 in 宽，原来那条
    # 「ε-约束扫描最优 (0.182343)」把图例撑到 305 px，比坐标区本身（222 px）还宽，
    # 于是图例只能向左溢出、看起来像贴在左上角。数值改到正文与图题里给全。
    ax.axhline(par['目标值'].min(), color=C_C, ls='--', lw=2.0,
               label='前沿最优 %.4f' % par['目标值'].min())
    ax.set_xlabel('迭代次数'); ax.set_ylabel('归一化目标值 F')
    ax.set_title('ALNS 收敛轨迹')
    ax.legend(loc='upper right', fontsize=9.5)
    ax3 = ax.twinx()
    ax3.step(tr['迭代'], tr['架次'], where='post', color=C_B, lw=1.6, alpha=0.85)
    ax3.set_ylabel('当前最优架次数', color=C_B)
    ax3.tick_params(axis='y', colors=C_B)
    ax3.grid(False)
    panel_tag(ax, '(a)')

    # (b) 消融：四项指标量纲差几个数量级，一律化成「相对基线」才画得在一张图上。
    # 画成**横向**分组柱：三栏并排后本栏只有 1.85 in 宽，四个消融名竖排时每个只分到
    # 55 px，而「③ALNS(顺序固定)」在 11 pt 下要 90 px —— 实测四个标签互相压住。
    # 横放后组名沿 y 轴一字排开，宽度随图幅自适应，不再有这个问题。
    ax = axes[1]
    base = abl.iloc[0]
    metrics = [('makespan', '完成时刻'), ('total_E', '总能耗'), ('tardiness', '加权时延')]
    # 第⑤项是阶段 11 的「三层全移（v2 引擎）」，协议与④逐项对齐（同轮数、同种子），
    # 故④→⑤的差就是移植本身的收益。名字表必须与 q2_ablation.csv 的行数严格同长：
    # 早先这里写死 4 项，而 y = np.arange(len(abl)) 随行数变，多出的第 5 行就会
    # 标签错位（不报错），故第⑤行落地时同步改成 5 项并加下面的断言把关。
    names = ['①基线', '②仅顺序', '③ALNS\n(顺序固定)', '④ALNS\n全量', '⑤三层\n全移']
    assert len(names) == len(abl), \
        f'消融标签 {len(names)} 项与 q2_ablation.csv 的 {len(abl)} 行不匹配'
    h = 0.26
    y = np.arange(len(abl))
    for k, (col, lab) in enumerate(metrics):
        ratio = abl[col].values / base[col]
        ax.barh(y + (k - 1) * h, ratio, h, label=lab, edgecolor='black', lw=0.4)
        # 加权时延在 ③④ 上真的是 0，零高柱画出来是空白，读者会以为是缺数据。
        # 就地标一个「≈0」，顺带把「时延被压到零」这个结论摆在柱位上。
        for yi, rv in zip(y + (k - 1) * h, ratio):
            if rv < 0.02:
                ax.text(0.012, yi, '≈0', ha='left', va='center', fontsize=9.5,
                        color='#4A4A4A')
    ax.axvline(1.0, color='black', ls='--', lw=1.5)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('相对基线的比值')
    ax.set_title('消融对照（基线 = 1）')
    # 右边留出 1.0~1.6 的空带：最长的一根柱正好到 1.0，空带里放图例与 ③④ 的说明，
    # 两者都不会压到柱子上。
    ax.set_xlim(0, 1.95)
    # 图例摆右上、说明摆右中：最长的一根柱到 1.0，而图例左边缘落在 1.16，两者不碰；
    # 说明缩成三行短句（本栏只有 114 px/单位，一行写 9 个字会顶出右边界）。
    ax.legend(fontsize=9.5, loc='upper right')
    # ③ 的完成时刻优于 ④，而 ④ 的可行集包含 ③ —— 这一点必须在图上点明，不能藏。
    # 只写文字不画箭头：从空带指向 ③ 的箭头会横穿 ③ 自己的「总能耗」柱。
    ax.text(1.0, 2.2, '③ 完成时刻\n优于 ④\n（可行集更大）', fontsize=9, color=C_C,
            va='center', ha='left')
    panel_tag(ax, '(b)')

    # (c) ε-约束扫描：架次数上限 → 实际架次 vs 完成时刻 / 能耗。
    # 六个点的「K≤xx」标注都挂在点上方 9 pt，顶端点若贴着轴顶，标注会被标题切掉。
    #
    # F11：q2_pareto.csv 现在同时含被支配档（末列「非支配」= 否），六档不全是前沿点。
    # 若仍把六点连成一条折线，就等于把被支配档也画成了前沿。故只有非支配档实心连线，
    # 被支配档画成空心点、不连线，并在图内用一行小字点明空心点的含义。
    ax = axes[2]
    ndm = (par['非支配'] == '是').to_numpy()
    xk, ym = par['实际架次'].to_numpy(float), (par['makespan_s'] / 3600).to_numpy(float)
    if ndm.any():
        ax.plot(xk[ndm], ym[ndm], '-o', color=C_A, ms=8, lw=2.2, label='非支配档')
    if (~ndm).any():
        ax.plot(xk[~ndm], ym[~ndm], 'o', mfc='white', mec='0.45', ms=7.5, mew=1.5,
                label='被支配档')
    # 量程按数据取，理由同图 5.3(a)：写死的量程在数据变动后会把点整批挤出坐标轴，
    # 而图上只留下一片空白，看不出是代码错了。
    ms = par['makespan_s'] / 3600
    ax.set_ylim(float(ms.min()) - 0.13 * (float(ms.max()) - float(ms.min())) - 0.06,
                float(ms.max()) + 0.30 * (float(ms.max()) - float(ms.min())))
    # 前沿上架次数相同的档（如 29 与 30）会挤在很近的横坐标上，标注一律挂上方就会
    # 叠字；按横向间距判断，靠近的改挂**下方**靠纵向错开。
    xs = par['实际架次'].to_numpy(float)
    for i, (_, row) in enumerate(par.iterrows()):
        near = any(abs(xs[j] - xs[i]) <= 2 and j != i for j in range(len(xs)))
        dy = -13 if near else 9
        ax.annotate('K≤%d' % row['K上限'], xy=(row['实际架次'], row['makespan_s'] / 3600),
                    xytext=(0, dy), textcoords='offset points', fontsize=9.5, ha='center',
                    va='bottom' if dy > 0 else 'top')
    ax.set_xlabel('实际架次数'); ax.set_ylabel('完成时刻 (h)', color=C_A)
    ax.tick_params(axis='y', colors=C_A)
    ax2 = ax.twinx()
    ax2.plot(xk[ndm], par['能耗kWh'].to_numpy(float)[ndm], '--s', color=C_C, ms=7, lw=2.0,
             label='能耗（非支配档）')
    ax2.plot(xk[~ndm], par['能耗kWh'].to_numpy(float)[~ndm], 's', mfc='white', mec=C_C,
             ms=6.5, mew=1.5, label='能耗（被支配档）')
    ax2.set_ylabel('总能耗 (kWh)', color=C_C)
    ax2.tick_params(axis='y', colors=C_C)
    en = par['能耗kWh']
    ax2.set_ylim(float(en.min()) - 0.10 * (float(en.max()) - float(en.min())) - 1.0,
                 float(en.max()) + 0.18 * (float(en.max()) - float(en.min())))
    ax2.grid(False)
    # 图例只留「实心/空心」这一条关键区分，能耗两条曲线合并说明，避免本栏（1.85 in）
    # 的图例撑到比坐标区还宽——图 6.2(a) 上已踩过这个坑。
    h1, l1 = ax.get_legend_handles_labels()
    ax.legend(h1, l1, fontsize=9, loc='lower right', framealpha=0.92, handlelength=1.6)
    ax.set_title('ε-约束扫描（架次数上限）')
    panel_tag(ax, '(c)')

    fig.tight_layout(w_pad=3.2)
    save(fig, 'fig_q2_alns.png')


def fig_q2_gantt():
    """升级版甘特图：按机型着色，叠加货箱交付硬时限标记。"""
    tp = rd('q2_transport_trips.csv')
    bd = rd('q2_box_delivery.csv')
    tp = tp.sort_values('开始时刻s').reset_index(drop=True)
    ypos = {t: i for i, t in enumerate(tp['架次编号'])}

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for _, r in tp.iterrows():
        y = ypos[r['架次编号']]
        ax.barh(y, r['返回O01时刻s'] - r['开始时刻s'], left=r['开始时刻s'], height=0.62,
                color=TYPE_COLOR[r['机型编号']], edgecolor='black', lw=0.5, zorder=3)
        ax.text(r['返回O01时刻s'] + 120, y, '%s｜%s' % (r['无人机编号'], r['访问服务区顺序']),
                va='center', fontsize=9.5)
    # 交付时刻：末箱交付落在架次末尾，用白点标出，便于与时限对照
    for _, r in bd.iterrows():
        if r['架次编号'] in ypos:
            ax.plot(r['交付完成时刻s'], ypos[r['架次编号']], 'o', ms=3.0,
                    color='white', mec='black', mew=0.4, zorder=4)
    ax.set_yticks(list(ypos.values())); ax.set_yticklabels(list(ypos.keys()))
    ax.invert_yaxis()
    ax.set_xlabel('时间 (s)'); ax.set_ylabel('运输架次')
    ax.set_title('问题二 运输调度甘特图（按机型着色）')
    ax.legend(handles=[Patch(color=TYPE_COLOR[t], label='机型 %s' % t) for t in 'ABC'] +
                      [plt.Line2D([], [], marker='o', ls='', mfc='white', mec='black',
                                  label='货箱交付时刻')],
              loc='upper right')
    ax.set_xlim(0, tp['返回O01时刻s'].max() * 1.30)
    fig.tight_layout()
    save(fig, 'fig_q2_gantt.png')


def _ribbon(ax, x0, x1, ya0, ya1, yb0, yb1, color, alpha=0.5):
    """一根桑基带：从 (x0, [ya0, ya1]) 流向 (x1, [yb0, yb1]) 的三次贝塞尔。"""
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch
    xm = (x0 + x1) / 2.0
    verts = [(x0, ya0), (xm, ya0), (xm, yb0), (x1, yb0),
             (x1, yb1), (xm, yb1), (xm, ya1), (x0, ya1), (x0, ya0)]
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4,
             Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(verts, codes), facecolor=color, edgecolor='none',
                           alpha=alpha, zorder=2))


def fig_q2_sankey():
    """桑基图：服务区 → 运输架次 → 机型。

    用 matplotlib 手绘而不是 plotly：plotly 出 PNG 要走 kaleido 拉起一个真实浏览器，
    本机被识别到的是第三方浏览器、启动还间歇性失败，一旦拉不起来整条出图流水线就断。
    桑基图的结构很规整，手绘几十行即可，且与其余各图共用同一套样式。
    """
    tp = rd('q2_transport_trips.csv')
    bd = rd('q2_box_delivery.csv')
    ttype = dict(zip(tp['架次编号'], tp['机型编号']))
    have = set(bd['架次编号'])
    trips = [t for t in tp.sort_values('开始时刻s')['架次编号'] if t in have]
    services = sorted(bd['服务区编号'].unique())
    types = ['A', 'B', 'C']
    layers = [services, trips, ['机型 ' + t for t in types]]
    tcol = {'A': C_A, 'B': C_B, 'C': C_C}

    cnt = bd.groupby(['服务区编号', '架次编号']).size()
    links = [(s, t, int(n), tcol[ttype[t]]) for (s, t), n in cnt.items() if t in ttype]
    links += [(t, '机型 ' + ttype[t], int(n), tcol[ttype[t]])
              for t, n in bd.groupby('架次编号').size().items() if t in ttype]

    # 节点高度 = 该节点进出流量的较大者；同层节点间留固定间隙，整体缩放到 [0,1]
    size = {}
    for lab in layers[0]:
        size[lab] = sum(v for s, _, v, _ in links if s == lab)
    for lab in layers[1]:
        size[lab] = max(sum(v for s, t, v, _ in links if t == lab),
                        sum(v for s, t, v, _ in links if s == lab))
    for lab in layers[2]:
        size[lab] = sum(v for s, t, v, _ in links if t == lab)

    gap = 0.10
    scale = min(1.0 / (sum(size[n] for n in layer) + gap * max(0, len(layer) - 1))
                for layer in layers)
    top, bot = {}, {}
    for layer in layers:
        h = sum(size[n] for n in layer) * scale + gap * max(0, len(layer) - 1) * scale
        y = 1.0 - (1.0 - h) / 2.0
        for n in layer:
            top[n] = y
            bot[n] = y - size[n] * scale
            y = bot[n] - gap * scale

    xw, xs = 0.055, [0.03, 0.50, 0.955]
    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    cout = {n: top[n] for layer in layers for n in layer}
    cin = {n: top[n] for layer in layers for n in layer}
    # 先画带再画节点，带被节点压住才不会有毛边露在框外
    for s, t, v, col in links:
        hgt = v * scale
        _ribbon(ax, xs[0] + xw if s in size and s in layers[0] else xs[1] + xw,
                xs[1] if t in layers[1] else xs[2],
                cout[s], cout[s] - hgt, cin[t], cin[t] - hgt, col, alpha=0.45)
        cout[s] -= hgt
        cin[t] -= hgt

    for li, layer in enumerate(layers):
        for n in layer:
            fc = ('#C44E52' if li == 0 else '#9A9A9A' if li == 1 else tcol[n[-1]])
            ax.add_patch(Rectangle((xs[li], bot[n]), xw, top[n] - bot[n],
                                   facecolor=fc, edgecolor='black', lw=0.5, zorder=3))
            # 服务区一律写全码（S001…）：省掉「S0」会和服务区编号体系对不上，读者
            # 拿图去核表时找不到对应行。架次标签正好落在飘带上，必须垫白底才读得出。
            if li == 1:
                ax.text(xs[li] + xw + 0.008, (top[n] + bot[n]) / 2, n, va='center',
                        fontsize=9.5, zorder=5,
                        bbox=dict(fc='white', ec='none', alpha=0.82, pad=1.0))
            else:
                ax.text(xs[li] - 0.006 if li == 0 else xs[li] + xw + 0.008,
                        (top[n] + bot[n]) / 2, n, va='center',
                        ha='right' if li == 0 else 'left', fontsize=9.5, zorder=5)

    ax.set_xlim(-0.05, 1.13); ax.set_ylim(-0.02, 1.02)
    ax.axis('off')
    ax.set_title('货箱流向：服务区 → 运输架次 → 机型', fontsize=18)
    ax.legend(handles=[Patch(color=tcol[t], label='机型 %s' % t) for t in types],
              loc='lower center', ncol=3, fontsize=12, frameon=False,
              bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout()
    save(fig, 'fig_q2_flow.png')


def fig_q2_exact():
    """小规模精确验证：Layer A 的最优性间隙 + Layer B 的时序层证书。"""
    ex = rd('q2_exact.csv')
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))

    # (a) Layer A：三个算例上 ALNS 与精确解完全重合，间隙为 0
    ax = axes[0]
    a = ex[ex['层级'] == 'A'].reset_index(drop=True)
    x = np.arange(len(a)); w = 0.34
    ax.bar(x - w / 2, a['精确能耗'], w, label='精确解 (ILP)', color=C_A,
           edgecolor='black', lw=0.5)
    ax.bar(x + w / 2, a['ALNS能耗'], w, label='ALNS', color=C_C, edgecolor='black',
           lw=0.5, hatch='//')
    for i, r in a.iterrows():
        # 加 0.0 是为了把 IEEE 的 -0.0 归一成 0.0：间隙本就是 0（ALNS 命中了精确最优），
        # 印成「-0.00%」会被当成排版错误。
        ax.text(i, max(r['精确能耗'], r['ALNS能耗']) * 1.03,
                '架次 %d vs %d\n能耗间隙 %.2f%%'
                % (r['精确架次'], r['ALNS架次'], r['能耗间隙pct'] + 0.0),
                ha='center', fontsize=10.5)
    ax.set_xticks(x)
    ax.set_xticklabels(['%s\n(%d 箱, %d 列)' % (r['算例'], r['箱数'], r['列数'])
                        for _, r in a.iterrows()], fontsize=10.5)
    ax.set_ylabel('总能耗 (kWh)')
    ax.set_ylim(0, a[['精确能耗', 'ALNS能耗']].values.max() * 1.32)
    ax.set_title('Layer A：组批层最优性间隙')
    ax.legend(fontsize=10.5, loc='upper left')
    panel_tag(ax, '(a)')

    # (b) Layer B：三根柱统一成同一个口径——「贪心解相对精确解的劣化」，否则三种
    # 互不可比的量画在一起就是误导。B1 比完成时刻、B1' 比加权时延，两者的精确解
    # 都已证到最优（gap 0.0000%）；B2 那一根是 MILP 自报的间隙，不是同一件事，
    # 故灰色＋斜纹并单独注明。两个精确最优解各自的硬时限问题也必须标在柱上。
    ax = axes[1]
    abl = rd('q2_ablation.csv')
    g_mk, g_td = float(abl.iloc[0]['makespan']), float(abl.iloc[0]['tardiness'])
    r1 = ex[ex['层级'] == 'B1'].iloc[0]
    r1p = ex[ex['层级'] == "B1'"].iloc[0]
    r2 = ex[ex['层级'] == 'B2'].iloc[0]
    # 劣化统一按 (启发式 − 精确) / 精确 定义，与表 6.x 的「精确相对启发式」同源；
    # 早先柱 0 用 (贪心−精确)/贪心、柱 1 用 (贪心−精确)/精确，同一张图上两种分母，
    # 与表里的数字对不上。
    degrade = [(g_mk - float(r1['回放makespan_s'])) / float(r1['回放makespan_s']) * 100.0,
               (g_td - float(r1p['回放加权时延'])) / float(r1p['回放加权时延']) * 100.0,
               float(r2['上下界间隙']) * 100.0]
    names = ['B1\n完成时刻准则\n(118 二元变量)', "B1'\n加权时延准则\n(118 二元变量)",
             'B2\n自由指派\n(630 二元变量)']
    bars = ax.bar(np.arange(3), degrade, 0.52, color=[C_A, C_B, '#B0B0B0'],
                  edgecolor='black', lw=0.5)
    bars[2].set_hatch('//')
    for i, v in enumerate(degrade):
        ax.text(i, v - 4.2 if i == 2 else v + 1.6, '%.2f%%' % v, ha='center',
                va='top' if i == 2 else 'bottom', fontsize=12.5, fontweight='bold',
                color='white' if i == 2 else 'black')
    # 两条说明各自居中在柱 0 / 柱 1 上，两柱中心相距 393 px。原来的句子第二行
    # 长到 460 px（半宽 230），而中心距只有 393 px —— 实测两行右左相接处叠字。
    # 改写成两行、每行都不超过 8 个全角字宽（≈336 px，半宽 168），留出 60 px 净空。
    for i, s in [(0, '精确解虽已证最优\n但违反 8 项硬时限'),
                 (1, '精确解虽已证最优\nmakespan 反升 4.59%')]:
        ax.text(i, 13, s, ha='center', fontsize=9.5, color=C_C)
    # B2 那根柱画的不是「劣化」而是 MILP 的上下界间隙，量级差一个数量级（90% 对
    # 2.4%/5.8%），且其可行上界本身还劣于贪心。量程按数据放开到柱顶之上，说明文字
    # 一并上移，避免被 y 轴上限切掉——写死 86 会让 90.36% 的柱顶和文字一起消失。
    ytop = max(degrade) * 1.22
    ax.text(2.42, ytop * 0.97, '这是 MILP 的上下界间隙、非劣化\n90 s 内未收敛，且上界反劣于贪心\n不计入最优性结论',
            ha='right', va='top', fontsize=9.5, color=C_C)
    ax.set_xticks(np.arange(3)); ax.set_xticklabels(names, fontsize=10.5)
    ax.set_ylabel('启发式相对精确解的劣化 (%)')
    ax.set_ylim(0, ytop)
    # 右边界放宽到 2.5：B2 那条说明文字右对齐在 x=2.42，默认的 5% 边距只到 2.39，
    # 文字尾巴会伸到坐标框外面去
    ax.set_xlim(-0.55, 2.50)
    ax.set_title('Layer B：给定指派下的调度质量')
    panel_tag(ax, '(b)')

    fig.tight_layout(w_pad=2.6)
    save(fig, 'fig_q2_exact.png')


# ---------------------------------------------------------------- 问题三

def _q3_context():
    """问题三出图需要轨迹采样，故这里才碰求解器；其余图一律只读 CSV。

    F05 之后问题三把**错峰后的最终方案**整份落盘（results/q3_solution.json），
    本函数直接按 trip_id 取回路线、机型、开始时刻与逐箱指派，不再走
    「读问题二方案 → 重解调度 → 按错峰表逐架次相加」这条二次拼装路径：那条
    路径在错峰口径改变时（级联传播、就绪性修复、弃飞）会与结果表悄悄分叉，
    而图上看不出来——图上少一段中继、时刻差几十秒，谁也不会察觉。
    轨迹采样仍调用 q3.sample_trip_trajectory，它只做几何采样、不含调度决策。
    """
    from core import load_data
    from q2 import precompute_geometry
    import q3
    d = load_data()
    d.geo_nodes, d.geo = precompute_geometry(d)
    with open(os.path.join(RES, 'q3_solution.json'), encoding='utf-8') as f:
        sol = json.load(f)
    box_by_id = {b['id']: b for bs in d.boxes_by_service.values() for b in bs}
    # tid 与 base_start 一并带出：`原开始时刻s` 在 CSV 里只留 3 位小数，而问题二
    # 基线时刻正是「优化前后对照图」第 (a) 行要用的量，取全精度的 base_start 才不会
    # 让重算出的直连/中断在边界采样点上因舍入而翻面。
    assignment = [dict(tid=t['trip_id'], type=t['type'], route=list(t['route']),
                       start=float(t['start']), base_start=float(t['base_start']),
                       boxes_at={s: [box_by_id[x] for x in ids]
                                 for s, ids in t['box_ids_by_service'].items()})
                  for t in sorted(sol['transport_trips'], key=lambda t: t['start'])]
    return d, q3, assignment


def fig_q3_timeline():
    """通信状态时序带图 + 中继机占用：中断归零的直观证据。"""
    ph = rd('q3_comm_phases.csv')
    rt = rd('q3_relay_trips.csv')
    trips = sorted(ph['运输架次编号'].unique())
    ypos = {t: i for i, t in enumerate(trips)}
    cmap = {'直连': C_DIRECT, '中继': C_RELAY, '中断': C_OUT}

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.2), sharex=True,
                             gridspec_kw=dict(height_ratios=[3, 1.15], hspace=0.12))

    # 上：每条运输架次的通信状态带。三色带里「中断」一格不出现，就是零中断的直接证据
    ax = axes[0]
    for _, r in ph.iterrows():
        if r['运输架次编号'] not in ypos:
            continue
        ax.barh(ypos[r['运输架次编号']], r['结束时刻s'] - r['开始时刻s'],
                left=r['开始时刻s'], height=0.72, color=cmap[r['保障方式']],
                edgecolor='none', zorder=3)
    ax.set_yticks(list(ypos.values())); ax.set_yticklabels(trips)
    ax.invert_yaxis()
    ax.set_ylabel('运输架次')
    ax.set_title('通信状态时序：直连 / 中继 / 中断（Δ t = 1 s 逐时刻判定）')
    cnt = ph['保障方式'].value_counts()
    handles = [Patch(color=cmap[k], label='%s（%d 段）' % (k, cnt.get(k, 0)))
               for k in ['直连', '中继', '中断']]
    ax.legend(handles=handles, loc='lower right', ncol=3, fontsize=10)
    # 结论不写进图注（图注只留图题），但**要摆在右上空白**而不是左下：左下那条
    # y=0.05 的带正好压在 T20 的行上，而 T20 的架次带从横轴 24% 处就开始了，
    # 13 pt 的 15 个字横向要走 32% 的宽度，实测文字尾巴糊在蓝色带上。
    # 右上（T01~T13 各行）的架次带最远只到 32%，文字右对齐放在 98% 处不会碰到。
    ax.text(0.985, 0.90, '中断段 0 段 —— 全过程连续通信', ha='right', va='center',
            transform=ax.transAxes, fontsize=13, color=C_RELAY, fontweight='bold')
    panel_tag(ax, '(a)')

    # 下：两架中继机的航链（首尾相接，每趟含准备/建链/周转固定开销）
    ax = axes[1]
    ruavs = sorted(rt['中继无人机编号'].unique())
    ry = {u: i for i, u in enumerate(ruavs)}
    # 一行 = 一次悬停站服务，而出动级的 `开始时刻s`/`返回O01时刻s` 在同一次出动的
    # 各行上取同值、`服务结束时刻s` 才是逐站推进的。逐行画会在同一个 y、同一个 x
    # 上叠出多根等长条，并把几个 `R03@S05` 标签压成一团（阶段 13 起一次出动可连做
    # 多站）。故先按 `出动编号` 归并成一次出动一根条：实体段取 [首次开始, 末次服务
    # 结束]、返航段取 [末次服务结束, 返回O01]，标签把该趟服务过的站一次列全。
    if '出动编号' in rt.columns:
        bars = [(g['中继无人机编号'].iloc[0], float(g['开始时刻s'].min()),
                 float(g['服务结束时刻s'].max()), float(g['返回O01时刻s'].iloc[0]),
                 '%s@%s' % (g['中继架次编号'].iloc[0],
                            '→'.join(str(s) for s in g['悬停站编号'])))
                for _, g in rt.groupby('出动编号', sort=False)]
    else:
        bars = [(r['中继无人机编号'], float(r['开始时刻s']), float(r['服务结束时刻s']),
                 float(r['返回O01时刻s']),
                 '%s@%s' % (r['中继架次编号'], r['悬停站编号']))
                for _, r in rt.iterrows()]
    for uav, t0, svc_end, t_ret, lab in bars:
        y = ry[uav]
        ax.barh(y, svc_end - t0, left=t0,
                height=0.5, color=C_RELAY, edgecolor='black', lw=0.5, zorder=3)
        ax.barh(y, t_ret - svc_end, left=svc_end,
                height=0.5, color=C_RELAY, alpha=0.35, edgecolor='black', lw=0.5, zorder=3)
        ax.text(t0 + 60, y, lab, va='center', fontsize=10, zorder=4)
    ax.set_yticks(list(ry.values())); ax.set_yticklabels(ruavs)
    ax.set_xlabel('时间 (s)'); ax.set_ylabel('中继无人机')
    # 纵轴下限留 1.0（原 -0.7）：图例放 lower right 时其顶边只到 axes 分数 ~0.10，
    # 而 R01 行在 y=0、柱底 -0.25，实测两者相切、图例压住 R06@S03 条及其淡绿返航段。
    # 下限压到 -1.0 后图例整体落进 y < -0.55 的空白，与柱底 -0.25 彻底分离，
    # 不必改 loc，也不必缩字号。
    ax.set_ylim(-1.0, len(ruavs) - 0.3)
    ax.legend(handles=[Patch(color=C_RELAY, label='悬停服务'),
                       Patch(color=C_RELAY, alpha=0.35, label='返航/周转')],
              loc='lower right', ncol=2)
    panel_tag(ax, '(b)')

    save(fig, 'fig_q3_timeline.png')


def fig_q3_analysis():
    """(a) 三口径通信占比 (b) 三维悬停站位 (c) 各架次中继时长占比。"""
    cov = rd('q3_coverage.csv')
    st = rd('q3_stations.csv')
    ph = rd('q3_comm_phases.csv')
    rel = rd('q3_relay_trips.csv')

    fig = plt.figure(figsize=(10.4, 3.9))

    # (a) 三种方案口径对比：无中继 → 序贯 → 联合优化，中断被逐级压到 0
    ax = fig.add_subplot(1, 3, 1)
    order = ['无中继', '序贯对照', '联合优化']
    cov = cov.set_index('口径').loc[order].reset_index()
    x = np.arange(len(cov))
    b1 = cov['直连时间占比'].values
    b2 = cov['中继时间占比'].values
    b3 = cov['中断时间占比'].values
    ax.bar(x, b1, 0.55, label='直连', color=C_DIRECT, edgecolor='black', lw=0.5)
    ax.bar(x, b2, 0.55, bottom=b1, label='中继', color=C_RELAY, edgecolor='black', lw=0.5)
    ax.bar(x, b3, 0.55, bottom=b1 + b2, label='中断', color=C_OUT, edgecolor='black', lw=0.5)
    for i in range(len(cov)):
        if b3[i] > 1e-9:
            ax.text(i, b1[i] + b2[i] + b3[i] / 2, '%.2f%%' % (b3[i] * 100), ha='center',
                    va='center', fontsize=11.5, color='white', fontweight='bold')
        else:
            ax.text(i, 0.5, '0.00%', ha='center', va='center', fontsize=11.5,
                    color='white', fontweight='bold')
        ax.text(i, b1[i] / 2, '%.1f%%' % (b1[i] * 100), ha='center', va='center', fontsize=11,
                color='white')
    ax.set_xticks(x); ax.set_xticklabels(order)
    ax.set_ylabel('时间占比')
    ax.set_ylim(0, 1.0)
    ax.set_title('通信时间占比（Δt = 1 s 积分）')
    ax.legend(loc='lower right', ncol=3, fontsize=10)
    panel_tag(ax, '(a)')

    # (b) 三维站位：水平位置 + 离地高度，运输轨迹用失效区间点云衬底
    ax = fig.add_subplot(1, 3, 2, projection='3d')
    d, q3, assignment = _q3_context()
    iv = rd('q3_intervals.csv')
    pts_all = []
    for _, r in iv.iterrows():
        k = int(r['运输架次编号'][1:]) - 1
        a = assignment[k]
        pts = q3.sample_trip_trajectory(d, d.transport_types[a['type']], a['route'],
                                        a['boxes_at'], t0=a['start'], dt=30.0)
        sel = [p for p in pts if r['区间起点s'] - 1 <= p[0] <= r['区间终点s'] + 1]
        pts_all += [(p[1], p[2], p[3]) for p in sel]
    P = np.array(pts_all)
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=5, c=C_OUT, alpha=0.35, label='失效区间轨迹点')
    # 站标只写站号：三站水平间距只有 0.01°~0.06°，连「(230 m AGL)」一起写，投影
    # 后长文字会吃到右侧的 z 轴刻度、或者压到另一站的标记上。离地高度改列在左上角
    # 的说明框里——信息一个不少，但不再和任何图元抢位置。
    ax.set_zlim(160, 800)
    # 站标改用**屏幕坐标偏移**（offset points），不用三维偏移。三维偏移量经过
    # elev=22 / azim=-58 的投影后方向会变：S03「远离质心」的那个方向恰好投影成
    # 指向屏幕内侧，文字于是正好落回自己的菱形标记上（截图实测确认被挡住三分之一）。
    # 屏幕偏移与视角解耦，按站序轮转「上 / 右上 / 左上」，白底 bbox 保证压住点云也
    # 读得清。若站点数超过下面 4 个模式，需重新目视一次排布。
    LBL_OFF = [(0, 16, 'center', 'bottom'), (16, 11, 'left', 'bottom'),
               (-16, 11, 'right', 'bottom'), (0, -16, 'center', 'top')]
    for i, (_, r) in enumerate(st.iterrows()):
        ax.scatter(r['悬停经度'], r['悬停纬度'], r['悬停海拔m'], marker='D', s=110,
                   color=C_A, edgecolor='black', depthshade=False)
        dxp, dyp, ha, va = LBL_OFF[i % len(LBL_OFF)]
        # 用 offset_copy 把「投影后的数据坐标」再沿屏幕方向平移若干磅；Annotation
        # 在 3D 轴上不接受三元 xy（实测抛 ValueError），Text3D 则原样使用传入的
        # transform，故这是 3D 轴里做屏幕偏移的可行写法。
        ax.text(r['悬停经度'], r['悬停纬度'], r['悬停海拔m'], r['悬停站编号'],
                transform=offset_copy(ax.transData, fig=fig, x=dxp, y=dyp, units='points'),
                fontsize=11.5, fontweight='bold', color=C_A, ha=ha, va=va,
                bbox=dict(fc='white', ec='none', alpha=0.75, pad=0.6))
    # 说明框摆在 (b) 编号下方：两者都在左上角，贴着放会叠在一起
    ax.text2D(0.02, 0.92,
              '\n'.join('%s 离地 %.0f m' % (r['悬停站编号'], r['悬停离地高度m'])
                        for _, r in st.iterrows()),
              transform=ax.transAxes, fontsize=9.5, va='top', ha='left',
              bbox=dict(fc='white', ec='#BBBBBB', alpha=0.85, boxstyle='round,pad=0.35'))
    ax.set_xlabel('经度 (°)', fontsize=10); ax.set_ylabel('纬度 (°)', fontsize=10)
    ax.set_zlabel('海拔 (m)', fontsize=10)
    ax.tick_params(labelsize=9)
    ax.set_title('中继悬停站三维部署')
    ax.view_init(elev=22, azim=-58)
    panel_tag(ax, '(b)', dx=0.02, dy=0.97)

    # (c) 各运输架次的中继依赖度：谁被中继救回来的一目了然
    ax = fig.add_subplot(1, 3, 3)
    g = ph.groupby(['运输架次编号', '保障方式'])['开始时刻s'].count().unstack(fill_value=0)
    g = g.reindex(sorted(g.index))
    tot = ph.groupby('运输架次编号').apply(
        lambda s: (s['结束时刻s'] - s['开始时刻s']).sum(), include_groups=False)
    relsec = ph[ph['保障方式'] == '中继'].groupby('运输架次编号').apply(
        lambda s: (s['结束时刻s'] - s['开始时刻s']).sum(), include_groups=False)
    frac = (relsec / tot).reindex(g.index).fillna(0.0)
    colors = [C_RELAY if v > 0 else '#BFBFBF' for v in frac.values]
    ax.barh(np.arange(len(frac)), frac.values * 100, color=colors, edgecolor='black', lw=0.4)
    for i, (name, v) in enumerate(frac.items()):
        if v > 0:
            ids = rel[rel['中继架次编号'].isin(
                ph[(ph['运输架次编号'] == name) & (ph['保障方式'] == '中继')]['中继架次编号'])]
            ax.text(v * 100 + 1.0, i, ','.join(sorted(set(ids['悬停站编号']))), va='center',
                    fontsize=9, color=C_A)
    ax.set_yticks(np.arange(len(frac))); ax.set_yticklabels(frac.index, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel('中继保障时长占该架次比例 (%)')
    ax.set_ylabel('运输架次')
    ax.set_title('各架次的中继依赖度')
    panel_tag(ax, '(c)')

    fig.tight_layout(w_pad=2.6)
    save(fig, 'fig_q3_analysis.png')


# ---------------------------------------------------------------- 问题四

RES_LABEL = ['A型运输无人机', 'B型运输无人机', 'C型运输无人机',
             'A型共享电池', 'B型共享电池', 'C型共享电池', '中继无人机', '中继能源组件']
RES_SHORT = ['A机', 'B机', 'C机', 'A电池', 'B电池', 'C电池', '中继机', '中继组件']


def fig_q4_resource():
    """(a) 资源需求雷达（对库存归一化） (b) 缺口柱状。"""
    cmp = rd('q4_comparison.csv').set_index('K')
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.9),
                             subplot_kw=dict(polar=True))
    # 雷达：需求/库存 = 1 的那圈就是库存线，越出去即缺口
    ang = np.linspace(0, 2 * np.pi, len(RES_LABEL), endpoint=False)
    ang_c = np.concatenate([ang, ang[:1]])
    cols = {2: C_A, 3: C_C}
    for K in (2, 3):
        ratios = [cmp.loc[K, '需求_' + k] / cmp.loc[K, '库存_' + k] for k in RES_LABEL]
        vals = ratios + ratios[:1]
        axes[0].plot(ang_c, vals, '-o', lw=2.2, ms=6, color=cols[K], label='K = %d' % K)
        axes[0].fill(ang_c, vals, color=cols[K], alpha=0.12)
    ones = [1.0] * (len(RES_LABEL) + 1)
    axes[0].plot(ang_c, ones, '--', color='black', lw=1.8, label='库存水平 (=1)')
    axes[0].set_xticks(ang)
    axes[0].set_xticklabels(RES_SHORT, fontsize=11.5)
    axes[0].set_ylim(0, 2.4)
    axes[0].set_yticks([0.5, 1.0, 1.5, 2.0])
    axes[0].set_yticklabels(['0.5', '1.0', '1.5', '2.0'], fontsize=10)
    axes[0].set_title('资源需求 / 库存（>1 即超配）', pad=22)
    # 图例必须**整体挪到极坐标框右外侧**：8 个轴名的角度均分，45° 的「B机」正好
    # 落在方框右上角，而 (1.22, 1.12) 的 upper right 图例左下角也要占这个角，
    # 实测把「B机」压在白色图例底上。改成锚在 (1.02, 1.00) 的 upper left 后，
    # 图例整块在方框之外，B机 恢复可见；tight_layout 会把这块宽度让出来。
    axes[0].legend(loc='upper left', bbox_to_anchor=(1.02, 1.00), fontsize=10.5)
    axes[0].grid(alpha=0.35)
    panel_tag(axes[0], '(a)')

    ax = axes[1]
    ax.remove()
    ax = fig.add_subplot(1, 2, 2)
    x = np.arange(len(RES_LABEL)); w = 0.38
    for i, K in enumerate((2, 3)):
        gap = [cmp.loc[K, '总量缺口_' + k] for k in RES_LABEL]
        ax.bar(x + (i - 0.5) * w, gap, w, color=cols[K], edgecolor='black', lw=0.5,
               label='K = %d' % K)
        for xi, gv in zip(x + (i - 0.5) * w, gap):
            if gv > 0:
                ax.text(xi, gv + 0.05, '%d' % gv, ha='center', fontsize=11.5,
                        fontweight='bold')
    # 8 个类目平分 ~4.2 in 的坐标区，间距只有 ~0.51 in，而「中继组件」4 个全角字
    # 在 11.5 pt 下要 0.64 in —— 实测末尾两个刻度名叠成「中继机中继组件」。
    # 转 40°、右对齐接在刻度下：横向投影 0.64*cos40° = 0.49 in < 0.51 in，正好分开。
    ax.set_xticks(x)
    ax.set_xticklabels(RES_SHORT, fontsize=11, rotation=40, ha='right',
                       rotation_mode='anchor')
    ax.set_ylabel('总量口径缺口（架/组）')
    # 上限随数据走，并留出比「最高柱 + 其数值标注」更大的余量：图例占坐标区高度的
    # 约五分之一，留 1.0 个数据单位即可保证它不压住任何柱顶。写死上限的旧版在
    # 缺口为 2 时把图例压到柱顶的「2」上，在缺口为 1 时又白白空掉半个坐标区。
    gap_max = max(cmp.loc[K, '总量缺口_' + k] for K in (2, 3) for k in RES_LABEL)
    ax.set_ylim(0, max(2.0, gap_max + 1.0))
    ax.set_title('现有库存下的资源缺口')
    # 图例放 upper center：那一带是类目中部，配合上面的余量后不会与任何柱顶相撞。
    ax.legend(fontsize=11, loc='upper center')
    # 红字放在左下方，让开 upper center 的图例与坐标区左上角的 (b) 编号。
    # 分区数按 K 现数、不写死：这里曾写「1023 / 28501」，是早先较粗的原子单元划分下的
    # 数字，比正文 8.3 节与 q4_all_partitions.csv 的 2047 / 86526 少一半还多。
    allp = rd('q4_all_partitions.csv')
    n_part = {K: int((allp['K'] == K).sum()) for K in (2, 3)}
    n_feas = {K: int(cmp.loc[K, '库存可行分区数']) for K in (2, 3)}
    assert n_feas[2] == 0 and n_feas[3] == 0, \
        '库存可行分区数不再是 0（K=2:%d，K=3:%d），本图红字须重写' % (n_feas[2], n_feas[3])
    ax.text(0.03, 0.72, '%d / %d 个分区中\n库存可行者 %d 个'
            % (n_part[2], n_part[3], n_feas[2]),
            transform=ax.transAxes, fontsize=12.5, color=C_C, fontweight='bold', va='top')
    panel_tag(ax, '(b)')

    fig.tight_layout(w_pad=2.0)
    save(fig, 'fig_q4_resource.png')


def _partition_map(ax, d, pos, part, K, tag, R):
    """一个 K 下的最优分区地图：服务区按任务组着色，标题与图例全部从结果表推。

    标题与图例一律**从结果表推**，不写字面量。这里曾写死「R=30」「任务组 1（13 个
    服务区）」「任务组 2（S009, S012）」：阶段 12 重解后实际是 R=29、组 1 有 14 个
    服务区、组 2 只有 S009，三个字面量全部过期，图与正文对不上却没有任何报错——
    正是「图与解同源」这条纪律要防的事故。
    """
    dem = d.dem._dem_nan
    ax.imshow(dem, extent=[d.dem.lon_min, d.dem.lon_max, d.dem.lat_min, d.dem.lat_max],
              origin='upper', cmap='Greys', aspect='auto', alpha=0.45)
    pk = part[part['K'] == K]
    # 组号 1/2/3 固定配色，K=2 时只用前两色——与 K=3 面板并排看时同组号同色
    gcol = {1: '#C44E52', 2: '#4C72B0', 3: '#55A868'}
    for _, r in pk.iterrows():
        for sid in str(r['服务区列表']).split(','):
            if sid in pos:
                ax.plot(*pos[sid], 'o', ms=11, color=gcol[r['任务组编号']],
                        mec='black', mew=0.7, zorder=4)
                # 保留 S 前缀：本图与 fig_q2_flow 是同一批服务区，那边写 S001，
                # 这边若只写 001，读者会以为又是另一套编号。
                ax.text(pos[sid][0], pos[sid][1] + 0.0022, sid, fontsize=7.8,
                        ha='center', zorder=5)
    ax.plot(d.O01['lon'], d.O01['lat'], '*', ms=18, color='gold', mec='black', mew=0.9,
            zorder=6)
    # 写「调度中心」而不是 O01：这个星标旁边就是服务区 S001 的标签，两者都写成
    # 三位数字会看混。标签摆在星标**右侧**（数据坐标 +0.006°，约 15 pt，大于星标
    # 10 pt 的半径）：原先放在正下方会顶住星标的下尖角，实测截图确认压字；而
    # (109.237~109.253, 23.0085) 一带没有任何服务区，横向摆放不会撞上别的标号。
    ax.text(d.O01['lon'] + 0.006, d.O01['lat'], '调度中心', fontsize=9, ha='left',
            va='center', fontweight='bold', zorder=6)
    ax.set_xlim(min(p[0] for p in pos.values()) - 0.01,
                max(p[0] for p in pos.values()) + 0.01)
    # 下边界留 0.030°（原 0.018 不够）：最低的两个点 S014(23.00521) 与 S006(23.0077)
    # 以及调度中心星标（23.00851）都落在图幅下沿，而左下角图例高约 0.49 in
    # ≈ 0.022°。按 -0.018 时图例顶边到 23.0072，实测既压住 S014/S006 的圆点、
    # 也顶住星标下尖角（截图逐点确认）。0.030° 后图例顶边降到 ~22.9975，
    # 与最低圆点的下缘 23.0015 留出 0.004°（≈8 pt）净空。
    ax.set_ylim(min(p[1] for p in pos.values()) - 0.030,
                max(p[1] for p in pos.values()) + 0.012)
    ax.set_xlabel('经度 (°)'); ax.set_ylabel('纬度 (°)')
    ax.set_title('$K=%d$ 最优分区（$R=%d$）' % (K, R))
    ax.legend(handles=[Patch(color=gcol[k],
                             label='任务组 %d（%d 个服务区）'
                                   % (k, len(str(pk[pk['任务组编号'] == k]['服务区列表'].iloc[0]).split(','))))
                       for k in sorted(pk['任务组编号'].unique())],
              loc='lower left', fontsize=9)
    panel_tag(ax, tag)


def fig_q4_partition():
    """(a) K=2 最优分区地图 (b) K=3 最优分区地图 (c) 原子单元工作量网络
    (d) 全部分区的资源规模分布。

    两个 K 各占一格而不是只画 K=2：(b) 的「一个大组 + 两个小组」正是
    §8 里「K=3 均衡度反而更差」的直接来源，只画 K=2 时这句话没有图可看。
    """
    part = rd('q4_partition.csv')
    units = rd('q4_units.csv')
    allp = rd('q4_all_partitions.csv')
    # 架次→服务区取自**问题三的最终运输方案**：问题四就是从这张表出发做绑定的，
    # 图与解同源。列名与问题二那张表一致（架次编号 / 访问服务区顺序）。
    tp = rd('q3_transport_trips.csv')
    # 两处 R 都从 q4_comparison.csv 读——R 是重解后会变的量，写进标题里就是下一个
    # 「图与正文对不上却无人报错」的隐患。
    _cmp = rd('q4_comparison.csv').set_index('K')

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.6))

    # (a)(b) 地图：两个 K 下的最优分区
    from core import load_data
    d = load_data()
    pos = {s['id']: (s['lon'], s['lat']) for s in d.services}
    _partition_map(axes[0][0], d, pos, part, 2, '(a)', int(_cmp.loc[2, '资源规模R']))
    _partition_map(axes[0][1], d, pos, part, 3, '(b)', int(_cmp.loc[3, '资源规模R']))

    # (c) 原子单元网络：节点面积 ∝ 工作量，边 = 两个单元被同一个中继悬停站保障。
    # 边判据必须真的落在两个**端点**上：原先写作 `s1 = {所有站}; if s1:`，s1 与
    # u1/u2 毫无关系、恒为非空，于是每个单元对都连边，画出来是完全图——一团线，
    # 看不出任何结构，等于没画。这里改成经「架次 → 所属单元」反查各站服务了哪些
    # 单元，再在**同站服务的单元**之间连边。
    import networkx as nx
    import itertools
    ax = axes[1][0]
    rel = rd('q3_intervals.csv')
    G = nx.Graph()
    for _, r in units.iterrows():
        svcs = str(r['服务区列表']).split(',')
        # 服务区一行两个：一行写四个会横向伸出节点外，两行正好缩在节点下方
        G.add_node(r['单元编号'], w=float(r['工作量h']),
                   lab='\n'.join(','.join(svcs[i:i + 2]) for i in range(0, len(svcs), 2)))
    unit_of_svc = {s: r['单元编号'] for _, r in units.iterrows()
                   for s in str(r['服务区列表']).split(',')}
    trip_unit = {}
    for _, r in tp.iterrows():
        for s in str(r['访问服务区顺序']).split('->'):
            if s in unit_of_svc:
                trip_unit.setdefault(r['架次编号'], set()).add(unit_of_svc[s])
    station_units = {}
    for _, r in rel.iterrows():
        for u in trip_unit.get(r['运输架次编号'], ()):
            station_units.setdefault(r['悬停站编号'], set()).add(u)
    for stn, us in station_units.items():
        for u1, u2 in itertools.combinations(sorted(us), 2):
            if G.has_edge(u1, u2):
                G[u1][u2]['st'].append(stn)
            else:
                G.add_edge(u1, u2, st=[stn])
    # 布局：spring_layout 对不连通图会把每个分量各自缩成一团、节点全部叠在一起，
    # 故按分量摆成圆环——位置完全确定，不依赖求解器。
    # 分量是**四个**（5 / 3 / 3 / 1 个单元；由 q4_units.csv 的单元与 q3_intervals.csv
    # 的共用悬停站推出）。早先按一行摆开、簇距 1.95、xlim 却写死 ±2.15 —— 12 个原子
    # 单元里 6 个被裁到画外，5 单元那一簇只在左边界露出一牙绿弧。图与正文「12 个原子
    # 单元」直接冲突，且越界不报错、只是安静地少画。现在坐标边界一律在**版面定稿后**
    # 按实际摆位反推（见函数末尾），并配一条越界断言。
    # 只能**单行四簇**：本面板宽而扁（定稿后坐标区约 290 pt × 184 pt），改成 2×2 时
    # 每簇的纵向预算只剩约 92 pt，装不下「圆环 + 簇名 + 图内说明」。
    comps = sorted((sorted(c) for c in nx.connected_components(G)), key=len, reverse=True)
    RING, CSP = 1.00, 1.80
    posn, comp_cx = {}, {}
    for ci, comp in enumerate(comps):
        # **不能用 nx.circular_layout(G.subgraph(comp))**：networkx 3.7 的 subgraph
        # 视图把节点名收进 set，其迭代序随 PYTHONHASHSEED 逐进程变化——实测同一份数据
        # 两次出图，3 节点那一簇的顺序一次是 ['U07','U01','U06']、另一次是 U01 打头，
        # 于是 U01 一会儿在左上、一会儿甩到最右侧。布局不可复现，纯净目录闭环复检的
        # 逐字节比对会直接失败。这里按已经排好序的 comp 手算圆环，公式与 circular_layout
        # 完全一致（θ = 2πk/n），但顺序是确定的。
        cx = (ci - (len(comps) - 1) / 2.0) * CSP
        n = len(comp)
        for k, node in enumerate(comp):
            th = 2.0 * math.pi * k / n
            # 单点分量摆在簇心：按 cos/sin 摆会被推到簇心右侧 RING，看着像摆歪了
            posn[node] = (RING * math.cos(th) + cx if n > 1 else cx,
                          RING * math.sin(th) if n > 1 else 0.0)
        comp_cx[ci] = cx
    # 节点面积系数 100/300：四簇并排时每簇横向只有约 65 pt，5 节点簇相邻节点的弦长约
    # 43 pt，系数再大，簇内最大节点（2.6113 h）就会吃满整个弦长、和邻节点糊在一起。
    sizes = [G.nodes[n]['w'] * 100 + 300 for n in G.nodes()]
    # 边上一律不写字：逐边写「S01」会沿边旋转、且正好压在节点上，改成每簇正上方写一次
    # 簇名（本来就是整簇同站），信息一点没少。
    nx.draw_networkx_edges(G, posn, ax=ax, alpha=0.55, edge_color='#8A8A8A',
                           width=3.0)
    nx.draw_networkx_nodes(G, posn, ax=ax, node_size=sizes, node_color=C_B,
                           edgecolors='black', linewidths=0.8, alpha=0.9)
    # 节点里只写**单元编号**：工作量已由面积编码，精确值与服务区清单都在 §8.1 的表里。
    # 旧版把「2.6h」和两行服务区清单一并画进节点，四簇并排的宽度下根本放不下——实测
    # 第二簇的服务区清单和第一簇的簇名互相压掉，字叠成一团。
    nx.draw_networkx_labels(G, posn, ax=ax, labels={n: n for n in G.nodes()},
                            font_size=8.5)
    _ttl = []
    for ci, comp in enumerate(comps):
        sub = G.subgraph(comp)
        stns = sorted({s for u, v in sub.edges() for s in G[u][v]['st']})
        # 簇名以**本簇最高节点**为基准上移：5 节点簇的顶节点在 0.951R、3 节点簇在
        # 0.866R，写死一个绝对 y 会让簇名一高一低，看着像摆歪了。
        # 簇名只写站号，不写「共用悬停站」四个字：后者 20 字实测约 190 pt，而相邻簇心
        # 间距只有约 65 pt，画出来会整块盖住下一簇的簇名。这层语义改由坐标区底部的
        # 图内说明交代。
        # 孤立点（U06：其运输架次在 q3_intervals.csv 里没有任何失效区间，全程直连）
        # 没有边，取不到共用站，如实标成「无失效区间」——空着会被读成漏画。
        _ttl.append('/'.join(stns) if stns else '无失效区间')
        ax.text(comp_cx[ci], max(posn[n][1] for n in comp) + 0.34, _ttl[ci],
                ha='center', va='bottom', fontsize=9.5, color=C_A,
                fontweight='bold')
    # 坐标边界、以及那行图内说明，一律留到 tight_layout **之后**再定：tight_layout
    # 之前量到的是默认网格里那一格的位置，与定稿后的坐标区差得很远。
    ax.set_title('原子任务单元与共用中继站')
    ax.axis('off')
    panel_tag(ax, '(c)')

    # (d) 全部分区的 R 分布：最优解落在分布的哪个位置，一眼可见。
    # 纵轴用**占比**而不是分区个数：两档分区数相差 42 倍（2047 / 86526），放在同一条
    # 计数轴上，K=2 的柱高不到 K=3 的 3%——图例写着「K = 2」却在图上看不到任何蓝色，
    # 等于图在说一件读者看不见的事。占比把两档拉回同一量级，绝对条数移进图例。
    ax = axes[1][1]
    vmax = {}
    for K, col in ((2, C_A), (3, C_C)):
        v = allp[allp['K'] == K]['资源规模R']
        cnt, _, _ = ax.hist(v, bins=range(int(v.min()), int(v.max()) + 2),
                            weights=np.full(len(v), 100.0 / len(v)), alpha=0.62,
                            color=col, edgecolor='black', lw=0.5,
                            label='K = %d（%d 个分区）' % (K, len(v)))
        vmax[K] = (float(v.min()), float(cnt.max()))
    # 两条「最优 R」标注都摆在柱顶之上、各自错开：先量出两档的峰高再定位置。
    top = max(m for _v, m in vmax.values())
    # 上限 1.70 倍峰高：只为把右上角图例顶到顶部、给下面两条标注腾地方。
    ax.set_ylim(0, top * 1.70)
    # 两条标注**不能摆在柱顶之上那一条**：实测图例包围盒横跨 x:[32.84, 50.33]，
    # 摆在 y = top*1.50 处的「最优 R=29」整块落在图例里被白底洗淡；「最优 R=33」
    # 在 y = top*1.25 处又啃到图例下沿。改摆进「柱顶之上、图例之下」那条空带
    # （x 从 30 到 41 之间的柱高都远低于峰高），蓝在上、红在下，上下错开。
    ymax = ax.get_ylim()[1]
    for K, col in ((2, C_A), (3, C_C)):
        vmin, _m = vmax[K]
        ax.axvline(vmin, color=col, ls='--', lw=2.0)
        # 标注紧贴各自的虚线右侧；不再画箭头（原来的箭头指向该 K 的最高柱顶，
        # 与实际想指的「虚线＝最优 R」并不一致，反而引错视线）。白底是必要的：
        # 「最优 R=29」的文本框横跨 x:[29.5, 36.1]，会把 x=33 那条红虚线切过去。
        ax.text(vmin + 0.5, ymax * (0.67 if K == 2 else 0.56),
                '最优 R=%d' % vmin, fontsize=11.5, color=col,
                ha='left', va='center',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.9, pad=1.5))
    ax.set_xlabel('资源规模 R = Σ$_g$ N$_{kg}^{req}$')
    ax.set_ylabel('分区占比（%）')
    ax.set_title('全部分区的资源规模分布')
    ax.legend(fontsize=10.5, loc='upper right')
    panel_tag(ax, '(d)')

    fig.tight_layout(w_pad=2.4, h_pad=2.6)
    # ---- (c) 的坐标边界：必须在**版面定稿后**按真实坐标区反推 ----
    # 为什么不能写死、也不能在画之前定：本面板宽而扁，横向要放下四个簇，而簇名是
    # 定宽的文字（点数不随数据单位走）。所以「一行能放几个字」与「x 轴跨多少数据单位」
    # 是互相决定的，只能先量出坐标区真实尺寸，再解下面这个一元的边界条件：
    #   要使半宽 half_pt 的簇名整块落在框内，需要 |cx| + half_pt/ppu ≤ S/2，
    #   而 ppu = Wpt/S ⇒ S ≥ 2|cx| / (1 − 2·half_pt/Wpt)。
    # 注意目标必须是 (c) 那个 axes：函数走到这里，`ax` 已经被 (d) 段改成 axes[1][1] 了。
    axc = axes[1][0]
    fig.canvas.draw()
    _bb = axc.get_window_extent()
    _Wpt = _bb.width * 72.0 / fig.dpi
    _Hpt = _bb.height * 72.0 / fig.dpi

    def _tw(txt, fs):
        """粗略字宽（磅）：中日韩全角按字号，其余按半字号。只用来给边界留净空。"""
        return sum(fs if ord(ch) > 0x2E80 else 0.5 * fs for ch in txt)

    _half = [0.5 * _tw(t, 9.5) for t in _ttl]          # 各簇簇名的半宽（磅）
    _ncl = len(comps)

    def _edges(ppu):
        """给定「每数据单位多少磅」，返回内容在 x 方向的左右边界（数据单位）。"""
        lo = min(comp_cx[ci] - max(RING, _half[ci] / ppu) for ci in range(_ncl))
        hi = max(comp_cx[ci] + max(RING, _half[ci] / ppu) for ci in range(_ncl))
        return lo, hi

    # 边界取在**内容**两侧而不是对称取 ±S/2：四簇里最左是 5 节点的圆环、最右是孤立的
    # 单点，内容本身就不对称，对称取限会把整块内容推得偏左，右边空出一大片。
    _S = 2.0 * max(abs(c) + RING for c in comp_cx) + 1.0
    for _ in range(6):                                  # 不动点迭代，3 轮内已收敛到 0.1%
        _ppu0 = _Wpt / _S
        _lo0, _hi0 = _edges(_ppu0)
        _S = (_hi0 - _lo0) + 12.0 / _ppu0               # 左右各留约 6 磅净空
    _ppu = _Wpt / _S
    _lo, _hi = _edges(_ppu)
    axc.set_xlim(_lo - 6.0 / _ppu, _hi + 6.0 / _ppu)
    # 纵向：上沿 = 簇名上沿（0.34 的间距 + 一行行高）；下沿 = 图内说明那一行。
    _ys = [p[1] for p in posn.values()]
    _line = 13.0 / _ppu
    _ybot = min(_ys) - 0.40 - _line - 0.10
    _ytop = max(_ys) + 0.34 + _line
    _yc = (_ytop + _ybot) / 2.0
    _need = _ytop - _ybot
    # 圆环要在最终版面上**等比例**（否则是椭圆）。等比例需要的纵向跨度由坐标区真实宽高
    # 比给出；若它比内容本身还窄（_need 更大），就退回 _need，宁可留白也不裁字。
    _span = max(_S * _Hpt / _Wpt, _need)
    axc.set_ylim(_yc - _span / 2.0, _yc + _span / 2.0)
    # 「节点面积」与簇名的含义写在坐标区底部，不塞进标题——标题一长就会横向伸进相邻
    # 面板（实测原标题 25 字宽 5.0 in，而单栏只有 3.3 in，直接压住 (d) 的标题）。
    # 措辞必须是「把该簇单元连起来的站」而不是「该簇共用的站」：U01 另有 S06 专供其
    # 自身运输架次，S06 不与任何别的单元共站、因而不产生边，也就不会出现在簇名里；
    # 写「共用」会让人误读成「该簇只由这些站保障」。
    axc.text(_lo + 6.0 / _ppu, _ybot + 0.10,
             '节点面积 ∝ 工作量；簇上方为把该簇单元连起来的中继悬停站',
             fontsize=9.0, color='#4A4A4A', va='bottom')
    # 硬门禁：全部单元必须落在坐标区内。这条断言就是为「一半单元被裁掉」加的——图形
    # 越界不会报错，只会安静地少画几个点，在缩略图上看不出来。
    _xl, _yl = axc.get_xlim(), axc.get_ylim()
    _out = [n for n, (x, y) in posn.items()
            if not (_xl[0] + 0.10 <= x <= _xl[1] - 0.10
                    and _yl[0] + 0.10 <= y <= _yl[1] - 0.10)]
    assert not _out, '这些原子单元落在坐标区外会被裁掉：%s' % sorted(_out)
    save(fig, 'fig_q4_partition.png')


# ---------------------------------------------------------------- 模型检验（§9）

# 四类资源的固定呈现顺序与配色，图 9.x 的 y 轴分组线与图例都取自此表
RES_CLASSES = [('uav:',  '运输无人机',   C_A),
               ('bat:',  '共享电池',     C_B),
               ('RUAV:', '中继无人机',   C_C),
               ('RBAT:', '中继能源组件', '#E8A33D')]
RES_COLOR = {p: c for p, _n, c in RES_CLASSES}


def _resource_usage(trips_file='q3_transport_trips.csv',
                    relay_file='q3_relay_trips.csv',
                    split_charge=False):
    """从 results/*.csv 重建「资源 → 占用区间」，口径与 code/q2.py、code/q3.py 逐行一致。

    不重算模型：架次时刻、机型与电池/组件指派一律读 CSV。唯一借用的模型要素是
    core.charge_time()——那是论文 §4 的两阶段充电公式本身，与 §9 表 9.2 脚注
    「返回时刻＋按能耗反算的充电时长」是同一句定义，复用而非另写一份。

    占用口径（右端一律取 CSV 的「返回O01时刻s」）：
      运输无人机  [t0, t1]                  q2.py: u['free_at']    = start + duration
      共享电池    [t0, t1 + 充电时长]        q2.py: bat['ready_at'] = 上式 + t_chg
      中继无人机  [t0, t1 + turnover]        q3.py: u['free_at']  = 返航 + turnover（按出动）
      中继组件    [t0, t1 + 充电时长]        q3.py: comps[...]    = 返航 + charge_time(...)
    运输架次读 q3_transport_trips.csv——那是**错峰后的最终方案**（F05 起由问题三
    直接把最终时刻落盘）。旧版读问题二的 q2_transport_trips.csv 再逐架次加上
    q3_stagger.csv 的推迟量，是同一件事的第二次拼装：两处口径一旦不同（例如
    问题三新增了就绪性修复而错峰表只记首轮推迟），图上与结果表就会各说一套。
    """
    tt = load_transport_uav_types()
    bats = load_batteries()
    rt = load_relay_type()
    _n_rc, tf_r = load_relay_batteries()

    trips = rd(trips_file)
    rel = rd(relay_file)

    # 电池占用区间的形状随 split_charge 变化：
    #   False（默认）→ (t0, 释放时刻, 架次编号)，与 `fig_resource_conflict` 的
    #                  重叠/间隙判据（比较 y[0] 与 x[1]）严格对齐；
    #   True        → (t0, 返航时刻, 释放时刻, 架次编号)，供甘特图把「随机执行」
    #                  与「返回后的两阶段充电」拆成两段画。
    # 默认值不改动任何既有调用点，`fig_resource_conflict` 的重建结果逐位不变。
    def _bat_iv(t0, t1, soc, cap, tid):
        tr = t1 + charge_time(soc, cap)
        return (t0, t1, tr, tid) if split_charge else (t0, tr, tid)

    use = {}
    for _, r in trips.iterrows():
        tid, g = r['架次编号'], r['机型编号']
        t0 = float(r['开始时刻s'])
        t1 = float(r['返回O01时刻s'])
        soc = 1.0 - float(r['架次能耗kWh']) / tt[g]['E_use']
        use.setdefault(('uav:', r['无人机编号']), []).append((t0, t1, tid))
        use.setdefault(('bat:', r['电池编号']), []).append(
            _bat_iv(t0, t1, soc, bats[g][1], tid))
    # 中继侧先按 `出动编号` 去重（阶段 13 起一行 = 一次悬停站服务，同一次出动的
    # 各行共享出动级的 `开始时刻s`/`返回O01时刻s`/`架次能耗kWh`）。机身与组件的
    # 占用对象都是**出动**：不去重会把同一次出动写成若干段首尾相同的占用，图上
    # 会显示成重叠、下面的 `assert n_ov == 0` 也会当场失败。旧表无该列时退回逐行，
    # 那正好等价于一次出动只服务一站。
    if '出动编号' in rel.columns:
        rel = rel.drop_duplicates(subset=['出动编号'], keep='first')
    for _, r in rel.iterrows():
        t0, t1 = float(r['开始时刻s']), float(r['返回O01时刻s'])
        soc = 1.0 - float(r['架次能耗kWh']) / rt['E_use']
        use.setdefault(('RUAV:', r['中继无人机编号']), []).append(
            (t0, t1 + rt['turnover'], r['中继架次编号']))
        use.setdefault(('RBAT:', r['能源组件编号']), []).append(
            _bat_iv(t0, t1, soc, tf_r, r['中继架次编号']))
    return {k: sorted(v) for k, v in use.items()}


def _ordered_keys(use):
    """按 RES_CLASSES 的类别顺序、类内按编号排序——刻意不用 set 迭代，保证逐字节可复现。"""
    return [k for pre, _n, _c in RES_CLASSES for k in sorted(x for x in use if x[0] == pre)]


def _min_gap(ivs):
    """相邻占用区间的最小间隙；只占用一次的资源返回 None（无周转）。"""
    return min((y[0] - x[1] for x, y in zip(ivs, ivs[1:])), default=None)


def fig_resource_conflict():
    """资源占用矩阵与周转间隙（图 9.x）。

    本图兼作 §9.1 三张检验表的**图示复核**：重叠数与最小间隙在此由 CSV 独立重算，
    必须与表 9.2/9.4 报告的「重叠 0 处」「最小间隙 +4×10^-4 s」一致；不一致说明
    占用口径与求解器不符，此处直接断言失败，而不是把图调好看。
    """
    from matplotlib.colors import ListedColormap

    use = _resource_usage()
    keys = _ordered_keys(use)

    # 判据的容差必须与**结果表的精度**对齐，不能取 1e-6。占用区间由 CSV 反算：
    # 时刻列留 3 位小数（±5e-4 s），返航时刻与充电时长各带一份，能耗列留 6 位
    # （经充电曲线斜率放大约 ±1.4e-4 s），合计不超过 1.5e-3 s。用 1e-6 去卡，
    # 边缘相接的两个架次（前一个「返航+充电」正好等于后一个起飞）会被舍入噪声
    # 判成重叠——实测正是如此：T02 的电池释放重建值 2195.771111 对 T15 的起飞
    # 2195.771，差 1.1e-4 s，纯粹是两边各留 3 位小数造成的。取 0.01 s，与
    # `verify.py` 中同一套反算所用的 TOL 保持一致：比舍入噪声高一个量级，
    # 又比任何真实的调度冲突小两个量级（本例真冲突是 1583 s）。
    CSV_TOL = 0.01
    n_ov = sum(1 for k in keys for x, y in zip(use[k], use[k][1:]) if y[0] < x[1] - CSV_TOL)
    # 只占用一次的资源没有相邻间隙（_min_gap 返回 None），统计时须剔除
    gb = [g for g in (_min_gap(use[k]) for k in keys if k[0] == 'bat:') if g is not None]
    gap_bat = min(gb)
    n_once = sum(1 for k in keys if _min_gap(use[k]) is None)
    print('[自校验] 资源数 = %d，重叠处数 = %d（容差 %.2f s），共享电池最小间隙 = %+.6e s（%d/%d 组电池有周转）'
          % (len(keys), n_ov, CSV_TOL, gap_bat, len(gb), sum(1 for k in keys if k[0] == 'bat:')))
    print('[自校验] 全场仅占用一次的资源 = %d 个' % n_once)
    assert n_ov == 0, '重建出的占用存在重叠 %d 处：口径与求解器不一致' % n_ov
    assert -CSV_TOL <= gap_bat < 1.0, '电池最小间隙 %.3e s 不在预期的亚秒量级' % gap_bat

    # ---- (a) 资源×时间占用矩阵 ----
    tmax = max(iv[1] for k in keys for iv in use[k])
    binw = 120.0
    nbin = int(math.ceil(tmax / binw))
    M = np.zeros((len(keys), nbin))
    for i, k in enumerate(keys):
        ci = [p for p, _n, _c in RES_CLASSES].index(k[0]) + 1
        for t0, t1, _lab in use[k]:
            a = max(0, int(t0 // binw))
            b = min(nbin - 1, int((t1 - 1e-9) // binw))
            M[i, a:b + 1] = ci

    fig, (axa, axb) = plt.subplots(
        1, 2, figsize=(10.4, 6.8),
        gridspec_kw=dict(width_ratios=[3.0, 1.3], wspace=0.05))

    cmap = ListedColormap(['#F4F4F4'] + [c for _p, _n, c in RES_CLASSES])
    axa.imshow(M, aspect='auto', cmap=cmap, vmin=0, vmax=4, interpolation='nearest',
               extent=[0, nbin * binw, len(keys) - 0.5, -0.5])
    acc = 0
    for pre, _n, _c in RES_CLASSES[:-1]:
        acc += sum(1 for k in keys if k[0] == pre)
        axa.axhline(acc - 0.5, color='white', lw=2.4)
        axa.axhline(acc - 0.5, color='#333333', lw=0.8)
    axa.grid(False)
    axa.set_xlabel('时间（s）')
    axa.set_yticks(list(range(len(keys))))
    axa.set_yticklabels([k[1] for k in keys], fontsize=8.5)
    for tick, k in zip(axa.get_yticklabels(), keys):
        tick.set_color(RES_COLOR[k[0]])
    axa.set_title('资源占用矩阵')
    panel_tag(axa, '(a)')
    # 图例移到坐标区外下方：放区内会压住 U05/U06 那几段占用块
    axa.legend(handles=[Patch(facecolor=RES_COLOR[p], label=n) for p, n, _c in RES_CLASSES],
               loc='upper center', bbox_to_anchor=(0.5, -0.105),
               ncol=4, frameon=False, fontsize=10.5)

    # ---- (b) 各资源最小周转间隙 ----
    ys = list(range(len(keys)))
    gaps = [_min_gap(use[k]) for k in keys]
    axb.barh(ys, [0.0 if g is None else g for g in gaps],
             color=[RES_COLOR[k[0]] for k in keys], height=0.62, alpha=0.92)
    # 仅占用一次的资源没有「相邻间隙」。**不能留空**——空行与「间隙恰为 0」的零长
    # 条形在视觉上完全一样，而本题确有多架无人机是 0 间隙紧接续，留空会被读成后者。
    # 用一个离散的灰色 ×（而非条形/色带）标记，形状上不可能与间隙数值混淆。
    once = [y for y, g in zip(ys, gaps) if g is None]
    if once:
        axb.plot([0.02] * len(once), once, transform=axb.get_yaxis_transform(),
                 ls='none', marker='x', ms=6, mew=1.5, color='#ADADAD', zorder=5)
    # linthresh 取 1e-4 而非 1：本题最紧的电池间隙只有 4.3e-4 s，若阈值取 1 则该值
    # 落在对称对数的线性段里、贴着 0 画成一粒看不见的点，与「间隙恰为 0」无法区分。
    # 取 1e-4 后 4.3e-4 进入对数段，成为一根肉眼可辨的短条，最紧资源才看得出来。
    axb.set_xscale('symlog', linthresh=1e-4)
    # 刻度必须显式给plain十进制：对称对数的默认刻度走 mathtext，负指数会渲染成
    # 「10⁻³」的上标减号 U+2212，而 SimSun 无此字形，XeTeX 静默丢字、matplotlib
    # 则替换成 dummy 方框——两种结局都是图上出现豆腐块。写死标签即可绕开。
    axb.set_xticks([0, 1e-3, 1e-1, 1e1, 1e3])
    axb.set_xticklabels(['0', '0.001', '0.1', '10', '1000'])
    axb.axvline(0.0, color=C_OUT, ls='--', lw=1.5)
    axb.set_ylim(len(keys) - 0.5, -0.5)
    axb.set_yticks([])
    axb.tick_params(left=False)
    axb.set_xlabel('最小周转间隙（s）')
    axb.set_title('周转间隙与冲突阈值')
    axb.text(0.97, 0.02, '红色虚线 = 冲突阈值 0 s\n全部资源均在其右侧',
             transform=axb.transAxes, ha='right', va='bottom', fontsize=9.5, color=C_OUT)
    if once:
        axb.legend(handles=[Line2D([], [], ls='none', marker='x', ms=6, mew=1.5,
                                   color='#ADADAD', label='仅占用一次（无周转）')],
                   loc='upper center', bbox_to_anchor=(0.5, -0.105),
                   frameon=False, fontsize=10.5)
    panel_tag(axb, '(b)')

    fig.tight_layout()
    save(fig, 'fig_resource_conflict.png')


def fig_solution_network():
    """跨问题方案网络：服务区 → 运输架次 → 机型／中继悬停站（图 9.x）。

    四层连边全部由独立脚本产出的 CSV 还原：服务区—架次与架次—机型取自
    q3_transport_trips.csv（问题三的最终方案，也即问题四的输入），
    架次—悬停站取自 q3_comm_phases.csv ⋈ q3_relay_trips.csv。

    **不使用任何 nx.*_layout**：networkx 3.7 的 subgraph 视图迭代 set，布局会随
    PYTHONHASHSEED 漂移、破坏逐字节可复现（fig_q4_partition 内已记录过同类事故）。
    此处四列 x 固定、y 由**已排序**节点表按序号等分算得，pos 一律显式传入。
    """
    import networkx as nx

    trips = rd('q3_transport_trips.csv')
    rtrips = rd('q3_relay_trips.csv')
    phases = rd('q3_comm_phases.csv')

    zones = sorted({z for s in trips['访问服务区顺序'] for z in str(s).split('->')})
    tids = sorted(trips['架次编号'])
    types = sorted(set(trips['机型编号']))
    stations = sorted(set(rtrips['悬停站编号']))

    z2t, t2type = [], {}
    for _, r in trips.iterrows():
        t2type[r['架次编号']] = r['机型编号']
        for z in str(r['访问服务区顺序']).split('->'):
            z2t.append((z, r['架次编号']))
    bind = phases[phases['中继架次编号'].notna()][['运输架次编号', '中继架次编号']].drop_duplicates()
    bind = bind.merge(rtrips[['中继架次编号', '悬停站编号']], on='中继架次编号')
    t2st = sorted(set(zip(bind['运输架次编号'], bind['悬停站编号'])))

    ez = sorted({(('z', z), ('t', t)) for z, t in z2t})
    # 注意：order 是「按机型分组」后的次序，与 tids（T01…T20）不同。eg 与 edge_color
    # 必须取同一个 order，否则颜色与边一一错位（曾因此画出「蓝色架次配绿色机型边」）。
    order = sorted(tids, key=lambda t: (t2type[t], t))
    eg = [(('t', t), ('g', t2type[t])) for t in order]
    print('[自校验] 服务区=%d 架次=%d 机型=%d 悬停站=%d | 边 服务区-架次=%d（去重前 %d）架次-机型=%d 架次-悬停站=%d'
          % (len(zones), len(tids), len(types), len(stations), len(ez), len(z2t), len(eg), len(t2st)))
    # 断言只查两类东西：**结构性不变量**（服务区 15 个、机型 3 种）与**与冻结方案的
    # 一致性**（架次数 = q3_solution.json 的 transport_trips 数、悬停站数 = 它的
    # station_ids 数）。不再写死 20 / 3 这类字面量——写死只在「方案恰好不变」时才对：
    # 重解一次它要么拦住一张完全正确的图，要么逼着人把数字改成新值、从而彻底失去
    # 把关作用。改成对 q3_solution.json 之后，这条断言把关的是「CSV 与冻结方案分叉」，
    # 而那正是它真正要防的事故。
    with open(os.path.join(RES, 'q3_solution.json'), encoding='utf-8') as _f:
        _sol = json.load(_f)
    assert (len(zones), len(types)) == (15, 3)
    assert len(tids) == len(_sol['transport_trips']), \
        '图上架次数 %d 与 q3_solution.json 的 %d 不符：CSV 与冻结方案已分叉' \
        % (len(tids), len(_sol['transport_trips']))
    assert len(stations) == len(_sol['station_ids']), \
        '图上悬停站数 %d 与 q3_solution.json 的 %d 不符' \
        % (len(stations), len(_sol['station_ids']))
    # 服务区访问共 z2t 次；去重后少掉的边是「两个不同架次访问了同一对服务区」
    assert len(ez) == len(set(z2t)) and len(eg) == len(tids)

    G = nx.Graph()
    G.add_edges_from(ez)
    G.add_edges_from(eg)
    G.add_edges_from([(('t', t), ('s', s)) for t, s in t2st])

    # 架次列**按机型分组排序**而不是按编号：这样「架次→机型」的连线成为三段互不交叉
    # 的平行束。若按 T01…T20 排，每一条架次→机型边都要横穿整个架次列，右侧会糊成一团。
    Y, X0, X1, X2, X3 = 15.0, 0.0, 1.25, 2.45, 3.30
    pos = {}
    for i, z in enumerate(zones):
        pos[('z', z)] = (X0, Y - i * (Y / max(1, len(zones) - 1)))
    yt = {t: Y - i * (Y / max(1, len(order) - 1)) for i, t in enumerate(order)}
    for t in order:
        pos[('t', t)] = (X1, yt[t])
    for g in types:
        yy = [yt[t] for t in order if t2type[t] == g]
        pos[('g', g)] = (X2, sum(yy) / len(yy))
    # 悬停站落在其所服务架次的纵坐标重心上，并强制彼此至少错开 1.35 个单位
    ysrv = {}
    for s in stations:
        yy = [yt[t] for t, ss in t2st if ss == s]
        ysrv[s] = sum(yy) / len(yy) if yy else 0.0
    sorder = sorted(stations, key=lambda s: -ysrv[s])
    for i, s in enumerate(sorder):
        if i:
            ysrv[s] = min(ysrv[s], ysrv[sorder[i - 1]] - 1.35)
    for s in stations:
        pos[('s', s)] = (X3, ysrv[s])

    fig, ax = plt.subplots(figsize=(10.6, 6.9))
    nx.draw_networkx_edges(G, pos, edgelist=ez, ax=ax,
                           edge_color='#D8D8D8', width=0.8, alpha=0.9)
    nx.draw_networkx_edges(G, pos, edgelist=[(('t', t), ('s', s)) for t, s in t2st], ax=ax,
                           edge_color=C_OUT, width=1.1, alpha=0.6, style='dashed')
    nx.draw_networkx_edges(G, pos, edgelist=eg, ax=ax,
                           edge_color=[TYPE_COLOR[t2type[t]] for t in order], width=1.6, alpha=0.85)

    nx.draw_networkx_nodes(G, pos, nodelist=[('z', z) for z in zones], ax=ax,
                           node_size=65, node_color='#D9D9D9',
                           edgecolors='#8A8A8A', linewidths=0.7)
    nx.draw_networkx_nodes(G, pos, nodelist=[('t', t) for t in order], ax=ax,
                           node_size=95, node_color=[TYPE_COLOR[t2type[t]] for t in order],
                           edgecolors='white', linewidths=0.7)
    nx.draw_networkx_nodes(G, pos, nodelist=[('g', g) for g in types], ax=ax,
                           node_size=1250, node_color=[TYPE_COLOR[g] for g in types],
                           edgecolors='white', linewidths=1.1)
    nx.draw_networkx_nodes(G, pos, nodelist=[('s', s) for s in stations], ax=ax,
                           node_size=1250, node_color=C_OUT, edgecolors='white',
                           linewidths=1.1, node_shape='s')

    # 标签一律加白底 bbox：架次编号写在架次列左侧，正好压在灰线上，不加底会糊
    bbox = dict(facecolor='white', edgecolor='none', alpha=0.72, pad=0.9)
    for z in zones:
        ax.text(X0 - 0.075, pos[('z', z)][1], z, ha='right', va='center',
                fontsize=10.5, color='#333333')
    for t in order:
        ax.text(X1 - 0.085, yt[t], t, ha='right', va='center',
                fontsize=9.5, color='#333333', bbox=bbox)
    nx.draw_networkx_labels(G, pos, labels={('g', g): g + ' 型' for g in types}, ax=ax,
                            font_size=12, font_color='white')
    nx.draw_networkx_labels(G, pos, labels={('s', s): s for s in stations}, ax=ax,
                            font_size=11.5, font_color='white')
    for x, lab in [(X0, '服务区（%d）' % len(zones)), (X1, '运输架次（%d）' % len(tids)),
                   (X2, '机型（按机型分组）'), (X3, '悬停站')]:
        ax.text(x, Y + 1.15, lab, ha='center', va='bottom',
                fontsize=12.5, fontweight='bold', color='#333333')

    # 图例放到坐标区上方：放左下会正好盖住 S015 的标签
    ax.legend(handles=[Line2D([], [], color='#C4C4C4', lw=1.6, label='服务区 → 架次（访问顺序）'),
                       Line2D([], [], color=TYPE_COLOR['C'], lw=1.8, label='架次 → 机型'),
                       Line2D([], [], color=C_OUT, lw=1.4, ls='--', label='架次 → 悬停站（中继保障）')],
              loc='lower center', bbox_to_anchor=(0.5, -0.055),
              ncol=3, frameon=False, fontsize=10.5)
    ax.set_xlim(-0.95, X3 + 0.55)
    ax.set_ylim(-1.1, Y + 2.3)
    ax.axis('off')
    ax.grid(False)
    fig.tight_layout()
    save(fig, 'fig_solution_network.png')


def fig_sens_panel():
    """§9.3 的多参数灵敏度与稳健性组合图（2×2）。

    四格依次对应四组扰动：(a) 问题二 5 种子稳定性（S4-a，`q2_seeds.csv`）、
    (b) 中继无人机架数（S4-b(i)）、(c) 合并窗口与审计步长（S4-b(ii)(iii)，
    两组都是「改了也几乎不动」的项，故合并成一格直接画出「平」）、
    (d) 库存阈值曲线（S4-c，`q4_sens.csv`）。

    **图里不写死任何数字**：基线取 `q3_coverage.csv` 的「联合优化」行，档位数与
    分区数取 `q4_comparison.csv`，散点与曲线一律来自各自的 `*_sens.csv`。
    这样做是因为本节的全部论点是「换个参数会变多少」，一旦图上的数字与表脱钩，
    图就会替表说话。
    """
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 8.2))

    # ---- (a) 问题二：4 个架次档 × 5 个种子的完工时刻 ----
    ax = axes[0, 0]
    sd = rd('q2_seeds.csv')
    KS = (20, 22, 25, 30)
    data = [sd[sd['K上限'] == K]['makespan_s'].values.astype(float) for K in KS]
    bp = ax.boxplot(data, positions=range(len(KS)), widths=0.5, patch_artist=True,
                    showfliers=False, medianprops=dict(color='black', lw=1.4))
    for b in bp['boxes']:
        b.set(facecolor='#DCE6F1', edgecolor=C_A, lw=1.1)
    # 逐种子散点用固定偏移错开，不用随机抖动——图必须每次运行逐位相同
    off = np.linspace(-0.17, 0.17, 5)
    for i, K in enumerate(KS):
        ax.plot(i + off, data[i], 'o', ms=5.5, color=C_A, alpha=0.9, zorder=3)
    # 交付档（K≤25）的档案值是 5 个种子里的最优，不是中位数，故单标红星
    scan = rd('q2_scan.csv')
    hit = scan[scan['K上限'] == 25]
    if len(hit) == 1:
        ax.plot([2], [float(hit['档案makespan_s'].iloc[0])], marker='*', ms=17,
                color=C_C, zorder=4, label='交付档取该档 5 种子的最优')
    ax.set_xticks(range(len(KS)))
    ax.set_xticklabels(['$K\\leq20$', '$K\\leq22$', '$K\\leq25$', '$K\\leq30$'])
    ax.set_xlabel('架次数约束档')
    ax.set_ylabel('完工时刻 / s')
    ax.set_title('多种子稳定性')
    ax.legend(loc='upper right', fontsize=10)
    n30 = sd[sd['K上限'] == 30][['架次', '能耗kWh', 'makespan_s', '加权时延']]
    # 注释框放左上偏中：左下被 K≤25 的箱与红星占着，右上被图例占着，
    # 只有 (0.02, 0.52) 一带是空的。
    ax.text(0.02, 0.52, '$K\\leq30$ 档：\n$%d/5$ 个种子四项指标逐位相同'
            % int(n30.value_counts().iloc[0]),
            transform=ax.transAxes, fontsize=10.5, va='center',
            bbox=dict(fc='white', ec='0.7', alpha=0.85, boxstyle='round,pad=0.3'))
    panel_tag(ax, '(a)')

    # ---- (b) 问题三：中继无人机的架数是不是紧的 ----
    ax = axes[0, 1]
    s3 = rd('q3_sens.csv')
    grp = s3[(s3['扰动组'] == '中继机数') & (s3['指标'] == '超额站·秒')]
    xs, ex = [], []
    for n in (1, 2, 3):
        xs.append(n)
        ex.append(float(grp[grp['取值'] == 'n_relay=%d' % n]['数值'].iloc[0]))
    ax.bar(range(len(xs)), ex, width=0.55, color=[C_C, C_B, C_B],
           edgecolor='0.3', lw=0.8)
    top = max(ex) if max(ex) > 0 else 1.0
    for i, v in enumerate(ex):
        ax.text(i, v + top * 0.035, '%.1f' % v, ha='center', fontsize=11.5)
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels(['$n_r=1$', '$n_r=2$（库存）', '$n_r=3$'])
    ax.set_xlabel('中继无人机可用架数')
    ax.set_ylabel('超额站$\\cdot$秒 / (站$\\cdot$s)')
    ax.set_ylim(0, top * 1.24)
    ax.set_title('中继机数的紧度')
    peak = float(s3[(s3['扰动组'] == '中继机数')
                    & (s3['指标'] == '同时最少站数峰值')]['数值'].iloc[0])
    ax.text(0.40, 0.74, '交付排班同一时刻\n最少需要 $%d$ 个悬停站' % int(peak),
            transform=ax.transAxes, fontsize=10.5,
            bbox=dict(fc='white', ec='0.7', alpha=0.85, boxstyle='round,pad=0.3'))
    panel_tag(ax, '(b)')

    # ---- (c) 合并窗口与审计步长：两组都该「平」，故并成一格画 ----
    ax = axes[1, 0]
    cov = rd('q3_coverage.csv')
    base = float(cov[cov['口径'] == '联合优化']['直连时间占比'].iloc[0])
    MW = ['gap=600, max_job=2400', 'gap=900, max_job=3600（交付档）',
          'gap=1200, max_job=4800']
    DTS = ['Δt=0.5 s', 'Δt=1 s', 'Δt=2 s']

    def pick(group, setting, metric='直连时间占比'):
        h = s3[(s3['扰动组'] == group) & (s3['取值'] == setting) & (s3['指标'] == metric)]
        return float(h['数值'].iloc[0])

    xs3 = [0, 1, 2]
    ax.plot(xs3, [pick('合并窗口', t) for t in MW], 'o-', color=C_A, lw=1.6,
            ms=7, label='服务时段合并窗口')
    ax.plot(xs3, [pick('审计步长', t) for t in DTS], 's--', color=C_C, lw=1.6,
            ms=6.5, label='逐时刻审计步长')
    ax.axhline(base, color='black', ls=':', lw=1.3)
    span = max(abs(pick('合并窗口', t) - base) for t in MW)
    span = max(span, max(abs(pick('审计步长', t) - base) for t in DTS))
    lo, hi = base - 2.6 * span, base + 1.2 * span
    ax.set_ylim(lo, hi)
    # 交付基线的名字直接写在虚线上，不占图例——三行图例会把右下角撑到审计步长的
    # 「宽档」点上去（那一段的纵坐标正好落在图例框里）。
    ax.text(0.99, (base - lo) / (hi - lo) + 0.025, '交付基线', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=10.5, color='0.25')
    ax.set_xticks(xs3)
    ax.set_xticklabels(['窄档', '交付档', '宽档'])
    ax.set_xlabel('参数档位（每组三档，中档为交付取值）')
    ax.set_ylabel('直连时间占比')
    ax.set_title('窗口与步长的不敏感性')
    ax.legend(loc='lower right', fontsize=10)
    rg_mw = max(pick('合并窗口', t) for t in MW) - min(pick('合并窗口', t) for t in MW)
    rg_dt = max(pick('审计步长', t) for t in DTS) - min(pick('审计步长', t) for t in DTS)
    # 极差恰为 0 时不能印成「0.0×10⁻⁶」——那是「算出来很小」，而事实是逐位相同，
    # 措辞要跟着事实走。
    mw_txt = '恒为 $0$' if rg_mw == 0 else '$%.1f\\times10^{-6}$' % (rg_mw * 1e6)
    ax.text(0.03, 0.06, '组内极差：合并窗口 %s\n审计步长 $%.1f\\times10^{-4}$'
            % (mw_txt, rg_dt * 1e4),
            transform=ax.transAxes, fontsize=10.5,
            bbox=dict(fc='white', ec='0.7', alpha=0.85, boxstyle='round,pad=0.3'))
    panel_tag(ax, '(c)')

    # ---- (d) 问题四：逐类各加 m 件后还剩多少个分区可选 ----
    ax = axes[1, 1]
    s4 = rd('q4_sens.csv')
    thr = s4[(s4['扰动组'] == '库存阈值·逐类各加m件') & (s4['指标'] == '库存可行分区数')]
    cmp4 = rd('q4_comparison.csv')
    npart = {int(r['K']): int(r['分区数']) for _, r in cmp4.iterrows()}
    for K, c, mk in ((2, C_A, 'o'), (3, C_C, 's')):
        rows = []
        for _, r in thr[thr['取值'].str.startswith('K=%d,' % K)].iterrows():
            rows.append((int(str(r['取值']).split('m=')[1]), float(r['数值'])))
        rows.sort()
        ax.plot([0] + [m for m, _ in rows], [0.0] + [v for _, v in rows],
                mk + '-', color=c, lw=1.6, ms=6.5,
                label='$K=%d$（共 $%d$ 个分区）' % (K, npart[K]))
    # 可行分区数可以取 0，普通对数轴画不出 0，故用 symlog：0 附近线性、以上取对数。
    ax.set_yscale('symlog', linthresh=1, linscale=0.5)
    ax.set_xlabel('现库存逐类各加 $m$ 件')
    ax.set_ylabel('库存可行的分区数')
    ax.set_title('库存阈值曲线')
    ax.set_xticks(range(0, 6))
    ax.legend(loc='lower right', fontsize=10)
    # 注释放左上：$m\\ge2$ 之后两条曲线都升到右上，左上才是空的。
    ax.text(0.03, 0.86, '$m=0$（现库存）时\n两个 $K$ 都是 $0$ 个可行分区',
            transform=ax.transAxes, fontsize=10.5, va='top',
            bbox=dict(fc='white', ec='0.7', alpha=0.85, boxstyle='round,pad=0.3'))
    panel_tag(ax, '(d)')

    fig.tight_layout(h_pad=3.0, w_pad=2.4)
    save(fig, 'fig_sens.png')


# ------------------------------------------- 问题三：优化前后对照与最小裕量

def _seg_spans(pts, states):
    """把逐采样点状态归并成 [(t1, t2, 状态), …]，边界口径与 q3.audit 的 b 序列相同。

    内部边界取相邻采样时刻的中点、首尾取端点。这样每段的长度恰是审计里那一段的
    时间权重，条形长度可以直接当占比读；若改用 pts[j-1][0] 收尾、下一段从 pts[j][0]
    起，每处边界都会漏掉一个采样步长，画出来的中断占比会比报出的数字小一截。
    """
    ts = [p[0] for p in pts]
    n = len(ts)
    b = [ts[0]] + [(ts[k - 1] + ts[k]) / 2.0 for k in range(1, n)] + [ts[-1]]
    out = []
    i = 0
    while i < n:
        j = i
        while j < n and states[j] == states[i]:
            j += 1
        out.append((b[i], b[j], states[i]))
        i = j
    return out


def fig_q3_plan_compare():
    """优化前后的运输—中继甘特对照（三行共享时间轴）。

    (a) 问题二基线的 24 个运输架次，按 `原开始时刻s`（取落盘的 base_start 全精度值）
    画，逐采样点用 q3.direct_ok 只读重算，蓝=直连、红=中断；(b) 问题三最终方案，
    通信分段直接读 q3_comm_phases.csv；(c) 两架中继机的悬停服务段。三行同一时间轴，
    「架次整体后移了多少」与「后移换来了什么」才能直接对着看。

    **内建门禁**：把中继整个拿掉后重算的直连／中断时间占比，必须与
    q3_coverage.csv 的「无中继」行逐位相符（6 位小数）。这张图的说服力全部建立在
    「同一套轨迹、同一个判据，去掉中继就是 28.3% 的中断」之上；重算一旦与结果表
    不符，说明图与解已经各说一套，此时宁可让脚本失败，也不留下一张自说自话的图。
    """
    d, q3, assign = _q3_context()
    cov = rd('q3_coverage.csv').set_index('口径')
    ph = rd('q3_comm_phases.csv')
    rt = rd('q3_relay_trips.csv')
    tt = rd('q3_transport_trips.csv')

    # 基线时刻与 CSV 交叉核对：CSV 的「原开始时刻s」只留 3 位小数，容差按舍入量级取
    csv_old = {r['架次编号']: float(r['原开始时刻s']) for _, r in tt.iterrows()}
    for a in assign:
        assert a['tid'] in csv_old, '架次 %s 不在 q3_transport_trips.csv 里' % a['tid']
        assert abs(a['base_start'] - csv_old[a['tid']]) <= 1e-3 + 1e-9, \
            '架次 %s 的基线时刻 json=%.6f 与 csv=%.3f 不符' % (
                a['tid'], a['base_start'], csv_old[a['tid']])

    segs_old, tot, T = {}, dict(direct=0.0, gap=0.0), 0.0
    for a in assign:
        t = d.transport_types[a['type']]
        pts = q3.sample_trip_trajectory(d, t, a['route'], a['boxes_at'],
                                        t0=a['base_start'], dt=q3.DT_AUDIT)
        w = q3.time_weights(pts)
        states = ['直连' if q3.direct_ok(d, lon, lat, alt) else '中断'
                  for _tt, lon, lat, alt in pts]
        for k, s in enumerate(states):
            tot['direct' if s == '直连' else 'gap'] += w[k]
        T += pts[-1][0] - pts[0][0]
        segs_old[a['tid']] = _seg_spans(pts, states)
    rec = {k: tot[k] / T for k in tot}
    for key, col in (('direct', '直连时间占比'), ('gap', '中断时间占比')):
        ref = float(cov.loc['无中继', col])
        assert abs(rec[key] - ref) < 5e-7, \
            '（a）行重算的%s = %.6f，与 q3_coverage.csv「无中继」行的 %.6f 不符' % (
                col, rec[key], ref)
    fr = float(cov.loc['联合优化', '中继时间占比'])
    print('[自校验] 优化前（无中继）重算：直连 %.6f / 中断 %.6f —— 与 q3_coverage.csv '
          '「无中继」行逐位相符' % (rec['direct'], rec['gap']))

    trips = sorted(csv_old)
    ypos = {x: i for i, x in enumerate(trips)}
    cmap = {'直连': C_DIRECT, '中继': C_RELAY, '中断': C_OUT}
    # 24 行 × 2 格，每格要给到 ~3.7 in 才有 0.15 in 行距，故整图取到 9.4 in 高；
    # hspace 由 0.15 加到 0.32，是因为上一格的 lower-left 图例与下一格的居中标题
    # 在 0.15 时会挤在同一条水平带上（192 dpi 截图实测两行字重叠）。
    fig, axes = plt.subplots(3, 1, figsize=(10.6, 9.4), sharex=True,
                             gridspec_kw=dict(height_ratios=[1.0, 1.0, 0.30],
                                              hspace=0.32))

    # (a) 优化前：问题二基线时刻，只判直连
    ax = axes[0]
    for tid in trips:
        for t1, t2, s in segs_old[tid]:
            ax.barh(ypos[tid], t2 - t1, left=t1, height=0.7,
                    color=cmap[s], edgecolor='none', zorder=3)
    ax.set_yticks(list(ypos.values())); ax.set_yticklabels(trips, fontsize=9)
    ax.invert_yaxis()
    ax.set_ylabel('运输架次')
    ax.set_title('优化前：问题二基线时刻，无中继保障')
    # 图例一律摆 lower left：末几个架次（T19~T24）从 2500 s 之后才起飞，左下角才是
    # 空格；放 lower right 会正好压在它们的条形上（截图逐行确认）。
    ax.legend(handles=[Patch(color=C_DIRECT, label='直连可用'),
                       Patch(color=C_OUT, label='无直连（此处即为中断）')],
              loc='lower left', ncol=1, fontsize=10.5)
    ax.text(0.985, 0.90, '直连 %.1f%%，中断 %.1f%%' % (rec['direct'] * 100, rec['gap'] * 100),
            ha='right', va='center', transform=ax.transAxes, fontsize=13,
            color=C_OUT, fontweight='bold')
    panel_tag(ax, '(a)')

    # (b) 优化后：问题三最终时刻，直连/中继/中断三段由审计结果直接给出
    ax = axes[1]
    for _, r in ph.iterrows():
        if r['运输架次编号'] not in ypos:
            continue
        ax.barh(ypos[r['运输架次编号']], r['结束时刻s'] - r['开始时刻s'],
                left=r['开始时刻s'], height=0.7, color=cmap[r['保障方式']],
                edgecolor='none', zorder=3)
    ax.set_yticks(list(ypos.values())); ax.set_yticklabels(trips, fontsize=9)
    ax.invert_yaxis()
    ax.set_ylabel('运输架次')
    ax.set_title('优化后：问题三联合调度时刻，中继补齐全部缺口')
    cnt = ph['保障方式'].value_counts()
    ax.legend(handles=[Patch(color=cmap[k], label='%s（%d 段）' % (k, cnt.get(k, 0)))
                       for k in ['直连', '中继', '中断']],
              loc='lower left', ncol=1, fontsize=10.5)
    ax.text(0.985, 0.90, '中断 %d 段，中继保障 %.1f%%'
            % (int(cnt.get('中断', 0)), fr * 100),
            ha='right', va='center', transform=ax.transAxes, fontsize=13,
            color=C_RELAY, fontweight='bold')
    panel_tag(ax, '(b)')

    # (c) 中继机出动：与上面两行共用时间轴，才能看出中继窗口正好扣住失效区间
    ax = axes[2]
    ruavs = sorted(rt['中继无人机编号'].unique())
    ry = {u: i for i, u in enumerate(ruavs)}
    for _, g in rt.groupby('出动编号', sort=False):
        u = g['中继无人机编号'].iloc[0]
        t0, t1 = float(g['开始时刻s'].min()), float(g['返回O01时刻s'].iloc[0])
        se = float(g['服务结束时刻s'].max())
        ax.barh(ry[u], se - t0, left=t0, height=0.5, color=C_RELAY,
                edgecolor='black', lw=0.5, zorder=3)
        ax.barh(ry[u], t1 - se, left=se, height=0.5, color=C_RELAY, alpha=0.35,
                edgecolor='black', lw=0.5, zorder=3)
        # 标签写在条形**之上**并加白底：架次标签是黑字，直接压在深绿的悬停服务段上
        # 会糊成一团（192 dpi 截图实测不可读）；白底后压在任何底色上都清楚。
        ax.text(t0, ry[u] + 0.34, '%s@%s' % (g['中继架次编号'].iloc[0],
                                             '→'.join(str(s) for s in g['悬停站编号'])),
                fontsize=8.5, va='bottom', zorder=6,
                bbox=dict(fc='white', ec='none', alpha=0.85, pad=0.8))
    ax.set_yticks(list(ry.values())); ax.set_yticklabels(ruavs)
    # 下界留到 2.45（原 1.65）：两架中继各占一行，1.65 时图例的上沿正好切在 R02
    # 条形的下缘上（截图确认），再往下让出 1.2 个单位，图例整块落在空白里。
    ax.set_ylim(2.45, -0.95)
    ax.set_xlabel('时间 (s)'); ax.set_ylabel('中继无人机')
    ax.set_title('中继机出动')
    ax.legend(handles=[Patch(facecolor=C_RELAY, edgecolor='black', label='悬停服务'),
                       Patch(facecolor=C_RELAY, alpha=0.35, edgecolor='black',
                             label='返航与周转')],
              loc='lower right', ncol=2, fontsize=10.5)
    panel_tag(ax, '(c)')

    save(fig, 'fig_q3_plan_compare.png')


def fig_q3_margin():
    """三类最小裕量：硬时限、直连、中继接入（1×3）。

    (a) 31 个带硬时限货箱的交付余量（q3_box_delivery.csv）；(b) 24 个运输架次在全
    轨迹上的最小直连裕量；(c) 19 个通信失效区间的最小接入裕量。(b)(c) 一律
    **读** q3_sens.csv 里 make_q3_sens.py 落下的逐对象明细行，本函数只画不算——
    同一批数字若在这里再调一次 direct_margin/access_margin，就有了两条算路，
    正文引用的宏与本图会各自独立地漂移。图末另把图中的极值与本表的合计行逐位
    核对，两边一旦分叉就当场报错。

    (c) 是这张图存在的理由：失效区间里运输机完全没有直连，全靠悬停站接住，
    接入裕量必须全部大于等于 0（q3.MARGIN_DB）。若有一段为负，说明覆盖矩阵与
    实际轨迹不一致——那属于求解器的错，必须在图上暴露出来，而不是抹掉。
    """
    s3 = rd('q3_sens.csv')
    bd = rd('q3_box_delivery.csv')

    def _detail(group, metric):
        sub = s3[s3['扰动组'] == group]
        if sub.empty:
            raise SystemExit('q3_sens.csv 缺 %s 组——先跑 code/make_q3_sens.py' % group)
        got = sub[sub['指标'] == metric]
        if len(got) != len(sub):
            raise SystemExit('%s 组的指标不全是 %s，无法逐对象取数' % (group, metric))
        return {str(r['取值']): float(r['数值']) for _, r in got.iterrows()}

    def _total(setting, metric):
        """取合计行（与正文宏同一个格子），用来核对图中极值。"""
        got = s3[(s3['扰动组'] == '交付解·最小裕量') & (s3['取值'] == setting)
                 & (s3['指标'] == metric)]
        if len(got) != 1:
            raise SystemExit('q3_sens.csv 的 交付解·最小裕量/%s/%s 命中 %d 行，应为 1 行'
                             % (setting, metric, len(got)))
        return float(got.iloc[0]['数值'])

    min_direct = _detail('交付解·最小裕量·逐架次', '最小直连裕量')
    min_access = _detail('交付解·最小裕量·逐区间', '最小接入裕量')
    n_neg = sum(1 for v in min_access.values() if v < -1e-6)
    print('[自校验] %d 个失效区间的最小接入裕量：最小 %+.4f dB，为负的 %d 个'
          % (len(min_access), min(min_access.values()), n_neg))
    assert n_neg == 0, '有 %d 个失效区间的接入裕量为负，覆盖矩阵与实际轨迹不一致' % n_neg

    fig = plt.figure(figsize=(10.8, 4.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.30, 1.0, 1.0], wspace=0.42)

    # (a) 硬时限余量：3 个数量级，故用对数横轴；「余量 > 0」这条判据用红字写在标题区
    ax = fig.add_subplot(gs[0, 0])
    hv = bd.dropna(subset=['硬时限余量s']).sort_values('硬时限余量s')
    ys = np.arange(len(hv))
    ax.barh(ys, hv['硬时限余量s'].values, height=0.72,
            color=[C_C if i == 0 else C_A for i in range(len(hv))])
    ax.set_yticks(ys)
    ax.set_yticklabels([str(c) for c in hv['货箱编号']], fontsize=6.6)
    # 下界留出 2 个单位，专给最紧箱的数值标注——31 行里最上 12 行的条形都止于
    # 2000 s 之前，但它们上方没有连续空白，格内任何位置都会压住某一行。
    ax.set_ylim(len(hv) + 2.0, -0.6)
    # linthresh 取 1 s：最紧的那个箱子只有 5.353 s，若阈值取 10 它会贴到 0 上看不见
    ax.set_xscale('symlog', linthresh=1.0)
    ax.set_xticks([0, 1, 10, 100, 1000, 10000])
    ax.set_xticklabels(['0', '1', '10', '100', '1000', '10000'])
    ax.axvline(0.0, color=C_C, ls='--', lw=1.6)
    ax.set_xlabel('交付余量（s）')
    ax.set_title('硬时限余量（%d 箱）' % len(hv))
    r0 = hv.iloc[0]
    ax.text(0.98, 0.012, '最紧 %s：%.3f s' % (r0['货箱编号'], r0['硬时限余量s']),
            transform=ax.transAxes, ha='right', va='bottom', fontsize=10, color=C_C,
            fontweight='bold')
    panel_tag(ax, '(a)')

    # (b) 最小直连裕量：全负说明这些架次的某一段确实没有直连，这正是中继存在的理由
    ax = fig.add_subplot(gs[0, 1])
    md = sorted(min_direct.items(), key=lambda kv: kv[1])
    ax.barh(np.arange(len(md)), [v for _k, v in md], height=0.72, color=C_A)
    ax.set_yticks(np.arange(len(md)))
    ax.set_yticklabels([k for k, _v in md], fontsize=8)
    # 底部同样留出 3 个单位的空白带专给标注：T01/T02/T07/T03/T10 这五根条形是
    # 正的（向右延伸），格内任何位置放文字都会压住它们，只有条形下方的空带是干净的。
    ax.set_ylim(len(md) + 3.0, -0.6)
    ax.axvline(0.0, color=C_C, ls='--', lw=1.6)
    ax.set_xlabel('最小直连裕量（dB）')
    ax.set_title('架次的最小直连裕量')
    n_ok = sum(1 for _k, v in md if v >= 0)
    # 措辞按**采样点上的最小值**写，不写「全程直连」：判据是 Δt = 1 s 的逐点采样，
    # 说「全程」就把一个采样结论说成了连续时间上的全称结论。
    ax.text(0.97, 0.012, '%d/%d 架次的最小直连裕量 $\\geq 0$；\n其余 %d 架次需中继'
            % (n_ok, len(md), len(md) - n_ok), transform=ax.transAxes, ha='right',
            fontsize=9.5, va='bottom',
            bbox=dict(fc='white', ec='0.7', alpha=0.9, boxstyle='round,pad=0.3'))
    panel_tag(ax, '(b)')

    # (c) 失效区间的接入裕量：全部 ≥ 0 才说明「零中断」不是把负裕量抹掉换来的
    ax = fig.add_subplot(gs[0, 2])
    ma = sorted(min_access.items(), key=lambda kv: kv[1])
    ax.barh(np.arange(len(ma)), [v for _g, v in ma], height=0.72, color=C_RELAY)
    ax.set_yticks(np.arange(len(ma)))
    ax.set_yticklabels([g for g, _v in ma], fontsize=8)
    ax.set_ylim(len(ma) - 0.4, -0.6)
    ax.axvline(0.0, color=C_C, ls='--', lw=1.6)
    ax.set_xlabel('最小接入裕量（dB）')
    ax.set_title('失效区间的中继接入裕量')
    # 注释放**右上**：前六个区间（G11/G01/G08/G02/G09/G04）的裕量都不到 5 dB，
    # 右上角对它们而言是空的；放右下会正好压住 G15/G18 那两根最长的条形。
    # 小数位与正文宏同为 4 位：图上写 +0.42、正文写 +0.4228 会让读者以为两处
    # 说的是两件事，而它们本是同一个数。
    ax.text(0.97, 0.93, '%d 个区间全部 $\\geq 0$\n最小 %+.4f dB'
            % (len(ma), min(v for _g, v in ma)), transform=ax.transAxes, ha='right',
            va='top', fontsize=9.5, bbox=dict(fc='white', ec='0.7', alpha=0.9,
                                            boxstyle='round,pad=0.3'))
    panel_tag(ax, '(c)')

    # 图与正文同源的最后一道闸：图里画出来的极值必须与正文宏取的那两个合计格子
    # 逐位相同。两条算路一旦分叉，这里当场报错，而不是让论文与自己的图各说一套。
    chk = [('最小直连裕量', min(v for _k, v in md), '全部架次取最小'),
           ('最小接入裕量', min(v for _g, v in ma), '全部失效区间取最小')]
    for metric, shown, setting in chk:
        tot = _total(setting, metric)
        if abs(shown - tot) > 1e-12:
            raise SystemExit('图 %s 的极值 %r 与 q3_sens.csv 合计行 %r 不符'
                             % (metric, shown, tot))
    print('[自校验] 图中的两个极值与 q3_sens.csv 合计行逐位一致')

    save(fig, 'fig_q3_margin.png')


def fig_q3_resource_gantt():
    """运输机与共享电池的占用甘特（含等待段与充电段）。

    (a) 8 架运输无人机的占用条：实心为该机在执行架次，浅色底为本机空闲等待；
    (b) 14 组共享电池：实心为随机执行架次，斜纹为返航后的两阶段充电。
    两条红色竖线是货箱硬时限（从 q3_box_delivery.csv 按数据取去重值），不写字面量
    ——硬时限是题目给的数，图上写错了不会有任何报错提醒。

    复用 `_resource_usage(split_charge=True)`：占用口径与 §9 表 9.2 的复核完全同一份
    代码路径，不另写一套反算，避免图与表各说一套。
    """
    use = _resource_usage(split_charge=True)
    keys = _ordered_keys(use)
    uavs = [k for k in keys if k[0] == 'uav:']
    bats = [k for k in keys if k[0] == 'bat:']
    # 释放时刻：电池是 4 元组 (t0, 返航, 释放, 编号)，机身是 3 元组 (t0, 返航, 编号)
    _rel = lambda iv: iv[2] if len(iv) == 4 else iv[1]
    tmax = max(_rel(iv) for k in keys for iv in use[k])
    dl = sorted(set(rd('q3_box_delivery.csv')['硬时限时刻s'].dropna().astype(float)))
    # 横轴上界取「最后释放时刻」与「最晚硬时限」的较大者：现方案的资源占用在 9200 s
    # 前后就结束了，而最晚的硬时限是 10800 s —— 若按前者截断，第三条硬时限线会落在
    # 图外，读者只看到两条线、以为硬时限只有两档。右边留出的空白本身也是信息：
    # 全部资源在最后一条硬时限之前就已释放完毕。
    xmax = max(tmax, max(dl)) * 1.005

    fig, axes = plt.subplots(2, 1, figsize=(10.4, 7.2), sharex=True,
                             gridspec_kw=dict(height_ratios=[len(uavs), len(bats)],
                                              hspace=0.16))

    # (a) 运输无人机：浅色底条铺满全场，实心条压在上面，露出的部分就是等待
    ax = axes[0]
    for i, k in enumerate(uavs):
        ax.barh(i, xmax, left=0.0, height=0.72, color='#D9D9D9', zorder=1)
        for iv in use[k]:
            ax.barh(i, iv[1] - iv[0], left=iv[0], height=0.72, color=C_A,
                    edgecolor='black', lw=0.4, zorder=3)
    ax.set_yticks(range(len(uavs)))
    ax.set_yticklabels([k[1] for k in uavs])
    ax.set_ylim(len(uavs) - 0.4, -0.6)
    busy = {k: sum(iv[1] - iv[0] for iv in use[k]) for k in uavs}
    # 利用率的分母用**机队完工时刻**（最后一架次返航）× 架数，而不是上面为画满
    # 硬时限线而撑宽的横轴上界——拿坐标轴范围当工期，等于把排版决定写成了指标。
    t_done = max(iv[1] for k in uavs for iv in use[k])
    util = 100.0 * sum(busy.values()) / (t_done * len(uavs))
    ax.set_ylabel('运输无人机')
    ax.set_title('运输无人机占用与等待')
    ax.text(0.985, 0.05, '机队利用率 %.0f%%（占用 %.0f s / 可用 %.0f s）'
            % (util, sum(busy.values()), t_done * len(uavs)),
            transform=ax.transAxes, ha='right', va='bottom', fontsize=10.5,
            bbox=dict(fc='white', ec='0.7', alpha=0.9, boxstyle='round,pad=0.3'))
    panel_tag(ax, '(a)')

    # (b) 共享电池：充电段用斜纹，与实心任务段在形状上就分得开（不只靠颜色）
    ax = axes[1]
    for i, k in enumerate(bats):
        for t0, t1, tr, _lab in use[k]:
            ax.barh(i, t1 - t0, left=t0, height=0.72, color=C_B,
                    edgecolor='black', lw=0.4, zorder=3)
            ax.barh(i, tr - t1, left=t1, height=0.72, facecolor='white',
                    edgecolor=C_B, lw=0.8, hatch='///', zorder=3)
    ax.set_yticks(range(len(bats)))
    ax.set_yticklabels([k[1] for k in bats], fontsize=9)
    ax.set_ylim(len(bats) - 0.4, -0.6)
    ax.set_xlabel('时间（s）')
    ax.set_ylabel('共享电池')
    ax.set_title('共享电池占用与两阶段充电')
    ax.legend(handles=[Patch(facecolor=C_B, edgecolor='black', label='随机执行架次'),
                       Patch(facecolor='white', edgecolor=C_B, hatch='///',
                             label='返航后充电')],
              loc='lower right', bbox_to_anchor=(1.0, 1.005), ncol=2, fontsize=10.5,
              frameon=False)
    panel_tag(ax, '(b)')

    # 硬时限线画在两格上：横轴共享，同一条线在两格里必须落在同一位置
    for ax in axes:
        for dlt in dl:
            ax.axvline(dlt, color=C_C, ls='--', lw=1.6, zorder=5)
        ax.set_xlim(0, xmax)
    # 两个图例都摆到坐标区**上外方**（锚在 (1.0, 1.005)、左对齐朝右展开）：格内右上方
    # 看着是空的，其实每一行都铺着「空闲等待」的灰底条，图例压上去就把等待段盖住了；
    # 标题居中、图例靠右，两者在 10.4 in 宽里相隔 3 in 以上，互不干涉。
    axes[0].legend(handles=[Patch(facecolor=C_A, edgecolor='black', label='执行架次'),
                            Patch(facecolor='#D9D9D9', label='空闲等待'),
                            Line2D([], [], color=C_C, ls='--', lw=1.6, label='货箱硬时限')],
                   loc='lower right', bbox_to_anchor=(1.0, 1.005), ncol=3, fontsize=10.5,
                   frameon=False)

    fig.tight_layout()
    save(fig, 'fig_q3_resource_gantt.png')


# ------------------------------------- 问题四：资源规模—均衡性前沿

def fig_q4_frontier():
    """88 573 个分区在（资源规模 R, 工作量均衡 CV_W）平面上的位置。

    (a) 每个分区一个点，另按 R 画出各 K 下「该资源规模里最均衡的那个分区」的阶梯
    前沿，并用星标标出两个推荐分区（R 与 CV 都从 q4_comparison.csv 读）；
    (b) CV_W 的经验分布，用来回答「推荐解的均衡度到底算好还是差」。

    散点用的是全部 88 573 行，不是抽样——抽样会让「前沿之下还有多少解」这个问题
    无法回答，而本图的全部意义就在于「前沿与云体的距离」。
    """
    allp = rd('q4_all_partitions.csv')
    cmp = rd('q4_comparison.csv').set_index('K')
    feas = int(cmp.loc[2, '库存可行分区数']) + int(cmp.loc[3, '库存可行分区数'])
    assert feas == 0, '库存可行分区数不再是 0（现为 %d），本图注释须重写' % feas
    # (b) 注解的方向依赖一个前提：推荐解的 R 是**全部同 K 分区里的最小值**（因为
    # 第一层目标就是最小化 R），故其均衡度必然落在 CV 分布的最差一端。若哪天换成
    # 先求均衡，那句话会整句反向却不会报错——故在此把前提钉住。
    for _K in (2, 3):
        _r_min = float(allp[allp['K'] == _K]['资源规模R'].min())
        if float(cmp.loc[_K, '资源规模R']) != _r_min:
            raise SystemExit('K=%d 的推荐资源规模 %g 不是全部分区的最小值 %g，'
                             '「先最小化 R」这一前提不成立，(b) 的注解须改写'
                             % (_K, float(cmp.loc[_K, '资源规模R']), _r_min))

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    ax = axes[0]
    for K, col, mk in ((2, C_A, 'o'), (3, C_C, 's')):
        g = allp[allp['K'] == K].sort_values('资源规模R')
        ax.scatter(g['资源规模R'], g['工作量均衡CV_W'], s=3.0, alpha=0.10,
                   color=col, edgecolors='none', rasterized=True)
        # 逐 R 的最优阶梯：同一资源规模下能达到的最小 CV
        fr = g.groupby('资源规模R')['工作量均衡CV_W'].min()
        ax.step(fr.index.values, fr.values, where='post', color=col, lw=2.2,
                label='$K=%d$ 阶梯前沿' % K, zorder=4)
        r_rec = float(cmp.loc[K, '资源规模R'])
        cv_rec = float(cmp.loc[K, '工作量均衡CV_W'])
        ax.plot([r_rec], [cv_rec], marker='*', ms=19, color=col, mec='black',
                mew=0.8, zorder=6, label='$K=%d$ 推荐解' % K)
        # 数值**不贴在星标旁**：K=2 那两行文本横跨 R≈31--34，正好压在 K=3 的星标上
        # （星标 zorder 更高，会把「0.7544」盖掉一半）。改成左上角一列带白底的数值，
        # 星标本体已经用颜色区分了 K，指向关系没有丢。
        ax.text(0.02, 0.97 - 0.055 * (K - 2),
                '$K=%d$ 推荐解：$R=%d$，$CV_W=%.4f$' % (K, r_rec, cv_rec),
                transform=ax.transAxes, fontsize=10, color=col, va='top',
                bbox=dict(fc='white', ec='none', alpha=0.85, boxstyle='round,pad=0.2'))
    ax.set_xlabel('资源规模 $R$')
    ax.set_ylabel('工作量均衡度 $CV_W$')
    ax.set_title('分区解的资源规模—均衡性前沿')
    ax.legend(loc='lower right', fontsize=9.5, framealpha=0.92)
    ax.text(0.02, 0.06, '共 %d 个分区，\n库存可行者 %d 个'
            % (len(allp), feas), transform=ax.transAxes, fontsize=10.5,
            color=C_OUT, fontweight='bold', va='bottom',
            bbox=dict(fc='white', ec='0.7', alpha=0.9, boxstyle='round,pad=0.3'))
    panel_tag(ax, '(a)')

    # (b) CV 的 ECDF：曲线越靠右，均衡度越差；推荐解落在分布的什么位置一眼可见
    ax = axes[1]
    for K, col in ((2, C_A), (3, C_C)):
        v = np.sort(allp[allp['K'] == K]['工作量均衡CV_W'].values.astype(float))
        ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=col, lw=2.0,
                label='$K=%d$（%d 个分区）' % (K, len(v)))
        cv_rec = float(cmp.loc[K, '工作量均衡CV_W'])
        # 与推荐解比较的方向**必须写对**：推荐解是「先最小化 R」挑出来的，R 取到全
        # 部分区的最小值，均衡度因此落在分布的最差一端。若按 (v <= cv_rec) 报成
        # 「优于其 X% 的同 K 分区」，X 会算成 100.0%——把「几乎最不均衡」印成
        # 「优于全部」，方向恰好相反。这里报「比它更均衡者占多少」。
        pct_better = float((v < cv_rec).mean()) * 100.0
        ax.axvline(cv_rec, color=col, ls='--', lw=1.6)
        # 竖排标注**不能挂在虚线上**：两条虚线只隔 0.081，而标注要占两行、旋转 90° 后
        # 两行是横向铺开的，实测 K=3 那整块压住了 K=2 行的「0.7544」。改到右下角图例
        # 之上横排：该带内两条 ECDF 都已升到 1.0，是整幅图里唯一没有被曲线穿过的空白。
        ax.text(0.985, 0.34 - 0.085 * (K - 2),
                '$K=%d$ 推荐解 $CV_W=%.4f$：同 $K$ 分区中 %.1f%% 比它更均衡'
                % (K, cv_rec, pct_better), transform=ax.transAxes, fontsize=9.5, color=col,
                ha='right', va='center',
                bbox=dict(fc='white', ec='none', alpha=0.85, boxstyle='round,pad=0.2'))
    ax.set_xlabel('工作量均衡度 $CV_W$')
    ax.set_ylabel('累计比例')
    ax.set_title('均衡度的经验分布')
    ax.set_ylim(0, 1.0)
    ax.legend(loc='lower right', fontsize=10)
    panel_tag(ax, '(b)')

    fig.tight_layout(w_pad=2.2)
    save(fig, 'fig_q4_frontier.png')


def main():
    fig_sens_panel()
    fig_q1_pareto()
    fig_q1_payload()
    fig_q2_alns()
    fig_q2_gantt()
    fig_q2_sankey()
    fig_q2_exact()
    fig_q3_timeline()
    fig_q3_analysis()
    fig_q3_plan_compare()
    fig_q3_margin()
    fig_q3_resource_gantt()
    fig_q4_resource()
    fig_q4_partition()
    fig_q4_frontier()
    fig_resource_conflict()
    fig_solution_network()
    print('全部升级版配图已生成 ->', FIG)


if __name__ == '__main__':
    main()
