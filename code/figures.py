# -*- coding: utf-8 -*-
# 本程序及代码是在人工智能工具辅助下完成的。
# 工具名称：DeepSeek；版本/型号：DeepSeek V4.1 Flash（API 模型名 deepseek-flash）；
# 开发机构：杭州深度求索人工智能基础技术研究有限公司；
# 版本发布日期：2026 年 9 月 10 日。

"""生成论文配图：地形图、问题一敏感性、问题二调度甘特图、问题三中继覆盖。"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os
from core import load_data, ll_to_xy
from q1 import max_safe_payload

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, '..', 'figures')
os.makedirs(FIG, exist_ok=True)
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False
# 出图按 300 dpi。图在版面里被缩到 0.6~0.95\textwidth，按 120 dpi 出图再缩，
# 有效分辨率只剩 180 dpi 左右，印出来笔画发虚；300 dpi 缩完仍有 250 dpi 以上。
rcParams['savefig.dpi'] = 300


def fig_map(d, relay_stations=None, name='fig_dem_map.png'):
    """地形图 + 服务区 + 调度中心 + （可选）中继站。"""
    dem = d.dem._dem_nan
    lon_min, lon_max = d.dem.lon_min, d.dem.lon_max
    lat_min, lat_max = d.dem.lat_min, d.dem.lat_max
    fig, ax = plt.subplots(figsize=(8, 7), dpi=120)
    im = ax.imshow(dem, extent=[lon_min, lon_max, lat_min, lat_max],
                   origin='upper', cmap='terrain', aspect='auto')
    # 服务区
    for s in d.services:
        ax.plot(s['lon'], s['lat'], 'o', ms=5, color='crimson', mec='white', mew=0.6)
        ax.text(s['lon'], s['lat'] + 0.002, s['id'][1:], fontsize=7, ha='center', color='black')
    # 调度中心
    ax.plot(d.O01['lon'], d.O01['lat'], '*', ms=16, color='gold', mec='black', mew=0.8, zorder=5)
    # 调度中心必须写**中文**、不能写 'O01'：上面服务区的标签是 s['id'][1:]（即 "001"…"015"），
    # 而 SimHei 下字母 O 与数字 0 同形，'O01' 渲染出来就是「001」，与 S001 的服务区标签
    # **完全撞号**（实测裁剪放大确认：星标下方印的是 001）。figures_adv.py 早已改成中文，
    # 这里同步。位置沿用星标下方 0.004°：实测 O01 最近的服务区是 S011（在其上方 +0.0234°）
    # 与 S006（在左侧 −0.0311°），下方那条带是空的，「调度中心」四字约 0.011° 宽，左右净空 ≥0.02°。
    ax.text(d.O01['lon'], d.O01['lat'] - 0.004, '调度中心', fontsize=9, ha='center', color='black', fontweight='bold')
    # 中继站
    if relay_stations:
        for st in relay_stations:
            ax.plot(st[0], st[1], 'D', ms=10, color='blue', mec='white', mew=0.8, zorder=6)
    # 聚焦服务区范围
    slon = [s['lon'] for s in d.services]; slat = [s['lat'] for s in d.services]
    ax.set_xlim(min(slon) - 0.01, max(slon) + 0.01)
    ax.set_ylim(min(slat) - 0.01, max(slat) + 0.01)
    ax.set_xlabel('经度 (°)'); ax.set_ylabel('纬度 (°)')
    ax.set_title('镇龙乡 DEM 地形与服务区分布')
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label('高程 (m)')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name))
    plt.close(fig)
    print('saved', name)


def fig_sensitivity(d, name='fig_q1_sensitivity.png'):
    """问题一：返航安全余量 ρ 对三机型 q_max 的影响。"""
    rhos = np.linspace(0.10, 0.40, 13)
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=120)
    for tid in ['A', 'B', 'C']:
        t = d.transport_types[tid]
        base = t['rho']
        mean_q = []
        for rho in rhos:
            t['rho'] = float(rho)
            mean_q.append(np.mean([max_safe_payload(d, t, d.si[s]) for s in d.S]))
        t['rho'] = base
        ax.plot(rhos * 100, mean_q, '-o', ms=4, label=f'机型 {tid}')
    ax.set_xlabel('返航安全余量 ρ (%)'); ax.set_ylabel('平均最大安全载荷 (kg)')
    ax.set_title('返航安全余量对最大安全载荷的影响')
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, name)); plt.close(fig)
    print('saved', name)


def fig_sensitivity2d(d, name='fig_q1_sens2d.png'):
    """问题一：ρ 与电池可用能量折减 κ 的协同扰动对平均最大安全载荷的影响。

    单因素曲线看不出参数耦合，这里给 (ρ, κ) 二维网格。三机型载荷量级差得远
    （A≈25、B≈30、C≈79），共用色标会把 A、B 压成一片，故每栏独立归一化。
    """
    rhos = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    kappas = [1.00, 0.95, 0.90, 0.85, 0.80]
    # 三栏横排压到 0.95\textwidth 后缩放比约 0.55，源字号必须放大到 ~13pt
    # 才能在排版后剩约 7pt；按 matplotlib 默认 10pt 出图会缩成 5.5pt，印出来看不清。
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.4), dpi=120)
    for k, (ax, tid) in enumerate(zip(axes, ['A', 'B', 'C'])):
        t = d.transport_types[tid]
        base_rho, base_E = t['rho'], t['E_use']
        M = np.zeros((len(rhos), len(kappas)))
        for i, rho in enumerate(rhos):
            for j, kap in enumerate(kappas):
                t['rho'] = rho
                t['E_use'] = base_E * kap
                M[i, j] = np.mean([max_safe_payload(d, t, d.si[s]) for s in d.S])
        t['rho'], t['E_use'] = base_rho, base_E
        im = ax.imshow(M, origin='lower', cmap='YlOrRd_r', aspect='auto',
                       extent=[-0.5, len(kappas) - 0.5, -0.5, len(rhos) - 0.5])
        # 色标是反向的（低载荷=深红），不能按数值大小猜字色；直接取该格的实际
        # 填充色算亮度，深底配白字，否则在深红格上写黑字根本看不见。
        for i in range(len(rhos)):
            for j in range(len(kappas)):
                r, g, b, _ = im.cmap(im.norm(M[i, j]))
                lum = 0.299 * r + 0.587 * g + 0.114 * b
                ax.text(j, i, '%.1f' % M[i, j], ha='center', va='center', fontsize=13,
                        color='white' if lum < 0.55 else 'black')
        ax.set_xticks(range(len(kappas)))
        ax.set_xticklabels(['%.0f%%' % (k * 100) for k in kappas])
        ax.set_yticks(range(len(rhos)))
        ax.set_yticklabels(['%.0f%%' % (r * 100) for r in rhos])
        ax.set_xlabel('电池可用能量 κ', fontsize=13)
        # 三栏的 ρ 刻度完全一致，只在最左栏写轴名；否则后两栏的 y 轴名会压到
        # 前一栏的色标标注上（0.95\textwidth 下横向根本挤不下两列文字）。
        if k == 0:
            ax.set_ylabel('返航安全余量 ρ', fontsize=13)
        ax.tick_params(labelsize=12)
        ax.set_title('机型 %s' % tid, fontsize=15)
        # 色标名用短标签：全称放在正文与图注里，竖排长名会顶到邻栏的刻度上。
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).set_label('载荷 (kg)', fontsize=12)
    fig.suptitle('ρ 与电池可用能量的协同扰动', fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.94], w_pad=2.2)
    fig.savefig(os.path.join(FIG, name)); plt.close(fig)
    print('saved', name)


def fig_gantt(assignment, name='fig_q2_gantt.png'):
    """问题二：运输无人机调度甘特图。"""
    types_color = {'A': '#4C72B0', 'B': '#55A868', 'C': '#C44E52'}
    uavs = sorted(set(a['uav'] for a in assignment))
    ypos = {u: i for i, u in enumerate(uavs)}
    fig, ax = plt.subplots(figsize=(9, 5), dpi=120)
    for a in assignment:
        y = ypos[a['uav']]
        ax.barh(y, a['duration'], left=a['start'], height=0.6,
                color=types_color[a['type']], edgecolor='black', lw=0.3)
    ax.set_yticks(list(ypos.values())); ax.set_yticklabels(list(ypos.keys()))
    ax.set_xlabel('时间 (s)'); ax.set_ylabel('运输无人机')
    ax.set_title('问题二 运输调度甘特图')
    from matplotlib.patches import Patch
    leg = [Patch(color=c, label=f'机型 {g}') for g, c in types_color.items()]
    ax.legend(handles=leg, loc='upper right')
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, name)); plt.close(fig)
    print('saved', name)


def main():
    d = load_data()
    fig_map(d)
    fig_sensitivity(d)
    fig_sensitivity2d(d)

    # 问题二：这里只取推荐方案的 assignment 供下面问题三使用，**不再画甘特图**。
    # fig_gantt() 与 figures_adv.py 的同名函数都写 figures/fig_q2_gantt.png，
    # 谁后跑谁覆盖——两个脚本合起来跑时，本脚本会把升级版甘特图覆盖回旧版。
    # 甘特图统一由 figures_adv.py 产出，故此处不再调用 fig_gantt()。
    from q2 import precompute_geometry, recommended
    d.geo_nodes, d.geo = precompute_geometry(d)
    trips, assignment, q2_m, q2_order = recommended(d)
    assignment = sorted(assignment, key=lambda x: x['start'])

    # 问题三：直连失效点 + 中继站地图。
    # 中继站必须**读** results/q3_stations.csv，不能手写坐标：手写的那三个点与
    # q3.py 联合优化实际选出的站无关（站数也会随错峰/换站而变），画出来是另一套
    # 东西，属于「图与解不一致」。失效点也按**错峰后**的时刻采样，否则画的是
    # 被推翻的旧排班。
    import pandas as pd
    from q3 import sample_trip_trajectory, direct_ok, shift_assignment
    st_df = pd.read_csv(os.path.join(HERE, '..', 'results', 'q3_stations.csv'), encoding='utf-8-sig')
    stations = [(float(r['悬停经度']), float(r['悬停纬度'])) for _, r in st_df.iterrows()]
    sg_df = pd.read_csv(os.path.join(HERE, '..', 'results', 'q3_stagger.csv'), encoding='utf-8-sig')
    delays = [float(r['推迟s']) for _, r in sg_df.sort_values('架次编号').iterrows()]
    assignment = shift_assignment(d, assignment, delays)
    dead_pts = []
    for a in assignment:
        t = d.transport_types[a['type']]
        pts = sample_trip_trajectory(d, t, a['route'], a['boxes_at'], t0=a['start'])
        for (tt, lon, lat, alt) in pts:
            if not direct_ok(d, lon, lat, alt):
                dead_pts.append((lon, lat))
    relay_stations = stations
    # 绘制失效点热力 + 中继站
    dem = d.dem._dem_nan
    lon_min, lon_max = d.dem.lon_min, d.dem.lon_max
    lat_min, lat_max = d.dem.lat_min, d.dem.lat_max
    fig, ax = plt.subplots(figsize=(8, 7), dpi=120)
    ax.imshow(dem, extent=[lon_min, lon_max, lat_min, lat_max], origin='upper',
              cmap='terrain', aspect='auto', alpha=0.85)
    ax.plot(d.O01['lon'], d.O01['lat'], '*', ms=16, color='gold', mec='black', mew=0.8, zorder=5)
    for s in d.services:
        ax.plot(s['lon'], s['lat'], 'o', ms=4, color='crimson', mec='white', mew=0.4)
    if dead_pts:
        dl = np.array(dead_pts)
        ax.scatter(dl[:, 0], dl[:, 1], s=6, c='red', alpha=0.5, label='直连失效点', zorder=3)
    for i, st in enumerate(relay_stations):
        ax.plot(st[0], st[1], 'D', ms=12, color='blue', mec='white', mew=0.8, zorder=6)
        ax.text(st[0], st[1] + 0.003, f'R{i+1}', color='blue', fontsize=9, ha='center', fontweight='bold')
    # 同 fig_map：'O01' 会被渲染成「001」，与 fig_dem_map 里的服务区编号撞号，改中文。
    ax.text(d.O01['lon'], d.O01['lat'] - 0.004, '调度中心', fontsize=9, ha='center', fontweight='bold')
    slon = [s['lon'] for s in d.services]; slat = [s['lat'] for s in d.services]
    ax.set_xlim(min(slon) - 0.01, max(slon) + 0.01)
    ax.set_ylim(min(slat) - 0.01, max(slat) + 0.01)
    ax.set_xlabel('经度 (°)'); ax.set_ylabel('纬度 (°)')
    ax.set_title('问题三 直连失效区域与中继悬停站')
    ax.legend(loc='upper right')
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_q3_relay.png')); plt.close(fig)
    print('saved fig_q3_relay.png')


if __name__ == '__main__':
    main()
