# -*- coding: utf-8 -*-
# 本程序及代码是在人工智能工具辅助下完成的。
# 工具名称：DeepSeek；版本/型号：DeepSeek V4.1 Flash（API 模型名 deepseek-flash）；
# 开发机构：杭州深度求索人工智能基础技术研究有限公司；
# 版本发布日期：2026 年 9 月 10 日。

"""独立复核脚本：只读结果表与原始数据，重新计算论文第 9 章的检验项。

与 q1~q4.py 的区别在于不调用任何求解器的中间状态——硬约束全部从
results/*.csv 重新算一遍，DEM 离散化误差从原始 .mat 重新采样一遍。
这样做的目的是排除"求解程序自报成功"这一类错误。

用法：python code/verify.py
输出：屏幕报告，对应论文表 9.1、9.2 与 9.4 节的数值。
"""
import sys
import itertools
import os

sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from core import (DEM_STEP, T_DEC, E_DEC, load_data, charge_time, ll_to_xy,
                  roundtrip_energy, segment_time, segment_geometry)

# 结果表按 T_DEC 位存时刻、E_DEC 位存能耗，故独立重算与表内值之间必然存在
# 半个末位的舍入差。容差取 1.01 个末位：既不会把舍入噪声误判为错误，
# 又比任何真实的模型偏差（>1e-3 量级）小得多。
TOL_T = 0.5 * 10 ** (-T_DEC) * 1.01
TOL_E = 0.5 * 10 ** (-E_DEC) * 1.01

RES = os.path.join(HERE, '..', 'results')
# 时刻列保留 T_DEC=3 位、能耗列保留 E_DEC=6 位。从这里反算共享电池的
# "返回时刻 + 充电时长" 时，两个时刻各带 ±5e-4 s，能耗舍入经充电曲线
# 斜率(|dT/dsoc| = T_full·0.65/0.9 ≈ 2167 s)放大后约 ±1.4e-4 s，
# 合计不超过 1.5e-3 s。取 0.01 s 作容差：比舍入噪声高一个量级留足余量，
# 又比任何真实的调度冲突小两个量级，不会把真冲突放过去。
TOL = 0.01


def check_hard_constraints(d):
    """表 9.1：问题二推荐方案的硬约束独立复核。"""
    trips = pd.read_csv(os.path.join(RES, 'q2_transport_trips.csv'), encoding='utf-8-sig')
    deliv = pd.read_csv(os.path.join(RES, 'q2_box_delivery.csv'), encoding='utf-8-sig')
    boxes = {b['id']: b for b in d.cargo}
    bad = []

    # 1) 能量裕度 E <= (1-rho) E_use
    ratios = []
    for _, r in trips.iterrows():
        t = d.transport_types[r['机型编号']]
        cap = (1 - t['rho']) * t['E_use']
        ratios.append(r['架次能耗kWh'] / cap)
        if r['架次能耗kWh'] > cap + 1e-9:
            bad.append('架次 %s 能耗超限' % r['架次编号'])
    ratios = np.array(ratios)

    # 2) 实体无人机不重叠：同一架无人机两架次的时间区间不得相交
    n_uav = 0
    for uid, g in trips.groupby('无人机编号'):
        iv = sorted(zip(g['开始时刻s'], g['返回O01时刻s'], g['架次编号']))
        for a, b in zip(iv, iv[1:]):
            if b[0] < a[1] - TOL:
                n_uav += 1
                bad.append('无人机 %s: %s 与 %s 重叠' % (uid, a[2], b[2]))

    # 3) 共享电池周转：电池占用到充电完成为止，充电未完不得再起飞
    n_bat, gaps = 0, []
    for bid, g in trips.groupby('电池编号'):
        gg = bid[0]
        iv = []
        for _, r in g.iterrows():
            soc = 1.0 - r['架次能耗kWh'] / d.transport_types[gg]['E_use']
            iv.append((r['开始时刻s'],
                       r['返回O01时刻s'] + charge_time(soc, d.batteries[gg][1]),
                       r['架次编号']))
        iv.sort()
        for a, b in zip(iv, iv[1:]):
            gaps.append(b[0] - a[1])
            if b[0] < a[1] - TOL:
                n_bat += 1
                bad.append('电池 %s: %s 与 %s 周转冲突' % (bid, a[2], b[2]))
    gaps = np.array(gaps)

    # 4) 物资时限：医疗物资的期望送达时刻、首批保障货箱的截止时刻
    n_late = 0
    for _, r in deliv.iterrows():
        b = boxes[r['货箱编号']]
        hard = None
        if b['category'] == '医疗物资' and b['expect'] is not None:
            hard = ('医疗期望', b['expect'])
        if b['first_batch']:
            hard = ('首批截止', b['deadline'])
        if hard and r['交付完成时刻s'] > hard[1] + TOL:
            n_late += 1
            bad.append('货箱 %s %s 超限' % (r['货箱编号'], hard[0]))

    # 5) 交付完整性
    dup = int(deliv['货箱编号'].duplicated().sum())
    miss = set(boxes) - set(deliv['货箱编号'])
    orphan = set(deliv['架次编号']) - set(trips['架次编号'])
    if miss:
        bad.append('漏交付 %d 箱' % len(miss))

    # 6) 资源库存
    inv = {}
    for g in ['A', 'B', 'C']:
        sub = trips[trips['机型编号'] == g]
        n_own = len([u for u in d.uavs if u['type'] == g])
        inv[g] = (sub['无人机编号'].nunique(), n_own,
                  sub['电池编号'].nunique(), d.batteries[g][0])
        if inv[g][0] > n_own or inv[g][2] > d.batteries[g][0]:
            bad.append('%s 型资源超配' % g)

    print('架次 %d，交付记录 %d，货箱总数 %d' % (len(trips), len(deliv), len(boxes)))
    print('  能量裕度占用比   最大 %.3f  均值 %.3f  越限 %d'
          % (ratios.max(), ratios.mean(), int((ratios > 1).sum())))
    print('  无人机时段重叠   %d 处' % n_uav)
    print('  电池周转冲突     %d 处（最小间隙 %.4f s，容差 %.2f s）'
          % (n_bat, gaps.min() if gaps.size else 0.0, TOL))
    print('  物资时限违约     %d 箱' % n_late)
    print('  交付完整性       重复 %d，漏交 %d，孤儿架次引用 %d'
          % (dup, len(miss), len(orphan)))
    for g in ['A', 'B', 'C']:
        a, b, c, e = inv[g]
        print('  %s 型资源         用机 %d/%d，用电池 %d/%d' % (g, a, b, c, e))
    print('  => %s' % ('硬约束全部满足' if not bad else '发现 %d 处问题' % len(bad)))
    for x in bad[:10]:
        print('     !', x)
    return not bad


def check_relay(d):
    """表 9.2：问题三中继方案的硬约束独立复核。

    同样只读 q3_relay_trips.csv / q3_comm_phases.csv 两张结果表：悬停离地高度
    不采信结果表里记的数值，而是拿表里的经纬度回原始 DEM 重新取地面高程再算；
    能源组件的周转时刻由"返回时刻 + 充电时长"从能耗列反算，与求解器内部状态无关。
    """
    rt = pd.read_csv(os.path.join(RES, 'q3_relay_trips.csv'), encoding='utf-8-sig')
    cp = pd.read_csv(os.path.join(RES, 'q3_comm_phases.csv'), encoding='utf-8-sig')
    R = d.relay_type
    cap = (1 - R['rho']) * R['E_use']
    bad = []

    # 1) 能量裕度
    ratios = (rt['架次能耗kWh'] / cap).to_numpy()
    for _, r in rt.iterrows():
        if r['架次能耗kWh'] > cap + 1e-9:
            bad.append('中继架次 %s 能耗超限' % r['中继架次编号'])

    # 2) 悬停离地高度 = 表内悬停海拔 − 该经纬度处的 DEM 地面高程。
    # 容差取 0.05 m，不是数值噪声而是结果表自身的精度：经纬度留 6 位（≈0.11 m）、
    # 海拔留 3 位（±5e-4 m）。方案按上限 300 m 设计（题给上限，越高覆盖越好），
    # 复核精度必须细于"贴限"的余量，否则会把合规方案误判为越限。
    HGT_TOL = 0.05
    agl = np.array([r['悬停海拔m'] - float(d.dem.sample(r['悬停经度'], r['悬停纬度']))
                    for _, r in rt.iterrows()])
    over_h = int((agl > R['max_hover_alt'] + HGT_TOL).sum())
    if over_h:
        bad.append('%d 个中继架次悬停高度超限' % over_h)

    # 3) 中继无人机：同一架的架次时段不得重叠
    n_uav = 0
    for rid, g in rt.groupby('中继无人机编号'):
        iv = sorted(zip(g['开始时刻s'], g['返回O01时刻s'], g['中继架次编号']))
        for a, b in zip(iv, iv[1:]):
            if b[0] < a[1] - TOL:
                n_uav += 1
                bad.append('中继 %s: %s 与 %s 重叠' % (rid, a[2], b[2]))

    # 4) 能源组件：占用持续到充电完成，充电未完不得再投入
    n_comp = 0
    for cid, g in rt.groupby('能源组件编号'):
        iv = []
        for _, r in g.iterrows():
            soc = 1.0 - r['架次能耗kWh'] / R['E_use']
            iv.append((r['开始时刻s'],
                       r['返回O01时刻s'] + charge_time(soc, d.relay_batteries[1]),
                       r['中继架次编号']))
        iv.sort()
        for a, b in zip(iv, iv[1:]):
            if b[0] < a[1] - TOL:
                n_comp += 1
                bad.append('能源组件 %s: %s 与 %s 周转冲突' % (cid, a[2], b[2]))

    # 5) 结果表自洽：标为「中继」的时段必须真的落在某个架次的服务窗口内，
    #    且该架次编号可查；标为「中断」的不得又挂着架次编号。
    #    这一条正是为了防住"把中断记成中继"这类表内自相矛盾。
    wins = {r['中继架次编号']: (r['建链完成时刻s'], r['服务结束时刻s'])
            for _, r in rt.iterrows()
            if r['服务结束时刻s'] > r['建链完成时刻s'] + 1e-9}
    n_mis = 0
    for _, r in cp.iterrows():
        rid = r['中继架次编号']
        has_id = isinstance(rid, str) and rid.strip() != ''
        if r['保障方式'] == '中继':
            if not has_id or rid not in wins:
                n_mis += 1
            elif not (wins[rid][0] - 1e-6 <= r['开始时刻s']
                      and r['结束时刻s'] <= wins[rid][1] + 1e-6):
                n_mis += 1
        elif r['保障方式'] == '中断' and has_id:
            n_mis += 1
    if n_mis:
        bad.append('通信保障表 %d 段的标注与其架次服务窗口不符' % n_mis)

    # 三段占比不在这里复算：本表按"段"记录，段首尾取点与逐时刻统计存在半个采样
    # 步长的口径差，照段复算会与逐时刻数字差约 0.5 个百分点，两个都对却看着像
    # 矛盾。占比的时间积分口径由下面的 check_relay_final 独立重采样复算（表 9.3），
    # 那里才是"中断 = 0"的复算处；本表只核资源约束与表内自洽。
    print('中继架次 %d，通信保障分段 %d' % (len(rt), len(cp)))
    print('  能量裕度占用比   最大 %.3f（可用 %.2f kWh/架次）  越限 %d'
          % (ratios.max(), cap, int((ratios > 1).sum())))
    print('  悬停离地高度     最大 %.1f m（上限 %.0f m）  越限 %d'
          % (agl.max(), R['max_hover_alt'], over_h))
    print('  中继时段重叠     %d 处' % n_uav)
    print('  能源组件周转冲突 %d 处' % n_comp)
    print('  结果表标注不符   %d 段' % n_mis)
    print('  => %s' % ('中继方案全部检验项通过' if not bad else '发现 %d 处问题' % len(bad)))
    for x in bad[:10]:
        print('     !', x)
    return not bad


def _q3_trips(d):
    """从 q2 结果表 + 错峰表重建**错峰后**的运输架次（路线、各区箱数、开始时刻）。"""
    trips = pd.read_csv(os.path.join(RES, 'q2_transport_trips.csv'), encoding='utf-8-sig')
    deliv = pd.read_csv(os.path.join(RES, 'q2_box_delivery.csv'), encoding='utf-8-sig')
    sg = pd.read_csv(os.path.join(RES, 'q3_stagger.csv'), encoding='utf-8-sig')
    shift = {r['架次编号']: float(r['推迟s']) for _, r in sg.iterrows()}
    out = []
    for _, r in trips.iterrows():
        tid = r['架次编号']
        route = [s for s in str(r['访问服务区顺序']).split('->') if s]
        sub = deliv[deliv['架次编号'] == tid]
        boxes_at = {sid: list(sub[sub['服务区编号'] == sid]['货箱编号']) for sid in route}
        out.append(dict(id=tid, type=r['机型编号'], route=route, boxes_at=boxes_at,
                        start=float(r['开始时刻s']) + shift.get(tid, 0.0),
                        dur=float(r['返回O01时刻s']) - float(r['开始时刻s'])))
    return out


def _q3_traj(d, a, dt):
    """独立重建架次三维轨迹：只用 core 的 segment_geometry，不调用 q3.py 的采样器。

    运动学与 q3.sample_trip_trajectory 同源（爬升垂直、巡航水平、下降垂直，
    投送期间定点悬停首尾各记一点），但这里是**另写一遍**——若只 import 求解器
    的采样器，采样这一层就失去了独立性，而"轨迹在哪里断了链路"正是本项要查的。
    """
    t = d.transport_types[a['type']]
    alt_o = d.O01['alt']

    def ll(nid):
        return (d.O01['lon'], d.O01['lat']) if nid == 'O01' else (d.si[nid]['lon'], d.si[nid]['lat'])

    def alt_of(nid):
        return alt_o if nid == 'O01' else d.si[nid]['alt'] + 30.0

    pts = []
    tc0 = a['start']
    pts.append((tc0, ll('O01')[0], ll('O01')[1], alt_o))
    tc0 += t['prep'] + t['load_per_box'] * sum(len(v) for v in a['boxes_at'].values())
    pts.append((tc0, ll('O01')[0], ll('O01')[1], alt_o))
    prev = 'O01'
    for sid in list(a['route']) + ['O01']:
        la, lb = ll(prev), ll(sid)
        za, zb = alt_of(prev), alt_of(sid)
        g = segment_geometry(d.dem, la[0], la[1], za, lb[0], lb[1], zb)
        for span, kind in ((g['climb'] / t['v_up'], 'up'),
                           (g['d'] / t['v_cruise'], 'cr'),
                           (g['descent'] / t['v_down'], 'dn')):
            n = max(2, int(np.ceil(span / dt)))
            for k in range(1, n + 1):
                fr = k / n
                if kind == 'up':
                    pts.append((tc0 + span * fr, la[0], la[1], za + (g['cruise_alt'] - za) * fr))
                elif kind == 'cr':
                    pts.append((tc0 + span * fr, la[0] + (lb[0] - la[0]) * fr,
                                la[1] + (lb[1] - la[1]) * fr, g['cruise_alt']))
                else:
                    pts.append((tc0 + span * fr, lb[0], lb[1],
                                g['cruise_alt'] - (g['cruise_alt'] - zb) * fr))
            tc0 += span
        if sid != 'O01':
            h = t['hand_base'] + t['hand_per_box'] * len(a['boxes_at'][sid])
            pts.append((tc0, lb[0], lb[1], zb))          # 悬停首点
            pts.append((tc0 + h, lb[0], lb[1], zb))      # 悬停末点，积分不留空洞
            tc0 += h
        prev = sid
    return pts


def _time_weights(ts):
    """时间占比口径：w_i = (t_{i+1} − t_{i−1})/2（端点取半格）。"""
    n = len(ts)
    ts = np.asarray(ts, dtype=float)
    if n == 1:
        return np.zeros(1)
    w = np.empty(n)
    w[0] = 0.5 * (ts[1] - ts[0])
    w[-1] = 0.5 * (ts[-1] - ts[-2])
    if n > 2:
        w[1:-1] = 0.5 * (ts[2:] - ts[:-2])
    return w


def check_relay_final(d, dt=1.0, frac_tol=0.01):
    """表 9.3：问题三**终点方案**的独立复核（对应 3.5/3.6/3.7 三项要求）。

    与 check_relay 的分工：那张表查的是"结果表内部自洽 + 资源硬约束"，不重算
    通信；这张表按 Δt=1 s **逐时刻重判通信状态**，是"中断 = 0"这个结论的独立
    复算。链路裕量直接用 core.CommModel.link_ok 重算，不调用 q3.py 里的
    direct_margin / access_margin / station_covers，也不调用它的终检 audit。

    三项要求逐条落地：
      3.5 无中继架次绑定的中继区间 = 0：q3_intervals.csv 每一行都必须有可查的
          中继架次编号，且该区间的时段必须落在这个架次的服务窗口内；
      3.6 占比按时间积分（不是采样点个数），并与 q3_coverage.csv 对照；
      3.7 逐时刻判：先判直连；失败则须存在一架**此刻正在服务**的中继，其所在
          悬停站对该点接入裕量 ≥ 0；两者皆不成立即为中断，必须为 0。
    """
    trips = _q3_trips(d)
    rt = pd.read_csv(os.path.join(RES, 'q3_relay_trips.csv'), encoding='utf-8-sig')
    stn = pd.read_csv(os.path.join(RES, 'q3_stations.csv'), encoding='utf-8-sig')
    iv = pd.read_csv(os.path.join(RES, 'q3_intervals.csv'), encoding='utf-8-sig')
    cp = pd.read_csv(os.path.join(RES, 'q3_comm_phases.csv'), encoding='utf-8-sig')
    cov = pd.read_csv(os.path.join(RES, 'q3_coverage.csv'), encoding='utf-8-sig')
    bad = []

    # ---- 3.5 中继区间必须绑定到真实存在、且此刻在役的中继架次 ----
    win = {r['中继架次编号']: (float(r['建链完成时刻s']), float(r['服务结束时刻s']))
           for _, r in rt.iterrows()}
    n_unbound = 0
    for _, r in iv.iterrows():
        rid = r['中继架次编号']
        if not (isinstance(rid, str) and rid.strip()):
            n_unbound += 1
            bad.append('失效区间 %s 无中继架次绑定' % r['失效区间编号'])
            continue
        if rid not in win:
            n_unbound += 1
            bad.append('失效区间 %s 绑定了不存在的中继架次 %s' % (r['失效区间编号'], rid))
            continue
        if not (win[rid][0] - 1e-6 <= r['区间起点s'] and r['区间终点s'] <= win[rid][1] + 1e-6):
            n_unbound += 1
            bad.append('失效区间 %s 的时段超出 %s 的服务窗口' % (r['失效区间编号'], rid))

    # ---- 3.7 逐时刻重判 + 3.6 时间占比 ----
    site = {r['悬停站编号']: (r['悬停经度'], r['悬停纬度'], r['悬停海拔m'])
            for _, r in stn.iterrows()}
    wins = []
    for _, r in rt.iterrows():
        s = site[r['悬停站编号']]
        if float(r['服务结束时刻s']) > float(r['建链完成时刻s']) + 1e-9:
            wins.append((float(r['建链完成时刻s']), float(r['服务结束时刻s']), s,
                         r['中继架次编号']))
    gw = (d.O01['lon'], d.O01['lat'], d.O01['alt'] + d.comm_params['gw_h'])
    Lmax_d, Lmax_a = d.comm.Lmax_uw_gw, d.comm.Lmax_uw_r

    tot = dict(direct=0.0, relay=0.0, gap=0.0)
    gaps = []
    labels = {}          # (架次, 采样下标) -> 判定，供与 q3_comm_phases 对照
    for a in trips:
        pts = _q3_traj(d, a, dt)
        w = _time_weights([p[0] for p in pts])
        for k, (tt, lon, lat, alt) in enumerate(pts):
            if d.comm.link_ok(d.dem, (lon, lat, alt), gw, Lmax_d)[1] <= Lmax_d + 1e-9:
                tot['direct'] += w[k]
                labels[(a['id'], k)] = '直连'
                continue
            hit = None
            for (w1, w2, s, rid) in wins:
                if w1 - 1e-6 <= tt <= w2 + 1e-6 and \
                        d.comm.link_ok(d.dem, (lon, lat, alt), s, Lmax_a)[1] <= Lmax_a + 1e-9:
                    hit = rid
                    break
            if hit:
                tot['relay'] += w[k]
                labels[(a['id'], k)] = '中继'
            else:
                tot['gap'] += w[k]
                labels[(a['id'], k)] = '中断'
                gaps.append((a['id'], tt, lon, lat, alt))
    T = sum(tot.values())
    frac = {k: (v / T if T > 0 else 0.0) for k, v in tot.items()}
    if tot['gap'] > 1e-9:
        bad.append('%d 个采样点处于中断，累计 %.1f s' % (len(gaps), tot['gap']))

    # ---- 3.6 与求解器自报的时间占比对照 ----
    row = cov[cov['口径'] == '联合优化']
    claim = None
    if len(row):
        claim = row.iloc[0]
        for k, col in (('direct', '直连时间占比'), ('relay', '中继时间占比'),
                       ('gap', '中断时间占比')):
            if abs(frac[k] - float(claim[col])) > frac_tol:
                bad.append('时间占比与 q3_coverage.csv 不符：%s 复核 %.4f vs 自报 %.4f'
                           % (k, frac[k], float(claim[col])))

    # ---- 3.6 分段表必须无缝铺满每个架次，且不得出现「中断」段 ----
    # 段按 [首采样点, 末采样点] 记，相邻两段之间天然差一个采样步长（后段首点就是
    # 前段末点的下一个采样点），故判据是 0 ≤ 间隙 ≤ dt，不是严格的 0。超过 dt 的
    # 空洞才是真的没标注；负间隙表示两段重叠，同样不正常。
    n_tile, n_gap_seg = 0, 0
    for tid, g in cp.groupby('运输架次编号'):
        g = g.sort_values('开始时刻s')
        cur = None
        for _, r in g.iterrows():
            if r['通信阶段'] == '中断':
                n_gap_seg += 1
            if cur is not None:
                hole = r['开始时刻s'] - cur
                if hole > dt + TOL or hole < -TOL:
                    n_tile += 1
            cur = r['结束时刻s']
    if n_tile:
        bad.append('通信保障表有 %d 处分段不连续（存在未标注的空洞）' % n_tile)
    if n_gap_seg:
        bad.append('通信保障表仍有 %d 段标注为「中断」' % n_gap_seg)
    if set(cp['运输架次编号']) != {a['id'] for a in trips}:
        bad.append('通信保障表未覆盖全部运输架次')

    print('失效区间 %d 个；中继架次 %d 个；通信保障分段 %d 段' % (len(iv), len(rt), len(cp)))
    print('  3.5 无中继架次绑定的中继区间 %d 个（要求 0）' % n_unbound)
    print('  3.6 时间占比（Δt=%.1fs，总时长 %.0f s）：直连 %.4f | 中继 %.4f | 中断 %.4f'
          % (dt, T, frac['direct'], frac['relay'], frac['gap']))
    if claim is not None:
        print('      与 q3_coverage.csv 自报值最大偏差 %.4f（容差 %.2f）'
              % (max(abs(frac[k] - float(claim[c])) for k, c in
                     (('direct', '直连时间占比'), ('relay', '中继时间占比'),
                      ('gap', '中断时间占比'))), frac_tol))
    print('  3.6 分段连续性：不连续 %d 处，标为「中断」的段 %d 段（均要求 0）'
          % (n_tile, n_gap_seg))
    print('  3.7 逐时刻重判的中断点 %d 个（要求 0）' % len(gaps))
    for g in gaps[:5]:
        print('     ! %s t=%.1f s (%.5f, %.5f, %.1f m)' % g)
    print('  => %s' % ('通信连续性独立复核通过' if not bad else '发现 %d 处问题' % len(bad)))
    for x in bad[:10]:
        print('     !', x)
    return not bad


def _node_pairs(d):
    nodes = [('O01', d.O01)] + [(s['id'], s) for s in d.services]
    return list(itertools.combinations(nodes, 2))


def _dense_cells(dem, lon_a, lat_a, lon_b, lat_b, step_m):
    """超密采样得到的像元集合，用作栅格遍历的参照。"""
    xa, ya = ll_to_xy(lon_a, lat_a); xb, yb = ll_to_xy(lon_b, lat_b)
    dist = float(np.hypot(xb - xa, yb - ya))
    n = max(2, int(np.ceil(dist / step_m)) + 1)
    tt = np.linspace(0, 1, n)
    fi = (dem.lat_max - (lat_a + (lat_b - lat_a) * tt)) / dem.dlat
    fj = ((lon_a + (lon_b - lon_a) * tt) - dem.lon_min) / dem.dlon
    ii = np.floor(fi).astype(int); jj = np.floor(fj).astype(int)
    ok = (ii >= 0) & (ii < dem.dem.shape[0]) & (jj >= 0) & (jj < dem.dem.shape[1])
    return set(zip(ii[ok].tolist(), jj[ok].tolist()))


def _max_over_cells(dem, cells):
    if not cells:
        return 0.0
    cells = list(cells)
    ii = np.fromiter((c[0] for c in cells), dtype=np.intp, count=len(cells))
    jj = np.fromiter((c[1] for c in cells), dtype=np.intp, count=len(cells))
    z = dem._dem_nan[ii, jj]
    z = z[np.isfinite(z)]
    return float(z.max()) if z.size else 0.0


def _legacy_max_elev(dem, lon_a, lat_a, lon_b, lat_b, step=5.0):
    """口径修正前的实现（保留以便量化口径差）：对双线性插值面按 step 采样取最大。"""
    xa, ya = ll_to_xy(lon_a, lat_a); xb, yb = ll_to_xy(lon_b, lat_b)
    dist = float(np.hypot(xb - xa, yb - ya))
    n = max(2, int(np.ceil(dist / step)) + 1)
    tt = np.linspace(0, 1, n)
    z = dem.sample(lon_a + (lon_b - lon_a) * tt, lat_a + (lat_b - lat_a) * tt)
    z = z[np.isfinite(z)]
    return float(z.max()) if z.size else 0.0


def check_dem_caliber(d, step_probe=0.5):
    """9.4 节：巡航高度所依据的 DEM 口径，及其与旧口径之差。

    题面附录 2 要求"计划巡航海拔取该航段所经过 DEM 像元的最高地面高程以上
    50 m"。本文对该量做栅格遍历（core.DEMGrid._cells_along，Amanatides–Woo），
    取航段穿过的每一个 30m 像元的**原始**高程最大值。这里独立复核两件事：

      (1) 遍历本身对不对。用超密采样得到的像元集合当参照，查有没有漏掉航段
          真正穿过的像元。漏掉即低估，是本项唯一的危险方向，必须为 0；
          多出来的像元只可能是航线擦角而过的格子，方向保守，单独计数。
      (2) 口径差有多大。同一批航段上比较"像元遍历"与"对双线性插值面采样取
          最大"两种口径——后者是本文早期版本用过的算法。

    刻意把 (2) 单独列出来，是因为旧的 check_dem_step 拿"1m 采样的插值面"当
    真值，按构造就测不到插值-像元之间的口径差，只能测出采样步长自身的残差，
    于是把一个量级更大的口径误差漏在了检查之外。
    """
    dem = d.dem
    pairs = _node_pairs(d)
    n_miss = n_extra = 0
    cap_miss = 0.0
    worst_trav = []
    for (i1, n1), (i2, n2) in pairs:
        trav = set(dem._cells_along(n1['lon'], n1['lat'], n2['lon'], n2['lat']))
        dense = _dense_cells(dem, n1['lon'], n1['lat'], n2['lon'], n2['lat'], step_probe)
        miss = dense - trav
        n_miss += len(miss); n_extra += len(trav - dense)
        if miss:
            # 漏像元直接折算成可能的高程低估量
            cap_miss = max(cap_miss, _max_over_cells(dem, dense) - _max_over_cells(dem, trav))
        worst_trav.append((len(miss), i1, i2))
    worst_trav.sort(reverse=True)
    print('(1) 栅格遍历 vs %.1fm 超密采样（%d 条航段）' % (step_probe, len(pairs)))
    print('    漏像元 %d 个（必须为 0，漏即低估），擦角多计像元 %d 个（方向保守）'
          % (n_miss, n_extra))
    if n_miss:
        print('    !! 漏像元最多：' + '，'.join('%s→%s %d个' % (w[1], w[2], w[0])
                                                for w in worst_trav[:3]))
    print('    漏像元导致的最大高程低估 %.3f m' % cap_miss)

    # (2) 口径差：像元原始值 vs 插值面采样
    rows = []
    for (i1, n1), (i2, n2) in pairs:
        e_cell = dem.max_elev_along(n1['lon'], n1['lat'], n2['lon'], n2['lat'])
        e_leg = _legacy_max_elev(dem, n1['lon'], n1['lat'], n2['lon'], n2['lat'])
        rows.append((e_cell - e_leg, i1, i2))
    arr = np.array([r[0] for r in rows])
    rows.sort(reverse=True)
    print('(2) 口径差（像元遍历 − 插值面采样，%d 条航段；正值 = 旧口径低估）' % len(rows))
    print('    低估 %d 条 (%.0f%%)，高估 %d 条，一致 %d 条'
          % (int((arr > 1e-9).sum()), 100.0 * (arr > 1e-9).mean(),
             int((arr < -1e-9).sum()), int((np.abs(arr) <= 1e-9).sum())))
    print('    最大低估 +%.2f m（%s→%s），最大高估 %.2f m，均值 %+.3f m，中位 %+.3f m'
          % (arr.max(), rows[0][1], rows[0][2], arr.min(), arr.mean(), np.median(arr)))
    print('    |口径差| ≥ 1/5/10 m 的航段数：%d / %d / %d'
          % (int((np.abs(arr) >= 1).sum()), int((np.abs(arr) >= 5).sum()),
             int((np.abs(arr) >= 10).sum())))
    print('    说明：低估 H_max -> 低估巡航海拔 -> 低估爬升量、时间与能耗，属**非保守**')
    print('    误差；高估方向同样存在（航线贴近像元边缘时插值混入了未经过的邻元）。')
    return n_miss, arr


def check_occlusion_step(d, step_coarse=DEM_STEP, step_fine=1.0):
    """9.4 节：DEM_STEP 现在只管视线遮挡判定，检验它是否改变遮挡结论。

    口径修正后，巡航高度改走像元遍历，采样步长只剩 line_occluded 一个用途。
    故这里问的不再是"最高地形被低估多少"，而是"步长是否会把遮挡判反"。
    除翻转次数外还报告连线相对地形的净空余量：余量贴近 0 的连线才是脆弱点，
    只看翻转率会把"本来就有几百米余量"的绝大多数连线混进来稀释掉。
    """
    dem = d.dem
    flips, clear = [], []
    for (i1, n1), (i2, n2) in _node_pairs(d):
        a1 = n1['alt'] if i1 == 'O01' else n1['alt'] + 30.0
        a2 = n2['alt'] if i2 == 'O01' else n2['alt'] + 30.0
        geo = segment_geometry(dem, n1['lon'], n1['lat'], a1, n2['lon'], n2['lat'], a2)
        pa = (n1['lon'], n1['lat'], geo['cruise_alt'])
        pb = (n2['lon'], n2['lat'], geo['cruise_alt'])
        oc_c = dem.line_occluded(pa, pb, step=step_coarse)
        oc_f = dem.line_occluded(pa, pb, step=step_fine)
        if oc_c != oc_f:
            flips.append((i1, i2, oc_c, oc_f))
        clear.append(_clearance(dem, pa, pb, step_fine))
    clear = np.array(clear)
    print('(3) 视线遮挡判定的步长敏感性（%d 条节点连线，%.0fm vs %.0fm）'
          % (len(clear), step_coarse, step_fine))
    print('    判定翻转 %d 条' % len(flips))
    for f in flips[:5]:
        print('      ! %s→%s 粗步长=%s 细步长=%s' % f)
    print('    沿线净空余量（正=不遮挡）：最小 %.1f m，中位 %.1f m，<1m 的连线 %d 条'
          % (clear.min(), np.median(clear), int((np.abs(clear) < 1.0).sum())))
    return len(flips)


def _clearance(dem, pa, pb, step=1.0):
    """视线连线相对沿途地形的净空余量(m)，负值表示被地形挡住。"""
    xa, ya = ll_to_xy(pa[0], pa[1]); xb, yb = ll_to_xy(pb[0], pb[1])
    dist = float(np.hypot(xb - xa, yb - ya))
    n = max(2, int(np.ceil(dist / step)) + 1)
    tt = np.linspace(0, 1, n)
    z = dem.sample(pa[0] + (pb[0] - pa[0]) * tt, pa[1] + (pb[1] - pa[1]) * tt)
    line_alt = pa[2] + (pb[2] - pa[2]) * tt
    ok = np.isfinite(z)
    if not ok.any():
        return np.inf
    return float(np.min(line_alt[ok] - z[ok]))


def check_bilinear_bound(d, n_probe=20000, seed=0):
    """9.4 节：双线性插值的自洽性——插值面被所在单元格四角高程夹住。

    **本项只证明插值作为一种"面"是自洽的，不能用来论证巡航高度口径正确。**
    恰恰相反：正因为插值值是四角高程的凸组合、必被四角夹住，它沿线的最大值
    结构性地不高于所经过像元的原始最大值——这正是旧口径系统性低估 H_max 的
    成因。该性质的实际用途只有一个：line_occluded 沿连线做插值采样判定遮挡时，
    不会因插值本身造出高于局部真实地形的虚假尖峰而误判为遮挡。
    """
    rng = np.random.default_rng(seed)
    lon = rng.uniform(d.dem.lon_min, d.dem.lon_max, n_probe)
    lat = rng.uniform(d.dem.lat_min, d.dem.lat_max, n_probe)
    z = d.dem.sample(lon, lat)
    ok = np.isfinite(z)
    lon, lat, z = lon[ok], lat[ok], z[ok]

    # 取每个探针所在单元格的四角原始高程
    fj = (lon - d.dem.lon_min) / d.dem.dlon
    fi = (d.dem.lat_max - lat) / d.dem.dlat
    j0 = np.floor(fj).astype(int)
    i0 = np.floor(fi).astype(int)
    m = (i0 >= 0) & (i0 + 1 < d.dem.dem.shape[0]) & (j0 >= 0) & (j0 + 1 < d.dem.dem.shape[1])
    i0, j0, z = i0[m], j0[m], z[m]
    Z = d.dem._dem_nan
    corners = np.vstack([Z[i0, j0], Z[i0, j0 + 1], Z[i0 + 1, j0], Z[i0 + 1, j0 + 1]])
    lo = np.nanmin(corners, axis=0)
    hi = np.nanmax(corners, axis=0)
    good = np.isfinite(lo) & np.isfinite(hi)
    over = np.sum((z[good] > hi[good] + 1e-9) | (z[good] < lo[good] - 1e-9))
    print('双线性插值探针 %d 个：越出四角高程区间的 %d 个（应为 0）' % (good.sum(), over))
    return over == 0


def check_q1(d):
    """表 9.0：问题一推荐组批方案的独立复核。

    只读 results/q1_recommended_batching.csv 与原始数据，用 core 的物理函数
    重新计算每架次的质量、体积、能耗与时间，不复用 q1.py 的任何中间量。
    """
    rec = pd.read_csv(os.path.join(RES, 'q1_recommended_batching.csv'),
                      encoding='utf-8-sig')
    summ = pd.read_csv(os.path.join(RES, 'q1_batching_summary.csv'),
                       encoding='utf-8-sig', index_col=0)
    box = {}
    for si_id in d.S:
        for b in d.boxes_by_service[si_id]:
            box[b['id']] = (si_id, b['mass'], b['volume'])

    seen, dup, orphan = [], 0, 0
    n_over_m = n_over_v = n_over_e = 0
    dE = dT = dSOC = 0.0
    for _, r in rec.iterrows():
        si_id = r['服务区编号']
        tid = r['机型编号']
        t = d.transport_types[tid]
        ids = [x for x in str(r['货箱编号列表']).split(';') if x]
        m = v = 0.0
        for bid in ids:
            if bid not in box:
                orphan += 1
                continue
            seen.append(bid)
            m += box[bid][1]
            v += box[bid][2]
        # 质量与体积必须与表内一致，且不超载
        dE_m = abs(m - float(r['总质量kg']))
        dE_v = abs(v - float(r['总体积m3']))
        if m > t['Q'] + 1e-6:
            n_over_m += 1
        if v > t['volume'] + 1e-6:
            n_over_v += 1
        if dE_m > 0.005 or dE_v > 5e-5:
            print('    !! %s 质量/体积与货箱清单不符 (d=%.4f, %.6f)'
                  % (r['架次编号'], dE_m, dE_v))
        # 能耗：用 core 独立重算往返能耗，并核对安全余量
        E, go, back = roundtrip_energy(t, d.dem, d.O01, d.si[si_id], m)
        if E > (1 - t['rho']) * t['E_use'] + 1e-9:
            n_over_e += 1
        dE = max(dE, abs(E - float(r['架次能耗kWh'])))
        dSOC = max(dSOC, abs((1.0 - E / t['E_use']) - float(r['返航SOC'])))
        # 时间：仿射式 T = (prep+hand_base+t_go+t_back) + (load+hand)*n
        t_aff = (t['prep'] + t['hand_base'] + segment_time(t, go)
                 + segment_time(t, back) + (t['load_per_box'] + t['hand_per_box']) * len(ids))
        dT = max(dT, abs(t_aff - float(r['往返时间s'])))
        if abs(m - float(r['总质量kg'])) > 0.005:
            print('    !! %s 质量不符' % r['架次编号'])
    cnt = {}
    for b in seen:
        cnt[b] = cnt.get(b, 0) + 1
    dup = sum(1 for v in cnt.values() if v > 1)
    total_boxes = sum(len(d.boxes_by_service[s]) for s in d.S)
    miss = total_boxes - len(cnt)

    # 逐机型架次：推荐方案里每个服务区只用一个机型，故该机型的架次数
    # 应等于 q1_batching_summary 中"被分配到该机型的那些服务区"的架次之和
    per_type = rec.groupby('机型编号').size().to_dict()
    assigned = rec.groupby('服务区编号')['机型编号'].first().to_dict()
    chk = {}
    for si_id, tid in assigned.items():
        chk[tid] = chk.get(tid, 0) + int(summ.loc[si_id, '%s_架次' % tid])
    type_ok = (chk == per_type)

    print('架次 %d，货箱引用 %d，货箱总数 %d' % (len(rec), len(seen), total_boxes))
    print('  覆盖完整性       重复 %d，漏交 %d，孤儿引用 %d' % (dup, miss, orphan))
    print('  载重/容积越限    质量 %d 处，体积 %d 处' % (n_over_m, n_over_v))
    print('  能耗安全余量越限 %d 处（判据 E <= (1-rho)E_use）' % n_over_e)
    print('  独立重算最大偏差 能耗 %.3e kWh，时间 %.3e s，返航SOC %.3e'
          % (dE, dT, dSOC))
    print('  机型架次核对     明细表 %s vs 汇总表 %s -> %s'
          % (per_type, chk, '一致' if type_ok else '不一致'))
    print('  舍入容差         能耗 %.1e kWh，时间 %.1e s（各 1 个末位）' % (TOL_E, TOL_T))
    ok = (dup == 0 and miss == 0 and orphan == 0 and n_over_m == 0
          and n_over_v == 0 and n_over_e == 0 and type_ok
          and dE <= TOL_E and dT <= TOL_T and dSOC <= TOL_E)
    print('  => %s' % ('问题一方案全部检验项通过' if ok else '存在问题，见上方 !! 行'))
    return ok


def main():
    d = load_data()
    print('=' * 72)
    print('表 9.0  问题一推荐组批方案独立复核')
    print('=' * 72)
    check_q1(d)
    print()
    print('=' * 72)
    print('表 9.1  问题二硬约束独立复核')
    print('=' * 72)
    check_hard_constraints(d)
    print()
    print('=' * 72)
    print('表 9.2  问题三中继方案独立复核')
    print('=' * 72)
    check_relay(d)
    print()
    print('=' * 72)
    print('表 9.3  通信连续性独立复核（Δt=1 s 逐时刻重判）')
    print('=' * 72)
    check_relay_final(d)
    print()
    print('=' * 72)
    print('9.4 节  DEM 口径与离散化误差')
    print('=' * 72)
    n_miss, _ = check_dem_caliber(d)
    print()
    check_occlusion_step(d)
    print()
    check_bilinear_bound(d)
    print('完成。')
    return n_miss


if __name__ == '__main__':
    main()
