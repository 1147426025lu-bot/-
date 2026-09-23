# -*- coding: utf-8 -*-
# 本程序及代码是在人工智能工具辅助下完成的。
# 工具名称：DeepSeek；版本/型号：DeepSeek V4.1 Flash（API 模型名 deepseek-flash）；
# 开发机构：杭州深度求索人工智能基础技术研究有限公司；
# 版本发布日期：2026 年 9 月 10 日。

"""
问题二：异构无人机多点多架次运输调度（不考虑通信保障）
  - 硬约束：货箱时限（医疗期望送达 / 首批截止）、载质量与体积、返航安全余量、
            无人机与共享电池数量、两阶段充电周转
  - 决策：货箱组批、服务区访问顺序、机型、执行无人机、共享电池、各架次开始时刻
  - 方法：构造式基线（紧迫货箱单独架次 + FFD 组批 + 贪心合并 + 紧迫度派工）
          → ALNS 自适应大邻域搜索（解 = 货箱→架次划分 + 派工优先级排列）
  - 目标：ε-约束扫描架次数 K，得 (架次, makespan, 加权时延, 能耗) 非支配前沿

时限口径：医疗物资的 expect 与首批保障的 deadline 是**硬约束**，任何 hard_viol>0
的解直接判为不可行（目标值取 +∞）并逐出解空间；非医疗非首批的 expect 只作软目标。
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import os
import json
import math
import random
import time
from core import (T_DEC, E_DEC, load_data, ll_to_xy, segment_time, segment_energy,
                  segment_geometry, equivalent_range, charge_time, max_safe_payload)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'results')
os.makedirs(OUT, exist_ok=True)

# 基线回归护栏：schedule(d, trips) 在 order=None 下必须逐位复现阶段 1.5 之后的结果。
# 护栏在 main() 里断言；任何一处改动若破坏向后兼容，这里先炸，不会静默改数。
BASE_REF = dict(n_trips=25, total_E=84.77853067400346,
                makespan=14367.235622004755, hard_viol=0,
                tardiness=743943.2211624251)
BASE_TOL = 1e-6

# 推荐方案的落盘缓存。solve_recommended 一次约需 10 分钟（6 个 K × 4000 迭代），
# 而 q3.py / q4.py / figures.py 都必须在**同一份** Q2 解上继续做（否则 Q2→Q3→Q4
# 链路各跑各的，正文数字对不上）。故权威解在 main() 里实跑一次后写盘，
# 三个下游脚本用 recommended() 毫秒级读回，保证「同一份解」而非「同一个算法」。
REC_CACHE = os.path.join(OUT, 'q2_recommended.json')


# ---------------------------------------------------------------------------
# 几何预计算
# ---------------------------------------------------------------------------
def precompute_geometry(d):
    """预计算所有有序节点对之间的航段几何量。节点集合 {O01} ∪ S1..S15。"""
    nodes = {'O01': dict(lon=d.O01['lon'], lat=d.O01['lat'], alt=d.O01['alt'])}
    for s in d.services:
        nodes[s['id']] = dict(lon=s['lon'], lat=s['lat'], alt=s['alt'] + 30.0)
    geo = {}
    for u, pu in nodes.items():
        for v, pv in nodes.items():
            if u == v:
                continue
            geo[(u, v)] = segment_geometry(d.dem, pu['lon'], pu['lat'], pu['alt'],
                                           pv['lon'], pv['lat'], pv['alt'])
    return nodes, geo


# ---------------------------------------------------------------------------
# 单架次执行计划（带备忘）
# ---------------------------------------------------------------------------
# 备忘键 = (机型, 路线, 每区箱数, 每区质量)。
# 依据：trip_plan 的能耗与时长只通过"每区箱数"（决定交接时长）与"每区总质量"
# （决定逐段剩余载荷）依赖货箱，与箱 id、单箱体积均无关；交付时刻在同一个服务区
# 内对全部货箱相同，故只需缓存"每区交付偏移"。体积单独在 type_feasible 里判。
# 键漏项会静默给出错值，是本设计最阴险的失败模式，故 q2_exact.py 配 1000 例
# 随机对拍（缓存路径 vs 无缓存直算）作为硬性门禁。
_TRIP_CACHE = {}
_TRIP_CACHE_MAX = 2_000_000


def trip_core_sig(d, t, route, cnt, mss):
    """架次能耗/时长/每区交付偏移。

    cnt/mss 为与 route 对齐的「每区箱数」「每区总质量」。这是全部物理量的真正
    自变量：能耗与时长只通过它们依赖货箱。ALNS 的插入评估每次都要试几十个候选，
    直接以这两个元组为键可以免去为每个候选重建 boxes_at 字典的开销。
    """
    key = (t['id'], tuple(route), tuple(cnt), tuple(mss))
    hit = _TRIP_CACHE.get(key)
    if hit is not None:
        return hit
    n_tot = sum(cnt)
    m_tot = sum(mss)
    t_cur = t['prep'] + t['load_per_box'] * n_tot
    prev = 'O01'
    payload = m_tot
    E = 0.0
    off = {}
    payloads = []
    for k, sid in enumerate(route):
        g = d.geo[(prev, sid)]
        E += segment_energy(t, g, payload)
        t_cur += segment_time(t, g)
        t_cur += t['hand_base'] + t['hand_per_box'] * cnt[k]
        off[sid] = t_cur
        payload -= mss[k]
        payloads.append((prev, sid, payload))
        prev = sid
    g = d.geo[(prev, 'O01')]
    E += segment_energy(t, g, 0.0)
    t_cur += segment_time(t, g)
    payloads.append((prev, 'O01', 0.0))
    usable = (1 - t['rho']) * t['E_use']
    out = dict(E=E, duration=t_cur, off=off, total_mass=m_tot,
               feasible=(E <= usable + 1e-9), payloads=payloads)
    if len(_TRIP_CACHE) < _TRIP_CACHE_MAX:
        _TRIP_CACHE[key] = out
    return out


def trip_core(d, t, route, boxes_at):
    """架次能耗/时长/每区交付偏移，仅依赖 (机型, 路线, 每区箱数, 每区质量)。"""
    cnt = [len(boxes_at[s]) for s in route]
    mss = [sum(b['mass'] for b in boxes_at[s]) for s in route]
    return trip_core_sig(d, t, route, cnt, mss)


def trip_plan(d, t, route, boxes_at):
    """
    计算一个运输架次的完整计划。
    route: 访问服务区顺序 list[str]
    boxes_at: dict[sid] -> list[box dict]（该架次在该服务区交付的货箱）
    返回 dict:
      E          总能耗(kWh)
      duration   架次总时长(s)：准备+装载+飞行+交接（不含返程后周转）
      deliver    dict[box_id] -> 相对架次开始时刻的交付完成时间(s)
      payloads   各航段载荷列表
      feasible   返航安全余量是否满足
    """
    c = trip_core(d, t, route, boxes_at)
    deliver = {}
    for sid in route:
        off = c['off'][sid]
        for b in boxes_at[sid]:
            deliver[b['id']] = off
    return dict(E=c['E'], duration=c['duration'], deliver=deliver,
                payloads=c['payloads'], feasible=c['feasible'],
                total_mass=c['total_mass'],
                total_volume=sum(b['volume'] for v in boxes_at.values() for b in v))


def trip_plan_direct(d, t, route, boxes_at):
    """无备忘的直算版本，只用于 q2_exact.py 的 1000 例随机对拍，不参与求解。"""
    total_boxes = sum(len(v) for v in boxes_at.values())
    total_mass = sum(b['mass'] for v in boxes_at.values() for b in v)
    t_cur = t['prep'] + t['load_per_box'] * total_boxes
    prev = 'O01'
    payload = total_mass
    E = 0.0
    deliver = {}
    payloads = []
    for sid in route:
        g = d.geo[(prev, sid)]
        E += segment_energy(t, g, payload)
        t_cur += segment_time(t, g)
        n = len(boxes_at[sid])
        t_cur += t['hand_base'] + t['hand_per_box'] * n
        for b in boxes_at[sid]:
            deliver[b['id']] = t_cur
        payload -= sum(b['mass'] for b in boxes_at[sid])
        payloads.append((prev, sid, payload))
        prev = sid
    g = d.geo[(prev, 'O01')]
    E += segment_energy(t, g, 0.0)
    t_cur += segment_time(t, g)
    payloads.append((prev, 'O01', 0.0))
    usable = (1 - t['rho']) * t['E_use']
    return dict(E=E, duration=t_cur, deliver=deliver, payloads=payloads,
                feasible=(E <= usable + 1e-9), total_mass=total_mass,
                total_volume=sum(b['volume'] for v in boxes_at.values() for b in v))


def type_feasible_sig(d, t, route, cnt, mss, v_tot):
    """机型 t 能否执行该架次（质量/体积/能量），以签名形式给入。"""
    if sum(mss) > t['Q'] + 1e-9 or v_tot > t['volume'] + 1e-9:
        return False, None
    plan = trip_core_sig(d, t, route, cnt, mss)
    return plan['feasible'], plan


def type_feasible(d, t, route, boxes_at):
    """机型 t 能否执行该架次（质量/体积/能量）。"""
    total_mass = sum(b['mass'] for v in boxes_at.values() for b in v)
    total_vol = sum(b['volume'] for v in boxes_at.values() for b in v)
    if total_mass > t['Q'] + 1e-9 or total_vol > t['volume'] + 1e-9:
        return False, None
    plan = trip_plan(d, t, route, boxes_at)
    return plan['feasible'], plan


def best_type_plan_sig(d, route, cnt, mss, v_tot):
    """签名形式：该架次能耗最低的可行机型；全不可行返回 (None, None)。"""
    best_g, best_plan = None, None
    for g in ('A', 'B', 'C'):
        fe, plan = type_feasible_sig(d, d.transport_types[g], route, cnt, mss, v_tot)
        if fe and (best_plan is None or plan['E'] < best_plan['E']):
            best_g, best_plan = g, plan
    return best_g, best_plan


def best_type_plan(d, route, boxes_at):
    """该架次能耗最低的可行机型；全不可行返回 (None, None)。"""
    cnt = [len(boxes_at[s]) for s in route]
    mss = [sum(b['mass'] for b in boxes_at[s]) for s in route]
    v_tot = sum(b['volume'] for v in boxes_at.values() for b in v)
    return best_type_plan_sig(d, route, cnt, mss, v_tot)


# ---------------------------------------------------------------------------
# 架次构造：FFD 单服务区组批 + 贪心合并为多服务区架次（基线）
# ---------------------------------------------------------------------------
def _pack_ffd(boxes, qmax, vmax):
    """FFD 装箱，返回 list of (idxs, mass, vol)。"""
    idx = sorted(range(len(boxes)), key=lambda i: -boxes[i]['mass'])
    batches = []
    for i in idx:
        b = boxes[i]
        placed = False
        for bt in batches:
            if bt[1] + b['mass'] <= qmax + 1e-9 and bt[2] + b['volume'] <= vmax + 1e-9:
                bt[1] += b['mass']; bt[2] += b['volume']; bt[0].append(i)
                placed = True; break
        if not placed:
            batches.append([[i], b['mass'], b['volume']])
    return batches


def _route_nn(d, stops):
    """从 O01 出发的最近邻路由，返回访问顺序 list[sid]。"""
    remaining = list(stops)
    route = []
    cur = 'O01'

    def dist_xy(a, b):
        pa = d.O01 if a == 'O01' else d.si[a]
        pb = d.O01 if b == 'O01' else d.si[b]
        xa, ya = ll_to_xy(pa['lon'], pa['lat']); xb, yb = ll_to_xy(pb['lon'], pb['lat'])
        return np.hypot(xb - xa, yb - ya)
    while remaining:
        nxt = min(remaining, key=lambda s: dist_xy(cur, s))
        route.append(nxt); remaining.remove(nxt); cur = nxt
    return route


def _is_urgent(b):
    return (b['category'] == '医疗物资') or b['first_batch']


def construct_trips(d, pack_type='C'):
    """返回 list of trip dict。紧急货箱单独成轻载架次；非紧急按 pack_type 容量 FFD+合并。"""
    tC = d.transport_types['C']
    tP = d.transport_types[pack_type]
    trips = []
    # 1) 紧急货箱：每服务区单独轻载架次（医疗+首批，通常 ~17kg）
    for sid in d.S:
        urgent = [b for b in d.boxes_by_service[sid] if _is_urgent(b)]
        if urgent:
            mass = sum(b['mass'] for b in urgent)
            vol = sum(b['volume'] for b in urgent)
            p0 = trip_plan(d, tC, [sid], {sid: urgent})
            trips.append(dict(route=[sid], boxes_at={sid: urgent}, mass=mass, vol=vol,
                              _dur=p0['duration'], urgent=True))
    # 2) 非紧急货箱：FFD（按 pack_type 能量限制）+ 贪心合并
    rest = []
    for sid in d.S:
        boxes = [b for b in d.boxes_by_service[sid] if not _is_urgent(b)]
        if not boxes:
            continue
        qmax = max_safe_payload(d, tP, d.si[sid])
        for idxs, mass, vol in _pack_ffd(boxes, min(tP['Q'], qmax), tP['volume']):
            bs = [boxes[i] for i in idxs]
            p0 = trip_plan(d, tP, [sid], {sid: bs})
            rest.append(dict(route=[sid], boxes_at={sid: bs}, mass=mass, vol=vol,
                             _dur=p0['duration'], urgent=False))
    # 合并非紧急架次（省架次）
    merged = True
    while merged:
        merged = False
        best = None; best_saving = -1e9; bi = bj = -1
        for i in range(len(rest)):
            for j in range(i + 1, len(rest)):
                ti, tj = rest[i], rest[j]
                new_mass = ti['mass'] + tj['mass']
                new_vol = ti['vol'] + tj['vol']
                if new_mass > tP['Q'] + 1e-9 or new_vol > tP['volume'] + 1e-9:
                    continue
                stops = list(set(ti['route']) | set(tj['route']))
                route = _route_nn(d, stops)
                boxes_at = {}
                for k, v in ti['boxes_at'].items():
                    boxes_at[k] = list(v)
                for k, v in tj['boxes_at'].items():
                    boxes_at.setdefault(k, []).extend(v)
                plan = trip_plan(d, tP, route, boxes_at)
                if not plan['feasible']:
                    continue
                saving = plan['duration'] - ti['_dur'] - tj['_dur']
                if -saving > best_saving:
                    best_saving = -saving; best = (route, boxes_at, new_mass, new_vol, plan)
                    bi, bj = i, j
        if best is not None:
            route, boxes_at, new_mass, new_vol, plan = best
            rest[bi] = dict(route=route, boxes_at=boxes_at, mass=new_mass, vol=new_vol,
                            _dur=plan['duration'], urgent=False)
            rest.pop(bj)
            merged = True
    trips.extend(rest)
    return trips


# ---------------------------------------------------------------------------
# 机型分配 + 调度（离散事件，共享电池两阶段充电）
# ---------------------------------------------------------------------------
def trip_sort_key(trip):
    """紧迫度排序键：硬时限（医疗期望/首批截止）优先，其次最小期望送达时间。"""
    hard = []
    soft = []
    for sid, bs in trip['boxes_at'].items():
        for b in bs:
            if b['category'] == '医疗物资':
                hard.append(b['expect'])
            elif b['first_batch']:
                hard.append(b['deadline'])
            if b['expect'] is not None:
                soft.append(b['expect'])
    return (min(hard) if hard else 1e9, min(soft) if soft else 1e9)


def schedule(d, trips, order=None, dispatch='earliest'):
    """
    对架次序列做机型分配与时间调度。

    order:    派工优先级序列（trips 下标的一个排列）。None = 按紧迫度排序，
              与历史行为逐位一致（main() 里有回归断言把关）。
    dispatch: 'earliest' = 选使开始时刻最早的机型（历史行为）；
              'ect'      = 选使完成时刻最早的机型。
    返回 assignment: list of dict（每架次：trip_idx/type/uav/battery/start/plan/
                                  deliver_abs/route/boxes_at）
    """
    types = ['A', 'B', 'C']
    # 资源状态：实体无人机（U01..U08）
    uavs = {g: [dict(id=u['id'], free_at=0.0) for u in d.uavs if u['type'] == g]
            for g in types}
    # 电池：初始时刻全部满电，ready_at=0
    batteries = {g: [dict(id=f'{g}B{k}', ready_at=0.0) for k in range(d.batteries[g][0])]
                 for g in types}
    if order is None:
        order = sorted(range(len(trips)), key=lambda i: trip_sort_key(trips[i]))
    else:
        assert sorted(order) == list(range(len(trips))), 'order 必须是 trips 下标的一个排列'

    assignment = []
    for idx in order:
        trip = trips[idx]
        # 选机型：默认使开始时刻最早（uav/battery 最早可用），并列取能耗低者
        best = None
        for g in types:
            feas, plan = type_feasible(d, d.transport_types[g], trip['route'], trip['boxes_at'])
            if not feas:
                continue
            u = min(uavs[g], key=lambda x: x['free_at'])
            bat = min(batteries[g], key=lambda x: x['ready_at'])
            start = max(u['free_at'], bat['ready_at'])
            if dispatch == 'earliest':
                better = (best is None or start < best['start']
                          or (abs(start - best['start']) < 1e-6 and plan['E'] < best['plan']['E']))
            else:
                ect = start + plan['duration']
                better = (best is None or ect < best['ect']
                          or (abs(ect - best['ect']) < 1e-6 and plan['E'] < best['plan']['E']))
            if better:
                best = dict(g=g, plan=plan, start=start, u=u, bat=bat, ect=start + plan['duration'])
        if best is None:
            # 资源终会空闲，唯一失败模式是该架次对 A/B/C 全不可行
            raise RuntimeError('架次不可行')
        g = best['g']; plan = best['plan']; start = best['start']
        u = best['u']; bat = best['bat']
        # 占用资源
        u['free_at'] = start + plan['duration']
        soc_end = 1.0 - plan['E'] / d.transport_types[g]['E_use']
        t_chg = charge_time(soc_end, d.batteries[g][1])
        bat['ready_at'] = start + plan['duration'] + t_chg
        deliver_abs = {bid: start + off for bid, off in plan['deliver'].items()}
        assignment.append(dict(trip_idx=idx, type=g, uav=u['id'], battery=bat['id'],
                               start=start, E=plan['E'], duration=plan['duration'],
                               plan=plan, deliver_abs=deliver_abs, route=trip['route'],
                               boxes_at=trip['boxes_at']))
    return assignment


def resource_usage(d, assignment):
    """
    把一份调度还原成「资源 → 占用区间」表，供外部（q3.py 的错峰搜索、verify.py
    的独立复核）在**不改动机型与指派**的前提下判断资源是否仍然不冲突。

    口径与 schedule() 内部完全一致，不另写一份：无人机占用 [start, start+duration]；
    共享电池占用延续到充电完成 [start, start+duration+charge_time(soc_end, T)]。
    返回 {'uav:U03': [(t0,t1,架次序号), ...], 'bat:AB2': [...]}，区间按开始时刻升序。
    """
    use = {}
    for k, a in enumerate(assignment):
        g = a['type']
        t1 = a['start'] + a['duration']
        soc_end = 1.0 - a['E'] / d.transport_types[g]['E_use']
        t_bat = t1 + charge_time(soc_end, d.batteries[g][1])
        use.setdefault('uav:' + a['uav'], []).append((a['start'], t1, k))
        use.setdefault('bat:' + a['battery'], []).append((a['start'], t_bat, k))
    for v in use.values():
        v.sort()
    return use


def resource_conflicts(d, assignment):
    """返回资源冲突的 (资源, 前架次, 后架次) 列表；空列表表示无冲突。"""
    bad = []
    for key, ivs in resource_usage(d, assignment).items():
        for x, y in zip(ivs, ivs[1:]):
            if y[0] < x[1] - 1e-6:
                bad.append((key, x[2], y[2]))
    return bad
def evaluate(d, assignment):
    makespan = max(a['start'] + a['duration'] for a in assignment)
    total_E = sum(a['E'] for a in assignment)
    n_trips = len(assignment)
    # 及时性：硬约束违反 + 软时延（带应急优先系数）
    hard_viol = 0
    tardiness = 0.0
    for a in assignment:
        for sid, bs in a['boxes_at'].items():
            for b in bs:
                t_del = a['deliver_abs'][b['id']]
                if b['category'] == '医疗物资':
                    if t_del > b['expect'] + 1e-6:
                        hard_viol += 1
                if b['first_batch']:
                    if t_del > b['deadline'] + 1e-6:
                        hard_viol += 1
                # 软时延（相对期望送达时间）
                if b['expect'] is not None:
                    tardiness += b['priority'] * max(0.0, t_del - b['expect'])
    return dict(makespan=makespan, total_E=total_E, n_trips=n_trips,
                hard_viol=hard_viol, tardiness=tardiness)


# ===========================================================================
# ALNS 自适应大邻域搜索
# ===========================================================================
ALNS_SEED = 20260923
K_TARGETS = [20, 22, 25, 30, 35, 40]

# 目标权重（时效优先）：加权时延为主，完成时间为辅。归一化基准取基线值，
# 使两项量纲无关、权重可直接解释。架次数不做平权加权，改用 ε-约束扫描。
W_TARD = 1.0
W_MS = 0.30
K_PEN = 0.50          # 超出 ε 上界的每架次罚（归一化后单位）
GROUP_EVAL_TOPK = 10  # 成组修复里真解码评估的候选上限（见 repair_group）


# --- 解的内部表示：trips = [{'route': [...], 'boxes': [...]}], order = [trips 下标]
def _group(boxes):
    by = {}
    for b in boxes:
        by.setdefault(b['service'], []).append(b)
    return by


def to_sched_trips(d, trips):
    """内部表示 -> schedule() 需要的 {route, boxes_at} 形式。"""
    return [dict(route=list(t['route']), boxes_at=_group(t['boxes']),
                 boxes=list(t['boxes'])) for t in trips]


def _default_order(d, trips):
    st = to_sched_trips(d, trips)
    return sorted(range(len(st)), key=lambda i: trip_sort_key(st[i]))


def _feasible_trip(d, t):
    """架次对至少一种机型可行。"""
    g, plan = best_type_plan(d, t['route'], _group(t['boxes']))
    return plan is not None


def objective(d, trips, order, k_target=None, ref_tard=1.0, ref_ms=1.0, mode='time',
              ref_E=1.0):
    """目标值。硬时限违反 => +inf（从解空间剔除）。

    mode='time'   时效优先：加权时延 + 0.3×makespan（归一化），架次数用 ε-约束罚。
    mode='energy' 字典序 (架次数, 能耗)，只用于与 Layer A 精确解同口径对比。

    mode='energy' 里能耗项按 ref_E 归一化，使该项与 SA 的温度同量级：若不做归一化，
    f≈10^4 而 T0=0.05，exp(-Δ/T) 恒下溢，接受准则退化为纯爬山，对精确解的对比不公。
    架次项仍取 1000，保证「架次数严格优先、同架次内才比能耗」的字典序语义（差值远大于 T）。
    """
    st = to_sched_trips(d, trips)
    try:
        asg = schedule(d, st, order=order)
    except RuntimeError:
        return float('inf'), None
    m = evaluate(d, asg)
    if m['hard_viol'] > 0:
        return float('inf'), m
    if mode == 'energy':
        return 1000.0 * m['n_trips'] + m['total_E'] / max(ref_E, 1e-9), m
    f = W_TARD * m['tardiness'] / ref_tard + W_MS * m['makespan'] / ref_ms
    if k_target is not None and m['n_trips'] > k_target:
        f += K_PEN * (m['n_trips'] - k_target)
    return f, m


# --- 破坏算子（7 个）-------------------------------------------------------
def _remove_boxes(trips, victims):
    """按 (架次下标, 货箱对象 id) 摘除货箱，返回 (新 trips, 被摘箱 list)。空架次直接删除。"""
    out, removed = [], []
    for i, t in enumerate(trips):
        keep = []
        for b in t['boxes']:
            if (i, id(b)) in victims:
                removed.append(b)
            else:
                keep.append(b)
        if keep:
            out.append(dict(route=[s for s in t['route']
                                   if any(x['service'] == s for x in keep)], boxes=keep))
    return out, removed


def _remove_by_pred(trips, pred):
    """按谓词摘除货箱，返回 (新 trips, 被摘箱 list)。"""
    out, removed = [], []
    for t in trips:
        keep = []
        for b in t['boxes']:
            if pred(b):
                removed.append(b)
            else:
                keep.append(b)
        if keep:
            out.append(dict(route=[s for s in t['route'] if any(x['service'] == s for x in keep)],
                            boxes=keep))
    return out, removed


def op_random_removal(d, trips, order, q, rng):
    allb = [(i, b) for i, t in enumerate(trips) for b in t['boxes']]
    rng.shuffle(allb)
    vic = set((i, id(b)) for i, b in allb[:q])
    return _remove_boxes(trips, vic)


def op_shaw_removal(d, trips, order, q, rng):
    """Shaw 相关移除：与种子箱在服务区/质量/期望时刻上最相关的 q 个。"""
    allb = [(i, b) for i, t in enumerate(trips) for b in t['boxes']]
    if not allb:
        return trips, []
    si, seed = allb[rng.randrange(len(allb))]
    def rel(ib):
        _, b = ib
        return (0.0 if b['service'] == seed['service'] else 1.0) \
            + abs(b['mass'] - seed['mass']) / 20.0 \
            + min(abs((b['expect'] or 0) - (seed['expect'] or 0)), 4000.0) / 4000.0
    allb.sort(key=rel)
    vic = set((i, id(b)) for i, b in allb[:q])
    return _remove_boxes(trips, vic)


def op_worst_trip_removal(d, trips, order, q, rng):
    """最差架次移除：反复拆掉"单位货箱代价最高"的架次，直到摘够 q 箱。"""
    trips = [dict(route=list(t['route']), boxes=list(t['boxes'])) for t in trips]
    removed = []
    while len(removed) < q and trips:
        worst, worst_cost = None, -1.0
        for i, t in enumerate(trips):
            g, plan = best_type_plan(d, t['route'], _group(t['boxes']))
            if plan is None:
                worst, worst_cost = i, float('inf')
                break
            cost = plan['E'] / max(1, len(t['boxes']))
            if cost > worst_cost:
                worst, worst_cost = i, cost
        if worst is None:
            break
        removed.extend(trips.pop(worst)['boxes'])
    return trips, removed


def op_service_removal(d, trips, order, q, rng):
    """整区移除：随机挑服务区，把该区全部货箱摘出。"""
    present = sorted({b['service'] for t in trips for b in t['boxes']})
    if not present:
        return trips, []
    rng.shuffle(present)
    vic_srv = set(present[:max(1, len(present) // 6)])
    return _remove_by_pred(trips, lambda b: b['service'] in vic_srv)


def op_nonurgent_removal(d, trips, order, q, rng):
    """仅非紧急移除：紧急（医疗/首批）货箱保持不动。"""
    non = [(i, b) for i, t in enumerate(trips) for b in t['boxes'] if not _is_urgent(b)]
    if not non:
        return trips, []
    rng.shuffle(non)
    vic = set((i, id(b)) for i, b in non[:q])
    return _remove_boxes(trips, vic)


def op_trip_destroy(d, trips, order, q, rng):
    """整架次拆解：随机挑一个多区架次整体拆开。"""
    multi = [i for i, t in enumerate(trips) if len(t['route']) > 1]
    if not multi:
        return op_random_removal(d, trips, order, q, rng)
    i = multi[rng.randrange(len(multi))]
    out = [dict(route=list(t['route']), boxes=list(t['boxes'])) for k, t in enumerate(trips) if k != i]
    return out, list(trips[i]['boxes'])


def op_order_perturb(d, trips, order, q, rng):
    """仅顺序扰动：货箱划分冻结，只重排派工优先级。"""
    new_order = list(order)
    if len(new_order) < 2:
        return [dict(route=list(t['route']), boxes=list(t['boxes'])) for t in trips], []
    a = rng.randrange(len(new_order))
    b = min(len(new_order), a + max(2, len(new_order) // 4))
    seg = new_order[a:b]
    rng.shuffle(seg)
    new_order[a:b] = seg
    return ([dict(route=list(t['route']), boxes=list(t['boxes'])) for t in trips], [], new_order)


# --- 修复算子（3 个）-------------------------------------------------------
def _insertion_candidates(d, trips, boxes):
    """把一组**同服务区**货箱整体插入各架次（或新开单区架次）的候选，按代理代价升序。

    传入单个货箱时须包成 [box]。**支持成组是必要的**：单个箱的增量插入看不到
    「两区各用一个 B 型」→「并成一架 C 型」这类**整组换型**协同收益。实测
    S013+S014 两区合并成一架 C 型能耗 5.0112 kWh，优于分开的两架；但只搬一个箱时
    增量 ΔE=+2.70，反而高于另起一架的 +1.73，故纯单箱贪心**永远不会合并**这两个区。
    成组插入才能评估整组迁移的真实代价。

    返回 [(cost, trip_idx 或 None, route, ΔE)]。
    代理代价 = 能耗增量 + 时长增量/1000（后者按 km 量级折算，避免量纲主导）。
    这里刻意不解码整个解（不跑 schedule），只做架次级局部评估；整解解码每轮仅一次。
    """
    if isinstance(boxes, dict):
        boxes = [boxes]
    sid = boxes[0]['service']
    n_add = len(boxes)
    m_add = sum(b['mass'] for b in boxes)
    v_add = sum(b['volume'] for b in boxes)
    Qmax = d.transport_types['C']['Q']
    Vmax = d.transport_types['C']['volume']
    cands = []
    for i, t in enumerate(trips):
        by = _group(t['boxes'])
        route0 = list(t['route'])
        cnt0 = [len(by[s]) for s in route0]
        mss0 = [sum(b['mass'] for b in by[s]) for s in route0]
        v0 = sum(b['volume'] for b in t['boxes'])
        # 容量预筛：先按最大机型（C）判，不通过就没有任何机型可行
        if sum(mss0) + m_add > Qmax + 1e-9 or v0 + v_add > Vmax + 1e-9:
            continue
        _, base_plan = best_type_plan_sig(d, route0, cnt0, mss0, v0)
        if base_plan is None:
            continue
        v_new = v0 + v_add
        if sid in route0:
            k = route0.index(sid)
            cnt = list(cnt0); cnt[k] += n_add
            mss = list(mss0); mss[k] += m_add
            variants = [(route0, cnt, mss)]
        else:
            variants = []
            for p in range(len(route0) + 1):
                variants.append((route0[:p] + [sid] + route0[p:],
                                 cnt0[:p] + [n_add] + cnt0[p:],
                                 mss0[:p] + [m_add] + mss0[p:]))
        for r, cnt, mss in variants:
            g, plan = best_type_plan_sig(d, r, cnt, mss, v_new)
            if plan is None:
                continue
            dE = plan['E'] - base_plan['E']
            dT = plan['duration'] - base_plan['duration']
            cands.append((dE + dT / 1000.0, i, r, dE))
    # 新开单区架次：该区整组自成一架，恒可行（各区满箱组批本就在基线解里出现）
    g, plan = best_type_plan_sig(d, [sid], [n_add], [m_add], v_add)
    if plan is None:
        raise RuntimeError('单区架次不可行，兜底动作失效')
    cands.append((plan['E'] + plan['duration'] / 1000.0, None, [sid], plan['E']))
    cands.sort(key=lambda x: x[0])
    return cands


def _best_insertion(d, trips, box):
    """返回 (trip_idx, route, cost, ΔE)。"""
    c = _insertion_candidates(d, trips, [box])
    return c[0][1], c[0][2], c[0][0], c[0][3]


def _sorted_by_urgency(boxes):
    return sorted(boxes, key=lambda b: (0 if _is_urgent(b) else 1,
                                        b['expect'] if b['expect'] is not None else 1e9,
                                        -b['mass']))


def repair_greedy(d, trips, removed, rng, ctx=None):
    trips = [dict(route=list(t['route']), boxes=list(t['boxes'])) for t in trips]
    for box in _sorted_by_urgency(removed):
        c = _best_insertion(d, trips, box)
        if c is None:
            return None
        i, route, _, _ = c
        if i is None:
            trips.append(dict(route=list(route), boxes=[box]))
        else:
            trips[i]['route'] = route
            trips[i]['boxes'].append(box)
    return trips


def repair_regret2(d, trips, removed, rng, ctx=None):
    """Regret-2：优先插入"次优代价 − 最优代价"最大的箱（犹豫最大的先定）。"""
    trips = [dict(route=list(t['route']), boxes=list(t['boxes'])) for t in trips]
    pending = list(removed)
    while pending:
        best_pick, best_box = None, None
        for box in pending:
            cands = _insertion_candidates(d, trips, [box])
            gap = (cands[1][0] - cands[0][0]) if len(cands) > 1 else 1e6
            if best_pick is None or gap > best_pick:
                best_pick, best_box = gap, box
        cands = _insertion_candidates(d, trips, [best_box])
        _, i, route, _ = cands[0]
        if i is None:
            trips.append(dict(route=list(route), boxes=[best_box]))
        else:
            trips[i]['route'] = route
            trips[i]['boxes'].append(best_box)
        pending.remove(best_box)
    return trips


def repair_newtrip(d, trips, removed, rng, ctx=None):
    """偏向新开架次：插入代价偏高时宁可另起一架，用于跳出局部最优。"""
    trips = [dict(route=list(t['route']), boxes=list(t['boxes'])) for t in trips]
    for box in _sorted_by_urgency(removed):
        i, route, cost, dE = _best_insertion(d, trips, box)
        # 阈值 0.35 kWh：只有插入代价明显偏高时才新开架次
        if i is None or dE > 0.35:
            trips.append(dict(route=[box['service']], boxes=[box]))
        else:
            trips[i]['route'] = route
            trips[i]['boxes'].append(box)
    return trips


def _apply_insert(trips, i, route, grp):
    """把整组 grp 落到第 i 个架次（i=None 则新开一架），返回新的 trips。"""
    out = [dict(route=list(t['route']), boxes=list(t['boxes'])) for t in trips]
    if i is None:
        out.append(dict(route=list(route), boxes=list(grp)))
    else:
        out[i]['route'] = list(route)
        out[i]['boxes'] = out[i]['boxes'] + list(grp)
    return out


def repair_group(d, trips, removed, rng, ctx=None):
    """成组修复：把被摘箱按**服务区**打包，整组一并并入某个现有架次，装不下才另起一架。

    与另三个修复算子的关键区别：**用真目标（解码后）而不是代理代价来选候选**。
    代理代价由 best_type_plan 给出，它假设每个架次都能自由选用最省机型，**完全看不到
    机队资源竞争**。实测 S013/S014/S015 三箱例：B 型无人机只有 2 架，三个单区架次无法
    同时用 B，S015 被迫改用 C（4.0825 vs 2.3114 kWh）；而把 S013+S014 并成一架 C 型
    反而腾出一架 B 给 S015，总能耗 7.3226 < 8.0345。这种**机队瓶颈型协同**在代理代价
    里表现为"合并更贵"，故必须真解码才看得见。ctx 为 None 时退化为代理代价（保底）。
    """
    by_svc = {}
    for b in removed:
        by_svc.setdefault(b['service'], []).append(b)
    for sid in sorted(by_svc, key=lambda s: -len(by_svc[s])):
        grp = _sorted_by_urgency(by_svc[sid])
        cands = _insertion_candidates(d, trips, grp)
        if ctx is None or len(cands) == 1:
            _, i, route, _ = cands[0]
            trips = _apply_insert(trips, i, route, grp)
            continue
        best, best_f = None, float('inf')
        for _, i, route, _ in cands[:GROUP_EVAL_TOPK]:
            cand = _apply_insert(trips, i, route, grp)
            f, _m = ctx['evaluate'](cand)
            if f < best_f:
                best_f, best = f, (i, route)
        if best is None:                      # 全部候选不可行，退回代理最优
            _, i, route, _ = cands[0]
            best = (i, route)
        trips = _apply_insert(trips, best[0], best[1], grp)
    return trips
    return trips


DESTROY_OPS = [op_random_removal, op_shaw_removal, op_worst_trip_removal,
               op_service_removal, op_nonurgent_removal, op_trip_destroy]
REPAIR_OPS = [repair_greedy, repair_regret2, repair_newtrip, repair_group]


# --- 主循环 ----------------------------------------------------------------
def alns(d, init_trips, k_target=None, n_iter=3000, seed=ALNS_SEED,
         use_order=True, freeze_boxes=False, ref_tard=1.0, ref_ms=1.0,
         trace=None, mode='time', ref_E=1.0, t0=None):
    """
    自适应大邻域搜索。
      use_order    False = 派工顺序恒按紧迫度（消融用）
      freeze_boxes True  = 只搜顺序（消融"仅顺序"用）
      t0           初始温度；None = 用 T0（时效口径的标定值）。
                   mode='energy' 时目标量级不同，必须由调用方给出匹配的 t0。
    返回 (best_trips, best_order, best_metrics, best_f)
    """
    rng = random.Random(seed)
    trips = [dict(route=list(t['route']), boxes=list(t['boxes'])) for t in init_trips]
    order = _default_order(d, trips)
    f_cur, m_cur = objective(d, trips, order, k_target, ref_tard, ref_ms, mode, ref_E)
    best_trips = [dict(route=list(t['route']), boxes=list(t['boxes'])) for t in trips]
    best_order, best_m, best_f = list(order), m_cur, f_cur

    nd, nr = len(DESTROY_OPS), len(REPAIR_OPS)
    w_d = [1.0] * nd
    w_r = [1.0] * nr
    s_d = [0.0] * nd
    s_r = [0.0] * nr
    c_d = [0] * nd
    c_r = [0] * nr
    SEG = 100
    RHO = 0.25
    T0 = 0.05 if t0 is None else float(t0)
    T1 = 1e-4
    n_boxes = sum(len(t['boxes']) for t in trips)

    def norm(w):
        s = sum(w)
        return [x / s for x in w]

    for it in range(n_iter):
        T = T0 * (T1 / T0) ** (it / max(1, n_iter - 1))
        pd, pr = norm(w_d), norm(w_r)
        ri = _roulette(pr, rng)
        # freeze_boxes（消融"仅顺序"）只跑顺序扰动，不参与破坏算子权重更新，
        # 故 di 置 None；直接复用 w_d 会越界。
        if freeze_boxes:
            op, di = op_order_perturb, None
        else:
            di = _roulette(pd, rng)
            op = DESTROY_OPS[di]
        q = rng.randint(2, max(2, min(12, n_boxes // 6)))
        res = op(d, trips, order, q, rng)
        if len(res) == 3:
            new_trips, removed, new_order_hint = res
        else:
            new_trips, removed = res
            new_order_hint = None
        if freeze_boxes:
            new_trips = [dict(route=list(t['route']), boxes=list(t['boxes'])) for t in trips]
        if not new_trips and not removed:
            continue
        try:
            if removed:
                # 只有成组修复需要真解码评估（代理代价看不到机队竞争，见 repair_group）；
                # 其余三个算子沿用代理代价，保持单轮解码一次的廉价性。
                ctx = None
                if REPAIR_OPS[ri] is repair_group:
                    ctx = dict(evaluate=lambda cand: objective(
                        d, cand, _remap_order(d, trips, order, cand),
                        k_target, ref_tard, ref_ms, mode, ref_E))
                new_trips = REPAIR_OPS[ri](d, new_trips, removed, rng, ctx)
        except RuntimeError:
            # 兜底动作失效（理论上不会发生）：丢弃该解，权重记 0 分
            new_trips = None
        if new_trips is None or not new_trips:
            if di is not None:
                w_d[di] = max(0.05, w_d[di] * (1 - RHO)); c_d[di] += 1
            continue
        # 新解的顺序：保留幸存架次的相对顺序，新架次按紧迫度插到末尾
        if new_order_hint is not None and len(new_order_hint) == len(new_trips):
            new_order = new_order_hint
        else:
            new_order = _remap_order(d, trips, order, new_trips)
        if not use_order:
            new_order = _default_order(d, new_trips)
        f_new, m_new = objective(d, new_trips, new_order, k_target, ref_tard, ref_ms,
                                 mode, ref_E)
        accept = False
        if math.isfinite(f_new):
            if f_new < f_cur:
                accept = True
            elif f_cur == float('inf'):
                accept = True
            elif T > 0 and rng.random() < math.exp(-(f_new - f_cur) / max(T, 1e-12)):
                accept = True
        if accept:
            improved_best = f_new < best_f
            trips, order, f_cur, m_cur = new_trips, new_order, f_new, m_new
            if improved_best:
                best_trips = [dict(route=list(t['route']), boxes=list(t['boxes'])) for t in trips]
                best_order, best_m, best_f = list(order), m_new, f_new
            sigma = 3.0 if improved_best else 1.0
        else:
            sigma = 0.0
        if di is not None:
            w_d[di] = (1 - RHO) * w_d[di] + RHO * sigma
            s_d[di] += sigma
            c_d[di] += 1
        w_r[ri] = (1 - RHO) * w_r[ri] + RHO * sigma
        s_r[ri] += sigma
        c_r[ri] += 1
        if (it + 1) % SEG == 0:
            for k in range(nd):
                if c_d[k]:
                    w_d[k] = max(0.05, 0.7 * w_d[k] + 0.3 * (s_d[k] / c_d[k]))
                s_d[k] = 0.0; c_d[k] = 0
            for k in range(nr):
                if c_r[k]:
                    w_r[k] = max(0.05, 0.7 * w_r[k] + 0.3 * (s_r[k] / c_r[k]))
                s_r[k] = 0.0; c_r[k] = 0
        if trace is not None and (it % max(1, n_iter // 60) == 0 or it == n_iter - 1):
            trace.append((it + 1, best_f if math.isfinite(best_f) else float('nan'),
                          best_m['makespan'] if best_m else float('nan'),
                          best_m['tardiness'] if best_m else float('nan'),
                          best_m['total_E'] if best_m else float('nan'),
                          best_m['n_trips'] if best_m else 0))
    return best_trips, best_order, best_m, best_f, (norm(w_d), norm(w_r))


def _roulette(p, rng):
    x = rng.random()
    acc = 0.0
    for i, pi in enumerate(p):
        acc += pi
        if x <= acc:
            return i
    return len(p) - 1


def _remap_order(d, old_trips, old_order, new_trips):
    """结构变化后重建派工顺序：幸存架次保持原相对顺序，新架次按紧迫度排到末尾。"""
    key_new = {}
    for i, t in enumerate(new_trips):
        key_new[(tuple(t['route']), tuple(sorted(b['id'] for b in t['boxes'])))] = i
    seq, used = [], set()
    for oi in old_order:
        t = old_trips[oi]
        k = (tuple(t['route']), tuple(sorted(b['id'] for b in t['boxes'])))
        ni = key_new.get(k)
        if ni is not None and ni not in used:
            seq.append(ni); used.add(ni)
    rest = [i for i in range(len(new_trips)) if i not in used]
    st = to_sched_trips(d, new_trips)
    rest.sort(key=lambda i: trip_sort_key(st[i]))
    return seq + rest


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def solve_recommended(d, n_iter=4000, verbose=False):
    """
    求解推荐方案（时效优先：硬时限零违反为不可谈判前提，其次加权时延与完成时间）。
    返回 (trips, assignment, metrics, info)。trips 为 schedule() 可直接消费的形式。
    """
    base_trips = construct_trips(d, pack_type='C')
    base_asg = schedule(d, base_trips)
    base_m = evaluate(d, base_asg)
    assert base_m['hard_viol'] == 0, '基线解违反硬时限，不应发生'
    ref_tard = max(base_m['tardiness'], 1.0)
    ref_ms = max(base_m['makespan'], 1.0)

    init = [dict(route=list(t['route']), boxes=[b for v in t['boxes_at'].values() for b in v])
            for t in base_trips]
    runs = {}
    for K in K_TARGETS:
        t0 = time.time()
        bt, bo, bm, bf, wts = alns(d, init, k_target=K, n_iter=n_iter,
                                  seed=ALNS_SEED + K, ref_tard=ref_tard, ref_ms=ref_ms)
        runs[K] = dict(trips=bt, order=bo, metrics=bm, f=bf,
                       secs=time.time() - t0, weights=wts)
        if verbose:
            print(f'  K<={K:2d}: 实际架次={bm["n_trips"]:2d} E={bm["total_E"]:7.3f} '
                  f'makespan={bm["makespan"]:9.1f} 时延={bm["tardiness"]:10.1f} '
                  f'硬违反={bm["hard_viol"]} ({runs[K]["secs"]:.1f}s)')
    # 推荐：时效优先口径下的最优可行解
    feas = {K: r for K, r in runs.items() if r['metrics'] and r['metrics']['hard_viol'] == 0}
    K_best = min(feas, key=lambda K: (feas[K]['f'], runs[K]['metrics']['n_trips']))
    rec = runs[K_best]
    st = to_sched_trips(d, rec['trips'])
    asg = schedule(d, st, order=rec['order'])
    m = evaluate(d, asg)
    assert m['hard_viol'] == 0, '推荐方案必须零硬约束违反'
    save_recommended(d, st, rec['order'], m)
    return st, asg, m, dict(runs=runs, K=K_best, base=base_m,
                            ref_tard=ref_tard, ref_ms=ref_ms, init=init)


# ---------------------------------------------------------------------------
# 推荐方案的落盘 / 读回（Q2 → Q3 → Q4 链路一致性）
# ---------------------------------------------------------------------------
def _box_index(d):
    return {b['id']: b for bs in d.boxes_by_service.values() for b in bs}


def save_recommended(d, st, order, m, path=None):
    """
    把推荐方案写盘。存「路线 + 各区货箱 id + 派工顺序」即可完整还原：
    schedule() 对同一 (trips, order) 是确定性函数，故读回后重解必得同一结果。
    """
    path = REC_CACHE if path is None else path

    def _num(v):
        # json 不认 numpy 标量；能转 float 的一律转，转不动的原样交给 json 报错
        try:
            return float(v)
        except (TypeError, ValueError):
            return v

    payload = dict(
        order=[int(i) for i in order],
        trips=[dict(route=list(t['route']),
                    boxes_at={s: [b['id'] for b in bs]
                              for s, bs in t['boxes_at'].items()})
               for t in st],
        metrics={k: _num(v) for k, v in m.items()})
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    return path


def load_recommended(d, path=None):
    """读回落盘的推荐方案，返回 (trips, order, metrics)。"""
    path = REC_CACHE if path is None else path
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'未找到 Q2 推荐方案缓存 {path}；请先运行 python code/q2.py')
    with open(path, 'r', encoding='utf-8') as f:
        payload = json.load(f)
    idx = _box_index(d)
    st = [dict(route=list(t['route']),
               boxes_at={s: [idx[bid] for bid in ids]
                         for s, ids in t['boxes_at'].items()})
          for t in payload['trips']]
    return st, [int(i) for i in payload['order']], payload['metrics']


def recommended(d, verbose=False):
    """
    Q3 / Q4 / 绘图统一入口：返回 (trips, assignment, metrics, order)，即 main()
    实跑并落盘的那一份推荐解。读回后会重解一次 schedule() 并断言与落盘指标逐位
    一致——若不一致说明 schedule() 的语义被改过，属必须立刻暴露的破坏性变更。
    """
    st, order, m_ref = load_recommended(d)
    asg = schedule(d, st, order=order)
    m = evaluate(d, asg)
    assert m['hard_viol'] == 0, '读回的推荐方案违反硬时限，缓存已失效'
    for k, v in m_ref.items():
        if isinstance(v, float):
            assert abs(m[k] - v) <= 1e-6 * max(1.0, abs(v)), \
                f'推荐方案读回不一致：{k} 缓存={v} 重解={m[k]}'
    if verbose:
        print(f'  读回 Q2 推荐方案：{m["n_trips"]} 架次 / {m["total_E"]:.3f} kWh / '
              f'makespan {m["makespan"]:.1f} s / 硬违反 {m["hard_viol"]}')
    return st, asg, m, order


def solve_subset(d, services, n_iter=3000, seed=ALNS_SEED):
    """
    在指定服务区子集上跑 ALNS，供 q2_exact.py 的 Layer A 与精确解做**同算例**对比。
    口径与 Layer A 一致：字典序 (架次数, 能耗)，硬时限仍为硬约束。
    初始解取「每箱一架次」（对 A 型恒可行），由 ALNS 负责合并。
    返回 (trips, assignment, metrics, order)。
    """
    boxes = [b for s in services for b in d.boxes_by_service[s]]
    if not boxes:
        raise ValueError('子集为空')
    init = [dict(route=[b['service']], boxes=[b]) for b in _sorted_by_urgency(boxes)]
    # 能耗归一化基准取初始解能耗：使目标量与 SA 温度同量级（见 objective 的说明）
    st0 = to_sched_trips(d, init)
    ref_E = max(evaluate(d, schedule(d, st0))['total_E'], 1e-9)
    bt, bo, bm, bf, _ = alns(d, init, k_target=None, n_iter=n_iter, seed=seed,
                             mode='energy', ref_E=ref_E, t0=0.5)
    st = to_sched_trips(d, bt)
    asg = schedule(d, st, order=bo)
    m = evaluate(d, asg)
    assert m['hard_viol'] == 0, '子集算例出现硬时限违反，不应发生'
    return st, asg, m, bo


def pareto_front(rows):
    """四维非支配前沿：架次少、makespan 小、时延小、能耗低。"""
    out = []
    for i, a in enumerate(rows):
        dominated = False
        for j, b in enumerate(rows):
            if i == j:
                continue
            if (b['n_trips'] <= a['n_trips'] and b['makespan'] <= a['makespan'] + 1e-9
                    and b['tardiness'] <= a['tardiness'] + 1e-9
                    and b['total_E'] <= a['total_E'] + 1e-9
                    and (b['n_trips'] < a['n_trips'] or b['makespan'] < a['makespan'] - 1e-9
                         or b['tardiness'] < a['tardiness'] - 1e-9
                         or b['total_E'] < a['total_E'] - 1e-9)):
                dominated = True
                break
        if not dominated:
            out.append(a)
    return out


def ablation(d, n_iter=3000, ref_tard=1.0, ref_ms=1.0):
    """消融：①基线 ②仅顺序 ③ALNS 关顺序 ④ALNS 全量。防止把顺序的功劳记到组批搜索上。"""
    base_trips = construct_trips(d, pack_type='C')
    init = [dict(route=list(t['route']), boxes=[b for v in t['boxes_at'].values() for b in v])
            for t in base_trips]
    out = []
    asg = schedule(d, base_trips)
    m = evaluate(d, asg)
    out.append(dict(name='①基线（FFD 组批 + 紧迫度派工）', **{k: m[k] for k in
                 ('n_trips', 'total_E', 'makespan', 'tardiness', 'hard_viol')}))
    bt, bo, bm, bf, _ = alns(d, init, k_target=None, n_iter=n_iter, seed=ALNS_SEED + 1,
                             use_order=True, freeze_boxes=True,
                             ref_tard=ref_tard, ref_ms=ref_ms)
    out.append(dict(name='②仅顺序（组批冻结）', **{k: bm[k] for k in
                 ('n_trips', 'total_E', 'makespan', 'tardiness', 'hard_viol')}))
    bt, bo, bm, bf, _ = alns(d, init, k_target=None, n_iter=n_iter, seed=ALNS_SEED + 2,
                             use_order=False, freeze_boxes=False,
                             ref_tard=ref_tard, ref_ms=ref_ms)
    out.append(dict(name='③ALNS（派工顺序固定）', **{k: bm[k] for k in
                 ('n_trips', 'total_E', 'makespan', 'tardiness', 'hard_viol')}))
    bt, bo, bm, bf, _ = alns(d, init, k_target=None, n_iter=n_iter, seed=ALNS_SEED + 3,
                             use_order=True, freeze_boxes=False,
                             ref_tard=ref_tard, ref_ms=ref_ms)
    out.append(dict(name='④ALNS 全量（组批 + 顺序）', **{k: bm[k] for k in
                 ('n_trips', 'total_E', 'makespan', 'tardiness', 'hard_viol')}))
    return out


def main():
    d = load_data()
    d.geo_nodes, d.geo = precompute_geometry(d)
    t_start = time.time()

    # --- 基线回归护栏 -------------------------------------------------------
    print('=' * 74)
    print('基线回归检查（order=None 必须逐位复现历史行为）')
    trips0 = construct_trips(d, pack_type='C')
    asg0 = schedule(d, trips0)
    m0 = evaluate(d, asg0)
    ok = True
    for k, ref in BASE_REF.items():
        got = m0[k]
        good = (abs(got - ref) <= BASE_TOL * max(1.0, abs(ref))) if isinstance(ref, float) \
            else (got == ref)
        ok &= good
        print(f'  {k:10s} 实测={got!r:24s} 护栏={ref!r:24s} {"✓" if good else "✗"}')
    assert ok, '基线回归失败：schedule(construct_trips()) 行为已改变'
    print('  基线回归通过 ✓')

    # --- 记忆化对拍（硬性门禁）---------------------------------------------
    print('=' * 74)
    print('trip_plan 记忆化对拍（缓存路径 vs 无缓存直算，1000 例）')
    n_bad = memo_crosscheck(d, n=1000)
    print(f'  不一致例数 = {n_bad}（必须为 0）')
    assert n_bad == 0, '记忆化键漏项，缓存值与直算不符'

    # --- ALNS 扫描 ε-约束 ---------------------------------------------------
    print('=' * 74)
    print('ε-约束扫描架次数 K（ALNS，时效优先目标）')
    st, asg, m, info = solve_recommended(d, n_iter=4000, verbose=True)
    print(f'  推荐（K<={info["K"]}）: 架次={m["n_trips"]} 能耗={m["total_E"]:.3f} kWh '
          f'makespan={m["makespan"]:.1f} s ({m["makespan"]/3600:.2f} h) '
          f'加权时延={m["tardiness"]:.0f} 硬违反={m["hard_viol"]}')

    # --- 消融 ---------------------------------------------------------------
    print('=' * 74)
    print('消融实验')
    abl = ablation(d, n_iter=3000, ref_tard=info['ref_tard'], ref_ms=info['ref_ms'])
    for r in abl:
        print(f'  {r["name"]:26s} 架次={r["n_trips"]:2d} 能耗={r["total_E"]:7.3f} '
              f'makespan={r["makespan"]:9.1f} 时延={r["tardiness"]:10.1f} 硬违反={r["hard_viol"]}')

    # --- 导出 ---------------------------------------------------------------
    export(d, asg, info, abl)
    print('=' * 74)
    print(f'已导出 q2_transport_trips.csv / q2_box_delivery.csv / q2_pareto.csv / '
          f'q2_alns_trace.csv / q2_ablation.csv  总耗时 {time.time()-t_start:.1f}s')


def memo_crosscheck(d, n=1000, seed=7):
    """随机 1000 例：缓存 trip_plan 与无缓存直算逐字段比对。"""
    rng = random.Random(seed)
    svc = d.S
    bad = 0
    for _ in range(n):
        k = rng.randint(1, 4)
        route = rng.sample(svc, k)
        boxes_at = {}
        for sid in route:
            pool = d.boxes_by_service[sid]
            nb = rng.randint(1, len(pool))
            boxes_at[sid] = rng.sample(pool, nb)
        g = rng.choice(['A', 'B', 'C'])
        t = d.transport_types[g]
        a = trip_plan(d, t, route, boxes_at)
        b = trip_plan_direct(d, t, route, boxes_at)
        same = (abs(a['E'] - b['E']) < 1e-12
                and abs(a['duration'] - b['duration']) < 1e-9
                and a['deliver'] == b['deliver']
                and a['feasible'] == b['feasible'])
        if not same:
            bad += 1
    return bad


def export(d, asg, info, abl):
    assignment_sorted = sorted(asg, key=lambda x: x['start'])
    q2_rows, deliver_rows = [], []
    for i, a in enumerate(assignment_sorted):
        q2_rows.append(dict(
            架次编号=f'T{i+1:02d}', 无人机编号=a['uav'], 机型编号=a['type'],
            电池编号=a['battery'], 开始时刻s=round(a['start'], T_DEC),
            访问服务区顺序='->'.join(a['route']), 返回O01时刻s=round(a['start'] + a['duration'], T_DEC),
            架次能耗kWh=round(a['E'], E_DEC)))
        for sid, bs in a['boxes_at'].items():
            for b in bs:
                deliver_rows.append(dict(
                    货箱编号=b['id'], 架次编号=f'T{i+1:02d}', 服务区编号=sid,
                    交付完成时刻s=round(a['deliver_abs'][b['id']], T_DEC)))
    pd.DataFrame(q2_rows).to_csv(os.path.join(OUT, 'q2_transport_trips.csv'), index=False)
    pd.DataFrame(deliver_rows).to_csv(os.path.join(OUT, 'q2_box_delivery.csv'), index=False)

    # Pareto / ε-约束扫描结果
    rows = []
    for K, r in sorted(info['runs'].items()):
        m = r['metrics']
        if m is None:
            continue
        rows.append(dict(K上限=K, 实际架次=m['n_trips'], 能耗kWh=round(m['total_E'], E_DEC),
                         makespan_s=round(m['makespan'], T_DEC), 加权时延=round(m['tardiness'], 3),
                         硬约束违反=m['hard_viol'], 目标值=round(r['f'], 6),
                         耗时s=round(r['secs'], 2)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, 'q2_pareto.csv'), index=False)

    # ALNS 收敛轨迹（重跑一次带 trace，用于绘图）
    tr = []
    alns(d, info['init'], k_target=info['K'], n_iter=1500, seed=ALNS_SEED + info['K'],
         ref_tard=info['ref_tard'], ref_ms=info['ref_ms'], trace=tr)
    pd.DataFrame(tr, columns=['迭代', '最优目标值', 'makespan_s', '加权时延', '能耗kWh', '架次']
                 ).to_csv(os.path.join(OUT, 'q2_alns_trace.csv'), index=False)

    pd.DataFrame(abl).to_csv(os.path.join(OUT, 'q2_ablation.csv'), index=False)


if __name__ == '__main__':
    main()
