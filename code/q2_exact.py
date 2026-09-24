# -*- coding: utf-8 -*-
# 本程序及代码是在人工智能工具辅助下完成的。
# 工具名称：DeepSeek；版本/型号：DeepSeek V4.1 Flash（API 模型名 deepseek-flash）；
# 开发机构：杭州深度求索人工智能基础技术研究有限公司；
# 版本发布日期：2026 年 9 月 10 日。

"""
问题二 小规模精确验证（独立脚本，不参与主求解）

Layer A  分区层：3 个服务区小算例上的**完全枚举 + 集合划分 ILP**。
         算例只含 3 个区时，任何架次的区域集合天然 ⊆ 这 3 个区，故 6 个排列即
         全部可能路径，区域集合与排列同时被穷举；列 = (路线排列 × 每区非空子集)。
         两层字典序：先最小化架次数，再在该架次数下最小化能耗。
         ALNS 用同一算例、同一目标口径（mode='energy'）对比，报最优性间隙。

Layer B  时序层：big-M 析取 MILP，精确解派工顺序与各架次开始时刻。
         B1 = 固定贪心那份（机型, 无人机, 电池）指派，只解排序与开始时刻
              → 给出"贪心调度在给定指派下已近最优"的证书。
         B2 = 机型固定、无人机与共享电池的**物理指派自由**，同样精确解。
         两者都配：贪心解暖可行性检验 + MIP 解回放一致性检验。

big-M 取值：M ≥ UB + max_p (dur_p + chg_p)。取 M = UB 会造成**假不可行**
（开始时刻之差最小可到 −UB，而资源占用最长可达 UB 之外），已实测踩坑，故这里
显式取 UB + max(dur+chg) 并在结果里回放验证。

适用范围如实声明：3 区算例里 8 架机对 ≤9 架次，**资源竞争几乎不存在**，故本脚本
验证的是组批/能耗/路径逻辑与时序逻辑，**不能**外推为"全局资源拥塞已证最优"。
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import os
import time
from itertools import combinations, permutations, product
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix
from core import load_data, charge_time
from q2 import (precompute_geometry, construct_trips, schedule, evaluate,
                best_type_plan, to_sched_trips, alns, solve_subset, _sorted_by_urgency)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'results')
os.makedirs(OUT, exist_ok=True)


# ===========================================================================
# Layer A：分区层
# ===========================================================================
def gen_columns(d, services, pool, cap=400000):
    """枚举全部候选架次列：路线排列 × 每区非空子集，仅保留至少一种机型可行的。

    调用 best_type_plan（含 60 次二分的载荷/能耗计算）之前先过一个**必要条件下筛**：
    全架次质量 ≤ 最大机型载重、体积 ≤ 最大机型容积。这一步把组合数压掉一两个数量级，
    且只剔除必然不可行的列，不影响枚举的完备性。
    """
    qmax = max(t['Q'] for t in d.transport_types.values())
    vmax = max(t['volume'] for t in d.transport_types.values())
    # 每区的非空子集及其 (质量, 体积)
    subs = {}
    for s in services:
        n = len(pool[s])
        subs[s] = [(tuple(c), sum(pool[s][i]['mass'] for i in c),
                    sum(pool[s][i]['volume'] for i in c))
                   for r in range(1, n + 1) for c in combinations(range(n), r)]
    total = 0
    for k in range(1, len(services) + 1):
        for rsub in combinations(services, k):
            for perm in permutations(rsub):
                acc = 1
                for s in perm:
                    acc *= len(subs[s])
                total += acc
    if total > cap:
        raise RuntimeError(f'候选组合上界 {total} 超过 cap={cap}，需缩小算例或提高 cap')
    cols = []
    for k in range(1, len(services) + 1):
        for rsub in combinations(services, k):
            for perm in permutations(rsub):
                for combo in product(*[subs[s] for s in perm]):
                    if sum(c[1] for c in combo) > qmax or sum(c[2] for c in combo) > vmax:
                        continue
                    boxes_at = {s: [pool[s][i] for i in c[0]]
                                for s, c in zip(perm, combo)}
                    g, plan = best_type_plan(d, list(perm), boxes_at)
                    if plan is None:
                        continue
                    cols.append(dict(route=list(perm), boxes_at=boxes_at,
                                     E=plan['E'], type=g))
    return cols


def solve_layerA(d, services, time_limit=300.0, verbose=False):
    """完全枚举 + 两层字典序集合划分 ILP。返回最优 (架次数, 能耗) 与列数。"""
    pool = {s: d.boxes_by_service[s] for s in services}
    boxes = [b for s in services for b in pool[s]]
    bidx = {b['id']: i for i, b in enumerate(boxes)}
    t0 = time.time()
    cols = gen_columns(d, services, pool)
    t_gen = time.time() - t0
    n, m = len(boxes), len(cols)
    if m == 0:
        raise RuntimeError('无可行列')
    # 覆盖矩阵：A[i, c] = 1 表示第 i 箱属于第 c 列
    A = lil_matrix((n, m))
    for c, col in enumerate(cols):
        for s, bs in col['boxes_at'].items():
            for b in bs:
                A[bidx[b['id']], c] = 1.0
    A = A.tocsr()
    ones = np.ones(n)
    integ = np.ones(m)
    bnd = Bounds(np.zeros(m), np.ones(m))
    # 第一层：最小架次数
    r1 = milp(np.ones(m), constraints=[LinearConstraint(A, ones, ones)],
              integrality=integ, bounds=bnd,
              options=dict(time_limit=time_limit, mip_rel_gap=0.0, presolve=True))
    if not r1.success:
        raise RuntimeError(f'Layer A 第一层未求解成功: {r1.message}')
    K = int(round(r1.fun))
    # 第二层：固定 K，最小能耗
    A2 = lil_matrix((n + 1, m))
    A2[:n, :] = A
    A2[n, :] = 1.0
    A2 = A2.tocsr()
    lo = np.concatenate([ones, [K - 1e-6]])
    hi = np.concatenate([ones, [K + 1e-6]])
    r2 = milp(np.array([c['E'] for c in cols]),
              constraints=[LinearConstraint(A2, lo, hi)],
              integrality=integ, bounds=bnd,
              options=dict(time_limit=time_limit, mip_rel_gap=0.0, presolve=True))
    if not r2.success:
        raise RuntimeError(f'Layer A 第二层未求解成功: {r2.message}')
    t_milp = time.time() - t0 - t_gen          # 两个 MILP 的求解时间
    t_tot = time.time() - t0                   # 列生成 + 两个 MILP，完整耗时
    if verbose:
        print(f'   列数={m} 列生成={t_gen:.1f}s 两阶段MILP={t_milp:.1f}s '
              f'合计={t_tot:.1f}s  K*={K}  E*={r2.fun:.6f} kWh')
    return dict(n_trips=K, energy=float(r2.fun), n_cols=m,
                t_gen=t_gen, t_milp=t_milp, t_exact=t_tot,
                gap=(max(r1.mip_gap or 0.0, r2.mip_gap or 0.0)
                     if hasattr(r2, 'mip_gap') else None),
                x=np.asarray(r2.x).ravel(), cols=cols)


def layerA_case(d, services, n_iter=3000, seed=11, time_limit=300.0):
    """一个算例：精确解 vs 同算例 ALNS。"""
    nb = sum(len(d.boxes_by_service[s]) for s in services)
    ex = solve_layerA(d, services, time_limit=time_limit, verbose=True)
    t0 = time.time()
    st, asg, m, order = solve_subset(d, services, n_iter=n_iter, seed=seed)
    t_alns = time.time() - t0
    assert len(asg) > 0 and sum(len(v) for a in asg for v in a['boxes_at'].values()) == nb, \
        'ALNS 解的货箱总数与算例不符'
    return dict(services=services, n_boxes=nb,
                exact_K=ex['n_trips'], exact_E=ex['energy'], n_cols=ex['n_cols'],
                alns_K=m['n_trips'], alns_E=m['total_E'],
                gap_K=m['n_trips'] - ex['n_trips'],
                gap_E=(m['total_E'] - ex['energy']) / ex['energy'] * 100.0,
                # 与 ALNS 比时间必须用**完整**耗时（列生成 + 两个 MILP），
                # 只报列生成时间会把这个"精确解"说快一个数量级
                t_exact=ex['t_exact'], t_gen=ex['t_gen'], t_milp=ex['t_milp'],
                t_alns=t_alns,
                # 两个 MILP 各自的 mip_gap（无此属性时为 None）：间隙为 0 才算
                # 证到最优。不写「超时」字段——本层没有捕获求解器状态，缺就不要编。
                gap=ex['gap'])


# ===========================================================================
# Layer B：时序层（big-M 析取 MILP）
# ===========================================================================
def _ub_sequential(d, asg):
    """
    big-M 的基准上界 UB：**计入共享电池充电占用**的安全串行排程完工时刻。

    构造一份必然可行的排程——完全串行，任一时刻只有一架次在飞：第 p 架次的开始
    时刻取前面所有架次的 (duration + 充电时长) 之和。无重叠即无资源冲突（无人机
    与电池即便同名也不会被同时占用），故它是**可行**排程；而任何可行排程的完工
    时刻都不可能超过一个可行排程，故它同时是我们需要的上界。

    旧版只累加 duration，漏掉了电池充电占用，取出的 UB 偏小 → M 偏小 →
    析取约束把「后一架次必须等前一架次连充电都完成」误编码成可以更早开始，
    进而把真正可行的排程判为不可行（假不可行）。这是本脚本初版真实踩过的坑的
    同一个根因，故此处显式计入充电，并在 solve_layerB 里回放验证该界不截掉可行解。
    """
    t = 0.0
    for a in sorted(asg, key=lambda x: x['start']):
        soc_end = 1.0 - a['plan']['E'] / d.transport_types[a['type']]['E_use']
        t += a['duration'] + charge_time(soc_end, d.batteries[a['type']][1])
    return t


def solve_layerB(d, asg, free_units=False, time_limit=90.0, criterion='makespan'):
    """
    精确解派工顺序与开始时刻（big-M 析取 MILP）。
      free_units=False (B1)：机型/无人机/电池全部沿用贪心指派，只解顺序与开始时刻。
      free_units=True  (B2)：机型沿用贪心指派，无人机与共享电池的物理指派自由。
      criterion='makespan'  目标 = 最小化最后完成时刻。
      criterion='tardiness' 目标 = 最小化加权时延 Σ w_k L_k，**并把硬时限写成约束**
                            （start_p ≤ 期限 − 交付偏移），即 Q2 真正采用的口径。
                            makespan 口径的最优解会违反硬时限（实测 8 项），
                            故只有本口径的解才能与 ALNS 推荐方案直接比较。

    析取编码：对每个可能冲突的架次对 {p,q} 设两个二元变量 z_pq / z_qp，
      z_pq = 1 ⇒ p 先于 q： start_p − start_q + M·z_pq ≤ M − dur_p
      z_qp = 1 ⇒ q 先于 p： start_q − start_p + M·z_qp ≤ M − dur_q
    **必须两个方向各一个变量**。只设一个 z 再令 z_qp = 1 − z_pq 会把
    「z_pq + z_qp ≤ 1」退化成「2·z_pq ≤ 1」，并让"同资源必排序"约束方向错乱——
    这是本脚本初版真实踩过的坑，故显式保留两方向变量。

    资源占用长度不同：无人机占用 dur，共享电池占用 dur + 充电时长，故两套析取分开写。
    固定指派下"是否共用资源"是常量，只对确实共用的对建变量；自由指派下同机型对都可能
    共用，故全部建变量并用 y 连接（y_pu + y_qu ≤ 1 + z_pq + z_qp）。

    返回 dict(makespan, starts, units, n_bin, status, gap, secs, M, ub, free_units)
    """
    n = len(asg)
    types = [a['type'] for a in asg]
    dur = np.array([a['duration'] for a in asg], dtype=float)
    # 电池在本架次结束后还要充电，充电期同样不可被别的架次占用
    chg = np.array([charge_time(1.0 - a['plan']['E'] / d.transport_types[a['type']]['E_use'],
                                d.batteries[a['type']][1]) for a in asg], dtype=float)
    ub = _ub_sequential(d, asg)
    M = ub + float(np.max(dur + chg))     # 见文件头：M 必须 ≥ UB + max τ
    Hmax = ub + float(np.max(dur + chg))
    # 上界不截掉可行解的**实证**：贪心排程（B1 口径下的一手可行解）的完工时刻与
    # 最大开始时刻都必须落在 Hmax 之内；若贴边，说明界太紧，M 也就不可信。
    greedy_done = float(np.max([a['start'] + a['duration'] for a in asg]))

    nv = n + 1
    rows_eq, rhs_eq, rows_le, rhs_le = [], [], [], []
    rows_ge, rhs_ge = [], []

    def eq(row, rhs):
        rows_eq.append(row); rhs_eq.append(rhs)

    def le(row, rhs):
        rows_le.append(row); rhs_le.append(rhs)

    # --- 时延口径：硬时限化为 start_p 的上界，并为每个软时延箱引入 L_k ≥ 0
    hard_ub = np.full(n, np.inf)
    L_aux, L_off, L_exp, L_w = {}, {}, {}, {}
    if criterion == 'tardiness':
        for p, a in enumerate(asg):
            for bs in a['boxes_at'].values():
                for b in bs:
                    off = a['plan']['deliver'][b['id']]
                    lim = None
                    if b['category'] == '医疗物资':
                        lim = b['expect']
                    if b['first_batch']:
                        lim = b['deadline'] if lim is None else min(lim, b['deadline'])
                    if lim is not None:
                        hard_ub[p] = min(hard_ub[p], lim - off)
                    if b['expect'] is not None:
                        L_aux[(p, b['id'])] = nv
                        L_off[(p, b['id'])] = off
                        L_exp[(p, b['id'])] = b['expect']
                        L_w[(p, b['id'])] = b['priority']
                        nv += 1
        if np.any(hard_ub < -1e-9):
            raise RuntimeError('硬时限在给定指派下不可行：存在架次最早交付即已超期')

    # --- 指派变量（仅 free_units）
    y_uav, y_bat = {}, {}
    if free_units:
        for p in range(n):
            g = types[p]
            for u in d.uavs:
                if u['type'] == g:
                    y_uav[(p, u['id'])] = nv; nv += 1
            for b in range(d.batteries[g][0]):
                y_bat[(p, b)] = nv; nv += 1

    # --- 需要析取的架次对（只有同机型才可能共用无人机或电池）
    pair_uav, pair_bat = [], []
    for p, q in combinations(range(n), 2):
        if types[p] != types[q]:
            continue
        if free_units:
            pair_uav.append((p, q)); pair_bat.append((p, q))
        else:
            if asg[p]['uav'] == asg[q]['uav']:
                pair_uav.append((p, q))
            if asg[p]['battery'] == asg[q]['battery']:
                pair_bat.append((p, q))

    zU, zB = {}, {}
    for (p, q) in pair_uav:
        zU[(p, q)] = nv; nv += 1
        zU[(q, p)] = nv; nv += 1
    for (p, q) in pair_bat:
        zB[(p, q)] = nv; nv += 1
        zB[(q, p)] = nv; nv += 1

    # makespan 定义：H ≥ start_p + dur_p
    for p in range(n):
        row = np.zeros(nv); row[p] = 1.0; row[n] = -1.0
        le(row, -dur[p])

    # 指派唯一性
    if free_units:
        for p in range(n):
            g = types[p]
            row = np.zeros(nv)
            for u in d.uavs:
                if u['type'] == g:
                    row[y_uav[(p, u['id'])]] = 1.0
            eq(row, 1.0)
            row = np.zeros(nv)
            for b in range(d.batteries[g][0]):
                row[y_bat[(p, b)]] = 1.0
            eq(row, 1.0)

    # 无人机析取（占用 dur）
    for (p, q) in pair_uav:
        zpq, zqp = zU[(p, q)], zU[(q, p)]
        row = np.zeros(nv); row[p] = 1.0; row[q] = -1.0; row[zpq] = M
        le(row, M - dur[p])
        row = np.zeros(nv); row[q] = 1.0; row[p] = -1.0; row[zqp] = M
        le(row, M - dur[q])
        row = np.zeros(nv); row[zpq] = 1.0; row[zqp] = 1.0
        if free_units:
            le(row, 1.0)          # 不同无人机时允许并行，故只要求互斥
        else:
            eq(row, 1.0)          # 固定指派下确实共用 ⇒ 必须一前一后
    # 电池析取（占用 dur + 充电）
    for (p, q) in pair_bat:
        zpq, zqp = zB[(p, q)], zB[(q, p)]
        row = np.zeros(nv); row[p] = 1.0; row[q] = -1.0; row[zpq] = M
        le(row, M - (dur[p] + chg[p]))
        row = np.zeros(nv); row[q] = 1.0; row[p] = -1.0; row[zqp] = M
        le(row, M - (dur[q] + chg[q]))
        row = np.zeros(nv); row[zpq] = 1.0; row[zqp] = 1.0
        if free_units:
            le(row, 1.0)
        else:
            eq(row, 1.0)

    # 自由指派：共用同一实体资源 ⇒ 必须被排序
    if free_units:
        for (p, q) in pair_uav:
            zpq, zqp = zU[(p, q)], zU[(q, p)]
            for u in d.uavs:
                if u['type'] != types[p]:
                    continue
                row = np.zeros(nv)
                row[y_uav[(p, u['id'])]] = 1.0
                row[y_uav[(q, u['id'])]] = 1.0
                row[zpq] = -1.0; row[zqp] = -1.0
                le(row, 1.0)
        for (p, q) in pair_bat:
            zpq, zqp = zB[(p, q)], zB[(q, p)]
            for b in range(d.batteries[types[p]][0]):
                row = np.zeros(nv)
                row[y_bat[(p, b)]] = 1.0
                row[y_bat[(q, b)]] = 1.0
                row[zpq] = -1.0; row[zqp] = -1.0
                le(row, 1.0)

    # 软时延： L_k ≥ start_p + off_k − expect_k
    for (p, bid), vi in L_aux.items():
        row = np.zeros(nv); row[p] = -1.0; row[vi] = 1.0
        rows_ge.append(row); rhs_ge.append(L_off[(p, bid)] - L_exp[(p, bid)])

    # 目标
    c = np.zeros(nv)
    if criterion == 'tardiness':
        for (p, bid), vi in L_aux.items():
            c[vi] = L_w[(p, bid)]
    else:
        c[n] = 1.0

    lb = np.zeros(nv)
    ubv = np.full(nv, 1.0)                # n+1 之后全为二元变量，上界 1
    ubv[:n + 1] = Hmax                    # start 与 H 连续
    integ = np.zeros(nv)
    integ[n + 1:] = 1
    if criterion == 'tardiness':
        for p in range(n):
            ubv[p] = min(Hmax, hard_ub[p])   # 硬时限直接压 start_p 的上界
        for vi in L_aux.values():
            ubv[vi] = np.inf                 # L_k 为连续非负变量
            integ[vi] = 0                    # 不能落在 integ[n+1:]=1 的整数块里

    cons = []
    if rows_eq:                           # B1 可能一条等式都没有，空矩阵会广播失败
        cons.append(LinearConstraint(np.array(rows_eq),
                                     np.array(rhs_eq), np.array(rhs_eq)))
    cons.append(LinearConstraint(np.array(rows_le), -np.inf, np.array(rhs_le)))
    if rows_ge:
        cons.append(LinearConstraint(np.array(rows_ge), np.array(rhs_ge), np.inf))
    t0 = time.time()
    r = milp(c, constraints=cons, integrality=integ, bounds=Bounds(lb, ubv),
             options=dict(time_limit=time_limit, mip_rel_gap=1e-9, presolve=True))
    secs = time.time() - t0
    if r.x is None:
        raise RuntimeError(f'Layer B 未给出解: {r.message}')
    x = np.asarray(r.x)
    starts = x[:n]
    # 还原每架次实际占用的 (无人机, 电池)
    units = []
    for p in range(n):
        if not free_units:
            units.append((asg[p]['uav'], asg[p]['battery']))
        else:
            g = types[p]
            u_best = max((u['id'] for u in d.uavs if u['type'] == g),
                         key=lambda uid: x[y_uav[(p, uid)]])
            b_best = max(range(d.batteries[g][0]), key=lambda b: x[y_bat[(p, b)]])
            units.append((u_best, f'{g}B{b_best}'))
    # 目标值与完工时刻是两回事：criterion='tardiness' 时 r.fun 是加权时延，把它
    # 塞进名为 makespan 的键里会让调用方把两个量混为一谈（旧版就是这样）。故分开：
    #   objective_value —— MILP 目标函数值
    #   makespan        —— 由 start+duration 独立算出的真实完工时刻
    #   proved_optimal  —— 是否已证最优（超时但有可行解 ≠ 已证最优）
    obj = float(r.fun)
    makespan = float(np.max(starts + dur))
    timed_out = (getattr(r, 'status', 0) == 1)
    return dict(objective_value=obj, makespan=makespan,
                criterion=criterion, starts=starts, units=units,
                n_bin=int(integ.sum()), status=r.message,
                proved_optimal=bool(r.success), timed_out=timed_out,
                incumbent=obj, best_bound=getattr(r, 'mip_dual_bound', None),
                gap=getattr(r, 'mip_gap', None), secs=secs,
                M=M, ub=ub, bound_headroom=Hmax - greedy_done,
                free_units=free_units)


def replay_check(d, asg, starts, units):
    """回放一致性检验：按资源占用独立重算一遍，确认无人机与电池都无重叠。

    只依据 (start, duration, charge) 与资源编号，不读取 MILP 内部状态。
    返回 (ok, 说明)。
    """
    uav_int, bat_int = {}, {}
    for p, a in enumerate(asg):
        u, b = units[p]
        uav_int.setdefault(u, []).append((starts[p], starts[p] + a['duration'], p))
        chg = charge_time(1.0 - a['plan']['E'] / d.transport_types[a['type']]['E_use'],
                          d.batteries[a['type']][1])
        bat_int.setdefault(b, []).append((starts[p], starts[p] + a['duration'] + chg, p))
    for name, book in (('无人机', uav_int), ('电池', bat_int)):
        for r, iv in book.items():
            iv.sort()
            for k in range(1, len(iv)):
                if iv[k][0] < iv[k - 1][1] - 1e-6:
                    return False, f'{name} {r} 冲突: {iv[k-1][:2]} 与 {iv[k][:2]}'
    return True, f'无人机 {len(uav_int)} 架 / 电池 {len(bat_int)} 组均无重叠'


def eval_starts(d, asg, starts):
    """由给定开始时刻重算指标（机型/指派沿用 asg）。"""
    hard = 0
    tard = 0.0
    ms = 0.0
    for p, a in enumerate(asg):
        ms = max(ms, starts[p] + a['duration'])
        for sid, bs in a['boxes_at'].items():
            for b in bs:
                t_del = starts[p] + a['plan']['deliver'][b['id']]
                if b['category'] == '医疗物资' and t_del > b['expect'] + 1e-6:
                    hard += 1
                if b['first_batch'] and t_del > b['deadline'] + 1e-6:
                    hard += 1
                if b['expect'] is not None:
                    tard += b['priority'] * max(0.0, t_del - b['expect'])
    return dict(makespan=float(ms), hard_viol=hard, tardiness=tard)


# ===========================================================================
def main():
    d = load_data()
    d.geo_nodes, d.geo = precompute_geometry(d)
    rows = []

    print('=' * 74)
    print('Layer A  分区层：3 服务区小算例（完全枚举 + 集合划分 ILP）')
    cases = [['S013', 'S014', 'S015'],      # 松紧混合，9 箱
             ['S009', 'S010', 'S011'],      # 同规模，时限分布不同
             ['S002', 'S009', 'S010']]      # 箱多压测，14 箱
    for svc in cases:
        nb = sum(len(d.boxes_by_service[s]) for s in svc)
        print(f'  算例 {"+".join(svc)}（{nb} 箱）')
        try:
            r = layerA_case(d, svc, n_iter=3000, seed=11)
            rows.append(dict(层级='A', 算例='+'.join(svc), 箱数=nb, 列数=r['n_cols'],
                             目标口径='能耗（字典序：架次→能耗）',
                             精确架次=r['exact_K'], 精确能耗=round(r['exact_E'], 6),
                             ALNS架次=r['alns_K'], ALNS能耗=round(r['alns_E'], 6),
                             架次间隙=r['gap_K'], 能耗间隙pct=round(r['gap_E'], 4),
                             上下界间隙=(None if r['gap'] is None else round(r['gap'], 9)),
                             已证最优=(None if r['gap'] is None else int(r['gap'] <= 1e-9)),
                             精确耗时s=round(r['t_exact'], 2),
                             ALNS耗时s=round(r['t_alns'], 2)))
            print(f'    精确 K*={r["exact_K"]} E*={r["exact_E"]:.4f} '
                  f'(列生成{r["t_gen"]:.0f}s + MILP{r["t_milp"]:.0f}s = {r["t_exact"]:.0f}s) | '
                  f'ALNS K={r["alns_K"]} E={r["alns_E"]:.4f} ({r["t_alns"]:.0f}s) | '
                  f'架次间隙={r["gap_K"]} 能耗间隙={r["gap_E"]:+.3f}%')
        except Exception as e:
            print(f'    [跳过] {e}')

    print('=' * 74)
    print('Layer B  时序层：big-M 析取 MILP（精确顺序与开始时刻）')
    trips = construct_trips(d, pack_type='C')
    asg = schedule(d, trips)
    m_g = evaluate(d, asg)
    print(f'  贪心参考: makespan={m_g["makespan"]:.1f}s 时延={m_g["tardiness"]:.0f} '
          f'硬违反={m_g["hard_viol"]}')
    greedy_units = [(a['uav'], a['battery']) for a in asg]
    greedy_starts = np.array([a['start'] for a in asg])
    for fu, crit, label in ((False, 'makespan', 'B1 固定指派·最短完工'),
                            (False, 'tardiness', "B1' 固定指派·加权时延"),
                            (True, 'makespan', 'B2 无人机/电池自由·最短完工')):
        try:
            r = solve_layerB(d, asg, free_units=fu, time_limit=90.0, criterion=crit)
            ok, msg = replay_check(d, asg, r['starts'], r['units'])
            # 暖可行性检验：贪心解本身必须在该模型里可行，否则模型有假不可行
            warm_ok, warm_msg = replay_check(d, asg, greedy_starts, greedy_units)
            mm = eval_starts(d, asg, r['starts'])
            imp = (m_g['makespan'] - mm['makespan']) / m_g['makespan'] * 100.0
            gap = r['gap'] if r['gap'] is not None else 0.0
            # 列名必须自带口径：层 B 的目标是「最短完工」或「最小加权时延」，同一个
            # 槽位在不同算例里装的是不同量纲的数。原来只写「精确能耗 / ALNS能耗」，
            # 读表的人会以为 B1 行的 9.7e5 是能耗，而它其实是加权时延。
            rows.append(dict(层级=label.split()[0], 算例=label, 箱数=80, 列数=r['n_bin'],
                             目标口径=r['criterion'],
                             精确架次=len(asg),
                             目标值=round(r['objective_value'], 6),
                             回放makespan_s=round(mm['makespan'], 3),
                             回放加权时延=round(mm['tardiness'], 3),
                             回放硬违反=int(mm['hard_viol']),
                             相对贪心makespan_pct=round(imp, 4),
                             上下界间隙=round(gap, 6),
                             已证最优=int(r['proved_optimal']), 超时=int(r['timed_out']),
                             大M=round(r['M'], 1), UB=round(r['ub'], 1),
                             界余量s=round(r['bound_headroom'], 1),
                             回放校验=int(bool(ok)), 贪心暖可行=int(bool(warm_ok)),
                             求解s=round(r['secs'], 2)))
            print(f'  {label}: 二元变量={r["n_bin"]} 目标值={r["objective_value"]:.1f} '
                  f'（{r["criterion"]} 口径）gap={gap:.4%} {r["secs"]:.1f}s '
                  f'{"已证最优" if r["proved_optimal"] else ("超时(有可行解，未证最优)" if r["timed_out"] else "未证最优")}')
            print(f'    回放后实际: makespan={mm["makespan"]:.1f}s '
                  f'(贪心 {m_g["makespan"]:.1f}s, {imp:+.2f}%) '
                  f'时延={mm["tardiness"]:.0f}（贪心 {m_g["tardiness"]:.0f}）'
                  f' 硬违反={mm["hard_viol"]}')
            print(f'    大M={r["M"]:.1f}（UB={r["ub"]:.1f} 含充电等待 + max τ）；'
                  f'界余量={r["bound_headroom"]:.0f}s（贪心完工距 Hmax，>0 说明界未截掉可行解）；'
                  f'贪心暖可行性={warm_ok}；MIP 解回放={ok}（{msg}）')
        except Exception as e:
            print(f'  {label}: [跳过] {e}')

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, 'q2_exact.csv'), index=False)
    print('=' * 74)
    print('已导出 q2_exact.csv')


if __name__ == '__main__':
    main()
