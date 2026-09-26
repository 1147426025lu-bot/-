# -*- coding: utf-8 -*-
# 本程序及代码是在人工智能工具辅助下完成的。
# 工具名称：DeepSeek；版本/型号：DeepSeek V4.1 Flash（API 模型名 deepseek-flash）；
# 开发机构：杭州深度求索人工智能基础技术研究有限公司；
# 版本发布日期：2026 年 9 月 10 日。

"""
问题四：救援任务分区与资源配置优化

 1) must-link：同一个运输架次服务的多个服务区必须留在同一任务组（拆开就要重排
    架次，问题二、三的全部结论作废），用并查集取连通分量，得**原子任务单元**。
 2) 完全枚举：单元数很少，K=2 的全部非空二分（2^n−2)/2、K=3 的全部 S(n,3) 种
    三分的数量级是 10^3，直接穷举即**全局最优**，不需要启发式。另用「组列集合
    划分 ILP」独立复核，与问题一的双路径验证同一范式。
 3) 资源核算：逐组算运输无人机、共享电池、中继无人机、中继能源组件的需求。
    中继能源组件按**峰值区间需求** max_t Σ 1(t ∈ [起飞, 返航+充电]) 核算，
    与运输共享电池的周转口径一致；不再取「组件数 = 中继机数」。
 4) 正式指标：资源规模 R_k、缺口 Gap_kg、冗余 Red_kg、工作量均衡 CV_W。
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import os
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy import sparse
from core import T_DEC, E_DEC, load_data, charge_time
from q2 import precompute_geometry
from solution_io import (load_solution, input_hashes, solver_hashes,
                         solver_hashes_text, Q3_SOLVER_FILES)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'results')
os.makedirs(OUT, exist_ok=True)

Q3_SOL = os.path.join(OUT, 'q3_solution.json')

TYPES = ['A', 'B', 'C']
RES_KEYS = [f'{g}_uav' for g in TYPES] + [f'{g}_bat' for g in TYPES] + ['relay', 'relay_comp']
RES_LABEL = {'A_uav': 'A型运输无人机', 'B_uav': 'B型运输无人机', 'C_uav': 'C型运输无人机',
             'A_bat': 'A型共享电池', 'B_bat': 'B型共享电池', 'C_bat': 'C型共享电池',
             'relay': '中继无人机', 'relay_comp': '中继能源组件'}


def peak_concurrency(intervals):
    """区间列表 [(s,e)] 的最大重叠数。"""
    if not intervals:
        return 0
    ev = []
    for (s, e) in intervals:
        ev.append((s, +1))
        ev.append((e, -1))
    ev.sort(key=lambda x: (x[0], x[1]))
    cur = peak = 0
    for (_, dk) in ev:
        cur += dk
        peak = max(peak, cur)
    return peak


# ---------------------------------------------------------------------------
# 1. 原子任务单元（must-link 并查集）
# ---------------------------------------------------------------------------
def atomic_units(d, assignment):
    """同一架次涉及的服务区连边，取连通分量作为不可再拆的原子单元。"""
    parent = {s: s for s in d.S}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a in assignment:
        for s in a['route'][1:]:
            ra, rb = find(a['route'][0]), find(s)
            if ra != rb:
                parent[ra] = rb
    groups = {}
    for s in d.S:
        groups.setdefault(find(s), []).append(s)
    units = [tuple(sorted(v)) for v in groups.values()]
    units.sort()
    return units


# ---------------------------------------------------------------------------
# 2. 完全枚举
# ---------------------------------------------------------------------------
def restricted_growth(n, K):
    """
    生成 n 个单元划入 K 个**非空且无序**组的全部方案（受限增长串），
    恰好枚举 S(n,K) 种，不重不漏——用 3^9 全排再对称去重会把每个分区
    数很多遍，且 K=2 时正好多算一倍。
    """
    seq = [0] * n

    def rec(i, mx):
        if i == n:
            if mx == K - 1:
                yield tuple(seq)
            return
        for g in range(min(mx + 1, K - 1) + 1):
            seq[i] = g
            yield from rec(i + 1, max(mx, g))
    if K > n:
        return
    yield from rec(0, -1)


def partition_sets(units, K):
    """把受限增长串还原成 [[单元下标, ...], ...]。"""
    for code in restricted_growth(len(units), K):
        groups = [[] for _ in range(K)]
        for i, g in enumerate(code):
            groups[g].append(i)
        yield groups


# ---------------------------------------------------------------------------
# 3. 资源核算
# ---------------------------------------------------------------------------
def group_resources(trip_ids, trips, relay_of_group):
    """
    一个任务组的资源需求。

    - 运输无人机：该组架次的 [开始, 完成] 峰值并发（同型分别算）
    - 共享电池  ：占用延续到充电完成，故区间取 [开始, 完成+充电时长]，与
                  q2.resource_usage 同口径
    - 中继无人机：该组失效区间所绑定的中继架次，其 [起飞, 返航] 峰值并发。
      一个中继架次若同时服务两个组的失效区间，两组各需配备一架——各组独立
      成队后无法共用，如实按重复计入（这正是分区带来的资源代价）。
    - 中继能源组件：同批架次的占用延续到充电完成，取峰值。
    """
    by_type = {g: [] for g in TYPES}
    for k in trip_ids:
        by_type[trips[k]['type']].append(trips[k])
    res = {}
    for g in TYPES:
        res[f'{g}_uav'] = peak_concurrency([(a['start'], a['end']) for a in by_type[g]])
        res[f'{g}_bat'] = peak_concurrency([(a['start'], a['bat_end']) for a in by_type[g]])
    # 中继无人机：占用到「返航 + 周转时间」为止，与 q3.schedule_relays 同口径。
    # 旧版只算到返航，少算了一段周转（本算例 300 s），于是同一条中继链上首尾
    # 相接的两个任务会被算成不重叠，峰值需求被系统性低估。
    # 再按**出动**归并：一次出动的多次访问共享同一段占用，逐行计会把它重复计入。
    relay_out = relay_outings(relay_of_group)
    relay_veh = [(x['start'], x['return_t'] + x['turnover']) for x in relay_out]
    relay_cmp = [(x['start'], x['cmp_end']) for x in relay_out]
    res['relay'] = peak_concurrency(relay_veh)
    res['relay_comp'] = peak_concurrency(relay_cmp)
    return res


def workload(trip_ids, trips, relay_of_group):
    """工作量 = 该组的运输飞行器时 + 中继飞行器时（架次占用总时长，秒）。"""
    w = sum(trips[k]['duration'] for k in trip_ids)
    # 中继的架次占用时长以**出动**计：一次出动服务多站时 `t_tot` 是整段行程的
    # 时长，同一次出动的各行取同值，逐行累加会成倍虚增工作量。
    w += sum(x['t_tot'] for x in relay_outings(relay_of_group))
    return w


def build_assignment(q3_sol, d):
    """
    由问题三方案构造资源核算用的运输架次记录：补齐充电结束时刻 `bat_end`
    （与 q2.resource_usage 同口径），并按开始时刻排序、写下标 `idx`。
    """
    out = []
    for t in q3_sol['transport_trips']:
        a = dict(t)
        a['E'] = float(t['energy_kwh'])
        a['end'] = t['start'] + t['duration']
        soc = 1.0 - a['E'] / d.transport_types[a['type']]['E_use']
        a['bat_end'] = a['end'] + charge_time(soc, d.batteries[a['type']][1])
        out.append(a)
    out.sort(key=lambda x: x['start'])
    for k, a in enumerate(out):
        a['idx'] = k
    return out


def build_relays(q3_sol, d):
    """由问题三方案构造中继记录：占用到「返航 + 周转」，能源组件到充电完成。

    一行 = 一次悬停站服务（与结果表同构）。自阶段 13 起一次出动可依次服务多个站，
    同一出动的各行共享 `start`/`return_time`/`energy_kwh`，故额外带上 `outing`；
    凡是按「占用」或「工作量」计的场合都必须先按 `outing` 归并（见
    `relay_outings`），否则同一次出动会被重复计入峰值。旧方案没有 `outing_id`
    字段，此时退回用 `relay_trip_id`，即每行自成一个出动——与旧口径逐位一致。
    """
    rt = d.relay_type
    return [dict(id=r['relay_trip_id'], station=r['station_id'], start=r['start'],
                 return_t=r['return_time'], t_tot=r['return_time'] - r['start'],
                 turnover=float(rt['turnover']),
                 outing=r.get('outing_id') or r['relay_trip_id'],
                 cmp_end=r['return_time'] + charge_time(
                     1.0 - r['energy_kwh'] / rt['E_use'], d.relay_batteries[1]))
            for r in q3_sol['relay_trips']]


def relay_outings(relays):
    """把中继记录按**出动**归并：一次出动 = 从 O01 出发到返回 O01 的完整行程。

    占用与工作量都以出动为单位：同一次出动的多次访问共享同一段机身占用与同一份
    能耗，逐行累加会把它们重复计入。归并键在 `build_relays` 里已经备好；返回的
    顺序按首次出现定序，不依赖字典迭代，保证可复现。
    """
    out, seen = [], set()
    for x in relays:
        k = x.get('outing') or x['id']
        if k in seen:
            continue
        seen.add(k)
        out.append(x)
    return out


def join_relays(q3_sol, relays):
    """
    外键连接（F03）：运输架次编号 → 服务它的中继架次集合，一律按 `trip_id` 查表。

    旧版 `trip_of = {int(s[1:]): i for i, s in enumerate(sorted(set(...)))}` 把
    失效表里出现过的编号按**排序位置**映射回架次下标：只要某架次没有失效区间，
    从它往后的映射就整体错位。这里按编号直接查，且**没有中继需求的运输架次其
    集合为空**，而不是从表里消失。未知编号一律显式报错。
    """
    relay_by_id = {x['id']: x for x in relays}
    trip_by_id = {t['trip_id']: t for t in q3_sol['transport_trips']}
    out = {t['trip_id']: [] for t in q3_sol['transport_trips']}
    for rec in q3_sol['intervals']:
        tid, rid = rec['trip_id'], rec['relay_trip_id']
        if tid not in trip_by_id:
            raise ValueError(f'失效区间 {rec["interval_id"]} 引用了未知运输架次 {tid}')
        if not rid:
            continue
        if rid not in relay_by_id:
            raise ValueError(f'失效区间 {rec["interval_id"]} 引用了未知中继架次 {rid}')
        if relay_by_id[rid] not in out[tid]:
            out[tid].append(relay_by_id[rid])
    return out


def color_intervals(intervals, tags, prefix):
    """
    区间着色：把 [(start, end)] 按开始时刻升序逐个分给**最早空闲**的资源，
    资源编号带组前缀（如 G1-U1），不跨组共用。

    这是「峰值重叠数 = 需要配备的资源数」这条口径的构造性证明：按开始时刻升序
    贪心分配，当且仅当所有已编号资源都在忙时才新开一个，故用到的编号数恰等于
    峰值重叠数。只报峰值而不给出一个真把每个区间落到具体编号上的分配，等于把
    「数量」当成「可实现」——两者中间差着这一步。

    返回 (分配表, 用到的资源数)。分配表每项 dict(资源编号, 起点, 终点, 标签)。
    """
    free, out = [], []
    for (s, e), tag in sorted(zip(intervals, tags), key=lambda p: (p[0][0], p[0][1], p[1])):
        hit = None
        for i, t_free in enumerate(free):
            if t_free <= s + 1e-9:
                hit = i
                break
        if hit is None:
            free.append(e)
            hit = len(free) - 1
        else:
            free[hit] = e
        out.append(dict(资源编号=f'{prefix}-{hit + 1}', 起点=s, 终点=e, 标签=tag))
    return out, len(free)


# ---------------------------------------------------------------------------
# 4. 组列集合划分 ILP（独立复核）
# ---------------------------------------------------------------------------
def solve_group_column_ilp(n_units, K, cost_of):
    """
    min Σ_g c_g z_g  s.t.  Σ_{g ∋ u} z_g = 1 ∀u（每个单元恰属一组）, Σ_g z_g = K。
    列 = 单元集合的每个非空子集（2^n−1 列）。与完全枚举走的是两条独立路径：
    枚举按「单元→组」的正向构造，ILP 按「组列」的集合划分松弛后取整，
    两者给出同一最优值才说明分区层的实现无误。
    """
    cols = []
    for mask in range(1, 1 << n_units):
        members = [u for u in range(n_units) if mask >> u & 1]
        cols.append((members, cost_of(members)))
    nc = len(cols)
    rows, ccols, vals = [], [], []
    for u in range(n_units):
        for c, (members, _) in enumerate(cols):
            if u in members:
                rows.append(u)
                ccols.append(c)
                vals.append(1.0)
    rows += [n_units] * nc
    ccols += list(range(nc))
    vals += [1.0] * nc
    A = sparse.csr_matrix((vals, (rows, ccols)), shape=(n_units + 1, nc))
    lb = np.array([1.0] * n_units + [float(K)])
    ub = np.array([1.0] * n_units + [float(K)])
    r = milp(np.array([c for _, c in cols]), constraints=LinearConstraint(A, lb, ub),
             integrality=np.ones(nc), bounds=Bounds(0, 1),
             options=dict(time_limit=60.0, presolve=True))
    if not r.success:
        return None
    picked = [cols[c][0] for c in range(nc) if r.x[c] > 0.5]
    return dict(obj=float(r.fun), groups=picked)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    d = load_data()
    d.geo_nodes, d.geo = precompute_geometry(d)
    rt = d.relay_type

    # 方案来源**唯一**：问题三落盘的完整方案（错峰后的运输时刻 + 中继绑定）。
    # 旧版走 `recommended(d)` 重新读问题二并重解调度，拿到的是**错峰前**的开始
    # 时刻，与问题三报出的联合方案不是同一个解——问题四的峰值并发、缺口、均衡度
    # 全都在算另一个方案。这里改为读问题三的方案，并先验 solution_id 与来源哈希。
    q3_sol = load_solution(Q3_SOL, stage='q3',
                           input_hashes_expect=input_hashes(),
                           solver_hashes_expect=solver_hashes(Q3_SOLVER_FILES),
                           solver_text_hashes_expect=solver_hashes_text(Q3_SOLVER_FILES))
    print(f'读取问题三方案 {q3_sol["solution_id"]}（来源 {q3_sol["source_solution_id"]}，'
          f'{len(q3_sol["transport_trips"])} 运输架次 / {len(q3_sol["relay_trips"])} 中继架次）')

    assignment = build_assignment(q3_sol, d)
    relays = build_relays(q3_sol, d)
    # 结果表的「一行 = 一次悬停站服务」不变，但一次出动可含多次访问：打印出两个数，
    # 免得把 10 次访问读成 10 架次（峰值并发与占用一律按出动计，见 relay_outings）。
    print('  其中中继出动 %d 次（%d 次悬停站服务）'
          % (len(relay_outings(relays)), len(relays)))
    relay_of_trip = join_relays(q3_sol, relays)

    units = atomic_units(d, assignment)
    n = len(units)
    print('=' * 74)
    print(f'运输架次 {len(assignment)} 个；原子任务单元 {n} 个')
    for u in units:
        print(f'   单元 {u}')

    # 库存（从原始数据推出，不写死）
    stock = {}
    for g in TYPES:
        stock[f'{g}_uav'] = sum(1 for u in d.uavs if u['type'] == g)
        stock[f'{g}_bat'] = int(d.batteries[g][0])
    stock['relay'] = len(d.relay_uavs)
    stock['relay_comp'] = int(d.relay_batteries[0])
    print('库存: ' + '  '.join(f'{RES_LABEL[k]}={stock[k]}' for k in RES_KEYS))

    # 单元 → 该单元的架次 / 中继架次
    unit_trips, unit_relays = [], []
    for u in units:
        us = set(u)
        tids = [a['idx'] for a in assignment if set(a['route']) <= us]
        unit_trips.append(tids)
        rr, seen = [], set()
        for k in tids:
            for x in relay_of_trip[assignment[k]['trip_id']]:
                if x['id'] not in seen:
                    seen.add(x['id'])
                    rr.append(x)
        unit_relays.append(rr)

    cache = {}

    def eval_group(members):
        key = tuple(sorted(members))
        if key in cache:
            return cache[key]
        tids = [k for m in members for k in unit_trips[m]]
        rr, seen = [], set()
        for m in members:
            for x in unit_relays[m]:
                if x['id'] not in seen:
                    seen.add(x['id'])
                    rr.append(x)
        res = group_resources(tids, assignment, rr)
        w = workload(tids, assignment, rr)
        out = (res, w, tids, rr)
        cache[key] = out
        return out

    rows, summary, all_rows = [], [], []
    for K in (2, 3):
        best = None
        n_part = 0
        relay_tot, n_feas = [], 0
        # 逐类资源在**全部**分区上的最小需求。论文要说「某一类缺 K 架」时，必须先
        # 分清这是「推荐分区的性质」还是「任何分区都躲不掉」——只有后者才谈得上
        # 结构性缺口。别把推荐分区那一侧的数字写成定律，也别把推荐分区的需求当成
        # 结构性需求。本算例实测（见 _run_logs/q4_final.log 的「全部分区中的最小需求」
        # 一行）：
        #   K=2：八类的 min_tot 全部 <= 库存（中继 2/2、B 型 2/2），没有任何一类
        #        是结构性缺口；库存可行分区数为 0 纯粹是「各类的最小值落在**不同**
        #        分区上」的联合约束结果。
        #   K=3：同上，**八类的最小缺口同样全为 0**（中继无人机 min_tot=2 恰等于
        #        库存 2）。本行在阶段 12 之前曾是「中继 min_tot=3 > 库存 2」——那是
        #        按「每个非空组各要 1 架」的先验推的，已被交付数据证伪（S006 的三个
        #        架次全程直连、无失效区间，它单独成组时中继需求为 0）。
        # 两个 K 都不可行，且成因相同：联合约束，不是任何单类短缺。论文必须这么说。
        min_tot = {k: 10 ** 9 for k in RES_KEYS}
        for groups in partition_sets(units, K):
            n_part += 1
            res_g, w_g = [], []
            tot = {k: 0 for k in RES_KEYS}
            # 分组独立口径（计划 §4.4 的 Gap_kg）：拆成 K 个互不共享的队伍后，
            # 每一组都要各自按库存配齐，故缺口按组逐个算再求和。
            gap = {k: 0 for k in RES_KEYS}
            red = {k: 0 for k in RES_KEYS}
            for members in groups:
                res, w, _, _ = eval_group(members)
                res_g.append(res)
                w_g.append(w)
                for k in RES_KEYS:
                    tot[k] += res[k]
                    gap[k] += max(0, res[k] - stock[k])
                    red[k] += max(0, stock[k] - res[k])
            R = sum(tot.values())
            for k in RES_KEYS:
                if tot[k] < min_tot[k]:
                    min_tot[k] = tot[k]
            W = np.array(w_g, dtype=float)
            cvw = float(W.std(ddof=0) / W.mean()) if W.mean() > 0 else 0.0
            # 总量口径：全队共用同一批库存时，总需求有没有超出库存。这才是"这套
            # 分区方案到底能不能落地"的判据；分组口径只说明组内是否够用。
            short = {k: max(0, tot[k] - stock[k]) for k in RES_KEYS}
            relay_tot.append(tot['relay'])
            if not any(short.values()):
                n_feas += 1
            # 逐个分区留档：绘图要的是 R 在**全部**分区上的分布，只留最优值画不出
            # 「最优离其余解有多远」，也画不出「可行解是不是压根不存在」。
            all_rows.append(dict(K=K, 资源规模R=R, 工作量均衡CV_W=round(cvw, 6),
                                 中继无人机总需求=tot['relay'],
                                 总量缺口合计=sum(short.values())))
            cand = dict(K=K, groups=[list(g) for g in groups], res=res_g, tot=tot,
                        gap=gap, red=red, short=short, R=R, cvw=cvw, W=W)
            if best is None or (cand['R'], cand['cvw']) < (best['R'], best['cvw']):
                best = cand
        summary.append((K, n_part, best, min(relay_tot), n_feas, min_tot))

    # 独立复核：组列集合划分 ILP
    check = {}
    for K, n_part, best, _rm, _nf, _mt in summary:
        cost_of = lambda members: sum(eval_group(members)[0].values())   # noqa: E731
        ilp = solve_group_column_ilp(n, K, cost_of)
        check[K] = ilp

    alloc_rows = []
    for K, n_part, best, relay_min, n_feas, min_tot in summary:
        print('=' * 74)
        print(f'K={K}：枚举 {n_part} 个分区（理论 {_stirling(n, K)} 个），'
              f'最优资源规模 R={best["R"]}，工作量均衡 CV_W={best["cvw"]:.4f}')
        if check[K]:
            print(f'   [复核] 组列集合划分 ILP 最优 R={check[K]["obj"]:.0f} '
                  f'（与枚举{"一致" if abs(check[K]["obj"] - best["R"]) < 1e-6 else "不一致 ⚠"}）')
        for gi, members in enumerate(best['groups']):
            us = sorted(s for m in members for s in units[m])
            res = best['res'][gi]
            print(f'   组{gi+1}: {",".join(us)}')
            print(f'      工作量 {best["W"][gi]/3600:.2f} h | ' +
                  '  '.join(f'{RES_LABEL[k]}={res[k]}' for k in RES_KEYS if res[k] > 0))
            rows.append(dict(K=K, 任务组编号=gi + 1, 服务区列表=','.join(us),
                             **{RES_LABEL[k]: res[k] for k in RES_KEYS},
                             工作量h=round(best['W'][gi] / 3600.0, 4)))

            # 峰值需求必须**可实现**：给出一个真把每个占用区间落到具体资源编号上的
            # 着色分配（按开始时刻升序分给最早空闲资源，编号带组前缀，不跨组共用），
            # 并核对用到的编号数恰等于峰值重叠数。
            _res, _w, g_tids, g_rr = eval_group(members)
            gp = f'G{gi + 1}'
            specs = []
            for g in TYPES:
                kk = [k for k in g_tids if assignment[k]['type'] == g]
                specs.append((f'{g}_uav', gp + '-U' + g,
                              [(assignment[k]['start'], assignment[k]['end']) for k in kk],
                              [assignment[k]['trip_id'] for k in kk]))
                specs.append((f'{g}_bat', gp + '-B' + g,
                              [(assignment[k]['start'], assignment[k]['bat_end']) for k in kk],
                              [assignment[k]['trip_id'] for k in kk]))
            # 中继的两类资源同样按**出动**计：一次出动的多次访问共享一段机身占用
            # 与一份能耗，逐行画占用会画出重叠，与 `group_resources` 的峰值口径不符。
            g_ro = relay_outings(g_rr)
            specs.append(('relay', gp + '-R',
                          [(x['start'], x['return_t'] + x['turnover']) for x in g_ro],
                          [x['id'] for x in g_ro]))
            specs.append(('relay_comp', gp + '-RC',
                          [(x['start'], x['cmp_end']) for x in g_ro],
                          [x['id'] for x in g_ro]))
            for key, prefix, ivs, tags in specs:
                alloc, n_used = color_intervals(ivs, tags, prefix)
                if n_used != res[key]:
                    raise AssertionError(
                        '%s 组%d 的 %s：着色用掉 %d 个资源，峰值需求却报 %d'
                        % (prefix, gi + 1, key, n_used, res[key]))
                for r in alloc:
                    alloc_rows.append(dict(K=K, 任务组编号=gi + 1, 资源类型=RES_LABEL[key],
                                           资源编号=r['资源编号'], 占用起点s=round(r['起点'], T_DEC),
                                           占用终点s=round(r['终点'], T_DEC), 对象编号=r['标签']))
            print(f'      资源分配可实现：着色编号数与峰值需求逐项一致'
                  f'（{sum(1 for s in specs)} 类资源）')
        # 两种口径必须分开列：混在一起会写出「需求 3、库存 2、缺口 0」这种自相矛盾
        # 的行——那是分组口径把缺口按组各算一遍再求和的结果，掩盖了总量超配。
        # 全队**单份**库存：N[r] = Σ_g n[g,r]（各组的同类需求相加），再与库存 S[r]
        # 比，Gap[r] = max(0, N[r] − S[r])。绝不是「每组各发一整套库存再比」——
        # 那等于把库存放大 K 倍，缺口必然恒为 0，等于没查。
        print('   ---- 单份库存口径（N[r]=Σ_g n[g,r] 与库存 S[r] 比；这才是能否落地的判据）----')
        for k in RES_KEYS:
            print(f'     {RES_LABEL[k]}: N=Σ_g n[g,r] {best["tot"][k]:2d}  S {stock[k]:2d}  '
                  f'Gap=max(0,N−S) {best["short"][k]:2d}  冗余 {max(0, stock[k] - best["tot"][k]):2d}')
        print('   ---- 分组独立口径（计划 §4.4 的 Gap_kg：每组各自按库存配齐）----')
        print('     Σ_g Gap_kg = %d，Σ_g Red_kg = %d'
              % (sum(best['gap'].values()), sum(best['red'].values())))
        print(f'   ---- 结论 ----')
        print(f'     资源规模 R_{K} = {best["R"]}；工作量均衡 CV_W = {best["cvw"]:.4f}')
        print(f'     全 {n_part} 个分区中，总量口径不缺任何资源的仅 {n_feas} 个'
              f'（占 {100.0*n_feas/n_part:.0f}%）；各分区中继无人机总需求最小 {relay_min}'
              f'，库存仅 {stock["relay"]}')
        print('     ---- 全部分区中的最小需求（判「结构性缺口」用这一行，'
              '不是推荐分区的需求）----')
        for k in RES_KEYS:
            print(f'       {RES_LABEL[k]}: 最小需求 {min_tot[k]:2d}  库存 {stock[k]:2d}  '
                  f'最小缺口 {max(0, min_tot[k] - stock[k]):2d}')
        if any(best['short'].values()):
            lack = '、'.join(f'{RES_LABEL[k]}缺 {best["short"][k]}'
                            for k in RES_KEYS if best['short'][k] > 0)
            print(f'     ⚠ 本分区方案在总量口径下不可行：{lack}'
                  f'（分区把队伍拆成互不共享的 K 份，中继与电池随之重复配备）')

    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'q4_partition.csv'), index=False)
    pd.DataFrame(all_rows).to_csv(os.path.join(OUT, 'q4_all_partitions.csv'), index=False)
    pd.DataFrame(alloc_rows).to_csv(os.path.join(OUT, 'q4_resource_allocation.csv'),
                                    index=False)

    # 比较指标表
    cmp_rows = []
    for K, n_part, best, relay_min, n_feas, min_tot in summary:
        row = dict(K=K, 分区数=n_part, 资源规模R=best['R'],
                   工作量均衡CV_W=round(best['cvw'], 6),
                   总量缺口=sum(best['short'].values()),
                   总量冗余=sum(max(0, stock[k] - best['tot'][k]) for k in RES_KEYS),
                   分组缺口合计=sum(best['gap'].values()),
                   分组冗余合计=sum(best['red'].values()),
                   库存可行分区数=n_feas, 中继总需求最小=relay_min)
        for k in RES_KEYS:
            row['需求_' + RES_LABEL[k]] = best['tot'][k]
            row['库存_' + RES_LABEL[k]] = stock[k]
            row['总量缺口_' + RES_LABEL[k]] = best['short'][k]
            row['分组缺口_' + RES_LABEL[k]] = best['gap'][k]
            # 全部分区中的最小需求：区分「推荐分区的缺口」与「躲不掉的结构性缺口」
            row['最小需求_' + RES_LABEL[k]] = min_tot[k]
            row['最小缺口_' + RES_LABEL[k]] = max(0, min_tot[k] - stock[k])
        cmp_rows.append(row)
    pd.DataFrame(cmp_rows).to_csv(os.path.join(OUT, 'q4_comparison.csv'), index=False)

    # 各单元工作量（供绘图）
    pd.DataFrame([dict(单元编号=f'U{i+1:02d}', 服务区列表=','.join(u),
                       运输架次数=len(unit_trips[i]),
                       工作量h=round(sum(assignment[k]['duration'] for k in unit_trips[i]) / 3600.0, 4))
                  for i, u in enumerate(units)]
                 ).to_csv(os.path.join(OUT, 'q4_units.csv'), index=False)
    print('=' * 74)
    print('已导出 q4_partition.csv / q4_comparison.csv / q4_units.csv / '
          'q4_all_partitions.csv / q4_resource_allocation.csv')


def _stirling(n, K):
    """第二类斯特林数：n 个单元划入 K 个非空无序组的方案数（用于核对枚举规模）。"""
    S = [[0] * (K + 1) for _ in range(n + 1)]
    S[0][0] = 1
    for i in range(1, n + 1):
        for j in range(1, min(i, K) + 1):
            S[i][j] = S[i - 1][j - 1] + j * S[i - 1][j]
    return S[n][K]


if __name__ == '__main__':
    main()
