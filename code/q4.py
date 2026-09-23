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
from q2 import precompute_geometry, recommended

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'results')
os.makedirs(OUT, exist_ok=True)

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
    relay_veh = [(x['start'], x['return_t']) for x in relay_of_group]
    relay_cmp = [(x['start'], x['cmp_end']) for x in relay_of_group]
    res['relay'] = peak_concurrency(relay_veh)
    res['relay_comp'] = peak_concurrency(relay_cmp)
    return res


def workload(trip_ids, trips, relay_of_group):
    """工作量 = 该组的运输飞行器时 + 中继飞行器时（架次占用总时长，秒）。"""
    w = sum(trips[k]['duration'] for k in trip_ids)
    w += sum(x['t_tot'] for x in relay_of_group)
    return w


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
    trips, assignment, q2_m, q2_order = recommended(d)
    assignment = sorted(assignment, key=lambda x: x['start'])
    for k, a in enumerate(assignment):
        a['end'] = a['start'] + a['duration']
        soc = 1.0 - a['E'] / d.transport_types[a['type']]['E_use']
        a['bat_end'] = a['end'] + charge_time(soc, d.batteries[a['type']][1])
        a['idx'] = k

    # 问题三的中继方案：直接读结果表，不再自己重跑一遍选站（消除双份真相源）
    iv = pd.read_csv(os.path.join(OUT, 'q3_intervals.csv'), encoding='utf-8-sig')
    rl = pd.read_csv(os.path.join(OUT, 'q3_relay_trips.csv'), encoding='utf-8-sig')
    rt = d.relay_type
    relays = []
    for _, r in rl.iterrows():
        soc_end = 1.0 - r['架次能耗kWh'] / rt['E_use']
        relays.append(dict(id=r['中继架次编号'], station=r['悬停站编号'],
                           start=float(r['开始时刻s']), return_t=float(r['返回O01时刻s']),
                           t_tot=float(r['返回O01时刻s']) - float(r['开始时刻s']),
                           cmp_end=float(r['返回O01时刻s']) + charge_time(soc_end, d.relay_batteries[1])))
    trip_of = {int(s[1:]): i for i, s in enumerate(sorted(set(iv['运输架次编号'])))}
    relay_by_id = {x['id']: x for x in relays}
    # 每个运输架次 → 服务它的中继架次集合
    relay_of_trip = {k: [] for k in range(len(assignment))}
    for _, r in iv.iterrows():
        rid = r['中继架次编号']
        if isinstance(rid, str) and rid.strip() and rid in relay_by_id:
            relay_of_trip[trip_of[int(r['运输架次编号'][1:])]].append(relay_by_id[rid])

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
            for x in relay_of_trip[k]:
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
        summary.append((K, n_part, best, min(relay_tot), n_feas))

    # 独立复核：组列集合划分 ILP
    check = {}
    for K, n_part, best, _rm, _nf in summary:
        cost_of = lambda members: sum(eval_group(members)[0].values())   # noqa: E731
        ilp = solve_group_column_ilp(n, K, cost_of)
        check[K] = ilp

    for K, n_part, best, relay_min, n_feas in summary:
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
        # 两种口径必须分开列：混在一起会写出「需求 3、库存 2、缺口 0」这种自相矛盾
        # 的行——那是分组口径把缺口按组各算一遍再求和的结果，掩盖了总量超配。
        print('   ---- 总量口径（全队共用库存；这才是能否落地的判据）----')
        for k in RES_KEYS:
            print(f'     {RES_LABEL[k]}: 总需求 {best["tot"][k]:2d}  库存 {stock[k]:2d}  '
                  f'缺口 {best["short"][k]:2d}  冗余 {max(0, stock[k] - best["tot"][k]):2d}')
        print('   ---- 分组独立口径（计划 §4.4 的 Gap_kg：每组各自按库存配齐）----')
        print('     Σ_g Gap_kg = %d，Σ_g Red_kg = %d'
              % (sum(best['gap'].values()), sum(best['red'].values())))
        print(f'   ---- 结论 ----')
        print(f'     资源规模 R_{K} = {best["R"]}；工作量均衡 CV_W = {best["cvw"]:.4f}')
        print(f'     全 {n_part} 个分区中，总量口径不缺任何资源的仅 {n_feas} 个'
              f'（占 {100.0*n_feas/n_part:.0f}%）；各分区中继无人机总需求最小 {relay_min}'
              f'，库存仅 {stock["relay"]}')
        if any(best['short'].values()):
            lack = '、'.join(f'{RES_LABEL[k]}缺 {best["short"][k]}'
                            for k in RES_KEYS if best['short'][k] > 0)
            print(f'     ⚠ 本分区方案在总量口径下不可行：{lack}'
                  f'（分区把队伍拆成互不共享的 K 份，中继与电池随之重复配备）')

    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'q4_partition.csv'), index=False)
    pd.DataFrame(all_rows).to_csv(os.path.join(OUT, 'q4_all_partitions.csv'), index=False)

    # 比较指标表
    cmp_rows = []
    for K, n_part, best, relay_min, n_feas in summary:
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
        cmp_rows.append(row)
    pd.DataFrame(cmp_rows).to_csv(os.path.join(OUT, 'q4_comparison.csv'), index=False)

    # 各单元工作量（供绘图）
    pd.DataFrame([dict(单元编号=f'U{i+1:02d}', 服务区列表=','.join(u),
                       运输架次数=len(unit_trips[i]),
                       工作量h=round(sum(assignment[k]['duration'] for k in unit_trips[i]) / 3600.0, 4))
                  for i, u in enumerate(units)]
                 ).to_csv(os.path.join(OUT, 'q4_units.csv'), index=False)
    print('=' * 74)
    print('已导出 q4_partition.csv / q4_comparison.csv / q4_units.csv / q4_all_partitions.csv')


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
