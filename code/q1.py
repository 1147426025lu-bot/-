# -*- coding: utf-8 -*-
# 本程序及代码是在人工智能工具辅助下完成的。
# 工具名称：DeepSeek；版本/型号：DeepSeek V4.1 Flash（API 模型名 deepseek-flash）；
# 开发机构：杭州深度求索人工智能基础技术研究有限公司；
# 版本发布日期：2026 年 9 月 10 日。

"""
问题一：单点往返运输能力与货箱组批方案

  1) 三种机型各服务区最大安全载荷（二分法）与可达性上界（闭式）
  2) 货箱组批：同质类多重集动态规划（精确最优）
       对照 1：首次适应降序（FFD）贪心装箱 —— 给出最优性间隙
       对照 2：对称破缺指派式整数规划 —— 独立复核最小架次数
  3) 字典序分层多目标（架次 -> 能耗 -> 作业时间）与帕累托前沿
  4) 返航安全余量敏感性：细扫 + 临界跳变点 rho* 二分定位

------------------------------------------------------------------ 三个关键事实

【事实 1（能量约束线性化）】机型 g 在服务区 i 的往返能耗 E(q) 关于载荷 q 单调递增，
故 E(q) <= (1-rho)E_use 等价于 q <= q_max(i,g)。把该上界并进载质量约束后，
组批问题中不再出现非线性项。见 max_safe_payload()。

【事实 2（装箱可精确求解）】各服务区的货箱按 (质量, 体积) 归类后只有不超过 4 类
（见 box_classes），同类货箱在全部约束与代价下完全可互换。因此不必在 2^n 个子集上枚举，
只需在"各类剩余件数"构成的多重集状态上做动态规划：状态数 = prod(n_c+1) <= 324，
可全局最优求解。见 dp_area()。

【事实 3（时间目标退化）】segment_time() 不含载荷项，故一架次的作业时间为
    T_j = [prep + hand_base + t_go + t_back] + [load_per_box + hand_per_box] * n_j，
其中 t_go/t_back 只由该服务区与机型的航段几何决定。对整区求和，因 sum_j n_j = N 为定值，
    T_total(k) = A * k + B,  A = prep + hand_base + t_go + t_back,  B = (load+hand)*N
即作业时间只是架次数的仿射函数，与货箱如何分配无关。故字典序第三层无自由度，
报出的时间为该仿射式取值（由 assert_time_affine() 数值复核）。

关于解耦性：问题一不跨服务区组批、也不调度实体无人机与共享电池，各服务区决策互不影响，
故"逐区字典序最优 => 全局字典序最优"（见 5.4 节引理）。
"""
import os
import sys
from functools import lru_cache

try:                                    # 中文控制台输出
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

from core import (T_DEC, E_DEC, load_data, segment_time, roundtrip_energy)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'results')
os.makedirs(OUT, exist_ok=True)

TYPES = ['A', 'B', 'C']
EPS = 1e-9
ILP_TIME_LIMIT = 60.0
# 多重集状态数上限。本数据集最大为 S001 的 3*3*4*9 = 324；
# 超过此值说明货箱异质性过强，DP 不再适用（此时应改用集合划分 ILP 并报告 MIP gap）。
STATE_CAP = 400000


# ---------------------------------------------------------------------------
# 1. 单点往返能力：最大安全载荷
# ---------------------------------------------------------------------------
def max_safe_payload(d, t, si):
    """机型 t 在服务区 si 的最大安全载荷(kg)：往返能耗<=(1-rho)E_use 的最大 q。

    E(q) 对 q 单调递增，故用二分法求根。满载已安全时直接返回 Q。
    """
    usable = (1 - t['rho']) * t['E_use']

    def energy(q):
        E, _, _ = roundtrip_energy(t, d.dem, d.O01, si, q)
        return E

    if energy(t['Q']) <= usable:
        return t['Q']
    lo, hi = 0.0, t['Q']
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if energy(mid) <= usable:
            lo = mid
        else:
            hi = mid
    return lo


def empty_reach_rho(d, t, si):
    """该服务区仍"能起飞"的 rho 上界（闭式）：空载往返能耗 E(0) 是 E(q) 的最小值，
    故 (1-rho)E_use >= E(0) 即 rho <= 1-E(0)/E_use。超过此值最大安全载荷降为 0。"""
    E0, _, _ = roundtrip_energy(t, d.dem, d.O01, si, 0.0)
    return 1.0 - E0 / t['E_use']


def feasible_rho_max(d, t, si_id):
    """该机型对该服务区仍"能把最大货箱运走"的 rho 上界（闭式）——真正的作业可行性界限。

    q_max(i,g) >= m_max(i)  <=>  E(m_max) <= (1-rho)E_use  <=>  rho <= 1 - E(m_max)/E_use。
    注意这比 empty_reach_rho 更紧：q_max 降到 0 以上但仍小于最大单箱质量时，
    该机型照样无法承运该区（最大那件货谁都装不下），此时服务区实际已不可作业。
    """
    mmax = max(b['mass'] for b in d.boxes_by_service[si_id])
    E, _, _ = roundtrip_energy(t, d.dem, d.O01, d.si[si_id], mmax)
    return 1.0 - E / t['E_use']


def scheme_rho_limit(d, profile='base'):
    """整个方案仍可行的 rho 上界（闭式）。

    服务区 i 可行 <=> 存在机型 g 使 rho <= rho_max(i,g)，即 rho <= max_g rho_max(i,g)。
    方案整体可行 <=> rho <= min_i max_g rho_max(i,g)。返回 (上界, 瓶颈服务区, 该区各机型上界)。
    """
    base = {tid: d.transport_types[tid]['rho'] for tid in TYPES}
    for tid in TYPES:
        d.transport_types[tid]['rho'] = 0.20
    lims, detail = [], {}
    for si_id in d.S:
        per = {tid: feasible_rho_max(d, d.transport_types[tid], si_id)
               for tid in TYPES}
        detail[si_id] = per
        lims.append((max(per.values()), si_id))
    for tid in TYPES:
        d.transport_types[tid]['rho'] = base[tid]
    lim, worst = min(lims)
    return lim, worst, detail[worst]


def trip_duration(d, t, si, n_boxes, go, back):
    """单次往返总作业时间(s)：准备 + 装载 + 去程 + 交接 + 返程。"""
    return (t['prep'] + t['load_per_box'] * n_boxes
            + segment_time(t, go) + segment_time(t, back)
            + t['hand_base'] + t['hand_per_box'] * n_boxes)


def time_affine_coeffs(d, t, si, n_total):
    """该服务区该机型的 T_total(k) = A*k + B 的两个系数（见 事实 3）。"""
    _, go, back = roundtrip_energy(t, d.dem, d.O01, si, 0.0)
    A = t['prep'] + t['hand_base'] + segment_time(t, go) + segment_time(t, back)
    B = (t['load_per_box'] + t['hand_per_box']) * n_total
    return A, B


def assert_time_affine(d, t, si, n_total, n_bins_list):
    """数值复核 事实 3：按架次逐个累加的时间与仿射式必须一致。

    n_bins_list 为若干"各架次箱数"划分，逐个与 A*k+B 比较，返回最大偏差(s)。
    """
    A, B = time_affine_coeffs(d, t, si, n_total)
    _, go, back = roundtrip_energy(t, d.dem, d.O01, si, 0.0)
    worst = 0.0
    for nb in n_bins_list:
        direct = sum(trip_duration(d, t, si, n, go, back) for n in nb)
        worst = max(worst, abs(direct - (A * len(nb) + B)))
    return worst


# ---------------------------------------------------------------------------
# 2. FFD 贪心装箱（对照基线）
# ---------------------------------------------------------------------------
def pack_boxes_ffd(boxes, qmax, vmax):
    """First-Fit-Decreasing 装箱：返回 list of dict(mass, vol, idxs)。"""
    idx = sorted(range(len(boxes)), key=lambda i: -boxes[i]['mass'])
    batches = []
    for i in idx:
        b = boxes[i]
        placed = False
        for bt in batches:
            if (bt['mass'] + b['mass'] <= qmax + EPS
                    and bt['vol'] + b['volume'] <= vmax + EPS):
                bt['mass'] += b['mass']; bt['vol'] += b['volume']; bt['idxs'].append(i)
                placed = True
                break
        if not placed:
            batches.append(dict(mass=b['mass'], vol=b['volume'], idxs=[i]))
    return batches


def service_batches_for_type(d, si_id, tid, qmax=None):
    """FFD 基线：该服务区在该机型下的组批列表、总架次、总能耗、总时间。"""
    t = d.transport_types[tid]
    si = d.si[si_id]
    boxes = d.boxes_by_service[si_id]
    if qmax is None:
        qmax = max_safe_payload(d, t, si)
    if qmax <= EPS:                       # 该机型对该服务区不可达
        return dict(type=tid, qmax=qmax, n_trips=0, total_E=0.0,
                    total_T=0.0, batches=[], reachable=False)
    batches = pack_boxes_ffd(boxes, qmax, t['volume'])
    total_E = total_T = 0.0
    for bt in batches:
        E, go, back = roundtrip_energy(t, d.dem, d.O01, si, bt['mass'])
        bt['E'] = E
        bt['t'] = trip_duration(d, t, si, len(bt['idxs']), go, back)
        bt['soc'] = 1.0 - E / t['E_use']
        total_E += bt['E']; total_T += bt['t']
    return dict(type=tid, qmax=qmax, n_trips=len(batches), total_E=total_E,
                total_T=total_T, batches=batches, reachable=True)


# ---------------------------------------------------------------------------
# 3. 货箱同质类与多重集状态（事实 2）
# ---------------------------------------------------------------------------
def box_classes(boxes):
    """把货箱按 (质量, 体积) 归并成同质类。

    返回 [(mass, vol, count, [原箱下标...]), ...]，按质量降序（利于模式枚举剪枝）。
    同类货箱在质量约束、体积约束、能耗、时间上完全等价，可互换。
    """
    groups = {}
    for i, b in enumerate(boxes):
        key = (round(b['mass'], 6), round(b['volume'], 6))
        groups.setdefault(key, []).append(i)
    out = [(k[0], k[1], len(v), list(v)) for k, v in groups.items()]
    out.sort(key=lambda x: (-x[0], -x[1]))
    return out


def enumerate_class_patterns(classes, qmax, vmax):
    """枚举全部可行的"架次模式"（各类取若干件的组合）。

    返回 [(counts, mass, vol), ...]，counts 为与 classes 对齐的取件数元组。
    质量/体积随取件数单调增，故超限即剪枝。
    """
    nc = len(classes)
    out = []

    def rec(i, counts, mass, vol):
        if i == nc:
            if any(counts):
                out.append((tuple(counts), mass, vol))
            return
        m, v, c = classes[i][0], classes[i][1], classes[i][2]
        for take in range(c + 1):
            nm = mass + take * m
            nv = vol + take * v
            if nm > qmax + EPS or nv > vmax + EPS:
                break
            rec(i + 1, counts + [take], nm, nv)

    rec(0, [], 0.0, 0.0)
    return out


def _dp_solve(classes, pats, E_pat):
    """多重集状态 DP 内核：返回 {state: {k: (最小能耗, 选中的模式下标)}} 的求解函数。

    状态 = 各类货箱剩余件数元组；决策 = 一个可行架次模式。
    为避免架次顺序造成重复计数，规定每步必须取走"当前剩余箱中类序号最小的一类"
    至少一件（标准规范形），故每个划分恰被枚举一次。
    """
    nc = len(classes)

    @lru_cache(maxsize=None)
    def solve(state):
        if not any(state):
            return {0: (0.0, None)}
        low = next(i for i in range(nc) if state[i] > 0)
        best = {}
        for pi, (counts, mass, vol) in enumerate(pats):
            if counts[low] == 0:
                continue                            # 规范形：必含最小类
            if any(counts[i] > state[i] for i in range(nc)):
                continue
            sub = solve(tuple(state[i] - counts[i] for i in range(nc)))
            for k, (e, _) in sub.items():
                ck, ce = k + 1, e + E_pat[pi]
                cur = best.get(ck)
                if cur is None or ce < cur[0] - 1e-12:
                    best[ck] = (ce, pi)
        return best

    return solve


def dp_area(d, si_id, tid, with_table=False):
    """该服务区该机型的精确最优组批（多重集状态动态规划）。

    返回 dict(front={k: 最小能耗}, classes, pats, E_pat, n_boxes, qmax)；
    不可达返回 None。with_table=True 时附带可回溯的状态表 dp['table']。
    """
    t = d.transport_types[tid]
    si = d.si[si_id]
    boxes = d.boxes_by_service[si_id]
    qmax = min(max_safe_payload(d, t, si), t['Q'])
    if qmax <= EPS:
        return None
    classes = box_classes(boxes)
    n_states = 1
    for c in classes:
        n_states *= (c[2] + 1)
    if n_states > STATE_CAP:
        raise RuntimeError('服务区 %s 多重集状态数 %d 超过上限，需改用集合划分 ILP'
                           % (si_id, n_states))

    pats = enumerate_class_patterns(classes, qmax, t['volume'])
    if not pats:
        return None
    E_pat = [roundtrip_energy(t, d.dem, d.O01, si, p[1])[0] for p in pats]
    solve = _dp_solve(classes, pats, E_pat)
    full = solve(tuple(c[2] for c in classes))
    if not full:
        return None                  # 无任何可行划分（单箱即超载），该机型对此区不可达
    front = {k: v[0] for k, v in full.items()}
    out = dict(front=front, classes=classes, pats=pats, E_pat=E_pat,
               n_boxes=len(boxes), qmax=qmax, n_states=n_states, n_patterns=len(pats))
    if with_table:
        out['table'] = solve
    return out


def reconstruct(dp, k):
    """按选定架次数 k 还原逐架次的"各类取件数"清单。"""
    state = tuple(c[2] for c in dp['classes'])
    pats = dp['pats']
    table = dp['table']
    out = []
    while any(state):
        e, pi = table(state)[k]
        counts = pats[pi][0]
        out.append((pi, counts))
        state = tuple(state[i] - counts[i] for i in range(len(state)))
        k -= 1
    return out


# ---------------------------------------------------------------------------
# 4. 独立复核 1：对称破缺指派式整数规划求最小架次数
# ---------------------------------------------------------------------------
def min_k_assignment_ilp(boxes, qmax, vmax, kmax):
    """把货箱指派到至多 kmax 个架次，最小化启用架次数（0-1 整数规划）。

    变量 y[i][j]（箱 i 放入架次 j）与 u[j]（架次 j 是否启用）。
    对称破缺：规定箱 i 只能进入编号 <= i 的架次，且 u[j] <= u[j-1]，
    从而每类等价指派只保留一个代表，消除架次编号置换对称。
    返回最小架次数；不可行返回 None。
    """
    n = len(boxes)
    m = np.array([b['mass'] for b in boxes], dtype=float)
    v = np.array([b['volume'] for b in boxes], dtype=float)
    ny = n * kmax
    nv = ny + kmax

    def Y(i, j):
        return i * kmax + j

    def U(j):
        return ny + j

    rows, lb, ub = [], [], []
    for i in range(n):                                     # 每箱恰好一次
        r = np.zeros(nv)
        for j in range(kmax):
            r[Y(i, j)] = 1.0
        rows.append(r); lb.append(1.0); ub.append(1.0)
    for j in range(kmax):                                  # 质量 / 体积 / 启用耦合
        r = np.zeros(nv)
        for i in range(n):
            r[Y(i, j)] = m[i]
        r[U(j)] = -qmax
        rows.append(r); lb.append(-np.inf); ub.append(0.0)
        r = np.zeros(nv)
        for i in range(n):
            r[Y(i, j)] = v[i]
        r[U(j)] = -vmax
        rows.append(r); lb.append(-np.inf); ub.append(0.0)
        for i in range(n):
            r = np.zeros(nv); r[Y(i, j)] = 1.0; r[U(j)] = -1.0
            rows.append(r); lb.append(-np.inf); ub.append(0.0)
    for j in range(1, kmax):                               # 架次按序启用
        r = np.zeros(nv); r[U(j)] = 1.0; r[U(j - 1)] = -1.0
        rows.append(r); lb.append(-np.inf); ub.append(0.0)
    for i in range(n):                                     # 箱 i 只进 j<=i 的架次
        for j in range(i + 1, kmax):
            r = np.zeros(nv); r[Y(i, j)] = 1.0
            rows.append(r); lb.append(0.0); ub.append(0.0)

    c = np.zeros(nv)
    for j in range(kmax):
        c[U(j)] = 1.0
    res = milp(c=c,
               constraints=LinearConstraint(csr_matrix(np.array(rows)),
                                            np.array(lb), np.array(ub)),
               integrality=np.ones(nv),
               bounds=Bounds(np.zeros(nv), np.ones(nv)),
               options=dict(time_limit=ILP_TIME_LIMIT))
    if not res.success:
        return None
    return int(round(res.fun))


# ---------------------------------------------------------------------------
# 5. 独立复核 2：集合划分 ILP 求固定架次数下的最小能耗
# ---------------------------------------------------------------------------
def energy_partition_ilp(d, si_id, tid, k, cap=20000):
    """枚举全部架次模式（原子集）并用集合划分 ILP 求恰好 k 架次下的最小总能耗。

    这是与 dp_area 完全独立的第二条求解路径（子集枚举 + 整数规划 vs 多重集 DP），
    仅在模式数不超过 cap 的服务区上运行，用于交叉验证。
    """
    t = d.transport_types[tid]
    si = d.si[si_id]
    boxes = d.boxes_by_service[si_id]
    n = len(boxes)
    qmax = min(max_safe_payload(d, t, si), t['Q'])
    if qmax <= EPS:
        return None
    m = [b['mass'] for b in boxes]
    v = [b['volume'] for b in boxes]
    size = 1 << n
    mass = [0.0] * size
    vol = [0.0] * size
    masks, E_of = [], []
    for mask in range(1, size):
        low = mask & (-mask)
        i = low.bit_length() - 1
        prev = mask ^ low
        mass[mask] = mass[prev] + m[i]
        vol[mask] = vol[prev] + v[i]
        if mass[mask] <= qmax + EPS and vol[mask] <= t['volume'] + EPS:
            if len(masks) >= cap:
                return None                     # 规模过大，交由 DP 处理
            masks.append(mask)
            E_of.append(roundtrip_energy(t, d.dem, d.O01, si, mass[mask])[0])
    ncol = len(masks)
    if ncol == 0:
        return None
    rows = [[float((mk >> b) & 1) for mk in masks] for b in range(n)]
    rows.append([1.0] * ncol)
    lbs = [1.0] * n + [float(k)]
    ubs = [1.0] * n + [float(k)]
    res = milp(c=np.array(E_of),
               constraints=LinearConstraint(csr_matrix(np.array(rows)),
                                            np.array(lbs), np.array(ubs)),
               integrality=np.ones(ncol),
               bounds=Bounds(0, 1),
               options=dict(time_limit=ILP_TIME_LIMIT))
    if not res.success:
        return None
    return float(res.fun)


# ---------------------------------------------------------------------------
# 6. 服务区选项与全局帕累托前沿
# ---------------------------------------------------------------------------
def area_options(d, si_id):
    """该服务区在三种机型（可混用）上的非支配选项：{k: [选项, ...]}。

    每个选项 = dict(k, E, T, type)。由于时间只依赖 k（事实 3），同 k 下只需比较能耗。
    """
    opts = {}
    for tid in TYPES:
        dp = dp_area(d, si_id, tid)
        if dp is None:
            continue
        A, B = time_affine_coeffs(d, d.transport_types[tid], d.si[si_id], dp['n_boxes'])
        for k, E in dp['front'].items():
            opts.setdefault(k, []).append(dict(k=k, E=E, T=A * k + B, type=tid))
    return opts


def aggregate_front(all_opts):
    """由各区选项合成全局 (架次数, 总能耗, 总时间) 帕累托前沿。

    各区解耦，逐区做 DP 式合并并按三目标支配关系剪枝，得到全局前沿。
    """
    states = [(0, 0.0, 0.0, [])]
    for si_id in sorted(all_opts):
        opts = all_opts[si_id]
        new = []
        for (K, E, T, pick) in states:
            for k in sorted(opts):
                for o in opts[k]:
                    new.append((K + k, E + o['E'], T + o['T'],
                                pick + [(si_id, k, o['type'])]))
        new.sort(key=lambda s: (s[0], round(s[1], 6), round(s[2], 3)))
        kept = []
        for s in new:
            if any(o[0] <= s[0] and o[1] <= s[1] + 1e-9 and o[2] <= s[2] + 1e-6
                   for o in kept):
                continue
            kept.append(s)
        states = kept
    return states


# ---------------------------------------------------------------------------
# 7. 主流程
# ---------------------------------------------------------------------------
def main():
    d = load_data()

    # ---------- 1) 最大安全载荷 ----------
    print('=' * 78)
    print('1) 三机型各服务区最大安全载荷 q_max (kg)')
    qmax_table = {tid: [round(max_safe_payload(d, d.transport_types[tid], d.si[s]), 2)
                        for s in d.S] for tid in TYPES}
    df_q = pd.DataFrame(qmax_table, index=d.S)
    df_q.index.name = '服务区'
    print(df_q.to_string())
    df_q.to_csv(os.path.join(OUT, 'q1_max_payload.csv'))

    # ---------- 2) 精确组批 + FFD 对照 + ILP 独立复核 ----------
    print('=' * 78)
    print('2) 各服务区各机型组批：多重集 DP（精确） vs FFD 贪心 vs 指派式 ILP 复核')
    summary, gap_rows = {}, []
    affine_worst = 0.0
    for si_id in d.S:
        summary[si_id] = {}
        for tid in TYPES:
            ffd = service_batches_for_type(d, si_id, tid)
            dp = dp_area(d, si_id, tid)
            if dp is None:
                summary[si_id][tid] = (0, 0.0, 0.0)
                gap_rows.append(dict(服务区=si_id, 机型=tid, 可达=0, FFD架次=0,
                                     DP架次=0, ILP架次=-1, FFD能耗kWh=0.0, DP能耗kWh=0.0))
                continue
            k_ex = min(dp['front'])
            E_ex = dp['front'][k_ex]
            A, B = time_affine_coeffs(d, d.transport_types[tid], d.si[si_id], dp['n_boxes'])
            summary[si_id][tid] = (k_ex, round(E_ex, 3), round(A * k_ex + B, 1))
            # 事实 3 数值复核：用该服务区的箱数做几种不同划分，逐一比对仿射式
            N = dp['n_boxes']
            splits = [[N // k_ex + (1 if i < N % k_ex else 0) for i in range(k_ex)]]
            if k_ex > 1:
                splits.append([N - k_ex + 1] + [1] * (k_ex - 1))
            affine_worst = max(affine_worst, assert_time_affine(
                d, d.transport_types[tid], d.si[si_id], N, splits))
            # 独立复核：对称破缺指派式 ILP
            qm = min(max_safe_payload(d, d.transport_types[tid], d.si[si_id]),
                     d.transport_types[tid]['Q'])
            ilp_k = min_k_assignment_ilp(d.boxes_by_service[si_id], qm,
                                         d.transport_types[tid]['volume'],
                                         kmax=min(len(d.boxes_by_service[si_id]), 12))
            gap_rows.append(dict(服务区=si_id, 机型=tid, 可达=1,
                                 FFD架次=ffd['n_trips'], DP架次=k_ex,
                                 ILP架次=(-1 if ilp_k is None else ilp_k),
                                 FFD能耗kWh=round(ffd['total_E'], 3),
                                 DP能耗kWh=round(E_ex, 3)))
    df_gap = pd.DataFrame(gap_rows)
    df_gap.to_csv(os.path.join(OUT, 'q1_ilp_gap.csv'), index=False)

    chk = df_gap[df_gap['可达'] == 1]
    bad = chk[(chk['ILP架次'] >= 0) & (chk['ILP架次'] != chk['DP架次'])]
    print('  DP 与指派式 ILP 的最小架次数一致: %d/%d%s'
          % (len(chk) - len(bad), len(chk),
             '' if len(bad) == 0 else '  !! 不一致: %s' % bad.to_dict('records')))
    print('  事实3 时间仿射式复核最大偏差: %.3e s' % affine_worst)
    nffd = int((chk['FFD架次'] > chk['DP架次']).sum())
    print('  FFD 非最优的 (服务区,机型) 组合: %d / %d' % (nffd, len(chk)))
    if nffd:
        print(chk[chk['FFD架次'] > chk['DP架次']]
              [['服务区', '机型', 'FFD架次', 'DP架次', 'FFD能耗kWh', 'DP能耗kWh']]
              .to_string(index=False))
    dE = float((chk['FFD能耗kWh'] - chk['DP能耗kWh']).sum())
    print('  FFD 相对精确解的总能耗冗余: %.3f kWh (%.2f%%)'
          % (dE, 100 * dE / chk['DP能耗kWh'].sum()))

    df_s = pd.DataFrame(index=d.S,
                        columns=[f'{t}_{m}' for t in TYPES for m in ('架次', '能耗', '时间')])
    for si_id in d.S:
        for tid in TYPES:
            n, e, tt = summary[si_id][tid]
            df_s.loc[si_id, f'{tid}_架次'] = n
            df_s.loc[si_id, f'{tid}_能耗'] = e
            df_s.loc[si_id, f'{tid}_时间'] = tt
    print('-' * 78)
    print(df_s.to_string())
    df_s.to_csv(os.path.join(OUT, 'q1_batching_summary.csv'))

    print('-' * 78)
    print('全场景汇总（精确最优）：')
    for tid in TYPES:
        n = sum(summary[s][tid][0] for s in d.S)
        e = sum(summary[s][tid][1] for s in d.S)
        tt = sum(summary[s][tid][2] for s in d.S)
        row = chk[chk['机型'] == tid]
        print('  机型%s: 架次 %d->%d  能耗 %.2f->%.2f kWh  时间 %.0f s (%.2f h)'
              % (tid, int(row['FFD架次'].sum()), n, float(row['FFD能耗kWh'].sum()), e,
                 tt, tt / 3600))

    # 第二条独立路径：集合划分 ILP 复核固定架次下的最小能耗
    print('-' * 78)
    print('  集合划分 ILP 能耗复核（模式数 <= 20000 的组合）：')
    nchk = nok = 0
    for si_id in d.S:
        for tid in TYPES:
            dp = dp_area(d, si_id, tid)
            if dp is None or dp['n_patterns'] > 20000:
                continue
            k_ex = min(dp['front'])
            E_ilp = energy_partition_ilp(d, si_id, tid, k_ex)
            if E_ilp is None:
                continue
            nchk += 1
            if abs(E_ilp - dp['front'][k_ex]) < 1e-6:
                nok += 1
            else:
                print('    !! %s %s k=%d  DP=%.6f  ILP=%.6f'
                      % (si_id, tid, k_ex, dp['front'][k_ex], E_ilp))
    print('    能耗最优值一致 %d / %d' % (nok, nchk))

    # ---------- 3) 逐区字典序最优方案 ----------
    print('=' * 78)
    print('3) 推荐组批方案：逐区字典序最优（架次 -> 能耗 -> 时间），机型可混用')
    all_opts = {si_id: area_options(d, si_id) for si_id in d.S}
    rec, rows = {}, []
    tot_k, tot_e, tot_t = 0, 0.0, 0.0
    for si_id in d.S:
        opts = all_opts[si_id]
        if not opts:
            continue
        k_best = min(opts)
        o = min(opts[k_best], key=lambda x: x['E'])       # 同架次下取能耗最小
        rec[si_id] = o['type']
        tot_k += o['k']; tot_e += o['E']; tot_t += o['T']
        dp = dp_area(d, si_id, o['type'], with_table=True)
        classes = dp['classes']
        bins = reconstruct(dp, k_best)
        boxes = d.boxes_by_service[si_id]
        pools = {ci: list(classes[ci][3]) for ci in range(len(classes))}
        tt = d.transport_types[o['type']]
        for bi, (pi, counts) in enumerate(bins):
            ids = []
            for ci, take in enumerate(counts):
                for _ in range(take):
                    ids.append(boxes[pools[ci].pop(0)]['id'])
            n_bin = sum(counts)
            t_one = (trip_duration(d, tt, d.si[si_id], n_bin, *roundtrip_energy(
                tt, d.dem, d.O01, d.si[si_id], dp['pats'][pi][1])[1:]))
            E_one = dp['E_pat'][pi]
            rows.append(dict(
                架次编号=f'{si_id}-{o["type"]}-{bi + 1}', 服务区编号=si_id,
                机型编号=o['type'], 货箱编号列表=';'.join(ids),
                总质量kg=round(dp['pats'][pi][1], 2),
                总体积m3=round(dp['pats'][pi][2], 4),
                往返时间s=round(t_one, T_DEC), 架次能耗kWh=round(E_one, E_DEC),
                返航SOC=round(1.0 - E_one / tt['E_use'], 6)))
    print('  服务区->机型:', rec)
    print('  推荐方案总计: 架次=%d, 能耗=%.2f kWh, 时间=%.2f h'
          % (tot_k, tot_e, tot_t / 3600))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'q1_recommended_batching.csv'), index=False)
    print('  已导出 q1_recommended_batching.csv，共 %d 架次' % len(rows))

    # ---------- 4) 帕累托前沿 ----------
    print('=' * 78)
    print('4) 帕累托前沿（架次数 - 总能耗 - 总作业时间）')
    states = aggregate_front(all_opts)
    par_rows, area_rows = [], []
    for (K, E, T, pick) in states:
        par_rows.append(dict(架次数=int(K), 总能耗kWh=round(E, 3),
                             总作业时间s=round(T, T_DEC), 总作业时间h=round(T / 3600, 3),
                             方案=';'.join('%s:%d%s' % (s, k, g) for s, k, g in pick)))
    df_par = pd.DataFrame(par_rows).sort_values(['架次数', '总能耗kWh'])
    df_par.to_csv(os.path.join(OUT, 'q1_pareto.csv'), index=False)
    for si_id in d.S:
        for k in sorted(all_opts[si_id]):
            for o in all_opts[si_id][k]:
                area_rows.append(dict(服务区=si_id, 机型=o['type'], 架次数=k,
                                      能耗kWh=round(o['E'], 3), 时间s=round(o['T'], 1)))
    pd.DataFrame(area_rows).to_csv(os.path.join(OUT, 'q1_pareto_area.csv'), index=False)
    print('  全局前沿点数 %d；架次数范围 %d--%d'
          % (len(df_par), int(df_par['架次数'].min()), int(df_par['架次数'].max())))
    print(df_par.head(10).to_string(index=False))

    # ---------- 5) 返航安全余量敏感性 ----------
    print('=' * 78)
    print('5) 返航安全余量敏感性（rho 细扫 + 临界跳变点 rho*）')
    rhos = [round(0.10 + 0.01 * i, 2) for i in range(31)]        # 0.10 ~ 0.40
    base = {tid: d.transport_types[tid]['rho'] for tid in TYPES}
    sens = {}
    for tid in TYPES:
        t = d.transport_types[tid]
        for rho in rhos:
            t['rho'] = rho
            sens[(tid, rho)] = [max_safe_payload(d, t, d.si[s]) for s in d.S]
        t['rho'] = base[tid]
    pd.DataFrame([dict(机型=tid, ρ=rho, 服务区=s, q_max=round(sens[(tid, rho)][j], 2))
                  for tid in TYPES for rho in rhos for j, s in enumerate(d.S)]
                 ).to_csv(os.path.join(OUT, 'q1_sensitivity.csv'), index=False)

    print('  机型   rho    平均q_max  最小q_max  最大q_max')
    for tid in TYPES:
        for rho in (0.10, 0.20, 0.30, 0.40):
            qs = sens[(tid, rho)]
            print('   %s    %.2f   %7.2f   %7.2f   %7.2f'
                  % (tid, rho, np.mean(qs), np.min(qs), np.max(qs)))
        print()

    cache = {}

    def total_trips(rho):
        """给定 rho 时字典序最优方案的总架次数；若有服务区三机型均不可作业则返回 None。

        不可作业 = 最大安全载荷低于该区最大单箱质量，即那件最大的货谁也装不下。
        此时方案整体不可行，绝不能把该区记作 0 架次（那会虚低总架次数）。
        """
        key = round(rho, 6)
        if key in cache:
            return cache[key]
        for tid in TYPES:
            d.transport_types[tid]['rho'] = rho
        tot = 0
        bad = []
        for si_id in d.S:
            k_best = None
            for tid in TYPES:
                dp = dp_area(d, si_id, tid)
                if dp is None:
                    continue
                kk = min(dp['front'])
                k_best = kk if k_best is None else min(k_best, kk)
            if k_best is None:
                bad.append(si_id)
            else:
                tot += k_best
        for tid in TYPES:
            d.transport_types[tid]['rho'] = base[tid]
        res = None if bad else tot
        cache[key] = res
        if bad:
            cache.setdefault('_bad', {})[key] = bad
        return res

    lim, worst_si, worst_per = scheme_rho_limit(d)
    print('  方案可行性上界（闭式）: rho <= %.4f' % lim)
    print('    瓶颈服务区 %s：%s' % (worst_si, {k: round(v, 4) for k, v in worst_per.items()}))

    grid = [(rho, total_trips(rho)) for rho in rhos]
    print('  rho-总架次阶梯（"x" = 方案不可行）：')
    print('   ', '  '.join('%.2f:%s' % (r, ('x' if n is None else n)) for r, n in grid))
    badmap = cache.get('_bad', {})
    if badmap:
        print('    失效区间示例: %s' % '; '.join(
            'rho=%.2f -> %s' % (r, ','.join(badmap[r])) for r in sorted(badmap)[:3]))

    crit = []
    for i in range(len(grid) - 1):
        (r1, n1), (r2, n2) = grid[i], grid[i + 1]
        if n1 is None or n2 is None or n1 == n2:
            continue                              # 只在实际可行的区间内找跳变
        lo, hi = r1, r2
        for _ in range(40):                       # 二分定位到 1e-4 以内
            mid = 0.5 * (lo + hi)
            if total_trips(mid) == n1:
                lo = mid
            else:
                hi = mid
        crit.append(dict(临界ρ=round(hi, 4), 跳变前架次=n1, 跳变后架次=n2, 服务区='全局'))
        print('   临界 rho* = %.4f：总架次 %d -> %d' % (hi, n1, n2))

    # 方案可行性上界：闭式与 DP 二分两条路径互相验证
    feas = [r for r, n in grid if n is not None]
    infeas = [r for r, n in grid if n is None]
    if feas and infeas:
        lo, hi = max(feas), min(infeas)
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if total_trips(mid) is not None:
                lo = mid
            else:
                hi = mid
        print('   方案失效临界 rho_crit（DP 二分）= %.4f ；闭式 = %.4f ；差 %.1e'
              % (hi, lim, abs(hi - lim)))
    else:
        print('   扫描区间内方案始终可行' if not infeas else '   扫描区间内方案始终不可行')
    crit.append(dict(临界ρ=round(hi, 4), 跳变前架次=-1, 跳变后架次=0,
                     服务区='方案失效(%s)' % worst_si))

    print('  各机型在各服务区的作业可行性上界 rho_max = 1 - E(m_max)/E_use：')
    for tid in TYPES:
        t = d.transport_types[tid]
        t['rho'] = 0.20
        vals = [(s, feasible_rho_max(d, t, s)) for s in d.S]
        t['rho'] = base[tid]
        w = min(vals, key=lambda x: x[1])
        print('   %s 型: 最紧 %s rho_max=%.4f （中位 %.4f）'
              % (tid, w[0], w[1], float(np.median([v for _, v in vals]))))
        crit.append(dict(临界ρ=round(w[1], 4), 跳变前架次=-1, 跳变后架次=0,
                         服务区='%s型/%s' % (tid, w[0])))
    pd.DataFrame(crit).to_csv(os.path.join(OUT, 'q1_critical_rho.csv'), index=False)

    # ---------- 6) 双因素协同扰动 ----------
    print('=' * 78)
    print('6) 双因素协同扰动敏感性（rho x 电池可用能量折减 kappa）')
    kappas = [1.00, 0.95, 0.90, 0.85, 0.80]
    g2 = {}
    for tid in TYPES:
        t = d.transport_types[tid]
        bE = t['E_use']
        for rho in rhos:
            for kap in kappas:
                t['rho'] = rho
                t['E_use'] = bE * kap
                g2[(tid, rho, kap)] = [max_safe_payload(d, t, d.si[s]) for s in d.S]
        t['rho'], t['E_use'] = base[tid], bE
    pd.DataFrame([dict(机型=tid, ρ=rho, κ=kap,
                       平均q_max=round(float(np.mean(g2[(tid, rho, kap)])), 2),
                       最小q_max=round(float(np.min(g2[(tid, rho, kap)])), 2))
                  for tid in TYPES for rho in rhos for kap in kappas]
                 ).to_csv(os.path.join(OUT, 'q1_sensitivity2d.csv'), index=False)

    print('   机型   基准(0.10,1.00)  仅rho->0.40  仅kap->0.80  两者同时   叠加预测   实际/叠加')
    for tid in TYPES:
        b0 = float(np.mean(g2[(tid, 0.10, 1.00)]))
        d_rho = b0 - float(np.mean(g2[(tid, 0.40, 1.00)]))
        d_kap = b0 - float(np.mean(g2[(tid, 0.10, 0.80)]))
        d_both = b0 - float(np.mean(g2[(tid, 0.40, 0.80)]))
        pred = d_rho + d_kap
        print('   %s      %8.2f      %8.2f    %8.2f    %8.2f   %8.2f   %6.2f'
              % (tid, b0, d_rho, d_kap, d_both, pred,
                 d_both / pred if pred else float('nan')))

    print('完成。结果已写入 results/ 目录。')


if __name__ == '__main__':
    main()
