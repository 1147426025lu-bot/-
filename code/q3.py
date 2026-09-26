# -*- coding: utf-8 -*-
# 本程序及代码是在人工智能工具辅助下完成的。
# 工具名称：DeepSeek；版本/型号：DeepSeek V4.1 Flash（API 模型名 deepseek-flash）；
# 开发机构：杭州深度求索人工智能基础技术研究有限公司；
# 版本发布日期：2026 年 9 月 10 日。

"""
问题三：通信约束下的运输—中继联合调度（连续通信是硬约束）

题面把「连续通信」与货箱时限、载荷能量、资源可用性并列为约束，故本问的可接受
方案必须满足：运输无人机在爬升、巡航、下降、投送的**全过程**通信不中断
（N_outage = 0）。中断比例不是可以报告的成绩，而是必须消掉的违反。

模型分五层：
  1) 轨迹层：逐架次三维轨迹采样（搜索用 Δt=2 s，终检用 Δt=1 s），逐点判直连裕量
  2) 覆盖层：失效区间 j 与候选悬停站 r 的 0-1 覆盖关系 a[j][r]。
     候选站 = 平面网格 × 离地高度 {50,100,150,200,250,300} m，须同时对 G01 回传可视；
     「覆盖」的判据是区间内**全部**采样点都可接入且回传可用，不是抽几个点看看。
  3) 选站层：集合覆盖 MILP，两层字典序——先最少站数，再最小中继能耗。
     若按粗步长判定的覆盖在真密度下不成立，剔除该列重解（列生成式修复环）。
  4) 精化层：站址 (lon,lat,离地高度) 连续坐标下降，最大化所辖区间的最小链路裕量。
  5) 时序层：运输架次开始时刻进入决策（错峰）以削减同时中继需求峰值；
     中继排班保留「建链晚于失效区间结束即弃飞」规则，避免空飞顶迟后续架次。

如实说明的口径问题（旧版两处会虚高覆盖率，本版已改）：
  - 旧版把「落在任一正在服务的中继架次时间窗内」一概记为「中继覆盖」，**不检查
    该中继站此刻是否真的能连通这架运输机**。本版逐点检查该站对该点的接入裕量。
  - 旧版覆盖率按「采样点个数」统计，而轨迹在起降/巡航/悬停各段的采样密度不同，
    点数占比不等于时间占比。本版按采样点的时间权重做积分，得真正的**时间占比**。
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import os
import itertools
import time as _time
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy import sparse
from core import (T_DEC, E_DEC, load_data, ll_to_xy, segment_geometry, segment_time,
                  relay_flight_time, relay_flight_energy, relay_hover_energy, charge_time,
                  box_index, weighted_tardiness, shifted_tardiness)
from q2 import precompute_geometry, recommended, resource_usage
import q3_chain
from q3_chain import schedule_relays as _schedule_relays_impl
from solution_io import (build_q3_solution, save_solution, load_solution,
                         inherit_trip_ids, check_foreign_keys, input_hashes,
                         solver_hashes, solver_hashes_text, Q2_SOLVER_FILES)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'results')
os.makedirs(OUT, exist_ok=True)

Q2_SOL = os.path.join(OUT, 'q2_solution.json')
Q3_SOL = os.path.join(OUT, 'q3_solution.json')

DT_SEARCH = 2.0                       # 搜索阶段轨迹采样步长(s)
DT_AUDIT = 1.0                        # 终检步长(s)：计划要求 1–2 s 覆盖全部轨迹点
COVER_STRIDE = 5                      # 覆盖矩阵初判时每隔几个采样点核一次（≈10 s）
GRID_STEP = 1200.0                    # 候选悬停站平面网格间距(m)
HOVER_SET = [50.0, 100.0, 150.0, 200.0, 250.0, 300.0]   # 悬停离地高度候选(m)
MERGE_GAP = 900.0                     # 同一站相邻失效区间间隔≤此值即并成一个中继架次(s)
MAX_JOB = 3600.0                      # 单个中继架次的服务时长上限(s)，防悬停能耗越限
MARGIN_DB = 0.0                       # 覆盖判定所需最小链路裕量(dB)
MAX_REPAIR_ROUNDS = 8                 # 就绪性修复环的最大轮数
_V2_CHECKED = False                   # q3_v2 自检只跑一次（见 solve_relay 的并列路径）
M_PER_DEG_LAT = 111132.95


def _m_per_deg_lon(lat):
    return 111320.0 * np.cos(np.deg2rad(lat))


# ===========================================================================
# 1. 轨迹层
# ===========================================================================
def sample_trip_trajectory(d, t, route, boxes_at, t0=0.0, dt=DT_SEARCH):
    """采样运输架次三维轨迹 → list of (t_abs, lon, lat, alt_abs)。"""
    pts = []
    t_cur = t0
    alt_o = d.O01['alt']

    def node_ll(nid):
        return (d.O01['lon'], d.O01['lat']) if nid == 'O01' else (d.si[nid]['lon'], d.si[nid]['lat'])

    pts.append((t_cur, d.O01['lon'], d.O01['lat'], alt_o))
    t_cur += t['prep'] + t['load_per_box'] * sum(len(v) for v in boxes_at.values())
    pts.append((t_cur, d.O01['lon'], d.O01['lat'], alt_o))
    prev = 'O01'
    prev_alt = alt_o
    for sid in route:
        g = d.geo[(prev, sid)]
        s = d.si[sid]
        alt_s = s['alt'] + 30.0
        lon_a, lat_a = node_ll(prev)
        lon_b, lat_b = node_ll(sid)
        tc = g['climb'] / t['v_up']
        n1 = max(2, int(np.ceil(tc / dt)))
        for k in range(1, n1 + 1):
            fr = k / n1
            pts.append((t_cur + tc * fr, lon_a, lat_a, prev_alt + (g['cruise_alt'] - prev_alt) * fr))
        t_cur += tc
        tcr = g['d'] / t['v_cruise']
        n2 = max(2, int(np.ceil(tcr / dt)))
        for k in range(1, n2 + 1):
            fr = k / n2
            pts.append((t_cur + tcr * fr, lon_a + (lon_b - lon_a) * fr,
                        lat_a + (lat_b - lat_a) * fr, g['cruise_alt']))
        t_cur += tcr
        td = g['descent'] / t['v_down']
        n3 = max(2, int(np.ceil(td / dt)))
        for k in range(1, n3 + 1):
            fr = k / n3
            pts.append((t_cur + td * fr, lon_b, lat_b,
                        g['cruise_alt'] - (g['cruise_alt'] - alt_s) * fr))
        t_cur += td
        h = t['hand_base'] + t['hand_per_box'] * len(boxes_at[sid])
        # 投送期间飞行器定点悬停，链路状态恒定；首尾各记一点，使终检的时间积分
        # 在这段静置区间上不留空洞（旧版只在结束时记一点，等于蒙掉整段投送时长）。
        pts.append((t_cur, lon_b, lat_b, alt_s))
        pts.append((t_cur + h, lon_b, lat_b, alt_s))
        t_cur += h
        prev = sid
        prev_alt = alt_s
    g = d.geo[(prev, 'O01')]
    lon_a, lat_a = node_ll(prev)
    tc = g['climb'] / t['v_up']
    n1 = max(2, int(np.ceil(tc / dt)))
    for k in range(1, n1 + 1):
        fr = k / n1
        pts.append((t_cur + tc * fr, lon_a, lat_a, prev_alt + (g['cruise_alt'] - prev_alt) * fr))
    t_cur += tc
    tcr = g['d'] / t['v_cruise']
    n2 = max(2, int(np.ceil(tcr / dt)))
    for k in range(1, n2 + 1):
        fr = k / n2
        pts.append((t_cur + tcr * fr, lon_a + (d.O01['lon'] - lon_a) * fr,
                    lat_a + (d.O01['lat'] - lat_a) * fr, g['cruise_alt']))
    t_cur += tcr
    td = g['descent'] / t['v_down']
    n3 = max(2, int(np.ceil(td / dt)))
    for k in range(1, n3 + 1):
        fr = k / n3
        pts.append((t_cur + td * fr, d.O01['lon'], d.O01['lat'],
                    g['cruise_alt'] - (g['cruise_alt'] - alt_o) * fr))
    pts.append((t_cur + td, d.O01['lon'], d.O01['lat'], alt_o))
    return pts


def time_weights(pts):
    """
    每个采样点代表的时间长度(s)：w_i = (t_{i+1} − t_{i−1})/2（端点取半）。
    轨迹在爬升/巡航/下降/投送各段采样密度不同，用点数占比会系统性高估
    采样密（即耗时短）的那一段。通信比例必须按时间加权，才是题面口径。
    """
    ts = [p[0] for p in pts]
    n = len(ts)
    w = np.zeros(n)
    for i in range(n):
        if i == 0:
            w[i] = 0.5 * (ts[1] - ts[0]) if n > 1 else 0.0
        elif i == n - 1:
            w[i] = 0.5 * (ts[-1] - ts[-2])
        else:
            w[i] = 0.5 * (ts[i + 1] - ts[i - 1])
    return w


# ===========================================================================
# 2. 链路层
# ===========================================================================
def _gw(d):
    return (d.O01['lon'], d.O01['lat'], d.O01['alt'] + d.comm_params['gw_h'])


def direct_margin(d, lon, lat, alt):
    """运输机 → G01 直连的链路裕量(dB)：≥0 表示可用。"""
    ok, L, _ = d.comm.link_ok(d.dem, (lon, lat, alt), _gw(d), d.comm.Lmax_uw_gw)
    return d.comm.Lmax_uw_gw - L


def direct_ok(d, lon, lat, alt):
    return direct_margin(d, lon, lat, alt) >= MARGIN_DB


def backhaul_margin(d, st):
    """中继站 → G01 回传裕量(dB)。与运输机位置无关，故对每个候选站只算一次。"""
    ok, L, _ = d.comm.link_ok(d.dem, st, _gw(d), d.comm.Lmax_r_gw)
    return d.comm.Lmax_r_gw - L


def access_margin(d, st, lon, lat, alt):
    """运输机 → 中继站接入裕量(dB)。"""
    ok, L, _ = d.comm.link_ok(d.dem, (lon, lat, alt), st, d.comm.Lmax_uw_r)
    return d.comm.Lmax_uw_r - L


def station_covers(d, st, lon, lat, alt, margin=MARGIN_DB):
    """悬停站 st 能否同时接住运输机并回传 G01。"""
    if access_margin(d, st, lon, lat, alt) < margin:
        return False
    return backhaul_margin(d, st) >= margin


def relay_trip_cost(d, st, t_service):
    """中继无人机 O01→悬停站(服务 t_service)→O01 的时间与能耗。"""
    rt = d.relay_type
    alt_o = d.O01['alt']
    g_out = segment_geometry(d.dem, d.O01['lon'], d.O01['lat'], alt_o, st[0], st[1], st[2])
    g_back = segment_geometry(d.dem, st[0], st[1], st[2], d.O01['lon'], d.O01['lat'], alt_o)
    t_flight_out = relay_flight_time(rt, g_out)
    t_flight = t_flight_out + relay_flight_time(rt, g_back)
    E_flight = relay_flight_energy(rt, g_out) + relay_flight_energy(rt, g_back)
    E_hover = relay_hover_energy(rt, t_service)
    return dict(t_flight=t_flight, t_flight_out=t_flight_out, E=E_flight + E_hover,
                E_flight=E_flight, E_hover=E_hover, t_tot=rt['prep'] + rt['link'] + t_flight + t_service,
                ground_elev=float(d.dem.sample(st[0], st[1])))


# ===========================================================================
# 3. 失效区间
# ===========================================================================
def dead_intervals(d, a, dt=DT_SEARCH):
    """
    识别某运输架次的直连失效区间。区间向外各扩一个采样点：判定只保证「采样点上
    直连可用」，两采样点之间是否曾经失效无从得知，外扩后中继须覆盖到最后一个
    直连可用的采样点，属保守处理（覆盖范围更大，不会漏保）。
    """
    t = d.transport_types[a['type']]
    pts = sample_trip_trajectory(d, t, a['route'], a['boxes_at'], t0=a['start'], dt=dt)
    ok = [direct_margin(d, lon, lat, alt) >= MARGIN_DB for (_, lon, lat, alt) in pts]
    out = []
    i = 0
    n = len(pts)
    while i < n:
        if ok[i]:
            i += 1
            continue
        j = i
        while j < n and not ok[j]:
            j += 1
        lo = max(0, i - 1)
        hi = min(n - 1, j)          # j 是第一个恢复直连的点，纳入覆盖范围
        out.append(dict(t_start=pts[lo][0], t_end=pts[hi][0], pts=pts[lo:hi + 1]))
        i = j
    return pts, out


# ===========================================================================
# 4. 候选悬停站
# ===========================================================================
def build_candidates(d, intervals, step_m=GRID_STEP, hover_set=HOVER_SET, verbose=False):
    """
    平面网格 × 离地高度集合 → 候选悬停站。先按「对 G01 回传可视」过滤：
    该条件只与站址本身有关，与运输机无关，故每个候选只需算一次。
    """
    pts = [p for iv in intervals for p in iv['pts']]
    if not pts:
        return []
    lons = [p[1] for p in pts]
    lats = [p[2] for p in pts]
    lat_mid = 0.5 * (min(lats) + max(lats))
    dlon = step_m / _m_per_deg_lon(lat_mid)
    dlat = step_m / M_PER_DEG_LAT
    nlon = max(2, int(np.ceil((max(lons) - min(lons)) / dlon)) + 1)
    nlat = max(2, int(np.ceil((max(lats) - min(lats)) / dlat)) + 1)
    cands = []
    for glon in np.linspace(min(lons), max(lons), nlon):
        for glat in np.linspace(min(lats), max(lats), nlat):
            z = float(d.dem.sample(glon, glat))
            if not np.isfinite(z):
                continue
            for h in hover_set:
                st = (float(glon), float(glat), z + h)
                if backhaul_margin(d, st) >= MARGIN_DB:
                    cands.append(dict(lon=float(glon), lat=float(glat), ground=z,
                                      agl=h, alt_abs=z + h))
    if verbose:
        print(f'候选悬停站: 网格 {nlon}×{nlat} × 高度 {len(hover_set)} → {len(cands)} 个')
    return cands


def _as_st(c):
    return (c['lon'], c['lat'], c['alt_abs'])


# ===========================================================================
# 5. 覆盖矩阵 + 集合覆盖 MILP
# ===========================================================================
def _access_range(d):
    """无遮挡时接入链路的最大距离(m)。超出此距离的候选站不可能覆盖该点。"""
    c = d.comm
    d_km = 10 ** ((c.Lmax_uw_r - 32.45 - 20 * np.log10(c.p['f'])) / 20)
    return d_km * 1000.0


def cover_matrix(d, intervals, cands, stride=COVER_STRIDE, verbose=False):
    """
    a[j] = 能**完整覆盖**区间 j 的候选站下标集合。
    初判按 stride 抽点（控算力），随后 verify_pairs 会在真密度下复核并剔除
    误判的正例——初判只允许产生假正例，不允许假负例，否则会丢掉可行站。
    """
    if not cands:
        return [set() for _ in intervals]
    R = _access_range(d)
    cxy = np.array([ll_to_xy(c['lon'], c['lat']) for c in cands])
    cover = []
    for iv in intervals:
        sp = iv['pts'][::stride]
        if sp[-1] is not iv['pts'][-1]:
            sp = list(sp) + [iv['pts'][-1]]
        # 剪枝：区间内最远点到站的距离必须仍在接入距离内
        pxy = np.array([ll_to_xy(p[1], p[2]) for p in iv['pts']])
        cand_idx = set()
        for k, p in enumerate(sp):
            px, py = ll_to_xy(p[1], p[2])
            near = np.where((cxy[:, 0] - px) ** 2 + (cxy[:, 1] - py) ** 2 <= R * R)[0]
            if k == 0:
                cand_idx = set(int(x) for x in near)
            else:
                cand_idx &= set(int(x) for x in near)
            if not cand_idx:
                break
        good = set()
        for ci in cand_idx:
            st = _as_st(cands[ci])
            if all(access_margin(d, st, p[1], p[2], p[3]) >= MARGIN_DB
                   for p in iv['pts'][::stride]):
                good.add(ci)
        cover.append(good)
    if verbose:
        print(f'覆盖矩阵: {len(intervals)} 个失效区间，'
              f'可覆盖站数 min={min((len(c) for c in cover), default=0)} '
              f'max={max((len(c) for c in cover), default=0)}，'
              f'无站可覆盖={sum(1 for c in cover if not c)}')
    return cover


def verify_pairs(d, intervals, cands, pairs, dt=DT_SEARCH):
    """在真采样密度下复核 (区间, 站) 对，返回不成立的子集。"""
    bad = set()
    for (j, ci) in pairs:
        st = _as_st(cands[ci])
        iv = intervals[j]
        t = d.transport_types[iv['type']]
        a = iv['assign']
        pts = sample_trip_trajectory(d, t, a['route'], a['boxes_at'], t0=a['start'], dt=dt)
        lo, hi = iv['t_start'], iv['t_end']
        seg = [p for p in pts if lo - 1e-6 <= p[0] <= hi + 1e-6]
        if not seg:
            seg = iv['pts']
        if not all(access_margin(d, st, p[1], p[2], p[3]) >= MARGIN_DB for p in seg):
            bad.add((j, ci))
    return bad


_SEG_CACHE = {}      # (架次, 窗口, 起始, dt) → 区间内采样点的几何坐标
_PAIR_OK = {}        # (架次, 窗口, 站坐标, dt) → 该 (区间, 站) 对是否真密度成立


def _iv_segment(d, iv, dt):
    """失效区间在给定步长下的采样点几何（去掉时刻），按区间缓存。

    换站修复每轮要试上百个 (区间, 站) 对，而重采样一条完整轨迹是这个检查里最贵的
    动作；不缓存就等于把重采样乘以候选站数。
    """
    key = (iv['trip'], round(iv['t_start'], 6), round(iv['t_end'], 6),
           round(iv['assign']['start'], 6), dt)
    seg = _SEG_CACHE.get(key)
    if seg is None:
        a = iv['assign']
        t = d.transport_types[iv['type']]
        pts = sample_trip_trajectory(d, t, a['route'], a['boxes_at'], t0=a['start'], dt=dt)
        lo, hi = iv['t_start'], iv['t_end']
        seg = [(p[1], p[2], p[3]) for p in pts if lo - 1e-6 <= p[0] <= hi + 1e-6]
        if not seg:
            seg = [(p[1], p[2], p[3]) for p in iv['pts']]
        _SEG_CACHE[key] = seg
    return seg


def pair_covers(d, iv, cand, dt=DT_AUDIT):
    """(区间, 站) 对在**真采样密度**下是否成立（区间内每一点都够接入裕量）。

    `cover_matrix` 按 stride=5（≈10 s）抽点初判，只保证**抽点上**够裕量；选站 MILP
    那条路有列生成式修复环按 DT_SEARCH 复核，而**换站修复 / 同时占站修复**是直接按
    `cover[j]` 里的候选改派的，改派出来的对子从未复核过。实测 24 架次方案：换站修复
    把 T06 的区间改派到 S02，抽点上全部通过，逐点看最小裕量 −1.20 dB，终检留下 213 s
    真中断——正是这一处漏网。默认步长取 DT_AUDIT：1 s 的采样点集包含 2 s 的，故这一
    条比 `verify_pairs` 还严，改派路径用得起（有区间级缓存）。
    """
    key = (iv['trip'], round(iv['t_start'], 6), round(iv['t_end'], 6),
           cand['lon'], cand['lat'], cand['alt_abs'], dt)
    hit = _PAIR_OK.get(key)
    if hit is None:
        st = _as_st(cand)
        hit = all(access_margin(d, st, p[0], p[1], p[2]) >= MARGIN_DB
                  for p in _iv_segment(d, iv, dt))
        _PAIR_OK[key] = hit
    return hit


def sel_pairs_ok(d, intervals, cands, sel_new, sel_old, dt=DT_AUDIT):
    """`sel_new` 相对 `sel_old` **改动过**的那些 (区间, 站) 对是否都过真密度复核。

    只查改动过的对：没动的对子是上一轮已经查过的，重复查一遍等于把缓存白建。
    """
    for j, ci in sel_new.items():
        if sel_old.get(j) == ci:
            continue
        if not pair_covers(d, intervals[j], cands[ci], dt=dt):
            return False
    return True


def solve_setcover(n_iv, cover, ecost, n_cand, time_limit=120.0):
    """
    集合覆盖 MILP，两层字典序：
      第一层 min Σ y_r           —— 最少悬停站数
      第二层 min Σ e_{jr} x_{jr} —— 给定站数下最小中继能耗
    约束：Σ_r x_{jr} = 1（每个失效区间必须被保障）；x_{jr} ≤ y_r（站没选就不能用）；
          Σ_r y_r ≤ K*（第二层的站数上界，来自第一层最优值）。
    返回 (pairs, x_sel, K, status)；无可行的区间集合会原样反映在 x_sel 中（缺失即无站可覆盖）。
    """
    pairs = [(j, ci) for j in range(n_iv) for ci in sorted(cover[j])]
    if not pairs:
        return [], {}, None, 'no-columns'
    pidx = {p: k for k, p in enumerate(pairs)}
    np_ = len(pairs)
    nv = n_cand + np_

    rows, cols, vals = [], [], []
    r = 0
    # 第一层：Σ_{r ∈ cover[j]} y_r ≥ 1
    for j in range(n_iv):
        for ci in sorted(cover[j]):
            rows.append(r); cols.append(ci); vals.append(1.0)
        r += 1
    A1 = sparse.csr_matrix((vals, (rows, cols)), shape=(r, nv))
    res1 = milp(np.concatenate([np.ones(n_cand), np.zeros(np_)]),
                constraints=LinearConstraint(A1, np.ones(r), np.full(r, np.inf)),
                integrality=np.concatenate([np.ones(n_cand), np.zeros(np_)]),
                bounds=Bounds(0, 1),
                options=dict(time_limit=time_limit, presolve=True))
    if not res1.success:
        return pairs, {}, None, 'infeasible:' + str(res1.message)
    K = int(round(res1.fun))

    # 第二层：固定站数上界，最小化指派能耗
    cost = np.zeros(nv)
    for (j, ci), k in pidx.items():
        cost[n_cand + k] = ecost[j][ci]
    rows, cols, vals, lb, ub = [], [], [], [], []
    r = 0
    for j in range(n_iv):
        for ci in sorted(cover[j]):
            rows.append(r); cols.append(n_cand + pidx[(j, ci)]); vals.append(1.0)
        lb.append(1.0); ub.append(1.0); r += 1
    for (j, ci), k in pidx.items():
        rows.append(r); cols.append(n_cand + k); vals.append(1.0)
        rows.append(r); cols.append(ci); vals.append(-1.0)
        lb.append(-np.inf); ub.append(0.0); r += 1
    for ci in range(n_cand):
        rows.append(r); cols.append(ci); vals.append(1.0)
    lb.append(-np.inf); ub.append(float(K)); r += 1

    A2 = sparse.csr_matrix((vals, (rows, cols)), shape=(r, nv))
    integrality = np.zeros(nv)
    integrality[:n_cand] = 1
    integrality[n_cand:] = 1
    res2 = milp(cost, constraints=LinearConstraint(A2, np.array(lb), np.array(ub)),
                integrality=integrality, bounds=Bounds(0, 1),
                options=dict(time_limit=time_limit, presolve=True))
    if not res2.success:
        return pairs, {}, K, 'phase2:' + str(res2.message)
    x = np.round(res2.x).astype(int)
    sel = {}
    for (j, ci), k in pidx.items():
        if x[n_cand + k] > 0.5:
            sel[j] = ci
    used = int(sum(1 for ci in range(n_cand) if x[ci] > 0.5))
    return pairs, sel, used, 'ok'


# ===========================================================================
# 6. 站址连续精化
# ===========================================================================
def refine_station(d, cand, pts, steps=(0.0009, 0.0003, 0.0001), hstep=(25.0, 10.0, 5.0)):
    """
    坐标下降：在 (经度, 纬度, 离地高度) 上最大化所辖区间所有采样点的最小接入裕量，
    同时保持回传可用与悬停高度不超上限。离散网格给的站址是粗糙的，这一步把
    「覆盖能力 vs 高度代价」的权衡在连续空间里做实。
    """
    best = dict(cand)
    best_m = min(access_margin(d, _as_st(best), p[1], p[2], p[3]) for p in pts) \
        if pts else -np.inf
    for ds, hs in zip(steps, hstep):
        improved = True
        while improved:
            improved = False
            for dlon, dlat, dh in ((ds, 0, 0), (-ds, 0, 0), (0, ds, 0), (0, -ds, 0),
                                   (0, 0, hs), (0, 0, -hs)):
                cand2 = dict(best)
                cand2['lon'] += dlon
                cand2['lat'] += dlat
                cand2['agl'] = float(np.clip(best['agl'] + dh, 10.0, d.relay_type['max_hover_alt']))
                z = float(d.dem.sample(cand2['lon'], cand2['lat']))
                if not np.isfinite(z):
                    continue
                cand2['ground'] = z
                cand2['alt_abs'] = z + cand2['agl']
                st2 = _as_st(cand2)
                if backhaul_margin(d, st2) < MARGIN_DB:
                    continue
                m2 = min(access_margin(d, st2, p[1], p[2], p[3]) for p in pts)
                if m2 > best_m + 1e-9:
                    best, best_m = cand2, m2
                    improved = True
    return best, best_m


# ===========================================================================
# 7. 峰值需求 / 错峰
# ===========================================================================
def peak_station_demand(win_by_station):
    """
    max_t #{站 r : t 落在 r 所辖的某个失效区间内}。
    同一站上的多个区间可以由同一架中继同时保障（一架中继悬停在一点即可），
    故峰值按**不同的站**计数，而不是按区间计数——按区间计会高估所需中继数。
    """
    ev = []
    for ci, wins in win_by_station.items():
        for (t1, t2) in wins:
            ev.append((t1, 1, ci))
            ev.append((t2, -1, ci))
    ev.sort(key=lambda x: (x[0], x[1]))
    cur = {}
    peak = 0
    for (_, dk, ci) in ev:
        cur[ci] = cur.get(ci, 0) + dk
        peak = max(peak, sum(1 for v in cur.values() if v > 0))
    return peak


def min_stations_exact(cover, active, cache=None):
    """
    覆盖区间集合 `active` 所需的**最少候选站数**（精确，分支定界）。

    `active` 是同一时刻全部处于失效中的区间下标。中继机悬停在一点只能充当一个
    站，故这个数就是该时刻的**中继机数下界**——它与具体选站方案无关，只由几何
    决定。先按可覆盖站数升序排，分支时优先钉死最难的那个区间，规模很小（同刻
    最多 5 段）故不必上 ILP。
    """
    key = frozenset(active)
    if cache is not None and key in cache:
        return cache[key]
    order = sorted(active, key=lambda j: len(cover[j]))
    best = [len(order) + 1]

    def rec(i, chosen):
        if len(chosen) >= best[0]:
            return
        if i == len(order):
            best[0] = len(chosen)
            return
        j = order[i]
        if cover[j] & chosen:
            rec(i + 1, chosen)
            return
        for c in cover[j]:
            rec(i + 1, chosen | {c})

    rec(0, frozenset())
    if cache is not None:
        cache[key] = best[0]
    return best[0]


def simultaneous_station_excess(win_by_iv, cover, n_relay, cache=None):
    """
    按时刻求「同时**必须**占用的悬停站数」超出中继机数的**超额站·秒**。

    与 `peak_station_demand` 的关键区别：这里数的是**最少站数下界**
    （`min_stations_exact`，在全部候选站上精确求解），而不是「当前选站方案里同时
    被用到的站数」。后者只是某一个选站方案的性质，会把**并不存在**的峰值算进去：
    本算例 t≈1027 s 处 4 段区间同时失效，现有选站方案占了 3 个站，但同一时刻只用
    2 个站就能全覆盖。按方案计数时这个 3 一直挂在目标上，削峰于是整轮都在推一堵
    推不动的墙——24 架次方案「峰值 3 → 3、0 个架次被推迟」就是这么来的。按下界
    计数，超额只出现在真正需要 3 个站的时刻（本算例 t∈(3550.7, 3655.0)），
    削峰的目标才与「2 架中继机能不能排得下」直接对应。

    返回 (超额站·秒, 最少站数峰值)。
    """
    ev = []
    for j, (t1, t2) in win_by_iv.items():
        ev.append((t1, 1, j))
        ev.append((t2, -1, j))
    ev.sort(key=lambda x: (x[0], x[1]))
    active = set()
    excess, peak, prev_t, i = 0.0, 0, None, 0
    while i < len(ev):
        t = ev[i][0]
        if prev_t is not None and active:
            m = min_stations_exact(cover, active, cache)
            peak = max(peak, m)
            if m > n_relay:
                excess += (t - prev_t) * (m - n_relay)
        while i < len(ev) and ev[i][0] == t:
            if ev[i][1] > 0:
                active.add(ev[i][2])
            else:
                active.discard(ev[i][2])
            i += 1
        prev_t = t
    return excess, peak


def _concurrency_violations(intervals, sel, n_relay):
    """
    列出「同一时刻被占用的**悬停站**数超过中继机数」的时段，按超出量降序。

    与 `simultaneous_station_excess` 的区别：那边问的是「几何上最少要几个站」，
    与选站方案无关；这里问的是「**当前这组站**在那一刻被占用了几个」——MILP 给的
    解未必就是最少站那组，而排班层只认当前这组站。
    """
    ev = []
    for j, iv in enumerate(intervals):
        ev.append((iv['t_start'], 1, j))
        ev.append((iv['t_end'], -1, j))
    ev.sort(key=lambda x: (x[0], x[1]))
    out = []
    active, prev, i = set(), None, 0
    while i < len(ev):
        t = ev[i][0]
        if prev is not None and active:
            used = {sel[j] for j in active if j in sel}
            if len(used) > n_relay:
                out.append(((t - prev) * (len(used) - n_relay), frozenset(active)))
        while i < len(ev) and ev[i][0] == t:
            if ev[i][1] > 0:
                active.add(ev[i][2])
            else:
                active.discard(ev[i][2])
            i += 1
        prev = t
    out.sort(key=lambda x: -x[0])
    return out


def enforce_concurrency(d, intervals, sel, cover, cands, n_relay,
                        max_rounds=12, verbose=False):
    """
    同时占站约束修复：把「某一时刻被占用的悬停站数 > 中继机数」压回中继机数以内。

    `solve_setcover` 只保证**覆盖**，不保证**同时占用**——它给出一组站，而某一时刻
    真正被占用几个站，取决于那一刻有哪些区间落在这些站上。实测：t≈1027 s 有 4 段
    区间同时失效，几何上 2 个站就能全覆盖，MILP 却选了 3 个（S01/S03/S04）。2 架
    中继机在物理上排不出 3 个站同时中继，排班层只能把其中一个架次整趟推到几小时
    之后——实测 T04 被推 7007.5 s，中断确实归零，代价是加权迟到 20 万：**可行但
    畸形**。根因就在这一步，不修它，后面的修复环只能拿推迟量去补。

    对每个超限时段，在该时段活跃区间的候选站并集里穷举 ≤n_relay 个站的组合，取
    「修复后总超限秒数最小、其次改动区间数最少」的那组，把活跃区间改派过去。组合
    规模很小（并集百来个站、2 站组合约一万），直接精确枚举。找不到可行组合的时段
    原样保留，并在返回值里如实体现，不静默当成已修好。

    返回 (新 sel, 修复前超限站·秒, 修复后超限站·秒)。
    """
    sel = dict(sel)
    viol0 = sum(w for w, _ in _concurrency_violations(intervals, sel, n_relay))
    viol_cur = viol0

    def _cost(j, ci):
        return relay_trip_cost(d, _as_st(cands[ci]),
                               intervals[j]['t_end'] - intervals[j]['t_start'])['E']

    for _ in range(max_rounds):
        viols = _concurrency_violations(intervals, sel, n_relay)
        if not viols:
            break
        _w, active = viols[0]
        act = sorted(active)
        hardest = min(act, key=lambda j: len(cover[j]))
        best = None         # (总超限秒数, 改动区间数, 新 sel)
        for c1 in sorted(cover[hardest]):
            rest = [j for j in act if c1 not in cover[j]]
            sets = [(c1,)] if not rest else []
            if rest and n_relay >= 2:
                inter = set(cover[rest[0]])
                for j in rest[1:]:
                    inter &= cover[j]
                sets = [(c1, c2) for c2 in sorted(inter) if c2 != c1]
            for S in sets:
                sel2 = dict(sel)
                for j in act:
                    opts = [c for c in S if c in cover[j]]
                    if not opts:
                        break
                    sel2[j] = min(opts, key=lambda c: _cost(j, c))
                else:
                    # 改派出来的新站必须过真密度复核：`cover[j]` 只是 stride=5 的抽点
                    # 初判，抽点上够裕量不等于区间内每一点都够（见 pair_covers）。
                    if not sel_pairs_ok(d, intervals, cands, sel2, sel):
                        continue
                    v = sum(w for w, _ in _concurrency_violations(intervals, sel2, n_relay))
                    chg = sum(1 for j in act if sel2[j] != sel[j])
                    key = (v, chg)
                    if best is None or key < best[0]:
                        best = (key, sel2)
        if best is None or best[0][0] >= viol_cur - 1e-9:
            break
        sel = best[1]
        viol_cur = best[0][0]
    if verbose and viol0 > 1e-9:
        print(f'  同时占站修复：超限 {viol0:.1f} → {viol_cur:.1f} 站·秒'
              f'（中继机 {n_relay} 架）{"，已压到约束内" if viol_cur <= 1e-9 else "，仍有超限"}')
    return sel, viol0, viol_cur


def trip_slack(d, assignment):
    """每个架次在不违反**硬时限**前提下可向后推迟的最大秒数。"""
    slack = []
    for a in assignment:
        s = np.inf
        for sid, bs in a['boxes_at'].items():
            for b in bs:
                lim = []
                if b['category'] == '医疗物资' and b['expect'] is not None:
                    lim.append(b['expect'])
                if b['first_batch'] and b['deadline'] is not None:
                    lim.append(b['deadline'])
                if lim:
                    s = min(s, min(lim) - a['deliver_abs'][b['id']])
        slack.append(max(0.0, float(s)) if np.isfinite(s) else np.inf)
    return slack


def shift_assignment(d, assignment, deltas):
    """
    按 deltas 平移各架次开始时刻，同步平移交付时刻；机型/指派/能耗不变。

    **一律返回新对象并复制可变成员**（route / boxes_at / deliver_abs）。旧版在
    位移量为 0 时直接把原对象塞进结果里，于是「新排班」与 `base_asg` 共享同一批
    嵌套字典——调用方一旦就地改其中一个字段（错峰、换站修复里都很自然），基线
    就被悄悄改掉，后续所有以基线为参照的差值全部失真。
    """
    out = []
    for a, dl in zip(assignment, deltas):
        b = dict(a)
        b['route'] = list(a['route'])
        b['boxes_at'] = {s: list(bs) for s, bs in a['boxes_at'].items()}
        b['deliver_abs'] = {k: v + dl for k, v in a['deliver_abs'].items()}
        if abs(dl) >= 1e-9:
            b['start'] = a['start'] + dl
        out.append(b)
    return out


def asg_digest(assignment):
    """排班的轻量摘要（开始时刻 / 时长 / 交付时刻），用于证明基线未被就地改动。"""
    return tuple((round(a['start'], 9), round(a['duration'], 9),
                  tuple(sorted((k, round(v, 9)) for k, v in a['deliver_abs'].items())))
                 for a in assignment)


def _apply_shift(intervals, assignment):
    """
    把失效区间随所属运输架次整体平移。逐次可调用：`iv['assign']` 记的是上一次
    已施加到的状态，故增量式平移等价于从原始状态一次算到位。
    轨迹形状不变、失效与否由**几何**决定，因此整段平移不会改变区间本身。
    """
    for iv in intervals:
        d0 = assignment[iv['trip']]['start'] - iv['assign']['start']
        if abs(d0) < 1e-9:
            continue
        iv['t_start'] += d0
        iv['t_end'] += d0
        iv['pts'] = [(p[0] + d0, p[1], p[2], p[3]) for p in iv['pts']]
        iv['assign'] = assignment[iv['trip']]


def chain_conflicts(d, assignment):
    """
    同资源链上的占用冲突，逐条带出**释放时刻**与**所需推迟量**。

    释放时刻必须按资源类型分别取，不能一律用 `start + duration`：
      · 运输无人机 —— 返航即释放；
      · 共享电池   —— 还要加上把返航 SOC 充回满电的充电时间（与 q2 的调度器
                      同一口径：直接复用 resource_usage 的区间右端，不另算一份）。
    旧版把两者的释放时刻都当成 `start + duration`，于是电池链上的冲突算出的
    `required_delay` 偏小；更糟的是当冲突**只**来自电池时该值可能 ≤ 0，
    cascade_shift 直接 `continue`，把「本来推得开」报成「推不动」。故这里把
    资源编号、释放时刻、下一架次起点、所需推迟量一并给出，调用方不再猜类型。
    """
    bad = []
    for key, ivs in resource_usage(d, assignment).items():
        for x, y in zip(ivs, ivs[1:]):
            if y[0] < x[1] - 1e-6:
                bad.append(dict(resource=key, prev=x[2], nxt=y[2],
                                resource_release=x[1], next_start=y[0],
                                required_delay=x[1] - y[0]))
    return bad


def _propagate(d, base_asg, deltas, slack, max_iter=400):
    """
    从给定推迟量出发，沿资源链把冲突**只加不减**地推平。

    单架次后移常常不是被硬时限挡住，而是被「同一条无人机或共享电池链上的下一
    架次」顶住：本算例里 U08 的链是 T03→T14→T18、U06 的链是
    T04→T12→T16→T19，都首尾相接，动一个就必须连锁后移。故这里逐次只推「被压
    住的那一架次」，推完重查，直到无冲突、或再无可推空间。

    「推不动」有两种，必须分清：下游那一架次自己的硬时限 slack 已经顶死（本算例
    T24 只有 1344.5 s 余量），于是无论怎么推都会留下残余重叠。此时**如实**返回
    ok=False 并把当前（仍冲突的）推迟量一并交出，由调用方决定是放弃这条动作还是
    改走别的路；绝不在返回值里假装已无冲突。

    返回 (deltas, assignment, 是否已无冲突)。
    """
    deltas = list(deltas)
    asg = shift_assignment(d, base_asg, deltas)
    for _ in range(max_iter):
        conf = chain_conflicts(d, asg)
        if not conf:
            return deltas, asg, True
        applied = False
        for c in conf:
            j, need = c['nxt'], c['required_delay']
            if need <= 1e-9 or deltas[j] + need > slack[j] + 1e-9:
                continue
            deltas[j] += need
            applied = True
        if not applied:
            return deltas, asg, False
        asg = shift_assignment(d, base_asg, deltas)
    return deltas, asg, False


def cascade_shift(d, base_asg, deltas, req, slack, max_iter=400):
    """
    在资源链上**能推多少推多少**地落定一组推迟量。

    与 `_propagate` 的分工：`_propagate` 只会把冲突往后推平、对「推不动的」如实
    报 False；本函数负责在推不动时**回退**——把请求的推迟量收小到链还能承受的
    最大值，而不是「要么全额、要么一点不动」。

    为什么必须回退（本项目实测踩过、代价是一个不可行解）：

      T17 与 T24 同属 U05 且**首尾相接**（T17 返航时刻 = T24 起飞时刻），T24 的
      硬时限 slack 只有 1344.5 s。而 T17 自身没有硬时限（slack = ∞），于是就绪性
      修复给它要了 1173.3 s 的后移。旧版只在**循环里**检查下游 slack，请求值本身
      不设链上上限，于是 T17 被推到 1583.0 s——比 T24 能承受的多 238.5 s。推不动
      就留着重叠返回，而调用方 `repair_readiness` 对 ok=False 照单全收：终检只查
      通信缺口、不查资源链，于是一架 U05 同时飞 T17 与 T24（重叠 1583 s）的方案
      被当作「零中断」交付。这不是精度问题，是硬约束被静默违反。

    回退用二分：可行集对增量单调（链上约束全是「下游不早于上游释放」），故
    「增量 d 可行 ⇒ 任何 d' < d 可行」，可直接取最大的可行增量。δ=0 恒可行前提是
    进来的 deltas 本身不冲突，故先做一次 `_propagate` 把关；那一步就不通过说明这
    条动作从一开始就不可行，原样回 ok=False，不返回任何仍冲突的状态。

    返回 (deltas, assignment, 是否可行且已无冲突)。
    """
    base_digest = asg_digest(base_asg)
    cur, cur_asg, ok0 = _propagate(d, base_asg, deltas, slack, max_iter)
    if not ok0:
        assert asg_digest(base_asg) == base_digest, \
            '基线排班在本轮传播中被就地改动，所有推迟量都不可信'
        return list(deltas), shift_assignment(d, base_asg, deltas), False
    for tr, need in req.items():
        cap = max(cur[tr], slack[tr]) if np.isfinite(slack[tr]) else cur[tr] + need
        target = min(cur[tr] + need, cap)
        if target <= cur[tr] + 1e-9:
            continue
        trial = list(cur)
        trial[tr] = target
        t2, a2, ok = _propagate(d, base_asg, trial, slack, max_iter)
        if ok:
            # 常见情形：全额推得开，一次传播即可，不必二分。
            cur, cur_asg = t2, a2
            continue
        lo, hi = cur[tr], target
        best, best_asg = cur, cur_asg
        for _ in range(12):                 # 12 次二分把区间收到全长的 1/4096
            mid = 0.5 * (lo + hi)
            trial = list(cur)
            trial[tr] = mid
            t2, a2, ok = _propagate(d, base_asg, trial, slack, max_iter)
            if ok:
                best, best_asg = t2, a2
                lo = mid
            else:
                hi = mid
        cur, cur_asg = best, best_asg
    assert asg_digest(base_asg) == base_digest, \
        '基线排班在本轮传播中被就地改动，所有推迟量都不可信'
    return cur, cur_asg, True


def _snapshot_intervals(intervals):
    return [(iv['t_start'], iv['t_end'], list(iv['pts']), iv['assign']) for iv in intervals]


def _reset_intervals(intervals, snap):
    for iv, (t1, t2, pts, asg) in zip(intervals, snap):
        iv['t_start'], iv['t_end'], iv['pts'], iv['assign'] = t1, t2, pts, asg


def _outage_proxy(intervals, relays, skipped):
    """
    不跑终检的快速缺口代理，返回 (缺口秒数, 无绑定区间数)：中继架次实际盖住的只是
    [建链完成, 服务结束]，故区间未被盖住的早段、以及弃飞任务的全段，都算缺口。

    必须**同时**返回无绑定区间数：只看秒数会被「把区间整个丢掉」这种退化动作骗到
    ——一个 621 s 的区间，迟建链缺口算出来是 831 s，丢掉之后反而只剩 621 s，代理
    量会把它当成改进而接受，结果换来一个违反硬性校验项「无中继架次绑定的中继区间
    = 0」的解。故两个量按字典序比较，无绑定数优先。

    该代理只用于搜索中的相对比较，最终方案一律以 Δt=1 s 的 audit 为准。
    """
    covered = {}
    for x in relays:
        for j in x['ivs']:
            covered[j] = x['link_done']
    tot, unbound = 0.0, 0
    for j, iv in enumerate(intervals):
        span = iv['t_end'] - iv['t_start']
        c = covered.get(j)
        if c is None:
            tot += span
            unbound += 1
        else:
            tot += min(span, max(0.0, c - iv['t_start']))
    return tot, unbound


def trip_damage(d, base_asg, deltas, idx=None):
    """按**架次**汇总的加权迟到（权重·s），只统计「被推迟过且真的迟了」的架次。

    与 `core.shifted_tardiness` 同源同口径（逐箱 优先系数 × max(0, 交付+位移−期望)），
    区别只在分组：搜索需要知道**是哪一架次**在贡献迟到，而不只是总量。

    缺了这一层，换站修复在缺口归零后就没有可排序的抓手，「缺口已归零」被当成收工
    信号，被推出去几千秒的架次再也没人过问——实测 24 架次方案里 T04 推 7313.6 s、
    独占 219447 加权迟到的绝大部分，而它的失效区间与 T05 的那段完全时间重叠、有
    15 个共用候选站，改派过去就能并进同一趟悬停、零推迟。`expect` 为空（题面没给
    期望送达时刻）的箱不计迟到，与 `weighted_tardiness` 一致。
    """
    idx = box_index(d) if idx is None else idx
    out = {}
    for k, dl in enumerate(deltas):
        if dl <= 1e-9:
            continue
        w = 0.0
        for boxes in base_asg[k]['boxes_at'].values():
            for bx in boxes:
                b = idx[bx['id']]
                if b['expect'] is None:
                    continue
                late = base_asg[k]['deliver_abs'][bx['id']] + dl - b['expect']
                if late > 1e-6:
                    w += b['priority'] * late
        if w > 1e-9:
            out[k] = w
    return out


def reassign_stations(d, base_asg, intervals, sel, cands, cover, slack, deltas0, snap,
                      verbose=False, max_try=15):
    """
    换站修复——补上集合覆盖看不见的那一维。

    集合覆盖只按「空间可覆盖 + 中继能耗」选站，**完全不知道中继机届时在不在位**：
    同一个失效区间往往有几十个站可选（实测 min=23），换一个「中继更早到位」的站
    就能把迟建链直接消掉，而这在时间维度上往往远优于继续推迟运输架次（推迟还要
    受硬时限与整条资源链的连环约束）。

    做法：反复按缺口从大到小取失效区间，逐个试它的替代站（按中继能耗升序），每组
    (sel, 推迟量) 都从原始状态整环重跑「合并 → 排班 → 就绪性修复」，用缺口代理
    量决定是否接受。代理量只用于相对比较，最终方案仍以 Δt=1 s 终检为准。

    **缺口归零之后不停手**：同一个换站动作既能把缺口消掉，也能把「为了消缺口而被
    推出去几千秒」的架次拉回来。判据仍是同一条字典序（无绑定数 → 缺口 → 加权迟到），
    故损伤修复不可能拿通信可行性去换交付质量；缺口为 0 时它只会在同为 0 的解里挑
    迟到更小的那个。焦点区间因此有两批：缺口最大的几段，和被推得最惨的几个架次。
    """
    # 货箱索引与原始排班在本环里是常量，建一次传下去——搜索每轮要试几十个动作，
    # 每次动作都要评一次迟到代价，重建索引会把这份开销乘以动作数。
    idx = box_index(d)
    # 邻接关系要按**错峰后**（即 snap 对应的）时刻判。搜索过程中 intervals 会随每轮
    # 推迟整体平移，被推了 7000 s 的架次其失效区间早已远离当初与之重叠的那些段，
    # 只看当前时刻就永远找不到「本可以同站合并」的邻居——T04 那类解的唯一入口正是
    # 这里。函数入口处 intervals 恰好停在 snap 状态，本表取的就是那一刻。
    snap_win = [(iv['t_start'], iv['t_end']) for iv in intervals]

    def attempt(sel_try, deltas_try):
        _reset_intervals(intervals, snap)
        relays, skipped, jobs, deltas, asg, rounds = repair_readiness(
            d, base_asg, intervals, sel_try, cands, deltas_try, slack, verbose=False)
        px, unbound = _outage_proxy(intervals, relays, skipped)
        # 迟到按 attempt 真正产出的 deltas 算，不按动作带进来的下限算：就绪性修复
        # 会在下限之上继续加推，最终落在哪只有跑完才知道。
        w_tard = shifted_tardiness(d, base_asg, deltas, idx)[0]
        # 资源链是否无冲突，是**排班是否成立**的前提，必须进判据：终检（audit）只
        # 看通信分段，一架无人机同时飞两个架次在它眼里完全正常。实测正是这一层
        # 缺失，让「零缺口 + 零迟到 + U05 同时飞 T17/T24」的解被选中并交付。
        n_conf = len(chain_conflicts(d, asg))
        return dict(sel=dict(sel_try), deltas=deltas, asg=asg, relays=relays, skipped=skipped,
                    jobs=jobs, rounds=rounds, px=px, unbound=unbound, w_tard=w_tard,
                    n_conf=n_conf)

    def better(new, old, eps=1e-6):
        """按 (资源链冲突处数, 无绑定区间数, 缺口秒数, 加权迟到) 字典序判优。

        第一层是**可行性的前提**：排班里同一条无人机/电池链不得重叠，这一条不
        成立时后面几层都无意义——一个「零缺口」的排班如果让一架无人机同时飞两个
        架次，它不是更好的解，它根本不是解。故它排在最前。

        其后两层是**硬**的：区间必须有中继绑定、且真的被盖住，这是题目对通信的要求，
        迟到再小也不能拿它换。最后一层才是交付侧：同为「缺口已归零」的两个方案，
        取迟到小的那个。

        为什么必须补第三层：本环的候选动作里有「让位」——把占住中继机的那些架次
        整体后移，好让缺口任务的窗口起点排到前面。挪哪个架次能腾出中继机，往往有
        多种选法，而**它们对交付侧的影响可以差出几个数量级**：实测被选中的那个
        动作把 T13（C 型，载 8 箱饮用水/应急食品）推后 5714.8 s，一家伙贡献了
        加权迟到 223301 权重·s，而同一轮里另外四个被推的架次（T15/T14/T19/T11）
        加起来是 0.0——它们的期望时刻本来就松。只比缺口时，这两种动作在判据里
        长得一模一样。

        只加这一层还不够：判据只有在**每个候选都被评过**之后才有意义。本轮已改为
        全候选评估（见下方循环），否则第一个改善的动作仍会被顺序决定，第三层形同
        虚设。另外这个量必须按「推迟之后」的交付时刻算，`shifted_tardiness` 早期
        有一版忘了把位移加进去，于是每个候选都算出 0、判据永远判平。
        """
        if new['n_conf'] != old['n_conf']:
            return new['n_conf'] < old['n_conf']
        if new['unbound'] != old['unbound']:
            return new['unbound'] < old['unbound']
        if abs(new['px'] - old['px']) > eps:
            return new['px'] < old['px'] - eps
        return new['w_tard'] < old['w_tard'] - eps

    best = attempt(sel, list(deltas0))
    for it in range(max_try):
        # 收敛判据只能是「缺口归零」。**不能**加成「没有弃飞架次就停」——缺口有
        # 两个来源：(a) 迟建链只盖住区间后半段、(b) 整段弃飞；而 (a) 根本不产生
        # any skipped 记录（实测本例 skipped 恒为 0、缺口全来自迟建链），照那个
        # 判据写会在第一轮直接退出，永远不去试任何替代站。
        #
        # 但「缺口归零」本身**不是**收工信号。零中断只说明通信可行，交付侧可能已经
        # 被就绪性修复推得面目全非——实测 24 架次方案里 T04 推 7313.6 s、独占
        # 219447 加权迟到的绝大部分。旧版本在这里无条件 `break`，于是「零中断但高
        # 迟到」的解一旦出现就再没有机会被换站修复碰过，哪怕它的失效区间与 T05 的
        # 那段完全重叠、改派过去即可零推迟。判据改为：缺口归零**且**没有任何架次
        # 在迟到，才停。缺口优先的字典序不变（见 better()），损伤动作只有在缺口同
        # 为 0 时才可能被接受，故这一步不会用通信可行性换交付质量。
        damage = trip_damage(d, base_asg, best['deltas'], idx)
        if best['px'] <= 1e-9 and not damage:
            break
        # 按缺口从大到小列出所有有缺口的区间。只取 argmax 是不够的：argmax 那个
        # 区间可能压根没有可用替代站，此时应当退而试次大的，而不是整体放弃。
        cov = {}
        for x in best['relays']:
            for j in x['ivs']:
                cov[j] = max(cov.get(j, -1e18), x['link_done'])
        gaps = []
        for j, iv in enumerate(intervals):
            c = cov.get(j)
            gap = (iv['t_end'] - iv['t_start']) if c is None else max(0.0, c - iv['t_start'])
            if gap > 1.0:
                gaps.append((gap, j))
        if not gaps and not damage:
            break
        gaps.sort(reverse=True)
        # 本轮要动的区间：缺口最大的两段，加上「被推得最惨」的两个架次名下的区间。
        # 两批都进候选，判据自己挑——有缺口时缺口动作必然更优（字典序第一层），
        # 缺口归零后剩下的只有损伤动作。上限 6 段，免得候选表随缺口数线性膨胀。
        focus = [j for _g, j in gaps[:2]]
        for k, _w in sorted(damage.items(), key=lambda kv: -kv[1])[:2]:
            focus += [j for j, iv in enumerate(intervals) if iv['trip'] == k]
        focus = list(dict.fromkeys(focus))[:6]

        def e_key(j, ci):
            return relay_trip_cost(d, _as_st(cands[ci]),
                                   intervals[j]['t_end'] - intervals[j]['t_start'])['E']

        # 组装本轮要试的动作。三类动作都必须是候选，缺任何一类都会在实测死结上判负：
        #   ① 单区间换站：把有缺口的那个区间挪到别的站。
        #   ② **整组换站**：把「同架次且窗口相邻（间隔 ≤ MERGE_GAP）」的若干区间
        #      一起挪到同一个站。缺了②就救不了实测里那个死结——G15 单独挪走，
        #      同架次的 G14 还留在旧站，原先把两段并成「一个站一个任务」的结构
        #      会裂成两个站两个任务，中继反而更赶不上，于是任何单区间动作都判负。
        #      实测 |cover[G14] ∩ cover[G15]| = 10 > 0，即这两段本可以同站合并，
        #      逐区间搜却永远发现不了。
        #   ③ **让位**：见下方注释——排班按窗口起点派发，顺序不是决策变量，于是
        #      缺口任务会被同一条中继链上「窗口起点更早」的任务挡在后面。
        moves = []
        for worst_j in focus:
            cur = best['sel'].get(worst_j)
            for ci in sorted((c for c in cover[worst_j] if c != cur),
                             key=lambda c: e_key(worst_j, c))[:3]:
                sel2 = dict(best['sel'])
                sel2[worst_j] = ci
                moves.append((f'G{worst_j+1:02d}→站{ci}', sel2, list(deltas0)))
            # 同架次的相邻区间（含自身）构成的换站组
            jt = intervals[worst_j]['trip']
            a, b = intervals[worst_j]['t_start'], intervals[worst_j]['t_end']
            grp = [worst_j] + [j2 for j2, iv2 in enumerate(intervals)
                               if j2 != worst_j and iv2['trip'] == jt
                               and iv2['t_start'] - b <= MERGE_GAP
                               and a - iv2['t_end'] <= MERGE_GAP]
            if len(grp) > 1:
                common = set(cover[grp[0]])
                for j2 in grp[1:]:
                    common &= cover[j2]
                names = '+'.join(f'G{j2+1:02d}' for j2 in sorted(grp))
                for ci in sorted(common,
                                 key=lambda c: sum(e_key(j2, c) for j2 in grp))[:3]:
                    sel2 = dict(best['sel'])
                    for j2 in grp:
                        sel2[j2] = ci
                    if sel2 != best['sel']:
                        moves.append((f'{names}→站{ci}', sel2, list(deltas0)))

        #   ④ **同站合并（跨架次）**：缺口区间等不到中继机，常常不是「没有站能覆盖
        #      它」，而是「覆盖它的站与同时段那段区间占的站不是同一个」——只有 2 架
        #      中继机时，同一时刻占用 3 个不同悬停站本身就是不可行，与被占的站是谁
        #      无关。①②看不到这一层：①按**单区间**的中继能耗排序取前 3 个候选，
        #      ②只在同一架次内找共站，而同时段冲突完全可能来自另一个架次。
        #      实测死结：24 架次方案里 T14 与 T15 的失效区间重叠 595 s，而
        #      |cover[T14] ∩ cover[T15]| = 0——这两段必须由两个不同的站保障；
        #      缺口那段 T16 却与 T14 有 5 个共用站、与 T15 有 2 个共用站，即
        #      「与相邻区间同站」本可把同时占站数从 3 压回 2，但逐区间搜索按单区间
        #      能耗取候选，永远取不到那几个共用站。故这里显式枚举共用站。
        #      多列候选只是多花时间，判据仍是 attempt 跑完的真值，不放宽任何标准。
        #
        #      **续驻那一半**：上面只讲「同一时刻」。还有一类同样看不见的冲突是
        #      「**先后**占住同一架中继机」：缺口区间等不到中继机，是因为前一段先把
        #      它钉在另一个站上了。若某个站同时覆盖两段、而两段间隔 ≤ MERGE_GAP，
        #      `merge_jobs` 会把它们并成**一个**中继架次——中继机不必返航、不必转场，
        #      一次悬停把两段都保障掉，冲突自然消失。故判据从「时间重叠」放宽到
        #      「间隔 ≤ MERGE_GAP」（时间重叠是间隔为负的特例），枚举方式不变。
        #      实测死结（本算例当前唯一残留缺口）：G11(T16, 站 S02, 3056–3917 s)
        #      晚建链 793.9 s，根因是 G06(T11, 站 S01, 2130–2552 s) 把 R02 钉在 S01
        #      直到 2552 s，返航 3030 s、再转场到 S02 建链已是 3850 s——固定开销
        #      1298 s 无从压缩。而 |cover[G06] ∩ cover[G11]| > 0 且两段间隔 504 s
        #      ≤ MERGE_GAP：把 G06 改派到 S02，R02 从 2130 s 一直悬停到 3917 s 即可，
        #      代价仅是 T11 后移 404 s（其硬时限 slack 为无穷），由就绪性修复环补上。
        for worst_j in focus:
            ivj = intervals[worst_j]
            a1, b1 = snap_win[worst_j]
            for j2, iv2 in enumerate(intervals):
                if j2 == worst_j:
                    continue
                # 时间重叠（间隔<0，同时占站的冲突）与近邻（间隔≤MERGE_GAP，
                # 先后占同一架中继机的冲突）都要枚举；更远的间隔合并不成一个架次。
                gap12 = max(iv2['t_start'] - ivj['t_end'],
                            ivj['t_start'] - iv2['t_end'])
                # 再按错峰后（snap）的时刻判一次。当前时刻与 snap 时刻在搜索早轮
                # 就已经分叉：被就绪性修复推走的架次，其区间按当前时刻看已经没邻居，
                # 而它当初与谁重叠是确定的。两套时刻取**更近**的那个，故候选集是
                # 旧版的超集，不存在「换掉一个好候选」的风险。
                a2, b2 = snap_win[j2]
                gap12 = min(gap12, max(a2 - b1, a1 - b2))
                if gap12 > MERGE_GAP:
                    continue
                common = cover[worst_j] & cover[j2]
                if not common:
                    continue
                for ci in sorted(common, key=lambda c: e_key(worst_j, c))[:2]:
                    # a) 只把缺口区间挪到对方的站；b) 两段一起挪到共用站
                    for both in (False, True):
                        sel2 = dict(best['sel'])
                        sel2[worst_j] = ci
                        if both:
                            sel2[j2] = ci
                        if sel2 != best['sel']:
                            tag = ('G%02d+G%02d' % (worst_j + 1, j2 + 1) if both
                                   else 'G%02d→G%02d 之站' % (worst_j + 1, j2 + 1))
                            moves.append((f'{tag}→站{ci}', sel2, list(deltas0)))

        # ③ 让位：schedule_relays 是按**窗口起点**依次派发的，顺序不是决策变量。
        # 于是「同一条中继链上更早的那个任务」会先占住中继机，缺口任务只能等它飞完
        # 一整趟（去程 + 服务 + 返航 + 周转）才轮得到——这正是 G15 的死结：R02 先去
        # S03 服务 T14/T16（窗口起点 5112.6 s），返航已是 6581.8 s，加周转 6881.8 s
        # 才起飞，而 G15 在 6762.6 s 就开了，迟 830.6 s。
        # 解法不需要改排班器：把占位那条任务所属的运输架次整体后移，让它的窗口起点
        # 越过缺口任务的窗口起点，t1 排序自然翻转，缺口任务就排到了前面。这里直接
        # 给这些架次一个**推迟下限**（repair_readiness 只会在此基础上再加，不会减），
        # 余下由就绪性修复环自己收敛。是否让得开由硬时限 slack 把关，让不开就跳过。
        for _g1, worst_j in gaps[:1]:
            t1j = intervals[worst_j]['t_start']
            riders = [x for x in best['relays'] if worst_j in x['ivs']]
            if not riders:
                continue
            rid = riders[0]['relay']
            for blk in best['relays']:
                if blk['relay'] != rid or blk['link_done'] >= t1j - 1e-6:
                    continue
                b1 = min(intervals[j2]['t_start'] for j2 in blk['ivs'])
                dur = max(0.0, blk['service_end'] - blk['link_done'])
                need = (t1j - b1) + dur + d.relay_type['turnover'] + 900.0
                d2, ok = list(deltas0), True
                for j2 in blk['ivs']:
                    k = intervals[j2]['trip']
                    if slack[k] == float('inf'):
                        d2[k] = max(d2[k], need)
                    elif d2[k] + need <= slack[k] + 1e-9:
                        d2[k] = max(d2[k], need)
                    else:
                        ok = False
                        break
                if not ok:
                    continue
                # 逐架次 slack 只保证「自己不被硬时限挡住」，管不了链上后一架次。
                # 让位动辄几千秒（实测 T17 被要求后移 1583 s，而同链的 T24 只有
                # 1344.5 s 余量），不在这里先走一遍级联，这条动作会带着一个推不平
                # 的排班进入 attempt，最终变成一架无人机同时飞两个架次。级联只可能
                # 再往后推，不会把 need 变小，故上面那条硬时限检查仍然必要。
                d2, _asg2, ok2 = cascade_shift(d, base_asg, d2, {}, slack)
                if not ok2:
                    continue
                names = '+'.join(sorted({f'T{intervals[j2]["trip"]+1:02d}'
                                         for j2 in blk['ivs']}))
                moves.append((f'让位：{names} 后移 ≥{need:.0f}s 以腾出 {rid}',
                              dict(best['sel']), d2))

        # **全部候选都评完再取最优**，不再「遇到第一个改善就 break」。
        #
        # 原来是取第一个改善的动作。加进第三层判据（加权迟到）之后这就不成立了：
        # 「让位 T14+T16 腾出 R02」把 T13 推后 5714.8 s、独占全部 223301 加权迟到，
        # 而同一轮里别的候选可能是零迟到的，但第一个改善的位置由列表顺序决定，
        # 判据再好也没机会比较。
        #
        # 试过用「按动作自带推迟下限估的迟到」排序把好的动作挪到前面——**没用**：
        # 那个估计只知道动作自己的下限，而真实损伤来自就绪性修复在其上的继续传播
        # （让位动作的 d2 只含 T14/T16，5714.8 s 是修复环后来加给 T13 的），所以
        # 每个候选估出来都是 0，排序是空操作。能分辨好坏的只有跑完 attempt 的真值。
        #
        # 代价可忽略：attempt 的主体内是 repair_readiness，实测单次 0.02 s、
        # 一轮十余个候选，合计零点几秒（本函数所在 solve_relay 总耗时约 158 s）。
        # 尾部本就从 snap 重放 best['asg'] 复原状态，故多跑几个候选不影响最终一致性。
        # 改派出来的新 (区间, 站) 对必须过真密度复核，与同时占站修复同一道关。
        # `cover[j]` 来自 stride=5 的抽点初判，抽点上够裕量不等于逐点都够——实测漏网
        # 的一例正是本环：T06 改派到 S02 后抽点全过、逐点最小裕量 −1.20 dB，终检
        # 留下 213 s 真中断。过滤器只可能剔除候选，不会放宽任何判据。
        moves = [(tag, sel2, d2) for (tag, sel2, d2) in moves
                 if sel_pairs_ok(d, intervals, cands, sel2, best['sel'])]

        # 两套时刻（当前 / snap）可能给出同一动作，按标签去重：标签由
        # (缺口区间, 对方区间, 候选站, 是否两段同移) 唯一决定，同标签必同 sel2。
        # 只省一次 attempt 的开销，不动候选集合。
        _seen, _uniq = set(), []
        for tag, sel2, d2 in moves:
            if tag in _seen:
                continue
            _seen.add(tag)
            _uniq.append((tag, sel2, d2))
        moves = _uniq

        improved, best_tag = False, None
        for tag, sel2, d2 in moves:
            r = attempt(sel2, d2)
            if better(r, best):
                best, improved, best_tag = r, True, tag
        if improved and verbose:
            print(f'  换站修复第 {it+1} 轮：{best_tag}，'
                  f'缺口 {best["px"]:.0f} s、无绑定 {best["unbound"]} 个、'
                  f'加权迟到 {best["w_tard"]:.0f}'
                  f'（{len(moves)} 个候选中选优'
                  f'{"，本轮为缺口归零后的交付修复" if best["px"] <= 1e-9 and gaps else ""}）')
        if not improved:
            if verbose:
                print(f'  换站修复：{len(moves)} 个候选动作均无改善，停止')
            break
    # 让 intervals 回到最优方案对应的状态。必须用 best 自己的推迟量（best['deltas']
    # / best['asg']）复原——「让位」这类动作的推迟量是带进来的下限，不属于错峰阶段
    # 的 deltas0，若照 deltas0 重跑就等于把该动作整个丢掉，代理算出的改进会被悄悄
    # 回滚（实测踩过：代理报缺口 0 s，终检却仍是 1.12%）。
    _reset_intervals(intervals, snap)
    _apply_shift(intervals, best['asg'])
    return (best['sel'], best['deltas'], best['asg'],
            best['relays'], best['skipped'], best['jobs'], best['rounds'])


def repair_readiness(d, base_asg, intervals, sel, cands, deltas, slack,
                     verbose=False, max_rounds=MAX_REPAIR_ROUNDS):
    """
    就绪性修复环——把「中继届时还没到位」翻译成对运输架次开始时刻的约束。

    错峰若只削「同时中继需求峰值」，遇到**峰值没超、但中继机此刻还没飞到**的情形
    就完全无能为力：中继机一趟的固定开销是 准备 180 s + 建链 30 s + 去程 + 周转
    300 s，前一趟没落地，后一趟再急也到不了。于是排班层只能弃飞，或者「迟建链」
    只盖住区间后半段，两者都在终检里表现为通信中断。

    本环逐轮把这两类缺口折算成推迟量，加到拥有该任务的运输架次上（只向后推，
    且受硬时限 slack 与无人机/电池占用双重把关），再整环重跑「合并 → 排班 →
    判定」。必须整环重跑：平移会改变同站区间之间 ≤MERGE_GAP 的相邻关系，合并
    结构随之变化，一次算完的解不成立。

    参数 base_asg 是错峰前的原始排班，deltas 是相对它的累计推迟量；每轮都用
    shift_assignment(d, base_asg, deltas) 重算，避免增量叠加漂移。
    """
    # `assignment` 必须在进环**之前**绑好：第 0 轮若 `not req`（交上来的方案本来就
    # 没有迟建链、也没有弃飞）会直接走下面那个 return，而它是函数局部名——不预置
    # 就会抛 UnboundLocalError，把「候选一上来就干净」这种**最好**的情形变成崩溃。
    assignment = shift_assignment(d, base_asg, deltas)
    for rnd in range(max_rounds):
        jobs = merge_jobs(intervals, sel)
        relays, skipped = schedule_relays(d, jobs, cands, verbose=False)
        req = {}

        def bump(trip, need):
            if need > req.get(trip, 0.0):
                req[trip] = need

        # (a) 迟建链：建链时刻晚于任务窗口起点，早段必然中断
        for x in relays:
            t1 = min(intervals[j]['t_start'] for j in x['ivs'])
            need = x['link_done'] - t1
            if need > 1e-6:
                j_first = min(x['ivs'], key=lambda j: intervals[j]['t_start'])
                bump(intervals[j_first]['trip'], need)
        # (b) 弃飞：窗口末端早于最早可建链时刻，整段无人保障
        for s in skipped:
            jb = s['job']
            j_last = max(jb['ivs'], key=lambda j: intervals[j]['t_end'])
            bump(intervals[j_last]['trip'], s['arrive'] - jb['t2'])

        if not req:
            if verbose and rnd:
                print(f'  就绪性修复：第 {rnd} 轮收敛，弃飞 0、迟建链 0')
            return relays, skipped, jobs, deltas, assignment, rnd

        cand, asg2, ok = cascade_shift(d, base_asg, deltas, req, slack)
        if not ok:
            # 进来的推迟量本身就沿资源链推不平（调用方按逐架次 slack 拼出的下限
            # 只保证「自己不被硬时限挡住」，不保证链上下一架次还接得住）。此时
            # 唯一正确的动作是**原地停手**：继续走下去等于把一个无人机同时飞两
            # 架次的排班当成可行解交出去，而终检只查通信缺口、查不到这一层。
            if verbose:
                print(f'  就绪性修复：第 {rnd+1} 轮的推迟量沿资源链推不平，'
                      f'放弃本轮（仍有 {len(req)} 个任务需要后移）')
            return relays, skipped, jobs, deltas, assignment, rnd
        if cand == deltas:
            if verbose:
                print(f'  就绪性修复：第 {rnd+1} 轮推不动了'
                      f'（受硬时限限制，仍有 {len(req)} 个任务需要后移），停止')
            return relays, skipped, jobs, deltas, assignment, rnd
        deltas, assignment = cand, asg2
        _apply_shift(intervals, assignment)
        if verbose:
            print(f'  就绪性修复第 {rnd+1} 轮：{len(req)} 个任务需后移，'
                  f'{sum(1 for i in range(len(deltas)) if deltas[i] > 0)} 个架次已推迟')
    jobs = merge_jobs(intervals, sel)
    relays, skipped = schedule_relays(d, jobs, cands, verbose=False)
    return relays, skipped, jobs, deltas, assignment, max_rounds


def stagger(d, assignment, dead_of_trip, slack, cover, verbose=False):
    """
    错峰：把运输架次开始时刻作为决策变量（只允许向后推迟，不早于 Q2 的资源可用
    时刻），目标是让「同时**必须**占用的悬停站数」不超过中继机数——2 架中继是硬
    瓶颈，削峰是零中断的关键。可行性只在两处把关：所有箱的硬时限不得违反；无人机
    与共享电池的占用区间不得重叠（复用 q2.resource_usage 的口径，不另写一份）。

    目标函数用**超额站·秒**（`simultaneous_station_excess`）而不是全局峰值：
    本算例里峰值 3 同时出现在 t≈1027 s 与 t≈3603 s 两处，前者只是当前选站方案
    多占了一个站（同一时刻 2 个站就够），后者才是真的排不下。用全局峰值当目标时，
    把后者削掉峰值仍是 3、判据看不出改善，搜索于是停在原地；超额站·秒只惩罚真
    正无解的那 104 s，改善方向与「2 架中继机够不够」完全一致。

    推迟一律经 `cascade_shift` 沿资源链传播后再评估，而不是一遇冲突就否决：本
    算例正是 T17 被同链的后续架次顶住，单架次后移全部被 `chain_conflicts` 拦下，
    而它只需后移 104 s 就能让 G12 完全错开 G10。

    返回 (新 assignment, 位移列表, 超额站·秒前, 超额站·秒后, 峰值前, 峰值后)。
    """

    n_relay = len(d.relay_uavs) if getattr(d, 'relay_uavs', None) else 2
    _cache = {}
    # 区间与站的绑定关系固定（选站由上层决定，错峰只动时刻），先把失效区间压平成
    # 一张 (架次, 站, 起, 止) 表并编上全局序号，`obj_of` 只需按位移量平移时刻。
    iv_tab = [(k, ci, t1, t2)
              for k, lst in enumerate(dead_of_trip) for (ci, t1, t2) in lst]

    def obj_of(asg):
        by_iv = {}
        for j, (k, _ci, t1, t2) in enumerate(iv_tab):
            base = asg[k]['start'] - assignment[k]['start']
            by_iv[j] = (t1 + base, t2 + base)
        return simultaneous_station_excess(by_iv, cover, n_relay, _cache)

    p0, pk0 = obj_of(assignment)
    deltas = [0.0] * len(assignment)
    cur = list(assignment)
    p_cur, pk_cur = p0, pk0
    caps = [min(sl, 1800.0) if np.isfinite(sl) else 1800.0 for sl in slack]
    grid = [30.0, 60.0, 120.0, 240.0, 480.0, 900.0, 1800.0]
    for k in sorted(range(len(assignment)), key=lambda i: -len(dead_of_trip[i])):
        if not dead_of_trip[k]:
            continue
        best_key, best_delta = (p_cur, pk_cur), None
        for dl in grid:
            if dl > caps[k] + 1e-9:
                break
            trial = list(deltas)
            trial[k] = dl
            d2, asg2, ok = cascade_shift(d, assignment, trial, {k: dl}, slack)
            if not ok:
                continue
            key2 = obj_of(asg2)
            if key2 < best_key:
                # 接受的是**级联后**的整组位移，不是单独一个 dl：若只记 dl 而丢掉
                # 沿资源链传播出来的部分，落定的排班仍会在链上冲突。
                best_key, best_delta = key2, d2
        if best_delta is not None:
            deltas = list(best_delta)
            cur = shift_assignment(d, assignment, deltas)
        p_cur, pk_cur = best_key
    if verbose:
        print(f'错峰: 超额站·秒 {p0:.1f} → {p_cur:.1f}，同时所需悬停站峰值 '
              f'{pk0} → {pk_cur}（共 {int(sum(1 for x in deltas if x > 0))} 个架次被推迟）')
    return cur, deltas, p0, p_cur, pk0, pk_cur


# ===========================================================================
# 8. 中继排班
# ===========================================================================
def merge_jobs(intervals, sel, gap=MERGE_GAP, max_job=MAX_JOB):
    """
    把同一站所辖的失效区间并成中继架次：间隔≤gap 就并，并后服务时长不超过
    max_job。并的收益是省下一整个中继架次的占用（准备+建链+往返+周转约
    800 s 的无人机时间），代价只是间隙里的悬停能耗（1.1 kW）。
    """
    by_st = {}
    for j, ci in sel.items():
        by_st.setdefault(ci, []).append(intervals[j])
    jobs = []
    for ci, ivs in by_st.items():
        ivs.sort(key=lambda x: x['t_start'])
        merged = []          # 每项 = dict(t1, t2, ivs=[区间下标])
        for iv in ivs:
            if merged and iv['t_start'] - merged[-1]['t2'] <= gap \
                    and iv['t_end'] - merged[-1]['t1'] <= max_job:
                merged[-1]['t2'] = max(merged[-1]['t2'], iv['t_end'])
                merged[-1]['ivs'].append(iv['idx'])
            else:
                merged.append(dict(t1=iv['t_start'], t2=iv['t_end'], ivs=[iv['idx']]))
        for m in merged:
            jobs.append(dict(ci=ci, t1=m['t1'], t2=m['t2'], ivs=m['ivs']))
    jobs.sort(key=lambda j: j['t1'])
    return jobs


# 中继排班是否允许「站与站之间直接转场」（阶段 13）。`False` 时退回「每次出动只
# 服务一个站、必返 O01」的旧规则，且走的是 `q3_chain` 里与旧实现同一段算术的分支，
# 因此与改动前的解逐位一致——这是「旧解仍可复现」的开关，也是一组回归用例。
ALLOW_CHAIN_RELAY = True


def schedule_relays(d, jobs, cands, verbose=False, allow_chain=None):
    """中继排班：按失效区间起点顺序贪心派发。

    排班算法自阶段 13 起迁到 `q3_chain.py`（允许一次出动内站间接续），本函数只是
    接线。保留「建链完成晚于区间结束即弃飞」规则——空飞同样占用中继与能源组件，
    会把后续本可按时到达的架次一起顶迟到。
    """
    if allow_chain is None:
        allow_chain = ALLOW_CHAIN_RELAY
    return _schedule_relays_impl(d, jobs, cands, verbose=verbose,
                                 allow_chain=allow_chain)


# ===========================================================================
# 9. 高密度终检
# ===========================================================================
def audit(d, assignment, relays, cands, dt=DT_AUDIT, verbose=False):
    """
    终检：以 Δt=1 s 重采全部轨迹点，逐时刻判定
      直连可用 → 直连；
      否则须存在一架中继，满足 (i) 该时刻落在它的实际服务窗口
      [建链完成, 服务结束] 内，(ii) **该架中继所在的站此刻真的能接通这个点**
      （接入裕量≥0），(iii) 该站对 G01 回传可用 → 中继；
      否则 → 中断。
    占比按采样点的时间权重积分，得时间占比。返回 (时间占比, 分段表, 中断明细)。
    """
    wins = [(x['link_done'], x['service_end'], x['station'], i)
            for i, x in enumerate(relays) if x['service_end'] > x['link_done'] + 1e-9]
    tot = dict(direct=0.0, relay=0.0, gap=0.0)
    segs, gaps = [], []
    for ai, a in enumerate(assignment):
        t = d.transport_types[a['type']]
        pts = sample_trip_trajectory(d, t, a['route'], a['boxes_at'], t0=a['start'], dt=dt)
        w = time_weights(pts)
        states = []
        for k, (tt, lon, lat, alt) in enumerate(pts):
            if direct_margin(d, lon, lat, alt) >= MARGIN_DB:
                states.append(('直连', '直连', '', w[k]))
                continue
            hit = None
            for (w1, w2, ci, ri) in wins:
                if w1 - 1e-6 <= tt <= w2 + 1e-6 and \
                        access_margin(d, _as_st(cands[ci]), lon, lat, alt) >= MARGIN_DB:
                    hit = ('中继', '中继', 'R%02d' % (ri + 1), w[k], ci)
                    break
            if hit:
                states.append(hit)
            else:
                states.append(('中断', '中断', '', w[k]))
                gaps.append(dict(trip=ai, t=tt, lon=lon, lat=lat, alt=alt))
        for s in states:
            tot[{'直连': 'direct', '中继': 'relay', '中断': 'gap'}[s[0]]] += s[3]
        # 共同边界序列 b_0 ≤ … ≤ b_n：内部边界取相邻采样时刻的中点，b_0=t_0、
        # b_n=t_{n-1}。于是采样点 k 独占的区间 [b_k, b_{k+1}] 的长度恰为
        # time_weights 给出的 w_k，分段表与上面报出的时间占比用的是**同一套**
        # 时间测度。旧写法用 pts[j-1][0] 收尾、下一段从 pts[j][0] 起，每处边界都
        # 漏掉一个采样步长 Δt（漏多少还随 dt 变），「相邻边界相等」无从谈起。
        ts = [p[0] for p in pts]
        nb = len(ts)
        b = [ts[0]] + [(ts[k - 1] + ts[k]) / 2.0 for k in range(1, nb)] + [ts[-1]]
        i = 0
        while i < nb:
            j = i
            while j < nb and states[j][:3] == states[i][:3]:
                j += 1
            segs.append(dict(trip=ai, phase=states[i][0], mode=states[i][1],
                             rid=states[i][2], t1=b[i], t2=b[j],
                             w=sum(w[i:j]), n_pts=j - i,
                             total=ts[-1] - ts[0]))
            i = j
    errs = check_phase_table(segs)
    if errs:
        raise AssertionError('通信分段表不自洽：' + '；'.join(errs[:3]))
    T = sum(tot.values())
    frac = {k: (v / T if T > 0 else 0.0) for k, v in tot.items()}
    frac['total_time'] = T
    if verbose:
        print(f'终检(Δt={dt:g}s，{T:.0f}s 总时长): 直连 {100*frac["direct"]:.2f}% | '
              f'中继 {100*frac["relay"]:.2f}% | 中断 {100*frac["gap"]:.2f}%')
        if gaps:
            print(f'  ⚠ 仍有 {len(gaps)} 个采样点处于中断，累计 '
                  f'{tot["gap"]:.0f} s')
    return frac, segs, gaps


# ===========================================================================
# 10. 序贯对照（不做通信协同的运输调度 / 无中继）
# ===========================================================================
def check_phase_table(segs, tol=1e-9):
    """
    通信分段表自洽性，**与采样步长无关**：只查表里写出的边界本身。

      ① 相邻边界相等：同一架次内后一段的起点必须逐位等于前一段的终点；
      ② 无空洞、无重叠：由 ① 直接保证，此处显式复算一遍差值；
      ③ 覆盖完整：同一架次所有段长之和 == 该架次轨迹总时长（time_weights 的口径）；
      ④ 段常性：每段内不得出现两种不同的通信阶段（阶段必须真的成段）。

    返回错误清单，空列表表示通过。旧版按「上一段收尾于 pts[j-1]、下一段起于
    pts[j]」写表，每处边界都差一个 Δt，把 dt 从 2 s 改到 1 s 表的数值就跟着变，
    这条校验根本立不住；现在边界由共同序列生成，校验与 dt 解耦。
    """
    errs = []
    by_trip = {}
    for s in segs:
        by_trip.setdefault(s['trip'], []).append(s)
    for tr, ss in sorted(by_trip.items()):
        ss.sort(key=lambda x: x['t1'])
        for x, y in zip(ss, ss[1:]):
            if abs(y['t1'] - x['t2']) > tol:
                errs.append('T%02d 分段边界不相等：%s 止于 %.9f，%s 起于 %.9f'
                            % (tr + 1, x['phase'], x['t2'], y['phase'], y['t1']))
        tot_seg = sum(x['t2'] - x['t1'] for x in ss)
        tot_ref = ss[0].get('total', None)
        if tot_ref is not None and abs(tot_seg - tot_ref) > 1e-6:
            errs.append('T%02d 分段总长 %.9f 与轨迹时长 %.9f 不符'
                        % (tr + 1, tot_seg, tot_ref))
        for x in ss:
            if x['t2'] < x['t1'] - tol:
                errs.append('T%02d 出现负长分段：%.9f → %.9f' % (tr + 1, x['t1'], x['t2']))
    return errs


def sequential_baseline(d, assignment, dt=DT_AUDIT):
    """
    对照实验：**完全不做通信协同**的运输调度下，直连能撑住多少。
    这是「题给约束被违反」的量化，不是成绩单——用于说明为什么必须做联合优化。
    """
    tot = dict(direct=0.0, gap=0.0)
    for a in assignment:
        t = d.transport_types[a['type']]
        pts = sample_trip_trajectory(d, t, a['route'], a['boxes_at'], t0=a['start'], dt=dt)
        w = time_weights(pts)
        for k, (_, lon, lat, alt) in enumerate(pts):
            if direct_margin(d, lon, lat, alt) >= MARGIN_DB:
                tot['direct'] += w[k]
            else:
                tot['gap'] += w[k]
    T = tot['direct'] + tot['gap']
    return dict(direct=tot['direct'] / T, outage=tot['gap'] / T, total_time=T)


# ===========================================================================
# 主流程
# ===========================================================================
def solve_relay(d, assignment, greedy=False, single_hover=False, verbose=True):
    """
    通信保障联合求解。greedy=True 时退化为「固定高度 + 贪心选站」的序贯做法，
    供消融对照用。返回 dict（含中继架次、悬停站、覆盖统计、终检结果）。
    """
    t0 = _time.time()
    hover_set = [300.0] if single_hover else HOVER_SET

    # 1) 逐架次失效区间（区间带所属架次信息，供真密度复核复用）
    intervals = []
    for ai, a in enumerate(assignment):
        pts, ivs = dead_intervals(d, a)
        for iv in ivs:
            iv['trip'] = ai
            iv['type'] = a['type']
            iv['assign'] = a
            iv['idx'] = len(intervals)
            intervals.append(iv)
    if verbose:
        print(f'失效区间: {len(intervals)} 个，涉及 '
              f'{len({iv["trip"] for iv in intervals})} 个运输架次')

    # 2) 候选站
    cands = build_candidates(d, intervals, hover_set=hover_set, verbose=verbose)
    if not cands:
        return dict(ok=False, reason='无候选悬停站', intervals=intervals, cands=[])

    # 3) 覆盖矩阵
    cover = cover_matrix(d, intervals, cands, verbose=verbose)
    if greedy:
        # 贪心集合覆盖（对照口径）：每轮取能多覆盖最多未覆盖区间的站
        unc = set(range(len(intervals)))
        sel = {}
        while unc:
            cnt = {}
            for j in unc:
                for ci in cover[j]:
                    cnt[ci] = cnt.get(ci, 0) + 1
            if not cnt:
                break
            best = max(cnt, key=lambda c: (cnt[c], -c))
            for j in list(unc):
                if best in cover[j]:
                    sel[j] = best
                    unc.discard(j)
    else:
        # 4) 集合覆盖 MILP + 真密度修复环
        ecost = []
        for j, iv in enumerate(intervals):
            row = {}
            for ci in cover[j]:
                st = _as_st(cands[ci])
                row[ci] = relay_trip_cost(d, st, iv['t_end'] - iv['t_start'])['E']
            ecost.append(row)
        pairs, sel, used, status = solve_setcover(len(intervals), cover, ecost, len(cands))
        if verbose:
            print(f'集合覆盖 MILP: {status}，选用悬停站 {used} 个')
        # 列生成式修复环：粗步长初判只允许产生假正例，真密度复核把不成立的
        # (区间, 站) 对剔掉后重解，直到 MILP 给出的指派在真密度下确实成立。
        for it in range(6):
            bad = verify_pairs(d, intervals, cands, list(sel.items()))
            if not bad:
                break
            if verbose:
                print(f'  真密度复核第 {it+1} 轮：剔除 {len(bad)} 个不成立的 (区间,站) 对')
            for (j, ci) in bad:
                cover[j].discard(ci)
            ecost = [{ci: relay_trip_cost(d, _as_st(cands[ci]),
                                          intervals[j]['t_end'] - intervals[j]['t_start'])['E']
                      for ci in cover[j]} for j in range(len(intervals))]
            pairs, sel, used, status = solve_setcover(len(intervals), cover, ecost, len(cands))
            if verbose:
                print(f'  重解: {status}，选用悬停站 {used} 个')
        else:
            bad = verify_pairs(d, intervals, cands, list(sel.items()))
            for (j, ci) in bad:
                cover[j].discard(ci)
                sel.pop(j, None)

    uncovered = [j for j in range(len(intervals)) if j not in sel]
    if verbose:
        print(f'已保障失效区间 {len(sel)}/{len(intervals)}，未保障 {len(uncovered)}')
        note = uncovered_note(uncovered)
        if note:
            print(note)

    # 5) 站址连续精化
    sel_stations = {}
    for j, ci in sel.items():
        sel_stations.setdefault(ci, []).append(j)
    refined = {}
    for ci, js in sel_stations.items():
        pts = [p for j in js for p in intervals[j]['pts']]
        c2, m2 = refine_station(d, cands[ci], pts)
        refined[ci] = c2
    for ci, c2 in refined.items():
        cands[ci] = c2

    # 6) 错峰削峰 + 就绪性修复 + 7) 中继排班（只在不贪心即真联合时启用）
    deltas = [0.0] * len(assignment)
    peak0 = peak1 = None
    repair_rounds = 0
    if not greedy and not single_hover:
        doft = [[] for _ in assignment]
        for j, ci in sel.items():
            doft[intervals[j]['trip']].append((ci, intervals[j]['t_start'], intervals[j]['t_end']))
        slack = trip_slack(d, assignment)
        base_asg = assignment        # 累计推迟量一律相对这个基准重算
        assignment, deltas, exc0, exc1, peak0, peak1 = stagger(
            d, assignment, doft, slack, cover, verbose=verbose)
        _apply_shift(intervals, assignment)
        # 6b) 同时占站约束修复。必须排在错峰**之后**：错峰先把「几何上同时要 3 个站」
        # 的时段消掉（本算例 104.3 s），几何下界落到中继机数以内，这一步才可能把
        # **当前这组站**的同时占用也压进去；排在错峰前则会对着一个本就无解的时段修。
        sel, viol0, viol1 = enforce_concurrency(
            d, intervals, sel, cover, cands, len(d.relay_uavs), verbose=verbose)
        if viol1 < viol0 - 1e-9:
            # 改派后新启用的站没做过第 5 步的连续精化，补一次（同一套坐标下降）。
            sel_stations = {}
            for j, ci in sel.items():
                sel_stations.setdefault(ci, []).append(j)
            for ci, js in sel_stations.items():
                pts = [p for j in js for p in intervals[j]['pts']]
                cands[ci], _m = refine_station(d, cands[ci], pts)
        # 削峰只治「同时所需悬停站数超过中继机数」；没超但中继机届时没飞到位的，交给下面两级修复：
        #   ① 就绪性修复（沿资源链级联后移运输架次，受硬时限 slack 把关）
        #   ② 换站修复（①顶死硬时限时改换「中继届时真在位」的站，补集合覆盖看不见的维度）
        # snap 必须取在 _apply_shift **之后**：repair_readiness 的轮 0 直接拿当前
        # intervals 去 merge_jobs，即它假定 intervals 的状态与传入的 deltas 对应，
        # 而各次 attempt 传的都是错峰后的 deltas，故基准快照就是错峰后的状态。
        snap = _snapshot_intervals(intervals)
        sel_a, deltas_a, asg_a, relays_a, skipped_a, jobs_a, repair_rounds = reassign_stations(
            d, base_asg, intervals, sel, cands, cover, slack, deltas, snap, verbose=verbose)
        snap_a = _snapshot_intervals(intervals)
        # 旧分支的缺口代理必须在 intervals 还停在 snap_a 状态时取（此刻正对应 relays_a）
        px_a = _outage_proxy(intervals, relays_a, skipped_a)

        # 并列路径：固定点迭代错峰（阶段 11 移植，见 q3_v2 的模块说明）。
        # 两条路都跑完再择优，旧路（上面这段）原地保留、逐位可复现。
        #
        # 择优判据必须用**缺口代理**打头，而不是「弃飞架次数」：`schedule_relays` 允许
        # 中继晚于区间起点到场（只把迟到记在 `late` 上），此时区间早段无人保障，而
        # 弃飞数为 0。实测正是这个差异：固定点解弃飞 0 却留下 4.22% 的真实中断。
        # 代理的两个量与 `_outage_proxy` 的说明一致，按 (无绑定区间数, 缺口秒数) 优先，
        # 其后才是「少推一点、早干完、少耗电」。
        sel, deltas, assignment, relays, skipped, jobs, repair_rounds = (
            sel_a, deltas_a, asg_a, relays_a, skipped_a, jobs_a, repair_rounds)
        _reset_intervals(intervals, snap)
        # 站址选择两条路共用（都用旧分支选出的 sel），故本次对照隔离出的正是
        # 「错峰/推迟策略」这一项，不含选站差异。
        global _V2_CHECKED
        try:
            import q3_v2
            if not _V2_CHECKED:
                q3_v2.self_check(d, base_asg, intervals, sel, cands, verbose=verbose)
                _V2_CHECKED = True
            r2 = q3_v2.solve_joint_v2(d, base_asg, intervals, sel, cands, slack,
                                      verbose=verbose)
        except Exception as e:                      # noqa: BLE001
            r2 = None
            print(f'  [v2 固定点] 未产出结果（{type(e).__name__}: {e}），沿用旧分支')
        if r2 is None:
            _reset_intervals(intervals, snap_a)
        else:
            px2 = _outage_proxy(intervals, r2['relays'], r2['skipped'])

            def _key(px, dl, asg, rls):
                done = max([x['return_t'] for x in rls], default=0.0)
                done = max(done, max((a['start'] + a['duration'] for a in asg), default=0.0))
                # 第三元由「累计推迟量」换成**加权迟到**：推迟量不是收益函数，
                # 推一个期望时刻很松的架次代价是 0、推一个很紧的代价是 优先系数×迟到秒，
                # 两者在 Σ推迟量 里长得一模一样。随后是完工、能耗，末位才是推迟量
                # （都无迟到时，少推一点仍是更干净的解）。
                # 中继能耗按**出动**求和（一排一行 = 一次悬停站服务，出动级的 E 在
                # 该出动的每一行上重复出现）。拼排班器名字取，不在这里另写一份。
                return (px[1], round(px[0], 6), round(weighted_tardiness(d, asg)[0], 6),
                        round(done, 6), round(q3_chain.outing_stats(rls)[1], 9),
                        round(sum(dl), 6))
            k1 = _key(px_a, deltas_a, asg_a, relays_a)
            k2 = _key(px2, r2['deltas'], r2['assignment'], r2['relays'])
            tag = (f'缺口 {k2[1]:.0f}s/无绑定 {k2[0]}、加权迟到 {k2[2]:.0f}、累计推迟 '
                   f'{sum(r2["deltas"]):.0f} s、联合完工 {k2[3]:.0f} s、中继能耗 '
                   f'{q3_chain.outing_stats(r2["relays"])[1]:.3f} kWh；'
                   f'旧：缺口 {k1[1]:.0f}s/无绑定 {k1[0]}、加权迟到 {k1[2]:.0f}、'
                   f'累计推迟 {sum(deltas_a):.0f} s、联合完工 {k1[3]:.0f} s')
            if k2 < k1:
                deltas, assignment = r2['deltas'], r2['assignment']
                relays, skipped, jobs = r2['relays'], r2['skipped'], r2['jobs']
                repair_rounds += r2['rounds']
                print(f'  [v2 固定点] 采用（{r2["reason"]}）：{tag}')
            else:
                _reset_intervals(intervals, snap_a)
                print(f'  [v2 固定点] 未采用（{r2["reason"]}）：{tag}')
    else:
        jobs = merge_jobs(intervals, sel)
        relays, skipped = schedule_relays(d, jobs, cands, verbose=verbose)
    if verbose:
        _n_out, _, _ = q3_chain.outing_stats(relays)
        print(f'中继架次 {_n_out} 个（{len(relays)} 次悬停站服务），'
              f'弃飞 {len(skipped)} 个，迟建链 {sum(1 for x in relays if x["late"])} 个')

    # 8) 中继硬约束自检
    rt = d.relay_type
    usable = (1 - rt['rho']) * rt['E_use']
    # 能量上限按**整次出动**核算：站间接续把多次服务叠在一次出动里，逐行比会
    # 只看单次悬停那一小截，把真正越限的出动放过去。故先按出动编号归并。
    outing_E = {}
    for x in relays:
        outing_E.setdefault(x['outing'], x['E'])
    over_E = ['%s(%.3f kWh)' % (od, E) for od, E in sorted(outing_E.items())
              if E > usable + 1e-9]
    over_H = [x for x in relays if x['hover_agl'] > rt['max_hover_alt'] + 1e-9]
    if over_E or over_H:
        raise RuntimeError('中继出动越限：能耗 %d 个（%s）、悬停高度 %d 个'
                           % (len(over_E), '、'.join(over_E), len(over_H)))
    # 每次出动的能耗必须等于其各次访问的份额之和（末次访问含末端返航段）。
    # 这条恒等式把「按行程核算」与「按访问报账」钉在一起，防止两处口径日后分叉。
    _sum_chk = {}
    for x in relays:
        _sum_chk[x['outing']] = _sum_chk.get(x['outing'], 0.0) + x['visit_E']
    _dev = max((abs(_sum_chk[od] - E) for od, E in outing_E.items()), default=0.0)
    if _dev > 1e-9:
        raise RuntimeError('中继出动能耗与各访问份额之和不符，最大偏差 %.3e kWh' % _dev)
    # 运输侧资源链自检：同一条无人机/共享电池链上不得有占用重叠。这一条是**硬约束**，
    # 而 `audit()` 只看通信分段、看不见它——实测正是这里缺一道关，让「U05 同时飞
    # T17 与 T24（重叠 1583 s）」的排班以「零中断」的名义落了盘，最后由画图脚本
    # `figures_adv.py` 的重建占用断言（`assert n_ov == 0`）才暴露出来。求解器自己
    # 必须先卡住，而不是等下游脚本替它把关。
    conf = chain_conflicts(d, assignment)
    if conf:
        print('!' * 74)
        print('资源链冲突：错峰方案仍有 %d 处无人机/电池占用重叠，方案不可行。' % len(conf))
        for c in conf[:5]:
            print('  %s：T%02d → T%02d 需再推 %.1f s（对方硬时限余量 %.1f s）'
                  % (c['resource'], c['prev'] + 1, c['nxt'] + 1,
                     c['required_delay'], trip_slack(d, assignment)[c['nxt']]))
        raise RuntimeError('错峰方案存在 %d 处资源链冲突，不得作为可行解交付' % len(conf))

    # ok=True 的含义仅限于「选站可行、中继架次不越限、资源链无冲突、排班已给出」。
    # 它**不是**交付侧的可行证书：硬时限是否满足、加权迟到多少、能耗合计几何，都要
    # 由调用方在最终错峰方案上重算（见 main 里的 F05 汇总）。当成 ok 的推论会漏检。
    return dict(ok=True, intervals=intervals, cands=cands, sel=sel, relays=relays,
                skipped=skipped, assignment=assignment, deltas=deltas,
                peak0=peak0, peak1=peak1, uncovered=uncovered, cover=cover,
                repair_rounds=repair_rounds, secs=_time.time() - t0, n_conf=0,
                certificate='选站/排班可行（含资源链无冲突），不含交付侧指标')


def uncovered_note(uncovered):
    if not uncovered:
        return ''
    return (f'  ⚠ {len(uncovered)} 个失效区间在任何高度、任何候选站址下都无法被完整覆盖，'
            f'这些区间属几何上不可达，须在报告中如实列出，不得假称零中断')


def main():
    d = load_data()
    d.geo_nodes, d.geo = precompute_geometry(d)

    # 复用问题二推荐方案（q2.py 实跑后落盘于 results/q2_recommended.json）
    trips, assignment, q2_m, q2_order = recommended(d)
    # 上游**完整方案**：数据/代码哈希必须与落盘时一致，否则这份上游已过期
    # 一并给文本规范化哈希：源码字节不一致时，加载器才能分辨「只是换行被改了」
    # 与「代码真改了」，并各自给出正确处置。缺这一项时两种原因混成一句
    # 「求解器源码已变」，读者只能靠刷新哈希蒙混过去。
    q2_sol = load_solution(Q2_SOL, stage='q2',
                           input_hashes_expect=input_hashes(),
                           solver_hashes_expect=solver_hashes(Q2_SOLVER_FILES),
                           solver_text_hashes_expect=solver_hashes_text(Q2_SOLVER_FILES))
    assignment = sorted(assignment, key=lambda x: x['start'])
    assignment = [dict(a, trip_idx=i) for i, a in enumerate(assignment)]
    # 架次编号**继承**上游方案，并逐架次核对指派未被重解改写
    trip_ids = inherit_trip_ids(assignment, q2_sol)
    base_starts = [a['start'] for a in assignment]
    print(f'上游方案 {q2_sol["solution_id"]}（{len(trip_ids)} 架次，'
          f'{trip_ids[0]}..{trip_ids[-1]}）')
    print('=' * 74)
    print(f'运输调度（问题二推荐方案）: {q2_m["n_trips"]} 架次 / {q2_m["total_E"]:.3f} kWh / '
          f'makespan {q2_m["makespan"]:.0f} s / 硬违反 {q2_m["hard_viol"]}')

    # 对照一：不做通信协同，直连能撑多少
    seq = sequential_baseline(d, assignment)
    print(f'[对照] 无中继的纯运输调度: 直连 {100*seq["direct"]:.2f}% | '
          f'中断 {100*seq["outage"]:.2f}%  ← 违反题给「连续通信」约束')

    # 对照二：序贯做法（固定 300 m、贪心选站、不错峰、不选高度、不做连续精化）
    seq2 = solve_relay(d, assignment, greedy=True, single_hover=True, verbose=False)
    f2 = None
    # 序贯对照的架次/能耗也按出动核（阶段 13 起两档共用同一个排班器）
    _seq2_out, _seq2_E, _ = (q3_chain.outing_stats(seq2['relays']) if seq2['ok']
                             else (0, 0.0, 0))
    if seq2['ok']:
        f2, _, _ = audit(d, assignment, seq2['relays'], seq2['cands'])
        print(f'[对照] 序贯（固定300m + 贪心选站 + 不错峰）: '
              f'直连 {100*f2["direct"]:.2f}% | 中继 {100*f2["relay"]:.2f}% | '
              f'中断 {100*f2["gap"]:.2f}%，中继架次 {_seq2_out}')

    # 联合优化
    print('-' * 74)
    res = solve_relay(d, assignment, verbose=True)
    if not res['ok']:
        raise RuntimeError(res['reason'])
    assignment = res['assignment']
    frac, segs, gaps = audit(d, assignment, res['relays'], res['cands'])
    print(f'[联合优化] 直连 {100*frac["direct"]:.2f}% | 中继 {100*frac["relay"]:.2f}% | '
          f'中断 {100*frac["gap"]:.2f}%')
    if gaps:
        print('!' * 74)
        print(f'零中断未达成：仍有 {frac["gap"]*frac["total_time"]:.0f} s'
              f'（{100*frac["gap"]:.2f}%）处于中断，共 {len(gaps)} 个采样点。')
        print(f'  未保障失效区间 {len(res["uncovered"])} 个；弃飞中继架次 {len(res["skipped"])} 个。')
        print('  以下如实记录，不得在报告中假称零中断：')
        for g in gaps[:10]:
            print(f'    {trip_ids[g["trip"]]} t={g["t"]:.0f}s '
                  f'({g["lon"]:.5f},{g["lat"]:.5f}) 海拔{g["alt"]:.0f}m')
        print('!' * 74)

    # ---- 导出 ----
    stations = {}
    for x in res['relays']:
        stations[x['station']] = x
    st_no = {ci: f'S{k:02d}' for k, ci in enumerate(sorted(stations), 1)}
    st_rows = []
    for k, (ci, x) in enumerate(sorted(stations.items()), 1):
        n_iv = sum(1 for j, c in res['sel'].items() if c == ci)
        st_rows.append(dict(悬停站编号=f'S{k:02d}', 悬停经度=round(x['lon'], 6),
                            悬停纬度=round(x['lat'], 6), 悬停海拔m=round(x['alt_abs'], 3),
                            悬停离地高度m=round(x['hover_agl'], 2),
                            地面高程m=round(x['ground_elev'], 2), 保障失效区间数=n_iv))
    pd.DataFrame(st_rows).to_csv(os.path.join(OUT, 'q3_stations.csv'), index=False)

    # ---- 完整最终方案落盘（先落盘，再由它派生全部 CSV：编号只有一个来源）----
    joint_done = max([x['return_t'] for x in res['relays']]
                     + [a['start'] + a['duration'] for a in assignment])
    # ---- F05：由**最终错峰方案**重算的一组交付侧指标 -------------------------
    # 旧版只打印联合完工时刻，其余（加权迟到、迟到箱数、最长迟到、最小硬时限
    # 余量）全靠调用方自己再算一遍，`solve_relay` 的 ok=True 也被当成完整可行的
    # 证书——实际它只保证「覆盖可行 + 零中断」，不含硬时限。这里一次性算清并落盘。
    #
    # 数据源是**错峰后的 assignment 本身**（不是 q3_sol）：q3_sol 在下面才由
    # build_q3_solution 造出来，此处引用它只会拿到一个未绑定的名字。两者同源——
    # build_q3_solution 的 transport_trips 正是逐字段抄自这个 assignment。
    idx_box = {b['id']: b for bs in d.boxes_by_service.values() for b in bs}
    # 加权迟到只留一份实现（core.weighted_tardiness）。此前这里与搜索里的退火路径
    # 各写一遍同一个公式，两处一旦有一处改了就是"同一指标两个值"；口径归一到
    # core 后，本文件只负责把结果落盘。verify.py 里**另有**一份独立重算——那是
    # 刻意保留的第二实现，不复用 core，用作复核。
    w_tard, n_late, max_late = weighted_tardiness(d, assignment, idx_box)
    min_margin = float('inf')
    min_margin_box = ''
    for t in assignment:
        for _sid, boxes in t['boxes_at'].items():
            for bx in boxes:
                bid = bx['id']
                b = idx_box[bid]
                t_del = t['deliver_abs'][bid]
                lim = []
                if b['category'] == '医疗物资' and b['expect'] is not None:
                    lim.append(b['expect'])
                if b['first_batch'] and b['deadline'] is not None:
                    lim.append(b['deadline'])
                if lim:
                    mg = min(lim) - t_del
                    if mg < min_margin:
                        min_margin, min_margin_box = mg, bid
    relay_done = max([x['return_t'] for x in res['relays']], default=0.0)
    trans_done = max(a['start'] + a['duration'] for a in assignment)
    # 中继侧的两个报告口径一律按**出动**核（阶段 13 起一次出动可服务多个站）：
    # 一排一行 = 一次悬停站服务，出动级的 E 在该出动的每一行上重复出现。
    n_out, relay_E, n_chain = q3_chain.outing_stats(res['relays'])
    q3_m = dict(direct=frac['direct'], relay=frac['relay'], gap=frac['gap'],
                total_time=frac['total_time'],
                n_relays=n_out, n_relay_visits=len(res['relays']), n_chain=n_chain,
                n_stations=len(stations),
                relay_energy_kwh=relay_E,
                skipped=len(res['skipped']), uncovered=len(res['uncovered']),
                joint_done=joint_done,
                makespan=trans_done,
                transport_energy_kwh=sum(a['E'] for a in assignment),
                weight_tardiness=w_tard, n_late_boxes=n_late, max_late_s=max_late,
                min_hard_margin_s=(0.0 if min_margin == float('inf') else min_margin),
                min_hard_margin_box=min_margin_box,
                transport_done_s=trans_done, relay_done_s=relay_done,
                joint_energy_kwh=(sum(a['E'] for a in assignment) + relay_E),
                n_staggered=sum(1 for x in res['deltas'] if x > 0),
                total_delay_s=sum(res['deltas']))
    if q3_m['min_hard_margin_s'] < -1e-6:
        raise AssertionError('最终方案存在硬时限违反：最小余量 %.6f s（%s）'
                             % (q3_m['min_hard_margin_s'], min_margin_box))
    q3_sol = build_q3_solution(d, q2_sol, assignment, res['relays'], res['intervals'],
                               st_no, q3_m, seed=0,
                               budget=dict(dt_search=DT_SEARCH, dt_audit=DT_AUDIT,
                                           margin_db=MARGIN_DB))
    # 外键校验：每条失效区间都必须解析到本方案内的运输架次与中继架次；空引用显式失败
    check_foreign_keys(q3_sol, q3_sol['intervals'], require_relay_binding=True)
    q3_id = save_solution(Q3_SOL, q3_sol)
    print(f'已落盘完整方案 {os.path.basename(Q3_SOL)}（solution_id={q3_id}，'
          f'来源 {q2_sol["solution_id"]}）')

    relay_recs = q3_sol['relay_trips']
    rid_by_index = {i: r['relay_trip_id'] for i, r in enumerate(relay_recs)}
    relay_xy = {x['station']: x for x in res['relays']}

    # 中继架次表：**一行 = 一次悬停站服务**。这是唯一的原子口径——`q3_intervals.csv`
    # 的区间绑定、`q3_comm_phases.csv` 的中继编号、以及 `verify.check_relay_final`
    # 的逐秒重判都以它为单位，因此不能改成「一行一次出动」。
    #
    # 自阶段 13 起一次出动可依次服务多个站，于是列分两层：
    #   出动级（同一次出动的各行取同值）：出动编号 / 中继无人机编号 / 能源组件编号 /
    #       开始时刻s（离开 O01）/ 返回O01时刻s / 架次能耗kWh（**整次出动**的能耗）
    #   访问级：中继架次编号 / 架次内序 / 悬停站与其坐标 / 建链完成时刻s / 服务结束时刻s /
    #       本访问能耗kWh（本次进场航段 + 本站悬停；该次出动的末次访问另含末端返航）
    # `架次能耗kWh` 是读表时**要按出动编号去重**的那个量；`本访问能耗kWh` 逐行相加
    # 才等于它。二者由 verify.py 分别独立复核（去重后的上限、逐行求和的一致性）。
    rt_rows = []
    for r in relay_recs:
        x = next(y for y in res['relays'] if st_no.get(y['station'], '') == r['station_id']
                 and abs(y['start'] - r['start']) <= 1e-9)
        rt_rows.append(dict(
            中继架次编号=r['relay_trip_id'], 出动编号=r['outing_id'],
            架次内序=r['seq_in_outing'],
            中继无人机编号=r['relay_uav_id'],
            能源组件编号=r['component_id'], 悬停站编号=r['station_id'],
            开始时刻s=round(r['start'], T_DEC),
            悬停经度=round(x['lon'], 6), 悬停纬度=round(x['lat'], 6),
            # 海拔留 3 位：悬停离地高度按上限取值时，只留 1 位会把 813.1891 舍成
            # 813.2，用表里数值回算离地高度就成了 300.011 m，看上去像越限。
            悬停海拔m=round(x['alt_abs'], 3), 建链完成时刻s=round(r['link_done'], T_DEC),
            服务结束时刻s=round(r['service_end'], T_DEC),
            返回O01时刻s=round(r['return_time'], T_DEC),
            架次能耗kWh=round(r['energy_kwh'], E_DEC),
            本访问能耗kWh=round(r['visit_energy_kwh'], E_DEC)))
    pd.DataFrame(rt_rows).to_csv(os.path.join(OUT, 'q3_relay_trips.csv'), index=False)

    # 通信分段表：编号一律取自方案，不按 `s['trip']+1` 现推。段边界是共同边界
    # 序列（同一架次内后段起点 == 前段终点），时长与终检的时间权重同源。
    comm_rows = []
    for s in segs:
        comm_rows.append(dict(运输架次编号=trip_ids[s['trip']], 通信阶段=s['phase'],
                              开始时刻s=round(s['t1'], T_DEC), 结束时刻s=round(s['t2'], T_DEC),
                              段时长s=round(s['t2'] - s['t1'], T_DEC),
                              采样点数=s['n_pts'],
                              保障方式=s['mode'], 中继架次编号=s['rid']))
    pd.DataFrame(comm_rows).to_csv(os.path.join(OUT, 'q3_comm_phases.csv'), index=False)
    if comm_rows:
        nobind = sum(1 for r in comm_rows if r['保障方式'] == '中继' and not r['中继架次编号'])
        assert nobind == 0, f'{nobind} 段中继没有绑定具体中继架次编号'
    ph_errs = check_phase_table(segs)
    if ph_errs:
        raise AssertionError('通信分段表不自洽：' + '；'.join(ph_errs[:3]))

    # 交付侧汇总表（F05）：一行一指标，均由最终方案重算，供论文与 Excel 直接取用
    m_rows = [
        ('直连时间占比', frac['direct'], '—'), ('中继时间占比', frac['relay'], '—'),
        ('中断时间占比', frac['gap'], '—'), ('轨迹总时长', frac['total_time'], 's'),
        ('运输完工时刻', q3_m['transport_done_s'], 's'),
        ('中继最晚返回时刻', q3_m['relay_done_s'], 's'),
        ('联合完工时刻', q3_m['joint_done'], 's'),
        ('运输能耗', q3_m['transport_energy_kwh'], 'kWh'),
        ('中继能耗', q3_m['relay_energy_kwh'], 'kWh'),
        ('合计能耗', q3_m['joint_energy_kwh'], 'kWh'),
        ('加权迟到', q3_m['weight_tardiness'], '权重·s'),
        ('迟到箱数', q3_m['n_late_boxes'], '个'),
        ('最长迟到', q3_m['max_late_s'], 's'),
        ('最小硬时限余量', q3_m['min_hard_margin_s'], 's'),
        ('错峰推迟架次数', q3_m['n_staggered'], '个'),
        ('累计推迟量', q3_m['total_delay_s'], 's'),
        # 阶段 13 起「中继架次」= 出动数（飞机出去几趟），与「悬停站服务次数」分开报：
        # 一次出动可依次服务多个站，两个数是不同的量，混用会把接续算成额外架次。
        ('中继架次', q3_m['n_relays'], '个'),
        ('悬停站服务次数', q3_m['n_relay_visits'], '次'),
        ('站间接续次数', q3_m['n_chain'], '次'),
        ('悬停站', q3_m['n_stations'], '个'),
        ('弃飞中继架次', q3_m['skipped'], '个'), ('未保障失效区间', q3_m['uncovered'], '个'),
    ]
    pd.DataFrame([dict(指标=k, 数值=round(float(v), 6), 单位=u) for k, v, u in m_rows]
                 ).to_csv(os.path.join(OUT, 'q3_metrics.csv'), index=False)

    # 失效区间级映射表：运输架次 ↔ 通信失效区间 ↔ 具体中继架次 ↔ 悬停站。
    # 问题四直接读这张表，不再自己重跑一遍中继选站——消除双份真相源。
    iv_rows = []
    for rec in q3_sol['intervals']:
        iv_rows.append(dict(失效区间编号=rec['interval_id'], 运输架次编号=rec['trip_id'],
                            区间起点s=round(rec['start'], T_DEC),
                            区间终点s=round(rec['end'], T_DEC),
                            悬停站编号=next((r['station_id'] for r in relay_recs
                                         if r['relay_trip_id'] == rec['relay_trip_id']), ''),
                            中继架次编号=rec['relay_trip_id']))
    pd.DataFrame(iv_rows).to_csv(os.path.join(OUT, 'q3_intervals.csv'), index=False)

    # 最终运输方案（错峰后）：编号来自方案；机型/实体/电池/逐箱交付一并落盘，
    # 问题四只读这张表与外键表，不再调用问题二的调度器
    tt_rows, bd_rows = [], []
    for t in q3_sol['transport_trips']:
        tt_rows.append(dict(
            架次编号=t['trip_id'], 无人机编号=t['uav'], 机型编号=t['type'],
            电池编号=t['battery'], 原开始时刻s=round(t['base_start'], T_DEC),
            开始时刻s=round(t['start'], T_DEC), 返回O01时刻s=round(t['return_time'], T_DEC),
            访问服务区顺序='->'.join(t['route']), 架次能耗kWh=round(t['energy_kwh'], E_DEC),
            服务区数=len(t['box_ids_by_service']),
            货箱数=sum(len(v) for v in t['box_ids_by_service'].values())))
        for sid, bids in sorted(t['box_ids_by_service'].items()):
            for bid in bids:
                b = idx_box[bid]
                t_del = t['deliver_abs'][bid]
                lim = []
                if b['category'] == '医疗物资' and b['expect'] is not None:
                    lim.append(b['expect'])
                if b['first_batch'] and b['deadline'] is not None:
                    lim.append(b['deadline'])
                bd_rows.append(dict(
                    货箱编号=bid, 架次编号=t['trip_id'], 服务区编号=sid,
                    类别=b['category'], 优先系数=b['priority'],
                    是否首批=('是' if b['first_batch'] else '否'),
                    期望送达时刻s=(None if b['expect'] is None else round(b['expect'], T_DEC)),
                    硬时限时刻s=(None if not lim else round(min(lim), T_DEC)),
                    交付完成时刻s=round(t_del, T_DEC),
                    硬时限余量s=(None if not lim else round(min(lim) - t_del, T_DEC)),
                    是否迟到=('是' if (b['expect'] is not None
                                   and t_del > b['expect'] + 1e-6) else '否')))
    pd.DataFrame(tt_rows).to_csv(os.path.join(OUT, 'q3_transport_trips.csv'), index=False)
    pd.DataFrame(bd_rows).to_csv(os.path.join(OUT, 'q3_box_delivery.csv'), index=False)
    if len(bd_rows) != sum(len(v) for t in q3_sol['transport_trips']
                           for v in t['box_ids_by_service'].values()):
        raise AssertionError('逐箱交付表行数与方案不符')

    # 覆盖率汇总（时间占比口径）
    cov_rows = [dict(口径='联合优化', 采样步长s=DT_AUDIT,
                     直连时间占比=round(frac['direct'], 6), 中继时间占比=round(frac['relay'], 6),
                     中断时间占比=round(frac['gap'], 6), 总时长s=round(frac['total_time'], 1),
                     中继架次=n_out, 悬停站=len(stations),
                     中继能耗kWh=round(relay_E, E_DEC)),
                dict(口径='序贯对照', 采样步长s=DT_AUDIT,
                     直连时间占比=round(f2['direct'], 6) if seq2['ok'] else None,
                     中继时间占比=round(f2['relay'], 6) if seq2['ok'] else None,
                     中断时间占比=round(f2['gap'], 6) if seq2['ok'] else None,
                     总时长s=round(f2['total_time'], 1) if seq2['ok'] else None,
                     中继架次=_seq2_out if seq2['ok'] else None,
                     悬停站=len({x['station'] for x in seq2['relays']}) if seq2['ok'] else None,
                     中继能耗kWh=round(_seq2_E, E_DEC) if seq2['ok'] else None),
                dict(口径='无中继', 采样步长s=DT_AUDIT,
                     直连时间占比=round(seq['direct'], 6), 中继时间占比=0.0,
                     中断时间占比=round(seq['outage'], 6), 总时长s=round(seq['total_time'], 1),
                     中继架次=0, 悬停站=0, 中继能耗kWh=0.0)]
    pd.DataFrame(cov_rows).to_csv(os.path.join(OUT, 'q3_coverage.csv'), index=False)

    sg = [dict(架次编号=trip_ids[i], 原开始时刻s=round(base_starts[i], T_DEC),
               错峰开始时刻s=round(assignment[i]['start'], T_DEC),
               推迟s=round(res['deltas'][i], T_DEC))
          for i in range(len(assignment))]
    pd.DataFrame(sg).to_csv(os.path.join(OUT, 'q3_stagger.csv'), index=False)

    print('-' * 74)
    print(f'悬停站 {len(stations)} 个；中继架次 {n_out} 个'
          f'（其中站间接续 {n_chain} 次，共 {len(res["relays"])} 次悬停站服务）；'
          f'中继总能耗 {relay_E:.3f} kWh；弃飞 {len(res["skipped"])} 个')
    print(f'交付侧：加权迟到 {q3_m["weight_tardiness"]:.1f} 权重·s'
          f'（迟到箱 {q3_m["n_late_boxes"]} 个，最长 {q3_m["max_late_s"]:.0f} s）；'
          f'最小硬时限余量 {q3_m["min_hard_margin_s"]:.0f} s'
          f'（{q3_m["min_hard_margin_box"]}）')
    print(f'能耗：运输 {q3_m["transport_energy_kwh"]:.3f} + 中继 '
          f'{q3_m["relay_energy_kwh"]:.3f} = 合计 {q3_m["joint_energy_kwh"]:.3f} kWh')
    print(f'// 运输完工 {q3_m["transport_done_s"]:.0f} s | '
          f'中继最晚返回 {q3_m["relay_done_s"]:.0f} s | '
          f'联合完工 {q3_m["joint_done"]:.0f} s')
    print(f'// 通信分段表：{len(comm_rows)} 段，共同边界校验通过'
          f'（相邻边界逐位相等、无空洞、段时长与时间权重同源）')
    print(f'// 资源链冲突按类型取释放时刻（电池含充电），'
          f'错峰推迟 {q3_m["n_staggered"]} 个架次、累计 {q3_m["total_delay_s"]:.0f} s')
    print(f'已导出 q3_solution.json / q3_transport_trips.csv / q3_box_delivery.csv / '
          f'q3_relay_trips.csv / q3_intervals.csv / q3_comm_phases.csv / q3_stations.csv / '
          f'q3_coverage.csv / q3_metrics.csv / q3_stagger.csv '
          f'（用时 {res["secs"]:.1f} s）')
    if gaps:
        # 结果表已落盘供诊断，但进程以非零码退出：零中断是本问的硬约束，
        # 未达成时绝不能让流水线当作成功继续往下走。
        raise SystemExit(f'零中断未达成：中断时间占比 {100*frac["gap"]:.3f}%')


if __name__ == '__main__':
    main()
