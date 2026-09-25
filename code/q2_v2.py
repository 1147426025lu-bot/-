# -*- coding: utf-8 -*-
# 本程序及代码是在人工智能工具辅助下完成的。
# 工具名称：DeepSeek；版本/型号：DeepSeek V4.1 Flash（API 模型名 deepseek-flash）；
# 开发机构：杭州深度求索人工智能基础技术研究有限公司；
# 版本发布日期：2026 年 9 月 10 日。

"""
问题二 v2 引擎：精确组批 + 瓶颈感知解码 + 多种子档间热启动。

本模块来自对一份外部解法包的复核与移植。**外部实现的物理模块 (core.py) 与我方
不同**（它逐点线采样求航段最高地面高程，我方是 Amanatides–Woo 像元遍历），
线采样会漏掉两像元之间的山脊、巡航高度偏低、爬降时间偏短，故它自报的完成时间
不能直接搬。本模块只取它的**算法层**，全部指标在我方 core 下重算。

外部函数 → 本模块函数对照（便于逐行核对；外部包已不随交付分发）：

    q2_opt.exact_pack            -> exact_pack
    q2_opt.exact_pack_mixed      -> exact_pack_mixed
    q2_opt.optimize_dispatch_order -> optimize_dispatch_order
    q2_alns.PlanCache            -> TripRouteCache
    q2_alns.decode 的选型项       -> q2.schedule(dispatch='balanced')（记账只留一份）
    q2_alns 的多种子/档间热启动    -> solve_recommended_v2

**不移植**的两处及理由：
  1. q2_opt.construct_trips_greedy —— 与我方 construct_trips 作用重叠，且其文件头
     docstring 声称的紧迫度项在代码里并未实现（tau 是死参数）。组批改用下面的
     精确子集 DP + 我方既有 merge_nonurgent。
  2. q2_alns 的「不可行解一律不接受」与「只留末端最优」—— 我方 alns 有更合适的
     兜底接受准则，且**保留可行档案**（外部没有档案，搜索途中路过但未成终态的
     可行解会丢）。这里复用 q2.alns 的档案，不放宽。

对外只暴露 solve_recommended_v2()，返回与 q2.solve_recommended 相同的四元组。
"""
import itertools
import math
import os
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')

import q2                                                          # noqa: E402
from core import load_data, segment_time, segment_energy, max_safe_payload  # noqa: E402

# 同一进程里只允许存在一份 q2 状态。GEO_EPOCH 与 _TRIP_CACHE 都是模块级全局，
# 「一个进程一份几何」这个前提就是靠它们成立的；若 `python code/q2.py --engine v2`
# 时本模块再 `import q2` 拿到第二份副本，就会有两个纪元、两份备忘，前提当场失效，
# 而且失效是静默的。这里把 q2 重绑到 __main__ 上那一份，从根上掐掉。
_main = sys.modules.get('__main__')
if (_main is not None and _main is not sys.modules.get(__name__)
        and all(hasattr(_main, a) for a in ('GEO_EPOCH', 'precompute_geometry', 'alns'))):
    q2 = _main

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'results')

TYPES = ('A', 'B', 'C')
# 路线全排列枚举的服务区数上限：5! = 120，再大就只取「按硬时限/近邻」的固定序。
PERM_MAX_STOPS = 5

# 机队规模（架）。选型代价按机队份额计价，见 _fleet_cost。
FLEET_N = {'A': 4, 'B': 2, 'C': 2}
# 选型代价里「时长」的权重 λ，单位 kWh/(机队·秒)。0 = 退化为纯能耗最小。
FLEET_LAMBDA = 0.0


def _fleet_cost(g, E, T):
    """选型代价 E + λ·T/n_g（kWh）。

    **为什么机队数要进分母。** 一架次占用的是该机型 `时长/n_g` 的「机队秒」，
    而完工时刻由最瓶颈的那个机型决定。原键只比 E，而 C 型每箱公里最省电
    （载 80 kg、L0=26000、8 kWh），于是架次构造器把 40% 的总时长堆到只占
    25% 机队的 2 架 C 机上：U07 从 0 s 飞到 8732.6 s 一秒不停，4 架 A 机
    4000 s 之后全部闲置——完工时刻被 C 机的串行时长钉死，与总工作量无关。

    这与 `q2.schedule(dispatch='balanced')` 的 rank_bonus 是同一个思想，但作用在
    **选型层**；派发层改不了选型，故那一层补不了这个缺口。

    λ=0 时本函数逐位等于旧的「能耗最小」，旧行为可复现。

    ⚠️ **λ>0 已实测被否，默认必须保持 0.0；不要重试。** 构造层单独跑时 λ=0.002
    把完工从 12597.3 s 压到 9794.7 s（-22%），看着很像收益；但进了完整引擎
    （`_probe/q2_v2_ab.py`，2 种子 × 1800 轮、全档链、档间热启动，唯一变量 λ）：

        λ=0     25 架次 / 71.057 kWh / 6550.9 s / 时延 0（K<=25 即达标）
        λ=0.002 26 架次 / 75.985 kWh / 7606.8 s / 时延 0（要放开到 K<=40 才达标）
        → +1055.9 s（+16.12%）、+4.928 kWh、+1 架次

    原因是**方向反了**：罚 C 逼着构造器改用更小的 A/B 架次，架次数变多、每架次
    的 prep（300 s）跟着多付一遍，Σ时长反而上升——而完工时刻的下界正是 Σ时长/8
    （λ=0 时为 46477.7/8 = 5809.7 s = 96.83 min）。C 型之所以被堆满，是因为它
    **每箱公里最省电、装得最多、架次最少、prep 付得最少**；这是效率的副产品，
    不是失衡的错误，用「少用 C」去纠它，付出的 Σ时长 比纠回来的失衡更多。

    构造层的 -22% 是假信号：那里没有 ALNS。ALNS 的修复算子按纯能耗选型（λ 进不去
    `q2.py`），它会把构造器让出的 C 又抢回来，于是 λ 只留下「初始解变差」这一半。

    **真正的下界不是 Σ时长/8，而要看机队是按机型锁死的。** 拿落盘的 20 架次方案
    实测：Σ时长 = 41726.8 s → Σ/8 = 5215.9 s = 86.93 min，可它实际完工 8732.6 s
    ——因为 U07（C 型）从 0 s 到 8732.6 s 一秒不停，完工时刻**等于**它的忙时，
    与 Σ/8 差了 3516.7 s。逐机占用：U07 100.0%、U08 89.7%、U06 87.6%、U05 65.4%、
    U01 42.1%、U02 46.3%、U03 23.7%、U04 23.0%。U01–U04 是 A 型机，**飞不了 C 的
    架次**：机型在机队里是固定的（4A+2B+2C），而方案把架次排成 6A:7B:7C，等于让
    每架 C 机摊 3.5 个架次、每架 A 机只摊 1.5 个。

    所以这个缺口由 `max_g(该机型总忙时 / 该机型机数)` 决定，而不是 Σ时长/8 决定。
    推论有两条，都别搞反：
      · 派发层（含下面的退火）**改不动**它——U07 已经 100% 忙，再怎么重排也压不下
        它的忙时，只能改选型；
      · 但要压的是**按机型分组的最大值**，不是加权总和。λ 是逐架次的求和罚项，
        压不出「最大值」这个形状——这与它被实测否掉是同一件事的两面，不是巧合。
    真正对形状的是「机型配比」，而它的代价也真实存在：A 型只载 25 kg（C 型 80 kg），
    把 C 的货挪给 A 要拆成更多架次，每架次多付一遍 prep。**这是一条诊断，不是已验证
    的改进**；动它之前必须先算清拆架次的代价，别重蹈 λ 的覆辙。
    """
    return E + FLEET_LAMBDA * T / FLEET_N[g]


def set_fleet_lambda(v):
    """设置选型代价权重，**返回被覆盖的旧值**。

    缓存按 λ 分代（见 TripRouteCache），故改 λ 不会读到旧解。返回旧值而非新值，
    是为了让调用方能写 `saved = set(v)` … `finally: set(saved)` 复位：λ 是模块级
    状态，`solve_recommended_v2` 返回后调用方还要跑消融，而消融第⑤行同样调
    `construct_trips_exact`，不复位就会带着上一轮的 λ 计算——它是要与第④行逐项
    对照的。这类残留不报错、只改数。
    """
    global FLEET_LAMBDA
    prev = FLEET_LAMBDA
    FLEET_LAMBDA = float(v)
    return prev


def _is_urgent(b):
    return q2._is_urgent(b)


_box_index = q2._box_index       # 货箱 id → 货箱，只有一份实现


# ---------------------------------------------------------------------------
# A1 精确组批：区内可行子集枚举 + 子集 DP
# ---------------------------------------------------------------------------
def _subsets_of(d, t, sid, boxes, qcap, vcap, t_fixed, t_per_box, e_cache):
    """枚举全部可行子集，返回 {mask: dict(mass, vol, n, E, T)}。

    只对**单区往返**计价：组批阶段不决定访问顺序，跨区合并交给 merge_nonurgent，
    故这里的 E/T 是「这批箱单独跑一趟该区」的能耗与时间。
    """
    go = d.geo[('O01', sid)]
    back = d.geo[(sid, 'O01')]
    masses = [b['mass'] for b in boxes]
    vols = [b['volume'] for b in boxes]
    subs = {}

    def trip_energy(m):
        if m not in e_cache:
            e_cache[m] = segment_energy(t, go, m) + segment_energy(t, back, 0.0)
        return e_cache[m]

    def dfs(start, mask, m, v, cnt):
        if mask:
            subs[mask] = dict(mass=m, vol=v, n=cnt, g=t['id'],
                              E=trip_energy(m), T=t_fixed + t_per_box * cnt)
        for j in range(start, len(masses)):
            nm = m + masses[j]
            nv = v + vols[j]
            if nm <= qcap and nv <= vcap:
                dfs(j + 1, mask | (1 << j), nm, nv, cnt + 1)

    dfs(0, 0, 0.0, 0.0, 0)
    return subs


def _subset_dp(subs, n):
    """Bellman 状态压缩：dp[mask] = 恰好覆盖 mask 的最优 (批次数, 能耗和, 时间和)。

    转移固定取「含 mask 中最小编号货箱」的可行子集，故不重不漏。
    候选稀疏时扫描含该箱的子集表，稠密时枚举子掩码——两条路等价，只是常数不同。
    """
    contain = [[] for _ in range(n)]
    for mk in subs:
        mm = mk
        while mm:
            lb = mm & -mm
            contain[lb.bit_length() - 1].append(mk)
            mm ^= lb
    N = 1 << n
    INF = 10 ** 9
    dn = [INF] * N
    dC = [0.0] * N
    dE = [0.0] * N
    dT = [0.0] * N
    choice = [0] * N
    dn[0] = 0
    for mask in range(1, N):
        lsb = mask & -mask
        k = lsb.bit_length() - 1
        rest = mask ^ lsb
        not_mask = (~mask) & (N - 1)
        n_submask = 1 << (mask.bit_count() - 1)
        best_key = None
        best = None
        if len(contain[k]) < n_submask:
            for f in contain[k]:
                if f & not_mask:
                    continue
                c = mask ^ f
                if dn[c] >= INF:
                    continue
                s = subs[f]
                key = (dn[c] + 1, dC[c] + _fleet_cost(s['g'], s['E'], s['T']),
                       dE[c] + s['E'], dT[c] + s['T'])
                if best_key is None or key < best_key:
                    best_key, best = key, f
        else:
            sub = rest
            while True:
                f = sub | lsb
                s = subs.get(f)
                if s is not None:
                    c = mask ^ f
                    if dn[c] < INF:
                        key = (dn[c] + 1, dC[c] + _fleet_cost(s['g'], s['E'], s['T']),
                               dE[c] + s['E'], dT[c] + s['T'])
                        if best_key is None or key < best_key:
                            best_key, best = key, f
                if sub == 0:
                    break
                sub = (sub - 1) & rest
        if best is None:
            raise RuntimeError('精确组批 DP 在 mask=%d 上无可行转移' % mask)
        choice[mask] = best
        dn[mask] = best_key[0]
        dC[mask] = best_key[1]
        dE[mask] = best_key[2]
        dT[mask] = best_key[3]

    out = []
    msk = N - 1
    while msk:
        f = choice[msk]
        s = subs[f]
        out.append(([i for i in range(n) if f >> i & 1], s))
        msk ^= f
    return out


def exact_pack(d, t, sid, boxes, qmax, n_max=16):
    """单机型精确装箱。字典序：批次数 → 能耗和 → 时间和。

    区箱数 > n_max 或存在单箱超限时回退 FFD（保证鲁棒，不因规模爆炸而阻塞）。
    返回 list of (boxes_sublist, mass, vol)。
    """
    n = len(boxes)
    if n > n_max or max(b['mass'] for b in boxes) > qmax + 1e-9:
        return [([boxes[i] for i in idxs], m, v)
                for idxs, m, v in q2._pack_ffd(boxes, qmax, t['volume'])]
    go = d.geo[('O01', sid)]
    back = d.geo[(sid, 'O01')]
    t_fixed = (t['prep'] + segment_time(t, go) + segment_time(t, back)
               + t['hand_base'])
    subs = _subsets_of(d, t, sid, boxes, qmax + 1e-7, t['volume'] + 1e-9,
                       t_fixed, t['load_per_box'] + t['hand_per_box'], {})
    out = []
    for idxs, s in _subset_dp(subs, n):
        out.append(([boxes[i] for i in idxs], s['mass'], s['vol']))
    return out


def exact_pack_mixed(d, sid, boxes, n_max=16):
    """多机型混合精确装箱。

    合并 A/B/C 三种机型的可行子集，按 mask 去重保留 (E, T) 最优的机型版本，再在
    合并候选池上跑子集 DP。重载子集自然由 C 型承载（A/B 装不下），轻载子集则可能
    由 A/B 型以更低单位航程能耗承载，在不增加批次数的前提下压低能耗与时间。
    返回 list of (boxes_sublist, mass, vol, type)。
    """
    n = len(boxes)
    tC = d.transport_types['C']
    if n > n_max or n == 0:
        qmax = min(tC['Q'], max_safe_payload(d, tC, d.si[sid]))
        return [(bs, m, v, 'C')
                for bs, m, v in q2._pack_ffd(boxes, qmax, tC['volume'])]

    best_by_mask = {}
    for g in TYPES:
        t = d.transport_types[g]
        qmax = min(t['Q'], max_safe_payload(d, t, d.si[sid]))
        if qmax + 1e-7 < max(b['mass'] for b in boxes):
            continue                      # 该机型有单箱超限，整型退出
        go = d.geo[('O01', sid)]
        back = d.geo[(sid, 'O01')]
        t_fixed = (t['prep'] + segment_time(t, go) + segment_time(t, back)
                   + t['hand_base'])
        subs = _subsets_of(d, t, sid, boxes, qmax + 1e-7, t['volume'] + 1e-9,
                           t_fixed, t['load_per_box'] + t['hand_per_box'], {})
        for mask, s in subs.items():
            prev = best_by_mask.get(mask)
            # 跨机型取舍用机队代价而不是纯能耗：见 _fleet_cost。λ=0 时两者逐位相同。
            if prev is None or _fleet_cost(g, s['E'], s['T']) < _fleet_cost(
                    prev['type'], prev['E'], prev['T']):
                best_by_mask[mask] = dict(s, type=g)

    if not best_by_mask:
        raise RuntimeError('%s: 三种机型均无法承载任一货箱' % sid)
    out = []
    for idxs, s in _subset_dp(best_by_mask, n):
        out.append(([boxes[i] for i in idxs], s['mass'], s['vol'], s['type']))
    return out


def construct_trips_exact(d, pack_mode='mixed'):
    """精确组批的初始解，返回与 q2.construct_trips 完全同形的 list of trip dict。

    pack_mode='mixed' 用多机型混合精确装箱；'single' 只用 pack_type='C' 的
    单机型精确装箱。紧急货箱（医疗 + 首批）每区单独成架次，与基线同规则——
    这条是硬时限的保证，不宜交给组批搜索去试探。
    """
    tC = d.transport_types['C']
    trips = []
    for sid in d.S:
        urgent = [b for b in d.boxes_by_service[sid] if _is_urgent(b)]
        if urgent:
            mass = sum(b['mass'] for b in urgent)
            vol = sum(b['volume'] for b in urgent)
            p0 = q2.trip_plan(d, tC, [sid], {sid: urgent})
            trips.append(dict(route=[sid], boxes_at={sid: urgent}, mass=mass, vol=vol,
                              _dur=p0['duration'], urgent=True))
    rest = []
    for sid in d.S:
        boxes = [b for b in d.boxes_by_service[sid] if not _is_urgent(b)]
        if not boxes:
            continue
        if pack_mode == 'mixed':
            packed = exact_pack_mixed(d, sid, boxes)
        else:
            qmax = max_safe_payload(d, tC, d.si[sid])
            packed = [(bs, m, v, 'C')
                      for bs, m, v in exact_pack(d, tC, sid, boxes, min(tC['Q'], qmax))]
        for bs, mass, vol, g in packed:
            p0 = q2.trip_plan(d, d.transport_types[g], [sid], {sid: bs})
            rest.append(dict(route=[sid], boxes_at={sid: bs}, mass=mass, vol=vol,
                             _dur=p0['duration'], urgent=False))
    trips.extend(q2.merge_nonurgent(d, rest, tC))
    return trips


# ---------------------------------------------------------------------------
# A2 解码层：路线全排列缓存
# ---------------------------------------------------------------------------
class TripRouteCache:
    """键 (几何纪元, frozenset(箱id), 机型) -> 最优 (route, plan)。

    路线选择：含硬时限箱 -> (最晚交付偏移, 能耗, 时长)；否则 (能耗, 时长, 交付偏移)。
    含硬时限的架次优先压缩最晚交付时刻，其余优先压能耗——与 q2 的紧迫度口径一致。

    键里带 GEO_EPOCH：缓存本身**不含几何**，若几何被重算而纪元不变，这里会给出
    按旧 DEM 算出的路线与能耗，是最难查的一类静默错误。见 q2.precompute_geometry。

    集合迭代一律 sorted()：字符串哈希受 PYTHONHASHSEED 影响，不排序会让
    「同种子不同进程得到不同答案」，直接摧毁可复现性。
    """

    def __init__(self, d):
        self.d = d
        self.boxes = _box_index(d)
        self.cache = {}
        self.epoch = q2.GEO_EPOCH
        # λ 也是对局量：同一 (箱集合, 机型) 在 λ 不同时最优路线可能不同，故入键。
        # 与 GEO_EPOCH 同理——键漏了自变量，改了参数却读到旧解，是最难查的一类错。
        self.lam = FLEET_LAMBDA

    def boxes_at_of(self, bids):
        ba = {}
        for bid in sorted(bids):
            b = self.boxes[bid]
            ba.setdefault(b['service'], []).append(b)
        return ba

    def get(self, bids, g):
        key = (self.epoch, self.lam, bids, g)
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        d = self.d
        t = d.transport_types[g]
        ba = self.boxes_at_of(bids)
        services = sorted(ba)
        mass = sum(b['mass'] for v in ba.values() for b in v)
        vol = sum(b['volume'] for v in ba.values() for b in v)
        hard = any(q2._is_urgent(self.boxes[bid]) for bid in bids)
        best = None
        if mass <= t['Q'] + 1e-9 and vol <= t['volume'] + 1e-9:
            perms = (itertools.permutations(services) if len(services) <= PERM_MAX_STOPS
                     else [tuple(services)])
            for perm in perms:
                feas, plan = q2.type_feasible(d, t, list(perm), ba)
                if not feas:
                    continue
                last = max(plan['deliver'].values())
                c = _fleet_cost(g, plan['E'], plan['duration'])
                k = ((last, c, plan['E']) if hard else (c, plan['E'], last))
                if best is None or k < best[0]:
                    best = (k, list(perm), plan)
        out = (best[1], best[2]) if best else None
        self.cache[key] = out
        return out

    def best_any(self, bids):
        """跨机型最优 (机型, route, plan)，按机队代价（λ=0 即能耗）；无可行返回 None。"""
        best = None
        for g in TYPES:
            r = self.get(bids, g)
            if r is None:
                continue
            route, plan = r
            k = (_fleet_cost(g, plan['E'], plan['duration']),
                 plan['E'], plan['duration'])
            if best is None or k < best[0]:
                best = (k, g, route, plan)
        return (best[1], best[2], best[3]) if best else None


def refine_routes(d, trips, cache):
    """把每个架次的路线换成「该箱集合下最优路线」；无可行者保持原路线。

    路线按**能耗最低的可行机型**选取。schedule(dispatch='balanced') 可能改派别的
    机型，届时沿用的仍是这条路线——它对该机型同样经过可行性校验，只是未必最优。
    这个近似是为了不把机型决策提前到组批阶段（机型取决于资源竞争，此时还不知道）。
    """
    out = []
    for tr in trips:
        bids = frozenset(b['id'] for v in tr['boxes_at'].values() for b in v)
        r = cache.best_any(bids)
        if r is None:
            out.append(tr)
            continue
        g, route, plan = r
        nt = dict(tr)
        nt['route'] = list(route)
        nt['_dur'] = plan['duration']
        out.append(nt)
    return out


# ---------------------------------------------------------------------------
# A1b F7 派发顺序局部寻优
# ---------------------------------------------------------------------------
def _vec(m):
    return (m['makespan'], m['total_E'], m['tardiness'])


def _dominates(a, b):
    """严格 Pareto 支配：各项不劣且至少一项严格更优。等值不接受，故搜索无环。"""
    return (all(x <= y + 1e-6 for x, y in zip(a, b))
            and any(x < y - 1e-6 for x, y in zip(a, b)))


def _priority_key(pair):
    """按题目指定的优先级字典序给 (档案条目, K 档) 排序：时延 → 完工 → 能耗 → 架次。

    **为什么不沿用 `f`。** `f = tardiness/ref_tard + 0.3*makespan/ref_ms` 把两者
    放在同一个标量里比，而两个参考量差着量级（本实例 ref_tard≈3.1e5、ref_ms≈1.3e4），
    于是 1 s 时延只值 1/3.1e5，1 s 完工值 0.3/1.3e4——**完工时刻的边际权重是时延的
    约 6 倍**。那是「先保完工、时延可让」的口径，与本题「物资必须在期望时间前送达、
    及时性是硬约束，其次才是全部任务完成时间，最后才是能耗与架次」正好相反：
    用 f 排序时，一个把时延抬高几千秒、但完工早几十秒的解会被选中。

    这里改成真正的字典序，四层依次比较，前一层严格更优即定胜负，不再互相折算。

    **为什么量化到 1e-6。** 两个解在时延上都「实际上是 0」时不该由浮点噪声决定胜负
    （0 与 3e-12 不是两种方案），量化后交给下一层比。同层内相差小于 1e-6 的也视作
    并列，继续往下比，故本键不引入任何偏好，只是把噪声挡在外面。
    """
    ent, K = pair
    m = ent['metrics']
    return (round(m['tardiness'], 6), round(m['makespan'], 6),
            round(m['total_E'], 6), m['n_trips'], K)


def optimize_dispatch_order(d, st, dispatch='balanced', rank_bonus=300.0,
                            order0=None, max_pass=200):
    """架次集合不变，只重排派发顺序。

    当前贪心调度「就绪即派」，容易把瓶颈机型先耗在非关键架次上；重排派工顺序等价于
    为即将空闲的无人机与共享电池**前瞻预约**下一任务。接受准则：硬违反恒为 0 且
    (makespan, 能耗, 加权时延) 严格 Pareto 改进——架次数不随顺序改变（能耗仅可能
    随机型重指派变化），故 recompute 一次才是准确判据。

    返回 (order, assignment, metrics)。
    """
    n = len(st)

    def ev(order):
        asg = q2.schedule(d, st, order=order, dispatch=dispatch, rank_bonus=rank_bonus)
        return asg, q2.evaluate(d, asg)

    if order0 is None:
        starts = [sorted(range(n), key=lambda i: q2.trip_sort_key(st[i]))]
        # 第二起点：同紧迫度层内时长降序（长架次早派，先占住重载机）
        dur = []
        for t in st:
            ds = [p['duration'] for g in TYPES
                  for fe, p in [q2.type_feasible(d, d.transport_types[g], t['route'],
                                                 t['boxes_at'])] if fe]
            dur.append(max(ds) if ds else 0.0)
        starts.append(sorted(range(n), key=lambda i: (q2.trip_sort_key(st[i])[0], -dur[i])))
    else:
        starts = [list(order0)]

    best = None
    for s0 in starts:
        order = list(s0)
        asg, m = ev(order)
        if m['hard_viol'] > 0:
            continue
        vec = _vec(m)
        for _ in range(max_pass):
            improved = False
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    cand = order[:i] + order[i + 1:]
                    cand = cand[:j] + [order[i]] + cand[j:]
                    asg_c, m_c = ev(cand)
                    if m_c['hard_viol'] > 0:
                        continue
                    vec_c = _vec(m_c)
                    if _dominates(vec_c, vec):
                        order, asg, m, vec = cand, asg_c, m_c, vec_c
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        if best is None or (_dominates(_vec(m), _vec(best[2]))
                            and not _dominates(_vec(best[2]), _vec(m))):
            best = (order, asg, m)
    if best is None:
        # 两个起点都违反硬时限：退回默认紧迫度序，交由调用方判断
        order = sorted(range(n), key=lambda i: q2.trip_sort_key(st[i]))
        asg, m = ev(order)
        return order, asg, m
    return best


# ---------------------------------------------------------------------------
# A2b 派发顺序的模拟退火（专攻尾部）
# ---------------------------------------------------------------------------
# 温度按初始 makespan 的相对比例给，故换实例不用重调。
DISPATCH_SA_T0 = 0.02      # 初温 ≈ 允许 2% 的恶化被接受
DISPATCH_SA_T1 = 2e-4      # 终温
DISPATCH_SA_SEED = 977     # 固定种子，保证与 ALNS 同为可复现搜索
# 标量化权重：时延 ≫ 完工 ≫ 能耗，与 _priority_key 同序。
DISPATCH_TARD_W = 1e6
DISPATCH_E_W = 1e-3


def _dispatch_scalar(m):
    """把 (时延, 完工, 能耗) 按题目优先级压成一个标量，供退火接受准则使用。

    字典序本身不能在退火里当目标——需要一个可比较大小的标量。权重取
    时延 1e6 ≫ 完工 1 ≫ 能耗 1e-3，与 `_priority_key` 的层序一致：本实例时延、
    完工都是 1e3–1e5 秒量级、能耗是 1e1–1e2 kWh 量级，故 1e6 的时延权重足以在
    双精度下压过完工差，而能耗只在时延、完工都并列时才起作用。

    硬违反**不**进这个标量：它是硬门禁，候选违反直接丢弃，不参与折算。
    """
    return (DISPATCH_TARD_W * m['tardiness'] + m['makespan']
            + DISPATCH_E_W * m['total_E'])


def _perturb(order, rng):
    """四种邻域随机取一：插入、交换、段搬移(or-opt)、段反转。

    只有插入是原来 optimize_dispatch_order 有的；段搬移与段反转能一次调整多个
    架次的相对次序，是退火消尾部时真正出力的一类移动（单点插入一次只能挪一架）。
    """
    n = len(order)
    if n < 2:
        return list(order)
    k = rng.randrange(4)
    new = list(order)
    if k == 0:                                  # 插入
        i, j = rng.randrange(n), rng.randrange(n)
        if i == j:
            return new
        x = new.pop(i)
        new.insert(j, x)
    elif k == 1:                                # 交换
        i, j = rng.randrange(n), rng.randrange(n)
        new[i], new[j] = new[j], new[i]
    elif k == 2:                                # or-opt：搬移长度 2–3 的整段
        L = min(rng.choice((2, 3)), n - 1)
        i = rng.randrange(n - L + 1)
        seg = new[i:i + L]
        rest = new[:i] + new[i + L:]
        j = rng.randrange(len(rest) + 1)
        new = rest[:j] + seg + rest[j:]
    else:                                       # 段反转
        i, j = rng.randrange(n), rng.randrange(n)
        a, b = min(i, j), max(i, j)
        new[a:b + 1] = new[a:b + 1][::-1]
    return new


def optimize_dispatch_order_sa(d, st, dispatch='balanced', rank_bonus=300.0,
                               order0=None, iters=6000, seed=DISPATCH_SA_SEED,
                               verbose=False):
    """在派发顺序上跑模拟退火，专攻「尾部」= 完工时刻 − Σ架次时长/机数。

    **为什么不能只靠严格 Pareto 的插入局部搜索。** `optimize_dispatch_order` 只接受
    (makespan, 能耗, 时延) 严格变好的移动，是纯爬山；而尾部恰恰要靠**中间态变差**的
    移动才消得掉——八台机最后一架次长短不一，得先把某台的架次让出去、让它空出一段，
    才可能把另一台的关键窗口提前，这一步在中间态看是变差的。实测尾部量级 1300 s
    （v2 现行解）到 3500 s（旧解），正是爬山够不到的地方。

    接受准则用 _dispatch_scalar（与题目优先级同序）；硬违反恒为硬门禁。
    返回 (order, assignment, metrics)。调用方应把结果与既有解按优先级比一次再采纳——
    本函数自身保证不返回比 order0 更差的解（best 初始即 order0）。
    """
    import random as _random
    rng = _random.Random(seed)
    n = len(st)
    if order0 is None:
        order0 = sorted(range(n), key=lambda i: q2.trip_sort_key(st[i]))

    def ev(order):
        asg = q2.schedule(d, st, order=order, dispatch=dispatch,
                          rank_bonus=rank_bonus)
        return asg, q2.evaluate(d, asg)

    cur_order = list(order0)
    cur_asg, cur_m = ev(cur_order)
    if cur_m['hard_viol'] > 0:
        # 起点就违反硬时限：不在不可行域里搜索，原样返回交由调用方判断
        return cur_order, cur_asg, cur_m
    cur_f = _dispatch_scalar(cur_m)
    best_order, best_asg, best_m, best_f = cur_order, cur_asg, cur_m, cur_f

    t0 = max(DISPATCH_SA_T0 * max(cur_m['makespan'], 1.0), 1e-9)
    t1 = max(DISPATCH_SA_T1 * max(cur_m['makespan'], 1.0), 1e-12)
    ratio = (t1 / t0) ** (1.0 / max(iters - 1, 1))
    T = t0
    n_acc = n_rej = 0
    for _ in range(iters):
        cand = _perturb(cur_order, rng)
        if cand == cur_order:
            T *= ratio
            continue
        asg_c, m_c = ev(cand)
        if m_c['hard_viol'] > 0:
            n_rej += 1
            T *= ratio
            continue
        f_c = _dispatch_scalar(m_c)
        df = f_c - cur_f
        if df <= 0.0 or rng.random() < math.exp(-df / max(T, 1e-12)):
            cur_order, cur_asg, cur_m, cur_f = cand, asg_c, m_c, f_c
            n_acc += 1
            if f_c < best_f - 1e-12:
                best_order, best_asg, best_m, best_f = cand, asg_c, m_c, f_c
        else:
            n_rej += 1
        T *= ratio
    if verbose:
        print(f'    退火派发顺序：{iters} 轮，接受 {n_acc}、拒 {n_rej}；'
              f'标量 {_dispatch_scalar(best_m):.1f}（起点 {_dispatch_scalar(cur_m):.1f}）')
    return best_order, best_asg, best_m


# ---------------------------------------------------------------------------
# A4 中继可行择序：问题二最优解是一个**等价类**，类内用第三问的门禁择序
# ---------------------------------------------------------------------------
RELAY_GATE_MAX_PROBE = 16          # 门禁评估的候选上限（超出的如实报数，不静默丢）
_RELAY_METRIC_KEYS = ('n_trips', 'makespan', 'total_E', 'tardiness', 'hard_viol')


def _relay_metric_key(m):
    """问题二**对外报告**的四项指标 + 硬违反，用于判「指标逐位不变」。"""
    return tuple(m[k] for k in _RELAY_METRIC_KEYS)


def _relay_probe(d, st, order, dispatch, rank_bonus):
    """把一个派工顺序交给第三问，返回 (字典序键, 明细)；不可行返回 (None, 原因)。

    键的顺序与 code/q3.py 内部的择优顺序一致：
        无绑定失效区间数 → 缺口秒数 → 累计推迟量 → 联合完工时刻 → 中继能耗
    「架次数 / 完工 / 能耗 / 加权时延」四项已在候选筛选时按逐位相等锁死，不重复。
    """
    import q3                                        # 延迟导入：q3 模块级 import q2
    asg = q2.schedule(d, st, order=order, dispatch=dispatch, rank_bonus=rank_bonus)
    # 与 q3.recommended() 同口径：按开始时刻重排后重编 trip_idx
    a = sorted(asg, key=lambda x: x['start'])
    a = [dict(x, trip_idx=i) for i, x in enumerate(a)]
    try:
        res = q3.solve_relay(d, a, verbose=False)
    except Exception as e:                           # noqa: BLE001
        return None, f'第三问抛错：{e}'
    if not res['ok']:
        return None, f'第三问不可行：{res["reason"]}'
    frac, _segs, gaps = q3.audit(d, res['assignment'], res['relays'], res['cands'])
    done = max([x['return_t'] for x in res['relays']] or [0.0])
    key = (len(res['uncovered']), round(float(frac['gap']), 12),
           round(float(sum(res['deltas'])), 6), round(done, 6),
           round(float(sum(x['E'] for x in res['relays'])), 9))
    return key, dict(gap=float(frac['gap']), n_gap_pts=len(gaps),
                     delay=float(sum(res['deltas'])), joint_done=float(done),
                     relay_E=float(sum(x['E'] for x in res['relays'])),
                     n_relays=len(res['relays']), n_skipped=len(res['skipped']))


def relay_feasible_order(d, st, asg, order, m, dispatch='balanced',
                         rank_bonus=300.0, max_probe=RELAY_GATE_MAX_PROBE,
                         verbose=True):
    """在**不改变问题二任何指标**的前提下，用第三问的门禁挑一个更好的派工顺序。

    为什么需要这一层：问题二的目标（加权时延 → 完工 → 能耗 → 架次）对「同一架
    无人机上两个架次谁先谁后」是盲的——该无人机的总占用时长不变，各项指标可以
    **逐位相同**；而第三问的通信保障对**绝对时刻**敏感：某个架次的失效区间若落在
    中继链的尾端，中继机返航周转赶不上，就会出现几百秒的迟建链。于是「问题二的
    最优解」实际上是一个等价类，类内怎么挑，问题二的目标函数给不出答案。

    候选生成是**确定性**的：按无人机编号升序、再按下标升序取「同机两架次互换」；
    筛选口径是 `q2.evaluate` 报告的指标**逐位相等**（不是容差）。由此：

      · 论文里报告的问题二数字一个都不变——这一层只动「谁先飞」，不动任何指标；
      · 同种子跑两遍得到同一个顺序；
      · 选不选得上，只看第三问的字典序（见 `_relay_probe`）。基线始终作为被挑战者
        参与比较，故门禁不会把一个已经更好的顺序换差。

    返回 (order, st, asg, m, gate)。gate 记录候选数、参与评估的条数、被上限截掉的
    条数与逐条判据，随 `info` 一起落盘，供事后回查「为什么选中了这条」。
    """
    import itertools

    t0 = time.time()
    order = list(order)
    base_key = _relay_metric_key(m)
    uav = {x['trip_idx']: x['uav'] for x in asg}
    slots = {}
    for k, i in enumerate(order):
        slots.setdefault(uav[i], []).append(k)

    cands = []
    for u in sorted(slots):
        ks = slots[u]
        for a, b in itertools.combinations(range(len(ks)), 2):
            o2 = list(order)
            o2[ks[a]], o2[ks[b]] = o2[ks[b]], o2[ks[a]]
            asg2 = q2.schedule(d, st, order=o2, dispatch=dispatch,
                               rank_bonus=rank_bonus)
            m2 = q2.evaluate(d, asg2)
            if _relay_metric_key(m2) == base_key:
                cands.append(dict(tag=f'{u} slot{ks[a]}<->{ks[b]}', order=o2,
                                  asg=asg2, m=m2))
    n_total, n_drop = len(cands), max(0, len(cands) - max_probe)
    if n_drop:
        cands = cands[:max_probe]

    rows, best_tag, best_key, best = [], 'base（原序）', None, None
    probes = [dict(tag='base（原序）', order=order, asg=asg, m=m)] + cands
    for pr in probes:
        key, det = _relay_probe(d, st, pr['order'], dispatch, rank_bonus)
        pr['key'], pr['detail'] = key, det
        rows.append(dict(tag=pr['tag'], key=key, detail=det))
        if verbose:
            if key is None:
                print(f'    中继门禁 {pr["tag"]:20s} 判不了：{det}')
            else:
                print(f'    中继门禁 {pr["tag"]:20s} 缺口 {100*det["gap"]:.4f}%'
                      f'（{det["n_gap_pts"]} 点）、中继 {det["n_relays"]} 架、'
                      f'推迟 {det["delay"]:.0f} s、联合完工 {det["joint_done"]:.1f} s、'
                      f'中继能耗 {det["relay_E"]:.4f} kWh')
        if key is not None and (best_key is None or key < best_key):
            best_tag, best_key, best = pr['tag'], key, pr
    if n_drop and verbose:
        print(f'    中继门禁：候选 {n_total} 条，超出上限 {max_probe} 的 {n_drop} 条'
              f'**未评估**（此处如实报数，不当作已评估）')

    gate = dict(n_swap_total=int(n_total), n_probed=len(probes), n_dropped=int(n_drop),
                max_probe=int(max_probe), chosen=str(best_tag), secs=time.time() - t0,
                rows=rows)
    if best is None or best['tag'] == 'base（原序）':
        if verbose:
            print(f'  中继可行择序：{n_total} 个等价候选全部评估完毕，'
                  f'原序仍是第三问门禁下的最优，顺序不变（{gate["secs"]:.1f}s）')
        return order, st, asg, m, gate

    order2, asg2, m2 = best['order'], best['asg'], best['m']
    # 门禁只允许在**指标逐位相同**的候选里挑，这里再核一遍——防止将来有人放宽
    # 候选筛选口径却忘了这一层，导致「换个顺序把问题二的数也换了」。
    assert _relay_metric_key(m2) == base_key, '择序不得改变问题二的报告指标'
    if verbose:
        print(f'  中继可行择序：{n_total} 个等价候选中选中 {best_tag}，'
              f'问题二指标逐位不变（{gate["secs"]:.1f}s）')
    return order2, st, asg2, m2, gate


# ---------------------------------------------------------------------------
# A3 多种子 + 档间热启动的 ε-约束扫描
# ---------------------------------------------------------------------------
def solve_recommended_v2(d, n_iter=3500, seeds=5, verbose=False,
                         k_targets=None, dispatch='balanced', rank_bonus=300.0,
                         patience=None, tier_budget=None, warm_start=True,
                         refine=True, save=True, fleet_lambda=None, sa_iters=4000,
                         relay_gate=True, relay_gate_max_probe=RELAY_GATE_MAX_PROBE):
    """v2 求解推荐方案的**唯一入口**：设定选型权重 λ 后转 `_solve_v2`，返回后复位。

    λ 是模块级状态（`_fleet_cost` 与 `TripRouteCache` 都读它），而调用方在本函数
    返回后还会继续跑消融——`ablation_row_v2` 同样调 `construct_trips_exact`。若不
    复位，消融第⑤行会在「上一步留下的 λ」下计算，而论文里它是要与第④行逐项对照的。
    这类残留不报错、只改数，故在这里用 try/finally 兜住，不给调用方留记忆负担。

    fleet_lambda 显式传参而不是让调用方先 set 一下：λ 会改变选型与缓存分代，必须
    与本次求解的其余预算一起**记进 info 并落盘**，否则无法从产物反推它是怎么算的。
    None = 用模块默认值 FLEET_LAMBDA。
    """
    lam = FLEET_LAMBDA if fleet_lambda is None else float(fleet_lambda)
    saved = set_fleet_lambda(lam)
    try:
        return _solve_v2(d, n_iter=n_iter, seeds=seeds, verbose=verbose,
                         k_targets=k_targets, dispatch=dispatch,
                         rank_bonus=rank_bonus, patience=patience,
                         tier_budget=tier_budget, warm_start=warm_start,
                         refine=refine, save=save, fleet_lambda=lam,
                         sa_iters=sa_iters, relay_gate=relay_gate,
                         relay_gate_max_probe=relay_gate_max_probe)
    finally:
        set_fleet_lambda(saved)


def _solve_v2(d, n_iter=3500, seeds=5, verbose=False,
              k_targets=None, dispatch='balanced', rank_bonus=300.0,
              patience=None, tier_budget=None, warm_start=True,
              refine=True, save=True, fleet_lambda=0.0, sa_iters=4000,
              relay_gate=True, relay_gate_max_probe=RELAY_GATE_MAX_PROBE):
    """v2 求解推荐方案，返回 (trips, assignment, metrics, info)。请走 solve_recommended_v2。

    与 q2.solve_recommended 的差别只有三处：初始解换成精确组批、解码用
    dispatch='balanced'、每档多种子且档间热启动。决策口径（硬时限零违反为前提，
    推荐解只从可行档案取）与下游接口完全一致。

    seeds        每档的独立种子数；种子由 ALNS_SEED + K*101 + s*7919 生成，互不重叠。
    patience     某档内连续多少轮档案无改进即停该档；None = 跑满。
    tier_budget  单档墙钟上限（秒）；None = 不限。超预算的档如实标注，不当成已收敛。
    warm_start   True = 上一档最优解热启动下一档；False = 每档都从初始解重开。
    save         False = 只探路，不写 results/q2_recommended.json 与 q2_solution.json。
                 探路轮必须用它：拿小预算的解覆盖正式产物，会让仓库停在
                 「方案记录与论文数字互不对应」的状态。
    fleet_lambda 本次生效的 λ（由外层设好并传入，此处只用于记录，不再改状态）。
    """
    t_start = time.time()
    base_trips = construct_trips_exact(d, pack_mode='mixed')
    if refine:
        base_trips = refine_routes(d, base_trips, TripRouteCache(d))
    base_asg = q2.schedule(d, base_trips, dispatch=dispatch, rank_bonus=rank_bonus)
    base_m = q2.evaluate(d, base_asg)
    assert base_m['hard_viol'] == 0, '精确组批初始解违反硬时限，不应发生'
    ref_tard = max(base_m['tardiness'], 1.0)
    ref_ms = max(base_m['makespan'], 1.0)
    if verbose:
        print(f'  初始解（精确组批，{len(base_trips)} 架次）: 能耗={base_m["total_E"]:.3f} kWh '
              f'makespan={base_m["makespan"]:.1f} s 时延={base_m["tardiness"]:.0f}')

    init = [dict(route=list(t['route']),
                 boxes=[b for v in t['boxes_at'].values() for b in v])
            for t in base_trips]

    # F7：先对初始解的派发顺序做一次局部寻优（架次集合不变，故不可能变差）
    st0 = q2.to_sched_trips(d, init)
    order0 = q2._default_order(d, init)
    order0, asg0, m0 = optimize_dispatch_order(d, st0, dispatch=dispatch,
                                               rank_bonus=rank_bonus, order0=order0)
    if verbose and m0['makespan'] < base_m['makespan'] - 1e-6:
        print(f'  F7 派发寻优: makespan {base_m["makespan"]:.1f} -> {m0["makespan"]:.1f} s')

    Ks = list(q2.K_TARGETS if k_targets is None else k_targets)
    runs = {}
    warm = (init, order0)
    for K in Ks:
        t_tier = time.time()
        arch, best_trace, best_f_seen = [], None, float('inf')
        seed_stats = []
        for s in range(seeds):
            if tier_budget is not None and time.time() - t_tier > tier_budget:
                if verbose:
                    print(f'  K<={K:2d}: 达到单档预算 {tier_budget:.0f}s，'
                          f'余下 {seeds - s} 个种子未跑（如实标注）')
                break
            seed = q2.ALNS_SEED + K * 101 + s * 7919
            tr = []
            bt, bo, bm, bf, _ = q2.alns(
                d, warm[0], k_target=K, n_iter=n_iter, seed=seed,
                ref_tard=ref_tard, ref_ms=ref_ms, trace=tr, archive=arch,
                dispatch=dispatch, rank_bonus=rank_bonus, patience=patience,
                init_order=warm[1])
            feas = bm['hard_viol'] == 0 and bm['n_trips'] <= K
            seed_stats.append(dict(seed=seed, feasible=feas,
                                   metrics=dict(bm), f=bf))
            if arch and arch[0]['f'] < best_f_seen - 1e-12:
                best_f_seen = arch[0]['f']
                best_trace = list(tr)
            if verbose:
                print(f'    seed {s} [{"可行" if feas else "违反档约束"}]: '
                      f'架次={bm["n_trips"]:2d} 能耗={bm["total_E"]:7.3f} '
                      f'makespan={bm["makespan"]:8.1f} 时延={bm["tardiness"]:9.0f} '
                      f'档案={len(arch)}')
        if not arch:
            # 该档全部种子均未找到可行解：如实记录「档无解」，不拿超 K 的解顶替
            runs[K] = dict(archive=[], metrics=None, secs=time.time() - t_tier,
                           trace=[], seeds=seed_stats, n_seeds_run=len(seed_stats),
                           converged=True)
            if verbose:
                print(f'  >> K={K} 档无解（{len(seed_stats)} 个种子均未满足档约束）')
            continue
        ent = arch[0]
        runs[K] = dict(trips=ent['trips'], order=ent['order'], metrics=dict(ent['metrics']),
                       f=ent['f'], archive=arch, secs=time.time() - t_tier,
                       trace=best_trace if best_trace else [],
                       seeds=seed_stats, n_seeds_run=len(seed_stats),
                       converged=(tier_budget is None
                                  or time.time() - t_tier <= tier_budget))
        if verbose:
            print(f'  >> K<={K} 档最优: 实际架次={ent["metrics"]["n_trips"]} '
                  f'能耗={ent["metrics"]["total_E"]:.3f} kWh '
                  f'makespan={ent["metrics"]["makespan"]:.1f} s '
                  f'时延={ent["metrics"]["tardiness"]:.0f}（{runs[K]["secs"]:.1f}s）')
        if warm_start:
            warm = ([dict(route=list(t['route']), boxes=list(t['boxes']))
                     for t in ent['trips']], list(ent['order']))

    # 每档的**代表解** = 该档全部档案条目里按题目优先级键最优的那条，而不是
    # archive[0]。档案是按 (f, n_trips) 排序、按 f 截断的，某档 f 最优的解时延未必
    # 为 0；而按题目优先级时延是第一层比较，该档真正该选中的很可能是被 f 挤到后面的
    # 那一条。若只取 archive[0]，新键就只能在「各档 f 冠军」之间挑，挑不到那些零时延
    # 解——键换了却挑不出差别，等于没换。
    # 12 条里挑优先级最优 == 该档在全档案上的优先级最优，故「逐档挑一条、跨档再比」
    # 与「把全体档案条目放一起比」等价：键相同，min 对同一键的分配律成立。
    reps = {}
    for K, r in sorted(runs.items()):
        if not r['archive']:
            continue
        # min 只比 key，不会再去比后面的 i/ent（dict 不可排序，会比出 TypeError）
        _, _i, _e = min(((_priority_key((ent, K)), i, ent)
                         for i, ent in enumerate(r['archive'])),
                        key=lambda p: p[0])
        reps[K] = dict(rank_in_arch=_i, entry=_e)
    if not reps:
        raise RuntimeError('ε-约束全部档位均未找到可行解（硬违反为 0 且 n_trips<=K）；'
                           '不得用超 K 的解顶替')

    # ---- 定型轮：**每一档**的代表解都过同一轮 F7 + 退火 ------------------------
    # 为什么每档都要跑、而不是只跑胜出的那一档：q2_pareto.csv 的一行 = 该档的交付解，
    # 而落盘的推荐方案就是其中一行。只定型胜出档，表内各行口径就不一致（一行定型后、
    # 其余定型前），更要命的是**落盘方案在表里查不到**——paper_metrics.py 的溯源断言
    # （推荐方案必须能在 q2_pareto.csv 里唯一命中）会当场拒绝，论文的推荐档数字也就
    # 无从与方案记录对齐。这一条不是可选收尾。
    # 故 reps[K]['metrics'] 存的是**定型后**的指标，q2.export() 从 reps 取每档落表；
    # runs[K]['metrics']（末态）与档案口径保持原义，一字不动。
    asg_of, order_of = {}, {}
    for K in sorted(reps):
        rep = reps[K]
        entry = rep['entry']
        st_k = q2.to_sched_trips(d, entry['trips'])
        order_k = list(entry['order'])
        asg_k = q2.schedule(d, st_k, order=order_k, dispatch=dispatch,
                            rank_bonus=rank_bonus)
        m_k = q2.evaluate(d, asg_k)
        assert m_k['hard_viol'] == 0, f'K<={K} 档代表解硬约束违反'
        assert m_k['n_trips'] <= K, f'K<={K} 档代表解超出档位，档案筛选失效'

        # 定型前再做一轮 F7：架次集合不变，只重排派工顺序。接受准则是
        # (makespan, 能耗, 加权时延) 的严格 Pareto 改进，故报告的数字只会更好、
        # 不会更差；n_trips 由构造不变，K 档约束仍成立。
        m_f7_pre = dict(m_k)
        order_p, asg_p, m_p = optimize_dispatch_order(
            d, st_k, dispatch=dispatch, rank_bonus=rank_bonus, order0=order_k)
        polished = _dominates(_vec(m_p), _vec(m_k))
        if polished:
            order_k, asg_k, m_k = order_p, asg_p, m_p
            assert m_k['hard_viol'] == 0 and m_k['n_trips'] <= K

        # 定型后再走一轮退火派发顺序。F7 是严格 Pareto 爬山，够不到「中间态变差才
        # 消得掉的尾部」；退火补的就是这一段。采纳门是 _priority_key 严格更优——与
        # 选择键同序（时延 → 完工 → 能耗 → 架次），故报告的数字只会更好、不会更差，
        # n_trips 由构造不变、K 档约束仍成立。seed 由档位派生，保证同种子跑两遍
        # 逐位一致。
        m_sa_pre = dict(m_k)
        sa_applied = False
        if sa_iters > 0:
            order_s, asg_s, m_s = optimize_dispatch_order_sa(
                d, st_k, dispatch=dispatch, rank_bonus=rank_bonus,
                order0=order_k, iters=sa_iters,
                seed=DISPATCH_SA_SEED + 101 * int(K))
            if (m_s['hard_viol'] == 0 and m_s['n_trips'] <= K
                    and _priority_key((dict(metrics=m_s), K))
                    < _priority_key((dict(metrics=m_k), K))):
                order_k, asg_k, m_k = order_s, asg_s, m_s
                sa_applied = True
                assert m_k['hard_viol'] == 0 and m_k['n_trips'] <= K
        asg_of[K], order_of[K] = asg_k, order_k
        rep.update(metrics=dict(m_k), trips=st_k,
                   polish=dict(applied=bool(polished),
                               before=_vec(m_f7_pre), after=_vec(m_sa_pre)),
                   sa=dict(iters=int(sa_iters), applied=bool(sa_applied),
                           before=_vec(m_sa_pre), after=_vec(m_k)))
        if verbose:
            # 关掉退火时必须说「未启用」。写成「未获改进」会让日志读者以为退火跑了
            # 但没赢，从而把「这一层无效」当成结论——两者是完全不同的事实。
            tag = '' if polished else '（F7 未获改进）'
            if sa_iters <= 0:
                tag += '（SA 未启用）'
            elif not sa_applied:
                tag += '（SA 未获改进）'
            print(f'  K<={K} 定型: 架次={m_k["n_trips"]:2d} '
                  f'能耗={m_k["total_E"]:7.3f} kWh '
                  f'makespan={m_k["makespan"]:8.1f} s 时延={m_k["tardiness"]:.0f}{tag}')

    # ---- 跨档择优：同一把优先级键，故各档定型解里最优的那条就是全局最优 --------
    K_best = min(sorted(reps),
                 key=lambda K: _priority_key((dict(metrics=reps[K]['metrics']), K)))
    rep_best = reps[K_best]
    entry = rep_best['entry']
    st, asg, m = rep_best['trips'], asg_of[K_best], rep_best['metrics']
    order_best = list(order_of[K_best])
    rank_in_arch = rep_best['rank_in_arch']
    m_before, polished = rep_best['polish']['before'], rep_best['polish']['applied']
    m_before_sa, sa_applied = rep_best['sa']['before'], rep_best['sa']['applied']
    if verbose and rank_in_arch:
        # 非 0 就说明「按优先级挑中的」不是该档 f 冠军。这行必须打出来，否则
        # 事后看日志只会觉得档案第一条被丢了。
        print(f'  定型解取自 K<={K_best} 档档案第 {rank_in_arch + 1} 条'
              f'（共 {len(runs[K_best]["archive"])} 条，按 f 排序本不在首位）：'
              f'时延={entry["metrics"]["tardiness"]:.0f} '
              f'makespan={entry["metrics"]["makespan"]:.1f} s '
              f'能耗={entry["metrics"]["total_E"]:.3f} kWh '
              f'架次={entry["metrics"]["n_trips"]}')
    # ---- 中继可行择序：在不改任何指标的前提下，用第三问门禁挑派工顺序 ----------
    # 放在这里而不是第三问里：verify.py 要求 q3 各架次的「原开始时刻」逐位等于 q2 的
    # 开始时刻，所以任何顺序调整都必须在**问题二落盘之前**完成，否则复核链断掉。
    gate = None
    if relay_gate:
        if not hasattr(d, 'geo'):
            raise RuntimeError('中继门禁需要已预计算的几何（d.geo）：'
                               '请先 precompute_geometry(d) 再求解')
        order_best, st, asg, m, gate = relay_feasible_order(
            d, st, asg, order_best, m, dispatch=dispatch, rank_bonus=rank_bonus,
            max_probe=relay_gate_max_probe, verbose=verbose)

    info = dict(runs=runs, K=K_best, base=base_m, ref_tard=ref_tard, ref_ms=ref_ms,
                # 定型轮的两段前后向量取自**胜出档**那一轮（已是 _vec 过的三元组），
                # 不再重复 _vec——它们和 reps[K_best] 是同一份记录。
                polish=dict(applied=bool(polished),
                            before=m_before, after=m_before_sa),
                sa=dict(iters=int(sa_iters), applied=bool(sa_applied),
                        before=m_before_sa, after=_vec(m)),
                # 各档定型后的代表解：q2.export() 用它写 q2_pareto.csv，从而保证
                # 「表里的一行」与「落盘的推荐方案」是同一个东西而不是两条路径各算各的。
                # 只放指标与目标值，不放 trips/order/指派（那些在 sol 里已有冻结版本，
                # 重复放进 info 只会让日志与内存翻倍）。
                reps={K: dict(metrics=dict(v['metrics']), f=v['entry']['f'],
                              rank_in_arch=v['rank_in_arch'],
                              arch_size=len(runs[K]['archive']))
                      for K, v in sorted(reps.items())},
                init=init, n_iter=n_iter, K_targets=list(Ks), engine='v2',
                seeds=int(seeds), patience=patience, tier_budget=tier_budget,
                warm_start=bool(warm_start), dispatch=dispatch,
                rank_bonus=float(rank_bonus), fleet_lambda=float(fleet_lambda),
                secs=time.time() - t_start,
                rec=dict(K=K_best, f=entry['f'],
                         # metrics 记的是**被选中的那条档案条目**（定型前）的指标，
                         # 与 f、rank_in_arch 同一出处，便于回查「档案里长什么样」；
                         # 交付的方案是它过完定型轮之后的 delivered，两者不同是正常的
                         # （v2 本次：档案 24 架次/70.201 kWh/6701.9 s → 交付
                         #  24 架次/69.342 kWh/6506.1 s）。写成同一个键名会让人以为
                         # 报告的数字就是档案里那条，故两个都留、名字分开。
                         metrics=dict(entry['metrics']),
                         delivered=dict(m),
                         # 该解在其档档案里的名次（0 = 按 f 排第一）。非 0 说明定型解
                         # 不是该档 f 冠军，而是按题目优先级从档案里另挑的——落盘时
                         # 不记这一项，事后就无法解释「为什么选中的不是档案第一条」。
                         rank_in_arch=rank_in_arch,
                         arch_size=len(runs[K_best]['archive']),
                         key='(tardiness, makespan, total_E, n_trips, K)',
                         trace=runs[K_best]['trace']),
                # 中继门禁的逐条判据：候选怎么来的、评估了几条、被上限截掉几条、
                # 最后选了谁。落盘是为了事后能回答「这条顺序凭什么被选中」，而不是
                # 只留一个「顺序是这么定的」。
                relay_gate=gate)
    info['order'] = list(order_best)
    if not save:
        info['solution_id'] = None
        if verbose:
            print('  （探路轮：未写盘，results/ 下的正式产物保持不动）')
        return st, asg, m, info
    q2.save_recommended(d, st, order_best, m, dispatch=dispatch,
                        rank_bonus=rank_bonus)
    sol = q2.build_q2_solution(d, asg, m, order_best, info, seed=q2.ALNS_SEED)
    info['solution_id'] = q2.save_solution(q2.SOL_CACHE, sol)
    if verbose:
        print(f'  已落盘完整方案 {os.path.basename(q2.SOL_CACHE)} '
              f'（solution_id={info["solution_id"]}，{len(sol["transport_trips"])} 架次）')
    return st, asg, m, info


def ablation_row_v2(d, n_iter=3000, ref_tard=1.0, ref_ms=1.0, seed=None,
                    rank_bonus=300.0, use_order=True):
    """消融第⑤行：三层全移，预算与④逐项对齐。

    与④的差别**只有三层移植本身**：初始解换成精确组批 + 路线全排列、解码换成
    dispatch='balanced'、初始解上再跑一次 F7 派发寻优；ALNS 的轮数、种子、
    use_order、freeze_boxes、温度口径与④完全相同。故④→⑤的差值可直接读作
    「我方组批/派发/解码 vs 三层移植」，不掺入搜索预算的差异。

    seed 缺省与④同值（ALNS_SEED+3）：换种子会把「种子运气」混进对照，测不出移植本身。
    """
    base = construct_trips_exact(d, pack_mode='mixed')
    trips0 = [dict(route=list(t['route']),
                   boxes=[b for v in t['boxes_at'].values() for b in v]) for t in base]
    st0 = q2.to_sched_trips(d, trips0)
    order0, _, _ = optimize_dispatch_order(d, st0, dispatch='balanced',
                                           rank_bonus=rank_bonus,
                                           order0=q2._default_order(d, trips0))
    _bt, _bo, bm, _bf, _w = q2.alns(
        d, trips0, k_target=None, n_iter=n_iter,
        seed=q2.ALNS_SEED + 3 if seed is None else seed,
        use_order=use_order, freeze_boxes=False, ref_tard=ref_tard, ref_ms=ref_ms,
        dispatch='balanced', rank_bonus=rank_bonus, init_order=order0)
    return dict(name='⑤三层全移（v2 引擎）',
                **{k: bm[k] for k in ('n_trips', 'total_E', 'makespan',
                                      'tardiness', 'hard_viol')})


# ---------------------------------------------------------------------------
# 自检：缓存纪要与解码退化
# ---------------------------------------------------------------------------
def self_check(d, verbose=True):
    """移植引入的两处新缓存与新解码档的硬性自检。返回 True 表示全过。

    1) 路线缓存与直算一致：同一 (箱集合, 机型) 下，缓存给的路线必须真是全排列最优。
    2) 几何纪元：重算几何后缓存必须落空、纪元必须递增。
    3) dispatch='balanced' 在 rank_bonus=0 时必须与 'earliest' 逐位一致
       （balen 的 score 退化为 start，此时两者是同一个规则）。
    """
    ok = True
    cache = TripRouteCache(d)
    boxes = _box_index(d)
    import random
    rng = random.Random(11)
    bad = 0
    for _ in range(60):
        sid = rng.choice(d.S)
        pool = d.boxes_by_service[sid]
        bids = frozenset(b['id'] for b in rng.sample(pool, rng.randint(1, min(4, len(pool)))))
        g = rng.choice(TYPES)
        r = cache.get(bids, g)
        ba = cache.boxes_at_of(bids)
        services = sorted(ba)
        best = None
        for perm in itertools.permutations(services):
            feas, plan = q2.type_feasible(d, d.transport_types[g], list(perm), ba)
            if not feas:
                continue
            hard = any(q2._is_urgent(boxes[b]) for b in bids)
            last = max(plan['deliver'].values())
            c = _fleet_cost(g, plan['E'], plan['duration'])
            k = (last, c, plan['E']) if hard else (c, plan['E'], last)
            if best is None or k < best[0]:
                best = (k, list(perm))
        if r is None:
            bad += (best is not None)
        elif best is None or r[0] != best[1]:
            bad += 1
    if bad:
        ok = False
    if verbose:
        print(f'  路线全排列缓存对拍：不一致 {bad} 例（必须为 0）')

    ep0 = q2.GEO_EPOCH
    TripRouteCache(d)                          # 填充前的世代
    q2.precompute_geometry(d)
    if q2.GEO_EPOCH != ep0 + 1 or q2._TRIP_CACHE:
        ok = False
        if verbose:
            print(f'  几何纪元断言失败：纪元 {ep0}->{q2.GEO_EPOCH}，'
                  f'_TRIP_CACHE={len(q2._TRIP_CACHE)} 条（均应为 纪元+1、0 条）')
    elif verbose:
        print(f'  几何纪元：重算后纪元 {ep0}->{q2.GEO_EPOCH}，备忘已清空 ✓')

    trips0 = q2.construct_trips(d, pack_type='C')
    a1 = q2.schedule(d, trips0, dispatch='earliest')
    a2 = q2.schedule(d, trips0, dispatch='balanced', rank_bonus=0.0)
    if len(a1) != len(a2) or any(x['type'] != y['type'] or abs(x['start'] - y['start']) > 1e-9
                                 for x, y in zip(a1, a2)):
        ok = False
        if verbose:
            print('  balanced(rank_bonus=0) 与 earliest 不一致（必须一致）')
    elif verbose:
        print('  balanced(rank_bonus=0) 与 earliest 逐位一致 ✓')

    # λ=0 必须逐位退化为旧的「能耗最小」，否则 C 层移植的收益无法与旧解干净对照
    lam_save = FLEET_LAMBDA
    try:
        set_fleet_lambda(0.0)
        worst_l = 0.0
        for g in TYPES:
            for g2 in TYPES:
                for E, T in ((1.234, 900.0), (0.0, 0.0), (7.5, 4321.0)):
                    worst_l = max(worst_l, abs(_fleet_cost(g, E, T) - E))
        if worst_l > 0.0:
            ok = False
            if verbose:
                print(f'  λ=0 退化断言失败：与纯能耗最大偏差 {worst_l:.3e}（必须为 0）')
        elif verbose:
            print('  λ=0 时选型代价逐位退化为能耗最小 ✓')
        ca = TripRouteCache(d)
        assert ca.lam == 0.0
    finally:
        set_fleet_lambda(lam_save)

    # λ 必须在退出时复位：调用方（solve_recommended_v2）靠它把状态还回去，
    # 若 setter 不返回旧值，第⑤行消融就会带着上一轮的 λ 计算，而它是要与第④行
    # 逐项对照的。这里验的是 setter 的返回契约，不是数值本身。
    before = FLEET_LAMBDA
    old = set_fleet_lambda(0.123)
    bad_setter = (abs(old - before) > 0.0 or abs(FLEET_LAMBDA - 0.123) > 0.0)
    set_fleet_lambda(before)
    if abs(FLEET_LAMBDA - before) > 0.0:
        bad_setter = True
    if bad_setter:
        ok = False
        if verbose:
            print('  set_fleet_lambda 返回契约失败：应返回被覆盖的旧值（否则外层 finally '
                  '复位成新值，等于没复位）')

    # 优先级字典序（见 _priority_key）逐层验证：每一层都要能压过后面所有层之和
    def _mk(tard, ms, e, n):
        return dict(metrics=dict(tardiness=tard, makespan=ms, total_E=e, n_trips=n))

    pk_cases = [
        # 时延更小者胜，哪怕完工、能耗、架次全输
        ((_mk(0.0, 9000.0, 99.0, 30), 30), (_mk(1000.0, 5000.0, 60.0, 20), 20), True),
        # 时延并列，比完工
        ((_mk(0.0, 5000.0, 99.0, 30), 30), (_mk(0.0, 6000.0, 60.0, 20), 20), True),
        # 时延、完工并列，比能耗
        ((_mk(0.0, 5000.0, 60.0, 30), 30), (_mk(0.0, 5000.0, 61.0, 20), 20), True),
        # 前三层并列，比架次
        ((_mk(0.0, 5000.0, 60.0, 20), 30), (_mk(0.0, 5000.0, 60.0, 25), 25), True),
        # 前三层与架次都并列，比 K 档
        ((_mk(0.0, 5000.0, 60.0, 20), 25), (_mk(0.0, 5000.0, 60.0, 20), 30), True),
        # 反向：上面第一条反过来必须判负，否则「更小者胜」只是巧合
        ((_mk(1000.0, 5000.0, 60.0, 20), 20), (_mk(0.0, 9000.0, 99.0, 30), 30), False),
        # 浮点噪声不构成时延优势：3e-12 量化后视作并列，于是由完工时刻决定胜负，
        # 噪声方（完工 6000 s）必须输给完工 3000 s 的一方——若量化失效，它会靠
        # 1e-12 的「时延优势」在第一层就赢，后面三层根本没机会比。
        ((_mk(3e-12, 6000.0, 99.0, 30), 30), (_mk(0.0, 3000.0, 60.0, 20), 20), False),
        # 全等则并列（键相等），由 min() 取先出现的，不引入偏好
        ((_mk(0.0, 5000.0, 60.0, 20), 25), (_mk(0.0, 5000.0, 60.0, 20), 25), False),
        # ↓ 标量化的定义域内用例：架次与 K 档**都相同**，胜负在第 1–3 层分出。
        #   上面 8 例全都动了架次或 K，故没有一例能检验 _dispatch_scalar——若不加这三条，
        #   标量检查会以「0 例全过」通过，看起来有覆盖实则什么都没验。
        ((_mk(0.0, 9000.0, 99.0, 25), 25), (_mk(1000.0, 5000.0, 60.0, 25), 25), True),
        ((_mk(0.0, 5000.0, 99.0, 25), 25), (_mk(0.0, 6000.0, 60.0, 25), 25), True),
        ((_mk(0.0, 5000.0, 60.0, 25), 25), (_mk(0.0, 5000.0, 61.0, 25), 25), True),
    ]
    n_bad_pk = sum(1 for a, b, first in pk_cases
                   if (_priority_key(a) < _priority_key(b)) != first
                   and not (not first and _priority_key(a) == _priority_key(b)))
    if n_bad_pk:
        ok = False
    if verbose:
        print(f'  优先级字典序断言：{len(pk_cases)} 例，失败 {n_bad_pk} 例（必须为 0）')

    # 退火标量化必须与 _priority_key 同序，否则接受准则会把解往反方向推。
    # **只在标量定义域上比**：SA 的架次集合是固定的，n_trips 与 K 全场不变，故第 4、5 层
    # （架次、K 档）在搜索中根本不参与——标量里也就没有这两项。凡在 n_trips/K 上分出胜负
    # 的用例都超出定义域，拿它们判标量是错的要求。这里据此只比第 1–3 层能分胜负的用例，
    # 并且**另设一条**断言把「第 4、5 层不与标量同序」这个已知边界钉住，防止哪天有人
    # 在 n_trips 会变的地方复用 _dispatch_scalar 却以为它管全局字典序。
    n_bad_sc = n_in_domain = n_out_domain_ok = 0
    for a, b, first in pk_cases:
        if not first:
            continue
        (ea, Ka), (eb, Kb) = a, b
        same_trips = ea['metrics']['n_trips'] == eb['metrics']['n_trips']
        in_domain = same_trips and Ka == Kb
        sa_s, sb_s = _dispatch_scalar(ea['metrics']), _dispatch_scalar(eb['metrics'])
        if in_domain:
            n_in_domain += 1
            if not (sa_s < sb_s):
                n_bad_sc += 1
        elif sa_s == sb_s:
            # 域外：标量必须**判平**（看不到架次/K），这正是它不能用于域外的证据
            n_out_domain_ok += 1
    if n_bad_sc:
        ok = False
    if verbose:
        print(f'  退火标量化与优先级同序：定义域内 {n_in_domain} 例，不同序 '
              f'{n_bad_sc} 例（必须为 0）；域外判平 {n_out_domain_ok} 例')
        print(f'    （标量不含架次/K 两项，仅适用于 SA 的定架次搜索；'
              f'全局择优选 _priority_key）')

    # _perturb 必须恒返回同一 multiset 的一个排列（长度、元素都对），且不原地改入参。
    # 顺序搜索里最隐蔽的错就是扰动把某个架次弄丢/复制，指标却还「看起来正常」。
    rng_p = random.Random(4242)
    base = list(range(40))
    n_bad_pert = 0
    for _ in range(2000):
        p = _perturb(base, rng_p)
        if sorted(p) != base or len(p) != len(base):
            n_bad_pert += 1
    if list(range(40)) != base:
        n_bad_pert += 1
    if n_bad_pert:
        ok = False
    if verbose:
        print(f'  退火扰动保序性：2000 例，破坏排列 {n_bad_pert} 例（必须为 0）')
    return ok


def compare_with_v1(verbose=True):
    """把上一版解与当前解并列写出 results/q2_port_compare.csv（移植对照，可交付）。"""
    import pandas as pd
    rows = []
    for tag, path in (('v1（阶段 10 旧解）', os.path.join(HERE, '..', '_legacy', 'q2_v1',
                                                         'q2_transport_trips.csv')),
                      ('v2（本次移植）', os.path.join(OUT, 'q2_transport_trips.csv'))):
        if not os.path.exists(path):
            if verbose:
                print(f'  跳过 {tag}：未找到 {path}')
            continue
        df = pd.read_csv(path)
        rows.append(dict(方案=tag, 架次数=len(df),
                         能耗kWh=round(df['架次能耗kWh'].sum(), 3),
                         makespan_s=round(df['返回O01时刻s'].max(), 1),
                         机型数=df['机型编号'].nunique()))
    if rows:
        out = os.path.join(OUT, 'q2_port_compare.csv')
        pd.DataFrame(rows).to_csv(out, index=False)
        if verbose:
            print(pd.DataFrame(rows).to_string(index=False))
            print(f'  已写出 {out}')
    return rows


def main():
    import argparse
    ap = argparse.ArgumentParser(description='问题二 v2 引擎（精确组批 + 瓶颈感知解码 + 多种子）')
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--iters', type=int, default=3500)
    ap.add_argument('--patience', type=int, default=None)
    ap.add_argument('--tier-budget', type=float, default=None)
    ap.add_argument('--only-K', type=int, default=None)
    ap.add_argument('--check', action='store_true', help='只跑自检')
    ap.add_argument('--warm-start', dest='warm_start', action='store_true', default=True)
    ap.add_argument('--no-warm-start', dest='warm_start', action='store_false')
    ap.add_argument('--save', dest='save', action='store_true', default=False,
                    help='写 results/（默认不写：本脚本按探路用，正式轮走 '
                         'python code/q2.py --engine v2）')
    args = ap.parse_args()

    d = load_data()
    d.geo_nodes, d.geo = q2.precompute_geometry(d)

    print('=' * 74)
    print('v2 引擎自检')
    assert self_check(d, verbose=True), 'v2 自检未通过'
    d.geo_nodes, d.geo = q2.precompute_geometry(d)

    if args.check:
        return
    print('=' * 74)
    print(f'ε-约束扫描（v2：精确组批 + 瓶颈感知解码 + 多种子档间热启动，'
          f'{args.seeds} 种子 × {args.iters} 轮）')
    st, asg, m, info = solve_recommended_v2(
        d, n_iter=args.iters, seeds=args.seeds, verbose=True,
        k_targets=[args.only_K] if args.only_K else None,
        patience=args.patience, tier_budget=args.tier_budget,
        warm_start=args.warm_start, save=args.save)
    print('=' * 74)
    print(f'推荐（取自 K<={info["K"]} 的可行档案）: 架次={m["n_trips"]} '
          f'能耗={m["total_E"]:.3f} kWh makespan={m["makespan"]:.1f} s '
          f'({m["makespan"] / 3600:.2f} h) 加权时延={m["tardiness"]:.0f} '
          f'硬违反={m["hard_viol"]}')
    n_no = [K for K, r in sorted(info['runs'].items()) if not r['archive']]
    if n_no:
        print(f'  无可行解的档位 {n_no} 按「未找到」如实报出')


if __name__ == '__main__':
    main()
