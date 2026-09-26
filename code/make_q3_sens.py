# -*- coding: utf-8 -*-
# 本程序及代码是在人工智能工具辅助下完成的。
# 工具名称：DeepSeek；版本/型号：DeepSeek V4.1 Flash（API 模型名 deepseek-flash）；
# 开发机构：杭州深度求索人工智能基础技术研究有限公司；
# 版本发布日期：2026 年 9 月 10 日。

"""问题三与通信侧的灵敏度扰动（只读派生，不写任何方案记录）。

为什么要有这个脚本
------------------
论文的问题三给出了一份零中断的联合方案（`results/q3_solution.json`），但方案里
有若干**本文自己定的**参数——中继机只有 2 架（题给，不能改）、服务时段合并窗口
（gap=900 s / max_job=3600 s，本文标定）、逐时刻审计步长（Δt=1 s，本文标定）。
只报一个「零中断」的结果，读者无从判断它是「参数选得好」还是「参数不敏感」。
本脚本对这三处各做一组扰动，并额外补一组通信侧参数（遮挡附加损耗与接收门限
各 ±3 dB）的扰动，把「结论对参数的依赖程度」如实落盘。

四条硬边界的执行方式
--------------------
1.  **不改任何求解器源码**。`core.py`/`q2.py`/`q3.py`/`q3_v2.py`/`q3_chain.py`/
    `solution_io.py` 是字节入哈希的，改一个字节就会作废已交付的方案记录。本脚本
    只**调用**它们的既有函数；对 `d.comm` 的替换是**运行期**换掉一个已加载对象
    的属性，不落盘、不写文件，源码字节不变（见 `_with_comm`）。
2.  **`save=False` 语义**：本脚本不调用任何导出函数，不写 `results/*_solution.json`，
    也不覆盖 `q3_coverage.csv` 等正式产物；产物只有 `results/q3_sens.csv` 一个。
3.  **先对上门禁再扰动**。扰动的前提是「当前代码能重算出已交付的那份方案」。
    故先按 `q3.py main()` 的同一条路径（`q2.recommended` → `q3.solve_relay`）重解
    一次，把 Δt=1 s 的终检占比、出动数与悬停站数与 `q3_coverage.csv` 的
    「联合优化」行逐位对比；对不上就**拒绝落盘并如实报告**，而不是拿一份基线都
    对不上的表当真。
4.  **扰动的口径要在表里写清楚**。凡「固定交付排班、只改某个参数」的扰动，其结论
    只能说「原方案对该参数有多敏感」，**不能**说「改成那个值以后还能不能排得下」——
    后者要重解，本脚本不做。每条记录都带 `扰动口径` 一列写明这一点。

用法
----
    python code/make_q3_sens.py            # 在工程根目录下运行
退出码 0 = 门禁通过并已写出 results/q3_sens.csv；1 = 门禁不通过（未写文件）。
"""
from __future__ import annotations

import io
import json
import os
import sys
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
RES = os.path.join(ROOT, 'results')
sys.path.insert(0, HERE)

import core                      # noqa: E402  只读调用
import q3                        # noqa: E402  只读调用
import q3_chain                  # noqa: E402  出动/能耗/接续的报告口径（q3 内部也用它）
from core import load_data       # noqa: E402
from q2 import precompute_geometry, recommended  # noqa: E402

# 门禁容差：占比逐位一致按 1e-6 判（与 solution_io 的浮点比较口径一致）
TOL = 1e-6

# 合并窗口的候选档：(gap, max_job)。交付档 (900, 3600) 必须在其中，
# 它同时充当本组扰动的内部对拍——重算出来必须与交付占比一致。
MERGE_WINDOWS = [(600.0, 2400.0), (900.0, 3600.0), (1200.0, 4800.0)]
MERGE_DELIVERED = (900.0, 3600.0)

# 通信侧扰动档：遮挡附加损耗 Lobs 与接收门限各 ±3 dB。
# 接收门限没有单独字段（= Psens + M），故按「Psens 下移 3 dB」等效于「门限抬 3 dB」，
# 即更严苛；反向亦然。两条链路的三组 Lmax 都由 CommModel 在构造时重算。
COMM_DELTAS = [-3.0, 0.0, 3.0]


def _rows(group, setting, metric, value, unit, caliber, note=''):
    """一行 = (扰动组, 取值, 指标, 数值)。长表比宽表好接：加一组扰动不用改表结构。"""
    return dict(扰动组=group, 取值=setting, 指标=metric,
                数值=float(value), 单位=unit, 扰动口径=caliber, 备注=note)


def _load_coverage():
    df = pd.read_csv(os.path.join(RES, 'q3_coverage.csv'), encoding='utf-8-sig')
    hit = df[df['口径'] == '联合优化']
    if len(hit) != 1:
        raise SystemExit('q3_coverage.csv 的「联合优化」行有 %d 行，应为 1 行' % len(hit))
    return hit.iloc[0]


def _frac_direct(d, assignment, dt):
    """按时间权重算直连占比。与 `q3.audit` 同一套权重口径，但不含中继判定。"""
    num = den = 0.0
    for a in assignment:
        t = d.transport_types[a['type']]
        pts = q3.sample_trip_trajectory(d, t, a['route'], a['boxes_at'],
                                        t0=a['start'], dt=dt)
        w = q3.time_weights(pts)
        for k, (_, lon, lat, alt) in enumerate(pts):
            den += w[k]
            if q3.direct_margin(d, lon, lat, alt) >= q3.MARGIN_DB:
                num += w[k]
    return num / den, den


def _with_comm(d, **delta):
    """在**运行期**换掉 `d.comm`，返回原对象以便复位。

    只改内存里的一个属性，不写任何文件，故入哈希的 `core.py` 字节不变。
    `_gw()` 读的是 `d.comm_params['gw_h']`（未扰动），`direct_margin`/`access_margin`
    读的是 `d.comm`，因此这里替换 `d.comm` 即可让判定口径整体平移。
    """
    p = dict(d.comm_params)
    for k, v in delta.items():
        p[k] = p[k] + v
    old = d.comm
    d.comm = core.CommModel(p)
    return old


def main():
    t_all = time.time()
    d = load_data()
    d.geo_nodes, d.geo = precompute_geometry(d)

    # ---- 1) 按交付链的同一条路径重解一次问题三 ----
    trips, assignment, q2_m, q2_order = recommended(d)
    print('读回问题二推荐方案：%d 架次 / %.3f kWh / makespan %.1f s'
          % (q2_m['n_trips'], q2_m['total_E'], q2_m['makespan']))

    # 架次编号**继承**上游方案，取的是**错峰之前**的时刻。这一句必须抢在
    # `solve_relay` 之前：错峰会改写 `start`，之后再比 `start` 必然全不符——
    # 那正是 `solution_io.inherit_trip_ids` 该报的错，不是它判错了。q3.py 的 main
    # 也是在这个执行点上取的编号，两处排序口径（按开始时刻升序）一致。
    assignment = sorted(assignment, key=lambda x: x['start'])
    with open(os.path.join(RES, 'q2_solution.json'), encoding='utf-8') as f:
        q2_sol = json.load(f)
    trip_ids = q3.inherit_trip_ids(assignment, q2_sol)
    print('架次编号继承上游方案 %s（%s..%s）'
          % (q2_sol['solution_id'], trip_ids[0], trip_ids[-1]))
    # 编号是**按位置**接到架次上的，故此后必须保证位置不变。先留一份身份指纹，
    # 重解之后逐位核对：错峰只改 `start`，机型/无人机/电池三样不会变。
    id_fp = [(a['type'], a['uav'], a['battery']) for a in assignment]

    t0 = time.time()
    res = q3.solve_relay(d, assignment, verbose=False)
    if not res['ok']:
        raise SystemExit('问题三重解失败：%s' % res.get('reason'))
    print('问题三重解完成（%.1f s）' % (time.time() - t0))

    asg = res['assignment']
    cands, cover, sel, intervals = res['cands'], res['cover'], res['sel'], res['intervals']
    if len(asg) != len(id_fp) or [(a['type'], a['uav'], a['battery']) for a in asg] != id_fp:
        raise SystemExit('重解后的架次顺序与继承编号时的顺序不一致——'
                         '编号是按位置接上去的，此时必须停，不能按位置硬套')
    print('架次顺序核对通过（%d 个架次，编号与位置一一对应）' % len(asg))

    # ---- 2) 门禁：重解的基线必须与交付结果逐位一致 ----
    frac, segs, gaps = q3.audit(d, asg, res['relays'], cands, dt=q3.DT_AUDIT)
    n_out, E_relay, n_chain = q3_chain.outing_stats(res['relays'])
    n_st = len({x['station'] for x in res['relays']})
    cov = _load_coverage()
    checks = [
        ('直连时间占比', frac['direct'], float(cov['直连时间占比'])),
        ('中继时间占比', frac['relay'], float(cov['中继时间占比'])),
        ('中断时间占比', frac['gap'], float(cov['中断时间占比'])),
    ]
    print('=' * 74)
    print('门禁：重解基线 vs q3_coverage.csv「联合优化」行')
    ok = True
    for name, got, exp in checks:
        good = abs(got - exp) <= TOL
        ok &= good
        print('  [%s] %-12s 重解=%.6f 交付=%.6f' % ('OK  ' if good else 'FAIL', name, got, exp))
    for name, got, exp in [('中继出动数', n_out, int(cov['中继架次'])),
                           ('悬停站数', n_st, int(cov['悬停站']))]:
        good = got == exp
        ok &= good
        print('  [%s] %-12s 重解=%d 交付=%d' % ('OK  ' if good else 'FAIL', name, got, exp))
    if gaps:
        ok = False
        print('  [FAIL] 重解出现 %d 个中断采样点，与交付的零中断不符' % len(gaps))
    print('=' * 74)
    if not ok:
        print('门禁不通过：当前代码重算不出已交付的问题三方案。')
        print('按本脚本的纪律，此时不落盘 results/q3_sens.csv——'
              '基线对不上的扰动表会把「换了参数」与「换了基线」混在一起，不能当证据。')
        return 1

    rows = []
    BASE_CAL = '固定交付排班，只改该参数；不含重解'

    # ---- 3) (i) 中继无人机数 n_relay ∈ {1,2,3} ----
    # 交付排班下「同一时刻最少需要几个悬停站」，再与 n_relay 比。
    # 超额站·秒 > 0 表示该排班在 n_relay 架中继下排不下（不是方案不可行，是排班不够）。
    win_by_iv = {j: (iv['t_start'], iv['t_end']) for j, iv in enumerate(intervals)}
    cache = {}
    print('-' * 74)
    print('(i) 中继无人机数扰动（交付排班固定）')
    for n in (1, 2, 3):
        excess, peak = q3.simultaneous_station_excess(win_by_iv, cover, n, cache)
        print('  n_relay=%d: 超额站·秒=%.1f，最少站数峰值=%d' % (n, excess, peak))
        rows.append(_rows('中继机数', 'n_relay=%d' % n, '超额站·秒', excess, '站·s', BASE_CAL,
                          '最少站数峰值 %d' % peak))
    # 峰值与 n_relay 无关，单列一条，正文引用它说明「同时最多要 3 个站」
    _ex0, peak0 = q3.simultaneous_station_excess(win_by_iv, cover, 2, cache)
    rows.append(_rows('中继机数', '不随 n_relay 变', '同时最少站数峰值', peak0, '个', BASE_CAL,
                      '该值只由失效区间与覆盖关系决定'))

    # ---- 4) (ii) 服务时段合并窗口 ----
    print('-' * 74)
    print('(ii) 服务时段合并窗口扰动')
    for gap, max_job in MERGE_WINDOWS:
        # `merge_jobs` 只给「要服务哪些站、各服务到哪一刻」的计划；`schedule_relays`
        # 返回的是 (trips, skipped) 二元组，**不是一个列表**——漏了拆包会把整个
        # 二元组当成 trips 传下去，报出「list indices must be integers」。
        jobs = q3.merge_jobs(intervals, sel, gap=gap, max_job=max_job)
        rt, skip = q3.schedule_relays(d, jobs, cands, verbose=False)
        o, E, ch = q3_chain.outing_stats(rt)
        f, _s, _g = q3.audit(d, asg, rt, cands, dt=q3.DT_AUDIT)
        tag = 'gap=%g, max_job=%g%s' % (gap, max_job,
                                        '（交付档）' if (gap, max_job) == MERGE_DELIVERED else '')
        print('  %s: 计划服务=%d 次，实际服务=%d 次，弃飞=%d，出动=%d 次，接续=%d 次，'
              '直连=%.6f 中继=%.6f 中断=%.6f，中继能耗=%.3f kWh'
              % (tag, len(jobs), len(rt), len(skip), o, ch,
                 f['direct'], f['relay'], f['gap'], E))
        # 交付档的重算必须与门禁基线一致，否则说明扰动链本身不可信
        if (gap, max_job) == MERGE_DELIVERED and (
                abs(f['direct'] - cov['直连时间占比']) > TOL
                or abs(f['gap'] - cov['中断时间占比']) > TOL
                or o != int(cov['中继架次'])):
            print('  ! 交付档重算与门禁基线不一致，合并窗口一组的结论不可信')
            return 1
        cal = '固定交付运输排班，按该窗口重组中继架次并重排班后重做终检'
        # 「计划」是合并规则给的架次数，「实际」是排班后真的飞成的架次数，
        # 两者之差即弃飞。只报计划数会把弃飞藏起来，故两个都落盘。
        rows.append(_rows('合并窗口', tag, '计划悬停站服务次数', len(jobs), '次', cal))
        rows.append(_rows('合并窗口', tag, '实际悬停站服务次数', len(rt), '次', cal))
        rows.append(_rows('合并窗口', tag, '弃飞架次数', len(skip), '次', cal))
        rows.append(_rows('合并窗口', tag, '中继出动次数', o, '次', cal))
        rows.append(_rows('合并窗口', tag, '站间接续次数', ch, '次', cal))
        rows.append(_rows('合并窗口', tag, '直连时间占比', f['direct'], '—', cal))
        rows.append(_rows('合并窗口', tag, '中继时间占比', f['relay'], '—', cal))
        rows.append(_rows('合并窗口', tag, '中断时间占比', f['gap'], '—', cal))
        rows.append(_rows('合并窗口', tag, '中继能耗kWh', E, 'kWh', cal))

    # ---- 5) (iii) 逐时刻审计步长 ----
    print('-' * 74)
    print('(iii) 审计步长扰动（交付方案固定）')
    for dt in (0.5, 1.0, 2.0):
        f, _s, g = q3.audit(d, asg, res['relays'], cands, dt=dt)
        print('  Δt=%.1f s: 直连=%.6f 中继=%.6f 中断=%.6f（中断点 %d 个）'
              % (dt, f['direct'], f['relay'], f['gap'], len(g)))
        cal = '交付方案固定，只改重采样的步长；不含重解'
        rows.append(_rows('审计步长', 'Δt=%g s' % dt, '直连时间占比', f['direct'], '—', cal))
        rows.append(_rows('审计步长', 'Δt=%g s' % dt, '中继时间占比', f['relay'], '—', cal))
        rows.append(_rows('审计步长', 'Δt=%g s' % dt, '中断时间占比', f['gap'], '—', cal))

    # ---- 6) (iv) 通信侧参数 ±3 dB ----
    # 只重算「运输轨迹上有多少时间可直连」这一件事，不重解中继方案。
    # 故结论只能说「对中继保障的需求量有多敏感」，不能说「新参数下方案还行不行」。
    print('-' * 74)
    print('(iv) 通信侧参数扰动（交付轨迹固定）')
    for dL in COMM_DELTAS:
        old = _with_comm(d, Lobs=dL)
        try:
            fr, tot = _frac_direct(d, asg, dt=q3.DT_AUDIT)
        finally:
            d.comm = old
        print('  Lobs %+.0f dB: 直连占比=%.6f（需中继 %.6f）' % (dL, fr, 1.0 - fr))
        cal = '交付轨迹固定，只改链路预算中的该项；不含重解中继方案'
        rows.append(_rows('通信·遮挡损耗', 'Lobs%+.0f dB' % dL, '直连时间占比', fr, '—', cal))
    for dP in COMM_DELTAS:
        old = _with_comm(d, Psens=dP)      # Psens 下移 = 门限抬高 = 更严苛
        try:
            fr, tot = _frac_direct(d, asg, dt=q3.DT_AUDIT)
        finally:
            d.comm = old
        print('  接收门限 %+.0f dB: 直连占比=%.6f（需中继 %.6f）' % (dP, fr, 1.0 - fr))
        cal = '交付轨迹固定，只改链路预算中的该项；不含重解中继方案'
        rows.append(_rows('通信·接收门限', '门限%+.0f dB' % dP, '直连时间占比', fr, '—', cal))

    # ---- 7) (v) 交付解的最小裕量 ----
    # 图 `fig_q3_margin` 要报三个「最小裕量」，其中两个（架次的最小直连裕量、失效区间
    # 的最小接入裕量）**不在任何结果表里**：它们只能由逐采样点重算得到。按本仓库的
    # 分层纪律，图脚本只读结果表、不落盘，故这两个量在这里算出来落进 q3_sens.csv，
    # 再由 paper_metrics.py 取成宏。第三个（最小硬时限余量）已在 q3_metrics.csv 的
    # 「最小硬时限余量」一行，不重复落盘。
    print('-' * 74)
    print('(v) 交付解的最小裕量（交付轨迹与站址固定，逐 Δt=1 s 采样点取最小）')
    st = pd.read_csv(os.path.join(RES, 'q3_stations.csv'),
                     encoding='utf-8-sig').set_index('悬停站编号')
    ivd = pd.read_csv(os.path.join(RES, 'q3_intervals.csv'), encoding='utf-8-sig')
    # 失效区间按**交付表**的架次编号归组；编号由上面的继承给出，不在这里现推。
    if len(trip_ids) != len(asg):
        raise SystemExit('继承到的架次编号 %d 个，与重解出的 %d 个架次不符——'
                         '此时按位置配对会把某个架次的区间记到别人头上'
                         % (len(trip_ids), len(asg)))
    iv_by_trip, iv_station = {}, {}
    for _, r in ivd.iterrows():
        iv_by_trip.setdefault(r['运输架次编号'], []).append(
            (float(r['区间起点s']), float(r['区间终点s']),
             r['悬停站编号'], r['失效区间编号']))
        iv_station[r['失效区间编号']] = r['悬停站编号']

    min_direct, min_access = {}, []
    for i, a in enumerate(asg):
        tid = trip_ids[i]
        t = d.transport_types[a['type']]
        pts = q3.sample_trip_trajectory(d, t, a['route'], a['boxes_at'],
                                        t0=a['start'], dt=q3.DT_AUDIT)
        min_direct[tid] = min(q3.direct_margin(d, lon, lat, alt)
                              for _t, lon, lat, alt in pts)
        for (a1, b1, sname, gid) in iv_by_trip.get(tid, ()):
            s = st.loc[sname]
            stp = (float(s['悬停经度']), float(s['悬停纬度']), float(s['悬停海拔m']))
            vals = [q3.access_margin(d, stp, lon, lat, alt)
                    for tt, lon, lat, alt in pts if a1 - 1e-9 <= tt <= b1 + 1e-9]
            if not vals:
                raise SystemExit('失效区间 %s 内没有任何轨迹采样点' % gid)
            min_access.append((gid, min(vals)))
    # 样本数必须与交付表一致：少了就是漏采了某个架次或某个区间，取出来的「最小」
    # 会偏大，而图与正文都会把它当成全体的最小来写。架次表按「有区间的架次」核，
    # 因为交付表里本来就只有这 19 个架次有失效区间。
    n_iv_trip = len({t for t in iv_by_trip})
    if len(min_direct) != len(asg) or len(min_access) != len(ivd):
        raise SystemExit('最小裕量的样本数与交付表不符：架次 %d/%d、失效区间 %d/%d'
                         % (len(min_direct), len(asg), len(min_access), len(ivd)))
    if set(iv_by_trip) - set(min_direct):
        raise SystemExit('交付表里出现了重解结果中不存在的架次编号：%s'
                         % sorted(set(iv_by_trip) - set(min_direct))[:5])
    print('  （含失效区间的架次 %d 个，全部在重解结果中解析到）' % n_iv_trip)
    n_neg = sum(1 for _g, v in min_access if v < -1e-6)
    if n_neg:
        raise SystemExit('有 %d 个失效区间的接入裕量为负，与交付的零中断不符——'
                         '此时应先在图上暴露出来，而不是把该区间从统计里去掉' % n_neg)
    worst_t = min(min_direct, key=min_direct.get)
    worst_g = min(min_access, key=lambda kv: kv[1])[0]
    acc_min = min(v for _g, v in min_access)
    n_ok_d = sum(1 for v in min_direct.values() if v >= 0)
    print('  直连：%d/%d 架次的最小直连裕量 ≥ 0；最小 %+.4f dB（%s）'
          % (n_ok_d, len(min_direct), min_direct[worst_t], worst_t))
    print('  接入：%d 个区间全部 ≥ 0；最小 %+.4f dB（%s）'
          % (len(min_access), acc_min, worst_g))
    CAL_M = '交付轨迹与站址固定，逐 Δt=1 s 采样点取最小；不含重解'
    # 逐对象的明细行也一并落盘，**不是**为了凑行数：图 `fig_q3_margin` 的 (b)(c)
    # 两格要逐架次、逐区间地画条形。若让图脚本自己去调 direct_margin/access_margin
    # 重算一遍，同一批数字就有了两条独立算路——正文引用宏（取自本表的合计行）、
    # 图里画的是另一条算路的结果，两边一旦分叉，论文会与自己的图互相矛盾而无人
    # 察觉。故此处把明细落盘，图与正文同源；图的注记另与本表的合计行逐位核对。
    rows.append(_rows('交付解·最小裕量', '全部架次取最小', '最小直连裕量',
                      min_direct[worst_t], 'dB', CAL_M, '出现在 %s' % worst_t))
    rows.append(_rows('交付解·最小裕量', '全部架次', '最小直连裕量非负的架次数',
                      n_ok_d, '个', CAL_M, '共 %d 个架次' % len(min_direct)))
    rows.append(_rows('交付解·最小裕量', '全部失效区间取最小', '最小接入裕量',
                      acc_min, 'dB', CAL_M, '出现在 %s' % worst_g))
    for ftid, v in min_direct.items():
        rows.append(_rows('交付解·最小裕量·逐架次', ftid, '最小直连裕量', v, 'dB',
                          CAL_M, '非负即该架次全轨迹无需中继'))
    for gid, v in min_access:
        rows.append(_rows('交付解·最小裕量·逐区间', gid, '最小接入裕量', v, 'dB',
                          CAL_M, '交付方案由 %s 保障' % iv_station.get(gid, '')))

    out = pd.DataFrame(rows)
    path = os.path.join(RES, 'q3_sens.csv')
    out.to_csv(path, index=False)
    print('=' * 74)
    print('已写出 %s（%d 行）' % (os.path.relpath(path, ROOT), len(out)))
    print('总耗时 %.1f s' % (time.time() - t_all))
    return 0


if __name__ == '__main__':
    sys.exit(main())
