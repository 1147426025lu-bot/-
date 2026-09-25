# -*- coding: utf-8 -*-
# 本程序及代码是在人工智能工具辅助下完成的。
# 工具名称：DeepSeek；版本/型号：DeepSeek V4.1 Flash（API 模型名 deepseek-flash）；
# 开发机构：杭州深度求索人工智能基础技术研究有限公司；
# 版本发布日期：2026 年 9 月 10 日。

"""
回归测试集

每个用例只做一件事：把**曾经出过错的那条推理**钉死。用例名与《整体修订实施指南》
§10.4 的编号对应（T03、T04、…），便于逐条核销。

覆盖情况（指南 §10.4）：T03–T22 全部落地；T01/T02 由 T-Q1/T-Q1b 覆盖（混合机型
可行域包含单机型、完整前沿保留互不支配标签）；T21 覆盖打包预检的清单校验，T22 覆盖
结果表回填的列名对齐。全阶段干净重跑的可追溯性由各脚本落盘的 solution_id/solver_hashes
与 verify.py 的独立重算共同承担，不在此重复。

设计约束：
  - 用例返回 `{name, passed, errors, metrics, tolerance}`；`main()` 汇总并在任何
    用例失败时以非零退出码结束。**空报告列表视为失败**——一个都没跑不算通过。
  - 需要真实数据的用例在读不到结果表时**跳过并记 skipped**，不计入「通过」，
    也不假装通过。
  - 断言的是**可复算的量**（秒数、编号、行数），不是「看起来对不对」。
  - 构造「表被改坏」的场景一律走 `_tamper()`：真的把结果表复制出来改掉再让
    verify 去读，而不是直接调用它的内部函数——否则证明不了检查项会拦下来。

运行：python code/tests_regression.py
"""
import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np                                                    # noqa: E402
import solution_io as S                                               # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, '..', 'results'))

CASES = []
_STDOUT_KEEP = []


def case(fn):
    CASES.append(fn)
    return fn


def result(name, passed, errors=None, metrics=None, tolerance=None, skipped=False):
    return dict(name=name, passed=bool(passed), errors=list(errors or []),
                metrics=metrics or {}, tolerance=tolerance, skipped=skipped)


def _need(*names):
    """结果表齐全才跑；缺表返回缺失清单。"""
    miss = [n for n in names if not os.path.exists(os.path.join(OUT, n))]
    return miss


# ===========================================================================
# A 组：结果接口与稳定编号（指南 §3）
# ===========================================================================
@case
def t_roundtrip():
    """方案读写往返：内容摘要稳定、逐字段无损。"""
    p = os.path.join(OUT, 'q2_solution.json')
    miss = _need('q2_solution.json')
    if miss:
        return result('T-R1 方案读写往返', True, skipped=True,
                      errors=['缺少 %s，请先运行 python code/q2.py' % miss])
    a = S.load_solution(p, stage='q2')
    body = {k: v for k, v in a.items() if k != 'solution_id'}
    b = json.loads(json.dumps(body, ensure_ascii=False))   # 过一遍 json 往返
    b['solution_id'] = a['solution_id']
    errs = []
    if S.content_id('q2', b) != a['solution_id']:
        errs.append('内容摘要经 json 往返后改变，序列化不是无损的')
    if json.dumps(body, sort_keys=True, ensure_ascii=False) != \
            json.dumps({k: v for k, v in b.items() if k != 'solution_id'},
                       sort_keys=True, ensure_ascii=False):
        errs.append('字段经 json 往返后不等')
    return result('T-R1 方案读写往返', not errs, errs,
                  metrics={'solution_id': a['solution_id'],
                           'n_trips': len(a['transport_trips'])},
                  tolerance='逐位相等')


@case
def t_tamper_detected():
    """方案文件被改动后必须拒绝使用（陈旧/被篡改的上游不得静默通过）。"""
    p = os.path.join(OUT, 'q2_solution.json')
    if _need('q2_solution.json'):
        return result('T-R2 篡改检出', True, skipped=True, errors=['缺 q2_solution.json'])
    sol = S.load_solution(p, stage='q2')
    sol['transport_trips'][0]['start'] += 1.0
    tmp = p + '.tamper'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(sol, f, ensure_ascii=False, sort_keys=True)
    errs = []
    try:
        S.load_solution(tmp, stage='q2')
        errs.append('篡改后的方案未报错')
    except ValueError:
        pass
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return result('T-R2 篡改检出', not errs, errs, tolerance='抛 ValueError')


@case
def t_solver_hash_mismatch():
    """求解器源码哈希不符时必须拒绝上游方案（代码改过，方案不再可复现）。"""
    p = os.path.join(OUT, 'q2_solution.json')
    if _need('q2_solution.json'):
        return result('T-R3 求解器哈希校验', True, skipped=True, errors=['缺方案文件'])
    fake = dict(S.solver_hashes(S.Q2_SOLVER_FILES))
    fake['code/q2.py'] = '0' * 64
    errs = []
    try:
        S.load_solution(p, stage='q2', solver_hashes_expect=fake)
        errs.append('求解器哈希不符却通过了校验')
    except ValueError:
        pass
    return result('T-R3 求解器哈希校验', not errs, errs, tolerance='抛 ValueError')


@case
def t_foreign_key():
    """外键校验：未知编号、空引用、重复编号都必须显式失败。"""
    p = os.path.join(OUT, 'q2_solution.json')
    if _need('q2_solution.json'):
        return result('T-R4 外键校验', True, skipped=True, errors=['缺方案文件'])
    sol = S.load_solution(p, stage='q2')
    tid0 = sol['transport_trips'][0]['trip_id']
    errs = []

    # 1) 合法引用通过
    ok_iv = [dict(interval_id='G01', trip_id=tid0, relay_trip_id='')]
    try:
        S.check_foreign_keys(sol, ok_iv, require_relay_binding=False)
    except ValueError as e:
        errs.append('合法引用被误判：%s' % e)

    # 2) 未知运输架次编号 → 必须失败（旧版按排序位置猜编号时此处会静默错配）
    try:
        S.check_foreign_keys(sol, [dict(interval_id='G01', trip_id='T99',
                                        relay_trip_id='')],
                             require_relay_binding=False)
        errs.append('未知运输架次编号 T99 未被检出')
    except ValueError:
        pass

    # 3) 需要中继绑定时空引用 → 必须失败
    try:
        S.check_foreign_keys(sol, [dict(interval_id='G01', trip_id=tid0,
                                        relay_trip_id='')],
                             require_relay_binding=True)
        errs.append('空的中继架次引用未被检出')
    except ValueError:
        pass

    # 4) 编号重复的方案 → 必须失败
    bad = json.loads(json.dumps(sol, ensure_ascii=False))
    bad['transport_trips'][1]['trip_id'] = bad['transport_trips'][0]['trip_id']
    try:
        S.validate_solution(bad)
        errs.append('重复架次编号未被检出')
    except ValueError:
        pass
    return result('T-R4 外键校验', not errs, errs, tolerance='抛 ValueError')


@case
def t_stable_numbering():
    """
    稳定编号（直接覆盖 F03 的错配）：把架次表按任意顺序重排、并整体重新排序，
    `trip_id` 必须**保持不变**，且按 trip_id 建的外键连接结果与行序无关。
    """
    p = os.path.join(OUT, 'q2_solution.json')
    if _need('q2_solution.json'):
        return result('T-R5 稳定编号', True, skipped=True, errors=['缺方案文件'])
    sol = S.load_solution(p, stage='q2')
    trips = sol['transport_trips']
    base = {t['trip_id']: (t['uav'], t['battery'], t['type']) for t in trips}

    rng = np.random.default_rng(20260924)
    perm = list(rng.permutation(len(trips)))
    shuffled = [trips[i] for i in perm]
    got = {t['trip_id']: (t['uav'], t['battery'], t['type']) for t in shuffled}
    errs = []
    if got != base:
        errs.append('重排行序后 trip_id → 实体指派的映射改变')

    # 行序打乱后，按 trip_id 的连接结果必须逐项相同
    by_id = S.trip_of(dict(sol, transport_trips=shuffled))
    for tid, v in base.items():
        if (by_id[tid]['uav'], by_id[tid]['battery'], by_id[tid]['type']) != v:
            errs.append('%s 的连接结果随行序改变' % tid)
    return result('T-R5 稳定编号', not errs, errs,
                  metrics={'n_trips': len(trips)}, tolerance='逐位相等')


@case
def t_inherit_mismatch():
    """下游重解出的指派若与上游方案不符，必须停止而不是继续算。"""
    p = os.path.join(OUT, 'q2_solution.json')
    if _need('q2_solution.json'):
        return result('T-R6 上游一致性', True, skipped=True, errors=['缺方案文件'])
    sol = S.load_solution(p, stage='q2')
    ups = sol['transport_trips']
    fake = [dict(type=t['type'], uav=t['uav'], battery=t['battery'],
                 start=t['start'], duration=t['duration'], E=t['energy_kwh'])
            for t in ups]
    errs = []
    ids = S.inherit_trip_ids(fake, sol)
    if ids != [t['trip_id'] for t in ups]:
        errs.append('一致输入下的编号继承结果不对')
    fake[0]['uav'] = '__不存在的无人机__'
    try:
        S.inherit_trip_ids(fake, sol)
        errs.append('指派不符时未停止')
    except ValueError:
        pass
    return result('T-R6 上游一致性', not errs, errs, tolerance='抛 ValueError')


# ===========================================================================
# B 组：问题一混合机型（F01/F02）
# ===========================================================================
_D = {}


def _data():
    """惰性加载数据（DEM 解析较慢，只在真要用时才付这份代价）。"""
    if 'd' not in _D:
        from core import load_data
        import q1
        _D['d'] = load_data()
        _D['q1'] = q1
    return _D['d'], _D['q1']


# 修订前基线（提交 3eef64c 的 q1 输出，论文 5.4 节报 18 架次 / 63.180 kWh）
BASE_Q1_K = 18
BASE_Q1_E = 63.180449701


def _single_type_best(d, q1, si_id, types):
    """旧口径（每区单一机型）下的最优 (k, E)。"""
    best = None
    for tid in types:
        dp = q1.dp_area(d, si_id, tid)
        if dp is None:
            continue
        k = min(dp['front'])
        cand = (k, dp['front'][k])
        if best is None or cand < best:
            best = cand
    return best


@case
def t_q1_mixed_gate():
    """
    门禁：把混合域限定为单机型时，必须复现修订前的逐区最优解与总数 18 架次 /
    63.180 kWh；放开混用后不得更差（架次不增、能耗不增）。
    """
    d, q1 = _data()
    errs = []
    old_k = old_e = 0
    new_k = new_e = 0.0
    worst = 0.0
    for si in d.S:
        b = _single_type_best(d, q1, si, q1.TYPES)
        if b is None:
            errs.append('%s 在单机型口径下不可作业' % si)
            continue
        dpm = q1.dp_area_mixed(d, si)
        if dpm is None:
            errs.append('%s 在混合机型口径下反而不存在可行解' % si)
            continue
        nk = min(dpm['front'])
        ne = min(x[0] for x in dpm['front'][nk])
        old_k += b[0]; old_e += b[1]
        new_k += nk; new_e += ne
        worst = max(worst, abs(b[1] - ne))
        if nk > b[0]:
            errs.append('%s 混用后架次数反而上升：%d > %d' % (si, nk, b[0]))
        if nk == b[0] and ne > b[1] + 1e-9:
            errs.append('%s 混用后同架次下能耗反而上升：%.9f > %.9f' % (si, ne, b[1]))
    if old_k != BASE_Q1_K or abs(old_e - BASE_Q1_E) > 1e-9:
        errs.append('单机型口径未复现基线：实得 %d 架次 / %.9f kWh，基线 %d / %.9f'
                    % (old_k, old_e, BASE_Q1_K, BASE_Q1_E))
    if new_k > old_k or new_e > old_e + 1e-9:
        errs.append('混用后总指标更差：%d/%.9f vs %d/%.9f' % (new_k, new_e, old_k, old_e))
    # 逐区最优值必须一致：混合域是单机型域的超集，最优值只可能更优
    return result('T-Q1 混合机型门禁', not errs, errs,
                  metrics={'单机型基线': '%d 架次 / %.6f kWh' % (old_k, old_e),
                           '混用后': '%d 架次 / %.6f kWh' % (new_k, new_e),
                           '节能': '%.2f%%' % (100 * (old_e - new_e) / old_e)},
                  tolerance='架次数相等；能耗 1e-9 绝对容差')


@case
def t_q1_front_dominance():
    """前沿自洽：每个 (区, k) 的标签两两互不支配，且能耗严格随 k 递减后回升不出现被支配点。"""
    d, q1 = _data()
    errs = []
    n_label = 0
    for si in d.S:
        dpm = q1.dp_area_mixed(d, si)
        for k, labs in dpm['front'].items():
            n_label += len(labs)
            for i, (e1, t1) in enumerate(labs):
                for j, (e2, t2) in enumerate(labs):
                    if i != j and e2 <= e1 + 1e-9 and t2 <= t1 + 1e-9:
                        errs.append('%s k=%d 标签 %d 被 %d 支配' % (si, k, i, j))
    return result('T-Q1b 前沿互不支配', not errs, errs,
                  metrics={'标签总数': n_label}, tolerance='两两互不支配')


@case
def t_q1_mixed_reconstruct():
    """回溯自洽：回溯出的逐架次模式必须恰好覆盖该区全部货箱，且能耗/时间与标签一致。"""
    d, q1 = _data()
    errs = []
    for si in d.S:
        dpm = q1.dp_area_mixed(d, si)
        for k, labs in dpm['front'].items():
            for li, (E, T) in enumerate(labs):
                bins = q1.reconstruct_mixed(dpm, k, li)
                if len(bins) != k:
                    errs.append('%s k=%d 标签%d 回溯出 %d 架次' % (si, k, li, len(bins)))
                    continue
                tot = [0] * len(dpm['classes'])
                for pi, cnt in bins:
                    for ci, v in enumerate(cnt):
                        tot[ci] += v
                want = [c[2] for c in dpm['classes']]
                if tot != want:
                    errs.append('%s k=%d 标签%d 货箱覆盖不符：%s vs %s'
                                % (si, k, li, tot, want))
                e_sum = sum(dpm['pats'][pi]['E'] for pi, _ in bins)
                t_sum = sum(dpm['pats'][pi]['T'] for pi, _ in bins)
                if abs(e_sum - E) > 1e-9 or abs(t_sum - T) > 1e-6:
                    errs.append('%s k=%d 标签%d 回溯聚合值与标签不符' % (si, k, li))
    return result('T-Q1c 回溯自洽', not errs, errs, tolerance='1e-9 / 1e-6')


# ===========================================================================
# G 组：验证程序本身（指南 §10.4 的 T03–T20）
#
# 这一组钉的是「曾经出过错的那条推理」，不是「代码现在跑得通」：多数用例
# 构造一个**必须被判错**的场景，再要求检查项确实报错——只报「跑通了」的
# 测试挡不住 F07 那类「把违约吞掉」的错误。
# ===========================================================================
import contextlib                                                # noqa: E402
import copy                                                      # noqa: E402
import importlib                                                 # noqa: E402
import io                                                        # noqa: E402
import shutil                                                    # noqa: E402
import tempfile                                                  # noqa: E402

import pandas as pd                                              # noqa: E402
import openpyxl                                                  # noqa: E402


def _mod(name):
    """按需 import（q3/q4 会连带载入 core 与 pandas，不必在收集用例时就付代价）。"""
    return importlib.import_module(name)


# verify 检查项读取的表；夹具只复制这些，避免整目录搬运。
_FIXTURE_TABLES = ['q2_transport_trips.csv', 'q2_box_delivery.csv', 'q2_pallet.csv',
                   'q1_recommended_batching.csv', 'q1_pareto.csv', 'q1_ilp_mixed.csv',
                   'q3_relay_trips.csv', 'q3_comm_phases.csv', 'q3_intervals.csv',
                   'q3_stations.csv', 'q3_coverage.csv', 'q3_transport_trips.csv',
                   'q3_box_delivery.csv', 'q3_metrics.csv', 'q3_stagger.csv']


@contextlib.contextmanager
def _tamper(mutate):
    """
    把结果表复制到临时目录、按 `mutate(tmp)` 改其中若干张，再让 verify 读临时目录。

    这是唯一能构造「表被人为改坏」场景的入口：verify 只吃 results/*.csv，
    要证明它会因此报错，就得真的给它一份坏表——而不是直接调用它内部的函数。
    """
    v = _mod('verify')
    tmp = tempfile.mkdtemp(prefix='regr_')
    old = v.RES
    try:
        for n in _FIXTURE_TABLES:
            src = os.path.join(OUT, n)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(tmp, n))
        mutate(tmp)
        v.RES = tmp
        yield v
    finally:
        v.RES = old
        shutil.rmtree(tmp, ignore_errors=True)


def _rd(tmp, name):
    return pd.read_csv(os.path.join(tmp, name), encoding='utf-8-sig')


def _wr(df, tmp, name):
    df.to_csv(os.path.join(tmp, name), index=False, encoding='utf-8-sig')


def _hit(report, *keys):
    """报告里是否出现了包含全部关键词的错误行。"""
    return any(all(k in e for k in keys) for e in report['errors'])


# --- T03 -------------------------------------------------------------------
@case
def t_box_ids_distinct():
    """
    T03：两个箱号不同但质量、体积相同的货箱，落到具体箱号后必须**各出现一次**。

    同类货箱在全部约束与代价下可互换，DP 只按「类」计数；把计数还原成箱号时
    若按类计数去重或按编号排序后截断，就会把两个箱子并成一个（或让同一个箱号
    发到两个架次上）。逐区检查：全方案箱号无重复、且恰好覆盖该区全部货箱。
    """
    d, q1 = _data()
    errs, n_pair, n_box = [], 0, 0
    for si in d.S:
        boxes = d.boxes_by_service[si]
        dp = q1.dp_area_mixed(d, si)
        if dp is None:
            errs.append('%s 无可行方案' % si)
            continue
        k = min(dp['front'])
        li = min(range(len(dp['front'][k])), key=lambda i: dp['front'][k][i][0])
        bins = q1.box_ids_of_bins(dp, k, li, boxes)
        ids = [x for _, _, xs in bins for x in xs]
        n_box += len(ids)
        if len(ids) != len(set(ids)):
            dup = sorted({x for x in ids if ids.count(x) > 1})
            errs.append('%s 箱号重复：%s' % (si, '、'.join(dup[:5])))
        if sorted(ids) != sorted(b['id'] for b in boxes):
            miss = sorted({b['id'] for b in boxes} - set(ids))
            extra = sorted(set(ids) - {b['id'] for b in boxes})
            errs.append('%s 箱号覆盖不符：缺 %s，多 %s'
                        % (si, '、'.join(miss[:5]) or '无', '、'.join(extra[:5]) or '无'))
        # 同 (质量, 体积) 且箱数 >= 2 的类，必须有几个箱号就出现几次
        for (m, v, c, pool) in dp['classes']:
            if c < 2:
                continue
            n_pair += 1
            want = sorted(boxes[i]['id'] for i in pool)
            got = sorted(x for x in ids if x in set(want))
            if got != want:
                errs.append('%s 同类(%s kg, %s m3) 的 %d 个箱号未各出现一次：%s'
                            % (si, m, v, c, got))
    return result('T03 同质异号箱各出现一次', not errs, errs,
                  metrics={'检查服务区': len(d.S), '检查同质类': n_pair,
                           '落位箱号': n_box}, tolerance='箱号集合逐位相等')


# --- T04 -------------------------------------------------------------------
@case
def t_relay_binding_by_id():
    """
    T04：失效表只含 T03 与 T20 时，中继只能绑到 T03/T20，**不得**误绑 T01。

    旧版按「排序后的位置」把编号映射回架次下标：失效表里第一个出现的编号
    （T03）会被映射到 0 号位，于是 T03 起的绑定整体错位两格。这里伪造一份
    「20 个架次、失效区间只落在 T03 与 T20」的方案，要求按编号查表的结果精确。
    """
    q4 = _mod('q4')
    d, _q1 = _data()
    n = 20
    trips = [dict(trip_id='T%02d' % (i + 1), type='A', start=float(i), duration=1.0,
                  energy_kwh=0.1, box_ids_by_service={'S001': ['B%02d' % i]})
             for i in range(n)]
    relays_in = [dict(relay_trip_id='R01', station_id='S01', start=0.0, return_time=10.0,
                      energy_kwh=0.5),
                 dict(relay_trip_id='R02', station_id='S02', start=20.0, return_time=30.0,
                      energy_kwh=0.5)]
    sol = dict(stage='q3', transport_trips=trips, relay_trips=relays_in,
               relay_uav_ids=['RU1', 'RU2'], station_ids=['S01', 'S02'],
               component_ids=['RC1', 'RC2'],
               intervals=[dict(interval_id='G01', trip_id='T03', relay_trip_id='R01'),
                          dict(interval_id='G02', trip_id='T20', relay_trip_id='R02')])
    relays = q4.build_relays(sol, d)
    errs = []
    try:
        got = q4.join_relays(sol, relays)
    except Exception as e:                                        # noqa: BLE001
        return result('T04 按编号绑定中继', False,
                      ['合法方案被拒：%r' % (e,)], tolerance='不抛异常')
    if len(got) != n:
        errs.append('连接表丢了架次：%d 个（应为 %d）——无中继需求的架次不应从表中消失'
                    % (len(got), n))
    want = {'T03': ['R01'], 'T20': ['R02']}
    for tid in ['T%02d' % (i + 1) for i in range(n)]:
        ids = sorted(x['id'] for x in got.get(tid, []))
        if ids != want.get(tid, []):
            errs.append('%s 的中继绑定为 %s，应为 %s' % (tid, ids, want.get(tid, [])))
    # 反向：未知编号必须显式报错，不得静默按位置凑
    bad_sol = dict(sol, intervals=[dict(interval_id='G01', trip_id='T99',
                                        relay_trip_id='R01')])
    try:
        q4.join_relays(bad_sol, relays)
        errs.append('未知运输架次编号 T99 未被检出')
    except ValueError:
        pass
    bad_sol = dict(sol, intervals=[dict(interval_id='G01', trip_id='T01',
                                        relay_trip_id='R99')])
    try:
        q4.join_relays(bad_sol, relays)
        errs.append('未知中继架次编号 R99 未被检出')
    except ValueError:
        pass
    return result('T04 按编号绑定中继', not errs, errs,
                  metrics={'架次数': n, '绑定 T03/T20': 'R01/R02'},
                  tolerance='绑定集合逐位相等')


# --- T05 -------------------------------------------------------------------
@case
def t_row_order_invariance():
    """
    T05：把运输架次表 / 中继表 / 失效区间表的**行序随机打乱**，指标与关联不变。

    外键连接一旦依赖行序（或依赖「排在第几个」），打乱输入就会得到另一份
    成本输入，进而改变分区最优值——而且不会有任何报错。
    """
    p = os.path.join(OUT, 'q3_solution.json')
    if _need('q3_solution.json'):
        return result('T05 行序无关', True, skipped=True, errors=['缺 q3_solution.json'])
    q4 = _mod('q4')
    d, _q1 = _data()
    sol = S.load_solution(p, stage='q3')
    rng = np.random.default_rng(20260924)

    def _snap(s):
        relays = q4.build_relays(s, d)
        rmap = q4.join_relays(s, relays)
        asg = q4.build_assignment(s, d)
        trips = {a['idx']: a for a in asg}
        tids_all = [a['idx'] for a in asg]
        res = q4.group_resources(tids_all, trips,
                                 [x for v in rmap.values() for x in v])
        return (tuple(sorted((k, tuple(sorted(x['id'] for x in v)))
                             for k, v in rmap.items())),
                tuple(sorted((k, v) for k, v in res.items())),
                q4.workload(tids_all, trips, [x for v in rmap.values() for x in v]))

    base = _snap(sol)
    sh = copy.deepcopy(sol)
    for key in ('transport_trips', 'relay_trips', 'intervals'):
        lst = sh[key]
        sh[key] = [lst[i] for i in rng.permutation(len(lst))]
    got = _snap(sh)
    errs = []
    for i, (a, b) in enumerate(zip(base, got)):
        if a != b:
            errs.append('打乱行序后第 %d 项结果改变（%s vs %s）'
                        % (i + 1, str(a)[:120], str(b)[:120]))
    return result('T05 行序无关', not errs, errs,
                  metrics={'架次': len(sol['transport_trips']),
                           '中继架次': len(sol['relay_trips']),
                           '失效区间': len(sol['intervals'])},
                  tolerance='逐位相等')


# --- T06 -------------------------------------------------------------------
_MED_BOX = 'S001-MED-01'


@case
def t_med_limit_not_overridden():
    """
    T06：同一箱既医疗又首批，医疗期限 100、首批 150、交付 120 → **医疗必须违约**。

    旧版用一个 hard 变量顺序覆盖，首批的 150 会把医疗的 100 盖掉，于是 120 被
    判为合规。这里在真实数据上把该箱的两条期限改成 100/150（数据里两者同为
    3600，改前无法区分），并把交付时刻改成 120。
    """
    if _need('q2_box_delivery.csv'):
        return result('T06 医疗期限不被首批覆盖', True, skipped=True,
                      errors=['缺 q2_box_delivery.csv'])
    d, _q1 = _data()
    box = next((b for b in d.cargo if b['id'] == _MED_BOX), None)
    if box is None or box['category'] != '医疗物资' or not box['first_batch']:
        return result('T06 医疗期限不被首批覆盖', True, skipped=True,
                      errors=['数据中找不到既医疗又首批的箱 %s' % _MED_BOX])
    old = (box['expect'], box['deadline'])

    def mutate(tmp):
        df = _rd(tmp, 'q2_box_delivery.csv')
        df.loc[df['货箱编号'] == _MED_BOX, '交付完成时刻s'] = 120.0
        _wr(df, tmp, 'q2_box_delivery.csv')

    errs = []
    try:
        box['expect'], box['deadline'] = 100.0, 150.0
        with _tamper(mutate) as v:
            r = v.check_hard_constraints(d)
            if r['passed']:
                errs.append('医疗期限 100 / 首批 150 / 交付 120 被判为合规')
            if not _hit(r, '医疗期望', '超限'):
                errs.append('未报出医疗期望超限（报告：%s）' % '；'.join(r['errors'][:3]))
            if _hit(r, _MED_BOX, '首批截止', '超限'):
                errs.append('把首批截止 150 也算成违约——两条期限必须分别判')
    finally:
        box['expect'], box['deadline'] = old
    return result('T06 医疗期限不被首批覆盖', not errs, errs,
                  metrics={'构造': '医疗 100 / 首批 150 / 交付 120', '箱号': _MED_BOX},
                  tolerance='必须报医疗违约，且不报首批违约')


# --- T07 -------------------------------------------------------------------
@case
def t_battery_charge_release():
    """
    T07：电池链的释放时刻必须含**充电时长**：返航 100、充满需 Δ、下次开始 Δ−40
    → 冲突恰为 40 秒。

    指南的样例写作「返航 100、充好 180、下次开始 140」，即充电 80 s；本算例
    电池满充时间为 1800 s 量级，80 s 的充电在该物理参数下不可能出现，故保留
    「冲突 40 秒」这个断言，把 Δ 取成真实的 `core.charge_time`。若释放时刻被
    当成「返航即释放」（旧口径），算出的所需推迟量会是负的 → 报「无冲突」。
    """
    q3 = _mod('q3')
    from core import charge_time as _ct
    d, _q1 = _data()
    g, soc = 'A', 0.5
    E = (1.0 - soc) * d.transport_types[g]['E_use']
    chg = _ct(soc, d.batteries[g][1])
    ret, dur = 100.0, 100.0
    nxt = ret + chg - 40.0
    asg = [dict(type=g, uav='U01', battery='A' + '_B1', start=0.0, duration=dur, E=E),
           dict(type=g, uav='U02', battery='A' + '_B1', start=nxt, duration=dur, E=E)]
    conf = q3.chain_conflicts(d, asg)
    errs = []
    if len(conf) != 1:
        errs.append('冲突条数 %d（应为 1）：%s' % (len(conf), conf))
    else:
        c = conf[0]
        if c['resource'] != 'bat:A_B1':
            errs.append('冲突落在 %s，应在共享电池链上' % c['resource'])
        if abs(c['required_delay'] - 40.0) > 1e-9:
            errs.append('所需推迟量 %.9f s，应为 40 s' % c['required_delay'])
        if abs(c['resource_release'] - (ret + chg)) > 1e-9:
            errs.append('释放时刻 %.6f，应为返航 %.0f + 充电 %.6f = %.6f'
                        % (c['resource_release'], ret, chg, ret + chg))
        if ret - nxt >= 0:
            errs.append('夹具失效：按「返航即释放」口径本应无冲突')
    return result('T07 电池充电占用（冲突 40 s）', not errs, errs,
                  metrics={'充电时长 s': round(chg, 6), '释放时刻 s': round(ret + chg, 6),
                           '下次开始 s': round(nxt, 6),
                           '冲突 s': 40.0}, tolerance='1e-9 绝对容差')


# --- T08 -------------------------------------------------------------------
@case
def t_relay_turnover_release():
    """
    T08：中继机占用到「返航 + 周转」：周转 30 s、前架返航 100、后架开始 110
    → 冲突 20 秒，该组需要 2 架中继。

    周转时间在数据里是 300 s；判据与数值无关，故把 `relay_type['turnover']`
    改成 30 s 复现指南的样例——占用公式走的是 q4 的真实代码路径（读 d 里的
    周转值），不是测试里另写一份。
    """
    q4 = _mod('q4')
    d, _q1 = _data()
    d30 = copy.copy(d)
    d30.relay_type = dict(d.relay_type, turnover=30.0)
    mk = lambda i, s, r: dict(relay_trip_id='R%02d' % i, station_id='S01', start=s,
                              return_time=r, energy_kwh=0.5)        # noqa: E731
    sol = dict(transport_trips=[], relay_trips=[mk(1, 0.0, 100.0), mk(2, 110.0, 210.0)],
               intervals=[], relay_uav_ids=['RU1'], station_ids=['S01'],
               component_ids=['RC1'])
    relays = q4.build_relays(sol, d30)
    errs = []
    iv = [(x['start'], x['return_t'] + x['turnover']) for x in relays]
    if abs(relays[0]['turnover'] - 30.0) > 1e-12:
        errs.append('周转时间未取自数据：%r' % relays[0]['turnover'])
    overlap = iv[0][1] - iv[1][0]
    if abs(overlap - 20.0) > 1e-9:
        errs.append('两架中继的占用重叠 %.9f s，应为 20 s（区间 %s）' % (overlap, iv))
    if q4.peak_concurrency(iv) != 2:
        errs.append('计周转后峰值 %d，应为 2' % q4.peak_concurrency(iv))
    if q4.peak_concurrency([(x['start'], x['return_t']) for x in relays]) != 1:
        errs.append('夹具失效：不计周转时本应只有 1 架在役')
    trips = {0: dict(type='A', start=0.0, end=100.0, bat_end=100.0),
             1: dict(type='A', start=110.0, end=210.0, bat_end=210.0)}
    res = q4.group_resources([0, 1], trips, relays)
    if res.get('relay') != 2:
        errs.append('group_resources 的中继需求 %r，应为 2（F09：占用到返航+周转）'
                    % res.get('relay'))
    return result('T08 中继周转占用（冲突 20 s）', not errs, errs,
                  metrics={'周转 s': 30.0, '占用区间': str(iv), '冲突 s': 20.0},
                  tolerance='1e-9 绝对容差')


# --- T09 -------------------------------------------------------------------
@case
def t_touching_intervals():
    """T09：占用 [0,10) 与 [10,20) 相接而不重叠 → 峰值 1，且不算冲突。"""
    q4 = _mod('q4')
    errs = []
    if q4.peak_concurrency([(0.0, 10.0), (10.0, 20.0)]) != 1:
        errs.append('相接区间的峰值不是 1')
    if q4.peak_concurrency([(0.0, 10.0), (10.0, 20.0), (5.0, 12.0)]) != 2:
        errs.append('存在真重叠时峰值不是 2（判据把重叠漏掉了）')
    return result('T09 区间相接不算冲突', not errs, errs,
                  metrics={'[0,10)∪[10,20) 峰值': 1}, tolerance='严格整数相等')


# --- T10 -------------------------------------------------------------------
@case
def t_delivery_integrity():
    """T10：重复箱、漏箱、未知箱、未知架次、未知资源编号——每种都必须判失败。"""
    if _need('q2_box_delivery.csv', 'q2_transport_trips.csv'):
        return result('T10 交付完整性', True, skipped=True,
                      errors=['缺 q2 结果表'])
    d, _q1 = _data()

    def mk(fn, table):
        def mutate(tmp):
            df = _rd(tmp, table)
            _wr(fn(df), tmp, table)
        return mutate

    def dup(df):
        return pd.concat([df, df.iloc[[0]]], ignore_index=True)

    def miss(df):
        return df.iloc[1:]

    def unknown_box(df):
        df.loc[df.index[0], '货箱编号'] = 'S001-XXX-99'
        return df

    def unknown_trip(df):
        df.loc[df.index[0], '架次编号'] = 'T99'
        return df

    def unknown_uav(df):
        df.loc[df.index[0], '无人机编号'] = 'U99'
        return df

    cases = [('重复箱号', mk(dup, 'q2_box_delivery.csv'), '重复'),
             ('漏箱', mk(miss, 'q2_box_delivery.csv'), '漏交付'),
             ('未知箱号', mk(unknown_box, 'q2_box_delivery.csv'), '未知箱号'),
             ('未知架次', mk(unknown_trip, 'q2_box_delivery.csv'), '不存在于架次表'),
             ('未知资源编号', mk(unknown_uav, 'q2_transport_trips.csv'), '不在库存')]
    errs, seen = [], []
    for name, mutate, key in cases:
        with _tamper(mutate) as v:
            r = v.check_hard_constraints(d)
        seen.append('%s:%s' % (name, '检出' if (not r['passed'] and _hit(r, key)) else '漏检'))
        if r['passed']:
            errs.append('%s 未被判失败' % name)
        elif not _hit(r, key):
            errs.append('%s 失败理由不含「%s」（%s）' % (name, key, '；'.join(r['errors'][:2])))
    return result('T10 交付完整性', not errs, errs, metrics={k: v for k, v in
                                                            (s.split(':') for s in seen)},
                  tolerance='五类错误均须判失败')


# --- T11 -------------------------------------------------------------------
@case
def t_resource_inventory():
    """T11：机型对但编号不在库存的实体（U99）必须失败——不能只数资源个数。"""
    if _need('q2_transport_trips.csv'):
        return result('T11 库存外的实体编号', True, skipped=True,
                      errors=['缺 q2_transport_trips.csv'])
    d, _q1 = _data()
    uav_ids = {u['id'] for u in d.uavs}
    fake = 'U99' if 'U99' not in uav_ids else 'U98'

    def mutate(tmp):
        df = _rd(tmp, 'q2_transport_trips.csv')
        df.loc[df.index[0], '无人机编号'] = fake
        _wr(df, tmp, 'q2_transport_trips.csv')

    with _tamper(mutate) as v:
        r = v.check_hard_constraints(d)
    errs = []
    if r['passed']:
        errs.append('库存外的无人机编号 %s 未被判失败（只数了个数？）' % fake)
    elif not _hit(r, '不在库存'):
        errs.append('失败理由不含「不在库存」：%s' % '；'.join(r['errors'][:2]))
    return result('T11 库存外的实体编号', not errs, errs,
                  metrics={'注入编号': fake, '在册编号数': len(uav_ids)},
                  tolerance='必须判失败')


# --- T12 -------------------------------------------------------------------
@case
def t_zero_relay():
    """T12：全部运输轨迹可直连（零中继）时，方案仍可导出并通过校验。"""
    p = os.path.join(OUT, 'q3_solution.json')
    if _need('q3_solution.json'):
        return result('T12 零中继方案', True, skipped=True, errors=['缺 q3_solution.json'])
    q4 = _mod('q4')
    d, _q1 = _data()
    sol = copy.deepcopy(S.load_solution(p, stage='q3'))
    sol['relay_trips'] = []
    for iv in sol['intervals']:
        iv['relay_trip_id'] = ''
    errs = []
    try:
        S.validate_solution(sol)
    except ValueError as e:
        errs.append('零中继方案自校验失败：%s' % e)
    try:
        S.check_foreign_keys(sol, sol['intervals'], require_relay_binding=False)
    except ValueError as e:
        errs.append('零中继方案外键校验失败：%s' % e)
    try:
        S.check_foreign_keys(sol, sol['intervals'], require_relay_binding=True)
        errs.append('要求中继绑定时空引用却通过了校验')
    except ValueError:
        pass
    relays = q4.build_relays(sol, d)
    if relays:
        errs.append('零中继方案构造出 %d 条中继记录' % len(relays))
    got = q4.join_relays(sol, relays)
    if len(got) != len(sol['transport_trips']) or any(got.values()):
        errs.append('零中继连接表不正确：%d 个架次，非空 %d 个'
                    % (len(got), sum(1 for v in got.values() if v)))
    return result('T12 零中继方案', not errs, errs,
                  metrics={'运输架次': len(sol['transport_trips']), '中继架次': 0},
                  tolerance='可导出且校验通过')


# --- T13 -------------------------------------------------------------------
@case
def t_phase_hole():
    """
    T13：两条通信阶段之间人为留 0.2 s 空洞 → 表格校验失败，**与采样 dt 无关**。

    判据只看表里写出的边界本身；哪怕把「允许的间隙」放宽到 1 s，0.2 s 的空洞
    也必须被抓住——旧版用「间隙 ≤ 采样步长」判，步长一改结论就跟着变。
    """
    v = _mod('verify')
    q3 = _mod('q3')
    span = {'T01': (0.0, 100.0)}
    ok = pd.DataFrame([dict(运输架次编号='T01', 开始时刻s=0.0, 结束时刻s=40.0, 通信阶段='直连'),
                       dict(运输架次编号='T01', 开始时刻s=40.0, 结束时刻s=100.0, 通信阶段='中继')])
    hole = pd.DataFrame([dict(运输架次编号='T01', 开始时刻s=0.0, 结束时刻s=40.0, 通信阶段='直连'),
                         dict(运输架次编号='T01', 开始时刻s=40.2, 结束时刻s=100.0, 通信阶段='中继')])
    overlap = pd.DataFrame([dict(运输架次编号='T01', 开始时刻s=0.0, 结束时刻s=40.2, 通信阶段='直连'),
                            dict(运输架次编号='T01', 开始时刻s=40.0, 结束时刻s=100.0, 通信阶段='中继')])
    short = pd.DataFrame([dict(运输架次编号='T01', 开始时刻s=0.0, 结束时刻s=40.0, 通信阶段='直连'),
                          dict(运输架次编号='T01', 开始时刻s=40.0, 结束时刻s=99.0, 通信阶段='中继')])
    errs = []
    if v.phase_boundary_errors(ok, span)['errors']:
        errs.append('相接的分段表被误判：%s' % v.phase_boundary_errors(ok, span)['errors'])
    for name, df, key in (('0.2 s 空洞', hole, 'n_break'), ('0.2 s 重叠', overlap, 'n_break'),
                          ('末段短了 1 s', short, 'n_span')):
        g = v.phase_boundary_errors(df, span)
        if g[key] != 1 or not g['errors']:
            errs.append('%s 未被检出（%s=%d）' % (name, key, g[key]))
    # 相邻边界判据与采样步长解耦：把容差从 0.01 s 放宽到 1 s，0.2 s 的错位照样成立
    for name, df in (('0.2 s 空洞', hole), ('0.2 s 重叠', overlap)):
        if v.phase_boundary_errors(df, span, tol=1.0)['n_break'] != 1:
            errs.append('%s 在容差放大到 1 s 后漏检——判据仍依赖采样步长' % name)
    segs = [dict(trip=0, t1=0.0, t2=40.0, phase='直连', total=100.0),
            dict(trip=0, t1=40.2, t2=100.0, phase='中继', total=100.0)]
    if not q3.check_phase_table(segs):
        errs.append('求解器侧 check_phase_table 未检出同一处空洞')
    return result('T13 分段表空洞（与 dt 无关）', not errs, errs,
                  metrics={'空洞 s': 0.2, '容差放宽到 1 s 仍须检出': '是'},
                  tolerance='逐位相接')


# --- T14 -------------------------------------------------------------------
@case
def t_binding_requires_available():
    """T14：区间绑定的中继必须**此刻在役**——绑一个时段对不上的中继必须失败。"""
    v = _mod('verify')
    win = {'R01': (100.0, 200.0), 'R02': (300.0, 400.0)}
    good = pd.DataFrame([dict(失效区间编号='G01', 区间起点s=120.0, 区间终点s=180.0,
                              中继架次编号='R01')])
    wrong = pd.DataFrame([dict(失效区间编号='G01', 区间起点s=120.0, 区间终点s=180.0,
                               中继架次编号='R02')])
    empty = pd.DataFrame([dict(失效区间编号='G01', 区间起点s=120.0, 区间终点s=180.0,
                               中继架次编号='')])
    unknown = pd.DataFrame([dict(失效区间编号='G01', 区间起点s=120.0, 区间终点s=180.0,
                                 中继架次编号='R09')])
    errs = []
    if v.interval_binding_errors(good, win):
        errs.append('时段落在服务窗口内的绑定被误判：%s'
                    % v.interval_binding_errors(good, win))
    for name, df in (('绑定了窗口外的中继', wrong), ('空绑定', empty),
                     ('绑定了不存在的中继', unknown)):
        if not v.interval_binding_errors(df, win):
            errs.append('%s 未被检出（该时段另有一架 R01 在役，正是会误绑的场景）' % name)
    return result('T14 绑定须落在中继服役窗口内', not errs, errs,
                  metrics={'R01 窗口': str(win['R01']), '区间': '[120, 180]'},
                  tolerance='错误清单非空')


# --- T15 -------------------------------------------------------------------
@case
def t_backhaul_required():
    """
    T15：接入可用而**站到网关的回传不可用** → 该站不得判为覆盖。

    把网关挪到 200 km 外重算：接入裕量（只与站点和机位有关）逐位不变，回传
    裕量必然变差。若覆盖判定漏掉回传这一条，站点会照旧判为可用——这正是
    「中继悬停着却传不回去」的错误。
    """
    q3 = _mod('q3')
    if _need('q3_stations.csv'):
        return result('T15 回传链路必须参与判定', True, skipped=True,
                      errors=['缺 q3_stations.csv'])
    d, _q1 = _data()
    st_df = pd.read_csv(os.path.join(OUT, 'q3_stations.csv'), encoding='utf-8-sig')
    r0 = st_df.iloc[0]
    st = (float(r0['悬停经度']), float(r0['悬停纬度']), float(r0['悬停海拔m']))
    p = (st[0], st[1], st[2] + 100.0)         # 机位取站点正上方 100 m：接入裕量必然充裕
    d_far = copy.copy(d)
    d_far.O01 = dict(d.O01, lon=d.O01['lon'] + 2.0)
    errs = []
    a0, a1 = q3.access_margin(d, st, *p), q3.access_margin(d_far, st, *p)
    b0, b1 = q3.backhaul_margin(d, st), q3.backhaul_margin(d_far, st)
    if not (a0 == a1):
        errs.append('挪动网关改变了接入裕量（%.6f → %.6f），夹具不成立' % (a0, a1))
    if not (b0 >= 0 > b1):
        errs.append('夹具不成立：原回传裕量 %.3f dB，挪远后 %.3f dB（裕量 >= 0 表示可用）'
                    % (b0, b1))
    if not q3.station_covers(d, st, *p):
        errs.append('原场景下站点本应可覆盖，却判为不可用（回传裕量 %.3f dB）' % b0)
    if q3.station_covers(d_far, st, *p):
        errs.append('回传不可用时站点仍被判为可覆盖（回传裕量 %.3f dB）' % b1)
    return result('T15 回传链路必须参与判定', not errs, errs,
                  metrics={'接入裕量 dB': '%.3f（挪网关后不变）' % a0,
                           '回传裕量 dB': '%.3f → %.3f' % (b0, b1)},
                  tolerance='覆盖判定必须随回传翻转')


# --- T15b ------------------------------------------------------------------
@case
def t_verifier_checks_backhaul():
    """
    T15b：**验证程序**本身必须查回传，而不只是求解器里的 station_covers 查。

    T15 只证明 q3.station_covers 会因回传不可用而翻转；`verify.check_relay_final`
    是另一条代码路径，它从前只调 link_ok(..., Lmax_a) 判接入，回传那一半从不
    参与，于是把 Lmax_r_gw 改成 -999 dB（回传物理上不可能）重跑，它照样
    报 passed=True。反向测试：真实参数下必须通过，把回传余量打到负值后必须失败。
    """
    v = _mod('verify')
    if _need('q3_transport_trips.csv', 'q3_relay_trips.csv', 'q3_stations.csv',
             'q3_intervals.csv', 'q3_comm_phases.csv', 'q3_coverage.csv'):
        return result('T15b 验证程序必须查回传', True, skipped=True,
                      errors=['缺问题三结果表，请先运行 python code/q3.py'])
    d, _q1 = _data()
    errs = []
    ok0 = v.check_relay_final(d)
    if not ok0['passed']:
        errs.append('真实参数下复核本应通过，却报失败：%s' % ok0['errors'][:2])
    # 浅拷贝 d 后再拷贝 comm，避免把缓存的 d 改坏、污染后续用例
    d_bad = copy.copy(d)
    d_bad.comm = copy.copy(d.comm)
    real = d_bad.comm.Lmax_r_gw
    d_bad.comm.Lmax_r_gw = -999.0
    bad = v.check_relay_final(d_bad)
    if bad['passed']:
        errs.append('把 Lmax_r_gw 改成 -999 dB（回传不可用）后复核仍报通过——'
                    '验证程序漏掉了回传链路')
    elif not any('回传' in e for e in bad['errors']):
        errs.append('失败原因里没有一条提到回传，可能不是回传判据触发：%s' % bad['errors'][:2])
    return result('T15b 验证程序必须查回传', not errs, errs,
                  metrics={'Lmax_r_gw dB': '%.2f（改为 -999 后复核失败 %d 条）'
                           % (real, len(bad['errors']))},
                  tolerance='回传不可用必须使复核失败')


# --- T-Q4m -----------------------------------------------------------------
@case
def t_q4_min_demand_enumerated():
    """
    T-Q4m：q4_comparison.csv 的「最小需求_*」两列必须真的是**全部分区上的下确界**。

    这两列是论文「某类资源在任一方案下都缺 N 架」这类全称断言的唯一数据依据，
    而 verify.py 只做表内自洽核对、不重跑枚举（重跑要用 q4 的分组层）。本用例
    对每个 K 重跑全部 S(n_units, K) 个分区，逐个算八类需求再取逐类最小值，与
    表内列对照（分区数随原子单元数变化，故只打印实测值，不写死）。

    **独立性边界**：本用例调用 q4.group_resources 与 q4.partition_sets，与
    q4.py 共享同一套分组资源核算，故它证明的是「枚举与计数一致」，**不是**
    「分组成本输入已由第二套独立实现验证」。后者由 T-R3 一类的资源区间重算
    用例负责，两者不可互相替代。
    """
    need = ['q4_partition.csv', 'q4_units.csv', 'q4_comparison.csv']
    miss = _need(*need)
    if miss:
        return result('T-Q4m 全部分区最小需求', True, skipped=True,
                      errors=['缺 %s，请先运行 python code/q4.py' % '、'.join(miss)])
    q4 = _mod('q4')
    import q2
    d, _q1 = _data()
    if not hasattr(d, 'geo'):
        d.geo_nodes, d.geo = q2.precompute_geometry(d)
    sol = S.load_solution(q4.Q3_SOL, stage='q3',
                          input_hashes_expect=q4.input_hashes(),
                          solver_hashes_expect=q4.solver_hashes(q4.Q3_SOLVER_FILES))
    asg = q4.build_assignment(sol, d)
    rr = q4.build_relays(sol, d)
    relay_of_trip = q4.join_relays(sol, rr)
    units = q4.atomic_units(d, asg)

    unit_trips, unit_relays = [], []
    for u in units:
        us = set(u)
        tids = [a['idx'] for a in asg if set(a['route']) <= us]
        unit_trips.append(tids)
        seen, rl = set(), []
        for k in tids:
            for x in relay_of_trip[asg[k]['trip_id']]:
                if x['id'] not in seen:
                    seen.add(x['id'])
                    rl.append(x)
        unit_relays.append(rl)

    cache = {}

    def demand(members):
        key = frozenset(members)
        if key not in cache:
            tids = [k for m in members for k in unit_trips[m]]
            seen, rl = set(), []
            for m in members:
                for x in unit_relays[m]:
                    if x['id'] not in seen:
                        seen.add(x['id'])
                        rl.append(x)
            cache[key] = q4.group_resources(tids, asg, rl)
        return cache[key]

    cmp_ = pd.read_csv(os.path.join(OUT, 'q4_comparison.csv'), encoding='utf-8-sig')
    part = pd.read_csv(os.path.join(OUT, 'q4_partition.csv'), encoding='utf-8-sig')
    labels = [c for c in part.columns
              if c not in ('K', '任务组编号', '服务区列表', '工作量h')]
    lab2key = {v: k for k, v in q4.RES_LABEL.items()}          # 中文列名 -> 内部键
    errs, n_part = [], 0
    n_by_k = {}
    min_all = {}
    for K in (2, 3):
        mn = {lab: 10 ** 9 for lab in labels}
        n_k = 0
        for groups in q4.partition_sets(units, K):
            n_part += 1
            n_k += 1
            tot = {lab: 0 for lab in labels}
            for members in groups:
                r = demand(members)
                for lab in labels:
                    tot[lab] += r[lab2key[lab]]
            for lab in labels:
                mn[lab] = min(mn[lab], tot[lab])
        min_all[K] = mn
        n_by_k[K] = n_k
        row = cmp_[cmp_['K'] == K]
        if not len(row):
            errs.append('q4_comparison.csv 缺 K=%d 行' % K)
            continue
        row = row.iloc[0]
        for lab in labels:
            col = '最小需求_' + lab
            if col not in cmp_.columns:
                errs.append('q4_comparison.csv 缺列 %s' % col)
            elif int(row[col]) != mn[lab]:
                errs.append('K=%d %s 的最小需求：独立枚举 %d vs 表内 %s'
                            % (K, lab, mn[lab], row[col]))
    return result('T-Q4m 全部分区最小需求', not errs, errs,
                  metrics={'枚举分区数': '%d（%s）' % (
                               n_part, ' + '.join('K=%d %d' % (K, n_by_k[K])
                                                  for K in sorted(n_by_k))),
                           'B 型机最小需求': 'K=2 %d | K=3 %d'
                           % (min_all[2]['B型运输无人机'], min_all[3]['B型运输无人机']),
                           '中继机最小需求': 'K=2 %d | K=3 %d'
                           % (min_all[2]['中继无人机'], min_all[3]['中继无人机'])},
                  tolerance='逐类逐 K 与表内列逐位相等')


# --- T16 -------------------------------------------------------------------
@case
def t_archive_k_cap():
    """T16：扫描上限 K 时，实际 K+1 架次的候选不得进入该档的可行档案。"""
    miss = _need('q2_scan.csv', 'q2_alns_trace.csv', 'q2_solution.json')
    if miss:
        return result('T16 可行档案不超 K', True, skipped=True,
                      errors=['缺 %s' % '、'.join(miss)])
    scan = pd.read_csv(os.path.join(OUT, 'q2_scan.csv'), encoding='utf-8-sig')
    tr = pd.read_csv(os.path.join(OUT, 'q2_alns_trace.csv'), encoding='utf-8-sig')
    sol = S.load_solution(os.path.join(OUT, 'q2_solution.json'), stage='q2')
    K = int(sol['budget']['K'])
    errs, n_arch = [], 0
    for _, r in scan.iterrows():
        k_cap = int(r['K上限'])
        if pd.isna(r['档案架次']):
            continue
        n_arch += 1
        if int(r['档案架次']) > k_cap:
            errs.append('K<=%d 的档案里出现 %d 架次（超过上限）'
                        % (k_cap, int(r['档案架次'])))
        if r['档案可行'] != '是':
            errs.append('K<=%d 有档案条数 %s 却标为不可行' % (k_cap, r['档案条数']))
    if '档案架次' in tr.columns:
        for _, r in tr.iterrows():
            if not pd.isna(r['档案架次']) and int(r['档案架次']) > K:
                errs.append('推荐档 K<=%d 的迭代轨迹里出现档案 %d 架次'
                            % (K, int(r['档案架次'])))
    n_rec = len(sol['transport_trips'])
    if n_rec > K:
        errs.append('推荐解 %d 架次超过其 K<=%d 的约束' % (n_rec, K))
    return result('T16 可行档案不超 K', not errs, errs,
                  metrics={'推荐 K': K, '推荐架次': n_rec, '有档案的档位': n_arch},
                  tolerance='档案架次数 <= K 上限')


# --- T17 -------------------------------------------------------------------
@case
def t_timeout_not_proven():
    """T17：求解器超时但有可行解 → 如实报「可行、未证最优」，不得报已证最优。"""
    if _need('q2_exact.csv'):
        return result('T17 超时≠已证最优', True, skipped=True, errors=['缺 q2_exact.csv'])
    df = pd.read_csv(os.path.join(OUT, 'q2_exact.csv'), encoding='utf-8-sig')
    if '已证最优' not in df.columns or '超时' not in df.columns:
        return result('T17 超时≠已证最优', False,
                      ['q2_exact.csv 缺少「已证最优/超时」状态列，超时与不可行无法区分'])
    errs, n_to, n_ok, n_na = [], 0, 0, 0
    for _, r in df.iterrows():
        lab = '%s/%s' % (r['层级'], r['算例'])
        if pd.isna(r['超时']):
            # Layer A（分区层）按设计不记求解器状态（见 q2_exact.py：没捕获就不编），
            # 故 T17 的「超时 ≠ 已证最优」对它不适用。如实计入「未报状态」一栏，
            # 不当成通过，也不静默丢弃：仍要求它自报的最优性标记与上下界间隙自洽。
            n_na += 1
            if pd.isna(r['上下界间隙']):
                errs.append('%s 既未报求解器状态，也未报上下界间隙，最优性无从判断' % lab)
            elif int(bool(r['已证最优'])) != int(float(r['上下界间隙']) <= 1e-9):
                errs.append('%s 已证最优标记与上下界间隙 %.3g 不符'
                            % (lab, float(r['上下界间隙'])))
            continue
        proved, to = int(r['已证最优']), int(r['超时'])
        if to:
            n_to += 1
            if proved:
                errs.append('%s 超时却标为已证最优' % lab)
            if pd.isna(r['目标值']):
                errs.append('%s 超时且未报出可行解——超时≠不可行' % lab)
            if pd.isna(r['上下界间隙']) or float(r['上下界间隙']) <= 1e-9:
                errs.append('%s 超时却报出 0 间隙（未证最优不能报零间隙）' % lab)
        else:
            n_ok += 1
            if not proved:
                errs.append('%s 未超时也未证最优，状态不明' % lab)
            elif not pd.isna(r['上下界间隙']) and float(r['上下界间隙']) > 1e-6:
                errs.append('%s 标为已证最优但上下界间隙 %.6f' % (lab, float(r['上下界间隙'])))
    return result('T17 超时≠已证最优', not errs, errs,
                  metrics={'超时行': n_to, '已证最优行': n_ok,
                           '未报求解器状态行': n_na},
                  tolerance='状态列自洽；未报状态的行按其自报的上下界间隙复核')


# --- T18 -------------------------------------------------------------------
@case
def t_stale_source_stop():
    """T18：下游引用的来源方案 id 不符时必须停止，而不是继续算另一份方案。"""
    q2p = os.path.join(OUT, 'q2_solution.json')
    q3p = os.path.join(OUT, 'q3_solution.json')
    if _need('q2_solution.json', 'q3_solution.json'):
        return result('T18 来源不符即停止', True, skipped=True, errors=['缺方案文件'])
    errs = []
    q3 = S.load_solution(q3p, stage='q3')
    try:
        S.load_solution(q3p, stage='q3', source_solution_id='q2-000000000000')
        errs.append('来源方案 id 不符却通过了校验')
    except ValueError:
        pass
    try:
        S.load_solution(q3p, stage='q3', source_solution_id=q3['source_solution_id'])
    except ValueError as e:
        errs.append('来源 id 正确时反而被拒：%s' % e)
    try:
        S.load_solution(q2p, stage='q2', source_solution_id='q2-000000000000')
        errs.append('问题二方案被当作 q3 引用的下游却未拒')
    except ValueError:
        pass
    return result('T18 来源不符即停止', not errs, errs,
                  metrics={'q3 来源': q3['source_solution_id']},
                  tolerance='抛 ValueError')


# --- T19 -------------------------------------------------------------------
@case
def t_energy_tamper_recompute():
    """T19：故意改一条架次能耗 → 指标重算必须与之不符（表内数字不是自证的）。"""
    if _need('q3_metrics.csv', 'q3_transport_trips.csv'):
        return result('T19 改动后重算不一致', True, skipped=True,
                      errors=['缺 q3_metrics.csv / q3_transport_trips.csv'])
    d, _q1 = _data()

    def mutate(tmp):
        df = _rd(tmp, 'q3_transport_trips.csv')
        df.loc[df.index[0], '架次能耗kWh'] = float(df.loc[df.index[0], '架次能耗kWh']) + 0.5
        _wr(df, tmp, 'q3_transport_trips.csv')

    errs = []
    with _tamper(lambda tmp: None) as v:
        base = v.check_q3_metrics(d)
    if not base['passed']:
        return result('T19 改动后重算不一致', False,
                      ['未改动的表本身就通不过重算，无法判定本用例：%s'
                       % '；'.join(base['errors'][:3])])
    with _tamper(mutate) as v:
        r = v.check_q3_metrics(d)
    if r['passed']:
        errs.append('某架次能耗被改 +0.5 kWh 后，指标重算仍判为一致')
    elif not _hit(r, '运输能耗'):
        errs.append('失败理由不含运输能耗：%s' % '；'.join(r['errors'][:2]))
    return result('T19 改动后重算不一致', not errs, errs,
                  metrics={'注入改动': '一条架次能耗 +0.5 kWh', '未改动时': '重算一致'},
                  tolerance='必须报不一致')


# --- T20 -------------------------------------------------------------------
@case
def t_exit_code_nonzero():
    """T20：任一子项失败、或一项都没跑，verify 的退出码必须非零。"""
    d, _q1 = _data()
    v = _mod('verify')
    errs = []
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        bad = [('构造的失败检查', lambda dd: v.rep('构造的失败检查', False, ['故意失败']))]
        rc_fail = v.main(d, checks=bad)
        rc_empty = v.main(d, checks=[])
        rc_ok = v.main(d, checks=[('构造的通过检查',
                                   lambda dd: v.rep('构造的通过检查', True))])
    if rc_fail == 0:
        errs.append('子项失败时退出码为 0')
    if rc_empty == 0:
        errs.append('一个检查都没跑时退出码为 0（空报告不得当作通过）')
    if rc_ok != 0:
        errs.append('子项全部通过时退出码非零：%r' % rc_ok)
    return result('T20 子项失败则退出码非零', not errs, errs,
                  metrics={'失败子项退出码': rc_fail, '空检查退出码': rc_empty,
                           '全通过退出码': rc_ok}, tolerance='失败→非零，通过→0')


# --- T21 -------------------------------------------------------------------
@case
def t_packaging_preflight():
    """T21：打包清单缺项时必须预检失败、不产出提交包；且清单不得与哈希口径脱节。

    交付一致性（审阅报告 P0-4）三件事在这里立成回归：①数据清单必须**等于**
    solution_io.input_hashes() 的键集——上一版手写 6 项、哈希 17 项，评委解压后
    每条方案都被判"输入数据不一致"；②第三方商业字体 LiSu.ttf 不得随包转发，
    同时又不能成为编译必需（gmcmthesis.cls 必须有字体回退）；③纯仓库检出
    （无赛题数据）时，预检要明确报"数据一个都没有"，而不是当作"没有数据要打"。

    预检里允许出现的缺失只有两类**环境性前提**：赛题数据（纯仓库检出时本来就没有）
    与论文 PDF main.pdf（由 latexmk 编出来；交付包只带编好的成品 论文/论文-队号.pdf，
    不在包根放 main.pdf）。这两项之外任何一项缺失都是真缺项，必须报错——本用例
    在工程目录与在解压后的交付包里都要能跑过，故按实际在位情况分别断言。
    """
    root = os.path.dirname(HERE)
    errs = []
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        bs = _mod('build_submission')
    except Exception as e:                                        # noqa: BLE001
        return result('T21 打包预检', False, ['无法导入 build_submission：%r' % (e,)])
    has_data = bool(bs.sio.input_hashes())
    has_pdf = all(os.path.exists(os.path.join(root, p)) for p in bs.PAPER_PDF)
    got0 = bs.preflight()
    if has_data and has_pdf:
        if got0:
            errs.append('当前工程预检就不通过：%s' % '；'.join(got0))
    else:
        allowed = ([] if has_data else ['原始数据']) + ([] if has_pdf else ['论文正文'])
        if len(got0) != len(allowed) or \
                not all(any(a in b for a in allowed) for b in got0):
            errs.append('缺数据/缺论文 PDF 时预检未按环境性缺失报出：%s（应为 %s）'
                        % (got0, allowed))
    if sorted(bs.DATA) != sorted(bs.sio.input_hashes()):
        errs.append('打包数据清单与 input_hashes() 不一致：清单 %d 项、哈希 %d 项'
                    % (len(bs.DATA), len(bs.sio.input_hashes())))
    for f in bs.LOCAL_ONLY:
        if f in bs.SUPPORT:
            errs.append('%s 是本地资源（授权不许再分发），不得列入 SUPPORT' % f)
    cls = open(os.path.join(root, 'gmcmthesis.cls'), encoding='utf-8').read()
    if 'IfFileExists{LiSu.ttf}' not in cls:
        errs.append('gmcmthesis.cls 缺字体回退：没有 LiSu.ttf 的机器编不过论文，'
                    '而该字体又不随包分发')
    keep_sup, keep_data = list(bs.SUPPORT), list(bs.DATA)
    try:
        bs.SUPPORT = keep_sup + ['code/__不存在的文件__.csv']
        bs.DATA = keep_data + ['数据/__不存在的数据__']
        # 环境性缺失（无数据 / 无论文 PDF）本来就在 got0 里，故期望值随之抬高
        want = 2 + len(got0)
        bad = bs.preflight()
        if len(bad) != want:
            errs.append('注入 2 项缺失却报出 %d 项（期望 %d）：%s' % (len(bad), want, bad))
        if not any('__不存在的文件__' in b for b in bad) or \
                not any('__不存在的数据__' in b for b in bad):
            errs.append('缺失项未被逐条列出：%s' % bad)
    finally:
        bs.SUPPORT, bs.DATA = keep_sup, keep_data
    if bs.preflight() != got0:
        errs.append('恢复清单后预检结果与注入前不一致')
    return result('T21 打包预检', not errs, errs,
                  metrics={'清单项': len(bs.PAPER_PDF) + len(keep_sup) + len(keep_data),
                           '数据清单': '%d 项（与 input_hashes() 同源）' % len(keep_data)},
                  tolerance='缺项必须逐条列出并使预检失败')


# --- T22 -------------------------------------------------------------------
@case
def t_excel_column_mapping():
    """T22：结果表回填必须按**列名**对齐——把源 CSV 的列序整体倒过来，填出的
    表必须逐格不变；多出一列未声明的列时必须中止，而不是静默错位。

    位置拷贝的失败方式是「填得挺像回事」：列数一变就整体串位，而串位后的表
    仍然合法，没有任何报错。故这里不比「填没填上」，比的是**两种列序填出的
    表逐格相同**，以及看守护栏真的会拦。
    """
    root = os.path.dirname(HERE)
    # fill_results 在模块级把 sys.stdout 换成 utf-8 包装器（脚本自己跑时是必要的，
    # 被当模块导入时会把本测试的输出流一并换掉），导入后立刻还原。
    keep_out = sys.stdout
    try:
        fr = _mod('fill_results')
    except Exception as e:                                            # noqa: BLE001
        return result('T22 结果表列名对齐', False, ['无法导入 fill_results：%r' % (e,)])
    finally:
        try:
            sys.stdout.flush()
        except Exception:                                             # noqa: BLE001
            pass
        # 必须留一个包装器的引用：TextIOWrapper 被回收时会 close() 掉底层 buffer，
        # 之后整个进程的 print 都会报 "I/O operation on closed file"。
        _STDOUT_KEEP.append(sys.stdout)
        sys.stdout = keep_out
    tmpl = os.path.join(root, '结果提交模板.xlsx')
    if not os.path.exists(tmpl):
        return result('T22 结果表列名对齐', True, skipped=True,
                      errors=['缺少《结果提交模板.xlsx》，跳过'])
    miss = _need(*[c for _, c, _ in fr.SHEETS])
    if miss:
        return result('T22 结果表列名对齐', True, skipped=True,
                      errors=['结果表未生成齐：%s' % '、'.join(miss)])

    keep_root, keep_res = fr.ROOT, fr.RES
    tmp = tempfile.mkdtemp(prefix='regr_xlsx_')
    try:
        fr.ROOT = tmp
        fr.RES = os.path.join(tmp, 'results')
        os.makedirs(fr.RES)
        shutil.copy2(tmpl, os.path.join(tmp, '结果提交模板.xlsx'))
        for _, csv, _ in fr.SHEETS:
            shutil.copy2(os.path.join(OUT, csv), os.path.join(fr.RES, csv))
        out = os.path.join(tmp, '结果提交表-已填写.xlsx')

        with contextlib.redirect_stdout(io.StringIO()):
            fr.main()
        base = _sheet_grid(out)

        # 源 CSV 列序整体倒过来，再填一次：按列名对齐则逐格相同
        for _, csv, _ in fr.SHEETS:
            p = os.path.join(fr.RES, csv)
            df = pd.read_csv(p, encoding='utf-8-sig')
            df[list(df.columns)[::-1]].to_csv(p, index=False, encoding='utf-8-sig')
        with contextlib.redirect_stdout(io.StringIO()):
            fr.main()
        rev = _sheet_grid(out)

        errs = []
        if base != rev:
            diff = [(s, i, j) for s in base for i, (a, b) in
                    enumerate(zip(base[s], rev[s])) for j, (x, y) in
                    enumerate(zip(a, b)) if x != y]
            errs.append('列序倒置后填出的表不一致（%d 格）：%s'
                        % (len(diff), diff[:5]))
        if not base:
            errs.append('一张表都没填出来')

        # 行数核销：每张表的表体行数必须等于源 CSV 行数
        wb = openpyxl.load_workbook(out)
        for sheet, csv, _ in fr.SHEETS:
            n_csv = len(pd.read_csv(os.path.join(fr.RES, csv), encoding='utf-8-sig'))
            n_out = sum(1 for r in wb[sheet].iter_rows(min_row=2, values_only=True)
                        if any(v is not None for v in r))
            if n_out != n_csv:
                errs.append('[%s] 表体 %d 行，源 CSV %d 行' % (sheet, n_out, n_csv))

        # 百分数只换算一次：Q1 返航SOC 列应为 CSV 的 100 倍
        q1 = [s for s in fr.SHEETS if s[0] == 'Q1_单点组批'][0]
        df1 = pd.read_csv(os.path.join(fr.RES, q1[1]), encoding='utf-8-sig')
        if ('Q1_单点组批', '返航SOC') in fr.PCT_COLS:
            v0 = wb['Q1_单点组批'].cell(row=2, column=_col_of(wb['Q1_单点组批'], '返航SOC')).value
            if abs(float(v0) - float(df1['返航SOC'].iloc[0]) * 100.0) > 1e-6:
                errs.append('返航SOC 未按 %% 换算：表内 %r，CSV %r'
                            % (v0, df1['返航SOC'].iloc[0]))

        # 护栏：源 CSV 多出一列且未声明不提交 → 必须中止
        extra = q1[1]
        p = os.path.join(fr.RES, extra)
        dfx = pd.read_csv(p, encoding='utf-8-sig')
        dfx['__多余列__'] = 1
        dfx.to_csv(p, index=False, encoding='utf-8-sig')
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                fr.main()
            errs.append('源表多出未声明列时未中止')
        except SystemExit:
            pass
        return result('T22 结果表列名对齐', not errs, errs,
                      metrics={'已填表数': len(base),
                               '表体总行数': sum(len(v) - 1 for v in base.values()),
                               '列序倒置后差异格': 0 if base == rev else -1},
                      tolerance='列序无关（逐格相同）；多列未声明必须中止')
    finally:
        fr.ROOT, fr.RES = keep_root, keep_res
        shutil.rmtree(tmp, ignore_errors=True)


def _sheet_grid(path):
    """把 xlsx 读成 {表名: [[表头], [行], ...]}。"""
    wb = openpyxl.load_workbook(path)
    out = {}
    for ws in wb.worksheets:
        rows = [[v for v in r] for r in ws.iter_rows(values_only=True)]
        rows = [r for r in rows if any(v is not None for v in r)]
        if rows:
            out[ws.title] = rows
    return out


def _col_of(ws, key):
    """按列名的**主干**找列号：模板表头带全角单位（「返航SOC（%）」），
    这里只用来定位，不参与对齐逻辑本身。"""
    for c in ws[1]:
        if c.value is not None and key in str(c.value):
            return c.column
    raise KeyError(key)


# ===========================================================================
# 汇总
# ===========================================================================
def main():
    reports = []
    for fn in CASES:
        try:
            reports.append(fn())
        except Exception as e:                                    # noqa: BLE001
            reports.append(result(fn.__name__, False,
                                  ['用例抛出未捕获异常：%r' % (e,)]))
    print('=' * 74)
    n_pass = n_skip = n_fail = 0
    for r in reports:
        tag = 'SKIP' if r['skipped'] else ('PASS' if r['passed'] else 'FAIL')
        if r['skipped']:
            n_skip += 1
        elif r['passed']:
            n_pass += 1
        else:
            n_fail += 1
        print(f'[{tag}] {r["name"]}')
        if r['metrics']:
            print('        ' + '  '.join(f'{k}={v}' for k, v in r['metrics'].items()))
        for e in r['errors']:
            print('        ! ' + str(e))
    print('-' * 74)
    print(f'通过 {n_pass}，失败 {n_fail}，跳过 {n_skip}')
    if not reports:
        print('!! 一个用例都没跑，按失败处理')
        return 1
    return 1 if n_fail else 0


if __name__ == '__main__':
    raise SystemExit(main())
