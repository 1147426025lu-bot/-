# -*- coding: utf-8 -*-
# 本程序及代码是在人工智能工具辅助下完成的。
# 工具名称：DeepSeek；版本/型号：DeepSeek V4.1 Flash（API 模型名 deepseek-flash）；
# 开发机构：杭州深度求索人工智能基础技术研究有限公司；
# 版本发布日期：2026 年 9 月 10 日。

"""
问题三 v2：运输重排 ↔ 中继保障 的固定点迭代错峰。

来源是对一份外部解法包（`code_opt/q3_opt.py`）的复核与部分移植。**只取不含高度
假设的联合调度部分**：外部把 S004/S005/S007/S013 的作业高度抬到离地 200–290 m
（`q3_opt.HIGH_ALT_DIRECT`），那违反 `problem.txt:46`（附录 2）「服务区作业高度取
其地面海拔以上 30 米」，故 `HIGH_ALT_DIRECT`/`analyze_communication`/`build_ha_geo`
/`retrip_ha` 一律不移植，本模块内所有作业高度仍由 `q3.sample_trip_trajectory` 的
`+30.0` 决定。也不移植其 `build_sorties` 跨簇链模型（一条 sortie 跨多个簇，与我方
`q3_relay_trips.csv`「一行 = 一个悬停站」的列语义、三条独立复核与 Excel 回填冲突）。

外部函数 → 本模块的对应关系：

    q3_opt.slack_tid            -> 直接用我方 q3.trip_slack（口径相同：医疗取 expect、
                                   首批取 deadline、其余不设限）
    q3_opt.resimulate_transport -> resimulate
    q3_opt 的 639-669 固定点     -> solve_joint_v2
    q3_opt.schedule_relays      -> 仍用我方 q3.schedule_relays
    q3_opt.build_spans(gap=600) -> 仍用我方 q3.merge_jobs(gap=900)

**与外部做法的实质差别（必须说清，否则会高估移植收益）**：外部自己写了一套双机
「直飞转场 / 回站换电」的中继仿真，并在簇划分上做 2^n 枚举；本模块不重复实现中继
排班，而是**直接用我方 `q3.schedule_relays` 作为内层可行性判定器**——同一份记账、
同一份能耗口径、同一份 CSV 语义，三条独立复核（`verify.check_relay`、
`verify.interval_binding_errors`、`fill_results` 的列消费）无需感知本次移植。
代价是失去外部的「不返场直接转场」模式（我方中继每趟都回 O01），
故外部那个 9669.6 s 的联合完工时刻在本模块口径下**不可能**复现，
本模块只主张「以我方口径，错峰结果是否更好」。

**要解决的问题**：我方 `q3.repair_readiness` 的就绪性修复是贪心级联，会沿资源链
把运输架次一路后推（实测把某架次推了 5714.8 s、累计 11862.3 s、加权时延 223301）。
本模块把它换成**受硬时限上界约束、且以「总释放量最小」为目标**的固定点：
每轮只把「中继实际晚到多少」作为该架次需要释放的量，摊到拥有该架次区间的运输
架次上，且**永不越过该架次的硬时限上界**（`q3.trip_slack`）。越过上界的路走不通，
就如实判该分支失败并回退旧分支，绝不为了消缺口去违反硬时限。
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')

import numpy as np                                                 # noqa: E402

import q3                                                          # noqa: E402
import q2                                                          # noqa: E402
from core import (charge_time, segment_geometry,                     # noqa: E402
                  relay_flight_energy, relay_hover_energy)

# 与 q2_v2 同理：`python code/q3.py` 时若本模块再 `import q3`，会拿到第二份 q3
# 副本（另一套模块级常量与状态）。本模块只调用 q3 的纯函数，数值上不会不同，
# 但「同一进程只有一份求解器模块」这条不变量值得守住，故把 q3 重绑到 __main__。
_main = sys.modules.get('__main__')
if (_main is not None and _main is not sys.modules.get(__name__)
        and all(hasattr(_main, a) for a in ('schedule_relays', 'merge_jobs', '_apply_shift'))):
    q3 = _main

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'results')

# 固定点迭代的最大轮数。外部取 12，这里一致：`release` 每轮单调不减且被
# trip_slack 夹住，12 轮足够看清「是否还在增长」，再多也只是重复同样的失败。
MAX_ROUNDS = 12
INF = float('inf')


def _sl(v):
    """slack 的比较值：无硬时限的架次用 INF，参与 min/max 时不失真。"""
    return INF if v is None or not np.isfinite(v) else float(v)


# ---------------------------------------------------------------------------
# 1) 运输重排：固定指派，按释放量重放资源链
# ---------------------------------------------------------------------------
def resimulate(d, base_asg, release=None):
    """给定逐架次释放量，重放运输资源链，返回新的 assignment。

    口径与 `q2.schedule` 内部完全一致（不另写一份）：
      开始 = max(该机型的无人机空闲, 该能源组件就绪, **基准开始 + 释放量**)
      能源组件就绪 = 开始 + 时长 + charge_time(soc_end, Tfull)
    `charge_time` 是两参数、soc_end 在前。

    以**基准开始**（而非上一轮的结果）为下限，是为了让 `release` 始终是「相对问题二
    原始排班的推迟量」——这正是 `q3_stagger.csv` 的 `推迟s` 列语义，也让每轮重放
    与轮次无关（增量式与一次算到位等价）。

    重放顺序取「基准开始时刻升序」，不是随便挑的：对任一资源，其占用者按派工顺序
    的开始时刻必然严格递增（后一架次至少要等前一架次结束），故按开始时刻排序等价于
    按每个资源各自的派工顺序重放，资源空闲时刻与 `q2.schedule` 逐位相同。因此
    release 全 0 时本函数必须逐位复现原排班——由 self_check 断言把关。
    """
    n = len(base_asg)
    rel = [0.0] * n if release is None else list(release)
    if len(rel) != n:
        raise ValueError('release 长度必须与 assignment 一致')
    order = sorted(range(n), key=lambda i: (base_asg[i]['start'], i))
    uav_free, bat_ready = {}, {}
    out = [None] * n
    for i in order:
        a = base_asg[i]
        g = a['type']
        st = max(uav_free.get(a['uav'], 0.0), bat_ready.get(a['battery'], 0.0),
                 a['start'] + rel[i])
        dur = a['duration']
        soc_end = 1.0 - a['E'] / d.transport_types[g]['E_use']
        uav_free[a['uav']] = st + dur
        bat_ready[a['battery']] = st + dur + charge_time(soc_end, d.batteries[g][1])
        shift = st - a['start']
        b = dict(a)
        b['route'] = list(a['route'])
        b['boxes_at'] = {s: list(bs) for s, bs in a['boxes_at'].items()}
        b['deliver_abs'] = {k: v + shift for k, v in a['deliver_abs'].items()}
        b['start'] = st
        out[i] = b
    return out


# ---------------------------------------------------------------------------
# 2) 释放量的分摊：总释放最小，且永不越过硬时限上界
# ---------------------------------------------------------------------------
def _spread(need, owners, release, slack):
    """把某个弃飞架次所需的释放量 need 摊到拥有其区间的运输架次上。

    优先给硬时限余量最大的架次（余量大的先推迟，对交付的影响最小），
    余量相同按下标升序，保证同种子可复现。单个架次的释放量**永不超过其 slack**：
    越过 slack 就是硬时限违反，那是不可谈判的失败，只能让分支整体作废。
    返回 (实际吸收量, 未能吸收的余量)。
    """
    left = need
    for i in sorted(owners, key=lambda k: (-_sl(slack[k]), k)):
        if left <= 1e-9:
            break
        room = _sl(slack[i]) - release[i]
        if room <= 1e-9:
            continue
        add = min(left, room)
        release[i] += add
        left -= add
    return need - left, left


# ---------------------------------------------------------------------------
# 3) 固定点：运输重排 ↔ 中继保障
# ---------------------------------------------------------------------------
def solve_joint_v2(d, base_asg, intervals, sel, cands, slack, verbose=False,
                   max_rounds=MAX_ROUNDS):
    """运输重排与中继保障的固定点迭代。

    每轮：
      ① 按当前 release 重放运输资源链 → 新排班；
      ② 把失效区间随所属架次整体平移（`_apply_shift` 在纯平移下精确，区间形状不变）；
      ③ 用我方 `merge_jobs` + `schedule_relays` 排中继，看哪些架次弃飞；
      ④ 弃飞架次 = 中继晚到，把「晚到多少」作为需要释放的量摊回其区间所属的运输架次。

    终止：弃飞清空（成功），或 release 不再增长（已顶到硬时限上界，判失败），
    或达到 max_rounds（判未收敛）。失败一律如实返回，由调用方回退旧分支，
    绝不以违反硬时限为代价强行消掉缺口。

    `intervals` 会被就地改写为与返回排班一致的状态；调用方若要保住原状态，
    需自行先 `q3._snapshot_intervals`。
    返回 dict，含 assignment/deltas/relays/skipped/jobs/rounds/converged/reason。
    """
    t0 = time.time()
    n = len(base_asg)
    release = [0.0] * n
    history = []
    asg = base_asg
    jobs, relays, skipped = [], [], []
    converged, reason = False, ''
    rounds = 0
    for rnd in range(1, max_rounds + 1):
        rounds = rnd
        asg = resimulate(d, base_asg, release)
        q3._apply_shift(intervals, asg)
        jobs = q3.merge_jobs(intervals, sel)
        relays, skipped = q3.schedule_relays(d, jobs, cands)
        total_delay = float(sum(release))
        history.append(dict(round=rnd, n_jobs=len(jobs), n_relays=len(relays),
                            n_skipped=len(skipped), total_delay=total_delay))
        n_late = sum(1 for x in relays if x['late'])
        if verbose:
            print(f'    固定点第 {rnd} 轮：中继架次 {len(relays)}、弃飞 {len(skipped)}、'
                  f'迟建链 {n_late}、累计释放 {total_delay:.0f} s')
        # 收敛判据必须同时要求「无弃飞」与「无迟建链」，缺一不可：本轮只以弃飞为
        # 判据时，第 3 轮弃飞清零就判成功，可真实缺口还有 1632 s——迟建链的中继
        # 只保住了区间尾段，区间早段无人保障。要的是零中断，不是零弃飞。
        if not skipped and not n_late:
            converged, reason = True, '弃飞与迟建链均已清空'
            break
        # 先把本轮所需释放量**全部算出来再一起施加**：逐条施加会让后面的架次看到
        # 前面刚改过的 release，摊分结果依赖 skipped 的遍历顺序，破坏可复现性。
        new = list(release)
        # 两类「保障不到位」都要摊回去：
        #   ① 中继根本没到（弃飞）—— 需求是把到达时刻压进区间终点 t2；
        #   ② 中继到了但来晚了（`late`）—— `schedule_relays` 允许建链完成晚于区间
        #      起点，此时区间早段无人保障、`_outage_proxy` 照算缺口，而弃飞数是 0。
        #      这一类必须把建链完成压到区间**起点 t1** 之前（不是 t2）：只压进 t2
        #      只能保住区间尾段。实测正是漏了 ② 才留下 4.22% 的真实中断。
        needs = []
        by_ivs = {frozenset(x['ivs']): x for x in relays}
        for sk in skipped:
            needs.append((sk['arrive'] - sk['job']['t2'], sk['job']))
        for job in jobs:
            x = by_ivs.get(frozenset(job['ivs']))
            if x is not None and x['late']:
                needs.append((x['link_done'] - job['t1'], job))
        needs.sort(key=lambda p: (-p[0], p[1]['t1']))
        absorbed_tot, unabsorbed_tot = 0.0, 0.0
        for need, job in needs:
            if need <= 1e-9:
                continue
            owners = {intervals[k]['trip'] for k in job['ivs']}
            got, left = _spread(need, owners, new, slack)
            absorbed_tot += got
            unabsorbed_tot += left
        if max(abs(x - y) for x, y in zip(new, release)) < 1e-9:
            # 一个字节都推不动了：要么所有 owner 的 slack 已耗尽，要么这些架次本就
            # 没有可推迟的余量。继续迭代只是重复同一结果。
            reason = ('release 已无增长空间：本轮需释放 %.0f s，'
                      '其中 %.0f s 超出所属架次的硬时限上界'
                      % (absorbed_tot + unabsorbed_tot, unabsorbed_tot))
            break
        release = new
    else:                       # 跑满 max_rounds 都没 break：release 一直在长，未收敛
        reason = f'{max_rounds} 轮内未收敛（release 仍在增长）'
    deltas = [a['start'] - b['start'] for a, b in zip(asg, base_asg)]
    return dict(assignment=asg, deltas=deltas, relays=relays, skipped=skipped,
                jobs=jobs, rounds=rounds, converged=converged, reason=reason,
                release=list(release), history=history,
                total_delay=float(sum(release)), secs=time.time() - t0,
                n_staggered=sum(1 for x in release if x > 1e-9))


# ---------------------------------------------------------------------------
# 4) 自检
# ---------------------------------------------------------------------------
def self_check(d, base_asg, intervals, sel, cands, verbose=True):
    """移植引入的两处新逻辑的硬性自检。返回 True 表示全过。

    1) 释放量全 0 时 `resimulate` 必须逐位复现原排班，且不产生资源冲突。
       这条是整套固定点的地基：不成立就说明「开始 = max(资源空闲, 基准+释放)」
       与我方 `q2.schedule` 的口径不一致，后面所有轮次都建立在错的基准上。
    2) 中继能耗口径：`q3.schedule_relays` 产出的每一次**出动**，其 E 必须等于
       沿该次出动实际路径（O01→站₁→…→站_n→O01）逐段重算的飞行能耗 + 各站悬停
       能耗之和。阶段 13 起一次出动可站间接续，逐行拿 `q3.relay_trip_cost` 去比
       在连飞下必然对不上（那正是接续省下来的转场），故改为按出动沿路径重算——
       这仍是独立于排班器内部记账的第二次计算。
    """
    ok = True
    same = resimulate(d, base_asg, [0.0] * len(base_asg))
    bad = [i for i, (x, y) in enumerate(zip(same, base_asg))
           if abs(x['start'] - y['start']) > 1e-9]
    conf = q2.resource_conflicts(d, same)
    if bad or conf:
        ok = False
    if verbose:
        print(f'  释放量为 0 的重放对拍：开始时刻不一致 {len(bad)} 例、'
              f'资源冲突 {len(conf)} 例（都必须为 0）')

    jobs = q3.merge_jobs(intervals, sel)
    relays, _sk = q3.schedule_relays(d, jobs, cands)
    by_out = {}
    for x in relays:
        by_out.setdefault(x['outing'], []).append(x)
    rt = d.relay_type
    worst = 0.0
    n_chain = 0
    for od in sorted(by_out):
        rows = sorted(by_out[od], key=lambda x: x['seq'])
        n_chain += len(rows) - 1
        prev = (d.O01['lon'], d.O01['lat'], d.O01['alt'])
        E = 0.0
        for x in rows:
            E += relay_flight_energy(rt, segment_geometry(
                d.dem, prev[0], prev[1], prev[2], x['lon'], x['lat'], x['alt_abs']))
            E += relay_hover_energy(rt, x['t_service'])
            prev = (x['lon'], x['lat'], x['alt_abs'])
        E += relay_flight_energy(rt, segment_geometry(
            d.dem, prev[0], prev[1], prev[2],
            d.O01['lon'], d.O01['lat'], d.O01['alt']))
        worst = max(worst, abs(E - rows[0]['E']))
    if worst > 1e-9:
        ok = False
    if verbose:
        print(f'  中继能耗口径对拍：{len(by_out)} 次出动（其中站间接续 {n_chain} 次），'
              f'沿实际路径重算与方案能耗的最大偏差 {worst:.3e} kWh（必须为 0）')
    return ok
