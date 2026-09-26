# -*- coding: utf-8 -*-
"""
阶段 13 的**冻结参照**：改动前的 `q3.schedule_relays` 逐字副本。

来源：阶段 13 之前的 `code/q3.py` 里的 `schedule_relays`，函数体一字未动（连注释
一起抄下来），只在文件头加了这段说明。用途只有一个——让「中继排班器的不连飞档与
旧规则逐位一致」这条验收在 `code/q3.py` 被改写**之后**仍然可复跑：

    `code/q3_chain.schedule_relays(allow_chain=False)` 必须与本模块的
    `schedule_relays` 在每一个字段上逐位相等（回归用例 T23）。

**它不在求解路径上**：`code/q3.py` 与 `code/q3_chain.py` 都不 import 它，论文里的
任何数字都不是它算的，因此它也不在 `solution_io.Q3_SOLVER_FILES` 的哈希清单里
（改了它不会、也不该使已落盘的方案失效）。留在这里而不是只放在 `_probe/`，是为了
让评审按《交付与复现说明》跑回归测试时 T23 能跑起来——旧规则可复现这件事本身就是
一条要交付的结论。

注意：本模块依赖 `q3.relay_trip_cost` / `q3._as_st` 与 `core.charge_time`——这三者
在阶段 13 中未改动，故冻结副本的算术与当时完全一致。
"""
from core import charge_time

from q3 import relay_trip_cost, _as_st


def schedule_relays(d, jobs, cands, verbose=False):
    """
    中继排班：每站去程时间已知，按失效区间起点顺序派发。保留「建链完成晚于
    区间结束即弃飞」规则——空飞同样占用中继与能源组件，会把后续本可按时到达
    的架次一起顶迟到。
    """
    rt = d.relay_type
    # 中继机清单从数据读（`d.relay_uavs`），不写死 ['R01','R02']：数据一旦增减
    # 机数，写死的版本会静默按 2 架排班，而报出的机队规模仍是数据里那个数。
    if not d.relay_uavs:
        raise RuntimeError('数据中继无人机清单为空，无法排班')
    relays = [dict(id=u['id'], free_at=0.0) for u in d.relay_uavs]
    comps = [dict(id=f'RC{k+1}', ready_at=0.0) for k in range(d.relay_batteries[0])]
    out_flight = {}
    for ci in {j['ci'] for j in jobs}:
        st = _as_st(cands[ci])
        out_flight[ci] = relay_trip_cost(d, st, 0.0)['t_flight_out']
    trips, skipped = [], []
    for job in jobs:
        ci = job['ci']
        st = cands[ci]
        tf = out_flight[ci]

        def arrive_time(r):
            return r['free_at'] + rt['prep'] + rt['link'] + tf

        r = min(relays, key=arrive_time)
        c = min(comps, key=lambda x: x['ready_at'])
        t_depart = job['t1'] - tf - rt['prep'] - rt['link']
        t_start = max(r['free_at'], c['ready_at'], t_depart, 0.0)
        arrive = t_start + rt['prep'] + rt['link'] + tf
        if arrive > job['t2'] + 1e-6:
            skipped.append(dict(job=job, arrive=arrive))
            continue
        t_service = max(0.0, job['t2'] - arrive)
        cost = relay_trip_cost(d, (st['lon'], st['lat'], st['alt_abs']), t_service)
        r['free_at'] = t_start + cost['t_tot'] + rt['turnover']
        soc_end = 1.0 - cost['E'] / rt['E_use']
        c['ready_at'] = t_start + cost['t_tot'] + charge_time(soc_end, d.relay_batteries[1])
        trips.append(dict(relay=r['id'], comp=c['id'], station=ci, ivs=list(job['ivs']),
                          start=t_start, lon=st['lon'], lat=st['lat'], alt_abs=st['alt_abs'],
                          hover_agl=st['agl'], ground_elev=st['ground'],
                          link_done=arrive, service_end=arrive + t_service,
                          return_t=t_start + cost['t_tot'], E=cost['E'], t_service=t_service,
                          late=(arrive > job['t1'] + 1e-6)))
    trips.sort(key=lambda x: x['start'])
    if verbose:
        print(f'中继架次: {len(trips)} 个（弃飞 {len(skipped)} 个）')
    return trips, skipped
