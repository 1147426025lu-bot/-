# -*- coding: utf-8 -*-
"""
第三问中继排班器（阶段 13）：允许「悬停站与悬停站直接转场」的连飞出动。

阶段 12 的诊断（`_run_logs/stage12_q3_diagnosis.md` 第 7 节）指出：现解 7 个中继
架次共 16835 s 飞行，其中去程 + 返航 8017 s（占 47.6%）全部是「基地 ↔ 悬停站」
的转场；而中继机可用能量 (1-ρ)·E_use = 2.56 kWh，最大的一架次只用 1.123 kWh
（44%）。也就是说**一架中继机在一次出动里连做两个站在能量上是允许的**，现在每次
都飞回 O01 再出去，那一段来回是纯浪费。

本模块把「一次出动 = 一次悬停站服务、必返 O01」放宽为
「一次出动 = 从 O01 出发、依次服务若干悬停站、最后返回 O01」：

    allow_chain=True   连飞档（本模块的主产物）
    allow_chain=False  退回旧规则，且**直接沿用旧实现的成本函数与状态更新，
                       故与 `q3.schedule_relays` 逐位一致**（不是「近似等价」）

口径与旧实现逐项对齐：
  · 建链完成时刻 = 飞抵悬停站的时刻 + 建链时长 link；晚于区间终点 t2 即弃飞
    （空飞同样占用机身与能源组件，会把后面本可按时到达的架次一起顶迟到）；
  · 必须悬停到 t2 才能离开（该站所辖失效区间要保障到这一刻）；
  · 能源上限按**整次出动**核算：(1-ρ)·E_use，含全部去程/转场、全部悬停、末端返航；
  · 机身占用到「返航 + turnover」，能源组件占用到「返航 + 充电完成」，
    充电时长由整次出动的累计能耗反算（core.charge_time）；
  · 能源组件池全局共享，且**派出即预约**：在派出的那一刻就把预估的释放时刻写进
    池子，而不是等返航再登记。旧实现正是这么做的，于是「已经飞出去的组件」不会
    被第二架次重复取用，也就不需要额外的占用标记。

连飞不是无代价的：它把多次服务的时间与航程叠在一次出动里，单次出动的能耗上升、
机身离开基地的时间变长，可能推迟后续派发。所以排班仍是按失效区间起点顺序的贪心，
每次取「建链完成最早」的一个（并列时取不连飞的那个——与阶段 13 的收益原型
`_probe/stage13_chain_full.py` 的取舍规则一致，那条路径上的收益已经过完整级联验证）：
  A 接续上一站：同一组件、不返基地，直接转场过去；
  B 返基地换组件后重新出动。
"""
from core import (segment_geometry, relay_flight_time, relay_flight_energy,
                  relay_hover_energy, charge_time)


def _leg(d, rt, p, q):
    """航段 (时间, 能耗)，p/q 为 (经度, 纬度, 绝对海拔)。"""
    g = segment_geometry(d.dem, p[0], p[1], p[2], q[0], q[1], q[2])
    return relay_flight_time(rt, g), relay_flight_energy(rt, g)


def outings(trips):
    """把 `schedule_relays` 的返回按**出动**归并：{出动编号: [访问行, ...]}。

    凡是要报「几个中继架次」「中继总能耗」的地方都必须走这里：一排一行 = 一次
    悬停站服务，出动级的 `E` 在该出动的每一行上重复出现，直接对 `trips` 求和会把
    接续出动的能耗按访问次数重复计入；`len(trips)` 数出来的也是访问数不是架次。
    """
    out = {}
    for x in trips:
        out.setdefault(x['outing'], []).append(x)
    return out


def outing_stats(trips):
    """(出动数, 中继总能耗 kWh, 站间接续次数)——报告口径的三个数一次算清。"""
    out = outings(trips)
    E = 0.0
    n_chain = 0
    for rows in out.values():
        E += rows[0]['E']
        n_chain += len(rows) - 1
    return len(out), E, n_chain


def schedule_relays(d, jobs, cands, verbose=False, allow_chain=True):
    """中继排班。返回 (trips, skipped)。

    `trips` 的**一行 = 一次悬停站服务**（与结果表 `q3_relay_trips.csv` 一行一致），
    同一次出动的多行共享 `出动编号`。各字段：
      访问级：station/lon/lat/alt_abs/hover_agl/ground_elev/link_done/service_end/
              t_service/visit_E/seq/chain
      出动级：outing/relay/comp/start/return_t/E（同一次出动内各行取同值）
      `E` 是**整次出动**的能耗，`visit_E` 是本次访问的能耗份额，二者满足
      Σ(同出动 visit_E) == E（由下方的构造保证，结果表与 verify.py 各自复核）。
    `skipped` 每项含 `job` 与有限的 `arrive`（修复环要拿它算需要推迟多少）。
    """
    rt = d.relay_type
    if not d.relay_uavs:
        raise RuntimeError('数据中继无人机清单为空，无法排班')
    cap = (1.0 - rt['rho']) * rt['E_use']
    o01 = (d.O01['lon'], d.O01['lat'], d.O01['alt'])
    T_full = d.relay_batteries[1]
    prep, link, turnover = rt['prep'], rt['link'], rt['turnover']

    uavs = [dict(id=u['id'], free_at=0.0, pos=None, pos_end=0.0, open_E=0.0,
                 comp=None, depart=0.0, seq=0, outing=None) for u in d.relay_uavs]
    comps = {'RC%d' % (k + 1): 0.0 for k in range(d.relay_batteries[0])}
    n_out = 0
    trips, skipped = [], []
    # 出动级账本按**出动编号**记，不挂在机身上：一架中继机在本次排班里会依次
    # 进行多次出动，挂在机身上的话，后一次出动会把前一次的记录冲掉。
    out_state = {}   # 出动编号 -> dict(vis_sum, ret, E_tot)
    out_rows = {}    # 出动编号 -> [访问行, ...]

    def settle(u):
        """连飞档专用：把机身与组件的投影重算为「以当前位置收尾」。

        位置（`pos`/`pos_end`/`open_E`）一变就调用一次。旧实现只在派出时写一次
        投影，因为它的出动从不延长；连飞会延长出动，所以必须跟着改写，否则池子
        里的组件释放时刻会停留在半途，后面的架次会提前取到还在充电的组件。
        """
        if u['pos'] is None:
            return
        st_ = out_state[u['outing']]
        tf_b, E_b = _leg(d, rt, u['pos'], o01)
        st_['ret'] = u['pos_end'] + tf_b
        st_['E_tot'] = st_['vis_sum'] + E_b
        u['free_at'] = st_['ret'] + turnover
        comps[u['comp']] = st_['ret'] + charge_time(
            1.0 - st_['E_tot'] / rt['E_use'], T_full)

    for job in jobs:
        ci = job['ci']
        st = cands[ci]
        pt = (st['lon'], st['lat'], st['alt_abs'])
        tf_out, E_out = _leg(d, rt, o01, pt)
        tf_back, E_back = _leg(d, rt, pt, o01)
        t2 = job['t2']

        def handoff(u):
            """候选 A：从上一站直接转场过去，同一组件、不返基地。"""
            if not (allow_chain and u['pos'] is not None):
                return None
            tf, E_f = _leg(d, rt, u['pos'], pt)
            ld = u['pos_end'] + tf + link
            if ld > t2 + 1e-6:
                return None
            svc = max(0.0, t2 - ld)
            v_E = E_f + relay_hover_energy(rt, svc)
            if u['open_E'] + v_E + E_back > cap + 1e-9:
                return None
            return dict(chain=True, link_done=ld, svc=svc, start=u['depart'],
                        comp=u['comp'], visit_E=v_E, pos_end=ld + svc)

        def relaunch(u):
            """候选 B：返基地换组件后重新出动。"""
            cid = min(comps, key=lambda k: comps[k])
            if not allow_chain:
                # 不连飞档：与旧实现 `q3.schedule_relays` **同一段算术**（同一个
                # 成本函数、同一个赋值顺序）。这里刻意不复用上面的分段写法：
                # 浮点加法不满足结合律，换个顺序就会在末位差 1 ulp，一旦落在
                # 结果表 3/6 位小数的舍入边界上，落盘字节就不再等于旧解。
                from q3 import relay_trip_cost
                c0 = relay_trip_cost(d, pt, 0.0)
                t_start = max(u['free_at'], comps[cid],
                              job['t1'] - c0['t_flight_out'] - prep - link, 0.0)
                arrive = t_start + prep + link + c0['t_flight_out']
                if arrive > t2 + 1e-6:
                    return None
                svc = max(0.0, t2 - arrive)
                cost = relay_trip_cost(d, pt, svc)
                return dict(chain=False, link_done=arrive, svc=svc, start=t_start,
                            comp=cid, visit_E=cost['E'], pos_end=arrive + svc,
                            E_tot=cost['E'], t_tot=cost['t_tot'],
                            soc=1.0 - cost['E'] / rt['E_use'])
            t_start = max(u['free_at'], comps[cid], job['t1'] - tf_out - prep - link, 0.0)
            ld = t_start + prep + link + tf_out
            if ld > t2 + 1e-6:
                return None
            svc = max(0.0, t2 - ld)
            v_E = E_out + relay_hover_energy(rt, svc)
            if v_E + E_back > cap + 1e-9:
                return None
            return dict(chain=False, link_done=ld, svc=svc, start=t_start,
                        comp=cid, visit_E=v_E, pos_end=ld + svc)

        # --- 选机选组件 ---
        best = None
        if allow_chain:
            for u in uavs:
                for o in (handoff(u), relaunch(u)):
                    if o is None:
                        continue
                    if best is None or (round(o['link_done'], 6), o['chain']) < \
                            (round(best[1]['link_done'], 6), best[1]['chain']):
                        best = (u, o)
        else:
            # 旧档：按机身可用时刻选机、按组件就绪时刻选组件。二者在旧实现里是
            # 两段独立贪心（先 `min(relays, key=arrive_time)`，再 `min(comps)`），
            # 这里逐字照搬，不做合并。
            u = min(uavs, key=lambda x: x['free_at'] + prep + link + tf_out)
            o = relaunch(u)
            best = None if o is None else (u, o)

        if best is None:
            # 弃飞。`arrive` 取「不受 t2 限制」的最早建链时刻：修复环要拿它算这条
            # job 需要推迟多少，给一个无法落地的下界会低估推迟量。
            earliest = None
            for u in uavs:
                o = handoff(u)
                if o is not None:
                    ld = o['link_done']
                else:
                    cid = min(comps, key=lambda k: comps[k])
                    ld = (max(u['free_at'], comps[cid],
                              job['t1'] - tf_out - prep - link, 0.0)
                          + prep + link + tf_out)
                earliest = ld if earliest is None else min(earliest, ld)
            skipped.append(dict(job=job, arrive=float(earliest)))
            continue

        u, o = best
        if o['chain']:
            out_state[u['outing']]['vis_sum'] += o['visit_E']
            u.update(pos=pt, pos_end=o['pos_end'], open_E=u['open_E'] + o['visit_E'],
                     seq=u['seq'] + 1)
            settle(u)
        else:
            n_out += 1
            od = 'D%02d' % n_out
            out_state[od] = dict(vis_sum=o['visit_E'], ret=0.0, E_tot=0.0)
            out_rows[od] = []
            u.update(pos=pt, pos_end=o['pos_end'], open_E=o['visit_E'],
                     comp=o['comp'], depart=o['start'], seq=1, outing=od)
            if allow_chain:
                settle(u)
            else:
                # 旧档：投影直接取自成本函数，不做第二次运算
                out_state[od]['ret'] = o['start'] + o['t_tot']
                out_state[od]['E_tot'] = o['E_tot']
                u['free_at'] = out_state[od]['ret'] + turnover
                comps[o['comp']] = out_state[od]['ret'] + charge_time(o['soc'], T_full)
        row = dict(relay=u['id'], comp=u['comp'], station=ci, ivs=list(job['ivs']),
                   start=u['depart'], lon=st['lon'], lat=st['lat'], alt_abs=st['alt_abs'],
                   hover_agl=st['agl'], ground_elev=st['ground'],
                   link_done=o['link_done'], service_end=o['pos_end'],
                   t_service=o['svc'], late=(o['link_done'] > job['t1'] + 1e-6),
                   outing=u['outing'], seq=u['seq'], chain=o['chain'],
                   E=0.0, return_t=0.0, visit_E=o['visit_E'])
        out_rows[u['outing']].append(row)
        trips.append(row)

    # 出动级字段定稿：`E` 与 `return_t` 随接续不断变长，派出那一刻写下的只是中间态，
    # 故统一在这里按该次出动的终态回填（一行 = 一次访问，一行内的出动级字段 =
    # 整次出动）。不连飞档的 `E_tot`/`ret` 取自成本函数，回填不改变它们的值。
    for od in sorted(out_rows):
        rows = out_rows[od]
        st_ = out_state[od]
        if allow_chain:
            # 出动的**末次**访问另计入末端返航段（末端返航不属于任何中间站）。
            # 于是「出动能耗 = 各访问份额之和」就是一条逐项可复算的恒等式，
            # 结果表与 verify.py 各自按自己那条路径复核。
            #
            # 这一条**对只有一次访问的出动同样成立**，不能写成 `len(rows) > 1`：
            # 份额的构造里，首站份额只有「进场航段 + 本站悬停」，末端返航从不预先
            # 摊进任何一站（`relaunch` 的 `visit_E = E_out + hover`、`settle` 的
            # `E_tot = vis_sum + E_back`）。漏掉单访问出动就会让该出动的份额之和
            # 比 `E` 少一整段返航（实测 0.3608 vs 0.5045 kWh），而_只_含连飞出动
            # 的算例恰好掩盖了这一点——T23 早期版本正是因此通过。
            last = rows[-1]
            _tf, E_b = _leg(d, rt, (last['lon'], last['lat'], last['alt_abs']), o01)
            last['visit_E'] = last['visit_E'] + E_b
            tot = sum(x['visit_E'] for x in rows)
            # `E_tot` 由 settle 在排班时累加（供组件充电时刻用），此处按份额重加
            # 一遍。两者是同一个量的两种加法顺序，浮点末位可能差 1 ulp，故取容差。
            if abs(tot - st_['E_tot']) > 1e-12:
                raise AssertionError(
                    '%s 出动能耗与各访问份额之和不符：%.12f vs %.12f'
                    % (od, tot, st_['E_tot']))
        for x in rows:
            x['E'] = st_['E_tot']
            x['return_t'] = st_['ret']
            x['n_visits'] = len(rows)

    trips.sort(key=lambda x: (x['start'], x['seq']))
    if verbose:
        n_chain = sum(1 for x in trips if x['chain'])
        print(f'中继架次: {len(trips)} 次访问 / {n_out} 次出动'
              f'（站间接续 {n_chain} 次，弃飞 {len(skipped)} 个）')
    return trips, skipped
