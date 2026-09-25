# -*- coding: utf-8 -*-
# 本程序及代码是在人工智能工具辅助下完成的。
# 工具名称：DeepSeek；版本/型号：DeepSeek V4.1 Flash（API 模型名 deepseek-flash）；
# 开发机构：杭州深度求索人工智能基础技术研究有限公司；
# 版本发布日期：2026 年 9 月 10 日。

"""独立复核脚本：只读结果表与原始数据，重新计算论文第 9 章的检验项。

与 q1~q4.py 的区别在于不调用任何求解器的中间状态——硬约束全部从
results/*.csv 重新算一遍，DEM 离散化误差从原始 .mat 重新采样一遍。
这样做的目的是排除"求解程序自报成功"这一类错误。

报告与退出码（F06）：每项检查返回 `{name, passed, errors, metrics, tolerance}`，
`main()` 汇总后以非零退出码结束——任何一项失败、或一项都没跑成，都算失败。
结果表缺失一律记 FAIL（本脚本是门禁，缺了输入就不能称"通过"），不写 `except: pass`，
也不把异常吞掉：检查自身抛错按"检查失败"记录并继续跑完其余项。
论文中只能引用本脚本 PASS 的项；未跑到、跳过或失败的项必须如实降级表述。

用法：python code/verify.py
输出：屏幕报告，对应论文表 9.0--9.3 与 9.4 节的数值；全部通过时退出码 0。
"""
import re
import sys
import itertools
import os

sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from core import (DEM_STEP, T_DEC, E_DEC, load_data, charge_time, ll_to_xy,
                  roundtrip_energy, segment_time, segment_geometry)

# 结果表按 T_DEC 位存时刻、E_DEC 位存能耗，故独立重算与表内值之间必然存在
# 半个末位的舍入差。容差取 1.01 个末位：既不会把舍入噪声误判为错误，
# 又比任何真实的模型偏差（>1e-3 量级）小得多。
TOL_T = 0.5 * 10 ** (-T_DEC) * 1.01
TOL_E = 0.5 * 10 ** (-E_DEC) * 1.01

# 上两个容差是**单值**的舍入界。凡"从结果表求和后复算"的量，误差随项数累加：
# n 个各带半个末位的数相加，误差上界是 n × 半个末位。用单值容差去卡和式，
# 会把纯粹的舍入噪声判成不符（20 个架次能耗之和最大可差 1.0e-5 kWh）。
# 因此涉及求和处一律按下式放大，并在容差说明里写明放大依据。
def tol_sum(n, unit):
    """n 项各带 ±unit 舍入的求和容差上界。"""
    return max(abs(n), 1) * unit


# q3.py 的 solve_relay 以 Q3_DT 为采样步长，分段表的段边界取"相邻采样时刻的中点"
# （见 q3.py 共同边界序列的注释）。于是某一段的末端最多可越过该架次服务窗口
# 半个采样步长——段内每个采样点都真的落在窗口里，越界只发生在段的右端点这半个
# 步长上。复核据此取 Q3_DT/2 作窗口容差，并把实测最大越界量打印出来：
# 放宽了多少必须看得见，不能悄悄抹平。
Q3_DT = 1.0

RES = os.path.join(HERE, '..', 'results')
# 时刻列保留 T_DEC=3 位、能耗列保留 E_DEC=6 位。从这里反算共享电池的
# "返回时刻 + 充电时长" 时，两个时刻各带 ±5e-4 s，能耗舍入经充电曲线
# 斜率(|dT/dsoc| = T_full·0.65/0.9 ≈ 2167 s)放大后约 ±1.4e-4 s，
# 合计不超过 1.5e-3 s。取 0.01 s 作容差：比舍入噪声高一个量级留足余量，
# 又比任何真实的调度冲突小两个量级，不会把真冲突放过去。
TOL = 0.01


# ---------------------------------------------------------------------------
# 报告骨架（F06）
# ---------------------------------------------------------------------------
def rep(name, passed, errors=None, metrics=None, tolerance=None):
    return dict(name=name, passed=bool(passed), errors=list(errors or []),
                metrics=metrics or {}, tolerance=tolerance)


def _inp(name):
    """读结果表；缺表直接抛 FileNotFoundError，由 main() 记为该检查失败。"""
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        raise FileNotFoundError('缺少结果表 results/%s，请先跑通对应的求解脚本' % name)
    return pd.read_csv(p, encoding='utf-8-sig')


# ---------------------------------------------------------------------------
# 表 9.0  问题一
# ---------------------------------------------------------------------------
def _parse_plan(s):
    """解析 q1_pareto.csv 的『方案』列：'S002:2B1+C1' -> {区: (架次数, {机型: 架次数})}。"""
    out = {}
    for item in str(s).split(';'):
        item = item.strip()
        if not item or ':' not in item:
            continue
        si, rest = item.split(':')
        m = re.match(r'^(\d+)(.*)$', rest)
        if not m:
            continue
        k = int(m.group(1))
        mix = {}
        for g, n in re.findall(r'([A-Za-z])(\d+)', m.group(2)):
            mix[g] = mix.get(g, 0) + int(n)
        out[si.strip()] = (k, mix)
    return out


def check_q1(d):
    """表 9.0：问题一推荐组批方案的独立复核。

    只读 results/q1_recommended_batching.csv 与原始数据，用 core 的物理函数
    重新计算每架次的质量、体积、能耗与时间，不复用 q1.py 的任何中间量。

    质量/体积与清单不符（F07）计入失败，不只是打印；机型架次核对不再假设
    "每个服务区只用一个机型"，而是逐区核对「架次数 + 机型组合」与帕累托表
    中同 (架次数, 能耗) 那一行是否一致。
    """
    rec = _inp('q1_recommended_batching.csv')
    par = _inp('q1_pareto.csv')
    box = {}
    for si_id in d.S:
        for b in d.boxes_by_service[si_id]:
            box[b['id']] = (si_id, b['mass'], b['volume'])
    tot_boxes = len(box)

    bad = []
    seen = []
    n_over_m = n_over_v = n_over_e = n_unknown_box = 0
    n_badm = n_badv = 0
    dE = dT = dSOC = 0.0
    sum_E = 0.0
    per_area = {}
    for _, r in rec.iterrows():
        si_id = r['服务区编号']
        tid = r['机型编号']
        if si_id not in d.si:
            bad.append('明细表出现未知服务区编号 %s' % si_id)
            continue
        if tid not in d.transport_types:
            bad.append('架次 %s 的机型编号 %s 不在机型表中' % (r['架次编号'], tid))
            continue
        t = d.transport_types[tid]
        ids = [x for x in str(r['货箱编号列表']).split(';') if x]
        m = v = 0.0
        for bid in ids:
            if bid not in box:
                n_unknown_box += 1
                bad.append('架次 %s 引用了未知货箱编号 %s' % (r['架次编号'], bid))
                continue
            seen.append(bid)
            m += box[bid][1]
            v += box[bid][2]
        # 质量与体积必须与表内一致，且不超载
        dE_m = abs(m - float(r['总质量kg']))
        dE_v = abs(v - float(r['总体积m3']))
        if dE_m > 0.005:
            n_badm += 1
            bad.append('架次 %s 质量与货箱清单不符：差 %.4f kg' % (r['架次编号'], dE_m))
        if dE_v > 5e-5:
            n_badv += 1
            bad.append('架次 %s 体积与货箱清单不符：差 %.6f m3' % (r['架次编号'], dE_v))
        if m > t['Q'] + 1e-6:
            n_over_m += 1
            bad.append('架次 %s 超载：%.3f > %.3f kg' % (r['架次编号'], m, t['Q']))
        if v > t['volume'] + 1e-6:
            n_over_v += 1
            bad.append('架次 %s 超容积：%.4f > %.4f m3' % (r['架次编号'], v, t['volume']))
        # 能耗：用 core 独立重算往返能耗，并核对安全余量
        E, go, back = roundtrip_energy(t, d.dem, d.O01, d.si[si_id], m)
        sum_E += E
        if E > (1 - t['rho']) * t['E_use'] + 1e-9:
            n_over_e += 1
            bad.append('架次 %s 能耗超安全余量：%.6f kWh' % (r['架次编号'], E))
        dE = max(dE, abs(E - float(r['架次能耗kWh'])))
        dSOC = max(dSOC, abs((1.0 - E / t['E_use']) - float(r['返航SOC'])))
        # 时间：仿射式 T = (prep+hand_base+t_go+t_back) + (load+hand)*n
        t_aff = (t['prep'] + t['hand_base'] + segment_time(t, go)
                 + segment_time(t, back) + (t['load_per_box'] + t['hand_per_box']) * len(ids))
        dT = max(dT, abs(t_aff - float(r['往返时间s'])))
        k, mix = per_area.setdefault(si_id, (0, {}))
        mix[tid] = mix.get(tid, 0) + 1
        per_area[si_id] = (k + 1, mix)

    cnt = {}
    for b in seen:
        cnt[b] = cnt.get(b, 0) + 1
    dup = sorted(b for b, c in cnt.items() if c > 1)
    miss = sorted(set(box) - set(cnt))
    if dup:
        bad.append('货箱被重复装载 %d 个：%s%s' % (len(dup), '、'.join(dup[:5]),
                                             ' 等' if len(dup) > 5 else ''))
    if miss:
        bad.append('漏装货箱 %d 个：%s%s' % (len(miss), '、'.join(miss[:5]),
                                        ' 等' if len(miss) > 5 else ''))

    # 与全局帕累托表核对：推荐方案必须是其中一行，且逐区「架次数 + 机型组合」相同
    n_trip = len(rec)
    hit, plan = None, None
    for _, r in par.iterrows():
        if int(r['架次数']) == n_trip and abs(float(r['总能耗kWh']) - sum_E) <= 5e-4:
            hit, plan = r, _parse_plan(r['方案'])
            break
    n_mix_mis = 0
    if hit is None:
        bad.append('推荐方案（%d 架次 / %.6f kWh）不在 q1_pareto.csv 中'
                   % (n_trip, sum_E))
    else:
        if set(plan) != set(per_area):
            n_mix_mis += 1
            bad.append('服务区集合与帕累托表不符：%s vs %s'
                       % (sorted(per_area), sorted(plan)))
        for si_id, (k, mix) in sorted(per_area.items()):
            if si_id not in plan:
                continue
            k0, mix0 = plan[si_id]
            if k != k0 or mix != mix0:
                n_mix_mis += 1
                bad.append('%s 与帕累托表不符：%d%s vs %d%s'
                           % (si_id, k, '+'.join('%s%d' % x for x in sorted(mix.items())),
                              k0, '+'.join('%s%d' % x for x in sorted(mix0.items()))))

    min_ratio = sum_E / max(1e-9, sum(
        (1 - d.transport_types[r['机型编号']]['rho'])
        * d.transport_types[r['机型编号']]['E_use'] for _, r in rec.iterrows()))
    metrics = [
        ('架次 / 货箱引用 / 货箱总数', '%d / %d / %d' % (n_trip, len(seen), tot_boxes)),
        ('覆盖完整性', '重复 %d，漏装 %d，未知箱号 %d' % (len(dup), len(miss), n_unknown_box)),
        ('载重/容积越限', '质量 %d 处，体积 %d 处' % (n_over_m, n_over_v)),
        ('质量/体积与清单不符', '%d / %d 行' % (n_badm, n_badv)),
        ('能耗安全余量越限', '%d 处（判据 E <= (1-rho)E_use）' % n_over_e),
        ('独立重算总能耗', '%.6f kWh（表内求和 %.6f）' % (sum_E, float(rec['架次能耗kWh'].sum()))),
        ('独立重算最大偏差', '能耗 %.3e kWh，时间 %.3e s，返航SOC %.3e' % (dE, dT, dSOC)),
        ('与帕累托表核对', ('逐区架次数/机型组合一致 %d 区' % len(per_area)) if not n_mix_mis
                          else '%d 处不符' % n_mix_mis),
        ('舍入容差', '能耗 %.1e kWh，时间 %.1e s（各 1 个末位）' % (TOL_E, TOL_T)),
    ]
    passed = (not bad and dE <= TOL_E and dT <= TOL_T and dSOC <= TOL_E)
    if dE > TOL_E:
        bad.append('能耗与表内值偏差 %.3e kWh 超出 1 个末位' % dE)
    if dT > TOL_T:
        bad.append('时间与表内值偏差 %.3e s 超出 1 个末位' % dT)
    if dSOC > TOL_E:
        bad.append('返航 SOC 与表内值偏差 %.3e 超出 1 个末位' % dSOC)
    return rep('T9.0 问题一推荐组批方案', passed, bad, dict(metrics),
               '逐位重算：能耗/时间/SOC 各 1 个末位；套用比 %.3f' % min_ratio)


def check_q1_ilp(d):
    """表 9.0b：问题一「真实箱号子集 + 机型」集合划分 ILP 与 DP 的交叉复核。

    读 q1_ilp_mixed.csv 的**状态列**：只有 status='ok' 且 (k, E) 都对上的算「一致」。
    列数超限、超时、N/A 一律不计入一致数——未复核不等于一致，两者必须分开报。
    """
    ilp = _inp('q1_ilp_mixed.csv')
    n_ok = n_bad = n_un = 0
    bad = []
    unver = []
    for _, r in ilp.iterrows():
        if r['状态'] != 'ok':
            n_un += 1
            unver.append('%s(%s)' % (r['服务区'], str(r['状态'])[:18]))
            continue
        if r['一致'] == '是':
            n_ok += 1
        else:
            n_bad += 1
            bad.append('%s DP=%s/%.6f  ILP=%s/%s'
                       % (r['服务区'], r['DP架次'], r['DP能耗kWh'], r['ILP架次'],
                          r['ILP能耗kWh']))
    if n_bad:
        bad.insert(0, '集合划分 ILP 与 DP 有 %d 个算例不一致' % n_bad)
    metrics = [('区域数', len(ilp)),
               ('一致 / 不一致 / 未复核', '%d / %d / %d' % (n_ok, n_bad, n_un)),
               ('未复核清单', '、'.join(unver) if unver else '无')]
    return rep('T9.0b 问题一集合划分 ILP 交叉复核', not n_bad, bad, dict(metrics),
               '两法最小架次数相等且该架次数下能耗差 < 1e-6 kWh')


# ---------------------------------------------------------------------------
# 表 9.1  问题二
# ---------------------------------------------------------------------------
def check_hard_constraints(d):
    """表 9.1：问题二推荐方案的硬约束独立复核。

    时限口径（F07）：一个货箱可能**同时**受「医疗期望送达」与「首批保障截止」
    两条限制，两条必须分别检查——旧版用一个 hard 变量顺序覆盖，医疗违约被首批
    截止吞掉。必需期限缺失（医疗箱无期望时刻、首批箱无截止时刻）按**输入错误**
    报出，不得当成"无限制"。重复箱号、孤儿架次引用、未知箱号/资源编号、
    机型与资源不符全部计入失败。
    """
    trips = _inp('q2_transport_trips.csv')
    deliv = _inp('q2_box_delivery.csv')
    boxes = {b['id']: b for b in d.cargo}
    uav_of = {u['id']: u for u in d.uavs}
    bad = []

    # 1) 能量裕度
    ratios = []
    n_unknown_type = 0
    for _, r in trips.iterrows():
        if r['机型编号'] not in d.transport_types:
            n_unknown_type += 1
            bad.append('架次 %s 的机型编号 %s 不在机型表中' % (r['架次编号'], r['机型编号']))
            continue
        t = d.transport_types[r['机型编号']]
        cap = (1 - t['rho']) * t['E_use']
        ratios.append(r['架次能耗kWh'] / cap)
        if r['架次能耗kWh'] > cap + 1e-9:
            bad.append('架次 %s 能耗超限 %.6f > %.6f' % (r['架次编号'], r['架次能耗kWh'], cap))
    ratios = np.array(ratios) if ratios else np.zeros(1)

    # 2) 实体无人机不重叠 + 编号/机型一致
    n_uav, n_uav_bad = 0, 0
    for uid, g in trips.groupby('无人机编号'):
        if uid not in uav_of:
            n_uav_bad += 1
            bad.append('无人机编号 %s 不在库存中' % uid)
            continue
        if set(g['机型编号']) != {uav_of[uid]['type']}:
            n_uav_bad += 1
            bad.append('无人机 %s 实际机型为 %s，表内却用于 %s 型架次'
                       % (uid, uav_of[uid]['type'], '、'.join(sorted(set(g['机型编号'])))))
        iv = sorted(zip(g['开始时刻s'], g['返回O01时刻s'], g['架次编号']))
        for a, b in zip(iv, iv[1:]):
            if b[0] < a[1] - TOL:
                n_uav += 1
                bad.append('无人机 %s: %s 与 %s 重叠' % (uid, a[2], b[2]))

    # 3) 共享电池周转：电池占用到充电完成为止，充电未完不得再起飞
    n_bat, n_bat_bad, gaps = 0, 0, []
    for bid, g in trips.groupby('电池编号'):
        gg = str(bid)[0]
        if gg not in d.batteries:
            n_bat_bad += 1
            bad.append('电池编号 %s 的机型前缀 %s 不在电池库存中' % (bid, gg))
            continue
        if set(g['机型编号']) != {gg}:
            n_bat_bad += 1
            bad.append('电池 %s 被 %s 型架次使用，与机型不符'
                       % (bid, '、'.join(sorted(set(g['机型编号'])))))
            continue
        iv = []
        for _, r in g.iterrows():
            soc = 1.0 - r['架次能耗kWh'] / d.transport_types[gg]['E_use']
            iv.append((r['开始时刻s'],
                       r['返回O01时刻s'] + charge_time(soc, d.batteries[gg][1]),
                       r['架次编号']))
        iv.sort()
        for a, b in zip(iv, iv[1:]):
            gaps.append(b[0] - a[1])
            if b[0] < a[1] - TOL:
                n_bat += 1
                bad.append('电池 %s: %s 与 %s 周转冲突' % (bid, a[2], b[2]))
    gaps = np.array(gaps)

    # 4) 物资时限：医疗期望与首批截止**分别**成立，缺失的必需期限按输入错误报出
    n_late = {'医疗期望': 0, '首批截止': 0}
    n_missing = 0
    lim_rows = []
    for _, r in deliv.iterrows():
        bid = r['货箱编号']
        if bid not in boxes:
            continue                       # 未知箱号在第 5 条统一报出
        b = boxes[bid]
        lims = []
        if b['category'] == '医疗物资':
            if b['expect'] is None:
                n_missing += 1
                bad.append('货箱 %s 为医疗物资但缺期望送达时刻（输入错误，不得视为无限制）' % bid)
            else:
                lims.append(('医疗期望', b['expect']))
        if b['first_batch']:
            if b['deadline'] is None:
                n_missing += 1
                bad.append('货箱 %s 为首批保障但缺首批截止时刻（输入错误，不得视为无限制）' % bid)
            else:
                lims.append(('首批截止', b['deadline']))
        for kind, lim in lims:
            t_del = float(r['交付完成时刻s'])
            lim_rows.append((bid, kind, lim, t_del))
            if t_del > lim + TOL:
                n_late[kind] += 1
                bad.append('货箱 %s %s 超限：交付 %.3f > 期限 %.3f' % (bid, kind, t_del, lim))

    # 5) 交付完整性：重复、漏交、未知箱号、孤儿架次引用
    dup = int(deliv['货箱编号'].duplicated().sum())
    if dup:
        bad.append('交付表有 %d 行重复箱号' % dup)
    unknown_box = sorted({x for x in deliv['货箱编号'] if x not in boxes})
    if unknown_box:
        bad.append('交付表出现 %d 个未知箱号：%s'
                   % (len(unknown_box), '、'.join(unknown_box[:5])))
    miss = sorted(set(boxes) - set(deliv['货箱编号']))
    if miss:
        bad.append('漏交付 %d 箱：%s%s' % (len(miss), '、'.join(miss[:5]),
                                        ' 等' if len(miss) > 5 else ''))
    orphan = sorted(set(deliv['架次编号']) - set(trips['架次编号']))
    if orphan:
        bad.append('交付表引用了 %d 个不存在于架次表的架次：%s'
                   % (len(orphan), '、'.join(orphan[:5])))
    empty_trips = sorted(set(trips['架次编号']) - set(deliv['架次编号']))

    # 6) 资源库存
    inv = {}
    for g in sorted(d.transport_types):
        sub = trips[trips['机型编号'] == g]
        n_own = len([u for u in d.uavs if u['type'] == g])
        inv[g] = (sub['无人机编号'].nunique(), n_own,
                  sub['电池编号'].nunique(), d.batteries[g][0])
        if inv[g][0] > n_own or inv[g][2] > d.batteries[g][0]:
            bad.append('%s 型资源超配：用机 %d/%d，用电池 %d/%d'
                       % ((g,) + inv[g]))

    metrics = [
        ('架次 / 交付记录 / 货箱总数', '%d / %d / %d' % (len(trips), len(deliv), len(boxes))),
        ('能量裕度占用比', '最大 %.3f  均值 %.3f  越限 %d'
         % (ratios.max(), ratios.mean(), int((ratios > 1).sum()))),
        ('无人机时段重叠 / 编号机型不符', '%d 处 / %d 处' % (n_uav, n_uav_bad)),
        ('电池周转冲突 / 编号机型不符', '%d 处（最小间隙 %.4f s） / %d 处'
         % (n_bat, gaps.min() if gaps.size else 0.0, n_bat_bad)),
        ('物资时限违约（医疗 / 首批）', '%d 箱 / %d 箱（检查 %d 条）'
         % (n_late['医疗期望'], n_late['首批截止'], len(lim_rows))),
        ('必需期限缺失', '%d 处（按输入错误计）' % n_missing),
        ('交付完整性', '重复 %d，未知箱号 %d，漏交 %d，孤儿架次 %d，空架次 %d'
         % (dup, len(unknown_box), len(miss), len(orphan), len(empty_trips))),
    ]
    for g in sorted(inv):
        a, b, c, e = inv[g]
        metrics.append(('%s 型资源（用机/库存，用电池/库存）' % g,
                        '%d/%d，%d/%d' % (a, b, c, e)))
    return rep('T9.1 问题二硬约束', not bad, bad, dict(metrics),
               '时刻 %.2f s；能量与时限按末位舍入容差' % TOL)


# ---------------------------------------------------------------------------
# 表 9.2  问题三中继方案
# ---------------------------------------------------------------------------
def check_relay(d):
    """表 9.2：问题三中继方案的硬约束独立复核。

    同样只读 q3_relay_trips.csv / q3_comm_phases.csv 两张结果表：悬停离地高度
    不采信结果表里记的数值，而是拿表里的经纬度回原始 DEM 重新取地面高程再算；
    能源组件的周转时刻由"返回时刻 + 充电时长"从能耗列反算，与求解器内部状态无关。
    """
    rt = _inp('q3_relay_trips.csv')
    cp = _inp('q3_comm_phases.csv')
    R = d.relay_type
    cap = (1 - R['rho']) * R['E_use']
    bad = []

    # 1) 能量裕度
    ratios = (rt['架次能耗kWh'] / cap).to_numpy()
    for _, r in rt.iterrows():
        if r['架次能耗kWh'] > cap + 1e-9:
            bad.append('中继架次 %s 能耗超限' % r['中继架次编号'])
    # 1b) 中继无人机与能源组件编号必须在册、机型一致
    n_uav_bad = 0
    uav_ids = {u['id'] for u in d.relay_uavs}
    for rid in sorted(set(rt['中继无人机编号'])):
        if rid not in uav_ids:
            n_uav_bad += 1
            bad.append('中继无人机编号 %s 不在册（在册：%s）'
                       % (rid, '、'.join(sorted(uav_ids))))

    # 2) 悬停离地高度 = 表内悬停海拔 − 该经纬度处的 DEM 地面高程。
    # 容差取 0.05 m，不是数值噪声而是结果表自身的精度：经纬度留 6 位（≈0.11 m）、
    # 海拔留 3 位（±5e-4 m）。方案按上限 300 m 设计（题给上限，越高覆盖越好），
    # 复核精度必须细于"贴限"的余量，否则会把合规方案误判为越限。
    HGT_TOL = 0.05
    agl = np.array([r['悬停海拔m'] - float(d.dem.sample(r['悬停经度'], r['悬停纬度']))
                    for _, r in rt.iterrows()])
    over_h = int((agl > R['max_hover_alt'] + HGT_TOL).sum())
    if over_h:
        bad.append('%d 个中继架次悬停高度超限' % over_h)

    # 3) 中继无人机：同一架的架次时段不得重叠
    n_uav = 0
    for rid, g in rt.groupby('中继无人机编号'):
        iv = sorted(zip(g['开始时刻s'], g['返回O01时刻s'], g['中继架次编号']))
        for a, b in zip(iv, iv[1:]):
            if b[0] < a[1] - TOL:
                n_uav += 1
                bad.append('中继 %s: %s 与 %s 重叠' % (rid, a[2], b[2]))

    # 4) 能源组件：占用持续到充电完成，充电未完不得再投入；编号须在册
    n_comp, n_comp_bad = 0, 0
    for cid, g in rt.groupby('能源组件编号'):
        if str(cid)[0] != 'R':
            n_comp_bad += 1
            bad.append('能源组件编号 %s 与中继机型不符' % cid)
        iv = []
        for _, r in g.iterrows():
            soc = 1.0 - r['架次能耗kWh'] / R['E_use']
            iv.append((r['开始时刻s'],
                       r['返回O01时刻s'] + charge_time(soc, d.relay_batteries[1]),
                       r['中继架次编号']))
        iv.sort()
        for a, b in zip(iv, iv[1:]):
            if b[0] < a[1] - TOL:
                n_comp += 1
                bad.append('能源组件 %s: %s 与 %s 周转冲突' % (cid, a[2], b[2]))
    n_comp_used = rt['能源组件编号'].nunique()
    if n_comp_used > d.relay_batteries[0]:
        bad.append('能源组件用件数 %d 超过库存 %d' % (n_comp_used, d.relay_batteries[0]))
    if rt['中继无人机编号'].nunique() > len(d.relay_uavs):
        bad.append('中继无人机用机数 %d 超过在册 %d'
                   % (rt['中继无人机编号'].nunique(), len(d.relay_uavs)))

    # 5) 结果表自洽：标为「中继」的时段必须真的落在某个架次的服务窗口内，
    #    且该架次编号可查；标为「中断」的不得又挂着架次编号。
    #    这一条正是为了防住"把中断记成中继"这类表内自相矛盾。
    wins = {r['中继架次编号']: (r['建链完成时刻s'], r['服务结束时刻s'])
            for _, r in rt.iterrows()
            if r['服务结束时刻s'] > r['建链完成时刻s'] + 1e-9}
    W_TOL = Q3_DT / 2.0 + 1e-6      # 见 Q3_DT 处的说明
    n_mis = 0
    over = 0.0
    for _, r in cp.iterrows():
        rid = r['中继架次编号']
        has_id = isinstance(rid, str) and rid.strip() != ''
        if r['保障方式'] == '中继':
            if not has_id or rid not in wins:
                n_mis += 1
            else:
                # 左端点是段的起点，采样点本身必在窗口内，按采样精度卡；
                # 右端点是"到下一个采样时刻的中点"，允许半个步长。
                d_left = wins[rid][0] - r['开始时刻s']
                d_right = r['结束时刻s'] - wins[rid][1]
                over = max(over, d_left, d_right)
                if d_left > 1e-6 or d_right > W_TOL:
                    n_mis += 1
        elif r['保障方式'] == '中断' and has_id:
            n_mis += 1
    if n_mis:
        bad.append('通信保障表 %d 段的标注与其架次服务窗口不符' % n_mis)

    # 三段占比不在这里复算：本表按"段"记录，段首尾取点与逐时刻统计存在半个采样
    # 步长的口径差，照段复算会与逐时刻数字差约 0.5 个百分点，两个都对却看着像
    # 矛盾。占比的时间积分口径由下面的 check_relay_final 独立重采样复算（表 9.3），
    # 那里才是"中断 = 0"的复算处；本表只核资源约束与表内自洽。
    metrics = [
        ('中继架次 / 通信保障分段', '%d / %d' % (len(rt), len(cp))),
        ('能量裕度占用比', '最大 %.3f（可用 %.2f kWh/架次）  越限 %d'
         % (ratios.max(), cap, int((ratios > 1).sum()))),
        ('悬停离地高度', '最大 %.1f m（上限 %.0f m）  越限 %d'
         % (agl.max(), R['max_hover_alt'], over_h)),
        ('中继无人机', '用机 %d/%d，编号不符 %d，时段重叠 %d'
         % (rt['中继无人机编号'].nunique(), len(d.relay_uavs), n_uav_bad, n_uav)),
        ('能源组件', '用件 %d/%d，编号不符 %d，周转冲突 %d'
         % (n_comp_used, d.relay_batteries[0], n_comp_bad, n_comp)),
        ('结果表标注不符', '%d 段' % n_mis),
        ('段边界越出服务窗口（最大）', '%.3f s（容差 %.2f = Δt/2，Δt=%.0f s）'
         % (over, W_TOL, Q3_DT)),
    ]
    return rep('T9.2 问题三中继方案', not bad, bad, dict(metrics),
               '时刻 %.2f s；悬停离地高度 %.2f m；段右端 Δt/2' % (TOL, HGT_TOL))


# ---------------------------------------------------------------------------
# 表 9.3c 问题三运输侧资源链（错峰后）
# ---------------------------------------------------------------------------
def check_q3_transport(d):
    """表 9.3c：**错峰之后**运输排班的资源链独立复核。

    为什么必须单独有这一张表：表 9.1 复核的是问题二自己的排班（读
    `q2_transport_trips.csv`），表 9.2 复核的是中继侧资源。第三问把运输架次的
    开始时刻整体后移了几千秒，而**后移之后**的无人机链与共享电池链此前没有任何
    复核覆盖——终端检查（表 9.3）只判通信分段，看不见运输资源。实测正是这个缺口
    放过了一个不可行解：U05 同时飞 T17 与 T24，重叠 1583 s，而全部既有检查都是
    「通过」。求解器侧已按硬约束拒绝它（`q3.solve_relay` 第 8 步），本表是与之
    **相互独立**的第二道关：全部数值由结果表自行反算，不调用 q3.py 的任何函数。

    只读 `q3_transport_trips.csv`（错峰后时刻与资源编号）、`q3_stagger.csv`
    （推迟量）、`q2_transport_trips.csv`（错峰前对照）。共享电池的释放时刻按
    「返航 + 按本架次能耗反算的 SOC 所需充电时长」重建，容差与表 9.1 同一口径。
    """
    trips = _inp('q3_transport_trips.csv')
    sg = _inp('q3_stagger.csv')
    q2t = _inp('q2_transport_trips.csv')
    uav_of = {u['id']: u for u in d.uavs}
    bad = []

    # 0) 架次集合必须与问题二一致：第三问只改时刻，不得增删架次、不得换机换型
    ids3 = list(trips['架次编号'])
    ids2 = list(q2t['架次编号'])
    n_set = 0
    if sorted(ids3) != sorted(ids2):
        n_set += 1
        bad.append('错峰后的架次集合与问题二不一致：多 %s、少 %s'
                   % (sorted(set(ids3) - set(ids2)), sorted(set(ids2) - set(ids3))))
    if len(set(ids3)) != len(ids3):
        n_set += 1
        bad.append('错峰后运输表存在重复架次编号')

    # 1) 实体无人机：在册、机型相符、同一架的架次时段不重叠
    n_uav, n_uav_bad = 0, 0
    for uid, g in trips.groupby('无人机编号'):
        if uid not in uav_of:
            n_uav_bad += 1
            bad.append('无人机编号 %s 不在库存中' % uid)
            continue
        if set(g['机型编号']) != {uav_of[uid]['type']}:
            n_uav_bad += 1
            bad.append('无人机 %s 实际机型为 %s，错峰后表内却用于 %s 型架次'
                       % (uid, uav_of[uid]['type'], '、'.join(sorted(set(g['机型编号'])))))
        iv = sorted(zip(g['开始时刻s'], g['返回O01时刻s'], g['架次编号']))
        for a, b in zip(iv, iv[1:]):
            if b[0] < a[1] - TOL:
                n_uav += 1
                bad.append('无人机 %s: %s 与 %s 时段重叠 %.3f s'
                           % (uid, a[2], b[2], a[1] - b[0]))

    # 2) 共享电池：编号与机型相符、周转（返航 + 充电）不冲突
    n_bat, n_bat_bad, gaps = 0, 0, []
    for bid, g in trips.groupby('电池编号'):
        gg = str(bid)[0]
        if gg not in d.batteries:
            n_bat_bad += 1
            bad.append('电池编号 %s 的机型前缀 %s 不在电池库存中' % (bid, gg))
            continue
        if set(g['机型编号']) != {gg}:
            n_bat_bad += 1
            bad.append('电池 %s 被 %s 型架次使用，与机型不符'
                       % (bid, '、'.join(sorted(set(g['机型编号'])))))
            continue
        iv = []
        for _, r in g.iterrows():
            soc = 1.0 - r['架次能耗kWh'] / d.transport_types[gg]['E_use']
            iv.append((r['开始时刻s'],
                       r['返回O01时刻s'] + charge_time(soc, d.batteries[gg][1]),
                       r['架次编号']))
        iv.sort()
        for a, b in zip(iv, iv[1:]):
            gaps.append(b[0] - a[1])
            if b[0] < a[1] - TOL:
                n_bat += 1
                bad.append('电池 %s: %s 与 %s 周转冲突 %.3f s'
                           % (bid, a[2], b[2], a[1] - b[0]))
    gaps = np.array(gaps) if gaps else np.zeros(1)

    # 3) 错峰表与运输表逐架次自洽：原开始 ≡ 问题二的开始时刻、推迟 ≡ 两者之差，
    #    且推迟量非负（第三问的设计只允许向后让，不允许把架次提前到资源就绪之前）。
    n_mis, n_neg, max_delay = 0, 0, 0.0
    q2start = {r['架次编号']: float(r['开始时刻s']) for _, r in q2t.iterrows()}
    s3 = {r['架次编号']: float(r['开始时刻s']) for _, r in trips.iterrows()}
    for _, r in sg.iterrows():
        tid = r['架次编号']
        if tid not in q2start or tid not in s3:
            n_mis += 1
            bad.append('错峰表架次 %s 在运输表中查无对应' % tid)
            continue
        if abs(r['原开始时刻s'] - q2start[tid]) > TOL:
            n_mis += 1
            bad.append('%s 的「原开始时刻」%.3f 与问题二开始时刻 %.3f 不符'
                       % (tid, r['原开始时刻s'], q2start[tid]))
        if abs(r['错峰开始时刻s'] - s3[tid]) > TOL:
            n_mis += 1
            bad.append('%s 的「错峰开始时刻」%.3f 与运输表开始时刻 %.3f 不符'
                       % (tid, r['错峰开始时刻s'], s3[tid]))
        if abs((r['错峰开始时刻s'] - r['原开始时刻s']) - r['推迟s']) > TOL:
            n_mis += 1
            bad.append('%s 的推迟量 %.3f 与前后两列之差不符' % (tid, r['推迟s']))
        if r['推迟s'] < -TOL:
            n_neg += 1
            bad.append('%s 被提前了 %.3f s（第三问只允许向后推迟）' % (tid, -r['推迟s']))
        max_delay = max(max_delay, float(r['推迟s']))
    if len(set(sg['架次编号'])) != len(sg):
        bad.append('错峰表存在重复架次编号')

    metrics = [
        ('架次集合与问题二一致', '%d 架次，不一致 %d 处' % (len(ids3), n_set)),
        ('无人机时段重叠 / 编号机型不符', '%d 处 / %d 处' % (n_uav, n_uav_bad)),
        ('共享电池周转冲突 / 编号机型不符', '%d 处（最小间隙 %.4f s） / %d 处'
         % (n_bat, gaps.min(), n_bat_bad)),
        ('错峰表与运输表不符', '%d 处' % n_mis),
        ('被提前的架次', '%d 个' % n_neg),
        ('最大推迟量', '%.1f s（共 %d 个架次被推迟）'
         % (max_delay, int((sg['推迟s'] > TOL).sum()))),
    ]
    return rep('T9.3c 问题三运输侧资源链', not bad, bad, dict(metrics),
               '时刻 %.2f s（结果表留 %d 位小数，反算含充电时长）' % (TOL, T_DEC))


# ---------------------------------------------------------------------------
# 表 9.3  通信连续性与交付侧指标
# ---------------------------------------------------------------------------
def _q3_trips(d):
    """从 q2 结果表 + 错峰表重建**错峰后**的运输架次（路线、各区箱数、开始时刻）。"""
    trips = _inp('q2_transport_trips.csv')
    deliv = _inp('q2_box_delivery.csv')
    sg = _inp('q3_stagger.csv')
    shift = {r['架次编号']: float(r['推迟s']) for _, r in sg.iterrows()}
    out = []
    for _, r in trips.iterrows():
        tid = r['架次编号']
        route = [s for s in str(r['访问服务区顺序']).split('->') if s]
        sub = deliv[deliv['架次编号'] == tid]
        boxes_at = {sid: list(sub[sub['服务区编号'] == sid]['货箱编号']) for sid in route}
        out.append(dict(id=tid, type=r['机型编号'], route=route, boxes_at=boxes_at,
                        start=float(r['开始时刻s']) + shift.get(tid, 0.0),
                        dur=float(r['返回O01时刻s']) - float(r['开始时刻s'])))
    return out


def _q3_traj(d, a, dt):
    """独立重建架次三维轨迹：只用 core 的 segment_geometry，不调用 q3.py 的采样器。

    运动学与 q3.sample_trip_trajectory 同源（爬升垂直、巡航水平、下降垂直，
    投送期间定点悬停首尾各记一点），但这里是**另写一遍**——若只 import 求解器
    的采样器，采样这一层就失去了独立性，而"轨迹在哪里断了链路"正是本项要查的。
    """
    t = d.transport_types[a['type']]
    alt_o = d.O01['alt']

    def ll(nid):
        return (d.O01['lon'], d.O01['lat']) if nid == 'O01' else (d.si[nid]['lon'], d.si[nid]['lat'])

    def alt_of(nid):
        return alt_o if nid == 'O01' else d.si[nid]['alt'] + 30.0

    pts = []
    tc0 = a['start']
    pts.append((tc0, ll('O01')[0], ll('O01')[1], alt_o))
    tc0 += t['prep'] + t['load_per_box'] * sum(len(v) for v in a['boxes_at'].values())
    pts.append((tc0, ll('O01')[0], ll('O01')[1], alt_o))
    prev = 'O01'
    for sid in list(a['route']) + ['O01']:
        la, lb = ll(prev), ll(sid)
        za, zb = alt_of(prev), alt_of(sid)
        g = segment_geometry(d.dem, la[0], la[1], za, lb[0], lb[1], zb)
        for span, kind in ((g['climb'] / t['v_up'], 'up'),
                           (g['d'] / t['v_cruise'], 'cr'),
                           (g['descent'] / t['v_down'], 'dn')):
            n = max(2, int(np.ceil(span / dt)))
            for k in range(1, n + 1):
                fr = k / n
                if kind == 'up':
                    pts.append((tc0 + span * fr, la[0], la[1], za + (g['cruise_alt'] - za) * fr))
                elif kind == 'cr':
                    pts.append((tc0 + span * fr, la[0] + (lb[0] - la[0]) * fr,
                                la[1] + (lb[1] - la[1]) * fr, g['cruise_alt']))
                else:
                    pts.append((tc0 + span * fr, lb[0], lb[1],
                                g['cruise_alt'] - (g['cruise_alt'] - zb) * fr))
            tc0 += span
        if sid != 'O01':
            h = t['hand_base'] + t['hand_per_box'] * len(a['boxes_at'][sid])
            pts.append((tc0, lb[0], lb[1], zb))          # 悬停首点
            pts.append((tc0 + h, lb[0], lb[1], zb))      # 悬停末点，积分不留空洞
            tc0 += h
        prev = sid
    return pts


def _time_weights(ts):
    """时间占比口径：w_i = (t_{i+1} − t_{i−1})/2（端点取半格）。"""
    n = len(ts)
    ts = np.asarray(ts, dtype=float)
    if n == 1:
        return np.zeros(1)
    w = np.empty(n)
    w[0] = 0.5 * (ts[1] - ts[0])
    w[-1] = 0.5 * (ts[-1] - ts[-2])
    if n > 2:
        w[1:-1] = 0.5 * (ts[2:] - ts[:-2])
    return w


def interval_binding_errors(iv, win):
    """
    失效区间 ↔ 中继架次的绑定校验：每条区间都必须绑定到**此刻在役**的中继架次。

    `win` = {中继架次编号: (服务开始, 服务结束)}。返回错误说明列表，空表示通过。
    只绑了编号、但区间时段落在该架次服务窗口之外，与绑了个不存在的编号同样算失败。
    """
    errs = []
    for _, r in iv.iterrows():
        rid = r['中继架次编号']
        if not (isinstance(rid, str) and rid.strip()):
            errs.append('失效区间 %s 无中继架次绑定' % r['失效区间编号'])
            continue
        if rid not in win:
            errs.append('失效区间 %s 绑定了不存在的中继架次 %s' % (r['失效区间编号'], rid))
            continue
        if not (win[rid][0] - 1e-6 <= r['区间起点s'] and r['区间终点s'] <= win[rid][1] + 1e-6):
            errs.append('失效区间 %s 的时段 [%.3f, %.3f] 超出 %s 的服务窗口 [%.3f, %.3f]'
                        % (r['失效区间编号'], r['区间起点s'], r['区间终点s'],
                           rid, win[rid][0], win[rid][1]))
    return errs


def phase_boundary_errors(cp, span, tol=None):
    """
    分段表连续性（F13）：同一架次内后一段的起点必须逐位等于前一段的终点，
    首段起点、末段终点必须等于该架次的起止时刻，且不得有「中断」段。

    判据只看表里写出的边界本身，**不含任何采样步长**——旧版按「间隙 ≤ 采样步长」
    判，把 dt 从 2 s 改到 1 s 结论就跟着变，那样的校验立不住。返回
    `{n_break, n_span, n_gap_seg, errors}`：0.2 s 的空洞是 0.2 s，与 dt 无关。
    """
    tol = TOL if tol is None else tol
    errs, n_break, n_span, n_gap_seg = [], 0, 0, 0
    for tid, g in cp.groupby('运输架次编号'):
        g = g.sort_values('开始时刻s')
        if tid not in span:
            n_break += 1
            errs.append('分段表出现架次表里没有的架次 %s' % tid)
            continue
        rows = list(g.iterrows())
        for i, (_, r) in enumerate(rows):
            if r['通信阶段'] == '中断':
                n_gap_seg += 1
            if i + 1 < len(rows):
                if abs(float(rows[i + 1][1]['开始时刻s']) - float(r['结束时刻s'])) > 1e-6:
                    n_break += 1
        if abs(float(rows[0][1]['开始时刻s']) - span[tid][0]) > tol or \
                abs(float(rows[-1][1]['结束时刻s']) - span[tid][1]) > tol:
            n_span += 1
    if n_break:
        errs.append('通信保障表有 %d 处相邻段首尾不相接（存在未标注的空洞或重叠）' % n_break)
    if n_span:
        errs.append('通信保障表有 %d 个架次的分段未铺满该架次起止时刻' % n_span)
    if n_gap_seg:
        errs.append('通信保障表仍有 %d 段标注为「中断」' % n_gap_seg)
    return dict(n_break=n_break, n_span=n_span, n_gap_seg=n_gap_seg, errors=errs)


def check_relay_final(d, dt=1.0, frac_tol=0.01):
    """表 9.3：问题三**终点方案**的独立复核（对应 3.5/3.6/3.7 三项要求）。

    与 check_relay 的分工：那张表查的是"结果表内部自洽 + 资源硬约束"，不重算
    通信；这张表按 Δt=1 s **逐时刻重判通信状态**，是"中断 = 0"这个结论的独立
    复算。链路裕量直接用 core.CommModel.link_ok 重算，不调用 q3.py 里的
    direct_margin / access_margin / station_covers，也不调用它的终检 audit。

    三项要求逐条落地：
      3.5 无中继架次绑定的中继区间 = 0：q3_intervals.csv 每一行都必须有可查的
          中继架次编号，且该区间的时段必须落在这个架次的服务窗口内；
      3.6 占比按时间积分（不是采样点个数），并与 q3_coverage.csv 对照；
      3.7 逐时刻判：先判直连；失败则须存在一架**此刻正在服务**的中继，其所在
          悬停站对该点接入裕量 ≥ 0 **且该站回传 G01 的裕量 ≥ 0**；两者皆不成立
          即为中断，必须为 0。回传这一半曾经漏检：只查接入时，把 Lmax_r_gw
          改成 -999 dB（回传物理上不可能）重跑，本检查照样通过。
    分段表（F13）：段边界是共同边界序列，相邻两段必须**严格首尾相接**
    （前段终点 == 后段起点），首段起点与末段终点必须等于该架次的起止时刻，
    不再用"间隙 ≤ 采样步长"这种依赖 dt 的松判据。
    """
    trips = _q3_trips(d)
    rt = _inp('q3_relay_trips.csv')
    stn = _inp('q3_stations.csv')
    iv = _inp('q3_intervals.csv')
    cp = _inp('q3_comm_phases.csv')
    cov = _inp('q3_coverage.csv')
    tt = _inp('q3_transport_trips.csv')
    bad = []

    # ---- 3.5 中继区间必须绑定到真实存在、且此刻在役的中继架次 ----
    win = {r['中继架次编号']: (float(r['建链完成时刻s']), float(r['服务结束时刻s']))
           for _, r in rt.iterrows()}
    unbound = interval_binding_errors(iv, win)
    n_unbound = len(unbound)
    bad.extend(unbound)
    # 3.5b 错峰表与架次表必须对得上（错峰后的方案只有一个真相源）
    n_shift_mis = 0
    if '原开始时刻s' in tt.columns and '开始时刻s' in tt.columns:
        for _, r in tt.iterrows():
            shift = float(r['开始时刻s']) - float(r['原开始时刻s'])
            got = [x for x in trips if x['id'] == r['架次编号']]
            if not got:
                n_shift_mis += 1
                continue
            if abs((got[0]['start'] - float(r['原开始时刻s'])) - shift) > TOL_T * 4:
                n_shift_mis += 1
    if n_shift_mis:
        bad.append('错峰后的架次表与 q2 起始时刻 + q3_stagger.csv 不一致：%d 行' % n_shift_mis)

    # ---- 3.7 逐时刻重判 + 3.6 时间占比 ----
    gw = (d.O01['lon'], d.O01['lat'], d.O01['alt'] + d.comm_params['gw_h'])
    Lmax_d, Lmax_a = d.comm.Lmax_uw_gw, d.comm.Lmax_uw_r
    Lmax_r = d.comm.Lmax_r_gw
    # 回传只与站址有关，与运输机位置无关，故每个悬停站只判一次；不通过的中继
    # 架次既不计入覆盖，也单独报错——它是「中继在役但传不回 G01」这种假覆盖。
    bh_cache = {}

    def _backhaul_ok(st):
        if st not in bh_cache:
            bh_cache[st] = d.comm.link_ok(d.dem, st, gw, Lmax_r)[1] <= Lmax_r + 1e-9
        return bh_cache[st]

    site = {r['悬停站编号']: (r['悬停经度'], r['悬停纬度'], r['悬停海拔m'])
            for _, r in stn.iterrows()}
    wins = []
    n_no_bh = 0
    for _, r in rt.iterrows():
        if r['悬停站编号'] not in site:
            bad.append('中继架次 %s 引用了不存在的悬停站 %s'
                       % (r['中继架次编号'], r['悬停站编号']))
            continue
        s = site[r['悬停站编号']]
        if float(r['服务结束时刻s']) > float(r['建链完成时刻s']) + 1e-9:
            ok_bh = _backhaul_ok(s)
            if not ok_bh:
                n_no_bh += 1
                bad.append('中继架次 %s 的悬停站 %s 回传 G01 裕量 < 0（%.2f dB），'
                           '该架次不能算作有效覆盖'
                           % (r['中继架次编号'], r['悬停站编号'],
                              d.comm.Lmax_r_gw -
                              d.comm.link_ok(d.dem, s, gw, Lmax_r)[1]))
            wins.append((float(r['建链完成时刻s']), float(r['服务结束时刻s']), s,
                         r['中继架次编号'], ok_bh))

    tot = dict(direct=0.0, relay=0.0, gap=0.0)
    gaps = []
    for a in trips:
        pts = _q3_traj(d, a, dt)
        w = _time_weights([p[0] for p in pts])
        for k, (ttt, lon, lat, alt) in enumerate(pts):
            if d.comm.link_ok(d.dem, (lon, lat, alt), gw, Lmax_d)[1] <= Lmax_d + 1e-9:
                tot['direct'] += w[k]
                continue
            hit = None
            for (w1, w2, s, rid, ok_bh) in wins:
                if ok_bh and w1 - 1e-6 <= ttt <= w2 + 1e-6 and \
                        d.comm.link_ok(d.dem, (lon, lat, alt), s, Lmax_a)[1] <= Lmax_a + 1e-9:
                    hit = rid
                    break
            if hit:
                tot['relay'] += w[k]
            else:
                tot['gap'] += w[k]
                gaps.append((a['id'], ttt, lon, lat, alt))
    T = sum(tot.values())
    frac = {k: (v / T if T > 0 else 0.0) for k, v in tot.items()}
    if tot['gap'] > 1e-9:
        bad.append('%d 个采样点处于中断，累计 %.1f s' % (len(gaps), tot['gap']))

    # ---- 3.6 与求解器自报的时间占比对照 ----
    row = cov[cov['口径'] == '联合优化']
    claim = None
    if len(row):
        claim = row.iloc[0]
        for k, col in (('direct', '直连时间占比'), ('relay', '中继时间占比'),
                       ('gap', '中断时间占比')):
            if abs(frac[k] - float(claim[col])) > frac_tol:
                bad.append('时间占比与 q3_coverage.csv 不符：%s 复核 %.4f vs 自报 %.4f'
                           % (k, frac[k], float(claim[col])))
    else:
        bad.append('q3_coverage.csv 缺少「联合优化」口径行，无法对照占比')

    # ---- 3.6 分段表：共同边界、首尾对齐、无「中断」段 ----
    span = {r['架次编号']: (float(r['开始时刻s']), float(r['返回O01时刻s']))
            for _, r in tt.iterrows()}
    pb = phase_boundary_errors(cp, span)
    bad.extend(pb['errors'])
    if set(cp['运输架次编号']) != {a['id'] for a in trips}:
        bad.append('通信保障表未覆盖全部运输架次')
    n_break, n_span, n_gap_seg = pb['n_break'], pb['n_span'], pb['n_gap_seg']

    metrics = [
        ('失效区间 / 中继架次 / 通信分段', '%d / %d / %d' % (len(iv), len(rt), len(cp))),
        ('3.5 无中继架次绑定的区间', '%d 个（要求 0）' % n_unbound),
        ('3.5b 错峰表与架次表不符', '%d 行' % n_shift_mis),
        ('3.7 回传 G01 不可用的在役中继', '%d 个（要求 0，判据 Lmax_r_gw = %.2f dB）'
         % (n_no_bh, Lmax_r)),
        ('3.6 时间占比（Δt=%.1fs，总时长 %.0f s）' % (dt, T),
         '直连 %.4f | 中继 %.4f | 中断 %.4f' % (frac['direct'], frac['relay'], frac['gap'])),
        ('3.6 分段连续性', '首尾不相接 %d 处，未铺满 %d 个架次，标为「中断」%d 段（均要求 0）'
         % (n_break, n_span, n_gap_seg)),
        ('3.7 逐时刻重判的中断点', '%d 个（要求 0）' % len(gaps)),
    ]
    if claim is not None:
        metrics.insert(4, ('3.6 与自报占比最大偏差',
                           '%.4f（容差 %.2f）'
                           % (max(abs(frac[k] - float(claim[c])) for k, c in
                                  (('direct', '直连时间占比'), ('relay', '中继时间占比'),
                                   ('gap', '中断时间占比'))), frac_tol)))
    for g in gaps[:5]:
        bad.append('中断点 %s t=%.1f s (%.5f, %.5f, %.1f m)' % g)
    return rep('T9.3 通信连续性（Δt=%.0f s 逐时刻重判）' % dt, not bad, bad, dict(metrics),
               '接入与回传同时 ≥ 0 dB；分段边界逐位相接；占比容差 %.2f；中断点 0 个' % frac_tol)


def _stirling2(n, k):
    """第二类 Stirling 数 S(n,k)：把 n 个可区分元素分成 k 个非空无标号组的方法数。"""
    prev = [1] + [0] * k
    for _ in range(n):
        cur = [0] * (k + 1)
        for j in range(1, k + 1):
            cur[j] = prev[j - 1] + j * prev[j]
        prev = cur
    return prev[k]


def check_q4(d):
    """表 9.4：问题四任务分区与资源配置的独立复核。

    只读 q4 的结果表重算，**不重跑分区枚举**（那是 q4.py 的全部代价）：枚举
    规模本身改用 Stirling 数核对，其余四项全部从交付表重算——
      1. 分区合法性：15 个服务区不重不漏；每组非空；原子单元不被拆开；且含有
         失效区间所在服务区的组至少配 1 架中继无人机（由 q3_intervals.csv 与
         q3_transport_trips.csv 反查失效服务区，不依赖 q4.py 的中继核算）；
      2. 资源规模 R：由 q4_partition.csv 逐组逐类需求相加，与 q4_comparison.csv 对照；
      3. 单份库存口径：N[r]=Σ_g n[g,r] 与全队**单份**库存比，Gap=max(0,N−S)，
         并与「最小需求」列自洽核对（最小需求 ≤ 推荐需求且最小缺口 = max(0, 最小需求 − 库存)）；
      4. 峰值需求可实现：q4_resource_allocation.csv 的着色分配中，每个
         (K, 组, 资源类型) 用到的不同资源编号数必须等于该组峰值需求，且同一
         编号不得跨组复用——否则「峰值需求」只是个没落到实处的计数。

    **本项测不到的**：「全部分区中的最小需求」那两列是 q4.py 枚举 364 个分区
    得到的结果，本检查只核对它与推荐需求、库存自洽（最小需求 ≤ 推荐需求，
    最小缺口 = max(0, 最小需求 − 库存)），**不重跑枚举**——重跑要用 q4 的分组
    层，那样就不再是独立复核。该列的真值由 tests_regression.py 的
    T-Q4m 用独立枚举核对。论文引用「最小需求」时只能称「与推荐分区自洽且经
    枚举复核」，不得称「由 verify.py 独立重算」。

    资源类别列名直接取自 q4_partition.csv 表头，不在此处复制一份标签表，
    以免求解器改了标签而复核仍按旧名对照。
    """
    part = _inp('q4_partition.csv')
    units = _inp('q4_units.csv')
    cmp_ = _inp('q4_comparison.csv')
    allp = _inp('q4_all_partitions.csv')
    alloc = _inp('q4_resource_allocation.csv')
    bad = []
    labels = [c for c in part.columns
              if c not in ('K', '任务组编号', '服务区列表', '工作量h')]
    stock = {f'{g}_uav': sum(1 for u in d.uavs if u['type'] == g) for g in 'ABC'}
    stock.update({f'{g}_bat': int(d.batteries[g][0]) for g in 'ABC'})
    stock['relay'] = len(d.relay_uavs)
    stock['relay_comp'] = int(d.relay_batteries[0])
    lab2key = {'A型运输无人机': 'A_uav', 'B型运输无人机': 'B_uav', 'C型运输无人机': 'C_uav',
               'A型共享电池': 'A_bat', 'B型共享电池': 'B_bat', 'C型共享电池': 'C_bat',
               '中继无人机': 'relay', '中继能源组件': 'relay_comp'}
    miss = [c for c in labels if c not in lab2key]
    if miss:
        bad.append('q4_partition.csv 出现无法对应库存的资源类别列：%s' % '、'.join(miss))
    keys = [lab2key[c] for c in labels if c in lab2key]

    # 「哪些服务区真的会失去视线」——用于下面那条**必要**条件：一个任务组只要
    # 含有失效区间所在的服务区，就必须至少配 1 架中继无人机。只用交付表
    # （q3_intervals.csv 的架次编号 + q3_transport_trips.csv 的路线）反查，
    # 不碰 q4.py 的任何计算，属独立复算。
    iv = _inp('q3_intervals.csv')
    tt = _inp('q3_transport_trips.csv')
    route_of = {r['架次编号']: [s for s in str(r['访问服务区顺序']).split('->') if s]
                for _, r in tt.iterrows()}
    iv_si, iv_unk = set(), set()
    for tid in set(iv['运输架次编号']):
        if tid not in route_of:
            iv_unk.add(tid)
            continue
        iv_si.update(route_of[tid])
    if iv_unk:
        bad.append('q3_intervals.csv 引用了 q3_transport_trips.csv 中不存在的架次：%s'
                   % '、'.join(sorted(iv_unk)))
    n_iv_grp = {}

    # ---- 1. 原子单元与分区合法性 ----
    all_si = set(d.S)
    unit_of = {}
    for _, r in units.iterrows():
        ss = [x for x in str(r['服务区列表']).split(',') if x]
        for s in ss:
            if s in unit_of:
                bad.append('服务区 %s 同时出现在单元 %s 与 %s' % (s, unit_of[s], r['单元编号']))
            unit_of[s] = r['单元编号']
    if set(unit_of) != all_si:
        bad.append('原子单元未覆盖全部服务区：多 %s，缺 %s'
                   % (sorted(set(unit_of) - all_si), sorted(all_si - set(unit_of))))
    n_split = 0
    for K, g in part.groupby('K'):
        seen = []
        for _, r in g.iterrows():
            seen.extend(x for x in str(r['服务区列表']).split(',') if x)
        if len(seen) != len(set(seen)):
            bad.append('K=%d 的分区里有服务区被重复分配' % K)
        if set(seen) != all_si:
            bad.append('K=%d 的分区未覆盖全部服务区' % K)
        if list(g['任务组编号']) != list(range(1, len(g) + 1)):
            bad.append('K=%d 的任务组编号不连续' % K)
        # 单元不可拆：同一单元的服务区必须落在同一组
        grp_of_si = {s: r['任务组编号'] for _, r in g.iterrows()
                     for s in str(r['服务区列表']).split(',') if s}
        for u, us in units.set_index('单元编号')['服务区列表'].items():
            gs = {grp_of_si.get(s) for s in str(us).split(',') if s}
            if len(gs) > 1:
                n_split += 1
                bad.append('K=%d：原子单元 %s 被拆到 %s 组（该单元的共享架次会失效）'
                           % (K, u, sorted(gs)))
        # 组级必要条件：含有失效区间所在服务区的组，必须至少配 1 架中继无人机。
        # 这一条取代了旧版「中继最小总需求 ≥ K」。旧版是一条**先验**（“每组自备
        # 护航”），本算例已用交付数据证伪：S006 的三个架次（T03/T06/T15）全程直连、
        # 在 q3_intervals.csv 里没有失效区间，它单独成组时中继需求为 0，因此
        # K=3 的最小总需求 2 < K=3 是真实值、不是漏算。先验不能当判据——组级核对
        # 比拿总数与 K 比更细，也更强。
        cnt_iv = 0
        for _, r in g.iterrows():
            gs = {s for s in str(r['服务区列表']).split(',') if s}
            if gs & iv_si:
                cnt_iv += 1
                if int(r['中继无人机']) < 1:
                    bad.append('K=%d 组%s 含失效区间所在的服务区，中继无人机需求却为 0'
                               % (K, r['任务组编号']))
        n_iv_grp[K] = cnt_iv

    # ---- 2/3. 资源规模与单份库存缺口 ----
    maxdR = maxdGap = 0.0
    for K, g in part.groupby('K'):
        row = cmp_[cmp_['K'] == K]
        if not len(row):
            bad.append('q4_comparison.csv 缺 K=%d 行' % K)
            continue
        row = row.iloc[0]
        tot = {k: int(g[lab].sum()) for lab, k in zip(labels, keys)}
        R = sum(tot.values())
        if row['资源规模R'] != R:
            bad.append('K=%d 的资源规模 R：复核 %d vs 自报 %s' % (K, R, row['资源规模R']))
        maxdR = max(maxdR, abs(R - float(row['资源规模R'])))
        for lab, k in zip(labels, keys):
            gap = max(0, tot[k] - stock[k])
            if row['总量缺口_' + lab] != gap:
                bad.append('K=%d %s 的单份库存缺口：复核 %d vs 自报 %s'
                           % (K, lab, gap, row['总量缺口_' + lab]))
            maxdGap = max(maxdGap, abs(gap - float(row['总量缺口_' + lab])))
            if row['需求_' + lab] != tot[k]:
                bad.append('K=%d %s 的推荐需求：复核 %d vs 自报 %s'
                           % (K, lab, tot[k], row['需求_' + lab]))
            # 「最小需求」必须是全部分区上的下确界：不得大于推荐分区的需求，
            # 且其缺口与库存自洽。这是全称断言「任何分区都缺」的唯一数据依据，
            # 核对它才不至于让一个手写的下界蒙混过关。
            mn = int(row['最小需求_' + lab])
            if mn > tot[k]:
                bad.append('K=%d %s 的最小需求 %d 大于推荐分区需求 %d（不是下确界）'
                           % (K, lab, mn, tot[k]))
            if row['最小缺口_' + lab] != max(0, mn - stock[k]):
                bad.append('K=%d %s 的最小缺口 %s 与 min(0, 最小需求−库存) 不符'
                           % (K, lab, row['最小缺口_' + lab]))
        # 全队量级的必要条件：只要有失效区间，中继无人机的最小总需求就 ≥ 1
        # （旧版拿它跟组数 K 比，见上面组级核对处的说明，那条先验已被证伪）。
        if len(iv) and '中继总需求最小' in row and int(row['中继总需求最小']) < 1:
            bad.append('K=%d 有 %d 条失效区间，中继无人机的最小总需求却报 %s'
                       % (K, len(iv), row['中继总需求最小']))

    # ---- 4. 峰值需求必须由着色分配落实 ----
    n_color_bad = 0
    for (K, gi, lab), g in alloc.groupby(['K', '任务组编号', '资源类型']):
        row = part[(part['K'] == K) & (part['任务组编号'] == gi)]
        if not len(row) or lab not in part.columns:
            bad.append('着色分配出现未知的 (K=%s, 组%s, %s)' % (K, gi, lab))
            continue
        need = int(row.iloc[0][lab])
        used = sorted(set(g['资源编号']))
        if len(used) != need:
            n_color_bad += 1
            bad.append('K=%s 组%s 的 %s：着色用掉 %d 个资源，峰值需求却报 %d'
                       % (K, gi, lab, len(used), need))
    # 同一资源编号不得跨组复用（题面「资源不得跨组调配」）
    for (K, lab, rid), g in alloc.groupby(['K', '资源类型', '资源编号']):
        if g['任务组编号'].nunique() > 1:
            n_color_bad += 1
            bad.append('K=%s 的资源 %s「%s」被 %s 个组共用'
                       % (K, lab, rid, g['任务组编号'].nunique()))

    # ---- 5. 枚举规模与可行性计数 ----
    n_units = len(units)
    n_wrong = 0
    for K in sorted(allp['K'].unique()):
        got = int((allp['K'] == K).sum())
        theo = _stirling2(n_units, K)
        if got != theo:
            n_wrong += 1
            bad.append('K=%s 的枚举行数 %d 与 Stirling 数 S(%d,%s)=%d 不符'
                       % (K, got, n_units, K, theo))
        if K in set(cmp_['K']):
            n_feas = int((allp[allp['K'] == K]['总量缺口合计'] <= 0).sum())
            if n_feas != int(cmp_[cmp_['K'] == K].iloc[0]['库存可行分区数']):
                bad.append('K=%s 的库存可行分区数：由 q4_all_partitions.csv 重算 %d，'
                           '自报 %s' % (K, n_feas,
                                        cmp_[cmp_['K'] == K].iloc[0]['库存可行分区数']))

    # 论文的全称断言只能建立在「全部分区的最小需求」上，故把这一行单独报出来
    mins = ' | '.join('%s %d' % (c, int(cmp_[cmp_['K'] == int(cmp_['K'].min())].iloc[0]
                                       ['最小需求_' + c]))
                      for c in labels if '最小需求_' + c in cmp_.columns)
    metrics = [
        ('原子单元 / 服务区', '%d / %d（单元不可拆违规 %d 处）' % (n_units, len(all_si), n_split)),
        ('资源规模 R 与自报的最大偏差', '%.0f（逐组逐类重算）' % maxdR),
        ('单份库存缺口与自报的最大偏差', '%.0f' % maxdGap),
        ('峰值需求的可实现性', '着色编号数或跨组复用不符 %d 处（要求 0）' % n_color_bad),
        ('枚举规模', ' '.join('K=%d %d 行' % (K, int((allp['K'] == K).sum()))
                              for K in sorted(allp['K'].unique()))
         + '（与 Stirling 数不符 %d 处）' % n_wrong),
        ('全部分区中的最小需求（K=%d）' % int(cmp_['K'].min()), mins),
        ('含失效区间的组数（须各配 ≥1 架中继）',
         ' '.join('K=%d %d 组' % (K, n_iv_grp[K]) for K in sorted(n_iv_grp))),
    ]
    return rep('表 9.4   问题四分区与资源配置独立复核', not bad, bad, dict(metrics),
               'R 与缺口逐位相等；着色编号数 = 峰值需求；枚举行数 = Stirling 数')


def check_q3_metrics(d):
    """表 9.3b：问题三交付侧与能耗指标，从**结果表**独立重算后与 q3_metrics.csv 对照。

    重算只用 q3_box_delivery.csv / q3_transport_trips.csv / q3_relay_trips.csv
    与原始货箱表：加权迟到 = Σ 优先系数 × max(0, 交付时刻 − 期望送达)，
    合计能耗 = 运输能耗 + 中继能耗，联合完工 = max(运输完工, 中继最晚返回)。
    """
    bd = _inp('q3_box_delivery.csv')
    tt = _inp('q3_transport_trips.csv')
    rt = _inp('q3_relay_trips.csv')
    qm = _inp('q3_metrics.csv')
    boxes = {b['id']: b for b in d.cargo}
    got = {r['指标']: float(r['数值']) for _, r in qm.iterrows()}
    need = ['加权迟到', '迟到箱数', '最长迟到', '最小硬时限余量', '运输完工时刻',
            '中继最晚返回时刻', '联合完工时刻', '运输能耗', '中继能耗', '合计能耗',
            '错峰推迟架次数', '累计推迟量']
    bad = [('q3_metrics.csv 缺指标 %s' % k) for k in need if k not in got]

    w_tard, n_late, max_late = 0.0, 0, 0.0
    min_margin, min_box = float('inf'), ''
    for _, r in bd.iterrows():
        bid = r['货箱编号']
        if bid not in boxes:
            bad.append('逐箱交付表出现未知箱号 %s' % bid)
            continue
        b = boxes[bid]
        t_del = float(r['交付完成时刻s'])
        if b['expect'] is not None:
            late = max(0.0, t_del - b['expect'])
            if late > 1e-6:
                n_late += 1
                max_late = max(max_late, late)
            w_tard += b['priority'] * late
        lim = []
        if b['category'] == '医疗物资' and b['expect'] is not None:
            lim.append(b['expect'])
        if b['first_batch'] and b['deadline'] is not None:
            lim.append(b['deadline'])
        if lim and min(lim) - t_del < min_margin:
            min_margin, min_box = min(lim) - t_del, bid
    trans_done = float(tt['返回O01时刻s'].max())
    relay_done = float(rt['返回O01时刻s'].max()) if len(rt) else 0.0
    e_tr = float(tt['架次能耗kWh'].sum())
    e_rl = float(rt['架次能耗kWh'].sum()) if len(rt) else 0.0
    sg = _inp('q3_stagger.csv')
    n_stag = int((sg['推迟s'].astype(float) > 0).sum())
    tot_delay = float(sg['推迟s'].sum())

    # 容差按"重算值由多少个带舍入的表内数相加而来"取（见 tol_sum）：
    # 加权迟到 = Σ 优先系数 × 交付时刻（后者留 T_DEC 位），误差上界 = Σ优先系数 × 半末位；
    # 能耗 = Σ 架次能耗（留 E_DEC 位），误差上界 = 架次数 × 半末位。
    prio_sum = sum(float(b['priority'] or 0.0) for b in d.cargo
                   if b['expect'] is not None)
    tol_tard = prio_sum * TOL_T
    tol_e_tr = tol_sum(len(tt), TOL_E)
    tol_e_rl = tol_sum(len(rt), TOL_E)
    got_vs = [('加权迟到', w_tard, tol_tard), ('迟到箱数', float(n_late), 0),
              ('最长迟到', max_late, TOL_T),
              ('最小硬时限余量', (0.0 if min_margin == float('inf') else min_margin), TOL_T),
              ('运输完工时刻', trans_done, TOL_T), ('中继最晚返回时刻', relay_done, TOL_T),
              ('联合完工时刻', max(trans_done, relay_done), TOL_T),
              ('运输能耗', e_tr, tol_e_tr), ('中继能耗', e_rl, tol_e_rl),
              ('合计能耗', e_tr + e_rl, tol_e_tr + tol_e_rl),
              ('错峰推迟架次数', float(n_stag), 0), ('累计推迟量', tot_delay, TOL_T * 4)]
    for k, v, tol in got_vs:
        if k not in got:
            continue
        if abs(got[k] - v) > tol:
            bad.append('%s 与独立重算不符：表内 %.6f vs 重算 %.6f（容差 %.1e）'
                       % (k, got[k], v, tol))
    if min_margin < -1e-6:
        bad.append('最小硬时限余量为负：%.6f s（%s）' % (min_margin, min_box))
    metrics = [('对照指标数', len(got_vs)),
               ('加权迟到', '表内 %.3f vs 重算 %.3f 权重·s' % (got.get('加权迟到', float('nan')), w_tard)),
               ('迟到箱数 / 最长迟到', '%d 个 / %.1f s' % (n_late, max_late)),
               ('最小硬时限余量', '%.3f s（%s）' % (min_margin, min_box)),
               ('运输 / 中继 / 合计能耗', '%.6f / %.6f / %.6f kWh' % (e_tr, e_rl, e_tr + e_rl)),
               ('联合完工时刻', '%.3f s（运输 %.3f，中继 %.3f）'
                % (max(trans_done, relay_done), trans_done, relay_done)),
               ('错峰', '推迟 %d 个架次，累计 %.1f s' % (n_stag, tot_delay))]
    return rep('T9.3b 问题三交付侧与能耗指标', not bad, bad, dict(metrics),
               '单值时刻 %.1e s；加权迟到 %.1e（%d 箱优先系数之和 × 半末位）；'
               '能耗 %.1e×项数（%d/%d 项）；计数严格相等'
               % (TOL_T, tol_tard, int(round(prio_sum)), TOL_E, len(tt), len(rt)))


# ---------------------------------------------------------------------------
# 9.4 节  DEM 口径与离散化
# ---------------------------------------------------------------------------
def _node_pairs(d):
    nodes = [('O01', d.O01)] + [(s['id'], s) for s in d.services]
    return list(itertools.combinations(nodes, 2))


def _dense_cells(dem, lon_a, lat_a, lon_b, lat_b, step_m):
    """超密采样得到的像元集合，用作栅格遍历的参照。"""
    xa, ya = ll_to_xy(lon_a, lat_a); xb, yb = ll_to_xy(lon_b, lat_b)
    dist = float(np.hypot(xb - xa, yb - ya))
    n = max(2, int(np.ceil(dist / step_m)) + 1)
    tt = np.linspace(0, 1, n)
    fi = (dem.lat_max - (lat_a + (lat_b - lat_a) * tt)) / dem.dlat
    fj = ((lon_a + (lon_b - lon_a) * tt) - dem.lon_min) / dem.dlon
    ii = np.floor(fi).astype(int); jj = np.floor(fj).astype(int)
    ok = (ii >= 0) & (ii < dem.dem.shape[0]) & (jj >= 0) & (jj < dem.dem.shape[1])
    return set(zip(ii[ok].tolist(), jj[ok].tolist()))


def _max_over_cells(dem, cells):
    if not cells:
        return 0.0
    cells = list(cells)
    ii = np.fromiter((c[0] for c in cells), dtype=np.intp, count=len(cells))
    jj = np.fromiter((c[1] for c in cells), dtype=np.intp, count=len(cells))
    z = dem._dem_nan[ii, jj]
    z = z[np.isfinite(z)]
    return float(z.max()) if z.size else 0.0


def _legacy_max_elev(dem, lon_a, lat_a, lon_b, lat_b, step=5.0):
    """口径修正前的实现（保留以便量化口径差）：对双线性插值面按 step 采样取最大。"""
    xa, ya = ll_to_xy(lon_a, lat_a); xb, yb = ll_to_xy(lon_b, lat_b)
    dist = float(np.hypot(xb - xa, yb - ya))
    n = max(2, int(np.ceil(dist / step)) + 1)
    tt = np.linspace(0, 1, n)
    z = dem.sample(lon_a + (lon_b - lon_a) * tt, lat_a + (lat_b - lat_a) * tt)
    z = z[np.isfinite(z)]
    return float(z.max()) if z.size else 0.0


def check_dem_caliber(d, step_probe=0.5):
    """9.4 节：巡航高度所依据的 DEM 口径，及其与旧口径之差。

    题面附录 2 要求"计划巡航海拔取该航段所经过 DEM 像元的最高地面高程以上
    50 m"。本文对该量做栅格遍历（core.DEMGrid._cells_along，Amanatides–Woo），
    取航段穿过的每一个 30m 像元的**原始**高程最大值。这里独立复核两件事：

      (1) 遍历本身对不对。用超密采样得到的像元集合当参照，查有没有漏掉航段
          真正穿过的像元。漏掉即低估，是本项唯一的危险方向，必须为 0；
          多出来的像元只可能是航线擦角而过的格子，方向保守，单独计数。
      (2) 口径差有多大。同一批航段上比较"像元遍历"与"对双线性插值面采样取
          最大"两种口径——后者是本文早期版本用过的算法。

    刻意把 (2) 单独列出来，是因为旧的 check_dem_step 拿"1m 采样的插值面"当
    真值，按构造就测不到插值-像元之间的口径差，只能测出采样步长自身的残差，
    于是把一个量级更大的口径误差漏在了检查之外。

    判据只取 (1) 的漏像元数：它违反的是"遍历必须覆盖航段实际经过的像元"这条
    硬要求。(2) 是量级测量，无对错，随报告一并给出。
    """
    dem = d.dem
    pairs = _node_pairs(d)
    n_miss = n_extra = 0
    cap_miss = 0.0
    worst_trav = []
    for (i1, n1), (i2, n2) in pairs:
        trav = set(dem._cells_along(n1['lon'], n1['lat'], n2['lon'], n2['lat']))
        dense = _dense_cells(dem, n1['lon'], n1['lat'], n2['lon'], n2['lat'], step_probe)
        miss = dense - trav
        n_miss += len(miss); n_extra += len(trav - dense)
        if miss:
            # 漏像元直接折算成可能的高程低估量
            cap_miss = max(cap_miss, _max_over_cells(dem, dense) - _max_over_cells(dem, trav))
        worst_trav.append((len(miss), i1, i2))
    worst_trav.sort(reverse=True)

    # (2) 口径差：像元原始值 vs 插值面采样
    rows = []
    for (i1, n1), (i2, n2) in pairs:
        e_cell = dem.max_elev_along(n1['lon'], n1['lat'], n2['lon'], n2['lat'])
        e_leg = _legacy_max_elev(dem, n1['lon'], n1['lat'], n2['lon'], n2['lat'])
        rows.append((e_cell - e_leg, i1, i2))
    arr = np.array([r[0] for r in rows])
    rows.sort(reverse=True)

    bad = []
    if n_miss:
        bad.append('栅格遍历漏掉 %d 个航段实际经过的像元（会低估 H_max，必须为 0）；'
                   '漏得最多的航段：%s'
                   % (n_miss, '，'.join('%s→%s %d个' % (w[1], w[2], w[0])
                                      for w in worst_trav[:3])))
    metrics = [
        ('节点连线数 / 超密采样步长', '%d 条 / %.1f m' % (len(pairs), step_probe)),
        ('(1) 漏像元 / 擦角多计像元', '%d 个（必须 0） / %d 个（方向保守）' % (n_miss, n_extra)),
        ('(1) 漏像元导致的最大高程低估', '%.3f m' % cap_miss),
        ('(2) 口径差（像元遍历 − 插值面采样）',
         '低估 %d 条 (%.0f%%)，高估 %d 条，一致 %d 条'
         % (int((arr > 1e-9).sum()), 100.0 * (arr > 1e-9).mean(),
            int((arr < -1e-9).sum()), int((np.abs(arr) <= 1e-9).sum()))),
        ('(2) 最大低估 / 最大高估 / 均值 / 中位',
         '+%.2f m（%s→%s） / %.2f m / %+.3f m / %+.3f m'
         % (arr.max(), rows[0][1], rows[0][2], arr.min(), arr.mean(), np.median(arr))),
        ('(2) |口径差| >= 1/5/10 m 的航段数', '%d / %d / %d'
         % (int((np.abs(arr) >= 1).sum()), int((np.abs(arr) >= 5).sum()),
            int((np.abs(arr) >= 10).sum()))),
        ('说明', '低估 H_max -> 低估巡航海拔 -> 低估爬升量、时间与能耗，属非保守误差'),
    ]
    return rep('T9.4a DEM 像元遍历与口径差', not bad, bad, dict(metrics),
               '漏像元 = 0（擦角多计不设限）')


def check_occlusion_step(d, step_coarse=DEM_STEP, step_fine=1.0):
    """9.4 节：DEM_STEP 现在只管视线遮挡判定，检验它是否改变遮挡结论。

    口径修正后，巡航高度改走像元遍历，采样步长只剩 line_occluded 一个用途。
    故这里问的不再是"最高地形被低估多少"，而是"步长是否会把遮挡判反"。
    判据直接取翻转条数：论文报的是"翻转 0 条"，翻转即该判定依赖步长、结论不可信。
    除翻转次数外还报告连线相对地形的净空余量：余量贴近 0 的连线才是脆弱点，
    只看翻转率会把"本来就有几百米余量"的绝大多数连线混进来稀释掉。
    """
    dem = d.dem
    flips, clear = [], []
    for (i1, n1), (i2, n2) in _node_pairs(d):
        a1 = n1['alt'] if i1 == 'O01' else n1['alt'] + 30.0
        a2 = n2['alt'] if i2 == 'O01' else n2['alt'] + 30.0
        geo = segment_geometry(dem, n1['lon'], n1['lat'], a1, n2['lon'], n2['lat'], a2)
        pa = (n1['lon'], n1['lat'], geo['cruise_alt'])
        pb = (n2['lon'], n2['lat'], geo['cruise_alt'])
        oc_c = dem.line_occluded(pa, pb, step=step_coarse)
        oc_f = dem.line_occluded(pa, pb, step=step_fine)
        if oc_c != oc_f:
            flips.append((i1, i2, oc_c, oc_f))
        clear.append(_clearance(dem, pa, pb, step_fine))
    clear = np.array(clear)
    bad = ['%s→%s 粗步长=%s 细步长=%s' % f for f in flips[:5]]
    if flips:
        bad.insert(0, '视线遮挡判定随采样步长翻转 %d 条（论文报 0 条）' % len(flips))
    metrics = [
        ('节点连线数（%.0fm vs %.0fm）' % (step_coarse, step_fine), len(clear)),
        ('判定翻转', '%d 条（论文报 0）' % len(flips)),
        ('沿线净空余量（正=不遮挡）', '最小 %.1f m，中位 %.1f m，<1m 的连线 %d 条'
         % (clear.min(), np.median(clear), int((np.abs(clear) < 1.0).sum()))),
    ]
    return rep('T9.4b 视线遮挡判定的步长敏感性', not flips, bad, dict(metrics),
               '翻转 0 条')


def _clearance(dem, pa, pb, step=1.0):
    """视线连线相对沿途地形的净空余量(m)，负值表示被地形挡住。"""
    xa, ya = ll_to_xy(pa[0], pa[1]); xb, yb = ll_to_xy(pb[0], pb[1])
    dist = float(np.hypot(xb - xa, yb - ya))
    n = max(2, int(np.ceil(dist / step)) + 1)
    tt = np.linspace(0, 1, n)
    z = dem.sample(pa[0] + (pb[0] - pa[0]) * tt, pa[1] + (pb[1] - pa[1]) * tt)
    line_alt = pa[2] + (pb[2] - pa[2]) * tt
    ok = np.isfinite(z)
    if not ok.any():
        return np.inf
    return float(np.min(line_alt[ok] - z[ok]))


def check_bilinear_bound(d, n_probe=20000, seed=0):
    """9.4 节：双线性插值的自洽性——插值面被所在单元格四角高程夹住。

    **本项只证明插值作为一种"面"是自洽的，不能用来论证巡航高度口径正确。**
    恰恰相反：正因为插值值是四角高程的凸组合、必被四角夹住，它沿线的最大值
    结构性地不高于所经过像元的原始最大值——这正是旧口径系统性低估 H_max 的
    成因。该性质的实际用途只有一个：line_occluded 沿连线做插值采样判定遮挡时，
    不会因插值本身造出高于局部真实地形的虚假尖峰而误判为遮挡。
    """
    rng = np.random.default_rng(seed)
    lon = rng.uniform(d.dem.lon_min, d.dem.lon_max, n_probe)
    lat = rng.uniform(d.dem.lat_min, d.dem.lat_max, n_probe)
    z = d.dem.sample(lon, lat)
    ok = np.isfinite(z)
    lon, lat, z = lon[ok], lat[ok], z[ok]

    # 取每个探针所在单元格的四角原始高程
    fj = (lon - d.dem.lon_min) / d.dem.dlon
    fi = (d.dem.lat_max - lat) / d.dem.dlat
    j0 = np.floor(fj).astype(int)
    i0 = np.floor(fi).astype(int)
    m = (i0 >= 0) & (i0 + 1 < d.dem.dem.shape[0]) & (j0 >= 0) & (j0 + 1 < d.dem.dem.shape[1])
    i0, j0, z = i0[m], j0[m], z[m]
    Z = d.dem._dem_nan
    corners = np.vstack([Z[i0, j0], Z[i0, j0 + 1], Z[i0 + 1, j0], Z[i0 + 1, j0 + 1]])
    lo = np.nanmin(corners, axis=0)
    hi = np.nanmax(corners, axis=0)
    good = np.isfinite(lo) & np.isfinite(hi)
    over = int(np.sum((z[good] > hi[good] + 1e-9) | (z[good] < lo[good] - 1e-9)))
    bad = [] if over == 0 else ['%d 个探针的插值值越出所在单元格四角高程区间' % over]
    metrics = [('有效探针数', int(good.sum())),
               ('越出四角高程区间的探针', '%d 个（应为 0）' % over)]
    return rep('T9.4c 双线性插值自洽性', over == 0, bad, dict(metrics),
               '插值值必须落在四角高程区间内（容差 1e-9 m）')


# ---------------------------------------------------------------------------
# 汇总（F06）
# ---------------------------------------------------------------------------
CHECKS = [
    ('表 9.0   问题一推荐组批方案独立复核', check_q1),
    ('表 9.0b  问题一集合划分 ILP 交叉复核', check_q1_ilp),
    ('表 9.1   问题二硬约束独立复核', check_hard_constraints),
    ('表 9.2   问题三中继方案独立复核', check_relay),
    ('表 9.3   通信连续性独立复核（Δt=1 s 逐时刻重判）', check_relay_final),
    ('表 9.3b  问题三交付侧与能耗指标独立重算', check_q3_metrics),
    ('表 9.3c  错峰后运输侧资源链独立复核', check_q3_transport),
    ('表 9.4   问题四分区与资源配置独立复核', check_q4),
    ('9.4 节   DEM 口径与离散化误差', check_dem_caliber),
    ('9.4 节   视线遮挡判定的步长敏感性', check_occlusion_step),
    ('9.4 节   双线性插值自洽性', check_bilinear_bound),
]


def _print_report(r):
    tag = 'PASS' if r['passed'] else 'FAIL'
    print('[%s] %s' % (tag, r['name']))
    if r['tolerance']:
        print('        容差：%s' % r['tolerance'])
    for k, v in r['metrics'].items():
        print('        %s = %s' % (k, v))
    for e in r['errors'][:12]:
        print('        ! %s' % e)
    if len(r['errors']) > 12:
        print('        ! …其余 %d 条同类问题见上表' % (len(r['errors']) - 12))


def run_checks(d, checks=None):
    """逐项跑检查；单项抛异常记为该检查失败并继续，不中断其余项。"""
    checks = CHECKS if checks is None else checks
    reports = []
    for title, fn in checks:
        print('=' * 74)
        print(title)
        print('=' * 74)
        try:
            r = fn(d)
        except Exception as e:                                        # noqa: BLE001
            r = rep(fn.__name__, False,
                    ['检查无法完成：%s: %s' % (type(e).__name__, e)])
        reports.append(r)
        _print_report(r)
        print()
    return reports


def summarize(reports, n_expect):
    """汇总并给出退出码：任一项失败、或没跑满，均返回 1。"""
    n_pass = sum(1 for r in reports if r['passed'])
    n_fail = len(reports) - n_pass
    print('=' * 74)
    print('独立复核汇总：通过 %d 项，失败 %d 项（共 %d 项检查）'
          % (n_pass, n_fail, len(CHECKS)))
    for r in reports:
        if not r['passed']:
            print('  [FAIL] %s（%d 处问题）' % (r['name'], max(1, len(r['errors']))))
    if not reports or len(reports) != n_expect:
        print('!! 检查项未全部执行，按失败处理')
        return 1
    print('结论：%s' % ('全部检验项通过——论文第 9 章可据此表述'
                      if n_fail == 0 else
                      '存在未通过项——论文中对应表述必须降级，不得写成"已复核"'))
    return 1 if n_fail else 0


def main(d=None, checks=None):
    """跑完全部检查并返回退出码。`d`/`checks` 只为回归测试留出入口。"""
    if d is None:
        d = load_data()
    checks = CHECKS if checks is None else checks
    return summarize(run_checks(d, checks), len(checks))


if __name__ == '__main__':
    raise SystemExit(main())
