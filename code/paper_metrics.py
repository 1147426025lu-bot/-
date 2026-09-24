# -*- coding: utf-8 -*-
# 本程序及代码是在人工智能工具辅助下完成的。
# 工具名称：DeepSeek；版本/型号：DeepSeek V4.1 Flash（API 模型名 deepseek-flash）；
# 开发机构：杭州深度求索人工智能基础技术研究有限公司；
# 版本发布日期：2026 年 9 月 10 日。

"""全篇数字的唯一来源：从 results/ 重算，产出 paper_metrics.json 与 LaTeX 宏。

论文里的每个数字都必须能在结果表里找到出处，且改动脚本后重跑一次就同步更新。
手工誊抄是这轮修订里最反复出错的环节——正文写着 63.180 kWh，代码已经跑到
59.236 kWh，两边都「看着没错」。这里把数字收成一个来源：正文只写宏
`\\mtQoneEnergy{}`，宏由本脚本按固定小数位生成，数值与出处一并落进 JSON 备查。

三条硬规定：
  1) **缺文件就失败**，不静默跳过。少一张表意味着少一个数字，而正文照旧会编译
     通过——那正是要避免的情形。
  2) **每个宏都带出处**（`sources` 里记着它读的是哪张表的哪一列/哪一行），
     便于复核。
  3) 只做「读表 → 取值 → 格式化」，不在这里做任何业务计算。要算的量一律由求解
     脚本落盘——本脚本再算一遍就成了第二套物理模型，两边不一致时无从判断谁对。

用法：python code/paper_metrics.py
输出：results/paper_metrics.json、sections/generated_metrics.tex
"""
import io
import json
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
RES = os.path.join(ROOT, 'results')
TEX = os.path.join(ROOT, 'sections', 'generated_metrics.tex')

# 小数位：能耗 3 位（与求解脚本 3 位一致）、时刻 1 位（秒）、比率 4 位
D_E, D_T, D_R = 3, 1, 4


def _need(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        raise SystemExit('缺少结果文件 results/%s。\n'
                         '本脚本只读结果、不猜数字：请先按依赖顺序重跑 '
                         'q1.py → q2.py → q3.py → q4.py。' % name)
    return p


def _csv(name, **kw):
    return pd.read_csv(_need(name), encoding='utf-8-sig', **kw)


def _json(name):
    with open(_need(name), encoding='utf-8') as f:
        return json.load(f)


def _metric(metrics_df, key):
    """从「一行一指标」的汇总表里取一个数（q3_metrics.csv 就是这种表）。"""
    hit = metrics_df[metrics_df['指标'] == key]
    if len(hit) != 1:
        raise SystemExit('指标表中「%s」出现 %d 次，应为 1 次' % (key, len(hit)))
    return float(hit['数值'].iloc[0])


def collect():
    """返回 (metrics, sources)：metrics 是 宏名 → 格式化后的字符串。"""
    m, src = {}, {}

    def put(key, value, source, nd=D_E, pct=False):
        if key in m:
            raise SystemExit('宏名重复：%s' % key)
        if isinstance(value, float):
            v = ('%.*f' % (nd, value))
        else:
            v = str(value)
        if pct:
            v = v + r'\%'
        m[key] = v
        src[key] = source

    # ---------------- 问题一 ----------------
    b = _csv('q1_recommended_batching.csv')
    par = _csv('q1_pareto.csv')
    put('QoneTrips', len(b), 'q1_recommended_batching.csv 行数')
    put('QoneEnergy', float(b['架次能耗kWh'].sum()),
        'q1_recommended_batching.csv 架次能耗kWh 求和')
    put('QoneMass', float(b['总质量kg'].sum()),
        'q1_recommended_batching.csv 总质量kg 求和')
    put('QoneMinSoc', float(b['返航SOC'].min()) * 100,
        'q1_recommended_batching.csv 返航SOC 最小值 ×100', nd=2, pct=True)
    put('QoneMaxRound', float(b['往返时间s'].max()),
        'q1_recommended_batching.csv 往返时间s 最大值', nd=D_T)
    put('QoneWorkTimeH', float(par['总作业时间h'].iloc[0]),
        'q1_pareto.csv 总作业时间h（推荐方案行）', nd=2)
    put('QoneWorkTimeS', float(par['总作业时间s'].iloc[0]),
        'q1_pareto.csv 总作业时间s（推荐方案行）', nd=D_T)
    put('QoneAreas', len(_csv('q1_max_payload.csv')), 'q1_max_payload.csv 行数')
    ilp = _csv('q1_ilp_mixed.csv')
    put('QoneIlpAgree', int(ilp['一致'].astype(str).str.strip().isin(
        ['True', 'TRUE', '一致', '是']).sum()),
        'q1_ilp_mixed.csv 一致列为真的行数', nd=0)
    put('QoneIlpCases', len(ilp), 'q1_ilp_mixed.csv 行数', nd=0)
    gap = _csv('q1_ilp_gap.csv')
    put('QoneIlpGapCases', len(gap), 'q1_ilp_gap.csv 行数', nd=0)

    # ---------------- 问题二 ----------------
    sol2 = _json('q2_solution.json')
    tt = _csv('q2_transport_trips.csv')
    put('QtwoTrips', len(tt), 'q2_transport_trips.csv 行数', nd=0)
    put('QtwoUavs', len(sol2['uav_ids']), 'q2_solution.json uav_ids', nd=0)
    put('QtwoEnergy', float(tt['架次能耗kWh'].sum()),
        'q2_transport_trips.csv 架次能耗kWh 求和')
    put('QtwoMakespan', float(tt['返回O01时刻s'].max()),
        'q2_transport_trips.csv 返回O01时刻s 最大值', nd=D_T)
    put('QtwoTardiness', float(sol2['metrics']['tardiness']),
        'q2_solution.json metrics.tardiness', nd=1)
    put('QtwoHardViol', int(sol2['metrics']['hard_viol']),
        'q2_solution.json metrics.hard_viol', nd=0)
    put('QtwoBoxes', len(_csv('q2_box_delivery.csv')), 'q2_box_delivery.csv 行数', nd=0)
    put('QtwoSeed', int(sol2['seed']), 'q2_solution.json seed', nd=0)
    scan = _csv('q2_scan.csv')
    put('QtwoScanRows', len(scan), 'q2_scan.csv 行数', nd=0)
    dom = _csv('q2_pareto.csv')
    put('QtwoParetoRows', len(dom), 'q2_pareto.csv 行数', nd=0)
    # 推荐档：扫描表里实际架次数最少、能耗最低的那一档（与求解脚本同序）
    rec = dom.sort_values(['实际架次', '能耗kWh']).iloc[0]
    put('QtwoRecK', int(rec['K上限']), 'q2_pareto.csv K上限（推荐档）', nd=0)
    put('QtwoRecTrips', int(rec['实际架次']), 'q2_pareto.csv 实际架次（推荐档）', nd=0)
    put('QtwoRecEnergy', float(rec['能耗kWh']), 'q2_pareto.csv 能耗kWh（推荐档）')
    put('QtwoRecMakespan', float(rec['makespan_s']),
        'q2_pareto.csv makespan_s（推荐档）', nd=D_T)
    ex = _csv('q2_exact.csv')
    put('QtwoExactCases', len(ex), 'q2_exact.csv 行数', nd=0)
    # 层 A 的行没有捕获求解器状态，该列留空；空值按 0 计，不得让 NaN 混进计数。
    put('QtwoExactTimeout', int(ex['超时'].fillna(0).sum()),
        'q2_exact.csv 超时列求和（空值按 0）', nd=0)
    put('QtwoExactProved', int(ex['已证最优'].fillna(0).sum()),
        'q2_exact.csv 已证最优列求和（空值按 0）', nd=0)

    # ---------------- 问题三 ----------------
    q3 = _csv('q3_metrics.csv')
    r3 = _csv('q3_relay_trips.csv')
    c3 = _csv('q3_comm_phases.csv')
    put('QthreeRelays', len(r3), 'q3_relay_trips.csv 行数', nd=0)
    put('QthreeStations', len(_csv('q3_stations.csv')), 'q3_stations.csv 行数', nd=0)
    put('QthreeSegs', len(c3), 'q3_comm_phases.csv 行数', nd=0)
    put('QthreeIntervals', len(_csv('q3_intervals.csv')), 'q3_intervals.csv 行数', nd=0)
    put('QthreeDirect', _metric(q3, '直连时间占比') * 100,
        'q3_metrics.csv 直连时间占比 ×100', nd=2, pct=True)
    put('QthreeRelayFrac', _metric(q3, '中继时间占比') * 100,
        'q3_metrics.csv 中继时间占比 ×100', nd=2, pct=True)
    put('QthreeGap', _metric(q3, '中断时间占比') * 100,
        'q3_metrics.csv 中断时间占比 ×100', nd=2, pct=True)
    put('QthreeTransEnergy', _metric(q3, '运输能耗'),
        'q3_metrics.csv 运输能耗')
    put('QthreeRelayEnergy', _metric(q3, '中继能耗'), 'q3_metrics.csv 中继能耗')
    put('QthreeJointEnergy', _metric(q3, '合计能耗'), 'q3_metrics.csv 合计能耗')
    put('QthreeTransDone', _metric(q3, '运输完工时刻'), 'q3_metrics.csv 运输完工时刻', nd=D_T)
    put('QthreeJointDone', _metric(q3, '联合完工时刻'), 'q3_metrics.csv 联合完工时刻', nd=D_T)
    put('QthreeTardiness', _metric(q3, '加权迟到'), 'q3_metrics.csv 加权迟到', nd=1)
    put('QthreeLateBoxes', int(_metric(q3, '迟到箱数')), 'q3_metrics.csv 迟到箱数', nd=0)
    put('QthreeMaxLate', _metric(q3, '最长迟到'), 'q3_metrics.csv 最长迟到', nd=D_T)
    put('QthreeMinMargin', _metric(q3, '最小硬时限余量'),
        'q3_metrics.csv 最小硬时限余量', nd=D_T)
    put('QthreeStaggered', int(_metric(q3, '错峰推迟架次数')),
        'q3_metrics.csv 错峰推迟架次数', nd=0)
    put('QthreeDelaySum', _metric(q3, '累计推迟量'), 'q3_metrics.csv 累计推迟量', nd=D_T)
    put('QthreeCoverageStep', float(_csv('q3_coverage.csv')['采样步长s'].max()),
        'q3_coverage.csv 采样步长s 最大值', nd=1)

    # ---------------- 问题四 ----------------
    cmp4 = _csv('q4_comparison.csv').iloc[0]
    put('QfourK', int(cmp4['K']), 'q4_comparison.csv K', nd=0)
    put('QfourR', int(cmp4['资源规模R']), 'q4_comparison.csv 资源规模R', nd=0)
    put('QfourCV', float(cmp4['工作量均衡CV_W']),
        'q4_comparison.csv 工作量均衡CV_W', nd=4)
    put('QfourRelayNeed', int(cmp4['中继总需求最小']),
        'q4_comparison.csv 中继总需求最小', nd=0)
    put('QfourGapSum', int(cmp4['总量缺口']), 'q4_comparison.csv 总量缺口', nd=0)
    put('QfourRedundancy', int(cmp4['总量冗余']), 'q4_comparison.csv 总量冗余', nd=0)
    put('QfourFeasible', int(cmp4['库存可行分区数']),
        'q4_comparison.csv 库存可行分区数', nd=0)
    put('QfourPartitions', int(len(_csv('q4_partition.csv'))),
        'q4_partition.csv 行数', nd=0)
    put('QfourAllPartitions', int(len(_csv('q4_all_partitions.csv'))),
        'q4_all_partitions.csv 行数', nd=0)
    put('QfourUnits', int(len(_csv('q4_units.csv'))), 'q4_units.csv 行数', nd=0)
    return m, src


# 宏名 → 正文里的含义（写进 JSON，便于「这个数字是干什么用的」当场可查）
DOC = {
    'QoneTrips': '问题一 总架次', 'QoneEnergy': '问题一 总能耗 kWh',
    'QoneMass': '问题一 总运载质量 kg', 'QoneMinSoc': '问题一 最小返航 SOC',
    'QoneMaxRound': '问题一 单架次最长往返时间 s', 'QoneWorkTimeH': '问题一 总作业时间 h',
    'QoneWorkTimeS': '问题一 总作业时间 s', 'QoneAreas': '服务区总数',
    'QoneIlpAgree': '问题一 与 ILP 复核一致的区数', 'QoneIlpCases': '问题一 参与复核的区数',
    'QoneIlpGapCases': '问题一 逐机型逐区算例数',
    'QtwoTrips': '问题二 运输架次数', 'QtwoUavs': '运输无人机架数',
    'QtwoEnergy': '问题二 总能耗 kWh', 'QtwoMakespan': '问题二 完工时刻 s',
    'QtwoTardiness': '问题二 加权时延', 'QtwoHardViol': '问题二 硬约束违反数',
    'QtwoBoxes': '交付箱次', 'QtwoSeed': 'ALNS 随机种子',
    'QtwoScanRows': 'ε-约束扫描档数', 'QtwoParetoRows': '非支配前沿档数',
    'QtwoRecK': '推荐档 K 上限', 'QtwoRecTrips': '推荐档实际架次',
    'QtwoRecEnergy': '推荐档能耗 kWh', 'QtwoRecMakespan': '推荐档完工时刻 s',
    'QtwoExactCases': '精确验证算例数', 'QtwoExactTimeout': '超时算例数',
    'QtwoExactProved': '已证最优算例数',
    'QthreeRelays': '问题三 中继架次数', 'QthreeStations': '悬停站数',
    'QthreeSegs': '通信分段数', 'QthreeIntervals': '通信失效区间数',
    'QthreeDirect': '直连时间占比', 'QthreeRelayFrac': '中继时间占比',
    'QthreeGap': '中断时间占比', 'QthreeTransEnergy': '问题三 运输能耗 kWh',
    'QthreeRelayEnergy': '问题三 中继能耗 kWh', 'QthreeJointEnergy': '问题三 合计能耗 kWh',
    'QthreeTransDone': '运输完工时刻 s', 'QthreeJointDone': '联合完工时刻 s',
    'QthreeTardiness': '问题三 加权迟到', 'QthreeLateBoxes': '迟到箱数',
    'QthreeMaxLate': '最长迟到 s', 'QthreeMinMargin': '最小硬时限余量 s',
    'QthreeStaggered': '错峰推迟架次数', 'QthreeDelaySum': '累计推迟量 s',
    'QthreeCoverageStep': '覆盖核算采样步长 s',
    'QfourK': '问题四 分区数 K', 'QfourR': '问题四 资源规模 R',
    'QfourCV': '问题四 工作量均衡 CV', 'QfourRelayNeed': '中继无人机总需求',
    'QfourGapSum': '单份库存下的总量缺口', 'QfourRedundancy': '总量冗余',
    'QfourFeasible': '库存可行分区数', 'QfourPartitions': '入选分区数',
    'QfourAllPartitions': '枚举分区总数', 'QfourUnits': '任务单元数',
}


def main():
    m, src = collect()
    unknown = sorted(set(m) - set(DOC))
    if unknown:
        raise SystemExit('以下宏没有写进 DOC 说明：%s' % unknown)
    out = {k: dict(value=v, meaning=DOC[k], source=src[k]) for k, v in m.items()}
    with open(os.path.join(RES, 'paper_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    lines = ['% 本文件由 code/paper_metrics.py 自动生成，请勿手工编辑。',
             '% 每个宏的数值与出处见 results/paper_metrics.json。',
             '']
    for k in sorted(m):
        lines.append('\\newcommand{\\mt%s}{%s}' % (k, m[k]))
    with open(TEX, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

    print('共 %d 个宏，已写入 %s 与 %s'
          % (len(m), os.path.relpath(os.path.join(RES, 'paper_metrics.json'), ROOT),
             os.path.relpath(TEX, ROOT)))
    for k in sorted(m):
        print('  \\mt%-22s %-12s  <- %s' % (k, m[k], src[k]))


if __name__ == '__main__':
    main()
