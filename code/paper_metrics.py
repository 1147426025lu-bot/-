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


# TeX 的控制序列名**只能由字母组成**。`\newcommand{\mtFoo2}{...}` 里 TeX 读到的是
# 控制序列 `\mtFoo` 后面跟着文本「2」，于是那个数字直接落进导言区，报出
# 「! LaTeX Error: Missing \begin{document}」——而报错行号指向的却是**下一条**
# \newcommand，极难定位。故宏名一律不含数字：K 档、件数一律按此表拼写。
# main() 里另有一条断言把这条规矩钉死，任何带数字的新宏都会当场被拒。
NUM_WORD = {1: 'One', 2: 'Two', 3: 'Three', 4: 'Four', 5: 'Five',
            20: 'Twenty', 22: 'TwentyTwo', 25: 'TwentyFive', 30: 'Thirty'}


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
    # nd=3：正文（摘要与 5.3 节）报的是 9.103 h，取 2 位会印成 9.10、
    # 与正文对不上。这里按正文所需精度取 3 位，数字仍只有这一个来源。
    put('QoneWorkTimeH', float(par['总作业时间h'].iloc[0]),
        'q1_pareto.csv 总作业时间h（推荐方案行）', nd=3)
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

    # ---- 问题一：ρ 与 κ 的耦合可加性（三个角点算例，供 §9.2 引用）-------------
    # 这组比值原先**手算后硬编码在 LaTeX 里**，并且印成了倒数（写 0.925、0.971，
    # 实为 7.38/6.83、26.99/26.21），方向也随之说反。改为脚本生成，杜绝再抄错。
    # 定义：比值 = 联合降幅 ÷（ρ 单独降幅 + κ 单独降幅）。>1 表示两因素同时劣化时
    # 的降幅**超过**各自之和（轻微超可加），<1 才表示一项被另一项抵消。
    s2 = _csv('q1_sensitivity2d.csv')

    def _avg(tid, rho, kap):
        hit = s2[(s2['机型'] == tid) & (s2['ρ'].round(4) == rho)
                 & (s2['κ'].round(4) == kap)]
        if len(hit) != 1:
            raise SystemExit('q1_sensitivity2d.csv 里 (机型=%s, ρ=%s, κ=%s) 命中 %d 行，应为 1'
                             % (tid, rho, kap, len(hit)))
        return float(hit['平均q_max'].iloc[0])

    ratios = {}
    for tid in ('A', 'B', 'C'):
        base = _avg(tid, 0.10, 1.00)
        d_rho = base - _avg(tid, 0.40, 1.00)
        d_kap = base - _avg(tid, 0.10, 0.80)
        d_jnt = base - _avg(tid, 0.40, 0.80)
        put('QoneSensRhoDrop%s' % tid, d_rho,
            'q1_sensitivity2d.csv：ρ 0.10→0.40 的平均载荷降幅 kg', nd=2)
        put('QoneSensKapDrop%s' % tid, d_kap,
            'q1_sensitivity2d.csv：κ 1.00→0.80 的平均载荷降幅 kg', nd=2)
        put('QoneSensJointDrop%s' % tid, d_jnt,
            'q1_sensitivity2d.csv：ρ 与 κ 同时劣化至 0.40/0.80 的平均载荷降幅 kg', nd=2)
        ssum = d_rho + d_kap
        ratios[tid] = (d_jnt / ssum) if ssum > 1e-12 else 1.0
        put('QoneSensRatio%s' % tid, ratios[tid],
            '联合降幅 ÷ (ρ 降幅 + κ 降幅)；>1 为轻微超可加', nd=3)
    put('QoneSensRatioMax', max(ratios.values()),
        '三个比值中的最大值（B 型）', nd=3)
    put('QoneSensExcessPct', (max(ratios.values()) - 1.0) * 100,
        '(最大比值 − 1) ×100，即联合降幅超出两项之和的百分点', nd=1, pct=True)

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
    # 这个宏叫「非支配前沿档数」，就必须数**非支配=是**的行，不能数文件行数。
    # q2_pareto.csv 是「全部扫描档 + 一列非支配标记」，行数 = 档数（6），而真正的
    # 前沿档是 5：K<=30 一档被 K<=35 严格占优（同为 29 架次，能耗/完成时间/加权时延
    # 三项同时更差），标了「否」。用 len(dom) 会让宏值 6 与 §6 正文「其余五档互不
    # 支配，共同构成四维非支配前沿」当场矛盾——两处都在同一篇论文里，评审一对就露。
    # 现在这两个宏没被正文引用，所以矛盾还没显形；正因为没显形才更要改，否则哪天
    # 有人写了「\mtQtwoParetoRows 档构成前沿」，错值会直接印进 PDF 而不报错。
    n_dom = int((dom['非支配'].astype(str).str.strip() == '是').sum())
    put('QtwoParetoRows', n_dom, 'q2_pareto.csv 中「非支配=是」的行数', nd=0)
    # 推荐档必须**由落盘方案的标识定**，不能在这里另按「架次、能耗排序第一行」重挑一遍。
    # 按排序重挑在旧口径下恰好与落盘方案重合（都是 K<=20），所以看不出问题；但选择键
    # 已改为题目优先级字典序（时延 → 完工 → 能耗 → 架次），选中的不再是最少架次那一档，
    # 排序重挑就会指向**另一个方案**——论文里的「推荐档能耗/完工/架次」会与
    # q2_solution.json 的实际推荐互相矛盾，而且不报错。
    # 改为按落盘方案的 (实际架次, 能耗) 在扫描表里定位它自己那一行；定位不到即说明
    # 论文要写的推荐方案不在扫描表里，这是必须炸出来的错误，不能退回排序。
    _n, _e = float(sol2['metrics']['n_trips']), float(sol2['metrics']['total_E'])
    _hit = dom[(dom['实际架次'].astype(float) == _n)
               & ((dom['能耗kWh'].astype(float) - _e).abs() < 1e-6)]
    if len(_hit) != 1:
        raise ValueError(
            'q2_solution.json 的推荐方案（架次=%g、能耗=%.6f kWh）在 q2_pareto.csv 中'
            '匹配到 %d 行，应为 1 行；论文推荐档必须能追溯到落盘方案本身'
            % (_n, _e, len(_hit)))
    rec = _hit.iloc[0]
    # 溯源串一律写「落盘推荐方案所在档」而非「推荐档」：前者说的是**这个值是那一行
    # 的字面拷贝**（因而与 q2_solution.json 同源、永不背离），后者听起来像「本脚本
    # 自己评出的推荐档」——一旦有人拿它当作独立判断，就会在下一次改选择键时再次分叉。
    put('QtwoRecK', int(rec['K上限']), 'q2_pareto.csv K上限（落盘推荐方案所在档）', nd=0)
    put('QtwoRecTrips', int(rec['实际架次']),
        'q2_pareto.csv 实际架次（落盘推荐方案所在档）', nd=0)
    put('QtwoRecEnergy', float(rec['能耗kWh']),
        'q2_pareto.csv 能耗kWh（落盘推荐方案所在档）')
    put('QtwoRecMakespan', float(rec['makespan_s']),
        'q2_pareto.csv makespan_s（落盘推荐方案所在档）', nd=D_T)
    ex = _csv('q2_exact.csv')
    put('QtwoExactCases', len(ex), 'q2_exact.csv 行数', nd=0)
    # 层 A 的行没有捕获求解器状态，该列留空；空值按 0 计，不得让 NaN 混进计数。
    put('QtwoExactTimeout', int(ex['超时'].fillna(0).sum()),
        'q2_exact.csv 超时列求和（空值按 0）', nd=0)
    put('QtwoExactProved', int(ex['已证最优'].fillna(0).sum()),
        'q2_exact.csv 已证最优列求和（空值按 0）', nd=0)

    # 多种子稳定性：q2_seeds.csv 一行 = 一个 (K 档, 种子) 组合的末态。
    # 这里只做「分组取极差 / 计数」，不重算任何指标——每个数都是表里已有值的差或计数。
    sd = _csv('q2_seeds.csv')
    put('QtwoSeedRuns', len(sd), 'q2_seeds.csv 行数（= 档数 × 种子数）', nd=0)
    _spread = (sd.groupby('K上限')[['makespan_s', '能耗kWh']].max()
               - sd.groupby('K上限')[['makespan_s', '能耗kWh']].min())
    # 交付档是 K<=25（QtwoRecK 与它同源）；两档极差都留一个宏，正文要拿它们对比。
    put('QtwoSeedSpreadKTwenty', float(_spread.loc[20, 'makespan_s']),
        'q2_seeds.csv K=20 档 makespan_s 极差（max-min）', nd=D_T)
    put('QtwoSeedSpreadKTwentyFive', float(_spread.loc[25, 'makespan_s']),
        'q2_seeds.csv K=25 档 makespan_s 极差（max-min）', nd=D_T)
    put('QtwoSeedSpreadKThirty', float(_spread.loc[30, 'makespan_s']),
        'q2_seeds.csv K=30 档 makespan_s 极差（max-min）', nd=D_T)
    put('QtwoSeedEnergySpreadMax', float(_spread['能耗kWh'].max()),
        'q2_seeds.csv 各档 能耗kWh 极差中的最大值', nd=D_E)
    # 「多少个种子直接找到零加权时延的解」按档分组数，比报极差更能说明档间差异。
    put('QtwoSeedZeroTardKTwentyFive', int((sd[sd['K上限'] == 25]['加权时延'] == 0).sum()),
        'q2_seeds.csv K=25 档中 加权时延=0 的种子数', nd=0)
    put('QtwoSeedZeroTardKThirty', int((sd[sd['K上限'] == 30]['加权时延'] == 0).sum()),
        'q2_seeds.csv K=30 档中 加权时延=0 的种子数', nd=0)
    # K=30 档 5 个种子里有 4 个逐位相同——收敛证据要数「最常出现的那组指标出现了几次」，
    # 不能拿「极差很小」代替：极差小只说明彼此接近，不说明落在同一份解上。
    _k30 = sd[sd['K上限'] == 30][['架次', '能耗kWh', 'makespan_s', '加权时延']]
    put('QtwoSeedSameKThirty', int(_k30.value_counts().iloc[0]),
        'q2_seeds.csv K=30 档四项指标逐位相同的种子数（众数计数）', nd=0)
    # 完工时刻的极差在窄档并不大，真正把「种子差异」放大的是加权时延（第一优先级
    # 判据）：K<=20 档 5 个种子能差出 7 万余秒。若只报 makespan 极差，会把这一档
    # 的种子敏感性说轻。故两项都留宏，正文并排写。
    for K in (20, 25, 30):
        put('QtwoSeedTardSpreadK%s' % NUM_WORD[K],
            float(sd[sd['K上限'] == K]['加权时延'].max()
                  - sd[sd['K上限'] == K]['加权时延'].min()),
            'q2_seeds.csv K=%d 档 加权时延 极差（max-min）' % K, nd=1)
    # 20 行必须全部可行，否则「5 个种子」这句话本身就不成立，正文不能写「各档种子
    # 全部可行」。故这里不是记一个数，而是先当作断言查一遍。
    n_feas = int((sd['是否可行'] == '是').sum())
    if n_feas != len(sd):
        raise SystemExit('q2_seeds.csv 有 %d/%d 行不可行，正文不能称各档种子全部可行'
                         % (len(sd) - n_feas, len(sd)))
    put('QtwoSeedFeas', n_feas, 'q2_seeds.csv 可行种子数（=总行数）', nd=0)
    # 交付档的**档案条目**（定型轮之前）与交付方案（定型轮之后）是两个不同的解：
    # 前者是 5 个种子跑出来的、后者再过 F7＋退火。正文要说清「种子实验量的是哪一头」，
    # 否则「交付解是 5 个种子里最优的那个」这句话是错的——交付方案其实严格优于
    # 全部 5 个种子的档案条目。故两个数都从结果表取，不在此处写死。
    _a25 = _csv('q2_scan.csv')
    _a25 = _a25[_a25['K上限'] == 25]
    if len(_a25) != 1:
        raise SystemExit('q2_scan.csv 的 K=25 有 %d 行，应为 1 行' % len(_a25))
    put('QtwoArchKTwentyFiveEn', float(_a25['档案能耗kWh'].iloc[0]),
        'q2_scan.csv K=25 档档案条目能耗 kWh（定型轮之前）', nd=D_E)
    put('QtwoArchKTwentyFiveMk', float(_a25['档案makespan_s'].iloc[0]),
        'q2_scan.csv K=25 档档案条目 makespan s（定型轮之前）', nd=D_T)

    # ---------------- 问题三 ----------------
    q3 = _csv('q3_metrics.csv')
    r3 = _csv('q3_relay_trips.csv')
    c3 = _csv('q3_comm_phases.csv')
    # 中继架次 = **出动**数。阶段 13 起 q3_relay_trips.csv 一行 = 一次悬停站服务，
    # 一次出动可依次服务多站（表中以「出动编号」分组、行数 = 访问数）。论文里说的
    # 「N 个中继架次」指飞机出去几趟，故按出动编号去重；表无该列时退回行数。
    n_relay = (r3['出动编号'].nunique() if '出动编号' in r3.columns else len(r3))
    put('QthreeRelays', n_relay,
        'q3_relay_trips.csv 出动编号去重行数（= 飞机出去几趟）', nd=0)
    # 中继架次（表内行数）= 一次悬停站服务，与上面的「出动」是两个不同的量：
    # 一次出动可依次服务多站，故 行数 >= 出动数。论文 §9 表 9.2 报的「7 个中继架次」
    # 用的是这一个，摘要里必须先说「出动」再说「架次」，不能混用。
    put('QthreeRelayVisits', len(r3),
        'q3_relay_trips.csv 行数（一行 = 一次悬停站服务）', nd=0)
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
    # 同一个量留两个精度是有意的。它要说明的是「离硬时限还剩多少」：1 位小数的
    # 5.4 会把 5.353 的紧张感抹平，而且 5.4 > 5.353 属于**向上取整一个正在收紧的
    # 安全余量**，越是当门禁用的数越不该这么写。故摘要（main.tex）与 7.7 节、9.1
    # 节的表与正文一律用 3 位小数的 Fine 版；D_T 版的 1 位小数只作为与其它时刻
    # 并排时的备选，当前没有引用点。两处都取同一个来源列，不存在谁抄谁的问题。
    put('QthreeMinMarginFine', _metric(q3, '最小硬时限余量'),
        'q3_metrics.csv 最小硬时限余量（3 位小数）', nd=3)
    put('QthreeStaggered', int(_metric(q3, '错峰推迟架次数')),
        'q3_metrics.csv 错峰推迟架次数', nd=0)
    put('QthreeDelaySum', _metric(q3, '累计推迟量'), 'q3_metrics.csv 累计推迟量', nd=D_T)
    put('QthreeCoverageStep', float(_csv('q3_coverage.csv')['采样步长s'].max()),
        'q3_coverage.csv 采样步长s 最大值', nd=1)

    # ---- 问题三：逐箱交付对照的跨表统计（供 7.7 节「零中断的代价」引用）-----------
    # 这几个量要把 q2 与 q3 两张逐箱表按货箱编号相减才得出（「被推迟的 6 个箱里只有
    # 2 个带硬时限」）。本脚本的规矩是不在这里做业务计算——再算一遍就成了第二套口径，
    # 所以它们由做那次连接与门禁的 make_delivery_table.py 落盘，这里只读。
    _ds = _csv('delivery_stats.csv')

    def _dst(key):
        hit = _ds[_ds['指标'] == key]
        if len(hit) != 1:
            raise SystemExit('delivery_stats.csv 里「%s」出现 %d 次，应为 1 次'
                             % (key, len(hit)))
        return float(hit['数值'].iloc[0])

    put('QthreeHardBoxes', int(_dst('有硬时限货箱数')),
        'delivery_stats.csv 有硬时限货箱数', nd=0)
    put('QthreeFirstBoxes', int(_dst('首批货箱数')),
        'delivery_stats.csv 首批货箱数', nd=0)
    put('QthreeDelayedBoxes', int(_dst('被推迟货箱数')),
        'delivery_stats.csv 被推迟货箱数', nd=0)
    put('QthreeDelayedHardBoxes', int(_dst('被推迟且带硬时限货箱数')),
        'delivery_stats.csv 被推迟且带硬时限货箱数', nd=0)
    put('QthreeStaggerTrips', int(_dst('被推迟架次数')),
        'delivery_stats.csv 被推迟架次数', nd=0)
    put('QthreeLooseStaggerTrips', int(_dst('无非时限箱的被推迟架次数')),
        'delivery_stats.csv 无非时限箱的被推迟架次数', nd=0)
    put('QthreeStaggerMinMargin', _dst('被推迟架次的最紧硬时限余量'),
        'delivery_stats.csv 被推迟架次的最紧硬时限余量', nd=1)
    put('QthreeTightTripDelay', _dst('最紧箱所在架次的推迟量'),
        'delivery_stats.csv 最紧箱所在架次的推迟量', nd=1)
    # 跨文件自洽：逐箱表导出的最小余量必须与 q3_metrics.csv 的那一行逐位相同。
    # 若不等，说明两张表对「硬时限」的口径已经分叉，正文任一处的引用都会失真。
    # 容差取 5e-4 而不是 1e-6：q3_box_delivery.csv 的「硬时限余量s」列只保留 3 位
    # 小数（逐箱表要贴进论文，写 5.353271 只是噪声），q3_metrics.csv 保留原值。
    # 故这里比对的是「按 3 位小数发布后是否同值」，而不是浮点逐位相等。
    if abs(_dst('最紧箱硬时限余量') - _metric(q3, '最小硬时限余量')) > 5e-4:
        raise SystemExit('逐箱表的最小硬时限余量 %.6f 与 q3_metrics.csv 的 %.6f 不等'
                         % (_dst('最紧箱硬时限余量'), _metric(q3, '最小硬时限余量')))

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
    # 逐 K 的分区数单列两个宏：正文写「2047 与 86 526 个分区中可行者为 0」时，
    # 这两个数此前是手写的字面量，而它们本来就在 q4_comparison.csv 的「分区数」列里。
    # （用 iterrows 而不是 itertuples：中文字段名虽合法，但不必赌 namedtuple 的改名规则）
    for _i, _r in _csv('q4_comparison.csv').iterrows():
        put('QfourPartK%s' % NUM_WORD[int(_r['K'])], int(_r['分区数']),
            'q4_comparison.csv K=%d 的分区数' % int(_r['K']), nd=0)
        # 上面 QfourR / QfourCV 只有推荐档（K=2）一份，而图 fig:q4-partition 的
        # 两条虚线要同时标出两 K 的最优资源规模，正文也要把两档的均衡度并排比较。
        # 若在 LaTeX 里手写 33 与 0.8353，就又是一处「正文比结果表先过期」的隐患。
        put('QfourRK%s' % NUM_WORD[int(_r['K'])], int(_r['资源规模R']),
            'q4_comparison.csv K=%d 的资源规模R' % int(_r['K']), nd=0)
        put('QfourCVK%s' % NUM_WORD[int(_r['K'])], float(_r['工作量均衡CV_W']),
            'q4_comparison.csv K=%d 的工作量均衡CV_W' % int(_r['K']), nd=4)
    put('QfourAllPartitions', int(len(_csv('q4_all_partitions.csv'))),
        'q4_all_partitions.csv 行数', nd=0)
    put('QfourUnits', int(len(_csv('q4_units.csv'))), 'q4_units.csv 行数', nd=0)

    # ---- S4-c：库存—可行分区数阈值（全部取自 q4_sens.csv 的「扰动组/取值/指标」三元组）----
    # 一律按三元组精确取行，命中数不为 1 就报错——避免以后有人改了 make_q4_sens.py
    # 的取值格式，这里静默取到空表或最后一行。
    sens4 = _csv('q4_sens.csv')

    def _sv(group, setting, metric):
        hit = sens4[(sens4['扰动组'] == group) & (sens4['取值'] == setting)
                    & (sens4['指标'] == metric)]
        if len(hit) != 1:
            raise KeyError('q4_sens.csv 的 %s / %s / %s 命中 %d 行，应为 1 行'
                           % (group, setting, metric, len(hit)))
        return float(hit['数值'].iloc[0])

    def _thr(K, mm):
        return int(_sv('库存阈值·逐类各加m件', 'K=%d, m=%d' % (K, mm), '库存可行分区数'))

    for K, ms in ((2, (1, 2, 3)), (3, (1, 2, 3, 4, 5))):
        for mm in ms:
            put('QfourThreshK%sM%s' % (NUM_WORD[K], NUM_WORD[mm]), _thr(K, mm),
                'q4_sens.csv K=%d 逐类各加 %d 件后的可行分区数' % (K, mm), nd=0)
    # 「最接近可行」的分区与其补配向量：正文要写「补到 X 件时可行分区数为 Y」。
    for K in (2, 3):
        put('QfourMinPatchK%s' % NUM_WORD[K],
            int(_sv('库存阈值·最小补配总量', 'K=%d' % K, '最小补配件数')),
            'q4_sens.csv K=%d 让分区可行的最小补配总量（八类求和）' % K, nd=0)
        put('QfourMinPatchFeasK%s' % NUM_WORD[K],
            int(_sv('库存阈值·最小补配总量', 'K=%d' % K, '补至该向量后的可行分区数')),
            'q4_sens.csv K=%d 补到最小补配向量后的可行分区数' % K, nd=0)
        put('QfourMinPatchCVK%s' % NUM_WORD[K],
            _sv('库存阈值·最小补配总量', 'K=%d' % K, '该分区工作量均衡CV_W'),
            'q4_sens.csv K=%d 最接近可行的那个分区的 CV_W' % K, nd=4)
        put('QfourRecGapFeasK%s' % NUM_WORD[K],
            int(_sv('库存阈值·按推荐缺口补齐', 'K=%d' % K, '库存可行分区数')),
            'q4_sens.csv K=%d 按推荐分区自身缺口补齐后的可行分区数' % K, nd=0)
    # 逐类各补到「该类在全部分区上的最小需求」所需的补配之和。实测为 0：八类各自
    # 都不缺，可行的分区却仍是 0 —— 这个 0 是「缺口属联合约束而非单类短缺」的直接证据，
    # 正文要拿它和 QfourMinPatchKTwo/K3 并排写，故必须由脚本产出而不是手写。
    pv = [_sv('库存阈值·最小补配总量', 'K=%d' % K, '逐类最小补配之和（对照）')
          for K in (2, 3)]
    if pv[0] != pv[1]:
        raise ValueError('q4_sens.csv 两个 K 的「逐类最小补配之和」不同（%s），'
                         '单一宏装不下，须改为分 K 两个宏' % pv)
    put('QfourPerClassMinPatch', int(pv[0]),
        'q4_sens.csv 逐类各补到该类最小需求所需的补配件数（两 K 同值）', nd=0)

    # ------------------------------------------------------------------
    # S4-b / S4-d 问题三与通信侧的扰动（q3_sens.csv）
    # ------------------------------------------------------------------
    sens3 = _csv('q3_sens.csv')

    def _s3(group, setting, metric):
        hit = sens3[(sens3['扰动组'] == group) & (sens3['取值'] == setting)
                    & (sens3['指标'] == metric)]
        if len(hit) != 1:
            raise KeyError('q3_sens.csv 的 %s / %s / %s 命中 %d 行，应为 1 行'
                           % (group, setting, metric, len(hit)))
        return float(hit['数值'].iloc[0])

    # (i) 中继机数：超额站·秒 > 0 表示交付排班在 n_relay 架中继下排不下。
    put('QthreeExcessROne', _s3('中继机数', 'n_relay=1', '超额站·秒'),
        'q3_sens.csv 交付排班在 n_relay=1 下的超额站·秒', nd=1)
    put('QthreeExcessRTwo', _s3('中继机数', 'n_relay=2', '超额站·秒'),
        'q3_sens.csv 交付排班在 n_relay=2 下的超额站·秒', nd=1)
    put('QthreePeakStations', int(_s3('中继机数', '不随 n_relay 变', '同时最少站数峰值')),
        'q3_sens.csv 交付排班同一时刻最少需要的悬停站数峰值', nd=0)

    # (ii) 合并窗口：三档的取值在表里带后缀，故按实际字符串取，不能按数值拼。
    MW = ['gap=600, max_job=2400', 'gap=900, max_job=3600（交付档）', 'gap=1200, max_job=4800']
    mw_d = [_s3('合并窗口', t, '直连时间占比') for t in MW]
    mw_o = [_s3('合并窗口', t, '中继出动次数') for t in MW]
    put('QthreeMergeDirectRange', max(mw_d) - min(mw_d),
        'q3_sens.csv 三档合并窗口下直连时间占比的极差', nd=6)
    put('QthreeMergeOutingRange', int(max(mw_o) - min(mw_o)),
        'q3_sens.csv 三档合并窗口下中继出动次数的极差', nd=0)
    put('QthreeMergeOutings', int(mw_o[1]), 'q3_sens.csv 交付档合并窗口下的中继出动次数', nd=0)
    put('QthreeMergeServices', int(_s3('合并窗口', MW[1], '计划悬停站服务次数')),
        'q3_sens.csv 交付档合并窗口下的悬停站服务次数', nd=0)
    put('QthreeMergeChained', int(_s3('合并窗口', MW[1], '站间接续次数')),
        'q3_sens.csv 交付档合并窗口下的站间接续次数', nd=0)

    # (iii) 审计步长：零中断不是把步长挑大挑小挑出来的。
    DTS = ['Δt=0.5 s', 'Δt=1 s', 'Δt=2 s']
    dt_d = [_s3('审计步长', t, '直连时间占比') for t in DTS]
    put('QthreeDtDirectRange', max(dt_d) - min(dt_d),
        'q3_sens.csv 三档审计步长下直连时间占比的极差', nd=6)
    put('QthreeDtGapMax', max(_s3('审计步长', t, '中断时间占比') for t in DTS),
        'q3_sens.csv 三档审计步长下中断时间占比的最大值', nd=6)

    # (iv) 通信侧 ±3 dB。遮挡附加损耗 Lobs 与接收门限 Psens 在链路预算里各以
    # −1 的系数出现一次（core.py:587 的 Pth=Psens+M 进 :597 的 Lmax 时带负号，
    # :611 的 L_path 含 +Lobs·occ），故同一 dB 偏移下两者对每个裕量的作用**恒等**。
    # 这条断言把这个恒等关系钉死：一旦不成立，说明链路预算的写法变了，正文就不能
    # 再把两者并成「一组」来写，必须当成两次独立扰动分别交代。
    for lo, hi in (('Lobs-3 dB', '门限-3 dB'), ('Lobs+3 dB', '门限+3 dB')):
        a = _s3('通信·遮挡损耗', lo, '直连时间占比')
        b = _s3('通信·接收门限', hi, '直连时间占比')
        if abs(a - b) > 1e-9:
            raise SystemExit('通信侧两条扰动不再等价（%s: %.6f vs %s: %.6f），'
                             '正文不能合并表述' % (lo, a, hi, b))
    lo3 = _s3('通信·遮挡损耗', 'Lobs-3 dB', '直连时间占比')
    hi3 = _s3('通信·遮挡损耗', 'Lobs+3 dB', '直连时间占比')
    put('QthreeCommLoDirect', lo3, 'q3_sens.csv 链路预算放宽 3 dB 后的直连时间占比', nd=6)
    put('QthreeCommHiDirect', hi3, 'q3_sens.csv 链路预算收紧 3 dB 后的直连时间占比', nd=6)
    put('QthreeCommSpread', hi3 - lo3,
        'q3_sens.csv 链路预算 ±3 dB 引起的直连时间占比变化幅度', nd=6)
    # 「需中继占比 = 1 − 直连占比」是同一件事的另一面，正文两处各要用一面，故都留宏，
    # 由脚本现算以免有人只改了直连那个宏而忘了这一对。
    put('QthreeCommLoRelay', 1.0 - lo3,
        'q3_sens.csv 链路预算放宽 3 dB 后的需中继时间占比（= 1 − 直连）', nd=6)
    put('QthreeCommHiRelay', 1.0 - hi3,
        'q3_sens.csv 链路预算收紧 3 dB 后的需中继时间占比（= 1 − 直连）', nd=6)

    # (v) 交付解的最小裕量。图 fig_q3_margin 的 (b)(c) 两格要报「最小直连裕量」与
    # 「最小接入裕量」，这两个量不在任何交付表里，故由 make_q3_sens.py 逐采样点
    # 重算后落进 q3_sens.csv（该脚本只读调用求解器，不写任何方案记录）。
    put('QthreeMinDirectMargin', _s3('交付解·最小裕量', '全部架次取最小', '最小直连裕量'),
        'q3_sens.csv 24 个运输架次的最小直连裕量（逐 Δt=1 s 采样点取最小）', nd=4)
    put('QthreeDirectOkTrips',
        int(_s3('交付解·最小裕量', '全部架次', '最小直连裕量非负的架次数')),
        'q3_sens.csv 最小直连裕量 ≥ 0 的运输架次数（其余需中继）', nd=0)
    put('QthreeMinAccessMargin', _s3('交付解·最小裕量', '全部失效区间取最小', '最小接入裕量'),
        'q3_sens.csv 19 个失效区间的最小接入裕量（逐 Δt=1 s 采样点取最小）', nd=4)
    return m, src


# 宏名 → 正文里的含义（写进 JSON，便于「这个数字是干什么用的」当场可查）
DOC = {
    'QoneTrips': '问题一 总架次', 'QoneEnergy': '问题一 总能耗 kWh',
    'QoneMass': '问题一 总运载质量 kg', 'QoneMinSoc': '问题一 最小返航 SOC',
    'QoneMaxRound': '问题一 单架次最长往返时间 s', 'QoneWorkTimeH': '问题一 总作业时间 h',
    'QoneWorkTimeS': '问题一 总作业时间 s', 'QoneAreas': '服务区总数',
    'QoneIlpAgree': '问题一 与 ILP 复核一致的区数', 'QoneIlpCases': '问题一 参与复核的区数',
    'QoneIlpGapCases': '问题一 逐机型逐区算例数',
    'QoneSensRhoDropA': 'A 型 仅 ρ 劣化的平均载荷降幅 kg',
    'QoneSensRhoDropB': 'B 型 仅 ρ 劣化的平均载荷降幅 kg',
    'QoneSensRhoDropC': 'C 型 仅 ρ 劣化的平均载荷降幅 kg',
    'QoneSensKapDropA': 'A 型 仅 κ 劣化的平均载荷降幅 kg',
    'QoneSensKapDropB': 'B 型 仅 κ 劣化的平均载荷降幅 kg',
    'QoneSensKapDropC': 'C 型 仅 κ 劣化的平均载荷降幅 kg',
    'QoneSensJointDropA': 'A 型 两因素同时劣化的平均载荷降幅 kg',
    'QoneSensJointDropB': 'B 型 两因素同时劣化的平均载荷降幅 kg',
    'QoneSensJointDropC': 'C 型 两因素同时劣化的平均载荷降幅 kg',
    'QoneSensRatioA': 'A 型 联合降幅 ÷ 两项降幅之和',
    'QoneSensRatioB': 'B 型 联合降幅 ÷ 两项降幅之和',
    'QoneSensRatioC': 'C 型 联合降幅 ÷ 两项降幅之和',
    'QoneSensRatioMax': '三个耦合比值中的最大值',
    'QoneSensExcessPct': '联合降幅超出两项之和的百分点',
    'QtwoTrips': '问题二 运输架次数', 'QtwoUavs': '运输无人机架数',
    'QtwoEnergy': '问题二 总能耗 kWh', 'QtwoMakespan': '问题二 完工时刻 s',
    'QtwoTardiness': '问题二 加权时延', 'QtwoHardViol': '问题二 硬约束违反数',
    'QtwoBoxes': '交付箱次', 'QtwoSeed': 'ALNS 随机种子',
    'QtwoScanRows': 'ε-约束扫描档数', 'QtwoParetoRows': '非支配前沿档数',
    'QtwoRecK': '推荐档 K 上限', 'QtwoRecTrips': '推荐档实际架次',
    'QtwoRecEnergy': '推荐档能耗 kWh', 'QtwoRecMakespan': '推荐档完工时刻 s',
    'QtwoExactCases': '精确验证算例数', 'QtwoExactTimeout': '超时算例数',
    'QtwoExactProved': '已证最优算例数',
    'QtwoSeedRuns': '多种子实验的 (档, 种子) 组合数',
    'QtwoSeedSpreadKTwenty': 'K=20 档 5 种子的完工时刻极差 s',
    'QtwoSeedSpreadKTwentyFive': 'K=25 档 5 种子的完工时刻极差 s',
    'QtwoSeedSpreadKThirty': 'K=30 档 5 种子的完工时刻极差 s',
    'QtwoSeedEnergySpreadMax': '各档 5 种子能耗极差的最大值 kWh',
    'QtwoSeedZeroTardKTwentyFive': 'K=25 档 5 种子中零加权时延的个数',
    'QtwoSeedZeroTardKThirty': 'K=30 档 5 种子中零加权时延的个数',
    'QtwoSeedSameKThirty': 'K=30 档四项指标逐位相同的种子数',
    'QthreeRelays': '问题三 中继出动次数', 'QthreeStations': '悬停站数',
    'QthreeRelayVisits': '问题三 中继架次数（= 悬停站服务次数）',
    'QthreeSegs': '通信分段数', 'QthreeIntervals': '通信失效区间数',
    'QthreeDirect': '直连时间占比', 'QthreeRelayFrac': '中继时间占比',
    'QthreeGap': '中断时间占比', 'QthreeTransEnergy': '问题三 运输能耗 kWh',
    'QthreeRelayEnergy': '问题三 中继能耗 kWh', 'QthreeJointEnergy': '问题三 合计能耗 kWh',
    'QthreeTransDone': '运输完工时刻 s', 'QthreeJointDone': '联合完工时刻 s',
    'QthreeTardiness': '问题三 加权迟到', 'QthreeLateBoxes': '迟到箱数',
    'QthreeMaxLate': '最长迟到 s', 'QthreeMinMargin': '最小硬时限余量 s',
    'QthreeMinMarginFine': '最小硬时限余量 s（3 位小数，与硬时限并排读时用）',
    'QthreeStaggered': '错峰推迟架次数', 'QthreeDelaySum': '累计推迟量 s',
    'QthreeCoverageStep': '覆盖核算采样步长 s',
    # 7.7 节逐箱交付对照的跨表统计（来源 results/delivery_stats.csv，见 make_delivery_table.py）
    'QthreeHardBoxes': '有硬时限的货箱数',
    'QthreeFirstBoxes': '其中有硬时限的首批货箱数',
    'QthreeDelayedBoxes': '第三问中被推迟的货箱数',
    'QthreeDelayedHardBoxes': '被推迟货箱中带硬时限者',
    'QthreeStaggerTrips': '被错峰推迟的架次数',
    'QthreeLooseStaggerTrips': '被推迟架次中一个硬时限箱都不载者',
    'QthreeStaggerMinMargin': '被推迟架次上最紧的硬时限余量 s',
    'QthreeTightTripDelay': '最紧箱所在架次的推迟量 s',
    'QfourK': '问题四 分区数 K', 'QfourR': '问题四 资源规模 R',
    'QfourCV': '问题四 工作量均衡 CV', 'QfourRelayNeed': '中继无人机总需求',
    'QfourGapSum': '单份库存下的总量缺口', 'QfourRedundancy': '总量冗余',
    'QfourFeasible': '库存可行分区数', 'QfourPartitions': '入选分区数',
    'QfourAllPartitions': '枚举分区总数', 'QfourUnits': '任务单元数',
    'QfourPartKTwo': 'K=2 的分区数', 'QfourPartKThree': 'K=3 的分区数',
    'QfourRKTwo': 'K=2 最优分区的资源规模 R',
    'QfourRKThree': 'K=3 最优分区的资源规模 R',
    'QfourCVKTwo': 'K=2 最优分区的工作量均衡 CV',
    'QfourCVKThree': 'K=3 最优分区的工作量均衡 CV',
    # S4-c 库存阈值：逐类各加 m 件后的可行分区数（正文给出的就是这条阈值曲线）
    'QfourThreshKTwoMOne': 'K=2 逐类各加 1 件后可行分区数',
    'QfourThreshKTwoMTwo': 'K=2 逐类各加 2 件后可行分区数',
    'QfourThreshKTwoMThree': 'K=2 逐类各加 3 件后可行分区数',
    'QfourThreshKThreeMOne': 'K=3 逐类各加 1 件后可行分区数',
    'QfourThreshKThreeMTwo': 'K=3 逐类各加 2 件后可行分区数',
    'QfourThreshKThreeMThree': 'K=3 逐类各加 3 件后可行分区数',
    'QfourThreshKThreeMFour': 'K=3 逐类各加 4 件后可行分区数',
    'QfourThreshKThreeMFive': 'K=3 逐类各加 5 件后可行分区数',
    'QfourMinPatchKTwo': 'K=2 让某个分区可行的最小补配件数',
    'QfourMinPatchKThree': 'K=3 让某个分区可行的最小补配件数',
    'QfourMinPatchFeasKTwo': 'K=2 补到最小补配向量后的可行分区数',
    'QfourMinPatchFeasKThree': 'K=3 补到最小补配向量后的可行分区数',
    'QfourMinPatchCVKTwo': 'K=2 最接近可行的那个分区的 CV_W',
    'QfourMinPatchCVKThree': 'K=3 最接近可行的那个分区的 CV_W',
    'QfourRecGapFeasKTwo': 'K=2 按推荐缺口补齐后的可行分区数',
    'QfourRecGapFeasKThree': 'K=3 按推荐缺口补齐后的可行分区数',
    'QfourPerClassMinPatch': '逐类补到该类最小值所需的补配件数（实测 0）',
    # S4-a 问题二 5 种子（16 个极差由 collect() 里同一个循环产出，说明文字同构，
    # 故在此按同一模板补 DOC，免得 16 行近乎重复的字面量把上面的表冲淡）
    'QtwoSeedFeas': 'q2_seeds.csv 中可行的种子数（=总行数）',
    'QtwoArchKTwentyFiveEn': 'K=25 档档案条目能耗 kWh（定型轮之前，区别于交付方案）',
    'QtwoArchKTwentyFiveMk': 'K=25 档档案条目完工时刻 s（定型轮之前）',
    # S4-b / S4-d 问题三与通信侧
    'QthreeExcessROne': '交付排班在 1 架中继下的超额站·秒（>0 即排不下）',
    'QthreeExcessRTwo': '交付排班在 2 架中继下的超额站·秒',
    'QthreePeakStations': '交付排班同一时刻最少需要的悬停站数峰值',
    'QthreeMergeDirectRange': '三档合并窗口下直连时间占比的极差',
    'QthreeMergeOutingRange': '三档合并窗口下中继出动次数的极差',
    'QthreeMergeOutings': '交付档合并窗口下的中继出动次数',
    'QthreeMergeServices': '交付档合并窗口下的悬停站服务次数',
    'QthreeMergeChained': '交付档合并窗口下的站间接续次数',
    'QthreeDtDirectRange': '三档审计步长下直连时间占比的极差',
    'QthreeDtGapMax': '三档审计步长下中断时间占比的最大值',
    'QthreeCommLoDirect': '链路预算放宽 3 dB 后的直连时间占比',
    'QthreeCommHiDirect': '链路预算收紧 3 dB 后的直连时间占比',
    'QthreeCommSpread': '链路预算 ±3 dB 引起的直连时间占比变化幅度',
    'QthreeCommLoRelay': '链路预算放宽 3 dB 后的需中继时间占比',
    'QthreeCommHiRelay': '链路预算收紧 3 dB 后的需中继时间占比',
    'QthreeMinDirectMargin': '运输架次的最小直连裕量 dB（负值即该架次某段无直连）',
    'QthreeDirectOkTrips': '最小直连裕量 ≥ 0 的运输架次数（其余需中继）',
    'QthreeMinAccessMargin': '失效区间的最小中继接入裕量 dB',
}
# 逐档的加权时延极差。K=22 档 5 个种子的加权时延逐位相同（极差 0），正文不引用，故不留空宏。
for _K in (20, 25, 30):
    DOC['QtwoSeedTardSpreadK%s' % NUM_WORD[_K]] = 'K=%d 档 5 个种子的加权时延极差 s' % _K


def main():
    m, src = collect()
    # 宏名一律不含数字，理由见 NUM_WORD 处的注释：带数字的名字在 TeX 里会被读成
    # 「控制序列 + 普通文本」，数字落进导言区，报出的却是**下一条** \newcommand
    # 上的 Missing \begin{document}。这条断言把该类缺陷挡在生成环节，而不是等到
    # 109 条报错之后再回头逐个宏名去查。
    bad = sorted(k for k in m if not k.isalpha())
    if bad:
        raise SystemExit('以下宏名含非字母字符，TeX 无法把它们当作控制序列名：%s\n'
                         '请按 NUM_WORD 的拼写表改名（如 K25 → KTwentyFive）。' % bad)
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
