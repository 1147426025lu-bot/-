# -*- coding: utf-8 -*-
# 本程序及代码是在人工智能工具辅助下完成的。
# 工具名称：DeepSeek；版本/型号：DeepSeek V4.1 Flash（API 模型名 deepseek-flash）；
# 开发机构：杭州深度求索人工智能基础技术研究有限公司；
# 版本发布日期：2026 年 9 月 10 日。

"""问题四的库存—可行分区数阈值曲线（只读派生，不写任何方案记录）。

为什么要有这个脚本
------------------
论文的问题四结论是「两种分区下都没有一个总量口径可行的分区」——这是一个
**否定式**结论，读者读完只知道「不行」，不知道「差多远、补到什么程度就行」。
本脚本把这句话升级为可执行的阈值：**逐类各加 m 件时，有多少个分区能落地**，
以及**最少补哪几件、补到哪个向量时第一个可行分区出现**。否定式结论加上阈值
才有决策价值，也才看得出缺口是「某一类短缺」还是「联合约束」造成的。

「联合约束」这一条是本脚本要单独说清的事
----------------------------------------
`q4.py` 已经把八类资源在**全部分区上的最小需求**算了出来（`min_tot`），实测
两个 K 下八类最小值**全部不超过库存**——也就是逐类看，库存没有一类是不够的。
但可行的分区数仍是 0。原因是各类的最小值落在**不同**的分区上：没有任何一个
分区能同时把所有类都压到各自的最小值。故本脚本把两个量并排列出：
「逐类各补到该类最小值所需的补配之和」（= 0）与「让某一个分区可行所需的最小
补配总量」（> 0）。两者之差就是联合约束的代价。把前者当成结论，会写出
「库存够用」这种与事实相反的判断。

三条硬边界的执行方式
--------------------
1.  **不改任何求解器源码**。`core.py`/`q2.py`/`q3.py`/`solution_io.py` 等是字节
    入哈希的，改一个字节就作废已交付的方案记录。本脚本只**调用** `q4.py` 的既有
    函数（`atomic_units`/`partition_sets`/`group_resources`/`workload` 等），并把
    `q4.py main()` 的枚举层原样重走一遍——`q4.py` 本身不在哈希表内，但也没有被
    修改，本脚本只是复用它。
2.  **`save=False` 语义**：不调用任何导出函数，不写 `results/*_solution.json`，也
    不覆盖 `q4_partition.csv`/`q4_all_partitions.csv` 等正式产物；产物只有
    `results/q4_sens.csv` 一个。
3.  **先对上门禁再报阈值**。先按 `q4.py main()` 的同一条路径重算，把
    「库存可行分区数」「推荐分区的 R 与 CV_W」「逐类全分区最小需求」「推荐分区总量
    缺口」与 `q4_comparison.csv` 逐位对比；对不上就**拒绝落盘并如实报告**——基线
    对不上的阈值表会把「补配的影响」与「基线的变化」混在一起，不能当证据。

口径
----
「可行」一律指**总量口径**：把该分区各组的同类需求相加得 N_r = Σ_g n[g,r]，
与**单份**库存 S_r 比，N_r ≤ S_r 对八类全部成立才算可行。这与
`q4_comparison.csv` 的「库存可行分区数」同口径。分组独立口径（每组各自配齐）
是另一件事，本脚本不混用。

用法
----
    python code/make_q4_sens.py            # 在工程根目录下运行
退出码 0 = 门禁通过并已写出 results/q4_sens.csv；1 = 门禁不通过（未写文件）。
"""
from __future__ import annotations

import os
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
RES = os.path.join(ROOT, 'results')
sys.path.insert(0, HERE)

import q4                         # noqa: E402  只读复用其枚举/核算层
from core import load_data        # noqa: E402
from q2 import precompute_geometry  # noqa: E402
from solution_io import (load_solution, input_hashes, solver_hashes,  # noqa: E402
                         solver_hashes_text, Q3_SOLVER_FILES)

Q3_SOL = os.path.join(RES, 'q3_solution.json')

# 逐类各加 m 件的扫描上限：到 n_part 就停，这里只是防跑飞的兜底。
M_CAP = 12

# 门禁容差：计数必须逐位相等；CV_W 是浮点，按 1e-6 判。
TOL = 1e-6

CAL_UNIFORM = '现库存逐类各加 m 件后，重数全部 K 组分区中总量口径可行者；不含重解'
CAL_RECGAP = '现库存 + 该 K 下推荐分区自己的总量缺口向量；不含重解'
CAL_PERVEC = '逐类各补到「该类在全部分区上的最小需求」；不含重解'
CAL_THRESH = '让「最接近可行」的那个分区落地的补配向量；不含重解'


def _rows(group, setting, metric, value, unit, caliber, note=''):
    """一行 = (扰动组, 取值, 指标, 数值)。长表比宽表好接：加一组不用改表结构。"""
    return dict(扰动组=group, 取值=setting, 指标=metric,
                数值=float(value), 单位=unit, 扰动口径=caliber, 备注=note)


def _load_cmp():
    """读 q4_comparison.csv，返回 {K: 该行的 Series}。"""
    df = pd.read_csv(os.path.join(RES, 'q4_comparison.csv'), encoding='utf-8-sig')
    out = {}
    for _, r in df.iterrows():
        out[int(r['K'])] = r
    if set(out) != {2, 3}:
        raise SystemExit('q4_comparison.csv 的 K 取值为 %s，应为 {2, 3}' % sorted(out))
    return out


def main():
    t_all = time.time()
    d = load_data()
    d.geo_nodes, d.geo = precompute_geometry(d)

    # ---- 1) 方案来源与 `q4.py main()` 完全一致：读问题三落盘的完整方案 ----
    q3_sol = load_solution(Q3_SOL, stage='q3',
                           input_hashes_expect=input_hashes(),
                           solver_hashes_expect=solver_hashes(Q3_SOLVER_FILES),
                           solver_text_hashes_expect=solver_hashes_text(Q3_SOLVER_FILES))
    print('读取问题三方案 %s（%d 运输架次 / %d 中继架次）'
          % (q3_sol['solution_id'], len(q3_sol['transport_trips']),
             len(q3_sol['relay_trips'])))

    assignment = q4.build_assignment(q3_sol, d)
    relays = q4.build_relays(q3_sol, d)
    relay_of_trip = q4.join_relays(q3_sol, relays)

    units = q4.atomic_units(d, assignment)
    n = len(units)
    print('运输架次 %d 个；原子任务单元 %d 个' % (len(assignment), n))

    # ---- 2) 库存：从原始数据推出，不写死（与 q4.py 逐行同源）----
    stock = {}
    for g in q4.TYPES:
        stock[f'{g}_uav'] = sum(1 for u in d.uavs if u['type'] == g)
        stock[f'{g}_bat'] = int(d.batteries[g][0])
    stock['relay'] = len(d.relay_uavs)
    stock['relay_comp'] = int(d.relay_batteries[0])
    stock_vec = np.array([stock[k] for k in q4.RES_KEYS], dtype=np.int64)
    print('库存: ' + '  '.join('%s=%d' % (q4.RES_LABEL[k], stock[k]) for k in q4.RES_KEYS))

    # ---- 3) 单元 → 该单元的架次 / 中继架次（照抄 q4.py）----
    unit_trips, unit_relays = [], []
    for u in units:
        us = set(u)
        tids = [a['idx'] for a in assignment if set(a['route']) <= us]
        unit_trips.append(tids)
        rr, seen = [], set()
        for k in tids:
            for x in relay_of_trip[assignment[k]['trip_id']]:
                if x['id'] not in seen:
                    seen.add(x['id'])
                    rr.append(x)
        unit_relays.append(rr)

    cache = {}

    def eval_group(members):
        key = tuple(sorted(members))
        if key in cache:
            return cache[key]
        tids = [k for m in members for k in unit_trips[m]]
        rr, seen = [], set()
        for m in members:
            for x in unit_relays[m]:
                if x['id'] not in seen:
                    seen.add(x['id'])
                    rr.append(x)
        res = q4.group_resources(tids, assignment, rr)
        w = q4.workload(tids, assignment, rr)
        out = (res, w, tids, rr)
        cache[key] = out
        return out

    # ---- 4) 枚举全部 K 组分区，留档每个分区的八类需求向量 ----
    # q4_all_partitions.csv 只有 5 列（R / CV_W / 中继总需求 / 缺口合计），
    # 没有逐类需求向量，因此阈值的计数必须在这里重走一遍枚举层，不能靠读表。
    enum = {}
    for K in (2, 3):
        t0 = time.time()
        tots, Rs, cvws, best, n_feas = [], [], [], None, 0
        min_tot = {k: 10 ** 9 for k in q4.RES_KEYS}
        for groups in q4.partition_sets(units, K):
            res_g, w_g = [], []
            tot = {k: 0 for k in q4.RES_KEYS}
            for members in groups:
                res, w, _, _ = eval_group(members)
                res_g.append(res)
                w_g.append(w)
                for k in q4.RES_KEYS:
                    tot[k] += res[k]
            R = sum(tot.values())
            for k in q4.RES_KEYS:
                if tot[k] < min_tot[k]:
                    min_tot[k] = tot[k]
            W = np.array(w_g, dtype=float)
            cvw = float(W.std(ddof=0) / W.mean()) if W.mean() > 0 else 0.0
            short = {k: max(0, tot[k] - stock[k]) for k in q4.RES_KEYS}
            if not any(short.values()):
                n_feas += 1
            tots.append([tot[k] for k in q4.RES_KEYS])
            Rs.append(R)
            cvws.append(cvw)
            cand = dict(K=K, groups=[list(g) for g in groups], res=res_g, tot=tot,
                        short=short, R=R, cvw=cvw)
            if best is None or (cand['R'], cand['cvw']) < (best['R'], best['cvw']):
                best = cand
        enum[K] = dict(tot=np.array(tots, dtype=np.int64), R=np.array(Rs, dtype=np.int64),
                       cvw=np.array(cvws, dtype=float), best=best,
                       n_feas=n_feas, min_tot=min_tot, n_part=len(tots))
        print('K=%d：枚举 %d 个分区（%.1f s），推荐 R=%d、CV_W=%.4f，'
              '库存可行 %d 个' % (K, len(tots), time.time() - t0, best['R'], best['cvw'], n_feas))

    # ---- 5) 门禁：重算的基线必须与 q4_comparison.csv 逐位一致 ----
    cmp_row = _load_cmp()
    print('=' * 74)
    print('门禁：重算基线 vs q4_comparison.csv')
    ok = True
    for K in (2, 3):
        e, r = enum[K], cmp_row[K]
        checks = [
            ('分区数', e['n_part'], int(r['分区数'])),
            ('库存可行分区数', e['n_feas'], int(r['库存可行分区数'])),
            ('推荐分区资源规模R', e['best']['R'], int(r['资源规模R'])),
            ('推荐分区总量缺口', sum(e['best']['short'].values()), int(r['总量缺口'])),
        ]
        for name, got, exp in checks:
            good = got == exp
            ok &= good
            print('  [%s] K=%d %-14s 重算=%s 交付=%s'
                  % ('OK  ' if good else 'FAIL', K, name, got, exp))
        good = abs(e['best']['cvw'] - float(r['工作量均衡CV_W'])) <= TOL
        ok &= good
        print('  [%s] K=%d %-14s 重算=%.6f 交付=%.6f'
              % ('OK  ' if good else 'FAIL', K, '工作量均衡CV_W', e['best']['cvw'],
                 float(r['工作量均衡CV_W'])))
        for k in q4.RES_KEYS:
            got, exp = e['min_tot'][k], int(r['最小需求_' + q4.RES_LABEL[k]])
            good = got == exp
            ok &= good
            if not good:
                print('  [FAIL] K=%d 最小需求_%s 重算=%d 交付=%d'
                      % (K, q4.RES_LABEL[k], got, exp))
        print('  [OK  ] K=%d 八类「全分区最小需求」与交付表逐位一致' % K
              if ok else '  [FAIL] K=%d 存在不一致的「最小需求」' % K)
    print('=' * 74)
    if not ok:
        print('门禁不通过：当前代码重算不出已交付的问题四枚举结果。')
        print('按本脚本的纪律，此时不落盘 results/q4_sens.csv——'
              '基线对不上的阈值表会把「补配的影响」与「换了基线」混在一起，不能当证据。')
        return 1

    # ---- 6) 阈值曲线 ----
    rows = []
    for K in (2, 3):
        e = enum[K]
        tot, n_part = e['tot'], e['n_part']

        # (i) 逐类各加 m 件：可行分区数随 m 的曲线
        print('-' * 74)
        print('(i) K=%d：逐类各加 m 件后的可行分区数' % K)
        m = 0
        while True:
            c = int(np.all(tot <= (stock_vec + m), axis=1).sum())
            print('  m=%d: 可行 %d / %d 个分区' % (m, c, n_part))
            rows.append(_rows('库存阈值·逐类各加m件', 'K=%d, m=%d' % (K, m),
                              '库存可行分区数', c, '个', CAL_UNIFORM,
                              '全部 %d 个分区中' % n_part))
            if c == n_part or m >= M_CAP:
                if c < n_part:
                    print('  ! m 到上限 %d 仍未全部可行，曲线按实测截断' % M_CAP)
                break
            m += 1

        # (ii) 按该 K 下推荐分区自己的总量缺口补齐
        gap_vec = np.array([e['best']['short'][k] for k in q4.RES_KEYS], dtype=np.int64)
        c = int(np.all(tot <= (stock_vec + gap_vec), axis=1).sum())
        gtxt = '、'.join('%s+%d' % (q4.RES_LABEL[k], gap_vec[i])
                        for i, k in enumerate(q4.RES_KEYS) if gap_vec[i] > 0)
        print('(ii) K=%d：按推荐缺口（%s）补齐后，可行 %d / %d 个分区'
              % (K, gtxt or '无需补配', c, n_part))
        rows.append(_rows('库存阈值·按推荐缺口补齐', 'K=%d' % K, '库存可行分区数', c,
                          '个', CAL_RECGAP, '补配向量：' + (gtxt or '无需补配')))

        # (iii) 对照：逐类各补到该类在全部分区上的最小值
        # 实测八类 min_tot 全部 <= 库存，故这个方案等于「不补」——列出它是为了
        # 把「逐类都够」与「没有一个分区同时够」并排放在同一张表里。
        per_vec = np.array([max(0, e['min_tot'][k] - stock[k]) for k in q4.RES_KEYS],
                           dtype=np.int64)
        c = int(np.all(tot <= (stock_vec + per_vec), axis=1).sum())
        print('(iii) K=%d：逐类各补到该类最小值（共 %d 件）后，可行 %d / %d 个分区'
              % (K, int(per_vec.sum()), c, n_part))
        rows.append(_rows('库存阈值·逐类补到最小值', 'K=%d' % K, '库存可行分区数', c,
                          '个', CAL_PERVEC,
                          '八类各自的补配合计仅 %d 件，但落不到同一个分区上' % int(per_vec.sum())))

        # (iv) 真正的最小补配：让「最接近可行」的那个分区落地的向量与代价
        short_mat = np.maximum(0, tot - stock_vec)
        tot_short = short_mat.sum(axis=1)
        i_min = int(np.argmin(tot_short))
        vec = short_mat[i_min]
        c = int(np.all(tot <= (stock_vec + vec), axis=1).sum())
        vtxt = '、'.join('%s+%d' % (q4.RES_LABEL[k], vec[i])
                        for i, k in enumerate(q4.RES_KEYS) if vec[i] > 0)
        print('(iv) K=%d：最小补配总量 %d 件（%s）⇒ 可行分区数 %d'
              % (K, int(vec.sum()), vtxt or '无', c))
        rows.append(_rows('库存阈值·最小补配总量', 'K=%d' % K, '最小补配件数',
                          int(vec.sum()), '件', CAL_THRESH, '八类求和；补配向量见下'))
        rows.append(_rows('库存阈值·最小补配总量', 'K=%d' % K, '补至该向量后的可行分区数',
                          c, '个', CAL_THRESH, '补配向量：' + (vtxt or '无')))
        # 该分区自身的两个指标单列成行：正文要拿它和推荐分区比（同为 R=33 时
        # CV_W 谁更小），写进备注自由文本就没法被 paper_metrics 取值。
        rows.append(_rows('库存阈值·最小补配总量', 'K=%d' % K, '该分区资源规模R',
                          int(e['R'][i_min]), '台', CAL_THRESH, '最接近可行的那个分区'))
        rows.append(_rows('库存阈值·最小补配总量', 'K=%d' % K, '该分区工作量均衡CV_W',
                          float(e['cvw'][i_min]), '—', CAL_THRESH, '最接近可行的那个分区'))
        for i, k in enumerate(q4.RES_KEYS):
            rows.append(_rows('库存阈值·最小补配总量', 'K=%d' % K,
                              '补配_' + q4.RES_LABEL[k], int(vec[i]), '件', CAL_THRESH,
                              '该分区 R=%d、CV_W=%.4f'
                              % (int(e['R'][i_min]), float(e['cvw'][i_min]))))
        # 对照量：逐类最小补配之和 vs 最小补配总量。前者 = 0 而后者 > 0，说明
        # 缺口不是任何一类短缺，是「各类最小值不在同一个分区上」的联合约束。
        rows.append(_rows('库存阈值·最小补配总量', 'K=%d' % K, '逐类最小补配之和（对照）',
                          int(per_vec.sum()), '件', CAL_THRESH,
                          '逐类看都不缺；与上一行之差即联合约束的代价'))

    out = pd.DataFrame(rows)
    path = os.path.join(RES, 'q4_sens.csv')
    out.to_csv(path, index=False)
    print('=' * 74)
    print('已写出 %s（%d 行）' % (os.path.relpath(path, ROOT), len(out)))
    print('总耗时 %.1f s' % (time.time() - t_all))
    return 0


if __name__ == '__main__':
    sys.exit(main())
