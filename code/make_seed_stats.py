# -*- coding: utf-8 -*-
# 本程序及代码是在人工智能工具辅助下完成的。
# 工具名称：DeepSeek；版本/型号：DeepSeek V4.1 Flash（API 模型名 deepseek-flash）；
# 开发机构：杭州深度求索人工智能基础技术研究有限公司；
# 版本发布日期：2026 年 9 月 10 日。

"""问题二多种子稳定性记录（只读派生，不写任何方案记录）。

为什么要有这个脚本
------------------
`results/q2_alns_trace.csv` 是**单次运行**的收敛曲线（`档案最优目标值` 全程恒定，
说明它记的是那一次搜索的过程），只贴这一条曲线无法回答「换一组种子结果会不会变」。
本题的交付解是在 5 个种子 × 4 个 K 档上跑出来的，每个种子的末态指标其实**已经记在
`info['runs'][K]['seeds']` 里**，只是从来没有落盘，所以论文里只能写「本文不做多种子
对照实验」。本脚本把这批已经算出来、却一直被丢掉的数字取出来写成 `q2_seeds.csv`。

与 solve_recommended_v2 的关系（不许改写的三条）
-----------------------------------------------
1.  **`save=False`**：绝不落盘 `results/q2_solution.json` 或任何正式产物。本脚本是
    只读派生，方案记录的字节由 `code/q2.py` 单独决定，两边不许互相污染。
2.  **`relay_gate=False`**：中继可行择序发生在档循环**之后**（`q2_v2.py` 里
    `relay_feasible_order` 的调用点在 `runs` 全部建好之后），故关掉它不改变任何
    种子记录，只是省掉一段与本脚本无关的开销。种子序列本身照旧由
    `q2.ALNS_SEED + K*101 + s*7919` 给出，与正式轮**逐位相同**。
3.  **调用参数逐项照抄正式轮**（`results/_交付与复现说明.txt:15`）：
    `--engine v2 --seeds 5 --iters 3500 --k-targets 20,22,25,30`。

硬门禁（不通过就不写文件）
--------------------------
写出前逐档核对 `results/q2_scan.csv` 的 `档案架次 / 档案能耗kWh / 档案makespan_s /
档案加权时延`——这四列是正式轮落盘的档案第一条。本脚本重跑同一套种子，若对不上，
说明**当前代码算不出已交付的那份结果**，此时正确做法是拒绝落盘并如实报告，
而不是把一份对不上的表写进 `results/` 冒充证据。

用法
----
    python code/make_seed_stats.py            # 在工程根目录下运行
退出码 0 = 门禁通过并已写出 `results/q2_seeds.csv`；1 = 门禁不通过（未写文件）。
"""
from __future__ import annotations

import os
import sys
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from core import T_DEC, E_DEC, load_data          # noqa: E402
import q2                                          # noqa: E402
import q2_v2                                       # noqa: E402

OUT = os.path.join(HERE, '..', 'results')
SCAN = os.path.join(OUT, 'q2_scan.csv')
DEST = os.path.join(OUT, 'q2_seeds.csv')

# 正式轮参数（照抄 _交付与复现说明.txt:15），改任何一项都会让下面的门禁失去意义
SEEDS = 5
N_ITER = 3500
K_TARGETS = [20, 22, 25, 30]
SA_ITERS = 4000
PATIENCE = max(1500, N_ITER // 3)

# 门禁比对的四列：(q2_scan.csv 列名, 种子记录里的键, 小数位)
GATE_COLS = [
    ('档案架次', 'n_trips', None),
    ('档案能耗kWh', 'total_E', E_DEC),
    ('档案makespan_s', 'makespan', T_DEC),
    ('档案加权时延', 'tardiness', 3),
]


def check_gate(runs: dict) -> list[str]:
    """逐档核对档案第一条与 q2_scan.csv。返回不一致的描述列表（空 = 通过）。"""
    if not os.path.exists(SCAN):
        return ['找不到 %s，无法核对' % os.path.relpath(SCAN, os.path.dirname(HERE))]
    ref = pd.read_csv(SCAN).set_index('K上限')
    bad = []
    for K in sorted(runs):
        got = runs[K]['archive'][0]['metrics'] if runs[K]['archive'] else None
        if K not in ref.index:
            bad.append('K=%d：q2_scan.csv 里没有这一档' % K)
            continue
        row = ref.loc[K]
        if got is None:
            if bool(row['档案可行']) if '档案可行' in row else True:
                bad.append('K=%d：正式轮有档案，本次无可行解' % K)
            continue
        for col, key, dec in GATE_COLS:
            want = row[col]
            if pd.isna(want):
                continue
            mine = got[key] if dec is None else round(got[key], dec)
            if abs(float(mine) - float(want)) > 1e-9:
                bad.append('K=%d %s：本次 %r ≠ 正式轮 %r' % (K, col, mine, want))
    return bad


def main() -> int:
    t0 = time.time()
    print('=' * 74)
    print('问题二多种子稳定性记录（%d 种子 × %s 档，%d 轮，耐心 %d）'
          % (SEEDS, K_TARGETS, N_ITER, PATIENCE))
    print('  本脚本 save=False、不写任何方案记录；参数照抄正式轮。')
    print('=' * 74)

    d = load_data()
    # 先算几何：self_check 与 ALNS 都要读 d.geo（与 q2.py:1329 的次序一致）
    d.geo_nodes, d.geo = q2.precompute_geometry(d)
    assert q2_v2.self_check(d, verbose=True), 'v2 引擎自检未通过'

    st, asg, m, info = q2_v2.solve_recommended_v2(
        d, n_iter=N_ITER, seeds=SEEDS, verbose=True, k_targets=K_TARGETS,
        patience=PATIENCE, sa_iters=SA_ITERS,
        save=False, relay_gate=False)

    runs = info['runs']

    # ---- 硬门禁：对不上就不写文件 -----------------------------------------
    print('=' * 74)
    print('门禁：逐档核对 q2_scan.csv 的档案第一条')
    bad = check_gate(runs)
    if bad:
        print('[FAIL] 与正式轮不一致，**不写 q2_seeds.csv**：')
        for b in bad:
            print('  - %s' % b)
        print('  这说明当前代码复现不出已交付的结果，请先查清再谈稳定性。')
        return 1
    print('[OK  ] 四档档案第一条与 q2_scan.csv 逐位一致')

    # ---- 落盘 --------------------------------------------------------------
    rows = []
    for K in sorted(runs):
        for i, s in enumerate(runs[K]['seeds']):
            mm = s['metrics']
            rows.append(dict(
                K上限=K, 种子序号=i, 种子=s['seed'],
                是否可行=('是' if s['feasible'] else '否'),
                架次=mm['n_trips'],
                能耗kWh=round(mm['total_E'], E_DEC),
                makespan_s=round(mm['makespan'], T_DEC),
                加权时延=round(mm['tardiness'], 3),
                硬违反=mm['hard_viol'],
                目标值f=round(s['f'], 6)))
    df = pd.DataFrame(rows)
    df.to_csv(DEST, index=False)

    print('=' * 74)
    print('已写出 %s（%d 行 = %d 档 × %d 种子）'
          % (os.path.relpath(DEST, os.path.dirname(HERE)), len(df),
             len(runs), SEEDS))
    summary = df.groupby('K上限').agg(
        可行数=('是否可行', lambda x: int((x == '是').sum())),
        架次_极差=('架次', lambda x: int(x.max() - x.min())),
        能耗_极差kWh=('能耗kWh', lambda x: round(x.max() - x.min(), E_DEC)),
        makespan_极差s=('makespan_s', lambda x: round(x.max() - x.min(), T_DEC)),
        加权时延_极差=('加权时延', lambda x: round(x.max() - x.min(), 3)))
    print(summary.to_string())
    print('  极差 = 该档 %d 个种子里的最大值减最小值，是「换一组种子会不会变」的直接答案。'
          % SEEDS)
    print('总耗时 %.1fs' % (time.time() - t0))
    return 0


if __name__ == '__main__':
    sys.exit(main())
