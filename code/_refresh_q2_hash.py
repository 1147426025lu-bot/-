# -*- coding: utf-8 -*-
# 一次性脚本：刷新 results/q2_solution.json 里的 solver_hashes。
#
# 背景：q2 于 20:32 跑完并落盘，之后为了修问题三导出崩掉的那行
# （solution_io.build_q3_solution 里 metrics 一律 float() 会把字符串字段
# min_hard_margin_box 转炸），又改了 code/solution_io.py。该文件同时列在
# Q2_SOLVER_FILES 里，于是 q2 的记录被判为「过期」，q3 拒绝加载。
#
# 这次改动落在 build_q3_solution 内，q2 的求解与导出路径**不经过该函数**；
# 但「不经过」是人工判断，不是重跑复现。因此本脚本做三件事：
#   1) 只允许 code/solution_io.py 一项哈希变化，多一项就中止；
#   2) 先用 q2 自己落盘的 CSV 复核 JSON 内容（架次数、逐箱交付、能耗）自洽；
#   3) 把旧/新哈希与旧/新 solution_id 写进 results/_q2_hash_refresh.txt 备查。
# 这一步是记账修补，不是重新求解——论文与最终汇报均须如实说明。
#
# ⚠️ 本脚本已被收紧，**当前状态下跑不动 2026-09-24 那一次刷新**：新增的证据
# 要求是「文本规范化哈希也不变」（见下方 real 判定），而那次改的是
# solution_io.build_q3_solution 里的类型转换，是真实代码改动，不是换行差异。
# 也就是说，那一次刷新的依据始终只是「人工判断该函数不在 q2 的路径上」——
# 记录文件里也是这么写的。阶段 11 的整链重跑会重新生成全部方案记录，使这次
# 刷新及其记录一并作废；重跑之后，本脚本只对「纯换行差异」生效，其余重跑。
import io
import json
import os
import sys
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

import solution_io as sio

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
RES = os.path.join(ROOT, 'results')
SOL = os.path.join(RES, 'q2_solution.json')
NOTE = os.path.join(RES, '_q2_hash_refresh.txt')


def main():
    with open(SOL, encoding='utf-8') as f:
        sol = json.load(f)
    old_id = sol['solution_id']
    old_h = dict(sol['solver_hashes'])
    new_h = sio.solver_hashes(sio.Q2_SOLVER_FILES)

    diff = [k for k in sorted(set(old_h) | set(new_h)) if old_h.get(k) != new_h.get(k)]
    if diff != ['code/solution_io.py']:
        raise SystemExit('预期只有 code/solution_io.py 一项哈希变化，实为 %s；'
                         '请重新运行 q2.py，不要刷新记录' % diff)

    # 上面那条只是**白名单**（「改的必须是这个文件」），它挡不住「改的就是这个
    # 文件的算法」——真正要证的是「代码一个字没动」。文本规范化哈希能直接给出
    # 这个证明：把换行归一后再算，两版逐字一致 ⇔ 这次改动只动了换行符。
    # 于是规则收紧为：**只有文本哈希也不变才允许刷新**，其余一律重跑。
    old_t = sol.get('solver_text_hashes', {})
    new_t = sio.solver_hashes_text(sio.Q2_SOLVER_FILES)
    real = [k for k in diff if old_t.get(k) != new_t.get(k)]
    if real:
        raise SystemExit(
            '以下源码的**文本内容**也已改变，不是换行差异：%s\n'
            '说明这次改动真的动了代码，刷新哈希会把它掩盖掉——必须重跑 q2.py。'
            % '、'.join(real))
    if not old_t:
        raise SystemExit(
            '该方案记录里没有 solver_text_hashes，无法判定改动是否只涉及换行。\n'
            '缺少这个证据时不能刷新哈希——请重跑 q2.py。')

    # ---- 1) 用 q2 自己落盘的 CSV 复核 JSON 内容自洽 ----
    tt = pd.read_csv(os.path.join(RES, 'q2_transport_trips.csv'), encoding='utf-8-sig')
    bd = pd.read_csv(os.path.join(RES, 'q2_box_delivery.csv'), encoding='utf-8-sig')
    trips = sol['transport_trips']
    assert len(trips) == len(tt) == sol['metrics']['n_trips'], '架次数与 CSV 不符'
    for t, row in zip(trips, tt.itertuples()):
        assert t['trip_id'] == getattr(row, '架次编号'), '架次编号与 CSV 不符'
        # CSV 里的能耗是 3 位小数，JSON 存全精度，容差取半个末位
        assert abs(t['energy_kwh'] - getattr(row, '架次能耗kWh')) < 5e-4, '能耗与 CSV 不符'
    n_box_sol = sum(len(v) for t in trips for v in t['box_ids_by_service'].values())
    assert n_box_sol == len(bd), '逐箱交付数与 CSV 不符'
    for k in ('makespan', 'tardiness', 'total_E', 'hard_viol'):
        assert k in sol['metrics'], '指标缺 %s' % k

    # ---- 2) 只换哈希，内容一字不动 ----
    sol['solver_hashes'] = new_h
    new_id = sio.save_solution(SOL, sol)
    back = sio.load_solution(SOL, stage='q2', input_hashes_expect=sio.input_hashes(),
                             solver_hashes_expect=sio.solver_hashes(sio.Q2_SOLVER_FILES))
    assert back['solution_id'] == new_id

    lines = [
        'q2 方案记录哈希刷新说明（人工干预，非重新求解）',
        '时间：%s' % time.strftime('%Y-%m-%d %H:%M:%S'),
        '原因：q2 跑完后为修问题三导出崩掉的一行改了 code/solution_io.py；',
        '      该文件同时计入 Q2_SOLVER_FILES，q2 记录因此被判过期。',
        '改动位置：solution_io.build_q3_solution 的 metrics 类型转换；',
        '          q2 的求解与导出路径不调用该函数（人工判断，非重跑复现）。',
        '',
        'solution_id: %s -> %s' % (old_id, new_id),
        '',
        'solver_hashes 变化项：%s' % '、'.join(diff),
    ]
    for k in diff:
        lines += ['  %s' % k, '    旧 %s' % old_h.get(k), '    新 %s' % new_h.get(k)]
    lines += ['',
              '内容复核：架次数、架次编号、逐架次能耗、逐箱交付数均与',
              '          results/q2_transport_trips.csv、results/q2_box_delivery.csv 一致。',
              '未复核：重跑一次 q2（约 43 分钟）能否复现同一份方案——未做。',
              '']
    with open(NOTE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print('已刷新 %s' % os.path.relpath(SOL, ROOT))
    print('  solution_id %s -> %s' % (old_id, new_id))
    print('  变化项 %s' % '、'.join(diff))
    print('  备查记录 %s' % os.path.relpath(NOTE, ROOT))


if __name__ == '__main__':
    main()
