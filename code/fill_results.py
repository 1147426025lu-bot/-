# -*- coding: utf-8 -*-
# 本程序及代码是在人工智能工具辅助下完成的。
# 工具名称：DeepSeek；版本/型号：DeepSeek V4.1 Flash（API 模型名 deepseek-flash）；
# 开发机构：杭州深度求索人工智能基础技术研究有限公司；
# 版本发布日期：2026 年 9 月 10 日。

"""把 results/*.csv 回填进《结果提交模板.xlsx》，生成可直接上传的结果表。

模板表头带全角单位括号（如“总质量（kg）”），CSV 列名是纯 ASCII（“总质量kg”），
两者列数、列序完全一致，因此按位置对应回填，模板表头一律保持原样——
表头是赛题规定的格式，不能改成 CSV 的写法。

用法：python code/fill_results.py
输出：结果提交表-已填写.xlsx
"""
import io
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd
import openpyxl

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
RES = os.path.join(ROOT, 'results')

# (模板工作表, 结果 CSV, 写入列数, 预期被丢弃且位于末位的列名)
#
# 模板是赛题下发的固定格式，列数不能改；CSV 是求解脚本的完整产物，列数也不能改
# （verify.py 与正文按列名读）。两者对不上的表在这里显式声明只回填前 n_write 列，
# 并把被丢弃的列名逐字写出做断言——否则将来 CSV 一旦调换列序，切片会静默丢错列。
SHEETS = [
    ('Q1_单点组批', 'q1_recommended_batching.csv', None, ()),
    ('Q2_运输架次', 'q2_transport_trips.csv', None, ()),
    ('Q2_逐箱交付', 'q2_box_delivery.csv', None, ()),
    ('Q3_中继架次', 'q3_relay_trips.csv', 11, ('悬停站编号',)),
    ('Q3_通信保障', 'q3_comm_phases.csv', None, ()),
    ('Q4_分区配置', 'q4_partition.csv', 11, ('工作量h',)),
]

# 按位置回填的前提是两边量纲一致。全列逐项核对后，唯一不一致的是 Q1 的
# 返航 SOC：模板表头写的是「返航SOC（%）」，而 CSV 与正文推导里 SOC 一律
# 用 0~1 的小数（s = 1 - E_p/E_g^use，见 4.3 节）。直接照搬会得到 0.62 这种
# 填进「%」列的数，量纲差 100 倍。这里显式换算，不改 CSV（正文引用的仍是小数）。
PCT_COLS = {('Q1_单点组批', '返航SOC')}


def main():
    src = os.path.join(ROOT, '结果提交模板.xlsx')
    if not os.path.exists(src):
        raise SystemExit('找不到《结果提交模板.xlsx》。\n'
                         '该文件是赛题下发的空模板，需与本脚本放在同一目录（工程根目录）。\n'
                         '支撑材料压缩包中已随包附带，解压后直接运行即可。')
    wb = openpyxl.load_workbook(src)
    total = 0
    for sheet, csv, n_write, dropped in SHEETS:
        ws = wb[sheet]
        df = pd.read_csv(os.path.join(RES, csv), encoding='utf-8-sig')
        if n_write is None:
            n_write = len(df.columns)
        # 模板表头占第 1 行；数据从第 2 行起按列序写入。
        # 判据必须是「第 1 行非空单元格数」——模板的 max_column 含残留格式列
        # （Q3_中继架次 14、Q4_分区配置 16、Q3_通信保障 23），按它判会误认为放得下。
        ncol = sum(1 for c in ws[1] if c.value is not None)
        if ncol != n_write:
            raise SystemExit('[%s] 列数不符：模板 %d，待写 %d' % (sheet, ncol, n_write))
        # 被丢弃的列必须恰好是预期的那几列，且都在末位
        if tuple(df.columns[n_write:]) != dropped:
            raise SystemExit('[%s] 丢弃列不符：预期 %r，实际 %r'
                             % (sheet, dropped, tuple(df.columns[n_write:])))
        df = df.iloc[:, :n_write]
        # 空白模板的表体里带着多余的格式化行（Q3_中继架次 29 行、Q4_分区配置 58 行、
        # Q3_通信保障 79 行），只从第 2 行往下覆盖而不清理，会把这些并不存在的
        # 旧方案一并交上去。故写入前先清空整个表体。
        if ws.max_row > 1:
            ws.delete_rows(2, ws.max_row - 1)
        # CSV 列名是 ASCII（返航SOC），模板表头带单位（返航SOC（%）），
        # 用后缀匹配把百分数列标出来，避免写错列。
        pct_idx = {j for j, c in enumerate(df.columns)
                   if (sheet, c.split('（')[0]) in PCT_COLS}
        if pct_idx:
            print('     （%s 按 %% 换算：×100）'
                  % '、'.join(df.columns[j] for j in sorted(pct_idx)))
        for i, (_, row) in enumerate(df.iterrows()):
            for j, v in enumerate(row):
                # openpyxl 不接受 numpy 标量，统一转成 Python 原生类型
                if hasattr(v, 'item'):
                    v = v.item()
                if j in pct_idx and isinstance(v, (int, float)):
                    v = round(v * 100.0, 4)
                ws.cell(row=2 + i, column=1 + j, value=v)
        # 清理是否真的生效、写入是否真的落满，都在这里当面核销
        if ws.max_row - 1 != len(df):
            raise SystemExit('[%s] 行数不符：写入后表体 %d 行，应为 %d 行'
                             % (sheet, ws.max_row - 1, len(df)))
        total += len(df)
        print('  %-12s <- %-32s %3d 行' % (sheet, csv, len(df)))
    out = os.path.join(ROOT, '结果提交表-已填写.xlsx')
    wb.save(out)
    print('共写入 %d 行，输出 %s' % (total, os.path.basename(out)))


if __name__ == '__main__':
    main()
