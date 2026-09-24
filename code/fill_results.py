# -*- coding: utf-8 -*-
# 本程序及代码是在人工智能工具辅助下完成的。
# 工具名称：DeepSeek；版本/型号：DeepSeek V4.1 Flash（API 模型名 deepseek-flash）；
# 开发机构：杭州深度求索人工智能基础技术研究有限公司；
# 版本发布日期：2026 年 9 月 10 日。

"""把 results/*.csv 回填进《结果提交模板.xlsx》，生成可直接上传的结果表。

模板表头带全角单位括号（如“总质量（kg）”），CSV 列名把单位并进名字（“总质量kg”），
**列名对得上、列位置未必对得上**：模板是赛题下发的固定格式不能改列，CSV 是求解
脚本的完整产物（verify.py 与正文按列名读）也不能为了对齐而删列，于是两边只要
有一方增列，按位置回填就会整体错位——把「悬停站编号」写进「开始时刻（s）」这种
错误不会有任何报错，填出来的表看着还挺像回事。

故这里按**列名**对齐：模板表头归一化（去掉括号单位）后与 CSV 列名逐一匹配，
措辞不同的走显式别名表；随后逐列核对单位，模板写了单位而 CSV 列名里没有的，
必须在 UNIT_CSV 里声明。最后核销三件事——模板每列都解析到了、CSV 每列都用上
或被逐名声明为不提交、单位一致——任何一条不成立就中止，绝不产出一份**看起来
完整**的结果表。模板表头与工作表名一律保持原样。

用法：python code/fill_results.py
输出：结果提交表-已填写.xlsx
"""
import io
import os
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd
import openpyxl

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
RES = os.path.join(ROOT, 'results')

# (模板工作表, 结果 CSV, 不提交的 CSV 列名)
#
# 模板是赛题下发的固定格式，列数不能改；CSV 是求解脚本的完整产物，列数也不能改
# （verify.py 与正文按列名读）。模板放不下的列在这里**逐名列明**，不靠「取前 n 列」
# 这种位置切片：切片在列序变动时会静默丢错列，而列名清单不会。
SHEETS = [
    ('Q1_单点组批', 'q1_recommended_batching.csv', ()),
    ('Q2_运输架次', 'q2_transport_trips.csv', ()),
    ('Q2_逐箱交付', 'q2_box_delivery.csv', ()),
    ('Q3_中继架次', 'q3_relay_trips.csv', ('悬停站编号',)),
    # 段时长s / 采样点数 是 F13 改用共同边界序列后新增的两列：段时长与逐时刻
    # 时间权重同源、采样点数用于核对分段未被细分。二者是复核用的旁证，模板没有
    # 对应列，故在此逐名列明不提交（而不是切片丢掉）。
    ('Q3_通信保障', 'q3_comm_phases.csv', ('段时长s', '采样点数')),
    ('Q4_分区配置', 'q4_partition.csv', ('工作量h',)),
]

# 模板列名 → CSV 列名：只在**措辞不同**时写，措辞相同的靠归一化自动匹配。
ALIAS = {
    ('Q4_分区配置', 'A型运输无人机数'): 'A型运输无人机',
    ('Q4_分区配置', 'B型运输无人机数'): 'B型运输无人机',
    ('Q4_分区配置', 'C型运输无人机数'): 'C型运输无人机',
    ('Q4_分区配置', 'A型电池组数'): 'A型共享电池',
    ('Q4_分区配置', 'B型电池组数'): 'B型共享电池',
    ('Q4_分区配置', 'C型电池组数'): 'C型共享电池',
    ('Q4_分区配置', '中继无人机数'): '中继无人机',
    ('Q4_分区配置', '中继能源组件数'): '中继能源组件',
}

# CSV 列名尾部可识别的单位。比对时把 'm³' 归一成 'm3'。
CSV_UNITS = ('kWh', 'kg', 'm3', 'dB', 'm', 's', 'h')

# 模板标了单位、而 CSV 列名里没有单位后缀的列：在此**显式声明**模板单位。
# 没声明的（无论哪边缺）一律报错——「模板写 kWh、CSV 是 kg」这类错位正是位置
# 拷贝最典型的失败方式，必须当场面质。
UNIT_CSV = {
    ('Q3_中继架次', '悬停经度'): '°',
    ('Q3_中继架次', '悬停纬度'): '°',
    ('Q1_单点组批', '返航SOC'): '%',
}

# 需要按百分数换算的列（模板列是「返航SOC（%）」，而 CSV 与正文一律用 0~1 的
# 小数：s = 1 - E_p/E_g^use，见 4.3 节）。直接照搬会得到 0.62 这种填进「%」列的
# 数，量纲差 100 倍。这里显式换算一次，不改 CSV（正文引用的仍是小数）。
PCT_COLS = {('Q1_单点组批', '返航SOC')}

BRACKET = re.compile(r'[（(]([^）)]*)[）)]')
UNIT_TAIL = re.compile(r'(kWh|kg|m3|dB|m|s|h)$')
# 括号里只有这些才算单位：模板还有「K（2或3）」这种括号，它不是单位，
# 若一并当成单位去比对，会逼着 CSV 也写个「2或3」后缀。
TPL_UNIT = re.compile(r'^(kWh|kg|m3|dB|m|s|h|°|%)$')


def _norm(s):
    """列名归一化：去掉括号内容、空格与下划线；'m³' 视作 'm3'。"""
    return re.sub(r'[\s_]', '', BRACKET.sub('', str(s))).replace('m³', 'm3')


def _key(s):
    """匹配用的键：归一化后再去掉尾部单位，于是「总质量kg」与「总质量（kg）」同键。"""
    return UNIT_TAIL.sub('', _norm(s))


def _tpl_unit(header):
    """模板表头括号里的单位（没有或不是单位则空串）。"""
    m = BRACKET.search(str(header))
    if not m:
        return ''
    u = m.group(1).strip().replace('m³', 'm3')
    return u if TPL_UNIT.match(u) else ''


def _csv_unit(col):
    """CSV 列名尾部的单位（没有则空串）。"""
    m = UNIT_TAIL.search(str(col))
    return m.group(1) if m else ''


def resolve(sheet, header, cols):
    """模板列名 → CSV 列名：先查别名表，再按 _key 匹配。返回列名或 None。

    必须以「唯一命中」为条件：两个 CSV 列归一到同一个键时（例如同时存在
    「总质量」与「总质量kg」）不算匹配——那时选哪一列都是猜。
    """
    if (sheet, header) in ALIAS:
        return ALIAS[(sheet, header)]
    hit = [c for c in cols if _key(c) == _key(header)]
    return hit[0] if len(hit) == 1 else None


def main():
    src = os.path.join(ROOT, '结果提交模板.xlsx')
    if not os.path.exists(src):
        raise SystemExit('找不到《结果提交模板.xlsx》。\n'
                         '该文件是赛题下发的空模板，需与本脚本放在同一目录（工程根目录）。\n'
                         '支撑材料压缩包中已随包附带，解压后直接运行即可。')
    wb = openpyxl.load_workbook(src)
    total = 0
    for sheet, csv, dropped in SHEETS:
        ws = wb[sheet]
        df = pd.read_csv(os.path.join(RES, csv), encoding='utf-8-sig')
        # 模板表头占第 1 行；判据取「第 1 行非空单元格数」，因为模板的 max_column
        # 含残留格式列（Q3_中继架次 14、Q4_分区配置 16、Q3_通信保障 23）。
        hdr = [c.value for c in ws[1] if c.value is not None]
        cols = list(df.columns)

        # ---- 列名映射：模板每列都要解析到，解析不到就报错 ----
        mapping, miss = {}, []
        for h in hdr:
            c = resolve(sheet, h, cols)
            if c is None:
                miss.append(h)
            else:
                mapping[h] = c
        if miss:
            raise SystemExit('[%s] 模板列在 %s 中找不到对应列：%s\n'
                             '  可选的 CSV 列：%s\n'
                             '  措辞不同请在 ALIAS 里登记，不要退回按位置回填。'
                             % (sheet, csv, '、'.join(miss), '、'.join(cols)))
        if len(set(mapping.values())) != len(mapping):
            raise SystemExit('[%s] 多个模板列解析到了同一个 CSV 列：%s' % (sheet, mapping))

        # ---- 不提交的列：必须逐名对上，且不得把要用的列漏掉 ----
        used = set(mapping.values())
        left = [c for c in cols if c not in used]
        if sorted(left) != sorted(dropped):
            raise SystemExit('[%s] 不提交的列不符：声明 %r，实际 %r'
                             % (sheet, tuple(dropped), tuple(left)))

        # ---- 单位核对：模板单位与 CSV 单位必须一致（含显式声明的例外）----
        bad_unit = []
        for h, c in mapping.items():
            tu, cu = _tpl_unit(h), _csv_unit(c)
            if not tu and not cu:
                continue
            if not cu:
                cu = UNIT_CSV.get((sheet, c), '')
                if not cu:
                    bad_unit.append('%s：模板单位「%s」，CSV 列「%s」无单位声明' % (h, tu, c))
                    continue
            if _norm(tu) != _norm(cu):
                bad_unit.append('%s：模板「%s」vs CSV「%s」' % (h, tu, cu))
        if bad_unit:
            raise SystemExit('[%s] 单位不一致（不得按位置硬塞）：\n  %s'
                             % (sheet, '\n  '.join(bad_unit)))

        # ---- 写入：按解析出的列名取值，模板表头原样保留 ----
        # 空白模板的表体里带着多余的格式化行（Q3_中继架次 29 行、Q4_分区配置 58 行、
        # Q3_通信保障 79 行），只从第 2 行往下覆盖而不清理，会把这些并不存在的
        # 旧方案一并交上去。故写入前先清空整个表体。
        if ws.max_row > 1:
            ws.delete_rows(2, ws.max_row - 1)
        n_pct = 0
        for i in range(len(df)):
            for j, h in enumerate(hdr):
                v = df[mapping[h]].iloc[i]
                # openpyxl 不接受 numpy 标量，统一转成 Python 原生类型
                if hasattr(v, 'item'):
                    v = v.item()
                if (sheet, mapping[h]) in PCT_COLS and isinstance(v, (int, float)):
                    v = round(v * 100.0, 4)
                    n_pct += 1
                ws.cell(row=2 + i, column=1 + j, value=v)
        if n_pct:
            print('     （%s 按 %% 换算 ×100，共 %d 格）'
                  % ('、'.join(c for s, c in PCT_COLS if s == sheet), n_pct))
        # 清理是否真的生效、写入是否真的落满，都在这里当面核销
        if ws.max_row - 1 != len(df):
            raise SystemExit('[%s] 行数不符：写入后表体 %d 行，应为 %d 行'
                             % (sheet, ws.max_row - 1, len(df)))
        total += len(df)
        print('  %-12s <- %-32s %3d 行  %2d 列（%s）'
              % (sheet, csv, len(df), len(hdr),
                 '列名对齐' if len(hdr) == len(cols)
                 else '列名对齐，模板少 %d 列：%s' % (len(cols) - len(hdr), '、'.join(dropped))))
    out = os.path.join(ROOT, '结果提交表-已填写.xlsx')
    wb.save(out)
    print('共写入 %d 行，输出 %s' % (total, os.path.basename(out)))


if __name__ == '__main__':
    main()
