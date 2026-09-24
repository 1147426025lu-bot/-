# -*- coding: utf-8 -*-
"""输入数据审计：原始 Excel 与模型实际取值逐格交叉核对（只读，不改任何文件）。

为什么需要这个脚本
------------------
2026 华为杯 D 题的《运输无人机数据.xlsx》里，「可用装载体积」三格的真实值是
0.06 / 0.073 / 0.25 m³，但单元格数字格式是 ``0.0``——Excel 默认**只显示一位小数**，
屏幕上看到的是 0.1 / 0.1 / 0.3。照屏幕抄数、或让 AI 读截图/文本，都会把 B 型机
0.073 抄成 0.1（容量虚增 37%），组批结果随之偏离。C 组的「第九通道」、E/F 组的
小字约束同理：**输入错一位，后面全盘皆错，而且不会报错**。

本脚本做两件独立的事：

1. **显示精度扫描**：遍历五份工作簿的每个数值单元格，解析 ``number_format``，
   若「按显示格式四舍五入后的值」不等于「单元格真实存储值」，就列为隐患格。
   这类格子不一定是错的（也许是显示冗余），但**凡是靠肉眼或截图读数的地方都必须
   按真值核**，所以逐个列出、供人工确认。
2. **逐格交叉核对**：模型实际用的每个参数，用 openpyxl 从原始单元格**独立再读一遍**，
   与 ``core.load_data()``（走 pandas）的取值逐字段比对，容差 0。
   两条读取路径不同，任一方被显示格式或手抄污染，这里就会不一致。

退出码：任一处不一致即非零，便于挂进 CI 或打包预检。
"""
import os
import re
import sys

import openpyxl
import pandas as pd

# 与 build_submission.py 并列放在仓库根目录：它属于开发期自检工具，不是求解代码，
# 既不在附录 A 的《求解代码与结果文件清单》里，也不随提交包分发（打包清单是白名单）。
ROOT = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(ROOT, 'code')
DATA = os.path.join(ROOT, '数据', '无人机应急物资运输基础数据')
sys.path.insert(0, CODE)

import core  # noqa: E402

WORKBOOKS = ['运输无人机数据.xlsx', '中继无人机数据.xlsx',
             '物资需求与配送时限.xlsx', '调度中心与服务区.xlsx',
             '通信链路参数.xlsx']


# ---------------------------------------------------------------------------
# 1. 显示精度扫描
# ---------------------------------------------------------------------------
def decimals_of(number_format):
    """把 number_format 解析成小数位数；不是定点小数格式则返回 None。

    只认 ``0.00`` / ``0.000`` / ``#,##0.0`` 这类定点格式（含千分位与百分号），
    科学计数、文本、General 一律返回 None——它们不存在「四舍五入掩盖真值」的问题。
    """
    if not number_format:
        return None
    fmt = str(number_format).split(';')[0].strip()
    if re.search(r'[eE][+-]', fmt):
        return None
    m = re.search(r'\.(0+)(?:%|$)', fmt)
    if m:
        return len(m.group(1))
    if re.fullmatch(r'[#0,]+%?', fmt):
        return 0
    return None


def displayed(v, d):
    """Excel 的显示值：四舍五入**逢五进一**。

    不能用 Python 的 ``round``——它是"银行家舍入"，``round(0.25, 1)`` 得 0.2，
    而 Excel 在 ``0.0`` 格式下显示 0.3。报表里的"屏幕所见"一列必须与 Excel 一致，
    否则这份审计自己就先错了。
    """
    from decimal import Decimal, ROUND_HALF_UP
    q = Decimal(1).scaleb(-d)
    return float(Decimal(repr(float(v))).quantize(q, rounding=ROUND_HALF_UP))


def scan_display_hazards():
    hazards = []
    for name in WORKBOOKS:
        path = os.path.join(DATA, name)
        wb = openpyxl.load_workbook(path, data_only=True)
        for ws in wb.worksheets:
            for row in ws.iter_rows():
                for c in row:
                    v = c.value
                    if not isinstance(v, (int, float)) or isinstance(v, bool):
                        continue
                    d = decimals_of(c.number_format)
                    if d is None:
                        continue
                    if abs(displayed(v, d) - float(v)) > 1e-12:
                        hazards.append((name, ws.title, c.coordinate, float(v),
                                        c.number_format, displayed(v, d)))
    return hazards


# ---------------------------------------------------------------------------
# 2. 逐格交叉核对
# ---------------------------------------------------------------------------
def cell(ws, r_iloc, c_iloc):
    """core.py 用 pandas ``df.iloc[r, c]``（0 基）读表；这里换算成 A1 坐标独立再读。"""
    return ws.cell(row=r_iloc + 1, column=c_iloc + 1).value


def build_checks():
    """返回 [(说明, 原始值, 代码值)]，覆盖模型实际使用的每一个输入参数。"""
    out = []
    d = core.load_data()

    # --- 运输机型 A/B/C：17 个数值字段 + 机型名 ---
    ws = openpyxl.load_workbook(os.path.join(DATA, '运输无人机数据.xlsx'),
                                data_only=True)['数据']
    tfields = ['name', 'empty_mass', 'Q', 'volume', 'v_cruise', 'L0', 'LF',
               'E_use', 'rho', 'prep', 'load_per_box', 'hand_base',
               'hand_per_box', 'v_up', 'v_down', 'eta_up', 'eta_down']
    for r, tid in [(2, 'A'), (3, 'B'), (4, 'C')]:
        got = d.transport_types[tid]
        for k, f in enumerate(tfields):
            raw = cell(ws, r, k + 1)
            val = got[f]
            if f == 'name':
                val = str(val).strip()
            elif f == 'rho':                      # core 把百分数除 100
                raw = float(raw) / 100.0
            out.append(('运输机型%s.%s' % (tid, f), raw, val))

    # --- 逐架运输无人机清单 U01..U08 ---
    for r, uid, tid in [(8, 'U01', 'A'), (9, 'U02', 'A'), (10, 'U03', 'A'),
                        (11, 'U04', 'A'), (12, 'U05', 'B'), (13, 'U06', 'B'),
                        (14, 'U07', 'C'), (15, 'U08', 'C')]:
        out.append(('逐架清单%s.编号' % uid, cell(ws, r, 0), uid))
        out.append(('逐架清单%s.机型' % uid, cell(ws, r, 1), tid))
    uavs = {u['id']: u for u in d.uavs}
    out.append(('逐架运输无人机架数', 8, len(uavs)))

    # --- 运输共享电池库存（行 19/20/21 = A/B/C）---
    for r, tid in [(19, 'A'), (20, 'B'), (21, 'C')]:
        n, chg = d.batteries[tid]
        out.append(('运输电池%s.组数' % tid, int(cell(ws, r, 1)), n))
        out.append(('运输电池%s.充电时间s' % tid, float(cell(ws, r, 2)), chg))

    # --- 中继机型 R：17 个数值字段 ---
    wr = openpyxl.load_workbook(os.path.join(DATA, '中继无人机数据.xlsx'),
                                data_only=True)['数据']
    rt = d.relay_type
    rfields = ['empty_mass', 'module_mass', 'takeoff_mass', 'v_cruise',
               'P_cruise', 'E_use', 'rho', 'prep', 'link', 'turnover',
               'v_up', 'v_down', 'eta_up', 'eta_down', 'P_hover', 'P_comm',
               'max_hover_alt']
    for k, f in enumerate(rfields):
        raw = cell(wr, 2, k + 2)
        val = rt[f]
        if f == 'rho':
            raw = float(raw) / 100.0
        out.append(('中继机型.%s' % f, raw, val))

    # --- 中继无人机与共享能源组件 ---
    out.append(('中继无人机架数', 2, len(d.relay_uavs)))
    rn, rchg = d.relay_batteries
    out.append(('中继能源组件.组数', int(cell(wr, 11, 1)), rn))
    out.append(('中继能源组件.充电时间s', float(cell(wr, 11, 2)), rchg))

    # --- 通信链路参数（iloc 行 2..15 → 列 4）---
    wc = openpyxl.load_workbook(os.path.join(DATA, '通信链路参数.xlsx'),
                                data_only=True)['数据']
    cp = d.comm_params      # 原始参数字典；d.comm 是封装后的链路模型，取不到原始值
    cfields = ['f', 'Lsys', 'Lobs', 'Psens', 'M', 'transport_Pt', 'transport_G',
               'relay_acc_Pt', 'relay_acc_G', 'relay_bh_Pt', 'relay_bh_G',
               'gw_Pt', 'gw_G', 'gw_h']
    for k, f in enumerate(cfields):
        out.append(('通信参数.%s' % f, cell(wc, k + 2, 4), cp[f]))

    # --- 调度中心 O01 与服务区 S001..S015 ---
    wn = openpyxl.load_workbook(os.path.join(DATA, '调度中心与服务区.xlsx'),
                                data_only=True)['数据']
    for k, attr in enumerate(['lon', 'lat', 'alt']):
        out.append(('调度中心O01.%s' % attr, cell(wn, 2, k + 2), d.O01[attr]))
    sid_map = {s['id']: s for s in d.services}
    for r in range(6, 21):
        sid = str(cell(wn, r, 0)).strip()
        for k, attr in enumerate(['lon', 'lat', 'alt', 'pop']):
            out.append(('%s.%s' % (sid, attr), cell(wn, r, k + 2), sid_map[sid][attr]))

    # --- 逐箱货箱清单（80 箱 × 质量/体积/时限/优先系数）---
    wb = openpyxl.load_workbook(os.path.join(DATA, '物资需求与配送时限.xlsx'),
                                data_only=True)
    boxws = wb['逐箱货箱清单']
    nbox = 0
    for r in range(2, 82):
        bid = str(boxws.cell(row=r, column=1).value).strip()
        if not bid or bid == 'None':
            continue
        nbox += 1
        b = next(x for x in d.cargo if x['id'] == bid)
        out.append(('%s.质量' % bid, boxws.cell(row=r, column=4).value, b['mass']))
        out.append(('%s.体积' % bid, boxws.cell(row=r, column=5).value, b['volume']))
        out.append(('%s.是否首批' % bid,
                    str(boxws.cell(row=r, column=6).value).strip(),
                    '是' if b['first_batch'] else '否'))
        for col, key in [(7, 'deadline'), (8, 'expect')]:
            raw = boxws.cell(row=r, column=col).value
            got = b[key]
            if raw is None and got is None:
                pass
            else:
                out.append(('%s.%s' % (bid, key), float(raw), got))
        out.append(('%s.优先系数' % bid,
                    boxws.cell(row=r, column=9).value, b['priority']))
    out.append(('货箱总数', 80, nbox))

    # --- 汇总表《数据》与逐箱清单必须自洽（箱数/单箱质量/单箱体积）---
    agg = pd.read_excel(os.path.join(DATA, '物资需求与配送时限.xlsx'),
                        sheet_name='数据')
    # 逐箱清单要带上箱号与首批标记，汇总表的「首批必须送达箱数」才能按列核对
    # （早先版本拿 m.index 去 isin 箱号，索引是 RangeIndex 整数、箱号是字符串，
    #   恒为 False，于是每行都数出 0——那是本脚本的错，不是数据错）
    boxes = pd.DataFrame([{'货箱编号': b['id'], '服务区编号': b['service'],
                           '物资类型': b['category'], '是否首批': bool(b['first_batch']),
                           '单箱质量（kg）': b['mass'], '单箱体积（m³）': b['volume']}
                          for b in d.cargo])
    for _, a in agg.iterrows():
        if pd.isna(a['服务区编号']):
            continue
        m = boxes[(boxes['服务区编号'] == a['服务区编号'])
                  & (boxes['物资类型'] == a['物资类型'])]
        out.append(('%s/%s.总需求箱数' % (a['服务区编号'], a['物资类型']),
                    int(a['总需求箱数']), len(m)))
        out.append(('%s/%s.首批必须送达箱数' % (a['服务区编号'], a['物资类型']),
                    int(a['首批必须送达箱数']), int(m['是否首批'].sum())))
        if len(m):
            out.append(('%s/%s.单箱质量' % (a['服务区编号'], a['物资类型']),
                        float(a['单箱质量（kg）']), float(m['单箱质量（kg）'].iloc[0])))
            out.append(('%s/%s.单箱体积' % (a['服务区编号'], a['物资类型']),
                        float(a['单箱体积（m³）']), float(m['单箱体积（m³）'].iloc[0])))
    return out


def same(raw, val):
    if isinstance(raw, str) or isinstance(val, str):
        return str(raw).strip() == str(val).strip()
    return abs(float(raw) - float(val)) <= 1e-12


def main():
    print('=' * 74)
    print('一、显示精度扫描：数字格式把真值四舍五入的单元格')
    print('=' * 74)
    hazards = scan_display_hazards()
    if not hazards:
        print('未发现"显示值 ≠ 存储值"的单元格。')
    else:
        print('%-24s %-12s %-6s %-14s %-8s %s' %
              ('工作簿', '工作表', '单元格', '存储真值', '显示格式', '屏幕所见'))
        for name, sheet, coord, v, fmt, shown in hazards:
            print('%-24s %-12s %-6s %-14r %-8s %r' %
                  (name, sheet, coord, v, fmt, shown))
        print('\n共 %d 格。这些格子按屏幕读会读错；本项目的求解代码走 '
              'pandas 读原始值，不受显示格式影响，但凡人工/截图核对处必须按真值核。'
              % len(hazards))
        print('D 题最典型的就是「可用装载体积」这三格：真值 0.06 / 0.073 / 0.25 m³，'
              '在 0.0 格式下屏幕显示 0.1 / 0.1 / 0.3。')

    print()
    print('=' * 74)
    print('二、逐格交叉核对：原始单元格 vs core.load_data() 实际取值')
    print('=' * 74)
    checks = build_checks()
    bad = []
    for name, raw, val in checks:
        if not same(raw, val):
            bad.append((name, raw, val))
    print('核对字段数 = %d，不一致 = %d' % (len(checks), len(bad)))
    for name, raw, val in bad:
        print('  [不符] %-34s 原始=%r  代码=%r' % (name, raw, val))
    if not bad:
        print('全部一致：模型用到的每个参数都与原始单元格逐位相同。')

    print()
    print('结论：显示精度隐患 %d 格（已逐个列出，供人工核）；'
          '输入取值不一致 %d 项。' % (len(hazards), len(bad)))
    return 1 if bad else 0


if __name__ == '__main__':
    raise SystemExit(main())
