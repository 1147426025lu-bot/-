# -*- coding: utf-8 -*-
# 本程序及代码是在人工智能工具辅助下完成的。
# 工具名称：DeepSeek；版本/型号：DeepSeek V4.1 Flash（API 模型名 deepseek-flash）；
# 开发机构：杭州深度求索人工智能基础技术研究有限公司；
# 版本发布日期：2026 年 9 月 10 日。

"""
D题 核心计算框架
山区洪涝灾害下无人机运输与通信协同优化

本模块统一实现：
  1) 基础数据加载（调度中心/服务区、货箱、运输/中继无人机、电池、通信参数）
  2) DEM 加载与空间查询（高程采样、沿线段地形净空、视线遮挡判定）
  3) 运输航段时间 / 能耗 / 等效航程 / 返航SOC 计算
  4) 中继无人机时间 / 能耗 计算
  5) 两阶段等效充电模型
  6) 通信链路预算与可用性判定

物理口径与单位严格遵循题目附录 2 / 附录 3。
"""
import numpy as np
import scipy.io as sio
import pandas as pd
import os

# ---------------------------------------------------------------------------
# 路径与常量
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')


def _resolve(*rel):
    """在候选位置里找第一个真实存在的路径，找不到就返回第一个候选。

    支撑材料压缩包里数据是保持 数据/ 层级整体带过去的，与工程目录同构；
    但为防有人解压后手工挪动了目录，这里再兜一层：按相对路径找不到时，
    退回只按文件名在 ROOT 下浅层搜一次。宁可多写十行，也不要让评委解压
    后跑出 FileNotFoundError —— 支撑材料跑不起来，等于结果不可追溯。
    """
    cands = [os.path.join(ROOT, *rel)]
    cands.append(os.path.join(ROOT, rel[0]))
    cands.append(os.path.join(ROOT, os.path.basename(rel[-1])))
    for c in cands:
        if os.path.exists(c):
            return c
    # 最后再按"目录名/文件名"扫两层
    leaf = os.path.basename(rel[-1])
    for base, dirs, files in os.walk(ROOT):
        if leaf in dirs or leaf in files:
            return os.path.join(base, leaf)
    return cands[0]


DATA_BASE = _resolve('数据', '无人机应急物资运输基础数据')
DEM_MAT = _resolve('数据', '镇龙乡地理空间数据', '镇龙乡及周边地理数据',
                   '数字高程模型数据（DEM）', '镇龙乡及周边30米DEM.mat')

G = 9.80665          # 重力加速度 m/s^2
J_PER_KWH = 3.6e6    # 1 kWh = 3.6e6 J

# 结果表数值精度。时刻列原先只留 1 位小数，复核脚本从表里反算共享电池的
# "返回时刻 + 充电时长" 时，两个时刻各带 ±0.05s 舍入、能耗舍入经充电曲线放大
# 约 ±0.014s，合计可达 0.11s，足以把周转余量伪装成约 0.03s 的冲突。
# 时刻留 3 位、能耗留 6 位后，该误差降到 2e-3 s 以下，与真实冲突区分得开。
T_DEC = 3            # 结果表中时刻(s)保留的小数位
E_DEC = 6            # 结果表中能耗(kWh)保留的小数位

# 视线遮挡判定的地形采样步长(m)：沿视线连线按此间隔取 DEM 高程点。
# 只服务于 line_occluded()。题面附录 3 的遮挡判据是"根据三维位置及 30m DEM
# 判断视线连线是否受到地形遮挡"，无"经过像元最高"的措辞，故按连线采样是自然读法。
# DEM 本身是 30m 栅格，步长越细越能捕捉两个栅格点之间的局部凸起。取 5m 的
# 依据不是"最高地形被低估多少"（那个量已改由像元遍历求得，与步长无关），
# 而是遮挡结论本身对步长是否敏感：120 条节点连线在 5m 与 1m 下判定翻转 0 条，
# 沿线净空余量最小 39.2m、中位 51.5m，无一贴近阈值（见 9.4 节）。
#
# 注意：巡航高度不归此参数管。附录 2 要求巡航海拔取"该航段所经过 DEM 像元的
# 最高地面高程"以上 50m，那是像元口径而非采样口径，由 DEMGrid.max_elev_along()
# 做栅格遍历求得，与 DEM_STEP 无关。
DEM_STEP = 5.0

# 坐标：局部平面近似（WGS84，参考纬度 23°N）
LAT0 = np.deg2rad(23.0)
M_PER_DEG_LAT = 111132.95
M_PER_DEG_LON = 111320.0 * np.cos(LAT0)


def ll_to_xy(lon, lat):
    """经纬度 -> 局部平面坐标(m)，以 O01 为原点由调用方平移。"""
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    x = lon * M_PER_DEG_LON
    y = lat * M_PER_DEG_LAT
    return x, y


# ---------------------------------------------------------------------------
# 1. 基础数据加载
# ---------------------------------------------------------------------------
def _clean_cols(df):
    df.columns = [str(c).strip() for c in df.columns]
    return df


def load_nodes():
    """返回 (O01 dict, service_areas list of dict)."""
    f = os.path.join(DATA_BASE, '调度中心与服务区.xlsx')
    df = pd.read_excel(f, header=None)
    # 行结构(header=None)：0=标题行；1=列名；2=O01 数据
    o01_lon = float(df.iloc[2, 2]); o01_lat = float(df.iloc[2, 3]); o01_alt = float(df.iloc[2, 4])
    O01 = dict(id='O01', name='凤丹村无人机调度中心',
               lon=o01_lon, lat=o01_lat, alt=o01_alt)
    services = []
    # 服务区：5=列名；6..20 = S001..S015
    for r in range(6, 6 + 15):
        sid = str(df.iloc[r, 0]).strip()
        name = str(df.iloc[r, 1]).strip()
        lon = float(df.iloc[r, 2]); lat = float(df.iloc[r, 3]); alt = float(df.iloc[r, 4])
        pop = int(df.iloc[r, 5])
        services.append(dict(id=sid, name=name, lon=lon, lat=lat, alt=alt, pop=pop))
    return O01, services


def load_transport_uav_types():
    """返回 dict: type -> 参数. 机型 A/B/C."""
    f = os.path.join(DATA_BASE, '运输无人机数据.xlsx')
    df = pd.read_excel(f, header=None)
    types = {}
    for r, tid in [(2, 'A'), (3, 'B'), (4, 'C')]:
        types[tid] = dict(
            id=tid,
            name=str(df.iloc[r, 1]).strip(),
            empty_mass=float(df.iloc[r, 2]),   # 含电池空载总质量 kg
            Q=float(df.iloc[r, 3]),            # 最大载货质量 kg
            volume=float(df.iloc[r, 4]),       # 可用装载体积 m^3
            v_cruise=float(df.iloc[r, 5]),     # 计划巡航速度 m/s
            L0=float(df.iloc[r, 6]),           # 空载标准航程 m
            LF=float(df.iloc[r, 7]),           # 满载标准航程 m
            E_use=float(df.iloc[r, 8]),        # 电池可用能量 kWh
            rho=float(df.iloc[r, 9]) / 100.0,  # 返航安全余量比例
            prep=float(df.iloc[r, 10]),        # 工位固定准备时间 s
            load_per_box=float(df.iloc[r, 11]),# 每箱装载时间 s
            hand_base=float(df.iloc[r, 12]),   # 接收点基础交接时间 s
            hand_per_box=float(df.iloc[r, 13]),# 每箱增加交接时间 s
            v_up=float(df.iloc[r, 14]),        # 最大爬升速度 m/s
            v_down=float(df.iloc[r, 15]),      # 最大下降速度 m/s
            eta_up=float(df.iloc[r, 16]),      # 爬升能耗效率
            eta_down=float(df.iloc[r, 17]),    # 下降能耗效率
        )
    return types


def load_transport_uavs():
    """返回 list of dict: 逐架无人机 U01..U08."""
    f = os.path.join(DATA_BASE, '运输无人机数据.xlsx')
    df = pd.read_excel(f, header=None)
    uavs = []
    rows = [(8, 'U01', 'A'), (9, 'U02', 'A'), (10, 'U03', 'A'), (11, 'U04', 'A'),
            (12, 'U05', 'B'), (13, 'U06', 'B'), (14, 'U07', 'C'), (15, 'U08', 'C')]
    for r, uid, tid in rows:
        uavs.append(dict(id=uid, type=tid, pos='O01'))
    return uavs


def load_batteries():
    """返回 dict: type -> (总数, 完全充电时间s)."""
    f = os.path.join(DATA_BASE, '运输无人机数据.xlsx')
    df = pd.read_excel(f, header=None)
    # 行 19/20/21 = A/B/C 电池
    return {
        'A': (int(df.iloc[19, 1]), float(df.iloc[19, 2])),
        'B': (int(df.iloc[20, 1]), float(df.iloc[20, 2])),
        'C': (int(df.iloc[21, 1]), float(df.iloc[21, 2])),
    }


def load_relay_type():
    """返回中继无人机机型参数 dict."""
    f = os.path.join(DATA_BASE, '中继无人机数据.xlsx')
    df = pd.read_excel(f, header=None)
    return dict(
        id='R',
        empty_mass=float(df.iloc[2, 2]),      # 含能源组件空载总质量 kg
        module_mass=float(df.iloc[2, 3]),     # 中继通信模块质量 kg
        takeoff_mass=float(df.iloc[2, 4]),    # 计划起飞总质量 kg
        v_cruise=float(df.iloc[2, 5]),        # 计划巡航速度 m/s
        P_cruise=float(df.iloc[2, 6]),        # 巡航功率 kW
        E_use=float(df.iloc[2, 7]),           # 能源组件可用能量 kWh
        rho=float(df.iloc[2, 8]) / 100.0,     # 返航电量下限比例
        prep=float(df.iloc[2, 9]),            # 工位固定准备时间 s
        link=float(df.iloc[2, 10]),           # 建链时间 s
        turnover=float(df.iloc[2, 11]),       # 架次周转时间 s
        v_up=float(df.iloc[2, 12]),           # 最大爬升速度 m/s
        v_down=float(df.iloc[2, 13]),         # 最大下降速度 m/s
        eta_up=float(df.iloc[2, 14]),         # 爬升能耗效率
        eta_down=float(df.iloc[2, 15]),       # 下降能耗效率
        P_hover=float(df.iloc[2, 16]),        # 悬停功率 kW
        P_comm=float(df.iloc[2, 17]),         # 通信附加功率 kW
        max_hover_alt=float(df.iloc[2, 18]),  # 最大悬停离地高度 m
    )


def load_relay_uavs():
    """返回 list: R01, R02."""
    f = os.path.join(DATA_BASE, '中继无人机数据.xlsx')
    df = pd.read_excel(f, header=None)
    return [dict(id='R01', type='R', pos='O01'), dict(id='R02', type='R', pos='O01')]


def load_relay_batteries():
    """返回 (总数, 完全充电时间s)."""
    f = os.path.join(DATA_BASE, '中继无人机数据.xlsx')
    df = pd.read_excel(f, header=None)
    return (int(df.iloc[11, 1]), float(df.iloc[11, 2]))


def load_cargo():
    """返回逐箱货箱清单 list of dict (80箱)."""
    f = os.path.join(DATA_BASE, '物资需求与配送时限.xlsx')
    df = pd.read_excel(f, sheet_name='逐箱货箱清单')
    boxes = []
    for _, row in df.iterrows():
        boxes.append(dict(
            id=str(row['货箱编号']).strip(),
            service=str(row['服务区编号']).strip(),
            category=str(row['物资类型']).strip(),
            mass=float(row['单箱质量（kg）']),
            volume=float(row['单箱体积（m³）']),
            first_batch=(str(row['是否首批保障']).strip() == '是'),
            deadline=float(row['首批截止时间（s）']) if pd.notna(row['首批截止时间（s）']) else None,
            expect=float(row['期望送达时间（s）']) if pd.notna(row['期望送达时间（s）']) else None,
            priority=float(row['应急优先系数']),
        ))
    return boxes


def load_comm_params():
    """返回通信参数 dict."""
    f = os.path.join(DATA_BASE, '通信链路参数.xlsx')
    df = pd.read_excel(f, header=None)
    p = dict(f=float(df.iloc[2, 4]),      # 载波频率 MHz
             Lsys=float(df.iloc[3, 4]),   # 系统损耗 dB
             Lobs=float(df.iloc[4, 4]),   # 地形遮挡附加损耗 dB
             Psens=float(df.iloc[5, 4]),  # 接收灵敏度 dBm
             M=float(df.iloc[6, 4]),      # 衰落裕量 dB
             transport_Pt=float(df.iloc[7, 4]), transport_G=float(df.iloc[8, 4]),
             relay_acc_Pt=float(df.iloc[9, 4]),  relay_acc_G=float(df.iloc[10, 4]),
             relay_bh_Pt=float(df.iloc[11, 4]),  relay_bh_G=float(df.iloc[12, 4]),
             gw_Pt=float(df.iloc[13, 4]),  gw_G=float(df.iloc[14, 4]),
             gw_h=float(df.iloc[15, 4]))   # 网关天线离地高度 m
    return p


# ---------------------------------------------------------------------------
# 2. DEM 加载与空间查询
# ---------------------------------------------------------------------------
class DEMGrid:
    def __init__(self, path=DEM_MAT):
        m = sio.loadmat(path)
        self.dem = m['dem'].astype(np.float64)          # (nlat, nlon)
        self.lat = m['latitude'].ravel().astype(np.float64)   # 降序
        self.lon = m['longitude'].ravel().astype(np.float64)  # 升序
        self.nodata = float(m['nodata'].ravel()[0])
        self.lat_max = self.lat[0]; self.lat_min = self.lat[-1]
        self.lon_min = self.lon[0]; self.lon_max = self.lon[-1]
        self.dlat = abs(self.lat[0] - self.lat[1])      # 降序步长
        self.dlon = self.lon[1] - self.lon[0]
        # NoData 用 NaN 替代以便插值忽略
        dem = self.dem.copy()
        dem[dem == self.nodata] = np.nan
        self._dem_nan = dem
        # max_elev_along 的备忘：游程遍历是纯 Python 循环，而同一对节点坐标会被
        # precompute_geometry、max_safe_payload 的 60 次二分、逐架次能耗重算反复
        # 问到，命中率接近 100%。键用传入的原始浮点坐标（调用方传的是同一批字面量，
        # 精确相等成立），不做舍入，避免"两次问同一线段却给出不同答案"。
        self._maxelev_cache = {}

    def sample(self, lon, lat):
        """双线性采样高程(m). 越界返回 NaN."""
        lon = np.asarray(lon, dtype=float); lat = np.asarray(lat, dtype=float)
        # 列索引 (lon 升序)
        fj = (lon - self.lon_min) / self.dlon
        # 行索引 (lat 降序)
        fi = (self.lat_max - lat) / self.dlat
        i0 = np.floor(fi).astype(int); j0 = np.floor(fj).astype(int)
        di = fi - i0; dj = fj - j0
        out = np.full(lon.shape, np.nan)
        valid = (i0 >= 0) & (i0 + 1 < self.dem.shape[0]) & (j0 >= 0) & (j0 + 1 < self.dem.shape[1])
        i0v = i0[valid]; j0v = j0[valid]
        z = self._dem_nan
        v00 = z[i0v, j0v]; v01 = z[i0v, j0v + 1]
        v10 = z[i0v + 1, j0v]; v11 = z[i0v + 1, j0v + 1]
        di_v = di[valid]; dj_v = dj[valid]
        top = v00 * (1 - dj_v) + v01 * dj_v
        bot = v10 * (1 - dj_v) + v11 * dj_v
        val = top * (1 - di_v) + bot * di_v
        out[valid] = val
        return out

    def _cells_along(self, lon_a, lat_a, lon_b, lat_b):
        """Amanatides–Woo 栅格遍历：返回水平航段穿过的全部 DEM 像元索引 (i, j)。

        像元 (i,j) 覆盖经度 [lon_min + j·dlon, lon_min + (j+1)·dlon]、
        纬度 [lat_max − (i+1)·dlat, lat_max − i·dlat]。从起点像元出发，每步走到
        下一个行边界与列边界中参数距离更近的那一侧，直到抵达终点像元。
        含起终点像元；越界像元丢弃（DEM 之外无高程可查）。
        """
        n_i, n_j = self.dem.shape
        # 分数栅格坐标：列沿经度升序，行沿纬度降序
        fj0 = (lon_a - self.lon_min) / self.dlon
        fi0 = (self.lat_max - lat_a) / self.dlat
        fj1 = (lon_b - self.lon_min) / self.dlon
        fi1 = (self.lat_max - lat_b) / self.dlat
        i, j = int(np.floor(fi0)), int(np.floor(fj0))
        i_end, j_end = int(np.floor(fi1)), int(np.floor(fj1))
        dfi, dfj = fi1 - fi0, fj1 - fj0
        step_i = 0 if abs(dfi) < 1e-15 else (1 if dfi > 0 else -1)
        step_j = 0 if abs(dfj) < 1e-15 else (1 if dfj > 0 else -1)
        if step_i:
            t_max_i = ((i + (1 if step_i > 0 else 0)) - fi0) / dfi
            t_delta_i = 1.0 / abs(dfi)
        else:
            t_max_i = t_delta_i = np.inf
        if step_j:
            t_max_j = ((j + (1 if step_j > 0 else 0)) - fj0) / dfj
            t_delta_j = 1.0 / abs(dfj)
        else:
            t_max_j = t_delta_j = np.inf
        cells = []
        if 0 <= i < n_i and 0 <= j < n_j:
            cells.append((i, j))
        if step_i == 0 and step_j == 0:      # 退化线段（两端点同像元）
            return cells
        # 循环次数上界 = 行跨度 + 列跨度，多给 2 步余量。用固定 range 而非
        # while，是为了在浮点异常时也一定退出，不会把整篇论文卡死在这里。
        for _ in range(abs(i_end - i) + abs(j_end - j) + 2):
            if i == i_end and j == j_end:
                break
            if t_max_i < t_max_j:
                i += step_i; t_max_i += t_delta_i
            else:
                j += step_j; t_max_j += t_delta_j
            if 0 <= i < n_i and 0 <= j < n_j:
                cells.append((i, j))
        return cells

    def max_elev_along(self, lon_a, lat_a, lon_b, lat_b):
        """航段所经过 DEM 像元的最高地面高程(m)。

        口径取自题面附录 2——"计划巡航海拔取该航段所经过 DEM 像元的最高地面
        高程以上 50 米"：遍历航段穿过的每一个 30m 像元，取原始高程最大值。

        这里刻意不改为对双线性插值面采样。插值值是像元四角高程的凸组合，
        其沿线最大值结构性地不高于原始像元最大值，用它定巡航高度会把高度
        系统性压低，进而低估爬升量、飞行时间与能耗——是**非保守**误差。
        反方向也不干净：航线贴近像元边缘时，插值会把并未进入的邻元高程混进来，
        于是同一个量在两个方向上都有偏。逐像元取原始值则无此问题。
        """
        key = (lon_a, lat_a, lon_b, lat_b)
        hit = self._maxelev_cache.get(key)
        if hit is not None:
            return hit
        cells = self._cells_along(lon_a, lat_a, lon_b, lat_b)
        out = 0.0
        if cells:
            ii = np.fromiter((c[0] for c in cells), dtype=np.intp, count=len(cells))
            jj = np.fromiter((c[1] for c in cells), dtype=np.intp, count=len(cells))
            z = self._dem_nan[ii, jj]
            z = z[np.isfinite(z)]
            if z.size:
                out = float(z.max())
        self._maxelev_cache[key] = out
        return out

    def line_occluded(self, p_a, p_b, step=DEM_STEP):
        """
        视线遮挡判定：p_a=(lon,lat,alt_abs_m), p_b 同理。
        沿三维连线采样，判断连线上方是否有 DEM 高于连线。返回 bool。
        """
        xa, ya = ll_to_xy(p_a[0], p_a[1]); xb, yb = ll_to_xy(p_b[0], p_b[1])
        d = float(np.hypot(xb - xa, yb - ya))
        n = max(2, int(np.ceil(d / step)) + 1)
        tt = np.linspace(0, 1, n)
        lons = p_a[0] + (p_b[0] - p_a[0]) * tt
        lats = p_a[1] + (p_b[1] - p_a[1]) * tt
        line_alt = p_a[2] + (p_b[2] - p_a[2]) * tt
        z = self.sample(lons, lats)
        finite = np.isfinite(z)
        if finite.sum() == 0:
            # 防御性分支，实测永不触发：本数据集 DEM 为 1309x1486，NoData 占比 0，
            # 任一落在范围内的连线都必然采到有效高程（越界只是两端点之外，不影响判据）。
            # 之所以保留 fail-open（判为不遮挡）而不是抛异常，是为了让一个残缺的 DEM
            # 不至于把通信可行性整体判死、连带把 q3/q4 全部拖崩。若将来换用带真实
            # NoData 空洞的 DEM，此处应改为按"不可判定"上报，而不是静默放行。
            return False
        # 若任一 DEM 点高于连线（留极小容差），则遮挡
        return bool(np.any(z[finite] > line_alt[finite] + 1e-3))


# ---------------------------------------------------------------------------
# 3. 运输航段物理模型
# ---------------------------------------------------------------------------
def equivalent_range(t, q):
    """机型 t 携带载荷 q(kg) 的等效航程(m)。"""
    q = float(q)
    if q < 0:
        q = 0.0
    if q > t['Q']:
        q = t['Q']
    return t['L0'] - (t['L0'] - t['LF']) * (q / t['Q']) ** 1.5


def segment_geometry(dem, lon_a, lat_a, alt_a, lon_b, lat_b, alt_b):
    """
    计算两节点间航段的几何量。
    alt_a/alt_b 为作业海拔（绝对高度, m，相对海平面）。
    返回 dict: d(水平距离m), cruise_alt, climb, descent。
    计划巡航海拔 = max(沿线最高地面高程+50, 起终作业高度)。
    """
    xa, ya = ll_to_xy(lon_a, lat_a); xb, yb = ll_to_xy(lon_b, lat_b)
    d = float(np.hypot(xb - xa, yb - ya))
    max_e = dem.max_elev_along(lon_a, lat_a, lon_b, lat_b)
    cruise_alt = max(max_e + 50.0, alt_a, alt_b)
    climb = max(0.0, cruise_alt - alt_a)
    descent = max(0.0, cruise_alt - alt_b)
    return dict(d=d, cruise_alt=cruise_alt, climb=climb, descent=descent)


def segment_time(t, geo):
    """航段飞行时间(s)：爬升 + 巡航 + 下降。"""
    return (geo['climb'] / t['v_up']
            + geo['d'] / t['v_cruise']
            + geo['descent'] / t['v_down'])


def segment_energy(t, geo, q):
    """
    航段运输能耗(kWh) = 水平巡航能耗 + 爬升附加能耗。
    水平巡航能耗 E_hor = E_use * d / L(q)（由等效航程反推）。
    爬升附加能耗 E_up = m_total * g * climb / eta_up（势能/效率）。
    q 为该航段剩余载荷(kg)。
    """
    q = float(q)
    L = equivalent_range(t, q)
    E_hor = t['E_use'] * geo['d'] / L if L > 0 else np.inf
    m_total = t['empty_mass'] + q
    E_up = (m_total * G * geo['climb'] / t['eta_up']) / J_PER_KWH
    return E_hor + E_up


def max_safe_payload(d, t, si):
    """机型 t 在服务区 si 的最大安全载荷(kg)：往返能耗≤(1-ρ)E_use 的最大 q。"""
    usable = (1 - t['rho']) * t['E_use']

    def energy(q):
        E, _, _ = roundtrip_energy(t, d.dem, d.O01, si, q)
        return E

    if energy(t['Q']) <= usable:
        return t['Q']
    lo, hi = 0.0, t['Q']
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if energy(mid) <= usable:
            lo = mid
        else:
            hi = mid
    return lo


def roundtrip_energy(t, dem, O01, si, q_out):
    """O01->Si(载q_out)->O01(空载) 往返总能耗(kWh)。"""
    # 作业高度
    alt_o = O01['alt']
    alt_s = si['alt'] + 30.0
    go = segment_geometry(dem, O01['lon'], O01['lat'], alt_o, si['lon'], si['lat'], alt_s)
    back = segment_geometry(dem, si['lon'], si['lat'], alt_s, O01['lon'], O01['lat'], alt_o)
    E = segment_energy(t, go, q_out) + segment_energy(t, back, 0.0)
    return E, go, back


# ---------------------------------------------------------------------------
# 4. 充电模型（两阶段等效充电）
# ---------------------------------------------------------------------------
def charge_time(soc_end, Tfull):
    """从任务结束 SOC=soc_end(0..1) 充至 100% 所需时间(s)。"""
    s = float(soc_end)
    if s < 0:
        s = 0.0
    if s > 1:
        s = 1.0
    if s < 0.90:
        return Tfull * (0.65 * (0.90 - s) / 0.90 + 0.35)
    else:
        return Tfull * 0.35 * (1 - s) / 0.10


# ---------------------------------------------------------------------------
# 5. 中继无人机物理模型
# ---------------------------------------------------------------------------
def relay_flight_time(t, geo):
    return (geo['climb'] / t['v_up'] + geo['d'] / t['v_cruise']
            + geo['descent'] / t['v_down'])


def relay_flight_energy(t, geo):
    """中继无人机航段能耗：水平巡航功率*时间 + 爬升附加能耗(取起飞总质量)。"""
    E_hor = t['P_cruise'] * (geo['d'] / t['v_cruise']) / 3600.0   # kWh
    E_up = (t['takeoff_mass'] * G * geo['climb'] / t['eta_up']) / J_PER_KWH
    return E_hor + E_up


def relay_hover_energy(t, hover_seconds):
    """悬停通信服务能耗：悬停功率 + 通信附加功率。"""
    return (t['P_hover'] + t['P_comm']) * hover_seconds / 3600.0   # kWh


# ---------------------------------------------------------------------------
# 6. 通信链路模型
# ---------------------------------------------------------------------------
class CommModel:
    def __init__(self, p):
        self.p = p
        self.Pth = p['Psens'] + p['M']   # 有效接收门限 dBm
        # 双向链路门限
        self.Lmax_uw_gw = self._bidir(p['transport_Pt'], p['transport_G'],
                                      p['gw_Pt'], p['gw_G'])
        self.Lmax_uw_r = self._bidir(p['transport_Pt'], p['transport_G'],
                                     p['relay_acc_Pt'], p['relay_acc_G'])
        self.Lmax_r_gw = self._bidir(p['relay_bh_Pt'], p['relay_bh_G'],
                                     p['gw_Pt'], p['gw_G'])

    def _bidir(self, Pta, Gta, Ptb, Gtb):
        a2b = Pta + Gta + Gtb - self.p['Lsys'] - self.Pth
        b2a = Ptb + Gtb + Gta - self.p['Lsys'] - self.Pth
        return min(a2b, b2a)

    def fspl(self, dist_km):
        return 32.45 + 20 * np.log10(self.p['f']) + 20 * np.log10(dist_km)

    def link_ok(self, dem, pa, pb, Lmax):
        """两点间链路是否可用（含地形遮挡）。pa/pb=(lon,lat,alt_abs_m)。"""
        xa, ya = ll_to_xy(pa[0], pa[1]); xb, yb = ll_to_xy(pb[0], pb[1])
        dx = float(xb - xa); dy = float(yb - ya); dz = float(pa[2] - pb[2])
        dist_km = np.sqrt(dx * dx + dy * dy + dz * dz) / 1000.0
        L_fspl = self.fspl(dist_km)
        occ = dem.line_occluded(pa, pb)
        L_path = L_fspl + self.p['Lobs'] * (1.0 if occ else 0.0)
        return L_path <= Lmax, L_path, occ


# ---------------------------------------------------------------------------
# 全局单例加载
# ---------------------------------------------------------------------------
class Data:
    def __init__(self):
        self.O01, self.services = load_nodes()
        self.transport_types = load_transport_uav_types()
        self.uavs = load_transport_uavs()
        self.batteries = load_batteries()
        self.relay_type = load_relay_type()
        self.relay_uavs = load_relay_uavs()
        self.relay_batteries = load_relay_batteries()
        self.cargo = load_cargo()
        self.comm_params = load_comm_params()
        self.dem = DEMGrid()
        self.comm = CommModel(self.comm_params)
        # 服务区索引
        self.si = {s['id']: s for s in self.services}
        # 货箱按服务区分组
        self.boxes_by_service = {}
        for b in self.cargo:
            self.boxes_by_service.setdefault(b['service'], []).append(b)
        # 节点编号 S001..S015
        self.S = [s['id'] for s in self.services]


def load_data():
    return Data()


if __name__ == '__main__':
    d = load_data()
    print('O01:', d.O01)
    print('services:', len(d.services))
    print('transport types:', {k: (v['Q'], v['volume'], v['E_use']) for k, v in d.transport_types.items()})
    print('batteries:', d.batteries)
    print('relay:', d.relay_type['takeoff_mass'], d.relay_type['E_use'])
    print('relay batteries:', d.relay_batteries)
    print('cargo boxes:', len(d.cargo))
    print('comm Lmax uw-gw:', d.comm.Lmax_uw_gw, 'uw-r:', d.comm.Lmax_uw_r, 'r-gw:', d.comm.Lmax_r_gw)
    print('DEM:', d.dem.dem.shape, 'elev range', np.nanmin(d.dem._dem_nan), np.nanmax(d.dem._dem_nan))
