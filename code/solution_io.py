# -*- coding: utf-8 -*-
# 本程序及代码是在人工智能工具辅助下完成的。
# 工具名称：DeepSeek；版本/型号：DeepSeek V4.1 Flash（API 模型名 deepseek-flash）；
# 开发机构：杭州深度求索人工智能基础技术研究有限公司；
# 版本发布日期：2026 年 9 月 10 日。

"""
跨问题结果接口（问题二 → 问题三 → 问题四 的唯一真相源）

解决的问题：旧版问题三、问题四**重新调用调度器／按排序位置猜编号**来复原上游
方案。调度器语义一旦改动，同一份缓存会还原成另一套实体指派；而按「排序后下标」
猜架次关联更会在上游筛掉若干架次时整体错位（如失效表只含 T03 起，T03 会被映射
到 0 号位）。两者都是**静默**错误——算出来的数字看着合理，关联却是错的。

本模块给出三条规矩：

 1. 方案落盘时保存**完整最终方案**，不只是生成方案的输入（路线 + 顺序）。每个
    架次带稳定 `trip_id`（问题三给中继架次编 `relay_trip_id`），**编号只在方案
    生成时定一次**，之后排序、筛选、绘图一律不得重新编号。
 2. 下游只按 `trip_id` / `relay_trip_id` 做**外键连接**，不从下标、排序位置或
    「T 编号减 1」反推关联。
 3. 引用校验**显式失败**：未知编号、空引用、字段缺失一律抛异常并列出全部问题项，
    不静默跳过、不降级继续。

`solution_id` 是内容摘要的确定性函数：只要方案内容变了，id 必变，下游据此拒绝
使用陈旧的上游文件。
"""
import hashlib
import json
import os

SCHEMA_VERSION = '1.0'

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
DATA_DIR = os.path.join(ROOT, '数据')

# 参与 solver_hashes 的源码文件。按阶段分开：只把**真正决定该阶段结果**的模块
# 计入，否则改 q4.py 也会让 q2_solution.json 判为过期，哈希校验就失去意义了。
Q2_SOLVER_FILES = ['core.py', 'q2.py', 'solution_io.py']
Q3_SOLVER_FILES = ['core.py', 'q2.py', 'q3.py', 'solution_io.py']

REQUIRED_FIELDS = ['schema_version', 'solution_id', 'stage', 'input_hashes',
                   'solver_hashes', 'seed', 'budget']


# ---------------------------------------------------------------------------
# 1. 哈希与摘要
# ---------------------------------------------------------------------------
def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _rel(path):
    return os.path.relpath(path, ROOT).replace('\\', '/')


def input_hashes(data_dir=None):
    """原始数据文件（含子目录）的 SHA-256 清单。数据换了，方案就不再有效。"""
    base = DATA_DIR if data_dir is None else data_dir
    out = {}
    for dirpath, _dirs, files in os.walk(base):
        for fn in sorted(files):
            if fn.startswith('~$') or fn.startswith('.'):
                continue
            p = os.path.join(dirpath, fn)
            out[_rel(p)] = sha256_file(p)
    return dict(sorted(out.items()))


SOLVER_FILES = Q3_SOLVER_FILES   # 兼容旧引用：默认取最全的一组


def solver_hashes(files=None):
    """求解器源码的 SHA-256 清单。代码改了，方案就不再可复现。"""
    out = {}
    for fn in (SOLVER_FILES if files is None else files):
        p = os.path.join(HERE, fn)
        if os.path.exists(p):
            out['code/' + fn] = sha256_file(p)
    return dict(sorted(out.items()))


def canonical_bytes(obj):
    """规范序列化：键排序、全精度、不转义非 ASCII。同内容必得同字节。"""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def content_id(stage, sol):
    """方案 id = 内容摘要。计算时剔除 solution_id 自身，得 'q2-xxxxxxxxxxxx'。"""
    body = {k: v for k, v in sol.items() if k != 'solution_id'}
    return '%s-%s' % (stage, hashlib.sha256(canonical_bytes(body)).hexdigest()[:12])


# ---------------------------------------------------------------------------
# 2. 稳定编号
# ---------------------------------------------------------------------------
def trip_id_of(rank):
    return 'T%02d' % (rank + 1)


def relay_trip_id_of(rank):
    return 'R%02d' % (rank + 1)


def assign_trip_ids(assignment):
    """
    按**开始时刻**升序给运输架次编 T01..Tn，同刻按原下标定序（确定性）。
    只在方案定稿时调用一次；返回与 assignment 等长的 id 列表。
    """
    ids = [None] * len(assignment)
    for rank, i in enumerate(sorted(range(len(assignment)),
                                    key=lambda i: (assignment[i]['start'], i))):
        ids[i] = trip_id_of(rank)
    return ids


def assign_relay_trip_ids(relays):
    """按开始时刻升序给中继架次编 R01..Rn，同刻按原下标定序。"""
    ids = [None] * len(relays)
    for rank, i in enumerate(sorted(range(len(relays)),
                                    key=lambda i: (relays[i]['start'], i))):
        ids[i] = relay_trip_id_of(rank)
    return ids


# ---------------------------------------------------------------------------
# 3. 读写
# ---------------------------------------------------------------------------
def save_solution(path, sol):
    """写盘前先自校验；solution_id 由内容算出并回填，不用调用方提供。"""
    sol = dict(sol)
    sol.pop('solution_id', None)
    sol['schema_version'] = SCHEMA_VERSION
    sol['solution_id'] = content_id(sol['stage'], sol)
    validate_solution(sol)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(sol, f, ensure_ascii=False, indent=1, sort_keys=True)
    os.replace(tmp, path)
    return sol['solution_id']


def load_solution(path, stage=None, source_solution_id=None,
                  input_hashes_expect=None, solver_hashes_expect=None):
    """
    读方案并**逐项校验**：字段完整、内容摘要自洽、阶段正确、来源 id 一致。
    任一项不符即抛异常——陈旧或串了来源的上游文件必须当场暴露，不能带着往下算。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'未找到方案文件 {path}；请先运行上游求解脚本生成它')
    with open(path, 'r', encoding='utf-8') as f:
        sol = json.load(f)
    validate_solution(sol)
    got = content_id(sol['stage'], sol)
    if got != sol['solution_id']:
        raise ValueError(
            f'{os.path.basename(path)} 内容摘要不符：文件内 {sol["solution_id"]}，'
            f'按内容重算 {got}——文件被改过或序列化有损')
    if stage is not None and sol['stage'] != stage:
        raise ValueError(f'{os.path.basename(path)} 阶段不符：期望 {stage}，实为 {sol["stage"]}')
    if source_solution_id is not None and sol.get('source_solution_id') != source_solution_id:
        raise ValueError(
            f'{os.path.basename(path)} 来源方案不符：期望 {source_solution_id}，'
            f'实为 {sol.get("source_solution_id")}——上游已重算，本文件已过期')
    if input_hashes_expect is not None:
        _diff('输入数据', input_hashes_expect, sol.get('input_hashes', {}))
    if solver_hashes_expect is not None:
        _diff('求解器源码', solver_hashes_expect, sol.get('solver_hashes', {}))
    return sol


def _diff(what, expect, got):
    bad = [k for k in sorted(set(expect) | set(got)) if expect.get(k) != got.get(k)]
    if bad:
        raise ValueError(f'{what}与方案记录不一致（{len(bad)} 项）：' + '、'.join(bad[:5]))


def validate_solution(sol):
    miss = [k for k in REQUIRED_FIELDS if k not in sol]
    if miss:
        raise ValueError('方案缺少必需字段：' + '、'.join(miss))
    if sol['schema_version'] != SCHEMA_VERSION:
        raise ValueError(f'方案 schema_version={sol["schema_version"]}，'
                         f'本模块只认 {SCHEMA_VERSION}')
    if sol['stage'] not in ('q2', 'q3'):
        raise ValueError(f'未知阶段 {sol["stage"]}')
    for k in ('input_hashes', 'solver_hashes'):
        if not isinstance(sol[k], dict) or not sol[k]:
            raise ValueError(f'{k} 必须是非空字典')
    if not isinstance(sol['seed'], int):
        raise ValueError('seed 必须是整数')
    if not isinstance(sol['budget'], dict):
        raise ValueError('budget 必须是字典')
    if sol['stage'] == 'q2':
        _validate_trips(sol)
    else:
        if not sol.get('source_solution_id'):
            raise ValueError('q3 方案必须记录 source_solution_id')
        _validate_trips(sol)
        _validate_relays(sol)
    return True


def _validate_trips(sol):
    trips = sol.get('transport_trips')
    if not isinstance(trips, list) or not trips:
        raise ValueError('transport_trips 必须是非空列表')
    seen = set()
    for t in trips:
        for k in ('trip_id', 'type', 'uav', 'battery', 'start', 'duration',
                  'box_ids_by_service'):
            if k not in t:
                raise ValueError(f'架次记录缺少字段 {k}：{t}')
        if t['trip_id'] in seen:
            raise ValueError(f'架次编号重复：{t["trip_id"]}')
        seen.add(t['trip_id'])
        if t['duration'] <= 0:
            raise ValueError(f'{t["trip_id"]} 作业时长非正：{t["duration"]}')
        if not t['box_ids_by_service']:
            raise ValueError(f'{t["trip_id"]} 没有服务任何服务区')
    ids = [t['trip_id'] for t in trips]
    if sorted(ids) != [trip_id_of(i) for i in range(len(trips))]:
        raise ValueError('架次编号不是连续的 T01..T%02d：%s' % (len(trips), sorted(ids)))
    return True


def _validate_relays(sol):
    relays = sol.get('relay_trips')
    if relays is None:
        raise ValueError('q3 方案必须含 relay_trips')
    seen = set()
    for r in relays:
        for k in ('relay_trip_id', 'relay_uav_id', 'component_id', 'station_id',
                  'start', 'return_time'):
            if k not in r:
                raise ValueError(f'中继架次记录缺少字段 {k}：{r}')
        if r['relay_trip_id'] in seen:
            raise ValueError(f'中继架次编号重复：{r["relay_trip_id"]}')
        seen.add(r['relay_trip_id'])
    return True


# ---------------------------------------------------------------------------
# 4. 外键校验（下游连接的唯一合法方式）
# ---------------------------------------------------------------------------
def trip_ids(sol):
    return [t['trip_id'] for t in sol['transport_trips']]


def relay_trip_ids(sol):
    return [r['relay_trip_id'] for r in (sol.get('relay_trips') or [])]


def trip_of(sol):
    return {t['trip_id']: t for t in sol['transport_trips']}


def relay_of(sol):
    return {r['relay_trip_id']: r for r in (sol.get('relay_trips') or [])}


def inherit_trip_ids(assignment, upstream_sol, tol=1e-6):
    """
    把上游方案（q2_solution）的 `trip_id` 接到**按开始时刻升序**排好的 assignment 上，
    并逐架次核对实体指派确实一致。

    这是「下游不得重跑调度器后自顾自编号」的执行点：`schedule()` 的语义一旦改变，
    同一份缓存会还原成另一套无人机/电池指派，此处立刻抛异常，而不是带着一份
    看着合理、关联却已错位的方案继续往下算。
    """
    ups = upstream_sol['transport_trips']          # 已按 trip_id 升序
    if len(assignment) != len(ups):
        raise ValueError('架次数不符：下游 %d，上游方案 %d——上游方案已过期'
                         % (len(assignment), len(ups)))
    for a, u in zip(assignment, ups):
        for f in ('type', 'uav', 'battery'):
            if a[f] != u[f]:
                raise ValueError(
                    f'{u["trip_id"]} 实体指派与上游方案不符：{f} 上游={u[f]!r} 重解={a[f]!r}'
                    f'——调度器语义已变，须重新生成问题二方案')
        for f, got in (('start', a['start']), ('duration', a['duration']),
                       ('energy_kwh', a['E'])):
            if abs(got - u[f]) > tol * max(1.0, abs(u[f])):
                raise ValueError(
                    f'{u["trip_id"]} {f} 与上游方案不符：上游={u[f]!r} 重解={got!r}')
    return [u['trip_id'] for u in ups]


def check_foreign_keys(sol, intervals=None, require_relay_binding=True):
    """
    校验方案内部及（可选的）失效区间表的外部引用。

    `intervals` 为 None 时只查方案内部一致性；传入区间记录列表时额外检查
    每条区间的 trip_id / relay_trip_id 都能在本方案中解析。
    **任何**未解析引用都收集后一次性抛出，不做静默跳过。
    """
    errs = []
    tids, rids = set(trip_ids(sol)), set(relay_trip_ids(sol))
    known = {'relay_uav_id': set(sol.get('relay_uav_ids') or []),
             'station_id': set(sol.get('station_ids') or []),
             'component_id': set(sol.get('component_ids') or [])}
    for r in (sol.get('relay_trips') or []):
        for field, pool in known.items():
            if pool and r[field] not in pool:
                errs.append(f'{r["relay_trip_id"]} 引用了未知 {field}={r[field]!r}')
    for i, iv in enumerate(intervals or []):
        tag = iv.get('interval_id') or f'第{i}条'
        tid = iv.get('trip_id')
        if tid not in tids:
            errs.append(f'失效区间 {tag} 引用未知运输架次 {tid!r}')
        rid = iv.get('relay_trip_id')
        if rid:
            if rid not in rids:
                errs.append(f'失效区间 {tag} 引用未知中继架次 {rid!r}')
        elif require_relay_binding:
            errs.append(f'失效区间 {tag} 没有绑定任何中继架次（引用为空）')
    if errs:
        raise ValueError('外键校验失败（%d 项）：\n  ' % len(errs) + '\n  '.join(errs[:20]))
    return True


# ---------------------------------------------------------------------------
# 5. 方案构造
# ---------------------------------------------------------------------------
def build_q2_solution(d, asg, m, order, info, seed, data_dir=None, code_dir=None):
    """
    由问题二的最终指派构造方案记录（**完整最终方案**，含实体指派与逐箱交付时刻）。

    `asg` 为 schedule() 的输出（未排序、未平移）；编号按开始时刻定序后一次定死。
    """
    ids = assign_trip_ids(asg)
    trips = []
    for a, tid in zip(asg, ids):
        trips.append(dict(
            trip_id=tid, type=a['type'], uav=a['uav'], battery=a['battery'],
            route=list(a['route']), start=float(a['start']),
            duration=float(a['duration']), return_time=float(a['start'] + a['duration']),
            energy_kwh=float(a['E']),
            box_ids_by_service={s: [b['id'] for b in bs] for s, bs in a['boxes_at'].items()},
            deliver_abs={b['id']: float(a['deliver_abs'][b['id']])
                         for s, bs in a['boxes_at'].items() for b in bs}))
    trips.sort(key=lambda t: t['trip_id'])
    uavs = sorted({t['uav'] for t in trips})
    sol = dict(
        stage='q2', source_solution_id=None,
        seed=int(seed),
        budget=dict(n_iter=int(info.get('n_iter', 0)), K=int(info.get('K', 0)),
                    K_targets=[int(k) for k in info.get('K_targets', [])]),
        input_hashes=input_hashes(data_dir),
        solver_hashes=solver_hashes(Q2_SOLVER_FILES),
        metrics={k: float(v) for k, v in m.items()},
        order=[int(i) for i in order],
        transport_trips=trips,
        uav_ids=uavs,
        battery_ids=sorted({t['battery'] for t in trips}),
    )
    return sol


def build_q3_solution(d, q2_sol, assignment, relays, intervals, st_no, m,
                      seed, budget, data_dir=None):
    """
    由问题三的最终错峰方案构造方案记录。

    运输架次**沿用问题二的 trip_id**（问题三只平移开始时刻，不重新分批，
    架次身份不变），中继架次在此新编 relay_trip_id。
    """
    # 架次身份来自上游：问题二方案里的 T 编号按开始时刻定序，此处 assignment
    # 也已按开始时刻排序，故按下标一一承接。数量必须相等，否则上游已变。
    q2_trips = q2_sol['transport_trips']
    if len(assignment) != len(q2_trips):
        raise ValueError('问题三运输架次数 %d 与问题二方案 %d 不符，上游方案已变'
                         % (len(assignment), len(q2_trips)))
    base = {t['trip_id']: t for t in q2_trips}
    tids = [t['trip_id'] for t in q2_trips]
    trips = []
    for a, tid in zip(assignment, tids):
        b = base[tid]
        trips.append(dict(
            trip_id=tid, type=a['type'], uav=a['uav'], battery=a['battery'],
            route=list(a['route']), start=float(a['start']),
            duration=float(a['duration']), return_time=float(a['start'] + a['duration']),
            energy_kwh=float(a['E']),
            box_ids_by_service={s: [x['id'] for x in bs]
                                for s, bs in a['boxes_at'].items()},
            deliver_abs={x['id']: float(a['deliver_abs'][x['id']])
                         for s, bs in a['boxes_at'].items() for x in bs},
            base_start=float(b['start'])))

    # 稳定排序：同刻架次保持列表原序（schedule_relays 已按开始时刻排过），
    # 不用 id() 之类跨进程不稳定的键
    relays = sorted(relays, key=lambda x: x['start'])
    rids = assign_relay_trip_ids(relays)
    relay_recs = []
    for x, rid in zip(relays, rids):
        relay_recs.append(dict(
            relay_trip_id=rid, relay_uav_id=x['relay'], component_id=x['comp'],
            station_id=st_no.get(x['station'], ''),
            start=float(x['start']), link_done=float(x['link_done']),
            service_end=float(x['service_end']), return_time=float(x['return_t']),
            energy_kwh=float(x['E']), hover_agl_m=float(x['hover_agl']),
            interval_ids=['G%02d' % (j + 1) for j in sorted(x['ivs'])],
            late=bool(x['late'])))

    iv_recs = []
    for j, iv in enumerate(intervals):
        iv_recs.append(dict(interval_id='G%02d' % (j + 1),
                            trip_id=tids[iv['trip']],
                            start=float(iv['t_start']), end=float(iv['t_end'])))

    relay_by_iv = {}
    for x, rid in zip(relays, rids):
        for j in x['ivs']:
            relay_by_iv[j] = rid
    for j, rec in enumerate(iv_recs):
        rec['relay_trip_id'] = relay_by_iv.get(j, '')

    sol = dict(
        stage='q3',
        source_solution_id=q2_sol['solution_id'],
        seed=int(seed),
        budget=dict(budget),
        input_hashes=input_hashes(data_dir),
        solver_hashes=solver_hashes(Q3_SOLVER_FILES),
        # 指标里并非全是数：min_hard_margin_box 记的是「余量最小的那个货箱编号」
        # （如 S008-WAT-01），供正文点名用。一律 float() 会在导出的最后一步炸掉，
        # 前面十几分钟的求解全部白跑。数值项转 float，其余原样保留。
        metrics={k: (float(v) if isinstance(v, (int, float)) and not isinstance(v, bool)
                     else v) for k, v in m.items()},
        transport_trips=trips,
        relay_trips=relay_recs,
        intervals=iv_recs,
        relay_uav_ids=sorted({r['relay_uav_id'] for r in relay_recs}),
        component_ids=sorted({r['component_id'] for r in relay_recs}),
        station_ids=sorted({r['station_id'] for r in relay_recs if r['station_id']}),
    )
    return sol
