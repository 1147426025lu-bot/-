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

哈希口径（三句话写死，改动前先想清楚会让哪些方案记录失效）：

 1. **原始输入一律按字节校验**（`input_hashes` → `sha256_file`），不做任何规范化。
    数据文件是二进制与第三者交付物，规范化它没有意义，只会削弱证据。
 2. **求解器源码同时记两套**：`solver_hashes` 是发布文件的字节哈希，判定「方案是否
    过期」只认它，逐字节严格；`solver_text_hashes` 把换行统一成 LF 后再算，**不参与
    判定**，只用来分辨不一致的原因（换行差异 vs 真改码）。
 3. 源码编码统一 UTF-8，换行由 `.gitattributes` 逐文件锁定（`core.py`/`q2.py` 为
    CRLF，`q3.py`/`q4.py`/`solution_io.py`/`q2_v2.py`/`q3_v2.py` 为 LF）。换行风格是
    记录的一部分，不是可有可无的格式偏好——字节哈希认的就是它。方案结构版本记在
    `SCHEMA_VERSION`。

这两套哈希合起来的作用是让「刷新哈希」不再是可选项：字节不一致时，文本哈希一致
即证明代码逐字未动（改回换行即可），文本哈希不一致即证明代码真的改了（必须重跑）。
见 `_diff_solver`。
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
# q2_v2.py / q3_v2.py 是阶段 11 移植来的算法层（--engine v2、解法二并列路径）。
# 它们不在这个表里就等于「论文的数来自一个不在记录里的文件」，故一并计入。
Q2_SOLVER_FILES = ['core.py', 'q2.py', 'q2_v2.py', 'solution_io.py']
Q3_SOLVER_FILES = ['core.py', 'q2.py', 'q2_v2.py', 'q3.py', 'q3_chain.py',
                   'q3_v2.py', 'solution_io.py']

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


def text_hash_file(path):
    """源码的**文本规范化**哈希：换行统一成 LF 后再按字节算 SHA-256。

    它不是 `sha256_file` 的替代品，而是用来**把两种字节差异分开**。核查源码
    身份时「字节不一致」有两种原因，处置正好相反：换行符变了（代码一个字没动）
    与代码真改了。只留字节哈希，这两者在方案记录里长得一模一样，读者无从判断
    该改回换行还是该重跑——于是最省事的做法就成了「刷新哈希」，而那恰好会把
    第二种（真改动）一并抹掉。正是这个歧义让「刷新哈希」看起来可用。

    刻意**不**解码成字符串：编码变了同样属于"内容变了"，必须按真改动处理。
    换行规范由 `.gitattributes` 逐文件锁定（core.py/q2.py 为 CRLF，其余源码
    为 LF），此处的归一化只用于判断，不改变发布文件的字节。

    归一化口径写死在此函数里，不开参数：口径一旦可调，两个不同的口径就能算出
    同一个值，「哈希一致」也就不再是任何事情的证据。
    """
    with open(path, 'rb') as f:
        b = f.read()
    return hashlib.sha256(b.replace(b'\r\n', b'\n').replace(b'\r', b'\n')).hexdigest()


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
    """求解器源码的**字节** SHA-256 清单。代码改了，方案就不再可复现。

    这是发布文件的字节哈希，也是判定「方案是否过期」的依据——它保持逐字节
    严格，不因换行规范化而放松。文本规范化哈希另见 `solver_hashes_text`，
    只用于诊断不一致的原因，不参与判定。
    """
    out = {}
    for fn in (SOLVER_FILES if files is None else files):
        p = os.path.join(HERE, fn)
        if os.path.exists(p):
            out['code/' + fn] = sha256_file(p)
    return dict(sorted(out.items()))


def solver_hashes_text(files=None):
    """与 `solver_hashes` **同键**的文本规范化哈希清单（换行统一成 LF）。

    两套哈希并列存进方案记录，才能回答「这次不一致是换行还是真改码」。
    两套都缺一不可：只存规范化哈希会漏掉编码/行内空白之外的真实改动判定，
    只存字节哈希则无法把误报和真改动分开。
    """
    out = {}
    for fn in (SOLVER_FILES if files is None else files):
        p = os.path.join(HERE, fn)
        if os.path.exists(p):
            out['code/' + fn] = text_hash_file(p)
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
                  input_hashes_expect=None, solver_hashes_expect=None,
                  solver_text_hashes_expect=None):
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
        _diff_solver(solver_hashes_expect, sol.get('solver_hashes', {}),
                     solver_text_hashes_expect, sol.get('solver_text_hashes', {}))
    return sol


def _diff(what, expect, got):
    bad = [k for k in sorted(set(expect) | set(got)) if expect.get(k) != got.get(k)]
    if bad:
        raise ValueError(f'{what}与方案记录不一致（{len(bad)} 项）：' + '、'.join(bad[:5]))


def _diff_solver(expect, got, expect_text, got_text):
    """源码字节哈希不一致时，用文本规范化哈希把两种原因分开报出来。

    只说「求解器源码已变」，读者无法区分「换行符被编辑器改了」和「代码真改了」
    ——而这两者的正确处置完全相反：前者改回换行即可，后者必须重跑。混成一条
    消息的后果是，最省事的处置（刷新哈希）对两种情况都"能用"，于是真改动也
    一并被抹掉。这里把原因写进异常，让误报和真改动各自可辨认、各自可处置。
    """
    bad = [k for k in sorted(set(expect) | set(got)) if expect.get(k) != got.get(k)]
    if not bad:
        return
    eol_only, real = [], []
    for k in bad:
        et, gt = (expect_text or {}).get(k), (got_text or {}).get(k)
        if et is not None and et == gt:
            eol_only.append(k)
        else:
            real.append(k)
    lines = [f'求解器源码与方案记录不一致（{len(bad)} 项）：' + '、'.join(bad[:5])]
    if eol_only:
        lines.append(
            '  · 仅换行符不同、文本内容逐字一致（%d 项：%s）——代码没动，这是误报；'
            '把换行改回 .gitattributes 规定的风格即可，不要刷新哈希'
            % (len(eol_only), '、'.join(eol_only[:5])))
    if real:
        lines.append(
            '  · 文本内容也已改变（%d 项：%s）——代码确有改动，方案记录必须作废，'
            '重跑该阶段，不得刷新哈希'
            % (len(real), '、'.join(real[:5])))
    raise ValueError('\n'.join(lines))


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
        solver_text_hashes=solver_hashes_text(Q2_SOLVER_FILES),
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

    # 稳定排序：同刻按架次内序（同一次出动的多次访问共享开始时刻，必须按访问
    # 顺序排，否则 R 编号会在同一次出动内部乱序）。刻意不用 id() 之类跨进程
    # 不稳定的键。
    relays = sorted(relays, key=lambda x: (x['start'], x.get('seq', 0)))
    rids = assign_relay_trip_ids(relays)
    relay_recs = []
    for x, rid in zip(relays, rids):
        # 一行 = 一次悬停站服务。`energy_kwh` 是**整次出动**的能耗（同一次出动的
        # 各行取同值，读表要按 outing_id 去重），`visit_energy_kwh` 是本次访问的
        # 份额（进场航段 + 本站悬停，末次访问另含末端返航），逐行相加等于前者。
        relay_recs.append(dict(
            relay_trip_id=rid, relay_uav_id=x['relay'], component_id=x['comp'],
            station_id=st_no.get(x['station'], ''),
            outing_id=x['outing'], seq_in_outing=int(x['seq']),
            start=float(x['start']), link_done=float(x['link_done']),
            service_end=float(x['service_end']), return_time=float(x['return_t']),
            energy_kwh=float(x['E']), visit_energy_kwh=float(x['visit_E']),
            hover_agl_m=float(x['hover_agl']),
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
        solver_text_hashes=solver_hashes_text(Q3_SOLVER_FILES),
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
