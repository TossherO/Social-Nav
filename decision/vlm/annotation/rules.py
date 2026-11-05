"""Human labels, deliberately separate from detector-center supervision."""
import hashlib
import json
import math


STATES = ('draft', 'skipped', 'complete')
STATE_NAMES = {'draft': '待标', 'skipped': '跳过', 'complete': '已标'}
RELATIONS = {'w': '同行', 't': '交谈', 'p': '拍照', 'o': '其他共同活动', 'u': '活动未知'}
VISIBILITY = {'clear': '清晰可见', 'partial': '部分遮挡', 'occluded': '完全遮挡',
              'outside': '视野外', 'unknown': '无法确定'}
MATCH = {'unique': '可唯一确认', 'ambiguous': '有歧义', 'none': '无合理匹配', 'insufficient': '证据不足'}
COVERAGE = {'represented': '已表示', 'missed': '漏检', 'misassociated': '关联错误',
            'no_rgb': '缺RGB', 'unknown': '无法确定'}
CONTINUITY = {'supported': '有在线保持依据', 'unsupported': '无可靠保持依据', 'unknown': '无法确定'}


def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(dumps(value).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def perception_continuity(row):
    """Map association changes do not change the continuous local pose lifecycle."""
    if row.get('health') != 'valid' or row.get('pose_health') != 'valid' or row.get('world_from_lidar') is None:
        return None
    binding = row.get('localization_binding')
    local = {key: binding.get(key) for key in ('session_id', 'generation', 'frame_id', 'source')} if binding else None
    return digest([row.get(key) for key in ('sequence_id', 'clock_domain', 'session_id', 'generation',
                                          'frame_id', 'detection_frame_id', 'pose_source')] + [local])


def ids(values, allowed, name):
    require(isinstance(values, list) and all(isinstance(v, str) for v in values), name + '必须是身份列表')
    require(len(values) == len(set(values)), name + '含重复身份')
    require(set(values) <= set(allowed), name + '引用了本观察中不存在的身份')


def t1_empty(binding):
    return dict(status='draft', groups=[], reference_groups=[], singles=[], unknown=[], note='', binding=binding)


def t2_empty(binding):
    return dict(status='draft', visibility='unknown', true_key=None, plausible=[], match='insufficient',
                coverage='unknown', continuity='unknown', flags=[], note='', binding=binding)


def validate_t1(label, keys, reference_keys=None):
    require(label.get('status') in STATES, '未知样本状态')
    require(isinstance(label.get('groups'), list), '缺少群组列表')
    used = []
    for group in label['groups']:
        members = group.get('members')
        ids(members, keys, '群组成员')
        require(len(members) >= 2, '一个群组至少两人')
        require(group.get('relation') in RELATIONS, '当前任务不支持此活动')
        roles = group.get('roles', [])
        ids(roles, members, '拍照角色')
        require(not roles or (group['relation'] == 'p' and len(roles) == 2), '拍照角色须为摄影者、主要被摄者')
        if group['relation'] == 'o' and label['status'] == 'complete':
            require(bool(group.get('activity', '').strip()), '其他共同活动需要简短说明')
        used.extend(members)
    for name in ('singles', 'unknown'):
        ids(label.get(name), keys, name)
        used.extend(label[name])
    require(len(used) == len(set(used)), '人物不能重复出现在群组、独立人物或归属未知中')
    covered = set()
    for group in label.get('reference_groups', []):
        members = group.get('members', [])
        ids(members, reference_keys if reference_keys is not None else members, '不可表达群组成员')
        require(len(members) >= 2 and group.get('relation') in RELATIONS, '不可表达群组格式错误')
        require(not set(members) <= set(keys), '有效PID群组应放入正式群组')
        covered.update(set(members) & set(keys))
    if label['status'] == 'complete':
        require(set(used) | covered == set(keys), '仍有未分类人物；隐藏人物也需要明确处理')


def validate_t2(label, keys, all_keys, target, person_for_key):
    require(label.get('status') in STATES, '未知样本状态')
    for field, choices in (('visibility', VISIBILITY), ('match', MATCH), ('coverage', COVERAGE),
                           ('continuity', CONTINUITY)):
        require(label.get(field) in choices, '未知T2字段：' + field)
    ids(label.get('plausible'), keys, '合理候选')
    true_key = label.get('true_key')
    require(true_key is None or true_key in all_keys, '真实对应不在当前观察')
    if label['status'] != 'complete':
        return
    if target.get('absent'):
        require(true_key is None, '无目标实例不能绑定场内真实目标')
    elif true_key is not None:
        known = person_for_key(true_key)
        require(known is not None and known == target.get('person_id'), '请先核对真实对应的人物身份关联')
    n = len(label['plausible'])
    require(label['match'] != 'unique' or n == 1, '唯一确认需要恰好一个合理候选')
    require(label['match'] != 'ambiguous' or n >= 2, '歧义需要至少两个合理候选')
    require(label['match'] != 'none' or n == 0, '无合理匹配时候选须为空')
    if label['coverage'] == 'represented' and label['visibility'] in ('clear', 'partial'):
        require(true_key in keys, '已表示的可见真实目标必须具有有效PID')
    if target.get('absent'):
        require(label['visibility'] in ('outside', 'unknown'), '无目标实例不能标为可见／遮挡')


def person_at(event, key, stamp_ns):
    result = {link['person_id'] for link in event.get('links', [])
              if link['key'] == key and link['start_ns'] <= stamp_ns <= link['end_ns']}
    require(len(result) <= 1, '同一轨迹时段存在冲突人物关联')
    return next(iter(result), None)


def validate_links(event):
    people = event.get('persons', {})
    require(isinstance(people, dict), '人物表错误')
    links = event.get('links', [])
    for i, link in enumerate(links):
        require(link.get('person_id') in people and isinstance(link.get('key'), str), '人物关联引用错误')
        require(type(link.get('start_ns')) is int and type(link.get('end_ns')) is int
                and link['start_ns'] <= link['end_ns'], '人物关联时间范围错误')
        for other in links[:i]:
            require(not (link['key'] == other['key'] and link['person_id'] != other['person_id']
                         and max(link['start_ns'], other['start_ns']) <= min(link['end_ns'], other['end_ns'])),
                    '同一轨迹的重叠时间段不能属于不同人物；请先拆分区间')
    for name in ('start', 'end'):
        require(isinstance(event.get(name), (int, float)) and math.isfinite(event[name]), '事件时间错误')
    require(0 <= event['start'] <= event['end'], '事件范围错误')


def rel_response(label, mapping):
    validate_t1(label, mapping)
    groups = []
    for group in label['groups']:
        roles = group.get('roles', [])
        members = [mapping[k] for k in group['members']]
        ordered = [mapping[k] for k in roles]
        members = ordered + sorted(set(members) - set(ordered)) if ordered else sorted(members)
        groups.append(dict(m=members, r=group['relation'], k=int(bool(roles))))
    return dict(g=sorted(groups, key=lambda g: min(g['m'])), u=sorted(mapping[k] for k in label['unknown']))
