"""Native function tools, checked model replies and explicit offline fixtures."""
import copy
from dataclasses import dataclass
import json
import uuid
import jsonschema


def parameters(properties=None, required=()):
    return dict(type='object', properties=properties or {}, required=list(required), additionalProperties=False)


TEXT = dict(type='string', minLength=1, maxLength=2048)
XY = dict(type='array', items=dict(type='number'), minItems=2, maxItems=2)
POSE = dict(type='array', items=dict(type='number'), minItems=3, maxItems=3)
GOAL = parameters({
    'goal_id': dict(TEXT,description='本任务内唯一的子任务身份；后续从当前要求中原样读取。'),
    'kind': dict(type='string',enum=['point','region'],description='point为点目标，region为人工标注区域。'),
    'target_id': dict(type='string',description='正式place_id或region_id，不是P/R显示编号；用户明确给出匿名点坐标时填空字符串。'),
    'description': dict(TEXT,description='对应原指令的这一子任务，不增加用户未要求的条件。'),
    'position': dict(XY,description='仅匿名点需要用户明确给出的地图[x,y]米；命名点可省略，区域不得填写。'),
    'required_yaw': dict(type='number',description='仅用户硬性指定的地图yaw弧度；自主选择的朝向不填这里。'),
    'facing': dict(type='string',description='仅用户要求面向且已标注的landmark_id；不填名称、P/R编号或自行创造的对象。'),
    'semantic_conditions': dict(type='array',items=TEXT,maxItems=8,
        description='仅原指令明确要求到达后用当前RGB确认的条件；普通到点停车不自动增加视觉条件。'),
}, ['goal_id', 'kind', 'target_id', 'description'])
TOOL_SPECS = {
    'record_task': ('只在requirements_registered=false时登记完整有序要求。目标身份未知可先查询；已登记后不重复调用，recorded不执行导航。',
                    parameters({'goals': dict(type='array',items=GOAL,minItems=1,maxItems=16,
                        description='按原指令顺序列出全部1至16项要求，不是工具调用计划。')}, ['goals'])),
    'lookup_places': ('检索正式标注点、区域和地标，返回身份与几何；目标名称不清楚时使用，不产生新对象。',
                      parameters({'name':dict(type='string',description='名称、别名或正式身份；空字符串返回全部登记对象。')}, ['name'])),
    'query_map': ('已有有效当前地图时无需重复查询；需要目标附近的更细局部范围时指定bounds。结果和下一轮图片同源，不移动轮椅或生成候选。',
                  parameters({'bounds': dict(type='array',items=dict(type='number'),minItems=4,maxItems=4,
                      description='当前地图米制[xmin,ymin,xmax,ymax]，xmin<xmax且ymin<ymax；省略时取自车附近。裁剪远处不代表那里有当前测量。')})),
    'observe_scene': ('当前RGB缺失或需要显式重新读取时使用，返回available或不可用原因；不移动轮椅、不主动推进录包，等新帧用wait_for_event。', parameters()),
    'save_keyframe': ('重要入口、岔路或任务线索值得后续比较时保存本轮已读RGB，返回历史图片引用。每修订最多24帧，同一观察不重复保存，不用于提前证明到达。',
                      parameters({'observation_id':dict(TEXT,description='当前visual_inputs.rgb所指的原观察身份，必须是本轮实际读过且仍有效的图像。'),
                                  'reason':dict(TEXT,description='简短说明要保留的可见线索及未来用途，不把过去图像写成当前执行证明。')}, ['observation_id','reason'])),
    'read_keyframe': ('根据当前keyframes索引重读过去RGB；最多两张历史图保留在上下文，裁剪不删除已保存记录。不会取得新观察或刷新源时间。',
                      parameters({'keyframe_id':dict(TEXT,description='本任务当前修订已经保存的keyframe_id，不能使用其他任务、其他修订或未来观察。')}, ['keyframe_id'])),
    'navigate': ('向导航提交下一未完成要求的连续位姿。使用当前地图view_id，不用静态图或历史图身份；有活动调用须先取消。拒绝时保留原参数，accepted仅为接收。',
                 parameters({'goal_id':dict(TEXT,description='当前next_goal_id，必须先登记要求。'),
                             'pose':dict(POSE,description='地图中的完整[x,y,yaw]，单位米、米、弧度，参考base_link；点遵守登记位置，区域内位置和非指定朝向由模型选择。'),
                             'view_id':dict(TEXT,description='本轮visual_inputs.maps/current_spatial_view给出的完整view_id，不是reference_map_id或图片编号。'),
                             'orientation_source': dict(type='string',enum=['user','semantic','autonomous'],
                                description='只从selected_goal.allowed_orientation_sources选择；user须有已登记required_yaw，否则用semantic或autonomous，不把资产旧yaw当用户要求。')},
                            ['goal_id', 'pose', 'view_id', 'orientation_source'])),
    'check_goal': ('对应导航已有执行完成且停稳反馈后检查当前要求。普通几何任务只需goal_id；视觉条件还需到达后当前RGB的判断。completed才可推进，unknown/not_met/unproven均不完成。',
                   parameters({'goal_id':dict(TEXT,description='当前未完成子任务的goal_id。'),
                               'observation_id':dict(TEXT,description='有视觉条件时填写本轮已读、到达后的当前RGB身份，不能填历史关键帧。'),
                               'semantic_result': dict(type='string',enum=['satisfied','not_satisfied','unknown'],
                                   description='当前RGB对全部已登记视觉条件的判断，证据不足填unknown，不代替几何与执行反馈。'),
                               'evidence':dict(TEXT,description='简短列出当前图像中实际可见的依据；无视觉条件时可省略。')}, ['goal_id'])),
    'get_navigation_state': ('当前摘要不够或收到导航事件后读取调用状态和最近完成记录，不刷新源测量时间；等执行结果优先wait_for_event。', parameters()),
    'cancel_navigation': ('撤销活动导航并等待对应取消或停稳反馈；取消请求本身不证明已经停稳。', parameters()),
    'wait_for_event': ('挂起模型决策，等待明确事件；不把等待当任务完成，也不反复查询同一状态。',
                       parameters({'event': dict(type='string',enum=['navigation','observation','user_continue'],
                           description='navigation等活动调用反馈；observation等新的原始观察；user_continue仅用于已确认停稳、且任务明确要求等用户继续。')}, ['event'])),
    'ask_user': ('目标身份、必要条件或证据无法自行澄清时询问；保留原任务，收到回答再继续。有活动导航必须先取消并确认。',
                 parameters({'question':dict(TEXT,description='只问当前缺少的具体信息，不要求用户反复确认已知条件。')}, ['question'])),
    'finish_task': ('全部登记要求已检查完成且停稳才能succeeded；shadow不能报告执行成功。failed记录不能继续的原因并处理活动调用停止。',
                    parameters({'status':dict(type='string',enum=['succeeded','failed'],description='本次任务结果，不能把接收、路径可行或图像接近目标当成功。'),
                                'evidence':dict(TEXT,description='实际完成依据或具体失败原因，不虚构到达。')}, ['status','evidence'])),
}


def action_schema():
    # Internal dispatch and explicit --actions fixtures, never the API response format.
    properties = {}
    for _, schema in TOOL_SPECS.values():
        properties.update(schema['properties'])
    return parameters({'tool': dict(type='string', enum=list(TOOL_SPECS)),
                       'arguments': parameters(properties)}, ['tool', 'arguments'])


def validate_call(value):
    jsonschema.validate(value, action_schema())
    jsonschema.validate(value['arguments'], TOOL_SPECS[value['tool']][1])
    return value


def native_tools():
    return [dict(type='function', function=dict(name=name, description=description, parameters=schema))
            for name, (description, schema) in TOOL_SPECS.items()]


@dataclass(frozen=True)
class Reply:
    message: dict
    action: dict | None

    @classmethod
    def parse(cls, message):
        """Validate the native reply without fixing content, arguments or identities."""
        if not isinstance(message,dict) or message.get('role') != 'assistant' or message.get('content') is not None and not isinstance(message['content'], str):
            raise ValueError('Expected an assistant message with optional text')
        calls = message.get('tool_calls') or []
        if not isinstance(calls, list) or len(calls) > 1:
            raise ValueError('Exactly one tool call per decision is supported')
        action = None
        if calls:
            call = calls[0]
            if (not isinstance(call,dict) or call.get('type') != 'function' or not isinstance(call.get('id'), str) or not call['id'] or
                    not isinstance(call.get('function'), dict)):
                raise ValueError('Native function call requires its original identity')
            function = call['function']
            if not isinstance(function.get('arguments'), str):
                raise ValueError('Function arguments must be a JSON string')
            action = validate_call(dict(tool=function.get('name'), arguments=json.loads(function['arguments'])))
        elif not (message.get('content') or '').strip():
            raise ValueError('Empty model reply')
        return cls(copy.deepcopy(message), action)

    @classmethod
    def fixture(cls, action):
        """Explicit synthetic action source; never adapts a production API reply."""
        validate_call(action)
        return cls.parse(dict(role='assistant', content=None, tool_calls=[dict(
            id='fixture_'+uuid.uuid4().hex, type='function', function=dict(name=action['tool'],
            arguments=json.dumps(action['arguments'], ensure_ascii=False, allow_nan=False)))]))
