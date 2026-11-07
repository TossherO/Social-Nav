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
    'goal_id': TEXT, 'kind': dict(type='string', enum=['point', 'region']), 'target_id': dict(type='string'),
    'description': TEXT, 'position': XY, 'required_yaw': dict(type='number'),
    'facing': dict(type='string'), 'semantic_conditions': dict(type='array', items=TEXT, maxItems=8),
}, ['goal_id', 'kind', 'target_id', 'description'])
TOOL_SPECS = {
    'record_task': ('登记原指令的有序点／区域要求，不填写自主选出的执行位姿，不自动执行。',
                    parameters({'goals': dict(type='array', items=GOAL, minItems=1, maxItems=16)}, ['goals'])),
    'lookup_places': ('查找登记地点；name为空返回目录。', parameters({'name': dict(type='string')}, ['name'])),
    'query_map': ('读取标注全局图、局部图及坐标；bounds为当前地图坐标[xmin,ymin,xmax,ymax]，不生成候选。',
                  parameters({'bounds': dict(type='array', items=dict(type='number'), minItems=4, maxItems=4)})),
    'observe_scene': ('读取当前RGB和同源场景，不移动或推进录包。', parameters()),
    'navigate': ('提交VLM直接选择的连续[x,y,yaw]；检查要求与几何，不修正位姿；accepted不是到达。',
                 parameters({'goal_id': TEXT, 'pose': POSE, 'view_id': TEXT,
                             'orientation_source': dict(type='string', enum=['user', 'semantic', 'autonomous'])},
                            ['goal_id', 'pose', 'view_id', 'orientation_source'])),
    'check_goal': ('用对应执行反馈检查当前子任务；有视觉条件时仅接受已读当前图像的语义判断。',
                   parameters({'goal_id': TEXT, 'observation_id': TEXT,
                               'semantic_result': dict(type='string', enum=['satisfied', 'not_satisfied', 'unknown']),
                               'evidence': TEXT}, ['goal_id'])),
    'get_navigation_state': ('查询当前调用和已确认到达记录。', parameters()),
    'cancel_navigation': ('撤销当前调用；等待对应取消或停稳反馈。', parameters()),
    'wait_for_event': ('等待navigation、observation或user_continue，不反复请求模型。',
                       parameters({'event': dict(type='string', enum=['navigation', 'observation', 'user_continue'])}, ['event'])),
    'ask_user': ('向用户澄清，保留原指令；活动导航应先取消。', parameters({'question': TEXT}, ['question'])),
    'finish_task': ('记录任务结果；成功须有本次任务的实际执行证明，影子不能成功。',
                    parameters({'status': dict(type='string', enum=['succeeded', 'failed']), 'evidence': TEXT}, ['status', 'evidence'])),
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
