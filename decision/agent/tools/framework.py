"""Tool descriptions, response envelope and per-tool argument validation."""
import jsonschema


def parameters(properties=None, required=()):
    return dict(type='object', properties=properties or {}, required=list(required), additionalProperties=False)


TEXT = dict(type='string', minLength=1, maxLength=2048)
TOOL_SPECS = {
    'lookup_places': ('查找登记地点；name为空返回目录。', parameters({'name': dict(type='string')}, ['name'])),
    'observe_scene': ('读取当前RGB和同源场景，不移动或推进录包。', parameters()),
    'navigate': ('内部检查几何并异步导航；accepted不是到达。', parameters({'place_id': TEXT}, ['place_id'])),
    'get_navigation_state': ('查询当前调用和已确认到达记录。', parameters()),
    'cancel_navigation': ('撤销当前调用；等待对应取消或停稳反馈。', parameters()),
    'wait_for_event': ('等待navigation、observation或user_continue，不反复请求模型。',
                       parameters({'event': dict(type='string', enum=['navigation', 'observation', 'user_continue'])}, ['event'])),
    'ask_user': ('向用户澄清，保留原指令；活动导航应先取消。', parameters({'question': TEXT}, ['question'])),
    'finish_task': ('记录任务结果；成功须有本次任务的实际执行证明，影子不能成功。',
                    parameters({'status': dict(type='string', enum=['succeeded', 'failed']), 'evidence': TEXT}, ['status', 'evidence'])),
}


def action_schema():
    # A small provider-independent envelope; tool-specific argument requirements
    # are additionally enforced locally. No response or arguments are repaired.
    properties = {}
    for _, schema in TOOL_SPECS.values():
        properties.update(schema['properties'])
    return parameters({'tool': dict(type='string', enum=list(TOOL_SPECS)),
                       'arguments': parameters(properties)}, ['tool', 'arguments'])


def validate_call(value):
    jsonschema.validate(value, action_schema())
    jsonschema.validate(value['arguments'], TOOL_SPECS[value['tool']][1])
    return value
