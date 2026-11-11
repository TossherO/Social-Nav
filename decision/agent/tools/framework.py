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
    'kind': dict(type='string',enum=['point','region','reference','circuit','sequence'],description='point为明确点，region为必须进入的区域，reference为自主选位姿的参考，circuit为按方向绕区域，sequence为用户要求按顺序到达的标注点或区域。'),
    'target_id': dict(type='string',description='正式place_id或region_id，不是P/R显示编号；用户给出匿名坐标或参考无需标注时填空字符串。'),
    'description': dict(TEXT,description='对应原指令的这一子任务，不增加用户未要求的条件。'),
    'position': dict(XY,description='仅匿名点需要用户明确给出的地图[x,y]米；命名点可省略，区域不得填写。'),
    'required_yaw': dict(type='number',description='仅用户硬性指定的地图yaw弧度；自主选择的朝向不填这里。'),
    'facing': dict(type='string',description='仅用户要求面向时填写一个已标注点或区域的正式ID字符串，不是ID数组；依据位置、通道和观察自主判断朝向，不强制指向几何中心。硬性视觉条件使用semantic_conditions，尽力满足的视觉偏好使用visual_preferences。'),
    'position_rule': dict(type='string',enum=['region_centroid'],description='仅区域目标使用；用户要求区域正中时使用面积质心，普通进入区域不填写。'),
    'max_distance_m': dict(type='number',exclusiveMinimum=0,maximum=50,description='reference的几何距离上限（米）；用户未明确距离时这是Agent选择的近似阈值，不应表述为用户指定。'),
    'direction': dict(type='string',enum=['clockwise','counterclockwise'],description='仅circuit使用；仅当用户指定绕行方向时登记。'),
    'checkpoints': dict(type='array',items=TEXT,minItems=2,maxItems=16,description='仅sequence使用；按用户要求的顺序填写正式点或区域ID，不含Agent自主添加的途经点。'),
    'semantic_conditions': dict(type='array',items=TEXT,maxItems=8,
        description='用户要求必须满足、需要到达后用当前RGB确认的条件；纯几何的附近要求可使用max_distance_m。'),
    'visual_preferences': dict(type='array',items=TEXT,maxItems=8,
        description='用户明确提出的尽量满足的视觉偏好；选择位姿时努力考虑，但不把偏好改写为必须全部满足的条件。'),
}, ['goal_id', 'kind', 'target_id', 'description'])
TOOL_SPECS = {
    'record_task': ('登记用户提出的全部有序任务要求；只在 requirements_registered=false 时调用。目标身份不清楚可先查询。登记不会执行导航。',
                    parameters({'goals': dict(type='array',items=GOAL,minItems=1,maxItems=16,
                        description='原生 JSON 数组，按原指令顺序放入1至16个要求对象；例如 [{"goal_id":"g1","kind":"point","target_id":"p1","description":"到p1"}]。不要把数组写成字符串。')}, ['goals'])),
    'lookup_places': ('检索正式标注点和区域，返回身份与几何；目标名称不清楚时使用，不产生新对象。',
                      parameters({'name':dict(type='string',description='名称、别名或正式身份；空字符串返回全部登记对象。')}, ['name'])),
    'query_map': ('查询静态先验地图的裁剪，可用bounds指定范围；不叠加当前局部占据或人物，不能据此判断当前环境空闲。结果和下一轮图片同源，不移动轮椅或生成候选。',
                  parameters({'bounds': dict(type='array',items=dict(type='number'),minItems=4,maxItems=4,
                      description='当前地图米制[xmin,ymin,xmax,ymax]，xmin<xmax且ymin<ymax；省略时取自车附近。裁剪远处不代表那里有当前测量。')})),
    'check_candidate': ('检查自主选择的候选base_link位姿：静态地图和可覆盖候选的同源局部占据中的完整车体轮廓与安全余量，以及指定标注点相对候选相机的近似水平视野。结果不证明动态障碍已排除，不检查路线、遮挡或实际可见性，不提交导航。',
                        parameters({'pose':dict(POSE,description='候选地图位姿[x,y,yaw]，米和弧度，参考base_link。'),
                                    'view_id':dict(TEXT,description='本轮当前空间图的完整view_id。'),
                                    'point_ids':dict(type='array',items=TEXT,maxItems=16,description='需要检查视野的已标注点ID；区域和未知目标不能当作点。')}, ['pose','view_id'])),
    'observe_scene': ('当前RGB缺失或需要显式重新读取时使用，返回available或不可用原因；不移动轮椅、不主动推进录包，等新帧用wait_for_event。', parameters()),
    'save_keyframe': ('重要入口、岔路或任务线索值得后续比较时保存本轮已读RGB，返回历史图片引用。每修订最多24帧，同一观察不重复保存，不用于提前证明到达。',
                      parameters({'observation_id':dict(TEXT,description='当前visual_inputs.rgb所指的原观察身份，必须是本轮实际读过且仍有效的图像。'),
                                  'reason':dict(TEXT,description='简短说明要保留的可见线索及未来用途，不把过去图像写成当前执行证明。')}, ['observation_id','reason'])),
    'read_keyframe': ('根据当前keyframes索引重读过去RGB；最多两张历史图保留在上下文，裁剪不删除已保存记录。不会取得新观察或刷新源时间。',
                      parameters({'keyframe_id':dict(TEXT,description='本任务当前修订已经保存的keyframe_id，不能使用其他任务、其他修订或未来观察。')}, ['keyframe_id'])),
    'navigate': ('向导航提交下一未完成要求的位姿。用pose自由选坐标，或用point_id和yaw指定标注点；只能使用其中一种。使用当前地图view_id，有活动调用须先取消。提交前检查最终位姿的已知车体占据；未知及完整路径由导航继续检查。accepted仅为接收。',
                 parameters({'goal_id':dict(TEXT,description='当前next_goal_id，必须先登记要求。'),
                             'pose':dict(POSE,description='自主选择的地图[x,y,yaw]，参考base_link；须满足已登记的点、区域和朝向要求。'),
                             'point_id':dict(TEXT,description='明确选择的已标注点的正式place_id；与yaw一起使用，不得与pose同时填写。'),
                             'yaw':dict(type='number',description='使用point_id时模型明确选择的地图yaw弧度，不自动采用标注朝向。'),
                             'view_id':dict(TEXT,description='本轮visual_inputs.maps/current_spatial_view给出的完整view_id，不是reference_map_id或图片编号。'),
                             'orientation_source': dict(type='string',enum=['user','semantic','autonomous'],
                                description='只从selected_goal.allowed_orientation_sources选择；user须有已登记required_yaw，否则用semantic或autonomous，不把资产旧yaw当用户要求。')},
                            ['goal_id', 'view_id', 'orientation_source'])),
    'navigate_route': ('提交自主选择的有序地图途经点。绕区域需首尾相接、按要求的方向完整绕一圈，所有线段而非仅端点均不得穿过区域；拒绝反馈可列出相交线段与闭合距离。仿真中自动推进并在终点停车；仅预检首段。accepted不表示到达，影子与实车不支持执行。',
                       parameters({'goal_id':dict(TEXT,description='当前next_goal_id。'),
                                   'view_id':dict(TEXT,description='当前地图view_id。'),
                                   'orientation_source':dict(type='string',enum=['user','semantic','autonomous'],description='终点朝向来源。'),
                                   'waypoints':dict(type='array',minItems=2,maxItems=32,items=XY,description='原生 JSON 二维数字数组，例如 [[1.0,2.0],[3.0,4.0]]；单位为米，按经过顺序排列，不要写成字符串。绕区域时首尾为同一点，逐段绕开区域轮廓；只用足以表达路线的必要点。中间点只要求经过。'),
                                   'direction':dict(type='string',enum=['clockwise','counterclockwise'],description='仅circuit使用；未登记用户方向时由Agent选择，不得与用户要求矛盾。'),
                                   'final_yaw':dict(type='number',description='终点base_link地图yaw弧度。')},
                                  ['goal_id','view_id','orientation_source','waypoints','final_yaw'])),
    'check_goal': ('对应导航已有执行完成且停稳反馈后检查当前要求。硬性视觉条件必须用到达后的当前RGB判断。视觉偏好有当前RGB时可记录实际结果；没有依据则保持未核实，不声称偏好已满足。completed才可推进。',
                   parameters({'goal_id':dict(TEXT,description='当前未完成子任务的goal_id。'),
                               'observation_id':dict(TEXT,description='判断硬性视觉条件或视觉偏好时填写本轮已读、到达后的当前RGB身份，不能填历史关键帧。'),
                               'semantic_result': dict(type='string',enum=['satisfied','not_satisfied','unknown'],
                                   description='当前RGB对全部已登记视觉条件的判断，证据不足填unknown，不代替几何与执行反馈。'),
                               'evidence':dict(TEXT,description='简短列出当前图像中实际可见的依据；无视觉条件时可省略。'),
                               'preference_result':dict(type='string',enum=['satisfied','partly_satisfied','not_satisfied','unknown'],
                                   description='仅对登记的视觉偏好，根据本轮实际读取的到达后当前RGB判断，不把部分可见说成全部可见。'),
                               'preference_evidence':dict(TEXT,description='本轮当前RGB实际支持的偏好判断，例如哪些目标可见；不要猜测或声称全局最优。'),
                               'facing_result':dict(type='string',enum=['satisfied','not_satisfied','unknown'],
                                   description='仅有facing要求时使用；基于本轮当前地图、位姿及可用观察判断实际朝向是否符合用户意图，证据不足填unknown。'),
                               'facing_evidence':dict(TEXT,description='简述本轮判断朝向的依据，例如相连走廊、入口及当前实际朝向；不要把静态地图当作实时无遮挡证明。')}, ['goal_id'])),
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
    if value['tool']=='navigate':
        args=value['arguments']
        if not (('pose' in args and 'point_id' not in args and 'yaw' not in args) or
                ('pose' not in args and 'point_id' in args and 'yaw' in args)):
            raise jsonschema.ValidationError('navigate requires pose or point_id and yaw, exclusively')
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
