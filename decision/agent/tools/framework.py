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
    'kind': dict(type='string',enum=['point','region','reference','route','sequence','skill'],description='point为明确点，region为必须进入的区域，reference为自主选位姿的参考，route为自主选择途经点的多点任务，sequence为用户要求按顺序到达的标注点或区域；skill用于跟随、观察、就座或乘梯。'),
    'target_id': dict(type='string',description='正式place_id或region_id，不是P/R显示编号；用户给出匿名坐标或参考无需标注时填空字符串。'),
    'description': dict(TEXT,description='对应原指令的这一子任务，不增加用户未要求的条件。'),
    'stage_type': dict(type='string',enum=['navigation','follow','observation','seating','elevator'],
        description='任务阶段所需技能；省略为navigation。observation只在当前位置停稳后观察，不能移动；用户要求先到某处时必须登记导航目标。follow启动跟随；seating目前只记录要求。'),
    'position': dict(XY,description='仅匿名点需要用户明确给出的地图[x,y]米；命名点可省略，区域不得填写。'),
    'floor_id': dict(TEXT,description='跨楼层人工测试时该阶段所在楼层；只使用当前任务提供的楼层身份。'),
    'to_floor': dict(TEXT,description='仅乘梯阶段填写目标楼层身份；由用户的跨楼层任务确定。'),
    'required_yaw': dict(type='number',description='仅用户硬性指定的地图yaw弧度；自主选择的朝向不填这里。'),
    'facing': dict(type='string',description='仅用户要求面向时填写一个已标注点或区域的正式ID字符串，不是ID数组；依据位置、通道和观察自主判断朝向，不强制指向几何中心。硬性视觉条件使用semantic_conditions，尽力满足的视觉偏好使用visual_preferences。'),
    'position_rule': dict(type='string',enum=['region_centroid'],description='仅区域目标使用；用户要求区域正中时使用面积质心，普通进入区域不填写。'),
    'max_distance_m': dict(type='number',exclusiveMinimum=0,maximum=50,description='仅用户明确指定参考对象的距离上限时填写，单位为米。'),
    'checkpoints': dict(type='array',items=TEXT,minItems=2,maxItems=16,description='仅sequence使用；按用户要求的顺序填写正式点或区域ID，不含Agent自主添加的途经点。'),
    'semantic_conditions': dict(type='array',items=TEXT,maxItems=8,
        description='仅用户明确要求必须满足、需到达后用RGB确认的条件；“尽量”要求只放visual_preferences，不要同时登记为硬条件。'),
    'visual_preferences': dict(type='array',items=TEXT,maxItems=8,
        description='用户提出的尽量满足的视觉偏好；不要把同一要求重复放入semantic_conditions，也不要推断用户要求固定面向或距离。'),
    'robot_inside_region_id': dict(TEXT,description='仅用户明确要求完成时轮椅位于某标注区域内时填写该区域的正式 ID。'),
}, ['goal_id', 'kind', 'target_id', 'description'])
TOOL_SPECS = {
    'pause_map_localization': ('在人工驾驶乘梯阶段关闭来源楼层观察定位并启动独立传感器观察，不要求停稳或来源地图关联有效。',
                               parameters({'goal_id':dict(TEXT,description='当前乘梯子任务的 goal_id。')},['goal_id'])),
    'restart_map_localization': ('出梯后留在目标楼层等候区。比较本轮局部雷达图和目标地图裁剪，选择车体初值。配准未成功时可重新选点。',
        parameters({'pose':dict(POSE,description='目标地图中的车体中心 [x米,y米,yaw弧度]'),
                    'evidence':dict(TEXT,description='本次扫描与目标地图中的对应结构')},['pose','evidence'])),
    'record_elevator_observation': ('结合任务、当前RGB、雷达及可选历史观察，判断是否完成乘梯；提供标准起点时判断是否已到标准起点并调整至参考朝向。不能确认时继续观察。判断不是目标地图定位结果。',
        parameters({'judgment':dict(type='string',enum=['completed','not_completed','unknown'],description='已完成、尚未完成或证据不足；若有标准起点，完成表示已到该点并调整至参考朝向。'),
                    'evidence':dict(type='string',minLength=1,maxLength=512,description='简述画面中直接可见的依据。'),
                    'retain_observation':dict(type='boolean',description='当前观察是否值得保留供本次乘梯后续判断参考；乘梯结束后自动清除。')},
                   ['judgment','evidence','retain_observation'])),
    'record_task': ('登记用户提出的全部有序任务要求；只在 requirements_registered=false 时调用。先到某处并尽量观察，登记导航目标及视觉偏好，不要只登记无法移动的 observation 阶段。目标身份不清楚可先查询。登记不会执行导航。',
                    parameters({'goals': dict(type='array',items=GOAL,minItems=1,maxItems=16,
                        description='按原指令顺序放入 1 至 16 个要求对象。')}, ['goals'])),
    'lookup_places': ('检索正式标注点和区域，返回身份与几何；目标名称不清楚时使用，不产生新对象。跨楼层任务可指定楼层。',
                      parameters({'name':dict(type='string',description='名称、别名或正式身份；空字符串返回全部登记对象。'),
                                  'floor_id':dict(TEXT,description='可选楼层身份；省略表示当前楼层。')}, ['name'])),
    'query_map': ('查询静态先验地图的裁剪，可用bounds指定范围；不叠加当前局部占据或人物，不能据此判断当前环境空闲。结果和下一轮图片同源，不移动轮椅或生成候选。',
                  parameters({'bounds': dict(type='array',items=dict(type='number'),minItems=4,maxItems=4,
                      description='当前地图米制[xmin,ymin,xmax,ymax]，xmin<xmax且ymin<ymax；省略时取自车附近。裁剪远处不代表那里有当前测量。')})),
    'check_candidate': ('检查自主选择的候选base_link位姿：静态地图和可覆盖候选的同源局部占据中的完整车体轮廓与安全余量，以及指定标注点相对候选相机的近似水平视野。结果不证明动态障碍已排除，不检查路线、遮挡或实际可见性，不提交导航。',
                        parameters({'pose':dict(POSE,description='候选地图位姿[x,y,yaw]，米和弧度，参考base_link。'),
                                    'view_id':dict(TEXT,description='原样填写本轮当前空间图提供的短 view_id。'),
                                    'point_ids':dict(type='array',items=TEXT,maxItems=16,description='需要检查视野的已标注点ID；区域和未知目标不能当作点。')}, ['pose','view_id'])),
    'observe_scene': ('当前RGB缺失或需要显式重新读取时使用，返回available或不可用原因；不移动轮椅、不主动推进录包，等新帧用wait_for_event。', parameters()),
    'select_follow_target': ('从本轮带编号的当前 RGB 人物候选中确认跟随对象，保存该人物的视觉参考。此操作不启动跟随或运动；观察不足时重新观察。',
                 parameters({'goal_id':dict(TEXT,description='当前跟随阶段的 goal_id。'),
                             'observation_id':dict(TEXT,description='本轮当前 RGB 的 observation_id。'),
                             'candidate_id':dict(type='integer',minimum=0,description='RGB 标注中的人物候选编号。'),
                             'target_description':dict(TEXT,description='与用户要求一致的目标外观描述。')},
                            ['goal_id','observation_id','candidate_id','target_description'])),
    'start_local_task': ('启动当前跟随或桌边停靠。跟随必须先调用 select_follow_target，再传其返回的 reference_id；系统使用已确认的描述，跟随时不需要重填描述。T2 running 后再通知人物行走。桌边停靠需要描述和本轮图像中的桌沿区域。',
        parameters({'goal_id':dict(TEXT,description='当前阶段的goal_id。'),
                    'task_type':dict(type='string',enum=['follow','seating'],description='必须与登记阶段的stage_type一致。'),
                    'target_description':dict(TEXT,description='桌边停靠时填写用户要求；跟随时无需填写，系统使用已确认参考中的人物描述。'),
                    'reference_id':dict(TEXT,description='跟随时填写 Agent 已确认的目标参考 reference_id；桌边停靠不使用。'),
                    'table_roi':dict(type='array',minItems=4,maxItems=4,items=dict(type='number',minimum=0,maximum=1),
                        description='仅 seating 必填，当前 RGB 中目标桌面近侧边缘的归一化 [左,上,右,下]。')},
                   ['goal_id','task_type'])),
    'finish_local_task': ('本轮当前RGB支持用户指定的局部任务结束条件时请求结束并等待停稳。follow 不以目标丢失或保持距离视为完成；seating 尚不可执行。',
        parameters({'goal_id':dict(TEXT,description='当前局部任务阶段的goal_id。'),
                    'observation_id':dict(TEXT,description='跟随结束时必须提供本轮实际读取并支持判断的当前RGB观察。'),
                    'evidence':dict(TEXT,description='跟随结束时必须提供当前图像如何支持用户指定结束条件的依据。')},
                   ['goal_id'])),
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
                             'view_id':dict(TEXT,description='原样填写本轮visual_inputs.maps/current_spatial_view给出的短view_id，不是reference_map_id或图片编号。'),
                             'orientation_source': dict(type='string',enum=['user','semantic','autonomous'],
                                description='只从selected_goal.allowed_orientation_sources选择；user须有已登记required_yaw，否则用semantic或autonomous，不把资产旧yaw当用户要求。')},
                            ['goal_id', 'view_id', 'orientation_source'])),
    'navigate_route': ('route 和 sequence 专用。提交有车体余量的有序 XY；绕区域时结合原任务选择少量位置，用户要求返回时才将返回位置作为硬条件。增加点时用 waypoint_reason 说明具体要求或地图依据。点间实际路径由导航器规划。若某点被拒，查看 diagnostics 的点序号及占据原因。仅仿真和人工驾驶支持。',
                       parameters({'goal_id':dict(TEXT,description='当前next_goal_id。'),
                                   'view_id':dict(TEXT,description='当前地图view_id。'),
                                   'orientation_source':dict(type='string',enum=['user','semantic','autonomous'],description='终点朝向来源。'),
                                   'waypoints':dict(type='array',minItems=2,maxItems=32,items=XY,description='原生 JSON 二维数字数组，单位为米，按实际经过顺序排列。需要返回时明确将首点列为末点；中间点只要求经过。'),
                                   'final_yaw':dict(type='number',description='终点base_link地图yaw弧度；用户没有朝向或视觉要求时可省略。'),
                                   'waypoint_reason':dict(type='string',minLength=1,maxLength=512,description='超过三个自主途经点时，说明用户明确要求、具体地图限制或实际规划反馈；描绘圆周和点间距离本身不是依据。该说明不是可行性证明。')},
                                  ['goal_id','view_id','orientation_source','waypoints'])),
    'check_goal': ('结合当前任务、三图和监控事件判断当前子任务是否完成；提交你的结论和依据。导航或跟随阶段仍须有对应完成和停稳反馈；独立observation阶段须停稳并读取该阶段开始后的RGB。硬性视觉条件必须用当前RGB判断。证据不足选择unknown；completed才可推进。',
                   parameters({'goal_id':dict(TEXT,description='当前未完成子任务的goal_id。'),
                               'review_result':dict(type='string',enum=['satisfied','not_satisfied','unknown'],description='你对整个当前子任务是否完成的判断；证据不足选unknown。'),
                               'review_evidence':dict(TEXT,description='结合实际观察、位置、路线和用户要求说明判断依据，不把导航接收当成到达。'),
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
    'get_navigation_state': ('当前摘要不够或收到导航、局部任务事件后读取调用状态和最近完成记录，不刷新源测量时间；等执行结果优先wait_for_event。', parameters()),
    'set_navigation_monitor': ('为当前子任务选择复核触发方式；导航状态事件默认触发，不控制运动。周期、进入区域、无进展、距离、转角、持续停稳和超时可组合；再次调用替换配置。',
        parameters({'goal_id':dict(TEXT,description='当前next_goal_id。'),
                    'focus':dict(type='string',minLength=1,maxLength=512,description='结合原任务简述需要关注的语义条件或进展。'),
                    'interval_s':dict(type='number',minimum=5,maximum=300,description='可选语义复核间隔，秒。'),
                    'region_ids':dict(type='array',items=TEXT,maxItems=8,description='可选的正式region_id；有效定位进入时产生事件。'),
                    'continuous_region_id':dict(TEXT,description='轮椅在该正式区域内时持续复核；进入即复核，离开或定位不可用时停止。'),
                    'review_mode':dict(type='string',enum=['after_result'],description='与 continuous_region_id 一起填写 after_result；每次判断结束并收到新观察后再复核。'),
                    'progress_interval_s':dict(type='number',minimum=10,maximum=300,description='可选的无测量进展复核间隔，秒；事件本身不表示故障。'),
                    'distance_m':dict(type='number',minimum=.5,maximum=50,description='可选累计行驶距离触发阈值，米。'),
                    'rotation_rad':dict(type='number',minimum=.2,maximum=6.3,description='可选累计转角触发阈值，弧度。'),
                    'stopped_s':dict(type='number',minimum=.5,maximum=300,description='可选连续定位估计停稳后触发，秒。'),
                    'timeout_s':dict(type='number',minimum=10,maximum=3600,description='可选活动阶段超时，一次性触发，秒；不自动判失败。')},
                   ['goal_id','focus'])),
    'record_monitor_review': ('记录条件式持续复核中尚未满足或证据不足的判断及其当前观察依据；记录后等待新观察继续复核，不代表任务完成。满足时调用 complete_subtask。',
        parameters({'goal_id':dict(TEXT,description='正在复核的当前子任务 ID。'),
                    'result':dict(type='string',enum=['not_satisfied','unknown'],description='本轮条件尚未满足，或当前观察不足以判断。'),
                    'evidence':dict(type='string',minLength=1,maxLength=512,description='简述本轮观察及未满足或无法判断的原因。')},
                   ['goal_id','result','evidence'])),
    'cancel_navigation': ('撤销活动导航或局部任务并等待对应取消或停稳反馈；取消请求本身不证明已经停稳。', parameters()),
    'wait_for_event': ('挂起模型决策，等待明确事件；不把等待当任务完成，也不反复查询同一状态。',
                       parameters({'event': dict(type='string',enum=['navigation','observation','monitor','user_continue'],
                           description='navigation等活动调用反馈；observation等新的原始观察；monitor等已配置的任务监控触发；user_continue仅用于已确认停稳、且任务明确要求等用户继续。')}, ['event'])),
    'ask_user': ('目标身份、必要条件或证据无法自行澄清时询问；保留原任务，收到回答再继续。有活动运动目标须先取消；已确认停稳的实机保持状态可以直接询问。',
                 parameters({'question':dict(TEXT,description='只问当前缺少的具体信息，不要求用户反复确认已知条件。')}, ['question'])),
    'complete_subtask': ('查看 completion_evidence 和本轮观察。你判断当前子任务已满足时，一次提交完成决定和依据。跟随会先请求停车，末段停稳后自动结束整项任务。',
        parameters({'goal_id':dict(TEXT,description='当前 next_goal_id。'),
                    'evidence':dict(TEXT,description='用本轮测量和观察说明已满足的要求，包括视觉偏好实际满足的程度。')},
                   ['goal_id','evidence'])),
    'fail_task': ('任务无法继续时说明原因，并处理当前活动目标的停止。',parameters({'reason':dict(TEXT,description='无法继续执行的具体原因。')},['reason'])),
    'finish_task': ('全部登记要求已检查完成且停稳才能succeeded；shadow不能报告执行成功。failed记录不能继续的原因并处理活动调用停止。',
                    parameters({'status':dict(type='string',enum=['succeeded','failed'],description='本次任务结果，不能把接收、路径可行或图像接近目标当成功。'),
                                'evidence':dict(TEXT,description='实际完成依据或具体失败原因，不虚构到达。')}, ['status','evidence'])),
}


monitor_fields = TOOL_SPECS['set_navigation_monitor'][1]['properties']
GOAL['properties'].update(
    monitor=parameters({key:value for key,value in monitor_fields.items() if key != 'goal_id'}),
    execution_notes=dict(TEXT,description='本阶段的工具使用注意事项、完成依据或异常处理。'),
    circuit=parameters({'turns':dict(type='integer',minimum=1),
                        'direction':dict(type='string',enum=['either','clockwise','counterclockwise'])},['turns']),
)
GOAL['properties']['circuit']['description']='明确绕标注区域时填写圈数，仅用户指定方向时填写 direction。'
GOAL['properties']['monitor']['description'] = '阶段进入时生效。可组合周期、区域和进展等触发条件；空对象表示只订阅状态事件。'
GOAL['allOf'] = [
    {'if':{'required':['circuit']},'then':{'properties':{'kind':{'const':'route'}}}},
    {'if':{'properties':{'kind':{'const':'sequence'}}},
     'then':{'properties':{'target_id':{'const':''}},'required':['checkpoints']}},
    {'if':{'properties':{'kind':{'const':'skill'}}},
     'then':{'required':['stage_type'],'properties':{'target_id':{'const':''},
             'stage_type':{'enum':['follow','seating','observation','elevator']}}}},
    {'if':{'required':['stage_type'],'properties':{'stage_type':{'const':'elevator'}}},
     'then':{'required':['floor_id','to_floor']}},
]

# The model-facing registration contract groups the target and constraints.
# Legacy recorded actions remain valid for replay, but are not advertised to
# the model. Conversion to the existing internal requirement is explicit.
REGISTRATION_TARGET = parameters({
    'id': dict(TEXT, description='正式标注点或区域 ID；自主选点可省略。'),
    'position': dict(XY, description='用户明确指定的匿名地图坐标 [x,y]，单位米。'),
    'checkpoints': dict(type='array', items=TEXT, minItems=2, maxItems=16,
                        description='sequence 的用户指定地点 ID，按要求顺序填写。'),
})
VISUAL_CONDITION = parameters({
    'description': dict(TEXT, description='用户提出的视觉要求。'),
    'priority': dict(type='string', enum=['required','preferred'],
                     description='required 为必须满足；preferred 为尽量满足。'),
}, ['description','priority'])
REGISTRATION_CONDITIONS = parameters({
    'required_yaw': GOAL['properties']['required_yaw'],
    'facing': dict(type='string',description='用户要求面向的正式点或区域 ID；结合连通关系判断朝向，不强制指向几何中心。'),
    'position_rule': GOAL['properties']['position_rule'],
    'max_distance_m': GOAL['properties']['max_distance_m'],
    'robot_inside_region_id': GOAL['properties']['robot_inside_region_id'],
    'visual': dict(type='array', items=VISUAL_CONDITION, maxItems=16,
                   description='必须满足和尽量满足的视觉要求；按用户原意区分。'),
})
REGISTRATION_GOAL = parameters({
    'goal_id': GOAL['properties']['goal_id'],
    'stage': dict(type='string', enum=['point','region','reference','route','sequence',
                                      'follow','observation','seating','elevator'],
                  description='导航目标类型或独立技能类型；无需额外填写 kind 或 stage_type。'),
    'description': GOAL['properties']['description'],
    'target': dict(REGISTRATION_TARGET, description='标注对象、用户指定坐标或有序地点；无目标时省略。'),
    'conditions': dict(REGISTRATION_CONDITIONS, description='用户明确提出的约束；无约束时省略。'),
    'floor_id': GOAL['properties']['floor_id'],
    'to_floor': GOAL['properties']['to_floor'],
    'monitor': GOAL['properties']['monitor'],
    'execution_notes': GOAL['properties']['execution_notes'],
}, ['goal_id','stage','description'])
REGISTRATION_ARGS = parameters({'goals': dict(type='array', items=REGISTRATION_GOAL,
    minItems=1,maxItems=16,description='按用户顺序登记的阶段数组；先登记要求，再在执行时选点。')}, ['goals'])
TOOL_SPECS['record_task'] = (TOOL_SPECS['record_task'][0], REGISTRATION_ARGS)

ACTION = parameters({'tool': dict(type='string', enum=list(TOOL_SPECS)),
                     'arguments': dict(type='object')}, ['tool', 'arguments'])
ACTION_VALIDATOR=jsonschema.Draft202012Validator(ACTION)
GOAL_VALIDATOR=jsonschema.Draft202012Validator(GOAL)
TOOL_VALIDATORS={name:jsonschema.Draft202012Validator(schema) for name,(_,schema) in TOOL_SPECS.items()}
# Direct programmatic callers and saved fixtures may still use the legacy
# schema; the advertised native tool above only exposes the new contract.
TOOL_VALIDATORS['record_task'] = jsonschema.Draft202012Validator(parameters({
    'goals': dict(type='array', items={}, minItems=1, maxItems=16)
}, ['goals']))
REGISTRATION_GOAL_VALIDATOR = jsonschema.Draft202012Validator(REGISTRATION_GOAL)
SCRIPTED_TOOLS={'check_goal','finish_local_task','finish_task'}
EXECUTION_TOOLS=tuple(name for name in TOOL_SPECS if name!='record_task' and name not in SCRIPTED_TOOLS)


def unique_fields(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError('Duplicate JSON field: '+name)
        result[name] = value
    return result


def validate_call(value, *, model_schema=False):
    ACTION_VALIDATOR.validate(value)
    # Includes overflowed exponents in native replies and non-finite fixture inputs.
    json.dumps(value['arguments'], allow_nan=False)
    if value['tool']=='record_task' and model_schema:
        jsonschema.Draft202012Validator(REGISTRATION_ARGS).validate(value['arguments'])
    else:
        TOOL_VALIDATORS[value['tool']].validate(value['arguments'])
    if value['tool']=='record_task' and not model_schema:
        for index,goal in enumerate(value['arguments']['goals']):
            validator = GOAL_VALIDATOR if isinstance(goal,dict) and 'kind' in goal and 'stage' not in goal else REGISTRATION_GOAL_VALIDATOR
            try:
                validator.validate(goal)
            except jsonschema.ValidationError as error:
                error.path.appendleft(index)
                error.path.appendleft('goals')
                raise
    if value['tool']=='navigate':
        args=value['arguments']
        if not (('pose' in args and 'point_id' not in args and 'yaw' not in args) or
                ('pose' not in args and 'point_id' in args and 'yaw' in args)):
            raise jsonschema.ValidationError('navigate requires pose or point_id and yaw, exclusively')
    return value


def native_tools(names=None):
    names=('record_task',*EXECUTION_TOOLS) if names is None else tuple(names)
    if len(set(names))!=len(names) or any(name not in TOOL_SPECS or name in SCRIPTED_TOOLS for name in names):
        raise ValueError('Unknown, replay-only or duplicate native tool')
    return [dict(type='function', function=dict(name=name, description=description, parameters=schema))
            for name, (description, schema) in TOOL_SPECS.items() if name in names]


@dataclass(frozen=True)
class Reply:
    message: dict
    action: dict | None
    argument_error: dict | None = None

    @classmethod
    def parse(cls, message, *, report_argument_error=False, model_schema=False):
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
            try:
                arguments = json.loads(function['arguments'], object_pairs_hook=unique_fields)
                action = validate_call(dict(tool=function.get('name'), arguments=arguments),model_schema=model_schema)
            except (ValueError,jsonschema.ValidationError) as error:
                if not report_argument_error or function.get('name') not in TOOL_SPECS:
                    raise
                detail = dict(status='refused', reason='invalid_tool_arguments', tool=function['name'],
                              path=list(getattr(error,'absolute_path',())), detail=str(error).splitlines()[0][:500])
                return cls(copy.deepcopy(message),None,detail)
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
