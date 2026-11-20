"""Read-only session projection for the local manual Agent window."""
import base64
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from difflib import SequenceMatcher


KINDS=('conversation_exchange','conversation_trimmed','tool_intent','model_action_applied',
       'api_request','api_response','api_failure','api_input_rejected','model_failed','model_protocol_rejected',
       'navigation_repeated_refusal',
       'manual_arrival_invalidated','manual_arrival_recovered',
       'elevator_monitor_started','elevator_model_failed','elevator_model_discarded',
       'floor_handoff_started','manual_floor_rebound','elevator_stage_completed',
       'model_result_discarded','manual_arrival_confirmation','manual_goal_evidence_wait_started',
       'manual_goal_evidence_wait_finished','manual_goal_evidence_refresh_failed','manual_goal_evidence_unproven','manual_route_progress',
       'manual_route_trace','manual_reference_status','manual_reference_region_conflict','operator_command','navigation_completed','navigation_ack','navigation_cancel_confirmed',
       'navigation_context_lost','task_connection_lost','user_event','subtask_skipped','registered_task_imported','task_created','assistant_message','model_action_missing','agent_run_finished',
       'route_repeated_refusal','reference_map_prepared','t2_replay','follow_target_confirmed','agent_bag_recording')

_LONG_HEX = re.compile(r'(?<![0-9a-fA-F])[0-9a-fA-F]{32,}(?![0-9a-fA-F])')


def goal_requirements(goal,names):
    """Describe registered requirements separately from the Agent's choice."""
    req=goal.get('requirement') or {};kind=req.get('kind')
    kinds={'point':'到指定点','region':'进入区域','reference':'靠近参考对象','circuit':'绕区域一圈','route':'多点路线',
           'sequence':'依次经过多个地点','skill':'技能任务'}
    directions={'clockwise':'顺时针','counterclockwise':'逆时针'}
    target=req.get('target_id');target_name=names.get(target,target) if target else None
    lines=['要求：'+str(req.get('description') or req.get('goal_id') or '未提供'),
           '类型：'+kinds.get(kind,str(kind or '未提供'))]
    if target_name:lines.append('目标：'+str(target_name))
    if kind=='circuit':
        direction=req.get('direction')
        lines.append('用户要求方向：'+directions.get(direction,str(direction)) if direction else '用户要求方向：未指定')
        selected=((goal.get('selected') or {}).get('route') or {})
        if selected.get('direction'):
            origin='用户指定' if selected.get('direction_source')=='user' else 'Agent 选择'
            lines.append('执行方案方向：'+directions.get(selected['direction'],selected['direction'])+'（'+origin+'）')
    fields={'position':'指定坐标','required_yaw':'要求朝向（弧度）','facing':'面向',
            'position_rule':'位置规则','max_distance_m':'距离上限（米）','checkpoints':'依次经过',
            'semantic_conditions':'必须满足','visual_preferences':'视觉偏好','stage_type':'任务阶段',
            'floor_id':'所在楼层','to_floor':'目标楼层','robot_inside_region_id':'完成时轮椅所在区域','wait_after':'完成后等待口令'}
    for key,label in fields.items():
        value=req.get(key)
        if value in (None,'',[]):continue
        if key=='robot_inside_region_id' and req.get('to_floor'):
            value=f"{req['to_floor']} · {value}"
        elif key in ('facing','robot_inside_region_id'):value=names.get(value,value)
        elif key=='stage_type' and value=='elevator':value='乘坐电梯'
        elif key=='checkpoints':value=[names.get(item,item) for item in value]
        elif key=='position_rule' and value=='region_centroid':value='区域正中'
        lines.append(label+'：'+('、'.join(map(str,value)) if isinstance(value,list) and key!='position' else str(value)))
    monitored=(req.get('monitor') or {}).get('continuous_region_id')
    if monitored:
        lines.append('持续复核区域：'+str(names.get(monitored,monitored)))
        if not req.get('robot_inside_region_id'):
            lines.append('完成时区域检查：未登记')
    if 'required_yaw' not in req and not req.get('facing'):lines.append('用户要求朝向：未指定')
    return '\n'.join(lines)


def display_value(value):
    """Hide internal identities in the GUI only; never modify the saved event."""
    if isinstance(value,dict):
        result={}
        for key,item in value.items():
            if (key in ('task_id','source_task_id','source_digest','call_id','tool_call_id','session_id','association_session') or
                    key.endswith(('_sha256','_fingerprint')) or
                    key.endswith('_binding') and key!='candidate_binding' and not isinstance(item,dict)):
                result[key]='[内部标识]'
            else:result[key]=display_value(item)
        return result
    if isinstance(value,list):return [display_value(item) for item in value]
    if isinstance(value,str):
        try:parsed=json.loads(value)
        except (ValueError,TypeError):parsed=None
        if isinstance(parsed,(dict,list)):
            return json.dumps(display_value(parsed),ensure_ascii=False,separators=(',',':'))
        return _LONG_HEX.sub('[内部标识]',value)
    return value


def connect_readonly(path):
    db=sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro',uri=True,timeout=.2)
    db.execute('PRAGMA busy_timeout=200')
    return db


def poll_session(db, cursor, limit=100):
    """Small incremental batches; API image payloads are loaded only on demand."""
    marks=','.join('?' for _ in KINDS)
    sql=("SELECT seq,wall_s,kind,CASE WHEN kind='api_request' THEN "
         "json_object('call_id',json_extract(value,'$.call_id'),'model',json_extract(value,'$.model'),"
         "'observation_id',json_extract(value,'$.observation_id')) WHEN kind='api_response' THEN "
         "json_object('call_id',json_extract(value,'$.call_id'),'content',json_extract(value,'$.content'),"
         "'full_return_s',json_extract(value,'$.full_return_s'),'finish_reason',json_extract(value,'$.finish_reason'),"
         "'message',json_extract(value,'$.message')) "
         "ELSE value END FROM events WHERE seq>? AND kind IN ("+marks+") ORDER BY seq LIMIT ?")
    db.execute('BEGIN')
    try:
        row=db.execute('SELECT value FROM state WHERE id=1').fetchone()
        events=[dict(seq=seq,wall_s=wall,kind=kind,value=json.loads(value)) for seq,wall,kind,value in
                db.execute(sql,(cursor,*KINDS,limit)).fetchall()]
        for event in events:
            if event['kind']=='api_request':
                request_row=db.execute('SELECT value FROM events WHERE seq=?',(event['seq'],)).fetchone()
                request_value=json.loads(request_row[0])
                if request_value.get('purpose')!='elevator_monitor':
                    event['context']=request_context(request_value)
        db.commit()
    except BaseException:
        db.rollback()
        raise
    return (None if row is None else json.loads(row[0])),events


def message_identity(message):
    def normalized(item):
        if isinstance(item,dict) and isinstance(item.get('url'),str) and item['url'].startswith('data:image/'):
            return {**{k:normalized(v) for k,v in item.items() if k not in ('cache_control','url')},
                    'session_image_sha256':hashlib.sha256(item['url'].encode()).hexdigest()}
        if isinstance(item,dict):return {k:normalized(v) for k,v in item.items() if k!='cache_control'}
        if isinstance(item,list):return [normalized(v) for v in item]
        return item
    message=normalized(message)
    content=message.get('content')
    if (isinstance(content,list) and len(content)==1 and isinstance(content[0],dict) and
            set(content[0])=={'type','text'} and content[0]['type']=='text'):
        message['content']=content[0]['text']
    raw=json.dumps(message,ensure_ascii=False,sort_keys=True,separators=(',',':'))
    return hashlib.sha256(raw.encode()).hexdigest()


def request_context(value):
    """Project actual request messages without copying image data into the GUI."""
    request=value.get('request') or {}
    referenced_images=value.get('image_storage')=='session_sha256'
    messages=[];image_index=0
    def clean(item, images):
        nonlocal image_index
        if isinstance(item,dict):
            if 'session_image_sha256' in item and referenced_images:
                images.append(image_index)
                image_index+=1
                return {**{k:clean(v,images) for k,v in item.items() if k not in ('session_image_sha256','cache_control')},
                        'url':'[图片]'}
            return {k:clean(v,images) for k,v in item.items() if k!='cache_control'}
        if isinstance(item,list):return [clean(v,images) for v in item]
        if isinstance(item,str) and item.startswith('data:image/') and ';base64,' in item:
            images.append(image_index);image_index+=1
            return '[图片]'
        return item
    for message in request.get('messages',[]):
        images=[];display=clean(message,images)
        content=display.get('content')
        if isinstance(content,str):summary=content
        else:
            descriptions=[]
            sources={'static_reference_map':'标注地图','runtime_current_state':'当前运行状态',
                     'current_rgb_observation':'当前 RGB 观察','current_spatial_view':'当前空间观察'}
            for part in content or []:
                if not isinstance(part,dict) or part.get('type')!='text':continue
                value=str(part.get('text',''))
                try:
                    parsed=json.loads(value)
                    if isinstance(parsed,dict) and parsed.get('source') in sources:
                        value=sources[parsed['source']]
                except (ValueError,TypeError):pass
                if value:descriptions.append(value[:120])
            summary=' · '.join(descriptions)
        summary=re.sub(r'\s*[\r\n]+\s*',' ',summary).strip()
        calls=display.get('tool_calls') or []
        if calls:
            names=', '.join(str(c.get('function',{}).get('name','')) for c in calls)
            summary=' '.join(part for part in (summary,names) if part)
        image_names=[]
        if isinstance(message.get('content'),list):
            previous=''
            labels={'static_reference_map':'标注地图','current_rgb_observation':'当前 RGB',
                    'trajectory_global_overview':'全局轨迹地图','target_or_robot_local_map':'历史局部地图',
                    'target_or_robot_static_crop':'静态地图附近放大图',
                    'historical_elevator_rgb':'历史乘梯 RGB','current_elevator_rgb':'当前乘梯 RGB',
                    'standard_start_reference_rgb':'标准起点历史 RGB','standard_start_reference_lidar':'标准起点历史雷达图',
                    'current_local_lidar':'当前局部雷达图','destination_waiting_area':'目标楼层地图',
                    'failed_pose_scan_overlay':'上次定位叠图'}
            for part in message['content']:
                if not isinstance(part,dict):continue
                if part.get('type')=='text':
                    previous=part.get('text','')
                    try:
                        parsed=json.loads(previous)
                        if isinstance(parsed,dict):previous=parsed.get('source',previous)
                    except (ValueError,TypeError):pass
                elif part.get('type')=='image_url':
                    image_names.append(next((name for key,name in labels.items() if key in previous),'输入图像'))
        messages.append(dict(role=message.get('role','unknown'),summary=summary[:400],images=images,image_names=image_names,
            identity=message_identity(message),tool_call_id=message.get('tool_call_id'),
            call_ids=[call.get('id') for call in calls]))
    return dict(messages=messages,tools=[x.get('function',{}).get('name') for x in request.get('tools',[])])


class ContextTimeline:
    """Align successive actual requests while retaining each message occurrence once."""
    def __init__(self):
        self.entries=[]
        self.active=[]
        self.pending_response=None

    def _append(self, message, seq, kind, index, *, pending):
        item=dict(message,seq=seq,kind=kind,index=index,active=True,pending=pending)
        self.entries.append(item)
        return len(self.entries)-1

    def request(self, context, seq):
        messages=context['messages']
        def alignment(message):
            if message['role']=='tool' and message.get('tool_call_id'):
                return ('tool',message['tool_call_id'])
            if message['role']=='assistant' and message.get('call_ids') and all(message['call_ids']):
                return ('assistant',*message['call_ids'])
            return ('message',message['identity'])
        before=[alignment(self.entries[i]) for i in self.active]
        after=[alignment(message) for message in messages]
        matcher=SequenceMatcher(a=before,b=after,autojunk=False)
        matched={}
        for left,right,count in matcher.get_matching_blocks():
            for offset in range(count):matched[right+offset]=self.active[left+offset]
        for i in self.active:self.entries[i]['active']=False
        active=[]
        for index,message in enumerate(messages):
            item=matched.get(index)
            if item is None:
                item=self._append(message,seq,'api_request',index,pending=False)
            else:
                entry=self.entries[item]
                if entry['identity']!=message['identity']:
                    entry.update(message,seq=seq,kind='api_request',index=index,representation_changed=True)
                entry.update(active=True,pending=False)
            active.append(item)
        self.active=active
        self.pending_response=None

    def response(self, message, seq):
        if not isinstance(message,dict):return
        item=request_context(dict(request=dict(messages=[message])))['messages'][0]
        index=self._append(item,seq,'api_response',None,pending=True)
        self.active.append(index)
        self.pending_response=index

    def exchange(self, messages, seq):
        for index,message in enumerate(messages):
            item=request_context(dict(request=dict(messages=[message])))['messages'][0]
            if (index==0 and self.pending_response is not None and
                    self.entries[self.pending_response]['identity']==item['identity']):
                self.pending_response=None
                continue
            self.active.append(self._append(item,seq,'conversation_exchange',index,pending=True))


def message_detail(message):
    """Present parsed tool arguments beside, never in place of, the original message."""
    calls=[]
    for call in message.get('tool_calls') or []:
        function=call.get('function') or {};raw=function.get('arguments')
        try:arguments=json.loads(raw) if isinstance(raw,str) else raw
        except ValueError:arguments='参数不是有效 JSON'
        calls.append(dict(id=call.get('id'),name=function.get('name'),arguments=arguments))
    return {'原始消息':message,'工具调用参数':calls} if calls else message


def select_detail(kind, value, images, message_index=None, image_index=None):
    """Limit message details to that message's images; retain whole-event view separately."""
    message = None
    if kind == 'api_request' and message_index is not None:
        message = value['request']['messages'][message_index]
    elif kind == 'api_response' and isinstance(value.get('message'), dict):
        message = value['message']
    elif kind == 'conversation_exchange' and message_index is not None:
        message = value['messages'][message_index]
    if message is None:
        return value, images, 0 if image_index is None else image_index
    indices = []
    def collect(item):
        if isinstance(item, dict):
            for child in item.values(): collect(child)
        elif isinstance(item, list):
            for child in item: collect(child)
        elif isinstance(item, str):
            match = re.fullmatch(r'\[输入图像 (\d+)(?:，见图像列表| 编码无效)\]', item)
            if match:
                index = int(match.group(1))-1
                if 0 <= index < len(images) and index not in indices: indices.append(index)
    collect(message)
    selected = [images[index] for index in indices]
    initial = indices.index(image_index) if image_index in indices else 0
    return message_detail(message), selected, initial


def event_detail(db, seq):
    row=db.execute('SELECT kind,value FROM events WHERE seq=?',(seq,)).fetchone()
    if row is None:raise ValueError('事件不存在')
    kind,value=row[0],json.loads(row[1])
    if kind=='api_request':
        from .core.store import restore_request_images
        value=restore_request_images(db,value)
    return format_event_detail(kind,value)


def format_event_detail(kind,value):
    """Prepare saved or in-memory request details without retaining images."""
    image_names=[]
    if kind=='api_request':
        for message in request_context(value)['messages']:
            image_names.extend(message['image_names'])
    images=[]
    def clean(item, name=''):
        if isinstance(item,dict):
            return {k:('[已隐藏]' if k.lower() in ('authorization','api_key','apikey','access_token') else clean(v,k))
                    for k,v in item.items()}
        if isinstance(item,list):return [clean(v,name) for v in item]
        if isinstance(item,str) and item.startswith('data:image/') and ';base64,' in item:
            header,payload=item.split(',',1)
            label=image_names[len(images)] if len(images)<len(image_names) else '输入图像'
            try: data=base64.b64decode(payload,validate=True)
            except ValueError:
                images.append((f'输入图像 {len(images)+1} · {label}（编码无效）',None))
                return f'[输入图像 {len(images)} 编码无效]'
            images.append((f'输入图像 {len(images)+1} · {label} ({header[5:].split(";")[0]})',data))
            return f'[输入图像 {len(images)}，见图像列表]'
        return item
    return kind,clean(value),images


def event_summary(event):
    kind,value=event['kind'],event['value']
    if kind=='agent_bag_recording':
        labels={'preparing':'准备中','recording':'录制中','finalizing':'正在关闭与校验',
                'saved':'已保存','failed':'失败','needs_review':'需要检查','finished':'已结束'}
        return '自动录包：'+labels.get(value.get('status'),str(value.get('status') or '未知'))+(('；'+str(value['bag'])) if value.get('bag') else '')+(('；'+str(value['error'])) if value.get('error') else '')
    if kind=='follow_target_confirmed':
        ref=value.get('reference') or {}
        return 'Agent 已确认跟随参考：候选 '+str(ref.get('candidate_id'))+'；'+str(ref.get('description') or '')
    if kind=='t2_replay':
        return 'T2 回放：'+str(value.get('status') or value.get('event') or '未知')+(
            '；选择 '+str(value['selection']) if value.get('selection') else '')+(
            '；'+str(value['reason']) if value.get('reason') else '')
    if kind=='subtask_skipped':return '操作者跳过当前子任务；该子任务未完成。'
    if kind=='registered_task_imported':return '已复用历史登记，从第一项重新执行。'
    if kind=='task_monitor_event':
        labels={'navigation_state_changed':'导航状态变化','operator_review_requested':'用户请求复核',
                'semantic_review_due':'周期复核','region_entered':'进入区域','no_measured_progress':'暂无测量进展',
                'distance_progress':'行驶距离','rotation_progress':'转角','stopped_duration':'持续停稳','monitor_timeout':'等待超时',
                'conditional_review_due':'条件式持续复核'}
        return '复核触发：'+'、'.join(labels.get(r.get('kind'),str(r.get('kind','未知'))) for r in value.get('reasons',[]))
    if kind=='conversation_exchange':
        parts=[]
        for message in value.get('messages',[]):
            role=message.get('role','')
            text=message.get('content')
            if text:parts.append(f'{role}: {str(text)[:250]}')
            for call in message.get('tool_calls') or []:
                parts.append('工具: '+str((call.get('function') or {}).get('name','')))
        return ' | '.join(parts) or '对话'
    if kind=='tool_intent':return '调用工具: '+str(value.get('tool',''))
    if kind=='api_request':return '模型请求 '+str(value.get('call_id',''))[:10]
    if kind=='api_response':return '模型返回: '+str(value.get('content') or value.get('finish_reason') or '')[:200]
    if kind=='api_input_rejected':
        return f"模型输入估算为 {value.get('estimated_input_tokens')}，超过本地限制 {value.get('limit')}；请求未发送。"
    if kind=='manual_route_progress':
        i=value.get('next_waypoint_index');n=value.get('waypoint_count')
        return f'路线进度：下一点 {i+1 if isinstance(i,int) else "?"}/{n}'
    if kind=='manual_arrival_invalidated':
        reasons={'movement_detected':'检测到运动','outside_position_tolerance':'离开目标位置容差',
                 'outside_heading_tolerance':'朝向超出要求','localization_identity_changed':'定位身份变化',
                 'map_association_changed':'地图关联变化','target_changed':'目标身份变化','clock_regressed':'反馈时间回退'}
        return '到达证据失效：'+reasons.get(value.get('reason'),str(value.get('reason') or '未知原因'))
    if kind=='manual_arrival_recovered':return '已重新记录定位和停稳到达证据。'
    if kind=='model_action_missing':
        return '模型只返回文字，将依据当前证据重新决策一次。' if value.get('correction_requested') else '模型未提交执行动作，任务已请求暂停。'
    if kind=='elevator_monitor_started':return '开始周期观察乘梯过程。'
    if kind=='elevator_model_failed':return '乘梯观察不可用：'+str(value.get('reason') or '未知原因')
    if kind=='elevator_model_discarded':return '乘梯阶段已变化，未应用旧观察结果。'
    if kind=='floor_handoff_started':return '已判断离开电梯，正在切换目标楼层地图。'
    if kind=='manual_floor_rebound':return '目标楼层定位已关联，原任务正在恢复。'
    if kind=='elevator_stage_completed':return '乘梯阶段已完成，继续后续子任务。'
    if kind=='manual_reference_region_conflict':return '参考路径穿过绕行区域；请手动停稳并等待新建议。'
    if kind=='manual_reference_status':
        status={'reference_ready':'已生成','waiting':'等待中','unavailable':'不可用'}.get(value.get('status'),value.get('status') or '未知')
        check={'clear':'绕行区域检查通过','unavailable':'绕行区域检查不可用','not_applicable':'无需绕行区域检查'}.get(value.get('region_check'))
        return '参考路径：'+str(status)+(('；'+check) if check else '')
    if kind in ('manual_goal_evidence_wait_started','manual_goal_evidence_wait_finished','manual_goal_evidence_unproven'):
        prefix=('证据已就绪' if value.get('ready') else
                '等待到达证据' if kind=='manual_goal_evidence_wait_started' else '到达证据不足')
        labels={'manual_arrival_evidence_missing_or_mismatched':'当前到达证据缺失或不匹配',
                'operator_confirmation_missing_or_mismatched':'到达证据缺失或不匹配',
                'stopped_estimate_unavailable':'当前停稳估计不可用','mapped_pose_unavailable':'地图定位不可用'}
        return prefix+('：'+', '.join(labels.get(issue,issue) for issue in value.get('issues',[])) if value.get('issues') else '')
    if kind=='manual_goal_evidence_refresh_failed':
        return '读取新反馈失败：'+str(value.get('error_type','未知错误'))
    if kind=='conversation_trimmed':return '模型上下文已裁剪；会话历史仍保留'
    if kind=='navigation_repeated_refusal':
        reason={'current_stage_does_not_support_navigation':'当前任务阶段不能导航',
                'model_input_context_unavailable':'原始任务或观察身份不可用',
                'model_input_expired_request_fresh_observation':'观察过期且无法重新核对',
                'candidate_footprint_occupied':'目标位置与占据区域冲突',
                'fresh_navigation_view_unavailable':'无法获取有效的新地图观察',
                'robot_pose_changed_since_model_input':'轮椅位姿已明显变化'}.get(value.get('reason'),value.get('reason') or '未知原因')
        return f"同一导航目标被拒绝 {value.get('count','?')} 次：{reason}；请调整任务或补充说明。"
    if kind=='route_repeated_refusal':
        reasons={'proposed_route_intersects_task_region':'路线穿过目标区域',
                 'circuit_waypoint_inside_region':'途经点位于绕行区域内',
                 'circuit_requires_three_distinct_waypoints':'绕行需要至少三个不同的途经点',
                 'waypoint_statically_occupied':'途经点在静态地图占据区域',
                 'waypoint_currently_occupied':'途经点被当前局部观测占据',
                 'waypoint_unknown':'途经点位置为未知',
                 'waypoint_outside_map':'途经点位于地图外',
                 'route_waypoints_review_required':'自主途经点较多，需要精简或说明保留原因',
                 'circuit_route_must_return_to_its_start':'路线未回到起点附近',
                 'proposed_route_does_not_make_requested_circuit':'绕行方向或圈数不符',
                 'proposed_route_excessive_reverse_winding':'路线包含过多反向绕行',
                 'model_input_expired_request_fresh_observation':'观察过期且无法重新核对',
                 'fresh_navigation_view_unavailable':'无法获取有效的新地图观察',
                 'spatial_source_changed':'地图观察已变化',
                 'robot_pose_changed_since_model_input':'轮椅位姿已明显变化'}
        reason=value.get('reason') or '未知原因'
        return f"同一路线被拒绝 {value.get('count','?')} 次：{reasons.get(reason,reason)}；请补充要求或调整任务。"
    return str(value.get('reason') or value.get('status') or kind)[:250]
