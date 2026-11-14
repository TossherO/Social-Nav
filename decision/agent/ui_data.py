"""Read-only session projection for the local manual Agent window."""
import base64
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from difflib import SequenceMatcher


KINDS=('conversation_exchange','conversation_trimmed','tool_intent','model_action_applied',
       'api_request','api_response','api_failure','model_failed','model_protocol_rejected',
       'model_result_discarded','manual_arrival_confirmation','manual_goal_evidence_wait_started',
       'manual_goal_evidence_wait_finished','manual_goal_evidence_refresh_failed','manual_goal_evidence_unproven','manual_route_progress',
       'manual_route_trace','manual_reference_status','operator_command','navigation_completed','navigation_ack','navigation_cancel_confirmed',
       'navigation_context_lost','task_connection_lost','user_event','task_created','assistant_message','agent_run_finished')


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
                event['context']=request_context(json.loads(request_row[0]))
        db.commit()
    except BaseException:
        db.rollback()
        raise
    return (None if row is None else json.loads(row[0])),events


def message_identity(message):
    def normalized(item):
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
    messages=[];image_index=0
    def clean(item, images):
        nonlocal image_index
        if isinstance(item,dict):
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
                    'trajectory_global_overview':'全局轨迹地图','target_or_robot_local_map':'局部地图'}
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


def event_detail(db, seq):
    row=db.execute('SELECT kind,value FROM events WHERE seq=?',(seq,)).fetchone()
    if row is None:raise ValueError('事件不存在')
    kind,value=row[0],json.loads(row[1])
    images=[]
    def clean(item, name=''):
        if isinstance(item,dict):
            return {k:('[已隐藏]' if k.lower() in ('authorization','api_key','apikey','access_token') else clean(v,k))
                    for k,v in item.items()}
        if isinstance(item,list):return [clean(v,name) for v in item]
        if isinstance(item,str) and item.startswith('data:image/') and ';base64,' in item:
            header,payload=item.split(',',1)
            try: data=base64.b64decode(payload,validate=True)
            except ValueError:
                images.append((f'输入图像 {len(images)+1}（编码无效）',None))
                return f'[输入图像 {len(images)} 编码无效]'
            images.append((f'输入图像 {len(images)+1} ({header[5:].split(";")[0]})',data))
            return f'[输入图像 {len(images)}，见图像列表]'
        return item
    return kind,clean(value),images


def event_summary(event):
    kind,value=event['kind'],event['value']
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
    if kind=='manual_route_progress':
        i=value.get('next_waypoint_index');n=value.get('waypoint_count')
        return f'路线进度：下一点 {i+1 if isinstance(i,int) else "?"}/{n}'
    if kind in ('manual_goal_evidence_wait_started','manual_goal_evidence_wait_finished','manual_goal_evidence_unproven'):
        prefix=('证据已就绪' if value.get('ready') else
                '等待到达证据' if kind=='manual_goal_evidence_wait_started' else '到达证据不足')
        return prefix+('：'+', '.join(value.get('issues',[])) if value.get('issues') else '')
    if kind=='conversation_trimmed':return '模型上下文已裁剪；会话历史仍保留'
    return str(value.get('reason') or value.get('status') or kind)[:250]
