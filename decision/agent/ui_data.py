"""Read-only session projection for the local manual Agent window."""
import base64
import json
from pathlib import Path
import sqlite3


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
         "'full_return_s',json_extract(value,'$.full_return_s'),'finish_reason',json_extract(value,'$.finish_reason')) "
         "ELSE value END FROM events WHERE seq>? AND kind IN ("+marks+") ORDER BY seq LIMIT ?")
    db.execute('BEGIN')
    try:
        row=db.execute('SELECT value FROM state WHERE id=1').fetchone()
        events=[dict(seq=seq,wall_s=wall,kind=kind,value=json.loads(value)) for seq,wall,kind,value in
                db.execute(sql,(cursor,*KINDS,limit)).fetchall()]
        db.commit()
    except BaseException:
        db.rollback()
        raise
    return (None if row is None else json.loads(row[0])),events


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
        if isinstance(item,str) and item.startswith(('data:image/png;base64,','data:image/jpeg;base64,')):
            if len(images)<6 and len(item)<=12_000_000:
                header,payload=item.split(',',1)
                try: data=base64.b64decode(payload,validate=True)
                except ValueError:return '[图像编码无效]'
                images.append((f'输入图像 {len(images)+1} ({header[5:].split(";")[0]})',data))
                return f'[输入图像 {len(images)}，见图像列表]'
            return '[图像过大或数量超过展示上限]'
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
