"""Read an audited task registration as a fresh manual-session input."""
import copy
import json
from pathlib import Path
import sqlite3

from perception.environment.context import fingerprint, layer_binding


def _annotations(value):
    return {kind:[{k:v for k,v in item.items() if k not in ('display_id','task_origin')}
                  for item in value.get(kind,[]) if item.get('place_id')!='mission_start']
            for kind in ('points','regions')}


def catalog_assets(catalogs):
    """Record stable content identities, excluding the task-local starting pose."""
    from ..adapters.shadow import load_catalog_map
    assets={}
    for floor,catalog in catalogs.items():
        asset=load_catalog_map(catalog)
        try:
            assets[floor]=dict(frame_id=catalog.value['frame_id'],
                localization_map_id=catalog.value.get('localization_map_id'),
                global_binding=layer_binding(asset.snapshot),annotations=_annotations(catalog.annotations()))
        finally:asset.close()
    return assets


def _execution_clarifications(db, after, before):
    for (raw,) in db.execute("SELECT value FROM events WHERE kind='api_request' AND seq>? AND seq<? ORDER BY seq",(after,before)):
        event=json.loads(raw)
        for message in (event.get('request') or {}).get('messages',[]):
            content=message.get('content')
            texts=[content] if isinstance(content,str) else [p.get('text') for p in content if isinstance(p,dict)] if isinstance(content,list) else []
            for text in texts:
                try:value=json.loads(text)
                except (ValueError,TypeError):continue
                if isinstance(value,dict) and value.get('source')=='registered_task_plan':
                    return value.get('clarifications',[])
    return None


def load_registered_task(path):
    """Use a consistent, read-only database snapshot; never load images or secrets."""
    path=Path(path).resolve()
    if not path.is_file():raise ValueError('历史会话文件不存在')
    try:
        db=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True,timeout=2.)
        try:
            db.execute('BEGIN')
            row=db.execute('SELECT value FROM state WHERE id=1').fetchone()
            if row is None:raise ValueError('历史会话缺少任务状态')
            state=json.loads(row[0]);revision=state.get('revision');task_id=state.get('task_id')
            instruction=state.get('instruction');floor=state.get('initial_floor')
            goals=[entry.get('requirement') for entry in state.get('task_goals',[])]
            if not isinstance(instruction,str) or not instruction.strip() or not goals or not floor:
                raise ValueError('历史会话没有完整的任务登记或起始楼层')
            selected=None
            for seq,kind,raw in db.execute("SELECT seq,kind,value FROM events WHERE kind IN ('model_action_applied','registered_task_imported') ORDER BY seq DESC"):
                event=json.loads(raw)
                if kind=='model_action_applied':
                    result=event.get('result') or {};context=event.get('context') or {}
                    saved=[g.get('requirement') for g in result.get('goals',[])]
                    if ((event.get('call') or {}).get('tool')!='record_task' or result.get('status')!='recorded' or
                            context.get('revision')!=revision or context.get('task_id')!=task_id or
                            result.get('original_instruction')!=instruction or saved!=goals):continue
                else:
                    if (event.get('revision')!=revision or event.get('task_id')!=task_id or
                            event.get('instruction')!=instruction or event.get('goals')!=goals):continue
                selected=(seq,kind,event);break
            if selected is None:raise ValueError('找不到与当前任务一致的成功登记记录')
            seq,kind,event=selected
            if kind=='registered_task_imported':
                clarifications=event.get('clarifications');assets=event.get('assets')
            else:
                next_revision=db.execute("SELECT COALESCE(MIN(seq),9223372036854775807) FROM events WHERE kind='task_created' AND seq>?",(seq,)).fetchone()[0]
                clarifications=_execution_clarifications(db,seq,next_revision)
                if clarifications is None:
                    if state.get('clarifications'):raise ValueError('无法核实登记时使用的澄清内容')
                    clarifications=[]
                row=db.execute("SELECT value FROM events WHERE kind='task_asset_provenance' AND seq<? ORDER BY seq DESC LIMIT 1",(seq,)).fetchone()
                assets=json.loads(row[0]).get('floors') if row else None
                if assets is None:
                    # Legacy sessions retained the original static map reference and map association.
                    row=db.execute("SELECT value FROM events WHERE kind='reference_map_prepared' AND seq<? ORDER BY seq DESC LIMIT 1",(seq,)).fetchone()
                    if row is None:raise ValueError('历史会话缺少静态地图与标注记录')
                    ref=json.loads(row[0]);context=event.get('context') or {};binding=context.get('map_binding') or {}
                    if not binding.get('valid') or not binding.get('map_id') or not ref.get('global_binding'):
                        raise ValueError('历史会话缺少有效定位地图或静态地图身份')
                    assets={floor:dict(frame_id=ref.get('frame_id'),localization_map_id=binding['map_id'],
                                       global_binding=ref['global_binding'],annotations=_annotations(ref.get('annotations') or {}))}
            if not isinstance(clarifications,list) or not isinstance(assets,dict):raise ValueError('历史任务缺少登记上下文或地图记录')
            needed={g.get('floor_id',floor) for g in goals}|{g['to_floor'] for g in goals if g.get('to_floor')}
            if not needed.issubset(assets):raise ValueError('历史跨楼层任务缺少部分楼层的地图来源，无法核实')
            binding=((event.get('context') or {}).get('map_binding') or {}) if kind=='model_action_applied' else (
                event.get('registration_map_binding') or {})
            if (goals[0].get('floor_id',floor)!=floor or not binding.get('valid') or
                    binding.get('map_id')!=assets[floor].get('localization_map_id')):
                raise ValueError('历史任务登记时的起始楼层或定位地图无法核实')
            result=dict(instruction=instruction,goals=goals,clarifications=clarifications,initial_floor=floor,
                        assets={key:assets[key] for key in needed},source_task_id=task_id,source_revision=revision,source_event_seq=seq)
            result['digest']=fingerprint(result)
            return result
        finally:db.close()
    except (sqlite3.Error,KeyError,TypeError,AttributeError,json.JSONDecodeError) as error:
        raise ValueError('无法读取完整的历史任务记录：'+type(error).__name__) from error


def verify_registered_task(plan, catalogs, initial_floor):
    if plan['initial_floor']!=initial_floor:raise ValueError('当前楼层与历史任务起始楼层不一致')
    if not set(plan['assets']).issubset(catalogs):raise ValueError('当前未配置历史任务所需的全部楼层')
    current=catalog_assets({floor:catalogs[floor] for floor in plan['assets']})
    for floor,old in plan['assets'].items():
        new=current[floor]
        for key,label in (('frame_id','坐标系'),('localization_map_id','定位地图'),
                          ('global_binding','静态地图'),('annotations','语义标注')):
            if not old.get(key) or old[key]!=new.get(key):
                raise ValueError(f'{floor} 的{label}与历史任务不一致或缺失')
    return copy.deepcopy(current)


def import_registered_task(session, plan, assets, source_path):
    """Re-register requirements with a new task identity; preserve no execution progress."""
    session.state['clarifications']=copy.deepcopy(plan['clarifications'])
    result=session.record_task(copy.deepcopy(plan['goals']))
    if result.get('status')!='recorded':raise ValueError('历史任务重新登记失败')
    session._save('registered_task_imported',dict(task_id=session.state['task_id'],revision=session.state['revision'],
        instruction=plan['instruction'],goals=copy.deepcopy(plan['goals']),clarifications=copy.deepcopy(plan['clarifications']),
        assets=assets,source_session=str(Path(source_path).resolve()),source_task_id=plan['source_task_id'],
        source_revision=plan['source_revision'],source_event_seq=plan['source_event_seq'],source_digest=plan['digest'],
        registration_map_binding=copy.deepcopy(session.navigation.snapshot().get('map_binding'))))
