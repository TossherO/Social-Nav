"""Console input for CLI workflows; user control bypasses model inference."""
import queue
import json
import sys
import threading


class Console:
    def __init__(self, *, structured=False):
        self.inbox = queue.Queue(32)
        self.structured = structured
        self.consumed = set()

        def read():
            for line in sys.stdin:
                self.inbox.put(line.strip())
        threading.Thread(target=read, daemon=True, name='agent-console').start()

    def poll(self, actor, emit):
        try:
            command = self.inbox.get_nowait()
        except queue.Empty:
            return
        try:
            if self.structured:
                result=dispatch_structured(command,actor,self.consumed)
                actor.session.store.event('operator_command',result)
                emit(result)
            else:
                dispatch(command, actor)
        except ValueError as error:
            emit(dict(event_rejected=str(error)))


def dispatch_structured(command, actor, consumed):
    """Apply a GUI command only to the task and goal observed by its sender."""
    try:
        value=json.loads(command)
    except (TypeError,ValueError):
        raise ValueError('Invalid GUI command JSON') from None
    if not isinstance(value,dict) or set(value)!={'command_id','name','task_id','revision','goal_id','call_id','text'}:
        raise ValueError('Invalid GUI command fields')
    command_id=value['command_id']
    if not isinstance(command_id,str) or not 1<=len(command_id)<=64:
        raise ValueError('Invalid GUI command identity')
    if command_id in consumed:
        return dict(event='operator_command',command_id=command_id,accepted=False,reason='duplicate_command')
    if len(consumed)>=2048:
        return dict(event='operator_command',command_id=command_id,accepted=False,reason='command_limit_reached')
    consumed.add(command_id)
    session=actor.session
    state=session.state
    current=next((g for g in session.task_goals.goals if g['status']!='completed'),None)
    goal_id=None if current is None else current['requirement']['goal_id']
    handle=session.skill.record.get('handle') or {}
    if (session.navigation.scope!='manual' or value['task_id']!=state['task_id'] or
            type(value['revision']) is not int or value['revision']!=state['revision'] or
            value['goal_id']!=goal_id or value['call_id']!=handle.get('call_id')):
        return dict(event='operator_command',command_id=command_id,accepted=False,reason='task_or_goal_changed')
    name=value['name'];text=value['text']
    if name not in ('continue','pause','resume','cancel','arrived','answer','replace'):
        return dict(event='operator_command',command_id=command_id,accepted=False,reason='unknown_command')
    if name in ('answer','replace'):
        if not isinstance(text,str) or not text.strip() or len(text)>4096:
            return dict(event='operator_command',command_id=command_id,accepted=False,reason='invalid_text')
    elif text is not None:
        return dict(event='operator_command',command_id=command_id,accepted=False,reason='unexpected_text')
    try:
        actor.event(name,text)
    except (ValueError,KeyError) as error:
        return dict(event='operator_command',command_id=command_id,accepted=False,reason=str(error)[:240])
    return dict(event='operator_command',command_id=command_id,accepted=True,name=name)


def dispatch(command, actor):
    if command in ('continue', 'pause', 'resume', 'cancel', 'arrived'):
        actor.event(command)
        return
    for prefix, name in (('replace ', 'replace'), ('answer ', 'answer')):
        if command.startswith(prefix) and command[len(prefix):].strip():
            actor.event(name, command[len(prefix):].strip())
            return
    raise ValueError('continue / pause / resume / cancel / arrived（手动模式） / answer 补充说明 / replace 新指令')
