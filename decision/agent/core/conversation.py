"""Bounded native conversation; current measurements are attached separately."""
import copy
from .store import encode


class Conversation:
    def __init__(self, instruction, catalog, emit, *, max_exchanges=12, history_bytes=12000):
        self.introduction = [dict(role='user', content=instruction), dict(role='user', content=encode(
            dict(source='initial_map_catalog', catalog=catalog)))]
        self.exchanges = []
        self.call_ids = set()
        self.revision = 0
        self.max_exchanges, self.history_bytes, self.emit = max_exchanges, history_bytes, emit

    def _append(self, messages):
        self.exchanges.append(copy.deepcopy(messages))
        self.revision += 1
        self.emit('conversation_exchange', dict(revision=self.revision, messages=messages))
        removed = 0
        while len(self.exchanges) > 1 and (len(self.exchanges) > self.max_exchanges or
                len(encode(self.exchanges).encode()) > self.history_bytes):
            self.exchanges.pop(0)
            removed += 1
        if removed:
            self.emit('conversation_trimmed', dict(exchanges=removed, revision=self.revision,
                      original_task_preserved=True, paired_tool_messages=True))

    def validate(self, reply):
        for call in reply.message.get('tool_calls') or []:
            if call['id'] in self.call_ids:
                raise ValueError('Native tool call identity was already consumed')

    def commit(self, reply, result=None):
        self.validate(reply)
        messages = [reply.message]
        if reply.action is not None:
            if result is None:
                raise ValueError('Tool reply requires its corresponding result')
            identity = reply.message['tool_calls'][0]['id']
            messages.append(dict(role='tool', tool_call_id=identity, content=encode(result)))
            self.call_ids.add(identity)
        self._append(messages)

    def notice(self, source, **value):
        self._append([dict(role='user', content=encode(dict(source=source, **value)))])

    def messages(self):
        return copy.deepcopy(self.introduction+[message for group in self.exchanges for message in group])

    def snapshot(self):
        return dict(revision=self.revision, exchanges=len(self.exchanges),
                    consumed_tool_calls=len(self.call_ids), automatic_resume=False)
