"""Console input for CLI workflows; user control bypasses model inference."""
import queue
import sys
import threading


class Console:
    def __init__(self):
        self.inbox = queue.Queue(32)

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
            dispatch(command, actor)
        except ValueError as error:
            emit(dict(event_rejected=str(error)))


def dispatch(command, actor):
    if command in ('continue', 'pause', 'resume', 'cancel'):
        actor.event(command)
        return
    for prefix, name in (('replace ', 'replace'), ('answer ', 'answer')):
        if command.startswith(prefix) and command[len(prefix):].strip():
            actor.event(name, command[len(prefix):].strip())
            return
    raise ValueError('continue / pause / resume / cancel / answer 补充说明 / replace 新指令')
