"""One background model request with cancellation and stale-result epochs."""
import copy
import threading


class AsyncModel:
    def __init__(self):
        self.lock = threading.Lock()
        self.epoch = 0
        self.thread = None
        self.result = None

    @property
    def busy(self):
        with self.lock:
            return self.result is not None or self.thread is not None and self.thread.is_alive()

    def submit(self, purpose, context, work):
        with self.lock:
            if self.result is not None or self.thread is not None and self.thread.is_alive():
                raise RuntimeError('One model request is already pending')
            epoch = self.epoch
            context = copy.deepcopy(context)

            def execute():
                try:
                    outcome = dict(ok=True, value=work())
                except Exception as error:
                    # HTTP and model clients persist their own redacted detail.
                    outcome = dict(ok=False, error_type=type(error).__name__)
                with self.lock:
                    self.result = dict(outcome, purpose=purpose, context=context, epoch=epoch)

            self.thread = threading.Thread(target=execute, daemon=True, name='agent-model')
            self.thread.start()

    def invalidate(self):
        with self.lock:
            self.epoch += 1

    def poll(self):
        with self.lock:
            result, self.result = self.result, None
            if result is not None:
                result['stale'] = result['epoch'] != self.epoch
            return result

    def close(self):
        self.invalidate()
        return dict(request_unresolved=self.busy, automatically_retried=False)
