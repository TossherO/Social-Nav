"""One background model request with cancellation and stale-result epochs."""
import copy
import threading
import time

_worker=threading.local()


def request_cancelled():
    flag=getattr(_worker,'cancel',None)
    return flag is not None and flag.is_set()


def request_timing(prepare_started_s):
    """Share process-monotonic timestamps across one worker request."""
    timing = getattr(_worker, 'timing', None)
    return timing if timing is not None else dict(prepare_started_s=prepare_started_s)


class AsyncModel:
    def __init__(self):
        self.lock = threading.Lock()
        self.epoch = 0
        self.thread = None
        self.result = None
        self.cancel = None
        self.closed = False

    @property
    def busy(self):
        with self.lock:
            return self.result is not None or self.thread is not None and self.thread.is_alive()

    def submit(self, purpose, context, work, *, prepare_started_s=None, request_index=None):
        with self.lock:
            if self.closed:
                raise RuntimeError('Model worker is closed')
            if self.result is not None or self.thread is not None and self.thread.is_alive():
                raise RuntimeError('One model request is already pending')
            epoch = self.epoch
            context = copy.deepcopy(context)
            cancel=self.cancel=threading.Event()
            timing=dict(request_index=request_index, prepare_started_s=prepare_started_s,
                        submitted_s=time.perf_counter())

            def execute():
                _worker.cancel=cancel
                _worker.timing=timing
                timing['worker_started_s']=time.perf_counter()
                try:
                    outcome = dict(ok=True, value=work())
                except Exception as error:
                    # HTTP and model clients persist their own redacted detail.
                    outcome = dict(ok=False, error_type=type(error).__name__,
                                   retry_after_s=getattr(error,'retry_after_s',None))
                with self.lock:
                    timing['worker_finished_s']=time.perf_counter()
                    if not self.closed:
                        self.result = dict(outcome, purpose=purpose, context=context, epoch=epoch,
                                           timing=timing, request_index=request_index)

            self.thread = threading.Thread(target=execute, daemon=True, name='agent-model')
            self.thread.start()

    def invalidate(self):
        with self.lock:
            self.epoch += 1
            if self.cancel is not None:self.cancel.set()

    def poll(self):
        with self.lock:
            result, self.result = self.result, None
            if result is not None:
                result['stale'] = result['epoch'] != self.epoch
            return result

    def close(self):
        with self.lock:
            self.closed = True
            self.epoch += 1
            if self.cancel is not None:
                self.cancel.set()
            self.result = None
            unresolved = self.thread is not None and self.thread.is_alive()
        return dict(request_unresolved=unresolved, automatically_retried=False)
