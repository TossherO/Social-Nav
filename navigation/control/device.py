"""GPU DWA evaluation with reusable buffers and explicit host exports."""

import ctypes as C
import math
from pathlib import Path

import numpy as np


_SCORE_BYTES = np.dtype(np.float64).itemsize


def _pointer(tensor):
    return None if tensor is None else tensor.data_ptr()


class DeviceDwaBatch:
    """One control thread owns the workspace and consumes each result before reuse."""

    def __init__(self, *, device='cuda:0', library=None, metrics=None):
        import torch

        self.torch = torch
        self.device = torch.device(device)
        self.metrics = metrics
        self.snapshot = None
        self._buffers = {}
        self._kernel_begin = torch.cuda.Event(enable_timing=True)
        self._kernel_end = torch.cuda.Event(enable_timing=True)
        self._result_ready = torch.cuda.Event()

        root = Path(__file__).resolve().parents[2]
        self.library = Path(
            library or root / 'outputs/runtime/build/dwa/libsocial_nav_dwa_cuda.so'
        )
        self.lib = C.CDLL(str(self.library))
        try:
            kernel = self.lib.sn_dwa_predict
        except AttributeError as error:
            raise RuntimeError('Rebuild the DWA CUDA library for finite-horizon prediction') from error
        kernel.argtypes = [
            C.c_void_p, C.c_int, C.c_void_p, C.c_void_p, C.c_int,  # targets, path, arc
            C.c_void_p, C.c_void_p, C.c_void_p, C.c_int, C.c_int,  # state, map, shape
            C.c_void_p, C.c_int, C.c_void_p, C.c_int,             # footprint, people
            C.c_void_p, C.c_int, C.c_void_p,                      # groups, social
            C.c_void_p, C.c_void_p,                               # poses, scores
            C.c_int, C.c_void_p,                                  # steps, stream
        ]

    def bind(self, snapshot):
        self.snapshot = snapshot

    def _buffer(self, name, shape, dtype, *, host=False):
        """Return a shaped view; grow the owning allocation only when necessary."""
        count = math.prod(shape)
        key = (name, host)
        value = self._buffers.get(key)
        if value is None or value.dtype != dtype or value.numel() < count:
            capacity = max(1, count, 0 if value is None else value.numel() * 2)
            if host:
                value = self.torch.empty(capacity, dtype=dtype, pin_memory=True)
            else:
                value = self.torch.empty(capacity, dtype=dtype, device=self.device)
            self._buffers[key] = value
        return value[:count].view(shape)

    def _upload_inputs(self, items):
        """Pack typed inputs into one pinned byte buffer and upload it once."""
        arrays = [np.ascontiguousarray(value) for _, value in items]
        size = sum(array.nbytes for array in arrays)
        host = self._buffer('inputs', (size,), self.torch.uint8, host=True)
        device = self._buffer('inputs', (size,), self.torch.uint8)
        raw = host.numpy()
        offset = 0
        views = []

        # Float64 arrays and two-column int32 group pairs both occupy multiples
        # of eight bytes, so every subsequent float64 view remains aligned.
        for (reason, _), array in zip(items, arrays):
            if array.dtype == np.float64:
                dtype = self.torch.float64
            elif array.dtype == np.int32:
                dtype = self.torch.int32
            else:
                raise ValueError('Unexpected native DWA input dtype')
            end = offset + array.nbytes
            raw[offset:end] = array.view(np.uint8).reshape(-1)
            views.append(device[offset:end].view(dtype).view(array.shape))
            offset = end
            if self.metrics:
                self.metrics.transfer('h2d', array.nbytes, reason)

        device.copy_(host, non_blocking=True)
        return views

    @staticmethod
    def _parameters(controller, grid, model, start, initial, progress):
        """Keep the flat double layout in sync with native/dwa.cu's q fields."""
        motion = controller.motion
        settings = controller.settings
        previous = initial if controller.previous is None else controller.previous
        parameters = np.r_[
            start, initial,
            motion['max_acceleration_mps2'], motion['max_angular_acceleration_rps2'],
            settings['prediction_s'], settings['sample_s'],
            grid.metadata['origin_xy'], grid.metadata['resolution'], progress, previous,
            [settings[key] for key in (
                'path_weight', 'progress_weight', 'heading_weight',
                'change_weight', 'clearance_weight',
            )],
            model.as_of - model.origin,
        ].astype(np.float64)
        if len(parameters) != 21:
            raise RuntimeError('Invalid DWA native parameter layout')
        return parameters

    def _download_scores(self, output, count):
        """Wait for this result before reading or reusing its pinned host storage."""
        host = self._buffer('output', output.shape, self.torch.uint8, host=True)
        host.copy_(output, non_blocking=True)
        self._result_ready.record()
        self._result_ready.synchronize()

        raw = host.numpy()
        score_bytes = count * _SCORE_BYTES
        scores = raw[:score_bytes].view(np.float64)
        if self.metrics:
            self.metrics.transfer(
                'd2h', scores.nbytes, 'dwa_candidate_summaries'
            )
            self.metrics.add(
                'dwa_gpu_event', self._kernel_begin.elapsed_time(self._kernel_end) / 1000
            )
        return scores

    def evaluate(self, controller, scene, model, groups, grid, dynamic, choices,
                 start, initial, path, prepared, progress):
        import torch

        if (self.snapshot is None or
                self.snapshot.metadata['snapshot_id'] != scene.geometry.metadata['snapshot_id']):
            raise ValueError('Device DWA needs the same sealed observation as its host safety capsule')
        layers = self.snapshot.consume()
        occupancy = layers.get('navigation_occupancy', layers['occupancy'])
        clearance = None if 'navigation_occupancy' in layers else layers.get('clearance')
        if dynamic.get('status') == 'people_prediction_over_historical_free':
            occupancy = torch.where(
                layers['person_background_free'] & (occupancy == 100),
                torch.zeros_like(occupancy), occupancy,
            )
            clearance = None
        if dynamic.get('fallback_circles'):
            # Exceptional stale/missing tracking packet: preserve the exact CPU
            # contact view. Normal delegated objects require no grid upload.
            values=np.array(grid.layers['occupancy'],copy=True)
            occupancy=torch.from_numpy(values).to(layers['occupancy'].device)
            clearance=None
            if self.metrics is not None:self.metrics.transfer('h2d',values.nbytes,'dwa_person_fallback_grid')
        self.navigation_occupancy = occupancy
        if (occupancy.dtype != torch.int8 or
                clearance is not None and clearance.dtype != torch.float32):
            raise ValueError('Unexpected device map dtypes')

        parameters = self._parameters(
            controller, grid, model, start, initial, progress
        )
        rotating = not len(prepared[2])
        if rotating and not hasattr(self.lib,'sn_dwa_supports_rotation'):
            raise RuntimeError('Rebuild the DWA CUDA library for rotation references')
        if not rotating and len(prepared[3]) != len(path):
            raise ValueError(
                'GPU DWA requires the same nondegenerate reference segments as the CPU adapter'
            )
        people = (np.c_[model.xy, model.velocity] if model.tracks else
                  np.empty((0, 4), np.float64))
        pairs = np.asarray(
            model.group_pairs(groups) if model.tracks else [], np.int32
        ).reshape(-1, 2)
        social = np.array([model.config[key] for key in (
            'personal_front_m', 'personal_back_m', 'personal_side_m',
            'individual_cost', 'group_cost', 'group_distance_m',
        )], np.float64)
        targets, reference, arc, state, footprint, people, pairs, social = self._upload_inputs([
            ('dwa_targets', choices),
            ('dwa_reference', path),
            ('dwa_reference_arc', np.zeros(len(path)) if rotating else prepared[3]),
            ('dwa_state', parameters),
            ('dwa_footprint', controller.geometry.footprint),
            ('dwa_people', people),
            ('dwa_groups', pairs),
            ('dwa_social', social),
        ])

        count = len(choices)
        settings = controller.settings
        steps = max(2, int(np.ceil(settings['prediction_s'] / settings['sample_s'])))
        self.poses = self._buffer('poses', (count, steps + 1, 4), torch.float64)
        output = self._buffer('output', (count * _SCORE_BYTES,), torch.uint8)
        score_bytes = count * _SCORE_BYTES
        scores = output[:score_bytes].view(torch.float64)

        self._kernel_begin.record()
        code = self.lib.sn_dwa_predict(
            _pointer(targets), count, _pointer(reference), _pointer(arc), len(path),
            _pointer(state), _pointer(occupancy), _pointer(clearance), *occupancy.shape,
            _pointer(footprint), len(controller.geometry.footprint),
            _pointer(people), len(people), _pointer(pairs), len(pairs), _pointer(social),
            _pointer(self.poses), _pointer(scores),
            steps, torch.cuda.current_stream(self.device).cuda_stream,
        )
        if code:
            raise RuntimeError('CUDA DWA kernel failed: ' + str(code))
        self._kernel_end.record()

        result = self._download_scores(output, count)
        finite = np.isfinite(result)
        ranked = [(float(score), index, None)
                  for index, score in enumerate(result) if finite[index]]
        return ranked, int(np.count_nonzero(~finite))

    def trajectory(self, j):
        out = self.poses[j].cpu().numpy()
        if self.metrics:
            self.metrics.transfer('d2h', out.nbytes, 'dwa_selected_prediction')
        return out
