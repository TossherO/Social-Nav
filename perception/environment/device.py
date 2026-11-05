"""Sealed GPU observations; persisted GridSnapshot remains a CPU-only contract."""

from dataclasses import dataclass
import copy
import uuid
import numpy as np

from .layers import GridSnapshot


@dataclass(frozen=True)
class DeviceGridSnapshot:
    metadata: dict
    _layers: dict
    ready: object
    owner: str

    @classmethod
    def seal(cls, metadata, layers):
        import torch

        if not layers or any(
            not t.is_cuda or tuple(t.shape) != tuple(metadata['shape'])
            for t in layers.values()
        ):
            raise ValueError('Device layers must share the explicit grid shape')
        devices = {t.device for t in layers.values()}
        if len(devices) != 1:
            raise ValueError('A device snapshot belongs to one GPU')
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(next(iter(devices))))
        return cls(copy.deepcopy(metadata), dict(layers), ready, uuid.uuid4().hex)

    def consume(self, stream=None):
        """Trusted same-process consumers only; no producer may mutate sealed tensors."""
        import torch

        stream = stream or torch.cuda.current_stream(next(iter(self._layers.values())).device)
        stream.wait_event(self.ready)
        for tensor in self._layers.values():
            tensor.record_stream(stream)
        return dict(self._layers)


    def materialize(self, *, consumer, fields=None, metrics=None):
        """Export only the named consumer's fields and return read-only CPU arrays."""
        if not consumer:
            raise ValueError('Host export needs a named consumer')
        selected = tuple(self._layers if fields is None else fields)
        if (not selected or len(set(selected)) != len(selected) or
                any(k not in self._layers for k in selected)):
            raise ValueError('Invalid device layer selection')
        import torch
        layers = self.consume()
        values=[layers[name].detach().contiguous().view(torch.uint8).reshape(-1) for name in selected]
        packed=torch.cat(values).cpu().numpy()
        host={};offset=0
        for name,value in zip(selected,values):
            # Resolve dtype without another device download.
            dtype=np.dtype(str(layers[name].dtype).removeprefix('torch.'))
            size=value.numel()
            host[name]=np.frombuffer(packed,dtype=dtype,count=layers[name].numel(),offset=offset).reshape(layers[name].shape)
            offset+=size
        if metrics is not None:
            metrics.transfer('d2h',sum(array.nbytes for array in host.values()),consumer)
            metrics.transfer('d2d',packed.nbytes,'device_export_pack')
        for array in host.values():
            array.flags.writeable = False
        return GridSnapshot(copy.deepcopy(self.metadata), host)

    @property
    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in self._layers.values())
