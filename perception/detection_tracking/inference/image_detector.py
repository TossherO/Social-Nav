"""Explicit PyTorch/TensorRT YOLO backend. No dependency/model auto-installation."""
import importlib.metadata
import json
import os
from pathlib import Path
import time

import numpy as np

from .rgb_cache import file_digest


def bind_device(device):
    """Freeze one logical GPU before Ultralytics can rewrite CUDA_VISIBLE_DEVICES."""
    import torch
    requested = str(device).replace('cuda:','')
    if requested == 'cpu':return 'cpu'
    if not requested.isdigit():raise ValueError('图像检测device须为GPU索引或cpu')
    if torch.cuda.is_initialized():
        if int(requested)>=torch.cuda.device_count():raise ValueError('图像检测GPU不在运行时的可见设备范围内')
        return 'cuda:'+requested
    else:
        visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        choices = visible.split(',') if visible is not None else None
        chosen = choices[int(requested)] if choices is not None else requested
        os.environ['CUDA_VISIBLE_DEVICES'] = chosen
        torch.cuda.init()
        torch.cuda.set_device(0)
    return 'cuda:0'


class ImageDetector:
    def __init__(self,config,device,*,metrics=None):
        self.metrics=metrics
        self.config = dict(config)
        backend = config['backend']
        if backend not in ('pytorch','tensorrt'):
            raise ValueError('图像检测后端必须显式选择pytorch或tensorrt')
        if config['precision'] not in ('fp32','fp16') or (backend=='tensorrt' and config['precision']!='fp16'):
            raise ValueError('图像检测精度须为fp32/fp16，当前TensorRT引擎要求fp16')
        path = Path(config['engine'] if backend=='tensorrt' else config['weights']).resolve()
        if not path.is_file():raise ValueError('图像模型不存在；请先显式准备：'+str(path))
        if path.suffix != ('.engine' if backend=='tensorrt' else '.pt'):
            raise ValueError('图像模型文件与所选后端不符')
        os.environ['YOLO_AUTOINSTALL'] = 'false'
        os.environ.setdefault('YOLO_CONFIG_DIR','/tmp/social-nav-ultralytics')
        self.device = bind_device(device)
        from ultralytics import YOLO
        self.model = YOLO(str(path),task='detect')
        self.metadata = dict(backend=backend,path=str(path),sha256=file_digest(path),
            ultralytics=importlib.metadata.version('ultralytics-opencv-headless'),
            torch=importlib.metadata.version('torch'),torchvision=importlib.metadata.version('torchvision'),
            imgsz=config['imgsz'],precision=config['precision'],confidence=config['confidence'],
            nms_iou=config['nms_iou'],classes=['person'],rect=False,batch=1)
        if self.device != 'cpu':
            import torch
            properties = torch.cuda.get_device_properties(self.device)
            self.metadata['gpu_uuid'] = str(properties.uuid)
            self.metadata['gpu_name'] = properties.name
        if backend=='tensorrt':
            import tensorrt
            from packaging.version import Version
            self.metadata['tensorrt'] = tensorrt.__version__
            self.metadata['tensorrt_distribution'] = importlib.metadata.version('tensorrt-cu13')
            build=json.loads(path.with_suffix('.engine.json').read_text())
            if (build['engine_sha256'] != self.metadata['sha256'] or build['imgsz'] != config['imgsz'] or
                    build['precision'] != config['precision'] or build['batch'] != 1 or
                    Version(build['tensorrt']).base_version != Version(tensorrt.__version__).base_version):
                raise ValueError('TensorRT引擎与构建来源、运行版本或推理配置不匹配')
            self.metadata['engine_build'] = build

    def detect(self,rgb):
        import torch
        rgb = np.asarray(rgb)
        if rgb.dtype != np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3:
            raise ValueError('YOLO输入必须为原图RGB uint8 HWC')
        start = time.perf_counter()
        # Ultralytics ndarray source is BGR; never silently swap the channel meaning.
        output = self.model.predict(source=np.ascontiguousarray(rgb[:,:,::-1]),
            device=self.device,imgsz=self.config['imgsz'],rect=False,batch=1,
            half=self.config['precision']=='fp16',conf=self.config['confidence'],
            iou=self.config['nms_iou'],classes=[0],verbose=False,save=False)[0]
        if output.names.get(0) != 'person':raise ValueError('模型类别0不是person，不能套用人物类别索引')
        detections = [dict(box=box[:4].tolist(),score=float(box[4]),class_id=int(box[5]))
                      for box in output.boxes.data.detach().cpu().numpy()]
        if self.metrics is not None and not self.device.startswith('cpu'):
            self.metrics.transfer('h2d',self.config['imgsz']**2*3,'image_network_uint8_input')
            self.metrics.transfer('d2h',output.boxes.data.numel()*output.boxes.data.element_size(),'image_boxes')
        return detections,time.perf_counter()-start
