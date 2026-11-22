"""TensorRT CUDA bindings for the local OSNet encoder."""

import json
import re


INPUT_NAME = 'rgb'
OUTPUT_NAME = 'embedding'
IMAGE_SHAPE = (3, 256, 128)
FEATURE_COUNT = 512


def metadata_path(path):
    return path.with_suffix('.engine.json')


def cuda_device(torch, requested):
    device = torch.device(requested if requested is not None else 'cuda')
    if device.type != 'cuda':
        raise ValueError('OSNet TensorRT requires a CUDA device')
    if not torch.cuda.is_available():
        raise RuntimeError('OSNet TensorRT requires available CUDA')
    index = torch.cuda.current_device() if device.index is None else device.index
    if index >= torch.cuda.device_count():
        raise ValueError(f'OSNet CUDA device is unavailable: cuda:{index}')
    return torch.device('cuda', index)


class TensorRtOsnet:
    def __init__(self, path, model, sha256, device):
        try:
            metadata = json.loads(metadata_path(path).read_text(encoding='utf-8'))
        except (OSError, ValueError) as error:
            raise ValueError('OSNet TensorRT requires readable engine metadata') from error
        onnx_sha256 = metadata.get('onnx_sha256') if isinstance(metadata, dict) else None
        if (not isinstance(metadata, dict) or metadata.get('engine_sha256') != sha256 or
                not isinstance(onnx_sha256, str) or not re.fullmatch(r'[0-9a-f]{64}', onnx_sha256) or
                metadata.get('precision') != 'fp32'):
            raise ValueError('OSNet TensorRT engine metadata does not match the asset')

        import tensorrt as trt
        import torch

        self.device = cuda_device(torch, device)
        capability = torch.cuda.get_device_capability(self.device)
        if (metadata.get('tensorrt_version') != trt.__version__ or
                metadata.get('compute_capability') != list(capability) or
                metadata.get('gpu_name') != torch.cuda.get_device_name(self.device)):
            raise RuntimeError('OSNet TensorRT engine requires the build TensorRT version and GPU model')
        self.onnx_sha256 = onnx_sha256
        self.tensorrt_version = trt.__version__
        self.torch = torch
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        with torch.cuda.device(self.device):
            self.engine = self.runtime.deserialize_cuda_engine(model)
            if self.engine is None:
                raise RuntimeError('Could not deserialize OSNet TensorRT engine')
            profile = self._check_bindings(trt)
            if metadata.get('batch_profile') != [shape[0] for shape in profile]:
                raise ValueError('OSNet TensorRT metadata batch profile does not match the engine')
            self.max_batch_size = profile[-1][0]
            self.context = self.engine.create_execution_context()
            if self.context is None:
                raise RuntimeError('Could not create OSNet TensorRT execution context')
            self.stream = torch.cuda.Stream(device=self.device)

    def _check_bindings(self, trt):
        engine = self.engine
        if (engine.num_io_tensors != 2 or engine.num_optimization_profiles != 1 or
                {engine.get_tensor_name(i) for i in range(2)} != {INPUT_NAME, OUTPUT_NAME}):
            raise ValueError('Unexpected OSNet TensorRT bindings or optimization profile')
        for name, mode, shape in ((INPUT_NAME, trt.TensorIOMode.INPUT, (-1, *IMAGE_SHAPE)),
                                  (OUTPUT_NAME, trt.TensorIOMode.OUTPUT, (-1, FEATURE_COUNT))):
            if (engine.get_tensor_mode(name) != mode or tuple(engine.get_tensor_shape(name)) != shape or
                    engine.get_tensor_dtype(name) != trt.float32 or
                    engine.get_tensor_location(name) != trt.TensorLocation.DEVICE or
                    engine.get_tensor_format(name) != trt.TensorFormat.LINEAR):
                raise ValueError(f'Unexpected OSNet TensorRT tensor binding: {name}')
        profile = [tuple(shape) for shape in engine.get_tensor_profile_shape(INPUT_NAME, 0)]
        if (len(profile) != 3 or any(shape[1:] != IMAGE_SHAPE for shape in profile) or
                not 1 == profile[0][0] <= profile[1][0] <= profile[2][0]):
            raise ValueError('Unexpected OSNet TensorRT batch profile')
        return profile

    def __call__(self, images):
        torch = self.torch
        context = self.context
        shape = (len(images), FEATURE_COUNT)
        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            if (not context.set_input_shape(INPUT_NAME, images.shape) or
                    tuple(context.get_tensor_shape(OUTPUT_NAME)) != shape):
                raise ValueError('Unexpected OSNet TensorRT batch shape')
            inputs = torch.from_numpy(images).to(self.device)
            outputs = torch.empty(shape, device=self.device, dtype=torch.float32)
            if (not context.set_tensor_address(INPUT_NAME, inputs.data_ptr()) or
                    not context.set_tensor_address(OUTPUT_NAME, outputs.data_ptr()) or
                    not context.execute_async_v3(self.stream.cuda_stream)):
                raise RuntimeError('OSNet TensorRT inference failed')
            # The blocking copy waits for this stream's inference before returning.
            return outputs.cpu().numpy()
