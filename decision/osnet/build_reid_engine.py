"""Build a local OSNet TensorRT engine from the frozen ONNX asset."""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

from .reid_trt import (
    FEATURE_COUNT, IMAGE_SHAPE, INPUT_NAME, OUTPUT_NAME, cuda_device, metadata_path,
)


def build(onnx, engine, *, batch_size=8, onnx_sha256=None, device=None):
    if onnx.suffix != '.onnx' or engine.suffix != '.engine':
        raise ValueError('OSNet engine build requires an .onnx input and an .engine output')
    if batch_size < 2:
        raise ValueError('OSNet dynamic maximum batch size must be at least 2')
    sidecar = metadata_path(engine)
    for path in (engine, sidecar):
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)

    model = onnx.read_bytes()
    actual_onnx_sha256 = hashlib.sha256(model).hexdigest()
    if onnx_sha256 is not None and actual_onnx_sha256 != onnx_sha256:
        raise ValueError('OSNet ONNX SHA256 mismatch')

    import tensorrt as trt
    import torch

    device = cuda_device(torch, device)
    logger = trt.Logger(trt.Logger.WARNING)
    batches = (1, 2, batch_size)
    with torch.cuda.device(device):
        builder = trt.Builder(logger)
        network = builder.create_network(0)
        parser = trt.OnnxParser(network, logger)
        if not parser.parse(model):
            errors = '\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise ValueError(f'Could not parse OSNet ONNX:\n{errors}')
        if network.num_inputs != 1 or network.num_outputs != 1:
            raise ValueError('OSNet ONNX must have one input and one output')
        for tensor, name, shape in ((network.get_input(0), INPUT_NAME, (-1, *IMAGE_SHAPE)),
                                    (network.get_output(0), OUTPUT_NAME, (-1, FEATURE_COUNT))):
            if tensor.name != name or tuple(tensor.shape) != shape or tensor.dtype != trt.float32:
                raise ValueError(f'Unexpected OSNet ONNX tensor: {tensor.name}')
        config = builder.create_builder_config()
        config.clear_flag(trt.BuilderFlag.TF32)
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 512 * 1024 ** 2)
        profile = builder.create_optimization_profile()
        profile.set_shape(INPUT_NAME, *((batch, *IMAGE_SHAPE) for batch in batches))
        if config.add_optimization_profile(profile) != 0:
            raise ValueError('Could not set OSNet TensorRT batch profile')
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError('Could not build OSNet TensorRT engine')
        plan = bytes(serialized)

    metadata = {
        'onnx_sha256': actual_onnx_sha256,
        'engine_sha256': hashlib.sha256(plan).hexdigest(),
        'precision': 'fp32',
        'batch_profile': list(batches),
        'tensorrt_version': trt.__version__,
        'cuda_build_version': torch.version.cuda,
        'gpu_name': torch.cuda.get_device_name(device),
        'compute_capability': list(torch.cuda.get_device_capability(device)),
    }
    engine.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=engine.parent) as directory:
        staged_plan = Path(directory) / engine.name
        staged_sidecar = metadata_path(staged_plan)
        staged_plan.write_bytes(plan)
        staged_sidecar.write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
        engine.hardlink_to(staged_plan)
        try:
            sidecar.hardlink_to(staged_sidecar)
        except OSError:
            engine.unlink()
            raise
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('onnx', type=Path)
    parser.add_argument('engine', type=Path)
    parser.add_argument('--batch-size', type=int, default=8, help='maximum candidates per GPU execution')
    parser.add_argument('--onnx-sha256', help='expected source ONNX SHA256')
    parser.add_argument('--device', help='Torch CUDA device, defaulting to the current visible device')
    args = parser.parse_args()
    metadata = build(args.onnx, args.engine, batch_size=args.batch_size,
                     onnx_sha256=args.onnx_sha256, device=args.device)
    print(metadata['engine_sha256'], args.engine)


if __name__ == '__main__':
    main()
