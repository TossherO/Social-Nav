"""Shared mixed-precision helpers for training and inference."""

from contextlib import nullcontext

import torch


SUPPORTED_PRECISIONS = ('fp32', 'fp16', 'bf16')
_ALIASES = {
    'float': 'fp32',
    'float32': 'fp32',
    'half': 'fp16',
    'float16': 'fp16',
    'bfloat16': 'bf16',
}


def normalize_precision(precision):
    """Return a canonical precision name and reject ambiguous values."""
    if precision is None:
        return 'fp32'
    value = str(precision).strip().lower()
    value = _ALIASES.get(value, value)
    if value not in SUPPORTED_PRECISIONS:
        supported = ', '.join(SUPPORTED_PRECISIONS)
        raise ValueError(f'Unsupported precision {precision!r}; choose one of: {supported}')
    return value


def resolve_precision(cli_precision=None, config_precision=None, use_amp=False):
    """Resolve CLI/config precision while preserving the old ``--use_amp`` flag."""
    if use_amp:
        if cli_precision is not None and normalize_precision(cli_precision) != 'fp16':
            raise ValueError('--use_amp is an alias for --precision fp16 and cannot be combined '
                             'with another precision')
        return 'fp16'
    if cli_precision is not None:
        return normalize_precision(cli_precision)
    return normalize_precision(config_precision)


def precision_dtype(precision):
    """Map a canonical precision to its autocast dtype."""
    precision = normalize_precision(precision)
    if precision == 'fp16':
        return torch.float16
    if precision == 'bf16':
        return torch.bfloat16
    return torch.float32


def autocast_context(precision, device_type='cuda'):
    """Return an autocast context; FP32 uses a zero-overhead null context."""
    precision = normalize_precision(precision)
    if precision == 'fp32':
        return nullcontext()
    return torch.autocast(
        device_type=device_type,
        dtype=precision_dtype(precision),
        enabled=True,
    )


def create_grad_scaler(precision, device_type='cuda', init_scale=32.0):
    """Create the persistent scaler used by mixed-precision training.

    Native BF16 normally does not require scaling.  This project uses spconv
    2.3.x, whose legacy autocast wrapper executes sparse convolutions in FP16
    even inside a BF16 autocast region, so scaling is kept enabled for both
    mixed-precision modes.
    """
    precision = normalize_precision(precision)
    return torch.amp.GradScaler(
        device_type,
        init_scale=init_scale,
        enabled=(precision != 'fp32'),
    )


def validate_precision_support(precision, device=None):
    """Fail early when the selected CUDA precision is unavailable."""
    precision = normalize_precision(precision)
    if precision == 'fp32':
        return
    if not torch.cuda.is_available():
        raise RuntimeError(f'{precision} was requested, but CUDA is not available')
    if precision == 'bf16' and not torch.cuda.is_bf16_supported():
        device_name = torch.cuda.get_device_name(device) if device is not None else 'current GPU'
        raise RuntimeError(f'bf16 is not supported by {device_name}')


def fp32_autocast_disabled():
    """Disable CUDA autocast around FP32-only custom CUDA extensions."""
    return torch.autocast(device_type='cuda', enabled=False)
