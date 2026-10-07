from contextlib import contextmanager
from typing import Set

import spconv
if float(spconv.__version__[2:]) >= 2.2:
    spconv.constants.SPCONV_USE_DIRECT_TABLE = False

try:
    import spconv.pytorch as spconv
except:
    import spconv as spconv

import torch
import torch.nn as nn


def find_all_spconv_keys(model: nn.Module, prefix="") -> Set[str]:
    """
    Finds all spconv keys that need to have weight's transposed
    """
    found_keys: Set[str] = set()
    for name, child in model.named_children():
        new_prefix = f"{prefix}.{name}" if prefix != "" else name

        if isinstance(child, spconv.conv.SparseConvolution):
            new_prefix = f"{new_prefix}.weight"
            found_keys.add(new_prefix)

        found_keys.update(find_all_spconv_keys(child, prefix=new_prefix))

    return found_keys


def replace_feature(out, new_features):
    if "replace_feature" in out.__dir__():
        # spconv 2.x behaviour
        return out.replace_feature(new_features)
    else:
        out.features = new_features
        return out


@contextmanager
def spconv_autocast_inference_fallback(module: nn.Module):
    """Use spconv's training kernel selector for autocast inference.

    spconv 2.3.x can fail to find an inference implicit-GEMM algorithm for
    some FP16 channel layouts (notably the 16 -> 16 VoxelRCNN stem).  Its
    training kernel path supports those layouts.  Only sparse convolution
    modules are toggled; BatchNorm and the enclosing detector remain in eval
    mode, and the caller's no-grad/inference context still controls autograd.
    """
    autocast_enabled = torch.is_autocast_enabled('cuda')
    if not autocast_enabled or module.training:
        yield
        return

    sparse_convs = [
        child for child in module.modules()
        if isinstance(child, spconv.conv.SparseConvolution)
    ]
    training_states = [child.training for child in sparse_convs]
    try:
        for child in sparse_convs:
            child.training = True
        yield
    finally:
        for child, training in zip(sparse_convs, training_states):
            child.training = training
