from .centerpoint import CenterPoint
from .voxel_rcnn import VoxelRCNN


def build_model(model_cfg, num_class, dataset):
    if model_cfg.NAME == 'CenterPoint':
        return CenterPoint(model_cfg, num_class, dataset)
    elif model_cfg.NAME == 'VoxelRCNN':
        return VoxelRCNN(model_cfg, num_class, dataset)
    else:
        raise NotImplementedError(f"Unknown model: {model_cfg.NAME}")


def load_data_to_gpu(batch_dict):
    import numpy as np
    import torch
    for key, val in batch_dict.items():
        if not isinstance(val, np.ndarray):
            continue
        elif key in ['frame_id', 'metadata', 'calib', 'image_paths', 'ori_shape', 'img_process_infos']:
            continue
        elif key in ['images']:
            continue
        elif key in ['image_shape']:
            batch_dict[key] = torch.from_numpy(val).int().cuda()
        else:
            batch_dict[key] = torch.from_numpy(val).float().cuda()
