# SimpleDet3D 代码框架

## 项目概述

SimpleDet3D 是一个简化的 3D 目标检测训练框架，基于 OpenPCDet 精简而来。仅支持两个模型（CenterPoint、VoxelRCNN）和两个数据集（CODa、Wheelchair）。模型采用直接构建方式（无注册机制），项目不安装到 Python 环境，直接从源码目录运行。

**支持的数据集和类别：**

| 数据集 | 类别 | 标注 | 说明 |
|--------|------|------|------|
| CODa | Car, Pedestrian, Cyclist | 有 | 原始 13 类映射为 3 类 |
| Wheelchair | Pedestrian | 人工 BEV 中心 + 伪框/伪点 | 支持弱监督迁移与推理 |

**支持的模型：**

| 模型 | 类型 | 参数量 | VFE |
|------|------|--------|-----|
| CenterPoint | 单阶段 anchor-free | ~4.2M | MeanVFE |
| VoxelRCNN | 两阶段（RPN + RoI 细化） | ~12.4M | DynamicVoxelVFE |

---

## 目录结构

```
simple_det3d/
├── setup.py                         # CUDA 扩展编译 (python setup.py build_ext --inplace)
├── compile.sh                       # 一键编译
├── train_8gpu.sh                    # 单个实验的单机 8 卡启动脚本
├── train_wheelchair_3stage.sh       # Wheelchair 三阶段训练串联、续训与 checkpoint 传递
├── requirements.txt
├── configs/                         # YAML 配置文件（8 个，支持继承）
│   ├── centerpoint_coda.yaml
│   ├── centerpoint_wheelchair.yaml
│   ├── voxelrcnn_coda.yaml
│   ├── voxelrcnn_coda_16line.yaml
│   ├── voxelrcnn_wheelchair.yaml
│   ├── voxelrcnn_wheelchair_weak_stage1.yaml
│   ├── voxelrcnn_wheelchair_weak_stage2.yaml
│   └── voxelrcnn_wheelchair_weak_stage3.yaml
├── simple_det3d/                    # 核心库
│   ├── ops/                         # CUDA 扩展（Python 封装 + C++/CUDA 源码）
│   │   ├── iou3d_nms/               #   3D IoU 计算 + 旋转 NMS
│   │   │   └── iou3d_nms_utils.py   #     7 个函数（IoU/NMS），封装 iou3d_nms_cuda
│   │   ├── roiaware_pool3d/         #   RoI 感知 3D 点云池化
│   │   │   └── roiaware_pool3d_utils.py  # 3 个函数/类（点-盒子归属、RoI 网格池化）
│   │   └── pointnet2/
│   │       ├── pointnet2_stack/     #   Stack 顺序 PointNet++（VoxelRCNN ROI 池化用）
│   │       │   ├── pointnet2_utils.py      # 9 个操作（ball query, FPS, 3NN, vector pool 等）
│   │       │   ├── voxel_query_utils.py    # 2 个类（VoxelQuery, VoxelQueryAndGrouping）
│   │       │   └── voxel_pool_modules.py   # NeighborVoxelSAModuleMSG（多尺度体素 SA）
│   │       └── pointnet2_batch/     #   Batch 顺序 PointNet++（固定 N 输入）
│   │           └── pointnet2_utils.py      # 8 个操作（FPS, Ball Query, Group, 3NN 等）
│   ├── models/                      # 模型实现（直接构建，无注册机制）
│   │   ├── __init__.py              #   build_model(), load_data_to_gpu()
│   │   ├── centerpoint.py           #   class CenterPoint
│   │   ├── voxel_rcnn.py            #   class VoxelRCNN
│   │   ├── vfe.py                   #   class MeanVFE, DynamicVoxelVFE, PFNLayerV2
│   │   ├── backbones_3d.py          #   class VoxelResBackBone8x, SparseBasicBlock
│   │   ├── backbones_2d.py          #   class HeightCompression, BaseBEVBackbone
│   │   ├── center_head.py           #   class SeparateHead, CenterHead
│   │   ├── voxel_rcnn_head.py       #   class VoxelRCNNHead, ProposalTargetLayer
│   │   └── model_utils.py           #   NMS 函数、heatmap 解码、Gaussian 工具（13 个函数）
│   ├── datasets/                    # 数据集
│   │   ├── coda_dataset.py          #   class CodaDataset（支持完整框和弱监督掩码）
│   │   ├── mixed_domain_dataset.py  #   CODa/Wheelchair 固定比例混合采样
│   │   ├── weak_label_utils.py      #   Wheelchair 伪框合法性/点数/时序过滤
│   │   ├── data_augmentor.py        #   class DataAugmentor, DataBaseSampler
│   │   └── data_processor.py        #   class PointFeatureEncoder, DataProcessor, VoxelGeneratorWrapper
│   ├── utils/                       # 工具函数
│   │   ├── config.py                #   load_config() YAML→EasyDict，支持 _BASE_CONFIG_ 继承
│   │   ├── common_utils.py          #   通用工具（numpy→torch、分布式、日志、随机种子等 19 个函数）
│   │   ├── box_utils.py             #   3D box 几何（corners、过滤、BEV、DIoU 等 5 个函数）
│   │   ├── box_coder_utils.py       #   box 编解码（4 个 coder 类）
│   │   ├── loss_utils.py            #   损失函数（12 个类 + 9 个函数）
│   │   ├── precision_utils.py       #   fp32/fp16/bf16、autocast 与 GradScaler 统一入口
│   │   └── spconv_utils.py          #   spconv 版本兼容、权重 key 查找、特征替换
│   └── train/                       # 训练模块
│       ├── train_utils.py           #   train_one_epoch(), train_model(), model_fn_decorator()
│       └── optimization/            #   优化器和学习率调度器
│           ├── __init__.py          #     build_optimizer(), build_scheduler()
│           ├── fastai_optim.py      #     OptimWrapper, FastAIMixedOptim（真实权重衰减+FP16）
│           └── learning_schedules_fastai.py  # OneCycle, CosineAnnealing, CosineWarmupLR
└── tools/                           # 入口脚本
    ├── train.py                     # 训练入口（含训练后自动评估）
    ├── test.py                      # 推理/评估入口（--infer_time, --save_vis）
    ├── generate_coda_info.py        # CODa 原始数据 → pkl 信息文件
    ├── generate_wheelchair_info.py  # Wheelchair 原始数据 → pkl 信息文件（无标注）
    ├── prepare_wheelchair_weak_data.py # 不完整标注 → 分级弱监督 pkl
    ├── visualize_wheelchair_weak_labels.py # 人工中心/伪框 BEV 核验
    ├── visualize_wheelchair_predictions.py # 模型推理 + Wheelchair 双视图 BEV 人工审计
    ├── compare_wheelchair_predictions.py   # 单帧/四帧预测与弱真值三栏对比
    ├── create_wheelchair_comparison_video.py # 完整序列三栏 BEV MP4
    └── generate_gt_database.py      # 生成 GT 采样数据库（用于 gt_sampling）
```

---

## 数据流

### 完整数据 Pipeline

```
[.pkl 信息文件]  +  [.bin 点云文件]
       |                    |
include_coda_data()   get_lidar_with_sweeps()
       |                    |
       v                    v
  self.infos       points (N, 5)  # [x, y, z, intensity, time]
       |                    |
       +--- __getitem__ ---+
                |
                v
        input_dict {points, frame_id, gt_boxes, gt_names, lidar2global}
                |
                v
        prepare_data(input_dict)
                |
    ┌───────────┼───────────┐
    │ [训练时]                │
    ▼                        │
DataAugmentor.forward()      │
    │                        │
    ├── DataBaseSampler()    │  # gt_sampling（仅 CODa，从预建数据库复制物体）
    ├── random_world_flip()  │  # 随机沿 X/Y 轴翻转（50% 概率）
    ├── random_world_rotation() │  # 随机绕 Z 轴旋转
    ├── random_world_scaling()  │  # 随机均匀缩放
    └── random_world_translation() │  # 高斯噪声平移（仅 Wheelchair）
    │                        │
    ▼                        │
set_lidar_aug_matrix()       │  # 记录从 flip/rotation/scale/translation 合成的 4×4 增广矩阵
    │                        │
过滤 gt_boxes（按 class_names，编码 class_id 到最后一列 → (M, 8)）
    │                        │
    ▼                        ▼
PointFeatureEncoder.forward()  # absolute_coordinates_encoding, use_lead_xyz=True
    │                         # 输出: points (N, C) [x, y, z, intensity, time]
    ▼
DataProcessor.forward()
    │
    ├── mask_points_and_boxes_outside_range()  # 按 point_cloud_range 裁剪
    ├── shuffle_points()                       # 随机打乱点顺序
    ├── transform_points_to_voxels()            # VoxelGeneratorWrapper (spconv)
    │   └── voxels (M, max_points, C), voxel_coords (M, 3), voxel_num_points (M,)
    └── sample_points()                        # 固定数量点采样（深度感知：优先保留远处点 >40m）
                |
                v
        data_dict {voxels, voxel_coords, voxel_num_points, gt_boxes, ...}
                |
                v
        collate_batch(batch_list)  # 静态方法，变长数据拼接 + 零填充
                |
                ▼
        batch_dict（numpy arrays）
                |
                ▼
        load_data_to_gpu(batch_dict)  # numpy → torch.float().cuda()
                |                      # 跳过 frame_id, metadata, calib, images 等
                ▼
        model(batch_dict)            # 遍历 self.module_list 中的每个模块
```

### 多帧 Sweep 拼接

`CodaDataset.get_lidar_with_sweeps()` 将历史帧点云通过位姿变换拼接到当前帧，增加点云密度：

1. 加载主帧点云并去除 ego 点（半径 1m），时间特征设为 0
2. 过滤非正、非有限、超过 `MAX_SWEEP_TIME_LAG` 或变换矩阵无效的历史帧
3. 选取 sweep 帧：`MAX_SWEEPS` 表示含当前帧的总帧数；`near` 按真实时间差选最近帧，`random` 从有效候选中随机抽样
4. 对每帧 sweep：加载 `.bin` → 去除 ego 点 → 乘以 `transform_matrix`（相对当前帧的位姿变换）
5. 拼接所有点云和时间通道 → `(N, 5)` 数组

### 类别平衡重采样

`CodaDataset.balanced_infos_resampling()` 基于类别频率做平衡重采样（参考 CBGS 论文, arXiv:1908.09492）。仅在 CODa 配置中启用 `BALANCED_RESAMPLING: true`。

### 空 GT 安全机制

在 `prepare_data` 中，如果训练样本经过所有过滤和增强后没有剩余 GT box，会随机重新采样另一个索引（`np.random.randint(len(self))`），防止模型见到空场景。

---

## 数据处理模块

### CodaDataset (`simple_det3d/datasets/coda_dataset.py`)

核心数据集类，同时支持 CODa 和 Wheelchair 数据集。

```python
class CodaDataset(torch.utils.data.Dataset):
    # 属性（在 __init__ 中设置，供模型构建使用）
    point_cloud_range: np.ndarray   # (6,) [xmin, ymin, zmin, xmax, ymax, zmax]
    voxel_size: list                # [vx, vy, vz]
    grid_size: list                 # [gx, gy, gz]（由 voxel_size 和 point_cloud_range 计算）
    class_names: list               # 类别名
    point_feature_encoder           # PointFeatureEncoder

    # 核心方法
    def __init__(self, dataset_cfg, class_names, training, root_path, logger)
    def __getitem__(self, index) -> dict
    def prepare_data(self, data_dict) -> dict
    @staticmethod
    def collate_batch(batch_list, _unused=False) -> dict
    def evaluation(self, det_annos, class_names, **kwargs) -> (str, dict)
    def generate_prediction_dicts(self, batch_dict, pred_dicts, class_names, output_path=None) -> list
    def create_groundtruth_database(self, ...)  # 离线生成 GT 数据库
```

**`collate_batch` 详细逻辑：**

| 字段 | 处理方式 |
|------|----------|
| `voxels`, `voxel_num_points` | 沿 dim=0 拼接 |
| `voxel_coords` | 第一列加 batch 索引后拼接 |
| `gt_boxes`, `roi_boxes` | 零填充到 batch 内最大数量，shape `(B, max_num, box_dim)` |
| `roi_scores`, `roi_labels` | 零填充到最大数量 |
| 其他标量/字符串 key | 直接 `np.stack` |

**`evaluation` 实现：**

自包含的 41-point interpolated AP 计算（不依赖外部库）：
- IoU 阈值: `{'Car': 0.7, 'Pedestrian': 0.5, 'Cyclist': 0.5}`
- 同时计算 AP3D（3D IoU）和 APBEV（BEV IoU）
- 返回 `(result_str, result_dict)`

### DataAugmentor (`simple_det3d/datasets/data_augmentor.py`)

支持五种数据增强，按配置顺序依次应用。所有增强对含速度的 box（dim>7）做相应变换。

| 增强类型 | 方法 | 说明 |
|----------|------|------|
| `gt_sampling` | `DataBaseSampler.__call__()` | 从 GT 数据库采样物体复制到当前场景（仅 CODa，BEV IoU 防重叠） |
| `random_world_flip` | `random_flip_along_x/along_y` | 沿 X/Y 轴翻转（50% 概率），同时翻转速度分量 |
| `random_world_rotation` | `global_rotation` | 绕 Z 轴随机旋转，同时旋转速度向量 |
| `random_world_scaling` | `global_scaling` | 随机均匀缩放（含 roi_boxes） |
| `random_world_translation` | `random_world_translation` | 高斯噪声平移（仅 Wheelchair，含 roi_boxes） |

**DataBaseSampler 采样逻辑：**
1. 对每个采样组（如 `Car:5, Pedestrian:5`），从预生成的 `.pkl` 数据库采样
2. 计算 BEV IoU，拒绝与已有 GT 或其他采样物体重叠的候选
3. 加载采样物体的点云 `.bin`，变换到当前场景
4. 从原始点云中移除采样物体范围内的点
5. 将采样物体点云和 box 拼接到当前场景

### DataProcessor (`simple_det3d/datasets/data_processor.py`)

采用两阶段初始化模式：构建时 `data_dict=None` 返回 `functools.partial`，运行时再传入实际数据。

```
DataProcessor.forward()
  ├── mask_points_and_boxes_outside_range  # 裁剪到 point_cloud_range
  ├── shuffle_points                       # 随机打乱点顺序
  ├── transform_points_to_voxels           # VoxelGeneratorWrapper.generate()
  │     spconv v1: VoxelGeneratorV2/VoxelGenerator
  │     spconv v2: cumm.tensorview → Point2VoxelCPU3d
  └── sample_points                       # 固定数量采样，优先保留远处点（>40m）
```

### PointFeatureEncoder

```python
def absolute_coordinates_encoding(self, points=None):
    # points=None → 返回特征维度
    # points  → 拼接 [x, y, z] + [intensity, time, ...] per used_feature_list
```

---

## 模型架构

### CenterPoint（单阶段 anchor-free 检测器）

```
batch_dict
  │
  ▼
MeanVFE.forward()
  输入: voxels (num_voxels, max_points_per_voxel, C)
  处理: 对每个 voxel 内所有点取均值（除以 clamp_min(voxel_num_points, 1)）
  输出: voxel_features (num_voxels, C)
  │
  ▼
VoxelResBackBone8x.forward()
  结构: 稀疏 3D 卷积 backbone，8x 下采样
    conv_input: SubMConv3d(1→16) + BN + ReLU
    conv1: 2× SparseBasicBlock(16, 16), stride=1
    conv2: SparseConv3d(stride=2, 16→32) + 2× SparseBasicBlock(32, 32)
    conv3: SparseConv3d(stride=2, 32→64) + 2× SparseBasicBlock(64, 64)
    conv4: SparseConv3d(stride=2, 64→128, pad=(0,1,1)) + 2× SparseBasicBlock(128, 128)
    conv_out: SparseConv3d(kernel=(3,1,1), stride=(2,1,1)), 128→128
  输出: encoded_spconv_tensor (C=128, stride=8), multi_scale_3d_features
  backbone_channels: {'x_conv1':16, 'x_conv2':32, 'x_conv3':64, 'x_conv4':128}
  │
  ▼
HeightCompression.forward()
  处理: 稀疏 tensor → .dense() → (N, C, D, H, W) → view(N, C*D, H, W)
  输出: spatial_features (B, C*D, H, W)
  │
  ▼
BaseBEVBackbone.forward()
  结构: 多级 2D 卷积 + 反卷积上采样（FPN 风格）
    blocks: ZeroPad2d(1)→Conv2d(stride)→BN→ReLU + Conv2d×N（逐级下采样）
    deblocks: ConvTranspose2d（上采样）
    多尺度特征沿 channel 维拼接
  输出: spatial_features_2d
  │
  ▼
CenterHead.forward()
  shared_conv: Conv2d(in, SHARED_CONV_CHANNEL, 3)→BN→ReLU
  SeparateHead（每 head 一组 task-specific 分支）:
    ├── hm_head:  Conv2d→BN→ReLU (×N) → Conv2d(out=num_classes)  # heatmap
    ├── box_head:  Conv2d→BN→ReLU (×N) → Conv2d(out=box_code_size) # 回归
    └── iou_head:  Conv2d→BN→ReLU (×N) → Conv2d(out=1)（可选）  # IoU 预测

  训练时:
    assign_targets() → 高斯热力图 + 回归目标
      1. GT box → heatmap 坐标（point_cloud_range, voxel_size, feature_map_stride）
      2. 计算高斯半径（gaussian_radius: 基于 box 尺寸和 min_overlap 的三次方程解取 min）
      3. draw_gaussian_to_heatmap: 在 heatmap 上绘制高斯峰值
      4. 编码回归目标: offset, z, log-dim, cos/sin rot, velocity（可选）
    get_loss():
      - hm_loss: FocalLossCenterNet (heatmap focal loss)，weight=cls_weight
      - loc_loss: RegLossCenterNet (L1 regression)，weight=loc_weight × code_weights
      - (可选) iou_loss: 预测 IoU 与 paired_boxes_iou3d_gpu 真值的 L1
      - (可选) iou_reg_loss: DIoU 回归正则化

  推理时:
    decode_bbox_from_heatmap():
      1. heatmap 取 top-K 峰值（max-pool NMS 预处理）
      2. 在峰值位置收集 offset, rotation, z, dim, velocity, iou
      3. 转换到世界坐标: feature_map_stride × voxel_size + point_cloud_range
      4. score_thresh + post_center_limit_range 过滤
      5. 可选 circle_nms（numba JIT 加速的贪心距离抑制）
      6. 返回 pred_boxes, pred_scores, pred_labels
  │
  ▼ (推理)
post_processing(): 收集最终预测，计算 recall（IoU_threshold ∈ RECALL_THRESH_LIST）
```

### VoxelRCNN（两阶段检测器）

第一阶段与 CenterPoint 共享相同 backbone 结构，关键差异如下。

**差异 1: DynamicVoxelVFE（替代 MeanVFE）**

```
输入: points (total_N, 6) [batch_idx, x, y, z, intensity, elongation]
  │
  1. 计算体素坐标: floor((points_xyz - point_cloud_range) / voxel_size)
  2. 过滤超出范围的点
  3. unique 聚合: batch_idx * scale_xyz + x * scale_yz + y * scale_z + z
  4. 按样本应用 DATA_PROCESSOR.MAX_NUMBER_OF_VOXELS:
     - train/test 分别使用对应上限（当前为 60000/80000）
     - 超限时按局部体素坐标哈希确定性选取，避免空间前缀偏置
     - 重映射逐点 voxel id；未超限时保持原路径
  5. 计算统计量:
     - f_cluster: 点相对于体素内平均位置的偏移
     - f_center: 点相对于体素中心（世界坐标）的偏移
  6. 构建特征向量:
     - absolute_xyz: [points[:, 1:], f_cluster, f_center]（+6 dims）
     - relative:     [points[:, 4:], f_cluster, f_center]（+3 dims）
     - 可选: |point_xyz|（距离特征，+1 dim）
  7. PFNLayerV2 堆叠:
     - 每层: Linear → BN → ReLU → scatter_max（按体素聚合）
     - 中间层: concat(per-point 特征, max-pooled 特征)（PointNet++ 风格）
     - 最后一层: 仅返回 max-pooled 特征
  输出: voxel_features, voxel_coords [batch_idx, z, y, x]
```

**差异 2: CenterHead 设置 `predict_boxes_when_training=True`**

训练时 CenterHead 也会生成 box 预测，用于第二阶段 ROI 提案。

**差异 3: 第二阶段 VoxelRCNNHead（ROI 细化）**

```
batch_dict（含 rois, roi_scores, roi_labels 来自 CenterHead）
  │
  ▼
ProposalTargetLayer.forward()  # 仅在训练时
  1. sample_rois_for_rcnn():
     - 计算 ROI 与 GT 的 3D IoU（可选 CLASS_AGNOSTIC 模式）
     - subsample_rois(): 按 IoU 阈值分 fg (>= FG_THRESH) / bg (< BG_THRESH)
     - bg 再分为 hard_bg（困难负样本）和 easy_bg
     - 按 FG_RATIO 比例采样，总数为 ROI_PER_IMAGE
  2. 计算回归目标:
     - reg_valid_mask: IoU >= REG_FG_THRESH
     - rcnn_cls_labels: 二值（'cls' 模式）或软标签（'roi_iou' 模式，线性插值）
  3. 编码: GT box 变换到 ROI 局部坐标系
     - 减去 ROI 中心，旋转到 ROI 朝向
     - heading 归一化到 [-pi/2, pi/2]
  │
  ▼
VoxelRCNNHead.roi_grid_pool()
  1. 为每个 ROI 生成 6×6×6 均匀网格点（get_dense_grid_points）
  2. 网格点旋转+平移到全局坐标（get_global_grid_points_of_roi）
  3. 对每个 feature source（来自 backbone 的 multi_scale_3d_features）:
     - 计算体素中心（get_voxel_centers）
     - 构建 voxel→point 索引（generate_voxel2pinds）
     - NeighborVoxelSAModuleMSG: 多尺度体素查询 + 分组 + MLP + 池化
  4. 拼接所有 source 的特征
  输出: pooled_features (B, num_rois, 6³, sum_mlp_channels)
  │
  ▼
SharedFC(256→256) → BN → ReLU → Dropout
  │
  ├── ClsFC(256→256) → BN → ReLU → cls_pred(256→num_class)
  └── RegFC(256→256) → BN → ReLU → reg_pred(256→code_size×num_class)
  │
  ▼
损失函数:
  - cls_loss: BCE（sigmoid, ignore label=-1）或 CrossEntropy
  - reg_loss: WeightedSmoothL1Loss（仅 fg ROIs, beta=1/9）
  - corner_loss: 解码预测框→世界坐标，8 对 8 角点 smoothL1（含 180° 翻转不变量）
  Box Coder: ResidualCoder (code_size=7, encode_angle_by_sincos=False)
  │
  ▼ (推理)
generate_predicted_boxes(): box_coder.decode_torch() → 局部坐标系→全局坐标
```

### CenterPoint vs VoxelRCNN 推理后处理差异

| 特性 | CenterPoint | VoxelRCNN |
|------|-------------|-----------|
| NMS 方式 | circle_nms 或 class_agnostic_nms | class_agnostic_nms 或 multi_classes_nms |
| multihead_label_mapping | 不支持 | 支持（多 head 类别映射） |
| OUTPUT_RAW_SCORE | 不支持 | 支持（输出 sigmoid 前的原始分数） |
| cls_preds_normalized | 自动 sigmoid | 检查标记后决定是否 sigmoid |
| roi recall 跟踪 | 不支持 | 支持（区分 roi 和 rcnn recall） |

---

## CUDA 扩展

### IoU & NMS (`ops/iou3d_nms/`)

| 函数 | 功能 | 输入形状 | 设备 |
|------|------|----------|------|
| `boxes_bev_iou_cpu(boxes_a, boxes_b)` | BEV IoU 矩阵 | `(N,7)`, `(M,7)` | CPU |
| `boxes_iou_bev(boxes_a, boxes_b)` | BEV IoU 矩阵 | `(N,7)`, `(M,7)` | GPU |
| `boxes_iou3d_gpu(boxes_a, boxes_b)` | 3D IoU 矩阵（BEV×高度/体积并） | `(N,7)`, `(M,7)` | GPU |
| `boxes_aligned_iou3d_gpu(boxes_a, boxes_b)` | 逐对 3D IoU | `(N,7)`, `(N,7)` | GPU |
| `paired_boxes_iou3d_gpu(boxes_a, boxes_b)` | 逐对 3D IoU | `(N,7)`, `(N,7)` | GPU |
| `nms_gpu(boxes, scores, thresh, pre_maxsize)` | 旋转 NMS（支持 pre_maxsize 截断） | `(N,7)`, `(N,)` | GPU |
| `nms_normal_gpu(boxes, scores, thresh)` | 普通 NMS | `(N,7)`, `(N,)` | GPU |

3D IoU 计算公式：`3D_IoU = (BEV_overlap × height_overlap) / (vol_a + vol_b - 3D_overlap)`，其中 BEV_overlap 由 CUDA 计算，height_overlap 由 Python 的 max/min/clamp 在 z 轴上计算。

### RoI-Aware Pooling (`ops/roiaware_pool3d/`)

| 函数/类 | 功能 |
|----------|------|
| `points_in_boxes_cpu(points, boxes)` | CPU 判断点是否在 box 内，返回 `(N, num_points)` 二值 mask |
| `points_in_boxes_gpu(points, boxes)` | GPU 批处理版本，返回 `(B, M)` 归属索引（-1=背景） |
| `RoIAwarePool3d(out_size, max_pts)` | 在 RoI 内部做规则网格池化（max/avg） |

### PointNet++ (`ops/pointnet2/`)

**Batch 版本**（`pointnet2_batch`）：操作 `(B, N, C)` 张量，所有 batch 元素 point 数相同。

| 操作 | 类名 | 功能 |
|------|------|------|
| 最远点采样 | `FarthestPointSampling` | `(B,N,3)` → `(B,npoint)` 索引 |
| 球查询 | `BallQuery` | `(B,N,3), (B,npoint,3)` → `(B,npoint,nsample)` |
| 特征分组 | `GroupingOperation` | `(B,C,N), (B,npoint,nsample)` → `(B,C,npoint,nsample)` |
| 球查询+分组 | `QueryAndGroup` | 组合 ball_query + grouping（含相对位置） |
| 全部分组 | `GroupAll` | 不分采样，所有点作为一组 |
| 3NN 插值 | `ThreeNN` + `ThreeInterpolate` | 距离倒数加权插值 |
| 特征收集 | `GatherOperation` | 按索引收集特征 |

**Stack 版本**（`pointnet2_stack`）：操作拼接后的 `(total_N, C)` 张量 + per-batch 计数数组 `[N1, N2, ...]`。用于 VoxelRCNN 的 ROI 网格池化（不同 ROI 点云数量差异大）。

| 操作 | 类名 | 功能 |
|------|------|------|
| 球查询 | `BallQuery` | 含 batch 边界处理和 empty_ball_mask |
| 特征分组 | `GroupingOperation` | 含 batch 边界处理 |
| 球查询+分组 | `QueryAndGroup` | empty ball 位置填零 |
| 堆叠最远点采样 | `StackFarthestPointSampling` | 支持 per-batch 不同 npoint |
| 3NN+插值 | `ThreeNN` + `ThreeInterpolate` | 含 batch 边界 |
| 两步向量池 kNN | `ThreeNNForVectorPoolByTwoStep` | 先找局部邻居，再找每 grid 的 3NN |
| 向量池化 | `VectorPoolWithVoxelQuery` | 3D 网格池化（含自动重试 buffer 扩展） |

**体素查询**（`voxel_query_utils.py`）：

| 类 | 功能 |
|----|------|
| `VoxelQuery` | 按体素坐标查邻居（替代球查询），需要 voxel→point 索引 |
| `VoxelQueryAndGrouping` | 体素查询+分组（自动处理 batch 偏移） |

**多尺度体素 SA**（`voxel_pool_modules.py`）：

| 类 | 功能 |
|----|------|
| `NeighborVoxelSAModuleMSG` | 多尺度体素 Set Abstraction，每 scale 独立 VoxelQueryAndGrouping + MLP，拼接输出 |

---

## 工具函数模块

### config.py — 配置加载

```python
load_config(yaml_path) -> EasyDict       # 加载 YAML，支持 _BASE_CONFIG_ 继承（递归合并）
log_config_to_file(cfg, pre, logger)     # 递归打印所有配置项
cfg_from_list(cfg_list, config)          # 命令行 key=value 覆盖（支持点号路径、EasyDict 子键、列表）

# 内部函数
_load_yaml_into_easydict(yaml_path)      # YAML → EasyDict（先试 FullLoader，fallback 到默认）
_merge_new_config(config, new_config)    # 递归合并，支持 _BASE_CONFIG_ 路径解析
```

### common_utils.py — 通用工具

```python
# tensor 转换
check_numpy_to_torch(x) -> (tensor, bool)   # numpy→torch，返回是否转换的标记

# 数学工具
limit_period(val, offset, period)            # 角度归一化到 [-offset*period, (1-offset)*period)
rotate_points_along_z(points, angle)         # (B,N,3+C) 绕 Z 轴旋转
angle2matrix(angle) -> (3,3)                 # 角度→旋转矩阵（注意与 rotate_points 符号约定不同）
mask_points_by_range(points, limit_range)    # 按 XY 范围过滤

# 体素工具
get_voxel_centers(coords, ds, vsize, range)  # voxel 索引→世界坐标（ZYX→XYZ 转换）
scatter_point_inds(indices, point_inds, shape) # 散射索引到密集 tensor
generate_voxel2pinds(sparse_tensor)          # 稀疏 tensor→voxel_to_point 索引

# 分布式
init_dist_slurm(tcp_port, local_rank)        # SLURM 环境初始化（读取 SLURM_PROCID 等）
init_dist_pytorch(tcp_port, local_rank)      # PyTorch spawn 初始化
get_dist_info(return_gpu_per_machine)        # 返回 rank, world_size
merge_results_dist(result_part, size, tmpdir) # pickle 合并多 rank 结果

# 其他
set_random_seed(seed)                        # 固定 random/numpy/torch/cudnn 种子
create_logger(log_file, rank, log_level)     # rank 0 全量日志，其他 rank 仅 ERROR
keep_arrays_by_name(gt_names, used_classes)  # 按类别过滤 GT 索引
drop_info_with_name(info, name)              # 从 dict 中移除指定 name 的所有条目

class AverageMeter:                          # 运行平均值追踪器
    update(val, n=1)                         # 更新 val/sum/count/avg
```

### box_utils.py — 3D Box 几何

```python
boxes_to_corners_3d(boxes3d)                        # (N,7) → (N,8,3) 角点（单位立方体缩放+旋转+平移）
mask_boxes_outside_range_numpy(boxes, limit_range)  # 按 3D 范围过滤 box（可选 corner 或 center 模式）
boxes3d_lidar_to_aligned_bev_boxes(boxes3d)          # 3D lidar box → BEV AABB（轴对齐包围框）
center_to_corner2d(center, dim)                     # 2D center → 4 角点
bbox3d_overlaps_diou(pred_boxes, gt_boxes)           # 逐对 3D DIoU（BEV 面积 + 3D 高度 + 中心距离归一化）
```

### box_coder_utils.py — Box 编解码

```python
class ResidualCoder:                  # 标准残差: res = (box - anchor) / diag_xy, log_ratio dims
    encode_torch(boxes, anchors) -> residuals
    decode_torch(encodings, anchors) -> boxes
    # 支持 encode_angle_by_sincos: cos/sin 编码（code_size +1）

class PointResidualCoder:             # 基于点的残差（相对于点坐标，而非 anchor）
    encode_torch(gt_boxes, points, gt_classes) -> residuals
    decode_torch(encodings, points, pred_classes) -> boxes
    # use_mean_size: 用 per-class 平均尺寸归一化

class PreviousResidualDecoder:        # 旧版解码器（w/l/h 约定, rg=rt+ra）
class PreviousResidualRoIDecoder:     # 旧版 RoI 解码器（rg=ra-rt）
```

### loss_utils.py — 损失函数

| 类/函数 | 用途 | 关键参数 |
|---------|------|----------|
| `SigmoidFocalClassificationLoss` | 多类分类 | alpha=0.25, gamma=2.0 |
| `WeightedSmoothL1Loss` | box 回归 | beta=1/9, code_weights |
| `WeightedL1Loss` | box 回归 | code_weights, AMP 混合精度 |
| `WeightedCrossEntropyLoss` | 分类 | 逐 anchor 权重 |
| `FocalLossCenterNet` | heatmap 损失 | CornerNet-style |
| `RegLossCenterNet` | heatmap 回归 | L1，在关键点收集特征 |
| `FocalLossSparse` | 稀疏 heatmap | 可变 N 输入 |
| `RegLossSparse` | 稀疏回归 | 可变 N 输入 |
| `IouLossSparse` | IoU 预测监督 | L1(预测 IoU, 3D IoU 真值) |
| `IouRegLossSparse` | DIoU 回归 | 1 - DIoU |
| `GaussianFocalLoss` | 高斯 heatmap | alpha=2, gamma=4 |
| `L1Loss` | 简单 L1 | 空 target 返回 0 |
| `get_corner_loss_lidar()` | 角点损失 | 8 对 8 角点 smoothL1（含翻转不变） |
| `neg_loss_cornernet()` | CenterNet focal | `(batch, c, h, w)` 密集 heatmap |
| `neg_loss_sparse()` | Sparse CenterNet focal | `(batch, c, n)` 稀疏 heatmap |
| `calculate_iou_loss_centerhead()` | CenterHead IoU | paired_boxes_iou3d_gpu |
| `calculate_iou_reg_loss_centerhead()` | CenterHead DIoU | bbox3d_overlaps_diou |

### spconv_utils.py — spconv 工具

```python
find_all_spconv_keys(model, prefix="") -> Set[str]  # 递归找所有 SparseConvolution.weight 的 key
replace_feature(out, new_features)                   # 替换稀疏 tensor 特征（兼容 spconv 1.x/2.x）
spconv_autocast_inference_fallback(module)           # spconv 2.3.x 半精度 eval kernel 兼容上下文
# 模块加载时: spconv >= 2.2 则设置 SPCONV_USE_DIRECT_TABLE = False
```

---

## 训练

### 训练入口 (`tools/train.py`)

```bash
python tools/train.py --cfg_file configs/<model>_<dataset>.yaml [可选参数]
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--batch_size` | 单卡 batch size | 配置文件值 |
| `--epochs` | 训练轮数 | 配置文件值 |
| `--workers` | 数据加载线程数 | 4 |
| `--precision` | `fp32` / `fp16` / `bf16`，命令行优先于 YAML | `PRECISION` |
| `--use_amp` | `--precision fp16` 的兼容别名 | False |
| `--extra_tag` | 实验标签 | default |
| `--ckpt` | 从检查点恢复（含 optimizer state） | 自动搜索最新 |
| `--pretrained_model` | 预训练权重（仅 model state） | 无 |
| `--fix_random_seed` | 固定随机种子（666） | False |
| `--ckpt_save_interval` | 检查点保存间隔（epoch） | 1 |
| `--max_ckpt_save_num` | 最多保留检查点数（删旧留新） | 30 |
| `--eval_interval` | 每 N 个 epoch 在 val 集评估一次（0=禁用） | 0 |
| `--root_path` | 覆盖 pkl 信息文件/GT 数据库目录 | 配置文件值 |
| `--raw_data_path` | 覆盖原始点云数据集根目录 | `DATA_CONFIG.RAW_DATA_PATH` |
| `--max_sweeps` | 同时覆盖目标域和源域的总输入帧数；0/1 表示仅当前帧 | 配置文件值 |

示例：

```bash
# 单卡训练
python tools/train.py --cfg_file configs/voxelrcnn_coda.yaml

# 4 卡 DDP 训练
torchrun --nproc_per_node=4 tools/train.py --cfg_file configs/voxelrcnn_coda.yaml

# 单卡训练 + 评估 + checkpoint 控制
python tools/train.py --cfg_file configs/voxelrcnn_coda.yaml \
    --eval_interval 3 --ckpt_save_interval 5 --max_ckpt_save_num 10
```

**多卡训练（DDP）：** 使用 `torchrun` 启动，自动处理 `DistributedDataParallel` 包装、`DistributedSampler` 数据分配、以及日志/checkpoint/评估的 rank 0 独占。`--batch_size` 始终指单卡 batch size。CUDA local device 在 NCCL process group 之前绑定，避免非零 rank 在逻辑 GPU 0 建立多余上下文。使用 `--log_gpu_memory` 可在普通训练日志间隔收集所有 rank 的输入点数、动态体素限流前后数量，以及 allocated/reserved/peak CUDA 显存。

**训练主流程：**

```
1. 解析参数 + 加载配置（YAML → EasyDict, 支持 _BASE_CONFIG_）
2. 创建输出目录: output/<TAG>/<extra_tag>/ckpt/
3. 设置 logger（文本文件）和 TensorBoard SummaryWriter
4. 构建训练数据集: CodaDataset(training=True) + DataLoader(pin_memory, shuffle)
5. 构建模型: build_model(model_cfg, num_class, dataset) → .cuda()
6. 按 `--precision` 创建全训练生命周期共享的 GradScaler（FP16/BF16 启用），并构建优化器
     adam/sgd:     标准 PyTorch 优化器
     adam_onecycle: OptimWrapper + Adam（真实 weight decay, AdamW 风格）
     adam_cosineanneal: 同上
7. 加载检查点:
     --pretrained_model: model.load_params_from_file()  # weights only
     --ckpt:      model.load_params_with_optimizer()    # weights + optimizer + GradScaler state
     自动:        扫描 ckpt_dir 中 .pth 文件，按 mtime 降序尝试加载
8. 构建调度器: build_scheduler(...)
     adam_onecycle:      OneCycle（LR 先升后降）
     adam_cosineanneal:  CosineAnnealing + CosineWarmupLR
     其他:               LambdaLR（epoch 衰减）+ 可选 CosineWarmupLR
9. (可选) 构建 val DataLoader（如果 --eval_interval > 0）
10. 训练循环 (train_model → train_one_epoch × total_epochs)
    - 每个 epoch 后保存 checkpoint
    - 每 eval_interval 个 epoch 自动在 val 集评估（eval_epoch）
    - 最佳模型自动保存为 ckpt/best_model.pth
11. 训练后最终评估：加载 best_model.pth 运行 eval_one_epoch
```

### 训练循环 (`simple_det3d/train/train_utils.py`)

**`train_one_epoch` 单 epoch 循环：**

1. 接收训练入口创建的持久 `GradScaler`；FP16/BF16 启用，FP32 为禁用状态（spconv 2.3.x 在 BF16 autocast 下的稀疏卷积实际使用 FP16）
2. `AverageMeter` 跟踪 `data_time`, `batch_time`, `forward_time`, `loss`
3. 每个 iteration:
   - 获取 batch（`StopIteration` 时重建迭代器）
   - `lr_scheduler.step(accumulated_iter, cur_epoch)`
   - 在所选 precision 的 autocast 上下文内调用 `model_fn_decorator` 返回的 `model_func`:
     1. `load_data_to_gpu(batch_dict)` — numpy → CUDA
     2. `model(batch_dict)` → `ret_dict, tb_dict, disp_dict`
     3. `loss = ret_dict['loss'].mean()`
     4. `model.module.update_global_step()`
   - `scaler.scale(loss).backward()` → `scaler.unscale_(optimizer)` → `clip_grad_norm_` → `scaler.step()` → `scaler.update()`
   - 定期日志（tqdm + tensorboard + text logger）

阶段 2 在 Student 前向之前先运行 EMA Teacher；伪标签注入后立即清空 Teacher
head 缓存并删除其预测和 `batch_dict`，使完整稀疏特征金字塔不与 Student
前向/反向图同时驻留。PyTorch CUDA reserved memory 仍可能保留历史峰值，因此
应结合 `--log_gpu_memory` 中的 allocated 与 reserved 判断实际张量占用和缓存。

**`train_model` 多 epoch 管理：**

- 外层 tqdm epoch 进度条
- 每个 epoch: 设置 `train_sampler.set_epoch(cur_epoch)`（DDP shuffle）
- Warmup 期内切换 `lr_warmup_scheduler`
- 每个 epoch 后保存检查点（`checkpoint_state` + `save_checkpoint`），包含 FP32 主权重、optimizer、precision 和 scaler 状态
- 自动清理旧检查点（保留最近 `max_ckpt_save_num` 个）
- 当 `eval_interval > 0` 时，每 N 个 epoch 自动触发以下流程:

```
eval_epoch(model, test_loader, cfg, logger, cur_epoch, tb_log, accumulated_iter)
  │
  ├── model.eval()
  ├── 遍历 test_loader
  │      └── model(batch) → pred_dicts
  │      └── dataset.generate_prediction_dicts() → det_annos
  ├── dataset.evaluation(det_annos, class_names) → AP3D/APBEV
  ├── 记录到 logger 和 TensorBoard（eval/ 前缀）
  ├── 如果当前 mean_metric > 历史最佳 → 保存 ckpt/best_model.pth
  └── model.train()
```

**`eval_epoch` 函数签名：**

```python
def eval_epoch(model, test_loader, cfg, logger, cur_epoch,
               tb_log=None, accumulated_iter=0, precision='fp32') -> dict
    # 轻量级评估：不做可视化、不保存结果文件
    # 返回 flat dict: {'Car_AP3D': 0.85, 'Pedestrian_AP3D': 0.62, ...}
```

**`train_model` 新增参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `eval_interval` | int | 0 | 每 N 个 epoch 评估一次（0=禁用） |
| `test_loader` | DataLoader | None | 验证集 DataLoader |
| `cfg` | EasyDict | None | 配置对象 |

### 优化器设计

**OptimWrapper**：包装标准优化器，提供属性式超参访问（`.lr`, `.mom`, `.beta`, `.wd`）。支持真实权重衰减（`param.data.mul_(1 - wd * lr)`），分离 BN/非BN 参数组。

**FastAIMixedOptim(OptimWrapper)**：FP32 master 参数 + FP16 前向。梯度从 model（FP16）→ master（FP32，除以 loss_scale）→ 更新 → 复制回 model。

**学习率调度器：**

| 调度器 | 说明 |
|--------|------|
| `OneCycle` | 1cycle 策略，LR 先升后降（cos），momentum 先降后升 |
| `CosineAnnealing` | 预热后余弦退火（LR_max → LR_max×0.001），手动管理 mom_phases |
| `CosineWarmupLR` | 余弦预热（eta_min → base_lr），PyTorch _LRScheduler 子类 |
| `LambdaLR` | 仅在 DECAY_STEP_LIST epoch 乘衰减因子 |

---

## 推理与评估

### 推理入口 (`tools/test.py`)

```bash
# 评估（AP3D/APBEV）
python tools/test.py --cfg_file configs/centerpoint_coda.yaml --ckpt <path>

# 测速
python tools/test.py --cfg_file configs/centerpoint_coda.yaml --ckpt <path> --infer_time

# BEV 可视化
python tools/test.py --cfg_file configs/centerpoint_coda.yaml --ckpt <path> --save_vis
```

**`eval_one_epoch` 流程：**

1. 遍历 DataLoader（`model.eval()`, `torch.no_grad()`）
2. `model(batch_dict)` → `pred_dicts, ret_dict`
3. 累积 recall（IoU ∈ RECALL_THRESH_LIST，如 [0.3, 0.5, 0.7]）
4. `dataset.generate_prediction_dicts()` → 标准标注格式
5. `dataset.evaluation()` → AP3D + APBEV（41-point interpolated）
6. 可选：保存 `result.pkl` 和 BEV 可视化 PNG

**`measure_inference_time`：** 10 warmup + 100 测量，含 `cuda.synchronize()`，报告 ms + FPS。

**`save_bev_visualization`：** 前 50 帧的 BEV 视角 PNG（Agg 后端）。配色：Car=红，Pedestrian=绿，Cyclist=蓝。

### Wheelchair 推理人工审计 (`tools/visualize_wheelchair_predictions.py`)

该独立入口面向不完整弱标注，不复用普通 AP 可视化的“完整 GT”假设。它按场景或
帧筛选测试样本，执行模型推理，再生成同尺度的双视图：左图仅绘制人工 BEV 中心及
过滤后的弱标签，右图仅绘制带朝向的预测框、置信度和框内当前帧点数。两图使用相同
点云和完整检测范围，不再混合叠加或使用局部视角。人工中心仍采用一对一半径匹配
计算召回式审计数据；未匹配预测标记为待复核，不视作假阳性。

帧选择支持 `missed`、`matched`、`crowded`、`low_support`、`random`、
`uniform` 和 `all`。输出包括 PNG 图片、`all_frames.csv`、
`selected_frames.csv`、`summary.json` 和原始 `predictions.pkl`，便于人工检查及
不同 checkpoint 间离线对比。

`compare_wheelchair_predictions.py` 读取单帧和四帧推理保存的 `predictions.pkl`，
按 `frame_id` 严格对齐后生成三栏 BEV：可用弱真值、单帧预测、四帧预测。默认均衡
抽取单帧占优、四帧占优、命中相同但单帧更干净、时间均匀样本，并在三栏中使用
完全相同的当前帧点云、坐标范围和显示阈值。

`create_wheelchair_comparison_video.py` 对序列 info 和两份预测做严格的三方
`frame_id` 对齐，按 info 的原始时序渲染全部帧并调用 ffmpeg 编码为 MP4。
左栏会根据 `center_label_is_exhaustive` 显式标记“已复核近似完整”或“弱标签
不完整”，防止将非穷举帧的漏标视为模型误检。

---

## 数据格式

### pkl 信息文件格式

```python
# 列表，每个元素为一帧
{
    'token': str,                        # 帧唯一标识（如 'seq0_frame_0001'）
    'lidar_path': str,                   # 点云 .bin 路径；新 pkl 相对 RAW_DATA_PATH，兼容旧绝对路径
    'gt_boxes': np.ndarray,              # (N, 7) float32 [x, y, z, dx, dy, dz, heading]
    'gt_names': np.ndarray,              # (N,) str 类别名
    'gt_cls_weights': np.ndarray,        # (N,) 弱标签分类/热图权重（可选）
    'gt_reg_weights': np.ndarray,        # (N,) 弱标签回归权重（可选）
    'gt_box_reg_mask': np.ndarray,       # (N, 8) CenterHead逐维回归掩码（可选）
    'label_is_exhaustive': bool,         # 中心/类别是否穷举，是否允许背景监督
    'center_label_is_exhaustive': bool,  # 上述语义的显式字段
    'is_target_domain': bool,            # 与穷举性独立的源/目标域标志
    'roi_supervision_enabled': bool,     # 是否对该帧计算ROI监督
    'sweeps': [                          # 多帧扫描列表（可为空）
        {
            'lidar_path': str,           #   扫描文件路径
            'time_lag': float,           #   时间差（秒）
            'transform_matrix': np.ndarray # (4, 4) 变换到当前帧的矩阵
        },
    ],
    'lidar2global': np.ndarray,          # (4, 4) float32, lidar→global 变换矩阵
}
```

### 点云格式

- `.bin` 文件：`(N, 4)` float32 `[x, y, z, intensity]`
- sweep 拼接后：`(N, 5)` float32 `[x, y, z, intensity, time]`

### 预测输出格式

```python
# pred_dicts: list of dict，每帧一个
{
    'pred_boxes':  torch.Tensor,  # (K, 7) [x, y, z, dx, dy, dz, heading]
    'pred_scores': torch.Tensor,  # (K,) 置信度分数
    'pred_labels': torch.Tensor,  # (K,) int, 类别 ID（1-indexed）
}

# generate_prediction_dicts 转换后:
{
    'name':        np.ndarray,    # (K,) str 类别名
    'score':       np.ndarray,    # (K,) float
    'boxes_lidar': np.ndarray,    # (K, 7) float32
    'pred_labels': np.ndarray,    # (K,) int
    'frame_id':    str,           # 帧 ID
}
```

### GT 数据库格式

`create_groundtruth_database()` 生成：

```
data/coda/
├── gt_database/
│   ├── 000000.bin              # 每个 GT 物体的点云
│   └── ...
└── coda_dbinfos_train.pkl      # 数据库索引
```

索引 pkl 格式（按类别组织）：
```python
{'Car': [{'name': 'Car', 'path': 'gt_database/xxx.bin', 'box3d_lidar': (7,),
          'num_points_in_gt': int, 'difficulty': int, ...}, ...], ...}
```

---

## 配置结构

### 加载机制

`load_config(yaml_path)` → `EasyDict`。支持多层 `_BASE_CONFIG_` 继承：递归加载所有 base，再用当前文件覆盖。

### 顶层 Section 总览

| Section | 关键字段 | 说明 |
|---------|---------|------|
| `CLASS_NAMES` | `list[str]` | 目标类别（CODa默认3类，Wheelchair弱迁移仅Pedestrian） |
| `PRECISION` | `fp32` / `fp16` / `bf16` | 训练与推理默认精度，可由 `--precision` 覆盖 |
| `DATA_CONFIG` | `DATA_PATH`, `RAW_DATA_PATH`, `POINT_CLOUD_RANGE`, `INFO_PATH`, `MAX_SWEEPS`, `MAX_SWEEP_TIME_LAG`, `SWEEP_SELECT_MODE`, `BALANCED_RESAMPLING` | 信息文件、原始点云路径、范围和时序融合参数 |
| `DATA_CONFIG.DATA_AUGMENTOR` | `AUG_CONFIG_LIST`, `DISABLE_AUG_LIST` | 增强策略列表 |
| `DATA_CONFIG.DATA_PROCESSOR` | `NAME` 列表 + 各处理器参数 | 数据处理 Pipeline |
| `DATA_CONFIG.POINT_FEATURE_ENCODING` | `encoding_type`, `src_feature_list`, `used_feature_list` | 点特征编码 |
| `MODEL.NAME` | `CenterPoint` / `VoxelRCNN` | 模型选择 |
| `MODEL.VFE` | VFE 配置 | 体素特征编码器 |
| `MODEL.BACKBONE_3D` | 3D 卷积参数 | 稀疏 3D backbone |
| `MODEL.MAP_TO_BEV` | 高度压缩参数 | Z→channel 折叠 |
| `MODEL.BACKBONE_2D` | `LAYER_NUMS`, `LAYER_STRIDES`, `NUM_FILTERS`, `UPSAMPLE_STRIDES`, `NUM_UPSAMPLE_FILTERS` | 2D BEV backbone (FPN) |
| `MODEL.DENSE_HEAD` | `SHARED_CONV_CHANNEL`, `CLASS_NAMES_EACH_HEAD`, `TARGET_ASSIGNER_CONFIG`, loss weights | Dense head 配置 |
| `MODEL.POST_PROCESSING` | `SCORE_THRESH`, `NMS_CONFIG`, `RECALL_THRESH_LIST`, `POST_CENTER_LIMIT_RANGE` | 后处理 |
| `MODEL.ROI_HEAD` | (VoxelRCNN 专属) ROI 头完整配置 | 二阶段头部 |
| `OPTIMIZATION` | `BATCH_SIZE_PER_GPU`, `NUM_EPOCHS`, `LR`, `WEIGHT_DECAY`, `GRAD_NORM_CLIP` | 优化参数 |

### 配置文件差异矩阵

**按数据集维度：**

| 配置项 | CODa | Wheelchair |
|--------|------|------------|
| `CLASS_NAMES` | `[Car, Pedestrian, Cyclist]` | `[Pedestrian, Cyclist]` |
| `POINT_CLOUD_RANGE` | `[-22, -22, -2, 22, 22, 6]` | `[0, -7.6, -2, 15.2, 7.6, 4]` |
| `BALANCED_RESAMPLING` | `true` | `false` |
| `MAX_SWEEPS` | `4`（当前帧 + 3 个历史帧） | `4`（当前帧 + 3 个历史帧） |
| `MAX_SWEEP_TIME_LAG` | `0.35` 秒 | `0.35` 秒（按 10 Hz 估算） |
| `SWEEP_SELECT_MODE` | `near` | `near` |
| `gt_sampling` | 暂时禁用（配置块已注释） | 无 |
| `random_world_translation` | 无 | `NOISE_TRANSLATE_STD: [0.5, 0.5, 0.5]` |
| `VOXEL_SIZE` | `[0.05, 0.05, 0.2]` | `[0.05, 0.05, 0.15]` |

**按模型维度（以 CODa 为例）：**

| 配置项 | CenterPoint | VoxelRCNN |
|--------|-------------|-----------|
| `VFE.NAME` | `MeanVFE` | `DynamicVoxelVFE` |
| `BACKBONE_2D.LAYER_NUMS` | `[5]` | `[5, 5]`（两阶段 2D backbone） |
| `BACKBONE_2D.UPSAMPLE_STRIDES` | `[1]` | `[1, 2]` |
| `DENSE_HEAD.SHARED_CONV_CHANNEL` | `64` | `128` |
| `DENSE_HEAD.TARGET_ASSIGNER_CONFIG.FEATURE_MAP_STRIDE` | `4` | `8` |
| `SCORE_THRESH` | `0.3` | `0.1` |
| `NMS_THRESH` | `0.1` | `0.7` |
| `ROI_HEAD` | 无 | 完整配置（RoI 网格池化、FC 层、损失权重等） |
| `BATCH_SIZE_PER_GPU` | `32` | `16` |
| `NUM_EPOCHS` | `20` | `30` |
| `LR` | `0.003` | `0.01` |

---

## 工具脚本

### downsample_coda_lidar.py

CODa OS1-128 点云的垂直通道降采样工具。源点云按 `(128, 1024, 4)` 还原，
在完整垂直视场内均匀选取 16 个通道，保持原文件名和序列目录结构后写入
`3d_comp_downsample/os1`。

```bash
python tools/downsample_coda_lidar.py \
    --data_root /share/public_datasets/VLA/nitrogen/CODa_full \
    --workers 8
```

默认会跳过尺寸正确的已有输出以支持断点续跑；`--overwrite` 可重新生成，
`--seq_ids 0 1 2` 可限制序列，`--limit 1` 可执行小规模检查。输出先写临时文件，
完整性检查通过后再原子替换目标文件。

### generate_coda_info.py

CODa 原始数据 → pkl 信息文件。

```bash
# 基础用法
python tools/generate_coda_info.py

# 降采样点云
python tools/generate_coda_info.py --pc_dir 3d_comp_downsample/os1

# 仅保留包含 Pedestrian 的帧
python tools/generate_coda_info.py --filter_class Pedestrian

# 完整参数
python tools/generate_coda_info.py \
    --data_root /share/public_datasets/VLA/nitrogen/CODa_full \
    --pc_dir 3d_comp_downsample/os1 \
    --filter_class Pedestrian \
    --output_dir data/coda_downsample \
    --sweep_frame_len 3 \
    --max_sweep_time_lag 0.35
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_root` | CODa 路径 | 数据集根目录 |
| `--output_dir` | `data/coda` | pkl 输出目录 |
| `--sweep_frame_len` | 3 | 最大历史 sweep 数，不含当前帧 |
| `--max_sweep_time_lag` | 0.35 | 最大历史时间差（秒） |
| `--seq_ids` | 全部 | 指定序列 ID 列表 |
| `--pc_dir` | `3d_comp/os1` | 点云子目录，降采样时改为 `3d_comp_downsample/os1` |
| `--filter_class` | 无 | 只保留包含指定类别（Car/Pedestrian/Cyclist）的帧 |

**处理逻辑：**

1. 扫描 `data_root/<pc_dir>/` 目录自动发现序列
2. 每个序列：加载标定（`calib_os1_to_base.yaml`，lidar→base）、位姿（`poses.txt`，6-DOF）、标注（cuboid JSON）
3. 13 个原始 CODa 类别 → 3 个目标类别（class mapping: Car, Pedestrian, Cyclist）
4. 从目标帧之前的连续原始 LiDAR 帧构建 sweep，可跨 metadata 标注划分
5. `lidar2global = base2global @ lidar2base`；sweep 用 `inv(T_current) @ T_sweep` 做相对变换
6. 同时按物理帧号窗口和 `max_sweep_time_lag` 过滤断点，并跳过无效位姿/变换
7. 数据校验：跳过缺失/损坏的点云、缺失标注、空标注、NaN/Inf box、非正尺寸等无效帧
8. 输出 `coda_infos_{train,val,test}.pkl` + 类别分布统计 + 跳过摘要

### generate_wheelchair_info.py

Wheelchair 数据集（无标注）→ pkl 信息文件。

```bash
python tools/generate_wheelchair_info.py [--data_root <path>] [--output_dir <path>]
                                         [--sweep_frame_len 3]
                                         [--max_sweep_time_lag 0.35]
                                         [--val_split 0.1]
```

**处理逻辑：**

1. 遍历 5 个类别目录（indoor1/indoor2/outdoor1/outdoor2/people），取最后 `val_split` 比例场景做验证
2. 每帧：加载可选 pose（`pose{id}: x y z qx qy qz qw`），按10Hz估算sweep time_lag；已知平台静止的`indoor1/people`在无pose时使用单位变换，其他位姿缺失或断点不做历史融合
3. gt_boxes/gt_names 为空 `(0, 7)`
4. 输出 `wheelchair_infos_{train,val}.pkl` + 统计

### prepare_wheelchair_weak_data.py

读取场景级图像伪标签和 `annotations_*.pkl` 人工 BEV 中心，枚举全部原始点云帧，
生成全量/室内/过渡、室内穷举中心、室内非穷举和穷举中心验证 info。人工标注键是
对应场景在主 pkl 中 `frames` 列表的零基下标，必须通过
`frames[k].frame_index` 映射到原始帧，不能使用固定帧号偏移。人工点与原有伪
3D框按互补标签独立保留，不进行距离关联或跨类型去重；人工标注帧上的二者并集
设置 `center_label_is_exhaustive=true`。该字段只允许中心分类和背景监督，不改变
伪框几何可信度。被拒绝的框会降级为中心监督或 ignore。

过滤阈值均可由命令行覆盖，并写入 JSON 汇总；无人工中心支撑的 A 级框默认
还要求图像分数不低于 0.6。只有存在正尺寸 Pedestrian
候选框的帧才实际读取点云做框内点数检查。无 pose 且已知平台静止的
`indoor1/people` 使用单位位姿，其余无有效 pose 的场景不参与时序支持判断。

### 弱监督训练运行时

- `MixedDomainDataset` 保留两路兼容接口；`MultiDomainDataset` 以确定性比例混合
  CODA、室内穷举中心和室内非穷举三路数据，并检查空间范围、体素和点特征兼容性。
- `CenterHead` 将 `heatmap_background_weight` 与逐目标 `gt_cls_weights` 分开：
  非穷举帧背景权重为0，穷举目标域帧默认0.25/阶段3为0.5，CODA为1；
  `gt_box_reg_mask` 继续负责逐维回归。
- CODA 使用常规IoU ROI分类与回归；非穷举Wheelchair帧屏蔽ROI监督；穷举中心帧
  使用0.5m正样本/1.0m负样本的中心式ROI分类，并始终屏蔽ROI框回归。
- EMA Teacher 仅在现有弱中心附近注入通过分数、尺寸、范围和当前帧点数过滤的
  几何。`is_target_domain` 与标签穷举性独立，因此穷举中心帧仍可接受几何升级；
  普通 checkpoint 额外保存 `ema_model_state`。
- `MODEL_FREEZE.MODULE_NAMES` 可冻结指定子模块并强制保持 eval 模式，用于
  target-only 阶段保护 z、尺寸、yaw 和 ROI 几何；对依赖 `self.training` 控制
  target assignment 的容器可用 `TRAIN_MODE_MODULE_NAMES` 保留根模块训练分支，
  同时让其参数化子模块保持 eval。

### generate_gt_database.py

```bash
python tools/generate_gt_database.py --cfg_file configs/<cfg>.yaml --split train
```

通过 `CodaDataset.create_groundtruth_database()` 实现：遍历所有帧→提取每个 GT box 内的点云（`roiaware_pool3d_utils.points_in_boxes_cpu`）→保存为 `.bin` + 生成索引 `.pkl`。

---

## 关键类的完整接口

### 模型入口 (`models/__init__.py`)

```python
def build_model(model_cfg, num_class, dataset) -> nn.Module
    # model_cfg.NAME == 'CenterPoint' → CenterPoint(...)
    # model_cfg.NAME == 'VoxelRCNN'  → VoxelRCNN(...)

def load_data_to_gpu(batch_dict) -> None
    # numpy.ndarray → torch.float().cuda()
    # 跳过: frame_id, metadata, calib, image_paths, images, ori_shape, img_process_infos
```

### CenterPoint (`models/centerpoint.py`)

```python
class CenterPoint(nn.Module):
    def __init__(self, model_cfg, num_class, dataset)
        # 从 dataset 获取 grid_size, point_cloud_range, voxel_size, num_point_features
        # 构建: MeanVFE → VoxelResBackBone8x → HeightCompression → BaseBEVBackbone → CenterHead
        # CenterHead: predict_boxes_when_training=False

    def forward(self, batch_dict) -> tuple
        # 训练: ({'loss': loss}, tb_dict, disp_dict)
        # 推理: (pred_dicts, recall_dicts)

    def get_training_loss(self) -> (loss, tb_dict, disp_dict)
    def post_processing(self, batch_dict) -> (pred_dicts, recall_dicts)

    # 权重加载（支持 spconv 1.x/2.x 权重格式自动转换）
    def load_params_from_file(self, filename, logger, to_cpu, pre_trained_path)
    def load_params_with_optimizer(self, filename, to_cpu, optimizer, logger) -> (it, epoch)
    def _load_state_dict(self, model_state_disk, *, strict) -> (state_dict, update_model_state)
```

### VoxelRCNN (`models/voxel_rcnn.py`)

```python
class VoxelRCNN(nn.Module):
    # 同 CenterPoint，差异:
    #   - VFE: DynamicVoxelVFE
    #   - CenterHead: predict_boxes_when_training=True
    #   - 可选 roi_head: VoxelRCNNHead

    def get_training_loss(self) -> (loss, tb_dict, disp_dict)
        # loss = loss_rpn + loss_rcnn（如果有 roi_head）

    def post_processing(self, batch_dict) -> (pred_dicts, recall_dicts)
        # 支持 multi_classes_nms（含 multihead_label_mapping）、OUTPUT_RAW_SCORE
```

### CenterHead (`models/center_head.py`)

```python
class CenterHead(nn.Module):
    def __init__(self, model_cfg, input_channels, num_class, class_names,
                 grid_size, point_cloud_range, voxel_size, predict_boxes_when_training)

    def forward(self, data_dict) -> dict
        # shared_conv → SeparateHead × N → pred_dicts
        # 训练: assign_targets（高斯热力图 + 回归目标）
        # 推理/predict_boxes_when_training: generate_predicted_boxes

    def assign_targets(self, gt_boxes, feature_map_size) -> dict
        # 返回: heatmaps, target_boxes, inds, masks, target_boxes_src

    def get_loss(self) -> (total_loss, tb_dict)
        # hm_loss(FocalLoss) + loc_loss(L1) + (可选) iou_loss + iou_reg_loss

    def generate_predicted_boxes(self, batch_size, pred_dicts) -> list
        # decode_bbox_from_heatmap → (可选) score rectification → NMS

    @staticmethod
    def reorder_rois_for_refining(batch_size, pred_dicts) -> (rois, roi_scores, roi_labels)
```

### VoxelRCNNHead (`models/voxel_rcnn_head.py`)

```python
class VoxelRCNNHead(nn.Module):
    def __init__(self, backbone_channels, model_cfg, point_cloud_range, voxel_size, num_class)
    def forward(self, batch_dict) -> dict
        # proposal_layer → (训练) assign_targets → roi_grid_pool → shared_fc → cls/reg preds

    def proposal_layer(self, batch_dict, nms_config)  # 从 CenterHead 输出生成 ROIs
    def assign_targets(self, batch_dict)              # ProposalTargetLayer + canonical 变换
    def roi_grid_pool(self, batch_dict)               # 多尺度 3D 体素查询池化 (6×6×6 网格)
    def get_loss(self, tb_dict=None)                   # BCE/CE + WeightedSmoothL1 + CornerLoss

class ProposalTargetLayer(nn.Module):
    def forward(self, batch_dict) -> targets_dict
    def sample_rois_for_rcnn(self, batch_dict)  # IoU 计算 + fg/bg 采样
```

### model_utils.py — NMS 和 Heatmap 解码

```python
# NMS
def class_agnostic_nms(box_scores, box_preds, nms_config, score_thresh)
def multi_classes_nms(cls_scores, box_preds, nms_config, score_thresh)
def class_specific_nms(box_scores, box_preds, box_labels, nms_config, score_thresh)

# Heatmap 工具
def gaussian_radius(height, width, min_overlap=0.5)    # 三次方程解取 min
def gaussian2D(shape, sigma=1)
def draw_gaussian_to_heatmap(heatmap, center, radius, k=1)

# Heatmap 解码
def decode_bbox_from_heatmap(heatmap, rot_cos, rot_sin, center, center_z, dim,
                             point_cloud_range, voxel_size, feature_map_stride,
                             vel=None, iou=None, K=100, circle_nms=False,
                             score_thresh=None, post_center_limit_range=None)

def _nms(heat, kernel=3)               # max-pool NMS
def _circle_nms(boxes, min_radius)     # numba JIT 加速
def _topk(scores, K=40)                # top-K 峰值位置和类别
```

---

## 模块间完整依赖关系

```
tools/train.py
  ├── utils/config.py              (load_config)
  ├── datasets/coda_dataset.py     (CodaDataset)
  ├── models/__init__.py           (build_model, load_data_to_gpu)
  └── train/                       (optimization, train_utils)

tools/test.py
  ├── utils/config.py
  ├── datasets/coda_dataset.py
  └── models/__init__.py

models/centerpoint.py
  ├── .vfe                         (MeanVFE)
  ├── .backbones_3d                (VoxelResBackBone8x, SparseBasicBlock)
  ├── .backbones_2d                (HeightCompression, BaseBEVBackbone)
  ├── .center_head                 (CenterHead, SeparateHead)
  └── ..ops/iou3d_nms              (iou3d_nms_utils — recall 计算)

models/voxel_rcnn.py
  ├── .vfe                         (DynamicVoxelVFE, PFNLayerV2)
  ├── .backbones_3d, .backbones_2d, .center_head
  ├── .voxel_rcnn_head             (VoxelRCNNHead, ProposalTargetLayer)
  │     ├── ..ops/pointnet2/pointnet2_stack/  (voxel_query_utils, voxel_pool_modules)
  │     └── ..ops/roiaware_pool3d/             (roiaware_pool3d_utils)
  └── .model_utils                 (NMS, heatmap 解码)

models/center_head.py
  ├── .model_utils                 (NMS, gaussian_radius, decode_bbox_from_heatmap)
  └── ..utils/loss_utils           (FocalLossCenterNet, RegLossCenterNet, IoU losses)

models/voxel_rcnn_head.py
  ├── .model_utils
  ├── ..utils/loss_utils           (WeightedSmoothL1Loss, get_corner_loss_lidar)
  ├── ..utils/box_coder_utils      (ResidualCoder)
  ├── ..ops/pointnet2_stack/       (NeighborVoxelSAModuleMSG 等)
  └── ..ops/iou3d_nms              (iou3d_nms_utils)

datasets/coda_dataset.py
  ├── .data_augmentor               (DataAugmentor, DataBaseSampler)
  │     └── .data_processor         (VoxelGeneratorWrapper → spconv)
  ├── .data_processor               (PointFeatureEncoder)
  ├── ..utils/box_utils, ..utils/common_utils
  ├── ..ops/iou3d_nms               (评估用)
  └── ..ops/roiaware_pool3d         (GT 数据库生成用)

utils/loss_utils.py
  └── ..ops/iou3d_nms               (iou3d_nms_utils — paired IoU 计算)
```

---

## 版本兼容性处理

### spconv 权重格式

`_load_state_dict()` 自动处理 spconv 1.x ↔ 2.x 权重差异：
- 1.x: `(K, C, R, S)` — 2.x: `(C, R, S, K)`
- 尝试 `transpose(-1, -2)` 转换；3D 卷积额外尝试 `permute(4, 0, 1, 2, 3)`
- 只加载 shape 匹配的权重，非严格模式下跳过不匹配的 key

### spconv 体素生成 API

`VoxelGeneratorWrapper` 自动适配：
- v1: `VoxelGeneratorV2` 或 `VoxelGenerator.generate()`（兼容 dict/tuple 返回）
- v2: `cumm.tensorview` + `Point2VoxelCPU3d.point_to_voxel()`

### spconv 特征替换

`replace_feature(out, new_features)` 兼容：
- 2.x: `out.replace_feature(new_features)`（方法调用）
- 1.x: `out.features = new_features`（属性赋值）

---

## 扩展与自定义

### 添加新模型

1. 在 `models/` 下创建新模型文件，继承 `nn.Module`
2. 实现 `__init__(model_cfg, num_class, dataset)` — 从 `dataset` 获取 grid_size/voxel_size/point_cloud_range/num_point_features
3. 实现 `forward(batch_dict)` — 训练返回 `({'loss': loss}, tb_dict, disp_dict)`，推理返回 `(pred_dicts, recall_dicts)`
4. 实现 `load_params_from_file()` 和 `load_params_with_optimizer()`
5. 在 `models/__init__.py` 的 `build_model()` 中添加分支
6. 创建对应的 YAML 配置文件

### 添加新数据集

1. 编写 `tools/generate_<dataset>_info.py`，生成符合 pkl 格式的信息文件
2. 如果数据格式不同，可创建新 Dataset 类；否则直接使用 `CodaDataset`
3. 创建 YAML 配置（指定 `CLASS_NAMES`, `DATA_PATH`, `INFO_PATH`, `POINT_CLOUD_RANGE` 等）
4. 如需 GT 采样增强，运行 `generate_gt_database.py` 预生成 GT 数据库

### 添加新数据增强

1. 在 `DataAugmentor` 类中添加新方法（签名: `(self, data_dict=None, config=None)`）
2. `data_dict is None` → 返回 `functools.partial`（延迟执行）；否则执行增强
3. 在 `AUG_CONFIG_LIST` 中添加对应条目
