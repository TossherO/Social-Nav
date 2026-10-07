# SimpleDet3D 使用指南

## 环境准备

### 1. 推荐版本与宿主机检查

| 组件 | 推荐版本 |
|------|----------|
| Python | 3.10 |
| PyTorch | 2.5.1 + CUDA 12.4 wheel |
| 系统 CUDA Toolkit | 12.x |
| spconv | `spconv-cu120==2.3.6` |
| NumPy | 1.26.4 |

先确认 GPU、CUDA 编译器和主机编译器可用：

```bash
nvidia-smi
nvcc --version
gcc --version
```

`nvidia-smi` 失败通常属于驱动、容器 GPU 映射或集群资源分配问题，conda
环境不能修复内核态驱动。系统 CUDA 12.9 与 PyTorch CUDA 12.4 同属 CUDA 12
大版本，编译时可能出现 minor-version warning；大版本不一致则不能忽略。

### 2. 创建独立 conda 环境

```bash
conda create -n simpledet3d \
    python=3.10 pip setuptools wheel ninja packaging -y
conda activate simpledet3d

which python
python --version
python -m pip --version
```

如果默认 conda 环境目录不可写，可用 `conda create -p <可写路径>` 创建并通过
完整路径激活。后续安装命令统一使用 `python -m pip`，避免调用系统 Python 的
pip。

### 3. 使用 pip 安装 PyTorch 和依赖

```bash
python -m pip install --upgrade pip setuptools wheel

python -m pip install \
    torch==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

python -m pip install \
    numpy==1.26.4 \
    easydict==1.13 \
    PyYAML==6.0.2 \
    tqdm==4.67.1 \
    tensorboardX==2.6.2.2 \
    scikit-image==0.24.0 \
    numba==0.60.0 \
    pyquaternion==0.9.9 \
    matplotlib==3.9.2

python -m pip install spconv-cu120==2.3.6

python -m pip install --no-index torch-scatter \
    -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
```

项目当前的 `requirements.txt` 只表达宽松依赖范围，直接安装可能替换已经固定的
PyTorch，且其中没有完整约束 `torch-scatter` 和 `pyquaternion`，新环境应优先
使用上面的版本化命令。项目不依赖 `torchvision` 和 `torchaudio`。

检查 Python 和二进制包是否来自当前 conda 环境：

```bash
python - <<'PY'
import sys
import torch
import spconv
import torch_scatter

print('Python executable:', sys.executable)
print('PyTorch:', torch.__version__, torch.__file__)
print('PyTorch CUDA:', torch.version.cuda)
print('spconv:', spconv.__version__)
print('torch-scatter:', torch_scatter.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('Compute capability:', torch.cuda.get_device_capability(0))
PY
```

`sys.executable` 和 `torch.__file__` 都应位于当前 conda 环境。若
`torch.__file__` 指向 `/usr/local/lib/python3.12/...`，说明系统 Python 路径污染
了环境；应检查并清除当前终端中的 `PYTHONPATH`，再重新激活环境。

### 4. CUDA 环境和扩展编译

只需要让 `CUDA_HOME` 随 `simpledet3d` 环境激活，可使用 conda 的环境级变量，
不会影响其他 conda 环境：

```bash
conda activate simpledet3d
conda env config vars set CUDA_HOME=/usr/local/cuda
conda deactivate
conda activate simpledet3d

echo "$CUDA_HOME"
```

`PATH` 和 `LD_LIBRARY_PATH` 仅在编译命令中临时设置，避免退出该环境后污染其他
环境：

```bash
cd /home/heyushen/code/SimpleDet3d

PATH="$CUDA_HOME/bin:$PATH" \
LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}" \
MAX_JOBS=8 \
python setup.py build_ext --inplace --force

find simple_det3d/ops -name '*.so' -print
```

内存不足时将 `MAX_JOBS` 降到 `2` 或 `1`。仓库从其他设备迁移来的 `.so` 和
`build/` 产物可能使用旧 PyTorch ABI，必须在新环境中带 `--force` 重编译。正常
应生成 iou3d_nms、roiaware_pool3d、pointnet2_stack 和 pointnet2_batch 四个
Python 3.10 扩展。

项目不需要执行 `pip install -e .`，应直接在源码根目录运行脚本。

### 5. 环境验收

先验证四个扩展能够导入：

```bash
python - <<'PY'
from simple_det3d.ops.iou3d_nms import iou3d_nms_cuda
from simple_det3d.ops.roiaware_pool3d import roiaware_pool3d_cuda
from simple_det3d.ops.pointnet2.pointnet2_stack import pointnet2_stack_cuda
from simple_det3d.ops.pointnet2.pointnet2_batch import pointnet2_batch_cuda

print('four CUDA extensions: OK')
PY
```

再执行最小 CUDA 算子测试：

```bash
python - <<'PY'
import torch
from simple_det3d.ops.iou3d_nms import iou3d_nms_utils

assert torch.cuda.is_available()
boxes = torch.tensor(
    [[0., 0., 0., 2., 2., 2., 0.]], device='cuda', dtype=torch.float32
)
iou = iou3d_nms_utils.boxes_iou3d_gpu(boxes, boxes)
assert torch.allclose(iou, torch.ones_like(iou), atol=1e-4)
print('CUDA IoU:', iou)
PY

python tools/train.py --help
python tools/test.py --help
python tools/generate_coda_info.py --help
```

完整验收条件为：Python 3.10、PyTorch 2.5.1、CUDA 可用、spconv 和
torch-scatter 可执行 GPU 运算、四个自定义扩展可导入、CUDA IoU 测试通过，且
入口脚本能够显示帮助。如果扩展导入出现 `undefined symbol`，优先检查
`sys.executable`/`torch.__file__`，然后在正确环境中重新强制编译；
`torch_scatter` 报同类错误时，应从上述 PyG wheel 地址强制重装匹配版本。

## 数据准备

### 生成 pkl 信息文件

**CODa 数据集**（使用专用脚本 `tools/generate_coda_info.py`）：

在生成 16 线信息文件前，先将原始 OS1-128 点云按垂直通道降采样。脚本默认处理
全部 22 个序列，输出约为原始点云体积的八分之一；已存在且尺寸正确的文件会自动
跳过，可以安全断点续跑：

```bash
python tools/downsample_coda_lidar.py \
    --data_root /share/public_datasets/VLA/nitrogen/CODa_full \
    --workers 8

# 只处理指定序列
python tools/downsample_coda_lidar.py --seq_ids 0 1 2 --workers 4

# 显式重新生成已有结果
python tools/downsample_coda_lidar.py --overwrite --workers 8
```

默认从 `3d_comp/os1` 读取 `(128, 1024, 4)` 点云，在完整垂直视场内均匀保留
16 个通道，并写入 `3d_comp_downsample/os1`。每个输出文件包含 16,384 个
`[x, y, z, intensity]` 点。可使用 `--limit 1 --output_dir /tmp/coda_downsample`
执行单帧检查而不写入正式目录。

```bash
# 一次性生成 train/val/test 三个 pkl（默认数据路径）
python tools/generate_coda_info.py

# 或指定数据路径和输出目录
python tools/generate_coda_info.py \
    --data_root /share/public_datasets/VLA/nitrogen/CODa_full \
    --output_dir data/coda

# 使用降采样点云
python tools/generate_coda_info.py \
    --pc_dir 3d_comp_downsample/os1 \
    --output_dir data/coda_downsample \
    --filter_class Pedestrian

# 仅保留包含 Pedestrian 的帧
python tools/generate_coda_info.py --filter_class Pedestrian

# 可选参数
#   --sweep_frame_len 3      # 历史帧数（默认 3，不含当前帧）
#   --max_sweep_time_lag 0.35 # 最大历史时间差，单位秒
#   --seq_ids 0 1 2          # 只处理指定序列
#   --pc_dir 3d_comp/os1     # 点云子目录（默认 3d_comp/os1）
#   --filter_class Pedestrian # 只保留含指定类别的帧
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_root` | `/share/public_datasets/VLA/nitrogen/CODa_full` | 数据集根目录 |
| `--output_dir` | `data/coda` | 输出目录 |
| `--sweep_frame_len` | 3 | 最多保存的历史帧数，不含当前帧 |
| `--max_sweep_time_lag` | 0.35 | 历史帧最大时间差，单位秒 |
| `--seq_ids` | 全部 0-21 | 指定处理的序列 |
| `--pc_dir` | `3d_comp/os1` | 点云子目录，指向降采样版本时用 `3d_comp_downsample/os1` |
| `--filter_class` | 无（保留全部） | 只保留包含指定类别（Car/Pedestrian/Cyclist）的帧 |

**数据校验：** 脚本会自动跳过以下无效帧并在摘要中报告：
- 缺失或损坏的点云文件（`bin_missing` / `bin_corrupt_or_empty`）
- 缺失的标注文件（`annotation_file_missing`）
- 标注为空的帧（`empty_gt_boxes`）
- 类别过滤后无匹配的帧（`filter_no_<类名>`）
- 损坏的 box 数据（`corrupt_boxes_nan_inf`、`non_positive_dimensions`）

> 生成的 pkl 文件位于 `data/coda/` 目录：`coda_infos_train.pkl`、`coda_infos_val.pkl`、`coda_infos_test.pkl`。
> 新生成的 pkl 使用相对于 CODa 根目录的点云路径，并通过配置中的 `RAW_DATA_PATH` 解析。
> sweep 从目标帧之前的连续原始 LiDAR 帧构建，可以跨 metadata 的标注划分；帧号断点和超过时间上限的历史帧会被跳过。
> 旧 pkl 中的绝对路径仍会自动重映射，异常时间差也会在加载时过滤；但要获得连续、等密度的历史帧，仍建议重新生成 pkl。

**Wheelchair 数据集**（使用 `tools/generate_wheelchair_info.py`）：

```bash
# 默认参数（数据路径 /home/heyushen/data/wheelchair-dataset）
python tools/generate_wheelchair_info.py

# 指定参数
python tools/generate_wheelchair_info.py \
    --data_root /home/heyushen/data/wheelchair-dataset \
    --output_dir data/wheelchair \
    --sweep_frame_len 3 \
    --max_sweep_time_lag 0.35

# 可选: 启用 old1/old2 额外数据（编辑脚本中 CATEGORY_DIRS 取消注释）
```

> 生成的 pkl: `data/wheelchair/wheelchair_infos_train.pkl` (70,978帧) / `wheelchair_infos_val.pkl` (8,175帧)
> **注意**: Wheelchair 原始数据无标注，gt_boxes 为空。仅可用于推理，训练需要先补充标注。

### 生成 GT 数据库（仅 gt_sampling 增强需要）

```bash
python tools/generate_gt_database.py --cfg_file configs/voxelrcnn_coda.yaml --split train
```

## 训练

### 单卡训练

```bash
# 常用命令模板
python tools/train.py --cfg_file configs/centerpoint_coda.yaml
python tools/train.py --cfg_file configs/centerpoint_wheelchair.yaml
python tools/train.py --cfg_file configs/voxelrcnn_coda.yaml
python tools/train.py --cfg_file configs/voxelrcnn_wheelchair.yaml

# 16 线降采样 CODa（需先生成 data/coda_downsample 中的信息文件）
python tools/train.py --cfg_file configs/voxelrcnn_coda_16line.yaml

# 常用可选参数
python tools/train.py --cfg_file configs/centerpoint_coda.yaml \
    --batch_size 4 \
    --epochs 30 \
    --workers 8 \
    --precision fp16 \
    --extra_tag my_exp \
    --ckpt <path> \
    --pretrained_model <path>

# 训练中自动评估：每 5 个 epoch 在 val 集上评估一次
python tools/train.py --cfg_file configs/voxelrcnn_coda.yaml --eval_interval 5

# 减少检查点保存频率：每 5 个 epoch 保存一次，最多保留 10 个
python tools/train.py --cfg_file configs/centerpoint_coda.yaml \
    --ckpt_save_interval 5 --max_ckpt_save_num 10
```

### 多卡训练（DDP）

使用 `torchrun` 启动，`--batch_size` 为**每张卡**的 batch size：

```bash
# 4 卡训练
torchrun --nproc_per_node=4 tools/train.py --cfg_file configs/voxelrcnn_coda.yaml

# 4 卡训练 + 评估
torchrun --nproc_per_node=4 tools/train.py \
    --cfg_file configs/voxelrcnn_coda.yaml \
    --batch_size 8 --eval_interval 5

# 指定 GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 tools/train.py \
    --cfg_file configs/voxelrcnn_coda.yaml
```

多卡训练特性：
- 使用 `DistributedDataParallel`（比 `DataParallel` 更高效）
- 日志、TensorBoard、checkpoint 保存仅在 rank 0 执行
- val 评估仅在 rank 0 执行
- 自动使用 `DistributedSampler` 分配数据，每张卡处理不同子集

**数据加载优化**（`--workers` / `persistent_workers`）：
- 默认 `--workers 4`，DataLoader 自动启用 `persistent_workers`
- 当 MAX_SWEEPS 较大时建议增加 worker 数，例如 `--workers 8`
- `persistent_workers` 让 worker 进程跨 epoch 存活，消除每 epoch 的 fork + 数据反序列化开销（约 3-5s/epoch）

**检查点管理**（`--ckpt_save_interval` / `--max_ckpt_save_num`）：
- 默认每个 epoch 保存一次，最多保留 30 个
- 训练轮数较多（如 30+ epoch）时可增大间隔以节省磁盘空间
- 旧检查点按修改时间排序后自动删除
- epoch checkpoint 同时保存历史 `best_metric`、对应 epoch 和指标名称；恢复时会
  与 `best_model.pth` 的元数据比较并取较优者，避免重启后用更差模型覆盖历史最佳
- 旧 checkpoint 没有最佳指标元数据时仍可恢复模型、优化器、GradScaler 和 EMA，
  但最佳模型跟踪会从恢复后的第一次验证重新开始

**训练中评估**（`--eval_interval`）：
- 每隔 N 个 epoch 在验证集上运行完整评估（AP3D/APBEV）
- 指标记录到 TensorBoard（`eval/` 前缀）和日志文件
- 最佳模型（所有类别 AP3D 均值最高）自动保存为 `ckpt/best_model.pth`
- 训练结束后会用最佳模型做最终评估
- 设为 `0` 或不指定则跳过（默认）

输出目录：`output/<配置名>/<extra_tag>/`（包含 `ckpt/`、`tensorboard/`、日志）

### 混合精度训练

训练和推理统一支持 `fp32`、`fp16`、`bf16`。默认读取 YAML 顶层的
`PRECISION`（当前配置默认均为 `fp32`），命令行 `--precision` 的优先级更高：

```bash
# FP16：启用 autocast 和动态 GradScaler
python tools/train.py --cfg_file configs/voxelrcnn_coda.yaml \
    --precision fp16 --extra_tag fp16

# BF16：H800 可用，稠密算子动态范围更大
python tools/train.py --cfg_file configs/voxelrcnn_coda.yaml \
    --precision bf16 --extra_tag bf16
```

`--use_amp` 仍可使用，但只是 `--precision fp16` 的兼容别名，不应与
`--precision fp32` 或 `--precision bf16` 同时传入。混合精度采用 FP32 主权重；
请不要额外调用 `model.half()`。混合精度 checkpoint 会保存 GradScaler 状态，使用
`--ckpt` 恢复时会一并恢复，旧的 FP32 checkpoint 也可直接用于 FP16/BF16 推理。

注意：spconv 2.3.6 的旧 autocast 实现会在 BF16 模式下仍以 FP16 执行稀疏卷积，
所以本项目对 FP16 和 BF16 训练都启用动态 GradScaler。若更关注依赖兼容性与吞吐，
优先使用 `fp16`；若更关注稠密网络部分的动态范围，可在 H800 上选择 `bf16`。
scaler 初值设为 32（已在两种模型上验证首步可更新），之后会自动动态调整。

## 测试/推理

```bash
# 评估
python tools/test.py --cfg_file configs/centerpoint_coda.yaml \
    --ckpt output/centerpoint_coda/default/ckpt/checkpoint_epoch_20.pth \
    --precision fp16

# 测速
python tools/test.py --cfg_file configs/centerpoint_coda.yaml \
    --ckpt <path> --infer_time --precision bf16

# BEV 可视化（保存 PNG）
python tools/test.py --cfg_file configs/centerpoint_coda.yaml \
    --ckpt <path> --save_vis --vis_output_dir vis_results
```

测试精度与 checkpoint 的训练精度相互独立；同一份 FP32 主权重 checkpoint 可以用
`--precision fp32`、`fp16` 或 `bf16` 评估。结果转为 NumPy 前会统一恢复为 FP32。

## Wheelchair 弱监督迁移

### 1. 审计并生成弱标签 info

先运行少量帧 dry-run，确认当前数据文件均可读取：

```bash
python tools/prepare_wheelchair_weak_data.py \
    --dry-run --max-frames-per-scene 3
```

人工标注键已经确认是主场景 pkl 中 `frames` 列表的零基下标，生成器会按
`frames[k].frame_index` 找到原始点云帧。可用三联图复查某一帧：左侧为
全局 BEV，中间为人工中心附近的局部 BEV，右侧为最高分合法伪框的局部侧视图：

```bash
MPLBACKEND=Agg python tools/visualize_wheelchair_weak_labels.py \
    --category indoor2 --scene sequence4_003 --frame 117
```

红色叉号为人工 BEV 中心，橙色加号为原伪中心，绿色为通过几何合法性检查的
原伪框。已完成的跨场景审计结果如下：

- 全部 9,705 个标注键都是对应场景 `frames` 的合法下标；81.88% 的键对应帧号
  不等于旧实现使用的 `k+1`。
- 修正后 weak-val 人工中心到最近点的中位距离为 0.018 m，全部中心在 0.2 m
  内存在点云回波；旧映射分别为 0.419 m 和 25.33%。
- 跨 884 帧抽样 1,200 个高分合法伪框，尺寸 `[3:6]` 的中位数为
  `[0.477, 0.323, 1.515] m`，明确表明第三维为高度。
- 采用 `z±dz/2` 时包含 96.46% 的框内 BEV 点；若误按底面中心 `[z,z+dz]`
  解释则仅包含 58.53%。因此已确认原始格式为
  `[x,y,z,dx,dy,dz,yaw]`，且 `z` 是几何中心。

```bash
python tools/prepare_wheelchair_weak_data.py \
    --data-root /home/heyushen/data/wheelchair-dataset \
    --annotation-root /home/heyushen/data/wheelchair_info \
    --output-dir data/wheelchair_weak
```

输出包括全部训练场景、纯室内训练场景、室内穷举/非穷举中心子集、过渡场景、
室内穷举中心验证集和 JSON 审计报告。关键分层文件为：

```text
wheelchair_weak_infos_train_indoor_exhaustive.pkl
wheelchair_weak_infos_train_indoor_non_exhaustive.pkl
wheelchair_weak_infos_val_indoor_exhaustive.pkl
```

人工 BEV 点与原有伪 3D 框是互补标注：已有框的目标不会重复点击，因此两类标签
独立保留，不按中心距离匹配、修正、降级或跨类型去重。人工标注帧中二者并集视为
中心/类别穷举标签，可以监督背景；这不会提高伪框 `z/size/yaw` 的可信度。伪框仍
仅经过几何、距离相关框内点数、图像分数和双向时序过滤，不可信框降级为中心监督。

所有过滤阈值都可通过 `--help` 查看和覆盖，并会完整写入
`wheelchair_weak_summary.json`。人工标注的 `frames[k]` 映射和原始伪框
`[x,y,z,dx,dy,dz,yaw]`/几何中心 `z` 均已通过上述结构、点云统计和可视化确认。
当前 `data/wheelchair_weak` 已按正确映射重建；修复前训练的三阶段模型仍须重新训练。

### 2. 三阶段训练

推荐使用仓库根目录的一键脚本。默认使用 8 张 GPU、FP32、单卡 batch size 16，
并从现有的 CODA-16line 四帧模型开始，依次完成 10/10/5 epoch 三个阶段：

```bash
conda activate simpledet3d
cd /home/heyushen/code/SimpleDet3d

# 先检查全部路径并打印三条实际命令，不启动训练
DRY_RUN=1 ./train_wheelchair_3stage.sh

# 正式启动
./train_wheelchair_3stage.sh
```

三个阶段使用相同的 `RUN_TAG`，但配置名不同，因此输出目录彼此独立。阶段 2
自动读取阶段 1 的最终 checkpoint；阶段 3 优先读取阶段 2 按
`CenterF1@0.50m` 选出的 `best_model.pth`，未生成最佳模型时自动退回阶段 2
最终 checkpoint。默认输出为：

```text
output/voxelrcnn_wheelchair_weak_stage1/wheelchair_exhaustive_center/
output/voxelrcnn_wheelchair_weak_stage2/wheelchair_exhaustive_center/
output/voxelrcnn_wheelchair_weak_stage3/wheelchair_exhaustive_center/
```

同一 `RUN_TAG` 重复运行时，已有最终 checkpoint 的阶段会被跳过，未完成阶段会
自动从其最新 checkpoint 续训。新实验应使用新的标签，避免复用旧结果：

```bash
RUN_TAG=wheelchair_run ./train_wheelchair_3stage.sh
```

单帧消融需要同时覆盖 Wheelchair 和混合训练中的 CODA 输入，并使用对应的
CODA-16line 单帧预训练权重。`MAX_SWEEPS=1` 表示只加载当前帧：

```bash
MAX_SWEEPS=1 \
CODA_CKPT=output/voxelrcnn_coda_16line/finetune_1frame_from_1frame/ckpt/checkpoint_epoch_20.pth \
RUN_TAG=wheelchair_exhaustive_center_1frame_10_10_5 \
./train_wheelchair_3stage.sh
```

必须使用新的 `RUN_TAG`，不能从四帧实验目录断点恢复。除输入 sweep 数和初始化
checkpoint 外，其余三阶段数据、损失和评估设置保持不变，便于与四帧结果公平比较。

常用覆盖参数如下；`BATCH_SIZE` 始终表示每张 GPU 的 batch size：

```bash
# 指定 GPU 数量、降低单卡 batch，并改用自定义 CODA 初始化权重
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
BATCH_SIZE=8 \
WORKERS=2 \
CODA_CKPT=/path/to/coda_16line_checkpoint.pth \
RUN_TAG=wheelchair_4gpu \
./train_wheelchair_3stage.sh

# 覆盖精度、轮数或验证间隔
PRECISION=fp16 \
STAGE1_EPOCHS=8 STAGE2_EPOCHS=10 STAGE3_EPOCHS=3 \
STAGE2_EVAL_INTERVAL=2 STAGE3_EVAL_INTERVAL=1 \
./train_wheelchair_3stage.sh
```

默认 `SKIP_COMPLETED=1`；设为 `0` 时不会主动跳过，但训练入口仍会从同一输出
目录自动恢复。脚本失败后保留相同 `RUN_TAG` 重新执行即可。阶段 2 开启 EMA
Teacher 并每轮验证，阶段 3 同样每轮验证；两者都会生成或更新
`best_model.pth`。

以下是与一键脚本等价的分阶段手动命令，适合单独调试某个阶段。

阶段 1 使用 CODA 与全部 Wheelchair 混合训练：

```bash
torchrun --standalone --nproc_per_node=8 tools/train.py \
    --cfg_file configs/voxelrcnn_wheelchair_weak_stage1.yaml \
    --pretrained_model <CODA-16line四帧checkpoint> \
    --extra_tag stage1_mixed
```

阶段 2 固定使用 25% CODA、50% 室内穷举中心帧、25% 室内非穷举帧，并启用
EMA Teacher。穷举帧训练 Dense 背景和 ROI 中心分类；Teacher 只升级已有弱中心
附近、同时通过几何与点数检查的框，默认不自行添加无锚点伪框：

```bash
torchrun --standalone --nproc_per_node=8 tools/train.py \
    --cfg_file configs/voxelrcnn_wheelchair_weak_stage2.yaml \
    --pretrained_model output/voxelrcnn_wheelchair_weak_stage1/stage1_mixed/ckpt/checkpoint_epoch_15.pth \
    --extra_tag stage2_indoor_ema --eval_interval 1
```

阶段 3 仅使用室内穷举中心帧做短周期精度固化：

```bash
torchrun --standalone --nproc_per_node=8 tools/train.py \
    --cfg_file configs/voxelrcnn_wheelchair_weak_stage3.yaml \
    --pretrained_model output/voxelrcnn_wheelchair_weak_stage2/stage2_indoor_ema/ckpt/best_model.pth \
    --extra_tag stage3_indoor
```

穷举中心验证使用“人工中心 + 同帧原始伪标签中心”的并集，报告
`CenterPrecision/Recall/F1@0.5m/1.0m`、FP/frame、中心误差和范围覆盖率；仍不报告
依赖可靠尺寸/yaw的 3D/BEV AP。`best_model.pth` 默认按 `CenterF1@0.50m` 选择。
Wheelchair 三阶段配置使用最终 `SCORE_THRESH=0.3`、评估
`CENTER_SCORE_THRESH=0.3`，并把模型最终评估/推理
`NMS_THRESH` 设为 `0.2`，以保留拥挤场景中的相邻行人。该覆盖不改变 Dense Head
proposal NMS 或 ROI Head 的训练/测试 proposal NMS；Stage 2/3 通过配置继承自动
获得相同设置。
阶段 2 的 EMA Teacher 会增加一次完整推理，训练显存和时间均高于阶段 1。
如需诊断动态点云造成的卡间显存差异，可在一键脚本中启用逐 rank 日志：

```bash
LOG_GPU_MEMORY=1 RUN_TAG=<新实验名> ./train_wheelchair_3stage.sh
```

也可在手动命令中添加 `--log_gpu_memory`。日志会输出每张卡当前 batch 的点数、
体素限流前后数量，以及 CUDA allocated/reserved/peak memory。该选项每个日志
间隔执行一次小规模 DDP 汇总，排障完成后可关闭。

阶段 2 的普通 epoch checkpoint 同时保存 Student 的 `model_state` 与
`ema_model_state`，使用 `--ckpt` 恢复时两者都会恢复。弱验证默认评估 EMA
Teacher，因此 `best_model.pth` 的主 `model_state` 也是被实际评估的 EMA 权重，
可直接用于下一阶段的 `--pretrained_model`。阶段 3 关闭 EMA，显式冻结 Dense
Head 的 `center_z/dim/rot` 与 ROI regression 分支，并强制目标域仅计算 xy 回归；
ROI classifier、共享特征和 backbone 继续训练以压制室内误检。

当前在线 EMA 是安全优先的第一版：它在同一增强坐标系中、仅对已有中心锚点
进行几何升级；尚未实现弱/强双视图一致性、无锚点 Teacher 新框、固定 Source
Teacher 蒸馏和周期伪标签记忆。没有额外验证前请保持 `ALLOW_UNMATCHED: false`，
因此无标注帧当前主要参与传感器统计适配，不会被自动当作负样本。

### 3. 推理结果 BEV 人工审计

`visualize_wheelchair_predictions.py` 会加载真实点云和训练 checkpoint，在测试集上
推理，并生成同尺度的“现有弱真值/模型预测”双视图 PNG、逐帧 CSV、原始预测 pkl
和汇总 JSON。
默认使用当前四帧三阶段实验的阶段 3 `best_model.pth`，并优先展示漏掉人工中心的帧：

```bash
conda activate simpledet3d
cd /home/heyushen/code/SimpleDet3d

python tools/visualize_wheelchair_predictions.py
```

默认结果保存在 `output/wheelchair_prediction_vis/<实验名_时间戳>/`。PNG 位于其中的
`images/`，图片文件名与 `selected_frames.csv` 的 `image_name` 列一一对应；可通过
远程 IDE 或下载到本地查看。常用抽样方式如下：

```bash
# 时间上均匀抽取 50 帧，显示 4 帧融合中的历史点
python tools/visualize_wheelchair_predictions.py \
    --selection uniform --num_samples 50 --show_history

# 优先看预测框多的复杂帧
python tools/visualize_wheelchair_predictions.py \
    --selection crowded --num_samples 40

# 只检查一个场景，或精确检查一帧
python tools/visualize_wheelchair_predictions.py \
    --scene indoor1/sequence2_001 --selection uniform --num_samples 20
python tools/visualize_wheelchair_predictions.py \
    --frame_id indoor1/sequence2_001/0001 --selection all

# 指定其他阶段的配置和权重
python tools/visualize_wheelchair_predictions.py \
    --cfg_file configs/voxelrcnn_wheelchair_weak_stage3.yaml \
    --ckpt output/voxelrcnn_wheelchair_weak_stage3/wheelchair_exhaustive_center_10_12_5/ckpt/best_model.pth \
    --selection uniform --num_samples 40

# 单帧 checkpoint 必须同时覆盖输入帧数，避免按 YAML 默认四帧进行推理
python tools/visualize_wheelchair_predictions.py \
    --cfg_file configs/voxelrcnn_wheelchair_weak_stage3.yaml \
    --ckpt output/voxelrcnn_wheelchair_weak_stage3/wheelchair_exhaustive_center_1frame_10_10_5/ckpt/best_model.pth \
    --max_sweeps 1 \
    --selection uniform --num_samples 40

# 后处理消融：覆盖最终 NMS，并按当前帧点数过滤历史残影候选
python tools/visualize_wheelchair_predictions.py \
    --cfg_file configs/voxelrcnn_wheelchair_weak_stage3.yaml \
    --ckpt <stage3-best-model.pth> \
    --final_nms_thresh 0.2 \
    --filter_by_point_support \
    --selection missed --num_samples 20
```

左右两图使用相同点云、范围和坐标比例。左图仅绘制现有标签：红色叉号是人工 BEV
中心，橙色虚框和加号是过滤后的弱 3D 框及非人工弱中心；右图仅绘制预测，框标签为
`置信度|当前帧框内点数`，点数小于 `--min_box_points` 时使用虚线。灰色点为当前帧，
浅蓝点（需 `--show_history`）为历史帧。可用 `--score_thresh` 调整显示阈值，默认 0.3。
`--max_sweeps` 覆盖本次推理使用的总帧数，`0/1` 都表示仅使用当前帧；该参数应与
checkpoint 的训练输入保持一致。
`--final_nms_thresh` 只覆盖本次推理的最终 NMS，便于固定 checkpoint 做消融；
`--filter_by_point_support` 使用与 EMA Teacher 相同的距离自适应当前帧点数规则
（默认 8m 内 5 点、8～15m 3 点、更远 1 点）。启用点过滤时，过滤后的结果写入
`predictions.pkl`，未过滤的模型输出另存为 `raw_predictions.pkl`。

必须注意：只有人工补全帧上的“人工中心 + 原始伪标签中心”并集近似穷举；在这些
验证帧上可把未匹配预测计为中心级假阳性。其余非穷举Wheelchair帧中的未匹配预测
仍只能列为“待复核”，不能用于计算precision。任何中心指标都不能替代3D/BEV AP。

### 4. 单帧与四帧三栏对比

先分别用 `visualize_wheelchair_predictions.py` 保存两套模型的全验证集
`predictions.pkl`，再运行：

```bash
python tools/compare_wheelchair_predictions.py \
    --single_predictions output/wheelchair_prediction_vis/exhaustive_center_1frame_10_10_5_missed20/predictions.pkl \
    --four_predictions output/wheelchair_prediction_vis/exhaustive_center_10_12_5_missed20/predictions.pkl \
    --num_samples 20 \
    --output_dir output/wheelchair_prediction_vis/comparison_1frame_vs_4frame_balanced20
```

每张 PNG 的三栏依次为可用真值/弱标签、单帧模型预测和四帧模型预测。默认样本由
单帧占优、四帧占优、命中相同但单帧预测更少、时间均匀抽样各占四分之一；也可
重复传入 `--frame_id <场景/帧>` 精确比较指定帧。三栏只绘制同一份当前帧点云，
但输入预测文件必须分别来自 `MAX_SWEEPS=1` 和 `MAX_SWEEPS=4` 的正确模型推理。

### 5. 完整序列单帧/四帧对比视频

使用两次全序列推理保存的 `predictions.pkl` 可生成“真值/可用弱标签、
单帧预测、4 帧预测”三联 MP4：

```bash
python tools/create_wheelchair_comparison_video.py \
    --root_path data/wheelchair_sequence/indoor2_sequence4_003_full \
    --raw_data_path /home/heyushen/data/wheelchair-dataset \
    --info_path wheelchair_weak_infos_train_all.pkl \
    --single_predictions output/wheelchair_sequence_video/sequence4_003_single_inference/predictions.pkl \
    --four_predictions output/wheelchair_sequence_video/sequence4_003_four_inference/predictions.pkl \
    --output_file output/wheelchair_sequence_video/indoor2_sequence4_003_1frame_vs_4frame.mp4 \
    --fps 10
```

视频按 info 中的帧顺序生成，且要求 info 与两份预测的 `frame_id` 集合
完全一致，任何缺帧、多帧或重复都会直接报错。已人工复核帧标为绿色
`REVIEWED`；其余帧用红色 `INCOMPLETE` 提示弱标签仍可能漏标。默认只保存
MP4 和摘要 JSON；调试时可加 `--keep_frames` 保留中间 PNG。若 ffmpeg 不在
`PATH` 中，使用 `--ffmpeg /path/to/ffmpeg` 显式指定。

## 配置文件速查

| 配置文件 | 模型 | 数据集 | 类别 |
|----------|------|--------|------|
| `configs/centerpoint_coda.yaml` | CenterPoint | CODa | Car, Pedestrian, Cyclist |
| `configs/centerpoint_wheelchair.yaml` | CenterPoint | Wheelchair | Pedestrian, Cyclist |
| `configs/voxelrcnn_coda.yaml` | VoxelRCNN | CODa | Car, Pedestrian, Cyclist |
| `configs/voxelrcnn_coda_16line.yaml` | VoxelRCNN | CODa 16 线降采样 | Pedestrian |
| `configs/voxelrcnn_wheelchair.yaml` | VoxelRCNN | Wheelchair | Pedestrian, Cyclist |
| `configs/voxelrcnn_wheelchair_weak_stage1.yaml` | VoxelRCNN | CODa + Wheelchair 弱监督 | Pedestrian |
| `configs/voxelrcnn_wheelchair_weak_stage2.yaml` | VoxelRCNN | 三路采样 + 穷举背景 + EMA | Pedestrian |
| `configs/voxelrcnn_wheelchair_weak_stage3.yaml` | VoxelRCNN | 穷举中心精度固化 | Pedestrian |

配置结构（修改配置时关注这些区域）：

| Section | 用途 |
|---------|------|
| `CLASS_NAMES` | 训练/评估的目标类别 |
| `DATA_CONFIG.DATA_PATH` | pkl 信息文件和 GT 数据库目录 |
| `DATA_CONFIG.RAW_DATA_PATH` | 原始点云数据集根目录 |
| `DATA_CONFIG.POINT_CLOUD_RANGE` | 点云裁剪范围 |
| `DATA_CONFIG.INFO_PATH` | pkl 信息文件路径 |
| `DATA_CONFIG.MAX_SWEEPS` | 总输入帧数（含当前帧）；当前推荐 4，0/1 为仅当前帧 |
| `DATA_CONFIG.MAX_SWEEP_TIME_LAG` | 允许加载的最大历史时间差（秒）；当前推荐 0.35 |
| `DATA_CONFIG.SWEEP_SELECT_MODE` | sweep 帧选取方式：`near`（默认，按真实时间差取最近帧）/ `random`（随机抽样） |
| `DATA_CONFIG.DATA_AUGMENTOR.AUG_CONFIG_LIST` | 数据增强策略 |
| `DATA_CONFIG.DATA_PROCESSOR[].VOXEL_SIZE` | 体素尺寸 |
| `MODEL.NAME` | CenterPoint / VoxelRCNN |
| `OPTIMIZATION.BATCH_SIZE_PER_GPU` | 单卡 batch size |
| `OPTIMIZATION.NUM_EPOCHS` | 训练轮数 |
| `OPTIMIZATION.LR` | 学习率 |

## 常见问题

**CUDA 扩展编译失败**：确认 CUDA Toolkit 与 PyTorch CUDA 版本匹配（当前环境 CUDA 12.x）。

**找不到数据文件**：使用 `--raw_data_path` 覆盖原始点云根目录；`--root_path` 仅覆盖 pkl/GT 数据库目录：
```bash
python tools/train.py --cfg_file configs/centerpoint_coda.yaml \
    --raw_data_path /share/public_datasets/VLA/nitrogen/CODa_full
```

**显存不足或卡间差异大**：VoxelRCNN 的动态体素化会真正执行
`DATA_PROCESSOR.transform_points_to_voxels.MAX_NUMBER_OF_VOXELS`，其数值表示
每个样本的非空体素上限（默认 train/test 为 60000/80000）。先使用
`--log_gpu_memory` 区分点/体素负载与 CUDA 缓存；仍不足时依次降低单卡 batch
size 和体素上限。不要在每个 iteration 调用 `torch.cuda.empty_cache()`。

**恢复训练**：
```bash
python tools/train.py --cfg_file configs/centerpoint_coda.yaml --ckpt output/.../checkpoint_epoch_10.pth
```

**TensorBoard**：
```bash
tensorboard --logdir output/centerpoint_coda/default/tensorboard
```
