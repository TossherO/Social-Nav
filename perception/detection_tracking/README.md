# 行人检测与跟踪

检测训练、离线推理与ROS 1接入统一放在本目录。核心处理相同的`Frame`，在线与离线不维护两套检测/跟踪算法。原SimpleDet3d、原始数据、训练输出和权重均不修改。

## 实现与边界

- `training/`：迁入原检测副本，源码历史清单不改写；不是推理服务。
- `inference/detector.py`：常驻单帧VoxelRCNN，复用副本预处理与模型，不读info pickle或GT。CUDA/Torch只在真实检测构造时导入。
- `inference/tracker.py`、`geometry.py`：CPU KF、Hungarian、旋转3D IoU/GIoU/DIoU，真实`dt`、秒级生命周期、中心距离门控；不依赖MMCV、FilterPy或ROS。
- `inference/dataset.py`：Wheelchair原始序列、通用JSONL帧清单、绑定输入指纹的检测缓存回放。
- `inference/pipeline.py`：坐标转换、可靠定位门控、失效/跳变重置、生命周期与共享快照转换。
- `inference/ros_inputs.py`：可独立测试的PointCloud2解码、定位缓存与单待处理帧队列。
- `ros/`：ROS 1 Python 3节点、消息与launch。本机没有ROS，catkin构建/真实运行尚未验证。

不包含定位/SLAM、地图、2D视觉检测、ReID、VLM在线渲染、导航或底盘控制。RGB作为时间匹配辅助输入保留，ROS发布选中的原Image；尚未实现3D框投影或VLM适配。

## 配置和指定模型

[inference.json](configs/inference.json)保存公共策略，[detector.yaml](configs/detector.yaml)明确部署模型。只读取忽略的`configs/detection_tracking.local.json`作为机器资源覆盖，不扫描其他local文件/API凭证。CLI权重优先于资源覆盖；ROS显式topic参数优先于配置。

本机使用用户指定Stage 3目录的`ckpt/best_model.pth`，实际路径登记在本地资源配置；历史日志记录最佳epoch为4、中心F1@0.50m为0.8772，不是新的完整3D框/跟踪成绩。

部署匹配实际训练日志而非训练YAML当前默认值：

| 项目 | 初始部署值 |
| --- | --- |
| 模型／类别 | VoxelRCNN／Pedestrian |
| 输入 | 单帧，`MAX_SWEEPS=1`，原始点云XYZI |
| 特征 | `[x,y,z,intensity/255,time_lag=0]` |
| 自车点剔除 | 删除`abs(x)<1 && abs(y)<1`的点，与训练一致 |
| 空间范围 | `[-4.8,-9.6,-1.5,14.4,9.6,4.5]`米 |
| 体素／精度 | `[0.05,0.05,0.2]`米／FP32 |
| 最终NMS | 0.2 |
| 候选／轨迹出生阈值 | 0.1／0.3 |

候选阈值0.1是低分支持关联适配，**不同于原日志最终检测阈值0.3**。低分框不创建轨迹、不作为KF测量；只支持已有轨迹，输出仍为`predicted`，最后真实观测时间不刷新。不能直接引用原成绩评价这套输出。

权重用`weights_only=True`加载，检查全部参数匹配，拒绝不完整加载。记录checkpoint摘要、配置/继承配置摘要、精度及二进制来源。模型尺寸/yaw来自弱监督与源域先验，不是人工完整3D真值，yaw也不是视线。

本机显式复用原检测仓库已编译的四个扩展，通过`extension_root`选与Python ABI匹配的二进制；Python模型只用本仓库副本，不自动编译/安装/修改环境。其他机器可单独编译副本扩展并取消该覆盖。

## 定位与时间

**原始ROS里程计不准确，不作为可靠定位。** 在线须接额外定位算法，本模块不实现定位。离线使用自带LiDAR位姿，校验有限值、四元数和刚体矩阵；缺失/损坏不换成单位矩阵，不按场景名猜平台静止。

- 检测保留在输入LiDAR系；跟踪使用定位输出的连续局部世界系，ROS初始名为`odom`，数据集为`dataset_world`。名称本身不证明定位精度。
- 定位对应车体时必须提供`localization_child_from_lidar`；同名LiDAR系才能明确使用单位外参。
- ROS仅选采集时刻之前且在`max_pose_age_s`内的定位，发布pose年龄，不用最新位姿替代，不读取未来轨迹平滑。
- 缺定位时仍给检测，但轨迹为空、`health=degraded`。定位丢失/跳变、序列/坐标系变化、大时间断点清空轨迹并改变生命周期，不能把自车运动/重定位当行人速度。
- roll/pitch后的yaw-only框变成重力对齐的保守直立包络，不宣称精确全姿态框。
- 速度为世界系运动，观测不足为`null`；旋转到车体系不自动变成相对速度。

Wheelchair当前按帧号与显式`nominal_rate_hz=10`估时，记录`timestamp_source=nominal_frame_index`。无真实时间时不宣称速度精确，可用`--timestamps`传JSON帧号到采集秒的映射。RGB同帧对应是数据集约定，不是重新核验过的硬件同步。检测单帧、跟踪时序、VLM三帧观察互不混淆。

## 离线运行

从仓库根运行。CPU检查/回放用既有NumPy/SciPy；真实检测用独立检测环境与GPU，本机入口见忽略的环境记录。目录必须是新的。

只读核对输入，不加载模型、不写结果：

```bash
python -m perception.detection_tracking.inference \
  --scene indoor2/sequence1_001 --start 30 --limit 3 --dry-run
```

检测＋跟踪（本机已配置权重和二进制位置）：

```bash
python -m perception.detection_tracking.inference \
  --scene indoor2/sequence1_001 --start 30 --limit 30 \
  --out outputs/perception/indoor_people_new
```

CPU缓存回放，不加载Torch/checkpoint：

```bash
python -m perception.detection_tracking.inference \
  --scene indoor2/sequence1_001 --start 30 --limit 30 \
  --detections-jsonl outputs/perception/indoor_people_new/observations.jsonl \
  --out outputs/perception/indoor_people_replay_new
```

`--data-root/--checkpoint/--extension-root`可覆盖，默认数据通过`data/datasets/wheelchair_raw`软链接。拒绝覆盖现有输出或写入共享数据/当前训练run。单帧推理错误记录invalid并停止，不重试或换模型；输入读取错误也停止。退出码0=全部valid、2=有degraded、1=失败。

产物为`settings.json`、`observations.jsonl`、`summary.json`，不复制点云/图片/权重。包括传感器检测、世界轨迹、位姿来源、健康、真实观测时间、输入指纹和耗时；缓存校验点云字节/形状、观察/序列ID、时间和传感器坐标。

通用数据集/rosbag导出可用JSONL清单（暂不内置CODa/JRDB专用转换器）：

```json
{"observation_id":"scene/0001","sequence_id":"scene","stamp_s":1.0,"timestamp_source":"sensor","pointcloud":"points/0001.bin","sensor_frame":"lidar","world_frame":"local_world","pose_source":"dataset_pose","pose_health":"valid","world_from_lidar":[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}
```

示例单位矩阵仅说明格式，不给缺位姿样本填假值。点文件为little-endian float32 XYZI，路径相对清单；可附`rgb_asset_uri/rgb_stamp_s`。用`--manifest path/to/frames.jsonl`代替`--scene`；同一序列重复/倒退时间报错，序列切换重置。

## ROS 1运行

需要兼容的ROS 1 Python 3、catkin和检测依赖。本机无ROS，本轮不安装或改变Python/Torch/spconv。若ROS与检测环境不兼容，需要单独桥接worker；**该双进程IPC尚未实现**，不假定复制PYTHONPATH即可解决ABI问题。

工作空间示例（不自动执行，已有链接不覆盖）：

```bash
mkdir -p /path/to/catkin_ws/src
ln -s /path/to/Social-Nav/perception/detection_tracking/ros \
  /path/to/catkin_ws/src/social_nav_people
cd /path/to/catkin_ws
catkin_make
source devel/setup.bash
roslaunch social_nav_people people.launch \
  project_root:=/path/to/Social-Nav localization_topic:=/localization/odom
```

package.xml的`0.0.0`仅满足catkin必填元数据，不给文件/产物添加发布编号。maintainer/许可证仍需所有者补齐，本轮不选择项目开源许可证。

| 项目 | 占位配置／规则 |
| --- | --- |
| 点云／RGB | `/points_raw`／`/camera/image_raw` |
| 原始里程计 | `/odom`仅记录，**不订阅、不用于定位** |
| 定位输出 | 默认null，配置额外定位算法的实际topic |
| 定位消息 | `odometry/pose_covariance/pose_stamped`三选一 |
| 定位frame | parent/child和固定外参必须一致；配置声明来源信任，不证明算法精度 |
| 固定标定 | `localization_child_from_lidar`用于坐标；`camera_calibration`为未来投影保留 |
| 输出 | `/perception/people`（People）、`/json`、`/rgb`、`/status` |

配置/本地覆盖可改全部字段，节点参数可覆盖topics。点云按字段偏移、类型、大小端和行填充解析，缺XYZ/intensity或非法消息拒绝；回调缓存，worker串行，最多一个待处理输入，忙时换掉旧待处理帧并统计丢帧。过旧输入、定位超时、乱序/时间回退明确处理；watchdog不是安全认证。

People.header保留采集时间及轨迹坐标系，检测使用独立detection_frame_id；has_localization/has_velocity/has_pose_age/has_rgb区分缺失与零。TrackState协方差为行优先10×10，顺序`x,y,z,dx,dy,dz,yaw,vx,vy,vz`，混合米/弧度及其导数单位。预测保留最后真实观测时间，confirmed不等于当前观测到；空轨迹/valid不代表自由空间或可行驶。

## 来源与验证

参考用户`simpletrack.zip`，SHA-256为`ba42dcf8b3bbcedb9996c2639095c5da408807d22b08829c876334fa54de1b99`；上游[3D SimpleTrack](https://github.com/tusen-ai/SimpleTrack)。未复制带缓存的整个包；保留核心方法并改写时间、CPU关联、配置和生命周期，**不声称逐字复现或原基准成绩**。

针对性CPU测试：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.test_detection_tracking
```

本轮用H800、既有simpledet3d环境和指定checkpoint完成以下检查，未训练/升级库：

| 检查 | 结果与证据 |
| --- | --- |
| `outdoor1/start_003`前12帧 | 12帧均有效，使用数据集位姿；高置信行人少，仅说明链路可运行；`outputs/perception/wheelchair_detection_tracking_smoke/` |
| `indoor2/sequence1_001`第31–54帧 | 24帧有效，保持同一轨迹ID，末帧confirmed且有速度估计；`outputs/perception/wheelchair_people_tracking_smoke/` |
| 上述24帧CPU缓存回放 | 全部有效，框与协方差数值一致；`outputs/perception/wheelchair_people_tracking_replay/` |

这些不是性能基准、身份正确性或精度评测。轨迹仍需人工视频审计/专用验证集；ROS构建/实机、定位精度、时钟/标定、VLM接入与闭环安全尚未验证。

本轮最终根级离线回归157项通过，其中本模块29项；87处本地文档链接和统一数据软链接检查通过。简要核验记录为`outputs/perception/verification.json`，不把旧GPU短片段结果当作后续每次改动的自动性能证明。
