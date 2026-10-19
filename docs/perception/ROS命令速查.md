# 感知ROS命令速查

所有命令从仓库／客户端根目录执行，**按任务选用，不整页顺序执行**。本文集中维护ROS、离线标定和bag标注的启动命令；运行机制见[实机感知](实机感知.md)，模型行为见[检测与跟踪](检测与跟踪.md)，标注交互与训练见[检测训练](检测训练.md)。Linux包装入口位于`perception/scripts/`，Windows使用Python模块入口。

常用顺序：**[准备](#准备) → [录包](#采集与bag处理) → [选起点并离线推理](#离线建图定位与检测) → [回看](#结果回看)**；已有在线传感器则直接看[在线推理](#在线建图定位与检测)。

标定看[离线标定与候选回放](#离线标定与候选回放)；人工数据准备看[本机标注与导出](#bag-行人中心标注)、[Linux服务](#远程服务linux)或[Windows客户端](#windows客户端)。

## 准备

每个操作终端按需填写下列变量；解释器、空闲GPU及已有资源的位置见[本机记录](../../ENVIRONMENT.local.md)。`GPU_ID`替换为实际可用编号，多卡用逗号分隔。系统ROS终端先退出已激活的Conda环境（`conda deactivate`），检测解释器单独指定；[远程标注HTTP服务](#远程服务linux)使用已激活的项目Conda环境，按该节准备。

```bash
P=perception/scripts
source "$P/ros1_env.sh"
BAG=/path/to/recording.bag
MAP=outputs/perception/maps/site_map              # 已有完整地图
POSE="${BAG%.bag}.initial_pose.json"                # 与该bag、地图绑定的人工起点
OUT=outputs/perception/session_new                # 本轮产物根目录，重复运行时更换
RESULT=outputs/perception/previous_run            # 要回看的既有运行目录，也可填完整地图目录
PEOPLE_PYTHON=/path/to/detector/env/bin/python
export CUDA_VISIBLE_DEVICES=GPU_ID                # 运行检测或PIN前设置
```

仅缺少构建产物或相关源码变更时构建；已有构建可跳过。`all`可换为`sensors`（驱动）、`mapping`（含消息及定位）或`interfaces`（仅消息）；并行数按本机资源设置。

```bash
BUILD_JOBS=2 bash "$P/build_ros1.sh" all
```

各任务的`--out`子目录必须尚不存在，不提前创建。bag回放、独立选点和结果查看自建隔离master；在线任务连接外部master。在线RGB显示仍为原图；离线点云／3D框投影由独立标定工具完成，采集与操作边界见[相机—雷达标定](实机感知.md#相机雷达无靶标标定)。

## 采集与bag处理

真实采集时先启动master、再启动驱动、最后录包，分别使用独立终端。每个终端先完成上述环境准备。

```bash
# 终端1：已有master时跳过；此处使用真实设备时钟。
roscore
```

```bash
# 终端2：仅在真实设备采集时设置；外部bag模拟在线应使用下方模拟时钟配置。
rosparam set /use_sim_time false
bash "$P/start_sensors.sh" --serial-front FRONT_CAMERA_SERIAL
# 可选：加 --rear --serial-rear REAR_CAMERA_SERIAL；序列号已写入本机配置时可省略。
# 仅雷达：将相机参数换为 --lidar-only；这不提供FAST-LIO所需的IMU。
```

```bash
# 终端3：保留一份RGB及其内参、原始点云、真实IMU和已有静态TF。
mkdir -p "$OUT"
rosbag record --lz4 -O "$OUT/capture.bag" /rslidar_points /camera/imu \
  /camera/color/image_raw /camera/color/camera_info /tf_static
```

先用`Ctrl+C`结束录包，再关闭驱动；需回放新包时将`BAG`改为该路径。后视／深度图像按任务另加，不必默认全录；若需记录已有动态TF或定位输出，再显式添加对应话题。

```bash
# 查看包；图形浏览需已安装rqt_bag。
rosbag info "$BAG"
rqt_bag "$BAG"

# 只在外部master上模拟在线输入时使用；不要与统一bag任务同时向该master回放。
rosparam set /use_sim_time true
rosbag play --clock --pause -r 0.5 "$BAG"

# 按原包记录时间切出[10,20)秒；输出新文件，数值按需要修改。
mkdir -p "$OUT"
BAG_T0=$(/usr/bin/python3 -c 'import rosbag,sys; b=rosbag.Bag(sys.argv[1]); print(b.get_start_time()); b.close()' "$BAG")
rosbag filter "$BAG" "$OUT/cut.bag" "$BAG_T0 + 10 <= t.to_sec() < $BAG_T0 + 20"

# 切包缺/tf_static时，使用已提供的恢复工具；实际位置见本机记录。
TF_RESTORE=/path/to/restore_tf_static.py
SOURCE_BAG=/path/to/original_with_tf_static.bag
/usr/bin/python3 "$TF_RESTORE" --source "$SOURCE_BAG" \
  --input "$OUT/cut.bag" --output-dir "$OUT/cut_with_tf"
```

切分条件使用bag记录时间，保留区间内全部话题；初始段之外的子包可能丢失只发布一次的静态TF。恢复工具只从原包补回已存在的变换，不能生成相机—雷达／车体标定；已含静态TF的包跳过。

## 离线标定与候选回放

相机—雷达标定使用独立项目，实际位置见[本机记录](../../ENVIRONMENT.local.md#感知ros工作区)，默认参数见[实机感知](实机感知.md#相机雷达无靶标标定)。以下入口不启动物理驱动，原bag只读；LI-Init二进制须带项目提供的纯雷达参考补丁，系统Python负责ROS，`SCIPY_PYTHON`负责拟合。

```bash
CALIB=outputs/perception/calibration/new_session
LI_INIT=/path/to/private/li_init
SCIPY_PYTHON=/path/to/scipy/env/bin/python
CAMERA_CALIB=perception/sensors/calibration/front_camera_lidar.yaml
TF_BAG=/path/to/same_device_with_tf_static.bag
CALIB_WORK=$(mktemp -d /tmp/social-nav-calibration.XXXXXX)

# 完整扫描的独立纯雷达参考；输出CSV必须尚不存在。
/usr/bin/python3 -m perception.sensors.calibration.replay_lidar \
  --bag "$BAG" --output "$CALIB/poses.csv" --binary "$LI_INIT" --subframes 1

# 原始IMU只提取到临时目录；不改变在线消息频率。
/usr/bin/python3 -m perception.sensors.calibration.lidar_imu extract \
  --bag "$BAG" --output "$CALIB_WORK/imu.npy"

# --same-device仅用于已确认同一台D435i的跨bag内部TF复用。
/usr/bin/python3 -m perception.sensors.calibration.lidar_imu compose \
  --camera "$CAMERA_CALIB" --tf-bag "$TF_BAG" --imu-bag "$BAG" --same-device \
  --output "$CALIB/initial.json"
"$SCIPY_PYTHON" -m perception.sensors.calibration.lidar_imu fit \
  --prior "$CALIB/initial.json" --poses "$CALIB/poses.csv" --imu "$CALIB_WORK/imu.npy" \
  --output "$CALIB" --validation-start 50

# 使用候选做整段回放；不会直接覆盖默认标定文件。
bash "$P/run.sh" --mode odometry --bag "$BAG" --out "$CALIB/replay" \
  --lidar-imu-calibration "$CALIB/optimized.yaml"
```

`--validation-start`按bag记录起点的相对秒数设置，须给求解和留出阶段留下有效覆盖；例子适合一分钟以上片段。默认推理已自动加载当前`lidar_imu.yaml`，无需执行以上求解。只有重新标定时才生成候选、拟合特征与对齐图；确认后按需保留参数、来源及必要结果，临时原始IMU提取可清理。候选覆盖也适用于建图、有图定位及在线FAST-LIO／ROLL，不用于PIN。

## bag 行人中心标注

此工具直接读取 bag，不需要运行 `roscore`、`rosbag play`、定位或 RViz。推荐先完成所有扫描的离线检测，再打开窗口复核；操作和监督边界见[检测训练](检测训练.md#rosbag人工中心复核与增量训练)。

```bash
ANNOTATION=data/annotations/rosbag_people/task_scene
ANNOTATION_GPU=0                             # 按本机空闲资源选择一张物理卡

# 创建工程，顺序检测全部扫描；同一工程可重复 --bag 添加多个来源。
bash "$P/annotate_people.sh" predict --bag "$BAG" --project "$ANNOTATION" \
  --detector-python "$PEOPLE_PYTHON" --gpu "$ANNOTATION_GPU" \
  --root-group ORIGINAL_RECORDING --location-group LOCATION --split train

# 打开／继续标注：窗口选择当前子 bag，只使用缓存／手工。
bash "$P/annotate_people.sh" open --project "$ANNOTATION"
# 可加 --sequence 子包文件名（或完整路径／来源ID）指定初始序列。
# 可加 --sample-interval 1 显式指定步长；省略则恢复各包设置，初次为1秒。

# 可选：导入有模型来源元数据的既有单帧候选；不把跟踪预测当标签。
bash "$P/annotate_people.sh" open --project "$ANNOTATION" \
  --detections /path/to/observations.jsonl

# 导出完整复核帧至新目录；不启动训练。
bash "$P/annotate_people.sh" export --project "$ANNOTATION" \
  --out perception/detection_tracking/training/simpledet3d/output/data_preparation/rosbag_centers/task_scene
```

`predict`顺序处理全部扫描，保留原始约10Hz时间戳；结果和完成统计集中于一个`annotations.sqlite`。同模型重复执行复用候选并保留人工编辑，换模型须新建工程，检测结束释放GPU。已有候选的工程可直接打开，无需重跑定位或检测。

| 参数／情况 | 用法 |
|---|---|
| 多个子bag或迁移路径 | 重复`--bag`添加／重定位来源；每包独立浏览，共用工程数据库 |
| 浏览片段 | `open --start/--duration`，偏移相对各包首帧点云记录时间 |
| 指定输入 | `--lidar-topic`覆盖点云话题，`--rgb-topic ''`关闭辅助图像 |
| 预计算资源 | `--detector-python`与`--gpu`仅用于`predict` |
| 图形兼容 | 默认OpenGL，必要时在`open`后加`--cloud-renderer software`；首次打开bag另有索引开销 |
| 导入旧候选 | `--detections`需要同目录`settings.json`或`manifest.json`中的模型摘要、单帧与范围记录，并逐扫描匹配指纹；缺来源元数据的旧`people.jsonl`应重新预计算 |

标注状态、快捷键、时间条、保存与来源划分只维护于[标注操作](检测训练.md#标注操作)。导出前须完成复核和划分，未标帧不视为无人背景。

### 远程服务（Linux）

HTTP服务复用项目的Python 3.12开发／检测Conda环境，系统ROS解释器子进程只读bag。先激活该Conda环境（本机环境名和路径见[本机记录](../../ENVIRONMENT.local.md)），首次使用时补齐以下服务依赖；已安装可跳过。客户端依赖单独安装，不把服务器环境复制到Windows。

```bash
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  -r perception/detection_tracking/annotation/requirements-service.txt

# 首次配置时复制；已有本机配置保留，按需编辑。
cp -n configs/annotation_service.example.json configs/annotation_service.local.json
```

本机JSON登记允许访问的工程目录、稳定的`server_id`和实际`ros_python`。示例默认`read_only: true`用于浏览；正式标注时将目标工程改为`false`并重启，客户端同时去掉`--read-only`。服务只增补协调／回执表，保留原标签、检测与输入绑定。需要做清空或冲突试验时，使用SQLite在线备份得到的独立工程。

```bash
# 默认仅监听127.0.0.1；多机时明确指定本机Tailscale IPv4。
ANNOTATION_IP=$(tailscale ip -4)
bash perception/scripts/serve_annotations.sh --config configs/annotation_service.local.json \
  --host "$ANNOTATION_IP" --port 8765
# Ctrl+C停止；不创建系统常驻服务、不开放公网监听。
```

服务需在仓库根目录运行，`serve_annotations.sh`负责ROS环境，默认使用当前激活的Conda环境中的`python`；HTTP解释器位置可用`SOCIAL_NAV_ANNOTATION_SERVICE_PYTHON`显式指定。工程必须已有索引与扫描绑定，通常直接复用`predict`完成的工程。服务不接受客户端指定任意服务器路径。连接无需访问令牌，客户端直接使用服务地址与工程ID；网络访问范围由Tailscale控制，服务仍仅允许监听回环或Tailscale地址。

生成轻量客户端包时使用新文件名；只包含固定代码清单、客户端依赖和从本页生成的说明，不带bag、权重、缓存或私有配置：

```bash
python -B -m perception.detection_tracking.annotation package-client \
  --out outputs/perception/annotation_client/client_new.zip
```

### Windows客户端

将服务器生成的`client.zip`传到Windows；可用已有的Tailscale SSH／SFTP或下载工具，无需传bag。以下命令在Windows PowerShell执行，解压到新的工作目录。Conda已可用时直接创建独立Python 3.11环境，普通包使用清华镜像。

```powershell
Expand-Archive .\client.zip -DestinationPath .\SocialNavAnnotator
Set-Location .\SocialNavAnnotator
$AnnotationEnv = 'social-nav-annotator'
$ExistingAnnotationEnvs = (conda env list --json | ConvertFrom-Json).envs
if ($ExistingAnnotationEnvs | Where-Object { (Split-Path $_ -Leaf) -eq $AnnotationEnv }) { throw '环境名已存在，请换新名称；不要覆盖已有环境。' }
conda create -n $AnnotationEnv python=3.11 pip -y --override-channels -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda activate $AnnotationEnv
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r .\perception\detection_tracking\annotation\requirements-client.txt

$AnnotationServer = 'http://LINUX_TAILSCALE_IP:8765'  # 替换IP；工程ID与服务端登记保持一致。
python -B -m perception.detection_tracking.annotation open --server $AnnotationServer --project-id wheelchair --read-only
```

以后启动时先激活客户端Conda环境，并进入解压后包含`perception/`的根目录，再执行上述`python -B -m … open`命令。

首次可用`--read-only`浏览三视图；正式标注按上节开放服务端写入并去掉该参数。显示不兼容时加`--cloud-renderer software`。窗口只负责绘制和编辑，不运行GPU检测，交互与同步行为见[检测训练](检测训练.md#远程标注与同步)。

可选`--sequence 子包名`、`--sample-interval 秒数`、`--cache-dir 本机目录`、`--image-encoding png`。默认缓存位于`$env:LOCALAPPDATA\SocialNav\annotation\<地址与工程对应目录>`；更换地址或缓存会打开另一份个人设置及待同步队列，旧草稿不会自动迁移。

发生冲突时优先使用窗口中的处理入口；需要保留为可读文件时指定原缓存目录和**新的输出文件**：

```powershell
python -B -m perception.detection_tracking.annotation export-drafts --cache-dir 'C:\path\to\original-cache' --out .\annotation_drafts.json
```

只有“服务器已保存”的已标帧才进入服务器导出；“已在本机暂存”不代表服务器同步完成。切换机器前先完成同步，退出时保留的待同步编辑需要使用原地址、工程与缓存重开恢复。Windows客户端无需source ROS环境，也不访问服务器SQLite文件共享。

## 离线建图定位与检测

以下默认使用ROLL／FAST-LIO。先选好场景对应的`BAG`与`MAP`；有起点标注就跳过选点。定位与无图命令同时启用检测、跟踪及RViz；只需要定位时去掉`--people-python "$PEOPLE_PYTHON"`。

```bash
# 建图：完成后地图位于该运行目录的map/；随后将MAP改为它。
bash "$P/run.sh" --mode mapping --bag "$BAG" --out "$OUT/mapping" --rviz

# 给bag保存起点：显示首帧点云与RGB，只选点，不运行定位；拒绝覆盖已有标注。
bash "$P/pick_start.sh" --bag "$BAG" --map "$MAP" --out "$POSE"

# 有图定位＋检测跟踪：使用已保存的起点。
bash "$P/run.sh" --mode localization --bag "$BAG" --map "$MAP" --pose-file "$POSE" \
  --people-python "$PEOPLE_PYTHON" --out "$OUT/bag_localization" --rviz

# 无图定位＋检测跟踪：不需要MAP或POSE。
bash "$P/run.sh" --mode odometry --bag "$BAG" \
  --people-python "$PEOPLE_PYTHON" --out "$OUT/bag_odometry" --rviz
```

| 按需调整 | 用法／限制 |
|---|---|
| 临时在地图选起点 | 有图命令将`--pose-file "$POSE"`换为`--initialization rviz`；配合`2D Pose Estimate` |
| 明确使用建图起点 | 换为`--initialization map-start`，仅确知同一起点时使用 |
| 已知六维起点 | 换为`--pose roll pitch yaw x y z`，弧度／米；各类起点选项互斥 |
| 慢放／截取片段 | 加`--rate 0.5`／`--start 5 --duration 12`；起点标注必须与偏移一致 |
| 对照标定候选 | 加`--lidar-imu-calibration /path/to/candidate.yaml`；只覆盖本次FAST-LIO／ROLL任务 |
| 不开RViz | 去掉`--rviz`；人工起点选择仍需窗口 |
| 建图同时检测 | 建图命令也可加`--people-python "$PEOPLE_PYTHON"` |
| 诊断／关闭建图回环 | 按需加`--keep-details`／`--loop-closure off`；后者仅ROLL建图 |

不要为这几条命令额外运行`roscore`或`rosbag play`。正常结束会保存结果；建图启用RViz时会继续显示最终地图，关闭窗口或再次`Ctrl+C`退出。
运行中的建图／有图／无图RViz默认显示青色同观测原始扫描；联合检测时扫描与框使用同次结果位姿，失效或超时清空。无图的`Recent scans`仅作可选累积诊断。

## 在线建图定位与检测

先按[采集部分](#采集与bag处理)启动外部master及点云／真实IMU输入，再在新终端选一条。以下两种定位均联合检测；只有建图／定位时同样去掉检测参数。运行入口不启动驱动或底盘。

```bash
bash "$P/run.sh" --source live --mode mapping --out "$OUT/live_mapping" --rviz

bash "$P/run.sh" --source live --mode localization --map "$MAP" --initialization rviz \
  --people-python "$PEOPLE_PYTHON" --out "$OUT/live_localization" --rviz

bash "$P/run.sh" --source live --mode odometry \
  --people-python "$PEOPLE_PYTHON" --out "$OUT/live_odometry" --rviz
```

选点时保持静止；前端支持运动初始化，健康有效后再使用结果。在线可加`--duration 60`限制墙钟时长，不接受`--start`、`--rate`或`--pose-file`；已知六维位姿可用`--pose`替代选点。`Ctrl+C`结束本次任务，保留外部master和驱动。

## 结果回看

只读已有结果，不运行检测或SLAM；同目录的`people.jsonl`会自动显示，无需再启动检测节点。

```bash
bash "$P/view.sh" --result "$RESULT"                             # 地图与完整轨迹全览
bash "$P/view.sh" --result "$RESULT" --play --points --rate 0.5  # 按同刻原始扫描播放
bash "$P/view.sh" --result "$RESULT" --play --points --frame lidar # 原生检测，含缺定位时刻
```

可加`--start 5`跳到指定秒、`--frame odom`看连续局部位姿；迁移后用`--bag PATH`／`--map PATH`重定位同一输入。比较目录需加`--phase localization --backend ROLL`（或`PIN`／`PIN_latest`）；单独查看PIN地图用`--backend-python "$PIN_PYTHON"`。空格暂停、滑条跳转；原始扫描按结束位姿刚性放置，未重新去畸变。

## 独立检测与显示

只在已有点云和可靠定位、且没有运行联合检测时使用。下例接FAST-LIO无图定位；ROLL来源将`localization_source`值改为`roll_fast_lio`，PIN改为`pin_slam`。

```bash
roslaunch social_nav_people people.launch project_root:="$PWD" \
  python_executable:="$PEOPLE_PYTHON" localization_source:=fast_lio rviz:=true

# 检测已运行且尚未启动显示适配器时，仅添加RViz显示。
roslaunch social_nav_people visualization.launch project_root:="$PWD" rviz:=true
```

去掉`rviz:=true`不打开窗口；`visualize:=true`只发布显示数据。缺有效定位时世界显示清空；单独显示时可加`sensor_only:=true rviz:=true`查看原生检测（先停止原显示适配器），常规有图／无图显示分别使用`map`／`odom`。图元颜色及过期规则见[显示说明](实机感知.md#检测跟踪的rviz显示)。

## PIN与旧地图

PIN是实验后端，使用独立、已准备的Python环境与空闲GPU；不支持默认任务的`--rviz`、ROLL的起点JSON或交互选点，ROLL／PIN地图不能混用。首次准备PIN环境时才执行安装行。

```bash
PIN_PYTHON=/path/to/pin/env/bin/python
"$PIN_PYTHON" -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  -r perception/mapping_localization/backends/pin_slam/requirements.txt

bash "$P/run.sh" --backend pin_slam --backend-python "$PIN_PYTHON" --mode mapping \
  --bag "$BAG" --step-replay --out "$OUT/pin_mapping"

# MAP改为PIN的完整地图；起点不同须显式加--pose（弧度／米）。
bash "$P/run.sh" --backend pin_slam --backend-python "$PIN_PYTHON" --mode localization \
  --bag "$BAG" --map "$MAP" --step-replay --out "$OUT/pin_localization"

# 导入历史ROLL楼层地图；MAP随后指向新导入目录，源地图保持只读。
bash "$P/import_map.sh" data/maps/wheelchair/legacy/floor "$OUT/imported_floor"
```

PIN逐帧回放等待同时间戳结果，回放耗时与在线处理时效分别统计。地图格式及恢复边界见[PIN说明](实机感知.md#pin-地图恢复与运行边界)。

## 常用诊断

以下用于在线／外部bag模拟在线的master。统一bag任务和查看器使用隔离master，默认外部终端不会自动连接；这类任务优先查看`run.json`，需要时加`--keep-details`保留诊断，不因外部终端查不到话题就判定失败。

```bash
rosnode list
rostopic list
rostopic hz /rslidar_points
rostopic hz /camera/imu
rostopic hz /camera/color/image_raw
rostopic echo -n 1 /localization/state
rostopic echo -n 1 /perception/people/status
rostopic echo -n 1 /roll/status                       # ROLL任务
rosparam get /use_sim_time
```

只查询与当前任务有关的项；状态频率不是端到端延迟，`VALID`也不代表精度通过。耗时、积压和失效解释见[运行状态与计时](实机感知.md#运行状态与计时)。
