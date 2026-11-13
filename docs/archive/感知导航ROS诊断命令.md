# 感知导航ROS诊断命令

以下保存可追溯的专项ROS工作流，日常推理与交互导航使用[当前运行手册](../系统运行手册.md)。归档不表示这些分支完成了本轮验证。

### ROS诊断准备

<details>
<summary>展开ROS环境和诊断库构建</summary>

ROS工具在匹配系统ROS的终端执行。只在缺少产物或相关源码变化时构建；`all`可换成`sensors`、`mapping`、`environment`或`interfaces`。

```bash
source perception/scripts/ros1_env.sh
BUILD_JOBS=2 bash perception/scripts/build_ros1.sh all
```

仅在使用CUDA环境诊断且缺构建时执行；编译器与计算能力沿用本机设置：

```bash
BUILD_JOBS=2 bash perception/scripts/build_ros1.sh environment
CUDACXX="$CUDA_COMPILER" CUDAHOSTCXX="$CUDA_HOST_COMPILER" \
  ENVIRONMENT_CUDA_ARCH="$CUDA_ARCH" BUILD_JOBS=2 bash perception/scripts/build_environment_cuda.sh
```

诊断bag入口会创建隔离master并自行回放，不要另外播放同一包。其他终端查看该任务时，要设置它对应的`ROS_MASTER_URI`。live诊断连接已有master；ROS查询命令只检查终端当前连接的master。

`--people-python`默认只运行三维检测跟踪。需要YOLO时显式选择`--image-association`或`--decision-project`，并准备RGB、CameraInfo、引擎和对应安装外参。`--no-image-association`可明确禁用，设备和外参分别由`--image-device N`与`--camera-calibration`指定。

旧ROS参数兼容`--map/--prior-map/--navigation`名称，下面示例统一采用明确的参数名。它们与runtime自己的参数不能混用。

</details>

<a id="离线建图定位与检测"></a>

### ROS离线建图定位与检测

<details>
<summary>展开隔离master中的建图、有图定位与无图定位</summary>

本节使用 ROS 诊断入口，它会创建隔离 master 并回放，不需要额外播放同包。初值方式互斥，`--pose-file` 必须对应 bag、地图和起始偏移。只做定位时省略检测参数；常规联合推理仍优先使用[直接运行时](../系统运行手册.md#日常运行与回看)。

先按[地图准备](../系统运行手册.md#地图准备)保存对应起点；下列三种任务分别选用。

```bash
source perception/scripts/ros1_env.sh
BAG=/path/to/recording.bag
LOCALIZATION_MAP=/path/to/complete_3d_localization_map
POSE="${BAG%.bag}.initial_pose.json"
OUT=outputs/perception/ros_bag_new
```

```bash
# 建图：完整三维地图位于该运行目录的map/。
bash "perception/scripts/run_ros_diagnostic.sh" --mode mapping --bag "$BAG" --out "$OUT/mapping" --rviz
```

```bash
# 有图定位＋检测跟踪：使用已保存的起点。
bash "perception/scripts/run_ros_diagnostic.sh" --mode localization --bag "$BAG" --localization-map "$LOCALIZATION_MAP" --pose-file "$POSE" \
  --people-python "$PEOPLE_PYTHON" --out "$OUT/bag_localization" --rviz
```

```bash
# 无图定位＋检测跟踪：不需要定位地图或起点。
bash "perception/scripts/run_ros_diagnostic.sh" --mode odometry --bag "$BAG" \
  --people-python "$PEOPLE_PYTHON" --out "$OUT/bag_odometry" --rviz
```

</details>

### 一次完成推理与决策标注准备

<details>
<summary>展开旧ROS联合准备；普通runtime结果导入见日常流程</summary>

本节是 ROS 联合准备方式，只用于 bag。模型环境必须同时具备三维检测与 YOLO 依赖和引擎，目标运行目录及标注工程目录均须不存在。给联合命令添加 `--decision-project` 即可在回放结束后建立标注索引。

```bash
source perception/scripts/ros1_env.sh
BAG=/path/to/recording.bag
LOCALIZATION_MAP=/path/to/complete_3d_localization_map
POSE="${BAG%.bag}.initial_pose.json"
OUT=outputs/perception/ros_decision_new
DECISION_PROJECT=data/annotations/vlm_decision/SESSION
bash "perception/scripts/run_ros_diagnostic.sh" --mode localization --bag "$BAG" --localization-map "$LOCALIZATION_MAP" --pose-file "$POSE" \
  --people-python "$PEOPLE_PYTHON" --out "$OUT/bag_localization" \
  --decision-project "$DECISION_PROJECT" --decision-root-group ORIGINAL_RECORDING
# 无图时使用 --mode odometry 并去掉 --localization-map／--pose-file。
# 旧安装的bag须用 --camera-calibration /path/to/matching_camera_lidar.yaml 指定对应外参。
```

同一次回放并行运行定位、三维检测跟踪与 YOLO 关联，分别保存定位记录、`people.jsonl` 和 `rgb_associations.jsonl`。结束后将二维结果导入工程的 `rgb_association/` 并登记默认来源，不再运行 YOLO。

人工标签留在 `DECISION_PROJECT`，不复制 bag、全量图像或点云。命令成功后按[决策标注手册](../decision/技术实现-VLM局部决策.md#决策任务标注)打开或服务该工程。

只有整条命令成功，才表示标注准备完成。`run.json` 保存联合状态、二维摘要、模型和标定，工程建立来源后不改写。未处理观察仍是 `not_processed`，不会自动补跑或换图。若仅导入失败，保留感知结果，按[恢复说明](../decision/技术实现-VLM局部决策.md#rgb框来源)处理。

</details>

<a id="在线建图定位与检测"></a>

### ROS在线建图定位与检测

<details>
<summary>展开外部master上的在线诊断</summary>

本节是ROS诊断。先启动外部 master、真实点云与 IMU，再运行任务。`--people-python` 只开启三维检测跟踪，原 RGB 显示需要相机话题，YOLO 还需显式 `--image-association` 和 CameraInfo。

Ctrl+C 结束任务但保留驱动，也可用 `--duration 60` 限时。在线不接受 bag 偏移、速率或起点文件。

```bash
source perception/scripts/ros1_env.sh
LOCALIZATION_MAP=/path/to/complete_3d_localization_map
OUT=outputs/perception/ros_live_new
```

```bash
bash "perception/scripts/run_ros_diagnostic.sh" --source live --mode mapping --out "$OUT/live_mapping" --rviz
```

```bash
bash "perception/scripts/run_ros_diagnostic.sh" --source live --mode localization --localization-map "$LOCALIZATION_MAP" --initialization rviz \
  --people-python "$PEOPLE_PYTHON" --out "$OUT/live_localization" --rviz
```

```bash
bash "perception/scripts/run_ros_diagnostic.sh" --source live --mode odometry \
  --people-python "$PEOPLE_PYTHON" --out "$OUT/live_odometry" --rviz
```

在线默认保存定位及三维结果。后续标注需原始图像／点云，采集时同步录bag。启用图像支路后还保存二维记录，标注导入可直接复用；此时`--rviz`显示沿用三维ID的YOLO匹配框，否则显示原RGB。

</details>

### 独立检测与显示

<details>
<summary>展开已有可靠定位时的ROS检测与显示节点</summary>

已有可靠定位时使用；ROLL来源将`fast_lio`改为`roll_fast_lio`。只看原生检测时另起显示适配器并加`sensor_only:=true`。

```bash
source perception/scripts/ros1_env.sh
```

```bash
roslaunch social_nav_people people.launch project_root:="$PWD" \
  python_executable:="$PEOPLE_PYTHON" localization_source:=fast_lio rviz:=true
```

```bash
# 检测已运行且尚未启动显示适配器时，仅添加RViz显示。
roslaunch social_nav_people visualization.launch project_root:="$PWD" rviz:=true
```

</details>

<a id="局部环境几何与查询"></a>

### ROS局部环境几何与查询

<details>
<summary>展开局部图诊断、快照与ROS服务查询</summary>

下面是ROS环境诊断，三种运行方式只选一种，已有适用结果时跳过构建与推理。实际图层含义见[环境专题](../perception/技术实现-环境地图与场景表示.md)。

局部图默认使用 CUDA、历史自由背景与近期障碍覆盖。CPU 参考需显式 `--environment-backend cpu`，该参数不是直接运行时参数。RViz 中可打开 `Historical free inference` 和 `Recent obstacle coverage` 查看两种来源。

构建见[ROS诊断准备](#ros诊断准备)。

```bash
source perception/scripts/ros1_env.sh
BAG=/path/to/recording.bag
LOCALIZATION_MAP=/path/to/complete_3d_localization_map
POSE="${BAG%.bag}.initial_pose.json"
OUT=outputs/perception/ros_environment_new
```

```bash
# 无图局部几何；在线话题把 --bag "$BAG" 换为 --source live。
bash "perception/scripts/run_ros_diagnostic.sh" --mode odometry --bag "$BAG" --environment --environment-backend cuda \
  --rviz --out "$OUT/environment_run"
```

```bash
# 有图并允许经过无图路段；初值和地图按已有定位流程选择。
COVERAGE=/path/to/localization_coverage.yaml
bash "perception/scripts/run_ros_diagnostic.sh" --mode localization --bag "$BAG" --localization-map "$LOCALIZATION_MAP" --pose-file "$POSE" \
  --environment --environment-backend cuda --allow-local-only --association-coverage "$COVERAGE" \
  --rviz --out "$OUT/environment_run"
```

需要检测时，在所选诊断命令中加`--people-python "$PEOPLE_PYTHON"`。可信区域通过[地图准备](../系统运行手册.md#地图准备)单独绘制。

```bash
# 建图并保存静态射线证据。
bash "perception/scripts/run_ros_diagnostic.sh" --mode mapping --bag "$BAG" --environment --environment-backend cuda --record-environment-scans \
  --out "$OUT/environment_run"
```

局部环境地图默认CUDA，上述显式参数可省略。需CPU参考或未构建CUDA库时显式加`--environment-backend cpu`；不会自动降级。设备编号由环境配置的`compute.device`指定，并受`CUDA_VISIBLE_DEVICES`约束。静态全图导出仍使用CPU。

```bash
# 停止运行后查看最后一份观测快照；--image可替代--show导出一张新图片。
"$ROS_PYTHON" -m perception.environment --snapshot "$OUT/environment_run/environment/layers.npz" --show
# 坐标单位米，使用快照记录的坐标系。
"$ROS_PYTHON" -m perception.environment --snapshot "$OUT/environment_run/environment/layers.npz" --query 0 0
```

```bash
rosservice call /perception/environment/status
# 把SNAPSHOT_ID替换为state中的实际身份；可用operation还包括crop_map和query_geometry。
rosservice call /perception/environment/query \
  'request_json: "{\"operation\":\"get_snapshot\",\"snapshot_id\":\"SNAPSHOT_ID\"}"'
```

</details>

<a id="统一场景与混合模式"></a>

### ROS统一场景与混合模式

<details>
<summary>展开场景服务、决策视图和覆盖对照</summary>

以下查询连接当前master；场景ID取发布元数据。混合路线加`--association-coverage COVERAGE.yaml`启用地图可信区域检查，参数与范围见[定位技术文档](../perception/技术实现-建图与定位.md#手动地图可信区域与混合定位)。

```bash
source perception/scripts/ros1_env.sh
BAG=/path/to/recording.bag
LOCALIZATION_MAP=/path/to/complete_3d_localization_map
POSE="${BAG%.bag}.initial_pose.json"
STATIC_MAP=/path/to/static_environment_map
COVERAGE=/path/to/localization_coverage.yaml
OUT=outputs/perception/ros_scene_new
```

```bash
bash "perception/scripts/run_ros_diagnostic.sh" --mode localization --bag "$BAG" --localization-map "$LOCALIZATION_MAP" --pose-file "$POSE" \
  --environment --allow-local-only --scene \
  --static-map "$STATIC_MAP" \
  --association-coverage "$COVERAGE" \
  --people-python "$PEOPLE_PYTHON" --out "$OUT/scene_run"
```

在另一个已设置对应`ROS_MASTER_URI`的终端查询正在运行的场景：

```bash
rosservice call /perception/scene/status
rostopic echo -n 1 /perception/scene
# 将SCENE_ID替换为场景中的observation_id；可额外绑定input_fingerprint。
rosservice call /perception/scene/query \
  'request_json: "{\"operation\":\"get_scene\",\"observation_id\":\"SCENE_ID\"}"'
# operation=get_rgb返回PNG/base64及像素摘要；几何操作为query_geometry／crop_map，须指定frame。
# 不需要先验视图时省略 --static-map；配合 --rviz 显示可撤销的历史占据层。
# get_prior查看先验状态；query_prior_geometry／crop_prior_map要求frame=map。
```

```bash
# 停止后生成最后一份场景的中性决策视图；输出目录必须尚不存在，不调用模型。
"$PEOPLE_PYTHON" -B -m decision.vlm.environment_inputs --scene "$OUT/scene_run/scene" --out "$OUT/scene_run/decision_context"
# 旧场景可显式添加 --static-map "$STATIC_MAP" 生成新的三栏视图，不修改源场景。
```

```bash
# 同一批真实前端／配准到达事件，对比保留／屏蔽覆盖；比较实际公共状态与事件重放。
"$ROS_PYTHON" -B -m perception.mapping_localization.experiment \
  --run "$OUT/scene_run" --out "$OUT/scene_run/comparison"
```

</details>

### 静态地图查询与场景重建

<details>
<summary>展开静态几何、局部裁剪与因果场景重建</summary>

查询使用环境包中的坐标系，下面位置和边界单位为米。

```bash
source perception/scripts/ros1_env.sh
STATIC_MAP=/path/to/static_environment_map

"$ROS_PYTHON" -B -m perception.environment query --static-map "$STATIC_MAP" --operation geometry \
  --frame map --position 0 0
```

```bash
"$ROS_PYTHON" -B -m perception.environment query --static-map "$STATIC_MAP" --operation crop \
  --frame map --bounds -2 -2 2 2 --layers occupancy ground_z
```

重建旧bag场景时，`RUN`必须是`SCENE`所属的定位运行；`--scenes`可列多个目录，不重跑检测或SLAM。

```bash
RUN=/path/to/localization_run
SCENE=/path/to/scene_from_that_run
OUT=outputs/perception/rebuilt_scenes_new

"$PEOPLE_PYTHON" -B -m perception.environment rebuild-scenes --run "$RUN" \
  --scenes "$SCENE" --out "$OUT/causal_scenes"
```

连续几何回放的起止值为相对该定位记录首时刻的秒数：

```bash
START_SECONDS=10
END_SECONDS=20

"$PEOPLE_PYTHON" -B -m perception.environment replay-window --run "$RUN" \
  --start-s "$START_SECONDS" --end-s "$END_SECONDS" --sampling recorded \
  --out "$OUT/geometry_window"
```

</details>

### RGB人物检测与三维关联

<details>
<summary>展开TensorRT引擎准备、离线缓存和ROS图像节点</summary>

新bag或live先按[日常流程](../系统运行手册.md#可选图像与决策输入)显式启用所需图像支路。已有二维记录的标注工程直接导入，无需再次离线推理。以下工具用于引擎准备、旧来源补充二维或专项比较，输入为已登记原bag及三维结果的决策观察工程。

Linux二维与三维检测共用项目环境；Windows不装YOLO。缺依赖时按[图像依赖](../../perception/detection_tracking/inference/requirements-image.txt)补齐，Torch／TorchVision从官方匹配源安装，普通包优先中国镜像、TensorRT二进制用NVIDIA源，不为本功能升级既有Torch。权重来源及关联规则见[推理专题](../perception/技术实现-检测与跟踪推理.md#rgb人物框与三维身份)。

```bash
DECISION_PROJECT=data/annotations/vlm_decision/SESSION
RGB_ASSOCIATIONS=outputs/perception/SESSION/rgb_association
```

```bash
# 仅缺TensorRT引擎时显式构建；目标GPU须与运行环境匹配，不覆盖现有引擎。
# 此命令可能耗时数分钟；浏览和服务启动不会自动构建。
"$PEOPLE_PYTHON" -B "perception/scripts/prepare_rgb_association.py" --export-engine --device 0
```

```bash
# 约10Hz既有三维观察逐帧准备；不重算SLAM或三维检测。
"$PEOPLE_PYTHON" -B "perception/scripts/prepare_rgb_association.py" \
  --project "$DECISION_PROJECT" --output "$RGB_ASSOCIATIONS" --device 0
# 小样本可加 --source 子bag文件名 --segments 20:25 65:70 --limit 150。
# 相同来源／配置可用原命令断点继续；改配置换新目录。
# 显式PyTorch备用：增加 --backend pytorch，输出须换新目录。
```

```bash
# 仅对含逐图二维检测的完整离线缓存：更改关联规则可复用检测，不读图、不加载GPU模型。
# 联合运行记录导入生成的缓存不适用此选项。
"$PEOPLE_PYTHON" -B "perception/scripts/prepare_rgb_association.py" \
  --project "$DECISION_PROJECT" --output outputs/perception/SESSION/rgb_association_refined \
  --detections-from "$RGB_ASSOCIATIONS"
```

上述`device 0`是`CUDA_VISIBLE_DEVICES`内的第一张卡；公共图像配置默认TensorRT FP16。独立工具只生成缓存，完成后一次登记；完整兼容缓存只核对来源和覆盖，不运行YOLO：

```bash
"$PEOPLE_PYTHON" -B -m decision.vlm.annotation prepare-rgb \
  --project "$DECISION_PROJECT" --output "$RGB_ASSOCIATIONS"
```

统一入口和标注`prepare-rgb`会登记工程默认来源，本机／远程标注及导出都从工程读取；切帧不推理，不复制全量图像或点云。来源切换与重审见[标注手册](../decision/技术实现-VLM局部决策.md#rgb框来源)。

以下独立节点仅用于专项联调。旧ROS入口`run_ros_diagnostic.sh`显式加`--image-association`或请求`--decision-project`时才管理该节点，不要重复启动；默认直接运行时不启动此ROS支路。使用同一master／时钟，保留CameraInfo；原检测节点的`/perception/people/rgb`必须可达：

```bash
source perception/scripts/ros1_env.sh
roslaunch social_nav_people rgb_association.launch \
  project_root:="$PWD" python_executable:="$PEOPLE_PYTHON" device:=0
# 原始相机话题不同可加 camera_info_topic:=/实际/camera_info。
rostopic echo -n 1 /perception/people/rgb_associations/status
rostopic echo -n 1 /perception/people/rgb_associations
```

新增消息后仅需构建`interfaces`。节点发布关联消息、保存用记录及`/perception/people/rgb_associations/image`叠框图，旧ROS入口显式启用图像支路时RViz订阅该图；原`/perception/people/rgb`不改写。关联失败和丢帧独立报告，三维检测继续运行；正式在线VLM适配仍待接通。

</details>

### 导航ROS诊断

<details>
<summary>展开ROS影子规划、安全节点与旧负载对照</summary>

需要已有感知话题、同一master与时钟；仅发布私有影子建议。

```bash
source perception/scripts/ros1_env.sh
STATIC_MAP=/path/to/static_environment_map
# 终端1：默认DWA；无图目标可省略静态包。
python -B -m navigation.planning online --geometry historical \
  --static-map "$STATIC_MAP" --shadow --collision-backend cuda
```

```bash
# 终端2：独立安全诊断。
source perception/scripts/ros1_env.sh
STATIC_MAP=/path/to/static_environment_map
python -B -m navigation.safety.shadow --geometry historical \
  --static-map "$STATIC_MAP" --collision-backend cpu
```

```bash
rostopic pub -1 /navigation/planning/goal std_msgs/String \
  'data: "{\"request_id\":\"manual-1\",\"task_id\":\"manual\",\"goal_revision\":1,\"mode\":\"local\",\"frame_id\":\"odom\",\"target_reference\":\"base_link\",\"goal\":[2.0,1.0],\"goal_behavior\":\"pass_through\"}"'
```

```bash
rostopic pub -1 /navigation/planning/cancel std_msgs/String \
  'data: "{\"task_id\":\"manual\",\"goal_revision\":1}"'
```

旧`navigation.planning.integration`曾用于ROS话题链的全负载与故障注入诊断，已在转入实机阶段时移出活跃源码。原脚本保存在`outputs/navigation/history/live_transition/source_records.zip`，仅供追溯；当前带目标测试使用`runtime`或RViz交互入口，不继续维护该旧命令。


</details>

### PIN与旧地图

<details>
<summary>展开PIN独立环境、建图定位和历史地图导入</summary>

PIN用独立环境、完整PIN地图和已知起点，不支持默认RViz／ROLL选点文件。下方安装只在新环境执行。

```bash
BAG=/path/to/recording.bag
LOCALIZATION_MAP=/path/to/complete_pin_map
OUT=outputs/perception/pin_session_new
PIN_PYTHON=/path/to/pin/env/bin/python
"$PIN_PYTHON" -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  -r perception/mapping_localization/backends/pin_slam/requirements.txt
```

```bash
bash "perception/scripts/run_ros_diagnostic.sh" --backend pin_slam --backend-python "$PIN_PYTHON" --mode mapping \
  --bag "$BAG" --step-replay --out "$OUT/pin_mapping"
```

```bash
# LOCALIZATION_MAP指向PIN完整地图；起点不同须显式加--pose（弧度／米）。
bash "perception/scripts/run_ros_diagnostic.sh" --backend pin_slam --backend-python "$PIN_PYTHON" --mode localization \
  --bag "$BAG" --localization-map "$LOCALIZATION_MAP" --step-replay --out "$OUT/pin_localization"
```

```bash
# 导入历史ROLL楼层地图；后续LOCALIZATION_MAP指向新导入目录，源地图保持只读。
bash "perception/scripts/import_map.sh" data/maps/wheelchair/legacy/floor "$OUT/imported_floor"
```

</details>

