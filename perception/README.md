# 感知域

`detection_tracking/`统一检测训练、离线推理与ROS 1入口；具体运行见
[检测跟踪模块说明](detection_tracking/README.md)。原始ROS里程计不作为可靠定位，在线须接定位算法输出。

## 检测训练副本

`detection_tracking/training/`已复制SimpleDet3d源码，不修改原仓库。

- `simpledet3d/`保留原`simple_det3d`、`configs`、`tools`、`tests`、`docs`、编译和训练脚本。
- `source_snapshot.json`记录122个原文件的SHA-256及982,403字节工作树快照。
- 未复制`data`、`output`、`build`、权重、pickle、`.so`、Git或原项目代理指导文件。
- 当前`data`是指向仓库`data/datasets`的软链接，不复制或移动原数据。
- 副本YAML/数据类/脚本/准备工具共13文件仅适配数据路径和本地准备输出；原SimpleDet3d仓库与`source_snapshot`不变，`--check`报告这些预期差异。

只检查复制完整性（不导入Torch/CUDA，不读checkpoint或pickle）：

```bash
python -m perception.detection_tracking.training --check
```

只打印三阶段DRY_RUN=1命令（不会执行）：

```bash
python -m perception.detection_tracking.training --print-command \
  --checkpoint /path/to/verified_coda_checkpoint.pth --conda-env simpledet3d
```

真正运行复制后的体系前：确认已有检测环境；核验已链接data中的info；
确认CODa/Wheelchair原始点云路径、初始化checkpoint、GPU数和单卡batch；
在复制版重新编译扩展；选新RUN_TAG并确认复制版output输出位置。
早期迁移仅绑定目录与路径；本轮另外验证了检测推理，不读取标注pickle或启动训练。
软链接不是只读保护；准备工具默认输出到本地output/data_preparation，
不要显式覆盖共享data目录。不要在VLM环境安装检测依赖或复用原output续训。
源文档：[代码框架](detection_tracking/training/simpledet3d/docs/code_framework.md)、
[使用指南](detection_tracking/training/simpledet3d/docs/usage_guide.md)、
[Wheelchair弱监督训练设计](detection_tracking/training/simpledet3d/docs/wheelchair_training_design.md)。三阶段为混合适配、EMA/室内监督、室内固化，
原训练参数/数据语义保留，原文档的物理路径以主项目软链接说明替代；
弱中心标签不是完整3D框真值。

## 推理与待实现模块

已实现：

- `detection_tracking/inference/`：单帧VoxelRCNN、数据集/JSONL读取、缓存回放、CPU跟踪。
- `detection_tracking/ros/`：ROS 1 Python 3节点及catkin消息/launch源码；真实运行待验证。
- 数据集位姿/定位输出的健康门控；缺位姿时仅检测，不伪造世界轨迹或速度。

以下仍为占位，不包含运行程序：

- `mapping/`：全局、局部/临时几何地图与更新版本。
- `localization/`：自车位姿、坐标变换和定位健康。
- `sensors/`：RGB/LiDAR/里程计输入、同步与标定。

未来向decision和navigation提供同一时钟/坐标定义下的真实观察；不输出社会真值。
SimpleTrack核心方法已适配，ROLL等定位算法尚未接入。接口转换见inference/pipeline.py。
本机没有ROS，不能据节点源码存在就声称在线接入完成；未接VLM在线窗口或底盘。
