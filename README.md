# Social-Nav

Social-Nav 面向移动机器人与智能轮椅的长程社交导航，贯通感知、任务决策、路径规划、运动控制与独立安全监督。

感知已提供传感器接入、建图与有图／无图定位、单帧检测跟踪、RViz与结果回看，以及本机／Windows远程行人中心标注和训练导出。默认建图／有图定位使用ROLL＋FAST-LIO，无图定位使用FAST-LIO；相机—雷达和雷达—IMU参数及适用边界见[实机感知](docs/perception/实机感知.md#标定与数据边界)。

VLM推理与离线调试优先使用自建远程SGLang服务，仅远程不可用或其GPU被占用时考虑本地备用方案。完整在线导航仍待接通，后续工作统一见[系统方案](docs/系统方案.md#83-待完成工作)，实际资源与检查记录见本机环境记录。

| 代码目录 | 职责 |
|---|---|
| `perception/` | 传感器接入、检测训练与跟踪、建图定位 |
| `decision/vlm/`、`decision/agent/` | 局部语义决策与外层任务编排；agent 尚未实现 |
| `navigation/` | 规划、控制与本地安全监督，当前为占位 |
| `interfaces/` | 共享类型与接口草案，尚未接通完整系统 |

模块登记见 [configs/system.json](configs/system.json)，它不是整机启动器；目录或类型存在不表示能力已验收。

## 文档入口

| 需要完成的工作 | 阅读入口 |
|---|---|
| 理解研究目标、系统分工与后续任务 | [系统方案](docs/系统方案.md) |
| 准备环境、迁移资源、理解目录和配置 | [开发指南](docs/开发指南.md) |
| 调整跨模块消息、坐标系、时间与失效语义 | [接口约定](docs/接口约定.md) |
| 构建、采集／bag处理、标定、定位检测、回看及标注启动命令 | [ROS命令速查](docs/perception/ROS命令速查.md) |
| 传感器、建图定位与检测跟踪的配置、运行机制及验证边界 | [实机感知](docs/perception/实机感知.md) |
| 使用当前相机—雷达、雷达—IMU参数，或重新标定 | [标定与数据边界](docs/perception/实机感知.md#标定与数据边界) |
| 检测模型与跟踪行为、数据集推理／缓存回放、输出语义 | [检测与跟踪](docs/perception/检测与跟踪.md) |
| 准备检测数据、编译、训练或评估 | [检测训练](docs/perception/检测训练.md) |
| 本机／远程标注行人中心、同步保存与导出微调数据 | [标注操作](docs/perception/检测训练.md#rosbag人工中心复核与增量训练)；安装启动见[标注命令](docs/perception/ROS命令速查.md#bag-行人中心标注) |
| 修改弱标签、损失或三阶段训练策略 | [弱监督设计](docs/perception/弱监督设计.md) |
| 默认远程VLM调用、服务启停、本地备用推理或便携包 | [VLM 部署与调用](docs/decision/VLM部署与调用.md) |
| 从零构建原生Conda与SGLang源码服务，使用默认DSPARK配置 | [SGLang服务部署方案](docs/decision/SGLang服务部署方案.md) |
| 设计数据划分、人工标注和实验指标 | [数据与评测](docs/experiments/数据与评测.md) |
| 准备T1/T2数据、本机／Windows远程标注与标签导出 | [决策任务标注](docs/decision/决策任务标注.md)；[实现方案](docs/decision/T1与T2标注程序实现方案.md) |
| 使用 JRDB 的字段、对齐、渲染与审核工具 | [JRDB 参考](docs/experiments/JRDB参考.md) |

开发约束见 [AGENTS.md](AGENTS.md)。本机解释器、资源位置和已执行检查只维护在忽略的 [ENVIRONMENT.local.md](ENVIRONMENT.local.md)；新机器按实际条件创建。

## 文档维护

当前项目文档集中于 `docs/`，最多采用“领域／专题”两级组织。系统方案维护设计与待办，开发指南维护公共准备流程，专题手册维护各自操作与实现边界；配置值以代码和配置文件为准，不在多份文档复制默认参数表。

代码模块不另维护重复 README。第三方源码文档、资产来源说明和独立便携包模板保留在其所属目录。VLM 部署指南由打包脚本直接从 `docs/` 复制，继续只有一个维护源。

历史资料：[旧版方案](docs/archive/旧版方案.md) · [实机感知整合记录](docs/archive/实机感知整合记录.md) · [AutoDL服务部署与调用](docs/archive/AutoDL服务部署与调用.md)。归档用于追溯，历史命令与环境不作为当前操作要求。
