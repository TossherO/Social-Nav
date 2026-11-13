# Social-Nav

Social-Nav 面向智能轮椅和移动机器人的长程社交导航，包含感知、局部视觉语言决策、长程任务组织、规划控制和独立安全监督。完整机器人在线系统仍在开发，接口存在不代表已完成实机验证。

先从[系统总览](docs/系统总览.md)了解职责和当前实现，再根据[开发指南](docs/开发指南.md)准备环境，按[系统运行手册](docs/系统运行手册.md)运行。实际参数以配置文件为准。

## 文档导航

- [接口约定](docs/接口约定.md)：时间、坐标、身份和跨模块数据。
- [后续开发测试汇总](docs/后续开发测试汇总.md)：各模块待完成工作和测试。
- [系统实机问题](docs/系统实机问题.md)：当前现场问题及下一步。
- [推理速度记录](docs/推理速度记录.md)：具有来源和适用范围的性能结果。

### 感知

- [建图与定位](docs/perception/技术实现-建图与定位.md)
- [检测与跟踪推理](docs/perception/技术实现-检测与跟踪推理.md)
- [检测训练](docs/perception/技术实现-检测训练.md)
- [环境地图与场景表示](docs/perception/技术实现-环境地图与场景表示.md)

### 决策

- [VLM 局部决策](docs/decision/技术实现-VLM局部决策.md)：T1 群组、T2 跟随、T3 就座、服务调用与标注。
- [Agent 长程导航](docs/decision/技术实现-Agent长程导航.md)：模型、工具、观察、任务和执行反馈。
- [SGLang 服务部署方案](docs/decision/SGLang服务部署方案.md)

### 导航

- [任务与路线](docs/navigation/技术实现-任务与路线.md)
- [路径规划](docs/navigation/技术实现-路径规划.md)
- [局部控制与安全](docs/navigation/技术实现-局部控制与安全.md)

[历史归档](docs/archive/)只供追溯，历史命令和结论不应直接视为当前系统状态。本机解释器和资源记录见被忽略的 `ENVIRONMENT.local.md`。

## 代码组织

感知在 `perception/`，局部决策与长程 Agent 在 `decision/`，规划、控制和安全在 `navigation/`；`runtime/` 负责运行调度与进程组织，`interfaces/` 定义跨模块数据，`configs/` 保存配置，`scripts/` 和 `tests/` 提供入口与验证。
