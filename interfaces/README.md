# 共享接口草案

[contracts.py](contracts.py)提供Python类型记录；检测跟踪核心已有`to_snapshot`转换，其余跨域仍待接入。
它本身不是ROS消息、序列化协议、调度器或安全实现。dataclass不校验数值、时间、坐标变换和响应Schema。
现有VLM的v/dt/ts/e/o/d及object不变，不把这些草案字段偷偷塞进HTTP输入。

## 单位、身份与缺失值

所有模块应明确米、弧度、秒，观察ID、坐标系、时钟域、源序列及版本。
缺位姿/速度/协方差为None，不当作零或单位位姿。空地图不表示自由空间。
TrackIdentity包含源轨迹与生命周期；模型窗口短PID不是长期人物身份。

## 数据方向与所有权

- 感知 → 决策/导航：`PerceptionSnapshot`、`PersonTrack`、`PoseEstimate`、`MapReference`。
- agent → VLM/导航：`AgentIntent`任务模式、描述/目标及模式版本，非底盘指令。
- VLM → 导航/agent：`VLMResultEnvelope`绑定观察、请求、映射/模式/候选版本。
- 规划 → 控制：`PathPlan`；控制的动力学轨迹与底盘指令协议在车辆参数明确后再定义。
- 导航 → agent：`NavigationFeedback`真实执行/停止/完成/失败，不能模拟成功。

## 接入与校验边界

未来适配器需独立验证响应、输入版本、源观测年龄和TTL，拒绝迟到/旧映射结果。
RGB资产目前只是引用，逐相机时间/标定/同步容差与ROS传输需在真实接入时扩展。
PersonTrack新增confirmed/score/covariance_order；感知快照明确health/reason。
Header记录timestamp_source，定位头保留定位测量时间，不伪装成点云采集时间。
检测跟踪协方差为行优先10x10，顺序x,y,z,dx,dy,dz,yaw,vx,vy,vz；未知速度仍为None。
缺可靠定位时只有传感器坐标检测，不输出伪造的世界轨迹。
ROS 1专用消息位于perception/detection_tracking/ros/msg；时间、缺失值有效位与共享类型对应，
本机尚未执行catkin构建或真实ROS运行，不宣称消息链路已验收。
接口版本升级与原实现分开测试；不可把类型文件存在当作三域集成已完成。
