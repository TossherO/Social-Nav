# 技术实现：VLM 局部决策

局部决策包含三类任务。T1 识别行人群组关系，T2 匹配并持续核对跟随目标，T3 从候选中选择目标点，当前主要用于轮椅在桌边停靠。三类任务均有 VLM 实现，T2 的本地行人重识别由独立的 `decision/osnet/` 包实现。本文维护共用的服务、观察、请求校验和标注流程。各任务的输入、算法和使用方式见下表。

| 任务 | 技术说明 |
|---|---|
| T1 群组识别 | 用三帧观察识别行人群组，输出关系和成员。[输入、判据与本地备用推理](技术实现-T1群组识别.md) |
| T2 行人跟随 | 从 Agent 选人开始，用 VLM 或本地 OSNet 持续核对身份并更新导航目标。[跟随、丢失与恢复](技术实现-T2行人跟随.md) |
| T3 桌边停靠 | 根据桌沿及车体几何生成候选，选点后核对接近路径和停稳反馈。[候选生成与调用](技术实现-T3桌边停靠.md) |

模型、默认后端和各任务 Prompt 由 [project.json](../../decision/vlm/configs/project.json)登记。本机使用的运行资源和已有结果保存在仓库根目录的 `ENVIRONMENT.local.md`。新服务器部署见[SGLang 服务部署方案](SGLang服务部署方案.md)。

外层 Agent 使用独立托管 API，负责长程任务登记、导航和局部技能调度。实现见[Agent 长程导航](技术实现-Agent长程导航.md)，命令见[系统运行手册](../系统运行手册.md)，后续事项见[开发测试汇总](../后续开发测试汇总.md)。

## 后端选择

局部 VLM 在线推理以及 Prompt、Schema 和语义调试默认调用远程 SGLang＋DSPARK。T2 另支持[显式选用托管 API 或本地 ReID](技术实现-T2行人跟随.md#观察与后端)。远程服务不可用或所需 GPU 被占用时，T1 可以使用 [Transformers＋XGrammar 备用入口](技术实现-T1群组识别.md#本地备用推理)。

切换后端必须记录原因。单次响应错误按所选后端的失败流程处理，禁止静默换模型或重放请求。代码检查和输入准备无需模型服务。

## 远程服务启停与访问（默认）

先使用已有 SSH 配置连接服务器，在服务目录手动启动：

```bash
ssh USER@SERVER
cd /path/to/sglang-service
tmux new -s sglang-service
bash start_server.sh > logs/server.log 2>&1
```

日志显示服务就绪后，可以脱离 tmux 会话。需要停止时回到该会话按 Ctrl+C。运行参数由[部署脚本](../../decision/vlm/deployment/start_server.sh)维护，客户端不会自动安装、编译或启动服务。

再在开发机建立 SSH 隧道：

```bash
ssh -N -L 30000:127.0.0.1:30000 USER@SERVER
```

客户端通过隧道访问本机服务端口。项目在线 T2 和 T3 从 `project.json` 读取 `deployment.host`、`deployment.port` 和 `model.served_name`。在车载机建立隧道时按这些值核对地址、端口和模型。用下一节的只读查询检查服务是否就绪并启用了 DSPARK。

## 仓库远程客户端

先创建私有配置，再填写隧道地址和准确的模型别名。下面的复制命令不会覆盖已有配置：

```bash
cp -n decision/vlm/configs/vlm_server.example.json configs/vlm_server.local.json
chmod 600 configs/vlm_server.local.json
python -B -m decision.vlm.server --server-info
```

[server.py](../../decision/vlm/server.py)使用这里的私有配置查询模型列表及可选运行元信息。要用它检查在线 T2、T3 实际访问的服务，请填写与 `project.json` 相同的地址、端口和模型。查询不发送推理请求。本地隧道可使用空 API Key，日志不输出凭证或服务错误正文。远程客户端只需 httpx 和 jsonschema。

T1 冻结输入可以复用 [vlm_api_common.make_body](../../decision/vlm/vlm_api_common.py)构造相同的 OpenAI 兼容请求。通用工具负责图像、流式解析及当前响应检查，不再维护旧数据集专用基准。

程序保留原始响应，检查 JSON 格式、当前人物身份、结束原因以及模型是否输出了推理内容。响应中有重复字段时，程序拒绝整条结果。流式请求中断时，已收到的输出正文仍可供复核。程序不会自动修复格式、补删身份或重试。

### T1 与 T3 的内存调用接口

T1 的 [GroupsDecision](../../decision/vlm/t1/online.py) 和 T3 的 [SeatingDecision](../../decision/vlm/t3/online.py) 共用[发送和校验流程](../../decision/vlm/online.py)。

T1 调用方在 `DecisionBuffer.consume()` 租约内使用 `observations.render_window()` 获取对应的三帧记录和 PNG 字节，再调用 `GroupsDecision.social(scene, window, images, now=...)`。接口校验图片摘要、人物编号与源场景，租约释放后 PNG 字节仍可使用。

`SeatingDecision.seating(scene, candidates, now=..., observation=...)` 使用当前场景、同源 RGB 和本次候选准备请求。候选可由[在线桌边工作器](技术实现-T3桌边停靠.md#在线选点与接近)生成，也可由离线调用方提供与本次观察对应的候选集。历史拟合上下文不能作为本轮输入。

T3 桌边工作器在发起请求和使用回复时，分别用采集端记录的源时刻与单调时钟读数推算当前源时间。回放时，经过的单调时间按回放倍率换算。两次检查都以请求所用观察的采集时刻为有效期起点。

调用 `infer()` 才会访问配置的远程 SGLang＋DSPARK。调用方保留完整返回，失败时可用 `discard()` 释放请求。`accept()` 核对 Schema、观察身份和有效期后，T1 返回带原始时间的人群关系，T3 返回局部坐标中的候选位姿。T1 的控制和独立安全消费仍待接入。

## 托管API对照（显式选用）

[hosted_evaluation.py](../../decision/vlm/hosted_evaluation.py)可按固定计划，通过 Agent 已有的托管 API 测试 T1 和 T3。它复用 Agent 的私有 HTTPS 配置、模型和共享费用账本，不改变局部 VLM 默认后端，也不作为失败后的自动备用。新 T2 协议尚未接入此入口。

计划指定各任务的冻结输入、原始三图、Prompt及摘要，参考答案另存。该入口保留三张原图；需要确认所用服务允许接收这些项目数据。

```bash
# 只检查计划，不读取API凭证或联网
python -B -m decision.vlm.hosted_evaluation --plan /path/to/plan.json
# 使用新的结果目录，明确执行计划中的请求
python -B -m decision.vlm.hosted_evaluation --plan /path/to/plan.json \
  --output /path/to/new_comparison --run
```

当前托管 API 拒绝数组的 `uniqueItems`。计划可显式设置 `schema_mode: portable`：T1 在传输 Schema 移除该关键字，T3 使用平面枚举对象；本地仍按完整原 Schema 检查去重和候选成员。请求失败或响应非法即停止，保留原文，不修复或自动重试。

时延分别记录首输出文本和完整返回，包含每次新建连接及上传，图像准备另计。与旧DSPARK记录比较时，还须说明模型、缓存、预热和测试日期不同，不能把差值单独归因于网络。

## 环境上下文与只读工具

[environment_inputs.py](../../decision/vlm/environment_inputs.py)从感知场景组织当前 RGB、人物、局部环境和可选先验。[Agent 环境工具](../../decision/agent/tools/environment.py)提供只读查询，不控制底盘。场景来源和含义见[环境地图与场景表示](../perception/技术实现-环境地图与场景表示.md)。

这些环境视图不能直接替换已经审核的任务三帧，也不能继承旧输入的人工标签。更换输入后，需要重新核对输入与标签的对应关系。

## 直接运行时的三帧输入与响应接纳

### 从感知观察生成三帧输入

统一 runtime 默认不加载 YOLO。显式记录任务窗口时，它先保存原扫描、RGB、相机内参和人物位姿的只读历史，再由决策后台运行 YOLO、建立二维框与三维人物的对应关系，最后渲染三帧。启用方法见[系统运行手册](../系统运行手册.md#日常运行与回看)。

采集端按整数纳秒选取扫描结束前最近的 RGB，允许同刻，最多回看 100 毫秒。三帧窗口用各自的扫描和 RGB 源时间复查配对，接受的图像年龄为 0 至 150 毫秒，含两个端点。

图像准备不会回写三维跟踪或环境地图，也不依赖低频 `--record-views`。准备输入本身不启动 VLM 服务或发送 VLM 请求。

| 实现 | 职责 |
|---|---|
| [runtime/decision.py](../../runtime/decision.py) | 缓存原始主机数组。消费者持有只读租约期间，槽位不能覆盖；后台忙时跳过窗口。 |
| [image_inputs.py](../../decision/vlm/image_inputs.py) | 为窗口运行 YOLO 和二维、三维关联。相机内参须与 RGB 的坐标系和分辨率一致，源时间不得晚于图像。缺少匹配内参时保留二维检测，三维关联标为不可用。 |
| [observations.py](../../decision/vlm/observations.py) | 选择前 1 秒、前 0.5 秒和当前观察，检查图像来源，保存三幅图、字段、人物映射和对应场景。 |
| [navigation.py](../../decision/vlm/navigation.py) | 用 `DecisionRequest`管理单个请求的发起、响应使用、过期和取消，委托导航 `TaskPolicy`检查结果。 |

当前窗口的人物来自全方向三维跟踪，只使用已确认且在该帧有观测的目标。RGB 只绘制与同源 YOLO 检测可靠匹配的框。图像必须来自当前或过去，且时间差满足要求；缺图不补造，定位或跟踪重置后清空历史。

生成的 `decision_inputs/<观察时间_标识>/input.json`和三幅 PNG 是新的派生输入。它们不包含人工答案，不继承旧标签，也不包含供规划使用的全套地图。

新后台窗口还保存图像模型、标定来源和逐观察关联记录。旧记录仍可只读复用。图像支路失败时，只将该支路标为不完整，主链继续运行；两者共用 GPU 时仍可能竞争资源。

`model_requests=0`表示没有 VLM 请求，不表示没有运行 YOLO；后者由 `image_detector_calls`另行记录。离线索引和二维缓存仍使用既有 `prepare/prepare-rgb`入口。

### 如何把响应交给导航

调用者需要保留发起请求时的原 `SceneInputs`。先把它和同源冻结窗口传给 `DecisionRequest.begin`，并使用返回的 Schema。收到响应后，再用最新场景和相同源时钟调用 `accept`。

人物映射仍以原窗口为准，不能用新窗口的 PID 解释旧响应。PID 是当前模型窗口内的编号，不是永久人物身份。响应即使非法或迟到，也保留在 `last_response`中，是否写入持久存储由调用者负责。

周期、响应期限和单请求策略由项目配置维护。该接口本身不发 HTTP 请求，不自动重试，也不会把 `plan`输出当作运动执行。T2 和 T3 已有专用后台工作器。T1 的群组结果仍待接入连续调度。

## 决策任务标注

`decision.vlm.annotation` 读取已有感知结果和原 bag，分别标注 T1 人物关系与 T2 目标身份；服务和数据与行人中心标注器独立。Linux 工程保存源时间、模型和标定身份，原 bag、感知结果及人工标签不被覆盖。未审核草稿、漏检和未知身份不能当作金标准，也不能用未来信息或人工身份修改模型输入。

在 Linux 选择尚不存在的工程目录：

```bash
python -B -m decision.vlm.annotation prepare \
  --results /path/to/perception_run --root-group recording_id \
  --output data/annotations/vlm_decision/new_project
python -B -m decision.vlm.annotation open --project data/annotations/vlm_decision/new_project
```

工程只保存来源和时间索引，访问图像或点云仍需要原 bag；已有二维关联沿用原记录，不默认重跑模型。旧数据缺关联时需要显式指定 `--with-rgb-associations`。跨机器迁移先核对原来源、校准和时间，不改写冻结摘要。

### RGB框来源

二维框与三维轨迹的对应结果称为 RGB 关联。已有工程可以独立准备或恢复该阶段，不必重新定位和三维推理：

```bash
python -B -m decision.vlm.annotation prepare-rgb --project data/annotations/vlm_decision/new_project
```

程序优先复用完整兼容缓存，有保存的二维记录时只做 CPU 导入；旧来源没有缓存或记录时才运行 YOLO。明确重算时使用 `--recompute --output` 指定新目录。打开、服务和导出均读取工程登记来源；来源缺失、未完成或不匹配会报错，不静默切换。无关联来源的旧工程仍使用其原投影模式。

只有可靠匹配到已有三维轨迹的二维框进入模型图。切换来源后重启窗口或服务；新框改变模型输入，已有标签须重新审核，过期输入不能正式导出。原始观察和人工标签不因此改写。

Linux 可为 Windows 客户端提供服务；只读是默认行为，`--write` 表示允许编辑，应绑定可信网络：

```bash
python -B -m decision.vlm.annotation serve \
  --project data/annotations/vlm_decision/new_project \
  --project-id wheelchair-vlm --host LINUX_TAILSCALE_IP --port 8766 --write
python -B -m decision.vlm.annotation package-client \
  --output outputs/decision/annotation_client/client.zip
```

Windows 解压便携包，在已有 Python 3.11 环境安装 `decision/vlm/annotation/requirements-client.txt`，然后连接：

```powershell
python -B -m decision.vlm.annotation open --server http://LINUX_TAILSCALE_IP:8766 --project-id wheelchair-vlm
```

T1 按观察分别记录同行、交谈、拍照、其他活动与未知归属，不自动把未处理人物当作独行。T2 记录初始可见描述、实际人物与轨迹片段、可见性和覆盖；三维跟踪 ID 可能改变，人工身份不回填模型输入。未同步的客户端草稿不是服务器已保存结果。

审核后在 Linux 导出到新目录；只有明确要求 `--include-inputs` 才复制不含标签答案的输入：

```bash
python -B -m decision.vlm.annotation export \
  --project data/annotations/vlm_decision/new_project --output outputs/decision/annotations/new_export
```
