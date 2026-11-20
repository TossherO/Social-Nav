# 技术实现：VLM 局部决策

局部 VLM 用于三类任务：T1 识别行人群组关系，T2 匹配并持续核对跟随目标，T3 从候选中选择目标点，当前主要用于轮椅在桌边停靠。本文说明模型服务的访问方式、任务输入，以及结果怎样交给导航接口。

模型、默认后端和各任务 Prompt 由 [project.json](../../decision/vlm/configs/project.json)登记。运行资源和已有结果见[本机记录](../../ENVIRONMENT.local.md)，新服务器从零安装见[SGLang 服务部署方案](SGLang服务部署方案.md)。

外层 Agent 使用独立托管 API，当前优先完备地图下的室内点和区域导航。它通过单模型工具循环读取地图、自主选择连续位姿并检查完成；桌边停靠和持续跟随已接入可选局部任务，T1 群组更新仍待接入。实现见[Agent 长程导航](技术实现-Agent长程导航.md)，命令见[系统运行手册](../系统运行手册.md)，后续事项见[开发测试汇总](../后续开发测试汇总.md)。

## 后端选择

局部 T1、T2、T3 的在线推理，以及离线 Prompt、Schema 和语义调试，默认调用远程 SGLang＋DSPARK。T2 可按测试需要显式选择托管 API。只有远程服务不可用，或所需 GPU 被其他任务占用时，才考虑本地 Transformers＋XGrammar。

切换后端必须记录原因。不能因为某次响应错误就静默换模型或重放请求。代码检查和输入准备不依赖模型服务；当前本地入口也不支持 T3。

持续 T2 的 DSPARK 后端使用 `project.json` 中的服务地址和模型名，并检查服务是否启用 DSPARK。可通过 `--t2-backend hosted` 选择 Agent 的托管 API，或在被忽略的 `configs/t2_backend.local.json` 中设置 `{"backend":"hosted"}`；命令行选项优先，均未指定时使用 `dspark`。托管模式复用 Agent 私有连接配置和用量账本，发送相同的三帧及可选目标参考，关闭思考并要求 JSON 对象；服务端不保证本地候选约束，因此客户端仍严格校验字段、选项和当前源观察。失败不修复、不自动切换后端或重试。两个后端均可通过运行时的 `--t2-jpeg-quality` 将观察图和参考图编码为 JPEG；源 PNG 的摘要仍用于验证三帧来源，实际发送的 JPEG 包含在请求指纹中。实机交互入口默认使用质量 80 的 JPEG，其他运行时入口的默认值和切换方法见[运行手册](../系统运行手册.md#bag-回放调试-t2-跟随)。运行时需显式启用跟随支路，服务和实机闭环的验证范围以实际测试为准。

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

客户端访问 `http://127.0.0.1:30000/v1`。如果改用其他本地端口，需要同时修改调用地址。SSH 能连接只说明网络路径可达，不能证明模型服务已经就绪。

## 仓库远程客户端

先创建私有配置，再填写隧道地址和准确的模型别名。下面的复制命令不会覆盖已有配置：

```bash
cp -n decision/vlm/configs/vlm_server.example.json configs/vlm_server.local.json
chmod 600 configs/vlm_server.local.json
python -B -m decision.vlm.server --server-info
```

[server.py](../../decision/vlm/server.py)只读查询模型列表及可选运行元信息，不发送推理请求。本地隧道不需要 API Key；日志不输出凭证或服务错误正文。远程客户端只需 httpx 和 jsonschema。

T2 和 [T3](#t3目标选择与就座)的任务入口见后文。T1 冻结输入可以复用 [vlm_api_common.make_body](../../decision/vlm/vlm_api_common.py)构造相同的 OpenAI 兼容请求。通用工具负责图像、流式解析及当前响应检查，不再维护旧数据集专用基准。

程序保留原始响应，检查 JSON 格式、当前人物身份、结束原因以及模型是否输出了推理内容。响应中有重复字段时，程序拒绝整条结果。流式请求中断时，已收到的输出正文仍可供复核。程序不会自动修复格式、补删身份或重试。

### T1 与 T3 的内存调用接口

[OnlineDecision](../../decision/vlm/online.py)提供不落盘的 T1／T3 单次请求接口。T3 已由在线桌边工作器调度，连续接近效果尚未实机验证；T1 结果持续接入控制和独立安全仍待实现。T1 调用方在 `DecisionBuffer.consume()` 的租约内使用 `observations.render_window()` 获取绑定的三帧记录和 PNG 字节，再调用 `OnlineDecision.social(scene, window, images, now=...)`。接口校验图片摘要、人物编号与源场景；租约释放后 PNG 字节仍可使用。

`OnlineDecision.seating(scene, candidates, now=..., observation=...)` 根据当前场景、同源 RGB 和本次候选生成模型请求。在线桌边停靠的候选由运行时根据 Agent 圈出的桌沿和对应点云生成。离线调用方也可以提供同一观察的候选集，历史拟合上下文不能作为本轮输入。

T3 桌边工作器在发起请求和使用回复时，分别用采集端记录的源时刻与单调时钟读数推算当前源时间。回放时，经过的单调时间按回放倍率换算。两次检查都以请求所用观察的采集时刻为有效期起点。

调用 `infer()` 才会访问配置的远程 SGLang＋DSPARK。调用方保留完整返回，失败时可用 `discard()` 释放请求。收到结果后，`accept()` 核对 Schema、观察身份和有效期。T1 返回带原始时间的人群关系，将其持续送入控制和独立安全仍待接入。T3 返回局部坐标中的候选位姿。桌边停靠线程检查当前路径，使用有效地图关联转换位姿。外层 Agent 提交目标前再核对任务、地图和车体足迹。

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

## 本地备用推理

[local_inference.py](../../decision/vlm/local_inference.py)使用原有 Transformers、Accelerate 和 XGrammar 引擎。模型从 FP8 显式反量化为 BF16，并按层分配到多张 GPU；不会自动下载权重或把模型卸载到 CPU。

这个入口读取一份由 `make_body` 生成的冻结请求 JSON，目前仅支持 `groups`。新 T2 协议尚未接入本地备用推理。

新机器按实际 Torch 和 CUDA 准备[本地依赖](../../decision/vlm/requirements-local.txt)及可选[融合内核](../../decision/vlm/requirements-local-kernels.txt)，普通包优先中国镜像。已有环境直接复用，路径和 GPU 选择见本机记录：

```bash
cp -n decision/vlm/configs/vlm_local.example.json configs/vlm_local.local.json
python -B -m decision.vlm.local_inference \
  --request /path/to/input/request.json --task groups --dry-run
python -B -m decision.vlm.local_inference \
  --request /path/to/input/request.json --task groups \
  --fallback-reason remote_unavailable --out /path/to/new_local_result
```

实际运行须填写 `--fallback-reason`，允许 `remote_unavailable`或 `remote_gpu_busy`。模型与 GPU 可在私有配置中填写，也可用 `--model-path`和 `--gpus`覆盖。输出位置必须在输入目录和模型目录之外。

`--dry-run`不读私有配置，不加载 GPU，不联网，也不写结果。正式运行只生成一次，原样保存 `response.json`和切换原因。

请求适配通过不等于实际 GPU 推理已验收，具体范围以本机记录为准。当前不能用该入口运行 T3。

## 环境上下文与只读工具

[environment_inputs.py](../../decision/vlm/environment_inputs.py)从感知场景组织当前 RGB、人物、局部环境和可选先验。[Agent 环境工具](../../decision/agent/tools/environment.py)提供只读查询，不控制底盘。场景来源和含义见[环境地图与场景表示](../perception/技术实现-环境地图与场景表示.md)。

这些环境视图不能直接替换已经审核的任务三帧，也不能继承旧输入的人工标签。更换输入后，需要重新核对输入与标签的对应关系。

## 直接运行时的三帧输入与响应接纳

### 从感知观察生成三帧输入

统一 runtime 默认不加载 YOLO。显式记录任务窗口时，它先保存原扫描、RGB、相机内参和人物位姿的只读历史，再由决策后台运行 YOLO、建立二维框与三维人物的对应关系，最后渲染三帧。启用方法见[系统运行手册](../系统运行手册.md#日常运行与回看)。

采集端按整数纳秒选取扫描结束前最近的 RGB，允许同刻，最多回看 100 毫秒。三帧窗口用各自的扫描和 RGB 源时间复查配对，接受的图像年龄为 0 至 150 毫秒，含两个端点。T2 另有 4 秒的结果有效期，从窗口最后一帧的扫描时刻算到控制线程尝试应用结果。

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

周期、响应期限和单请求策略由项目配置维护。该接口本身不发 HTTP 请求，不自动重试，也不会把 `plan`输出当作运动执行。统一 runtime 的自动模型调度仍待接入。

## 实机T1群组输入

T1 使用三帧 RGB 和全高度单帧扫描 BEV。标注审核与导出共用 `Store.model_image`；输入审核使用无损模型图，预览叠层不进入模型输入。

当前 PID 覆盖已确认且有观测的全方向三维轨迹，不按相机视野筛选。缺少 RGB 或可靠图像框时，字段保持 `null`；浏览点云不能代替原始扫描。

| 字段 | 含义 |
|---|---|
| `v` | 最后一份观察中的有效 PID 集合。 |
| `dt`、`ts` | 三帧相对秒数、时间来源及 `clock_domain`。 |
| `e` | 固定米制显示范围 `[xmin,xmax,ymin,ymax]`。 |
| `o` | 每行包含 `[pid,三个XY,三个RGB框,三个人物像素高度]`。RGB 框按各轴归一化到 0–1000。 |
| `d` | 每行包含 `[pid1,pid2,三个同一时刻的米制距离]`。 |

每个 XY 都属于它所在观察的雷达坐标系，因此不能用跨帧坐标差直接计算行人速度。每张 BEV 中的历史轨迹已补偿自车运动；位姿失效时，不伪造世界轨迹。

T1 使用[默认英文 Prompt](../../decision/vlm/configs/prompts/wheelchair_t1_groups.txt)，按 `walking`、`talking`、`photographing`和 `other`组织群组。每个 PID 全局最多出现一次，没有群组时返回 `{}`。[最终英文候选](../../decision/vlm/configs/prompts/wheelchair_t1_groups_en_candidate.txt)仅供显式选择，不会自动切换。动态 Schema 和成员检查由 `vlm_api_common.groups_schema / validate_groups`维护。

标注器的富信息导出仍包含 `g/u`、未知、角色和覆盖信息，不能直接当作紧凑响应的答案。当前关系识别的离线开发已收尾，独立金标准、在线响应处理和导航消费仍需分别验收。最终本机结果由环境记录定位。

## T2持续跟随决策

非执行观察可以显式启用原始 RGB 实验模式。Agent 初次选人仍使用 YOLO 候选并保存选中人物裁剪；进入 T2 后不运行二维检测和二维／三维关联。每次模型请求提供参考裁剪，以及三组原始 RGB 和独立三维俯视图。模型依据视觉、空间和时间信息选择已确认三维轨迹，选择结果不具有独立的二维关联证明；无把握时应等待。RGB 图像按配置编码，不叠加检测框或人物编号。源观察、轨迹身份、时间及结果校验仍然有效。

该模式仅供人工驾驶观察与参考路径预览，不能开启底盘自动执行。现有跟随结束检查仍要求当前 RGB 与目标轨迹的二维关联证据，因此实验模式不能自动证明跟随结束，也不能用于完整任务验收。原始 RGB 模式下的轨迹选择准确率和实时性尚需验证。

[follow_online.py](../../decision/vlm/follow_online.py)提供持续跟随的请求接口。输入是按源时间排序的三帧 RGB／BEV、目标描述、可选固定人物参考和当前观察候选。预测轨迹与真实观察使用不同样式。模型返回两个字段：选中人物时如 `{"target":"P0","observation":null}`；没有人物时如 `{"target":null,"observation":"A"}`，其中 A 是本轮观察点 ID。`target` 和 `observation` 在当前模型分词器中各为单 token，整条 JSON 并非单 token。程序严格检查两项互斥及本轮映射，并校验图像和场景来源，再构造供跟踪层核验的目标参考。

启用 `--agent-follow-candidates` 后，运行时在跟随开始前提供带编号的人物候选，Agent 根据当前图像和任务描述确认目标。系统保存该候选的原图与人物裁剪，并在启动跟随时核对候选、观察和任务身份；T2 使用这张由 Agent 确认的裁剪作为参考。启动后，运行时持续采集因果三帧并在后台准备请求，同一时刻最多一个请求在途。收到并消费结果后使用最新可用观察发起下一次请求，不再按固定 2 秒周期调度。运行时按本次映射与连续三维轨迹应用结果，可纠正目标或进入搜索。结果须在源观察后 4 秒内应用；连续跟踪的视觉确认最多保留 6 秒，遮挡后重新观察到同一连续轨迹时可以继续运动并尽快重新判断，但不会刷新原视觉时间。模型请求失败不自动重试。

`project.json` 中的 `inference.period_s` 仍供通用决策数据录制器使用，不控制 T2 或在线 T3 的连续请求频率。实际调用频率受新观察和模型返回速度限制。

搜索始终提供等待选项；地图关联与车体扫掠有效时，还提供原地转向及向最后观察位置短距离靠近的可行候选。每步最多 0.8 米，运动限制在丢失起点 1.5 米范围内，执行前按当前地图重新检查；缺少有效地图时只能等待，因此尚不能保证抵达较远的最后位置。安全监督仍独立生效。

目前已通过合成控制联动和模拟模型回复的短 bag 影子回放。托管 API 使用 JSON Schema 约束测试了两组真实人物输入和一组移除可选人物的模拟输入，三次回复均符合两字段协议。实际 DSPARK 的正则约束、现场识别、遮挡找回和运动效果仍待验证；少量格式测试不能代表这些能力。

## T3目标选择与就座

T3 根据自然语言指令，从给定候选中选择目标。当前重点是轮椅在桌边停靠，不包含人体转移；狭窄通道倒退让行尚未实现。

就座是 Agent 的可选阶段。轮椅停稳后，Agent 在当前 RGB 中指明目标桌面近侧的边缘范围。T3 工作器使用相机标定把同源扫描与该范围对应，在当前帧验证桌沿后按车体轮廓提出候选。局部 DSPARK 选择候选，运行时检查当前完整车体占据和接近路径，随后按当前地图关联转换目标。靠近过程中，工作器把已观察桌沿投影到新图像，并用新扫描核对桌沿；每次模型返回后使用最新观察进行下一次选择，直到阶段完成、取消或失败。Agent 对新候选再次检查当前静态与局部足迹，再更新同一导航目标；小于 0.15 米且朝向变化不超过 0.15 弧度的更新保留原目标，避免反复重置到达判断。到达后的停车确认期间不替换目标；确认停稳后如新结果有明显变化，重新发起受检查的接近目标。桌沿无法继续确认、模型失败或身份变化时停止并交由 Agent 处理。Agent 结合实测位置、朝向、新 RGB 和定位估计的停稳作完成决定。当前提取器依赖浅色表面和足够的雷达回波；连续停靠尚未实机验证。[在线工作器](../../runtime/seating.py)

代码集中在 [seating/](../../decision/vlm/seating)，统一入口为 `python -m decision.vlm.seating`，各子命令可用 `--help`查看参数：

| 文件 | 职责 |
|---|---|
| [geometry.py](../../decision/vlm/seating/geometry.py) | 处理桌子轮廓和辅助桌边，采样、排序候选，提出观察位姿假设，并逐候选检查是否可规划。 |
| [inputs.py](../../decision/vlm/seating/inputs.py) | 从 bag 抽帧，渲染并冻结单帧或三帧输入，准备辅助桌边复核。 |
| [inference.py](../../decision/vlm/seating/inference.py) | 记录候选集合身份，生成动态 Schema，调用远程模型并保存原始响应。 |
| [stage.py](../../decision/vlm/seating/stage.py) | 离线检验就座的阶段转换、停止确认和反馈适配。 |
| [在线桌边选点](../../runtime/seating.py) | 从同源 RGB 和雷达生成候选，通过局部模型和路径检查后，将建议交给当前 Agent 子任务。 |

### 输入与候选生成

离线评测入口使用近似桌子轮廓和三帧 RGB、BEV。窗口与 T1、T2 一致，选择前 1 秒、前 0.5 秒和当前的实际过去观察。图像共享布局、字体、浅色高度点云及颜色条，但不共用任务标签。

每幅 BEV 使用对应帧的雷达坐标。过去帧显示当时扫描、自车和已知先验，当前帧显示本次 C 候选。保存输入时共同记录来源、RGB 时间、位姿、图像摘要和候选集合，拒绝未来观察或跨定位会话拼接。

| 输入 | 需要提供的内容 |
|---|---|
| 指令 | 要选择方便的桌边停靠位置，并允许多个合理位置。略斜停、脚踏进入桌下等要求应明确写出。 |
| 桌子先验 | 同一局部坐标中的近似矩形、假设高度、来源和最早可用时间。补全角点不能当作实测。 |
| 当前观察 | 有效算法定位、原始扫描及当前或过去 RGB，不要求已有 T1、T2 标签。 |
| 车体参考 | 显式提供足迹和雷达与车体的相对位置。历史资料只作离线假设，见[导航几何参考](../navigation/技术实现-路径规划.md#轮椅几何历史参考)。 |

候选生成沿四条桌边采样位置、朝向和带符号的前缘间距。每一种朝向分别检查足迹是否超出桌边范围，再按近期桌边下方回波、直线距离和转角排序。筛选时先保留不同桌边，再按空间间隔保留最多 8 个候选，策略由 [seating_geometry.json](../../decision/vlm/configs/seating_geometry.json)维护。

前缘间距是“假设足迹最前点到桌边”的距离。只有明确允许探索桌下位置时才使用负值，不能据此允许整个车体或乘员穿过桌面。

近期回波只提供部分高度带的阻碍线索，没有回波不能证明空闲。当前排序不使用真实路径代价，自动生成候选的 `geometry`仍为 `unknown`。模型可以选出语义合理的位置，但执行前还须几何检查。候选 ID 只在本集合内有效，对应关系和响应含义见[接口约定](../接口约定.md#t3目标点选择)。

辅助桌边流程从当前 RGB 的人工区域选回波，累积过去证据拟合支持段，并保留来源截止时间，不自动外推为桌角。对应子命令为 `prepare-edge`和 `refine-edge`。`prepare`用于单帧场景或扫描诊断，当前轮廓流程使用下面的 `prepare-window`。

### 准备与调用

已有同源感知结果时，先用系统 ROS 解释器抽取少量观察。`offsets`相对 bag 起点，示例用三个相隔 0.5 秒的时刻组成窗口：

```bash
/usr/bin/python3 -B -m decision.vlm.seating extract \
  --run /path/to/perception_run --bag /path/to/source.bag \
  --offsets 31 31.5 32 --out /path/to/new_observations
```

定位失效时，抽帧不会退回旧的有效帧；RGB 只使用当前或过去消息。程序保留原运行时钟：直接运行时为 `runtime_bag/runtime_live`，旧 ROS 回放为 `ros_sim_time`。只有来源没有记录且无法明确推断时，才指定 `--clock-domain`，不能用它覆盖冲突记录。

需要规划对照时，在抽帧命令中增加 `--scenes /path/to/current_scene`。当前扫描必须与场景时间、位姿及定位会话一致，也可以依次提供三份同时间场景。派生观察保留原扫描 ID 和场景摘要，不修改源文件，也不为旧冻结候选重新赋予身份。

随后在项目环境生成候选和模型输入，再显式调用远程服务：

```bash
python -B -m decision.vlm.seating prepare-window \
  --observations /path/to/new_observations/O00 /path/to/new_observations/O01 /path/to/new_observations/O02 \
  --table-prior /path/to/table_prior.json \
  --vehicle-reference configs/wheelchair_geometry_reference.json \
  --instruction "在这张桌子旁选择方便的轮椅就座位置，允许略斜停靠，脚踏可以进入桌下。" \
  --out /path/to/new_input
python -B -m decision.vlm.seating run \
  --input /path/to/new_input --base-url http://127.0.0.1:30000/v1
```

准备阶段不调用模型。也可以用 `--candidates /path/to/candidates.json`替代先验、车体和指令参数，读取完整且来源已记录的候选集合。`--current-only`仅用于单帧图像对照，生成候选时仍可使用因果历史。

输出目录必须为新目录。推理前先确认远程 DSPARK 可用，每份输入显式预热一次、正式请求一次，保存原始结果，失败即停。新实验使用新的 `--result-name`，手工指定 Prompt 使用 `--prompt`，不自动转到本地模型。

候选与同一份真实场景对应时，可独立检查导航能否规划到各点，不调用 VLM：

```bash
python -B -m decision.vlm.seating check-navigation \
  --input /path/to/new_input --scene /path/to/current_scene \
  --out /path/to/new_navigation_check
```

该命令复用当前参考规划、足迹和运动配置，检查最多 8 个候选。桌边候选的车体假设必须与导航一致；运动配置不允许倒退时，不规划倒退动作。

结果另存为 `navigation.json`，可查看路径或拒绝原因。它不会修改冻结候选的 `geometry=unknown`，也不会自动改变模型答案或证明实际停靠成功。当前候选排序仍只用局部证据、距离和转角，没有用这些结果重新训练或替换策略。

统一入口按子命令加载依赖：`extract`用系统 ROS，其余准备和推理用项目环境。Prompt、近距路由和 BEV 范围由 [project.json](../../decision/vlm/configs/project.json)维护。路由仅按当前假设车体与候选的距离和转角决定；近距时补充小幅调整提示，不根据模型答案临时更换 Prompt。

模型先判断 RGB 是否足以比较停靠位置，再选择候选。历史帧可以帮助确认桌子，但绿色先验轮廓不能替代当前视觉证据。朝向沙发等无关区域时可以返回证据不足；看到桌沿或桌下时，不要求完整桌面都可见。

同一段开发数据的对照不能证明三帧必然优于单帧，也不能证明已找到全局最方便位置。后续评测见[开发测试汇总](../后续开发测试汇总.md)。

### 阶段与完成反馈

`from decision.vlm.seating.stage import SeatingStage, CompletionLimits`提供进程内 API，不是 ROS 运动节点。调用者显式提供任务身份、完成容差和超时，依次调用 `select → geometry → observe`，并周期调用 `tick`检查超时。

选点后，阶段先请求验证目标。只有收到同任务、同候选且尚未过期的几何反馈，才请求导航。`n/u`当前直接返回失败原因，不自动移动或重试。

成功需要同时满足位置、朝向、桌边关系、定位和几何条件。实际线速度与角速度还须达到停车要求，并连续保持；随后阶段请求停止，收到新鲜停止确认后才返回成功。取消、超时和失效时，也根据控制是否激活请求停止，缺反馈不能算完成。

完成容差由实际设备和任务定义提供，不能用演示末帧或模型回答代替。导航适配接口为 `observe_navigation(request, feedback, navigation_state=..., geometry_valid=..., table_relation_valid=..., now_s=...)`：

- `request`来自同任务的 `TaskPolicy`。
- `feedback`包含任务、目标修订、原请求摘要、定位会话、源时钟和时间、实测 `pose/velocity`、状态及停止确认，并声明真实来源和 `execution_feedback=true`。
- 默认只接受 `scope=robot`。仿真须同时提供 `simulation=True`与 `scope=simulation`，结果保留仿真标记。
- 影子建议和请求接收 ACK 均拒绝。`navigation_state`、桌边关系及几何有效性须来自对应检查，不能由 VLM 或候选选择补造。

### 闭环联调待办

桌边停靠沿用本地参考规划、控制和独立安全。现场需核实以下情况。

1. 核对当前目标桌、候选车体与桌沿间隙、脚踏和乘员包络，分别评估空闲轮廓、接近路径和实际停车位置。
2. 接通观察位姿的导航执行和补充观察调度。现有 `observation_goals`只根据桌外位置、相机视野比例和直线距离提出假设，没有处理遮挡，必须用实拍检查观察收益。没有可达观察点或补充次数用尽时返回原因。
3. 用局部服务及实际驾驶反馈验证正常停稳、无候选、服务失败、取消及超时。候选失效或新观察到达时，由 Agent 决定是否重选。

后续测试分别评价候选生成、模型选择、主动观察和真实执行。同一 bag 的视角来自原驾驶过程，不能记为算法执行观察目标的成果。唯一的本机结果入口见[环境记录](../../ENVIRONMENT.local.md)。

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
