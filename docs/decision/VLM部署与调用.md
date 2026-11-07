# VLM 部署与调用

局部 VLM 用于三类任务：T1 识别行人群组关系，T2 匹配并持续核对跟随目标，T3 从候选中选择目标点，当前主要用于轮椅在桌边停靠。本文说明模型服务的访问方式、任务输入，以及结果怎样交给导航接口。

模型、默认后端和各任务 Prompt 由 [project.json](../../decision/vlm/configs/project.json)登记。运行资源和已有结果见[本机记录](../../ENVIRONMENT.local.md)，新服务器从零安装见[SGLang 服务部署方案](SGLang服务部署方案.md)。

外层 Agent 使用独立托管 API，当前优先完备地图下的室内点和区域导航。它通过单模型工具循环读取地图、自主选择连续位姿并检查完成，不采用本文局部 T3 的候选选点或模型服务。实现见[Agent 技术实现](Agent技术实现.md)，命令见[任务与测试](长程导航任务与评测计划.md#运行与测试操作)，研究与模型选型见[技术计划](Agent技术开发计划.md#模型与调用)。

## 后端选择

局部 T1、T2、T3 的在线推理，以及离线 Prompt、Schema 和语义调试，优先调用远程 SGLang＋DSPARK。只有远程服务不可用，或所需 GPU 被其他任务占用时，才考虑本地 Transformers＋XGrammar。

切换后端必须记录原因。不能因为某次响应错误就静默换模型或重放请求。代码检查和输入准备不依赖模型服务；当前本地入口也不支持 T3。

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

程序分别检查 JSON 格式、当前人物身份的语义、正常结束和非思考输出。原始响应始终保留，不修复格式、不补删身份，也不自动重试。

## 托管API对照（显式选用）

[hosted_evaluation.py](../../decision/vlm/hosted_evaluation.py)可按固定计划，通过Agent已有的托管API测试T1、T2和T3。它复用Agent的私有HTTPS配置、模型和共享费用账本，独立于Agent工具循环；不会改变局部VLM默认后端，也不作为失败后的自动备用。当前开发对照结果由[本机记录](../../ENVIRONMENT.local.md)定位。

计划指定各任务的冻结输入、原始三图、Prompt及摘要，参考答案另存。T2按本次有效回答推进历史。与Agent单张缩略图输入不同，该入口保留三张原图，方便与既有任务结果对照；需要确认所用服务允许接收这些项目数据。

```bash
# 只检查计划，不读取API凭证或联网
python -B -m decision.vlm.hosted_evaluation --plan /path/to/plan.json
# 使用新的结果目录，明确执行计划中的请求
python -B -m decision.vlm.hosted_evaluation --plan /path/to/plan.json \
  --output /path/to/new_comparison --run
```

当前托管API拒绝数组的`uniqueItems`。计划可显式设置`schema_mode: portable`：T1在传输Schema移除该关键字，T2／T3使用平面枚举对象；本地仍按完整原Schema检查去重、PID、状态组合及候选成员。没有实测原`oneOf`的服务端支持情况，不能宣称此模式与DSPARK具有相同的服务端约束。请求失败或响应非法即停止，保留原文，不修复或自动重试。

时延分别记录首输出文本和完整返回，包含每次新建连接及上传，图像准备另计。与旧DSPARK记录比较时，还须说明模型、缓存、预热和测试日期不同，不能把差值单独归因于网络。

## 本地备用推理

[local_inference.py](../../decision/vlm/local_inference.py)使用原有 Transformers、Accelerate 和 XGrammar 引擎。模型从 FP8 显式反量化为 BF16，并按层分配到多张 GPU；不会自动下载权重或把模型卸载到 CPU。

这个入口读取一份由 `make_body`生成的冻结请求 JSON，支持 `groups`、`appearance`和 `follow`。它保持原图像字节、顺序、Prompt、Schema 和输出预算，但不保证与远程逐字一致。

新机器按实际 Torch 和 CUDA 准备[本地依赖](../../decision/vlm/requirements-local.txt)及可选[融合内核](../../decision/vlm/requirements-local-kernels.txt)，普通包优先中国镜像。已有环境直接复用，路径和 GPU 选择见本机记录：

```bash
cp -n decision/vlm/configs/vlm_local.example.json configs/vlm_local.local.json
python -B -m decision.vlm.local_inference \
  --request /path/to/input/request.json --task follow --dry-run
python -B -m decision.vlm.local_inference \
  --request /path/to/input/request.json --task follow \
  --fallback-reason remote_unavailable --out /path/to/new_local_result
```

实际运行须填写 `--fallback-reason`，允许 `remote_unavailable`或 `remote_gpu_busy`。模型与 GPU 可在私有配置中填写，也可用 `--model-path`和 `--gpus`覆盖。输出位置必须在输入目录和模型目录之外。

`--dry-run`不读私有配置，不加载 GPU，不联网，也不写结果。正式运行只生成一次，原样保存 `response.json`和切换原因。需要历史时，调用者依据此前接受的结果构造，不能读取参考答案。

请求适配通过不等于实际 GPU 推理已验收，具体范围以本机记录为准。当前不能用该入口运行 T3。

## 环境上下文与只读工具

[environment_inputs.py](../../decision/vlm/environment_inputs.py)从感知场景组织当前 RGB、人物、局部环境和可选先验。[Agent 环境工具](../../decision/agent/tools/environment.py)提供只读查询，不控制底盘。场景来源和含义见[环境地图与场景表示](../perception/技术实现-环境地图与场景表示.md)。

这些环境视图不能直接替换已经审核的任务三帧，也不能继承旧输入的人工标签。更换输入后，需要重新核对输入与标签的对应关系。

## 直接运行时的三帧输入与响应接纳

### 从感知观察生成三帧输入

统一 runtime 默认不加载 YOLO。显式记录任务窗口时，它先保存原扫描、RGB、相机内参和人物位姿的只读历史，再由决策后台运行 YOLO、建立二维框与三维人物的对应关系，最后渲染三帧。启用方法见[感知导航运行手册](../感知导航运行手册.md#感知导航直接运行时)。

图像准备不会回写三维跟踪或环境地图，也不依赖低频 `--record-views`。准备输入本身不启动 VLM 服务或发送 VLM 请求。

| 实现 | 职责 |
|---|---|
| [runtime/decision.py](../../runtime/decision.py) | 缓存原始主机数组。消费者持有只读租约期间，槽位不能覆盖；后台忙时跳过窗口。 |
| [image_inputs.py](../../decision/vlm/image_inputs.py) | 在可选后台运行 YOLO 和二维、三维关联，根据原观察及相机内参提供图像辅助，不成为导航前置依赖。 |
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

T1 使用三帧 RGB 和全高度单帧扫描 BEV。标注审核与导出共用 `Store.model_image`，绘制来源和范围见[标注手册](决策任务标注.md#模型图像与输入审核)。

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

## T2外观匹配与因果身份回放

[follow.py](../../decision/vlm/follow.py)提供独立离线诊断，复用标注器的三帧 RGB、扫描 BEV 和 HTTP 工具，不调用 T1，也不控制底盘。

`prepare`只读标注工程中的固定事实，另存明确标记的开发输入及参考，不修改标签状态、人物关联或审核时使用的输入。描述参考必须来自同一采集，且不晚于当前观察。人工身份、合理候选及参考答案不会进入模型请求。

### 外观匹配

先准备少量窗口，再手动启动远程 DSPARK 服务并建立连接：

```bash
# 先冻结少量外观匹配窗口；新目录不覆盖既有结果
python -B -m decision.vlm.follow prepare \
  --project /path/to/decision_annotation_project \
  --out /path/to/new_t2_appearance --limit 20

# 手动启动已有远程DSPARK服务并建立所需连接后执行
python -B -m decision.vlm.follow run \
  --out /path/to/new_t2_appearance \
  --base-url http://127.0.0.1:30000/v1
```

外观匹配输出 `id/match`。`match`可以是 `unique`、`ambiguous`、`none`或 `insufficient`，只有 `unique`携带当前有效 PID，其余 `id`均为 `-1`。Prompt 由[外观配置](../../decision/vlm/configs/prompts/wheelchair_t2_appearance.txt)维护。

这个接口只判断当前外观匹配，不输出完整跟随状态。目标没有 PID，也不能自动解释为目标已经离开。

### 按历史顺序回放

需要检验历史时，在同一目标任务中按时间选取已有观察点：

```bash
python -B -m decision.vlm.follow prepare \
  --project /path/to/decision_annotation_project \
  --out /path/to/new_t2_replay \
  --target TARGET_TASK_ID --period-s 2 --limit 8
python -B -m decision.vlm.follow run \
  --out /path/to/new_t2_replay \
  --base-url http://127.0.0.1:30000/v1
```

`TARGET_TASK_ID`是标注工程的目标任务 ID，不是模型 PID。`--period-s`规定已有观察点的最小采样间隔；指定时刻没有标注时，实际间隔可能更长，不补造连续观测。

回放使用[跟随 Prompt](../../decision/vlm/configs/prompts/wheelchair_t2_follow.txt)，返回 `id/state`。`state`包括 `visible`、`occluded`、`uncertain`、`lost`和 `not_found`。历史只由本次运行的正式响应推进，预热不会推进历史；非法响应会终止该序列，不借用真值继续。

程序根据带有生命周期信息的轨迹身份，把上一次接受的目标映射到当前 PID，不能直接复制旧编号。Prompt 只允许用最后一张图 IMAGE_2 左侧 RGB 确认当前视觉状态。标签框、有效三维 PID 和历史外观，都不能单独证明当前可见。

短时保持身份需要有效位姿、连续轨迹、允许的年龄，以及尚未撤销的状态。`occluded`不会刷新最后视觉确认时刻。目前只支持原目标仍在当前有效三维候选中的保持，纯预测轨迹候选尚未接入。

新输入与导航 `TaskPolicy`默认从 `tasks.follow.visual_memory_s`读取保持时间，并与 HTTP 响应期限分别检查。这个开发值考虑了一次推理周期、响应期限和少量调度余量，避免正常调用间隔内过早丢掉已确认身份；它不放宽场景、轨迹或控制反馈的新鲜度要求。旧冻结实验继续使用自身 settings，不重算历史成绩。

### 结果保存与验证范围

每次运行包含一次明确记录的预热和一遍正式请求，不自动续跑或重试。需要比较另一份 Prompt 时，可使用 `--prompt /path/to/prompt.txt --result-name NAME`，在新子目录保存独立历史和原始响应；也可用 `--result-out /path/to/new_result`指定独立结果目录，复用只读输入而不复制图像。两个结果位置选项互斥，都禁止覆盖已有结果。

原生标注草稿提供固定身份、可见性和覆盖事实，不等于描述唯一性或完整五状态金标准。评测须分别报告身份命中、当前可见状态和端到端覆盖。RGB 关联框如果覆盖了前景路人，即使轨迹身份正确，也不能算当前视觉确认。

独立新场景、完整状态覆盖、真实重捕获、T1 与 T2 联合请求、在线时效处理和导航执行仍需验证。保持年龄等诊断设置不是实机安全参数；本机结果由环境记录定位。

## T3目标选择与就座

T3 根据自然语言指令，从给定候选中选择目标。当前重点是轮椅在桌边停靠，不包含人体转移；狭窄通道倒退让行尚未实现。

就座拟作为 Agent 调用的独立阶段：确认桌子，必要时移动观察，再选点、接近停靠，最后确认停止并返回结果。当前已有输入准备、候选、选点和反馈驱动的阶段核心，主动观察调度、Agent 服务及真实运动闭环仍未接通。

代码集中在 [seating/](../../decision/vlm/seating)，统一入口为 `python -m decision.vlm.seating`，各子命令可用 `--help`查看参数：

| 文件 | 职责 |
|---|---|
| [geometry.py](../../decision/vlm/seating/geometry.py) | 处理桌子轮廓和辅助桌边，采样、排序候选，提出观察位姿假设，并逐候选检查是否可规划。 |
| [inputs.py](../../decision/vlm/seating/inputs.py) | 从 bag 抽帧，渲染并冻结单帧或三帧输入，准备辅助桌边复核。 |
| [inference.py](../../decision/vlm/seating/inference.py) | 记录候选集合身份，生成动态 Schema，调用远程模型并保存原始响应。 |
| [stage.py](../../decision/vlm/seating/stage.py) | 管理就座状态、完成条件、停止确认和导航反馈适配；实际反馈源仍待接入。 |

### 输入与候选生成

当前主入口使用近似桌子轮廓和三帧 RGB、BEV。窗口与 T1、T2 一致，选择前 1 秒、前 0.5 秒和当前的实际过去观察。图像共享布局、字体、浅色高度点云及颜色条，但不共用任务标签。

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

同一段开发数据的对照不能证明三帧必然优于单帧，也不能证明已找到全局最方便位置。评测范围见[数据与评测](../experiments/数据与评测.md#t3选点与就座阶段)。

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

地图表示、参考规划、局部控制和独立安全已有实现，T3 也有同源数据检查、逐候选规划和阶段反馈适配。下一步是把这些接口接入持续任务执行，不再针对同一批窗口反复搜索 Prompt。未完成事项统一见[感知导航待办](../感知导航待办与数据采集.md)，具体接入要点如下：

1. 接入人工地图桌子标注、地图身份和有效地图到局部坐标的变换，核对车体、脚踏、扶手和乘员包络，再检查净空、路径及到达方式。
2. 接通观察位姿的导航执行和补充观察调度。现有 `observation_goals`只根据桌外位置、相机视野比例和直线距离提出假设，没有处理遮挡，必须用实拍检查观察收益。没有可达观察点或补充次数用尽时返回原因。
3. 接 Agent 工具以及真实导航和停止反馈，验收成功、失败、取消和超时。目标调度采用事件触发：证据充分时主要做选点，取得新观察或原目标失效后才重选；正常运动由本地导航维持目标，不采用持续周期调用。

后续测试分别评价候选生成、模型选择、主动观察和真实执行。同一 bag 的视角来自原驾驶过程，不能记为算法执行观察目标的成果。唯一的本机结果入口见[环境记录](../../ENVIRONMENT.local.md)。
