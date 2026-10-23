# VLM 部署与调用

在线推理与离线Prompt、Schema、语义和决策逻辑调试均优先使用自有服务器的SGLang＋DSPARK，通过Tailscale与SSH隧道访问。仅远程服务不可用或其所需GPU被其他任务占用时，才考虑本地Transformers＋Accelerate＋XGrammar备用方案。两种入口共用冻结模型输入、Prompt与动态Schema。服务器首次安装和构建只见 [SGLang服务部署方案](SGLang服务部署方案.md)；本文维护后端选择、服务启停与调用。

服务参数由 `decision/vlm/deployment/start_server.sh` 维护，环境与路径由同目录 `runtime_env.sh` 维护；公共客户端策略由 `decision/vlm/configs/project.json` 维护。具体机器、解释器、资源路径、已完成检查和当前运行状态只记录在 `ENVIRONMENT.local.md`。开发环境与模型服务环境分开，客户端只需httpx和jsonschema。

## 后端选择

先选择远程服务，确认服务就绪及所需GPU资源；这里的GPU占用指其他任务造成的资源冲突，SGLang自身常驻权重、显存和正常计算不算冲突。仅远程不可用或存在上述占用时，再按本机记录考虑本地备用，保留相同输入与约束并记录切换原因及实际后端。模型响应不合法或语义不一致仍按失败记录，不触发静默换后端重跑。

`project.json`的`inference.offline_backend`登记默认使用`sglang`。远程入口为`python -m decision.vlm.benchmark_vlm`，本地备用入口为`python -m decision.vlm.local_inference`；两者仍显式选择，不会自动探测GPU占用或切换后端。纯输入、配置与单元检查无需模型推理，`--dry-run`继续不联网、不读取私有连接配置。

## 远程服务启停与访问（默认）

通过已有SSH配置连接自有服务器，进入该机服务根目录；主机别名和路径以本机记录为准。已有环境直接启动，不重复安装或编译：

```bash
ssh USER@SERVER
cd /path/to/sglang-service
tmux new -s sglang-service
# 在tmux会话内执行；使用已测试的默认DSPARK配置。
set -o pipefail
bash start_server.sh 2>&1 | tee logs/server.log
```

等待日志报告就绪。用 `Ctrl+B`、`D` 脱离会话，`tmux attach -t sglang-service` 返回；在服务终端按 `Ctrl+C` 停止。服务器重启后需手动启动。示例只维护最近一轮日志，需保留的故障证据先归入同一实验结果。只有明确需要时才切换配置：`SGLANG_PROFILE=mtp` 使用内置MTP，`mtp_previous` 和 `baseline` 保留旧对照入口；切换前停止已有服务。

在开发机另一个终端建立隧道并保持它运行：

```bash
ssh -N -L 30000:127.0.0.1:30000 USER@SERVER
```

服务器监听回环地址；开发机客户端使用 `http://127.0.0.1:30000/v1`。本机端口被占用时可改变隧道左端端口，并同步客户端地址。SSH可连接不表示模型服务已就绪；服务关闭时不要把连接拒绝当成环境损坏。

## 仓库远程客户端

仓库入口统一使用 `decision.vlm`，旧 `scripts.*` VLM包装已移除。从仓库根目录选择已有客户端解释器，并仅在首次准备时建立私有配置：

```bash
cp -n decision/vlm/configs/vlm_server.example.json configs/vlm_server.local.json
chmod 600 configs/vlm_server.local.json
```

填写隧道对应的 `base_url`、模型别名 `qwen3.8-27b-fp8`，本地隧道无需API Key；`allow_insecure_http` 保持false。不要覆盖已有配置或提交连接信息。模型ID为空时只允许发现唯一匹配的模型，不静默换模型或服务。

按目的选择检查，已有有效结果无需重跑：

```bash
# 离线读取公共配置和输入，不读取私有连接配置，也不发送请求。
python -m decision.vlm.benchmark_vlm --dry-run --limit 24

# 只GET模型列表；输出使用新的临时目录。
python -m decision.vlm.benchmark_vlm --check --out /tmp/vlm_endpoint_check

# 三个场景的代表输入；会提交三次推理。
python -m decision.vlm.benchmark_vlm --limit 3 --out /tmp/vlm_three_scenes

# 仅在需要完整耗时比较时，选择串行或固定到达轮次。
python -m decision.vlm.benchmark_vlm --limit 24 --out /tmp/vlm_serial
python -m decision.vlm.benchmark_vlm --mode paced --rate-hz 0.5 \
  --limit 24 --out /tmp/vlm_paced
```

以上目录必须尚不存在，可用 `--bundle` 指定其他冻结输入。正式计时前可用少量代表性多图输入初始化视觉路径，并单独记录；纯文本 `--warmups 1` 只预热固定前缀，不能覆盖首次视觉执行。比较配置时保持请求、轮次条件和缓存状态可比，客户端不自动清服务缓存。

CLI优先于配置。只传 `--rate-hz` 时deadline沿用1/rate；同时传 `--deadline-s` 则独立设置完成时限。HTTP timeout是等待上限，不是机器人TTL。客户端固定非思考、temperature=0、单在途、零重试和object输出策略；非法配置直接报错。

## 本地备用推理

满足前述备用条件后，先按本机记录核对模型、可用GPU与运行条件，再显式使用本地入口。本地使用独立Conda环境，与检测、ROS及远程服务环境分开。首次安装在本机执行；已有环境直接复用，普通Python库优先中国镜像，PyTorch与TorchVision使用官方cu130源：

```bash
conda create -n social-nav-vlm --override-channels -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge python=3.12 pip
conda activate social-nav-vlm
python -m pip install --index-url https://download.pytorch.org/whl/cu130 \
  'torch==2.13.0+cu130' 'torchvision==0.28.0+cu130'
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  -r decision/vlm/requirements-local.txt
conda env config vars set PYTHONNOUSERSITE=1 'PYTHONPATH=' 'PYTHONHOME='
conda deactivate
conda activate social-nav-vlm
```

从仓库根目录准备专用本机资源覆盖，只填写模型路径和GPU，不读取远程API配置：

```bash
cp -n decision/vlm/configs/vlm_local.example.json configs/vlm_local.local.json
# 在该文件填写实际模型目录和可用GPU，然后运行：
python -m decision.vlm.local_inference --dry-run --limit 3
python -m decision.vlm.local_inference --limit 3 --out /tmp/vlm_local_debug
```

可用 `--model-path`、`--gpus 0,1,2,3` 显式覆盖资源，`--bundle` 指定相同协议的冻结输入，`--case-id` 选择具体窗口。默认只跑三个代表场景，输出必须是新目录且位于模型、冻结输入之外。模型常驻单个进程，多卡按层分摊一份模型，逐个处理样本，不创建HTTP服务、不联网取模型或自动重试。

需要GDN融合加速时，在该独立环境额外安装 [内核依赖](../../decision/vlm/requirements-local-kernels.txt)。`fla-core`提供当前模型需要的算子，不必安装FLA的模型层包；`causal-conv1d`是原生CUDA扩展。先按本机记录设置与Torch匹配的CUDA Toolkit及`CC`／`CXX`，再执行：

```bash
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  -c decision/vlm/requirements-local-kernels.txt ninja
CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS=4 python -m pip install \
  -i https://pypi.tuna.tsinghua.edu.cn/simple --no-build-isolation \
  -c decision/vlm/requirements-local.txt -r decision/vlm/requirements-local-kernels.txt
python -m pip check
```

原生扩展按当前Torch／CUDA／ABI构建，避免使用与系统glibc不匹配的预编译包；FLA的Triton算子在首次使用时编译，初始化耗时应与稳定请求耗时分开。结果元数据的`gdn`记录48层所选预填充、递归解码和卷积实现；`fast_path_available`或导入成功只证明可选择该路径，实际启用还应由少量请求中的函数调用／GPU执行验证。缺少这些依赖时仍可用PyTorch兼容实现，不能将其速度当作融合算子成绩。

针对当前FP8资产，本地显式反量化为BF16加载，保留原始文件；这不能恢复量化前精度，也不保证与服务器FP8＋DSPARK逐字一致。权重分摊预算由公共配置维护，另需给视觉激活和缓存留出显存；设备不足时报错，不静默卸载到CPU或切换远程。模型路径与GPU实际情况见本机记录，不把另一设备的资源选择直接照搬。

两端共用相同三图字节、`IMAGE_i`顺序、system／user文本、聊天模板、非思考设置和输出上限。本地使用相同版本XGrammar编译每个窗口的动态Schema，在 `generate()` 的每一步约束token，并为每次生成新建匹配状态；空白规则与现有SGLang一致。结构约束之外仍执行同一PID成员、跨组互斥、角色和完整结束校验，保留原始响应，错误或截断立即停止。

只保留 `settings.json`、`requests.jsonl` 和 `summary.json`：记录模型小资产摘要、精度与设备分摊、实际输入token及图像网格、生成token和原始／合法结果。加载与生成时间用于排错，离线不判定在线deadline，也不伪装成HTTP／网络性能。`reference_request_sha256` 是同输入的参考HTTP正文摘要，正文未发送，不能当作一次真实网络请求。

需要与已有远程结果对齐时，在本地命令中显式加 `--reference /path/to/remote_result`；只在全部本地生成结束后读取参考答案，要求原始输入指纹和参考正文摘要一致，再比较原始文本、关系结构与输入token数。参考结果不进入Prompt、不指导重试或输出修复。初次用少量代表样本确认输入和约束，后续按实际语义问题追加；复用已运行的远程服务与仍有效的结果，不为每次修改重复部署或重跑整套基线。

## 独立客户端与部署材料

已有服务器客户端可直接复用。需要迁移时，由现有打包器生成新目录和ZIP，不覆盖旧包：

```bash
# 普通包只包含冻结输入、Prompt、客户端与默认值。
python -m decision.vlm.package_vlm_test --out /tmp/jrdb_client

# 首次部署新服务器时，另附当前源码构建材料与两份手册。
python -m decision.vlm.package_vlm_test --with-deployment --out /tmp/jrdb_service_bundle
```

可用 `--source` 指定输入。打包只复制每case白名单metadata／三图、固定Prompt、客户端及公开配置；不带原生社会标签、人工答案、源人物身份、权重、凭证、环境、编译缓存或旧结果，也不安装、构建、启动服务或请求模型。

普通包README来自便携客户端模板。含部署材料的包额外携带当前启动／环境脚本、cu129约束、源码补丁与构建入口，以及本手册和SGLang手册；它不再携带替换原环境的旧安装器。新包导出公共默认值，已经填写的 `server_config.json` 和已有结果不随源码更新覆盖。

在解压包根目录运行；仅全新客户端环境需要安装其依赖，普通库优先中国镜像：

```bash
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
# 填写server_config.json后，使用相同客户端命令。
python benchmark.py --dry-run --limit 24
python benchmark.py --limit 3 --out /tmp/jrdb_portable_smoke
```

服务器本地调用使用回环地址；开发机调用使用SSH隧道。两端使用同一输入与推理设置，但分别创建新的结果目录，不复制填写过的连接配置。

## 请求、结果与耗时

JSON Schema由客户端提交，返回后独立检查Schema、PID成员、角色、模型ID、正常流结束和非思考状态。保留原始响应，不修复格式、补删身份、改变关系或追加重试；合法JSON不证明语义正确，也不单独证明服务端硬约束已生效。默认忽略环境代理、单连接、不跟随重定向；可信代理只能通过 `--proxy` 显式指定。

serial逐个等待完整结果；paced按计划到达，单请求在途，忙时丢弃而不排队。失败立即停止，停止后的未提交任务单独记录；迟到但合法的结果仍统计。机器人是否接受结果仍需未来的观察时间、身份、任务与TTL适配器。

| 字段 | 含义与范围 |
|---|---|
| `prepare_s` | 读取已渲染图片、base64与请求编码 |
| `response_headers_s`／`first_event_s` | HTTP开始至响应头／首个SSE事件，首事件可能只有元数据 |
| `first_output_text_s`／`last_output_text_s` | 首段／末段最终content，末段不一定是流结束 |
| `stream_complete_s`／`api_complete_s` | 完整流／HTTP上下文结束，须有DONE和正常finish_reason |
| `validation_s` | 本地输出校验 |
| `end_to_end_s` | 请求准备至完整校验，不含模型加载、实时感知或源渲染 |
| `arrival_to_result_s` | paced计划到达至完整结果，包含启动滞后 |

每轮保存 `settings.json`、`model_check.json`、`requests.jsonl` 和 `summary.json`；显式文本预热另记，不混入场景耗时。统计计划、提交、合法、丢弃和取消数量，以及合法子集耗时和按时覆盖。必要原始结果与唯一汇总集中在同一实验目录，临时检查不进入长期产物。

只读比较同请求结果，可在仓库运行：

```bash
python -m decision.vlm.deployment.compare_results \
  --reference /path/to/reference --candidate /path/to/candidate
```

含部署材料包可使用 `python compare_results.py`。比较器要求请求／输入指纹绑定一致，不仅凭case名配对，也不改摘要绕过差异。远程与同机差值包含客户端、链路、服务、缓存和轮次波动，不是纯RTT或GPU时间。API的缓存值为0也不证明没有缓存，需结合服务设置及已有日志。

默认JRDB输入是带oracle几何与PID的冻结开发窗口，没有人工语义金标准；输入生成、标签和审核边界见仓库 `docs/experiments/JRDB参考.md`。三图短测通过不能代替长期运行、满上下文、语义准确率或实机感知—渲染—决策链路验证。T2／T3输入、在线状态接收和外层agent仍待实现。已有耗时报告由 `ENVIRONMENT.local.md` 定位，不在本手册重复维护成绩表。

## 环境上下文与只读工具

感知环境的运行和来源边界由[建图与定位](../perception/技术实现-环境地图与场景表示.md#静态环境与视觉记忆)维护。决策侧有两个独立数据适配入口，均不发起模型请求或控制底盘：

| 入口 | 用途与约束 |
|---|---|
| [environment_inputs.py](../../decision/vlm/environment_inputs.py)的`prepare_scene_context`／`render_environment_context` | 直接消费[统一场景](../perception/技术实现-环境地图与场景表示.md#统一场景输入)，绑定RGB、几何、人物、记忆和可选先验关联。RGB＋BEV中人物使用P编号，保留轨迹身份与观察／预测状态，历史位置使用H编号；不投影未经同安装标定绑定的RGB框。基础`prepare_environment_context`仍可用于仅几何／RGB输入。 |
| [environment_tools.py](../../decision/agent/environment_tools.py)的`query_scene`／`query_environment` | 前者查询同一场景的元数据、RGB和几何，后者查询已有环境包的地点、观察及地图。请求绑定身份／指纹，保持未知语义和响应预算；没有导航目标执行或agent主循环。 |

统一场景视图使用`environment_scene_context`并保留原场景指纹，基础视图仍为`environment_context`；不自动替换既有REL三帧输入、Prompt或Schema，也不复用原协议人工标签。正式模型任务接入与语义评价仍待完成。渲染命令见[场景速查](../perception/感知操作速查.md#统一场景与混合模式)，复用已有NumPy／OpenCV及Pillow／Matplotlib，不增加普通HTTP客户端的依赖或启动模型服务。
