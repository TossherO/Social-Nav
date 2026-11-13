# AutoDL 服务部署与调用（历史归档）

本文保留此前云端容器／CUDA 13部署方案，仅供追溯。旧原环境安装器及其profile已从当前代码移除，既有便携包和用户结果保持原样；不要按本文操作当前自有服务器或开发环境。当前操作见 [VLM部署与调用](../decision/技术实现-VLM局部决策.md)与 [SGLang服务部署方案](../decision/SGLang服务部署方案.md)。

在Ubuntu 20.04／glibc 2.31的双4090／550驱动设备，通过原生Conda与SGLang 0.5.19源码构建服务环境，按[SGLang服务部署方案](../decision/SGLang服务部署方案.md)执行。下面的安装器与服务包装适用于既有AutoDL／CUDA 13路线，不用于该新环境；客户端及便携包规则仍由本手册维护。

适用基础镜像：**PyTorch 2.12＋Python 3.12＋CUDA 13.0**。目标服务为冻结的 `Qwen3.8-27B-FP8`，串行、非思考、JSON Schema、每两秒调用一次。普通 AutoDL 容器即可，不需要嵌套 Docker。

## 仓库客户端与便携包

本指南是项目 `docs/decision/技术实现-VLM局部决策.md` 的唯一维护源；带部署工具的便携包将它复制为README。下面先区分仓库入口与独立包，后续安装/服务命令均在已解压的便携包根目录执行。

仓库内工具统一使用 `decision.vlm` 命令入口，旧 `scripts` VLM 包装已移除。T1渲染/人工审核和源数据字段由 `docs/experiments/JRDB参考.md`维护；T2/T3专用输入、在线VLM状态适配与外层agent仍待实现，不根据静态Schema推断能力完成。

从仓库根目录选择客户端解释器：

```bash
cp -n decision/vlm/configs/vlm_server.example.json configs/vlm_server.local.json
chmod 600 configs/vlm_server.local.json

# 离线加载公共配置/输入，不读取私有连接配置或发送请求。
python -m decision.vlm.benchmark_vlm --dry-run --limit 24

# 生成普通独立客户端；加入部署工具时使用第二条，分别选择新输出。
python -m decision.vlm.package_vlm_test --out outputs/t1/portable_new
python -m decision.vlm.package_vlm_test --with-deployment \
  --out outputs/t1/autodl_sglang_bundle_new
```

仅在实际调用前填写本机连接配置；不覆盖已有配置。打包只复制Prompt、每case白名单metadata/三图、同一客户端和默认值快照；不复制native、人工标签、源人物身份、权重、凭证或旧结果，也不安装环境或请求模型。可用 `--source` 指定新bundle，输出目录/ZIP已存在时拒绝覆盖，已有ZIP不会自动更新。

普通包使用便携测试模板作为README；部署包使用本指南作为README，通用客户端模板另存benchmark_guide.md。新包的默认值由公共配置导出，已经填写的server_config.json及用户结果不随源码更新覆盖。

仓库客户端的实际调用例子：

```bash
# 仅GET检查；其后两条会提交模型推理，按实际目的选取。
python -m decision.vlm.benchmark_vlm --check --out outputs/t1/check_new
python -m decision.vlm.benchmark_vlm --bundle outputs/t1/dev_new \
  --limit 3 --out outputs/t1/smoke_new
python -m decision.vlm.benchmark_vlm --mode paced --rate-hz 0.5 \
  --limit 24 --out outputs/t1/paced_new
```

CLI优先于配置；只覆盖rate未指定deadline时，deadline沿用1/rate兼容规则；同时传两者则独立控制。只改deadline不改变到达率。HTTP timeout是等待上限，不是机器人TTL。固定非思考、temperature=0、单在途、零重试和object策略由客户端校验，不支持的配置报错。

仅在唯一匹配模型ID存在时允许自动发现，否则明确填写；不静默换服务或付费API。默认忽略环境代理、单连接、不跟随重定向；可信代理须显式 `--proxy`，认证信息不放URL或公开日志。

## 请求与计时

serial测逐个请求完整结果；paced按计划到达，单请求在途，忙时丢弃而不排队。失败立即停止，不重试或修复输出；停止后的未提交任务单独记录。迟到但合法的结果仍统计，是否被机器人接受要由未来身份/任务/TTL适配器决定。

TTFT为第一段最终文本，SSE首事件可能仅含元数据；完整JSON和正常流结束之后才得到可用结果。日志字段分工为：

| 字段 | 含义与范围 |
|---|---|
| prepare_s | 当前请求构造、读取图片/base64及JSON编码，源图已离线渲染 |
| response_headers_s / first_event_s | HTTP开始到响应头/首个SSE事件，均非纯模型执行 |
| first_output_text_s / last_output_text_s | 第一段/最后一段最终content，后者未必是流结束 |
| stream_complete_s / api_complete_s | 完整流/HTTP上下文结束，需要DONE和正常finish_reason |
| validation_s | Schema、成员、结束状态等局部校验 |
| end_to_end_s | 请求准备到完整校验，不含实时感知、源渲染或模型加载 |
| arrival_to_result_s | paced计划到达至完整结果，含启动滞后 |

运行保存settings.json、model_check.json、requests.jsonl和summary.json；一次纯文本预热仅显式warmups=1时使用，单独记录，不混入场景延迟。记录计划/提交/合法/丢弃/取消数量以及全请求、合法子集和按时覆盖，JSON合法不证明语义正确。

输入运输指纹、请求指纹和人工审核指纹用途不同，不跨用途比较；结果比较必须来自相同输入。远程与同机日志差值是链路综合差，不能当纯RTT。日志仍含人物关系或模型路径，分享前检查，不能当无隐私资料。

重复输入的缓存状态会影响耗时；比较远程与同机调用前，核对实际服务参数及缓存日志。服务未启用缓存报告时，API的 `cached_tokens=0` 不证明没有缓存，不能把首次输入与已缓存输入的差值归因于网络。

## 先理解环境边界

本包按用户选择，**直接修改运行安装脚本的原有 Python 环境，不创建新的虚拟环境**。执行安装后，镜像原来的 PyTorch 2.12 将替换为 **2.13.0＋cu130**，并安装 SGLang **0.5.19** 及其配套依赖，因为该 SGLang 发行元数据要求 `torch==2.13.0`。Python 3.12、宿主机驱动和系统 CUDA Toolkit 不由脚本修改；但 TorchVision、TorchAudio、Transformers 等现有 Python 包可能被替换。不要通过 `--no-deps` 强行保留不兼容的 Torch。

**仅在准备用于该服务的 AutoDL 容器中执行安装，不在本项目开发机的 `python312` 或检测环境中执行。**安装前保存可恢复的容器镜像，停止自己正在运行的模型服务；原环境的其他应用可能受影响。包版本记录不是完整环境备份，脚本不提供自动回滚。若必须保留 Torch 2.12，不执行本包安装，应另选并验证完整依赖组合。

关键约束由安装 profile 维护：仓库路径为 `decision/vlm/deployment/installation_profile.json`，便携包内为 `installation_profile.json`，包括 Torch、TorchVision、TorchAudio、SGLang、原生 kernel、FlashInfer、Transformers、XGrammar 和客户端依赖。便携包另外导出 `requirements.server.txt` 供阅读；安装器使用同一 profile 中的固定版本。

这是关键依赖约束，**不是全部传递依赖的永久锁文件**。预下载的 wheelhouse 另附全部实际解析版本及 SHA-256 锁文件，离线安装时使用该完整锁文件。安装后必须通过 `pip check`，并保存实际包版本。0.5.19 曾在本机 CUDA 12/H800 上测试；本包的 CUDA 13/远程 GPU 组合尚未执行实际安装或推理，不承诺直接复现旧延迟。

依据：2026-10-07 核对的 [SGLang 官方发行依赖](https://pypi.org/pypi/sglang/0.5.19/json)、[PyTorch CUDA 13 轮子](https://download.pytorch.org/whl/cu130/torch/)及 [SGLang 安装文档](https://docs.sglang.io/docs/get-started/install)。不使用浮动的 `latest`。

安装索引同时包含 PyPI、PyTorch cu130 和 [NVIDIA 官方索引](https://pypi.nvidia.com/cuda-tile/)。2026-10-08 已核对 `cuda-tile==1.6.0rc5` 的 Python 3.12/Linux x86_64 wheel 在 NVIDIA 索引中；PyPI 该版本只有用于获取 wheel 的占位源码包。当前安装器与 `requirements.server.txt`均包含 NVIDIA 索引和此依赖，不需要手动预装、源码编译或改用另一版本。

## 1. 租用与存储

- 推荐先用一张 **4090 48GB** 与原应用作对照，或使用更大显存的兼容 GPU。普通 4090 24GB 不在本包部署范围内。CPU/内存也影响视觉预处理和加载，建议至少 32GB 内存，条件允许选择 64GB。
- 驱动需要支持 CUDA 13，官方最低驱动主版本为 **580**；这只是最低条件，JIT/PTX/算子仍需真实运行验证。镜像名称不能证明宿主机驱动兼容。[NVIDIA 兼容说明](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
- 原有 Python 环境所在磁盘需容纳 CUDA/PyTorch 包替换时的峰值空间，系统盘建议扩到 60GB 或按实际使用量增加。安装器检查 `sys.prefix` 所在磁盘至少有 20GiB 空闲；依赖和临时解包可能需要更多空间，也需核对 `/tmp`。模型与数据盘建议预留至少 60GB，100GB 更方便。
- 模型、测试包和结果放 `/root/autodl-tmp`；环境继续使用镜像原有路径，不另建 `/root/socialnav-vlm-env`。用 `python -c 'import sys; print(sys.executable); print(sys.prefix)'` 确认安装目标；数据盘上的环境不会包含在系统镜像中。
- 系统盘可保存为私有镜像，数据盘需要单独保留/迁移。关机不等于永久备份，实例释放或磁盘故障仍可能丢失文件。[AutoDL 镜像](https://www.autodl.com/docs/image/)、[磁盘说明](https://api.autodl.com/docs/env/)

上传并解压 `autodl_sglang_bundle.zip`，进入包目录。例如：

```bash
cd /root/autodl-tmp
unzip autodl_sglang_bundle.zip
cd autodl_sglang_bundle

nvidia-smi
python --version
df -h / /root/autodl-tmp /dev/shm
```

ZIP 不含模型、账号信息、API Key、完整 JRDB 或旧实验结果。包含 24 个开发窗口、72 张既有渲染图、固定 Prompt、动态 Schema，以及与仓库相同的测速实现。图片含真实行人，仅按数据集许可用于自己的研究测试，不当作可任意公开再分发的素材。

## 2. 查看并执行原环境安装

在 AutoDL 基础镜像的 Python 3.12 中执行；以下 `python` 均指同一个原有解释器。如果终端激活了其他环境，先切回期望修改的环境。若镜像设置了 `PYTHONPATH`/`PYTHONHOME`，先在这个终端执行 `unset PYTHONPATH PYTHONHOME`，避免安装目标与实际导入路径不一致，不修改全局 shell 配置。

第一次**只显示计划**，列出目标解释器、环境路径、版本替换警告和 pip 命令，不联网、不创建目录：

```bash
python deploy.py install
```

确认已保存容器镜像、目标解释器正确且可以替换现有包后执行。两项显式确认不可省略：

```bash
python deploy.py install --execute --allow-env-change --out results/install
```

行为边界：

- 用运行脚本的 `sys.executable -m pip` 修改**当前 Python 环境**；不运行 `venv`、不切换解释器，也不使用 `--user` 把包装进另一个位置。
- 从 PyPI、PyTorch 官方 cu130 及 NVIDIA 官方索引升级到固定的关键版本，其他依赖仅在需要时调整；仅接受 wheel，不在安装阶段自动编译第三方源码。正常服务启动时仍可能 JIT 编译兼容算子。
- 不修改 Python 本身、驱动、系统 CUDA 或 `.bashrc`，不使用 `--break-system-packages` 绕过系统保护。
- 修改前写入 `installed_packages_before.json` 和 `installation_started.json`；完成安装和 `pip check` 后写入 `installed_packages_after.json`、`installation.json`。只保存包名/版本，不记录凭证或认证 URL。
- 不自动下载权重、启动服务、发测试请求、改用其他模型或追加安装重试。
- 记录目录必须是新的。失败时保存 `installation_failed.json`；当前环境可能已被部分修改，不自动修复/卸载/回滚。保留记录并诊断，必要时从安装前镜像重建容器，而不是在半修改环境中反复盲装。
- `--min-free-gib` 可提高空间门槛，不建议调低来绕过不足。安装日志留在当前终端；不把代理配置、认证 URL 或全部环境变量写入产物。

下载需要代理时，只设置在该远程机器上确实可用、可信的 `http_proxy`/`https_proxy`。不要复制原服务器的公司内网代理，也不要写入 Git/ZIP。安装器忽略 pip 私有配置，使用明确的官方索引；HTTP 测速客户端则默认忽略环境代理。

完成后，后续服务命令继续使用同一个原有解释器。关闭或重启安装前已运行的 Python/Jupyter 进程，不在旧进程中继续使用已加载的 Torch/native 模块：

```bash
python -m pip check
python deploy.py inspect --out results/environment
```

`inspect` 默认只读取包元数据，不初始化 GPU；写入 `preflight.json` 和无认证 URL 的包名/版本清单。所有结果目录都必须是**新目录**。

### 2.1 推荐：预下载资源，正式容器仅离线安装

下载不需要 GPU，也不会安装依赖；可以在现有 Linux x86_64/Python 3.12 机器上完成，再通过网页上传到 AutoDL 文件存储。文件存储不需要 GPU 实例开机，但存储本身不是执行下载命令的计算环境；正式实例选择同一地区，挂载路径为 `/root/autodl-fs`。[AutoDL 文件存储](https://www.autodl.com/docs/fs/)、[不开实例上传](https://www.autodl.com/docs/scp/)。

准备阶段的命令如下，`--out`必须是新目录；若已经拿到本项目准备的`autodl_wheelhouse.zip`，直接跳过下载：

```bash
# 只打印下载计划，不联网、不修改环境
python deploy.py download --out /path/to/autodl_wheelhouse --archive

# 仅下载官方 wheel 和间接依赖，生成完整锁文件与独立 ZIP；不安装
python deploy.py download --out /path/to/autodl_wheelhouse --archive --execute
```

下载完成后生成`requirements.lock`与`wheelhouse_manifest.json`，锁定全部解析包名、版本和 wheel SHA-256，并执行一次`pip install --dry-run --ignore-installed --no-index --require-hashes`验证依赖闭包；这不是实际安装、原生模块导入或 GPU 验证。wheel 的平台标签保存在清单中，远程 Linux 的 glibc/ABI 也必须兼容；本轮 SGLang wheel 要求 glibc 至少 2.34，Ubuntu 22.04/24.04 的常规基础环境满足这一项，但仍需核对实际镜像。下载失败不会生成完成清单，不把零散文件视为完整资源。

两个 ZIP 分开保存：

- `autodl_sglang_bundle.zip`：脚本、指南与24例/72图；不含第三方 wheel、模型或已有凭证。
- `autodl_wheelhouse.zip`：wheel、全部依赖锁文件、完成清单；不含模型、代理配置或凭证。wheel本身已压缩，因此ZIP使用存储模式并支持ZIP64，大包上传端也须支持其实际大小。

将两个 ZIP 上传到文件存储，正式实例启动后解压。下面假设文件位于`/root/autodl-fs/socialnav_assets/`，且目标解压目录尚未存在：

```bash
cd /root/autodl-tmp
unzip /root/autodl-fs/socialnav_assets/autodl_sglang_bundle.zip
unzip /root/autodl-fs/socialnav_assets/autodl_wheelhouse.zip
cd autodl_sglang_bundle

# 使用该远程容器原有 Python；这里只打印计划
python deploy.py install --wheelhouse ../autodl_wheelhouse

# 已保存镜像、确认允许替换原有包后执行；绝不退回在线索引
python deploy.py install --wheelhouse ../autodl_wheelhouse \
  --execute --allow-env-change --out results/install_offline
python deploy.py inspect --out results/environment_offline
```

可直接把`--wheelhouse`指向已解压的文件存储目录，避免再复制一份依赖；模型建议仍放本地数据盘以加快加载。安装执行前完整核对 wheel、锁文件与安装 profile 的绑定；安装使用`--no-index --find-links --require-hashes --only-binary=:all:`，同时忽略 pip 配置和版本检查，不因缺包而联网、编译源码或更改版本。完整锁文件会固定间接依赖，原环境其他包仍可能冲突，`pip check`失败必须诊断，不自动删包。安装确认、前后记录与失败处理规则与在线方式相同。

离线安装只消除 Python 包下载；模型权重另行上传。服务首次加载/JIT还可能需要工具链或算子缓存，wheel依赖闭包通过不代表所有运行期资源已经就绪。不要把网络依赖诊断或GPU启动成绩写成下载阶段已验证。

## 3. 上传并核对模型

将当前机器的 `/share/public_models/qwen/Qwen3.8-27B-FP8` 整个目录上传到：

```text
/root/autodl-tmp/models/Qwen3.8-27B-FP8
```

也可使用单独导出的 `outputs/t1/Qwen3.8-27B-FP8.zip`，约 **28.8GiB**，支持 ZIP64；它不在脚本包或 wheel 包中。把模型 ZIP 和同名 `.sha256` 校验文件上传到文件存储，例如 `/root/autodl-fs/socialnav_assets/`，正式实例启动后校验并解压到本地数据盘。下面假设目标模型目录尚未存在：

```bash
cd /root/autodl-fs/socialnav_assets
sha256sum -c Qwen3.8-27B-FP8.sha256
# 只有上一步显示 OK 才继续
mkdir -p /root/autodl-tmp/models
unzip -n Qwen3.8-27B-FP8.zip -d /root/autodl-tmp/models
```

ZIP 内只有当前模型目录的公开文件，保留 66 个权重分片及原有配置/模板，不重新量化，不带下载缓存或凭证。解压后路径就是上述模型路径。旁边的 `.verification.json` 记录逐文件 SHA-256、权重索引/文件结构核验、原始 CRC 校验差异和传输包摘要；这验证复制完整性，不代表官方模型身份、运行兼容或 GPU 推理已经验证。当前源目录的 `chat_template.jinja`、`generation_config.json`、`tokenizer_config.json` 与其旧 `crc32.txt` 不一致，原文件与旧校验表均如实保留，传输核对以本次 `.sha256` 和清单为准。

若 ZIP 也放本地数据盘，ZIP 加解压目录约占 **57.6GiB**，还要预留 wheel、环境和结果空间；建议至少 100GB 数据盘，或把 ZIP 留在文件存储、仅将模型解压到本地。不要覆盖已有模型目录或根据旧校验表修改配置。

也可以指定别的实际目录，但不要自行猜测 Hugging Face 仓库 ID 或用同名模型替换。必须带齐 `config.json`、权重索引、所有分片、Tokenizer、聊天模板和视觉处理器配置；本机记录约 31GB/66 分片仅供核对，不硬编码为所有模型的要求。目录叫“3.8”不代表配置架构类必须包含“3.8”。

执行小型 GPU/资源检查，不加载模型、不发送 VLM 请求：

```bash
python deploy.py inspect \
  --gpu \
  --model-path /root/autodl-tmp/models/Qwen3.8-27B-FP8 \
  --out results/gpu_preflight
```

核对依赖版本、CUDA 13、驱动、显存、原生模块导入；只做一个极小 CUDA 矩阵计算。模型检查读取索引和小配置、确认分片可达与非空、现有 FP8 和视觉配置，并记录小文件 SHA-256。**不会读取并哈希全部 31GB 权重，也不证明权重身份或模型加载成功。**完整性严谨比对可在传输端另做全分片摘要。

如果原应用仍在运行，先用 `nvidia-smi` 查清所属进程，再停止你自己的旧服务。脚本不会杀其他进程腾显存，也不会因缺卡自动降级到 CPU。

## 4. 启动基线服务

先查看命令，零服务启动：

```bash
python deploy.py serve \
  --model-path /root/autodl-tmp/models/Qwen3.8-27B-FP8
```

确认后，在一个独立终端或已有的 `tmux` 会话执行：

```bash
python deploy.py serve \
  --model-path /root/autodl-tmp/models/Qwen3.8-27B-FP8 \
  --out results/server_baseline --execute
```

基线：`127.0.0.1:30000`，模型别名 `qwen3.8-27b-fp8`，一个运行请求，上下文 12288，静态显存比例 0.85，非量化模块 BF16，已有 FP8 权重，XGrammar。BF16 参数**不表示把 FP8 权重变成 BF16**；不进行在线重新量化。

不强制 FA3/DeepGEMM/Hopper 专用内核，不默认开启 MTP，不禁用默认的缓存或 CUDA Graph。具体支持以安装版本的 `python -m sglang.launch_server --help` 为准。[SGLang 服务参数](https://docs.sglang.io/docs/advanced_features/server_arguments)

终端会显示自己启动的 PID 和 `server.log` 路径；在另一终端查看日志：

```bash
tail -f results/server_baseline/server.log
```

首次加载、图捕获或 JIT 可能耗时较长，不计入单请求耗时。脚本前台管理服务，Ctrl+C 只停止自己创建的进程组，无自动重启或失败回退。日志同时记录 `preflight.json`、`launch.json` 和退出时的 `exit.json`；不要只因进程存在就认为服务 ready。

## 5. 服务器内测试：先零推理，再三例

新开终端，回到同一个包目录并确认使用与安装/服务相同的 Python 环境。包内 `server_config.json` 预填本地地址和模型别名，API Key 为空；基线只通过本地地址或 SSH 隧道使用，不向公网裸露服务。

```bash
# 验证 24 例资产、Prompt、Schema 和代码 SHA；零网络、零凭证读取
python benchmark.py --dry-run --limit 24

# 只 GET 模型列表及可选服务元数据；零推理
python benchmark.py \
  --check --server-info --out results/check

# 三个场景各一例；仅 3 个图片推理请求
python benchmark.py --out results/smoke
```

要求三例均合法、正常结束、没有隐藏思考，且模型别名匹配。JSON Schema 请求之后还有本地结构/成员/角色校验，不自动修格式、补身份、切换服务或重试；第一个失败任务即停止。

确认正常且确需完整测速时，**只运行一轮**：

```bash
python benchmark.py \
  --limit 24 --server-info --out results/serial
```

随后只有要验证每两秒到达的调度时才运行：

```bash
python benchmark.py \
  --mode paced --rate-hz 0.5 --deadline-s 2 \
  --limit 24 --out results/paced_2s
```

忙时丢弃到达样本，不堆积队列。3 例 smoke、24 例 serial 和 24 例 paced 是可分别选择的阶段，并非要求每次全部运行。默认不发送额外 warm-up，避免把重复窗口预热当作真实新图加速。

## 6. 自己电脑上的远程调用

将同一个 ZIP 解压到电脑。可以在独立客户端环境安装包内 `requirements.txt`；只需 `httpx` 和 `jsonschema`，**不要在电脑安装 `requirements.server.txt`**。原有合适的环境也可以直接使用。

例如在电脑建立客户端虚拟环境（Python 3.10 或更新；Windows 可用对应 `Scripts/python.exe`）：

```bash
python -m venv .client-env
.client-env/bin/python -m pip install -r requirements.txt
```

将 AutoDL 控制台的 SSH 地址、端口代入，保持隧道终端运行：

```bash
ssh -p SSH_PORT -N -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:30000:127.0.0.1:30000 root@SSH_HOST
```

这里 `SSH_PORT`、`SSH_HOST` 是待替换的示意文本，不是环境变量。电脑侧继续使用包内 `http://127.0.0.1:30000/v1`。如电脑 30000 已占用，可把隧道左端口换成 30001，同时只改电脑的 `server_config.json`。[AutoDL 隧道说明](https://api.autodl.com/docs/ssh_proxy/)

在电脑的客户端环境执行相同的 dry-run/check，然后按需跑一轮串行：

```bash
python benchmark.py --dry-run --limit 24
python benchmark.py --check --out results/remote_check
python benchmark.py --limit 24 --out results/remote_serial
```

上述 `python` 指已激活的客户端解释器。使用相同图片/Prompt/Schema/参数，尽量相近时段测本机与远程。基线绑定本地地址，没有 API Key，SSH 提供加密和访问控制；如果改成公网服务，必须另行设计认证和 HTTPS，不能只把监听地址改为 `0.0.0.0`。

## 7. 解读与比较结果

每次测速输出 `model_check.json`、`settings.json`、`requests.jsonl`、`summary.json`。完整耗时包括客户端图片读取/编码、HTTP、流结束和校验，不含实时检测/跟踪/图像渲染、服务启动或模型加载。TTFT 从 HTTP 请求开始到第一段最终文本，**不是可执行决策返回时间**。

只读比较同机与远程结果（先将两份结果放在一台机器），不发送请求：

```bash
python compare_results.py \
  --reference results/serial --candidate results/remote_serial
```

也支持历史 `results1.zip`、`results2.zip`，只读取所选阶段的两个日志，不解压或读取连接配置：

```bash
python compare_results.py \
  --reference results2.zip --candidate results1.zip --phase serial
```

比较器要求配对合法样本具有一致的输入/请求指纹，拒绝仅凭相同 case 名作对照。不同输入或参数不能强行比较。差值包含客户端、网络、转发、服务/缓存和运行波动，**不是精确的 GPU 时间或纯网络 RTT**；不要把历史 H800 与新 4090 的差值解释为纯部署优化收益。

这 24 例只是开发输入，使用 oracle 几何/PID，没有本任务完整人工金标准。格式合法不等于群组语义准确；串行 2 秒内完成也不等于 0.5Hz 实机闭环验收。

## 8. 优化与常见问题

先保存可用基线，再一次只改变一项，以新的服务/结果目录记录：

- 显存不足：确认旧服务是否占用；可显式降低 `serve --context-length 8192` 或 `--mem-fraction-static 0.80`，不自动改图像/Schema。降低显存比例会减少缓存容量，并非必然解决所有预填充峰值。
- 缓存差异：可显式 `serve --disable-radix-cache` 作对照；返回 cached_tokens=0 不能证明真实后端缓存关闭。重复窗口测速也不代表新图在线延迟。
- 图捕获失败：可显式 `serve --disable-cuda-graph` 定位；不把诊断关闭项静默当作新的默认优化。
- 原生模块导入失败、无对应 wheel、驱动或 PTX 错误：保留失败类型/远程日志，先核对版本与GPU，不在指定依赖替换之外盲目追加升级，也不自动换回 CUDA 12。仅凭 import 通过仍不能证明所有算子路径可用。
- `cuda-tile==1.6.0rc5`找不到：先打印安装计划，确认其中有`--extra-index-url https://pypi.nvidia.com`，并检查该远程机器可访问 NVIDIA 索引。当前包已修正旧包的索引遗漏，不应改版本、删依赖或用`--no-deps`绕过。
- `pip check` 失败：原镜像的额外包也可能要求旧版 Torch/Transformers。先分析具体冲突和该容器是否需要这些应用，不自动卸载未知依赖；必要时恢复安装前镜像再调整方案。修改了 Python 环境，不等于要升级宿主机驱动或重装系统 CUDA。
- 缺 CUDA Toolkit/JIT 编译器：检查 `nvcc --version` 与镜像实际 CUDA 路径，必要时换包含所需工具的基础镜像；不要修改宿主机驱动或把 CUDA_HOME 指向不存在的目录。
- Schema/MTP：先验证多模态＋Schema 基线。MTP、FP8 KV、视觉图捕获及其他后端需要该 GPU/版本单独验证，本包不默认启用、不追加反复调用来筛选。
- `deploy.py inspect` 包版本不一致会返回非零；不是自动修复器。若你主动修改版本，应先更新部署实现和来源约束、重新生成包，而不是改 SHA 绕过检查。
- 文件 SHA 不一致：重新解压完整包；不要混用不同 ZIP 的 benchmark、common、defaults 或图片。可编辑的连接配置不纳入冻结资产摘要。

## 9. 保存环境与回传结果

运行正常后停止自己的服务，在 AutoDL 控制台关机并保存系统镜像。不要将 API Token、SSH 私钥或含凭证的下载配置保存在要分享的镜像中。模型和测试包所在数据盘单独保留；新实例需要重新确认驱动/GPU与数据路径，保存镜像并不自动恢复运行中的服务。

分析时提供新的 serial/remote_serial/paced 结果目录、环境 `preflight.json`、包版本列表和 `launch.json` 即可。不要回传填写过的 `server_config.json`、SSH 凭证、代理信息或整个虚拟环境；分享 server.log 前自行审查。既有输入与结果不覆盖。

仓库内重新打包入口：`python -m decision.vlm.package_vlm_test --with-deployment --out outputs/t1/autodl_sglang_bundle`。输出目录/ZIP 已存在时拒绝覆盖，重新构建要选新的语义化名字；工具源码位于decision/vlm/deployment，实际大图资产只在便携包内白名单复制。

新生成部署包中的`README.md`为本指南，`benchmark_guide.md`为通用客户端测速说明；普通测试包则直接使用`README.md`作为测速说明。按用户要求，本项目仅保留当前 AutoDL 完整包，不再保留旧的隔离安装包或增量更新包；云端文件不会自动更新，既有实验结果不清理。

## 10. 已上传旧包时如何更新

上传当前完整的`autodl_sglang_bundle.zip`，先解压到新的临时目录；不要直接覆盖正在使用的旧包目录。只将下列六个公共文件复制到旧目录：`deploy.py`、`installation_profile.json`、`requirements.server.txt`、`README.md`、`manifest.json`、`benchmark_guide.md`。`manifest.json`必须与这些文件一起更新，不能只替换脚本。

**不替换** `server_config.json`、`cases/`、`system_prompt.txt`、`results/`及模型目录；完整包中的连接配置为空，不能覆盖已经填写的配置。旧`README.txt`不再使用，留在云端旧目录中不影响验证。

例如旧包位于`/root/autodl-tmp/autodl_sglang_bundle`，将当前ZIP上传到其父目录后执行。`autodl_install_refresh`必须是尚未存在的新目录：

```bash
cd /root/autodl-tmp
unzip autodl_sglang_bundle.zip -d autodl_install_refresh
cp autodl_install_refresh/autodl_sglang_bundle/{deploy.py,installation_profile.json,requirements.server.txt,README.md,manifest.json,benchmark_guide.md} autodl_sglang_bundle/
cd autodl_sglang_bundle

# 使用准备修改的原有 Python，先检查目标和资产；不会安装或调用模型
python -c 'import sys; print(sys.executable); print(sys.prefix)'
python benchmark.py --dry-run --limit 24
python deploy.py install
```

确认解释器正确、已保存容器镜像后，再执行第2节的安装命令。只替换文件不会安装依赖、迁移或删除旧虚拟环境；若此前激活了旧服务虚拟环境，应先退出并选择镜像原有 Python，不能仍使用 `/root/socialnav-vlm-env/bin/python`执行新的安装步骤。

这次六文件替换适用于此前提供的原包，并加入了`download`与`install --wheelhouse`入口：测速客户端、24窗口/72图、Prompt和Schema均未变化。如云端改动了公共资产，dry-run会拒绝，不应修改摘要绕过。要使用离线安装，还需上传并解压完整的`autodl_wheelhouse.zip`。安装仍使用新的记录目录，原`results/install`已经存在时不要覆盖。已有结果只保留，不自动重测。
