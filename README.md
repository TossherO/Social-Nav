# Social-Nav

感知 / 决策 / 导航系统骨架。更新：2026-10-07。

当前已纳入检测训练、共享检测跟踪推理核心、ROS 1接入源码和VLM公开数据验证流程，完整在线导航系统尚未实现。VLM主线为冻结 **Qwen3.8-27B-FP8**、远程手动SGLang部署、JSON Schema输出、每2秒调用一次（0.5Hz），不微调VLM。

本仓库作为正式初始实现，文件名、项目配置、协议名和新生成产物不附加项目版本号。模型/依赖名称、标准API路径、数据来源提交及运行时状态编号保留；已有冻结样例、用户结果和历史归档不改写。工具ZIP按用户要求更新或清理，不覆盖已填写的连接配置和用户结果。

## 项目结构

| 目录 | 职责与当前状态 |
| --- | --- |
| `perception/` | `detection_tracking/`统一检测训练与推理；离线核心已验证，ROS 1节点源码待真实环境验证。定位算法和建图仍未实现，原始里程计不作为可靠位姿。 |
| `decision/vlm/` | 现有VLM流程的真实实现及公开配置，包括JRDB渲染、审核、输出校验和远程测速。 |
| `decision/agent/` | 未来长程任务分解、模式切换、恢复与技能编排的预留位置。 |
| `navigation/` | 路径规划、运动控制和安全模块，目前均为未实现占位。 |
| `interfaces/` | 感知结果已有类型转换，其余跨域接入仍是草案，不能据此运行整机。 |

旧的 `python -m scripts.<VLM入口>` 仍由薄包装转发；新代码使用 `decision.vlm` 导入路径。

## 文档入口

建议先阅读[项目框架](docs/项目框架.md)。`docs/`包含四份当前文档和一份明确标识的旧版方案：

- [当前整体方案](docs/当前整体方案.md)：研究目标、三任务规约、系统设计与安全边界。
- [纯公开数据可行性验证方案](docs/纯公开数据可行性验证方案.md)：不依赖实机采集的验证流程。
- [JRDB数据与测试实现](docs/JRDB数据与测试实现.md)：数据和标注格式、当前实现及测试证据。
- [项目框架](docs/项目框架.md)：目录职责、配置、命令、产物与复现方法。
- [旧版：基于VLM的长程社交导航](docs/旧版_基于VLM的长程社交导航.md)：历史技术方案，不作为当前实施要求，历史命令不保证可执行。

模块入口：[感知](perception/README.md) · [决策](decision/README.md) · [导航](navigation/README.md) · [共享接口](interfaces/README.md)。项目说明统一使用Markdown；配置、Prompt、依赖清单等纯文本文件保持其专用名称。

检测训练体系的原详细文档保留于[检测源码文档目录](perception/detection_tracking/training/simpledet3d/docs)。
检测跟踪的配置、离线推理、缓存回放、ROS 1与定位边界见[模块说明](perception/detection_tracking/README.md)。
AutoDL原环境依赖替换、手动SGLang服务及同机/远程测试见[部署指南](decision/vlm/deployment/README.md)。安装显式确认后将远程容器原有Torch 2.12替换为2.13+cu130，不创建服务虚拟环境；不修改本机开发环境。可用`python -m decision.vlm.package_vlm_test --with-deployment --out outputs/t1/autodl_sglang_bundle`生成带现有24例输入的独立ZIP；打包不安装环境或调用模型。部署工具支持无GPU预下载完整wheel依赖及哈希锁文件，正式容器使用`install --wheelhouse`离线安装；wheel资源另存`outputs/t1/autodl_wheelhouse.zip`，不塞入轻量测试包。

## 配置与数据

- 系统级配置：[configs/system.json](configs/system.json)。
- VLM配置：[decision/vlm/configs/project.json](decision/vlm/configs/project.json)。
- 数据默认统一通过 `data/datasets/` 下的软链接访问；检测副本的 `data` 也指向该入口。仅调整链接目标即可重定位，不移动原始数据；SUB审计和冻结测速均有默认别名。具体映射见[项目框架](docs/项目框架.md#201-统一数据软链接)。
- 测速默认值读取公共 `inference` / `benchmark` 配置块，CLI参数优先覆盖。新便携包导出经过哈希校验的默认值快照，已有ZIP不自动更新；`--deadline-s` 与HTTP超时独立。
- 服务配置示例为[vlm_server.example.json](decision/vlm/configs/vlm_server.example.json)，使用时复制到 `configs/vlm_server.local.json` 并在本机填写。`configs/*.local.json` 为私有配置，不提交或分享。

## 按需检查

以下命令在仓库根目录、具备对应依赖的Python环境中运行；不要求每次修改全部执行。

系统结构和数据链接检查不启动进程：

```bash
python -m scripts.system_status
python -m scripts.dataset_links
```

VLM输入与配置变更时可执行离线检查，不发送模型请求：

```bash
python -m decision.vlm.benchmark_vlm --dry-run --limit 24
```

离线回归测试不调用模型、API或GPU。局部修改先运行相关测试，代码修改完成后再集中执行完整回归；仅修改文档时检查相关链接和示例即可。

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests
```

检测历史源码审计不读取数据或权重，也不是训练可运行性检查：

```bash
python -m perception.detection_tracking.training --check
```

只读核对三帧数据集点云及自带位姿（不加载模型）：

```bash
python -m perception.detection_tracking.inference --scene outdoor1/start_003 --limit 3 --dry-run
```

真实检测使用已配置的检测环境及GPU；机器权重/二进制位置保存在忽略的`configs/detection_tracking.local.json`，不读取API配置。不默认使用原始`/odom`；在线需另外接入定位算法输出。

检测副本已做数据路径与无编号命名适配，上述历史审计会报告预期差异并返回非零退出码。原仓库和清单中的原始文件摘要不变；不要将其作为日常必须通过的门槛。

## 历史记录与边界

数据、原始JRDB、模型、Conda环境和既有便携ZIP保留原位。历史备份与验证记录位于 `outputs/archive/`：

- `cleanup_20261005/`：早期清理归档。
- `restructure_20261007/`：系统结构重构。
- `maintenance_20261007/`：完整性维护。
- `dataset_links_20261007/`：统一数据链接适配。

早期重构与路径适配未执行模型调用。本轮检测跟踪实现另外执行了少量GPU离线验证，证据见模块说明；当前有ROS 1节点源码，但本机没有ROS运行环境，未验证catkin构建或真实消息接入，仍无底盘运行程序。
