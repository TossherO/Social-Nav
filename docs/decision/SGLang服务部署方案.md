# SGLang 服务部署方案：原生 Conda、源码构建与 DSPARK

本文用于在自有 Ubuntu 20.04、glibc 2.31 服务器上，从零部署局部 VLM 服务。方案使用独立 Python 3.12 Conda 环境、CUDA 12.9 工具链、官方 cu129 PyTorch，以及固定的 SGLang 0.5.19 源码。推理时用两张 GPU 做张量并行，加载 Qwen FP8 目标模型和公开 DSPARK 草稿模型。

源码构建用于适应当前系统的 glibc，系统驱动和 glibc 保持原值。既有实现已用项目冻结三图输入完成服务器本机调用、开发机经 SSH 调用和短时 0.5 Hz 测试。实际机器、结果和服务状态由 [ENVIRONMENT.local.md](../../ENVIRONMENT.local.md)定位，不能据这些测试认定满上下文或更高并发已经验收。

这项部署只服务局部 T1、T2、T3，外层 Agent 使用独立托管 API。已有服务器不必重装，启停和访问直接看[VLM 部署与调用](技术实现-VLM局部决策.md)。本文只维护从零安装和最终构建材料。

服务参数由部署脚本维护，依赖版本以 `config/constraints-cu129.txt`为准。该文件约束关键版本，并不是所有传递依赖的永久锁文件。

## 1. 准备目录与部署材料

下面的安装命令都在目标服务器的 Bash 终端执行。先确认机器为 Linux x86_64，检查 glibc、现有驱动、所选两张 GPU 的空闲资源，以及部署盘和临时目录的空间：

```bash
getconf GNU_LIBC_VERSION
nvidia-smi
df -h "$HOME" /tmp
```

当前方案沿用已经验证的 CUDA 12 驱动路线。新设备仍需按真实驱动核对 [CUDA 兼容规则](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)，不能根据镜像名称判断兼容性。

缺少基础工具时才安装以下系统包，不需要为本方案重装系统 CUDA 或驱动：

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  curl ca-certificates git build-essential libnuma1 libibverbs1 ffmpeg tmux
```

将仓库的 `decision/vlm/deployment/`完整复制到服务器，或使用服务器已有仓库。`SGLANG_MATERIALS`指向部署材料目录，`SGLANG_ROOT`是将要创建的服务目录；服务目录必须尚不存在：

```bash
export SGLANG_MATERIALS=/path/to/Social-Nav/decision/vlm/deployment
export SGLANG_ROOT="$HOME/sglang-service"
mkdir "$SGLANG_ROOT"
mkdir -p "$SGLANG_ROOT/build" "$SGLANG_ROOT/config" \
  "$SGLANG_ROOT/models" "$SGLANG_ROOT/logs" "$SGLANG_ROOT/cache"
cp "$SGLANG_MATERIALS/start_server.sh" "$SGLANG_MATERIALS/runtime_env.sh" "$SGLANG_ROOT/"
cp "$SGLANG_MATERIALS/build/build_sglang.sh" \
  "$SGLANG_MATERIALS/build/source-cu129.patch" "$SGLANG_ROOT/build/"
cp "$SGLANG_MATERIALS/config/constraints-cu129.txt" "$SGLANG_ROOT/config/"
```

完成后，服务目录中应有启动和环境脚本、构建材料、版本约束文件，以及模型、日志和缓存目录。模型、环境和源码各保留一个正式位置，已有大资产直接复用。

服务根目录只保留两份本机说明，分别介绍技术实现和环境测试记录。正式测试结果集中在开发仓库，服务器保留复现所需的冻结输入和客户端即可。

## 2. 新建独立Conda环境

已有 Miniconda 管理器时直接使用；仅在没有管理器时安装。下例把管理器放在服务目录之外，安装器用完删除，也不修改全局 Shell 配置：

```bash
export SGLANG_CONDA_ROOT="$HOME/miniconda3"
# 仅当上面的管理器不存在时执行以下三行。
curl -fL https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh \
  -o /tmp/Miniconda3-sglang.sh
bash /tmp/Miniconda3-sglang.sh -b -p "$SGLANG_CONDA_ROOT"
rm /tmp/Miniconda3-sglang.sh
```

接着创建专用环境。这个环境不替换项目开发环境或系统 ROS 环境：

```bash
source "$SGLANG_CONDA_ROOT/etc/profile.d/conda.sh"
export SGLANG_CONDA_ENV="$SGLANG_CONDA_ROOT/envs/sglang-cu129"
conda create -y -p "$SGLANG_CONDA_ENV" --override-channels \
  -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge \
  python=3.12 pip gcc_linux-64=13 gxx_linux-64=13 sysroot_linux-64=2.28 \
  'cmake>=3.26' ninja pkg-config 'libprotobuf>=5,<7'
conda activate "$SGLANG_CONDA_ENV"
conda install -y --override-channels \
  -c nvidia/label/cuda-12.9.2 \
  -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge \
  cuda-toolkit=12.9.2
source "$SGLANG_ROOT/runtime_env.sh"

export PIP_CONFIG_FILE=/dev/null
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
unset PIP_EXTRA_INDEX_URL
python -m pip install --upgrade pip setuptools wheel packaging build \
  'setuptools-rust>=1.11' 'setuptools-scm>=8'
```

`runtime_env.sh`设置环境内编译器、ptxas、运行库和缓存路径，并隔离其他 Python 路径。Conda sysroot 只用于编译，不替换系统 glibc；CUDA stubs 也不加入运行库搜索路径。

如果管理器或环境不在默认位置，每次调用构建或启动脚本前，都要设置 `SGLANG_CONDA_ROOT`和 `SGLANG_CONDA_ENV`，不要把某台机器的路径写入源码。

## 3. 安装cu129依赖

下面所有运行依赖使用同一约束文件。PyTorch 从官方 cu129 源安装，普通 Python 包优先清华镜像，专属 CUDA wheel 使用发行方源：

```bash
export SGLANG_CONSTRAINTS="$SGLANG_ROOT/config/constraints-cu129.txt"
python -m pip install --only-binary=:all: \
  --index-url https://download.pytorch.org/whl/cu129 \
  -c "$SGLANG_CONSTRAINTS" \
  'torch==2.13.0+cu129' 'torchvision==0.28.0+cu129' 'torchaudio==2.11.0+cu129'

python -m pip install --only-binary=:all: \
  --index-url https://pypi.nvidia.com \
  --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  -c "$SGLANG_CONSTRAINTS" 'cuda-tile==1.6.0rc5'

python -m pip install --only-binary=:all: \
  --extra-index-url https://docs.sglang.ai/whl/cu129/ \
  -c "$SGLANG_CONSTRAINTS" \
  'https://github.com/sgl-project/whl/releases/download/v0.4.6.post1/sglang_kernel-0.4.6.post1+cu129-cp310-abi3-manylinux2014_x86_64.whl' \
  'https://github.com/sgl-project/whl/releases/download/v0.1.7/sgl_deep_gemm-0.1.7+cu129-py3-none-manylinux2014_x86_64.whl' \
  'sgl-deep-ep==0.1.2+cu129'
```

其中内核的 `cp310-abi3`wheel 可以用于 Python 3.12，不要求切换到 Python 3.10。安装了某个原生包，不代表启动时一定使用它的专用算子；实际后端仍由最终启动脚本决定。本方案不启用 CUDA 13 或 Hopper 专用路径。

## 4. 固定源码、适配依赖并构建

SGLang 0.5.19 的官方 Python 3.12 Linux wheel 要求 glibc 至少为 2.34，而目标系统是 2.31。因此，本方案从固定源码构建本机 wheel，不修改已有 wheel 的平台标签。[发行文件](https://pypi.org/project/sglang/0.5.19/#files)保留了对应来源。

获取完整仓库，以保留 Rust workspace 和 Cargo 锁文件：

```bash
export SGLANG_SRC="$SGLANG_ROOT/build/sglang"
git clone --depth 1 --branch v0.5.19 \
  https://github.com/sgl-project/sglang.git "$SGLANG_SRC"
test "$(git -C "$SGLANG_SRC" rev-parse HEAD)" = 0bcd822377da7b5718e674eaf9c870d349424dd1
git -C "$SGLANG_SRC" apply "$SGLANG_ROOT/build/source-cu129.patch"
```

补丁只调整四项依赖选择：把 cuda-python 限制在 CUDA 12 范围，FlashInfer 和 Humming 改用 cu12 选项，CUTLASS 去掉 cu13 额外选项。算法源码不变，源码声明和约束文件共同防止依赖解析回 CUDA 13。

已经应用补丁的源码不要重复执行上述步骤，也不要给他人的源码副本打补丁。随后使用独立 Rust 工具链和 RsProxy 依赖镜像构建：

```bash
export CARGO_HOME="$SGLANG_ROOT/build/cargo"
export RUSTUP_HOME="$SGLANG_ROOT/build/rustup"
export RUSTUP_DIST_SERVER=https://rsproxy.cn
export RUSTUP_UPDATE_ROOT=https://rsproxy.cn/rustup
curl -fL https://sh.rustup.rs -o /tmp/rustup-sglang.sh
sh /tmp/rustup-sglang.sh -y --no-modify-path \
  --profile minimal --default-toolchain 1.92.0
rm /tmp/rustup-sglang.sh
cat > "$CARGO_HOME/config.toml" <<'TOML'
[source.crates-io]
replace-with = "rsproxy-sparse"
[source.rsproxy-sparse]
registry = "sparse+https://rsproxy.cn/index/"
TOML

bash "$SGLANG_ROOT/build/build_sglang.sh"
python -m pip install --only-binary=:all: \
  --extra-index-url https://docs.sglang.ai/whl/cu129/ \
  --extra-index-url https://pypi.nvidia.com \
  -c "$SGLANG_CONSTRAINTS" \
  "$SGLANG_ROOT/build/wheelhouse/sglang-0.5.19-cp312-cp312-linux_x86_64.whl"
python -m pip check
```

构建脚本使用当前 cu129 Torch、Conda 编译器和 protoc，关闭 Python 构建隔离，并构建全部四个 Rust 扩展。wheel 写入 `build/wheelhouse/`，构建日志写入 `logs/build.log`。脚本只构建，不安装或启动服务；后面的 pip 命令负责安装和检查依赖。

构建失败时，先查看日志中的首个实际错误。必要时降低 `CARGO_BUILD_JOBS`并复用缓存，不跳过扩展、不另装 CPU LibTorch，也不因一个构建错误重建整套环境。

## 5. 准备固定目标与草稿模型

已有可信模型时直接复用，分别用 `SGLANG_MODEL_DIR`和 `SGLANG_DRAFT_MODEL_DIR`指定路径。新机器可以通过 hf-mirror 下载下面固定 revision 的公开资产。下载列表不包含模型仓库 Python 代码，也不执行远程模型代码：

```bash
export HF_ENDPOINT=https://hf-mirror.com
python - <<'PY'
import json
import os
from pathlib import Path
from huggingface_hub import snapshot_download

root = Path(os.environ['SGLANG_ROOT']) / 'models'
models = [
    ('Qwen/Qwen3.8-27B-FP8', '017b9c7af6b5689d5dd426a76e0bc077eb5ca20a',
     'Qwen3.8-27B-FP8', ['*.json', '*.safetensors', '*.jinja', 'README.md', 'LICENSE*', '*.txt']),
    ('RadixArk/Qwen3.8-27B-DSpark', 'b9a5dbdf03bc999c6c73c426b19c2d9041cea393',
     'Qwen3.8-27B-DSpark', ['config.json', 'model.safetensors', 'README.md']),
]
for repo, revision, name, files in models:
    target = root / name
    if target.exists():
        raise FileExistsError(f'复用已有资产或检查未完成下载：{target}')
    snapshot_download(repo_id=repo, revision=revision, local_dir=target, allow_patterns=files)
    (target / 'SOURCE.json').write_text(json.dumps({
        'repo_id': repo, 'revision': revision, 'transport': os.environ['HF_ENDPOINT']
    }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
PY
```

目标模型保留发布的 FP8 权重，以及完整 Tokenizer、聊天模板和视觉处理器。DSPARK 草稿使用原 BF16 Safetensors，由 SGLang 原生 `DSparkDraftModel`加载，来源见[公开草稿模型](https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark)。

模型身份由来源和必要配置确认，不能因为文件夹改名就修改配置架构。下载或迁移时保留来源记录和已有摘要，不反复哈希整套目标权重；目录可访问也不代表模型已经能够运行。

## 6. 启动最终DSPARK配置

先确认所选两张 GPU 空闲，再按机器实际情况设置 `SGLANG_GPUS`。默认 GPU 选择和完整服务参数都由启动脚本维护；模型路径与默认位置不同时，再设置前述两个模型变量。

进入服务目录，在 tmux 会话中启动：

```bash
cd "$SGLANG_ROOT"
tmux new -s sglang-service
# 以下在tmux会话内执行。
set -o pipefail
bash start_server.sh 2>&1 | tee logs/server.log
```

默认配置使用双卡 DSPARK，采用静态验证，草稿模型使用独立 FlashInfer 后端。目标模型的注意力、线性注意力和 FP8 GEMM 使用 Triton，视觉部分使用 SDPA，采样使用 PyTorch。

配置保留 float32 GDN 状态、原 KV 精度和解码 CUDA 图，关闭此前出现显存不足的预填充图捕获。驱动库搜索路径只用于 JIT 链接，不把 stubs 当作真实驱动加载。服务监听回环地址，模型别名和 JSON Schema 接口与客户端保持一致。

等待服务就绪后，按 VLM 手册选少量已有冻结任务输入，检查真实多图请求和完整 Schema 响应。初始化与正式计时分开记录，纯文本检查不能代替视觉路径。已有有效结果不需要因文档改写重跑；声明的上下文容量也不代表满容量和额外并发已经验证。远程访问和停止方法见[VLM 手册](技术实现-VLM局部决策.md#远程服务启停与访问默认)。

## 7. 维护与排错

| 现象 | 先检查什么 |
|---|---|
| glibc 或 wheel 不兼容 | 确认正在使用本机构建的 SGLang wheel，再定位具体不兼容依赖。不改平台标签或系统 glibc。 |
| 解析到 CUDA 13 或错误 Torch | 确认四项源码补丁和 cu129 约束同时生效，不跳过依赖检查。 |
| Rust、C++ 或 protoc 构建失败 | 从首个错误检查固定源码、独立工具链、Conda 编译器和 protoc，保留有效缓存。 |
| JIT 找不到 `-lcuda` | 检查系统驱动库及启动脚本的编译链接路径。stubs 不加入运行库路径。 |
| 显存不足或图捕获失败 | 检查两卡空闲情况及最终 DSPARK 配置，特别是预填充图捕获是否关闭。不恢复失败候选，也不照搬 48 GB 单卡参数。 |
| 双卡通信或 PTX 失败 | 针对实际算子和拓扑诊断。自定义 AllReduce 标志没有禁用，并不证明该功能已生效；不要盲换驱动或 Torch。 |
| JSON 或视觉推理失败 | 保留原始错误和响应，区分输入、Schema、算子和显存问题，不修复模型回答或自动重试。 |
| 端口连接被拒绝 | 检查服务是否就绪，以及 SSH 隧道是否建立。服务已停止时按需手动启动。 |

本机记录维护实际环境、源码和模型来源，以及真正执行的检查。完整依赖可直接查询现有 Conda 和 pip 元数据，不重复导出多份快照。

目录中保留必要源码、补丁、wheel、运行和构建入口，以及仍有效的缓存。安装器、一次性脚本和重复测试结果可以清理；汇总与复现必需的原始结果集中保存。服务和编译检查按实际变更执行，不把全套检查作为每次启动或文档修改的前置。
