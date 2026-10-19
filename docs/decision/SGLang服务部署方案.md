# SGLang服务部署方案：原生Conda、源码构建与DSPARK

最终方案面向自有Ubuntu 20.04／glibc 2.31服务器：独立Python 3.12 Conda环境、CUDA 12.9工具链与官方cu129 PyTorch、固定SGLang 0.5.19源码构建、双卡张量并行、Qwen FP8目标模型与公开DSPARK草稿。系统驱动和glibc保持原值；SGLang官方wheel的glibc门槛由本机构建处理。当前实现已通过项目冻结三图输入的服务器本机调用、开发机经SSH调用及短时0.5Hz测试，实际机器信息、成绩与服务状态由 `ENVIRONMENT.local.md` 定位。

本文只维护从零安装与最终构建材料。该远程服务是项目VLM推理与离线调试的首选，资源选择与本地备用条件见 [VLM部署与调用](VLM部署与调用.md)；已有服务器直接按该手册启停和测试，不重复安装。具体服务参数以部署脚本为准，依赖约束以 `config/constraints-cu129.txt` 为准；它是关键版本约束，不是全部传递依赖的永久锁文件。

## 1. 准备目录与部署材料

安装命令全部在目标服务器的Bash终端执行。先确认Linux x86_64、glibc、现有驱动、两张所选卡的空闲资源及部署盘空间：

```bash
getconf GNU_LIBC_VERSION
nvidia-smi
df -h "$HOME" /tmp
```

当前方案针对已验证的CUDA 12驱动路线；新设备仍须核对真实兼容性，不按镜像名称判断。[CUDA兼容规则](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)。缺少基础工具时才安装系统包，不重装系统CUDA或驱动：

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  curl ca-certificates git build-essential libnuma1 libibverbs1 ffmpeg tmux
```

先在开发机生成含部署材料的便携包并传至服务器，方法见VLM手册。在其解压根目录设置 `SGLANG_MATERIALS="$PWD"`；若服务器已有本仓库，改指向 `decision/vlm/deployment/`，两者提供相同实现。首次创建的服务目录必须尚不存在：

```bash
export SGLANG_MATERIALS="$PWD"
export SGLANG_ROOT="$HOME/sglang-service"
mkdir "$SGLANG_ROOT"
mkdir -p "$SGLANG_ROOT/build" "$SGLANG_ROOT/config" \
  "$SGLANG_ROOT/models" "$SGLANG_ROOT/logs" "$SGLANG_ROOT/cache"
cp "$SGLANG_MATERIALS/start_server.sh" "$SGLANG_MATERIALS/runtime_env.sh" "$SGLANG_ROOT/"
cp "$SGLANG_MATERIALS/build/build_sglang.sh" \
  "$SGLANG_MATERIALS/build/source-cu129.patch" "$SGLANG_ROOT/build/"
cp "$SGLANG_MATERIALS/config/constraints-cu129.txt" "$SGLANG_ROOT/config/"
```

模型、环境、源码和缓存直接放各自唯一目录，不复制已有大资产。服务根目录只维护技术实现和环境／测试记录两份本机MD；实际测试结果集中在开发仓库，服务器保留冻结输入与客户端即可。

## 2. 新建独立Conda环境

使用已有Miniconda管理器；没有时才安装。下例将管理器放在服务目录之外，临时安装器用完即删，不修改全局Shell配置：

```bash
export SGLANG_CONDA_ROOT="$HOME/miniconda3"
# 仅当上面的管理器不存在时执行以下三行。
curl -fL https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh \
  -o /tmp/Miniconda3-sglang.sh
bash /tmp/Miniconda3-sglang.sh -b -p "$SGLANG_CONDA_ROOT"
rm /tmp/Miniconda3-sglang.sh
```

从管理器创建新的专用环境，不替换项目开发或ROS环境：

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

`runtime_env.sh` 固定环境内编译器、ptxas、运行库与缓存路径，并隔离其他Python路径。Conda sysroot仅供编译，不替换系统glibc；CUDA stubs不进入运行库搜索路径。非默认Conda路径在每次调用构建／启动脚本前设置 `SGLANG_CONDA_ROOT` 和 `SGLANG_CONDA_ENV`，不把其他机器的路径写死在源码里。

## 3. 安装cu129依赖

所有运行依赖安装都引用同一约束文件。PyTorch使用官方cu129源；普通Python包优先清华，专属CUDA wheel使用发行方源：

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

内核的 `cp310-abi3` wheel可用于Python 3.12，不要求使用Python 3.10。安装某个原生包不表示启用了它的专用算子；本方案运行后端仍由最终启动脚本选择，不启用CUDA 13或Hopper专用路径。

## 4. 固定源码、适配依赖并构建

SGLang官方Python 3.12 Linux wheel要求glibc至少2.34，当前系统为2.31，因此直接构建固定源码，不改wheel标签。[发行文件](https://pypi.org/project/sglang/0.5.19/#files)。获取完整仓库，保持Rust workspace与Cargo锁文件：

```bash
export SGLANG_SRC="$SGLANG_ROOT/build/sglang"
git clone --depth 1 --branch v0.5.19 \
  https://github.com/sgl-project/sglang.git "$SGLANG_SRC"
test "$(git -C "$SGLANG_SRC" rev-parse HEAD)" = 0bcd822377da7b5718e674eaf9c870d349424dd1
git -C "$SGLANG_SRC" apply "$SGLANG_ROOT/build/source-cu129.patch"
```

补丁只适配四项依赖：cuda-python改为CUDA 12范围、FlashInfer和Humming使用cu12选项、CUTLASS取消cu13额外选项；算法源码不改。已应用补丁的源码不要再次执行上述安装步骤，也不对他人的源码副本打补丁。依赖声明与约束共同防止解析回CUDA 13路线。

使用独立Rust工具链与RsProxy依赖镜像：

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

构建入口使用当前cu129 Torch、Conda编译器和protoc，关闭Python构建隔离，构建全部四个Rust扩展；输出与唯一构建日志分别位于 `build/wheelhouse/` 和 `logs/build.log`。该脚本只构建，不安装或启动服务。失败时看首个实际错误，按需降低 `CARGO_BUILD_JOBS` 并复用缓存，不跳过扩展、另装CPU LibTorch或重建整套环境。

## 5. 准备固定目标与草稿模型

已有可信模型直接复用，可分别设置 `SGLANG_MODEL_DIR` 与 `SGLANG_DRAFT_MODEL_DIR`。新机器通过hf-mirror取得以下固定revision的公开资产，不下载或执行模型仓库Python代码：

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

模型身份与必要配置按来源绑定，不因文件夹名称修改配置架构。目标保持发布的FP8权重与完整Tokenizer、聊天模板和视觉处理器；DSPARK使用原BF16 Safetensors，由当前SGLang原生 `DSparkDraftModel` 加载。下载／迁移时保留来源和已有资产摘要，不反复哈希整套目标权重；目录可达不证明模型可以运行。[公开草稿](https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark)。

## 6. 启动最终DSPARK配置

确认所选两卡空闲，按实际条件设置 `SGLANG_GPUS`；默认值与全部服务参数只由启动脚本维护。模型路径不同时设置上述两个模型变量。然后按VLM手册进入tmux启动：

```bash
cd "$SGLANG_ROOT"
tmux new -s sglang-service
# 以下在tmux会话内执行。
set -o pipefail
bash start_server.sh 2>&1 | tee logs/server.log
```

默认使用双卡DSPARK、静态验证和独立草稿FlashInfer；目标注意力／线性注意力／FP8 GEMM使用Triton，视觉使用SDPA，采样使用PyTorch。保留float32 GDN状态、原KV精度与解码CUDA图，关闭已出现显存不足的预填充图捕获。驱动库搜索路径只用于JIT链接，不将stubs加载为真实驱动。服务监听回环地址，模型别名与JSON Schema接口保持一致。

等待就绪后，用便携客户端的三个代表场景确认真实多图与完整JSON响应，初始化与计时分开记录；已有有效测试无需因文档调整重跑。远程访问、停止和完整对比均见VLM手册；纯文本检查不能代替视觉路径。上下文包含图像、文本和输出预算，声明容量不代表已验证满容量或额外并发。

## 7. 维护与排错

| 现象 | 处理 |
|---|---|
| glibc／wheel不兼容 | 确认使用本地SGLang wheel，定位具体其他依赖；不改平台标签或系统glibc |
| 解析到CUDA 13或错误Torch | 确认四项源码补丁和cu129约束同时生效，不跳过依赖检查 |
| Rust／C++／protoc失败 | 查看首个构建错误，检查固定源码、独立工具链、Conda编译器与protoc，复用有效缓存 |
| JIT找不到 `-lcuda` | 确认系统驱动库及启动脚本的编译链接路径，stubs不加入运行库路径 |
| 显存不足／图捕获失败 | 确认两卡空闲及最终DSPARK配置，尤其预填充图关闭；不恢复失败候选或照搬48GB单卡参数 |
| 双卡通信／PTX失败 | 针对实际算子与拓扑诊断；自定义AllReduce标志未禁用不证明其已生效，不盲换驱动或Torch |
| JSON／视觉失败 | 保留原始错误和响应，区分输入、Schema、算子与显存，不修复回答或自动重试 |
| 端口连接拒绝 | 确认服务实际就绪及SSH隧道，服务已停止时先按需手动启动 |

本机记录维护实际环境、源码／模型来源及已执行检查；完整依赖直接查询现有Conda／pip元数据，不再导出多份快照。只保留必要源码、补丁、wheel、运行／构建入口和有效缓存；临时脚本、安装器及重复测试结果清理，汇总和复现必需原始结果集中到同一实验目录。服务与编译检查按实际变更执行，不将全套检查作为每次启动或文档修改的前置步骤。
