#!/usr/bin/env bash
# 使用已准备的任务场景配置进行 Wheelchair 增量微调。

set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

usage() {
    cat <<'EOF'
用法:
  ./train_wheelchair_task_scene.sh
  DRY_RUN=1 ./train_wheelchair_task_scene.sh
  MAX_SWEEPS=3 RUN_TAG=task_scene_3frame_new ./train_wheelchair_task_scene.sh

常用环境变量:
  TRAIN_CFG            默认 data/wheelchair_det_extra/wheelchair_20261016.yaml
                       应为已准备的 TASK_SCENE 配置；相对路径以脚本目录为准
  CUDA_VISIBLE_DEVICES 默认 0,1
  NPROC_PER_NODE       默认 2
  MAX_SWEEPS           默认 1，包含当前帧；0 按单帧处理
  RUN_TAG              默认 task_scene_<帧数>frame；新实验请更换名字
  PRECISION            默认 fp32，可选 fp32/fp16/bf16
  WORKERS              默认 2（每个训练进程）
  BATCH_SIZE           可覆盖每卡 batch；默认沿用 YAML
  EPOCHS               可覆盖训练轮数；默认沿用 YAML
  EPOCH_SAMPLES        可覆盖每轮采样预算；默认沿用 YAML
  EVAL_INTERVAL        默认 1；0 关闭训练中验证
  PRETRAINED_MODEL     可覆盖初始化权重；默认按帧数自动选择 Stage 3 最佳权重
  CKPT                 可指定完整恢复断点；优先于已有断点及初始化
  FIX_RANDOM_SEED      默认 1，使用训练入口的固定随机种子；0 关闭
  CKPT_SAVE_INTERVAL   默认 1
  MAX_CKPT_SAVE_NUM    默认 30
  LOG_GPU_MEMORY       默认 0；1 记录各 rank 显存
  PYTHON               默认 python，仅用于 DRY_RUN 配置检查
  DRY_RUN              默认 0；1 仅解析配置、选择权重及打印命令

脚本复用 tools/train.py 的初始化与自动恢复逻辑，不重新导出数据、准备配置或运行对比。
所有相对文件路径均以脚本目录为准。改变输入、帧数或采样策略后使用新 RUN_TAG。
EOF
}

if (( $# > 0 )); then
    if [[ "$1" == "-h" || "$1" == "--help" ]] && (( $# == 1 )); then
        usage
        exit 0
    fi
    echo "[ERROR] 本脚本不接受位置参数；请通过环境变量覆盖设置。" >&2
    usage >&2
    exit 2
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

TRAIN_CFG="${TRAIN_CFG:-data/wheelchair_det_extra/wheelchair_20261016.yaml}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MAX_SWEEPS="${MAX_SWEEPS:-1}"
PRECISION="${PRECISION:-fp32}"
WORKERS="${WORKERS:-2}"
BATCH_SIZE="${BATCH_SIZE:-}"
EPOCHS="${EPOCHS:-}"
EPOCH_SAMPLES="${EPOCH_SAMPLES:-}"
EVAL_INTERVAL="${EVAL_INTERVAL:-1}"
PRETRAINED_MODEL="${PRETRAINED_MODEL:-}"
CKPT="${CKPT:-}"
FIX_RANDOM_SEED="${FIX_RANDOM_SEED:-1}"
CKPT_SAVE_INTERVAL="${CKPT_SAVE_INTERVAL:-1}"
MAX_CKPT_SAVE_NUM="${MAX_CKPT_SAVE_NUM:-30}"
LOG_GPU_MEMORY="${LOG_GPU_MEMORY:-0}"
PYTHON="${PYTHON:-python}"
DRY_RUN="${DRY_RUN:-0}"

require_integer() {
    local name=$1 value=$2 minimum=$3
    if [[ ! "${value}" =~ ^[0-9]+$ ]] || (( 10#${value} < minimum )); then
        echo "[ERROR] ${name} 必须是大于等于 ${minimum} 的整数，当前值: ${value}" >&2
        return 1
    fi
}

require_integer MAX_SWEEPS "${MAX_SWEEPS}" 0
MAX_SWEEPS="$((10#${MAX_SWEEPS}))"
if (( MAX_SWEEPS < 1 )); then
    MAX_SWEEPS=1
fi
RUN_TAG="${RUN_TAG:-task_scene_${MAX_SWEEPS}frame}"

on_error() {
    local exit_code=$?
    echo "[ERROR] 任务场景微调失败（退出码 ${exit_code}）。" >&2
    echo "输入与配置不变时，可用相同 RUN_TAG=${RUN_TAG} 续训。" >&2
    exit "${exit_code}"
}
trap on_error ERR

for name in NPROC_PER_NODE CKPT_SAVE_INTERVAL MAX_CKPT_SAVE_NUM; do
    require_integer "${name}" "${!name}" 1
done
for name in WORKERS EVAL_INTERVAL; do
    require_integer "${name}" "${!name}" 0
done
for name in BATCH_SIZE EPOCHS EPOCH_SAMPLES; do
    if [[ -n "${!name}" ]]; then
        require_integer "${name}" "${!name}" 1
    fi
done
for name in FIX_RANDOM_SEED LOG_GPU_MEMORY DRY_RUN; do
    if [[ "${!name}" != "0" && "${!name}" != "1" ]]; then
        echo "[ERROR] ${name} 只能取 0 或 1。" >&2
        exit 2
    fi
done
case "${PRECISION}" in
    fp32|fp16|bf16) ;;
    *) echo "[ERROR] PRECISION 只能取 fp32、fp16 或 bf16。" >&2; exit 2 ;;
esac
if [[ ! "${RUN_TAG}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "[ERROR] RUN_TAG 必须以字母或数字开头，且只能包含字母、数字、点、下划线和连字符。" >&2
    exit 2
fi
IFS=',' read -r -a VISIBLE_GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
if (( ${#VISIBLE_GPU_IDS[@]} < NPROC_PER_NODE )); then
    echo "[ERROR] NPROC_PER_NODE 超过 CUDA_VISIBLE_DEVICES 中的设备数。" >&2
    exit 2
fi
if [[ ! -s "${TRAIN_CFG}" ]]; then
    echo "[ERROR] 缺少任务场景配置: ${TRAIN_CFG}" >&2
    exit 1
fi

train_args=(
    --cfg_file "${TRAIN_CFG}" --extra_tag "${RUN_TAG}"
    --max_sweeps "${MAX_SWEEPS}" --precision "${PRECISION}" --workers "${WORKERS}"
    --eval_interval "${EVAL_INTERVAL}" --ckpt_save_interval "${CKPT_SAVE_INTERVAL}"
    --max_ckpt_save_num "${MAX_CKPT_SAVE_NUM}"
)
for option in BATCH_SIZE:--batch_size EPOCHS:--epochs EPOCH_SAMPLES:--epoch_samples \
              PRETRAINED_MODEL:--pretrained_model CKPT:--ckpt; do
    name="${option%%:*}"
    if [[ -n "${!name}" ]]; then
        train_args+=("${option#*:}" "${!name}")
    fi
done
if [[ "${FIX_RANDOM_SEED}" == "1" ]]; then
    train_args+=(--fix_random_seed)
fi
if [[ "${LOG_GPU_MEMORY}" == "1" ]]; then
    train_args+=(--log_gpu_memory)
fi
command=(torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" tools/train.py "${train_args[@]}")

echo "========== Wheelchair 任务场景微调 =========="
echo "项目目录: ${PROJECT_DIR}"
echo "配置: ${TRAIN_CFG}"
echo "GPU: ${CUDA_VISIBLE_DEVICES} (进程数: ${NPROC_PER_NODE})"
echo "精度/单卡 batch/workers: ${PRECISION}/${BATCH_SIZE:-YAML默认}/${WORKERS}"
echo "输入帧数: ${MAX_SWEEPS}  训练轮数: ${EPOCHS:-YAML默认}"
echo "实验标签: ${RUN_TAG}"
printf '  '
printf '%q ' "${command[@]}"
printf '\n'

if [[ "${DRY_RUN}" == "1" ]]; then
    "${PYTHON}" tools/train.py "${train_args[@]}" --dry_run
    exit 0
fi

command -v torchrun >/dev/null || {
    echo "[ERROR] 当前环境找不到 torchrun，请先激活检测开发环境。" >&2
    exit 1
}
"${command[@]}"
