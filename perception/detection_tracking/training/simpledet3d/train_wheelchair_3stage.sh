#!/usr/bin/env bash
# 单机多卡顺序完成 Wheelchair 三阶段弱监督迁移训练。

set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

usage() {
    cat <<'EOF'
用法:
  ./train_wheelchair_3stage.sh
  DRY_RUN=1 ./train_wheelchair_3stage.sh

常用环境变量:
  CUDA_VISIBLE_DEVICES  默认 0,1,2,3,4,5,6,7
  NPROC_PER_NODE        默认 8
  BATCH_SIZE            默认 16（每张 GPU）
  WORKERS               默认 2（每个训练进程）
  PRECISION             默认 fp32，可选 fp32/fp16/bf16
  MAX_SWEEPS            默认留空并使用 YAML 配置；设为 1 时仅使用当前帧
  RUN_TAG               默认 wheelchair_exhaustive_center
  CODA_CKPT             阶段 1 的 CODA-16line 初始化权重
  STAGE{1,2,3}_EPOCHS   默认 10/10/5
  STAGE{1,2,3}_EVAL_INTERVAL 默认 0/1/1
  CKPT_SAVE_INTERVAL    默认 1，必须整除三个阶段的 epoch 数
  LOG_GPU_MEMORY        默认 0；设为 1 时记录各 rank 点/体素数和 CUDA 显存
  SKIP_COMPLETED        默认 1
  DRY_RUN               默认 0
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

# 限制每个 DDP/DataLoader 进程创建的 CPU 线程数，避免多进程线程竞争。
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

# 所有参数均可在命令前通过同名环境变量覆盖。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
PRECISION="${PRECISION:-fp32}"
MAX_SWEEPS="${MAX_SWEEPS:-}"
WORKERS="${WORKERS:-2}"
BATCH_SIZE="${BATCH_SIZE:-16}"
RUN_TAG="${RUN_TAG:-wheelchair_exhaustive_center}"

STAGE1_EPOCHS="${STAGE1_EPOCHS:-10}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-10}"
STAGE3_EPOCHS="${STAGE3_EPOCHS:-5}"
STAGE1_EVAL_INTERVAL="${STAGE1_EVAL_INTERVAL:-0}"
STAGE2_EVAL_INTERVAL="${STAGE2_EVAL_INTERVAL:-1}"
STAGE3_EVAL_INTERVAL="${STAGE3_EVAL_INTERVAL:-1}"
CKPT_SAVE_INTERVAL="${CKPT_SAVE_INTERVAL:-1}"
MAX_CKPT_SAVE_NUM="${MAX_CKPT_SAVE_NUM:-30}"
LOG_GPU_MEMORY="${LOG_GPU_MEMORY:-0}"

# 重复执行同一 RUN_TAG 时跳过已有最终 epoch checkpoint 的阶段；未完成阶段
# 会由 tools/train.py 自动加载其输出目录中最新的 checkpoint 继续训练。
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
DRY_RUN="${DRY_RUN:-0}"

CODA_CKPT="${CODA_CKPT:-output/voxelrcnn_coda_16line/finetune_4frame_from_4frame/ckpt/checkpoint_epoch_20.pth}"
if [[ "${CODA_CKPT}" != /* ]]; then
    CODA_CKPT="${PROJECT_DIR}/${CODA_CKPT}"
fi

STAGE1_CFG="configs/voxelrcnn_wheelchair_weak_stage1.yaml"
STAGE2_CFG="configs/voxelrcnn_wheelchair_weak_stage2.yaml"
STAGE3_CFG="configs/voxelrcnn_wheelchair_weak_stage3.yaml"

STAGE1_DIR="${PROJECT_DIR}/output/voxelrcnn_wheelchair_weak_stage1/${RUN_TAG}"
STAGE2_DIR="${PROJECT_DIR}/output/voxelrcnn_wheelchair_weak_stage2/${RUN_TAG}"
STAGE3_DIR="${PROJECT_DIR}/output/voxelrcnn_wheelchair_weak_stage3/${RUN_TAG}"
STAGE1_FINAL="${STAGE1_DIR}/ckpt/checkpoint_epoch_${STAGE1_EPOCHS}.pth"
STAGE2_FINAL="${STAGE2_DIR}/ckpt/checkpoint_epoch_${STAGE2_EPOCHS}.pth"
STAGE3_FINAL="${STAGE3_DIR}/ckpt/checkpoint_epoch_${STAGE3_EPOCHS}.pth"

CURRENT_STAGE="启动前检查"
on_error() {
    local exit_code=$?
    echo >&2
    echo "[ERROR] ${CURRENT_STAGE}失败（退出码 ${exit_code}）。" >&2
    echo "修复问题后用相同 RUN_TAG 重新执行，脚本会跳过已完成阶段并续训未完成阶段。" >&2
    exit "${exit_code}"
}
trap on_error ERR

require_file() {
    local path=$1
    local description=$2
    if [[ ! -s "${path}" ]]; then
        echo "[ERROR] 缺少${description}: ${path}" >&2
        return 1
    fi
}

require_dir() {
    local path=$1
    local description=$2
    if [[ ! -d "${path}" ]]; then
        echo "[ERROR] 缺少${description}: ${path}" >&2
        return 1
    fi
}

require_nonnegative_integer() {
    local name=$1
    local value=$2
    if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
        echo "[ERROR] ${name} 必须是非负整数，当前值: ${value}" >&2
        return 1
    fi
}

require_positive_integer() {
    local name=$1
    local value=$2
    require_nonnegative_integer "${name}" "${value}"
    if (( value < 1 )); then
        echo "[ERROR] ${name} 必须大于 0，当前值: ${value}" >&2
        return 1
    fi
}

validate_switch() {
    local name=$1
    local value=$2
    if [[ "${value}" != "0" && "${value}" != "1" ]]; then
        echo "[ERROR] ${name} 只能取 0 或 1，当前值: ${value}" >&2
        return 1
    fi
}

print_command() {
    printf '  '
    printf '%q ' "$@"
    printf '\n'
}

run_stage() {
    local stage_name=$1
    local cfg_file=$2
    local pretrained_model=$3
    local epochs=$4
    local eval_interval=$5
    local final_checkpoint=$6

    CURRENT_STAGE="${stage_name}"

    if [[ "${DRY_RUN}" == "0" && "${SKIP_COMPLETED}" == "1" && -s "${final_checkpoint}" ]]; then
        echo
        echo "[SKIP] ${stage_name}已完成: ${final_checkpoint}"
        return 0
    fi

    if [[ "${DRY_RUN}" == "0" ]]; then
        require_file "${pretrained_model}" "${stage_name}初始化 checkpoint"
    fi

    local command=(
        torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}"
        tools/train.py
        --cfg_file "${cfg_file}"
        --pretrained_model "${pretrained_model}"
        --extra_tag "${RUN_TAG}"
        --precision "${PRECISION}"
        --workers "${WORKERS}"
        --batch_size "${BATCH_SIZE}"
        --epochs "${epochs}"
        --eval_interval "${eval_interval}"
        --ckpt_save_interval "${CKPT_SAVE_INTERVAL}"
        --max_ckpt_save_num "${MAX_CKPT_SAVE_NUM}"
    )
    if [[ "${LOG_GPU_MEMORY}" == "1" ]]; then
        command+=(--log_gpu_memory)
    fi
    if [[ -n "${MAX_SWEEPS}" ]]; then
        command+=(--max_sweeps "${MAX_SWEEPS}")
    fi

    echo
    echo "[RUN] ${stage_name}"
    echo "  配置: ${cfg_file}"
    echo "  初始化: ${pretrained_model}"
    echo "  输出: ${final_checkpoint}"
    print_command "${command[@]}"

    if [[ "${DRY_RUN}" == "1" ]]; then
        return 0
    fi

    "${command[@]}"
    require_file "${final_checkpoint}" "${stage_name}最终 checkpoint"
}

echo "========== Wheelchair 三阶段训练 =========="
echo "项目目录: ${PROJECT_DIR}"
echo "GPU: ${CUDA_VISIBLE_DEVICES} (进程数: ${NPROC_PER_NODE})"
echo "精度/单卡 batch/workers: ${PRECISION}/${BATCH_SIZE}/${WORKERS}"
echo "输入帧数 MAX_SWEEPS: ${MAX_SWEEPS:-配置文件默认}"
echo "实验标签: ${RUN_TAG}"
echo "阶段 epoch: ${STAGE1_EPOCHS} -> ${STAGE2_EPOCHS} -> ${STAGE3_EPOCHS}"
echo "逐 rank 显存诊断: ${LOG_GPU_MEMORY}"

command -v torchrun >/dev/null || {
    echo "[ERROR] 当前环境找不到 torchrun，请先激活 simpledet3d conda 环境。" >&2
    exit 1
}

case "${PRECISION}" in
    fp32|fp16|bf16) ;;
    *)
        echo "[ERROR] PRECISION 只能取 fp32、fp16 或 bf16，当前值: ${PRECISION}" >&2
        exit 1
        ;;
esac

if [[ ! "${RUN_TAG}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "[ERROR] RUN_TAG 必须以字母或数字开头，且只能包含字母、数字、点、下划线和连字符: ${RUN_TAG}" >&2
    exit 1
fi

require_positive_integer NPROC_PER_NODE "${NPROC_PER_NODE}"
require_nonnegative_integer WORKERS "${WORKERS}"
require_positive_integer BATCH_SIZE "${BATCH_SIZE}"
require_positive_integer STAGE1_EPOCHS "${STAGE1_EPOCHS}"
require_positive_integer STAGE2_EPOCHS "${STAGE2_EPOCHS}"
require_positive_integer STAGE3_EPOCHS "${STAGE3_EPOCHS}"
require_nonnegative_integer STAGE1_EVAL_INTERVAL "${STAGE1_EVAL_INTERVAL}"
require_nonnegative_integer STAGE2_EVAL_INTERVAL "${STAGE2_EVAL_INTERVAL}"
require_nonnegative_integer STAGE3_EVAL_INTERVAL "${STAGE3_EVAL_INTERVAL}"
require_positive_integer CKPT_SAVE_INTERVAL "${CKPT_SAVE_INTERVAL}"
require_positive_integer MAX_CKPT_SAVE_NUM "${MAX_CKPT_SAVE_NUM}"
if [[ -n "${MAX_SWEEPS}" ]]; then
    require_nonnegative_integer MAX_SWEEPS "${MAX_SWEEPS}"
fi
validate_switch SKIP_COMPLETED "${SKIP_COMPLETED}"
validate_switch DRY_RUN "${DRY_RUN}"
validate_switch LOG_GPU_MEMORY "${LOG_GPU_MEMORY}"

for stage_epochs in "${STAGE1_EPOCHS}" "${STAGE2_EPOCHS}" "${STAGE3_EPOCHS}"; do
    if (( stage_epochs % CKPT_SAVE_INTERVAL != 0 )); then
        echo "[ERROR] CKPT_SAVE_INTERVAL=${CKPT_SAVE_INTERVAL} 必须整除各阶段 epoch 数；${stage_epochs} 无法整除。" >&2
        exit 1
    fi
done

IFS=',' read -r -a VISIBLE_GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
if (( ${#VISIBLE_GPU_IDS[@]} < NPROC_PER_NODE )); then
    echo "[ERROR] NPROC_PER_NODE=${NPROC_PER_NODE} 超过 CUDA_VISIBLE_DEVICES 中的设备数（${#VISIBLE_GPU_IDS[@]}）。" >&2
    exit 1
fi

require_file "${CODA_CKPT}" "CODA-16line 四帧预训练 checkpoint"
require_file "${PROJECT_DIR}/${STAGE1_CFG}" "阶段 1 配置"
require_file "${PROJECT_DIR}/${STAGE2_CFG}" "阶段 2 配置"
require_file "${PROJECT_DIR}/${STAGE3_CFG}" "阶段 3 配置"
require_file "${PROJECT_DIR}/data/coda_downsample/coda_infos_train.pkl" "CODA-16line 训练信息"
require_file "${PROJECT_DIR}/data/wheelchair_weak/wheelchair_weak_infos_train_all.pkl" "Wheelchair 全场景训练信息"
require_file "${PROJECT_DIR}/data/wheelchair_weak/wheelchair_weak_infos_train_indoor.pkl" "Wheelchair 室内训练信息"
require_file "${PROJECT_DIR}/data/wheelchair_weak/wheelchair_weak_infos_train_indoor_exhaustive.pkl" "Wheelchair 室内穷举中心训练信息"
require_file "${PROJECT_DIR}/data/wheelchair_weak/wheelchair_weak_infos_train_indoor_non_exhaustive.pkl" "Wheelchair 室内非穷举训练信息"
require_file "${PROJECT_DIR}/data/wheelchair_weak/wheelchair_weak_infos_val_indoor_exhaustive.pkl" "Wheelchair 室内穷举中心验证信息"
require_dir "${PROJECT_DIR}/data/coda_raw" "CODA 原始数据目录"
require_dir "${PROJECT_DIR}/data/wheelchair_raw" "Wheelchair 原始数据目录"

run_stage \
    "阶段 1/3：CODA + Wheelchair 混合适配" \
    "${STAGE1_CFG}" "${CODA_CKPT}" \
    "${STAGE1_EPOCHS}" "${STAGE1_EVAL_INTERVAL}" "${STAGE1_FINAL}"

run_stage \
    "阶段 2/3：室内穷举背景 + EMA Teacher" \
    "${STAGE2_CFG}" "${STAGE1_FINAL}" \
    "${STAGE2_EPOCHS}" "${STAGE2_EVAL_INTERVAL}" "${STAGE2_FINAL}"

if [[ "${DRY_RUN}" == "1" ]]; then
    STAGE2_TRANSFER="${STAGE2_DIR}/ckpt/best_model.pth"
elif [[ -s "${STAGE2_DIR}/ckpt/best_model.pth" ]]; then
    STAGE2_TRANSFER="${STAGE2_DIR}/ckpt/best_model.pth"
else
    echo "[WARN] 阶段 2 未生成 best_model.pth，将使用最终 epoch checkpoint。" >&2
    STAGE2_TRANSFER="${STAGE2_FINAL}"
fi

run_stage \
    "阶段 3/3：穷举中心精度固化" \
    "${STAGE3_CFG}" "${STAGE2_TRANSFER}" \
    "${STAGE3_EPOCHS}" "${STAGE3_EVAL_INTERVAL}" "${STAGE3_FINAL}"

if [[ "${DRY_RUN}" == "1" ]]; then
    echo
    echo "[DRY-RUN] 参数与路径检查通过，未启动训练。"
    exit 0
fi

if [[ -s "${STAGE3_DIR}/ckpt/best_model.pth" ]]; then
    FINAL_MODEL="${STAGE3_DIR}/ckpt/best_model.pth"
else
    FINAL_MODEL="${STAGE3_FINAL}"
fi

echo
echo "========== 三阶段训练全部完成 =========="
echo "最终候选模型: ${FINAL_MODEL}"
echo "阶段 1 输出: ${STAGE1_DIR}"
echo "阶段 2 输出: ${STAGE2_DIR}"
echo "阶段 3 输出: ${STAGE3_DIR}"
