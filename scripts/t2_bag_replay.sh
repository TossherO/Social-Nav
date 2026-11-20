#!/usr/bin/env bash
# 仅供离线观察。独立 ROS master 避免与正在运行的实机会话共用话题。
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
usage() {
  printf '%s\n' '用法: bash scripts/t2_bag_replay.sh --bag 文件 [--floor floor1|floor2] [--start 秒] [--rate 倍率] [--initial-z 米] [--pose-file 文件] [--reuse-session 文件 --auto-follow [--auto-follow-single-person]] [--ros-port 端口] [--t2-backend hosted|dspark] [--t2-image-format jpeg|png] [--t2-jpeg-quality 1..95]' \
    '启动独立 ROS、RViz 和 Agent 界面；不会连接底盘。默认二楼、1 倍速和 hosted。'
}
bag='' floor=floor2 start=0 rate=1 pose_file='' initial_z='' reuse_session='' auto_follow=0 auto_follow_single_person=0 port=11329 backend=hosted image_format=auto jpeg_quality=70
while (($#)); do
  case "$1" in
    --help|-h) usage; exit 0;;
    --auto-follow) auto_follow=1; shift;;
    --auto-follow-single-person) auto_follow_single_person=1; shift;;
    --bag|--floor|--start|--rate|--pose-file|--initial-z|--reuse-session|--ros-port|--t2-backend|--t2-image-format|--t2-jpeg-quality)
      if (($# < 2)); then usage >&2; exit 2; fi
      case "$1" in
        --bag) bag=$2;; --floor) floor=$2;; --start) start=$2;; --rate) rate=$2;;
        --pose-file) pose_file=$2;; --initial-z) initial_z=$2;; --reuse-session) reuse_session=$2;; --ros-port) port=$2;; --t2-backend) backend=$2;;
        --t2-image-format) image_format=$2;; --t2-jpeg-quality) jpeg_quality=$2;;
      esac
      shift 2;;
    *) echo "未知参数：$1" >&2; usage >&2; exit 2;;
  esac
done
[[ -f "$bag" ]] || { echo '请使用 --bag 指定存在的 bag 文件。' >&2; exit 2; }
[[ "$floor" == floor1 || "$floor" == floor2 ]] || { echo '楼层只能是 floor1 或 floor2。' >&2; exit 2; }
[[ "$backend" == hosted || "$backend" == dspark ]] || { echo 'T2 后端只能是 hosted 或 dspark。' >&2; exit 2; }
[[ "$image_format" == auto || "$image_format" == png || "$image_format" == jpeg ]] || { echo '图像格式只能是 jpeg 或 png。' >&2; exit 2; }
[[ "$jpeg_quality" =~ ^[1-9][0-9]?$ ]] && ((jpeg_quality<=95)) || { echo 'JPEG 质量需在 1 至 95 之间。' >&2; exit 2; }
if [[ "$image_format" == auto ]]; then
  if [[ "$backend" == hosted ]]; then image_format=jpeg; else image_format=png; fi
fi
if [[ "$auto_follow" == 1 && ! -f "$reuse_session" ]]; then echo '--auto-follow 需要用 --reuse-session 指定已登记任务的 session.sqlite。' >&2; exit 2; fi
if [[ "$auto_follow_single_person" == 1 && "$auto_follow" != 1 ]]; then echo '--auto-follow-single-person 需要同时启用 --auto-follow。' >&2; exit 2; fi
if [[ -n "$reuse_session" && "$auto_follow" != 1 ]]; then echo '--reuse-session 需与 --auto-follow 一起使用；也可在窗口中手动选择历史任务。' >&2; exit 2; fi
[[ "$port" =~ ^[0-9]+$ ]] && ((port>=1024 && port<=65535)) || { echo 'ROS 端口需在 1024 至 65535 之间。' >&2; exit 2; }
[[ -f outputs/runtime/environment/activate.sh ]] || { echo '缺少本机运行环境 activate.sh。' >&2; exit 1; }
source outputs/runtime/environment/activate.sh
source perception/scripts/ros1_env.sh
export ROS_MASTER_URI="http://127.0.0.1:$port" ROS_HOSTNAME=127.0.0.1 ROS_IP=127.0.0.1
python -B - "$port" <<'PY'
import socket, sys
with socket.socket() as s:
    try: s.bind(('127.0.0.1', int(sys.argv[1])))
    except OSError as error: raise SystemExit('回放 ROS 端口不可用，请使用 --ros-port 指定其他端口：'+str(error))
PY
if [[ "$floor" == floor1 ]]; then config=navigation/commissioning/interactive_agent_floor1.yaml
else config=navigation/commissioning/interactive_agent.yaml; fi
catalog=$(python -B - "$config" <<'PY'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1]))['catalog'])
PY
)
session="outputs/navigation/bag_agent_replay/$(date +%Y%m%d_%H%M%S_%N)"
mkdir -p "$session/ros_logs"
export ROS_LOG_DIR="$PWD/$session/ros_logs"
ros_pid='' ui_pid=''
cleanup() {
  if [[ -n "$ui_pid" ]]; then kill "$ui_pid" 2>/dev/null || true; wait "$ui_pid" 2>/dev/null || true; fi
  if [[ -n "$ros_pid" ]]; then kill "$ros_pid" 2>/dev/null || true; wait "$ros_pid" 2>/dev/null || true; fi
}
trap cleanup EXIT
roscore -p "$port" >"$session/roscore.log" 2>&1 & ros_pid=$!
python -B - <<'PY'
import time, rosgraph
deadline=time.monotonic()+15
while True:
    try: rosgraph.Master('/bag_replay_start').getPid(); break
    except Exception:
        if time.monotonic()>=deadline: raise SystemExit('独立 ROS master 启动失败，请查看 roscore.log')
        time.sleep(.1)
PY
ui_args=(--replay --runtime-socket "$session/interactive/run/commands.sock" --catalog "$catalog")
if [[ "$auto_follow" == 1 ]]; then ui_args+=(--reuse-session "$reuse_session" --auto-follow); fi
if [[ "$auto_follow_single_person" == 1 ]]; then ui_args+=(--auto-follow-single-person); fi
python -B -m decision.agent.manual_ui "${ui_args[@]}" & ui_pid=$!
args=(--bag "$bag" --config "$config" --output "$session/interactive" --start "$start" --rate "$rate"
      --agent-observations --agent-follow-candidates --t2-backend "$backend" --record-t2-requests)
pose_note='在 RViz 选择 bag 起始位置与方向；定位就绪后可开始任务。'
if [[ -n "$pose_file" ]]; then
  args+=(--pose-file "$pose_file")
  pose_note='已指定初值文件。等待地图关联就绪，并在 RViz 核对扫描是否对齐。'
fi
if [[ -n "$initial_z" ]]; then args+=(--initial-z "$initial_z"); fi
if [[ "$image_format" == jpeg ]]; then args+=(--t2-jpeg-quality "$jpeg_quality"); fi
printf '%s\n' "回放记录：$session" "$pose_note" \
  '终端输入 p 暂停、r 继续、q 退出。暂停期间不会产生新的观测；结束后需重新启动。' \
  "T2 图像格式：$image_format；模型请求会发送 bag 图像及相关观察至所选服务。"
python -B -m runtime.interactive "${args[@]}"
