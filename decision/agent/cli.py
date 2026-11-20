"""Command parsing and workflow selection for the long-horizon Agent."""
import argparse
import json
from pathlib import Path
from . import CONFIG_DIR, ROOT
from .tools.places import Catalog
from .workflows.common import emit


def main():
    parser = argparse.ArgumentParser(description='单VLM工具循环：仿真、观察及显式开启的实机目标执行')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('catalog', help='只读查看地点，不读取API凭证')
    p.add_argument('--catalog', default=str(CONFIG_DIR/'floor1.json'))
    p.add_argument('--map-condition', choices=('complete','simple','none'))
    p = sub.add_parser('status', help='只读检查任务，不自动恢复执行')
    p.add_argument('--session', required=True)
    for name in ('run', 'shadow', 'execute', 'manual'):
        p = sub.add_parser(name)
        p.add_argument('--instruction', required=name!='manual')
        p.add_argument('--catalog', default=str(CONFIG_DIR/('simulation.json' if name == 'run' else 'floor1.json')))
        p.add_argument('--map-condition', choices=('complete','simple','none'), help='地图先验条件；当前仅complete支持目标执行')
        p.add_argument('--output', required=True)
        p.add_argument('--actions', help='显式工具动作JSON数组；不读取凭证或调用API，不接收旧计划格式')
        p.add_argument('--max-requests', type=int, default=512)
        p.add_argument('--api-config', default=str(ROOT/'configs/agent_api.local.json'))
        p.add_argument('--cache-mode',choices=('explicit','implicit'),default='implicit',
                       help='默认隐式缓存；explicit显式标记稳定前缀，不自动降级或重试')
        p.add_argument('--thinking-mode',choices=('planning','off','low'),default='planning',
                       help='planning 仅登记阶段使用指定思考强度；off 和 low 分别用于全程关闭或开启的对照')
        p.add_argument('--planning-thinking-mode',choices=('off','low','medium'),default='low',
                       help='planning 模式下任务登记阶段的思考强度，默认 low；登记后关闭思考')
        p.add_argument('--interactive', action='store_true')
        p.add_argument('--no-memory', action='store_true', help='关闭附加证据记忆，仍保留原任务与有界工具历史')
        if name == 'run':
            p.add_argument('--max-s', type=float, default=180)
            p.add_argument('--scripted-user-events', action='store_true', help='仅合成测试显式生成并记录继续事件')
        elif name == 'shadow':
            group = p.add_mutually_exclusive_group(required=True)
            group.add_argument('--observations')
            group.add_argument('--runtime-socket')
            p.add_argument('--sequences', nargs='+', type=int)
            p.add_argument('--duration', type=float, default=60)
            p.add_argument('--gui-commands', action='store_true', help=argparse.SUPPRESS)
            p.add_argument('--require-bag', action='store_true', help='仅连接不启用底盘的 bag 回放会话')
            p.add_argument('--reuse-session', help='复用历史任务登记，不复用执行进度')
            p.add_argument('--reuse-digest', help=argparse.SUPPRESS)
            p.add_argument('--auto-follow', action='store_true', help='回放定位就绪后自动启动已登记的第一项跟随任务')
            p.add_argument('--auto-follow-single-person',action='store_true',
                           help='回放时用当前唯一具有可靠二维与三维关联的行人启动跟随')
        else:
            p.add_argument('--runtime-socket', required=True)
            p.add_argument('--duration', type=float, default=600)
            if name == 'manual':
                p.add_argument('--gui-commands', action='store_true', help=argparse.SUPPRESS)
                p.add_argument('--cross-floor', action='store_true', help='启用人工驾驶跨楼层任务与乘梯观察')
                p.add_argument('--reuse-session', help='从只读历史会话复用已登记要求，从第一项重新执行')
                p.add_argument('--reuse-digest', help=argparse.SUPPRESS)
                p.add_argument('--record-bag',action='store_true',help='任务登记后自动录制原始导航传感器话题')
                p.add_argument('--bag-output-root',default='/home/tl/rosbag',help='自动录包输出根目录')
    p = sub.add_parser('evaluate', help='运行少量合成工具反馈场景，不调用API')
    p.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'catalog':
        emit(Catalog.load(args.catalog,condition=args.map_condition).prompt())
    elif args.command == 'status':
        import sqlite3
        db = sqlite3.connect(Path(args.session).resolve().as_uri()+'?mode=ro', uri=True)
        try:
            row = db.execute('SELECT value FROM state WHERE id=1').fetchone()
            emit(dict(task=None if row is None else json.loads(row[0]), auto_resume=False,
                      recovery_status='requires_navigation_reconciliation'))
        finally:
            db.close()
    elif args.command == 'evaluate':
        from .workflows.evaluate import evaluate
        evaluate(Path(args.output))
    else:
        if not 1 <= args.max_requests <= 4096:
            raise ValueError('Use one to 4096 decision turns')
        if args.command == 'run':
            if args.max_s <= 0:
                raise ValueError('Positive simulation time required')
            from .workflows.run import run_simulation
            run_simulation(args)
        elif args.command == 'shadow':
            from .workflows.shadow import run
            run(args)
        elif args.command == 'execute':
            from .workflows.live import run
            run(args)
        else:
            from .workflows.manual import run
            run(args)


if __name__ == '__main__':
    main()
