"""Command parsing and workflow selection for the long-horizon Agent."""
import argparse
import json
from pathlib import Path
from . import CONFIG_DIR, ROOT
from .tools.places import Catalog
from .workflows.common import emit


def main():
    parser = argparse.ArgumentParser(description='单VLM工具循环：仿真与真实观察影子测试，无底盘输出')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('catalog', help='只读查看地点，不读取API凭证')
    p.add_argument('--catalog', default=str(CONFIG_DIR/'floor1.json'))
    p = sub.add_parser('status', help='只读检查任务，不自动恢复执行')
    p.add_argument('--session', required=True)
    for name in ('run', 'shadow'):
        p = sub.add_parser(name)
        p.add_argument('--instruction', required=True)
        p.add_argument('--catalog', default=str(CONFIG_DIR/('simulation.json' if name == 'run' else 'floor1.json')))
        p.add_argument('--output', required=True)
        p.add_argument('--actions', help='显式工具动作JSON数组；不读取凭证或调用API，不接收旧计划格式')
        p.add_argument('--max-requests', type=int, default=32)
        p.add_argument('--api-config', default=str(ROOT/'configs/agent_api.local.json'))
        p.add_argument('--interactive', action='store_true')
        p.add_argument('--no-memory', action='store_true', help='关闭附加证据记忆，仍保留原任务与有界工具历史')
        if name == 'run':
            p.add_argument('--max-s', type=float, default=180)
            p.add_argument('--scripted-user-events', action='store_true', help='仅合成测试显式生成并记录继续事件')
        else:
            group = p.add_mutually_exclusive_group(required=True)
            group.add_argument('--observations')
            group.add_argument('--runtime-socket')
            p.add_argument('--sequences', nargs='+', type=int)
            p.add_argument('--duration', type=float, default=60)
    p = sub.add_parser('evaluate', help='运行少量合成工具反馈场景，不调用API')
    p.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'catalog':
        emit(Catalog.load(args.catalog).value)
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
        if not 1 <= args.max_requests <= 64:
            raise ValueError('Use one to sixty-four decision turns')
        if args.command == 'run':
            if args.max_s <= 0:
                raise ValueError('Positive simulation time required')
            from .workflows.run import run_simulation
            run_simulation(args)
        else:
            from .workflows.shadow import run
            run(args)


if __name__ == '__main__':
    main()
