"""Fixed allowlist Windows/Linux client package; no datasets or local settings."""
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[3]
FILES = ('__init__.py', '__main__.py', 'rules.py', 'store.py', 'backend.py', 'remote.py', 'assets.py',
         'app.py', 'views.py', 'cloud_view.py', 'requirements-client.txt')


def package_client(output):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('客户端包已存在，请选择新文件')
    output.parent.mkdir(parents=True, exist_ok=True)
    paths = [Path('decision/__init__.py'), Path('decision/vlm/__init__.py'), Path('decision/vlm/scene_inputs.py'),
             Path('decision/vlm/trajectory_inputs.py')]
    paths += [Path('decision/vlm/annotation') / name for name in FILES]
    guide = (ROOT / 'docs/decision/技术实现-VLM局部决策.md').read_text(encoding='utf-8')
    guide = '# T1、T2 决策任务标注\n\n' + guide.split('## 决策任务标注\n', 1)[1]
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(ROOT / path, str(path))
        archive.writestr('使用说明.md', guide)
    return len(paths) + 1
