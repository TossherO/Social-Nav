"""Build a portable source client from a fixed allowlist, never recurse into configs/data."""
from pathlib import Path
import re
import zipfile

from .data import safe_directory


ROOT = Path(__file__).resolve().parents[3]
FILES = [
    'perception/__init__.py', 'perception/detection_tracking/__init__.py',
    'perception/detection_tracking/inference/__init__.py', 'perception/detection_tracking/inference/config.py',
    'perception/sensors/__init__.py', 'perception/sensors/pointcloud.py',
]
FILES += ['perception/detection_tracking/annotation/' + name for name in (
    '__init__.py', '__main__.py', 'app.py', 'appearance.py', 'backend.py', 'cache.py', 'cloud_view.py',
    'data.py', 'protocol.py', 'remote.py', 'timeline.py', 'requirements-client.txt')]


def section(path, heading):
    lines = (ROOT / path).read_text(encoding='utf-8').splitlines()
    start = lines.index(heading)
    depth = len(heading.split(' ', 1)[0])
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith('#') and
                len(lines[i].split(' ', 1)[0]) <= depth), len(lines))
    return '\n'.join(lines[start:end]).strip()


def package_client(output):
    output = Path(output)
    safe_directory(output.parent).mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream:
        with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for name in FILES:
                archive.write(ROOT / name, name)
            guide = '# Windows远程标注客户端\n\n由项目当前操作手册生成；服务器需先完成工程登记并启动。\n\n'
            guide += section('docs/感知导航运行手册.md', '### Windows客户端')
            guide += '\n\n' + section('docs/perception/技术实现-检测训练.md', '### 远程标注与同步') + '\n'
            # The portable bundle does not ship repository documentation.
            guide = re.sub(r'\[([^\]]+)\]\((?!https?://)[^)]+\)', r'\1', guide)
            archive.writestr('使用说明.md', guide.encode('utf-8'))
    return str(output)
