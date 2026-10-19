"""Persistent ROS-interpreter child. stdin JSON requests; stdout length + ZIP, no pickle."""
import argparse
import json
import struct
import sys

from .data import FrameLoader
from .protocol import MAX_BUNDLE, pack_assets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', required=True)
    args = parser.parse_args()
    loader = FrameLoader(args.project, read_only=True)
    output = sys.stdout.buffer
    try:
        for line in sys.stdin.buffer:
            if len(line) > 4096:
                raise ValueError('资产请求过长')
            try:
                request = json.loads(line)
                frame, points, image, stamp, notice = loader.read(request['id'])
                payload = b'Z' + pack_assets(frame, points, image, stamp, notice, request.get('encoding', 'jpeg'))
            except Exception as error:
                payload = b'E' + json.dumps(dict(error=type(error).__name__, detail=str(error)), ensure_ascii=False).encode('utf-8')
            if len(payload) > MAX_BUNDLE + 1:
                raise ValueError('资产响应过长')
            output.write(struct.pack('!I', len(payload)) + payload)
            output.flush()
    finally:
        loader.close()


if __name__ == '__main__':
    main()
