"""Downsample CODa OS1 point clouds from 128 vertical channels to 16.

CODa stores each compensated OS1 frame as 128 x 1024 points, with four
float32 features per point: ``[x, y, z, intensity]``.  The output preserves
the original filenames and directory layout so that ``generate_coda_info.py``
can index the downsampled stream without any special-case handling.

Example:
    python tools/downsample_coda_lidar.py \
        --data_root data/coda_raw
"""

import argparse
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = str(PROJECT_ROOT / 'data/coda_raw')
DEFAULT_INPUT_DIR = '3d_comp/os1'
DEFAULT_OUTPUT_DIR = str(PROJECT_ROOT / 'output/data_preparation/coda_raw/3d_comp_downsample/os1')
NUM_FEATURES = 4


def parse_args():
    parser = argparse.ArgumentParser(
        description='Downsample CODa OS1 point clouds by vertical channel',
    )
    parser.add_argument('--data_root', type=str, default=DEFAULT_DATA_ROOT,
                        help='Path to the CODa dataset root')
    parser.add_argument('--input_dir', type=str, default=DEFAULT_INPUT_DIR,
                        help='Input directory, absolute or relative to data_root')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='Output directory, absolute or relative to data_root')
    parser.add_argument('--input_channels', type=int, default=128,
                        help='Number of vertical channels in each input frame')
    parser.add_argument('--output_channels', type=int, default=16,
                        help='Number of vertical channels to retain')
    parser.add_argument('--columns', type=int, default=1024,
                        help='Number of horizontal samples per channel')
    parser.add_argument('--seq_ids', type=int, nargs='+', default=None,
                        help='Sequence IDs to process (default: all sequences)')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel worker processes')
    parser.add_argument('--overwrite', action='store_true',
                        help='Replace existing outputs instead of resuming')
    parser.add_argument('--limit', type=int, default=None,
                        help='Process at most this many files (for smoke tests)')
    return parser.parse_args()


def resolve_path(data_root, value):
    """Resolve a CLI path relative to the dataset root."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = data_root / path
    return path.resolve()


def select_channel_indices(input_channels=128, output_channels=16):
    """Select uniformly spaced source channels over the full vertical FoV."""
    if input_channels <= 0:
        raise ValueError('input_channels must be positive')
    if output_channels <= 0:
        raise ValueError('output_channels must be positive')
    if output_channels > input_channels:
        raise ValueError('output_channels cannot exceed input_channels')

    if output_channels == 1:
        return np.array([input_channels // 2], dtype=np.int64)

    indices = np.rint(
        np.linspace(0, input_channels - 1, output_channels)
    ).astype(np.int64)
    if len(np.unique(indices)) != output_channels:
        raise ValueError('channel selection produced duplicate indices')
    return indices


def downsample_points(points, input_channels=128, output_channels=16,
                      columns=1024):
    """Return a channel-wise downsampled ``(N, 4)`` point array."""
    points = np.asarray(points)
    expected_shape = (input_channels * columns, NUM_FEATURES)
    if points.shape != expected_shape:
        raise ValueError(
            f'expected point shape {expected_shape}, got {points.shape}'
        )
    if points.dtype != np.float32:
        raise ValueError(f'expected float32 points, got {points.dtype}')
    if not np.all(np.isfinite(points)):
        raise ValueError('point cloud contains NaN or Inf')

    channel_indices = select_channel_indices(input_channels, output_channels)
    frame = points.reshape(input_channels, columns, NUM_FEATURES)
    return np.ascontiguousarray(frame[channel_indices].reshape(-1, NUM_FEATURES))


def downsample_file(input_path, output_path, input_channels=128,
                    output_channels=16, columns=1024, overwrite=False):
    """Downsample one frame and atomically install its output file."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    expected_output_size = (
        output_channels * columns * NUM_FEATURES * np.dtype(np.float32).itemsize
    )

    if output_path.exists() and not overwrite:
        if output_path.stat().st_size == expected_output_size:
            return 'skipped'
        raise FileExistsError(
            f'existing output has an invalid size: {output_path}; '
            'use --overwrite to replace it'
        )

    values = np.fromfile(input_path, dtype=np.float32)
    expected_values = input_channels * columns * NUM_FEATURES
    if values.size != expected_values:
        raise ValueError(
            f'{input_path} contains {values.size} float32 values; '
            f'expected {expected_values}'
        )

    sampled = downsample_points(
        values.reshape(-1, NUM_FEATURES),
        input_channels=input_channels,
        output_channels=output_channels,
        columns=columns,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(
        f'.{output_path.name}.{os.getpid()}.tmp'
    )
    try:
        sampled.tofile(temp_path)
        if temp_path.stat().st_size != expected_output_size:
            raise IOError(f'incomplete output written to {temp_path}')
        os.replace(temp_path, output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    return 'written'


def _process_task(task):
    input_path, output_path, input_channels, output_channels, columns, overwrite = task
    try:
        status = downsample_file(
            input_path=input_path,
            output_path=output_path,
            input_channels=input_channels,
            output_channels=output_channels,
            columns=columns,
            overwrite=overwrite,
        )
        return status, str(input_path), ''
    except Exception as error:
        return 'failed', str(input_path), str(error)


def discover_sequences(input_dir, seq_ids=None):
    if not input_dir.is_dir():
        raise FileNotFoundError(f'input directory not found: {input_dir}')

    if seq_ids is not None:
        sequence_dirs = [input_dir / str(seq_id) for seq_id in seq_ids]
        missing = [path for path in sequence_dirs if not path.is_dir()]
        if missing:
            raise FileNotFoundError(
                'sequence directories not found: '
                + ', '.join(str(path) for path in missing)
            )
        return sequence_dirs

    sequence_dirs = [path for path in input_dir.iterdir() if path.is_dir()]
    try:
        return sorted(sequence_dirs, key=lambda path: int(path.name))
    except ValueError as error:
        raise ValueError(
            f'non-numeric sequence directory found below {input_dir}'
        ) from error


def build_tasks(input_dir, output_dir, sequence_dirs, input_channels,
                output_channels, columns, overwrite, limit=None):
    tasks = []
    for sequence_dir in sequence_dirs:
        output_sequence_dir = output_dir / sequence_dir.name
        for input_path in sorted(sequence_dir.glob('*.bin')):
            tasks.append((
                input_path,
                output_sequence_dir / input_path.name,
                input_channels,
                output_channels,
                columns,
                overwrite,
            ))
            if limit is not None and len(tasks) >= limit:
                return tasks
    return tasks


def main():
    args = parse_args()
    if args.workers <= 0:
        raise ValueError('workers must be positive')
    if args.limit is not None and args.limit <= 0:
        raise ValueError('limit must be positive')

    data_root = Path(args.data_root).expanduser().resolve()
    input_dir = resolve_path(data_root, args.input_dir)
    output_dir = resolve_path(data_root, args.output_dir)
    if input_dir == output_dir:
        raise ValueError('input_dir and output_dir must be different')

    channel_indices = select_channel_indices(
        args.input_channels, args.output_channels,
    )
    sequence_dirs = discover_sequences(input_dir, args.seq_ids)
    tasks = build_tasks(
        input_dir=input_dir,
        output_dir=output_dir,
        sequence_dirs=sequence_dirs,
        input_channels=args.input_channels,
        output_channels=args.output_channels,
        columns=args.columns,
        overwrite=args.overwrite,
        limit=args.limit,
    )
    if not tasks:
        raise RuntimeError(f'no .bin files found below {input_dir}')

    expected_output_points = args.output_channels * args.columns
    print(f'Input directory:  {input_dir}')
    print(f'Output directory: {output_dir}')
    print(f'Sequences:        {[path.name for path in sequence_dirs]}')
    print(f'Files:            {len(tasks)}')
    print(f'Channels:         {channel_indices.tolist()}')
    print(f'Output points:    {expected_output_points} per frame')
    print(f'Workers:          {args.workers}')

    results = Counter()
    failures = []
    if args.workers == 1:
        result_iter = map(_process_task, tasks)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=args.workers)
        result_iter = executor.map(_process_task, tasks, chunksize=16)

    try:
        for index, (status, input_path, message) in enumerate(result_iter, start=1):
            results[status] += 1
            if status == 'failed':
                failures.append((input_path, message))
                print(f'ERROR: {input_path}: {message}')
            if index % 500 == 0 or index == len(tasks):
                print(
                    f'Progress: {index}/{len(tasks)} '
                    f'(written={results["written"]}, '
                    f'skipped={results["skipped"]}, '
                    f'failed={results["failed"]})'
                )
    finally:
        if executor is not None:
            executor.shutdown()

    print('Downsampling complete:')
    print(f'  written: {results["written"]}')
    print(f'  skipped: {results["skipped"]}')
    print(f'  failed:  {results["failed"]}')
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
