#!/usr/bin/env python3
"""Unpack retained source/debs into ignored local build prefixes, without sudo."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile


ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / 'data/cache/perception'
OUTPUT = ROOT / 'outputs/perception'


def unpack(archive, destination, expected=None):
    if expected and hashlib.sha256(archive.read_bytes()).hexdigest() != expected:
        raise ValueError('Source archive differs from retained dependency record: ' + archive.name)
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as source:
        for member in source.getmembers():
            candidate = (destination / member.name).resolve()
            if candidate != destination.resolve() and destination.resolve() not in candidate.parents:
                raise ValueError('Archive path escapes source directory')
            if member.issym() or member.islnk():
                base = candidate.parent if member.issym() else destination
                target = (base / member.linkname).resolve()
                if target != destination.resolve() and destination.resolve() not in target.parents:
                    raise ValueError('Archive link escapes source directory')
        source.extractall(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=('interfaces', 'mapping', 'sensors', 'all'), default='all')
    args = parser.parse_args()
    if ROOT not in OUTPUT.resolve().parents:
        raise ValueError('Build output must resolve inside the project before writing dependencies')
    mapping = args.profile in ('mapping', 'all')
    record = json.loads((CACHE / 'roll_dependencies.json').read_text()) if mapping else None
    prefix = OUTPUT / 'dependencies/roll'
    if mapping and not (prefix / '.prepared').exists():
        prefix.mkdir(parents=True, exist_ok=True)
        for package in record['packages']:
            deb = CACHE / 'roll_debs' / package['file']
            if hashlib.sha256(deb.read_bytes()).hexdigest() != package['sha256']:
                raise ValueError('Retained dependency differs: ' + deb.name)
            subprocess.run(['dpkg-deb', '-x', str(deb), str(prefix)], check=True)
        ceres = prefix / 'usr/lib/cmake/Ceres/CeresTargets-none.cmake'
        ceres.write_text(ceres.read_text().replace('/usr/lib/x86_64-linux-gnu/libglog.so',
            str(prefix / 'usr/lib/x86_64-linux-gnu/libglog.so')))
        (prefix / '.prepared').write_text('Retained focal dependencies, extracted locally.\n')
    sources = OUTPUT / 'sources'
    if mapping and not (sources / 'gtsam-4.0.2/CMakeLists.txt').exists():
        unpack(CACHE / 'gtsam-4.0.2.tar.gz', sources, record['gtsam']['sha256'])
    if args.profile in ('sensors', 'all') and not (sources / 'librealsense/CMakeLists.txt').exists():
        unpack(CACHE / 'librealsense-2.54.2.tar.gz', sources / 'librealsense')
    workspace = OUTPUT / 'ros1_ws/src'
    workspace.mkdir(parents=True, exist_ok=True)
    packages = {
        'social_nav_localization': 'perception/mapping_localization/ros',
        'social_nav_people': 'perception/detection_tracking/ros',
        'roll': 'perception/mapping_localization/backends/roll_fast_lio/roll',
        'fast_lio': 'perception/mapping_localization/backends/roll_fast_lio/fastlio-loc',
        'rslidar_sdk': 'perception/sensors/drivers/rslidar_sdk',
        'realsense2_camera': 'perception/sensors/drivers/realsense-ros/realsense2_camera',
        'realsense2_description': 'perception/sensors/drivers/realsense-ros/realsense2_description',
        'wheelchair_sensors': 'perception/sensors/ros/wheelchair_sensors',
    }
    for name, relative in packages.items():
        link = workspace / name
        target = ROOT / relative
        if link.is_symlink():
            if link.resolve() != target.resolve():
                raise ValueError('Workspace link points elsewhere: ' + name)
        elif link.exists():
            raise ValueError('Refusing existing workspace source: ' + name)
        else:
            link.symlink_to(target)
    print('ROS workspace and retained dependencies prepared under outputs/perception.')


if __name__ == '__main__':
    main()
