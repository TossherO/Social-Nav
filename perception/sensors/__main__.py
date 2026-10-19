"""Manually start the selected sensor drivers; never starts chassis feedback."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--serial-front')
    parser.add_argument('--serial-rear')
    parser.add_argument('--lidar-only', action='store_true')
    parser.add_argument('--rear', action='store_true')
    parser.add_argument('--fps', type=int, choices=(15, 30), default=15)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    local = root / 'configs/perception.local.json'
    device = json.loads(local.read_text()).get('sensors', {}) if local.is_file() else {}
    front = args.serial_front or device.get('serial_front', '')
    rear = args.serial_rear or device.get('serial_rear', '')
    if not args.lidar_only and not front:
        parser.error('Select the front camera by --serial-front or configs/perception.local.json')
    if args.rear and (not rear or args.lidar_only):
        parser.error('--rear requires a rear serial and enabled cameras')
    command = ['roslaunch', 'wheelchair_sensors', 'sensors.launch',
               'serial_a:=' + front, 'serial_b:=' + rear,
               'camera_a:=' + str(not args.lidar_only).lower(),
               'camera_b:=' + str(args.rear).lower(), 'fps:=' + str(args.fps)]
    os.execvp(command[0], command)


if __name__ == '__main__':
    main()
