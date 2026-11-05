"""Declared sensor-clock sessions and time-calibration applicability, without ROS."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

from perception.mapping_localization.results import file_fingerprint


def clock_group(value):
    if value is None:
        return None
    if not isinstance(value, str) or not value or any(c.isspace() for c in value):
        raise ValueError('Sensor clock group must be a nonempty identifier without whitespace')
    return value


def session_path(bag):
    return Path(str(bag) + '.clock.json')


def record_bag_session(bag, group, output=None):
    """The caller declares driver continuity; never infer it from a bag date."""
    bag = Path(bag).expanduser().resolve()
    group = clock_group(group)
    if group is None:
        raise ValueError('Recording a sensor session requires a clock group')
    record = dict(entity_type='sensor_clock_session', clock_group=group,
                  declaration='operator_declared', bag=str(bag), bag_fingerprint=file_fingerprint(bag))
    target = Path(output).expanduser() if output is not None else session_path(bag)
    # Do not follow or replace an existing sidecar, including a dangling link.
    with target.open('x') as stream:
        stream.write(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    return record


def time_calibration_report(config, *, bag=None, group=None, session_file=None, previous=None, source_file=None):
    if not isinstance(config, dict):
        raise ValueError('Time calibration configuration must be a mapping')
    common = config.get('common', {})
    calibration = config.get('calibration') or {}
    if not isinstance(common, dict) or not isinstance(calibration, dict):
        raise ValueError('Time calibration common/calibration entries must be mappings')
    offset = common.get('imu_time_offset', 0.)
    if type(offset) not in (int, float) or not math.isfinite(offset):
        raise ValueError('common.imu_time_offset must be a finite number in seconds')
    heuristic = common.get('time_sync_en', False)
    if type(heuristic) is not bool:
        raise ValueError('common.time_sync_en must be boolean')
    calibrated = clock_group(calibration.get('deployment_clock_group'))
    selected = clock_group(group)
    origin = 'argument' if selected is not None else 'unknown'
    fingerprint = None if bag is None else file_fingerprint(bag)
    session = None
    session_bytes = None
    if session_file is not None and bag is None:
        raise ValueError('--sensor-session is a bag-bound record and cannot be used with live input')
    path = Path(session_file).expanduser() if session_file is not None else session_path(bag) if bag is not None else None
    if path is not None and (session_file is not None or path.exists() or path.is_symlink()):
        session_bytes = path.read_bytes()
        session = json.loads(session_bytes)
        if (not isinstance(session, dict) or session.get('entity_type') != 'sensor_clock_session' or
                session.get('declaration') != 'operator_declared' or session.get('bag_fingerprint') != fingerprint):
            raise ValueError('Sensor session record does not identify this bag: ' + str(path))
        recorded = clock_group(session.get('clock_group'))
        if recorded is None:
            raise ValueError('Sensor session record has no clock group: ' + str(path))
        if selected is not None and selected != recorded:
            raise ValueError('--sensor-clock-group conflicts with the bag sensor session record')
        selected, origin = recorded, 'bag_session_record'
    # Only an explicit input declaration bound to this same bag can be reused.
    # In particular, calibration.deployment_clock_group is never an input identity.
    if selected is None and bag is not None and isinstance(previous, dict):
        if previous.get('input_bag_fingerprint') == fingerprint:
            selected = clock_group(previous.get('input_clock_group'))
            if selected is not None:
                origin = 'source_run'
    matched = None if selected is None or calibrated is None else selected == calibrated
    issues = []
    if matched is None:
        issues.append('Sensor clock session applicability is unconfirmed; no driver continuity is inferred.')
    if 'imu_time_offset' not in common:
        issues.append('No IMU time offset is configured; the input reader uses its zero default.')
    if heuristic:
        issues.append('Heuristic time synchronization is requested; this check does not verify its effect.')
    source = None
    if source_file is not None:
        source_path = Path(source_file).expanduser().resolve()
        source = dict(path=str(source_path), sha256=hashlib.sha256(source_path.read_bytes()).hexdigest())
    return dict(offset_s=offset, offset_configured='imu_time_offset' in common, heuristic_sync_requested=heuristic,
                calibration_clock_group=calibrated, input_clock_group=selected,
                input_clock_group_source=origin, input_bag_fingerprint=fingerprint,
                session_record=None if session is None else dict(path=str(path.resolve()),
                    sha256=hashlib.sha256(session_bytes).hexdigest()),
                clock_group_match=matched, calibration_source=source,
                declared_calibration_status=calibration.get('status'),
                declared_validation_scope=calibration.get('validation_scope'),
                time_offset_validity=calibration.get('time_offset_validity'),
                validation_performed=False, issues=issues)


def check_time_calibration(config, **options):
    report = time_calibration_report(config, **options)
    state = 'unknown' if report['clock_group_match'] is None else 'matched' if report['clock_group_match'] else 'mismatch'
    display = dict(offset_ms=report['offset_s'] * 1000, input_clock_group=report['input_clock_group'],
                   calibration_clock_group=report['calibration_clock_group'], scope=state,
                   heuristic_sync_requested=report['heuristic_sync_requested'],
                   declared_status=report['declared_calibration_status'],
                   validation='not_performed_by_startup_check', calibration_source=report['calibration_source'])
    print('[IMU time calibration] ' + json.dumps(display, ensure_ascii=False, allow_nan=False), file=sys.stderr)
    if report['clock_group_match'] is False:
        raise ValueError('IMU time calibration clock group mismatch: input=' + report['input_clock_group'] +
                         ', calibration=' + report['calibration_clock_group'] +
                         '; select a calibration for this sensor session')
    for message in report['issues']:
        print('WARNING: ' + message, file=sys.stderr)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bag', required=True)
    parser.add_argument('--clock-group', required=True,
                        help='Operator-declared continuous sensor/driver session; reuse across bags until a restart')
    parser.add_argument('--out', help='New session JSON; defaults to BAG.clock.json, never overwritten')
    args = parser.parse_args()
    record = record_bag_session(args.bag, args.clock_group, args.out)
    print(json.dumps(dict(clock_group=record['clock_group'], declaration=record['declaration'],
                         output=str(Path(args.out) if args.out else session_path(Path(args.bag).expanduser().resolve())))))


if __name__ == '__main__':
    main()
