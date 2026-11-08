"""Local two-stage low-speed measurement; prints plan unless --execute is given."""
import hashlib
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "navigation/commissioning/measure_remaining.py"
spec = importlib.util.spec_from_file_location("local_existing_measure_remaining", SOURCE)
existing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(existing)

existing.STEPS = (
    ("forward_active_zero", dict(maneuver="forward", speed=.10, pulse_s=2.), None),
    ("reverse_active_zero", dict(maneuver="reverse", pulse_s=2.), None),
)
original_plan = existing.plan


def plan(names=None):
    result = original_plan(names)
    result.update(
        scope="profile_gaps_forward_reverse_observation_not_navigation",
        expected_elapsed_s=[30, 60],
        motor_command_limit="Forward <=0.10 m/s; reverse fixed at 0.05 m/s; 2s each, active zero after each.",
        preparation="Stationary, unoccupied, joystick centered, electric mode available, clear level front/back, physical stop supervision.",
        calibration_acceptance=False,
        source_sha256={
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__).resolve(), SOURCE,
                ROOT / "navigation/commissioning/measure_observer.py",
                ROOT / "navigation/control/ground_probe.py",
            )
        },
    )
    return result


existing.plan = plan

if __name__ == "__main__":
    raise SystemExit(existing.main())
