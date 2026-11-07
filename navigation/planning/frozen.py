"""Plan once at an immutable bag observation time, with no execution outlet."""
from navigation.control.shadow import observed_feedback
from .continuity import reference_envelope
from .planner import Planner


def preview(request, scene, rows, controller, planning, *, asset=None, dt=.1):
    now = scene.record['as_of']
    clock = scene.record['clock_domain']
    if clock != 'runtime_bag':
        raise ValueError('Frozen planning is restricted to bag observations')
    scene.verify()
    planner = Planner(planning, controller.geometry, require_actual_start=True)
    result = planner.query(request, scene=scene, asset=asset, now=now, clock_domain=clock)
    reference = reference_envelope(result, request, scene, asset)
    feedback = observed_feedback(rows, scene, controller.geometry, controller.motion, now=now)
    controller.clear()
    proposal = controller.command(request, scene, feedback, reference, now=now, clock_domain=clock, dt=dt)
    trajectory = None if controller.result is None else controller.result.get('trajectory')
    return dict(scope='frozen_bag_observation', executable=False, execution_feedback=False,
                task_completed=False, as_of=now, scene_binding=scene.record['input_fingerprint'],
                request=request, reference=reference, proposal=proposal, feedback=feedback,
                trajectory=trajectory, reason=result.get('reason') or proposal.get('reason'),
                reference_available=result['status']=='path_found', generated=proposal.get('status')=='candidate')
