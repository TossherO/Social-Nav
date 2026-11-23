import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from perception.environment.context import fingerprint
from scripts.field_video.reference import load_references


class FieldVideoReferenceTests(unittest.TestCase):
    def setUp(self):
        self.route = dict(t=10., floor="floor2", points=[(3., 4.), (5., 6.)], selected_pose=(5., 6., 1.),
                          waypoint_index=0, handle=dict(task_id="task", call_id="call", goal_revision=2))
        self.association = dict(valid=True, map_id="map2", map_frame="map", map_revision=0,
                                association_session="association", association_generation=0)
        self.pose = dict(t=10., floor="floor2", x=0., y=0., yaw=0., local_identity=["session", 0, "odom"],
                         map_binding=self.association)
        self.data = dict(capture_start=9., routes=[self.route], poses=[self.pose],
                         maps={"floor2": dict(frame_id="map", map_id="map2", snapshot_id="static",
                                              layers_sha256="layers")})

    def test_offline_reference_uses_current_leg_only_and_clears_on_blockage(self):
        next_leg = dict(self.route, t=13., waypoint_index=1)
        self.data["routes"] += [dict(self.route, t=11., blocked=True), next_leg, dict(t=15., floor="floor2", points=[])]
        self.data["poses"].append(dict(self.pose, t=13., x=3., y=4.))
        with tempfile.TemporaryDirectory() as demo, patch("scripts.field_video.reference.StaticReference.plan") as plan, \
                patch.dict("os.environ", {"SOCIAL_NAV_PYTHON": ""}):
            plan.side_effect = [[(0., 0.), (3., 4.)], [(3., 4.), (5., 6.)]]
            references, failures = load_references(Path(demo), self.data, 16.)

        self.assertEqual([(ref["t"], ref["until"], ref["source"]) for ref in references],
                         [(10., 11., "offline"), (13., 15., "offline")])
        self.assertEqual([call.args[0]["waypoint_index"] for call in plan.call_args_list], [0, 1])
        self.assertFalse(failures)

    def test_archived_reference_requires_matching_goal_and_is_cleared_by_next_cycle(self):
        for goal in ([3., 4.], [5., 6.]):
            with self.subTest(goal=goal), tempfile.TemporaryDirectory() as demo, \
                    patch.dict("os.environ", {"SOCIAL_NAV_PYTHON": ""}):
                request = dict(request_id="call:0", task_id="manual-preview:task:call", goal_revision=200,
                               goal=goal, mode="mapped", frame_id="map", target_reference="base_link",
                               map_snapshot_id="static")
                source = dict(clock_domain="runtime_live", as_of=10.02, association=self.association,
                              local_session="session", local_generation=0,
                              static_map=dict(snapshot_id="static", layers_sha256="layers"))
                result = dict(status="path_found", current_path_valid=True, frame_id="map",
                              input_binding=source, request=request, path=[[0., 0., 0.], [*goal, 0.]])
                result["result_binding"] = fingerprint(result)
                envelope = dict(result=result, as_of=10.02, clock_domain="runtime_live", context_binding="context")
                binding = envelope["binding"] = fingerprint(envelope)
                planning = Path(demo) / "run/planning"
                replies = planning / "replies"
                replies.mkdir(parents=True)
                (replies / "1.json").write_text(json.dumps(dict(result=envelope)))
                cycles = [dict(stamp_s=10.1, dt=.1, reference_binding=binding),
                          dict(stamp_s=10.15, dt=.05, reference_binding=None)]
                (planning / "cycles.jsonl").write_text("\n".join(map(json.dumps, cycles)))
                references, failures = load_references(Path(demo), self.data, 12.)
                self.assertFalse(failures)
                if goal == [3., 4.]:
                    self.assertEqual(references, [dict(t=10.1, until=10.15, floor="floor2",
                                                       points=[(0., 0.), (3., 4.)], source="archived")])
                else:
                    self.assertEqual(references, [])
