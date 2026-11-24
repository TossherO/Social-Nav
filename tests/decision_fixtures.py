"""Synthetic runtime observations and mocked DSPARK transport for decision tests."""
import hashlib
import json

import httpx
import numpy as np

from decision.vlm.observations import render_window
from perception.environment.context import fingerprint
from perception.environment.layers import load_config as environment_config
from perception.environment.scene import assemble_scene
from runtime.decision import DecisionBuffer
from tests.navigation_fixtures import grid
from tests.test_environment_scene import person
from tests.test_navigation_prediction import track


def observation(stamp, generation=0, *, stamp_ns=None):
    snapshot = grid(generation=generation)
    snapshot.metadata['stamp_s'] = stamp
    row = person(stamp)
    if stamp_ns is not None:
        row['stamp_ns'] = stamp_ns
    row.update(observation_id=str(round(stamp * 1e9)), pose_health='valid', pose_source='localization',
               pose_stamp_s=stamp, timestamp_source='sensor_header',
               world_from_lidar=snapshot.metadata['world_from_lidar'],
               localization_binding=dict(session_id='test', generation=generation, frame_id='odom', source='fast_lio'))
    row['tracks'] = [dict(track(4, 2, (.5, 0, 0), stamp=stamp), lifecycle_generation=7, state='observed')]
    row['covariance_order'] = ['x', 'y', 'z', 'dx', 'dy', 'dz', 'yaw', 'vx', 'vy', 'vz']
    row['input_fingerprint'] = fingerprint(row)
    arrays = dict(raw=np.array([[1., 0., 0., 3.], [2., 1., 1., 5.]], np.float32), rgb=np.full((48, 64, 3), 120, np.uint8))
    association = {k: row[k] for k in ('session_id', 'generation', 'sequence_id', 'clock_domain', 'observation_id', 'stamp_s')}
    association.update(perception_input_fingerprint=row['input_fingerprint'],
                       image_input=dict(raw_rgb_sha256=hashlib.sha256(arrays['rgb'].tobytes()).hexdigest(),
                                        rgb_stamp_ns=round((stamp - .02) * 1e9), image_size=[64, 48]),
                       result=dict(status='complete', tracks={'a/g7': dict(status='matched', rgb_box=[20, 5, 35, 40])}))
    scene = assemble_scene(snapshot, clock_domain='synthetic', settings=environment_config()['scene'],
                           people=row, stamp_ns=stamp_ns)
    return row, arrays, association, scene


def candidate_set(scene):
    record = scene.record
    return dict(instruction='在桌边就座', observation_id=record['observation_id'], stamp_s=record['as_of'],
                **{k: record[k] for k in ('frame_id', 'clock_domain', 'session_id', 'generation')}, revision=0,
                candidates=[dict(id=0, pose=[4., 2., 0.], motion='forward', geometry='unknown')])


def two_person_window(start=10., *, include_predicted=False, predict_last=False):
    buffer = DecisionBuffer(max_frames=16, slot_bytes=32768)
    try:
        for i in range(12):
            stamp = round(start + i / 10, 1)
            row, arrays, association, scene = observation(stamp)
            if predict_last and i == 11:
                row['tracks'][0].update(state='predicted', last_observed_stamp_s=round(stamp - .1, 1))
            row['tracks'].append(dict(track(4., 2.5, id='b', stamp=stamp), lifecycle_generation=7, state='observed'))
            row.pop('input_fingerprint')
            row['input_fingerprint'] = fingerprint(row)
            association['perception_input_fingerprint'] = row['input_fingerprint']
            association['result']['tracks']['b/g7'] = dict(status='matched', rgb_box=[40, 5, 55, 40])
            scene = assemble_scene(scene.geometry, clock_domain='synthetic', settings=environment_config()['scene'], people=row)
            buffer.push(row, arrays, association, scene=scene)
        lease = buffer.acquire()
        with buffer.consume(lease) as read:
            record, images = render_window(lease['entries'], lease['history'], read,
                                           extent=[-4, 4, -4, 4], view_binding={}, include_predicted=include_predicted)
        return scene, record, images
    finally:
        buffer.close()


def follow_search(scene, _context):
    result = dict(clock_domain=scene.record['clock_domain'], source_stamp_s=scene.record['as_of'],
                  scene_binding=scene.record['input_fingerprint'], candidates=[dict(id=0, kind='wait', pose=None)])
    result['binding'] = fingerprint(result)
    return result


def dspark_client(raw, *, ready=True, complete=True, model='qwen3.8-27b-fp8', finish='stop', reasoning=False,
                  tokens=None, gate=None):
    calls = []

    def respond(request):
        calls.append(request)
        if request.url.path == '/v1/models':
            return httpx.Response(200 if ready else 503, json={'data': [{'id': 'qwen3.8-27b-fp8'}]})
        if request.url.path == '/get_server_info':
            return httpx.Response(200, json={'server_args': {'speculative_algorithm': 'DSPARK'}})
        if gate is not None:
            gate.wait(3)
        delta = dict(content=raw)
        if reasoning:
            delta['reasoning_content'] = 'hidden'
        event = dict(model=model, choices=[dict(index=0, delta=delta, finish_reason=finish)])
        if tokens is not None:
            event['usage'] = dict(completion_tokens=tokens)
        body = 'data: ' + json.dumps(event) + '\n\n'
        if complete:
            body += 'data: [DONE]\n\n'
        return httpx.Response(200, text=body)

    transport = httpx.MockTransport(respond)
    return lambda **kwargs: httpx.Client(transport=transport, **kwargs), calls


def follow_client(selection, *, gate=None):
    if selection.startswith('P'):
        response = dict(target=selection, observation=None)
    else:
        response = dict(target=None, observation=selection)
    factory, _ = dspark_client(json.dumps(response), gate=gate)
    return factory
