"""System-ROS bounded RGB extraction; no detection, localization or model run."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3


def extract(bag_path, requests, output):
    import cv2
    import rosbag
    import rospy
    from perception.sensors.image import rgb_array
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    db = sqlite3.connect(output)
    db.execute('CREATE TABLE images (seq INTEGER PRIMARY KEY, metadata TEXT, png BLOB)')
    with rosbag.Bag(str(bag_path), 'r') as bag:
        for request in requests:
            cutoff = request['stamp_s']
            latest = None
            for _, msg, _ in bag.read_messages(topics=['/camera/color/image_raw'],
                    start_time=rospy.Time.from_sec(cutoff-.7), end_time=rospy.Time.from_sec(cutoff+.05)):
                stamp = msg.header.stamp.to_sec()
                if stamp <= cutoff and (latest is None or stamp > latest.header.stamp.to_sec()):
                    latest = msg
            if latest is None or not 0 <= cutoff-latest.header.stamp.to_sec() <= .15:
                raise ValueError('Missing causal RGB for '+str(request['seq']))
            pixels = rgb_array(latest)
            ok, encoded = cv2.imencode('.png', pixels[:, :, ::-1])
            if not ok:
                raise ValueError('PNG encoding failed')
            metadata = dict(request, rgb_stamp_s=latest.header.stamp.to_sec(), frame_id=latest.header.frame_id,
                            topic='/camera/color/image_raw', bag=str(bag_path), pixel_sha256=hashlib.sha256(pixels.tobytes()).hexdigest())
            db.execute('INSERT INTO images VALUES (?,?,?)', (request['seq'], json.dumps(metadata), encoded.tobytes()))
    db.commit()
    db.close()
    print(json.dumps(dict(images=len(requests), source='past_RGB_only', detection_runs=0, localization_runs=0)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bag')
    p.add_argument('--requests')
    p.add_argument('--rgb-db', nargs='+', help='Bind previously extracted RGB to its existing source scenes; use project Python')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    if args.rgb_db:
        bind(args.rgb_db, args.output)
    elif args.bag and args.requests:
        extract(args.bag, json.loads(Path(args.requests).read_text()), args.output)
    else:
        p.error('Use --rgb-db, or both --bag and --requests')


def bind(paths, output):
    import io
    import numpy as np
    from PIL import Image
    from perception.environment.scene import SceneInputs
    from ..tools.observations import Observation, ObservationSet
    target = ObservationSet(output, create=True)
    count = 0
    try:
        for path in paths:
            source = sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro', uri=True)
            try:
                for metadata, png in source.execute('SELECT metadata,png FROM images ORDER BY seq'):
                    m = json.loads(metadata)
                    pixels = np.asarray(Image.open(io.BytesIO(png)).convert('RGB'))
                    if hashlib.sha256(pixels.tobytes()).hexdigest() != m['pixel_sha256']:
                        raise ValueError('Extracted RGB changed')
                    scene = SceneInputs.load(m['scene'])
                    if scene.record['as_of'] != m['stamp_s']:
                        raise ValueError('RGB pairing cutoff does not match source scene')
                    observation = Observation.from_scene(scene, rgb=dict(pixels=pixels, stamp_s=m['rgb_stamp_s'],
                        frame_id=m['frame_id'], source={k: m[k] for k in ('bag', 'topic', 'rgb_stamp_s')}),
                        source=dict(kind='causal_saved_scene_with_bag_rgb', scene=m['scene'], bag=m['bag'], offset_s=m['offset_s']))
                    target.add(observation)
                    count += 1
            finally:
                source.close()
    finally:
        target.close()
    print(json.dumps(dict(observations=count, model_calls=0, input_scope='derived_development_observations')))


if __name__ == '__main__':
    main()
