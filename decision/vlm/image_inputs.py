"""Optional YOLO/3D association for VLM windows, outside perception/navigation."""
import copy
from pathlib import Path

import numpy as np
import yaml


ROOT=Path(__file__).resolve().parents[2]


def camera_calibration(path=None):
    from perception.detection_tracking.inference.rgb_cache import file_digest
    path=Path(path or ROOT/'perception/sensors/calibration/front_camera_lidar.yaml')
    return dict(yaml.safe_load(path.read_text())['lidar_to_camera_optical'],sha256=file_digest(path))


class ImagePreparer:
    """Created lazily by the optional decision worker, never the core scan loop."""
    def __init__(self,policy=None,calibration=None):
        import torch
        from perception.detection_tracking.inference.config import load_config,project_path
        from perception.detection_tracking.inference.image_detector import ImageDetector
        self.policy=copy.deepcopy(policy if policy is not None else load_config()['image_association'])
        for key in ('engine','weights'):self.policy[key]=str(project_path(self.policy[key]))
        self.calibration=camera_calibration(calibration)
        self.transform=np.asarray(self.calibration['matrix'])
        self.stream=torch.cuda.Stream(device=0);self.calls=0
        with torch.cuda.stream(self.stream):
            self.detector=ImageDetector(self.policy,'0')
            self.detector.detect(np.zeros((640,640,3),np.uint8))
        self.metadata=dict(detector=self.detector.metadata,calibration=self.calibration,
            policy=self.policy,scope='optional_vlm_image_preparation_not_navigation_input')

    def prepare(self,entries,history,read_arrays):
        import torch
        from perception.detection_tracking.inference.rgb_association import ImageAssociator,people_in_lidar,causal_geometry
        from perception.detection_tracking.inference.rgb_recording import observation_record
        from .trajectory_inputs import local_scope
        matcher=ImageAssociator(self.policy);result=[]
        past=[h['result'] for h in history]
        with torch.cuda.stream(self.stream):
            for entry in entries:
                if entry['association'] is not None:result.append(entry);continue
                arrays=read_arrays(entry['buffer']);row=entry['people'];image=entry['image_metadata']
                pixels=arrays['rgb'];rgb_ns=image['rgb_stamp_ns'];rgb_stamp=rgb_ns/1e9
                if (str(image.get('rgb_frame') or '').lstrip('/')!=self.calibration['target_frame'].lstrip('/') or
                        row['detection_frame_id'].lstrip('/')!=self.calibration['source_frame'].lstrip('/')):
                    raise ValueError('VLM image calibration does not match the observed camera/LiDAR frames')
                detections,elapsed=self.detector.detect(pixels);self.calls+=1
                people=causal_geometry(people_in_lidar(row),row,past,rgb_stamp,.05)
                association=matcher.step(people,detections,self.transform,image.get('camera_info'),
                    row['stamp_s']-rgb_stamp,row['stamp_s'],local_scope(row),health=row['health'])
                record=observation_record(row,pixels=pixels,rgb_stamp_ns=rgb_ns,camera_info=image.get('camera_info'),
                    calibration=self.calibration,detector=self.detector.metadata,detections=detections,
                    result=association,detector_s=elapsed)
                result.append(dict(entry,association=record))
        return tuple(result)
