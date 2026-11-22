"""Local T2 appearance matching against an immutable, agent-selected person crop."""

from collections import deque
from dataclasses import dataclass
import hashlib
import io
import math
from pathlib import Path

import numpy as np
from PIL import Image

from interfaces.runtime import source_stamp_ns


def _osnet_pixels(crops):
    # Torchreid evaluation uses RGB, 256x128 (H,W), and ImageNet normalization.
    images = np.stack([np.asarray(crop.resize((128, 256), Image.Resampling.BILINEAR))
                       for crop in crops]).astype(np.float32) / np.float32(255)
    images -= np.array([.485, .456, .406], np.float32)
    images /= np.array([.229, .224, .225], np.float32)
    return np.ascontiguousarray(images.transpose(0, 3, 1, 2))


class OsnetEmbedder:
    """OSNet x0.5 MSMT17 with an explicitly selected ONNX or TensorRT asset."""

    def __init__(self, path, *, sha256=None, device=None):
        self.path = Path(path).resolve()
        model = self.path.read_bytes()
        self.sha256 = hashlib.sha256(model).hexdigest()
        if sha256 is not None and self.sha256 != sha256:
            raise ValueError('OSNet model SHA256 mismatch')
        if self.path.suffix == '.engine':
            from .reid_trt import TensorRtOsnet

            self.engine = TensorRtOsnet(self.path, model, self.sha256, device)
            self.backend = 'tensorrt_cuda'
            self.onnx_sha256 = self.engine.onnx_sha256
            self.device = str(self.engine.device)
            self.tensorrt_version = self.engine.tensorrt_version
            return
        if self.path.suffix != '.onnx':
            raise ValueError('OSNet requires an .onnx or .engine model')

        import cv2

        self.engine = None
        self.backend = 'opencv_cpu'
        self.onnx_sha256 = self.sha256
        self.device = 'cpu'
        self.tensorrt_version = None
        self.net = cv2.dnn.readNetFromONNX(model)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def __call__(self, crops):
        if not len(crops):
            return np.empty((0, 512), dtype=np.float32)
        if self.engine is not None:
            capacity = self.engine.max_batch_size
            return np.concatenate([
                self.engine(_osnet_pixels(crops[start:start + capacity]))
                for start in range(0, len(crops), capacity)
            ])
        self.net.setInput(_osnet_pixels(crops))
        features = self.net.forward()
        if features.shape != (len(crops), 512):
            raise ValueError('Unexpected OSNet batch output shape')
        return features


@dataclass(frozen=True)
class ReidPolicy:
    min_score: float = .65
    min_margin: float = .08
    min_reference_score: float = .55
    confirmation_frames: int = 2
    max_gap_s: float = 1.
    min_box_width_px: int = 16
    min_box_height_px: int = 32
    min_detection_score: float = .35
    gallery_size: int = 3
    gallery_min_reference_score: float = .82
    gallery_min_margin: float = .15
    gallery_min_detection_score: float = .65
    gallery_update_frames: int = 3
    gallery_interval_s: float = .5
    position_slack_m: float = .65
    max_person_speed_mps: float = 2.

    def __post_init__(self):
        probabilities = (self.min_score, self.min_margin, self.min_reference_score,
                         self.min_detection_score, self.gallery_min_reference_score,
                         self.gallery_min_margin, self.gallery_min_detection_score)
        if (any(not 0 < value <= 1 for value in probabilities) or
                not self.min_reference_score <= self.min_score <= self.gallery_min_reference_score or
                not self.min_margin <= self.gallery_min_margin or
                not self.min_detection_score <= self.gallery_min_detection_score):
            raise ValueError('Invalid ReID thresholds')
        counts = (self.confirmation_frames, self.gallery_size, self.gallery_update_frames,
                  self.min_box_width_px, self.min_box_height_px)
        if (any(type(value) is not int or value < 1 for value in counts) or
                self.confirmation_frames < 2 or self.gallery_update_frames < self.confirmation_frames or
                self.gallery_size > 8):
            raise ValueError('Invalid ReID confirmation or gallery size')
        limits = (self.max_gap_s, self.gallery_interval_s, self.position_slack_m, self.max_person_speed_mps)
        if any(not math.isfinite(value) or value <= 0 for value in limits):
            raise ValueError('Invalid ReID time or position limits')


def _normalized(encoder, crops):
    features = np.asarray(encoder(crops), dtype=np.float32)
    if features.ndim != 2 or features.shape[0] != len(crops) or features.shape[1] == 0:
        raise ValueError('Invalid ReID embedding batch shape')
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or (norms < 1e-8).any():
        raise ValueError('Invalid ReID feature values')
    return features / norms


class FollowReid:
    """One task's visual reference and sequential confirmation state, owned by the T2 worker."""

    def __init__(self, encoder, reference_crop, reference, *, policy=ReidPolicy()):
        if hashlib.sha256(reference_crop).hexdigest() != reference['crop_sha256']:
            raise ValueError('ReID needs the original selected crop')
        with Image.open(io.BytesIO(reference_crop)) as raw:
            crop = raw.convert('RGB')
        if crop.width < policy.min_box_width_px or crop.height < policy.min_box_height_px:
            raise ValueError('Selected crop is too small for OSNet')
        self.encoder = encoder
        self.policy = policy
        self.anchor = _normalized(encoder, [crop])[0]
        self.anchor.setflags(write=False)
        self.scope = (reference['clock_domain'], tuple(reference['local_identity']))
        self.last_stamp = source_stamp_ns(reference['rgb_stamp_s'], reference.get('rgb_stamp_ns'))
        self.gallery = deque(maxlen=policy.gallery_size)
        self.gallery_stamp = self.last_stamp
        self.gallery_run = 0
        self.active = None
        self.pending = None
        self.last_supported = None

    def interrupt(self):
        """Require fresh confirmation after missing input while keeping the trusted reference."""
        self.active = None
        self.pending = None
        self.gallery_run = 0

    def _plausible(self, position, stamp, previous):
        if previous is None:
            return True
        prior_position, prior_stamp = previous
        elapsed = (stamp - prior_stamp) * 1e-9
        return 0 < elapsed and math.dist(position, prior_position) <= (
            self.policy.position_slack_m + elapsed * self.policy.max_person_speed_mps)

    def step(self, pixels, candidate_record, *, track_positions):
        stamp = candidate_record['rgb_stamp_ns']
        if type(stamp) is not int or stamp <= 0:
            self.interrupt()
            raise ValueError('ReID requires a source RGB timestamp in nanoseconds')
        scope = (candidate_record['clock_domain'], tuple(candidate_record['local_identity']))
        result = dict(status='unconfirmed', reason=None, candidate_id=None, identity=None,
                      lifecycle_generation=None, best_candidate_id=None, second_candidate_id=None,
                      score=None, reference_score=None, margin=None, confirmation_count=0,
                      rgb_stamp_ns=stamp)
        if scope != self.scope:
            self.interrupt()
            return dict(result, reason='source_scope_changed')
        if stamp <= self.last_stamp:
            return dict(result, reason='source_not_new')
        if stamp - self.last_stamp > self.policy.max_gap_s * 1e9:
            self.interrupt()
        self.last_stamp = stamp
        try:
            return self._match(pixels, candidate_record, track_positions, result, stamp)
        except Exception:
            self.interrupt()
            raise

    def _match(self, pixels, candidate_record, track_positions, result, stamp):
        policy = self.policy
        candidates = candidate_record['candidates']
        association_available = candidate_record['association_status'] == 'complete'
        if not candidates:
            self.interrupt()
            return dict(result, reason='no_detections' if association_available else 'association_unavailable')
        image = Image.fromarray(pixels)
        if image.mode != 'RGB':
            raise ValueError('ReID requires RGB pixels')
        crops = [image.crop((math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2)))
                 for x1, y1, x2, y2 in (candidate['box'] for candidate in candidates)]
        features = _normalized(self.encoder, crops)
        anchor_scores = np.clip(features @ self.anchor, -1., 1.)
        scores = anchor_scores
        if self.gallery:
            # Gallery evidence can improve a match, with at least half of the score from the fixed crop.
            gallery_scores = np.clip((features @ np.asarray(self.gallery).T).max(axis=1), -1., 1.)
            scores = np.maximum(anchor_scores, (anchor_scores + gallery_scores) * .5)
        ranked = np.argsort(-scores, kind='stable')
        index = ranked[0]
        score = float(scores[index])
        reference_score = float(anchor_scores[index])
        margin = None if len(ranked) == 1 else score - float(scores[ranked[1]])
        selected = candidates[index]
        result.update(best_candidate_id=selected['id'],
                      second_candidate_id=None if len(ranked) == 1 else candidates[ranked[1]]['id'],
                      score=score, reference_score=reference_score, margin=margin)
        crop = crops[index]
        if (crop.width < policy.min_box_width_px or crop.height < policy.min_box_height_px or
                selected['score'] < policy.min_detection_score):
            self.interrupt()
            return dict(result, reason='appearance_low_quality')
        if score < policy.min_score:
            self.interrupt()
            return dict(result, reason='below_similarity')
        if margin is not None and margin < policy.min_margin:
            self.interrupt()
            return dict(result, reason='appearance_ambiguous')
        if reference_score < policy.min_reference_score:
            self.interrupt()
            return dict(result, reason='below_reference_similarity')
        track = selected.get('track')
        position = track_positions.get(selected['id'])
        if (not association_available or track is None or position is None or len(position) != 2 or
                not all(math.isfinite(value) for value in position)):
            self.interrupt()
            return dict(result, reason='appearance_without_track')
        position = tuple(float(value) for value in position)
        identity = tuple(track['identity'])
        lifecycle = track['lifecycle_generation']
        key = identity, lifecycle
        if not self._plausible(position, stamp, self.last_supported):
            self.interrupt()
            return dict(result, reason='position_inconsistent')

        status = 'same'
        count = policy.confirmation_frames
        if self.active != key:
            self.active = None
            self.gallery_run = 0
            pending = self.pending
            count = 1
            if pending is not None and pending[0] == key:
                if not self._plausible(position, stamp, pending[2]):
                    self.interrupt()
                    return dict(result, reason='position_inconsistent')
                count = pending[1] + 1
            self.pending = (key, count, (position, stamp))
            if count < policy.confirmation_frames:
                return dict(result, reason='awaiting_confirmation', confirmation_count=count)
            status = 'recovered'
        self.active = key
        self.pending = None
        self.last_supported = (position, stamp)

        # Keep only well-separated examples judged high-confidence against the immutable crop alone.
        reference_others = np.delete(anchor_scores, index)
        reference_margin = 2. if not len(reference_others) else reference_score - float(reference_others.max())
        high_quality = (reference_score >= policy.gallery_min_reference_score and
                        reference_margin >= policy.gallery_min_margin and
                        selected['score'] >= policy.gallery_min_detection_score)
        self.gallery_run = self.gallery_run + 1 if high_quality else 0
        if (self.gallery_run >= policy.gallery_update_frames and
                stamp - self.gallery_stamp >= policy.gallery_interval_s * 1e9):
            self.gallery.append(features[index].copy())
            self.gallery_stamp = stamp
        return dict(result, status=status, reason='confirmed', candidate_id=selected['id'],
                    identity=list(identity), lifecycle_generation=lifecycle, confirmation_count=count)
