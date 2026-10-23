"""Distance diagnostics for already matched, center-only detections."""
import numpy as np


class DistanceCenterMetrics:
    """Aggregate global matches; partial annotations never supply negatives."""

    def __init__(self, config, match_threshold):
        self.edges = np.asarray(config['EDGES_M'], dtype=np.float64)
        self.weights = np.asarray(config['WEIGHTS'], dtype=np.float64)
        if (self.edges.ndim != 1 or not len(self.edges) or self.edges[0] != 0
                or not np.isfinite(self.edges).all() or np.any(np.diff(self.edges) <= 0)):
            raise ValueError('DISTANCE_EVAL.EDGES_M must start at zero and strictly increase')
        if (self.weights.shape != self.edges.shape or not np.isfinite(self.weights).all()
                or np.any(self.weights <= 0) or np.any(np.diff(self.weights) > 0)):
            raise ValueError('DISTANCE_EVAL.WEIGHTS must be positive, nonincreasing, one per bin')
        self.threshold = float(match_threshold)
        self.counts = {name: np.zeros(len(self.edges), dtype=np.int64)
                       for name in ('known_gt', 'known_tp', 'gt', 'tp', 'fp', 'empty_fp')}
        self.complete_frames = self.empty_frames = 0

    def _bins(self, xy):
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        return np.searchsorted(self.edges[1:], np.linalg.norm(xy, axis=1), side='right')

    def add_frame(self, gt_xy, pred_xy, matches, *, complete, empty):
        gt_bins, pred_bins = self._bins(gt_xy), self._bins(pred_xy)
        matched_gt = np.asarray([match[2] for match in matches], dtype=np.int64)
        matched_pred = np.asarray([match[1] for match in matches], dtype=np.int64)
        size = len(self.edges)
        self.counts['known_gt'] += np.bincount(gt_bins, minlength=size)
        self.counts['known_tp'] += np.bincount(gt_bins[matched_gt], minlength=size)
        if not complete:
            return
        self.complete_frames += 1
        self.counts['gt'] += np.bincount(gt_bins, minlength=size)
        self.counts['tp'] += np.bincount(gt_bins[matched_gt], minlength=size)
        unmatched = np.ones(len(pred_bins), dtype=bool)
        unmatched[matched_pred] = False
        self.counts['fp'] += np.bincount(pred_bins[unmatched], minlength=size)
        if empty:
            self.empty_frames += 1
            self.counts['empty_fp'] += np.bincount(pred_bins, minlength=size)

    @staticmethod
    def _rate(numerator, denominator):
        return float(numerator / denominator) if denominator else float('nan')

    def result(self):
        metrics, lines = {}, []
        suffix = f'@{self.threshold:.2f}m'
        for index, low in enumerate(self.edges):
            high = f'{self.edges[index + 1]:g}' if index + 1 < len(self.edges) else 'inf'
            name = f'{low:g}-{high}m'
            prefix = 'Distance/' + name + '/'
            known_gt, known_tp, gt, tp, fp, empty_fp = (
                int(self.counts[key][index])
                for key in ('known_gt', 'known_tp', 'gt', 'tp', 'fp', 'empty_fp'))
            metrics[prefix + 'LabeledCenterCount'] = known_gt
            metrics[prefix + 'CenterHit' + suffix] = self._rate(known_tp, known_gt)
            if self.complete_frames:
                metrics.update({
                    prefix + 'ExhaustiveCenterCount': gt,
                    prefix + 'CenterTruePositiveCount' + suffix: tp,
                    prefix + 'CenterMissedCount' + suffix: gt - tp,
                    prefix + 'CenterRecall' + suffix: self._rate(tp, gt),
                    prefix + 'CenterFalsePositiveCount' + suffix: fp,
                    prefix + 'CenterFalsePositivesPerFrame': fp / self.complete_frames,
                    prefix + 'EmptyFalsePositiveCount': empty_fp,
                    prefix + 'EmptyFalsePositivesPerFrame': self._rate(empty_fp, self.empty_frames),
                })
                lines.append(f'  Range {name}: GT={gt}, TP={tp}, FN={gt-tp}, FP={fp}, '
                             f'Recall={self._rate(tp, gt):.4f}, FP/frame={fp/self.complete_frames:.4f}')
            else:
                lines.append(f'  Range {name}: known centers={known_gt}, '
                             f'Hit={self._rate(known_tp, known_gt):.4f} (positive-only)')
        if self.complete_frames:
            tp = float(self.counts['tp'] @ self.weights)
            fn = float((self.counts['gt'] - self.counts['tp']) @ self.weights)
            fp = float(self.counts['fp'] @ self.weights)
            f1 = 2 * tp / (2 * tp + fn + fp) if tp + fn else float('nan')
            metrics.update({
                'DistanceWeightedTruePositiveCount' + suffix: tp,
                'DistanceWeightedMissedCount' + suffix: fn,
                'DistanceWeightedFalsePositiveCount' + suffix: fp,
                'DistanceWeightedPrecision' + suffix: self._rate(tp, tp + fp),
                'DistanceWeightedRecall' + suffix: self._rate(tp, tp + fn),
                'DistanceWeightedF1' + suffix: f1,
                'DistanceWeightedFalsePositivesPerFrame': fp / self.complete_frames,
            })
            lines.append(f'  Distance-weighted F1{suffix}: {f1:.4f}')
        return metrics, lines
