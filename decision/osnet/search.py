"""Classify local T2 loss using source observations and choose one checked approach."""

from collections import deque
import math

from perception.environment.context import fingerprint


def _overlaps(a, b):
    overlap = (max(0., min(a[2], b[2]) - max(a[0], b[0])) *
               max(0., min(a[3], b[3]) - max(a[1], b[1])))
    return overlap > 0 and overlap >= .25 * min((a[2] - a[0]) * (a[3] - a[1]),
                                                (b[2] - b[0]) * (b[3] - b[1]))


def _turn_samples(samples, center, first_center, width):
    if len(samples) < 4 or abs(center - .5) < .15 or abs(center - first_center) < .08:
        return None
    first, latest = samples[0], samples[-1]
    # Use displacements that span multiple samples when the camera runs quickly.
    preceding_index = next((index for index in range(len(samples) - 2, 1, -1)
                            if math.dist(samples[index]['position'], latest['position']) >= .12), None)
    if preceding_index is None:
        return None
    preceding = samples[preceding_index]
    for corner in reversed(samples[1:preceding_index]):
        incoming = [b - a for a, b in zip(first['position'], corner['position'])]
        outbound = [b - a for a, b in zip(corner['position'], preceding['position'])]
        final = [b - a for a, b in zip(preceding['position'], latest['position'])]
        incoming_length, outbound_length, final_length = [math.hypot(*xy) for xy in (incoming, outbound, final)]
        left_box, right_box = corner['box'], latest['box']
        image_progress = (right_box[0] + right_box[2] - left_box[0] - left_box[2]) / (2 * width)
        if (min(incoming_length, outbound_length) >= .12 and
                sum(a * b for a, b in zip(outbound, final)) >= .87 * outbound_length * final_length and
                sum(a * b for a, b in zip(incoming, outbound)) <= .65 * incoming_length * outbound_length and
                (center - .5) * image_progress >= .015):
            return first, corner, preceding, latest
    return None


def _departure(samples, min_height):
    if len(samples) < 2:
        return None, None
    first, latest = samples[0], samples[-1]
    if math.dist(first['position'], latest['position']) < .25:
        return None, None
    x1, y1, x2, y2 = latest['box']
    width, height = latest['image_size']
    center = (x1 + x2) / (2 * width)
    first_center = (first['box'][0] + first['box'][2]) / (2 * width)
    turn = _turn_samples(samples, center, first_center, width)
    if turn is not None:
        return 'turned_toward_image_side', turn
    if ((x1 <= .04 * width or center <= .18) and first_center - center >= .12 or
            (x2 >= .96 * width or center >= .82) and center - first_center >= .12):
        return 'left_image_edge', None

    if abs(center - .5) >= .19 and (center - .5) * (center - first_center) >= .025:
        for corner in samples[1:-1]:
            incoming = [b - a for a, b in zip(first['position'], corner['position'])]
            outgoing = [b - a for a, b in zip(corner['position'], latest['position'])]
            incoming_length, outgoing_length = math.hypot(*incoming), math.hypot(*outgoing)
            if (incoming_length >= .18 and outgoing_length >= .25 and
                    sum(a * b for a, b in zip(incoming, outgoing)) <= .5 * incoming_length * outgoing_length):
                return 'turned_toward_image_side', None

    box_height = y2 - y1
    if y1 > .02 * height and y2 < .98 * height:
        for index, intermediate in enumerate(samples[1:-1], 1):
            previous_height = intermediate['box'][3] - intermediate['box'][1]
            if latest['distance'] - intermediate['distance'] < .12 or box_height >= previous_height:
                continue
            for older in samples[:index]:
                old_height = older['box'][3] - older['box'][1]
                if (math.dist(older['position'], latest['position']) >= .25 and
                        intermediate['distance'] - older['distance'] >= .12 and
                        latest['distance'] - older['distance'] >= .3 and previous_height < old_height and
                        (box_height <= .6 * old_height or
                         box_height <= .75 * old_height and box_height <= max(1.5 * min_height, .12 * height))):
                    return 'receding_in_view', None
    return None, None


class FollowSearch:
    """Track one loss episode from the last visual confirmation accepted by control."""

    wait_ns = 1_000_000_000
    max_gap_ns = 500_000_000
    max_initial_loss_gap_ns = 1_500_000_000
    evidence_max_age_s = 45.

    def __init__(self):
        self.samples = deque(maxlen=64)
        self.last_binding = None
        self.attempted = False
        self.defer()

    def defer(self):
        """Stop a continuation and require consecutive loss before any unused attempt."""
        self.first_loss_ns = self.latest_loss_ns = self.loss_reason = None
        self.loss_count = 0
        self.continuation = None
        self.departure_yaw_rad = None

    def observe(self, source, decision, *, last, positions, pose_xy, observed, association_tracks, policy):
        stamp = source['rgb_stamp_ns']
        last_binding = None if last is None else last.get('binding')
        if self.last_binding != last_binding:
            self.defer()
            self.attempted = False
            self.last_binding = last_binding
        result = dict(state='unknown', reason=decision['reason'], last_evidence_binding=last_binding,
                      first_loss_rgb_stamp_ns=None, loss_count=0, search_ready=False,
                      departure_yaw_rad=None, departure_evidence=None)
        if decision['reason'] == 'source_not_new':
            return result
        if decision['status'] in ('same', 'recovered'):
            self.defer()
            if pose_xy is not None:
                candidate_id = decision['candidate_id']
                candidate = next(item for item in source['candidates'] if item['id'] == candidate_id)
                position = positions[candidate_id]
                self.samples.append(dict(scene_binding=source['scene_binding'], stamp=stamp,
                    run_start=decision['status'] == 'recovered',
                    support_stamp_s=candidate['track']['support_stamp_s'],
                    track=(tuple(decision['identity']), decision['lifecycle_generation']), position=position,
                    distance=math.dist(position, pose_xy), box=candidate['box'], image_size=source['image_size']))
            return dict(result, state='confirmed')
        if decision['reason'] == 'awaiting_confirmation':
            self.defer()
            return dict(result, state='recovering')
        if (last_binding is None or last['clock_domain'] != source['clock_domain'] or
                last['local_identity'] != source['local_identity'] or
                not 0 <= source['stamp_s'] - last['stamp_s'] < self.evidence_max_age_s or
                observed is None or pose_xy is None or source['association_status'] != 'complete'):
            self.defer()
            return result

        track = (tuple(last['identity']), last['lifecycle_generation'])
        samples = list(self.samples)
        anchor_index = next((i for i in reversed(range(len(samples)))
                             if samples[i]['scene_binding'] == last['scene_binding']), None)
        if anchor_index is None:
            self.defer()
            return result
        anchor = samples[anchor_index]
        if (anchor['track'] != track or anchor['image_size'] != source['image_size'] or
                math.dist(anchor['position'], last['position_xy_m']) > 1e-6):
            self.defer()
            return result
        history = [anchor]
        for prior in reversed(samples[:anchor_index]):
            newer = history[-1]
            if (newer['run_start'] or prior['track'] != track or prior['image_size'] != anchor['image_size'] or
                    newer['stamp'] - prior['stamp'] > self.max_gap_ns or
                    anchor['stamp'] - prior['stamp'] > 1_500_000_000):
                break
            history.append(prior)
        departure, turn = _departure(history[::-1], policy.min_box_height_px)
        track_key = str(track[0][2]) + '/g' + str(track[1])
        association = association_tracks.get(track_key, {})
        if association.get('status') in ('time_skew', 'unconfirmed'):
            self.defer()
            return result
        previous_regions = [anchor['box']]
        if association.get('proposal_box') is not None:
            previous_regions.append(association['proposal_box'])
        nearby = [item for item in source['candidates']
                  if item['id'] == association.get('detection_index') or
                  any(_overlaps(item['box'], region) for region in previous_regions)]
        fragments = []
        if departure == 'receding_in_view':
            previous_height = anchor['box'][3] - anchor['box'][1]
            for item in nearby:
                x1, y1, x2, y2 = item['box']
                if (y2 - y1 < policy.min_box_height_px or
                        x2 - x1 < policy.min_box_width_px and y2 - y1 <= .75 * previous_height):
                    fragments.append(item)
        if association.get('status') == 'ambiguous' or len(nearby) > len(fragments):
            self.defer()
            return dict(result, state='occluded', reason='target_region_ambiguous')

        absent = (decision['reason'] == 'no_detections' or
                  decision['reason'] == 'below_similarity' and decision['score'] <= policy.min_reference_score or
                  decision['reason'] == 'appearance_low_quality' and bool(fragments) and
                  all(item in fragments for item in source['candidates']))
        if not departure or not absent:
            self.defer()
            if not departure and decision['reason'] == 'no_detections' and track in observed:
                return dict(result, state='holding', reason='isolated_detector_dropout')
            return result

        if (self.latest_loss_ns is None or stamp - self.latest_loss_ns > self.max_gap_ns or
                self.loss_reason != departure):
            self.defer()
            if not 0 < stamp - anchor['stamp'] <= self.max_initial_loss_gap_ns:
                return dict(result, reason='loss_onset_unobserved')
            self.first_loss_ns = stamp
        self.latest_loss_ns = stamp
        self.loss_reason = departure
        self.loss_count += 1
        ready = (self.loss_count >= 3 and stamp - self.first_loss_ns >= self.wait_ns and
                 (not self.attempted or self.continuation is not None))
        result.update(state='lost', reason=departure, first_loss_rgb_stamp_ns=self.first_loss_ns,
                      loss_count=self.loss_count, search_ready=ready)
        if turn is not None:
            corner, latest = turn[1]['position'], turn[-1]['position']
            self.departure_yaw_rad = result['departure_yaw_rad'] = math.atan2(latest[1] - corner[1], latest[0] - corner[0])
            result['departure_evidence'] = [dict(scene_binding=item['scene_binding'], rgb_stamp_ns=item['stamp'],
                support_stamp_s=item['support_stamp_s'], position_xy_m=list(item['position'])) for item in turn]
        return result

    def choose(self, search, *, last, scene):
        """Select or renew the fixed checked pose for an already qualified loss."""
        record = scene.record
        stamp, domain = record['as_of'], record['clock_domain']
        if (search['scene_binding'] != record['input_fingerprint'] or search['clock_domain'] != domain or
                search['source_stamp_s'] != stamp or
                search['binding'] != fingerprint({key: value for key, value in search.items() if key != 'binding'})):
            raise ValueError('follow_search_observation_mismatch')
        wait = next(candidate for candidate in search['candidates'] if candidate['kind'] == 'wait')
        if (last.get('binding') != self.last_binding or self.loss_count < 3 or
                self.latest_loss_ns - self.first_loss_ns < self.wait_ns):
            return wait
        pose_size = 3 if self.departure_yaw_rad is not None else 2
        checked = [candidate for candidate in search['candidates']
                   if candidate['kind'] == 'approach' and len(candidate['pose']) == pose_size and
                   candidate.get('check', {}).get('valid') is True and
                   candidate['check'].get('source_stamp_s') == stamp and
                   candidate['check'].get('clock_domain') == domain]
        current = self.continuation
        if current is not None:
            refresh = next((candidate for candidate in checked
                            if candidate.get('continuation_id') == current['intent_id'] and
                            candidate['pose'] == current['target']), None)
            if refresh is not None and stamp < current['expires_s']:
                return refresh
            self.defer()
            return wait
        candidate = next((candidate for candidate in checked if 'continuation_id' not in candidate), None)
        expires_s = last['stamp_s'] + self.evidence_max_age_s
        if self.attempted or candidate is None or stamp >= expires_s:
            return wait
        self.attempted = True
        self.continuation = dict(intent_id=fingerprint([search['binding'], candidate['id']]),
                                 kind='approach', target=list(candidate['pose']), expires_s=expires_s)
        return candidate
