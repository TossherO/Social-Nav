"""Optional bounded evidence memory, separate from current execution proof."""
import copy
from perception.environment.context import fingerprint


class EvidenceMemory:
    def __init__(self, entries=None, *, capacity=24):
        self.entries = copy.deepcopy(entries or [])[-capacity:]
        self.capacity = capacity

    def remember(self, result, context, *, place_id, clock_domain, map_binding):
        entry = dict(task_id=context['task_id'], place_id=place_id,
                     observation_id=context['observation_id'], input_binding=context['observation_binding'],
                     source_stamp_s=context['source_stamp_s'], clock_domain=clock_domain,
                     local_identity=copy.deepcopy(context['local_identity']), map_binding=copy.deepcopy(map_binding),
                     semantic=copy.deepcopy(result), spatial_valid=True, invalidation_reasons=[])
        entry['evidence_id'] = fingerprint(entry)
        if not any(e['evidence_id'] == entry['evidence_id'] for e in self.entries):
            self.entries.append(entry)
            self.entries = self.entries[-self.capacity:]

    def invalidate(self, *, local_identity, clock_domain, map_binding):
        changed = []
        for entry in self.entries:
            reasons = []
            if entry['clock_domain'] != clock_domain:
                reasons.append('clock_changed')
            if entry['local_identity'] != local_identity:
                reasons.append('localization_changed')
            if entry['map_binding'] != map_binding:
                reasons.append('map_binding_changed')
            if reasons and entry['spatial_valid']:
                entry.update(spatial_valid=False, invalidation_reasons=reasons)
                changed.append(entry['evidence_id'])
        return changed

    def recall(self, *, place_ids, now, clock_domain, limit=4):
        # Preserve contradictory observations individually. These are historical
        # model judgments, never current route clearance or arrival proof.
        selected = [e for e in self.entries if e['place_id'] in place_ids and
                    (e['clock_domain'] != clock_domain or e['source_stamp_s'] <= now)]
        return [dict(copy.deepcopy(e), historical_only=True, proves_arrival=False,
                     proves_current_clearance=False) for e in selected[-limit:]]

    def snapshot(self):
        return copy.deepcopy(self.entries)
