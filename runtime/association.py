"""Bounded causal association updates before scene construction, never after sealing."""
from collections import OrderedDict
import json
import math


class AssociationInbox:
    def __init__(self,sock,metrics):
        self.socket=sock;self.socket.setblocking(False);self.metrics=metrics
        self.cache=OrderedDict();self.received=0;self.used=0;self.rejected=0

    @staticmethod
    def _key(value):
        return (value['local_session_id'],value['local_generation'],value['local_frame'],value['stamp_s'])

    def select(self,original):
        while True:
            try:payload=self.socket.recv(65536)
            except BlockingIOError:break
            value=json.loads(payload);self.received+=1
            stamp=value['stamp_s'];measured=value.get('measurement_stamp_s')
            if (not math.isfinite(stamp) or measured is None or not math.isfinite(measured) or measured>stamp):
                self.rejected+=1;continue
            key=self._key(value);previous=self.cache.get(key)
            if previous is None or measured>=previous['measurement_stamp_s']:
                self.cache[key]=value;self.cache.move_to_end(key)
            while len(self.cache)>64:self.cache.popitem(last=False)
        value=self.cache.get(self._key(original))
        if (value is None or value['map_id']!=original['map_id'] or
                value.get('asset_fingerprint')!=original.get('asset_fingerprint') or
                original.get('measurement_stamp_s') is not None and value['measurement_stamp_s']<original['measurement_stamp_s']):
            return original
        self.used+=1
        self.metrics.trace('association_selected',measurement_stamp_s=original['stamp_s'],
            old_association_stamp_s=original.get('measurement_stamp_s'),association_stamp_s=value['measurement_stamp_s'],valid=value['valid'])
        return value

    def close(self):
        self.socket.close()
        return dict(received=self.received,used=self.used,rejected=self.rejected)
