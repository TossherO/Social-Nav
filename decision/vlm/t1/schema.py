"""Response constraints and validation for T1 pedestrian groups."""


def groups_schema(pids):
    if (not isinstance(pids,list) or any(type(p) is not int or not 0 <= p <= 99 for p in pids)
            or len(set(pids)) != len(pids)):
        raise ValueError('Invalid current PID set')
    if len(pids)<2:
        return dict(type='object',additionalProperties=False,properties={})
    value=dict(type='array',minItems=1,maxItems=len(pids)//2,
               items=dict(type='array',minItems=2,maxItems=len(pids),uniqueItems=True,
                          items=dict(type='integer',enum=pids)))
    return dict(type='object',additionalProperties=False,
                properties={name:value for name in ('walking','talking','photographing','other')})


def validate_groups(raw,pids,schema=None):
    from jsonschema import Draft202012Validator
    Draft202012Validator(schema or groups_schema(pids)).validate(raw)
    claimed=[]
    for activity,groups in raw.items():
        if activity not in {'walking','talking','photographing','other'}:
            raise ValueError('Unsupported activity')
        for members in groups:
            if len(members)<2 or any(type(p) is not int or p not in pids for p in members):
                raise ValueError('Invalid group membership')
            claimed.extend(members)
    if len(set(claimed)) != len(claimed):
        raise ValueError('A PID may occur only once across all groups')
