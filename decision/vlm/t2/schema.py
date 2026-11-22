"""T2 transport constraint and selection validation."""
from ..vlm_api_common import parse_response


def selection_regex(choices):
    persons = [label for label, value in choices.items() if value['kind'] == 'person']
    observations = [label for label, value in choices.items() if value['kind'] == 'observation']
    variants = []
    if persons:
        variants.append(r'\{"target":"(' + '|'.join(persons) + r')","observation":null\}')
    variants.append(r'\{"target":null,"observation":"(' + '|'.join(observations) + r')"\}')
    return '(' + '|'.join(variants) + ')'


def parse_selection(raw, choices):
    """Require exactly one target or observation from the corresponding set."""
    try:
        value = parse_response(raw)
    except (TypeError, ValueError):
        raise ValueError('Invalid T2 JSON') from None
    if not isinstance(value, dict) or set(value) != {'target', 'observation'}:
        raise ValueError('Invalid T2 fields')
    target, observation = value['target'], value['observation']
    if isinstance(target, str) and observation is None and choices.get(target, {}).get('kind') == 'person':
        return target
    if target is None and isinstance(observation, str) and choices.get(observation, {}).get('kind') == 'observation':
        return observation
    raise ValueError('Invalid T2 selection')
