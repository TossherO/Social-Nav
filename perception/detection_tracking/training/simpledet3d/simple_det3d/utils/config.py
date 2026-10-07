import os
from pathlib import Path

import yaml
from easydict import EasyDict


def load_config(yaml_path):
    """Load a YAML config file with optional _BASE_CONFIG_ inheritance.

    The _BASE_CONFIG_ value (if present) is a path relative to the YAML file
    being loaded.  The base config is loaded first, then overridden by the
    current file's values.

    Args:
        yaml_path: str or Path – path to the top-level YAML config.

    Returns:
        EasyDict – the merged configuration.
    """
    config = EasyDict()
    _merge_new_config(config, _load_yaml_into_easydict(yaml_path), base_dir=os.path.dirname(os.path.abspath(yaml_path)))
    return config


def log_config_to_file(cfg, pre='cfg', logger=None):
    """Recursively log every key/value in an EasyDict config."""
    for key, val in cfg.items():
        if isinstance(cfg[key], EasyDict):
            logger.info('----------- %s -----------' % (key))
            log_config_to_file(cfg[key], pre=pre + '.' + key, logger=logger)
            continue
        logger.info('%s.%s: %s' % (pre, key, val))


def cfg_from_list(cfg_list, config):
    """Set config keys via list (e.g., from command line key=value pairs).

    Args:
        cfg_list: list[str] – flat list of key=value strings.  Keys use
            dot-notation to address nested dicts (e.g. 'OPTIM.LEARNING_RATE').
        config: EasyDict – the configuration to update in-place.
    """
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = config
        for subkey in key_list[:-1]:
            assert subkey in d, 'NotFoundKey: %s' % subkey
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d, 'NotFoundKey: %s' % subkey
        try:
            value = literal_eval(v)
        except Exception:
            value = v

        if type(value) != type(d[subkey]) and isinstance(d[subkey], EasyDict):
            key_val_list = value.split(',')
            for src in key_val_list:
                cur_key, cur_val = src.split(':')
                val_type = type(d[subkey][cur_key])
                cur_val = val_type(cur_val)
                d[subkey][cur_key] = cur_val
        elif type(value) != type(d[subkey]) and isinstance(d[subkey], list):
            val_list = value.split(',')
            for k, x in enumerate(val_list):
                val_list[k] = type(d[subkey][0])(x)
            d[subkey] = val_list
        else:
            assert type(value) == type(d[subkey]), \
                'type {} does not match original type {}'.format(type(value), type(d[subkey]))
            d[subkey] = value


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_yaml_into_easydict(yaml_path):
    with open(yaml_path, 'r') as f:
        try:
            raw = yaml.safe_load(f, Loader=yaml.FullLoader)
        except Exception:
            raw = yaml.safe_load(f)
    return EasyDict(raw)


def _merge_new_config(config, new_config, base_dir=None):
    if '_BASE_CONFIG_' in new_config:
        base_path = new_config['_BASE_CONFIG_']
        if base_dir is not None:
            base_path = os.path.join(base_dir, base_path)
        base_path = os.path.abspath(base_path)
        base_config = _load_yaml_into_easydict(base_path)
        # Resolve inheritance recursively so stage-specific configs can inherit
        # a Wheelchair weak-supervision config that itself inherits the CODA
        # model architecture.
        _merge_new_config(config, base_config, base_dir=os.path.dirname(base_path))

    for key, val in new_config.items():
        if key == '_BASE_CONFIG_':
            continue
        if not isinstance(val, dict):
            config[key] = val
            continue
        if key not in config:
            config[key] = EasyDict()
        _merge_new_config(config[key], val, base_dir=base_dir)

    return config
