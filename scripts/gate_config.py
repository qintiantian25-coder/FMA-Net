import configparser
import os


DEFAULT_GATE_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fusion_gate.cfg')


def _to_bool(value):
    return str(value).strip().lower() in ('1', 'true', 'yes', 'y', 'on')


def read_gate_section(config_path, section_name):
    """Read one section from gate config file. Returns empty dict if file/section is missing."""
    if not config_path or (not os.path.exists(config_path)):
        return {}

    parser = configparser.ConfigParser()
    with open(config_path, 'r', encoding='utf-8') as f:
        parser.read_file(f)

    if not parser.has_section(section_name):
        return {}

    return {k: v for k, v in parser.items(section_name)}


def pick_value(cli_value, section_dict, key, cast_fn, default_value):
    """Use CLI value first, then config section value, then fallback default."""
    if cli_value is not None:
        return cli_value

    raw = section_dict.get(key)
    if raw is None:
        return default_value

    try:
        if cast_fn is bool:
            return _to_bool(raw)
        return cast_fn(raw)
    except (TypeError, ValueError):
        return default_value

