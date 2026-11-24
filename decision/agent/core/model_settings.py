"""Hosted Agent model choices and provider-specific credential resolution."""
import json
import os
from pathlib import Path
import re
from urllib.parse import urlsplit


DEFAULT_MODEL = 'qwen3.8-flash'
GPT_MODEL = 'gpt-6.1-terra'
OPENAI_ENDPOINT = 'https://api.openai.com/v1/responses'
MODEL_EFFORTS = {
    DEFAULT_MODEL: ('off', 'low', 'medium'),
    GPT_MODEL: ('off', 'low', 'medium', 'high', 'xhigh'),
}


def default_efforts(model_name):
    """Return the registration, execution and monitor defaults."""
    if model_name == GPT_MODEL:
        return 'low', 'low', 'low'
    if model_name == DEFAULT_MODEL:
        return 'low', 'off', 'off'
    raise ValueError(f'Unknown Agent model: {model_name}')


def validate_effort(model_name, effort):
    if model_name not in MODEL_EFFORTS:
        raise ValueError(f'Unknown Agent model: {model_name}')
    if effort not in MODEL_EFFORTS[model_name]:
        raise ValueError(f'{model_name} supports these reasoning modes: {", ".join(MODEL_EFFORTS[model_name])}')
    return effort


def _api_endpoint(url, route):
    if not isinstance(url, str):
        raise ValueError('Expected an HTTPS API address')
    url = url.rstrip('/')
    parsed = urlsplit(url)
    if (parsed.scheme != 'https' or not parsed.hostname or parsed.username is not None
            or parsed.password is not None or parsed.query or parsed.fragment):
        raise ValueError('API requires HTTPS and credentials outside the URL')
    return url if url.endswith(route) else url + route


def credentials_for(model_name, private_config):
    """Resolve the endpoint and credentials belonging to the selected model."""
    if model_name not in MODEL_EFFORTS:
        raise ValueError(f'Unknown Agent model: {model_name}')
    config_path = Path(private_config)
    if model_name == GPT_MODEL:
        settings = json.loads(config_path.read_text()) if config_path.is_file() else {}
        endpoint = _api_endpoint(settings.get('openai_base_url', OPENAI_ENDPOINT), '/responses')
        key = settings.get('openai_api_key')
        # A custom service uses its own configured key, never the global OpenAI credential.
        if not key and endpoint == OPENAI_ENDPOINT:
            key = os.environ.get('OPENAI_API_KEY', '').strip()
        if not key:
            filename = settings.get('openai_api_key_file')
            if not isinstance(filename, str) or not filename.strip():
                raise ValueError('Configure openai_api_key or openai_api_key_file for this endpoint')
            key_path = config_path.parent / Path(filename).expanduser()
            contents = key_path.read_text().strip()
            # Accept a plain key or a single sk- key in a labelled local text file.
            matches = re.findall(r'(?<![\w-])sk-[a-zA-Z0-9_-]{16,}(?![\w-])', contents)
            if len(matches) != 1:
                raise ValueError('Expected a single OpenAI API key in openai_api_key_file')
            key = matches[0]
    else:
        settings = json.loads(config_path.read_text())
        endpoint = _api_endpoint(settings['base_url'], '/chat/completions')
        key = os.environ.get('SOCIAL_NAV_AGENT_API_KEY') or settings.get('api_key')
    if not isinstance(key, str) or not key or any(c.isspace() for c in key):
        raise ValueError(f'Missing or invalid API key for {model_name}')
    return endpoint, key
