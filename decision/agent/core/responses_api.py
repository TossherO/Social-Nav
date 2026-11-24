"""Translate Agent dialogue and native tool calls for the OpenAI Responses API."""


def response_input(messages):
    items = []
    pending = set()
    seen = set()
    for message in messages:
        role, content = message['role'], message.get('content')
        if role == 'tool':
            call_id = message['tool_call_id']
            if call_id not in pending or not isinstance(content, str):
                raise ValueError('Tool output requires its original function call and text')
            pending.remove(call_id)
            items.append(dict(type='function_call_output', call_id=call_id, output=content))
            continue
        if role not in ('system', 'developer', 'assistant', 'user') or pending:
            raise ValueError('Unexpected role or unanswered function call in dialogue')
        if isinstance(content, list):
            content = [input_part(part) for part in content]
        if content:
            items.append(dict(role=role, content=content))
        for call in message.get('tool_calls') or ():
            call_id = call['id']
            if (role != 'assistant' or call['type'] != 'function' or not isinstance(call_id, str)
                    or not call_id or call_id in seen):
                raise ValueError('Historical function call requires a unique original identity')
            seen.add(call_id)
            pending.add(call_id)
            items.append(dict(type='function_call', call_id=call_id, **call['function']))
    if pending:
        raise ValueError('Historical function call has no tool output')
    return items


def input_part(part):
    if part['type'] == 'text':
        return dict(type='input_text', text=part['text'])
    if part['type'] == 'image_url':
        image = part['image_url']
        return dict(type='input_image', image_url=image['url'], detail=image.get('detail', 'auto'))
    raise ValueError('Unsupported Responses API input content')


def request_body(history, thinking_mode):
    return dict(model=history['model'], input=response_input(history['messages']),
                tools=[dict(type='function', strict=False, **tool['function']) for tool in history['tools']],
                tool_choice=history['tool_choice'], parallel_tool_calls=False,
                max_output_tokens=history['max_output_tokens'],
                reasoning=dict(effort='none' if thinking_mode == 'off' else thinking_mode),
                stream=True, store=False)


TERMINAL_EVENTS = frozenset(('response.completed', 'response.incomplete', 'response.failed', 'error'))
TEMPORARY_ERROR_CODES = frozenset(('server_error', 'internal_server_error', 'rate_limit_exceeded',
                                   'service_unavailable', 'overloaded', 'engine_overloaded',
                                   'server_overloaded', 'temporarily_unavailable'))


def has_output_delta(event):
    kind = event.get('type')
    if kind == 'response.output_item.added':
        return event['item']['type'] == 'function_call'
    return kind in ('response.output_text.delta', 'response.function_call_arguments.delta',
                    'response.reasoning_summary_text.delta', 'response.reasoning_text.delta',
                    'response.refusal.delta') and bool(event.get('delta'))


def normalized_usage(response):
    usage = response.get('usage')
    if usage is None:
        return None
    return dict(prompt_tokens=usage['input_tokens'], completion_tokens=usage['output_tokens'],
                total_tokens=usage.get('total_tokens'),
                prompt_tokens_details=usage.get('input_tokens_details'),
                completion_tokens_details=usage.get('output_tokens_details'))


def completed_message(response):
    if response['status'] != 'completed' or not isinstance(response['output'], list):
        raise ValueError('Responses API output requires a completed response')
    text, calls, reasoning = [], [], []
    for item in response['output']:
        kind = item['type']
        if item.get('status', 'completed') != 'completed':
            raise ValueError('Unexpected unfinished Responses API output item')
        if kind == 'function_call':
            calls.append(dict(id=item['call_id'], type='function',
                              function=dict(name=item['name'], arguments=item['arguments'])))
        elif kind == 'message' and item['role'] == 'assistant':
            for part in item['content']:
                if part['type'] != 'output_text':
                    raise ValueError('Unexpected assistant response content')
                text.append(part['text'])
        elif kind == 'reasoning':
            for part in item.get('summary', []):
                if part['type'] != 'summary_text':
                    raise ValueError('Unexpected reasoning summary content')
                reasoning.append(part['text'])
        else:
            raise ValueError('Unexpected Responses API output item')
    return text, calls, reasoning
