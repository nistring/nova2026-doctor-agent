"""Claude (Anthropic API) for dev-side calls, with the same complete() interface as nova_local.Endpoint.

Used for knowledge/case generation, self-grading, the local virtual patient and examiner, and as a local stand-in
for the Doctor model. The submitted agent never imports this: it only calls client.chat (gpt-6-luna on the server).
Credentials: ANTHROPIC_API_KEY (dev/.env), read by anthropic.Anthropic().
"""
import json
import os
import subprocess
import sys
import tempfile
import threading
from datetime import date
from pathlib import Path

import anthropic
from nova_sdk import NovaError

DEFAULT_MODEL = 'claude-sonnet-5-5'
# Role alias → Claude model, by tier (user choice): luna (Doctor) is the light tier, terra (virtual patient and
# examiner) the middle, sol (generation and grading) the top. CLAUDE_<ALIAS>_MODEL in dev/.env overrides one role.
ALIAS_MODELS = {'luna': 'claude-haiku-5-5', 'haiku': 'claude-haiku-5-5', 'terra': 'claude-sonnet-5-5',
                'sol': 'claude-opus-5-5'}
JSON_HINT = '\n\n출력은 JSON 객체 하나만 씁니다(설명·코드 블록 없이).'
# USD per 1M tokens: input, output, cache read, 5-minute cache write
PRICES = {'claude-opus-5-5': (4.0, 20.0, 0.20, 5.0), 'claude-sonnet-5-5': (2.0, 10.0, 0.20, 2.5),
          'claude-haiku-5-5': (0.10, 0.50, 0.01, 0.125)}
LEDGER = Path(__file__).resolve().parent / 'data' / 'usage' / 'claude.log'
_lock = threading.Lock()


def spend(usage, model):
    price_in, price_out, price_read, price_write = PRICES.get(model, PRICES[DEFAULT_MODEL])
    return (usage.input_tokens * price_in + usage.output_tokens * price_out
            + (usage.cache_read_input_tokens or 0) * price_read + (usage.cache_creation_input_tokens or 0) * price_write) / 1e6


def total_spend():
    """USD recorded so far (all days) and today."""
    if not LEDGER.exists():
        return 0.0, 0.0
    rows = [line.split('\t') for line in LEDGER.read_text(encoding='utf-8').splitlines() if line]
    today = str(date.today())
    return round(sum(float(r[2]) for r in rows), 2), round(sum(float(r[2]) for r in rows if r[0] == today), 2)


def record_spend(model, usage):
    with _lock:
        LEDGER.parent.mkdir(parents=True, exist_ok=True)
        with LEDGER.open('a', encoding='utf-8') as ledger:
            ledger.write(f'{date.today()}\t{model}\t{spend(usage, model):.5f}\n')


def read_image(png_bytes, instruction, model=DEFAULT_MODEL, max_tokens=8000):
    """Text from one page image (OCR-style transcription)."""
    import base64
    client = anthropic.Anthropic(max_retries=6)
    content = [{'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png',
                                            'data': base64.standard_b64encode(png_bytes).decode('ascii')}},
               {'type': 'text', 'text': instruction}]
    with client.beta.messages.stream(model=model, max_tokens=max_tokens, messages=[{'role': 'user', 'content': content}],
                                     output_config={'effort': 'low'}, betas=['server-side-fallback-2026-07-01'],
                                     fallbacks='default') as stream:
        message = stream.get_final_message()
    record_spend(model, message.usage)
    if message.stop_reason == 'refusal':
        return ''
    return ''.join(block.text for block in message.content if block.type == 'text')


class ClaudeEndpoint:
    def __init__(self, model=DEFAULT_MODEL):
        self.model, self.base_url = model, 'api.anthropic.com'
        self.client = anthropic.Anthropic(max_retries=6)  # the SDK retries 408/409/429/5xx with backoff

    def complete(self, messages, max_tokens=2048, json_mode=False, reasoning_effort=None, temperature=None):
        """(text, stop_reason, input_tokens, output_tokens). temperature is ignored (not supported on this model)."""
        system = '\n\n'.join(m['content'] for m in messages if m['role'] == 'system')
        if json_mode and system:
            system += JSON_HINT
        request = {
            'model': self.model,
            # Thinking is always on for this model and counts toward max_tokens: leave room for the answer.
            'max_tokens': max(max_tokens, 8000),
            'messages': [m for m in messages if m['role'] != 'system'],
            # Simulator calls pass no effort; they are short role-play/lookup tasks.
            'output_config': {'effort': reasoning_effort or 'low'},
        }
        if 'haiku' not in self.model:  # Haiku has no server-side fallback (the parameter is a 400)
            request.update(betas=['server-side-fallback-2026-07-01'], fallbacks='default')  # re-run a declined request
        if system:
            request['system'] = [{'type': 'text', 'text': system, 'cache_control': {'type': 'ephemeral'}}]
        message = None
        for attempt in range(3):  # the SDK retries before a stream starts; a stream cut mid-way is retried here
            try:
                with self.client.beta.messages.stream(**request) as stream:
                    message = stream.get_final_message()
                break
            except anthropic.APIStatusError as exc:
                sys.stderr.write(f'[claude] {self.model}: HTTP {exc.status_code} {exc.message[:200]}\n')
                if exc.status_code < 500 and exc.status_code != 429:
                    raise NovaError('PROVIDER_UNAVAILABLE', 503) from None
            except Exception as exc:  # connection reset or read error while streaming
                sys.stderr.write(f'[claude] {self.model}: stream error {type(exc).__name__}: {exc}\n')
        if message is None:
            raise NovaError('PROVIDER_UNAVAILABLE', 503)
        usage = message.usage
        record_spend(self.model, usage)
        tokens_in = usage.input_tokens + (usage.cache_read_input_tokens or 0) + (usage.cache_creation_input_tokens or 0)
        if message.stop_reason == 'refusal':
            sys.stderr.write(f'[claude] {self.model}: refusal ({getattr(message.stop_details, "category", None)})\n')
            return '', 'refusal', tokens_in, usage.output_tokens
        text = ''.join(block.text for block in message.content if block.type == 'text')
        return text, message.stop_reason, tokens_in, usage.output_tokens


class ClaudeCodeEndpoint:
    """The same complete() through headless Claude Code (`claude -p`), billed to the logged-in Claude plan instead of the
    API key. Each call starts a process (about 3 s), max_tokens is not enforced, and the calls count toward the plan's
    usage limits, which interactive Claude Code sessions share. Chosen per role with CLAUDE_CODE_ROLES (dev/.env)."""

    def __init__(self, model=DEFAULT_MODEL):
        self.model, self.base_url = model, 'claude-code'

    def complete(self, messages, max_tokens=2048, json_mode=False, reasoning_effort=None, temperature=None):
        system = '\n\n'.join(m['content'] for m in messages if m['role'] == 'system')
        if json_mode and system:
            system += JSON_HINT
        user = '\n\n'.join(m['content'] for m in messages if m['role'] != 'system')
        # without the API key in its environment, claude uses the logged-in plan; CLAUDECODE would mark it as nested
        env = {k: v for k, v in os.environ.items() if k not in ('ANTHROPIC_API_KEY', 'CLAUDECODE')}
        with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False, encoding='utf-8') as prompt_file:
            prompt_file.write(system or 'You are a helpful assistant.')  # a file: Windows caps command lines at 32k
        command = ['claude', '-p', '--model', self.model, '--system-prompt-file', prompt_file.name, '--tools', '',
                   '--strict-mcp-config', '--no-session-persistence', '--output-format', 'json',
                   '--effort', reasoning_effort or 'low']
        try:
            for attempt in range(3):
                done = subprocess.run(command, input=user, capture_output=True, text=True, encoding='utf-8', env=env,
                                      timeout=900)
                try:
                    result = json.loads(done.stdout)
                except ValueError:
                    result = None
                if result and not result.get('is_error'):
                    break
                sys.stderr.write(f'[claude-code] {self.model}: exit {done.returncode} '
                                 f'{(done.stdout or done.stderr)[-300:]!r}\n')
            else:
                raise NovaError('PROVIDER_UNAVAILABLE', 503)
        finally:
            os.unlink(prompt_file.name)
        with _lock:
            LEDGER.parent.mkdir(parents=True, exist_ok=True)
            with LEDGER.open('a', encoding='utf-8') as ledger:  # plan usage, not API spend: logged at 0 USD
                ledger.write(f'{date.today()}\tclaude-code/{self.model}\t0\n')
        usage = result.get('usage') or {}
        tokens_in = sum(int(usage.get(k) or 0) for k in ('input_tokens', 'cache_read_input_tokens',
                                                          'cache_creation_input_tokens'))
        return result.get('result') or '', result.get('stop_reason') or 'stop', tokens_in, int(usage.get('output_tokens') or 0)
