"""Shared helpers for dev scripts (not shipped). Keys come from dev/.env or the environment, never from code.

Backends (DEV_BACKEND in dev/.env):
  nova    competition API outside a session with the development key (자가 채점·데이터 생성용; daily limits luna 1,000 /
          sol 100 successful calls per team). Rules forbid connecting the key to coding tools, so this backend refuses to
          run inside Claude Code: run those commands in your own terminal.
  openai  personal OpenAI key (OPENAI_API_KEY).
"""
import json
import os
import sys
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEV = ROOT / 'dev'
DEVKIT = ROOT / 'nova-devkit'
sys.path.insert(0, str(DEVKIT))

from nova_local import Endpoint, first_json  # noqa: E402
from nova_sdk import Client, NovaError  # noqa: E402

OPENAI = 'https://api.openai.com/v1'
MODELS = {'luna': 'gpt-6-luna', 'sol': 'gpt-6.1-sol'}           # OpenAI names (override with LUNA_MODEL / SOL_MODEL)
NOVA = {'luna': ('openai/gpt-6-luna', 8000, 1000), 'sol': ('openai/gpt-6.1-sol', 32000, 100)}  # id, max_tokens, calls/day
SECRET = {'NOVA_DEV_KEY', 'NOVA_RUN_TOKEN'}                        # never copied into os.environ by load_env
USAGE = DEV / 'data' / 'usage'
_lock = threading.Lock()


def _dotenv():
    path = DEV / '.env'
    pairs = {}
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            key, sep, value = line.strip().partition('=')
            key = key.strip().removeprefix('export ').strip()
            if sep and key and not key.startswith('#'):
                pairs[key] = value.strip().strip('"\'')
    return pairs


def load_env():
    """dev/.env into os.environ (existing variables win); the development key is kept out."""
    for key, value in _dotenv().items():
        if key not in SECRET:
            os.environ.setdefault(key, value)
    return os.environ


def dev_key():
    if os.environ.get('CLAUDECODE'):
        raise SystemExit('대회 규칙상 개발 키는 Claude Code 같은 코딩 도구에서 쓸 수 없습니다. 이 명령은 직접 연 터미널에서 실행하세요.')
    key = os.environ.get('NOVA_DEV_KEY') or _dotenv().get('NOVA_DEV_KEY')
    if not key:
        raise SystemExit('NOVA_DEV_KEY가 없습니다(dev/.env 또는 환경변수).')
    return key


def usage_today():
    """Successful competition-API calls today (KST), from an append-only local ledger."""
    path = USAGE / f'{datetime.now(timezone(timedelta(hours=9))):%Y-%m-%d}.log'
    lines = path.read_text(encoding='utf-8').split() if path.exists() else []
    return path, {alias: lines.count(alias) for alias in NOVA}


class NovaEndpoint:
    """Competition /v1/chat/completions outside a session, with the same complete() interface as nova_local.Endpoint."""

    def __init__(self, alias):
        self.alias, (self.model, self.max_tokens, self.daily) = alias, NOVA[alias]
        self.base_url = load_env().get('NOVA_API_BASE', 'https://nova.snubhai.org')
        self.key = dev_key()

    def complete(self, messages, max_tokens=2048, json_mode=False, reasoning_effort=None, temperature=None):
        path, used = usage_today()
        if used[self.alias] >= self.daily - 5:
            raise NovaError('LOCAL_DAILY_BUDGET')  # keep a margin under the team's daily limit
        if self.alias == 'luna' and reasoning_effort not in (None, 'low', 'medium', 'high'):
            reasoning_effort = 'high'
        client = Client(api_base=self.base_url, token=self.key, sample=True, timeout=300)
        request_id = str(uuid.uuid4())
        for attempt in range(6):
            try:
                completion = client.chat(messages, max_tokens=min(max_tokens, self.max_tokens), model=self.model,
                                         reasoning_effort=reasoning_effort, request_id=request_id)
                break
            except NovaError as exc:
                if attempt < 5 and exc.code in ('PROVIDER_UNAVAILABLE', 'REQUEST_PENDING', 'API_UNAVAILABLE'):
                    time.sleep(3 + 5 * attempt)  # same request id: the server may still be working on it
                    continue
                if attempt < 5 and exc.code == 'PROVIDER_RESPONSE_INVALID':
                    request_id = str(uuid.uuid4())  # empty answer: ask again
                    continue
                sys.stderr.write(f'[nova api] {self.model}: {exc.code}\n')
                raise
        with _lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('a', encoding='utf-8') as ledger:
                ledger.write(self.alias + '\n')
        choice, usage = completion['choices'][0], completion.get('usage') or {}
        return (choice['message'].get('content') or '', choice.get('finish_reason') or 'stop',
                int(usage.get('prompt_tokens') or 0), int(usage.get('completion_tokens') or 0))


def endpoint(alias='sol'):
    """Model endpoint for dev-side calls. alias names the role: 'luna' (Doctor, the server's gpt-6-luna), 'terra'
    (virtual patient and examiner; Claude backend only) or 'sol' (generation and grading, the server's gpt-6.1-sol).
    The agent only uses client.chat."""
    env = load_env()
    backend = env.get('DEV_BACKEND', 'nova')
    if backend == 'claude':
        from claude_backend import ALIAS_MODELS, DEFAULT_MODEL, ClaudeCodeEndpoint, ClaudeEndpoint
        model = env.get(f'CLAUDE_{alias.upper()}_MODEL', ALIAS_MODELS.get(alias, DEFAULT_MODEL))
        if alias in env.get('CLAUDE_CODE_ROLES', '').replace(' ', '').split(','):  # e.g. terra,sol: billed to the plan
            return ClaudeCodeEndpoint(model)
        return ClaudeEndpoint(model)
    alias = {'terra': 'luna', 'haiku': 'luna'}.get(alias, alias)  # tiers that exist only on the Claude backend
    if backend == 'nova':
        return NovaEndpoint(alias)
    return Endpoint(env.get('DEV_LLM_URL', OPENAI), env.get(alias.upper() + '_MODEL', MODELS[alias]),
                    env.get('OPENAI_API_KEY', ''), timeout=900)


def ask_json(ep, system, user, effort='medium', max_tokens=16000, tries=2):
    """One JSON object from a dev-side model call, or None."""
    for _ in range(tries):
        text, finish, _, _ = ep.complete([{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
                                         max_tokens=max_tokens, json_mode=True, reasoning_effort=effort)
        value = first_json(text)
        if value is not None:
            return value
        sys.stderr.write(f'[ask_json] unparsable output (finish={finish}), retrying\n')
    return None


def export_ref(ref):
    """agent/ as of a git tag or commit, exported to a temporary folder that is removed when the process exits."""
    import atexit
    import io
    import shutil
    import subprocess
    import tarfile
    import tempfile
    tar = subprocess.run(['git', 'archive', '--format=tar', ref, 'agent'], cwd=ROOT, capture_output=True, check=True).stdout
    target = Path(tempfile.mkdtemp(prefix=f'agent-{ref}-'))
    atexit.register(shutil.rmtree, target, ignore_errors=True)
    tarfile.open(fileobj=io.BytesIO(tar)).extractall(target)
    return target / 'agent'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=1), encoding='utf-8')
