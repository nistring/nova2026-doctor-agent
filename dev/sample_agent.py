"""Run OUR agent (agent/run.py) on published server samples with the development key, and save each session like a
local run (results/server/<version>/<case>-<time>/transcript.json, record.json, stderr.txt) so dev/judge.py works on it.
Run this in your own terminal: competition rules forbid using the development key from coding tools (Claude Code).

  python dev/sample_agent.py --list
  python dev/sample_agent.py --case PUBLIC_SAMPLE_UUID [--reps 2]
  python dev/sample_agent.py --all [--reps 2]            # every published sample (team limit: 100 sessions/day)
  python dev/sample_agent.py --all --ref v6             # run a tagged/committed version instead of the working agent/
"""
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

from common import DEVKIT, ROOT, dev_key, export_ref, load_env, read_json, write_json


def agent_dir():
    """The agent to run: the working agent/ folder, or `--ref <git ref>` exported to a temporary folder (removed at exit)."""
    if '--ref' not in sys.argv:
        return ROOT / 'agent', 'working'
    ref = sys.argv[sys.argv.index('--ref') + 1]
    return export_ref(ref), ref


AGENT_DIR, VERSION = agent_dir()
sys.path.insert(0, str(AGENT_DIR))
import run as agent  # noqa: E402
from nova_sdk import Client, NovaError  # noqa: E402


class RecordingClient(Client):
    """nova_sdk.Client that also keeps the transcript and Doctor usage, in the devkit transcript format."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.events, self.record, self.status = [], None, 'active'
        self.calls = self.tokens_in = self.tokens_out = 0
        self.turn = 0

    def action(self, action, *, request_id=None, **fields):
        try:
            reply = super().action(action, request_id=request_id, **fields)
        except NovaError as exc:
            if exc.code in agent.TURN_REFUSALS:
                self.turn += 1
                self.events.append({'turn': self.turn, 'action': action, 'content': fields.get('content'), 'refused': exc.code})
            raise
        if action == 'DIAGNOSE':
            self.record, self.status = {**fields['note'], 'diagnosis': fields['diagnosis']}, 'completed'
            self.events.append({'turn': self.turn, 'action': 'DIAGNOSE', 'diagnosis': fields['diagnosis']})
        else:
            self.turn = reply.get('turn', self.turn + 1)
            self.events.append({'turn': self.turn, 'action': action, 'content': fields.get('content'), 'response': reply})
            print(f"T{self.turn:02d} {action} {fields.get('content')} → {reply.get('content') or reply.get('result')}")
        return reply

    def chat(self, messages, **kwargs):
        completion = super().chat(messages, **kwargs)
        usage = completion.get('usage') or {}
        self.calls += 1
        self.tokens_in += int(usage.get('prompt_tokens') or 0)
        self.tokens_out += int(usage.get('completion_tokens') or 0)
        return completion


def local_case(opening):
    """The devkit sample folder whose opening matches this server sample (for grading against the script)."""
    for path in (DEVKIT / 'samples').glob('*/case.json'):
        if read_json(path)['start_information']['opening'].strip() == opening.strip():
            return path.parent.name
    return 'unknown'


class Tee:
    """stderr that is also kept: the agent's [agent] log (failed or rejected model answers) is saved with the run."""

    def __init__(self, stream):
        self.stream, self.lines = stream, []

    def write(self, text):
        self.lines.append(text)
        return self.stream.write(text)

    def flush(self):
        self.stream.flush()


def run_case(case_id, cards):
    client = RecordingClient(api_base=load_env().get('NOVA_API_BASE', 'https://nova.snubhai.org'), token=dev_key(), sample=True)
    started = time.time()
    client.create_sample(case_id)
    session = client.session()
    name = local_case(session['initial']['opening'])
    print(f'== {name} ({case_id})')
    sys.stderr = log = Tee(sys.__stderr__)
    try:
        print(agent.consult(client, cards))
    except NovaError as exc:
        print(f'agent ended with {exc.code}')
    finally:
        sys.stderr = sys.__stderr__
    out = ROOT / 'results' / 'server' / VERSION / f'{name}-{datetime.now():%m%d-%H%M%S}'
    summary = {'case_id': name, 'version': VERSION, 'server_case_id': case_id, 'status': client.status, 'turns': client.turn,
               'doctor_calls': client.calls, 'doctor_input_tokens': client.tokens_in, 'doctor_output_tokens': client.tokens_out,
               'seconds': round(time.time() - started, 1)}
    write_json(out / 'transcript.json', {'summary': summary, 'start_information': session['initial'], 'events': client.events})
    if client.record:
        write_json(out / 'record.json', client.record)
    (out / 'stderr.txt').write_text(''.join(log.lines), encoding='utf-8')
    print(f'saved {out.relative_to(ROOT)}: {summary}')


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--case')
    group.add_argument('--all', action='store_true')
    group.add_argument('--list', action='store_true')
    parser.add_argument('--reps', type=int, default=1)
    parser.add_argument('--ref', help='git tag/commit of the agent to run instead of the working agent/')
    args = parser.parse_args()
    client = Client(api_base=load_env().get('NOVA_API_BASE', 'https://nova.snubhai.org'), token=dev_key(), sample=True)
    cases = client._request('GET', '/v1/sample/cases')['cases']
    if args.list:
        for case in cases:
            print(case['case_id'], local_case(case['initial']['opening']), case['initial']['opening'][:50])
        return
    cards = agent.prepare()
    for case_id in ([c['case_id'] for c in cases] if args.all else [args.case]) * args.reps:
        run_case(case_id, cards)


if __name__ == '__main__':
    main()
