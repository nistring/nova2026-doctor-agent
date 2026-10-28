"""Rebuild a transcript and record from the agent's stderr (shown per case in 샘플 검증), for grading and calibration.

  python dev/import_stderr.py --case C10 --log path/to/stderr.txt [--tag v5.1]
      -> results/official/<tag>/<case>-<n>/{transcript.json, record.json}
  python dev/judge.py --case nova-devkit/samples/C10 --run results/official/<tag>/<case>-<n>
Then put the server's area scores for the same case next to the local ones in dev/calibration.csv.
"""
import argparse
import json
import re

from common import ROOT, write_json

TURN = re.compile(r'^\[agent\] T(\d+) (?:의사: (.*?) / 환자: (.*)|진찰\((.*?)\): (.*)|\((SAY|EXAM) "(.*)" 거절됨: (\w+)\))$')


def parse(lines):
    events, diagnosis, note = [], None, None
    for line in lines:
        line = line.rstrip('\n')
        if line.startswith('[agent] DIAGNOSIS '):
            diagnosis = line[len('[agent] DIAGNOSIS '):]
        elif line.startswith('[agent] NOTE '):
            note = json.loads(line[len('[agent] NOTE '):])
        elif match := TURN.match(line):
            turn = int(match.group(1))
            if match.group(2) is not None:
                events.append({'turn': turn, 'action': 'SAY', 'content': match.group(2), 'response': {'content': match.group(3)}})
            elif match.group(4) is not None:
                events.append({'turn': turn, 'action': 'EXAM', 'content': match.group(4), 'response': {'result': match.group(5)}})
            else:
                events.append({'turn': turn, 'action': match.group(6), 'content': match.group(7), 'refused': match.group(8)})
    return events, diagnosis, note


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', required=True, help='public sample folder name, e.g. C10')
    parser.add_argument('--log', required=True)
    parser.add_argument('--tag', default='validation')
    args = parser.parse_args()
    events, diagnosis, note = parse(open(args.log, encoding='utf-8').readlines())
    base = ROOT / 'results' / 'official' / args.tag
    n = 1 + sum(1 for _ in base.glob(f'{args.case}-*'))
    out = base / f'{args.case}-{n}'
    turns = max((e['turn'] for e in events), default=0)
    summary = {'case_id': args.case, 'status': 'completed' if note else 'unknown', 'turns': turns,
               'doctor_calls': None, 'doctor_input_tokens': 0, 'doctor_output_tokens': 0, 'source': 'stderr'}
    write_json(out / 'transcript.json', {'summary': summary, 'events': events})
    if note:
        write_json(out / 'record.json', {**note, 'diagnosis': diagnosis})
    print(f'{out.relative_to(ROOT)}: {len(events)} turns parsed, record {"found" if note else "missing"}')


if __name__ == '__main__':
    main()
