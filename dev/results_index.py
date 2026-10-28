"""Write results/README.md: what each folder holds and one score row per server version and local bench.

  python dev/results_index.py
"""
import json

from common import ROOT

RESULTS = ROOT / 'results'
AREAS = ['total', 'info', 'dx', 'tests', 'safety', 'comm', 'eff']
HEADER = """# results (git-ignored)

| folder | what | written by |
|---|---|---|
| `server/<version>/<case>-<MMDD-HHMMSS>/` | **external**: our agent on the competition server's public samples (luna Doctor, server patient) | `dev/sample_agent.py` (user-run) |
| `local/<tag>/<case>-r<n>/` + `summary.json` | **internal**: devkit simulator (Haiku Doctor, Sonnet patient), graded by Opus | `dev/bench.py` |
| `official/<tag>/<case>-<n>/` | 샘플 검증 stderr pasted from the submission page, rebuilt into a transcript | `dev/import_stderr.py` |

In each run folder: `transcript.json`, `record.json`, `stderr.txt` (server) or `stdout.txt` (local), and grades named by grader:
`grade-opus.json` (Opus, judge re-splits the model record each run), `grade-opus-fixed.json` (Opus, fixed checklist
`dev/data/checklists/` + script-extraction rate), `grade-sol.json` (competition sol with our rubric, user-run).
Official sample-validation scores are in `dev/calibration.csv`. Regenerate this file with `python dev/results_index.py`.
"""


def mean_row(grades):
    grades = [g for g in grades if g.get('scores')]
    if not grades:
        return None
    return {k: sum(g['scores'][k] for g in grades) / len(grades) for k in AREAS}, len(grades)


def fmt(row):
    means, n = row
    return ' | '.join(f'{means[k]:.1f}' for k in AREAS) + f' | {n}'


def main():
    out = [HEADER, '## server (external)\n', '| version | grader | ' + ' | '.join(AREAS) + ' | runs | turns |',
           '|---|---|' + '---|' * (len(AREAS) + 2)]
    for version in sorted((RESULTS / 'server').iterdir(), key=lambda p: int(p.name[1:]) if p.name[1:].isdigit() else 0):
        runs = [r for r in version.iterdir() if r.is_dir()]
        turns = []
        for r in runs:
            try:
                turns.append(json.loads((r / 'transcript.json').read_text(encoding='utf-8'))['summary']['turns'])
            except (OSError, KeyError, ValueError):
                pass
        for grader in ('grade-opus-fixed.json', 'grade-opus.json', 'grade-sol.json'):
            row = mean_row([json.loads((r / grader).read_text(encoding='utf-8')) for r in runs if (r / grader).exists()])
            if row:
                out.append(f"| {version.name} | {grader[6:-5]} | {fmt(row)} | {'/'.join(map(str, sorted(turns)))} |")
    out += ['', '## local (internal)\n', '| tag | ' + ' | '.join(AREAS) + ' | runs |', '|---|' + '---|' * (len(AREAS) + 1)]
    for tag in sorted((RESULTS / 'local').iterdir(), key=lambda p: p.stat().st_mtime):
        summary = tag / 'summary.json'
        if summary.exists():
            row = mean_row(json.loads(summary.read_text(encoding='utf-8')))
            if row:
                out.append(f'| {tag.name} | {fmt(row)} |')
    (RESULTS / 'README.md').write_text('\n'.join(out) + '\n', encoding='utf-8')
    print(f"wrote {RESULTS / 'README.md'}")


if __name__ == '__main__':
    main()
