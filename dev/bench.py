"""Run an agent over a case set several times with the devkit simulator, grade each run, and summarise.

  python dev/bench.py --tag v1 --reps 3                                   # samples, luna doctor, terra patient, sol judge
  python dev/bench.py --tag v1 --cases dev/data/synth --reps 2
  python dev/bench.py --tag base --agent nova-starter/run.py              # baseline for comparison
  python dev/bench.py --tag v6-again --ref v6                             # a committed version (git tag/commit)

Each run is a separate dev/local_run.py process (a fresh state per case, like the server). Results: results/local/<tag>/.
With DEV_BACKEND=nova (competition API) run this in your own terminal, not in Claude Code. One local run costs about
40 luna calls (Doctor + virtual patient + examiner); a luna judge call 1 more. Daily team limit: luna 1,000, sol 100.
"""
import argparse
import os
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common import DEV, DEVKIT, ROOT, endpoint, export_ref, load_env, read_json, usage_today, write_json
from judge import AREAS, efficiency, grade_run


def case_dirs(spec):
    dirs = []
    for part in spec.split(','):
        path = ROOT / part if part != 'samples' else DEVKIT / 'samples'
        dirs += [path] if (path / 'case.json').exists() else sorted(p.parent for p in path.glob('*/case.json'))
    return dirs


def run_once(agent, case_dir, out, doctor, patient):
    if (out / 'transcript.json').exists():
        return out  # already ran (use a new --tag to repeat)
    done = subprocess.run([sys.executable, str(DEV / 'local_run.py'), '--case', str(case_dir), '--agent', str(agent),
                           '--out', str(out), '--doctor', doctor, '--patient', patient],
                          env={**os.environ, 'PYTHONUTF8': '1'}, capture_output=True, text=True, encoding='utf-8')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'stdout.txt').write_text(done.stdout + '\n--- stderr ---\n' + done.stderr, encoding='utf-8')
    return out if (out / 'transcript.json').exists() else None


def fmt(values):
    return f'{statistics.mean(values):5.1f}±{statistics.pstdev(values):4.1f}' if values else '  -  '


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', required=True)
    parser.add_argument('--cases', default='samples', help="'samples', a case folder, or a folder of case folders (comma-separated)")
    parser.add_argument('--agent', default='agent/run.py')
    parser.add_argument('--ref', help='git tag/commit of agent/ to run instead of the working tree')
    parser.add_argument('--reps', type=int, default=1)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--judge', default='sol', choices=['luna', 'terra', 'sol', 'none'])
    parser.add_argument('--doctor', default='luna')
    parser.add_argument('--patient', default='terra', help="virtual patient and examiner ('luna' with DEV_BACKEND=nova)")
    args = parser.parse_args()

    agent = export_ref(args.ref) / 'run.py' if args.ref else ROOT / args.agent
    base = ROOT / 'results' / 'local' / args.tag
    jobs = [(c, base / f'{c.name}-r{r}') for c in case_dirs(args.cases) for r in range(1, args.reps + 1)]
    print(f'{len(jobs)} runs of {args.ref or args.agent} → {base.relative_to(ROOT)}')
    with ThreadPoolExecutor(args.workers) as pool:
        run_dirs = list(pool.map(lambda job: run_once(agent, job[0], job[1], args.doctor, args.patient), jobs))
    ep = None if args.judge == 'none' else endpoint(args.judge)

    def grade_job(item):
        (case_dir, out), run_dir = item
        if run_dir is None:
            return {'case': case_dir.name, 'run': str(out), 'failed': True}
        summary = read_json(run_dir / 'transcript.json')['summary']
        row = {'case': case_dir.name, 'run': str(run_dir.relative_to(ROOT)), 'status': summary['status'],
               'turns': summary['turns'], 'calls': summary['doctor_calls'], 'failed': summary['status'] != 'completed',
               'cost_krw': efficiency(summary['turns'], summary['doctor_input_tokens'], summary['doctor_output_tokens'])[1]}
        if ep:
            try:
                graded = grade_run(case_dir, run_dir, ep)
                row.update(scores=graded['scores'], detail=graded['detail'])
            except Exception as exc:  # one failed grading must not lose the whole table; rerun the tag to retry
                print(f'grading failed for {row["run"]}: {type(exc).__name__}: {exc}')
        return row

    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(grade_job, zip(jobs, run_dirs)))
    write_json(base / 'summary.json', results)

    columns = ['total', *AREAS, 'eff'] if ep else []  # raw judge scores; 'official' = dev/judge.py official_estimate
    print(f"{'case':8s} " + ' '.join(f'{c:>11s}' for c in [*columns, 'official', 'turns', 'calls', 'cost']) + '  failed')
    for case in [*sorted({r['case'] for r in results}), 'ALL']:
        rows = [r for r in results if case in ('ALL', r['case'])]
        ok = [r for r in rows if not r.get('failed')]
        cells = [[r['scores'][c] for r in ok if 'scores' in r] for c in columns]
        cells += [[r['official_est']['total'] for r in ok if 'official_est' in r]] if ep else []
        cells += [[r[k] for r in ok] for k in ('turns', 'calls', 'cost_krw')]
        print(f'{case:8s} ' + ' '.join(f'{fmt(v):>11s}' for v in cells) + f"  {sum(1 for r in rows if r.get('failed'))}")
    print('competition API calls today:', usage_today()[1])
    if load_env().get('DEV_BACKEND') == 'claude':
        from claude_backend import total_spend
        print('Claude spend (USD) total / today: %.2f / %.2f' % total_spend())


if __name__ == '__main__':
    main()
