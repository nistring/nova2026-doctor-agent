"""Run one agent on one case with the devkit simulator, using dev/common.py endpoints for Doctor and patient.

  python dev/local_run.py --case nova-devkit/samples/C10 --out results/local/try/C10
(bench.py calls this once per run in a separate process so every case starts from a fresh state, like the server.)
"""
import argparse
import time
from pathlib import Path

from common import DEVKIT, ROOT, endpoint
from develop import load_agent  # nova-devkit/develop.py (on sys.path via common)
from nova_local import LocalClient, NovaError, Simulator, load_case, save_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', required=True)
    parser.add_argument('--agent', default=str(ROOT / 'agent' / 'run.py'))
    parser.add_argument('--out', required=True)
    parser.add_argument('--doctor', default='luna')
    parser.add_argument('--patient', default='terra')
    parser.add_argument('--minutes', type=float, default=30)
    args = parser.parse_args()

    case = load_case(args.case)
    simulator = Simulator(case, doctor=endpoint(args.doctor), sim=endpoint(args.patient), minutes=args.minutes, log=print)
    client = LocalClient(simulator)
    agent = load_agent(args.agent)
    started = time.time()
    try:
        state = agent.prepare()
        client.ready()
        agent.consult(client, state)
    except NovaError as exc:
        print(f'agent ended with an unhandled error: {exc.code}')
    summary = save_result(simulator, Path(args.out), started)
    print(summary)


if __name__ == '__main__':
    assert DEVKIT.exists()
    main()
