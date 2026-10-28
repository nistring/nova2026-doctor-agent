"""Print the grader's 정보수집 checklist for a bench tag: every item with its status per run, the items missed most
often, and the grader's notes per area. Also written to results/<tag>/checklist.md.

  python dev/show_checklist.py results/local/v34-ext
Status: ✅ asked/examined and recorded · △ asked but recorded poorly · ✗ not asked/examined.
"""
import sys
from collections import defaultdict
from pathlib import Path

from common import ROOT, read_json

MARK = {2: '✅', 1: '△', 0: '✗'}
AREAS = {'dx': '진단추론', 'tests': '검사·추적', 'safety': '안전', 'comm': '의사소통'}


def grams(text):
    text = ''.join(text.split())
    return {text[i:i + 2] for i in range(len(text) - 1)}


def same(a, b):
    """Item texts are written anew by the grader each run: match them by character-bigram overlap."""
    x, y = grams(a), grams(b)
    return bool(x and y) and len(x & y) / len(x | y) >= 0.45


def case_table(runs):
    rows = []  # [item, field, {run: status}]
    for run in runs:
        for item in ((run.get('detail') or {}).get('info') or {}).get('items') or []:
            if not isinstance(item, dict) or not item.get('item'):
                continue
            row = next((r for r in rows if same(r[0], item['item'])), None)
            if row is None:
                row = [item['item'], item.get('field', ''), {}]
                rows.append(row)
            row[2][run['run']] = int(item.get('status', 0))
    return rows


def main():
    base = Path(sys.argv[1])
    results = [r for r in read_json(base / 'summary.json') if r.get('detail')]
    out = []
    for case in sorted({r['case'] for r in results}):
        runs = [r for r in results if r['case'] == case]
        names = [Path(r['run']).name for r in runs]
        rows = case_table(runs)
        if not rows:
            out.append(f'## {case}\n\n(no item list in these results: grade with the current dev/judge.py)\n')
            continue
        counts = [((r['detail'].get('info') or {}).get('checklist') or {}) for r in runs]
        out.append(f'## {case} — ' + ', '.join(f"{n}: {c.get('full', 0)}✅ {c.get('partial', 0)}△ / {c.get('items', 0)}"
                                                 for n, c in zip(names, counts)))
        out.append('\n| 항목 | 칸 | ' + ' | '.join(names) + ' |\n|---|---|' + '---|' * len(names))
        for text, field, status in sorted(rows, key=lambda r: (r[1], r[0])):
            out.append(f'| {text} | {field} | ' + ' | '.join(MARK.get(status.get(r["run"]), '·') for r in runs) + ' |')
        missed = sorted((r for r in rows if sum(1 for v in r[2].values() if v < 2) >= max(1, len(runs) // 2 + 1)),
                        key=lambda r: -sum(1 for v in r[2].values() if v == 0))
        out.append('\n**Mostly missed:** ' + (', '.join(r[0] for r in missed[:20]) or '—'))
        notes = defaultdict(list)
        for run in runs:
            for key, name in AREAS.items():
                why = (run['detail'].get(key) or {}).get('why')
                if why:
                    notes[name].append(why)
        out.append('\n' + '\n'.join(f'- **{name}:** ' + ' / '.join(texts) for name, texts in notes.items()) + '\n')
    text = '\n'.join(out)
    (base / 'checklist.md').write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
