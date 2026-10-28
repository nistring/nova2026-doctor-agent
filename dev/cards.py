"""Knowledge-card pipeline: one JSON part per presentation and stage, composed into agent/knowledge/cp.json.

  base    dev/data/cards/base/<id>.json    dev/build_knowledge.py  (평가목표집 + model knowledge)
  refine  dev/data/cards/refine/<id>.json  dev/build_guide.py      (기본진료수행지침 + guideline excerpts)
  edit    dev/data/cards/edit/<id>.json    dev/edit_cards.py       (exam/history curation)
  dx      dev/data/cards/dx/<id>.json      dev/build_dx.py         (per-cause follow-up questions; exam ids refer to the
                                                                    edit stage's numbering, so rebuild dx after re-editing)
  exambank dev/data/cards/exambank/<id>.json dev/build_exam_bank.py (per-cause diagnosis-specific exam maneuvers; added
                                                                    on top of any stage as `exam_bank`)

compose(id) layers base → refine → edit → dx; merge() writes every presentation of dev/data/cp_sections.json.
Earlier states of cp.json live in git history, not in copies.
"""
import re
import sys

from common import DEV, ROOT, read_json, write_json

sys.path.insert(0, str(ROOT / 'agent'))
from brain import BANNED, BUNDLED, COMMON_EXAMS  # noqa: E402

CARDS = ROOT / 'agent' / 'knowledge' / 'cp.json'
STAGES = ('base', 'refine', 'edit', 'dx')
SCALES = re.compile(r'PHQ|GAD|BDI|BAI|GDS|C-SSRS|AUDIT|CAGE|DAST|ISI|MDQ|척도|설문|선별도구|자살위험 평가|우울 선별|불안 평가')
OPEN_ENDED = re.compile(r'말씀해|설명해|이야기해|순서로|그 밖에|다른 (증상|불편|걱정)|어떻게 지내')


def part(stage, card_id):
    return DEV / 'data' / 'cards' / stage / f'{card_id}.json'


def order(card_id):
    return tuple(int(n) for n in card_id.split('-'))


def stub(section):
    """Card for a presentation with no generated base: name and the cause list from the 평가목표집 text."""
    causes = re.search(r'❑ 원인\n(.*?)\n❑', section['text'], re.S)
    lines = [re.sub(r'^\s*(\d+|[가-힣])\.\s*', '', l).strip() for l in (causes.group(1).splitlines() if causes else [])]
    return {'id': section['id'], 'name': section['name'], 'type': '진단', 'kw': [], 'stub': True,
            'causes': [{'dx': l, 'tier': ''} for l in lines if l], 'hx': [], 'exams': [], 'plan': {}}


def short_phrase(phrase):
    phrase = (phrase or '').strip()
    return re.sub(r'\s*\([^)]*\)\s*$', '', phrase) if len(phrase) > 30 else phrase


def guard(card):
    """Constraints the agent relies on: sendable exams and questions, no questionnaires among the tests."""
    exams = [{**e, 'phrase': short_phrase(e.get('phrase'))} for e in card.get('exams', [])]
    card['exams'] = [e for e in exams if 0 < len(e['phrase']) <= 30 and not any(w in e['phrase'] for w in BUNDLED)
                     and '활력' not in e['phrase']]
    card['exam_bank'] = [{'dx': b['dx'], 'exams': [short_phrase(e) for e in b['exams'] if 0 < len(short_phrase(e)) <= 30
                                                   and not any(w in e for w in BUNDLED) and '활력' not in e]}
                         for b in card.get('exam_bank', [])]
    if not card['exam_bank']:
        card.pop('exam_bank')
    card['hx'] = [h for h in card.get('hx', []) if 0 < len(h.get('q', '')) <= 30 and not any(w in h['q'] for w in BANNED)
                  and not OPEN_ENDED.search(h['q'])]  # open questions only get the main symptom back
    plan = dict(card.get('plan') or {})
    plan['tests'] = [t for t in plan.get('tests', []) if not (isinstance(t, dict) and SCALES.search(t.get('t', '')))]
    card['plan'] = plan
    exam_ids = {e['id'] for e in card['exams']} | set(COMMON_EXAMS)
    for note in card.get('dx_notes', []):
        if 'hx' in note:
            note['hx'] = [h for h in note['hx'] if 0 < len(h.get('q', '')) <= 30
                          and not any(w in h['q'] for w in BANNED) and not OPEN_ENDED.search(h['q'])]
        if 'exams' in note:
            note['exams'] = [e for e in note['exams'] if e in exam_ids]
    return card


def apply_edit(card, edit):
    card = {**card, 'exams': edit['exams'], 'hx': edit['hx']}
    known = {t.get('t') for t in card.get('plan', {}).get('tests', []) if isinstance(t, dict)}
    plan = dict(card.get('plan') or {})
    plan['tests'] = plan.get('tests', []) + [m for m in edit.get('moved_to_tests', []) if m.get('t') not in known]
    return {**card, 'plan': plan}


def compose(section, upto='edit'):  # 'dx' (v13 follow-ups) is not shipped: no measurable gain on the bench
    card_id = section['id']
    card = read_json(part('base', card_id)) if part('base', card_id).exists() else stub(section)
    if STAGES.index(upto) >= 1 and part('refine', card_id).exists():
        card = {**card, **read_json(part('refine', card_id)), 'id': card_id, 'name': section['name']}
        card.pop('stub', None)
    if STAGES.index(upto) >= 2 and part('edit', card_id).exists():
        card = apply_edit(card, read_json(part('edit', card_id)))
    if STAGES.index(upto) >= 3 and part('dx', card_id).exists():
        card = {**card, 'dx_notes': read_json(part('dx', card_id))['dx_notes']}
    if part('exambank', card_id).exists():
        card = {**card, 'exam_bank': read_json(part('exambank', card_id))['banks']}
    for n, exam in enumerate(card.get('exams', []), 1):
        exam['id'] = f'e{n}'
    for n, item in enumerate(card.get('hx', []), 1):
        item['id'] = f'h{n}'
    for n, note in enumerate(card.get('dx_notes', []), 1):  # follow-up question ids: d2-3 = cause 2, question 3
        for m, item in enumerate(note.get('hx', []), 1):
            item['id'] = f'd{n}-{m}'
    return card


def sections():
    return read_json(DEV / 'data' / 'cp_sections.json')


def merge(upto='edit'):
    cards = sorted((guard(compose(s, upto)) for s in sections()), key=lambda c: order(c['id']))
    write_json(CARDS, cards)
    done = {stage: sum(part(stage, c['id']).exists() for c in cards) for stage in STAGES}
    print(f"{CARDS.relative_to(ROOT)}: {len(cards)} presentations (parts: {done}), "
          f"{sum(len(c['exams']) for c in cards)} exams, {sum(len(c['hx']) for c in cards)} history items")
    return cards


if __name__ == '__main__':
    merge()
