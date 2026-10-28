"""Exam bank: per presentation and cause, the diagnosis-specific physical-exam maneuvers an exam table would list
(knowledge transfer from Opus; luna's own picks were often not table rows: neck circumference, Babinski, visual fields).

  python dev/build_exam_bank.py              # missing parts -> dev/data/cards/exambank/<id>.json, then merge (dev/cards.py)
  python dev/build_exam_bank.py --only 40 28

The public samples are not shown to the model. One call per presentation.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor

import cards
from common import ROOT, ask_json, endpoint, read_json, write_json

import sys
sys.path.insert(0, str(ROOT / 'agent'))
from brain import COMMON_EXAMS  # noqa: E402

SYSTEM = """당신은 CPX(의사국가시험 실기) 신체진찰 출제위원입니다. JSON 하나만 출력합니다.
임상표현 하나와 그 원인 진단 목록을 받습니다. 원인마다, 그 진단을 확인하거나 감별하려고 외래에서 시행하는 진단 특이 신체진찰
동작을 중요한 순서로 4~8개 씁니다. 각 동작은 진찰 결과표의 한 행이 되는 구체적인 동작 하나입니다:
"한국어 이름 (English name)" 형식, 30자 이하(넘으면 영어를 뺌), 여러 동작·부위 전체를 묶지 않음, 활력징후 측정은 제외.
한 진찰표 행을 하위 소견으로 쪼개지 않습니다(안징후 검사에 안검 지연·안구 돌출이, Tanner 평가에 고환 용적이, 신체 계측에 키·체중·
머리둘레가 들어 있음: 같은 행을 두 번 쓰면 같은 결과가 나와 턴만 씀).
원인 이름에 여러 질환이 묶여 있으면(예: "알코올·약물 사용 또는 수면제 의존") 그 임상표현에서 가장 흔한 것 위주로 씁니다.
아래 [공통 진찰] 중 그 진단에서 특히 중요한 것(예: 갑상샘항진증의 심부건반사·사지 근력)은 넣고, 나머지는 코드가 따로 합니다. 검사(혈액·영상)는 넣지 않습니다.
{"banks": [{"dx": "원인 이름(목록 그대로)", "exams": ["갑상선 잡음 청진 (Thyroid bruit)", ...]}]}"""


def build(card):
    user = (f"[임상표현] {card['id']}. {card['name']}\n[원인]\n" + '\n'.join(f"- {c['dx']}" for c in card.get('causes', [])[:12])
            + '\n[카드 진찰]\n' + '\n'.join(f"- {e['phrase']}" for e in card.get('exams', []))
            + '\n[공통 진찰]\n' + ', '.join(COMMON_EXAMS.values()))
    try:
        answer = ask_json(endpoint('sol'), SYSTEM, user, effort='medium', max_tokens=16000) or {}
    except Exception as exc:  # an overloaded API: this card stays missing and is built on the next run
        print(f"{card['id']}: {type(exc).__name__}")
        return card['id'], 0, 0
    banks = [{'dx': str(b.get('dx', '')).strip(), 'exams': [str(e).strip() for e in b.get('exams') or [] if str(e).strip()]}
             for b in answer.get('banks') or [] if isinstance(b, dict) and b.get('dx')]
    if banks:
        write_json(cards.part('exambank', card['id']), {'banks': banks})
    return card['id'], len(banks), sum(len(b['exams']) for b in banks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--only', nargs='*')
    args = parser.parse_args()
    deck = read_json(cards.CARDS)
    todo = [c for c in deck if (c['id'] in args.only if args.only else not cards.part('exambank', c['id']).exists())]
    with ThreadPoolExecutor(4) as pool:
        for card_id, n, m in pool.map(build, todo):
            print(f'{card_id}: {n} causes, {m} exams' if n else f'{card_id}: FAILED')
    cards.merge()


if __name__ == '__main__':
    main()
