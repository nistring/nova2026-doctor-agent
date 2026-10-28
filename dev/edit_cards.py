"""Edit the exam and history lists of every knowledge card so the agent's EXAM/SAY requests are accepted and useful.

  python dev/edit_cards.py [--only 25-1 28]   # edit parts → dev/data/cards/edit/<id>.json, then merge (dev/cards.py)
  python dev/edit_cards.py --only none        # merge only

- exams: single physical maneuvers / bedside observations, named like OSCE exam-table rows ("한국어 (English)").
  Questionnaires (PHQ-9, C-SSRS, ...) become history questions or P.tests; labs and calculations are dropped.
- hx: duplicates and near-duplicates merged, bundled questions split, off-topic items removed, interview order kept.
The public samples' exam-table row names are given only as a naming-style reference (not as content).
"""
import argparse
import glob
import json
import sys
from concurrent.futures import ThreadPoolExecutor

import cards as stages
from common import DEVKIT, ROOT, ask_json, endpoint, read_json, write_json

sys.path.insert(0, str(ROOT / 'agent'))
from brain import COMMON_EXAMS  # noqa: E402

STYLE = sorted({r['exam'] for p in glob.glob(str(DEVKIT / 'samples' / '*' / 'case.json'))
                for r in read_json(p)['exam_results']})

SYSTEM = f"""당신은 의사국가시험 실기(CPX) 출제·채점 전문가입니다. AI 의사가 쓰는 임상표현 지식 카드의 진찰 목록과 문진 목록을 다듬습니다.
가상 환자 시험에서 의사는 EXAM으로 "진찰 동작 하나"를 요청하고, 진찰 담당자는 그것을 진찰표의 한 행에 대응시켜 결과를 줍니다.
진찰이 아닌 요청(문진·설문·혈액/영상 검사·계산·기록)이나 여러 동작을 묶은 요청은 거절되고 턴만 씁니다.

[진찰표 행 이름의 예(이름 형식 참고용)]
{chr(10).join(STYLE)}

[공통 기본 진찰(카드에 넣지 않음)]
{', '.join(COMMON_EXAMS.values())}, 활력징후 측정

규칙:
1. exams: 이 임상표현의 원인을 가르는 진찰 8~12개. 각각 시진·촉진·타진·청진·신경학적 검사·침상 측정(기립성 혈압, 신체 계측, 사춘기 성숙도,
   인지기능 선별검사 등 진찰표에 들어갈 수 있는 것)처럼 진찰 담당자가 그 자리에서 수행하는 동작 하나.
   phrase는 위 예처럼 "한국어 이름 (English name)" 형식이며 30자를 넘으면 영어 부분을 뺍니다. "및", "/", "전체", "전신", "모두"를 쓰지 않습니다.
   region은 Skin|HEENT|Neck|Chest|Abdomen|Extremities|Neuro|Mental|Other, why는 감별 목적 한 줄.
   설문·척도(PHQ-9, GDS, AUDIT, C-SSRS, 자살위험 평가, 통증 척도 등)는 exams에서 빼고, 환자에게 묻는 질문이면 hx로, 검사 계획이면 moved_to_tests로 옮깁니다.
   혈액·영상 검사, 계산(BMI 계산 등)은 빼고 필요하면 moved_to_tests로 옮깁니다. 공통 기본 진찰과 겹치는 것은 뺍니다.
2. hx: 30~40개, 진료 순서(주 증상 → 동반 증상 → 감별 질문 → 과거력·약물·알레르기 → 가족력 → 사회력 → 계통 문진)로.
   같은 것을 묻는 항목은 하나로 합치고, 한 질문에 두 가지를 묻는 항목은 나누되 짝 증상("열이나 오한")은 허용합니다.
   이 임상표현과 관계없는 항목은 지우고, 빠진 핵심 감별 질문·red flag가 있으면 더합니다.
   q는 30자 이하 존댓말 질문 하나("전부", "모두", "모든", "다 말씀" 금지). tag는 필수/감별/위험, field는 cc/hpi/pmh/fh/sh/ros.
3. 기존 카드의 의학 내용을 바꾸지 말고 목록만 다듬습니다.
JSON: {{"exams": [{{"id": "e1", "phrase": "", "region": "", "why": ""}}], "hx": [{{"id": "h1", "field": "", "tag": "", "item": "", "q": ""}}],
 "moved_to_tests": [{{"t": "검사·설문 이름", "why": "이유"}}]}}"""


def edit(card):
    user = (f"임상표현 {card['id']}. {card['name']} ({card.get('type', '진단')}형)\n원인: "
            + ', '.join(c['dx'] for c in card.get('causes', [])) + '\n\n[현재 exams]\n'
            + json.dumps(card.get('exams', []), ensure_ascii=False) + '\n\n[현재 hx]\n' + json.dumps(card.get('hx', []), ensure_ascii=False))
    try:
        result = ask_json(endpoint('sol'), SYSTEM, user, effort='medium', max_tokens=24000)
    except Exception as exc:
        print(f"error {card['id']}: {exc}")
        return
    if not result or not result.get('hx') or not result.get('exams'):
        print(f"FAILED {card['id']} {card['name']}")
        return
    write_json(stages.part('edit', card['id']), result)
    print(f"ok {card['id']} {card['name']}: exams {len(card.get('exams', []))}→{len(result['exams'])}, "
          f"hx {len(card.get('hx', []))}→{len(result['hx'])}, moved {[m['t'] for m in result.get('moved_to_tests', [])]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--only', nargs='*')
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    cards = [stages.compose(s, upto='refine') for s in stages.sections()]
    todo = [c for c in cards if (args.only and c['id'] in args.only)
            or (not args.only and not stages.part('edit', c['id']).exists())]
    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(edit, todo))
    stages.merge()


if __name__ == '__main__':
    main()
