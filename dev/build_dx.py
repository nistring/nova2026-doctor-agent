"""Per-cause follow-up questions for every knowledge card (stage 'dx'): what to ask once a cause leads the differential.

  python dev/build_dx.py [--only 28 40]   # dx parts → dev/data/cards/dx/<id>.json, then merge (dev/cards.py)

Each cause gets aliases (to match the Doctor's diagnosis names in code), the key findings line, 3-6 history questions
that confirm it and gauge its severity, course and complications, and pointers to exams already on the card.
The agent shows the questions of the top 1-2 diagnoses next to the remaining checklist, so a light Doctor model does not
have to come up with these follow-ups itself.
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor

import cards as stages
from common import ROOT, ask_json, endpoint, write_json

sys.path.insert(0, str(ROOT / 'agent'))
from brain import COMMON_EXAMS  # noqa: E402

SYSTEM = """당신은 의사국가시험 실기(CPX) 출제·채점 전문가입니다. AI 의사가 쓰는 임상표현 지식 카드에 원인별 확인 문진을 더합니다.
CPX 채점표는 주호소의 공통 문진에 더해, 가장 유력한 진단을 확인하고 그 정도·경과·합병증을 묻는 질문을 채점합니다
(예: 갑상샘항진증이면 손떨림·눈 증상·목 앞 부음·근력 저하, 수면제 의존이면 하루 총 복용량·여러 병원 처방·감량 시도·금단 증상).
AI 의사는 유력한 진단이 정해지면 그 원인의 질문 목록을 받아 묻습니다.

규칙:
1. dx_notes: [원인]의 각 원인마다 하나(거의 같은 원인은 합침). dx는 교과서·ICD의 일반적인 질환명.
2. aliases: 의사가 쓰는 진단명 안에서 이 원인을 찾을 짧은 부분 문자열 2~6개(한국어 표기 변형과 영어, 예: "갑상샘항진", "갑상선기능항진",
   "그레이브스", "Graves", "hyperthyroid"). "질환", "장애", "증후군"처럼 다른 원인에도 들어가는 말만으로 된 것은 쓰지 않습니다.
3. key: 이 원인을 시사하는 핵심 문진·진찰 소견 한 줄([현재 dx_notes]에 있으면 다듬어 씀).
4. hx: 이 원인이 가장 유력할 때 확인할 문진 3~6개. 전형 증상(계통별), 정도·빈도·양, 경과, 유발 요인, 이전 치료, 합병증·위험 징후 순.
   약물·물질이 원인이면 종류·하루 총량·기간·구한 경로(여러 병원 처방)·끊거나 줄인 시도·금단 증상을 묻습니다.
   [현재 hx]에 이미 있는 질문과 같은 것은 쓰지 않습니다. q는 30자 이하 존댓말 질문 하나("전부", "모두", "모든", "다 말씀" 금지),
   짝 증상("열이나 오한")만 묶습니다. item은 채점표 항목 이름(짧게).
5. exams: 이 원인을 확인하거나 배제하는 진찰을 [진찰 목록]의 id로 0~3개(목록에 없는 진찰은 쓰지 않음).
JSON: {"dx_notes": [{"dx": "", "aliases": [""], "key": "", "hx": [{"item": "", "q": ""}], "exams": ["e1", "c5"]}]}"""


def build(card):
    exams = [f'{k}: {v}' for k, v in COMMON_EXAMS.items()] + [f"{e['id']}: {e['phrase']}" for e in card.get('exams', [])]
    user = (f"임상표현 {card['id']}. {card['name']} ({card.get('type', '진단')}형)\n[원인]\n"
            + '\n'.join(f"- {c['dx']} ({c.get('tier', '')})" for c in card.get('causes', []))
            + '\n\n[현재 dx_notes]\n' + json.dumps(card.get('dx_notes', []), ensure_ascii=False)
            + '\n\n[현재 hx]\n' + '\n'.join(h['q'] for h in card.get('hx', []))
            + '\n\n[진찰 목록]\n' + '\n'.join(exams))
    try:
        result = ask_json(endpoint('sol'), SYSTEM, user, effort='medium', max_tokens=24000)
    except Exception as exc:
        print(f"error {card['id']}: {exc}")
        return
    notes = (result or {}).get('dx_notes')
    if not isinstance(notes, list) or not notes:
        print(f"FAILED {card['id']} {card['name']}")
        return
    write_json(stages.part('dx', card['id']), {'dx_notes': notes})
    print(f"ok {card['id']} {card['name']}: {len(card.get('dx_notes', []))}→{len(notes)} causes, "
          f"{sum(len(n.get('hx', [])) for n in notes)} questions")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--only', nargs='*')
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--no-merge', action='store_true', help='write parts only (agent/knowledge/cp.json unchanged)')
    args = parser.parse_args()
    cards = [stages.compose(s, upto='edit') for s in stages.sections()]
    todo = [c for c in cards if not c.get('stub') and ((args.only and c['id'] in args.only)
            or (not args.only and not stages.part('dx', c['id']).exists()))]
    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(build, todo))
    if not args.no_merge:
        stages.merge(upto='dx')


if __name__ == '__main__':
    main()
