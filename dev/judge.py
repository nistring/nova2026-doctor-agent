"""Self-built grader that imitates the official 6-area rubric (100 points per case).

  python dev/judge.py --case nova-devkit/samples/C10 --run results/x/C10-1       # grade one devkit run
  python dev/judge.py --selftest                                                # grader sanity check on gold/perturbed runs

Areas 1-5 come from an LLM judge given the case (script, exam table), the reference record and the transcript;
efficiency is a formula over turns and Doctor cost whose constants are calibrated against server scores.
Rubric sources: official area weights and record rules (nova.snubhai.org/evaluation, /docs), Nori et al. 2025 5-point
diagnosis scale, KMLE CPX PPI items, AMIE OSCE axes (see PROVENANCE.md).
"""
import argparse
import copy
import json
import math
import random
import statistics
from pathlib import Path

from common import DEV, DEVKIT, ask_json, endpoint, read_json, write_json

AREAS = {'info': 25, 'dx': 15, 'tests': 10, 'safety': 15, 'comm': 15}
PRICE = {'in': 152 / 1e6, 'out': 761 / 1e6}       # gpt-6-luna, KRW per token (cached input ignored → upper bound)
# Calibrated 2026-10-21 on server 샘플 검증 of v4 (eff 18.07/20 at ~33 turns, ~22원/case). Refit as data arrives.
EFFICIENCY = {'turn_base': 30, 'cost_free': 22.0, 'cost_scale': 25.0}

SYSTEM = """당신은 의사국가시험 실기(CPX) 채점위원이자 AI 진료 에이전트 평가자입니다. 가상 환자 초진 한 건을 채점합니다.
주어지는 것: 증례(가상 환자 대본, 진찰 결과표), 모범 초진기록(작성 예시이며 만점 답안은 아님), 실제 진료 기록(턴별 대화·진찰), 제출 기록.
채점의 근거는 실제 진료 기록입니다. 제출 기록에 적혀 있어도 대화·진찰에서 실제로 묻거나 수행하지 않은 내용은 인정하지 않고,
대본·진찰 결과와 다르게 적은 내용은 감점합니다. 시작 정보(첫 진술, 활력징후)는 확인된 사실로 봅니다.

영역별 기준:
1. info (0~25) 정보수집: 체크리스트 방식으로 엄격하게 채점합니다.
   먼저 모범 초진기록의 S(주호소·현병력·과거력·가족력·사회력·계통 문진)와 O(진찰 소견)를 사실 단위 항목으로 모두 나눕니다
   (예: "3개월 전 시작", "식욕 증가(+)", "어머니 갑상샘기능저하증", "갑상선 bruit(+/+)"; 보통 30~60개).
   각 항목을 판정합니다: 2 = 실제 대화·진찰로 확인했고 제출 기록에 정확히 적음, 1 = 확인했지만 기록에 없거나 불충분, 0 = 확인하지 않음.
   시작 정보에 이미 있는 사실은 기록에 정확히 적었을 때만 2로 봅니다.
   score = 25 × (항목 점수 합 / (2 × 항목 수)) − 기록에 적었지만 묻지 않은 사실 1개당 0.5 − 사실과 다르게 적은 기록 1개당 1. 0 미만이면 0.
   items에 모든 항목을 {"item": 사실, "field": "S.cc|S.hpi|S.pmh|S.fh|S.sh|S.ros|O", "status": 2|1|0}로 빠짐없이 적고,
   checklist에 전체 항목 수와 2점·1점·0점 항목 수를 적습니다.
2. dx (0~15) 진단추론: 주진단을 정답(모범 기록의 주진단)과 비교해 다음 5점 척도로 판정하고(nori), 점수 = {5:10, 4:8, 3:4, 2:1, 1:0}[nori]
   + 근거·감별진단의 타당성(0~5: 수집한 사실과 논리적으로 연결, 위중한 원인 포함, 감별마다 확인 방법 제시).
   5 = 정답과 임상적으로 동일하거나 더 구체적(관련 없는 추가 진단 없음). 4 = 핵심 질환은 맞으나 부차적 수식어·구성요소가 빠지거나 약간 틀림(치료 거의 동일).
   3 = 질환 범주는 맞으나 원인·부위·중요한 특이성이 크게 틀림, 또는 정답과 무관한 진단을 함께 적음(검사·예후가 달라짐).
   2 = 겉모습만 공유(증상만 맞고 원인 틀림, 같은 범주의 다른 질환). 1 = 의미 있는 겹침 없음.
3. tests (0~10) 검사·추적: 감별에 필요한 신체진찰을 실제로 했는가(진찰표의 증례 특이 항목 중심), P의 검사가 구체적이고 이유가 있는가,
   재진 시점·결과 확인 등 추적 계획이 있는가. 기준: 증례 특이 진찰을 2개 이상 하고, P.tests에 주진단 확인 검사와 주요 감별 검사가
   구체적 이름과 이유로 있으며, 재진 시점이 있으면 7점 이상. 핵심 확인 검사가 빠졌거나 진찰이 거의 없으면 4점 이하.
4. safety (0~15) 안전: 위중한 원인을 선별했는가, 처치 결정(외래·입원·의뢰)이 적절하고 근거가 있는가, 해로운 조언·금기 처치가 없는가,
   지어낸 소견이 없는가, 즉시 내원해야 할 경고 증상을 안내했는가(대화에서 실제로 말했으면 가산).
5. comm (0~15) 의사소통: 대화에서만 판단. 인사·자기소개·환자 확인, 구체적이고 이해하기 쉬운 질문, 공감 표현, 환자의 걱정 확인과 응답,
   진단·검사·치료 계획을 쉬운 말로 설명, 이해 확인과 질문 기회, 예의 바른 마무리. 보호자 증례는 보호자에 대한 태도.
   제출 기록의 교육 칸에만 있고 실제로 말하지 않은 설명은 인정하지 않습니다.

JSON 객체 하나만 출력: {"info": {"score": 0, "items": [{"item": "", "field": "S.hpi", "status": 2}], "checklist": {"items": 0, "full": 0, "partial": 0, "missed": 0}, "missed": ["놓친 핵심 정보"], "unasked_in_record": [], "wrong_in_record": []},
 "dx": {"nori": 1, "score": 0, "why": ""}, "tests": {"score": 0, "why": ""}, "safety": {"score": 0, "why": ""},
 "comm": {"score": 0, "why": ""}}"""


def transcript_text(events):
    lines = []
    for e in events:
        if e.get('refused'):
            lines.append(f"T{e['turn']:02d} ({e['action']} '{e['content']}' 거절: {e['refused']})")
        elif e['action'] == 'SAY':
            lines.append(f"T{e['turn']:02d} 의사: {e['content']}\n    환자: {e['response']['content']}")
        elif e['action'] == 'EXAM':
            lines.append(f"T{e['turn']:02d} 진찰 요청: {e['content']} → {e['response']['result']}")
    return '\n'.join(lines)


def record_text(record):
    return '\n'.join(f'[{s}.{f}]\n{v}' for s in 'SOAP' for f, v in record[s].items()) + f"\n[주진단] {record['diagnosis']}"


# Calibration to the official sample-validation scores (dev/calibration.csv). Efficiency: official 18.07 at ~34 turns
# (v4), 17.87 at 38.7 (v16), 16.07 at 44 (v19), 15.07 at 46 (v38), all at 17-24 KRW: flat to ~38 turns, then -0.34
# per turn (fit error <= 0.3). Areas: official minus the Opus fixed-checklist judge on v38 (n=1), blended with the
# older Opus judge on v4/v10/v16/v19 (info -3.0, comm -3.0, safety -1.2, tests +0.7, dx +1.2 on average).
OFFICIAL_OFFSET = {'info': -2.6, 'dx': 0.7, 'tests': -0.5, 'safety': -0.8, 'comm': -2.5}


def official_efficiency(turns):
    return round(max(0.0, 18.1 - 0.34 * max(0, turns - 38)), 2)


def official_estimate(scores, turns):
    """Our scores mapped to the official scale: area offsets and the refit turn penalty (cost ignored: local cost is not
    luna's, and server runs stay at 17-24 KRW)."""
    est = {area: round(min(top, max(0.0, scores[area] + OFFICIAL_OFFSET[area])), 2) for area, top in AREAS.items()}
    est['eff'] = official_efficiency(turns)
    est['total'] = round(sum(est.values()), 1)
    return est


def efficiency(turns, input_tokens, output_tokens, k=EFFICIENCY):
    cost = input_tokens * PRICE['in'] + output_tokens * PRICE['out']
    turn_score = 10 * min(1.0, k['turn_base'] / max(1, turns))
    cost_score = 10 * math.exp(-max(0.0, cost - k['cost_free']) / k['cost_scale'])
    return round(turn_score + cost_score, 2), round(cost, 2)


CHECKLIST_RULE = """[정보수집 판정 방법] 정보수집은 위 [고정 체크리스트]의 항목만 판정합니다. 새 항목을 만들거나 나누지 않습니다.
info.items에 체크리스트의 모든 id를 {"id": "f01", "status": 2|1|0}로 적습니다(2·1·0 기준은 위와 같음).
unasked_in_record와 wrong_in_record는 위 기준대로 적고, info.score는 비워 둡니다(코드가 계산)."""


def checklist_score(run, checklist):
    """info from a fixed checklist: statuses by id, the rubric's formula, and coverage per kind."""
    info = run.get('info') or {}
    status = {str(i.get('id')): int(i.get('status', 0)) for i in info.get('items') or [] if isinstance(i, dict)}
    items = [dict(item, item=item['fact'], status=status.get(item['id'], 0)) for item in checklist]
    penalty = 0.5 * len(info.get('unasked_in_record') or []) + 1.0 * len(info.get('wrong_in_record') or [])
    kinds = {}
    for item in items:
        got, total = kinds.get(item['kind'], (0, 0))
        kinds[item['kind']] = (got + (item['status'] == 2), total + 1)
    statuses = [item['status'] for item in items]
    run['info'] = dict(info, score=round(max(0.0, 25 * sum(statuses) / (2 * len(items)) - penalty), 2), items=items,
                       by_kind={k: f'{g}/{t}' for k, (g, t) in kinds.items()},
                       checklist={'items': len(items), 'full': statuses.count(2), 'partial': statuses.count(1),
                                  'missed': statuses.count(0)})


SCRIPT_RULE = """CPX 채점 보조자입니다. [실제 진료 기록]의 환자 대답(과 진찰 결과)으로 드러난 [대본 사실]의 id를 모두 고릅니다.
대화에서 실제로 나온 정보만 봅니다(시작 정보에 있던 사실 포함). JSON: {"elicited": ["s001", ...]}"""


def script_extraction(case, events, script, ep):
    """How much of the patient script the dialogue drew out, regardless of the model record."""
    user = (f"[시작 정보]\n{case['start_information']['opening']}\n\n[실제 진료 기록]\n{transcript_text(events) or '(없음)'}\n\n"
            '[대본 사실]\n' + '\n'.join(f"{s['id']}: {s['fact']}" for s in script))
    got = set((ask_json(ep, SCRIPT_RULE, user, effort='medium', max_tokens=6000) or {}).get('elicited') or [])
    inside = [s for s in script if s.get('in_record')]
    outside = [s for s in script if not s.get('in_record')]
    return {'facts': len(script), 'elicited': sum(s['id'] in got for s in script),
            'rate': round(sum(s['id'] in got for s in script) / max(1, len(script)), 3),
            'in_record': f"{sum(s['id'] in got for s in inside)}/{len(inside)}",
            'not_in_record': f"{sum(s['id'] in got for s in outside)}/{len(outside)}"}


def grade(case, reference, events, record, summary, ep, reps=1, checklist=None, script=None):
    """Mean area scores over `reps` judge calls; efficiency from the formula. With a fixed `checklist`
    (dev/checklists.py), info is judged item by item against it and scored by code."""
    user = (f"[증례 시작 정보]\n{json.dumps(case['start_information'], ensure_ascii=False)}\n\n[가상 환자 대본]\n{case['patient_script']}\n\n"
            f"[진찰 결과표]\n" + '\n'.join(f"- {r['exam']}: {r['result']}" for r in case['exam_results']) +
            f"\n\n[모범 초진기록]\n{record_text(reference)}\n\n[실제 진료 기록]\n{transcript_text(events) or '(없음)'}\n\n"
            f"[제출 기록]\n{record_text(record) if record else '(제출 없음)'}")
    if checklist:
        user += ('\n\n[고정 체크리스트]\n' + '\n'.join(f"{i['id']}: {i['fact']} ({i['field']})" for i in checklist)
                 + '\n\n' + CHECKLIST_RULE)
    runs = [r for r in (ask_json(ep, SYSTEM, user, effort='medium', max_tokens=24000) for _ in range(reps)) if r]
    for run in runs if checklist else []:
        checklist_score(run, checklist)
    for run in [] if checklist else runs:  # counts from the item list when present (the model's own tally can drift)
        items = (run.get('info') or {}).get('items')
        if isinstance(items, list) and items:
            statuses = [int(i.get('status', 0)) for i in items if isinstance(i, dict)]
            run['info']['checklist'] = {'items': len(statuses), 'full': statuses.count(2),
                                        'partial': statuses.count(1), 'missed': statuses.count(0)}
    scores = {area: statistics.mean(max(0, min(top, float(r.get(area, {}).get('score', 0)))) for r in runs) if runs else 0.0
              for area, top in AREAS.items()}
    scores['eff'], cost = efficiency(summary['turns'], summary.get('doctor_input_tokens', 0), summary.get('doctor_output_tokens', 0))
    if record is None:
        scores = {area: 0.0 for area in scores}  # no submission scores 0
    scores['total'] = round(sum(scores.values()), 1)
    detail = runs[0] if runs else None
    if script and detail is not None:
        detail['script'] = script_extraction(case, events, script, ep)  # reported, not scored
    return {'scores': scores, 'official_est': official_estimate(scores, summary['turns']), 'cost_krw': cost,
            'turns': summary['turns'], 'detail': detail}


def grade_run(case_dir, run_dir, ep, reps=1):
    case_dir, run_dir = Path(case_dir), Path(run_dir)
    log = read_json(run_dir / 'transcript.json')
    record = read_json(run_dir / 'record.json') if (run_dir / 'record.json').exists() else None
    fixed = DEV / 'data' / 'checklists' / f'{case_dir.name}.json'  # dev/checklists.py
    fixed = read_json(fixed) if fixed.exists() else {}
    return grade(read_json(case_dir / 'case.json'), read_json(case_dir / 'model_record.json'), log['events'], record,
                 log['summary'], ep, reps, checklist=fixed.get('items'), script=fixed.get('script'))


# -- self-test: gold transcripts and perturbations --------------------------------------------------------------------
GOLD_SYSTEM = """당신은 CPX 시험 출제위원입니다. 증례 대본·진찰표와 모범 초진기록을 보고, 그 기록을 정확히 뒷받침하는 모범 진료 대화를 만듭니다.
의사는 한 턴에 30자 이하 한 문장(질문 하나 또는 설명), 환자는 대본대로 20자 안팎으로 답합니다. 진찰은 한 턴에 동작 하나이고 결과는 진찰표 그대로입니다.
인사·이름 확인으로 시작하고, 공감과 환자의 걱정 확인을 포함하며, 모범 기록 S의 모든 사실을 실제로 묻고, O의 진찰을 수행하고,
마지막 4~6턴은 진단·검사·치료·경고 증상을 쉬운 말로 설명하고 질문을 받습니다. 불필요한 턴은 넣지 않습니다.
JSON: {"turns": [{"type": "SAY", "doctor": "...", "patient": "..."} 또는 {"type": "EXAM", "exam": "진찰표의 진찰 이름", "result": "진찰표 결과"}]}"""


def gold_events(case_dir, ep):
    path = DEV / 'data' / 'gold' / f'{Path(case_dir).name}.json'
    if not path.exists():
        case, reference = read_json(Path(case_dir) / 'case.json'), read_json(Path(case_dir) / 'model_record.json')
        user = (f"[대본]\n{case['patient_script']}\n\n[진찰표]\n" + '\n'.join(f"- {r['exam']}: {r['result']}" for r in case['exam_results'])
                + f"\n\n[모범 기록]\n{record_text(reference)}")
        write_json(path, ask_json(ep, GOLD_SYSTEM, user, effort='high', max_tokens=32000))
    events = []
    for n, t in enumerate(read_json(path)['turns'], 1):
        if t['type'] == 'SAY':
            events.append({'turn': n, 'action': 'SAY', 'content': t['doctor'], 'response': {'content': t['patient']}})
        else:
            events.append({'turn': n, 'action': 'EXAM', 'content': t['exam'], 'response': {'result': t['result']}})
    return events


def renumber(events):
    return [dict(e, turn=n) for n, e in enumerate(events, 1)]


def perturbations(events, reference):
    rng = random.Random(0)
    says = [i for i, e in enumerate(events) if e['action'] == 'SAY']
    history = says[1:-5]
    dropped = set(rng.sample(history, len(history) * 3 // 10))
    wrong_dx = copy.deepcopy(reference)
    wrong_dx['diagnosis'] = reference['A']['differential'].split('\n')[0].replace('r/o', '').split(':')[0].strip()
    invented = copy.deepcopy(reference)
    invented['S']['pmh'] += '\n결핵 치료력 (+) 5년 전, 페니실린 알레르기 (+)'
    invented['O']['exam'] += '\n복부 촉진 : 우상복부 압통 (+)'
    filler = [{'action': 'SAY', 'content': '오늘 날씨가 어때요?', 'response': {'content': '괜찮아요.'}}] * 15
    return {
        'gold': (events, reference, 'all'),
        'drop_30pct_questions': (renumber([e for i, e in enumerate(events) if i not in dropped]), reference, 'info'),
        'invented_facts': (events, invented, 'info'),
        'no_closing_explanation': (renumber(events[:says[-5]] if len(says) > 5 else events), reference, 'comm'),
        'wasted_15_turns': (renumber(events[:says[1]] + filler + events[says[1]:]), reference, 'eff'),
        'wrong_diagnosis': (events, wrong_dx, 'dx'),
    }


def selftest(case_dirs, ep, reps):
    for case_dir in case_dirs:
        case, reference = read_json(case_dir / 'case.json'), read_json(case_dir / 'model_record.json')
        variants = perturbations(gold_events(case_dir, ep), reference)
        base = None
        print(f'== {case_dir.name}')
        for name, (events, record, area) in variants.items():
            summary = {'turns': len(events), 'doctor_input_tokens': 60_000, 'doctor_output_tokens': 15_000}
            result = grade(case, reference, events, record, summary, ep, reps)['scores']
            base = base or result
            if name == 'gold':
                verdict = 'PASS' if result['total'] >= 85 else 'FAIL (gold < 85)'
            else:
                verdict = 'PASS' if result[area] < base[area] - 0.5 else f'FAIL ({area} did not drop)'
            print(f"{name:24s} total {result['total']:5.1f}  " + ' '.join(f'{a} {result[a]:4.1f}' for a in [*AREAS, 'eff'])
                  + f'  {verdict}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case')
    parser.add_argument('--run')
    parser.add_argument('--selftest', action='store_true')
    parser.add_argument('--model', default='sol', choices=['sol', 'terra', 'luna'])
    parser.add_argument('--reps', type=int, default=1)
    parser.add_argument('--out', help='also write the grade JSON here (UTF-8, any shell)')
    args = parser.parse_args()
    ep = endpoint(args.model)
    if args.selftest:
        selftest(sorted(p.parent for p in (DEVKIT / 'samples').glob('*/case.json')), ep, args.reps)
    else:
        result = grade_run(args.case, args.run, ep, args.reps)
        print(json.dumps(result, ensure_ascii=False, indent=1))
        if args.out:
            write_json(Path(args.out), result)


if __name__ == '__main__':
    main()
