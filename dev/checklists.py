"""Fixed grading checklists: one atomic-fact list per case, built once from the patient script, the model record and
the exam table, so every run of a case is graded against the same items (the judge used to re-split the model
record per run: 45-62 items for the same case).

  python dev/checklists.py                 # missing checklists for the samples and the synthetic cases
  python dev/checklists.py --only C10 S03  # (re)build selected cases

Output: dev/data/checklists/<case>.json = {"case": id, "items": [{"id", "fact", "field", "block", "kind"}],
  "script": [{"id", "fact", "topic", "in_record"}]}  -- the second list: every fact of the patient script, for the
  script-extraction rate (how much of the script the dialogue drew out, graded record or not)
"""
import argparse
from concurrent.futures import ThreadPoolExecutor

from common import DEV, DEVKIT, ask_json, endpoint, read_json, write_json
from judge import record_text

OUT = DEV / 'data' / 'checklists'
SYSTEM = """당신은 CPX 채점표 작성위원입니다. 증례의 정보수집 채점표를 사실 단위로 만듭니다. JSON 하나만 출력합니다.
주어지는 것: 시작 정보, 가상 환자 대본(문단별 주제로 쓰임), 모범 초진기록, 진찰 결과표.
- 모범 초진기록 S(주호소·현병력·과거력·가족력·사회력·계통 문진)와 O(진찰 소견)에 있는 사실을 빠짐없이 채점 단위로 씁니다
  (예: "식욕 증가(+)", "어머니 갑상샘기능저하증", "혈변(-)", "갑상선 bruit(+/+)"). 한 질문·한 진찰로 함께 얻는 밀접한 사실
  (예: "56kg→49kg, 약 7kg 감소", "눈 뻑뻑함·충혈감", "슬개건 3+/3+, 아킬레스건 2+/2+")은 한 항목으로 묶습니다. 보통 증례당 40~70개.
- 대본에 있지만 모범 기록에 없는 사실은 넣지 않습니다. 활력징후는 한 항목입니다.
- field: S.cc | S.hpi | S.pmh | S.fh | S.sh | S.ros | O
- block: 그 사실이 나오는 대본 문단의 주제를 짧게(예: "경과", "갑상샘 전형 증상", "소화기", "생리", "눈·목", "가족력", "걱정")
- kind: cc(시작 정보에 이미 있는 사실) | dx(정답 진단의 전형 증상·소견, 원인 노출) | diff(감별을 가르는 양성·음성 소견) |
  routine(과거력·약·알레르기·수술·가족력·흡연·음주·직업·동거·월경/임신) | ice(환자의 걱정·기대·궁금증) | exam(O의 진찰 소견)
{"items": [{"fact": "", "field": "", "block": "", "kind": ""}]}"""


SCRIPT = """CPX 채점 보조자입니다. [환자 대본]을 의사가 문진으로 얻을 수 있는 사실 단위로 빠짐없이 나눕니다(한 질문으로 함께 얻는
밀접한 사실은 한 항목). 시작 정보에 이미 있는 사실도 포함합니다. 각 사실이 [모범 초진기록] S에 들어 있는지 표시합니다.
JSON: {"facts": [{"fact": "", "topic": "문단 주제", "in_record": true}]}"""


def case_dirs():
    return [p.parent for p in sorted((DEVKIT / 'samples').glob('*/case.json'))] + \
           [p.parent for p in sorted((DEV / 'data' / 'synth').glob('*/case.json'))]


def build(case_dir):
    case, reference = read_json(case_dir / 'case.json'), read_json(case_dir / 'model_record.json')
    user = (f"[시작 정보]\n{case['start_information']['opening']}\n활력징후: {case['start_information']['vital_signs']}\n\n"
            f"[가상 환자 대본]\n{case['patient_script']}\n\n[모범 초진기록]\n{record_text(reference)}\n\n[진찰 결과표]\n"
            + '\n'.join(f"- {r['exam']}: {r['result']}" for r in case['exam_results']))
    path = OUT / f'{case_dir.name}.json'
    old = read_json(path) if path.exists() else {}
    items = old.get('items')
    if not items:
        answer = ask_json(endpoint('sol'), SYSTEM, user, effort='high', max_tokens=16000) or {}
        items = [dict(item, id=f'f{n:02d}') for n, item in enumerate(answer.get('items') or [], 1) if item.get('fact')]
    script_user = (f"[환자 대본]\n{case['start_information']['opening']}\n{case['patient_script']}\n\n"
                   f"[모범 초진기록]\n{record_text(reference)}")
    facts = (ask_json(endpoint('sol'), SCRIPT, script_user, effort='high', max_tokens=16000) or {}).get('facts') or []
    script = [dict(fact, id=f's{n:03d}') for n, fact in enumerate(facts, 1) if fact.get('fact')]
    if items:
        write_json(path, {'case': case_dir.name, 'items': items, 'script': script})
    return case_dir.name, len(items), len(script)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--only', nargs='*')
    args = parser.parse_args()
    todo = [d for d in case_dirs() if (d.name in args.only if args.only else not (OUT / f'{d.name}.json').exists())]
    with ThreadPoolExecutor(6) as pool:
        for name, n, m in pool.map(build, todo):
            print(f'{name}: {n} items, {m} script facts' if n else f'{name}: FAILED')


if __name__ == '__main__':
    main()
