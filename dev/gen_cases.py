"""Generate synthetic CPX-style cases (same schema as the public samples) as a held-out dev set.

  python dev/gen_cases.py --n 15            # -> dev/data/synth/S01..S15/{case.json, model_record.json}

Presentations are drawn from the 평가목표집 list, excluding those of the public samples. One public sample is shown to the
model as a format example only; the generated case must use a different presentation and diagnosis.
"""
import argparse
import random
from concurrent.futures import ThreadPoolExecutor

from common import DEV, DEVKIT, ask_json, endpoint, read_json, write_json

SAMPLE_CPS = {'25-1', '28', '40'}
COUNSELING = {'2-1', '12', '24', '32', '35-1', '35-2', '37', '38'}
OUT = DEV / 'data' / 'synth'

SYSTEM = """당신은 한국 의사국가시험 실기(CPX) 출제위원입니다. 주어진 임상표현으로 새 가상 환자 증례를 하나 만듭니다.
형식 예시와 똑같은 JSON 구조를 쓰되 내용은 완전히 새로 만듭니다(예시의 질환·인물·소견을 재사용하지 않음).
- start_information.opening: 환자(소아면 보호자 포함)의 1인칭 서술로 이름·나이·성별·내원 이유를 3~5문장. vital_signs는 예시와 같은 형식.
- patient_script: 대본(2,000자 안팎). 현병력, 동반 증상과 중요한 음성 증상, 과거력·약물·알레르기, 가족력, 사회력, 환자의 걱정과 질문을
  환자 1인칭으로 서술. 진단을 가르는 핵심 단서는 물어봐야만 나오는 곳에 둡니다.
- exam_results: 20~25행. 공통 행(전반적 외관 시진, 의식·지남력 평가, 활력징후 측정, 결막·공막 시진, 갑상선 촉진, 흉곽 시진, 폐 청진,
  심음 청진, 복부 시진, 복부 촉진, 간·비장 촉진, 하지 말초 진찰, 보행 관찰)과 증례 특이 행. 이름은 "한국어 (English)" 형식.
  결과는 짧고 구체적으로(예: "머피 징후(-)", "슬개건 3+/3+").
- model_record: 예시와 같은 S·O·A·P 칸과 diagnosis. 대본과 진찰표에서 확인 가능한 사실만 사용. 감별진단 4~6개, 검사마다 이유.
- case_id는 주어진 값, answer에는 정답 주진단과 핵심 감별 포인트를 적습니다.
JSON: {"case": {"case_id": "", "start_information": {"opening": "", "vital_signs": ""}, "patient_script": "", "exam_results": [{"exam": "", "result": ""}]},
       "model_record": {...}, "answer": {"diagnosis": "", "key_points": [""]}}"""


def generate(case_id, section, twist, example, ep):
    folder = OUT / case_id
    if (folder / 'case.json').exists():
        return
    user = (f"case_id: {case_id}\n임상표현: {section['id']}. {section['name']}\n"
            f"{'첫 호소와 실제 원인이 다른 반전 증례로 만드세요(예: 불면으로 왔지만 수면제 의존).' if twist else '전형적인 증례로 만드세요.'}\n\n"
            f"[평가목표집 원문]\n{section['text']}\n\n[형식 예시: case.json]\n{example[0]}\n\n[형식 예시: model_record.json]\n{example[1]}")
    result = ask_json(ep, SYSTEM, user, effort='high', max_tokens=32000)
    case = (result or {}).get('case') or {}
    if not case.get('exam_results') or not case.get('patient_script') or not (result or {}).get('model_record'):
        print(f'FAILED {case_id} (incomplete output)')  # a truncated answer must not become a case
        return
    write_json(folder / 'case.json', {**result['case'], 'case_id': case_id})
    write_json(folder / 'model_record.json', result['model_record'])
    write_json(folder / 'meta.json', {'cp': section['id'], 'name': section['name'], 'twist': twist, 'answer': result.get('answer')})
    print(f"ok {case_id} {section['name']}: {result.get('answer', {}).get('diagnosis')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=15)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--model', default='sol', choices=['sol', 'luna'])
    args = parser.parse_args()
    sections = [s for s in read_json(DEV / 'data' / 'cp_sections.json') if s['id'] not in SAMPLE_CPS]
    rng = random.Random(args.seed)
    counseling = [s for s in sections if s['id'] in COUNSELING]
    diagnostic = [s for s in sections if s['id'] not in COUNSELING]
    picked = rng.sample(counseling, 2) + rng.sample(diagnostic, args.n - 2)
    sample = DEVKIT / 'samples' / 'C10'
    example = ((sample / 'case.json').read_text(encoding='utf-8'), (sample / 'model_record.json').read_text(encoding='utf-8'))
    ep = endpoint(args.model)
    jobs = [(f'S{n:02d}', s, n % 4 == 0) for n, s in enumerate(picked, 1)]
    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(lambda job: generate(*job, example, ep), jobs))


if __name__ == '__main__':
    main()
