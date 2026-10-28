"""Build agent/knowledge/cp.json from the 2020 KMLE CPX 평가목표집 (one sol call per clinical presentation).

  python dev/build_knowledge.py --split          # PDF text -> dev/data/cp_sections.json (no model calls)
  python dev/build_knowledge.py                  # generate missing base parts (dev/data/cards/base/), then merge (dev/cards.py)
  python dev/build_knowledge.py --only 28 40     # (re)generate selected presentations

The public samples are deliberately NOT given to the model here (overfitting guard); check_coverage() uses them afterwards.
"""
import argparse
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor

import cards
from common import DEV, ROOT, ask_json, endpoint, write_json

PDF = ROOT / 'resources' / '[붙임] 의사 국가시험 실기시험 평가목표 개정판[2020].pdf'
TEXT = DEV / 'data' / 'goals.txt'
SECTIONS = DEV / 'data' / 'cp_sections.json'

SYSTEM = """당신은 한국 의사국가시험 실기(CPX) 교육 전문가이자 일차진료 전문의입니다.
임상표현 하나에 대해, 가상 환자를 외래에서 처음 진료하는 AI 의사가 쓸 지식 카드를 JSON으로 만듭니다.
제약: 의사는 한 번에 30자 이하 한국어 문장 하나로 말하고, 환자는 한 질문에 하나만 짧게 답합니다.
신체진찰은 한 번에 구체적인 동작 하나만 요청할 수 있고, 예선에는 혈액·영상 검사가 없습니다(계획에만 씁니다).

출력 형식(JSON 객체 하나):
{"type": "진단" 또는 "상담",
 "kw": ["환자 첫마디에 나올 만한 한국어 단어 5~12개"],
 "speaker_hint": "본인" 또는 "보호자 가능"(소아·의식저하 등),
 "causes": [{"dx": "질환", "tier": "흔함" 또는 "위중"}],   // 8~15개, 위중(놓치면 안 되는) 질환 포함
 "hx": [{"id": "h1", "field": "cc|hpi|pmh|fh|sh|ros", "tag": "필수|감별|위험", "item": "확인할 내용",
         "q": "환자에게 할 질문(30자 이하, 존댓말 구어체, 질문 하나)"}],   // 25~40개
 "exams": [{"id": "e1", "phrase": "진찰 동작 이름 (English name)", "region": "Skin|HEENT|Neck|Chest|Abdomen|Extremities|Neuro|Mental|Other", "why": "감별 목적"}],  // 6~12개
 "plan": {"tests": [{"t": "검사", "why": "이유"}], "tx": ["치료"], "dispo": ["처치 결정 기준"],
          "fu": ["추적 계획"], "red_flags": ["즉시 내원해야 할 증상"], "edu": ["환자 교육 요점"]},
 "counsel": ["상담형일 때 대화로 다뤄야 할 요점"]}

규칙:
- hx는 실제 진료 순서(주호소 특성 OPQRST → 동반 증상 → 감별 질문 → 과거력·약물·알레르기 → 가족력 → 사회력 → 계통 문진)로 정렬합니다.
- '필수'는 이 임상표현이면 늘 묻는 항목, '감별'은 특정 원인을 가르는 항목, '위험'은 위중한 원인의 red flag입니다.
- q에는 "전부", "모두", "모든", "다 말씀" 같은 표현을 쓰지 않습니다(환자가 답을 거부합니다). 두 증상을 한 질문에 묶을 때는 "열이나 오한 있으세요?"처럼 짧게만.
- 소아 증례가 가능하면 보호자에게 하는 질문(출생력, 발달, 예방접종, 성장 기록 등)을 포함합니다.
- exams.phrase는 표준 진찰 동작 하나의 이름입니다(예: "갑상선 촉진 (Thyroid palpation)", "머피 징후 (Murphy sign)",
  "기립성 혈압 측정 (Orthostatic BP)", "인지기능 선별검사 (Cognitive screening)"). 30자 이하. "및", "/", "전체", "전신"을 쓰지 않습니다.
  공통 기본 진찰(전반적 외관, 의식, 결막, 갑상선 촉진, 폐 청진, 심음 청진, 복부 시진·촉진, 간·비장 촉진, 말초 부종, 보행)은 넣지 말고
  이 임상표현의 원인을 가르는 진찰만 넣습니다. 활력징후 측정은 넣지 않습니다(이미 주어짐).
- plan은 한국 일차진료 기준으로, 원인별로 필요한 검사에 이유를 붙이고 위중 원인 배제 계획을 포함합니다.
- 상담형(예방접종, 산전 진찰, 금연·음주 상담, 나쁜 소식 전하기, 가정폭력, 자살 등)이면 counsel을 충실히 채웁니다. 진단형이면 counsel은 []입니다."""


def extract_text():
    if not TEXT.exists():
        TEXT.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['pdftotext', '-enc', 'UTF-8', '-layout', str(PDF), str(TEXT)], check=True)
    return TEXT.read_text(encoding='utf-8')


def split_sections(text):
    """A presentation (or sub-presentation like '25-1. 성장 지연') is an indented heading followed by '❑ 중요성'."""
    lines = [l.replace('\f', '') for l in text.splitlines()]
    starts = []
    for i, line in enumerate(lines):
        match = re.match(r'^\s*(\d{1,2}(?:-\d)?)\.\s*(\S.*?)\s*$', line)
        following = next((l for l in lines[i + 1:i + 4] if l.strip()), '')
        if match and following.strip().startswith('❑ 중요성'):
            starts.append((i, match.group(1), match.group(2)))
    sections = []
    for n, (start, number, name) in enumerate(starts):
        end = starts[n + 1][0] if n + 1 < len(starts) else len(lines)
        body = '\n'.join(l.rstrip() for l in lines[start:end] if l.strip() and not re.fullmatch(r'\s*\d+\s*', l))
        sections.append({'id': number, 'name': name, 'text': body})
    return [s for s in sections if not s['name'].startswith('기본진료술기')]


def generate(section, ep, effort):
    try:
        card = ask_json(ep, SYSTEM, f'임상표현 {section["id"]}. {section["name"]}\n\n평가목표집 원문:\n{section["text"]}',
                        effort=effort, max_tokens=24000)
    except Exception as exc:  # one failed presentation must not stop the others; rerun to fill gaps
        card = None
        print(f'error {section["id"]}: {exc}')
    if card is None:
        print(f'FAILED {section["id"]} {section["name"]}')
        return
    write_json(cards.part('base', section['id']), {'id': section['id'], 'name': section['name'], **card})
    print(f'ok {section["id"]} {section["name"]}: hx {len(card.get("hx", []))}, exams {len(card.get("exams", []))}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', action='store_true')
    parser.add_argument('--only', nargs='*', help='presentation ids such as 28 25-1')
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--model', default='sol', choices=['sol', 'luna'])
    args = parser.parse_args()
    sections = split_sections(extract_text())
    write_json(SECTIONS, sections)
    print(f'{len(sections)} sections: ' + ', '.join(f'{s["id"]}.{s["name"]}' for s in sections))
    if args.split:
        return
    todo = [s for s in sections if (args.only and s['id'] in args.only)
            or (not args.only and not cards.part('base', s['id']).exists())]
    ep = endpoint(args.model)
    effort = 'medium' if args.model == 'sol' else 'high'
    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(lambda s: generate(s, ep, effort), todo))
    cards.merge()


if __name__ == '__main__':
    main()
