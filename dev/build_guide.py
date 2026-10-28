"""Refine agent knowledge cards with the 기본진료수행지침 (OCR text) and permissively licensed guideline documents.

  python dev/build_guide.py --chapters    # chapters in dev/data/guide_ocr/ → dev/data/guide_chapters.json (no model calls)
  python dev/build_guide.py --map         # one call: chapters ↔ presentations, search terms → dev/data/guide_map.json
  python dev/build_guide.py [--only 28]   # refine parts → dev/data/cards/refine/<id>.json, then merge (dev/cards.py)

Guideline excerpts come only from documents whose license allows adapted summaries (see ALLOWED); the guide and the
documents are paraphrased into checklists and plan items, not copied. Sources per card are kept in "sources".
"""
import argparse
import csv
import json
import math
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import cards as stages
from common import DEV, ROOT, ask_json, endpoint, read_json, write_json

OCR = DEV / 'data' / 'guide_ocr'
CHAPTERS = DEV / 'data' / 'guide_chapters.json'
MAP = DEV / 'data' / 'guide_map.json'
CORPUS = ROOT / 'nova-guidelines-2026'
ALLOWED = {'CC BY', '공공 도메인', 'CC0', 'CC BY-NC', '공공누리 제0유형'}  # adaptation permitted (non-commercial use)


# -- the OCR'd guide ---------------------------------------------------------------------------------------------------
def find_chapters():
    pages = sorted(OCR.glob('p*.md'), key=lambda p: int(p.stem[1:]))
    chapters = []
    for page in pages:
        text = page.read_text(encoding='utf-8')
        match = re.search(r'Chapter\s*(\d+)', text[:400])
        if match:
            heading = [l.strip('# ').strip() for l in text[match.end():match.end() + 300].splitlines() if l.strip('# ').strip()]
            chapters.append({'page': int(page.stem[1:]), 'chapter': int(match.group(1)), 'title': ' / '.join(heading[:2])})
    for n, chapter in enumerate(chapters):
        chapter['last'] = (chapters[n + 1]['page'] - 1) if n + 1 < len(chapters) else int(pages[-1].stem[1:])
    write_json(CHAPTERS, chapters)
    print(f'{len(chapters)} chapters: ' + '; '.join(f"p{c['page']} {c['title']}" for c in chapters))
    return chapters


def chapter_text(chapter, limit=24000):
    text = '\n'.join((OCR / f'p{n:03d}.md').read_text(encoding='utf-8') for n in range(chapter['page'], chapter['last'] + 1)
                     if (OCR / f'p{n:03d}.md').exists())
    return text[:limit]


MAP_SYSTEM = """두 목록을 대응시킵니다. A는 교재(기본진료수행지침)의 장 제목, B는 의사국시 실기 임상표현입니다.
B의 각 임상표현마다 내용이 맞는 A의 장(시작 쪽 번호)을 0~2개 고르고, 그 임상표현의 주요 원인 질환 5~8개에 대해
진료지침 검색어를 영어와 한국어로 만듭니다(예: "Graves disease hyperthyroidism diagnosis treatment", "갑상샘항진증").
JSON: {"<임상표현 id>": {"chapters": [시작 쪽 번호], "queries": ["검색어", ...]}}"""


def build_map(chapters, cards):
    user = ('[A 교재 장]\n' + '\n'.join(f"{c['page']}: {c['title']}" for c in chapters) + '\n\n[B 임상표현]\n'
            + '\n'.join(f"{c['id']}. {c['name']} — " + ', '.join(x['dx'] for x in c.get('causes', [])[:10]) for c in cards))
    result = ask_json(endpoint('sol'), MAP_SYSTEM, user, effort='medium', max_tokens=32000)
    write_json(MAP, result)
    print(f'mapped {len(result or {})} presentations')
    return result


# -- guideline retrieval (small BM25 over allowed documents) ----------------------------------------------------------
def tokens(text):
    text = text.lower()
    words = re.findall(r'[a-z][a-z0-9\-]{2,}', text)
    hangul = re.findall(r'[가-힣]+', text)
    return words + [w[i:i + 2] for w in hangul for i in range(max(1, len(w) - 1))]


class Index:
    def __init__(self):
        rows = [r for r in csv.DictReader(open(CORPUS / 'catalog.csv', encoding='utf-8')) if r['라이선스'] in ALLOWED]
        self.docs, self.tf = [], []
        for row in rows:
            path = CORPUS / row['파일명']
            head = path.read_text(encoding='utf-8', errors='ignore')[:6000] if path.exists() else ''
            self.docs.append({**row, 'path': path})
            self.tf.append(Counter(tokens(row['제목'] * 3 + ' ' + head)))
        self.df = Counter(t for tf in self.tf for t in tf)
        self.avg = sum(sum(tf.values()) for tf in self.tf) / max(1, len(self.tf))

    def search(self, query, k=3):
        terms = set(tokens(query))
        n = len(self.docs)
        def score(i):
            tf, length = self.tf[i], sum(self.tf[i].values())
            return sum(math.log(1 + (n - self.df[t] + 0.5) / (self.df[t] + 0.5)) * tf[t] * 2.2
                       / (tf[t] + 1.2 * (0.25 + 0.75 * length / self.avg)) for t in terms if t in tf)
        ranked = sorted(range(n), key=score, reverse=True)[:k]
        return [self.docs[i] for i in ranked if score(i) > 0]


SECTION = re.compile(r'diagnos|evaluat|treat|manag|recommend|follow|refer|진단|검사|치료|관리|권고|추적|의뢰', re.I)


def excerpt(doc, limit=2500):
    """The recommendation/diagnosis/treatment sections of one document, trimmed."""
    text = doc['path'].read_text(encoding='utf-8', errors='ignore')
    text = text.split('---', 2)[-1] if text.startswith('---') else text
    sections = re.split(r'\n(?=#{1,4} )', text)
    picked = [s for s in sections if SECTION.search(s.split('\n', 1)[0])] or sections[:2]
    body = '\n'.join(picked)[:limit]
    return f"[{doc['제목']} — {doc['발행기관']} {doc['연도']}, {doc['라이선스']}]\n{body}"


# -- card refinement ---------------------------------------------------------------------------------------------------
REFINE_SYSTEM = """당신은 한국 의사국가시험 실기(CPX) 교육 전문가이자 일차진료 전문의입니다. 임상표현 하나의 지식 카드를 개선합니다.
입력: 현재 카드(JSON), 교재 기본진료수행지침의 해당 장(OCR 텍스트), 진료지침 발췌.
출력: 같은 형식의 개선된 카드 JSON 하나. 형식과 제약은 현재 카드와 같습니다:
- hx: 30~40개, 진료 순서대로. 교재의 병력청취 항목(주 증상 확인, 동반 증상, 감별 질문, 과거력·약물·가족력·사회력 등)을 빠짐없이 반영하고
  tag(필수/감별/위험)와 field(cc/hpi/pmh/fh/sh/ros)를 붙입니다. q는 30자 이하 존댓말 질문 하나("전부/모두/모든" 금지).
- exams: 교재의 신체진찰 항목 중 감별에 쓰이는 구체적 진찰 동작 8~12개(공통 기본 진찰·활력징후 제외). phrase는 30자 이하 표준 진찰 이름.
- plan: 교재의 진단 계획·치료·환자교육과 진료지침 권고를 반영해 tests(검사와 이유), tx, dispo, fu, red_flags, edu를 채웁니다.
  주요 원인 질환별 핵심 차이가 드러나도록 구체적으로(검사 이름, 약 계열, 재진 시점).
- causes는 흔한 것과 위중한 것(tier)을 함께, counsel은 상담형일 때 상담 요점.
- 새로 "dx_notes": [{"dx": "질환", "key": "이 질환을 시사하는 핵심 문진·진찰 소견 한 줄"}] 를 주요 원인 5~10개에 대해 씁니다.
- 새로 "sources": ["교재 쪽 번호", "진료지침 제목"] 를 씁니다.
교재와 지침의 문장을 그대로 옮기지 말고 체크리스트와 요점으로 다시 씁니다. 근거가 약한 내용은 넣지 않습니다."""


def refine(card, guide_map, chapters, index):
    entry = (guide_map or {}).get(card['id'], {})
    guide = '\n\n'.join(f"(교재 p{c['page']}~{c['last']}: {c['title']})\n{chapter_text(c)}"
                        for c in chapters if c['page'] in entry.get('chapters', []))
    seen, notes = set(), []
    for query in entry.get('queries', [])[:8]:
        for doc in index.search(query, k=2):
            if doc['파일명'] not in seen:
                seen.add(doc['파일명'])
                notes.append(excerpt(doc))
    user = (f"[현재 카드]\n{json.dumps({k: v for k, v in card.items() if k != 'stub'}, ensure_ascii=False)}\n\n"
            f"[기본진료수행지침]\n{guide or '(해당 장 없음)'}\n\n[진료지침 발췌]\n" + ('\n\n'.join(notes[:8]) or '(없음)'))
    try:
        result = ask_json(endpoint('sol'), REFINE_SYSTEM, user, effort='medium', max_tokens=32000)
    except Exception as exc:
        print(f"error {card['id']}: {exc}")
        return
    if not result or not result.get('hx'):
        print(f"FAILED {card['id']} {card['name']}")
        return
    write_json(stages.part('refine', card['id']), {**result, 'id': card['id'], 'name': card['name']})
    print(f"ok {card['id']} {card['name']}: hx {len(result['hx'])}, exams {len(result.get('exams', []))}, "
          f"guide pages {entry.get('chapters')}, docs {len(notes)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chapters', action='store_true')
    parser.add_argument('--map', action='store_true')
    parser.add_argument('--only', nargs='*')
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    chapters = find_chapters() if args.chapters or not CHAPTERS.exists() else read_json(CHAPTERS)
    if args.chapters:
        return
    cards = [stages.compose(s, upto='base') for s in stages.sections()]
    guide_map = build_map(chapters, cards) if args.map or not MAP.exists() else read_json(MAP)
    if args.map:
        return
    index = Index()
    print(f'guideline index: {len(index.docs)} documents with adaptation-friendly licenses')
    todo = [c for c in cards if (args.only and c['id'] in args.only)
            or (not args.only and not stages.part('refine', c['id']).exists())]
    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(lambda c: refine(c, guide_map, chapters, index), todo))
    stages.merge()


if __name__ == '__main__':
    main()
