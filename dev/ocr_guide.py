"""OCR the scanned 기본진료수행지침 (resources/기본진료수행지침.pdf) page by page with Claude vision.

  python dev/ocr_guide.py                 # all pages → dev/data/guide_ocr/p001.md … (resumes; skips pages already done)
  python dev/ocr_guide.py --pages 60-80   # a range
  python dev/ocr_guide.py --join          # dev/data/guide_ocr.md (all pages, with page markers)
"""
import argparse
from concurrent.futures import ThreadPoolExecutor

import pymupdf

import common  # noqa: F401  (loads dev/.env, sets sys.path)
from claude_backend import read_image, total_spend
from common import DEV, ROOT, load_env

PDF = ROOT / 'resources' / '기본진료수행지침.pdf'
OUT = DEV / 'data' / 'guide_ocr'
INSTRUCTION = """이 이미지는 한국 의학 교재(기본진료수행지침)의 한 페이지입니다. 페이지의 글을 빠짐없이 그대로 옮겨 적으세요.
- 제목·소제목은 마크다운 제목(#, ##)으로, 목록은 목록으로, 표는 마크다운 표로 옮깁니다.
- 순서도나 그림은 [그림: 내용을 한두 줄로] 로 적고, 그림 안의 글자는 옮깁니다.
- 머리글·꼬리말·쪽번호는 빼고, 내용을 요약하거나 덧붙이지 않습니다. 빈 페이지면 (빈 페이지)라고만 씁니다.
마크다운 본문만 출력하세요."""


def ocr(doc, index):
    path = OUT / f'p{index + 1:03d}.md'
    if path.exists():
        return
    png = doc[index].get_pixmap(matrix=pymupdf.Matrix(2, 2)).tobytes('png')
    try:
        text = read_image(png, INSTRUCTION)
    except Exception as exc:  # leave the page for a rerun
        print(f'page {index + 1}: {type(exc).__name__} {exc}')
        return
    path.write_text(text, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pages', help='1-based range like 60-80')
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--join', action='store_true')
    args = parser.parse_args()
    load_env()
    doc = pymupdf.open(PDF)
    if args.join:
        pages = sorted(OUT.glob('p*.md'))
        (DEV / 'data' / 'guide_ocr.md').write_text(
            '\n\n'.join(f'<!-- page {p.stem[1:]} (printed {int(p.stem[1:]) - 2}) -->\n' + p.read_text(encoding='utf-8')
                        for p in pages), encoding='utf-8')
        print(f'joined {len(pages)} pages')
        return
    first, last = (int(x) for x in args.pages.split('-')) if args.pages else (1, len(doc))
    OUT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(args.workers) as pool:  # PyMuPDF documents are not thread-safe: one per worker call
        list(pool.map(lambda i: ocr(pymupdf.open(PDF), i), range(first - 1, last)))
    print(f'{len(list(OUT.glob("p*.md")))} pages done; Claude spend total/today: %.2f / %.2f' % total_spend())


if __name__ == '__main__':
    main()
