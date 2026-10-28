# Data provenance and reproduction

Competition rules require keeping the source, license, and generation process of every external resource used.

| Artifact | Source | How it was produced | Reproduce |
|---|---|---|---|
| `dev/data/goals.txt`, `dev/data/cp_sections.json` | 한국보건의료인국가시험원, 「의사 국가시험[실기] 평가목표집 개정판」(2020) (`resources/`) | `pdftotext -enc UTF-8 -layout`, split by presentation headings | `python dev/build_knowledge.py --split` |
| `agent/knowledge/cp.json` (shipped) | The section above + model knowledge of gpt-6.1-sol | One sol call per presentation (prompt in `dev/build_knowledge.py`); the model writes its own checklists and plan summaries. No guideline text is copied. The public samples are not shown to the model. | `python dev/build_knowledge.py` |
| `dev/data/guide_ocr/*` (dev only) | 한국보건의료인국가시험원 「기본진료수행지침」 (`resources/기본진료수행지침.pdf`, scanned) | Page-by-page transcription with Claude Sonnet 5.5 vision (`dev/ocr_guide.py`) | `python dev/ocr_guide.py` |
| `agent/knowledge/cp.json` v2 (shipped) | v1 cards + the guide chapter for each presentation + excerpts from guideline documents licensed CC BY / CC BY-NC / CC0 / public domain / 공공누리 제0유형 only (no ND licenses) | One Claude Sonnet 5.5 call per presentation rewrites the card as checklists and plan points (no verbatim text); each card lists its `sources`. v1 kept as `dev/data/cp_v1.json` | `python dev/build_guide.py --chapters`, `--map`, then `python dev/build_guide.py` |
| `dev/data/cards/dx/*` → `dx_notes` in `agent/knowledge/cp.json` (shipped) | The composed cards (causes, existing `dx_notes`, history and exam lists) + model knowledge of Claude Opus 5.5 | One Opus call per presentation (prompt in `dev/build_dx.py`) writes, per cause, aliases, a key-findings line, 3-6 follow-up questions and pointers to existing exam ids. The prompt's two examples (hyperthyroidism, sedative dependence) are general CPX knowledge chosen after the grader's comments on local runs of the public samples; no sample case text is shown to the model. No guideline text is copied. | `python dev/build_dx.py`, then `python dev/cards.py` |
| `dev/data/synth/*` (dev only) | Synthetic; written by gpt-6.1-sol from the 평가목표집 section, with public sample C10 shown as a format example only | `dev/gen_cases.py` (seeded presentation choice) | `python dev/gen_cases.py --n 15` |
| `dev/data/gold/*` (dev only) | Synthetic ideal transcripts for the public samples, written by gpt-6.1-sol from the case script and model record | `dev/judge.py` (`gold_events`) | `python dev/judge.py --selftest` |

Grader rubric (`dev/judge.py`) references:
- Official area weights and record rules: https://nova.snubhai.org/evaluation/ , https://nova.snubhai.org/docs/
- Diagnosis 5-point scale: Nori et al., "Sequential Diagnosis with Language Models", arXiv:2506.22405 (2025)
- CPX PPI items (관계 형성, 청취, 공감, 정보 전달, 마무리): KMLE CPX literature; 평가목표집 "의사소통/환자의사관계"
- AMIE OSCE axes (managing patient concerns, explanation, empathy): Tu et al., arXiv:2401.05654 (2024)

Agent design reference: MAI-DxO (Nori et al., 2025), adapted to one structured call per batch of actions.

Not used at runtime: the guideline corpus (`nova-guidelines-2026/`), the benchmark datasets, and any external API.
| `dev/data/checklists/*` (dev only) | Each case's patient script, model record and exam table (public samples and `dev/data/synth`) | One Claude Opus 5.5 call per case (prompt in `dev/checklists.py`) splits the model record into grading-unit facts with the script paragraph and a kind (cc/dx/diff/routine/ice/exam); `dev/judge.py` grades against it when present | `python dev/checklists.py` |
| `dev/data/cards/exambank/*` → `exam_bank` in `agent/knowledge/cp.json` (shipped) | Each card's presentation, cause list (first 12) and card exams + model knowledge of Claude Opus 5.5 (43 cards via the API, 12 via headless Claude Code when the API credit ran out) | One call per presentation (prompt in `dev/build_exam_bank.py`) lists, per cause, 4-8 diagnosis-specific exam maneuvers named like exam-table rows. The public samples are not shown | `python dev/build_exam_bank.py` |
