# N.O.V.A. 2026 Doctor Agent (v44)

Our entry for N.O.V.A. 2026 (분당서울대학교병원 의료인공지능센터 + elice). The agent interviews a Korean-speaking virtual patient
in CPX style, performs physical exams, and submits an initial S/O/A/P record with one main diagnosis. v44 is our best
officially validated version.

Diagram of the agent structure with an example encounter (one editable slide): https://claude.ai/artifact/S68fWuVTF2E1aUAtEDdDcF

## Score (official sample validation, 3 public cases)

| 정보수집 | 진단추론 | 검사·추적 | 안전 | 의사소통 | 효율 | total |
|---|---|---|---|---|---|---|
| 12.24 / 25 | 15.00 / 15 | 7.44 / 10 | 12.50 / 15 | 6.72 / 15 | 17.39 / 20 | **71.29** |

Every scored version is in `dev/calibration.csv`.

## How an encounter runs

Code fixes the order and the turn budget (`TARGET_TURNS = 42`). The competition model is called only through
`brain.ask(client, enc, task)`.

1. Greeting and one open question.
2. Broad history to turn 5 (`first`: picks a presentation card from `knowledge/cp.json`).
3. Hypothesis round 1 (`hypothesis`): 8 card checklist items plus up to 5 extra questions.
4. Substance-use screen (code, regex triggered).
5. Hypothesis round 2 (`predict`): the facts expected for the leading diagnosis, sized to the turns left.
6. Routine history (code): past history, medication, family, social.
7. Physical exams: up to 5 diagnosis-specific picks from the card's exam bank, then core exam rows (14 at most).
8. Closing dialog (`close`, then `talk` once per turn). It covers diagnosis, concern, tests, treatment, cause,
   prognosis, lifestyle and red flags, then asks for questions and says goodbye.
9. Record (`record`) and audit (`audit`), then `client.diagnose()`.

## Layout

| path | what |
|---|---|
| `agent/run.py` | encounter loop, turn budget, closing |
| `agent/brain.py` | prompts, `ask()`, validation, record building |
| `agent/knowledge/cp.json` | 55 presentation cards: checklists, exam bank, diagnosis notes, plans |
| `dev/` | offline tests, local bench, Opus judge, card-building pipeline, zip builder (not shipped) |
| `dev/PROVENANCE.md` | sources and generation of every data file |

## Not included

The competition kit (`nova-starter/`, `nova-devkit/`, `nova-samples-2026/`, `nova_sdk.py`), the guideline corpus, the
KMLE CPX objectives and 기본진료수행지침 PDFs, their OCR and section text, and the per-stage card parts built from
them are not redistributed. Get the kit from the competition site and place it next to `agent/`. Copy
`nova-starter/nova_sdk.py` into `agent/` to run or package the agent.

```bash
python dev/test_agent.py   # offline tests (Python 3.12)
python dev/make_zip.py     # dist/submission.zip
```
