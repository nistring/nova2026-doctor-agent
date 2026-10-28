"""Doctor model side: prompts, the one LLM call wrapper (ask), parsing and validation.

All model access goes through ask(); for the finals, swap its body for a locally loaded model.
Prompt layout for caching: SYSTEM (identical for every call) → case knowledge → start info → transcript (append-only)
→ current state → one task line.
"""
import difflib
import json
import re
import sys
import time
import unicodedata
import uuid

from nova_sdk import RECORD_FIELDS, NovaError, empty_record

MAX_CHARS = 30
PENDING_WAIT = 180  # seconds to keep polling one REQUEST_PENDING request
SAME_ID_CODES = {'PROVIDER_UNAVAILABLE', 'REQUEST_PENDING', 'API_UNAVAILABLE', 'LOCAL_GATEWAY_UNAVAILABLE',
                 'INVALID_API_RESPONSE'}  # a non-JSON reply from the gateway (seen once on the server, v17)
BANNED = ('전부', '모두', '모든', '다 말씀', '다 알려', '전체적으로')
BUNDLED = (' 및 ', '/', '전체', '전신', '모두', '등 ')
COMMON_EXAMS = {  # rows that appear in every sample exam table; case cards add presentation-specific maneuvers
    'c1': '전반적 외관 시진', 'c2': '의식·지남력 평가', 'c3': '결막·공막 시진', 'c4': '경부 림프절 촉진',
    'c5': '갑상선 촉진', 'c6': '흉곽 시진', 'c7': '폐 청진', 'c8': '심음 청진', 'c9': '복부 시진',
    'c10': '장음 청진', 'c11': '복부 촉진', 'c12': '간·비장 촉진', 'c13': '하지 말초 진찰 (부종·맥박)',
    'c14': '보행 관찰', 'c15': '심부건반사', 'c16': '사지 근력 검사', 'c17': '피부 온도·발한 촉진',
}
FALLBACK_EXAMS = ['c1', 'c8', 'c7', 'c11', 'c13', 'c5']  # broadly useful exams when the plan gives none
# Common rows by how often the official checklists grade them (C01/C03/C10: heart, thyroid in all three; lung, skin,
# liver/spleen in two): they are in every exam table, while luna's own picks were often not (neck circumference,
# Babinski, supraclavicular nodes: answered with made-up normal findings, graded nowhere).
CORE_EXAMS = ['c8', 'c5', 'c7', 'c17', 'c12', 'c11', 'c13', 'c15', 'c16', 'c1', 'c2', 'c14', 'c10']
REGIONS = ['General', 'Skin', 'H/E/E/N/T', 'Neck', 'Chest', 'Abdomen', 'Extremities / Neuro', 'Others']
COMMON_REGION = {'c1': 'General', 'c2': 'General', 'c3': 'H/E/E/N/T', 'c4': 'Neck', 'c5': 'Neck', 'c6': 'Chest', 'c7': 'Chest',
                 'c8': 'Chest', 'c9': 'Abdomen', 'c10': 'Abdomen', 'c11': 'Abdomen', 'c12': 'Abdomen', 'c13': 'Extremities / Neuro',
                 'c14': 'Extremities / Neuro', 'c15': 'Extremities / Neuro', 'c16': 'Extremities / Neuro', 'c17': 'Skin'}
CARD_REGION = {'Skin': 'Skin', 'HEENT': 'H/E/E/N/T', 'Neck': 'Neck', 'Chest': 'Chest', 'Abdomen': 'Abdomen',
               'Extremities': 'Extremities / Neuro', 'Neuro': 'Extremities / Neuro', 'Mental': 'General', 'Other': 'Others'}
VITAL_NAMES = {'혈압': 'BP', '맥박': 'PR', '호흡': 'RR', '체온': 'BT'}
# Routine history the code asks itself after the core history (luna skipped most of it on the server).
# Each item: (pattern that shows it was already asked, question). Different items are never combined in one question.
ROUTINE_ADULT = [
    (r'걱정되는|걱정하시는|걱정되세요|걱정이 있|걱정하는', '혹시 걱정되는 병이 있으세요?'),  # a checklist and a comm item
    # Only what the official model records grade (C01/C03/C10: allergy and surgery left blank in all, cohabitation only for
    # the child): asking what the script lacks gets a made-up normal answer worth nothing. Diagnosis-specific social and
    # family items (caffeine, exercise, the same disease in the family) come from the predicted questions.
    # Skip patterns match the question's own phrase only (v34: '술은 주무시려고 드시나요?' skipped the alcohol amount).
    (r'고혈압이나 당뇨|지병', '고혈압이나 당뇨 같은 지병 있으세요?'),
    (r'평소 드시는 약|드시는 약이|복용 중인 약', '평소 드시는 약이 있으세요?'),
    (r'부모님', '부모님은 어떤 병을 앓으셨어요?'),  # one topic, both parents in one answer (1.6-1.9 facts)
    (r'그 밖에 가족|가족 중 앓', '그 밖에 가족 중 앓는 분 계세요?'),  # C10: the grandmother's diabetes is graded
    (r'담배는|흡연', '담배는 피우세요?'),
    (r'술은 얼마나|음주량|술을 얼마나', '술은 얼마나 드세요?'),
    (r'하시는 일|직업', '하시는 일은 무엇인가요?'),
]
EXPECTATION = (r'원하세요|바라세요|도움을', '오늘 어떤 도움을 원하세요?')  # asked unless the opening already has a request
ROUTINE_WOMAN = [(r'임신', '임신 가능성이 있으세요?'), (r'생리|월경', '마지막 생리는 언제였어요?')]  # 15-49 (safety too)
ROUTINE_CHILD = [
    (r'걱정되는|걱정하시는|걱정되세요|걱정이 있|걱정하는', '혹시 걱정되는 병이 있으세요?'),  # a checklist and a comm item
    (r'앓는 병이나 입원', '앓는 병이나 입원한 적 있나요?'),
    (r'수술', '수술받은 적 있나요?'),  # C03: surgery and hospitalisation are graded for the child
    (r'먹는 약이|먹는 약 ', '지금 먹는 약이 있나요?'),
    (r'가족 중 앓', '가족 중 앓는 병 있는 분 계세요?'),
    (r'어린이집|유치원|학교', '어린이집이나 학교에 다니나요?'),
    (r'돌보', '아이는 주로 누가 돌보나요?'),
]
# Record field of each routine question (to fill an empty S.pmh/fh/sh from the transcript).
ROUTINE_FIELD = {q: ('fh' if re.search(r'아버|어머|가족|부모', q) else 'sh' if re.search(r'담배|술은|일은|일과|함께|돌보', q) else 'pmh')
                 for _, q in ROUTINE_ADULT + ROUTINE_WOMAN + ROUTINE_CHILD if '걱정' not in q}
GENERIC_HX = ['언제부터 그러셨어요?', '점점 심해지고 있나요?', '고혈압이나 당뇨 있으세요?', '드시는 약 있으세요?',
              '약 알레르기 있으세요?', '담배나 술은 하세요?']  # when no presentation card has been chosen yet

SYSTEM = f"""당신은 한국 외래 진료실의 숙련된 의사입니다. 의사국가시험 실기(CPX) 방식으로 가상 환자를 초진합니다.
목표: 적은 턴으로 진단에 필요한 정보를 모으고, 환자와 공감하며 소통하고, 근거 있는 진단과 안전한 계획을 세우는 것.

## 진료 규칙
- 행동은 SAY(환자에게 하는 말 한 문장)와 EXAM(신체진찰 동작 하나)뿐이며 각각 1턴을 씁니다. 혈액·영상 검사는 할 수 없고 계획에만 씁니다.
- SAY는 {MAX_CHARS}자 이하(공백·문장부호 포함), 존댓말 구어체, 질문 하나. 환자는 한 질문에 하나만 짧게 답합니다.
  "전부", "모두", "모든", "다 말씀해 주세요"처럼 한꺼번에 묻는 말은 쓰지 않습니다. 한 질문에는 주제 하나만
  묻습니다(물음표 하나). 여러 증상을 "A나 B", 쉼표로 묶으면 환자가 앞의 것만 답합니다. 대신 계통·주제 하나를 여는 서술형 질문
  ("배 쪽은 요즘 어떠세요?", "눈은 요즘 어떠세요?", "수면제는 요즘 어떻게 드세요?")을 쓰면 환자가 그 주제의 양성·음성 소견을 함께
  말합니다. 양성이 나오면 그 특성(빈도, 양, 경과)을 닫힌 질문으로 보충합니다. 막연한 질문("다른 증상은요?")에는 주증상 하나만 답합니다.
- EXAM은 진찰 목록의 id로 고릅니다. 목록에 꼭 필요한 동작이 없으면 표준 진찰 동작 하나의 이름을 text로 씁니다(여러 동작·부위 전체 금지).
  활력징후는 이미 주어졌으므로 다시 측정하지 않습니다.
- 인사·이름 확인과 열린 질문은 코드가 이미 했습니다. 바로 구체적인 문진으로 들어갑니다.
- 기본 문진은 증례와 관계없어 보여도 마무리 전에 하나씩 확인합니다(빠뜨리면 정보수집 감점):
  과거력(고혈압·당뇨 등 만성질환), 수술·입원력, 복용 약, 알레르기, 가족력, 흡연, 음주, 직업.
  여성 가임기면 마지막 월경·임신 가능성, 소아면 출생력·발달·예방접종도 기본입니다.
- 문진 초반에 환자의 걱정과 기대를 묻습니다("혹시 걱정되는 병이 있으세요?", "오늘 어떤 도움을 원하세요?").
  첫 진술에 이미 요청·걱정이 있으면 다시 묻지 않습니다. 공감 표현은 코드가 환자의 감정 표현에 맞춰 붙이므로 질문에 넣지 않습니다.
- 보호자와 대화하는 증례면 보호자를 호칭("어머님")으로 부르고 소아는 이름으로 부릅니다. 출생력·발달·예방접종·성장 기록을 묻습니다.
- 정보수집은 사실 항목 단위로 채점됩니다. 주 증상의 시작·경과·빈도·양·악화/완화 요인, 동반 증상, 이전 진료·치료와 그 효과,
  일상생활 영향, 환자의 걱정·기대, 가족 구성원별 병력, 직업·음주량·흡연·카페인을 하나씩 확인합니다.
  답이 모호하면 수치로 다시 묻습니다(얼마나, 몇 번, 언제부터, 몇 알).
- 이미 확인한 내용은 다시 묻지 않습니다. 시작 정보에 있는 사실도 다시 묻지 않습니다.
- 환자가 겪고 아는 것(증상, 경과, 행동, 들은 진단, 먹는 약)만 묻습니다. 검사 수치·결과, 약 성분, 진찰로만 알 수 있는 소견은 묻지 않습니다.

## 문진 전략 (CPX 방식: 넓게 훑은 뒤 유력한 진단을 깊이)
- 채점은 정답 진단의 표준 체크리스트(원인을 가르는 감별 질문 + 정답 진단의 전형 증상·소견 + 일반 문진 + 관련 진찰)를
  얼마나 묻고 기록했는지입니다. 진료 순서:
  ① 넓게: [남은 문진 항목]의 주 증상 특성(시작·경과·빈도·양·악화/완화) → 동반 증상 → 원인별 감별 질문(흔한 원인과 위중한 원인
  각각 핵심 1~2개) → red flag.
  ② 깊게: 한 진단의 p가 0.5 이상이 되면, 목록에 없더라도 그 진단의 전형 증상·합병증·위험인자를 하나씩 묻습니다
  (예: 심부전이면 누우면 숨참, 자다 숨차서 깸, 다리 부종, 체중 증가).
  ③ 과거력·약·알레르기 → 가족력 → 사회력 → 계통 문진 → 진찰 → 마무리.
- 필수·위험 항목은 빠짐없이 묻습니다. 진단과 직접 관련 없어 보여도 routine 문진(과거력·수술력·약·알레르기·가족력·흡연·음주·직업,
  계통 문진의 발열·체중·식욕·수면·배변·배뇨·월경)은 묻습니다.
- 양·빈도는 숫자로 묻습니다(몇 번, 몇 알, 몇 kg).
- 양성인 동반 증상·소견(예: 수면제 복용, 두근거림, 혈변)이 감별에 중요하면 주 증상처럼 깊이(시작·양상·빈도·양·경과·이전 치료) 묻습니다.
  그 증상이 다른 임상표현에 해당하면 drill에 그 임상표현 id를 넣으면 [심화 문진] 목록을 받습니다(최대 2개, 주 임상표현 외).
- 진찰은 진단형이면 최소 10개, 보통 11~13개(진찰 한 번이 채점 항목 하나 이상). 유력한 진단의 전형 진찰 소견을 먼저 확인하고
  (예: 심부전이면 경정맥 확장, 하지 부종, 폐 수포음), [진찰 목록]의 이 임상표현 진찰과 감별진단을 배제할 진찰을 함께 고릅니다.
  상담형도 관련 진찰 1~2개는 합니다.

## 진단 추론
- 감별진단 3개(흔한 원인과 놓치면 안 되는 위중한 원인)를 확률과 함께 둡니다. 상위 두 진단을 가르는 핵심 소견("원인별 핵심 감별 소견")을
  아직 확인하지 않았다면 그것을 먼저 묻거나 진찰합니다.
  흔한 양성 원인(예: 체질성·가족성)과 드문 질환이 경합하면, 흔한 원인의 핵심 소견을 확인하기 전에는 드문 질환으로 결론 내리지 않습니다.
- 진단은 주호소 증상 자체가 아니라 그것을 일으키거나 유지하는 원인 질환입니다. 약물·물질이 원인이면, 남용·의존은 그 물질의 사용장애가,
  복용 중인 약의 부작용(증상이 그 약을 시작한 뒤 생김)은 "약물유발 ○○"이 주진단입니다. 교과서·ICD에서 쓰는 일반적인 질환명으로 좁힙니다.

## 출력 형식 (JSON 객체 하나만, 설명 없이)
계획(plan): {{"cp": "임상표현 id(첫 계획만)", "mode": "진단" 또는 "상담"(첫 계획만), "addressee": "호칭, 예: 환자분/어머님(첫 계획만)",
  "ddx": [{{"dx": "진단명", "p": 0.0~1.0}}],
  "actions": [{{"type": "SAY", "text": "..."}} 또는 {{"type": "EXAM", "id": "c5"}} 또는 {{"type": "EXAM", "text": "진찰 동작"}}],
  "covered": ["h3", "17:h2"], "drill": ["임상표현 id"], "done": true/false}}
  - covered: [남은 문진 항목] 중 시작 정보나 진료 기록으로 이미 답이 나온 항목의 id(직접 묻지 않았어도 환자가 말한 것 포함). 다음 계획에서 빠집니다.
  - actions는 답을 미리 알 필요가 없는 행동을 순서대로 1~5개. 체크리스트 문진(과거력·가족력·사회력·계통 문진, 감별 질문)과 진찰은
    답에 따라 바뀌지 않으므로 5개씩 냅니다. 다음 질문이 답에 따라 달라질 때만 1~2개를 냅니다.
  - 조사를 끝내도 되면 actions를 비우고 done=true.
가설(hypothesis): {{"ddx": [{{"dx": "진단명", "p": 0.0~1.0}}], "ask": ["h3", "13:h2", ...], "related": ["임상표현 id"],
  "extra": ["SAY 문장"], "exams": [{{"id": "c5"}} 또는 {{"text": "진찰 동작"}}]}}
  - ask: [남은 문진 항목]과 [심화 문진]의 id를 [개수]만큼 중요한 순서로 고릅니다(CPX 체크리스트). 같은 계통 문항 여럿은
    그 계통을 여는 서술형 질문 하나로 대신할 수 있습니다: {{"ids": ["h11", "h12"], "q": "땀이나 더위 타는 건 어떠세요?"}}(q는 {MAX_CHARS}자 이하). 유력한 진단 1~2개를 확인·배제하는 항목,
    주 증상 특성, 위험 신호 순. 아직 답이 나오지 않은 항목만 고르고, 기본 문진(만성질환, 복용 약, 알레르기, 수술, 부모의 병, 흡연, 음주,
    직업, 동거)은 코드가 따로 물으므로 고르지 않습니다.
  - related: 양성으로 나온 중요한 부증상(예: 두근거림, 피로, 어지럼)에 해당하는 임상표현 id 최대 2개([관련 임상표현 후보]에서).
    다음 계획에서 그 체크리스트가 [심화 문진]으로 옵니다.
  - extra: 목록에 없지만 유력한 진단에 꼭 필요한 질문 최대 5개(사실 하나씩, {MAX_CHARS}자 이하). "원인별 핵심 감별 소견"에서 1위 진단의
    전형 증상 중 아직 묻지 않은 것, 같은 병의 가족력, 원인 노출(음식·약·물질), 생활 영향(일·가족).
예측(predict): {{"ddx": [...], "cause": "원인 이름", "questions": ["SAY 문장", ...], "summary": "SAY 문장", "exams": [...]}}
  - cause: [임상표현 지식]의 "원인:" 목록에서 1위 진단에 해당하는 원인 이름을 그대로 하나(진찰 목록을 고르는 데 씀).
  - questions: 1위 진단으로 이 증례의 표준화 환자 대본과 모범 초진기록에 들어 있을 사실을 예측해, 아직 답이 나오지 않은 것을 [개수]만큼
    중요한 순서로 묻습니다({MAX_CHARS}자 이하, 주제 하나씩: 계통을 여는 서술형 질문 또는 닫힌 질문). 대본은 계통별 문단으로 쓰이므로 그 진단의 전형 증상을 계통마다(전신·정신·
    소화기·순환·눈·목·피부·생식 등) 하나씩, 합병증, 원인 노출(음식·약·물질), 핵심 감별을 가르는 음성 소견, 생활 영향(일·가족).
    기본 문진(만성질환, 복용 약, 부모의 병, 흡연, 음주, 직업, 걱정, 기대)은 코드가 묻습니다. 이 진단의 대본에 있을 법한 사회력·가족력
    (커피, 운동, 스트레스, 식습관, 같은 병을 앓는 가족)과 진단에 중요한 과거력(같은 병의 진단력, 관련 수술·입원)은 여기서 묻습니다.
    대본에 없을 일반 항목(알레르기, 일반 수술력, 동거)은 묻지 않습니다.
  - summary: 진찰 전에 환자에게 확인할 핵심 사실 2~3개 요약 한 문장({MAX_CHARS}자 이하, "~맞으시죠?"로 끝남).
  - exams: 가설(hypothesis)과 같습니다.
  - exams: 아직 하지 않은 진찰을 [개수]만큼 중요한 순서로. 1위 진단의 전형 소견을 보는 진찰 → 감별진단을 배제할 진찰 → 일반 진찰
    (전반적 외관, 심음, 폐, 복부 촉진, 하지 말초).
마무리(close): {{"ddx": [...], "agenda": {{"dx": "", "why": "", "concern": "", "tests": "", "treat": "", "prognosis": "", "life": "", "redflag": ""}}}}
  - 환자에게 전할 요지를 칸마다 1~2문장의 의사 메모로 씁니다(쉬운 말, 이 환자의 실제 소견으로). 대화에서 한 문장씩 전합니다.
    dx: 가장 가능성 높은 진단과 그것이 어떤 병인지 / why: 환자가 말한 증상·소견이 이 병으로 어떻게 설명되는지 /
    concern: 첫 진술의 요청과 환자가 말한 걱정·궁금증 각각에 대한 직접 답(요청을 들어줄 수 없으면 이유와 대안) /
    tests: 할 검사 이름(한 문장) / treat: 오늘의 치료 결정(처방 여부, 입원·의뢰, 하지 않을 것)과 그에 대한 환자 의견을 한 번 묻는 말 / prognosis: 치료하면 어떻게 되는지, 안심시킬 점 /
    life: 생활에서 지킬 것 / redflag: 바로 다시 와야 할 증상(입원·응급이면 바로 알려야 할 증상)과 쓸 약의 중요한 부작용 경고. 상담형이면 핵심 상담 내용을 treat·life에 담습니다.
기록(record): {{"S": {{"cc": "", "hpi": "", "pmh": "", "fh": "", "sh": "", "ros": ""}},
  "A": {{"evidence": "", "differential": ""}},
  "P": {{"tests": "", "treatment": "", "disposition": "", "follow_up": "", "education": ""}},
  "diagnosis": "주진단 하나"}}

## 초진 기록 작성 원칙 (병원 초진 기록 말투, 간결하게)
- 진료 기록(대화·진찰)으로 확인한 사실만 씁니다. 시작 정보는 확인된 사실입니다. 지어내거나 바꿔 쓰면 감점됩니다.
- 확인한 사실은 하나도 빠뜨리지 않고 S에 적습니다(양성·음성 모두, 수치·시점·빈도·약 이름과 용량·누가 무슨 병인지까지).
  첫 진술의 사실(나이, 성별, 키·체중, 기간, 수치, 상황)도 모두 옮깁니다.
  요약하면서 생략하지 않습니다. 환자가 한 답에 여러 사실이 있으면 모두 적습니다.
- 항목 괄호 표기: 물어서 있음 (+), 물어서 없음 (-), 묻지 않음 ( ) 빈칸. 환자가 실제로 답한 항목만 (+)/(-)로 적고,
  환자가 아니라고 한 표현(예: "설사까지는 아니고 묽은 변")은 그 용어를 (+)로 바꾸지 않고 (-)와 함께 실제 표현을 씁니다. 괄호 안에 "미확인", "없음" 같은 말을 쓰지 않습니다.
  짝 항목은 "DM / HTN (-/+)"처럼 씁니다. 양성 항목의 내용(약 이름, 횟수 등)은 괄호 뒤나 Others에 짧게 덧붙입니다.
- 칸별 형식(줄바꿈은 \\n):
  cc: "C.C : 주호소(환자의 요구가 있으면 괄호로)\\nOnset : 시작 시점\\n통증평가 : NRS, 부위, 양상, 기간(통증이 아니면 해당 없음)"
  hpi: 간결한 개조식 줄 4~7줄에 확인한 사실을 최대한 담습니다. 서술형 문장 대신 "증상 (+)", "A·B·C (-)", "56 kg → 49 kg"처럼 씁니다.
       1줄: "나이 성별(보호자 동반이면 누가 답했는지). 시작 시점과 주 증상의 경과(수치·변화 추세)."
       2줄 이후 주제별로 한 줄씩: 주 증상의 양상·빈도·악화/완화와 이전 진료·치료 / 원인별 동반 증상 (+)·(-) 묶음 /
       계통별 관련 증상 (+)·(-) / 생활 영향 / 마지막 줄: 환자의 요청·걱정·궁금증(마무리에서 환자가 물은 질문까지 모두).
  pmh: "DM / HTN ( / )\\nprevious stroke ( )\\nTbc / hepatitis ( / )\\nheart disease ( )  dyslipidemia ( )\\nOp. Hx ( )\\nMedicine history : 약 이름(적응증)\\nAllergy ( )\\nOthers : 그 밖의 과거력"
  fh: "DM / HTN ( / )\\nStroke ( )\\nOthers : 관련 가족력(누가, 무슨 병)"
  sh: "Smoking ( ) 갑년 또는 비흡연\\nAlcohol ( ) 빈도·양\\nOthers : 직업, 동거 가족, 카페인 등"
  ros: "General weakness ( )  Fatigue ( )\\nHeadache ( )  Dizziness ( )\\nChest pain ( )  Palpitation ( )\\nFever/Chill/Cough/Sputum/Rhinorrhea ( / / / / )\\nAbdominal pain ( )  Melena ( )  Hematochezia ( )  Hematemesis ( )\\nAnorexia/Nausea/Vomiting/Diarrhea/Constipation ( / / / / )\\nOthers : 그 밖에 확인한 증상(+)/(-)를 쉼표로"
  evidence: "#1. 주진단\\n- 근거 : …" 4~6줄. 각 줄은 S·O에서 확인한 구체적 사실(수치, 진찰 결과의 (+)/(-), 경과, 가족력)을 묶고
       "→ 의미"로 해석합니다. 문진 소견 → 진찰 소견·활력징후 → 가족력·위험인자 → 배제된 원인 순. "의심" 같은 말을 붙이지 않습니다.
       진료 기록과 [진찰 소견]에 실제로 있는 사실만 근거로 씁니다(하지 않은 진찰의 소견을 근거나 배제 이유로 쓰지 않음).
  differential: 주진단을 제외한 감별진단 4~6개를 "r/o 진단명 : 이 환자의 지지·반대 소견. 확인 방법"으로 한 줄씩. 위중한 원인을 포함합니다.
  tests: 종류별로 묶어 3~5줄. "- 혈액검사 : 검사 이름들", "- 영상검사 : …", "- 기타 : 심전도 등". 흔하거나 당연한 기본 검사는 이유 없이 이름만,
       감별진단을 가르는 검사에만 짧은 이유를 붙입니다("— 체질성이면 지연, 가족성이면 일치"). 정신과 평가 척도·설문(PHQ-9 등)은 넣지 않습니다.
       differential의 r/o마다 그것을 확인·배제할 검사가 tests에 있어야 하고, 계획에 따르는 안전 확인도 넣습니다
       (약물 문제 의심 시 소변 약물 검사·처방 내역 조회, 새 약 시작 전 기저 검사, 영상·약물 전 임신 확인 등).
  treatment: "- " 줄 목록. 약물은 계열·대표 약 이름과 시작 조건, 비약물 치료, 기존 약 유지·조정, 하지 말아야 할 처치.
  disposition: "- 외래/입원/응급/의뢰 결정 : 근거(활력징후·중증 징후 유무)" 와 의뢰 과, 가족·보호자 역할. 오늘의 결정(오늘 처방하는지,
       무엇을 하지 않는지, 언제 결과를 보는지)을 분명히 씁니다.
  follow_up: "- n주 후 재진 : 확인할 것", 이후 추적 간격, "- 즉시 내원 : 경고 증상".
  education: "- 주제 : 내용" 4~6줄(질환 설명, 위험성, 약 사용·주의, 생활 관리, 다음 검사 안내). 대화에서 실제로 설명한 내용을 먼저 씁니다.
- diagnosis: 가장 가능성 높은 진단 하나를 교과서·ICD에서 쓰는 일반적인 질환명으로 "한국어 진단명 (English name)" 형식으로 씁니다.
  주호소 증상이 아니라 원인 질환을 쓰고(진단 추론 참고), 특정 약 이름 대신 약물 계열의 일반명을 씁니다. "원인 미상", "의증", "의심",
  증상명은 쓰지 않습니다. evidence·differential에 반대 소견을 적은 진단은 주진단으로 고르지 않습니다.
  원인이 정해지면 "질환(원인)" 형태로 씁니다(예: "급성 신우신염 (Acute pyelonephritis)", "철결핍빈혈 (Iron deficiency anemia)").
  증상·발작 유형·검사 이상이 아니라 질환을 씁니다(예: 발작 유형 대신 해당 뇌전증 증후군). 동반 질환·다른 진단을 덧붙이지 않습니다.
- 기록 전체는 간결하게 쓰되 확인한 정보는 최대한 담습니다. 같은 사실을 여러 칸에 반복하지 않습니다(HPI에 쓴 동반 증상은 ROS에 (+)/(-)만).
- 안전: 위중한 원인을 배제하지 못했으면 그 계획을 쓰고, 해로운 처치(급격한 약 중단, 금기 약물)를 권하지 않습니다."""

TASKS = {
    'first': '[과제] 임상표현 목록에서 이 환자에게 가장 맞는 id 하나를 cp에 넣고(반드시 목록의 id), 계획(plan) JSON을 출력하세요.',
    'plan': '[과제] 진료 기록을 반영해 감별진단을 갱신하고 다음 계획(plan) JSON을 출력하세요.',
    'hypothesis': '[과제] 진료 기록을 반영해 감별진단을 갱신하고 가설(hypothesis) JSON을 출력하세요. 개수는 위 [개수]를 따릅니다.',
    'predict': '[과제] 진료 기록을 반영해 예측(predict) JSON을 출력하세요. 개수는 위 [개수]를 따릅니다.',
    'close': '[과제] 조사를 마쳤습니다. 마무리(close) JSON을 출력하세요.',
    'talk': '[과제] 다음에 할 말 하나를 {"say": "…", "covered": ["다 전한 id"]} JSON으로 출력하세요.',
    'audit': '[과제] 빠진 사실을 {"add": [...]} JSON으로 출력하세요.',
    'shorten': '[과제] 위 [고칠 문장]을 같은 순서로, 뜻은 그대로 두고 각각 30자 이하(공백 포함)의 완결된 문장으로 고쳐 {"lines": [...]} JSON으로 출력하세요. 긴 문장은 완결된 두 문장으로 나눕니다.',
    'record': '[과제] 진료가 끝났습니다. 기록(record) JSON을 출력하세요. O(진찰 소견)는 아래 [진찰 소견]을 코드가 그대로 쓰므로 출력하지 않습니다.',
}
# Top-level keys each task's answer must have. Without them parse_json would accept an inner object
# (e.g. {"dx":…,"p":…} from a malformed plan), and the encounter would stall with no actions.
REQUIRED = {'first': ('cp', 'actions'), 'plan': ('actions',), 'hypothesis': ('ask', 'exams'), 'predict': ('questions', 'exams'), 'close': ('agenda',),
            'talk': ('say',), 'audit': ('add',), 'shorten': ('lines',), 'record': ('S', 'A', 'P')}
# The closing dialog: what the patient is told, in order (CPX 환자 교육 items the comm graders asked for), one short call per turn.
AGENDA = ('dx', 'concern', 'tests', 'treat', 'why', 'prognosis', 'life', 'redflag')  # tests/treat early: the patient's follow-ups eat turns
# (v42 server comm, all three cases: 'which tests were not explained', the worries not answered directly)
TALK_SYSTEM = f"""당신은 한국 외래 의사이고, 진료를 마무리하며 환자에게 설명을 한 문장씩 말합니다. JSON 객체 하나만 출력합니다.
- say: 25자 안팎(최대 {MAX_CHARS}자, 공백 포함)의 완결된 존댓말 문장 하나에 생각 하나. 넘으면 버려집니다. 의학 용어는 쉬운 말로. 쉼표로 나열하지 않습니다.
- 증상·검사·약·치료는 구체적인 이름으로 말합니다("이런 증상", "그것" 같은 지시어로 대신하지 않음).
- 진단명은 쉬운 말 풀이와 함께 말합니다(예: "~가 의심돼요. 쉽게 말해 ~한 상태예요." 두 문장으로 나눠도 됩니다).
- [환자의 마지막 말]이 질문이면 이번 문장은 그 질문에 대한 구체적인 답입니다. 답이 [전할 내용]의 남은 항목이면 그 내용으로 답하고 covered에 넣습니다.
  걱정·두려움을 말했으면 그 걱정을 받아 주고 근거를 들어 안심시키거나 계획을 말합니다.
- 아니면 [전할 내용]에서 남은 첫 항목을 이어서 전합니다. 검사와 치료는 한 문장에 함께 말해도 됩니다(예: "피검사로 확인하고 약을 시작해요"). 한 항목을 1~2문장에 걸쳐 전할 수 있고, 다 전한 항목의 id를 covered에 넣습니다.
- 검사는 무슨 검사를 하겠다는 것만 한 문장으로 말합니다. 결과가 나오는 시간, 예상 수치, 입원 기간처럼 지금 알 수 없는 것은
  지어내지 않고, 물으면 "검사 결과를 보고 자세히 설명드릴게요"처럼 짧게 답합니다.
- [마무리 대화]에서 이미 한 말을 되풀이하지 않습니다. 이해 확인 질문과 끝인사는 따로 하므로 쓰지 않습니다."""
AUDIT_SYSTEM = """당신은 초진 기록 검토자입니다. JSON 객체 하나만 출력합니다.
[환자 문답]의 환자 답 하나하나에 대해, 그 답의 사실이 [초진 기록 S]에 그대로 있는지 확인합니다: 약 이름·용량, 수치, 시점·기간,
빈도, 경과(예: "반 알 → 1알 → 3~4알"), 부정 표현("설사까지는 아님"), 누가 무슨 병인지. 빠졌거나 줄여 쓴 것은 환자 답에 가깝게
개조식 한 줄로 넣을 칸과 함께 씁니다(줄여 쓴 것은 자세한 줄을 새로 씀). 환자가 말하지 않은 것은 쓰지 않습니다. 없으면 빈 목록입니다.
{"add": [{"field": "hpi|pmh|fh|sh|ros", "line": "…"}]}"""
SHORT_SYSTEMS = {'talk': TALK_SYSTEM, 'audit': AUDIT_SYSTEM}
FEELINGS = [(r'걱정', '걱정되셨겠어요.'), (r'무서|두려|겁나|불안', '많이 불안하셨겠어요.'),
            (r'힘들|괴로|속상|우울|스트레스', '많이 힘드셨겠어요.')]
NEGATED = re.compile(r'^(아니|아뇨)|없어요|않아요|없었어요|안 해요')
OPEN_QUESTION = '어떤 점이 불편하신지 자세히 말씀해 주세요.'


def nfc(text):
    return unicodedata.normalize('NFC', str(text or '')).strip()


def parse_json(text, require=()):
    """First JSON object in the model output (code fences or prose around it are fine) that has the `require` keys."""
    decoder = json.JSONDecoder()
    for match in re.finditer(r'\{', text or ''):
        try:
            value, _ = decoder.raw_decode(text, match.start())
        except ValueError:
            continue
        if isinstance(value, dict) and all(key in value for key in require):
            return value
    return None


def log(message):
    sys.stderr.write(f'[agent] {message}\n')


DOCTOR = '안녕하세요, 의사 김민수입니다.'
GOODBYE = '오늘 수고하셨어요. 결과 보고 다시 뵐게요.'
EXAM_INTRO = '이제 진찰할게요. 불편하시면 말씀해 주세요.'  # CPX: explain before the physical exam
GOODBYE_ADMIT = '오늘 수고하셨어요. 입원해서 잘 치료받으세요.'
ASK_QUESTIONS = '이해되셨어요? 궁금한 점 있으세요?'
RED_FLAG = re.compile(r'바로|즉시|응급|당장')


def red_flag_line(card):
    """A spoken warning when the closing lines have none: the card's first short red flag, else a general one."""
    for flag in (card or {}).get('plan', {}).get('red_flags', []):
        flag = re.split(r'[,(·]', str(flag))[0].strip()
        flag = flag[:-1] + '으면' if flag.endswith('음') else flag[:-1] + '하면' if flag.endswith('함') else f'{flag} 있으면'
        line = f'{flag} 바로 오세요.'  # "소변이 나오지 않음" → "소변이 나오지 않으면 바로 오세요."
        if len(line) <= MAX_CHARS:
            return line
    return '증상이 심해지면 바로 병원에 오세요.'


def greeting(opening):
    """First SAY: greeting, self-introduction and identity check (CPX communication items) in one turn."""
    match = re.search(r'(?:이름은|저는|나는)\s*([가-힣]{2,4}?)(?:이고|이며|입니다|이에요|예요|이다)', opening)
    if not match:
        return f'{DOCTOR} 성함이 어떻게 되세요?'
    name = match.group(1)
    if is_guardian(opening):
        given = name[1:] if len(name) == 3 else name
        given += '이' if (ord(given[-1]) - 0xAC00) % 28 else ''  # 도현 → 도현이, 지수 → 지수
        parent = '어머님' if re.search('엄마|어머니', opening) else '아버님' if re.search('아빠|아버지', opening) else '보호자분'
        return f'{DOCTOR} {given} {parent}이시죠?'
    return f'{DOCTOR} {name}님 맞으시죠?'


def childbearing(opening):
    """A woman aged 15-49 by the opening ("서른두 살 여자"): pregnancy and the last period are routine questions."""
    return bool(re.search(r'여자|여성', opening) and re.search(r'(열[다여일아]|스물|서른|마흔)\S*\s*살|(1[5-9]|[2-4]\d)\s*살', opening))


def is_guardian(opening):
    return '보호자' in opening or bool(re.search(r'(엄마|아빠|어머니|아버지).{0,15}(왔|함께|데리고)', opening))


def open_question(opening):
    """Second SAY: an open question (the patient mostly restates the chief complaint, but CPX grades asking it)."""
    return '어떤 점이 걱정되시는지 자세히 말씀해 주세요.' if is_guardian(opening) else OPEN_QUESTION


def fit_one(text):
    """A closing line within 30 characters without breaking a sentence: drop a leading interjection ("네, ") or split at
    sentence ends only; [] when that is not enough (a word split read "…상태로" / "보입니다." on the v21 bench)."""
    text = nfc(text)
    if len(text) <= MAX_CHARS:
        return [text] if text else []
    text = re.sub(r'^(네|예|아|음|자|그럼)[,.!]?\s+', '', text)
    parts = [part for part in re.split(r'(?<=[.!?])\s+', text) if part]
    return parts if all(len(part) <= MAX_CHARS for part in parts) else []


def split_say(text):
    """An explanation longer than one SAY becomes several SAYs (≤30 chars), split at sentence and comma breaks."""
    text = nfc(text)
    if len(text) <= MAX_CHARS:
        return [text] if text else []
    words = [w for piece in re.split(r'(?<=[.!?,])\s+', text) for w in ([piece] if len(piece) <= MAX_CHARS else piece.split())]
    chunks = ['']
    for word in words:
        joined = f'{chunks[-1]} {word}'.strip()
        if len(joined) <= MAX_CHARS:
            chunks[-1] = joined
        else:
            chunks.append(word[:MAX_CHARS])
    return [c for c in chunks if c]


def chat_waiting(client, enc, task, messages, max_tokens, effort, request_id):
    """client.chat, polling the same request while the server answers REQUEST_PENDING (v30 server: a high-effort
    record call gave up after ~7 s twice and the fallback record was submitted in 2 of 3 cases)."""
    waited = 0
    while True:
        try:
            return client.chat(messages, max_tokens=max_tokens, reasoning_effort=effort, request_id=request_id)
        except NovaError as exc:
            if exc.code != 'REQUEST_PENDING' or waited >= PENDING_WAIT or enc.seconds_left() < 60:
                raise
            time.sleep(10)
            waited += 10
            log(f'{task}: request pending, waited {waited} s')


def ask(client, enc, task, extra='', effort='medium', max_tokens=8000, tries=3):
    """The Doctor model's JSON answer for `task` (with its REQUIRED keys), or None. Swap point for the finals."""
    if task in SHORT_SYSTEMS:  # short prompts for the closing lines and the record audit keep these calls cheap
        messages = [{'role': 'system', 'content': SHORT_SYSTEMS[task]}, {'role': 'user', 'content': f'{extra}\n\n{TASKS[task]}'}]
    else:
        messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': context(enc, task, extra)}]
    request_id = str(uuid.uuid4())
    for attempt in range(tries + 2):
        try:
            completion = chat_waiting(client, enc, task, messages, max_tokens, effort, request_id)
        except NovaError as exc:
            if exc.code in SAME_ID_CODES and attempt < tries + 1:
                time.sleep(2 + 3 * attempt)  # same request: the server may still be working on it
                continue
            if exc.code == 'PROVIDER_RESPONSE_INVALID' and attempt < tries:
                request_id, max_tokens = str(uuid.uuid4()), min(16000, int(max_tokens * 1.5))  # empty answer: more room
                continue
            if exc.code == 'TOKEN_LIMIT_EXCEEDED' and max_tokens > 8000:
                request_id, max_tokens = str(uuid.uuid4()), 8000  # documented cap outside sessions; be safe inside too
                continue
            log(f'chat failed: {exc.code}')
            return None  # call/token limits or a persistent failure: callers fall back
        text = completion['choices'][0]['message']['content']
        value = parse_json(text, REQUIRED[task])
        if value is not None:
            return value
        log(f'{task}: no JSON with {REQUIRED[task]} ({len(text or "")} chars): {(text or "")[:300]!r}')
        request_id = str(uuid.uuid4())  # unparsable output: ask again as a new request
    return None


# -- prompt pieces ---------------------------------------------------------------------------------------------------
def rank_cards(cards, opening, top=5):
    def score(card):
        names = [w for w in re.split(r'[/()\s]+', card['name']) if len(w) >= 2]
        return sum(opening.count(w) for w in card.get('kw', []) + names if w)
    return sorted(cards, key=score, reverse=True)[:top]


def card_text(card):
    """The part of the card that does not change during the encounter (stays in the cached prompt prefix)."""
    lines = [f"[임상표현 지식] {card['id']}. {card['name']} ({card.get('type', '진단')}형)",
             '원인: ' + ', '.join(f"{c['dx']}({c.get('tier', '')})" for c in card.get('causes', []))]
    if card.get('dx_notes'):
        lines.append('원인별 핵심 감별 소견 (주진단을 정하기 전에 문진·진찰로 확인):')
        lines += [f"- {n.get('dx')}: {n.get('key')}" for n in card['dx_notes'] if isinstance(n, dict)]
    plan = card.get('plan', {})
    lines.append('계획 참고: ' + json.dumps(plan, ensure_ascii=False))
    if card.get('counsel'):
        lines.append('상담 요점: ' + ' / '.join(card['counsel']))
    return '\n'.join(lines)


def candidates_text(candidates, cards):
    """Keyword-ranked candidates in detail, then every presentation by name (the ranking can miss)."""
    detail = '\n'.join(f"{c['id']}. {c['name']} ({c.get('type', '진단')}형) — 원인: "
                       + ', '.join(x['dx'] for x in c.get('causes', [])[:8]) for c in candidates)
    return f"[임상표현 후보]\n{detail}\n그 밖의 임상표현: " + ', '.join(f"{c['id']}. {c['name']}" for c in cards)


def remaining_text(enc):
    """Checklist items not yet covered and exams not yet done (shrinks as the encounter goes on)."""
    hx = [h for h in enc.card.get('hx', []) if h['id'] not in enc.covered]
    lines = [f'[남은 문진 항목] (id [분류/칸] 내용 — 질문 예시) {len(enc.card.get("hx", [])) - len(hx)}개 확인됨']
    lines += [f"{h['id']} [{h.get('tag', '')}/{h.get('field', '')}] {h.get('item', '')} — {h['q']}" for h in hx]
    lines.append('[진찰 목록] (id: 동작, 이미 한 진찰은 뺐음)')
    lines += [f'{k}: {v}' for k, v in COMMON_EXAMS.items() if v not in enc.done_texts]
    lines += [f"{e['id']}: {e['phrase']} — {e.get('why', '')}" for e in enc.card.get('exams', [])
              if exam_phrase(enc, e) not in enc.done_texts]
    for cp_id in enc.drills:  # related presentations: their history questions and key exams, ids prefixed "17:"
        card = enc.cards[cp_id]
        items = [h for h in card.get('hx', []) if h.get('field') == 'hpi'
                 and f"{cp_id}:{h['id']}" not in enc.covered and h['q'] not in enc.done_texts][:15]
        lines.append(f"[심화 문진: {cp_id}. {card['name']}] (관련 소견을 주 증상처럼 깊이 확인)")
        lines += [f"{cp_id}:{h['id']} [{h.get('tag', '')}] {h.get('item', '')} — {h['q']}" for h in items]
        lines += [f"{cp_id}:{e['id']}: {e['phrase']} — {e.get('why', '')}" for e in card.get('exams', [])[:3]
                  if exam_phrase(enc, {'id': f"{cp_id}:{e['id']}"}) not in enc.done_texts]
    return '\n'.join(lines)


def context(enc, task, extra=''):
    initial = enc.initial
    parts = [card_text(enc.card) if enc.card else candidates_text(enc.candidates, enc.cards.values()),
             f"[시작 정보]\n첫 진술: {initial['opening']}\n활력징후: {initial['vital_signs']}",
             '[진료 기록]\n' + ('\n'.join(enc.lines) or '(아직 없음)')]
    if enc.card and task in ('plan', 'hypothesis', 'predict'):
        parts.append(remaining_text(enc))
    if enc.card and task == 'hypothesis':
        parts.append('[관련 임상표현 후보] ' + ', '.join(f"{c['id']}. {c['name']}" for c in enc.cards.values()
                                                   if c['id'] != enc.card['id']))
    parts.append('[현재 상태]\n' + enc.status_text())
    if extra:
        parts.append(extra)
    parts.append(TASKS[task])
    return '\n\n'.join(parts)


def dx_key(name):
    """A diagnosis name for matching: no parentheses or spaces, one spelling of 갑상선/갑상샘."""
    return re.sub(r'\([^)]*\)|\s', '', str(name or '')).replace('갑상선', '갑상샘')


def bank_exams(enc):
    """The card's exam-bank maneuvers for the cause closest to the leading diagnosis ([] when nothing is close)."""
    top = dx_key(enc.cause or enc.top_dx())  # the card cause luna named in `predict`, else its leading diagnosis
    banks = (enc.card or {}).get('exam_bank') or []
    scored = [(difflib.SequenceMatcher(None, top, dx_key(b['dx'])).ratio(), b) for b in banks]
    score, bank = max(scored, key=lambda pair: pair[0], default=(0, None))
    return list(bank['exams']) if bank and score >= 0.4 else []


def card_question(enc, key):
    """The card's question for a checklist id ("h3" on the main card, "13:h2" on a related one), or ''."""
    cp_id, _, item = str(key).strip().rpartition(':')
    card = enc.cards.get(cp_id) if cp_id else enc.card
    return next((nfc(h['q']) for h in (card or {}).get('hx', []) if h['id'] == item), '')


SUBSTANCE = re.compile(r'약|수면제|술|진통제|안정제|담배')
TOLERANCE = re.compile(r'늘었|늘어|늘려|더 먹|더 마셔|많아졌|처음엔.{0,20}(요즘|지금)|알까지|알로|알에서|에서 (이젠|지금은|요즘은)')  # a dose that went from half a tablet to several
STOPPING = re.compile(r'안 드시|안 먹|끊|줄이|거르|못 드시')
WITHDRAWAL = re.compile(r'불안|떨|땀|두근|못 자|초조|잠을 못|잠이 안')
HIGH_DOSE = re.compile(r'(세|네|다섯|여섯|일곱|여덟|서너|[3-9]) ?알')  # tolerance stated as a dose ('요즘은 밤에 여섯 알')
CUT_DOWN = re.compile(r'줄여 ?보|줄이려|끊으려|끊어 ?보|줄이다|끊다가')


SCREENED = ('수면제', '안정제', '술')  # not 진통제: high-dose NSAIDs (S04 ulcer bleeding) are a cause, not a dependence
HEAVY_USE = re.compile(r'없이는|없으면|못 자|떨어질까|매일|더 먹')


def substance_screen(enc):
    """Use-disorder screening questions (tolerance, withdrawal, cut-down) for a drug or alcohol the patient uses heavily,
    not yet asked. luna anchored on its first hypothesis (v36 C01: several tablets a night and fear of running
    out, never asked about escalation or stopping, diagnosed chronic insomnia)."""
    pairs = [line.split(' 의사: ', 1)[1].split(' / 환자: ', 1) for line in enc.lines if ' / 환자: ' in line and ' 의사: ' in line]
    asked = ' '.join(q for q, _ in pairs)
    for name in SCREENED:
        if any(name in q + a and (HIGH_DOSE.search(a) or TOLERANCE.search(a) or HEAVY_USE.search(a)) for q, a in pairs):
            questions = [(r'늘었|늘어', f'{name} 양이 점점 늘었나요?'), (r'안 드시면|끊으면|안 먹으면', f'{name} 안 드시면 어떠세요?'),
                         (r'줄이거나|줄여|끊어', f'{name} 줄이거나 끊어 보신 적 있나요?')]
            return [q for pattern, q in questions if not (name in asked and re.search(pattern, asked))]
    return []


def dependence_note(enc):
    """A diagnosis check for the record when the answers show tolerance and withdrawal for a drug or substance (DSM use
    disorder criteria). luna kept "chronic insomnia" in C01 on four server versions despite both in its own record."""
    pairs = [line.split(' 의사: ', 1)[1].split(' / 환자: ', 1) for line in enc.lines if ' / 환자: ' in line and ' 의사: ' in line]
    talk = [(q, a) for q, a in pairs if SUBSTANCE.search(q + a)]
    tolerance = any(TOLERANCE.search(a) or HIGH_DOSE.search(a) for _, a in talk)
    withdrawal = any(WITHDRAWAL.search(a) and (STOPPING.search(q) or STOPPING.search(a)) for q, a in talk)
    cut_down = any(CUT_DOWN.search(a) for _, a in talk)
    if tolerance + withdrawal + cut_down < 2:  # DSM: two or more criteria
        return ''
    return ('[진단 점검] 환자 답에서 같은 약물·물질의 사용장애 기준(내성·고용량, 중단 시 금단, 감량 시도 실패) 중 2개 이상이 확인됐습니다. '
            'DSM 기준상 그 물질(계열)의 사용장애가 주진단 후보입니다. 주호소가 물질 사용 자체로 설명되면(불면, 조절 실패, 금단) '
            '사용장애가 주진단이고, 장기 손상(간·췌장·위장관 출혈 등) 때문에 왔으면 그 질환이 주진단이며 사용장애는 감별·계획에 씁니다.')


def feeling(reply):
    """A short empathy phrase for an emotional patient reply ('' for none or a negated one: "스트레스는 없어요")."""
    if NEGATED.search(reply or ''):
        return ''
    return next((phrase for pattern, phrase in FEELINGS if re.search(pattern, reply or '')), '')


def agenda(raw, enc):
    """[(id, point)] in AGENDA order from the close answer; a red-flag point always, a basic one when the call failed."""
    raw = raw if isinstance(raw, dict) else {'dx': f'검사로 확인해야 하지만 {enc.top_dx()} 가능성이 높아요.',
                                               'tests': '정확한 원인은 검사로 확인하겠습니다.'}  # said as is if talk fails too
    items = [(key, nfc(to_text(raw.get(key)))) for key in AGENDA]
    items = [(key, point) for key, point in items if point]
    if not any(key == 'redflag' for key, _ in items):
        items.append(('redflag', red_flag_line(enc.card)))
    return items


def talk_text(enc, items, done, start, turn_now=None):
    """The talk call's whole context: agenda with done marks and the closing dialog so far (not the history)."""
    lines = [f"[진단] {enc.top_dx()}", f"[호칭] {enc.addressee or '환자분'}", '[전할 내용] (id 상태: 요지)']
    lines += [f"{key} ({'전함' if key in done else '남음'}): {point}" for key, point in items]
    lines.append('[마무리 대화]\n' + ('\n'.join(enc.lines[start:]) or '(아직 없음. 첫 문장입니다)'))
    if enc.lines[start:] and ' / 환자: ' in enc.lines[-1]:
        lines.append('[환자의 마지막 말] ' + enc.lines[-1].split(' / 환자: ', 1)[1])
    if turn_now:
        lines.append(f'[이번 차례] {turn_now}')
    return '\n'.join(lines)


# -- validation ------------------------------------------------------------------------------------------------------
def exam_phrase(enc, action):
    key = str(action.get('id', ''))
    phrase = COMMON_EXAMS.get(key)
    cp_id, _, exam_id = key.rpartition(':')  # "17:e2" = exam e2 of drill card 17
    card = enc.cards.get(cp_id) if cp_id else enc.card
    if phrase is None and card:
        phrase = next((e['phrase'] for e in card.get('exams', []) if e['id'] == exam_id), None)
    phrase = nfc(phrase or action.get('text'))
    return phrase if len(phrase) <= MAX_CHARS else base_name(phrase)  # drop the English name if it does not fit


def base_name(text):
    """An action without its trailing English name: "정신상태검사 (Mental status exam)" and "정신상태검사" are one exam."""
    return re.sub(r'\s*\([^)]*\)\s*$', '', text)


def validate(enc, actions, limit=5):
    """Keep only actions the server will accept and that are new; returns [(kind, text)]."""
    kept, done = [], {base_name(t) for t in enc.done_texts}
    for action in actions if isinstance(actions, list) else []:
        if not isinstance(action, dict):
            continue
        kind = str(action.get('type', '')).upper()
        text = nfc(action.get('text')) if kind == 'SAY' else exam_phrase(enc, action) if kind == 'EXAM' else ''
        if not 0 < len(text) <= MAX_CHARS or base_name(text) in done | {base_name(t) for _, t in kept}:
            continue
        if kind == 'SAY' and any(word in text for word in BANNED):
            continue
        if kind == 'EXAM' and (any(word in text for word in BUNDLED) or '활력' in text
                               or ('계측' in text and not is_guardian(enc.initial['opening']))):  # adult tables lack it
            continue
        kept.append((kind, text))
    return kept[:limit]


# -- record ----------------------------------------------------------------------------------------------------------
def exam_region(card, phrase):
    for key, name in COMMON_EXAMS.items():
        if name == phrase:
            return COMMON_REGION[key]
    for exam in (card or {}).get('exams', []):
        if phrase in (exam['phrase'], re.sub(r'\s*\([^)]*\)\s*$', '', exam['phrase'])):
            return CARD_REGION.get(exam.get('region'), 'Others')
    return 'Others'


def exam_text(vital_signs, exams, card):
    """O.exam written by code from what was actually examined: V/S line, then findings grouped by region with turns."""
    vitals = []
    for line in vital_signs.splitlines():
        name, _, value = line.partition(':')
        vitals.append(f"{VITAL_NAMES.get(name.strip(), name.strip())} {' '.join(value.split())}".strip())
    lines = ['V/S : ' + ', '.join(v for v in vitals if v) + ' (시작 정보)']
    grouped = {}
    for turn, name, result in exams:
        grouped.setdefault(exam_region(card, name), []).append(f'[T{turn:02d}] {name} - {result}')
    lines += [f"{region} : {'; '.join(grouped[region])}" for region in REGIONS if region in grouped]
    return '\n'.join(lines)


def to_text(value):
    if isinstance(value, list):
        return '\n'.join(map(str, value))
    if isinstance(value, dict):
        return '\n'.join(f'{k} : {v}' for k, v in value.items())
    return str(value or '')


def find_key(value, key):
    """The first value stored under `key` (any case) anywhere in a nested answer, or None."""
    if isinstance(value, dict):
        for name, inner in value.items():
            if str(name).lower() == key:
                return inner
        value = list(value.values())
    for inner in value if isinstance(value, list) else []:
        found = find_key(inner, key)
        if found is not None:
            return found
    return None


def routine_answers(enc, field):
    """S.pmh/fh/sh from the code-asked routine questions and their answers, for a record that left the field empty."""
    asked = [q for q, f in ROUTINE_FIELD.items() if f == field]
    lines = [line.split(' 의사: ', 1)[1] for line in enc.lines if ' 의사: ' in line and any(q in line for q in asked)]
    return '\n'.join('Others : ' + line.replace(' / 환자: ', ' → ') for line in lines)


def audit_text(enc, note):
    """The audit call's whole context: the doctor-patient exchanges and the written S."""
    talk = [line.split(' 의사: ', 1)[1] for line in enc.lines if ' / 환자: ' in line and ' 의사: ' in line]
    record = '\n'.join(f'[{field}]\n{note["S"][field]}' for field in RECORD_FIELDS['S'])
    return '[환자 문답]\n' + '\n'.join(talk) + '\n\n[초진 기록 S]\n' + record


def add_lines(note, answer):
    """Append the audit's lines to their S fields (at most 12, only fields that exist, no exact repeats)."""
    added = 0
    for item in (answer or {}).get('add') or []:
        if not isinstance(item, dict):
            continue
        field, line = str(item.get('field', '')).strip(), nfc(item.get('line'))
        if field in note['S'] and line and line not in note['S'][field] and added < 12:
            note['S'][field] = (note['S'][field] + '\n' + line).strip()
            added += 1
    return added


def to_note(enc, result):
    note, missing = empty_record(), []
    for section in ('S', 'A', 'P'):
        values = result.get(section) if isinstance(result.get(section), dict) else {}
        for field in RECORD_FIELDS[section]:
            value = values.get(field)
            if value is None or not to_text(value).strip():
                value = find_key(result, field)  # luna/Haiku sometimes put S.pmh… beside S (v23 bench: 2 of 9 records)
            note[section][field] = to_text(value).strip()
            if not note[section][field] and field in ('pmh', 'fh', 'sh'):
                note[section][field] = routine_answers(enc, field)
                missing.append(f'{section}.{field}')
    if missing:
        log(f'record: {", ".join(missing)} empty in the answer, filled from the routine history')
    note['O']['exam'] = enc.exam_text()  # findings verbatim with turns: the model's rewrite kept 83% of them (v26 bench)
    diagnosis = ' '.join(to_text(result.get('diagnosis')).split()) or enc.top_dx()
    return note, diagnosis.encode()[:450].decode(errors='ignore')


def fallback_note(enc):
    """Record without the model: the transcript itself, so nothing unconfirmed is written."""
    note = empty_record()
    note['S']['cc'] = 'C.C : ' + enc.initial['opening'][:200]
    note['S']['hpi'] = '\n'.join(line for line in enc.lines if '의사:' in line or '환자:' in line)
    note['O']['exam'] = enc.exam_text()
    dx = enc.top_dx()
    note['A']['evidence'] = f'#1. {dx}'
    note['A']['differential'] = '\n'.join(f"r/o {d['dx']}" for d in enc.ddx[1:6] if isinstance(d, dict))
    return note, dx
