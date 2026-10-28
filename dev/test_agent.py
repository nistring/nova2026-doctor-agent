"""Offline tests for agent/ with a scripted fake client (no model or network calls).

  python dev/test_agent.py
"""
import json
import re
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'agent'))
import brain  # noqa: E402
import run  # noqa: E402
from nova_sdk import RECORD_FIELDS, NovaError  # noqa: E402

brain.time.sleep = run.time.sleep = lambda seconds: None

CARD = {'id': '40', 'name': '체중 감소', 'type': '진단', 'kw': ['살', '체중'], 'causes': [{'dx': '갑상샘항진증', 'tier': '흔함'}],
        'hx': [{'id': 'h1', 'field': 'hpi', 'tag': '필수', 'item': '시작', 'q': '언제부터 빠지셨어요?'},
               {'id': 'h2', 'field': 'hpi', 'tag': '필수', 'item': '식욕', 'q': '식욕은 어떠세요?'}],
        'exams': [{'id': 'e1', 'phrase': '갑상선 안징후 검사 (Thyroid eye signs, lid lag)', 'region': 'HEENT'}],
        'plan': {'tests': [{'t': 'TFT', 'why': '갑상샘'}]}}
OTHER = {'id': '28', 'name': '수면장애', 'type': '진단', 'kw': ['잠'], 'causes': [], 'hx': [], 'exams': [], 'plan': {}}

PLAN_FIRST = {'cp': '40', 'mode': '진단', 'addressee': '환자분', 'ddx': [{'dx': '갑상샘항진증', 'p': 0.6}],
              'actions': [{'type': 'SAY', 'text': '안녕하세요, 언제부터 빠졌나요?'}, {'type': 'SAY', 'text': '식욕은 어떠세요?'}]}
PLAN_NEXT = {'ddx': [{'dx': '갑상샘항진증(그레이브스병)', 'p': 0.8}], 'actions': [{'type': 'EXAM', 'id': 'e1'}], 'done': False}
PLAN_DONE = {'ddx': [{'dx': '갑상샘항진증(그레이브스병)', 'p': 0.8}], 'actions': [], 'done': True}
HYPO = {'ddx': [{'dx': '그레이브스병', 'p': 0.7}], 'ask': ['h1', 'h2', 'h9'], 'related': ['28'],
        'extra': ['손이 떨리세요?', '눈이 튀어나온 느낌이 있나요?', '가' * 31, '목 앞이 부었나요?', '불안하세요?', '여섯 번째는 버려요?'],
        'exams': [{'id': 'c8'}, {'text': '갑상선 잡음 청진'}, '심부건반사']}  # h2 already asked, h9 unknown
PREDICT = {'ddx': [{'dx': '그레이브스병', 'p': 0.8}], 'questions': ['피부가 축축해지셨나요?', '손이 떨리세요?'],  # 2nd: already asked
           'exams': HYPO['exams']}
CLOSE = {'agenda': {'dx': '갑상샘 호르몬이 많은 병이에요.', 'tests': '피검사를 해 볼게요.', 'redflag': '가슴이 심하게 뛰면 바로 오세요.'}}
ANSWER = '보통 1년 이상 꾸준히 드세요.'
MORE = '치료하면 대부분 좋아져요.'
RECORD = {'S': {'cc': 'C.C : 체중 감소', 'hpi': '3개월', 'pmh': '', 'fh': '', 'sh': '', 'ros': ['Palpitation (+)']},
          'A': {'evidence': '#1. 갑상샘항진증', 'differential': 'r/o 당뇨'},
          'P': {'tests': '- TFT : 확인', 'treatment': '', 'disposition': '', 'follow_up': '', 'education': ''},
          'diagnosis': '갑상샘항진증(그레이브스병)'}


class FakeClient:
    def __init__(self, doctor=None, minutes=30, say_errors=(), exam_errors=(), chat_errors=(), replies=None):
        self.turn, self.calls, self.diagnosed, self.chat_ids, self.say_ids = 0, 0, None, [], []
        self.doctor = doctor or self.default_doctor
        self.deadline = (datetime.now(timezone.utc) + timedelta(minutes=minutes)).isoformat()
        self.say_errors, self.exam_errors, self.chat_errors = list(say_errors), list(exam_errors), list(chat_errors)
        self.plans = 0
        self.said, self.replies, self.prompts, self.systems = [], replies or {}, [], []

    def default_doctor(self, user):
        if '가장 맞는 id 하나를' in user:
            return PLAN_FIRST
        if '가설(hypothesis)' in user:
            return HYPO
        if '예측(predict)' in user:
            return PREDICT
        if '다음 계획' in user:
            self.plans += 1
            return PLAN_NEXT if self.plans == 1 else PLAN_DONE
        if '마무리(close)' in user:
            return CLOSE
        if '기록(record)' in user:
            return RECORD
        if '[전할 내용]' in user:  # talk: answer the last question when told to, else say the first point left
            if '마지막 질문' in user:
                return {'say': ANSWER, 'covered': []}
            if '전할 내용은 다 전했습니다' in user:
                return {'say': MORE, 'covered': []}  # the same line again is dropped: the closing then moves on
            key, point = re.search(r'^(\w+) \(남음\): (.*)$', user, re.M).groups()
            return {'say': point, 'covered': [key]}
        return {}

    def session(self):
        return {'initial': {'opening': '살이 많이 빠져서 왔어요.', 'vital_signs': '혈압 : 128/84 mmHg\n맥박 : 96 회/분'},
                'turns_remaining': 50 - self.turn, 'deadline': self.deadline}

    def chat(self, messages, *, max_tokens, reasoning_effort=None, request_id=None, **_):
        self.calls += 1
        self.chat_ids.append(request_id)
        self.systems.append(messages[0]['content'])
        self.prompts.append(messages[-1]['content'])
        if self.chat_errors:
            raise NovaError(self.chat_errors.pop(0))
        value = self.doctor(messages[-1]['content'])
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        return {'choices': [{'message': {'content': text}}]}

    def _step(self, errors, text):
        if len(text) > 30:
            raise ValueError('too long')
        if self.turn >= 50:
            raise NovaError('TURN_LIMIT_REACHED')
        if errors:
            code = errors.pop(0)
            if code in run.TURN_REFUSALS:
                self.turn += 1
            raise NovaError(code) if isinstance(code, str) else code
        self.turn += 1
        return self.turn

    def say(self, text, *, request_id=None):
        self.say_ids.append(request_id)
        self.first_say = getattr(self, 'first_say', text)
        turn = self._step(self.say_errors, text)
        self.said.append(text)
        return {'status': 'OK', 'content': self.replies.get(text, '네.'), 'turn': turn, 'turns_remaining': 50 - turn}

    def exam(self, text, *, request_id=None):
        turn = self._step(self.exam_errors, text)
        return {'status': 'OK', 'result': '돌출(+/+)', 'turn': turn, 'turns_remaining': 50 - turn}

    def diagnose(self, *, note, diagnosis, request_id=None):
        assert set(note) == set(RECORD_FIELDS) and all(set(note[s]) == set(f) for s, f in RECORD_FIELDS.items())
        assert all(isinstance(v, str) for s in note.values() for v in s.values())
        assert diagnosis and '\n' not in diagnosis and len(diagnosis.encode()) <= 500
        self.diagnosed = (note, diagnosis)
        return {'status': 'completed'}


RUN_BATCH = run.run_batch


class AlwaysAsking(dict):
    def get(self, key, default=None):
        return '그럼 어떻게 하면 되나요?'


def consult(client):
    return run.consult(client, [OTHER, CARD])


class AgentTests(unittest.TestCase):
    def test_happy_path(self):
        client = FakeClient()
        consult(client)
        note, diagnosis = client.diagnosed
        self.assertEqual(diagnosis, '갑상샘항진증(그레이브스병)')
        self.assertTrue(client.first_say.startswith('안녕하세요, 의사'))
        self.assertIn('H/E/E/N/T : [T05] 갑상선 안징후 검사 - 돌출(+/+)', note['O']['exam'])  # English name dropped (>30 chars)
        self.assertIn('V/S : BP 128/84 mmHg, PR 96 회/분 (시작 정보)', note['O']['exam'])
        self.assertEqual(note['S']['ros'], 'Palpitation (+)')
        # greeting + open question + 2 SAY + 1 EXAM (broad phase ends at T5) + 1 card item + 4 extra questions (max 5)
        # + 13 routine (with the expectation question) + 3 hypothesis exams + 8 owed exams (12 minimum) + 3 agenda lines + question check + goodbye
        self.assertEqual(client.turn, 39)  # routine of 9; exams up to the budget; one extra explanation line
        self.assertEqual(client.said[1], brain.OPEN_QUESTION)
        self.assertTrue(any('바로' in text for text in client.said))  # a spoken red-flag warning is guaranteed
        self.assertLessEqual(client.calls, 15)  # plans, 2 selections, close, talks (incl. one extra explanation), record, audit

    def test_chat_retry_reuses_request_id(self):
        client = FakeClient(chat_errors=['PROVIDER_UNAVAILABLE', 'REQUEST_PENDING'])
        consult(client)
        self.assertEqual(client.chat_ids[0], client.chat_ids[1])
        self.assertEqual(client.chat_ids[1], client.chat_ids[2])
        self.assertIsNotNone(client.diagnosed)

    def test_pending_request_is_polled_with_the_same_id(self):
        client = FakeClient(chat_errors=['REQUEST_PENDING'] * 6)  # v30 server: a slow record call gave up after ~7 s
        consult(client)
        self.assertEqual(len(set(client.chat_ids[:7])), 1)
        self.assertNotIn('C.C : 살이 많이', client.diagnosed[0]['S']['cc'])  # not the fallback record

    def test_empty_answer_uses_new_request_id(self):
        client = FakeClient(chat_errors=['PROVIDER_RESPONSE_INVALID'])
        consult(client)
        self.assertNotEqual(client.chat_ids[0], client.chat_ids[1])

    def test_say_retry_same_id_and_one_turn(self):
        client = FakeClient(say_errors=['PROVIDER_UNAVAILABLE'])
        consult(client)
        self.assertEqual(client.say_ids[0], client.say_ids[1])
        self.assertEqual(client.turn, 39)  # same as the happy path: the retried SAY used one turn

    def test_refused_exam_uses_turn_and_is_not_repeated(self):
        client = FakeClient(exam_errors=['EXAM_NOT_SPECIFIC'])
        consult(client)
        self.assertEqual(client.turn, 40)  # refused exam used a turn, and 12 other exams are still done (common rows refill)
        self.assertNotIn('[T05]', client.diagnosed[0]['O']['exam'])

    def test_model_always_failing_still_diagnoses(self):
        client = FakeClient(doctor=lambda user: 'not json')
        consult(client)
        note, diagnosis = client.diagnosed
        self.assertEqual(diagnosis, '진단 미정')
        self.assertIn('언제부터 그러셨어요?', note['S']['hpi'])  # generic fallback questions (no presentation chosen)
        self.assertLessEqual(client.calls, 60)

    def test_unexpected_exception_still_diagnoses(self):
        client = FakeClient(say_errors=[RuntimeError('boom')])
        consult(client)
        self.assertIsNotNone(client.diagnosed)

    def test_stop_code_goes_to_record(self):
        client = FakeClient(say_errors=['TURN_LIMIT_REACHED'])
        consult(client)
        self.assertIsNotNone(client.diagnosed)

    def test_deadline_skips_to_record(self):
        client = FakeClient(minutes=3)
        consult(client)
        self.assertEqual(client.turn, 0)
        self.assertIsNotNone(client.diagnosed)

    def test_covered_items_leave_the_prompt(self):
        enc = run.Encounter(FakeClient().session(), [CARD])
        enc.card = CARD
        self.assertIn('언제부터 빠지셨어요?', brain.context(enc, 'plan'))
        enc.update({'covered': ['h1']})
        prompt = brain.context(enc, 'plan')
        self.assertNotIn('언제부터 빠지셨어요?', prompt)
        self.assertIn('1개 확인됨', prompt)
        enc.record_turn('EXAM', '갑상선 촉진', '비대(+)', 3)
        self.assertNotIn('c5: 갑상선 촉진', brain.context(enc, 'plan'))
        self.assertNotIn('[남은 문진 항목]', brain.context(enc, 'record'))

    def test_routine_history_asked_once_skipping_asked_items(self):
        client = FakeClient()
        consult(client)
        routine = [q for _, q in brain.ROUTINE_ADULT]
        self.assertEqual([q for q in client.said if q in routine], routine)  # all, in order, once (model finished early)
        self.assertLess(client.said.index(routine[-1]), client.said.index(CLOSE['agenda']['dx']))  # before the closing
        enc = run.Encounter(FakeClient().session(), [CARD])
        enc.card, enc.done_texts = CARD, {'담배는 하루 몇 개비 피우세요?', '평소 드시는 약이 있으신가요?'}
        said = []
        run.run_batch = lambda client, enc, actions, stop_on_question=True: said.extend(t for _, t in actions)
        try:
            run.routine(None, enc, now=True)
        finally:
            run.run_batch = RUN_BATCH
        self.assertNotIn('담배는 피우세요?', said)
        self.assertIn('술은 얼마나 드세요?', said)  # narrow patterns: a nearby question does not skip it
        self.assertNotIn('평소 드시는 약이 있으세요?', said)
        self.assertIn('하시는 일은 무엇인가요?', said)
        self.assertFalse(any('알레르기' in q or '수술' in q for q in said))  # not graded in the official model records
        enc = run.Encounter({'initial': {'opening': '보호자인 엄마가 아이를 데리고 왔다.', 'vital_signs': ''}}, [CARD])
        enc.card, said[:] = CARD, []
        run.run_batch = lambda client, enc, actions, stop_on_question=True: said.extend(t for _, t in actions)
        try:
            run.routine(None, enc, now=True)
        finally:
            run.run_batch = RUN_BATCH
        child = [q for _, q in brain.ROUTINE_CHILD]
        self.assertEqual(said, [child[0], brain.EXPECTATION[1], *child[1:]])  # no request in the opening

    def test_record_exam_written_by_model_or_code(self):
        enc = run.Encounter(FakeClient().session(), [CARD])
        enc.card = CARD
        enc.record_turn('EXAM', '갑상선 촉진', '비대(+)', 3)
        note, _ = brain.to_note(enc, {**RECORD, 'O': {'exam': 'V/S : BP 128/84 mmHg\nNeck : 갑상선 비대(+) (T03)'}})
        self.assertIn('Neck : [T03] 갑상선 촉진 - 비대(+)', note['O']['exam'])  # always code-built, the model's O ignored
        note, _ = brain.to_note(enc, {**RECORD, 'O': {'exam': ''}})
        self.assertIn('[T03] 갑상선 촉진 - 비대(+)', note['O']['exam'])

    def test_final_question_is_answered_past_target(self):
        old = run.TARGET_TURNS
        run.TARGET_TURNS = 5  # the encounter is already over budget when the closing starts
        try:
            client = FakeClient(replies={brain.ASK_QUESTIONS: '약은 얼마나 먹어야 하나요?'})
            consult(client)
        finally:
            run.TARGET_TURNS = old
        self.assertIn('보통 1년 이상 꾸준히 드세요.', client.said)
        self.assertEqual(client.said[-1], brain.GOODBYE)

    def test_drill_card_questions_and_exams(self):
        drill = {'id': '13', 'name': '두근거림', 'type': '진단', 'causes': [], 'plan': {},
                 'hx': [{'id': 'h1', 'field': 'hpi', 'tag': '필수', 'item': '빈도', 'q': '얼마나 자주 두근거리세요?'},
                        {'id': 'h2', 'field': 'fh', 'tag': '필수', 'item': '가족력', 'q': '가족 중 심장병 있으세요?'}],
                 'exams': [{'id': 'e1', 'phrase': '요골동맥 맥박 촉진', 'region': 'Extremities'}]}
        enc = run.Encounter(FakeClient().session(), [CARD, drill])
        enc.card = CARD
        enc.update({'drill': ['13', '40', '99']})  # own card and unknown ids are ignored
        self.assertEqual(enc.drills, ['13'])
        prompt = brain.context(enc, 'plan')
        self.assertIn('13:h1', prompt)
        self.assertNotIn('13:h2', prompt)  # only hpi-level items are drilled
        self.assertEqual(brain.validate(enc, [{'type': 'EXAM', 'id': '13:e1'}]), [('EXAM', '요골동맥 맥박 촉진')])
        enc.update({'covered': ['13:h1']})
        self.assertNotIn('13:h1', brain.context(enc, 'plan'))

    def test_malformed_plan_is_retried_not_taken_as_an_inner_object(self):
        # v10 on the server: luna's plan JSON did not parse as a whole, the first decodable object was {"dx":…,"p":…},
        # so no presentation and no actions were taken and C03/C10 ended after 10 turns.
        broken = '{"cp": "40", "ddx": [{"dx": "갑상샘항진증", "p": 0.6}], "actions": [{"type": "SAY" "text": "식욕은요?"}]}'
        self.assertEqual(brain.parse_json(broken), {'dx': '갑상샘항진증', 'p': 0.6})
        self.assertIsNone(brain.parse_json(broken, brain.REQUIRED['first']))
        broken_left = [broken, broken]
        client = FakeClient(doctor=lambda user: broken_left.pop() if broken_left else client.default_doctor(user))
        consult(client)
        self.assertNotIn(brain.GENERIC_HX[0], client.said)  # retried inside the first ask: no card-less fallback questions
        self.assertIn('식욕은 어떠세요?', client.said)

    def test_exam_repeated_with_or_without_english_name_is_dropped(self):
        enc = run.Encounter(FakeClient().session(), [CARD])
        enc.card = CARD
        enc.done_texts.add('정신상태검사 (Mental status exam)')
        self.assertEqual(brain.validate(enc, [{'type': 'EXAM', 'text': '정신상태검사'},
                                              {'type': 'EXAM', 'text': '갑상선 촉진'}, {'type': 'EXAM', 'id': 'c5'}]),
                         [('EXAM', '갑상선 촉진')])

    def test_cut_off_record_is_rejected(self):
        cut = '{"S": {"cc": "C.C : 객혈", "hpi": "3주"}, "A": {"evidence": "#1. 폐결핵"'  # max_tokens hit mid-answer
        self.assertIsNone(brain.parse_json(cut, require=('S', 'A', 'P')))
        self.assertEqual(brain.parse_json(cut), {'cc': 'C.C : 객혈', 'hpi': '3주'})  # what the old code took as the record
        client = FakeClient(doctor=lambda user: cut if '기록(record)' in user else FakeClient().default_doctor(user))
        consult(client)
        note = client.diagnosed[0]
        self.assertTrue(note['S']['cc'].startswith('C.C') and '의사:' in note['S']['hpi'])  # fallback record, not an empty S

    def test_long_closing_line_is_split(self):
        parts = brain.split_say('검사 결과를 보고 갑상샘 약을 시작할 예정이에요, 1~2주 뒤 다시 오세요.')
        self.assertTrue(len(parts) == 2 and all(len(p) <= 30 for p in parts))
        self.assertEqual(brain.split_say('짧은 설명이에요.'), ['짧은 설명이에요.'])

    def test_validate_filters(self):
        enc = run.Encounter(FakeClient().session(), [CARD])
        enc.card = CARD
        kept = brain.validate(enc, [
            {'type': 'SAY', 'text': '가' * 31}, {'type': 'SAY', 'text': '증상을 모두 말씀해 주세요'},
            {'type': 'EXAM', 'text': '복부 시진 및 촉진'}, {'type': 'EXAM', 'text': '활력징후 측정'},
            {'type': 'EXAM', 'id': 'c5'}, {'type': 'EXAM', 'id': 'c5'}, {'type': 'SAY', 'text': '식욕은 어떠세요?'},
            {'type': 'TEST', 'text': '피검사'}, 'junk'])
        self.assertEqual(kept, [('EXAM', '갑상선 촉진'), ('SAY', '식욕은 어떠세요?')])

    def test_hypothesis_questions_then_routine_then_exams(self):
        client = FakeClient()
        consult(client)
        said = client.said
        self.assertEqual(said[4:9], ['언제부터 빠지셨어요?', *HYPO['extra'][:2], *HYPO['extra'][3:5]])  # h1, then extras (max 5)
        self.assertEqual(said[9], PREDICT['questions'][0])  # the second list predicts the script's facts
        self.assertEqual(said[10], brain.ROUTINE_ADULT[0][1])
        self.assertNotIn(brain.EXAM_INTRO, said)  # v41: no exam introduction (official comm flat, the turn costs efficiency)  # routine history follows the hypothesis questions
        hypotheses = [p for p in client.prompts if '가설(hypothesis)' in p or '예측(predict)' in p]
        self.assertEqual(len(hypotheses), 2)
        self.assertIn('[개수] questions 8개', hypotheses[0])
        self.assertIn('[관련 임상표현 후보] 28. 수면장애', hypotheses[0])
        self.assertIn('[심화 문진: 28. 수면장애]', hypotheses[1])  # the related presentation's checklist
        self.assertIn('손이 떨리세요?', hypotheses[1])  # the second list is chosen after the first answers
        self.assertNotIn(brain.ROUTINE_ADULT[-1][1], hypotheses[1])  # ... before the routine history,
        self.assertIn('[개수] questions 7개', hypotheses[1])  # what is left after 9 routine items and 10 owed exams
        self.assertIn('[남은 문진 항목]', hypotheses[0])
        exam = client.diagnosed[0]['O']['exam']
        n = int(re.search(r'\[T(\d+)\] 갑상선 잡음 청진', exam).group(1))  # specific first, then the core rows in order
        self.assertIn(f'[T{n + 1}] 심음 청진', exam)
        self.assertIn(f'[T{n + 2}] 갑상선 촉진', exam)

    def test_failed_hypothesis_falls_back_to_the_plan_loop(self):
        fake = FakeClient()
        client = FakeClient(doctor=lambda user: 'not json' if '가설(hypothesis)' in user else fake.default_doctor(user))
        consult(client)
        self.assertEqual(sum('가설(hypothesis)' in p for p in client.prompts), 5)  # one ask (with retries), never again
        routine = [q for _, q in brain.ROUTINE_ADULT]
        self.assertEqual([q for q in client.said if q in routine], routine)
        self.assertEqual(client.said[-1], brain.GOODBYE)

    def test_empathy_prefix_for_emotional_replies(self):
        enc = run.Encounter(FakeClient().session(), [CARD])
        enc.record_turn('SAY', '약은요?', '약 줄이면 잠 못 잘까 봐 무서워요.', 3)
        self.assertEqual(enc.empathize('하루 몇 알 드세요?'), '많이 불안하셨겠어요. 하루 몇 알 드세요?')
        self.assertEqual(enc.empathize('하루 몇 알 드세요?'), '하루 몇 알 드세요?')  # owed once only
        enc.record_turn('SAY', '스트레스는요?', '아니요, 스트레스는 없어요.', 4)
        self.assertEqual(enc.empathize('잠은 어떠세요?'), '잠은 어떠세요?')  # negated
        enc.record_turn('SAY', '기분은요?', '요즘 많이 힘들어요.', 5)
        self.assertEqual(enc.empathize('가족 중 같은 병을 앓으신 분이 계신가요?'), '가족 중 같은 병을 앓으신 분이 계신가요?')  # too long
        enc.empathy = run.MAX_EMPATHY
        enc.record_turn('SAY', '걱정은요?', '걱정돼요.', 6)
        self.assertEqual(enc.empathize('언제부터요?'), '언제부터요?')  # cap reached

    def test_closing_dialog_uses_short_prompt_and_answers_the_last_question(self):
        client = FakeClient(replies={brain.ASK_QUESTIONS: '약은 얼마나 먹어야 하나요?'})
        consult(client)
        talks = [s for s, p in zip(client.systems, client.prompts) if '[전할 내용]' in p]
        self.assertEqual(len(talks), 6)  # 3 agenda points + 2 extra explanations (the repeat ends it) + the answer
        self.assertTrue(all(s == brain.TALK_SYSTEM for s in talks))
        self.assertTrue(all('[진료 기록]' not in p for p in client.prompts if '[전할 내용]' in p))  # history left out
        closing = client.said[client.said.index(CLOSE['agenda']['dx']):]
        self.assertEqual(closing, [CLOSE['agenda']['dx'], CLOSE['agenda']['tests'], CLOSE['agenda']['redflag'], MORE,
                                   brain.ASK_QUESTIONS, ANSWER, brain.GOODBYE])

    def test_closing_never_passes_turn_50_and_red_flag_is_kept(self):
        fake, lines = FakeClient(), iter(f'설명을 조금 더 드릴게요 {n}.' for n in range(100))
        doctor = lambda user: {'say': next(lines), 'covered': []} if '[전할 내용]' in user else fake.default_doctor(user)
        old = run.TARGET_TURNS
        run.TARGET_TURNS = 60  # agenda never covered, budget past the server limit
        try:
            client = FakeClient(doctor=doctor, replies=AlwaysAsking(on=True))  # the patient asks after every line (non-empty: replies or {})
            consult(client)
        finally:
            run.TARGET_TURNS = old
        self.assertEqual(client.turn, 50)
        self.assertEqual(client.said[-1], brain.GOODBYE)
        ask = client.said.index(brain.ASK_QUESTIONS)
        self.assertEqual(len(client.said) - ask, 4)  # check at T47, two answers, goodbye at T50
        self.assertTrue(any(brain.RED_FLAG.search(text) for text in client.said[ask - 2:ask]))  # forced before the check

    def test_repeated_talk_line_does_not_stall_the_closing(self):
        fake = FakeClient()
        doctor = lambda user: {'say': '검사를 해 볼게요.', 'covered': []} if '[전할 내용]' in user else fake.default_doctor(user)
        client = FakeClient(doctor=doctor)
        consult(client)
        self.assertEqual(client.said.count('검사를 해 볼게요.'), 1)
        self.assertEqual(client.said[-1], brain.GOODBYE)
        self.assertLess(client.turn, 40)

    def test_long_talk_line_is_asked_again_and_last_words_are_shown(self):
        fake, prompts = FakeClient(), []

        def doctor(user):
            if '[전할 내용]' in user:
                prompts.append(user)
                if '넘었습니다' not in user:
                    return {'say': '지금 나타나는 체중 감소와 두근거림은 갑상샘 때문입니다.', 'covered': []}
            return fake.default_doctor(user)
        client = FakeClient(doctor=doctor, replies={CLOSE['agenda']['dx']: '치료는 어떻게 하나요?'})
        consult(client)
        self.assertNotIn('때문입니다.', client.said)  # never the word-split tail of the long line
        self.assertIn('30자를 넘었습니다', prompts[1])
        self.assertTrue(any('[환자의 마지막 말] 치료는 어떻게 하나요?' in p for p in prompts))

    def test_closing_line_is_never_split_mid_sentence(self):
        self.assertEqual(brain.fit_one('네, 약을 천천히 줄이면 대부분 잠을 되찾을 수 있어요.'), ['약을 천천히 줄이면 대부분 잠을 되찾을 수 있어요.'])
        self.assertEqual(brain.fit_one('갑상샘이 너무 활발해요. 약으로 호르몬을 낮추면 체중도 돌아와요.'),
                         ['갑상샘이 너무 활발해요.', '약으로 호르몬을 낮추면 체중도 돌아와요.'])
        self.assertEqual(brain.fit_one('지금 드시는 수면제에 몸이 의존하게 된 상태로 보입니다요.'), [])

    def test_pregnancy_question_for_women_of_childbearing_age_and_red_flag_grammar(self):
        self.assertTrue(brain.childbearing('서른두 살 여자이다'))
        self.assertFalse(brain.childbearing('예순두 살 여자이다'))
        self.assertFalse(brain.childbearing('스물두 살 남자이다'))
        enc = run.Encounter({'initial': {'opening': '이름은 한지아이고, 나는 서른두 살 여자이다.', 'vital_signs': ''}}, [CARD])
        enc.card, said = CARD, []
        run.run_batch = lambda client, enc, actions, stop_on_question=True: said.extend(t for _, t in actions)
        try:
            run.routine(None, enc, now=True)
        finally:
            run.run_batch = RUN_BATCH
        self.assertEqual(said[-2:], [q for _, q in brain.ROUTINE_WOMAN])
        self.assertEqual(brain.red_flag_line({'plan': {'red_flags': ['소변이 아예 나오지 않음']}}), '소변이 아예 나오지 않으면 바로 오세요.')

    def test_record_fields_found_elsewhere_or_filled_from_routine(self):
        enc = run.Encounter(FakeClient().session(), [CARD])
        enc.card = CARD
        enc.record_turn('SAY', '담배는 피우세요?', '하루 반 갑 피워요.', 20)
        enc.record_turn('SAY', '부모님은 어떤 병을 앓으셨어요?', '어머니가 갑상샘 약 드세요.', 21)
        broken = {'S': {'cc': 'C.C : 체중 감소', 'hpi': '3개월'}, 'pmh': 'DM / HTN (-/-)', 'A': {}, 'P': {}}
        note, _ = brain.to_note(enc, broken)
        self.assertEqual(note['S']['pmh'], 'DM / HTN (-/-)')  # stored beside S
        self.assertEqual(note['S']['sh'], 'Others : 담배는 피우세요? → 하루 반 갑 피워요.')
        self.assertEqual(note['S']['fh'], 'Others : 부모님은 어떤 병을 앓으셨어요? → 어머니가 갑상샘 약 드세요.')

    def test_summary_before_exams_expectation_and_admission_goodbye(self):
        fake = FakeClient()

        def doctor(user):
            if '가설(hypothesis)' in user:
                return HYPO
            if '예측(predict)' in user:
                return {**PREDICT, 'summary': '정리하면 석 달간 7kg 빠지신 거죠?'}
            if '마무리(close)' in user:
                return {'agenda': {**CLOSE['agenda'], 'treat': '오늘 입원해서 수액 치료를 합니다.'}}
            return fake.default_doctor(user)
        client = FakeClient(doctor=doctor)
        consult(client)
        self.assertIn('갑상선 잡음 청진', client.diagnosed[0]['O']['exam'])
        self.assertNotIn('정리하면 석 달간 7kg 빠지신 거죠?', client.said)  # v41: no summary turn
        self.assertEqual(client.said[-1], brain.GOODBYE_ADMIT)
        enc = run.Encounter({'initial': {'opening': '수면제를 처방해 달라고 왔다.', 'vital_signs': ''}}, [CARD])
        enc.card, said = CARD, []
        run.run_batch = lambda client, enc, actions, stop_on_question=True: said.extend(t for _, t in actions)
        try:
            run.routine(None, enc, now=True)
        finally:
            run.run_batch = RUN_BATCH
        self.assertNotIn(brain.EXPECTATION[1], said)  # the opening already states the request

    def test_bundled_card_items_are_one_question_covering_all_ids(self):
        fake = FakeClient()
        bundle = {'ask': [{'ids': ['h1', 'h2'], 'q': '언제부터, 얼마나 빠지셨어요?'}], 'extra': [], 'related': []}
        client = FakeClient(doctor=lambda u: {**HYPO, **bundle} if '가설(hypothesis)' in u else fake.default_doctor(u))
        consult(client)
        self.assertIn('언제부터, 얼마나 빠지셨어요?', client.said)
        second = next(p for p in client.prompts if '예측(predict)' in p)
        self.assertIn('2개 확인됨', second)  # both ids left the checklist

    def test_dependence_check_and_adult_anthropometry(self):
        enc = run.Encounter(FakeClient().session(), [CARD])
        enc.record_turn('SAY', '수면제 양이 점점 늘었나요?', '네, 반 알에서 네 알로 늘었어요.', 10)
        self.assertEqual(brain.dependence_note(enc), '')  # tolerance alone
        enc.record_turn('SAY', '약을 안 드시면 어떠세요?', '불안하고 손이 떨려요.', 11)
        self.assertIn('사용장애', brain.dependence_note(enc))
        enc.card = CARD
        self.assertEqual(brain.validate(enc, [{'type': 'EXAM', 'text': '신체 계측 (Anthropometry)'}]), [])  # adult
        child = run.Encounter({'initial': {'opening': '엄마가 보호자로 함께 왔다.', 'vital_signs': ''}}, [CARD])
        child.card = CARD
        self.assertEqual(len(brain.validate(child, [{'type': 'EXAM', 'text': '신체 계측'}])), 1)

    def test_substance_screening_questions(self):
        enc = run.Encounter(FakeClient().session(), [CARD])
        enc.record_turn('SAY', '수면제는 요즘 어떻게 드세요?', '요즘은 자기 전에 다섯 알 먹어요.', 16)
        self.assertEqual(brain.substance_screen(enc), ['수면제 양이 점점 늘었나요?', '수면제 안 드시면 어떠세요?',
                                                      '수면제 줄이거나 끊어 보신 적 있나요?'])
        drinker = run.Encounter(FakeClient().session(), [CARD])
        drinker.record_turn('SAY', '술은 얼마나 드세요?', '매일 소주 두 병씩 마셔요.', 20)
        self.assertEqual(len(brain.substance_screen(drinker)), 3)  # the same three DSM questions for alcohol
        enc.record_turn('SAY', '수면제 양이 점점 늘었나요?', '네, 한 알에서 다섯 알로 늘었어요.', 17)
        self.assertNotIn('수면제 양이 점점 늘었나요?', brain.substance_screen(enc))
        calm = run.Encounter(FakeClient().session(), [CARD])
        calm.record_turn('SAY', '술은 얼마나 드세요?', '한 달에 한두 번 가볍게 마셔요.', 20)
        self.assertEqual(brain.substance_screen(calm), [])  # light use: no screening

    def test_record_audit_adds_missing_answered_facts(self):
        fake = FakeClient()
        audit = {'add': [{'field': 'sh', 'line': '평생 비흡연'}, {'field': 'nope', 'line': 'x'}, {'field': 'hpi', 'line': ''}]}
        client = FakeClient(doctor=lambda u: audit if '[초진 기록 S]' in u else fake.default_doctor(u))
        consult(client)
        note = client.diagnosed[0]
        self.assertTrue(note['S']['sh'].endswith('평생 비흡연'))
        self.assertEqual(sum(1 for s in client.systems if s == brain.AUDIT_SYSTEM), 1)  # one short-prompt call

    def test_exam_bank_for_the_named_cause_comes_first(self):
        banked = {**CARD, 'exam_bank': [{'dx': '갑상샘항진증', 'exams': ['경골앞 점액부종 확인', '갑상선 잡음 청진 (Thyroid bruit)']}]}
        fake = FakeClient()
        client = FakeClient(doctor=lambda u: {**PREDICT, 'cause': '갑상샘항진증'} if '예측(predict)' in u else fake.default_doctor(u))
        run.consult(client, [OTHER, banked])
        exam = client.diagnosed[0]['O']['exam']
        n = int(re.search(r'\[T(\d+)\] 경골앞 점액부종 확인', exam).group(1))
        self.assertIn(f'[T{n + 1}] 갑상선 잡음 청진', exam)  # the bank's order, before luna's picks and the core rows
        self.assertLess(n, int(re.search(r'\[T(\d+)\] 심음 청진', exam).group(1)))

    def test_guardian_open_question(self):
        self.assertEqual(brain.open_question('엄마가 보호자로 함께 왔다.'), '어떤 점이 걱정되시는지 자세히 말씀해 주세요.')
        self.assertEqual(brain.open_question('살이 빠져서 왔어요.'), brain.OPEN_QUESTION)


if __name__ == '__main__':
    unittest.main(verbosity=1)
