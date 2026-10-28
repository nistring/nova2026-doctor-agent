"""NOVA 예선 Doctor Agent (MAI-DxO 변형: 묶음 계획 + 재계획).

흐름: 인사·열린 질문 → 첫 계획(임상표현 선택·감별진단·첫 행동 묶음) → 묶음 실행(LLM 호출 없음) → 재계획 … (BROAD_END까지)
→ 가설 목록(진단별 질문을 코드가 차례로, 2회) → 기본 문진 → 진찰 → 마무리 대화(agenda + 턴마다 짧은 talk 호출)
→ 초진 기록(높은 추론) → DIAGNOSE. 모든 LLM 호출은 brain.ask() 하나를 거칩니다.
"""
import json
import re
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import brain
from nova_sdk import NovaError, run_agent

AGENT_VERSION = 'v44'    # stderr에 기록됨. git 태그와 맞춰 올림
TARGET_TURNS = 42       # 목표 총 턴. 공식 효율: 38~40턴 17.87(v16), 44턴 16.07(v19), 46턴 15.07(v38), 40턴 넘으면 턴당 ~0.4
MAX_TURNS = 50          # 서버 상한. 끝인사는 늘 이 안에서 마지막
CLOSE_TURNS = 6         # 마무리 대화 턴: 설명 4문장 + 질문 확인 + 끝인사(상담형은 +2). 공식 의사소통은 6.9~7.3에서 평평(v16~v38)
MAX_EXAMS = 14          # 진찰 상한(상담형은 2)
MIN_EXAMS, MIN_EXAMS_COUNSEL = 11, 1  # 조사를 끝내기 전 최소 진찰 수(남은 턴이 이만큼이면 진찰만 함). 진찰표 20~24행 중 채점 ~14행
BROAD_END = 5           # 이 턴까지 첫 계획으로 넓게 묻고, 그다음 카드 체크리스트 선택(hypothesis)으로 묻습니다
HYPO_FIRST, HYPO_MORE = 8, 8  # 첫 선택의 질문 수, 기본 문진 뒤 두 번째 선택의 상한(남은 턴에 맞춤)
MAX_EXTRA = 5           # 카드 밖 질문 상한: 카드 문항은 턴당 0.73항목, 카드 밖 질문은 0.34(v23 bench 9건). 진단 특이 심화용
MAX_EMPATHY = 3         # 감정 표현에 붙이는 공감 문구 상한(형식적 반복 방지)
MAX_DRILLS = 2          # 깊이 물어볼 관련 임상표현(다른 카드의 문진 목록) 수
ROUTINE_AT = 12         # 이 턴이 지나면 코드가 기본 문진(과거력·약·가족력·사회력)을 묻습니다. luna는 거의 묻지 않음(v11·v14 서버)
TIME_MARGIN = 300       # 마감 이 초 전부터는 기록 작성으로 넘어감
ONE_AT_A_TIME = '한 번에 하나씩 물어봐 주세요.'
STOP_CODES = {'TURN_LIMIT_REACHED', 'SESSION_EXPIRED', 'SESSION_CLOSED', 'RUN_NOT_ACTIVE'}
TURN_REFUSALS = {'EXAM_NOT_SPECIFIC', 'EXAM_NOT_RECOGNIZED', 'EMPTY_CONTENT', 'CONTENT_TOO_LONG'}
MODEL_SIDE = {'PATIENT_RESPONSE_INVALID', 'EXAMINER_RESPONSE_INVALID'}


def log(message):
    sys.stderr.write(f'[agent] {message}\n')


class Stop(Exception):
    """The session cannot take more actions."""


class Encounter:
    def __init__(self, session, cards):
        self.initial = session['initial']
        self.deadline = session.get('deadline')
        self.turn = 50 - session.get('turns_remaining', 50)
        self.candidates = brain.rank_cards(cards, self.initial['opening'])
        self.cards = {c['id']: c for c in cards}
        self.card, self.mode, self.addressee = None, '진단', ''
        self.ddx, self.lines, self.exams, self.done_texts, self.covered = [], [], [], set(), set()
        self.drills = []  # related presentations whose checklists are explored too
        self.exam_count = 0
        self.routine_done = False
        self.hypo_rounds = 0
        self.cause = ''  # card cause for the exam bank, from the predict call
        self.feeling, self.empathy = '', 0  # empathy phrase owed for the last emotional reply, and how many were said

    # budgets
    def close_turns(self):
        return CLOSE_TURNS + (2 if self.mode == '상담' else 0)

    def investigate_left(self):
        return TARGET_TURNS - self.close_turns() - self.turn

    def exams_left(self):
        return (2 if self.mode == '상담' else MAX_EXAMS) - self.exam_count

    def seconds_left(self):
        if not self.deadline:
            return float('inf')
        return datetime.fromisoformat(self.deadline).timestamp() - time.time()

    def status_text(self):
        ddx = ', '.join(f"{d.get('dx')}({d.get('p')})" for d in self.ddx if isinstance(d, dict)) or '(없음)'
        exam_only = '\n※ 남은 조사 턴은 모두 진찰(EXAM)에 씁니다.' if 0 < self.investigate_left() <= self.exams_needed() else ''
        return (f'사용한 턴: {self.turn} / 목표 {TARGET_TURNS}\n'
                f'남은 조사 턴(SAY·EXAM): {max(0, self.investigate_left())}, 남은 진찰 가능 수: {max(0, self.exams_left())}, '
                f'아직 필요한 최소 진찰 수: {self.exams_needed()}, 마무리 턴: {self.close_turns()}{exam_only}\n'
                f"진료 유형: {self.mode}형, 대화 상대 호칭: {self.addressee or '미정'}\n감별진단: {ddx}")

    # transcript
    def record_turn(self, kind, text, reply, turn):
        self.turn = turn or self.turn + 1
        self.done_texts.add(text)
        if kind == 'SAY':
            self.lines.append(f'T{self.turn:02d} 의사: {text} / 환자: {reply}')
            self.feeling = brain.feeling(reply)
        else:
            self.exam_count += 1
            self.exams.append((self.turn, text, reply))
            self.lines.append(f'T{self.turn:02d} 진찰({text}): {reply}')
        log(self.lines[-1])  # stderr is shown per case in 샘플 검증: the transcript can be rebuilt from it

    def exam_text(self):
        return brain.exam_text(self.initial['vital_signs'], self.exams, self.card)

    def empathize(self, text):
        """The next history question, with an empathy phrase in front when the last reply was emotional and it fits."""
        phrase, self.feeling = self.feeling, ''
        if not phrase or self.empathy >= MAX_EMPATHY or '겠어요' in text or len(phrase) + 1 + len(text) > brain.MAX_CHARS:
            return text
        self.empathy += 1
        return f'{phrase} {text}'

    def exams_needed(self):
        return max(0, (MIN_EXAMS_COUNSEL if self.mode == '상담' else MIN_EXAMS) - self.exam_count)

    def top_dx(self):
        first = next((d for d in self.ddx if isinstance(d, dict) and d.get('dx')), None)
        return str(first['dx']) if first else '진단 미정'

    def update(self, plan):
        if not isinstance(plan, dict):
            return
        if isinstance(plan.get('ddx'), list) and plan['ddx']:
            self.ddx = plan['ddx'][:7]
        if isinstance(plan.get('covered'), list):
            self.covered |= {str(item) for item in plan['covered']}
        for cp_id in plan.get('drill') or [] if isinstance(plan.get('drill'), list) else []:
            cp_id = str(cp_id).split('.')[0].strip()
            if (cp_id in self.cards and cp_id not in self.drills and len(self.drills) < MAX_DRILLS
                    and (self.card is None or cp_id != self.card['id'])):
                self.drills.append(cp_id)
                log(f"drill into {cp_id} {self.cards[cp_id]['name']}")
        if plan.get('cp') is not None and self.card is None:
            cp_id = str(plan['cp']).split('.')[0].strip()  # "28", 28 or "28. 수면장애"
            self.card = self.cards.get(cp_id)
            if self.card is None:
                return log(f'unknown presentation id {plan["cp"]!r}')
            self.mode = '상담' if plan.get('mode') == '상담' or self.card.get('type') == '상담' else '진단'
            self.addressee = brain.nfc(plan.get('addressee'))[:10]


def act(client, enc, kind, text):
    """One SAY/EXAM with retries. Returns the reply text, '' if refused (turn used), None if nothing happened."""
    request_id = str(uuid.uuid4())
    for wait in (2, 5, 10, None):
        try:
            reply = client.say(text, request_id=request_id) if kind == 'SAY' else client.exam(text, request_id=request_id)
        except ValueError:
            return None  # rejected by the SDK before sending: no turn used
        except NovaError as exc:
            if wait and exc.code in brain.SAME_ID_CODES:
                time.sleep(wait)
                continue
            if wait and exc.code in MODEL_SIDE:
                request_id = str(uuid.uuid4())
                time.sleep(wait)
                continue
            if exc.code in STOP_CODES:
                raise Stop(exc.code) from exc
            if exc.code in TURN_REFUSALS:
                enc.done_texts.add(text)
                try:
                    enc.turn = 50 - client.session()['turns_remaining']
                except (NovaError, KeyError):
                    enc.turn += 1
                enc.lines.append(f'T{enc.turn:02d} ({kind} "{text}" 거절됨: {exc.code})')
                log(enc.lines[-1])
                return ''
            log(f'{kind} failed: {exc.code}')
            return None
        answer = reply.get('content') if kind == 'SAY' else reply.get('result')
        enc.record_turn(kind, text, brain.nfc(answer), reply.get('turn'))
        return brain.nfc(answer)
    return None


def run_batch(client, enc, actions, stop_on_question=True, keep_going=False):
    """Execute planned actions in order; stop early when the plan may no longer fit (code lists: `keep_going`)."""
    for kind, text in actions:
        if enc.investigate_left() <= 0 or enc.seconds_left() < TIME_MARGIN:
            return
        if kind == 'EXAM' and enc.exams_left() <= 0:
            continue
        if kind == 'SAY' and enc.investigate_left() <= enc.exams_needed():
            continue  # the remaining turns are reserved for the minimum physical exam
        if kind == 'SAY':
            enc.done_texts.add(text)  # the question itself, so it is not asked again without the empathy phrase
            text = enc.empathize(text)
        answer = act(client, enc, kind, text)
        if not keep_going and (answer == '' or answer is None or answer == ONE_AT_A_TIME):
            return
        if stop_on_question and kind == 'SAY' and answer.endswith('?'):
            return


def routine_actions(enc):
    """The routine history items not yet asked in any form (guardian set for a child)."""
    child = any(word in enc.addressee for word in ('어머님', '아버님', '보호자')) or '보호자' in enc.initial['opening']
    asked = ' '.join(enc.done_texts)
    opening = enc.initial['opening']
    items = brain.ROUTINE_CHILD if child else brain.ROUTINE_ADULT + (brain.ROUTINE_WOMAN if brain.childbearing(opening) else [])
    if not re.search(r'원해|원한다|달라|받고 싶|받으러|싶어서', opening):
        items = items[:1] + [brain.EXPECTATION] + items[1:]  # right after the concern question
    return brain.validate(enc, [{'type': 'SAY', 'text': q} for pattern, q in items if not re.search(pattern, asked)],
                          limit=len(items))


def routine(client, enc, now=False):
    """Once per encounter, after ROUTINE_AT (or `now`): the routine history, asked by code because the model skips it."""
    if enc.routine_done or not enc.card or not (now or enc.turn >= ROUTINE_AT):
        return
    enc.routine_done = True
    actions = routine_actions(enc)
    log(f'routine history: {len(actions)} items')
    run_batch(client, enc, actions, stop_on_question=False)


def fallback_actions(enc, history_done=False):
    """Used when planning gives nothing: owed exams once history is done (or only exam turns remain), else the next
    unasked must-ask questions."""
    if enc.exams_needed() and (history_done or enc.investigate_left() <= enc.exams_needed()):
        # the presentation's own exams first, then broadly useful ones, then the other common rows (never run out)
        ids = [e['id'] for e in (enc.card or {}).get('exams', [])] + brain.FALLBACK_EXAMS + list(brain.COMMON_EXAMS)
        return brain.validate(enc, [{'type': 'EXAM', 'id': key} for key in ids])[:enc.exams_needed()]
    questions = ([h['q'] for h in enc.card.get('hx', []) if h.get('tag') == '필수' and h['id'] not in enc.covered]
                 if enc.card else brain.GENERIC_HX)
    return brain.validate(enc, [{'type': 'SAY', 'text': q} for q in questions if q not in enc.done_texts][:3])


def fit_lines(client, enc, lines):
    """Closing lines within 30 characters: ask once to rewrite long ones, split what is still too long."""
    lines = [brain.nfc(line) for line in lines or [] if brain.nfc(line)]
    if any(len(line) > brain.MAX_CHARS for line in lines):
        rewritten = brain.ask(client, enc, 'shorten', '[고칠 문장]\n' + '\n'.join(lines), effort='low',
                              max_tokens=4000, tries=1) or {}
        if isinstance(rewritten.get('lines'), list) and rewritten['lines']:
            lines = [brain.nfc(line) for line in rewritten['lines'] if brain.nfc(line)]
    return [part for line in lines for part in brain.split_say(line)]


def exam_actions(items):
    """Hypothesis exams as validate() input: {"id": …}, {"text": …} or a bare maneuver name."""
    return [{'type': 'EXAM', **item} if isinstance(item, dict) else {'type': 'EXAM', 'text': item}
            for item in items if isinstance(item, (dict, str))] if isinstance(items, list) else []


# Comm graders on every v36 server case: the explanation was too short (what the disease is, why, treatment, prognosis,
# a direct answer to each worry). Turns left after the agenda go to that.
MORE_EXPLANATION = ('전할 내용은 다 전했습니다. 병이 생긴 원인, 치료 방법(약과 생활), 앞으로의 경과, 환자가 말한 걱정 각각에 대한 '
                    '구체적인 답 중 아직 말하지 않은 것 하나를 쉬운 말로 말합니다.')
SPECIFIC_EXAMS = 5  # luna's diagnosis-specific picks kept; the rest comes from the core common rows


def exam_order(enc, actions):
    """Up to SPECIFIC_EXAMS diagnosis-specific maneuvers from the list (v22: luna lists common rows head to toe), then
    the core common rows in grading order (v34 server: luna's common picks were conjunctiva, cervical nodes, chest
    inspection, while heart, skin, liver/spleen went undone)."""
    common = set(brain.COMMON_EXAMS.values())
    bank = brain.validate(enc, [{'type': 'EXAM', 'text': e} for e in brain.bank_exams(enc)], limit=SPECIFIC_EXAMS)
    picked = [a for a in actions if a[1] not in common and a not in bank]
    specific = (bank + picked)[:SPECIFIC_EXAMS]  # Opus exam bank for the leading diagnosis first, then luna's picks
    core = brain.validate(enc, [{'type': 'EXAM', 'id': key} for key in brain.CORE_EXAMS], limit=len(brain.CORE_EXAMS))
    return specific + [a for a in core if a not in specific]


def screen(client, enc):
    """Ask the use-disorder screening questions once heavy drug or alcohol use shows (code-side, against anchoring)."""
    questions = brain.substance_screen(enc)
    if questions:
        log('substance screening: ' + ' | '.join(questions))
        run_batch(client, enc, [('SAY', q) for q in questions], stop_on_question=False, keep_going=True)


def hypothesize(client, enc):
    """Deep history after the broad phase: two card selections asked in order by code (the second sized to leave the
    routine history and the minimum exams their turns), then the routine history and the exams, specific ones first.
    False if no hypothesis parsed."""
    hypothesis = None
    for first in (True, False):
        enc.hypo_rounds += 1
        # the second list gets what is left after the routine history and the minimum exams are set aside
        count = HYPO_FIRST if first else min(HYPO_MORE, enc.investigate_left() - enc.exams_needed() - len(routine_actions(enc)))
        if count <= 0 or enc.seconds_left() < TIME_MARGIN:
            break
        # first: card items (breadth); second: the facts predicted for the leading diagnosis' script (depth). Offline on
        # 9 cases a 25-question prediction covered 50% of the graded facts, the encounter 53%, together 77%.
        answer = brain.ask(client, enc, 'hypothesis' if first else 'predict',
                           f'[개수] questions {count}개, exams {max(0, enc.exams_left())}개', effort='high', max_tokens=12000)
        if not answer:
            break
        hypothesis = answer
        enc.cause = brain.nfc(answer.get('cause')) or enc.cause
        enc.update(answer)
        if isinstance(answer.get('related'), list):
            enc.update({'drill': answer['related']})  # related presentations' checklists come in the second selection
        picked = {}  # question text -> card ids it covers: a single id, or a bundle {"ids": [...], "q": "..."}
        for key in answer.get('ask') or []:
            if isinstance(key, dict) and brain.nfc(key.get('q')) and isinstance(key.get('ids'), list):
                picked[brain.nfc(key['q'])] = ','.join(str(k) for k in key['ids'])
            elif brain.card_question(enc, key):
                picked[brain.card_question(enc, key)] = str(key)
        extra = [q for q in answer.get('extra') or [] if isinstance(q, str)][:MAX_EXTRA]
        if not first:
            extra = [q for q in answer.get('questions') or [] if isinstance(q, str)]
        actions = brain.validate(enc, [{'type': 'SAY', 'text': q} for q in list(picked) + extra], limit=count)
        log(f'hypothesis {enc.hypo_rounds} ({enc.top_dx()}): ' + ' | '.join(f'{picked.get(t, "+")} {t}' for _, t in actions))
        run_batch(client, enc, actions, stop_on_question=False, keep_going=True)
        enc.covered |= {key for text, keys in picked.items() if text in enc.done_texts for key in keys.split(',')}
        screen(client, enc)
    routine(client, enc, now=True)  # reliable checklist items, after both lists (their turns were set aside)
    screen(client, enc)  # the medication question may be the first to show it
    if hypothesis is None:
        return False
    exams = exam_order(enc, brain.validate(enc, exam_actions(hypothesis.get('exams')), limit=MAX_EXAMS))
    log('hypothesis exams: ' + ' | '.join(text for _, text in exams))
    run_batch(client, enc, exams, stop_on_question=False, keep_going=True)
    return True


def investigate(client, enc):
    stalls = 0
    while enc.investigate_left() > 0 and enc.seconds_left() > TIME_MARGIN and stalls < 2:
        if enc.card and not enc.hypo_rounds and enc.turn >= BROAD_END:
            hypothesize(client, enc)  # the plan loop below then uses any turns left (all of them if it failed)
            continue
        routine(client, enc)
        task = 'first' if enc.card is None else 'plan'  # keep asking until the model names a presentation
        plan = brain.ask(client, enc, task)
        enc.update(plan)
        if task == 'first':
            log(f"presentation: {enc.card['id'] + ' ' + enc.card['name'] if enc.card else 'not chosen'}")
        actions = brain.validate(enc, (plan or {}).get('actions'))
        if not actions and plan and plan.get('done') and not enc.exams_needed():
            if enc.routine_done:
                return
            routine(client, enc, now=True)  # the model finished early: routine history before the closing
            continue
        turn = enc.turn
        batch = actions or fallback_actions(enc, history_done=bool(plan and plan.get('done')))
        if enc.card and not enc.hypo_rounds:
            batch = batch[:max(1, BROAD_END - enc.turn)]  # the broad phase ends at BROAD_END
        run_batch(client, enc, batch)
        stalls = stalls + 1 if enc.turn == turn else 0


def is_question(reply):
    return reply.rstrip().endswith('?') or any(word in reply for word in ('까요', '나요', '궁금해요', '알고 싶'))


def say(client, enc, lines):
    """Say closing lines in order; False when one could not be sent."""
    for kind, text in brain.validate(enc, [{'type': 'SAY', 'text': line} for line in lines], limit=max(1, len(lines))):
        if act(client, enc, kind, text) is None:
            return False
    return True


def talk(client, enc, items, done, start, turn_now=None):
    """The next closing line(s) from one short talk call and the agenda ids it finished; the next point if it failed."""
    answer = brain.ask(client, enc, 'talk', brain.talk_text(enc, items, done, start, turn_now), effort='low',
                       max_tokens=3000, tries=1) or {}
    if len(brain.nfc(answer.get('say'))) > brain.MAX_CHARS:  # split by words it reads broken ("…갑상샘" / "때문입니다.")
        note = f"앞의 답({brain.nfc(answer.get('say'))})이 {brain.MAX_CHARS}자를 넘었습니다. 같은 뜻을 {brain.MAX_CHARS}자 이하 한 문장으로."
        answer = brain.ask(client, enc, 'talk', brain.talk_text(enc, items, done, start, f'{turn_now or ""} {note}'.strip()),
                           effort='low', max_tokens=3000, tries=1) or answer
    covered = answer.get('covered') if isinstance(answer.get('covered'), list) else []
    covered = {str(key) for key in covered} & {key for key, _ in items}
    lines = brain.fit_one(answer.get('say'))
    if not lines and answer.get('say'):  # still over 30 after the retry: rewrite or split, never stay silent (v22 S03)
        lines = fit_lines(client, enc, [answer.get('say')])
    if not lines:
        key, point = next(((key, point) for key, point in items if key not in done), items[-1])
        lines, covered = fit_lines(client, enc, [point]), {key}
    return lines, covered


def close(client, enc):
    """Dialog closing: an agenda from one close call, then one short talk call per turn, answering the patient's
    questions and emotions on the way. The red flag is guaranteed, the last question answered, the goodbye always last."""
    if enc.seconds_left() < TIME_MARGIN:
        return
    result = brain.ask(client, enc, 'close') or {}
    enc.update(result)
    items = brain.agenda(result.get('agenda'), enc)
    log('agenda: ' + ' | '.join(f'{key}: {point}' for key, point in items))
    start, done = len(enc.lines), set()
    # explanation up to TARGET-2 (v38 server: 4 lines, T41-44, then T46 goodbye with the 2 answer turns unused); the check,
    # the answers and the goodbye may run to MAX_TURNS
    last = min(MAX_TURNS - 4, max(TARGET_TURNS - 2, enc.turn + 4))
    while enc.turn < last and enc.seconds_left() > TIME_MARGIN / 2:  # the whole closing budget (v36: 2 of 8 turns unused)
        extra = len(done) >= len(items)
        forced = 'redflag' not in done and enc.turn == last - 1
        hint = ('바로 와야 할 증상(redflag)을 구체적인 증상 이름으로 전합니다.' if forced else MORE_EXPLANATION if extra else None)
        lines, covered = talk(client, enc, items, done, start, hint)
        if forced:
            lines = lines if any(brain.RED_FLAG.search(line) for line in lines) else [brain.red_flag_line(enc.card)]
            covered |= {'redflag'}
        before = enc.turn
        if not say(client, enc, lines[:last - enc.turn]):
            break
        done |= covered
        if enc.turn == before:  # nothing new was said (a repeated line is dropped by validate): move on
            if extra:
                break
            done.add(next(key for key, _ in items if key not in done))
    if 'redflag' not in done and enc.turn < MAX_TURNS - 2:
        say(client, enc, [brain.red_flag_line(enc.card)])
    if enc.seconds_left() > TIME_MARGIN / 2 and enc.turn < MAX_TURNS - 1:
        reply = act(client, enc, 'SAY', brain.ASK_QUESTIONS)
        while reply and is_question(reply) and enc.turn < MAX_TURNS - 1 and enc.seconds_left() > TIME_MARGIN / 2:
            lines, covered = talk(client, enc, items, done, start, '환자의 마지막 질문에 답합니다.')
            before = enc.turn
            if not say(client, enc, lines[:MAX_TURNS - 1 - enc.turn]) or enc.turn == before:
                break
            done |= covered
            reply = enc.lines[-1].split(' / 환자: ', 1)[-1]  # the patient may ask again: answered while turns last
    if enc.turn < MAX_TURNS:
        admitted = any(key == 'treat' and '입원' in point for key, point in items)
        say(client, enc, [brain.GOODBYE_ADMIT if admitted else brain.GOODBYE])


def finish(client, enc):
    result = None
    if enc.seconds_left() > 150:
        extra = '[진찰 소견] (받은 결과 그대로. O.exam에 부위별로 옮겨 씀)\n' + enc.exam_text()
        note = brain.dependence_note(enc)
        if note:
            log(note)
            extra += '\n\n' + note
        effort = 'high' if enc.seconds_left() > TIME_MARGIN else 'medium'
        result = brain.ask(client, enc, 'record', extra, effort=effort, max_tokens=16000, tries=1)
        if result is None and enc.seconds_left() > 120:
            result = brain.ask(client, enc, 'record', extra, effort='medium', max_tokens=12000, tries=1)
    note, diagnosis = brain.to_note(enc, result) if isinstance(result, dict) else brain.fallback_note(enc)
    if isinstance(result, dict) and enc.seconds_left() > 120:
        # v38 server: 10% of the graded facts were asked but left out of the record (none of them cost a turn)
        audit = brain.ask(client, enc, 'audit', brain.audit_text(enc, note), effort='medium', max_tokens=6000, tries=1)
        log(f'record audit: {brain.add_lines(note, audit)} lines added')
    log(f'DIAGNOSIS {diagnosis}')
    log('NOTE ' + json.dumps(note, ensure_ascii=False))
    request_id = str(uuid.uuid4())
    for wait in (2, 5, 10, None):
        try:
            return client.diagnose(note=note, diagnosis=diagnosis, request_id=request_id)
        except NovaError as exc:
            if wait and exc.code in brain.SAME_ID_CODES:
                time.sleep(wait)
                continue
            log(f'diagnose failed: {exc.code}')
            raise


def prepare():
    path = Path(__file__).resolve().parent / 'knowledge' / 'cp.json'
    return json.loads(path.read_text(encoding='utf-8'))


def consult(client, cards):
    enc = Encounter(client.session(), cards)
    log(f'version {AGENT_VERSION}')
    try:
        if enc.seconds_left() > TIME_MARGIN:
            act(client, enc, 'SAY', brain.greeting(enc.initial['opening']))
            act(client, enc, 'SAY', brain.open_question(enc.initial['opening']))  # comm graders asked for one
        investigate(client, enc)
        close(client, enc)
    except Stop as stop:
        log(f'stopped: {stop}')
    except Exception as exc:  # never lose the case: record what we have
        log(f'unexpected {type(exc).__name__}: {exc}')
    return finish(client, enc)


if __name__ == '__main__':
    run_agent(prepare, consult)
