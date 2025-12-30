from typing import Dict, List, Any, Tuple, Optional, Iterable
from db import inc_error_feature, get_top_error_features
from collections import defaultdict
from data import COLOURS, FRUITS
from dataclasses import dataclass, asdict
from db import (
    init_db,
    seed_words,
    get_or_create_user,
    upsert_state,
    get_state_or_default,
    ensure_topic_path,
    get_active_topic,
    set_topic_completed_and_unlock_next,
    get_word_ids_for_topic,
    get_word_by_id,
    prune_user_error_stats
)

ALPHA = 0.25  # how fast mastery updates
MIN_GAP = 1  # soonest review (in questions)
MAX_GAP = 25  # latest review (in questions)
FAIL_FAST_THRESH = 0.6  # if score below this, schedule very soon
HISTORY_SIZE = 3
NEW_PER_LESSON = 5
RETRY_THRESH = 0.1
LESSON_LEN = 20

TARGET_MASTERY = 0.9
MIN_ATTEMPTS_PER_WORD = 2
MAX_COUNT_CAP = 5

ACCENT_MAP = {
    "é": "e", "è": "e", "ê": "e", "ë": "e",
    "à": "a", "â": "a",
    "ç": "c",
    "î": "i", "ï": "i",
    "ô": "o",
    "ù": "u", "û": "u", "ü": "u",
}

@dataclass
class EngineState:
    t: int
    question_in_lesson: int
    lesson_word_ids: List[int]
    recent_history: List[int]
    lesson_error_counts: Dict[str, int]
    progress: Dict[int, Dict[str, Any]]
    new_queue: List[int]
    topic_slug: str
    current_word_id: Optional[int] = None

    # pending retry state
    pending_retry_word_id: Optional[int] = None
    pending_retry_used: bool = False

#Character is input, if it exists in ACCENT_MAP, the base character is returned
#Normalises characters so that those that onlt differ by accents are treated as the same underlying character
def deaccent(ch: str) -> str:
    return ACCENT_MAP.get(ch, ch)

def topic_mastered(conn, user_id: int, language_code: str, topic_slug: str) -> bool:
    word_ids = get_word_ids_for_topic(conn, language_code, topic_slug)
    if not word_ids:
        return False

    for wid in word_ids:
        mastery, attempts, _ = get_state_or_default(conn, user_id, wid)
        if mastery < TARGET_MASTERY:
            return False

    return True

def load_progress(conn, user_id: int, word_ids: List[int]) -> dict:
    """
    Returns: progress[word_id] = {"mastery": float, "attempts": int, "next_due_t": int}
    """
    progress = {}
    for wid in word_ids:
        mastery_val, attempts_val, next_due_val = get_state_or_default(conn, user_id, wid)
        progress[wid] = {"mastery": mastery_val, "attempts": attempts_val, "next_due_t": next_due_val}
    return progress


def levenshtein(inp: str, target: str) -> int:

    '''
    Compute the Levenshtein edit distance between two strings.

    The distance is the minimum number of single-character edits
    (insertions, deletions, or substitutions) required to transform
    `inp` into `target`.

    This is used to measure graded similarity between a user's answer
    and the correct word, rather than relying on a binary match.'''


    lenInp, lenTarget = len(inp), len(target)

    '''Handles base case if either strings are empty, Defensive programming that prevents unexpected results
       when carrying out unit tests, or if validation is not present in a database'''
    if lenInp == 0: return lenTarget  # empty input
    if lenTarget == 0: return lenInp  # empty dataset value

    dp = list(range(lenTarget + 1))  # stores a single row of a the table that keeps track of edits needed to be made
    for i in range(1, lenInp + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, lenTarget + 1):
            cur = dp[j]  # Holds value of the most recent last element in the queue.
            cost = 0 if inp[i - 1] == target[j - 1] else 1
            # lowest cost between insertion, deletion and substituition
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[lenTarget]  # Returns value in bottom left corner


def levenshtein_ops(inp: str, target: str):
    """
    Return the sequence of edit operations required to transform `inp` into `target`.

    Each operation is one of:
      - ("del", char)            : deletion
      - ("ins", char)            : insertion
      - ("sub", from_char, to_char) : substitution

    This is used to extract *error patterns* from an incorrect answer,
    rather than just measuring how far off it was.
    """
    #Ignore case and whitespace
    inp = inp.lower().strip()
    target = target.lower().strip()
    lenInp, lenTarget = len(inp), len(target)

    # dp[i][j] = distance between inp[:i] and target[:j]
    dp = [[0]*(lenTarget+1) for _ in range(lenInp+1)]
    for i in range(lenInp+1): dp[i][0] = i
    for j in range(lenTarget+1): dp[0][j] = j

    for i in range(1, lenInp+1):
        for j in range(1, lenTarget+1):
            cost = 0 if inp[i-1] == target[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,      # delete
                dp[i][j-1] + 1,      # insert
                dp[i-1][j-1] + cost  # substitute/match
            )

    # backtrace to ops
    i, j = lenInp, lenTarget
    ops = []  # list of tuples like ("sub", from_char, to_char) or ("ins", char) or ("del", char)
    while i > 0 or j > 0:
        if i > 0 and dp[i][j] == dp[i-1][j] + 1:
            ops.append(("del", inp[i-1]))
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j-1] + 1:
            ops.append(("ins", target[j-1]))
            j -= 1
        else:
            # diagonal
            if i > 0 and j > 0:
                if inp[i-1] != target[j-1]:
                    ops.append(("sub", inp[i-1], target[j-1]))
                i -= 1
                j -= 1

    ops.reverse()
    return ops

def similarity(inp: str, target: str) -> float:
    """
    Compute a normalized similarity score between user input and the target word.

    The score is based on Levenshtein edit distance and is scaled to [0, 1]:
      - 1.0  -> exact match
      - 0.0  -> completely different
      - (0,1)-> partial match

    This provides a graded correctness signal rather than a binary correct/incorrect.
    """
    # Normalising to lower case + removing whitespace
    normalised_inp, normalised_target = inp.lower().strip(), target.lower().strip()

    # Calculates distance between input and target word
    dist = levenshtein(normalised_inp, normalised_target)

    return max(0.0, 1.0 - dist / max(len(normalised_inp), len(normalised_target)))


def getvalidInput(userPrompt: str) -> str:
    '''Used for validation, ensures user does not enter empty space
        Returns cleaned string'''
    validInput = False
    while validInput == False:
        user_input = input(userPrompt)
        # Remove leading/trailing whitespace
        cleanedUserInput = user_input.strip()

        # Validation: block empty or whitespace-only input
        if (cleanedUserInput == ""):
            print("Please enter a valid answer before continuing.")
        else:
            validInput = True  # stops loop as user has entered a valid value
    return cleanedUserInput


def pattern_bonus(candidate_text: str, top_features: Iterable[Tuple[str, int]]) -> float:
    '''This is a personalisation heuristic, it nudges content selection towards words that contain
        patterns the user has previously struggled with.
        Example: If a candidate often mixes up the order of the letters 'g' and 'e' in words
        words like rouge and orange will be weighted higher to practise this'''
    text_to_score = candidate_text.lower() #Candidate text being weighted
    bonus = 0.0
    for feat, count in top_features:
        strength = min(MAX_COUNT_CAP, count)  # cap effect

        if feat.startswith("swap:"):
            bigram = feat.split(":", 1)[1] #stores letters to swap
            if bigram in text_to_score:
                bonus += 0.15 * strength #increase priority

        elif feat == "accent:issue":
            if any(ch in text_to_score for ch in "éèêëàâçîïôùûü"):
                bonus += 0.10 * strength

        elif feat.startswith("sub:"):
            rhs = feat.split("->")[-1]  # the "to" character
            if rhs and rhs in text_to_score:
                bonus += 0.05 * strength
    return bonus

def top_from_dict(freq_dict, limit: int = 10) -> List[Tuple[Any, int]]:
    """
    Return the top `limit` (key, count) pairs from a frequency dictionary.

    Items are ordered by descending count. This is used to surface the most
    frequent error patterns (per-lesson or lifetime) for feedback and weighting.
    """
    return sorted(freq_dict.items(), key=lambda kv: kv[1], reverse=True)[:limit]


def pick_next_word_id(conn, current_time: int, lesson_word_ids: List[int], recent_history: List[int], progress:dict, top_feats: List[Tuple[str, int]]) -> int:
    """
    Smarter review picker:
    1) Build due list from introduced words (or all words if nothing introduced yet).
    2) If due list is non-empty, prefer it.
    3) BUT if due list collapses to 1 word and it's in recent_history, widen to introduced pool
        so we avoid spamming the same word.
    4) Choose the lowest-mastery word from the final candidate set.
    """
    # 1) Prefer reviewing words we've already introduced
    introduced_ids = [wid for wid in lesson_word_ids if progress[wid]["attempts"] > 0]
    if introduced_ids:
        lesson_pool_for_review = introduced_ids
    else:
        lesson_pool_for_review = lesson_word_ids  # early on, nothing introduced yet

    # 2) Due words within that pool
    due_word_ids = []
    # Moving through the list and then adding words to due state, if they need to be practiced
    # 1) Collect due words (within the lesson pool only), a word being due depends on it's mastery level
    for word_id in lesson_pool_for_review:
        if progress[word_id]["next_due_t"] <= current_time:
            due_word_ids.append(word_id)

    # 3) Start candidates as due if possible, otherwise use review_pool
    #Lists are mutable so [:] creates a copy of the list, changing candidate_ids will not change the original lesson_word_ids
    candidate_ids = due_word_ids if len(due_word_ids) > 0 else lesson_pool_for_review[:]
    print("Len_candidates: ",len(candidate_ids), "Len recent history: ",len(recent_history))

    # 4) Avoid recent repeats when we have options
    filtered_candidates = []
    for word_id in candidate_ids:
        if word_id not in recent_history:
            filtered_candidates.append(word_id)

    if len(filtered_candidates) > 0:
        candidate_ids = filtered_candidates

    # 5) If we're still stuck with one option (usually fail-fast spamming),
    #    widen temporarily to avoid situations like "Grey, Grey, Grey..."

    if len(candidate_ids) == 1:
        widened_pool = lesson_pool_for_review[:]  # introduced words (or all if none introduced)
        widened_filtered = [wid for wid in widened_pool if wid not in recent_history]
        if widened_filtered:
            candidate_ids = widened_filtered
        else:
            candidate_ids = widened_pool
    best_id = candidate_ids[0]
    best_score = -1e9 # ensure first candidate always wins comparison

    for wid in candidate_ids:
        w = get_word_by_id(conn, wid)
        score = (1.0 - progress[wid]["mastery"]) + pattern_bonus(w["answer"], top_feats)
        if score > best_score: #scores checked here
            best_score = score
            best_id = wid

    return best_id


def update_mastery(old_mastery: float, score: float, alpha: float = ALPHA) -> float:
    """Exponential moving average update toward the latest evidence."""
    return old_mastery + alpha * (score - old_mastery)


def gap_from_mastery(mastery: float, score: float) -> int:
    """
    Compute the spacing gap (in number of questions) before the next review.

    Policy:
    - If the latest attempt score is very poor, schedule an immediate retry
      (fail-fast behaviour).
    - Otherwise, increase the spacing nonlinearly with mastery:
      low mastery → short gaps, high mastery → long gaps.
    """
    if score < FAIL_FAST_THRESH:
        return MIN_GAP

    # Smoothly scale gap between MIN_GAP and MAX_GAP based on mastery
    # (m^2 grows slowly at first, faster as mastery approaches 1)
    # So further apart when mastery is high and close repetition when mastery is low
    scaled = MIN_GAP + (MAX_GAP - MIN_GAP) * (mastery ** 2)

    # Convert to an integer number of questions (at least MIN_GAP)
    return max(MIN_GAP, int(round(scaled)))


def should_ask_new(question_index: int) -> bool:
    # 1-based question_index: 1,5,9,13,17 => 5 new words over 20 questions
    return (question_index - 1) % 4 == 0


def count_due_candidates(current_time: int, lesson_word_ids: List[int], progress: dict) -> int:
    '''This function returns the number of review candidates available at the current time
        Allows the user to consolidate words they would have already seen
        If early in the lesson, consider all lesson words'''
    introduced_ids = [wid for wid in lesson_word_ids if progress[wid]["attempts"] > 0]
    pool = introduced_ids if introduced_ids else lesson_word_ids
    due = [wid for wid in pool if progress[wid]["next_due_t"] <= current_time]
    return len(due) if due else len(pool)


def engine_next_question(conn, user_id: int, state: EngineState, language_code: str) -> Tuple[EngineState, Dict[str, Any]]:
    """
    Driver for question selection.

    Responsibilities:
    1) Advance the lesson clock (t, question_in_lesson).
    2) Apply scheduling policy to choose either:
       - a NEW word (curriculum progression / interleaving), or
       - a REVIEW word (spaced repetition / reinforcement).
    3) Return a small payload for UI/CLI display.
    """
    state.t += 1
    state.question_in_lesson += 1

    t = state.t
    q = state.question_in_lesson

    # 1) Pending retry gets priority (one immediate retry)
    if state.pending_retry_word_id is not None and not state.pending_retry_used:
        word_id = state.pending_retry_word_id
        state.pending_retry_used = True
        state.current_word_id = word_id
        w = get_word_by_id(conn, word_id)
        return state, {
            "t": t,
            "question_number": q,
            "topic": state.topic_slug,
            "question": {"word_id": word_id, "prompt": w["prompt"]},
            "pending_retry": True,
        }

    introduced_ids = [wid for wid in state.lesson_word_ids if state.progress[wid]["attempts"] > 0]
    introduced_count = len(introduced_ids)

    pool = introduced_ids if introduced_ids else state.lesson_word_ids
    due = [wid for wid in pool if state.progress[wid]["next_due_t"] <= t]
    review_candidate_count = len(due) if due else len(pool)

    # 3) Decide whether to introduce a new word or pick a review word
    if state.new_queue and (introduced_count < 3 or should_ask_new(q) or review_candidate_count <= 1):
        word_id = state.new_queue.pop(0)

    else:
        top_feats = top_from_dict(state.lesson_error_counts, limit=10)
        word_id = pick_next_word_id(
            conn, t, state.lesson_word_ids, state.recent_history, state.progress, top_feats
        )

    state.current_word_id = word_id
    w = get_word_by_id(conn, word_id)

    return state, {
        "t": t,
        "question_number": q,
        "topic": state.topic_slug,
        "question": {"word_id": word_id, "prompt": w["prompt"]},
        "pending_retry": False
    }

def engine_start(conn, user_id: int, language_code: str, topic_slug: str, lesson_word_ids: List[int]) -> EngineState:

    """
    Initialise an EngineState for a new lesson session.

    - Loads persisted per-word spaced-repetition state (mastery/attempts/next due).
    - Seeds a queue of new (unseen) lesson items to be introduced gradually.
    - Resets lesson-scoped tracking (recent history, error counts, retry flags).
    """

    progress = load_progress(conn, user_id, lesson_word_ids)

    # NEW items to introduce this lesson: words with zero attempts, capped per lesson.
    # The driver (engine_next_question) controls when these are interleaved.
    new_queue = [wid for wid in lesson_word_ids if progress[wid]["attempts"] == 0][:NEW_PER_LESSON]

    # Create a fresh engine state for this lesson run.
    state = EngineState(
        t=0,
        question_in_lesson=0,
        lesson_word_ids=lesson_word_ids,
        recent_history=[],
        lesson_error_counts=defaultdict(int),
        progress=progress,
        new_queue=new_queue,
        topic_slug=topic_slug,
        current_word_id=None,
        pending_retry_word_id=None,
        pending_retry_used=False
    )

    return state

def engine_submit_answer(conn, user_id: int, state: EngineState, user_answer: str, language_code: str = "fr",) -> Tuple[EngineState, Dict[str, Any]]:
    """
    Grades an answer for the current word, updates mastery + scheduling + error stats,
    and returns a result payload. Does NOT automatically pick the next question.
    Call engine_next_question() after this to get the next prompt.
    """
    if state.current_word_id is None:
        raise ValueError("No current_word_id set. Call engine_next_question() first.")

    word_id = state.current_word_id
    w = get_word_by_id(conn, word_id)
    correct = w["answer"]

    score = similarity(user_answer, correct)

    ua = user_answer.lower().strip()
    ta = correct.lower().strip()

    # 1) Only log patterns if not exact match
    if ua != ta:
        ops = levenshtein_ops(user_answer, correct)
        for op in ops:
            if op[0] == "sub":
                _, a, b = op
                feat = f"sub:{a}->{b}"
                inc_error_feature(conn, user_id, language_code, feat, 1)
                state.lesson_error_counts[feat] += 1

                # detect accent-only substitutions
                if deaccent(a) == deaccent(b) and a != b:
                    feat2 = "accent:issue"
                    inc_error_feature(conn, user_id, language_code, feat2, 1)
                    state.lesson_error_counts[feat2] += 1

            elif op[0] == "del":
                _, a = op
                feat = f"del:{a}"
                inc_error_feature(conn, user_id, language_code, feat, 1)
                state.lesson_error_counts[feat] += 1

            elif op[0] == "ins":
                _, b = op
                feat = f"ins:{b}"
                inc_error_feature(conn, user_id, language_code, feat, 1)
                state.lesson_error_counts[feat] += 1

    # 2) Pending retry scheduling: if very poor score, retry once next question
    if score < RETRY_THRESH and not state.pending_retry_used:
        state.pending_retry_word_id = word_id
    else:
        state.pending_retry_word_id = None
        state.pending_retry_used = False

    # 3) Update per-word spaced repetition state
    prev_mastery = state.progress[word_id]["mastery"]
    prev_attempts = state.progress[word_id]["attempts"]

    attempts = prev_attempts + 1
    mastery = update_mastery(prev_mastery, score)
    next_due_t = state.t + gap_from_mastery(mastery, score)

    upsert_state(conn, user_id, word_id, mastery, attempts, state.t, next_due_t)

    # sync progress cache
    state.progress[word_id]["mastery"] = mastery
    state.progress[word_id]["attempts"] = attempts
    state.progress[word_id]["next_due_t"] = next_due_t

    # 4) Update recent history with the word that was just asked
    state.recent_history.append(word_id)
    if len(state.recent_history) > HISTORY_SIZE:
        state.recent_history.pop(0)

    # 5) Lesson summary gating (only emit per-lesson patterns every LESSON_LEN questions)
    lesson_complete = (state.question_in_lesson % LESSON_LEN == 0)

    out: Dict[str, Any] = {
        "result": {
            "score": score,
            "correct_answer": correct,
            "mastery": mastery,
            "was_exact_match": (ua == ta),
        },
        "lesson_complete": lesson_complete,
    }
    #If lesson finished, show patterns to user
    if lesson_complete:
        #patterns learnt this lesson
        out["learned_patterns_this_lesson"] = top_from_dict(state.lesson_error_counts, limit=5)
        #lifetime patterns (throughout practice in total)
        out["learned_patterns_lifetime"] = get_top_error_features(conn, user_id, language_code, limit=5)

        remaining_new = [wid for wid in state.lesson_word_ids if state.progress[wid]["attempts"] == 0]
        state.new_queue = remaining_new[:NEW_PER_LESSON]

        # reset lesson-scoped stuff
        state.lesson_error_counts = defaultdict(int)
        state.question_in_lesson = 0
        state.recent_history = []

        # clear retry carry-over
        state.pending_retry_word_id = None
        state.pending_retry_used = False
    else:
        out["learned_patterns_this_lesson"] = []
        out["learned_patterns_lifetime"] = []

    row = conn.execute(
        "SELECT COUNT(*) FROM user_error_stats WHERE user_id=? AND language_code=?",
        (user_id, language_code)
    ).fetchone()
    if row and row[0] > 300:
        prune_user_error_stats(conn, user_id, language_code, keep_top=200)

    return state, out

def get_topic_mastery_snapshot(conn, user_id: int, language_code: str, topic_slug: str):
    """
    Return a per-word mastery snapshot for a given topic.

    This is a read-only reporting helper used for:
    - end-of-lesson summaries
    - debugging / inspection
    - progress visualisation in UI or CLI

    Each row contains the word metadata along with the user's current
    mastery level and attempt count.
    """
    word_ids = get_word_ids_for_topic(conn, language_code, topic_slug)
    rows = []
    for wid in word_ids:
        m, attempts, _ = get_state_or_default(conn, user_id, wid)
        w = get_word_by_id(conn, wid)
        rows.append({
            "word_id": wid,
            "prompt": w["prompt"],
            "answer": w["answer"],
            "mastery": m,
            "attempts": attempts,
        })
    return rows


def session_runner_cli(conn, user_id: int, language_code: str, topic_slug: str, lesson_word_ids: List[int],num_questions: int = 20):
    """
    Run a single lesson in the command-line interface.

    This is a thin wrapper around the engine that:
    - initializes session state,
    - loops through question selection + user input + grading

    The core scheduling and learning logic lives in engine_* functions;
    this function is intentionally I/O-focused.
    """

    # 1) create engine state
    state = engine_start(conn, user_id, language_code, topic_slug, lesson_word_ids)

    for i in range(num_questions):
        # 2) get next question
        state, q = engine_next_question(conn, user_id, state, language_code)
        word_id = q["question"]["word_id"]
        prompt = q["question"]["prompt"]

        # 3) ask user
        user_ans = getvalidInput(f"[t={q['t']}] What is '{prompt}' in French? ")

        # 4) submit answer
        state, result = engine_submit_answer(conn, user_id, state, user_ans, language_code)

        # 5) show immediate feedback (CLI only)
        exact = result["result"]["was_exact_match"]
        print("Correct!" if exact else f"Not quite. Correct: {result['result']['correct_answer']}")
        print(f"Score: {result['result']['score']:.2f} | Mastery now: {result['result']['mastery']:.2f}")

        # 6) if lesson summary is produced only every 20, show it when available
        if result.get("lesson_complete"):
            print("\n--- Lesson Summary ---")
            pats = result.get("learned_patterns_this_lesson", [])
            if pats:
                for feat, cnt in pats:
                    print(f"  - {feat}: {cnt}")
            else:
                print("  (no common errors this lesson)")
            print("----------------------\n")
    print("\n--- Lesson Mastery Snapshot ---")
    snapshot = get_topic_mastery_snapshot(conn, user_id, language_code, topic_slug)

    snapshot.sort(key=lambda x: x["mastery"])

    for row in snapshot:
        if row["mastery"] >= TARGET_MASTERY:
            status = "On target"
        elif row["mastery"] > 0:
            status = "Below target"
        else:
            status = "Not seen yet"

        print(
            f"  {row['prompt']:8} -> {row['answer']:8} : "
            f"{row['mastery']:.2f} {status}"
        )

    print("-------------------------------\n")

    return state

def main():
    """
      CLI entrypoint.

      Responsibilities:
      - Initialise database and seed demo content (idempotent seeding expected).
      - Create/load a demo user.
      - Ensure the topic progression path exists for the user.
      - Run lessons in a loop:
          * pick active topic
          * run one lesson session via session_runner_cli (engine-backed)
          * check mastery and unlock next topic if completed
      """
    conn = init_db("word_learning.db")

    # Seed data once
    seed_words(conn, COLOURS, language_code="fr", topic_slug="colours")
    seed_words(conn, FRUITS,  language_code="fr", topic_slug="fruits")

    user_id = get_or_create_user(conn, "cli_user")
    language_code = "fr"

    ensure_topic_path(conn, user_id, language_code)

    lesson_index = 1

    while True:
        topic_slug = get_active_topic(conn, user_id, language_code)

        print("\n====================")
        print(f"Lesson {lesson_index} — Topic: {topic_slug}")
        print("====================\n")

        lesson_word_ids = get_word_ids_for_topic(conn, language_code, topic_slug)
        if not lesson_word_ids:
            print("No words found for topic. Exiting.")
            break

        #  Run ONE lesson via the engine-backed CLI wrapper
        session_runner_cli(
            conn=conn,
            user_id=user_id,
            language_code=language_code,
            topic_slug=topic_slug,
            lesson_word_ids=lesson_word_ids,
            num_questions=20,
        )

        # Check topic mastery AFTER lesson
        if topic_mastered(conn, user_id, language_code, topic_slug):
            print(f"\n Mastered topic '{topic_slug}'! Unlocking next topic...")
            set_topic_completed_and_unlock_next(conn, user_id, topic_slug, language_code)

            next_topic = get_active_topic(conn, user_id, language_code)
            if next_topic == topic_slug:
                print("\n All topics completed!")
                break
        else:
            print(f"\n Topic '{topic_slug}' not mastered yet. More practice needed.")

        choice = input("\nStart next lesson? (Enter = yes, q = quit): ").strip().lower()
        if choice == "q":
            break

        lesson_index += 1


if __name__ == "__main__":
    main()
