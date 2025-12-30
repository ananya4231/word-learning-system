import os
from fastapi import FastAPI #creates web server + endpoints
from fastapi.middleware.cors import CORSMiddleware #Lets a browser call API
from pydantic import BaseModel #Validates JSON input for requests
from collections import defaultdict
from typing import Dict, Any, List, Tuple
from data import COLOURS, FRUITS

from app import (
    engine_start,
    engine_next_question,
    engine_submit_answer,
    get_topic_mastery_snapshot,
    topic_mastered,
    load_progress,
    pattern_bonus,
    EngineState,
    TARGET_MASTERY,
    HISTORY_SIZE,          # if API enforces history length (ideally engine does)
    NEW_PER_LESSON,        # if lesson structure is exposed
    RETRY_THRESH,          # if frontend wants retry hints
    FAIL_FAST_THRESH      # if frontend shows explanations

)

from db import (
    init_db,
    seed_words,
    get_or_create_user,
    ensure_topic_path,
    get_active_topic,
    set_topic_completed_and_unlock_next,
    get_word_ids_for_topic,
    get_word_by_id,
    get_state_or_default,
    upsert_state,
    inc_error_feature,
    get_top_error_features,
    prune_user_error_stats,
)

app = FastAPI()

#CORS so a browser frontend can call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#Connecting to SQLite

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_TO_DB = os.getenv("WORD_LEARNING_DB", os.path.join(BASE_DIR, "word_learning.db"))
conn = init_db(PATH_TO_DB)

seed_words(conn, COLOURS, language_code="fr", topic_slug="colours")
seed_words(conn, FRUITS,  language_code="fr", topic_slug="fruits")

#Storing per-user session state
SESSIONS: Dict[int, Dict[str, Any]] = {}

class StartReq(BaseModel):
    username: str
    language_code: str = "fr"

class AnswerReq(BaseModel):
    user_id: int
    word_id: int
    user_answer: str
    language_code: str = "fr"


#Pydantic models used to define API contract
#Schemas that describe the shape of the JSON data the API expects
@app.post("/start")
def start(req: StartReq):
    user_id = get_or_create_user(conn, req.username)
    ensure_topic_path(conn, user_id, req.language_code)

    topic_slug = get_active_topic(conn, user_id, req.language_code)
    lesson_word_ids = get_word_ids_for_topic(conn, req.language_code, topic_slug)

    if not lesson_word_ids:
        return {"error": f"No words found for topic '{topic_slug}' / language '{req.language_code}'"}

    state = engine_start(conn, user_id, req.language_code, topic_slug, lesson_word_ids)
    state, qpayload = engine_next_question(conn, user_id, state, req.language_code)

    SESSIONS[user_id] = state

    return {
        "user_id": user_id,
        "topic": topic_slug,
        "target_mastery": TARGET_MASTERY,
        **qpayload,
    }

#Endpoint to confirm the API is running
@app.get("/health")
def health():
    return {"ok": True}

@app.get("/")
def root():
    return {"message": "API running. See /docs"}


@app.post("/answer")
def answer(req: AnswerReq):
    if req.user_id not in SESSIONS:
        return {"error": "No active session. Call /start first."}

    state: EngineState = SESSIONS[req.user_id]

    # grade + update mastery + log patterns + (pruning if you put it in engine)
    state, result_payload = engine_submit_answer(conn, req.user_id, state, req.user_answer, req.language_code)
    snapshot = None
    if result_payload.get("lesson_complete"):
        snapshot = get_topic_mastery_snapshot(conn, req.user_id, req.language_code, state.topic_slug)
        snapshot.sort(key=lambda r: r["mastery"])  # weakest first

    # topic unlock check (still DB-level; fine in API)
    topic = state.topic_slug
    if topic_mastered(conn, req.user_id, req.language_code, topic):
        set_topic_completed_and_unlock_next(conn, req.user_id, topic, req.language_code)
        new_topic = get_active_topic(conn, req.user_id, req.language_code)

        if new_topic != topic:
            # refresh the engine state for the new topic
            lesson_word_ids = get_word_ids_for_topic(conn, req.language_code, new_topic)
            state = engine_start(conn, req.user_id, req.language_code, new_topic, lesson_word_ids)

    # get next question
    state, qpayload = engine_next_question(conn, req.user_id, state, req.language_code)

    SESSIONS[req.user_id] = state

    return {
        **result_payload,
        "topic": state.topic_slug,
        "next_question": qpayload["question"],
        "lesson_mastery": (
            [{"prompt": r["prompt"], "answer": r["answer"], "mastery": r["mastery"], "attempts": r["attempts"]}
             for r in snapshot]
            if snapshot is not None else []
        ),
    }


