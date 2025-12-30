import sqlite3
from typing import List

TOPIC_PATH = ["colours", "fruits"]  # extend later
def init_db(db_path: str = "word_learning.db") -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON;")

    conn.execute("""
    CREATE TABLE IF NOT EXISTS users (
        user_id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL
    );
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS words (
        word_id INTEGER PRIMARY KEY AUTOINCREMENT,
        language_code TEXT NOT NULL,
        topic_slug TEXT NOT NULL,
        prompt TEXT NOT NULL,
        answer TEXT NOT NULL,
        UNIQUE(language_code, topic_slug, prompt, answer)
    );
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS user_word_state (
        user_id INTEGER NOT NULL,
        word_id INTEGER NOT NULL,
        mastery REAL NOT NULL,
        attempts INTEGER NOT NULL,
        last_seen_t INTEGER,
        next_due_t INTEGER NOT NULL,
        PRIMARY KEY (user_id, word_id),
        FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
        FOREIGN KEY (word_id) REFERENCES words(word_id) ON DELETE CASCADE
    );
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS user_topic_state (
        user_id INTEGER NOT NULL,
        language_code TEXT NOT NULL,
        topic_slug TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'locked',   -- locked | active | completed
        completed_at INTEGER,
        PRIMARY KEY (user_id, language_code, topic_slug),
        FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
    );
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS user_error_stats (
            user_id INTEGER NOT NULL,
            language_code TEXT NOT NULL,
            feature TEXT NOT NULL,
            count INTEGER NOT NULL,
            PRIMARY KEY (user_id, language_code, feature),
            FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
        );
        """)


    conn.commit()
    return conn

#Takes a list of vocabulary items and makes sure they exist in the database for a given language and
def seed_words(conn: sqlite3.Connection, words: list, language_code="fr", topic_slug="colours"):
    for w in words:
        #For each word, either insert into the table or ignore if already present (as words has UNIQUE constraint)
        #Carrying out parameter binding to prevent SQL Injection, passing shape and raw values separately
        conn.execute("""
        INSERT OR IGNORE INTO words (language_code, topic_slug, prompt, answer)
        VALUES (?, ?, ?, ?)
        """, (language_code, topic_slug, w["prompt"], w["answer"]))
    conn.commit()

#If a username is new, adds it to the database and return the userIDs
def get_or_create_user(conn: sqlite3.Connection, username: str) -> int:
    conn.execute("INSERT OR IGNORE INTO users (username) VALUES (?)", (username,))

    #Retrieves the row corresponding to the user
    row = conn.execute("SELECT user_id FROM users WHERE username = ?", (username,)).fetchone()
    conn.commit()

    #Returns userID
    return row[0]

#Create or update the learning state based on user and word, creates state first time, and updates every time user practises word
def upsert_state(conn: sqlite3.Connection, user_id: int, word_id: int, mastery: float, attempts: int, last_seen_t: int, next_due_t: int):
    #Enables storage of the learner's progress
    conn.execute("""
    INSERT INTO user_word_state (user_id, word_id, mastery, attempts, last_seen_t, next_due_t)
    VALUES (?, ?, ?, ?, ?, ?)
    ON CONFLICT(user_id, word_id) DO UPDATE SET
      mastery=excluded.mastery,
      attempts=excluded.attempts,
      last_seen_t=excluded.last_seen_t,
      next_due_t=excluded.next_due_t
    """, (user_id, word_id, mastery, attempts, last_seen_t, next_due_t))
    conn.commit()

#Returns information about the learning state of the user, or creates a default if user has never attempted a word before
def get_state_or_default(conn: sqlite3.Connection, user_id: int, word_id: int):
    #Returns learning state if one exists
    row = conn.execute("""
        SELECT mastery, attempts, next_due_t
        FROM user_word_state
        WHERE user_id = ? AND word_id = ?
    """, (user_id, word_id)).fetchone()

    #If row doesn't exist set mastery = 0.3 as a default value - New words are likely to be weaker, but not completely unknown
    #0.3 is chosen as a prior, to stabilise the progression of the mastery scores, this would be callibrated from real data
    if row is None:
        return 0.3, 0, 0  # default mastery, attempts, next_due_t
    return float(row[0]), int(row[1]), int(row[2])

TOPIC_PATH = ["colours", "fruits"]  # Can extend later

#Initialises and validates user's topic progression, guarantees a starting point for the user
def ensure_topic_path(conn: sqlite3.Connection, user_id: int, language_code: str = "fr"):
    # Insert path rows as locked, Making sure there's a row for each topic
    for slug in TOPIC_PATH:
        conn.execute("""
        INSERT OR IGNORE INTO user_topic_state (user_id, language_code, topic_slug, status)
        VALUES (?, ?, ?, 'locked')
        """, (user_id, language_code, slug))

    # Checking if any topics are active
    row = conn.execute("""
        SELECT topic_slug
        FROM user_topic_state
        WHERE user_id=? AND language_code=? AND status='active'
        LIMIT 1
    """, (user_id, language_code)).fetchone()

    #If none of the topics are active set the first topic as active
    #TOPIC_PATH[0] = 'colours'
    if row is None:
        conn.execute("""
        UPDATE user_topic_state
        SET status='active'
        WHERE user_id=? AND language_code=? AND topic_slug=?
        """, (user_id, language_code, TOPIC_PATH[0]))

    #Making changes permanent
    conn.commit()

def get_active_topic(conn: sqlite3.Connection, user_id: int, language_code: str = "fr") -> str:
    row = conn.execute("""
        SELECT topic_slug
        FROM user_topic_state
        WHERE user_id=? AND language_code=? AND status='active'
        LIMIT 1
    """, (user_id, language_code)).fetchone()
    return row[0] if row else TOPIC_PATH[0]

def set_topic_completed_and_unlock_next(conn: sqlite3.Connection, user_id: int, current_slug: str, language_code: str = "fr"):
    # mark completed
    conn.execute("""
        UPDATE user_topic_state
        SET status='completed', completed_at=COALESCE(completed_at, strftime('%s','now'))
        WHERE user_id=? AND language_code=? AND topic_slug=?
    """, (user_id, language_code, current_slug))

    # unlock next
    idx = TOPIC_PATH.index(current_slug)
    if idx + 1 < len(TOPIC_PATH):
        next_slug = TOPIC_PATH[idx + 1]
        conn.execute("""
            UPDATE user_topic_state
            SET status='active'
            WHERE user_id=? AND language_code=? AND topic_slug=? AND status='locked'
        """, (user_id, language_code, next_slug))

    conn.commit()

def get_word_ids_for_topic(conn: sqlite3.Connection, language_code: str, topic_slug: str) -> List[int]:
    rows = conn.execute("""
        SELECT word_id
        FROM words
        WHERE language_code=? AND topic_slug=?
        ORDER BY word_id
    """, (language_code, topic_slug)).fetchall()
    return [r[0] for r in rows]

def get_word_by_id(conn: sqlite3.Connection, word_id: int) -> dict:
    row = conn.execute("""
        SELECT word_id, language_code, topic_slug, prompt, answer
        FROM words
        WHERE word_id=?
    """, (word_id,)).fetchone()
    if row is None:
        raise KeyError(f"word_id {word_id} not found")
    return {"id": row[0], "language_code": row[1], "topic_slug": row[2], "prompt": row[3], "answer": row[4]}

#Increments a counter for a specific error pattern
def inc_error_feature(conn, user_id: int, language_code: str, feature: str, delta: int = 1):
    conn.execute("""
    INSERT INTO user_error_stats (user_id, language_code, feature, count)
    VALUES (?, ?, ?, ?)
    ON CONFLICT(user_id, language_code, feature)
    DO UPDATE SET count = count + ?
    """, (user_id, language_code, feature, delta, delta))

#Returns the user's learned behaviour summary, returns lifetime view of user errors
def get_top_error_features(conn, user_id: int, language_code: str, limit: int = 10):
    rows = conn.execute("""
        SELECT feature, count
        FROM user_error_stats
        WHERE user_id=? AND language_code=?
        ORDER BY count DESC
        LIMIT ?
    """, (user_id, language_code, limit)).fetchall()
    return [(r[0], int(r[1])) for r in rows]

#Keep only the top-N most frequent error features for this user+language.
#Deletes long-tail noise so the table can't grow without bound.
def prune_user_error_stats(conn, user_id: int, language_code: str, keep_top: int = 200) -> None:
    conn.execute("""
        DELETE FROM user_error_stats
        WHERE user_id = ? AND language_code = ?
          AND feature NOT IN (
              SELECT feature
              FROM user_error_stats
              WHERE user_id = ? AND language_code = ?
              ORDER BY count DESC
              LIMIT ?
          )
    """, (user_id, language_code, user_id, language_code, keep_top))
    conn.commit()
