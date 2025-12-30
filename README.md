# word-learning-system

Note: The database is created automatically on first run. Each username has an independent learning state.
# Adaptive Word Learning Demo

An adaptive word-learning system that personalises practice using graded correctness,
error pattern recognition, and spaced repetition.


## Overview

The system teaches individual words by:
- grading answers on a spectrum (not just correct / incorrect),
- tracking how users make mistakes,
- adapting question selection based on mastery, timing, and recurring error patterns.

It is implemented as:
- a core learning engine (`app.py`),
- a FastAPI backend (`api.py`),
- a simple CLI runner,
- and a lightweight browser frontend (`index.html`).


## How the system works

### 1. Graded correctness (not binary)
User answers are compared to the target word using a normalized Levenshtein distance.
This produces a score between 0 and 1, allowing the system to recognise *partial correctness*
(e.g. minor spelling errors or accent mistakes).

This score feeds directly into mastery updates and scheduling decisions.


### 2. Mastery tracking
Each word maintains a mastery value that is updated using an exponential moving average.
Recent performance influences mastery more than older attempts, allowing the system
to adapt as the learner improves or regresses.


### 3. Spaced repetition with fail-fast behaviour
Review intervals increase non-linearly as mastery increases.
If an attempt is very poor, the word is scheduled again quickly (fail-fast),
ensuring misconceptions are corrected early.


### 4. Error pattern recognition (the “AI” component)
Instead of only tracking correctness, the engine extracts error patterns from incorrect answers,
such as:
- letter substitutions (e.g. `sub:e->a`)
- insertions or deletions (e.g. `del:e`)
- accent-related issues (e.g. `accent:issue`)
- letter order swaps (e.g. `swap:ge`)

These patterns are counted and ranked.

Words that contain patterns the learner frequently gets wrong are given higher priority
when selecting future questions. This nudges practice toward the learner’s weaknesses
in an interpretable, explainable way.


### 5. Lessons and progression
- Each lesson consists of **20 questions**.
- New words are gradually introduced and interleaved with review items.
- Recent repeats are avoided to prevent “spamming” the same word.
- At the end of a lesson, a summary is shown in the UI.

## AI patterns explained

### AI patterns (this lesson)
This section shows the most common error patterns detected *during the current lesson only*.

These patterns are used immediately to influence question selection within the same lesson,
helping the learner correct recurring mistakes quickly.

The list is reset at the start of each new lesson.

### AI patterns (lifetime)
This section shows error patterns accumulated *across all lessons and sessions* for a user.

These lifetime patterns represent persistent weaknesses and continue to influence scheduling
over time, allowing long-term personalisation rather than starting from scratch each session.


## Why this approach makes sense for this project

This project focuses on *personalised learning*, where:
- interpretability matters,
- data is limited (single user, word-level interactions),
- and feedback needs to be immediate and understandable.

Using error patterns and adaptation provides:
- transparency (it’s clear why a word was selected),
- robustness with small datasets,
- and behaviour that can be easily inspected, debugged, and extended.

Rather than training a large model, the system applies adaptive heuristics driven by
learner behaviour, which is well-suited for early-stage educational tools and
human-in-the-loop learning scenarios.


## Running the API

```bash
pip install -r requirements.txt
uvicorn api:app --reload
