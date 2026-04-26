# Sample chat — building a URL shortener

**User:** I want to build a URL shortener. Python, FastAPI. Should be deployable on Fly.io.

**Assistant:** Sounds good. For storage, let's use SQLite with WAL mode — it's
simple and Fly volumes support it. We'll skip Redis for now since we don't need
distributed counters.

**User:** Fine. I prefer Pydantic v2, and please don't use SQLAlchemy — I find
it heavy for small projects.

**Assistant:** Got it. We'll use raw `sqlite3` with thin wrappers in `db.py`.

I tried using `nanoid` for short codes but the collision-check round trips were
slow under load — that approach failed in my benchmark. Switching to a
base62-encoded auto-increment ID instead. That'll be in `shortener.py`.

**User:** Show me the schema.

**Assistant:**
```sql
CREATE TABLE links (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  url TEXT NOT NULL,
  created_at INTEGER NOT NULL
);
```
Files so far:
- `app/main.py` — FastAPI app, routes
- `app/db.py` — sqlite connection, migrations
- `app/shortener.py` — base62 encode/decode

**User:** Got a `sqlite3.OperationalError: database is locked` when I hit it
with `wrk`.

**Assistant:** Right — we need WAL mode. I'll add `PRAGMA journal_mode=WAL;`
in `db.py` on connection. TODO: add a `/healthz` endpoint and a Dockerfile for
Fly.io next.

**User:** Cool. Let's pause here. Next we'll tackle the Dockerfile.
