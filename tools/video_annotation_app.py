import argparse
import cgi
import hashlib
import hmac
import json
import math
import mimetypes
import random
import re
import secrets
import sqlite3
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "annotation_workspace" / "annotations.sqlite3"
DEFAULT_UPLOAD_DIR = ROOT / "annotation_workspace" / "uploads"
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
CLIENT_DISCONNECT_ERRORS = (BrokenPipeError, ConnectionAbortedError, ConnectionResetError)


def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def parse_json_body(handler):
    length = int(handler.headers.get("Content-Length", "0") or 0)
    if length <= 0:
        return {}
    raw = handler.rfile.read(length)
    return json.loads(raw.decode("utf-8"))


def dumps(data):
    return json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def hash_password(password, salt=None):
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", str(password).encode("utf-8"), salt.encode("utf-8"), 120000)
    return f"{salt}${digest.hex()}"


def verify_password(password, password_hash):
    if not password_hash or "$" not in password_hash:
        return False
    salt, expected = password_hash.split("$", 1)
    actual = hash_password(password, salt).split("$", 1)[1]
    return hmac.compare_digest(actual, expected)


def public_user(row):
    data = dict(row)
    data.pop("password_hash", None)
    return data


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_jsonl_text(text):
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def infer_split(value):
    name = Path(str(value)).stem.lower()
    if name in {"train", "val", "test"}:
        return name
    for part in name.replace("-", "_").split("_"):
        if part in {"train", "val", "test"}:
            return part
    return None


def video_id_base(vid):
    parts = str(vid).rsplit("_", 2)
    if len(parts) == 3:
        return parts[0]
    return str(vid)


def clip_ids_for_windows(windows, duration, clip_len):
    ids = set()
    n_clips = max(1, int(math.ceil(float(duration) / clip_len)))
    for start, end in windows:
        start_idx = max(0, int(math.floor(float(start) / clip_len)))
        end_idx = min(n_clips - 1, max(0, int(math.ceil(float(end) / clip_len)) - 1))
        ids.update(range(start_idx, end_idx + 1))
    return sorted(ids)


def normalize_windows(windows, duration):
    normalized = []
    duration = float(duration or 0)
    for item in windows or []:
        if isinstance(item, dict):
            start = item.get("start", item.get("startTimeSec", 0))
            end = item.get("end", item.get("endTimeSec", 0))
        else:
            start, end = item[0], item[1]
        start = max(0.0, float(start))
        end = max(0.0, float(end))
        if duration > 0:
            start = min(start, duration)
            end = min(end, duration)
        if end > start:
            normalized.append([round(start, 3), round(end, 3)])
    normalized.sort(key=lambda x: (x[0], x[1]))
    return normalized


def normalize_single_scores(scores, n_items, default=4):
    normalized = []
    for idx in range(n_items):
        value = scores[idx] if idx < len(scores or []) else default
        if isinstance(value, list):
            nums = [float(v) for v in value if isinstance(v, (int, float))]
            value = round(sum(nums) / len(nums)) if nums else default
        try:
            value = int(round(float(value)))
        except (TypeError, ValueError):
            value = default
        normalized.append(max(0, min(4, value)))
    return normalized


def unpack_saliency(value, n_windows=0):
    if isinstance(value, str):
        try:
            value = json.loads(value or "[]")
        except json.JSONDecodeError:
            value = []
    if isinstance(value, dict):
        window_scores = normalize_single_scores(value.get("window_scores") or [], n_windows)
        raw_clip_scores = value.get("clip_scores") or {}
        clip_scores = {}
        for key, score in raw_clip_scores.items():
            try:
                clip_id = int(key)
                score = int(round(float(score)))
            except (TypeError, ValueError):
                continue
            clip_scores[str(clip_id)] = max(0, min(4, score))
        return {"window_scores": window_scores, "clip_scores": clip_scores}
    return {"window_scores": normalize_single_scores(value or [], n_windows), "clip_scores": {}}


def pack_saliency(window_scores, clip_scores=None, n_windows=0):
    packed = unpack_saliency(
        {"window_scores": window_scores or [], "clip_scores": clip_scores or {}},
        n_windows,
    )
    return packed


def qv_saliency_scores(single_scores, n_items):
    scores = normalize_single_scores(single_scores, n_items)
    return [[score, score, score] for score in scores]


def window_scores_from_qv(row):
    windows = row.get("relevant_windows") or []
    clip_ids = row.get("relevant_clip_ids") or []
    saliency = row.get("saliency_scores") or []
    clip_score = {}
    for idx, clip_id in enumerate(clip_ids):
        if idx >= len(saliency):
            continue
        value = saliency[idx]
        if isinstance(value, list):
            nums = [float(v) for v in value if isinstance(v, (int, float))]
            value = round(sum(nums) / len(nums)) if nums else 4
        clip_score[int(clip_id)] = max(0, min(4, int(round(float(value)))))
    scores = []
    for start, end in windows:
        ids = clip_ids_for_windows([[start, end]], row.get("duration", 0), 2.0)
        vals = [clip_score[i] for i in ids if i in clip_score]
        scores.append(round(sum(vals) / len(vals)) if vals else 4)
    return normalize_single_scores(scores, len(windows))


def clip_scores_from_qv(row):
    clip_ids = row.get("relevant_clip_ids") or []
    saliency = row.get("saliency_scores") or []
    out = {}
    for idx, clip_id in enumerate(clip_ids):
        if idx >= len(saliency):
            continue
        value = saliency[idx]
        if isinstance(value, list):
            nums = [float(v) for v in value if isinstance(v, (int, float))]
            value = round(sum(nums) / len(nums)) if nums else 4
        try:
            out[str(int(clip_id))] = max(0, min(4, int(round(float(value)))))
        except (TypeError, ValueError):
            continue
    return out


def qv_saliency_from_annotation(windows, saliency_data, clip_ids, clip_len=2.0):
    scores = unpack_saliency(saliency_data, len(windows))
    window_scores = scores["window_scores"]
    clip_scores = scores["clip_scores"]
    out = []
    for clip_id in clip_ids:
        if str(int(clip_id)) in clip_scores:
            value = clip_scores[str(int(clip_id))]
            out.append([value, value, value])
            continue
        clip_start = float(clip_id) * clip_len
        clip_end = clip_start + clip_len
        value = 4
        for idx, (start, end) in enumerate(windows):
            if clip_end > float(start) and clip_start < float(end):
                value = window_scores[idx]
                break
        out.append([value, value, value])
    return out


def init_db(db_path):
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                qid INTEGER PRIMARY KEY,
                query TEXT NOT NULL,
                duration REAL NOT NULL,
                vid TEXT NOT NULL,
                video_path TEXT,
                group_id INTEGER,
                split TEXT,
                status TEXT NOT NULL DEFAULT 'todo',
                claimed_by TEXT,
                claimed_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_tasks_vid ON tasks(vid);
            CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
            CREATE INDEX IF NOT EXISTS idx_tasks_split ON tasks(split);

            CREATE TABLE IF NOT EXISTS task_groups (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                notes TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'published',
                claimed_by TEXT,
                claimed_at TEXT,
                created_by TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_task_groups_status ON task_groups(status);
            CREATE INDEX IF NOT EXISTS idx_task_groups_claimed_by ON task_groups(claimed_by);

            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                role TEXT NOT NULL CHECK(role IN ('annotator', 'reviewer')),
                display_name TEXT,
                password_hash TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS annotations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                qid INTEGER NOT NULL,
                annotator TEXT NOT NULL,
                windows_json TEXT NOT NULL DEFAULT '[]',
                saliency_json TEXT NOT NULL DEFAULT '[]',
                notes TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'draft',
                reviewed_by TEXT,
                reviewed_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(qid, annotator),
                FOREIGN KEY(qid) REFERENCES tasks(qid)
            );
            CREATE INDEX IF NOT EXISTS idx_annotations_qid ON annotations(qid);
            CREATE INDEX IF NOT EXISTS idx_annotations_status ON annotations(status);
            """
        )
        task_cols = {r[1] for r in conn.execute("PRAGMA table_info(tasks)").fetchall()}
        if "group_id" not in task_cols:
            conn.execute("ALTER TABLE tasks ADD COLUMN group_id INTEGER")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_group ON tasks(group_id)")
        group_cols = {r[1] for r in conn.execute("PRAGMA table_info(task_groups)").fetchall()}
        if "notes" not in group_cols:
            conn.execute("ALTER TABLE task_groups ADD COLUMN notes TEXT NOT NULL DEFAULT ''")
        user_cols = {r[1] for r in conn.execute("PRAGMA table_info(users)").fetchall()}
        if "password_hash" not in user_cols:
            conn.execute("ALTER TABLE users ADD COLUMN password_hash TEXT")
        conn.execute(
            """
            INSERT INTO users(username, role, display_name, password_hash, created_at, updated_at)
            VALUES('annotator1', 'annotator', 'annotator1', ?, ?, ?)
            ON CONFLICT(username) DO NOTHING
            """,
            (hash_password("annotator1"), now_iso(), now_iso()),
        )
        conn.execute(
            """
            INSERT INTO users(username, role, display_name, password_hash, created_at, updated_at)
            VALUES('reviewer1', 'reviewer', 'reviewer1', ?, ?, ?)
            ON CONFLICT(username) DO NOTHING
            """,
            (hash_password("reviewer1"), now_iso(), now_iso()),
        )
        conn.execute("UPDATE users SET role='annotator', display_name=COALESCE(display_name,'annotator1'), updated_at=? WHERE username='annotator1'", (now_iso(),))
        conn.execute("UPDATE users SET role='reviewer', display_name=COALESCE(display_name,'reviewer1'), updated_at=? WHERE username='reviewer1'", (now_iso(),))
        for row in conn.execute("SELECT username FROM users WHERE password_hash IS NULL OR password_hash=''").fetchall():
            conn.execute(
                "UPDATE users SET password_hash=?, updated_at=? WHERE username=?",
                (hash_password(row[0]), now_iso(), row[0]),
            )


class Store:
    def __init__(self, db_path):
        self.db_path = str(db_path)
        init_db(db_path)

    def connect(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def get_user(self, username):
        username = str(username or "").strip()
        if not username:
            raise PermissionError("login required")
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
            if row:
                return dict(row)
        raise PermissionError("user not found")

    def register_user(self, username, password, role, display_name=None):
        username = str(username or "").strip()
        password = str(password or "")
        role = str(role or "annotator").strip()
        if not username:
            raise ValueError("username is required")
        if len(password) < 4:
            raise ValueError("password must be at least 4 characters")
        if role not in {"annotator", "reviewer"}:
            raise ValueError("role must be annotator or reviewer")
        with self.connect() as conn:
            if conn.execute("SELECT 1 FROM users WHERE username=?", (username,)).fetchone():
                raise ValueError("username already exists")
            conn.execute(
                """
                INSERT INTO users(username, role, display_name, password_hash, created_at, updated_at)
                VALUES(?,?,?,?,?,?)
                """,
                (username, role, display_name or username, hash_password(password), now_iso(), now_iso()),
            )
            row = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
        return public_user(row)

    def login_user(self, username, password):
        username = str(username or "").strip()
        password = str(password or "")
        if not username or not password:
            raise ValueError("username and password are required")
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
        if not row or not verify_password(password, row["password_hash"]):
            raise PermissionError("invalid username or password")
        return public_user(row)

    def set_user(self, username, role, display_name=None):
        user = self.get_user(username)
        return public_user(user)

    def list_users(self):
        with self.connect() as conn:
            return [public_user(r) for r in conn.execute("SELECT * FROM users ORDER BY username").fetchall()]

    def require_role(self, username, roles):
        user = self.get_user(username)
        if user["role"] not in set(roles):
            raise PermissionError(f"{user['username']} is {user['role']}, required: {', '.join(roles)}")
        return user

    def stats(self):
        with self.connect() as conn:
            task_rows = conn.execute(
                "SELECT status, COUNT(*) n FROM tasks GROUP BY status"
            ).fetchall()
            ann_rows = conn.execute(
                "SELECT status, COUNT(*) n FROM annotations GROUP BY status"
            ).fetchall()
            return {
                "tasks": {r["status"]: r["n"] for r in task_rows},
                "annotations": {r["status"]: r["n"] for r in ann_rows},
                "users": {
                    r["role"]: r["n"]
                    for r in conn.execute("SELECT role, COUNT(*) n FROM users GROUP BY role").fetchall()
                },
                "groups": {
                    r["status"]: r["n"]
                    for r in conn.execute("SELECT status, COUNT(*) n FROM task_groups GROUP BY status").fetchall()
                },
                "total_tasks": conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0],
                "unpublished_videos": conn.execute("""
                    SELECT COUNT(*) FROM (
                        SELECT COALESCE(NULLIF(vid, ''), 'qid-' || qid) AS video_key
                        FROM tasks
                        WHERE group_id IS NULL AND video_path IS NOT NULL AND video_path != ''
                        GROUP BY video_key
                    )
                """).fetchone()[0],
            }

    def list_groups(self, params):
        where = []
        values = []
        status = params.get("status")
        user = params.get("user")
        user_info = self.get_user(user)
        if status and status != "all":
            where.append("g.status=?")
            values.append(status)
        if user_info["role"] == "annotator":
            where.append("(g.claimed_by IS NULL OR g.claimed_by='' OR g.claimed_by=?)")
            values.append(user_info["username"])
        sql_where = "WHERE " + " AND ".join(where) if where else ""
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT g.*,
                    (
                        SELECT COUNT(*) FROM (
                            SELECT COALESCE(NULLIF(t2.vid,''), 'qid-' || t2.qid) AS video_key
                            FROM tasks t2
                            WHERE t2.group_id=g.id
                            GROUP BY video_key
                        )
                    ) AS video_count,
                    (
                        SELECT COUNT(*) FROM (
                            SELECT COALESCE(NULLIF(t2.vid,''), 'qid-' || t2.qid) AS video_key
                            FROM tasks t2
                            WHERE t2.group_id=g.id
                            GROUP BY video_key
                            HAVING SUM(CASE WHEN t2.status='approved' THEN 1 ELSE 0 END)=COUNT(*)
                        )
                    ) AS approved_count,
                    (
                        SELECT COUNT(*) FROM (
                            SELECT COALESCE(NULLIF(t2.vid,''), 'qid-' || t2.qid) AS video_key
                            FROM tasks t2
                            WHERE t2.group_id=g.id AND EXISTS (
                                SELECT 1 FROM annotations a WHERE a.qid=t2.qid AND json_array_length(a.windows_json)>0
                            )
                            GROUP BY video_key
                        )
                    ) AS annotated_count,
                    (
                        SELECT COUNT(*) FROM (
                            SELECT COALESCE(NULLIF(t2.vid,''), 'qid-' || t2.qid) AS video_key
                            FROM tasks t2
                            WHERE t2.group_id=g.id AND EXISTS (
                                SELECT 1 FROM annotations a WHERE a.qid=t2.qid AND a.status='submitted'
                            )
                            GROUP BY video_key
                        )
                    ) AS submitted_count,
                (
                    SELECT COUNT(*) FROM (
                        SELECT COALESCE(NULLIF(t2.vid,''), 'qid-' || t2.qid) AS video_key
                        FROM tasks t2
                        WHERE t2.group_id=g.id AND EXISTS (
                            SELECT 1 FROM annotations a WHERE a.qid=t2.qid AND a.status='draft'
                        )
                        GROUP BY video_key
                    )
                ) AS draft_count,
                (
                    SELECT COUNT(*) FROM (
                        SELECT COALESCE(NULLIF(t2.vid,''), 'qid-' || t2.qid) AS video_key
                        FROM tasks t2
                        WHERE t2.group_id=g.id AND EXISTS (
                            SELECT 1 FROM annotations a WHERE a.qid=t2.qid AND a.status='rejected'
                        )
                        GROUP BY video_key
                    )
                ) AS rejected_count,
                SUM(CASE WHEN t.claimed_by IS NOT NULL AND t.claimed_by!='' THEN 1 ELSE 0 END) AS claimed_item_count,
                CASE WHEN g.claimed_by=? THEN 1 ELSE 0 END AS is_mine
                FROM task_groups g
                LEFT JOIN tasks t ON t.group_id=g.id
                {sql_where}
                GROUP BY g.id
                ORDER BY
                    CASE WHEN g.claimed_by=? THEN 0 WHEN g.claimed_by IS NULL THEN 1 ELSE 2 END,
                    g.id DESC
                """,
                [user_info["username"], *values, user_info["username"]],
            ).fetchall()
            return [dict(r) for r in rows]

    def get_group(self, group_id, user=None):
        user_info = self.get_user(user)
        with self.connect() as conn:
            group = conn.execute("SELECT * FROM task_groups WHERE id=?", (int(group_id),)).fetchone()
            if not group:
                return None
            if user_info["role"] == "annotator" and group["claimed_by"] not in (None, "", user_info["username"]):
                raise PermissionError("annotators cannot view task groups claimed by others")
            tasks = conn.execute(
                """
                SELECT t.*,
                    (SELECT COUNT(*) FROM annotations a WHERE a.qid=t.qid) AS ann_count,
                    (SELECT COUNT(*) FROM annotations a WHERE a.qid=t.qid AND a.status='approved') AS approved_count,
                    (SELECT COUNT(*) FROM annotations a WHERE a.qid=t.qid AND a.status='submitted') AS submitted_count,
                    (SELECT COUNT(*) FROM annotations a WHERE a.qid=t.qid AND a.status='rejected') AS rejected_count,
                    (SELECT status FROM annotations a WHERE a.qid=t.qid AND a.annotator=? LIMIT 1) AS my_status,
                    (SELECT json_array_length(windows_json) FROM annotations a WHERE a.qid=t.qid AND a.annotator=? LIMIT 1) AS my_window_count,
                    (SELECT json_array_length(windows_json) FROM annotations a WHERE a.qid=t.qid AND a.status IN ('submitted','approved') ORDER BY updated_at DESC LIMIT 1) AS display_window_count,
                    (SELECT updated_at FROM annotations a WHERE a.qid=t.qid AND a.annotator=? LIMIT 1) AS my_updated_at,
                    (SELECT annotator FROM annotations a WHERE a.qid=t.qid AND a.status='submitted' ORDER BY updated_at DESC LIMIT 1) AS submitted_annotator
                FROM tasks t
                WHERE t.group_id=?
                ORDER BY t.qid
                """,
                (user_info["username"], user_info["username"], user_info["username"], int(group_id)),
            ).fetchall()
        data = dict(group)
        data["tasks"] = [dict(r) for r in tasks]
        return data

    def refresh_group_status(self, conn, group_id):
        if group_id is None:
            return
        row = conn.execute(
            """
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN t.status='approved' THEN 1 ELSE 0 END) AS approved_count,
                SUM(CASE WHEN EXISTS (
                    SELECT 1 FROM annotations a WHERE a.qid=t.qid AND a.status='submitted'
                ) THEN 1 ELSE 0 END) AS submitted_count,
                SUM(CASE WHEN EXISTS (
                    SELECT 1 FROM annotations a WHERE a.qid=t.qid AND a.status='rejected'
                ) THEN 1 ELSE 0 END) AS rejected_count,
                MAX(CASE WHEN t.claimed_by IS NOT NULL AND t.claimed_by!='' THEN 1 ELSE 0 END) AS claimed
            FROM tasks t
            WHERE t.group_id=?
            """,
            (group_id,),
        ).fetchone()
        if not row or row["total"] == 0:
            return
        status = "published"
        if row["approved_count"] == row["total"]:
            status = "approved"
        elif row["submitted_count"]:
            status = "submitted"
        elif row["rejected_count"]:
            status = "rejected"
        elif row["claimed"]:
            status = "claimed"
        conn.execute("UPDATE task_groups SET status=?, updated_at=? WHERE id=?", (status, now_iso(), group_id))

    def publish_video_groups(self, data):
        user = data.get("user")
        self.require_role(user, {"reviewer"})
        group_size = max(1, int(data.get("group_size") or 1))
        requested_name = str(data.get("name_prefix") or "标注任务1").strip() or "标注任务1"
        match = re.match(r"^(.*?)(\d+)$", requested_name)
        if match:
            name_base = match.group(1) or "任务"
            requested_start = int(match.group(2))
        else:
            name_base = requested_name
            requested_start = 1
        with self.connect() as conn:
            max_existing = 0
            for row in conn.execute("SELECT name FROM task_groups").fetchall():
                m = re.match(rf"^{re.escape(name_base)}(\d+)$", str(row["name"]))
                if m:
                    max_existing = max(max_existing, int(m.group(1)))
            next_index = max(requested_start, max_existing + 1)
            rows = conn.execute(
                """
                SELECT qid, vid FROM tasks
                WHERE group_id IS NULL
                  AND video_path IS NOT NULL
                  AND video_path != ''
                ORDER BY qid
                """
            ).fetchall()
            videos = []
            seen = set()
            qids_by_video = {}
            for row in rows:
                video_key = row["vid"] or f"qid-{row['qid']}"
                qids_by_video.setdefault(video_key, []).append(row["qid"])
                if video_key not in seen:
                    seen.add(video_key)
                    videos.append(video_key)
            plans = []
            raw_groups = data.get("groups") or []
            if raw_groups:
                total_requested = 0
                for idx, item in enumerate(raw_groups, start=1):
                    count = int(item.get("video_count") or item.get("count") or 0)
                    if count <= 0:
                        raise ValueError("each task must contain at least one video")
                    total_requested += count
                    plans.append(
                        {
                            "name": str(item.get("name") or f"{name_base}{idx}").strip() or f"{name_base}{idx}",
                            "notes": str(item.get("notes") or ""),
                            "video_count": count,
                        }
                    )
                if total_requested != len(videos):
                    raise ValueError(f"task video counts must sum to {len(videos)}")
            else:
                for idx in range(0, len(videos), group_size):
                    plans.append(
                        {
                            "name": f"{name_base}{next_index + len(plans)}",
                            "notes": "",
                            "video_count": len(videos[idx : idx + group_size]),
                        }
                    )
            created = []
            offset = 0
            for plan in plans:
                chunk_videos = videos[offset : offset + plan["video_count"]]
                offset += plan["video_count"]
                chunk_qids = [qid for video in chunk_videos for qid in qids_by_video.get(video, [])]
                cur = conn.execute(
                    """
                    INSERT INTO task_groups(name, notes, status, created_by, created_at, updated_at)
                    VALUES(?,?,?,?,?,?)
                    """,
                    (plan["name"], plan["notes"], "published", user, now_iso(), now_iso()),
                )
                group_id = cur.lastrowid
                conn.executemany(
                    """
                    UPDATE tasks
                    SET group_id=?, status='todo', claimed_by=NULL, claimed_at=NULL, updated_at=?
                    WHERE qid=?
                    """,
                    [(group_id, now_iso(), qid) for qid in chunk_qids],
                )
                created.append(
                    {
                        "id": group_id,
                        "name": plan["name"],
                        "notes": plan["notes"],
                        "video_count": len(chunk_videos),
                        "query_count": len(chunk_qids),
                    }
                )
        return {"created_groups": created, "videos_published": len(videos), "group_size": group_size}

    def reset_workspace(self, user, keep_users=True):
        self.require_role(user, {"reviewer"})
        with self.connect() as conn:
            target_qids = [
                row["qid"]
                for row in conn.execute(
                    """
                    SELECT qid FROM tasks
                    WHERE group_id IS NULL
                      AND video_path IS NOT NULL
                      AND video_path != ''
                    """
                ).fetchall()
            ]
            if target_qids:
                placeholders = ",".join("?" for _ in target_qids)
                conn.execute(f"DELETE FROM annotations WHERE qid IN ({placeholders})", target_qids)
                conn.execute(f"DELETE FROM tasks WHERE qid IN ({placeholders})", target_qids)
            if not keep_users:
                conn.execute("DELETE FROM users")
        return {"reset": True, "keep_users": keep_users, "deleted_tasks": len(target_qids)}

    def delete_group(self, group_id, user):
        self.require_role(user, {"reviewer"})
        group_id = int(group_id)
        with self.connect() as conn:
            group = conn.execute("SELECT * FROM task_groups WHERE id=?", (group_id,)).fetchone()
            if not group:
                raise ValueError("task group not found")
            conn.execute(
                """
                UPDATE tasks
                SET group_id=NULL, status='todo', claimed_by=NULL, claimed_at=NULL, updated_at=?
                WHERE group_id=?
                """,
                (now_iso(), group_id),
            )
            conn.execute("DELETE FROM task_groups WHERE id=?", (group_id,))
        return {"deleted": True, "group_id": group_id, "name": group["name"]}
    def claim_group(self, group_id, user):
        self.require_role(user, {"annotator"})
        with self.connect() as conn:
            group = conn.execute("SELECT * FROM task_groups WHERE id=?", (int(group_id),)).fetchone()
            if not group:
                raise ValueError("task group not found")
            if group["claimed_by"] and group["claimed_by"] != user:
                raise PermissionError("task group is already claimed")
            conn.execute(
                """
                UPDATE task_groups
                SET status='claimed', claimed_by=?, claimed_at=COALESCE(claimed_at, ?), updated_at=?
                WHERE id=?
                """,
                (user, now_iso(), now_iso(), int(group_id)),
            )
            conn.execute(
                """
                UPDATE tasks
                SET claimed_by=?, claimed_at=COALESCE(claimed_at, ?), status='draft', updated_at=?
                WHERE group_id=?
                """,
                (user, now_iso(), now_iso(), int(group_id)),
            )
            self.refresh_group_status(conn, int(group_id))
        return self.get_group(group_id, user)

    def import_jsonl(self, path, split=None, video_root=None):
        rows = read_jsonl(path)
        split = split or infer_split(path)
        return self.import_rows(rows, split=split, video_root=video_root)

    def import_jsonl_text(self, text, split=None, video_root=None, filename=None):
        rows = parse_jsonl_text(text)
        split = split or infer_split(filename or "")
        return self.import_rows(rows, split=split, video_root=video_root)

    def import_rows(self, rows, split=None, video_root=None):
        added = 0
        updated = 0
        with self.connect() as conn:
            next_qid = conn.execute("SELECT COALESCE(MAX(qid), -1) + 1 FROM tasks").fetchone()[0]
            for row in rows:
                source_qid = row.get("qid")
                qid = int(source_qid if source_qid is not None else next_qid)
                next_qid = max(next_qid, qid + 1)
                vid = str(row["vid"])
                video_path = row.get("video_path")
                if not video_path and video_root:
                    video_path = find_video_path(video_root, vid)
                existing = conn.execute("SELECT vid FROM tasks WHERE qid=?", (qid,)).fetchone()
                if existing and existing["vid"] != vid:
                    qid = next_qid
                    next_qid += 1
                    existing = None
                conn.execute(
                    """
                    INSERT INTO tasks(qid, query, duration, vid, video_path, split, status, created_at, updated_at)
                    VALUES(?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(qid) DO UPDATE SET
                        query=excluded.query,
                        duration=excluded.duration,
                        vid=excluded.vid,
                        video_path=COALESCE(excluded.video_path, tasks.video_path),
                        split=COALESCE(excluded.split, tasks.split),
                        updated_at=excluded.updated_at
                    """,
                    (
                        qid,
                        str(row.get("query", "")),
                        float(row.get("duration", 0)),
                        vid,
                        video_path,
                        split or row.get("split"),
                        "todo",
                        now_iso(),
                        now_iso(),
                    ),
                )
                if row.get("relevant_windows"):
                    clip_ids = row.get("relevant_clip_ids") or clip_ids_for_windows(
                        row["relevant_windows"], row.get("duration", 0), 2.0
                    )
                    saliency = pack_saliency(
                        window_scores_from_qv(row),
                        clip_scores_from_qv(row),
                        len(row["relevant_windows"]),
                    )
                    conn.execute(
                        """
                        INSERT INTO annotations(qid, annotator, windows_json, saliency_json, notes, status, created_at, updated_at)
                        VALUES(?,?,?,?,?,?,?,?)
                        ON CONFLICT(qid, annotator) DO UPDATE SET
                            windows_json=excluded.windows_json,
                            saliency_json=excluded.saliency_json,
                            status=excluded.status,
                            updated_at=excluded.updated_at
                        """,
                        (
                            qid,
                            "imported",
                            json.dumps(row["relevant_windows"], ensure_ascii=False),
                            json.dumps(saliency, ensure_ascii=False),
                            "imported from jsonl",
                            "approved",
                            now_iso(),
                            now_iso(),
                        ),
                    )
                    conn.execute(
                        "UPDATE tasks SET status='approved', updated_at=? WHERE qid=?",
                        (now_iso(), qid),
                    )
                added += 0 if existing else 1
                updated += 1 if existing else 0
        return {"rows": len(rows), "added": added, "updated": updated}

    def import_videos(self, directory):
        directory = Path(directory)
        files = [p for p in directory.rglob("*") if p.suffix.lower() in VIDEO_EXTS]

        def video_import_sort_key(path):
            parts = [part.lower() for part in path.parts]
            if any(part in {"single", "单视频"} for part in parts):
                priority = 0
            elif any(part in {"multi", "视频"} for part in parts):
                priority = 2
            else:
                priority = 1
            return (priority, str(path).lower())
        with self.connect() as conn:
            next_qid = conn.execute("SELECT COALESCE(MAX(qid), -1) + 1 FROM tasks").fetchone()[0]
            added = 0
            for path in sorted(files, key=video_import_sort_key):
                vid = path.stem
                if conn.execute("SELECT 1 FROM tasks WHERE vid=?", (vid,)).fetchone():
                    continue
                duration = 0.0
                try:
                    import cv2
                    cap = cv2.VideoCapture(str(path))
                    if cap.isOpened():
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                        cap.release()
                        if fps > 0 and frames > 0:
                            duration = frames / fps
                except Exception:
                    pass
                conn.execute(
                    """
                    INSERT INTO tasks(qid, query, duration, vid, video_path, split, status, created_at, updated_at)
                    VALUES(?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        next_qid,
                        "",
                        duration,
                        vid,
                        str(path),
                        None,
                        "todo",
                        now_iso(),
                        now_iso(),
                    ),
                )
                next_qid += 1
                added += 1
        return {"found": len(files), "added": added}

    def attach_uploaded_video(self, path, qid=None, match_stem=None):
        path = Path(path)
        vid = match_stem or path.stem
        with self.connect() as conn:
            if qid is not None:
                conn.execute(
                    "UPDATE tasks SET video_path=?, updated_at=? WHERE qid=?",
                    (str(path), now_iso(), int(qid)),
                )
                return {"attached": 1, "qid": int(qid), "video_path": str(path)}
            rows = conn.execute(
                "SELECT qid, vid FROM tasks WHERE vid=? OR vid LIKE ? ORDER BY qid",
                (vid, f"{vid}_%")
            ).fetchall()
            for row in rows:
                conn.execute(
                    "UPDATE tasks SET video_path=?, updated_at=? WHERE qid=?",
                    (str(path), now_iso(), row["qid"]),
                )
            if rows:
                return {"attached": len(rows), "qid": rows[0]["qid"], "video_path": str(path)}
            next_qid = conn.execute("SELECT COALESCE(MAX(qid), -1) + 1 FROM tasks").fetchone()[0]
            conn.execute(
                """
                INSERT INTO tasks(qid, query, duration, vid, video_path, status, created_at, updated_at)
                VALUES(?,?,?,?,?,?,?,?)
                """,
                (next_qid, "", 0.0, vid, str(path), "todo", now_iso(), now_iso()),
            )
            return {"attached": 1, "qid": next_qid, "video_path": str(path), "created_task": True}

    def list_tasks(self, params):
        where = []
        values = []
        status = params.get("status")
        split = params.get("split")
        user = params.get("user")
        q = params.get("q")
        if status and status != "all":
            where.append("t.status=?")
            values.append(status)
        if split and split != "all":
            where.append("COALESCE(t.split,'')=?")
            values.append(split)
        if q:
            where.append("(t.query LIKE ? OR t.vid LIKE ? OR CAST(t.qid AS TEXT)=?)")
            values.extend([f"%{q}%", f"%{q}%", q])
        sql_where = "WHERE " + " AND ".join(where) if where else ""
        limit = min(int(params.get("limit", 200)), 1000)
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT t.*,
                    (SELECT COUNT(*) FROM annotations a WHERE a.qid=t.qid) AS ann_count,
                    (SELECT COUNT(*) FROM annotations a WHERE a.qid=t.qid AND a.status='approved') AS approved_count,
                    (SELECT status FROM annotations a WHERE a.qid=t.qid AND a.annotator=? LIMIT 1) AS my_status
                FROM tasks t
                {sql_where}
                ORDER BY t.qid
                LIMIT ?
                """,
                [user or "", *values, limit],
            ).fetchall()
            return [dict(r) for r in rows]

    def get_task(self, qid, user=None):
        with self.connect() as conn:
            task = conn.execute("SELECT * FROM tasks WHERE qid=?", (qid,)).fetchone()
            if not task:
                return None
            user_info = self.get_user(user) if user else {"role": "reviewer", "username": None}
            if user_info["role"] == "reviewer":
                anns = conn.execute(
                    "SELECT * FROM annotations WHERE qid=? ORDER BY updated_at DESC", (qid,)
                ).fetchall()
            else:
                anns = conn.execute(
                    "SELECT * FROM annotations WHERE qid=? AND annotator=? ORDER BY updated_at DESC",
                    (qid, user_info["username"]),
                ).fetchall()
        data = dict(task)
        data["annotations"] = [annotation_to_json(r) for r in anns]
        data["my_annotation"] = next(
            (a for a in data["annotations"] if a["annotator"] == user), None
        )
        return data

    def save_task(self, data):
        user = data.get("user")
        user_info = self.get_user(user)
        qid = data.get("qid")
        with self.connect() as conn:
            existing = None
            if qid is not None:
                existing = conn.execute("SELECT * FROM tasks WHERE qid=?", (int(qid),)).fetchone()
                if not existing:
                    raise ValueError("task not found")
            if user_info["role"] == "annotator":
                if existing:
                    group = conn.execute("SELECT * FROM task_groups WHERE id=?", (existing["group_id"],)).fetchone()
                    if existing["claimed_by"] not in (None, "", user_info["username"]) and (not group or group["claimed_by"] != user_info["username"]):
                        raise PermissionError("annotators can only edit queries in their own claimed task group")
                    if existing["status"] == "approved":
                        raise PermissionError("approved queries cannot be edited by annotators")
                else:
                    group_id = data.get("group_id")
                    if group_id is None:
                        raise PermissionError("annotators can only add queries inside a claimed task group")
                    group = conn.execute("SELECT * FROM task_groups WHERE id=?", (int(group_id),)).fetchone()
                    if not group or group["claimed_by"] != user_info["username"]:
                        raise PermissionError("annotators can only add queries inside their own claimed task group")
            if qid is None:
                qid = conn.execute("SELECT COALESCE(MAX(qid), -1) + 1 FROM tasks").fetchone()[0]
            group_id = data.get("group_id")
            claimed_by = data.get("claimed_by")
            claimed_at = data.get("claimed_at")
            if existing:
                group_id = existing["group_id"]
                claimed_by = existing["claimed_by"]
                claimed_at = existing["claimed_at"]
            elif user_info["role"] == "annotator":
                claimed_by = user_info["username"]
                claimed_at = now_iso()
            conn.execute(
                """
                INSERT INTO tasks(qid, query, duration, vid, video_path, group_id, split, status, claimed_by, claimed_at, created_at, updated_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(qid) DO UPDATE SET
                    query=excluded.query,
                    duration=excluded.duration,
                    vid=excluded.vid,
                    video_path=excluded.video_path,
                    group_id=excluded.group_id,
                    split=excluded.split,
                    claimed_by=excluded.claimed_by,
                    claimed_at=excluded.claimed_at,
                    updated_at=excluded.updated_at
                """,
                (
                    int(qid),
                    str(data.get("query", "")),
                    float(data.get("duration", 0)),
                    str(data.get("vid", "")),
                    data.get("video_path"),
                    group_id,
                    data.get("split"),
                    data.get("status", "draft" if user_info["role"] == "annotator" else "todo"),
                    claimed_by,
                    claimed_at,
                    now_iso(),
                    now_iso(),
                ),
            )
            if group_id is not None:
                self.refresh_group_status(conn, group_id)
        return self.get_task(int(qid), data.get("user"))

    def delete_task(self, qid, user):
        user_info = self.get_user(user)
        qid = int(qid)
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM tasks WHERE qid=?", (qid,)).fetchone()
            if not row:
                raise ValueError("task not found")
            group_id = row["group_id"]
            if user_info["role"] == "annotator":
                group = conn.execute("SELECT * FROM task_groups WHERE id=?", (group_id,)).fetchone()
                if row["claimed_by"] not in (None, "", user_info["username"]) and (not group or group["claimed_by"] != user_info["username"]):
                    raise PermissionError("annotators can only delete queries in their own claimed task group")
                if row["status"] == "approved":
                    raise PermissionError("approved queries cannot be deleted by annotators")
            remaining = None
            conn.execute("DELETE FROM annotations WHERE qid=?", (qid,))
            conn.execute("DELETE FROM tasks WHERE qid=?", (qid,))
            if group_id is not None:
                remaining = conn.execute(
                    "SELECT COUNT(*) FROM tasks WHERE group_id=?", (group_id,)
                ).fetchone()[0]
                if remaining == 0:
                    conn.execute("DELETE FROM task_groups WHERE id=?", (group_id,))
                else:
                    conn.execute(
                        "UPDATE task_groups SET updated_at=? WHERE id=?",
                        (now_iso(), group_id),
                    )
        return {"deleted": True, "qid": qid, "deleted_empty_group": bool(group_id is not None and remaining == 0)}

    def claim(self, user):
        self.require_role(user, {"annotator"})
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT qid FROM tasks
                WHERE status IN ('todo','draft','rejected')
                  AND (claimed_by IS NULL OR claimed_by='' OR claimed_by=?)
                ORDER BY CASE WHEN claimed_by=? THEN 0 ELSE 1 END, qid
                LIMIT 1
                """,
                (user, user),
            ).fetchone()
            if not row:
                return None
            conn.execute(
                "UPDATE tasks SET claimed_by=?, claimed_at=?, status='draft', updated_at=? WHERE qid=?",
                (user, now_iso(), now_iso(), row["qid"]),
            )
            return self.get_task(row["qid"], user)

    def save_annotation(self, data):
        qid = int(data["qid"])
        editor = str(data.get("user") or "anonymous")
        editor_user = self.get_user(editor)
        annotator = str(data.get("annotator") or editor)
        if editor_user["role"] == "annotator" and annotator != editor:
            raise PermissionError("annotators can only edit their own annotations")
        if editor_user["role"] == "annotator" and str(data.get("status") or "draft") == "approved":
            raise PermissionError("annotators cannot approve annotations")
        status = str(data.get("status") or "draft")
        posted_duration = float(data.get("duration") or 0)
        duration = posted_duration
        with self.connect() as conn:
            existing_annotation = conn.execute(
                "SELECT status FROM annotations WHERE qid=? AND annotator=?",
                (qid, annotator),
            ).fetchone()
            if editor_user["role"] == "annotator" and status == "draft" and existing_annotation:
                if existing_annotation["status"] in {"submitted", "approved"}:
                    return self.get_task(qid, editor)
            task = conn.execute("SELECT duration FROM tasks WHERE qid=?", (qid,)).fetchone()
            if task:
                stored_duration = float(task["duration"] or 0)
                if stored_duration > 0:
                    duration = stored_duration
                elif posted_duration > 0:
                    conn.execute(
                        "UPDATE tasks SET duration=?, updated_at=? WHERE qid=?",
                        (posted_duration, now_iso(), qid),
                    )
            windows = normalize_windows(data.get("windows", []), duration)
            saliency = pack_saliency(data.get("saliency") or [], data.get("clip_scores") or {}, len(windows))
            conn.execute(
                """
                INSERT INTO annotations(qid, annotator, windows_json, saliency_json, notes, status, created_at, updated_at)
                VALUES(?,?,?,?,?,?,?,?)
                ON CONFLICT(qid, annotator) DO UPDATE SET
                    windows_json=excluded.windows_json,
                    saliency_json=excluded.saliency_json,
                    notes=excluded.notes,
                    status=excluded.status,
                    reviewed_by=CASE WHEN excluded.status='approved' THEN annotations.reviewed_by ELSE NULL END,
                    reviewed_at=CASE WHEN excluded.status='approved' THEN annotations.reviewed_at ELSE NULL END,
                    updated_at=excluded.updated_at
                """,
                (
                    qid,
                    annotator,
                    json.dumps(windows, ensure_ascii=False),
                    json.dumps(saliency, ensure_ascii=False),
                    str(data.get("notes", "")),
                    status,
                    now_iso(),
                    now_iso(),
                ),
            )
            task_status = "submitted" if status == "submitted" else "draft"
            if status == "approved":
                task_status = "approved"
            conn.execute(
                "UPDATE tasks SET status=?, updated_at=? WHERE qid=? AND status!='approved'",
                (task_status, now_iso(), qid),
            )
            row = conn.execute("SELECT group_id FROM tasks WHERE qid=?", (qid,)).fetchone()
            if row:
                self.refresh_group_status(conn, row["group_id"])
        return self.get_task(qid, editor)

    def submit_group(self, data):
        user = str(data.get("user") or "anonymous")
        self.require_role(user, {"annotator"})
        group_id = int(data["group_id"])
        qids = [int(qid) for qid in (data.get("qids") or [])]
        with self.connect() as conn:
            group = conn.execute("SELECT * FROM task_groups WHERE id=?", (group_id,)).fetchone()
            if not group:
                raise ValueError("task group not found")
            if group["claimed_by"] != user:
                raise PermissionError("annotators can only submit their own claimed task group")
            if not qids:
                qids = [
                    r["qid"]
                    for r in conn.execute("SELECT qid FROM tasks WHERE group_id=? ORDER BY qid", (group_id,)).fetchall()
                ]
            if not qids:
                raise ValueError("no queries to submit")

            rows = conn.execute(
                f"""
                SELECT qid, windows_json FROM annotations
                WHERE annotator=? AND qid IN ({','.join('?' for _ in qids)})
                """,
                (user, *qids),
            ).fetchall()
            has_any_window = any(json.loads(row["windows_json"] or "[]") for row in rows)
            if not has_any_window:
                raise ValueError("无法提交：当前提交范围内没有任何 query 标注片段")

            submitted = 0
            for qid in qids:
                row = conn.execute(
                    "SELECT id, status FROM annotations WHERE qid=? AND annotator=?",
                    (qid, user),
                ).fetchone()
                if row and row["status"] == "approved":
                    continue
                if row:
                    conn.execute(
                        "UPDATE annotations SET status='submitted', updated_at=? WHERE id=?",
                        (now_iso(), row["id"]),
                    )
                else:
                    conn.execute(
                        """
                        INSERT INTO annotations(qid, annotator, windows_json, saliency_json, notes, status, created_at, updated_at)
                        VALUES(?,?,?,?,?,?,?,?)
                        """,
                        (qid, user, "[]", "[]", "", "submitted", now_iso(), now_iso()),
                    )
                conn.execute(
                    "UPDATE tasks SET status='submitted', updated_at=? WHERE qid=? AND status!='approved'",
                    (now_iso(), qid),
                )
                submitted += 1
            self.refresh_group_status(conn, group_id)
        return {"submitted": submitted, "group": self.get_group(group_id, user)}
    def review(self, data):
        qid = int(data["qid"])
        annotator = str(data.get("annotator", ""))
        reviewer = str(data.get("reviewer") or data.get("user") or "reviewer")
        self.require_role(reviewer, {"reviewer"})
        decision = str(data.get("decision"))
        if decision not in {"approved", "rejected"}:
            raise ValueError("decision must be approved or rejected")
        with self.connect() as conn:
            task = conn.execute("SELECT qid, vid, group_id FROM tasks WHERE qid=?", (qid,)).fetchone()
            if not task:
                raise ValueError("task not found")
            if task["group_id"] is None:
                video_tasks = conn.execute(
                    "SELECT qid FROM tasks WHERE vid=? AND group_id IS NULL ORDER BY qid",
                    (task["vid"],),
                ).fetchall()
            else:
                video_tasks = conn.execute(
                    "SELECT qid FROM tasks WHERE vid=? AND group_id=? ORDER BY qid",
                    (task["vid"], task["group_id"]),
                ).fetchall()
            video_qids = [int(r["qid"]) for r in video_tasks]
            placeholders = ",".join("?" for _ in video_qids)
            params = [decision, reviewer, now_iso(), now_iso(), annotator, *video_qids]
            result = conn.execute(
                f"""
                UPDATE annotations
                SET status=?, reviewed_by=?, reviewed_at=?, updated_at=?
                WHERE annotator=? AND qid IN ({placeholders}) AND status='submitted'
                """,
                params,
            )
            if result.rowcount == 0:
                raise ValueError("no submitted annotations found for this video")
            for item_qid in video_qids:
                approved = conn.execute(
                    "SELECT COUNT(*) FROM annotations WHERE qid=? AND status='approved'", (item_qid,)
                ).fetchone()[0]
                submitted = conn.execute(
                    "SELECT COUNT(*) FROM annotations WHERE qid=? AND status='submitted'", (item_qid,)
                ).fetchone()[0]
                rejected = conn.execute(
                    "SELECT COUNT(*) FROM annotations WHERE qid=? AND status='rejected'", (item_qid,)
                ).fetchone()[0]
                if approved:
                    status = "approved"
                elif submitted:
                    status = "submitted"
                elif rejected:
                    status = "rejected"
                else:
                    status = "draft"
                conn.execute("UPDATE tasks SET status=?, updated_at=? WHERE qid=?", (status, now_iso(), item_qid))
            if task["group_id"] is not None:
                self.refresh_group_status(conn, task["group_id"])
        return self.get_task(qid, reviewer)

    def review_batch(self, data):
        reviewer = str(data.get("reviewer") or data.get("user") or "reviewer")
        self.require_role(reviewer, {"reviewer"})
        decision = str(data.get("decision"))
        if decision not in {"approved", "rejected"}:
            raise ValueError("decision must be approved or rejected")
        group_id = int(data["group_id"])
        vids = [str(v) for v in (data.get("vids") or []) if str(v)]
        if not vids:
            raise ValueError("no videos selected")
        reviewed = 0
        skipped = 0
        with self.connect() as conn:
            group = conn.execute("SELECT * FROM task_groups WHERE id=?", (group_id,)).fetchone()
            if not group:
                raise ValueError("task group not found")
            for vid in vids:
                video_tasks = conn.execute(
                    """
                    SELECT qid FROM tasks
                    WHERE group_id=? AND COALESCE(NULLIF(vid,''), 'qid-' || qid)=?
                    ORDER BY qid
                    """,
                    (group_id, vid),
                ).fetchall()
                video_qids = [int(r["qid"]) for r in video_tasks]
                if not video_qids:
                    skipped += 1
                    continue
                placeholders = ",".join("?" for _ in video_qids)
                result = conn.execute(
                    f"""
                    UPDATE annotations
                    SET status=?, reviewed_by=?, reviewed_at=?, updated_at=?
                    WHERE qid IN ({placeholders}) AND status='submitted'
                    """,
                    [decision, reviewer, now_iso(), now_iso(), *video_qids],
                )
                if result.rowcount == 0:
                    skipped += 1
                    continue
                reviewed += result.rowcount
                for item_qid in video_qids:
                    approved = conn.execute(
                        "SELECT COUNT(*) FROM annotations WHERE qid=? AND status='approved'", (item_qid,)
                    ).fetchone()[0]
                    submitted = conn.execute(
                        "SELECT COUNT(*) FROM annotations WHERE qid=? AND status='submitted'", (item_qid,)
                    ).fetchone()[0]
                    rejected = conn.execute(
                        "SELECT COUNT(*) FROM annotations WHERE qid=? AND status='rejected'", (item_qid,)
                    ).fetchone()[0]
                    if approved:
                        status = "approved"
                    elif submitted:
                        status = "submitted"
                    elif rejected:
                        status = "rejected"
                    else:
                        status = "draft"
                    conn.execute("UPDATE tasks SET status=?, updated_at=? WHERE qid=?", (status, now_iso(), item_qid))
            self.refresh_group_status(conn, group_id)
        return {"reviewed": reviewed, "skipped": skipped, "group": self.get_group(group_id, reviewer)}

    def split(self, train_ratio, val_ratio, seed, source):
        tasks = self.export_rows(source=source, include_test_labels=True, split_filter=None)
        by_video = {}
        for row in tasks:
            by_video.setdefault(video_id_base(row["vid"]), []).append(row["qid"])
        vids = sorted(by_video)
        random.Random(seed).shuffle(vids)
        n = len(vids)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        split_map = {}
        for idx, vid in enumerate(vids):
            split = "train" if idx < n_train else "val" if idx < n_train + n_val else "test"
            for qid in by_video[vid]:
                split_map[qid] = split
        with self.connect() as conn:
            for qid, split in split_map.items():
                conn.execute("UPDATE tasks SET split=?, updated_at=? WHERE qid=?", (split, now_iso(), qid))
        return {
            "videos": len(vids),
            "tasks": len(split_map),
            "train_videos": n_train,
            "val_videos": n_val,
            "test_videos": max(0, n - n_train - n_val),
        }

    def export_rows(self, source="approved", include_test_labels=False, split_filter=None, group_filter=None, clip_len=2.0):
        where = []
        values = []
        if split_filter:
            where.append("COALESCE(t.split,'')=?")
            values.append(split_filter)
        if group_filter is not None:
            where.append("t.group_id=?")
            values.append(int(group_filter))
        if source == "approved":
            where.append("EXISTS (SELECT 1 FROM annotations a WHERE a.qid=t.qid AND a.status='approved')")
        sql_where = "WHERE " + " AND ".join(where) if where else ""
        with self.connect() as conn:
            tasks = conn.execute(f"SELECT * FROM tasks t {sql_where} ORDER BY qid", values).fetchall()
            rows = []
            for task in tasks:
                row = {
                    "qid": task["qid"],
                    "query": task["query"],
                    "duration": int(task["duration"]) if float(task["duration"]).is_integer() else task["duration"],
                    "vid": task["vid"],
                }
                is_test = task["split"] == "test"
                ann = conn.execute(
                    """
                    SELECT * FROM annotations
                    WHERE qid=? AND status='approved'
                    ORDER BY reviewed_at DESC, updated_at DESC
                    LIMIT 1
                    """,
                    (task["qid"],),
                ).fetchone()
                if ann and (include_test_labels or not is_test):
                    windows = json.loads(ann["windows_json"])
                    clip_ids = clip_ids_for_windows(windows, task["duration"], clip_len)
                    saliency = qv_saliency_from_annotation(
                        windows,
                        json.loads(ann["saliency_json"] or "[]"),
                        clip_ids,
                    )
                    row["relevant_clip_ids"] = clip_ids
                    row["saliency_scores"] = saliency
                    row["relevant_windows"] = windows
                rows.append(row)
            return rows

    def export_group_files(self, output_dir, source="approved", group_id=None, clip_len=2.0):
        output_dir = Path(output_dir)
        with self.connect() as conn:
            if group_id in (None, "", "null"):
                groups = conn.execute("SELECT * FROM task_groups ORDER BY id").fetchall()
            else:
                groups = conn.execute("SELECT * FROM task_groups WHERE id=?", (int(group_id),)).fetchall()
        summary = {}
        for group in groups:
            rows = self.export_rows(
                source=source,
                include_test_labels=True,
                group_filter=group["id"],
                clip_len=clip_len,
            )
            safe_name = re.sub(r'[\\/:*?"<>|]+', "_", str(group["name"] or f"group_{group['id']}"))
            safe_name = safe_name.strip(" ._") or f"group_{group['id']}"
            path = output_dir / f"group_{group['id']}_{safe_name}_release.jsonl"
            write_jsonl(path, rows)
            summary[str(group["id"])] = {"name": group["name"], "path": str(path), "rows": len(rows)}
        return summary

    def export_files(self, output_dir, source="approved", strip_test_labels=True, clip_len=2.0):
        output_dir = Path(output_dir)
        summary = {}
        for split in ["train", "val", "test"]:
            rows = self.export_rows(
                source=source,
                include_test_labels=not (strip_test_labels and split == "test"),
                split_filter=split,
                clip_len=clip_len,
            )
            path = output_dir / f"highlight_{split}_release.jsonl"
            write_jsonl(path, rows)
            summary[split] = {"path": str(path), "rows": len(rows)}
        return summary

def annotation_to_json(row):
    data = dict(row)
    data["windows"] = json.loads(data.pop("windows_json") or "[]")
    saliency = unpack_saliency(data.pop("saliency_json") or "[]", len(data["windows"]))
    data["saliency"] = saliency["window_scores"]
    data["clip_scores"] = saliency["clip_scores"]
    return data


def find_video_path(video_root, vid):
    root = Path(video_root)
    candidates = [str(vid), video_id_base(vid)]
    for stem in candidates:
        for ext in VIDEO_EXTS:
            path = root / f"{stem}{ext}"
            if path.exists():
                return str(path)
    return None


class AppHandler(BaseHTTPRequestHandler):
    store = None
    args = None

    def log_message(self, fmt, *args):
        print("%s - %s" % (self.address_string(), fmt % args))

    def send_json(self, data, status=HTTPStatus.OK):
        body = dumps(data)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_html(self):
        body = INDEX_HTML.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        try:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self.send_html()
            elif parsed.path == "/api/stats":
                self.send_json(self.store.stats())
            elif parsed.path == "/api/users":
                self.send_json({"users": self.store.list_users()})
            elif parsed.path == "/api/tasks":
                qs = {k: v[0] for k, v in parse_qs(parsed.query).items()}
                self.send_json({"tasks": self.store.list_tasks(qs)})
            elif parsed.path == "/api/groups":
                qs = {k: v[0] for k, v in parse_qs(parsed.query).items()}
                self.send_json({"groups": self.store.list_groups(qs)})
            elif parsed.path.startswith("/api/group/"):
                group_id = int(parsed.path.rsplit("/", 1)[1])
                qs = {k: v[0] for k, v in parse_qs(parsed.query).items()}
                group = self.store.get_group(group_id, qs.get("user"))
                self.send_json({"group": group} if group else {"error": "not found"}, HTTPStatus.OK if group else HTTPStatus.NOT_FOUND)
            elif parsed.path.startswith("/api/task/"):
                qid = int(parsed.path.rsplit("/", 1)[1])
                qs = {k: v[0] for k, v in parse_qs(parsed.query).items()}
                task = self.store.get_task(qid, qs.get("user"))
                self.send_json({"task": task} if task else {"error": "not found"}, HTTPStatus.OK if task else HTTPStatus.NOT_FOUND)
            elif parsed.path.startswith("/media/"):
                qid = int(parsed.path.rsplit("/", 1)[1])
                task = self.store.get_task(qid)
                if not task or not task.get("video_path"):
                    self.send_error(HTTPStatus.NOT_FOUND, "video path not set")
                    return
                self.send_file(Path(task["video_path"]))
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except CLIENT_DISCONNECT_ERRORS:
            return
        except Exception as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self):
        try:
            parsed = urlparse(self.path)
            if parsed.path == "/api/upload_videos":
                self.handle_upload_videos()
                return
            data = parse_json_body(self)
            if parsed.path == "/api/register":
                self.send_json({"user": self.store.register_user(data.get("username"), data.get("password"), data.get("role"), data.get("display_name"))})
            elif parsed.path == "/api/login":
                self.send_json({"user": self.store.login_user(data.get("username"), data.get("password"))})
            elif parsed.path == "/api/user":
                self.send_json({"user": self.store.set_user(data.get("username"), data.get("role"), data.get("display_name"))})
            elif parsed.path == "/api/import_jsonl":
                self.store.require_role(data.get("user"), {"reviewer"})
                result = self.store.import_jsonl(
                    data["path"], data.get("split"), data.get("video_root")
                )
                self.send_json(result)
            elif parsed.path == "/api/import_jsonl_content":
                self.store.require_role(data.get("user"), {"reviewer"})
                result = self.store.import_jsonl_text(
                    data["content"], data.get("split"), data.get("video_root"), data.get("filename")
                )
                self.send_json(result)
            elif parsed.path == "/api/import_videos":
                self.store.require_role(data.get("user"), {"reviewer"})
                self.send_json(self.store.import_videos(data["path"]))
            elif parsed.path == "/api/task":
                self.send_json({"task": self.store.save_task(data)})
            elif parsed.path == "/api/delete_task":
                self.send_json(self.store.delete_task(data.get("qid"), data.get("user")))
            elif parsed.path == "/api/claim":
                task = self.store.claim(str(data.get("user") or "anonymous"))
                self.send_json({"task": task})
            elif parsed.path == "/api/publish_groups":
                self.send_json(self.store.publish_video_groups(data))
            elif parsed.path == "/api/reset_workspace":
                self.send_json(self.store.reset_workspace(data.get("user"), bool(data.get("keep_users", True))))
            elif parsed.path == "/api/delete_group":
                self.send_json(self.store.delete_group(data.get("group_id"), data.get("user")))
            elif parsed.path == "/api/claim_group":
                self.send_json({"group": self.store.claim_group(data.get("group_id"), str(data.get("user") or "anonymous"))})
            elif parsed.path == "/api/annotation":
                self.send_json({"task": self.store.save_annotation(data)})
            elif parsed.path == "/api/submit_group":
                self.send_json(self.store.submit_group(data))
            elif parsed.path == "/api/review":
                self.send_json({"task": self.store.review(data)})
            elif parsed.path == "/api/review_batch":
                self.send_json(self.store.review_batch(data))
            elif parsed.path == "/api/split":
                self.store.require_role(data.get("user"), {"reviewer"})
                self.send_json(
                    self.store.split(
                        float(data.get("train_ratio", 0.8)),
                        float(data.get("val_ratio", 0.1)),
                        int(data.get("seed", 2024)),
                        str(data.get("source", "all")),
                    )
                )
            elif parsed.path == "/api/export_group":
                self.store.require_role(data.get("user"), {"reviewer"})
                self.send_json(
                    self.store.export_group_files(
                        data.get("output_dir") or str(ROOT / "annotation_workspace" / "exports"),
                        source=str(data.get("source", "approved")),
                        group_id=data.get("group_id"),
                        clip_len=float(data.get("clip_len", 2.0)),
                    )
                )
            elif parsed.path == "/api/export":
                self.store.require_role(data.get("user"), {"reviewer"})
                self.send_json(
                    self.store.export_files(
                        data.get("output_dir") or str(ROOT / "annotation_workspace" / "exports"),
                        source=str(data.get("source", "approved")),
                        strip_test_labels=bool(data.get("strip_test_labels", True)),
                        clip_len=float(data.get("clip_len", 2.0)),
                    )
                )
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def handle_upload_videos(self):
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": self.headers.get("Content-Type"),
                "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
            },
        )
        upload_dir = DEFAULT_UPLOAD_DIR
        upload_dir.mkdir(parents=True, exist_ok=True)
        qid = form.getfirst("qid")
        user = form.getfirst("user")
        self.store.require_role(user, {"reviewer"})
        qid = int(qid) if qid not in (None, "", "null") else None
        fields = form["files"] if "files" in form else []
        if not isinstance(fields, list):
            fields = [fields]
        saved = []
        for field in fields:
            if not getattr(field, "filename", None):
                continue
            filename = Path(field.filename).name
            suffix = Path(filename).suffix.lower()
            if suffix not in VIDEO_EXTS:
                continue
            target = upload_dir / filename
            counter = 1
            while target.exists():
                target = upload_dir / f"{Path(filename).stem}_{counter}{suffix}"
                counter += 1
            with target.open("wb") as f:
                while True:
                    chunk = field.file.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
            saved.append(
                self.store.attach_uploaded_video(
                    target,
                    qid=qid if len(fields) == 1 else None,
                    match_stem=Path(filename).stem,
                )
            )
        self.send_json({"saved": saved, "count": len(saved)})

    def send_file(self, path):
        path = Path(path)
        if not path.exists() or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        size = path.stat().st_size
        start = 0
        end = size - 1
        status = HTTPStatus.OK
        range_header = self.headers.get("Range")
        if range_header and range_header.startswith("bytes="):
            status = HTTPStatus.PARTIAL_CONTENT
            range_value = range_header.replace("bytes=", "", 1)
            first, _, last = range_value.partition("-")
            start = int(first) if first else 0
            end = int(last) if last else size - 1
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with path.open("rb") as f:
            f.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = f.read(min(1024 * 512, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except CLIENT_DISCONNECT_ERRORS:
                    break
                remaining -= len(chunk)


INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>QV-M2 视频标注工具</title>
  <style>
    :root { color-scheme: light; --ink:#18202a; --muted:#627080; --line:#d9e0e7; --panel:#f7f9fb; --accent:#0f766e; --warn:#b45309; --bad:#b91c1c; }
    * { box-sizing: border-box; }
    body { margin:0; font-family: "Segoe UI", Arial, sans-serif; color:var(--ink); background:#fff; }
    header { display:flex; align-items:center; gap:16px; padding:14px 18px; border-bottom:1px solid var(--line); background:#fff; position:sticky; top:0; z-index:2; }
    h1 { font-size:18px; margin:0; white-space:nowrap; }
    input, select, button, textarea { font:inherit; }
    input, select, textarea { border:1px solid var(--line); border-radius:6px; padding:8px 10px; background:#fff; }
    button { border:1px solid var(--line); border-radius:6px; padding:8px 11px; background:#fff; cursor:pointer; }
    button.primary { background:var(--accent); color:#fff; border-color:var(--accent); }
    button.warn { color:var(--warn); }
    button.bad { color:var(--bad); }
    main { display:grid; grid-template-columns: 360px minmax(0,1fr); min-height:calc(100vh - 58px); }
    aside { border-right:1px solid var(--line); background:var(--panel); padding:12px; overflow:auto; }
    section { padding:14px 18px; min-width:0; }
    .row { display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
    .stack { display:flex; flex-direction:column; gap:10px; }
    .box { border:1px solid var(--line); border-radius:8px; padding:10px; background:#fff; }
    .task { padding:9px; border:1px solid var(--line); border-radius:7px; margin-bottom:8px; background:#fff; cursor:pointer; }
    .task.active { border-color:var(--accent); box-shadow:0 0 0 2px rgba(15,118,110,.12); }
    .task small { color:var(--muted); display:block; margin-top:4px; }
    .query { width:100%; min-height:72px; }
    video { width:100%; max-height:52vh; background:#111; border-radius:8px; }
    .timeline { position:relative; height:42px; border:1px solid var(--line); border-radius:6px; background:#edf2f6; overflow:hidden; cursor:crosshair; }
    .seg { position:absolute; top:6px; height:28px; background:rgba(15,118,110,.65); border:1px solid #075f59; border-radius:4px; }
    .playhead { position:absolute; top:0; bottom:0; width:2px; background:#dc2626; }
    table { width:100%; border-collapse:collapse; }
    td, th { border-bottom:1px solid var(--line); padding:7px; text-align:left; vertical-align:top; }
    th { color:var(--muted); font-weight:600; }
    .muted { color:var(--muted); }
    .grow { flex:1; min-width:160px; }
    .pill { display:inline-block; border:1px solid var(--line); border-radius:999px; padding:2px 8px; font-size:12px; color:var(--muted); background:#fff; }
    .group-grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap:8px; }
    .group-video { border:1px solid var(--line); border-radius:7px; padding:8px; background:#fff; cursor:pointer; }
    .group-video.active { border-color:var(--accent); box-shadow:0 0 0 2px rgba(15,118,110,.12); }
    .clip-grid { display:grid; grid-template-columns: repeat(auto-fill, minmax(118px, 1fr)); gap:6px; margin-top:8px; }
    .clip-cell { border:1px solid var(--line); border-radius:6px; padding:6px; background:#f8fafc; }
    .clip-cell input { width:64px; padding:4px 6px; }
    @media (max-width: 900px) { main { grid-template-columns:1fr; } aside { border-right:0; border-bottom:1px solid var(--line); max-height:42vh; } }
  </style>
</head>
<body>
<header>
  <h1>QV-M2 视频标注</h1>
  <input id="user" value="annotator1" title="用户名">
  <select id="role" title="角色"><option value="annotator">标注员</option><option value="reviewer">审核员</option></select>
  <button onclick="loginUser()">登录/创建账号</button>
  <span id="roleBadge" class="pill">annotator</span>
  <button data-annotator-only onclick="claimTask()">领取任务</button>
  <button onclick="refresh()">刷新</button>
  <span id="stats" class="muted"></span>
</header>
<main>
  <aside class="stack">
    <div class="box stack">
      <div class="row">
        <select id="status" onchange="refresh()">
          <option value="all">全部状态</option><option value="todo">todo</option><option value="draft">draft</option>
          <option value="submitted">submitted</option><option value="approved">approved</option><option value="rejected">rejected</option>
        </select>
        <select id="split" onchange="refresh()">
          <option value="all">全部集合</option><option value="train">train</option><option value="val">val</option><option value="test">test</option>
        </select>
      </div>
      <input id="search" placeholder="搜索 qid / vid / query" onkeydown="if(event.key==='Enter') refresh()">
    </div>
    <div class="box stack" data-reviewer-only>
      <b>导入</b>
      <input id="jsonlFile" type="file" accept=".jsonl,.json,.txt" onchange="importJsonlFile()">
      <input id="jsonlPath" value="data/highlight_test_release.jsonl">
      <div class="row"><input id="jsonlSplit" placeholder="split，可空" class="grow"><button onclick="importJsonl()">导入 JSONL</button></div>
      <div class="row">
        <input id="videoFiles" type="file" accept="video/*,.mp4,.avi,.mov,.mkv,.webm,.m4v" multiple onchange="selectVideoFiles()" style="display:none">
        <button type="button" onclick="$('videoFiles').click()">选择视频文件</button>
        <input id="videoDirFiles" type="file" webkitdirectory directory multiple onchange="selectVideoDirectory()" style="display:none">
        <button type="button" onclick="$('videoDirFiles').click()">选择视频目录</button>
      </div>
      <div id="selectedVideoSummary" class="muted">未选择视频</div>
      <button id="confirmVideosBtn" onclick="confirmSelectedVideos()" disabled>确认视频</button>
      <input id="videoRoot" placeholder="服务器本机目录路径，例如 D:\videos">
      <button onclick="importVideos()">扫描视频目录</button>
      <label class="stack">
        <span class="muted">每个任务的视频数</span>
        <input id="groupSize" type="number" min="1" value="5">
      </label>
      <label class="stack">
        <span class="muted">任务名前缀/起始名</span>
        <input id="groupPrefix" value="标注任务1">
      </label>
      <button id="publishGroupsBtn" class="primary" onclick="publishGroups()" disabled>按视频数发布任务</button>
      <button class="bad" onclick="resetWorkspace()">清空当前扫描视频</button>
    </div>
    <div class="box stack" data-reviewer-only>
      <b>划分与导出</b>
      <div class="row"><input id="trainRatio" value="0.8" class="grow"><input id="valRatio" value="0.1" class="grow"><input id="seed" value="2024" class="grow"></div>
      <button onclick="splitData()">按视频比例划分</button>
      <input id="outDir" value="annotation_workspace/exports">
      <select id="exportSource"><option value="all">全部任务</option><option value="approved">仅审核通过</option></select>
      <button class="primary" onclick="exportData()">导出 release JSONL</button>
      <select id="exportGroup"><option value="">全部任务包</option></select>
      <button class="primary" onclick="exportByGroup()">按任务包导出 JSONL</button>
    </div>
    <div class="box stack">
      <div class="row"><b>任务包</b><button onclick="refreshGroups()">刷新任务包</button></div>
      <div id="groups"></div>
    </div>
    <details data-reviewer-only>
      <summary class="muted">历史单条任务</summary>
      <div id="tasks"></div>
    </details>
  </aside>
  <section class="stack">
    <div id="empty" class="box muted">请选择或领取一个任务。</div>
    <div id="groupPanel" class="box stack" style="display:none"></div>
    <div id="editor" class="stack" style="display:none">
      <div class="box stack">
        <div class="row">
          <b id="title"></b><span id="taskMeta" class="pill"></span>
          <button data-reviewer-only onclick="saveTask()">保存任务信息</button>
          <button data-reviewer-only class="bad" onclick="deleteCurrentTask()">删除任务</button>
          <button onclick="prevGroupVideo()">上一个视频</button>
          <button onclick="nextGroupVideo()">下一个视频</button>
        </div>
        <textarea id="query" class="query" placeholder="query"></textarea>
        <div class="row">
          <input id="vid" class="grow" placeholder="vid">
          <input id="duration" type="number" step="0.001" placeholder="duration">
          <input id="videoPath" class="grow" placeholder="本地视频路径">
          <label data-reviewer-only><input id="bindVideoFile" type="file" accept="video/*,.mp4,.avi,.mov,.mkv,.webm,.m4v" style="display:none" onchange="uploadCurrentVideo()"> <button type="button" onclick="$('bindVideoFile').click()">选择视频</button></label>
        </div>
      </div>
      <video id="video" controls></video>
      <div class="timeline" id="timeline" onmousedown="timelineDown(event)"></div>
      <div class="row">
        <button onclick="markStart()">设为开始</button>
        <button onclick="markEnd()">设为结束并添加</button>
        <input id="segStart" type="number" step="0.001" placeholder="start">
        <input id="segEnd" type="number" step="0.001" placeholder="end">
        <button onclick="addWindow()">添加片段</button>
      </div>
      <div class="box stack">
        <b>片段与显著性</b>
        <table><thead><tr><th>start</th><th>end</th><th>我的 score</th><th></th></tr></thead><tbody id="windows"></tbody></table>
        <textarea id="notes" placeholder="备注"></textarea>
        <div class="row">
          <button onclick="saveAnnotation('draft')">保存草稿</button>
          <button class="primary" onclick="saveAnnotation('submitted')">提交审核</button>
        </div>
      </div>
      <div class="box stack">
        <b>审核</b>
        <div id="annotations"></div>
      </div>
    </div>
  </section>
</main>
<script>
let tasks = [], groups = [], current = null, currentGroup = null, groupTasks = [], windows = [], windowScores = [], clipScores = {}, expandedWindow = null, dragStart = null, currentUser = null, activeAnnotator = null;
let pendingVideoFiles = [], confirmedVideoCount = 0;
const $ = id => document.getElementById(id);
async function api(path, opts={}) {
  const res = await fetch(path, {headers:{'Content-Type':'application/json'}, ...opts});
  const data = await res.json();
  if (data.error) throw new Error(data.error);
  return data;
}
function isReviewer(){ return currentUser && currentUser.role === 'reviewer'; }
async function loginUser() {
  const data = await api('/api/user', {method:'POST', body:JSON.stringify({username:$('user').value, role:$('role').value})});
  currentUser = data.user;
  $('role').value = currentUser.role;
  $('roleBadge').textContent = currentUser.role === 'reviewer' ? '审核员' : '标注员';
  applyRoleUi();
  refresh();
}
function applyRoleUi(){
  document.querySelectorAll('[data-reviewer-only]').forEach(el => el.style.display = isReviewer() ? '' : 'none');
  document.querySelectorAll('[data-annotator-only]').forEach(el => el.style.display = isReviewer() ? 'none' : '');
  updatePublishState();
}
function updatePublishState() {
  const confirmBtn = $('confirmVideosBtn');
  const publishBtn = $('publishGroupsBtn');
  if (confirmBtn) confirmBtn.disabled = pendingVideoFiles.length === 0;
  if (publishBtn) publishBtn.disabled = confirmedVideoCount <= 0;
}
async function refresh() {
  const params = new URLSearchParams({user:$('user').value, status:$('status').value, split:$('split').value, q:$('search').value});
  const [list, groupList, stats] = await Promise.all([api('/api/tasks?' + params), api('/api/groups?user=' + encodeURIComponent($('user').value)), api('/api/stats')]);
  tasks = list.tasks; renderTasks(); $('stats').textContent = `任务 ${stats.total_tasks} | ` + JSON.stringify(stats.tasks);
  groups = groupList.groups; renderGroups();
  updateNextGroupName();
  applyRoleUi();
}
async function refreshGroups() {
  const data = await api('/api/groups?user=' + encodeURIComponent($('user').value));
  groups = data.groups; renderGroups();
}
function renderGroups() {
  $('groups').innerHTML = groups.map(g => {
    const mine = g.claimed_by === $('user').value;
    const free = !g.claimed_by;
    const action = free && !isReviewer()
      ? `<button onclick="claimGroup(${g.id})">接取</button>`
      : `<button onclick="openGroup(${g.id})">打开</button>`;
    return `<div class="task">
      <b>#G${g.id} ${escapeHtml(g.name)}</b> <span class="pill">${g.status}</span>
      <span class="pill">${g.video_count || 0} 个视频</span>
      <small>${g.claimed_by ? '已被 ' + escapeHtml(g.claimed_by) + ' 接取' : '未被接取'}${mine ? ' · 我的任务' : ''}</small>
      ${action}</div>`;
  }).join('');
  updateNextGroupName();
}
function updateNextGroupName() {
  const input = $('groupPrefix');
  if (!input || document.activeElement === input) return;
  const names = (groups || []).map(g => g.name || '').filter(Boolean);
  let bestBase = input.value.replace(/\d+$/, '') || 'part';
  let bestNum = 0;
  for (const name of names) {
    const m = String(name).match(/^(.*?)(\d+)$/);
    if (!m) continue;
    const n = Number(m[2]);
    if (n >= bestNum) { bestBase = m[1] || bestBase; bestNum = n; }
  }
  if (bestNum > 0) input.value = `${bestBase}${bestNum + 1}`;
}
function renderTasks() {
  $('tasks').innerHTML = tasks.map(t => `<div class="task ${current&&current.qid===t.qid?'active':''}" onclick="loadTask(${t.qid})">
    <b>#${t.qid}</b> <span class="pill">${t.status}</span> <span class="pill">${t.split||'-'}</span>
    <small>${escapeHtml(t.vid)} · ${t.duration}s · 标注 ${t.ann_count}</small>
    <div>${escapeHtml((t.query||'').slice(0,110))}</div></div>`).join('');
}
async function loadTask(qid) {
  const data = await api(`/api/task/${qid}?user=${encodeURIComponent($('user').value)}`);
  current = data.task;
  activeAnnotator = isReviewer() ? null : $('user').value;
  windows = (current.my_annotation && current.my_annotation.windows) || [];
  windowScores = (current.my_annotation && current.my_annotation.saliency) || windows.map(() => 4);
  clipScores = (current.my_annotation && current.my_annotation.clip_scores) || {};
  expandedWindow = null;
  $('empty').style.display='none'; $('editor').style.display='flex';
  $('groupPanel').style.display = currentGroup ? 'flex' : 'none';
  $('title').textContent = `任务 #${current.qid}`;
  $('taskMeta').textContent = `${current.status} / ${current.split || '-'}`;
  $('query').value = current.query || ''; $('vid').value = current.vid || ''; $('duration').value = current.duration || '';
  $('videoPath').value = current.video_path || ''; $('notes').value = current.my_annotation ? current.my_annotation.notes : '';
  $('video').src = current.video_path ? `/media/${current.qid}` : '';
  renderWindows(); renderAnnotations(); renderTasks(); renderGroupPanel();
  applyRoleUi();
}
async function openGroup(groupId) {
  const data = await api(`/api/group/${groupId}?user=${encodeURIComponent($('user').value)}`);
  currentGroup = data.group;
  groupTasks = currentGroup.tasks || [];
  renderGroupPanel();
  if (!groupTasks.length) return alert('这个任务包没有视频');
  await loadTask(groupTasks[0].qid);
}
async function claimGroup(groupId) {
  const data = await api('/api/claim_group', {method:'POST', body:JSON.stringify({user:$('user').value, group_id:groupId})});
  currentGroup = data.group;
  groupTasks = currentGroup.tasks || [];
  renderGroupPanel();
  await refreshGroups();
  if (groupTasks.length) await loadTask(groupTasks[0].qid);
}
async function publishGroups() {
  if (confirmedVideoCount <= 0) return alert('请先选择并确认视频');
  const result = await api('/api/publish_groups', {
    method:'POST',
    body:JSON.stringify({user:$('user').value, group_size:Number($('groupSize').value), name_prefix:$('groupPrefix').value})
  });
  confirmedVideoCount = 0;
  $('selectedVideoSummary').textContent = '任务已发布，请继续选择新视频';
  updatePublishState();
  alert(JSON.stringify(result)); refresh();
}
async function resetWorkspace() {
  if (!confirm('确定清空旧任务、旧任务包和旧标注？账号会保留。')) return;
  const result = await api('/api/reset_workspace', {
    method:'POST',
    body:JSON.stringify({user:$('user').value, keep_users:true})
  });
  current = null; currentGroup = null; groupTasks = []; windows = []; windowScores = []; tasks = []; groups = [];
  $('editor').style.display = 'none';
  $('empty').style.display = 'block';
  $('tasks').innerHTML = '';
  $('groups').innerHTML = '';
  alert(JSON.stringify(result));
  refresh();
}
function currentGroupIndex(){
  if (!current || !groupTasks.length) return -1;
  return groupTasks.findIndex(t => t.qid === current.qid);
}
async function prevGroupVideo(){
  const idx = currentGroupIndex();
  if (idx > 0) await loadTask(groupTasks[idx - 1].qid);
}
async function nextGroupVideo(){
  const idx = currentGroupIndex();
  if (idx >= 0 && idx < groupTasks.length - 1) await loadTask(groupTasks[idx + 1].qid);
}
async function claimTask() { const data = await api('/api/claim', {method:'POST', body:JSON.stringify({user:$('user').value})}); if (data.task) { current=data.task; await loadTask(current.qid); } else alert('没有可领取任务'); }
async function importJsonl() { alert(JSON.stringify(await api('/api/import_jsonl', {method:'POST', body:JSON.stringify({user:$('user').value, path:$('jsonlPath').value, split:$('jsonlSplit').value || null, video_root:$('videoRoot').value || null})}))); refresh(); }
async function importJsonlFile() {
  const file = $('jsonlFile').files[0]; if (!file) return;
  const content = await file.text();
  const result = await api('/api/import_jsonl_content', {method:'POST', body:JSON.stringify({user:$('user').value, content, filename:file.name, split:$('jsonlSplit').value || null, video_root:$('videoRoot').value || null})});
  alert(JSON.stringify(result)); $('jsonlFile').value = ''; refresh();
}
async function importVideos() {
  const result = await api('/api/import_videos', {method:'POST', body:JSON.stringify({user:$('user').value, path:$('videoRoot').value})});
  confirmedVideoCount = result.added || 0;
  $('selectedVideoSummary').textContent = confirmedVideoCount
    ? `已确认 ${confirmedVideoCount} 个视频，可以发布任务`
    : '没有发现新的可发布视频';
  updatePublishState();
  alert(JSON.stringify(result)); refresh();
}
function setPendingVideos(files, label) {
  pendingVideoFiles = files;
  confirmedVideoCount = 0;
  $('selectedVideoSummary').textContent = files.length
    ? `${label}：包含 ${files.length} 个视频，点击“确认视频”后上传`
    : '未选择视频';
  updatePublishState();
}
function selectVideoFiles() {
  const files = Array.from($('videoFiles').files); if (!files.length) return;
  setPendingVideos(files, '已选择视频文件');
}
function selectVideoDirectory() {
  const files = Array.from($('videoDirFiles').files)
    .filter(f => /\.(mp4|avi|mov|mkv|webm|m4v)$/i.test(f.name));
  if (!files.length) { $('videoDirFiles').value = ''; return alert('目录中没有支持的视频文件'); }
  setPendingVideos(files, '已选择视频目录');
}
async function confirmSelectedVideos() {
  if (!pendingVideoFiles.length) return;
  await uploadVideoFileList(pendingVideoFiles, null);
}
async function uploadVideoFileList(files, inputId) {
  const fd = new FormData(); fd.append('user', $('user').value); files.forEach(f => fd.append('files', f));
  const res = await fetch('/api/upload_videos', {method:'POST', body:fd});
  const data = await res.json(); if (data.error) throw new Error(data.error);
  confirmedVideoCount = data.count || 0;
  pendingVideoFiles = [];
  if (inputId) $(inputId).value = '';
  $('videoFiles').value = '';
  $('videoDirFiles').value = '';
  $('selectedVideoSummary').textContent = `已确认 ${confirmedVideoCount} 个视频，可以发布任务`;
  updatePublishState();
  alert(JSON.stringify(data)); refresh();
}
async function uploadCurrentVideo() {
  if (!current) return alert('请先选择任务');
  const file = $('bindVideoFile').files[0]; if (!file) return;
  const fd = new FormData(); fd.append('user', $('user').value); fd.append('qid', current.qid); fd.append('files', file);
  const res = await fetch('/api/upload_videos', {method:'POST', body:fd});
  const data = await res.json(); if (data.error) throw new Error(data.error);
  $('bindVideoFile').value = ''; await loadTask(current.qid); refresh();
}
async function saveTask() {
  const body = {qid: current ? current.qid : null, query:$('query').value, vid:$('vid').value, duration:Number($('duration').value), video_path:$('videoPath').value, user:$('user').value};
  const data = await api('/api/task', {method:'POST', body:JSON.stringify(body)}); current = data.task; await loadTask(current.qid); refresh();
}
async function deleteCurrentTask() {
  if (!current) return;
  if (!confirm(`确定删除任务 #${current.qid}？相关标注也会删除。`)) return;
  const result = await api('/api/delete_task', {method:'POST', body:JSON.stringify({user:$('user').value, qid:current.qid})});
  current = null; windows = []; windowScores = [];
  $('editor').style.display = 'none';
  $('empty').style.display = 'block';
  if (currentGroup) {
    const data = await api(`/api/group/${currentGroup.id}?user=${encodeURIComponent($('user').value)}`);
    currentGroup = data.group;
    groupTasks = currentGroup ? (currentGroup.tasks || []) : [];
  }
  alert(JSON.stringify(result));
  refresh();
}
function markStart(){ $('segStart').value = $('video').currentTime.toFixed(3); }
function markEnd(){ $('segEnd').value = $('video').currentTime.toFixed(3); addWindow(); }
function addWindow(){
  let s = Number($('segStart').value), e = Number($('segEnd').value);
  if (!(e > s)) return alert('end 必须大于 start');
  windows.push([s,e]); windowScores.push(4);
  const paired = windows.map((w, i) => ({w, score: windowScores[i] ?? 4})).sort((a,b)=>a.w[0]-b.w[0]);
  windows = paired.map(x => x.w); windowScores = paired.map(x => x.score);
  renderWindows();
}
function renderWindows(){
  $('windows').innerHTML = windows.map((w,i)=>`<tr><td>${w[0]}</td><td>${w[1]}</td>
    <td><input type="number" min="0" max="4" value="${windowScores[i] ?? 4}" data-score="${i}"></td>
    <td><button class="bad" onclick="windows.splice(${i},1);windowScores.splice(${i},1);renderWindows()">删除</button></td></tr>`).join('');
  renderTimeline();
}
function renderTimeline(){
  const tl = $('timeline'), dur = Number($('duration').value || $('video').duration || 1);
  tl.innerHTML = windows.map(w => `<div class="seg" style="left:${100*w[0]/dur}%;width:${100*(w[1]-w[0])/dur}%"></div>`).join('') + '<div id="playhead" class="playhead"></div>';
}
function timelineDown(ev){
  const rect = $('timeline').getBoundingClientRect(), dur = Number($('duration').value || $('video').duration || 1);
  dragStart = Math.max(0, Math.min(dur, (ev.clientX-rect.left)/rect.width*dur));
  document.onmouseup = e => {
    const end = Math.max(0, Math.min(dur, (e.clientX-rect.left)/rect.width*dur));
    $('segStart').value = Math.min(dragStart,end).toFixed(3); $('segEnd').value = Math.max(dragStart,end).toFixed(3);
    document.onmouseup = null;
  };
}
$('video').ontimeupdate = () => { const ph=$('playhead'); if(ph){ const dur=Number($('duration').value || $('video').duration || 1); ph.style.left=(100*$('video').currentTime/dur)+'%'; } };
async function saveAnnotation(status){
  const saliency = windows.map((_,i)=>Number(document.querySelector(`[data-score="${i}"]`).value || 4));
  const annotator = activeAnnotator || $('user').value;
  const body = {qid:current.qid, user:$('user').value, annotator, duration:Number($('duration').value), windows, saliency, notes:$('notes').value, status};
  const data = await api('/api/annotation', {method:'POST', body:JSON.stringify(body)}); current=data.task; renderAnnotations(); refresh();
}
function renderAnnotations(){
  $('annotations').innerHTML = (current.annotations||[]).map(a => `<div class="box">
    <div><b>${escapeHtml(a.annotator)}</b> <span class="pill">${a.status}</span> <span class="muted">${a.updated_at}</span></div>
    <div>窗口: ${escapeHtml(JSON.stringify(a.windows))}</div><div class="muted">${escapeHtml(a.notes||'')}</div>
    <button data-reviewer-only onclick="loadCorrection('${escapeJs(a.annotator)}')">载入修正</button>
    <button data-reviewer-only class="primary" onclick="review('${escapeJs(a.annotator)}','approved')">通过</button>
    <button data-reviewer-only class="warn" onclick="review('${escapeJs(a.annotator)}','rejected')">驳回</button></div>`).join('');
  applyRoleUi();
}
function loadCorrection(annotator){
  const ann = (current.annotations || []).find(a => a.annotator === annotator);
  if (!ann) return;
  activeAnnotator = annotator;
  windows = JSON.parse(JSON.stringify(ann.windows || []));
  windowScores = (ann.saliency || []).map(x => Array.isArray(x) ? Math.round(x.reduce((a,b)=>a+Number(b||0),0)/x.length) : Number(x || 4));
  while (windowScores.length < windows.length) windowScores.push(4);
  $('notes').value = ann.notes || '';
  renderWindows();
}
async function review(annotator, decision){ const data=await api('/api/review',{method:'POST',body:JSON.stringify({qid:current.qid, annotator, decision, user:$('user').value})}); current=data.task; renderAnnotations(); refresh(); }
async function splitData(){ alert(JSON.stringify(await api('/api/split',{method:'POST',body:JSON.stringify({user:$('user').value, train_ratio:Number($('trainRatio').value), val_ratio:Number($('valRatio').value), seed:Number($('seed').value), source:'all'})}))); refresh(); }
async function exportData(){ alert(JSON.stringify(await api('/api/export',{method:'POST',body:JSON.stringify({user:$('user').value, output_dir:$('outDir').value, source:$('exportSource').value, strip_test_labels:true})}))); }
function escapeHtml(s){ return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
function escapeJs(s){ return String(s).replace(/\\/g,'\\\\').replace(/'/g,"\\'"); }
showAuth();
</script>
</body>
</html>"""


INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>QV-M2 视频标注平台</title>
  <style>
    :root { --ink:#17202a; --muted:#667587; --line:#d8e0e8; --panel:#f6f8fa; --accent:#0f766e; --bad:#b91c1c; --warn:#a16207; }
    * { box-sizing:border-box; }
    body { margin:0; font-family:"Segoe UI", Arial, sans-serif; color:var(--ink); background:#fff; }
    header { display:flex; align-items:center; gap:10px; padding:12px 16px; border-bottom:1px solid var(--line); position:sticky; top:0; z-index:5; background:#fff; }
    h1 { margin:0; font-size:18px; white-space:nowrap; }
    input, select, textarea, button { font:inherit; }
    input, select, textarea { border:1px solid var(--line); border-radius:6px; padding:8px 10px; background:#fff; }
    button { border:1px solid var(--line); border-radius:6px; padding:8px 11px; background:#fff; cursor:pointer; }
    button:disabled { opacity:.45; cursor:not-allowed; }
    button.primary { background:var(--accent); color:#fff; border-color:var(--accent); }
    button.bad { color:var(--bad); }
    button.warn { color:var(--warn); }
    main { display:grid; grid-template-columns:360px minmax(0,1fr); min-height:calc(100vh - 58px); }
    aside { border-right:1px solid var(--line); background:var(--panel); padding:12px; overflow:auto; }
    section { padding:14px 18px; min-width:0; }
    .row { display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
    .stack { display:flex; flex-direction:column; gap:10px; }
    .box { border:1px solid var(--line); border-radius:8px; padding:10px; background:#fff; }
    .task, .group-video { border:1px solid var(--line); border-radius:7px; padding:9px; background:#fff; cursor:pointer; }
    .task { margin-bottom:8px; }
    .task.active, .group-video.active { border-color:var(--accent); box-shadow:0 0 0 2px rgba(15,118,110,.12); }
    .task small, .group-video small, .muted { color:var(--muted); }
    .pill { display:inline-block; border:1px solid var(--line); border-radius:999px; padding:2px 8px; font-size:12px; color:var(--muted); background:#fff; }
    .grow { flex:1; min-width:160px; }
    .query { width:100%; min-height:64px; }
    video { width:100%; max-height:50vh; background:#111; border-radius:8px; }
    .timeline { position:relative; height:42px; border:1px solid var(--line); border-radius:6px; background:#eef3f7; overflow:hidden; cursor:crosshair; }
    .seg { position:absolute; top:6px; height:28px; background:rgba(15,118,110,.65); border:1px solid #075f59; border-radius:4px; }
    .playhead { position:absolute; top:0; bottom:0; width:2px; background:#dc2626; }
    table { width:100%; border-collapse:collapse; }
    th,td { border-bottom:1px solid var(--line); padding:7px; text-align:left; vertical-align:top; }
    th { color:var(--muted); font-weight:600; }
    .group-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:8px; }
    .clip-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(120px,1fr)); gap:6px; margin-top:8px; }
    .clip-cell { border:1px solid var(--line); border-radius:6px; padding:6px; background:#f8fafc; }
    .clip-cell input { width:64px; padding:4px 6px; }
    @media (max-width:900px){ main{grid-template-columns:1fr;} aside{border-right:0;border-bottom:1px solid var(--line);max-height:44vh;} }
  </style>
</head>
<body>
<header>
  <h1>QV-M2 视频标注平台</h1>
  <input id="user" value="annotator1" title="账号">
  <select id="role" title="角色"><option value="annotator">标注员</option><option value="reviewer">审核员</option></select>
  <button onclick="loginUser()">登录/创建账号</button>
  <span id="roleBadge" class="pill">标注员</span>
  <button onclick="refresh()">刷新</button>
  <span id="stats" class="muted"></span>
</header>
<main>
  <aside class="stack">
    <div class="box stack">
      <div class="row">
        <select id="status" onchange="refresh()"><option value="all">全部状态</option><option value="published">published</option><option value="claimed">claimed</option><option value="submitted">submitted</option><option value="approved">approved</option></select>
        <select id="split" onchange="refresh()"><option value="all">全部集合</option><option value="train">train</option><option value="val">val</option><option value="test">test</option></select>
      </div>
      <input id="search" placeholder="搜索 qid / vid / query" onkeydown="if(event.key==='Enter') refresh()">
    </div>
    <div class="box stack" data-reviewer-only>
      <b>导入</b>
      <input id="jsonlFile" type="file" accept=".jsonl,.json,.txt" onchange="importJsonlFile()">
      <input id="jsonlPath" value="data/highlight_test_release.jsonl">
      <div class="row"><input id="jsonlSplit" class="grow" placeholder="split，可空"><button onclick="importJsonl()">导入 JSONL</button></div>
      <div class="row">
        <input id="videoFiles" type="file" accept="video/*,.mp4,.avi,.mov,.mkv,.webm,.m4v" multiple onchange="selectVideoFiles()" style="display:none">
        <button type="button" onclick="$('videoFiles').click()">选择视频文件</button>
        <input id="videoDirFiles" type="file" webkitdirectory directory multiple onchange="selectVideoDirectory()" style="display:none">
        <button type="button" onclick="$('videoDirFiles').click()">选择视频目录</button>
      </div>
      <div id="selectedVideoSummary" class="muted">尚未选择视频</div>
      <button id="confirmVideosBtn" onclick="confirmSelectedVideos()" disabled>确认视频</button>
      <input id="videoRoot" placeholder="服务器本机目录路径，例如 D:\videos">
      <button onclick="importVideos()">扫描视频目录</button>
      <div id="publishPlanner" class="box stack" style="display:none">
        <div class="row">
          <label class="stack grow"><span class="muted">发布任务数量</span><input id="publishTaskCount" type="number" min="1" value="1" onchange="buildPublishPlan()"></label>
          <label class="stack grow"><span class="muted">任务名前缀</span><input id="groupPrefix" value="标注任务" onchange="buildPublishPlan()"></label>
        </div>
        <div id="publishPlanSummary" class="muted"></div>
        <div id="publishPlanList" class="stack"></div>
      </div>
      <button id="publishGroupsBtn" class="primary" onclick="publishGroups()" disabled>发布任务</button>
      <button class="bad" onclick="resetWorkspace()">清空当前扫描视频</button>
    </div>
    <div class="box stack" data-reviewer-only>
      <b>划分与导出</b>
      <div class="row"><input id="trainRatio" value="0.8" class="grow"><input id="valRatio" value="0.1" class="grow"><input id="seed" value="2024" class="grow"></div>
      <button onclick="splitData()">按视频划分训练/验证/测试</button>
      <input id="outDir" value="annotation_workspace/exports">
      <select id="exportSource"><option value="approved">仅导出审核通过</option><option value="all">导出全部任务</option></select>
      <button class="primary" onclick="exportData()">导出 release JSONL</button>
      <select id="exportGroup"><option value="">全部任务包</option></select>
      <button class="primary" onclick="exportByGroup()">按任务包导出 JSONL</button>
    </div>
    <div class="box stack">
      <div class="row"><b>任务包</b><button onclick="refreshGroups()">刷新任务包</button></div>
      <div id="groups"></div>
    </div>
  </aside>
  <section class="stack">
    <div id="empty" class="box muted">请选择任务包。标注员接取任务包后，右侧会显示包内全部视频和完成状态。</div>
    <div id="groupPanel" class="box stack" style="display:none"></div>
    <div id="editor" class="stack" style="display:none">
      <div class="box stack">
        <div class="row">
          <b id="title"></b><span id="taskMeta" class="pill"></span>
          <button data-reviewer-only onclick="saveTask()">保存视频信息</button>
          <button data-reviewer-only class="bad" onclick="deleteCurrentTask()">删除视频任务</button>
          <button onclick="prevGroupVideo()">上一个</button>
          <button onclick="nextGroupVideo()">下一个</button>
        </div>
        <textarea id="query" class="query" placeholder="query"></textarea>
        <div class="row">
          <input id="vid" class="grow" placeholder="vid">
          <input id="duration" type="number" step="0.001" placeholder="duration">
          <input id="videoPath" class="grow" placeholder="视频路径">
          <label data-reviewer-only><input id="bindVideoFile" type="file" accept="video/*,.mp4,.avi,.mov,.mkv,.webm,.m4v" style="display:none" onchange="uploadCurrentVideo()"> <button type="button" onclick="$('bindVideoFile').click()">选择视频</button></label>
        </div>
      </div>
      <video id="video" controls></video>
      <div class="timeline" id="timeline" onmousedown="timelineDown(event)"></div>
      <div class="row">
        <button onclick="markStart()">设为开始</button>
        <button onclick="markEnd()">设为结束并添加</button>
        <input id="segStart" type="number" step="0.001" placeholder="start">
        <input id="segEnd" type="number" step="0.001" placeholder="end">
        <button onclick="addWindow()">添加片段</button>
      </div>
      <div class="box stack">
        <b>相关片段与显著性分数</b>
        <table><thead><tr><th>start</th><th>end</th><th>整段默认分数</th><th>clip 细分</th><th></th></tr></thead><tbody id="windows"></tbody></table>
        <textarea id="notes" placeholder="备注"></textarea>
        <div class="row">
          <button onclick="saveAnnotation('draft')">保存草稿</button>
          <button class="primary" onclick="saveAnnotation('submitted')">提交当前视频审核</button>
        </div>
      </div>
      <div class="box stack">
        <b>审核记录</b>
        <div id="annotations"></div>
      </div>
    </div>
  </section>
</main>
<script>
let tasks=[], groups=[], current=null, currentGroup=null, groupTasks=[], windows=[], windowScores=[], clipScores={}, expandedWindow=null, dragStart=null, currentUser=null, activeAnnotator=null;
let pendingVideoFiles=[], confirmedVideoCount=0, publishPlan=[];
const CLIP_LEN=2.0;
const $=id=>document.getElementById(id);
async function api(path, opts={}){const res=await fetch(path,{headers:{'Content-Type':'application/json'},...opts}); const data=await res.json(); if(data.error) throw new Error(data.error); return data;}
function isReviewer(){return currentUser&&currentUser.role==='reviewer';}
async function loginUser(){const data=await api('/api/user',{method:'POST',body:JSON.stringify({username:$('user').value,role:$('role').value})}); currentUser=data.user; $('role').value=currentUser.role; $('roleBadge').textContent=isReviewer()?'审核员':'标注员'; applyRoleUi(); refresh();}
function applyRoleUi(){document.querySelectorAll('[data-reviewer-only]').forEach(el=>el.style.display=isReviewer()?'':'none'); updatePublishState();}
function updatePublishState(){
  if($('confirmVideosBtn')) $('confirmVideosBtn').disabled=pendingVideoFiles.length===0;
  if($('publishPlanner')) $('publishPlanner').style.display=confirmedVideoCount>0?'flex':'none';
  validatePublishPlan();
}
async function refresh(){
  const params=new URLSearchParams({user:$('user').value,status:$('status').value,split:$('split').value,q:$('search').value});
  const [list,groupList,stats]=await Promise.all([api('/api/tasks?'+params),api('/api/groups?user='+encodeURIComponent($('user').value)),api('/api/stats')]);
  tasks=list.tasks;
  groups=groupList.groups;
  const availableVideoCount=Number(stats.unpublished_videos||0);
  $('stats').textContent=`视频任务 ${stats.total_tasks} | 可发布 ${availableVideoCount}`;
  if($('selectedVideoSummary') && !confirmedVideoCount) $('selectedVideoSummary').textContent=availableVideoCount?`有 ${availableVideoCount} 个未发布视频；扫描或确认视频后再发布任务`:'没有可发布视频';
  renderGroups();
  updateNextGroupName();
  buildPublishPlan();
  applyRoleUi();
}
async function refreshGroups(){const data=await api('/api/groups?user='+encodeURIComponent($('user').value)); groups=data.groups; renderGroups();}
function renderGroups(){const user=$('user').value; $('groups').innerHTML=groups.map(g=>{const free=!g.claimed_by; const mine=g.claimed_by===user; const total=Number(g.video_count||0); const annotated=Number(g.annotated_count||0); const submitted=Number(g.submitted_count||0); const action=free&&!isReviewer()?`<button onclick="claimGroup(${g.id})">接取</button>`:`<button onclick="openGroup(${g.id})">查看</button>`; return `<div class="task ${currentGroup&&currentGroup.id===g.id?'active':''}"><div><b>#G${g.id} ${escapeHtml(g.name)}</b> <span class="pill">${escapeHtml(g.status)}</span></div><div class="row"><span class="pill">${total} 视频</span><span class="pill">已标注 ${annotated}</span><span class="pill">待审核 ${submitted}</span></div><small>${g.claimed_by?'已接取: '+escapeHtml(g.claimed_by):'未接取'}${mine?' · 我的任务':''}</small><div>${action}</div></div>`}).join('');}
function updateNextGroupName(){
  const input=$('groupPrefix');
  if(!input||document.activeElement===input)return;
  let base=input.value.replace(/\d+$/,'')||'标注任务', best=0;
  for(const g of groups){const m=String(g.name||'').match(/^(.*?)(\d+)$/); if(m&&Number(m[2])>=best){base=m[1]||base; best=Number(m[2]);}}
  if(best>0 && !confirmedVideoCount) input.value=base;
}
function buildPublishPlan(){
  if(!$('publishPlanList')) return;
  const total=Number(confirmedVideoCount||0);
  let count=Math.max(1, Number($('publishTaskCount')&&$('publishTaskCount').value||1));
  count=Math.min(count, Math.max(1,total||1));
  if($('publishTaskCount')) $('publishTaskCount').value=count;
  const prefix=(($('groupPrefix')&&$('groupPrefix').value)||'标注任务').trim()||'标注任务';
  const base=Math.floor(total/count), extra=total%count;
  publishPlan=Array.from({length:count},(_,i)=>({name:`${prefix}${i+1}`,video_count:base+(i<extra?1:0),notes:''}));
  renderPublishPlan();
}
function renderPublishPlan(){
  if(!$('publishPlanList')) return;
  $('publishPlanList').innerHTML=publishPlan.map((p,i)=>`<div class="box stack"><div class="row"><b>任务 ${i+1}</b><label class="stack grow"><span class="muted">名称</span><input value="${escapeHtml(p.name)}" oninput="publishPlan[${i}].name=this.value;validatePublishPlan()"></label><label class="stack" style="width:120px"><span class="muted">视频数</span><input type="number" min="1" value="${p.video_count}" oninput="publishPlan[${i}].video_count=Number(this.value||0);validatePublishPlan()"></label></div><textarea placeholder="备注" oninput="publishPlan[${i}].notes=this.value">${escapeHtml(p.notes||'')}</textarea></div>`).join('');
  validatePublishPlan();
}
function validatePublishPlan(){
  const total=Number(confirmedVideoCount||0);
  const sum=publishPlan.reduce((n,p)=>n+Number(p.video_count||0),0);
  const ok=total>0 && publishPlan.length>0 && sum===total && publishPlan.every(p=>Number(p.video_count||0)>0 && String(p.name||'').trim());
  if($('publishPlanSummary')) $('publishPlanSummary').textContent=total>0?`已确认 ${total} 个视频，当前分配 ${sum} 个，剩余 ${total-sum} 个。`:'';
  if($('publishGroupsBtn')) $('publishGroupsBtn').disabled=!ok;
  return ok;
}
function clearPublishSelection(){
  confirmedVideoCount=0;
  publishPlan=[];
  if($('publishPlanList')) $('publishPlanList').innerHTML='';
  if($('publishPlanSummary')) $('publishPlanSummary').textContent='';
  if($('selectedVideoSummary')) $('selectedVideoSummary').textContent='没有可发布视频';
  updatePublishState();
}
async function deleteGroup(groupId, name){
  if(!confirm(`确认删除任务包 "${name}"（#G${groupId}）？\n\n视频任务和已有标注会保留，可重新发布到新的任务包。`))return;
  await api('/api/delete_group',{method:'POST',body:JSON.stringify({user:$('user').value,group_id:groupId})});
  if(currentGroup&&currentGroup.id===groupId){currentGroup=null; groupTasks=[]; current=null; $('groupPanel').style.display='none'; $('editor').style.display='none'; $('empty').style.display='block';}
  await refresh();
  clearPublishSelection();
}
async function openGroup(groupId){const data=await api(`/api/group/${groupId}?user=${encodeURIComponent($('user').value)}`); currentGroup=data.group; groupTasks=currentGroup.tasks||[]; $('empty').style.display='none'; renderGroupPanel(); renderGroups(); if(groupTasks.length) await loadTask(groupTasks[0].qid);}
async function claimGroup(groupId){const data=await api('/api/claim_group',{method:'POST',body:JSON.stringify({user:$('user').value,group_id:groupId})}); currentGroup=data.group; groupTasks=currentGroup.tasks||[]; $('empty').style.display='none'; renderGroupPanel(); await refreshGroups(); if(groupTasks.length) await loadTask(groupTasks[0].qid);}
function renderGroupPanel(){const panel=$('groupPanel'); if(!currentGroup){panel.style.display='none';return;} const total=groupTasks.length; const annotated=groupTasks.filter(t=>Number(t.my_window_count||0)>0||Number(t.ann_count||0)>0).length; const submitted=groupTasks.filter(t=>Number(t.submitted_count||0)>0||t.my_status==='submitted').length; const approved=groupTasks.filter(t=>Number(t.approved_count||0)>0||t.status==='approved').length; const remaining=Math.max(0,total-annotated); const canSubmit=!isReviewer()&&currentGroup.claimed_by===$('user').value; const cards=groupTasks.map(t=>{const status=t.my_status||t.status||'todo'; const checked=status==='submitted'?'checked':''; const cb=canSubmit?`<input type="checkbox" class="submit-qid" value="${t.qid}" ${checked} onclick="event.stopPropagation()">`:''; return `<div class="group-video ${current&&current.qid===t.qid?'active':''}" onclick="loadTask(${t.qid})"><div class="row">${cb}<b>#${t.qid}</b><span class="pill">${escapeHtml(status)}</span></div><small>${escapeHtml(t.vid||'')}</small><div>${escapeHtml((t.query||'').slice(0,90))}</div><small>${t.my_window_count?`片段 ${t.my_window_count}`:'未标注'}${t.submitted_count?' · 待审核 '+t.submitted_count:''}</small></div>`}).join(''); const submitActions=canSubmit?`<div class="row"><button class="primary" onclick="submitGroupVideos('selected')">提交选中视频审核</button><button onclick="submitGroupVideos('all')">提交本任务全部已标注视频</button></div>`:''; panel.innerHTML=`<div class="row"><b>任务包 #G${currentGroup.id} ${escapeHtml(currentGroup.name||'')}</b><span class="pill">${escapeHtml(currentGroup.status||'')}</span><span class="pill">${currentGroup.claimed_by?'已接取: '+escapeHtml(currentGroup.claimed_by):'未接取'}</span></div><div class="row"><span class="pill">视频 ${total}</span><span class="pill">已标注 ${annotated}</span><span class="pill">剩余 ${remaining}</span><span class="pill">已提交审核 ${submitted}</span><span class="pill">已通过 ${approved}</span></div>${isReviewer()?'<div class="muted">点击任务包内已提交的视频，查看并审核标注结果。</div>':''}${submitActions}<div class="group-grid">${cards}</div>`; panel.style.display='flex';}
async function submitGroupVideos(mode){let qids=[]; if(mode==='selected'){qids=Array.from(document.querySelectorAll('.submit-qid:checked')).map(x=>Number(x.value)); if(!qids.length)return alert('请先勾选要提交审核的视频');} const result=await api('/api/submit_group',{method:'POST',body:JSON.stringify({user:$('user').value,group_id:currentGroup.id,qids})}); currentGroup=result.group; groupTasks=currentGroup.tasks||[]; renderGroupPanel(); refresh();}
async function loadTask(qid){const data=await api(`/api/task/${qid}?user=${encodeURIComponent($('user').value)}`); current=data.task; activeAnnotator=isReviewer()?null:$('user').value; windows=(current.my_annotation&&current.my_annotation.windows)||[]; windowScores=(current.my_annotation&&current.my_annotation.saliency)||windows.map(()=>4); clipScores=(current.my_annotation&&current.my_annotation.clip_scores)||{}; expandedWindow=null; $('empty').style.display='none'; $('editor').style.display='flex'; $('groupPanel').style.display=currentGroup?'flex':'none'; $('title').textContent=`视频任务 #${current.qid}`; $('taskMeta').textContent=`${current.status} / ${current.split||'-'}`; $('query').value=current.query||''; $('vid').value=current.vid||''; $('duration').value=current.duration||''; $('videoPath').value=current.video_path||''; $('notes').value=current.my_annotation?current.my_annotation.notes:''; $('video').src=current.video_path?`/media/${current.qid}`:''; renderWindows(); renderAnnotations(); renderGroupPanel(); applyRoleUi();}
function currentGroupIndex(){return current&&groupTasks.length?groupTasks.findIndex(t=>t.qid===current.qid):-1;}
async function prevGroupVideo(){const i=currentGroupIndex(); if(i>0) await loadTask(groupTasks[i-1].qid);}
async function nextGroupVideo(){const i=currentGroupIndex(); if(i>=0&&i<groupTasks.length-1) await loadTask(groupTasks[i+1].qid);}
async function importJsonl(){alert(JSON.stringify(await api('/api/import_jsonl',{method:'POST',body:JSON.stringify({user:$('user').value,path:$('jsonlPath').value,split:$('jsonlSplit').value||null,video_root:$('videoRoot').value||null})}))); refresh();}
async function importJsonlFile(){const file=$('jsonlFile').files[0]; if(!file)return; const content=await file.text(); alert(JSON.stringify(await api('/api/import_jsonl_content',{method:'POST',body:JSON.stringify({user:$('user').value,content,filename:file.name,split:$('jsonlSplit').value||null,video_root:$('videoRoot').value||null})}))); $('jsonlFile').value=''; refresh();}
async function importVideos(){
  const result=await api('/api/import_videos',{method:'POST',body:JSON.stringify({user:$('user').value,path:$('videoRoot').value})});
  confirmedVideoCount=result.added||0;
  if(confirmedVideoCount<=0 && result.found>0){
    const stats=await api('/api/stats');
    confirmedVideoCount=Number(stats.unpublished_videos||0);
  }
  $('selectedVideoSummary').textContent=confirmedVideoCount?`已确认 ${confirmedVideoCount} 个视频，可发布任务`:'没有可发布视频';
  buildPublishPlan(); updatePublishState(); refresh();
}
function setPendingVideos(files,label){pendingVideoFiles=files; confirmedVideoCount=0; publishPlan=[]; if($('publishPlanList')) $('publishPlanList').innerHTML=''; $('selectedVideoSummary').textContent=files.length?`${label}: 已选择 ${files.length} 个视频，点击确认后才能发布任务`:'尚未选择视频'; updatePublishState();}
function selectVideoFiles(){const files=Array.from($('videoFiles').files); if(files.length) setPendingVideos(files,'视频文件');}
function selectVideoDirectory(){const files=Array.from($('videoDirFiles').files).filter(f=>/\.(mp4|avi|mov|mkv|webm|m4v)$/i.test(f.name)); if(!files.length){$('videoDirFiles').value=''; return alert('目录里没有支持的视频文件');} setPendingVideos(files,'视频目录');}
async function confirmSelectedVideos(){if(!pendingVideoFiles.length)return; const fd=new FormData(); fd.append('user',$('user').value); pendingVideoFiles.forEach(f=>fd.append('files',f)); const res=await fetch('/api/upload_videos',{method:'POST',body:fd}); const data=await res.json(); if(data.error)throw new Error(data.error); confirmedVideoCount=data.count||0; pendingVideoFiles=[]; $('videoFiles').value=''; $('videoDirFiles').value=''; $('selectedVideoSummary').textContent=`已确认 ${confirmedVideoCount} 个视频，可发布任务`; buildPublishPlan(); updatePublishState(); refresh();}
async function publishGroups(){if(confirmedVideoCount<=0)return alert('请先选择并确认视频'); if(!validatePublishPlan()) return alert('请确保每个任务视频数大于 0，且总和等于已确认视频数'); const result=await api('/api/publish_groups',{method:'POST',body:JSON.stringify({user:$('user').value,name_prefix:$('groupPrefix').value,groups:publishPlan})}); confirmedVideoCount=0; publishPlan=[]; if($('publishPlanList')) $('publishPlanList').innerHTML=''; $('selectedVideoSummary').textContent='任务已发布，请选择新视频后继续发布'; updatePublishState(); alert(JSON.stringify(result)); refresh();}
async function resetWorkspace(){if(!confirm('确认删除所有已扫描/导入但未发布的视频任务？已发布的任务包和已有标注都会保留。'))return; const result=await api('/api/reset_workspace',{method:'POST',body:JSON.stringify({user:$('user').value,keep_users:true})}); current=null; currentGroup=null; groupTasks=[]; windows=[]; windowScores=[]; clipScores={}; confirmedVideoCount=0; publishPlan=[]; if($('publishPlanList')) $('publishPlanList').innerHTML=''; if($('publishPlanSummary')) $('publishPlanSummary').textContent=''; if($('selectedVideoSummary')) $('selectedVideoSummary').textContent='没有可发布视频'; $('editor').style.display='none'; $('groupPanel').style.display='none'; $('empty').style.display='block'; updatePublishState(); alert(JSON.stringify(result)); refresh();}
async function uploadCurrentVideo(){if(!current)return; const file=$('bindVideoFile').files[0]; if(!file)return; const fd=new FormData(); fd.append('user',$('user').value); fd.append('qid',current.qid); fd.append('files',file); const res=await fetch('/api/upload_videos',{method:'POST',body:fd}); const data=await res.json(); if(data.error)throw new Error(data.error); $('bindVideoFile').value=''; await loadTask(current.qid);}
async function saveTask(){const body={qid:current?current.qid:null,query:$('query').value,vid:$('vid').value,duration:Number($('duration').value),video_path:$('videoPath').value,user:$('user').value}; const data=await api('/api/task',{method:'POST',body:JSON.stringify(body)}); current=data.task; await loadTask(current.qid); refresh();}
async function deleteCurrentTask(){if(!current||!confirm(`确认删除视频任务 #${current.qid}？`))return; await api('/api/delete_task',{method:'POST',body:JSON.stringify({user:$('user').value,qid:current.qid})}); current=null; $('editor').style.display='none'; if(currentGroup) await openGroup(currentGroup.id); refresh();}
function markStart(){$('segStart').value=$('video').currentTime.toFixed(3);}
function markEnd(){$('segEnd').value=$('video').currentTime.toFixed(3); addWindow();}
function addWindow(){let s=Number($('segStart').value),e=Number($('segEnd').value); if(!(e>s))return alert('end 必须大于 start'); windows.push([s,e]); windowScores.push(4); sortWindows(); renderWindows();}
function sortWindows(){const paired=windows.map((w,i)=>({w,score:windowScores[i]??4})).sort((a,b)=>a.w[0]-b.w[0]); windows=paired.map(x=>x.w); windowScores=paired.map(x=>x.score);}
function clipIdsForWindow(w){const dur=Number($('duration').value||$('video').duration||0); const maxClip=Math.max(0,Math.ceil(dur/CLIP_LEN)-1); const a=Math.max(0,Math.floor(Number(w[0])/CLIP_LEN)); const b=Math.min(maxClip,Math.max(0,Math.ceil(Number(w[1])/CLIP_LEN)-1)); const ids=[]; for(let i=a;i<=b;i++)ids.push(i); return ids;}
function renderWindows(){const rows=windows.map((w,i)=>{const clips=expandedWindow===i?`<tr><td colspan="5"><div class="clip-grid">${clipIdsForWindow(w).map(cid=>`<div class="clip-cell">clip ${cid}<br><small>${(cid*CLIP_LEN).toFixed(1)}-${((cid+1)*CLIP_LEN).toFixed(1)}s</small><br><input type="number" min="0" max="4" value="${clipScores[cid]??windowScores[i]??4}" data-clip="${cid}"></div>`).join('')}</div></td></tr>`:''; return `<tr><td>${w[0]}</td><td>${w[1]}</td><td><input type="number" min="0" max="4" value="${windowScores[i]??4}" data-score="${i}"></td><td><button onclick="expandedWindow=expandedWindow===${i}?null:${i};renderWindows()">clip 评分</button></td><td><button class="bad" onclick="windows.splice(${i},1);windowScores.splice(${i},1);renderWindows()">删除</button></td></tr>${clips}`}).join(''); $('windows').innerHTML=rows; renderTimeline();}
function renderTimeline(){const tl=$('timeline'),dur=Number($('duration').value||$('video').duration||1); tl.innerHTML=windows.map(w=>`<div class="seg" style="left:${100*w[0]/dur}%;width:${100*(w[1]-w[0])/dur}%"></div>`).join('')+'<div id="playhead" class="playhead"></div>';}
function timelineDown(ev){const rect=$('timeline').getBoundingClientRect(),dur=Number($('duration').value||$('video').duration||1); dragStart=Math.max(0,Math.min(dur,(ev.clientX-rect.left)/rect.width*dur)); document.onmouseup=e=>{const end=Math.max(0,Math.min(dur,(e.clientX-rect.left)/rect.width*dur)); $('segStart').value=Math.min(dragStart,end).toFixed(3); $('segEnd').value=Math.max(dragStart,end).toFixed(3); document.onmouseup=null;};}
$('video').ontimeupdate=()=>{const ph=$('playhead'); if(ph){const dur=Number($('duration').value||$('video').duration||1); ph.style.left=(100*$('video').currentTime/dur)+'%';}};
async function saveAnnotation(status){const saliency=windows.map((_,i)=>Number(document.querySelector(`[data-score="${i}"]`).value||4)); document.querySelectorAll('[data-clip]').forEach(el=>clipScores[el.dataset.clip]=Number(el.value||4)); const body={qid:current.qid,user:$('user').value,annotator:activeAnnotator||$('user').value,duration:Number($('duration').value),windows,saliency,clip_scores:clipScores,notes:$('notes').value,status}; const data=await api('/api/annotation',{method:'POST',body:JSON.stringify(body)}); current=data.task; if(currentGroup) await openGroup(currentGroup.id); else renderAnnotations(); refresh();}
function renderAnnotations(){const anns=current.annotations||[]; $('annotations').innerHTML=anns.length?anns.map(a=>`<div class="box"><div><b>${escapeHtml(a.annotator)}</b> <span class="pill">${a.status}</span> <span class="muted">${a.updated_at}</span></div><div>片段: ${escapeHtml(JSON.stringify(a.windows))}</div><div class="muted">${escapeHtml(a.notes||'')}</div><button data-reviewer-only onclick="loadCorrection('${escapeJs(a.annotator)}')">载入修正</button><button data-reviewer-only class="primary" onclick="review('${escapeJs(a.annotator)}','approved')">通过</button><button data-reviewer-only class="warn" onclick="review('${escapeJs(a.annotator)}','rejected')">驳回</button></div>`).join(''):'<span class="muted">暂无标注</span>'; applyRoleUi();}
function loadCorrection(annotator){const ann=(current.annotations||[]).find(a=>a.annotator===annotator); if(!ann)return; activeAnnotator=annotator; windows=JSON.parse(JSON.stringify(ann.windows||[])); windowScores=(ann.saliency||[]).map(Number); clipScores=ann.clip_scores||{}; while(windowScores.length<windows.length)windowScores.push(4); $('notes').value=ann.notes||''; renderWindows();}
async function review(annotator,decision){const data=await api('/api/review',{method:'POST',body:JSON.stringify({qid:current.qid,annotator,decision,user:$('user').value})}); current=data.task; renderAnnotations(); if(currentGroup) await openGroup(currentGroup.id); refresh();}
async function splitData(){alert(JSON.stringify(await api('/api/split',{method:'POST',body:JSON.stringify({user:$('user').value,train_ratio:Number($('trainRatio').value),val_ratio:Number($('valRatio').value),seed:Number($('seed').value),source:'all'})}))); refresh();}
async function exportData(){alert(JSON.stringify(await api('/api/export',{method:'POST',body:JSON.stringify({user:$('user').value,output_dir:$('outDir').value,source:$('exportSource').value,strip_test_labels:true})})));}
async function exportByGroup(){const gid=$('exportGroup').value||null;alert(JSON.stringify(await api('/api/export_group',{method:'POST',body:JSON.stringify({user:$('user').value,output_dir:$('outDir').value,source:$('exportSource').value,group_id:gid})})));}
function showAuth(){
  currentUser=null;
  document.querySelector('header').style.display='none';
  document.querySelector('main').style.display='none';
  if(!$('authPanel')){
    document.body.insertAdjacentHTML('afterbegin', `<div id="authPanel" style="min-height:100vh;display:flex;align-items:center;justify-content:center;background:#f6f8fa;padding:20px"><div class="box stack" style="width:min(440px,100%);padding:18px"><h2 style="margin:0">QV-M2 &#35270;&#39057;&#26631;&#27880;&#24179;&#21488;</h2><div class="row"><button id="loginTab" class="primary" onclick="setAuthMode('login')">&#30331;&#24405;</button><button id="registerTab" onclick="setAuthMode('register')">&#21019;&#24314;&#36134;&#21495;</button></div><input id="authUsername" placeholder="&#36134;&#21495;"><input id="authPassword" type="password" placeholder="&#23494;&#30721;"><div id="registerFields" class="stack" style="display:none"><input id="authDisplayName" placeholder="&#26174;&#31034;&#21517;&#31216;&#65292;&#21487;&#31354;"><select id="authRole"><option value="annotator">&#26631;&#27880;&#21592;</option><option value="reviewer">&#23457;&#26680;&#21592;</option></select></div><button class="primary" onclick="submitAuth()">&#36827;&#20837;&#31995;&#32479;</button><div id="authMsg" class="muted"></div><div class="muted">&#40664;&#35748;&#36134;&#21495;&#65306;annotator1 / annotator1, reviewer1 / reviewer1.</div></div></div>`);
  }
  $('authPanel').style.display='flex';
  setAuthMode('login');
}
function setAuthMode(mode){
  window.authMode=mode;
  $('registerFields').style.display=mode==='register'?'flex':'none';
  $('loginTab').className=mode==='login'?'primary':'';
  $('registerTab').className=mode==='register'?'primary':'';
  $('authMsg').textContent='';
}
async function submitAuth(){
  try{
    const payload={username:$('authUsername').value,password:$('authPassword').value,role:$('authRole')?$('authRole').value:'annotator',display_name:$('authDisplayName')?$('authDisplayName').value:''};
    const data=await api(window.authMode==='register'?'/api/register':'/api/login',{method:'POST',body:JSON.stringify(payload)});
    currentUser=data.user;
    $('user').value=currentUser.username;
    $('role').value=currentUser.role;
    $('roleBadge').textContent=currentUser.role==='reviewer'?'审核员':'标注员';
    $('authPanel').style.display='none';
    document.querySelector('header').style.display='flex';
    document.querySelector('main').style.display='grid';
    ensureAccountBar();
    applyRoleUi();
    refresh();
  }catch(err){$('authMsg').textContent=err.message;}
}
function ensureAccountBar(){
  let info=$('accountInfo');
  if(!info){
    document.querySelector('header').insertAdjacentHTML('beforeend','<span id="accountInfo" class="pill"></span><button id="logoutBtn" onclick="logout()">退出登录</button>');
    info=$('accountInfo');
  }
  $('user').style.display='none';
  $('role').style.display='none';
  const oldLoginButton=document.querySelector('header button[onclick="loginUser()"]');
  if(oldLoginButton) oldLoginButton.style.display='none';
  info.textContent=`${currentUser.display_name||currentUser.username} / ${currentUser.role==='reviewer'?'审核员':'标注员'}`;
}
function logout(){
  current=null; currentGroup=null; groupTasks=[]; tasks=[]; groups=[]; windows=[]; windowScores=[]; clipScores={};
  $('editor').style.display='none'; $('groupPanel').style.display='none'; $('empty').style.display='block';
  showAuth();
}
function escapeHtml(s){return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function escapeJs(s){return String(s).replace(/\\/g,'\\\\').replace(/'/g,"\\'");}

function taskHasMyAnnotation(t){
  return Number(t.my_window_count||0)>0 || ['draft','submitted','approved','rejected'].includes(t.my_status||'');
}
function taskSubmitted(t){
  return Number(t.submitted_count||0)>0 || t.my_status==='submitted';
}
function taskApproved(t){
  return Number(t.approved_count||0)>0 || t.status==='approved' || t.my_status==='approved';
}
function taskRejected(t){
  return Number(t.rejected_count||0)>0 || t.status==='rejected' || t.my_status==='rejected';
}
function statusText(s){
  return ({todo:'未标注',draft:'草稿',submitted:'待审核',approved:'已完成',rejected:'已退回',claimed:'已接取',published:'可接取'}[s]||s||'未标注');
}
function selectedQueryAnnotation(){
  if(!current) return null;
  if(isReviewer()){
    return (current.annotations||[]).find(a=>a.status==='submitted') || (current.annotations||[])[0] || null;
  }
  return current.my_annotation || null;
}
function videoGroups(){
  const map=new Map();
  for(const t of groupTasks||[]){
    const key=t.vid||`qid-${t.qid}`;
    if(!map.has(key)) map.set(key,{vid:key,tasks:[],queryCount:0,annotated:0,submitted:0,approved:0,rejected:0});
    const item=map.get(key);
    item.tasks.push(t);
    item.queryCount++;
    if(taskHasMyAnnotation(t) || (isReviewer() && Number(t.ann_count||0)>0)) item.annotated++;
    if(taskSubmitted(t)) item.submitted++;
    if(taskApproved(t)) item.approved++;
    if(taskRejected(t)) item.rejected++;
  }
  return Array.from(map.values());
}
function pickTaskForVideo(vid){
  const items=(groupTasks||[]).filter(t=>(t.vid||`qid-${t.qid}`)===vid);
  if(!items.length) return null;
  if(isReviewer()) return items.find(taskSubmitted) || items[0];
  return items.find(t=>!taskHasMyAnnotation(t)) || items.find(t=>t.my_status!=='submitted') || items[0];
}
async function loadVideoTask(vid){
  const task=pickTaskForVideo(vid);
  if(task) await loadTask(task.qid);
}
async function refreshCurrentGroup(){
  if(!currentGroup) return;
  const data=await api(`/api/group/${currentGroup.id}?user=${encodeURIComponent($('user').value)}`);
  currentGroup=data.group;
  groupTasks=currentGroup?(currentGroup.tasks||[]):[];
  renderGroupPanel();
  renderQueryList();
  renderGroups();
}
function renderQueryList(){
  if(!current) return;
  let holder=$('queryList');
  if(!holder){
    const box=document.createElement('div');
    box.id='queryList';
    box.className='box stack';
    const queryBox=$('query').parentElement;
    queryBox.parentElement.insertBefore(box, queryBox.nextSibling);
    holder=box;
  }
  const same=(groupTasks||[]).filter(t=>t.vid===current.vid);
  if(!same.length){holder.style.display='none';return;}
  holder.style.display='flex';
  holder.innerHTML=`<div class="row"><b>Query 卡组</b><span class="pill">${escapeHtml(current.vid||'')}</span></div><div class="muted">选择一个 query 后，下方只显示这个 query 对应的片段标注。</div><div class="group-grid">`+
    same.map(t=>{
      const status=t.my_status||t.status||'todo';
      const count=isReviewer()?Number(t.display_window_count||t.my_window_count||0):Number(t.my_window_count||0);
      const tags=[
        `<span class="pill">${statusText(status)}</span>`,
        `<span class="pill">片段 ${count}</span>`,
        taskSubmitted(t)?'<span class="pill">待审核</span>':'',
        taskApproved(t)?'<span class="pill">已完成</span>':'',
        taskRejected(t)?'<span class="pill">已退回</span>':''
      ].join('');
      return `<div class="task ${current&&current.qid===t.qid?'active':''}" onclick="loadTask(${t.qid})"><div class="row"><b>#${t.qid}</b>${tags}</div><div>${escapeHtml(t.query||'未填写 query')}</div></div>`;
    }).join('')+'</div>';
  applyRoleUi();
}
function ensureSelectedQueryBox(){
  let holder=$('selectedQueryBox');
  if(!holder){
    const box=document.createElement('div');
    box.id='selectedQueryBox';
    box.className='box stack';
    const timeline=$('timeline');
    timeline.parentElement.insertBefore(box, timeline);
    holder=box;
  }
  return holder;
}
function renderSelectedQueryBox(){
  if(!current) return;
  const ann=selectedQueryAnnotation();
  const status=(ann&&ann.status)||current.my_status||current.status||'todo';
  const holder=ensureSelectedQueryBox();
  const rejected=ann&&ann.status==='rejected';
  holder.innerHTML=`<div class="row"><b>当前 Query #${current.qid}</b><span class="pill">${statusText(status)}</span><span class="pill">片段 ${windows.length}</span></div><div class="box" style="background:#f8fafc">${escapeHtml(current.query||'未填写 query')}</div>${rejected?'<div class="box" style="border-color:#a16207;color:#7c2d12;background:#fff7ed">该标注已被审核员退回，请修改片段后重新提交审核。</div>':''}`;
}
function renderGroups(){
  const user=$('user').value;
  const active=groups.filter(g=>g.status!=='approved');
  const done=groups.filter(g=>g.status==='approved');
  const renderList=list=>list.map(g=>{
    const free=!g.claimed_by;
    const mine=g.claimed_by===user;
    const total=Number(g.video_count||0);
    const annotated=Number(g.annotated_count||0);
    const submitted=Number(g.submitted_count||0);
    const rejected=Number(g.rejected_count||0);
    const approved=Number(g.approved_count||0);
    const remaining=Math.max(0,total-approved);
    const action=free&&!isReviewer()?`<button onclick="claimGroup(${g.id})">接取</button>`:`<button onclick="openGroup(${g.id})">查看</button>`;
    const deleteAction=isReviewer()?`<button class="bad" onclick="deleteGroup(${g.id}, '${escapeJs(g.name)}')">删除任务包</button>`:'';
    const warn=rejected?`<div class="box" style="border-color:#a16207;color:#7c2d12;background:#fff7ed">有 ${rejected} 个视频被退回，需要修改后重新提交。</div>`:'';
    return `<div class="task ${currentGroup&&currentGroup.id===g.id?'active':''}"><div><b>#G${g.id} ${escapeHtml(g.name)}</b> <span class="pill">${statusText(g.status)}</span></div><div class="row"><span class="pill">总视频 ${total}</span><span class="pill">已完成 ${approved}</span><span class="pill">剩余 ${remaining}</span>${submitted?`<span class="pill">待审核 ${submitted}</span>`:''}${rejected?`<span class="pill">退回 ${rejected}</span>`:''}</div><small>${g.claimed_by?'已接取: '+escapeHtml(g.claimed_by):'未接取'}${mine?' · 我的任务':''}</small>${warn}<div>${action}${deleteAction}</div></div>`;
  }).join('');
  $('groups').innerHTML=`<b>进行中 / 待处理</b>${renderList(active)||'<div class="muted">暂无待处理任务</div>'}<b>已完成内容</b>${renderList(done)||'<div class="muted">暂无已完成任务</div>'}`;
  const exportGroup=$('exportGroup');
  if(exportGroup){
    const currentValue=exportGroup.value;
    exportGroup.innerHTML='<option value="">全部任务包</option>'+groups.map(g=>`<option value="${g.id}">#G${g.id} ${escapeHtml(g.name)} (${Number(g.video_count||0)} 视频)</option>`).join('');
    exportGroup.value=currentValue;
  }
}
async function deleteGroup(groupId, name){
  if(!confirm(`确认删除任务包 "${name}"（#G${groupId}）？\n\n视频任务和已有标注会保留，可重新发布到新的任务包。`))return;
  await api('/api/delete_group',{method:'POST',body:JSON.stringify({user:$('user').value,group_id:groupId})});
  if(currentGroup&&currentGroup.id===groupId){currentGroup=null; groupTasks=[]; current=null; $('groupPanel').style.display='none'; $('editor').style.display='none'; $('empty').style.display='block';}
  await refresh();
  clearPublishSelection();
}
async function openGroup(groupId){
  const data=await api(`/api/group/${groupId}?user=${encodeURIComponent($('user').value)}`);
  currentGroup=data.group; groupTasks=currentGroup.tasks||[];
  $('empty').style.display='none';
  renderGroupPanel(); renderGroups();
  const vids=videoGroups();
  if(vids.length) await loadVideoTask(vids[0].vid);
}
async function claimGroup(groupId){
  const data=await api('/api/claim_group',{method:'POST',body:JSON.stringify({user:$('user').value,group_id:groupId})});
  currentGroup=data.group; groupTasks=currentGroup.tasks||[];
  $('empty').style.display='none';
  renderGroupPanel(); await refreshGroups();
  const vids=videoGroups();
  if(vids.length) await loadVideoTask(vids[0].vid);
}
function renderGroupPanel(){
  const panel=$('groupPanel');
  if(!currentGroup){panel.style.display='none';return;}
  const vids=videoGroups();
  const total=vids.length;
  const annotated=vids.filter(v=>v.annotated>0).length;
  const submitted=vids.filter(v=>v.submitted>0).length;
  const approved=vids.filter(v=>v.approved===v.queryCount && v.queryCount>0).length;
  const rejected=vids.filter(v=>v.rejected>0).length;
  const remaining=Math.max(0,total-approved);
  const canSubmit=!isReviewer()&&currentGroup.claimed_by===$('user').value;
  const makeCards=(list, preferredStatus)=>list.map(v=>{
    const active=current&&current.vid===v.vid;
    const checked=v.submitted>0?'checked':'';
    const first=v.tasks[0]||{};
    const cb=canSubmit?`<input type="checkbox" class="submit-vid" value="${escapeHtml(v.vid)}" ${checked} onclick="event.stopPropagation()">`:'';
    return `<div class="group-video ${active?'active':''}" onclick="loadVideoTask('${escapeJs(v.vid)}','${preferredStatus||''}')"><div class="row">${cb}<b>${escapeHtml(v.vid)}</b></div><div class="row"><span class="pill">query ${v.queryCount}</span><span class="pill">已标注 ${v.annotated}</span><span class="pill">待审核 ${v.submitted}</span><span class="pill">已完成 ${v.approved}</span><span class="pill">退回 ${v.rejected}</span></div><small>${escapeHtml((first.query||'').slice(0,96))}</small></div>`;
  }).join('');
  const pendingCards=makeCards(vids.filter(v=>v.approved!==v.queryCount));
  const doneCards=makeCards(vids.filter(v=>v.approved===v.queryCount && v.queryCount>0));
  const submitActions=canSubmit?`<div class="row"><button class="primary" onclick="submitGroupVideos('selected')">提交选中视频审核</button><button onclick="submitGroupVideos('all')">提交本任务全部已标注视频</button></div>`:'';
  panel.innerHTML=`<div class="row"><b>任务包 #G${currentGroup.id} ${escapeHtml(currentGroup.name||'')}</b><span class="pill">${statusText(currentGroup.status)}</span><span class="pill">${currentGroup.claimed_by?'已接取: '+escapeHtml(currentGroup.claimed_by):'未接取'}</span></div><div class="row"><span class="pill">视频 ${total}</span><span class="pill">已标注 ${annotated}</span><span class="pill">剩余 ${remaining}</span><span class="pill">待审核 ${submitted}</span><span class="pill">已完成 ${approved}</span><span class="pill">退回 ${rejected}</span></div>${isReviewer()?'<div class="muted">点击视频后，在下方 Query 卡组中选择具体 query 查看并审核。</div>':''}${submitActions}<b>进行中 / 待处理</b><div class="group-grid">${pendingCards||'<span class="muted">暂无待处理视频</span>'}</div><b>已完成</b><div class="group-grid">${doneCards||'<span class="muted">暂无已完成视频</span>'}</div>`;
  panel.style.display='flex';
}
async function submitGroupVideos(mode){
  let qids=[];
  if(mode==='selected'){
    const vids=new Set(Array.from(document.querySelectorAll('.submit-vid:checked')).map(x=>x.value));
    if(!vids.size) return alert('请先勾选要提交审核的视频');
    qids=(groupTasks||[]).filter(t=>vids.has(t.vid||`qid-${t.qid}`)).map(t=>Number(t.qid));
  }
  const result=await api('/api/submit_group',{method:'POST',body:JSON.stringify({user:$('user').value,group_id:currentGroup.id,qids})});
  currentGroup=result.group; groupTasks=currentGroup.tasks||[];
  renderGroupPanel(); renderQueryList(); refresh();
}
async function loadTask(qid){
  const data=await api(`/api/task/${qid}?user=${encodeURIComponent($('user').value)}`);
  current=data.task; activeAnnotator=isReviewer()?null:$('user').value;
  windows=(current.my_annotation&&current.my_annotation.windows)||[];
  windowScores=(current.my_annotation&&current.my_annotation.saliency)||windows.map(()=>4);
  clipScores=(current.my_annotation&&current.my_annotation.clip_scores)||{};
  expandedWindow=null;
  $('empty').style.display='none'; $('editor').style.display='flex'; $('groupPanel').style.display=currentGroup?'flex':'none';
  $('title').textContent=`标注项 #${current.qid}`;
  $('taskMeta').textContent=`${current.status} / ${current.split||'-'}`;
  $('query').value=current.query||''; $('vid').value=current.vid||''; $('duration').value=current.duration||''; $('videoPath').value=current.video_path||''; $('notes').value=current.my_annotation?current.my_annotation.notes:'';
  $('video').src=current.video_path?`/media/${current.qid}`:'';
  renderWindows(); renderAnnotations(); renderGroupPanel(); renderQueryList(); renderSelectedQueryBox(); applyRoleUi();
}
function currentVideoIndex(){const vids=videoGroups(); return current?vids.findIndex(v=>v.vid===current.vid):-1;}
async function prevGroupVideo(){const vids=videoGroups(); const i=currentVideoIndex(); if(i>0) await loadVideoTask(vids[i-1].vid);}
async function nextGroupVideo(){const vids=videoGroups(); const i=currentVideoIndex(); if(i>=0&&i<vids.length-1) await loadVideoTask(vids[i+1].vid);}
if($('video')) $('video').onloadedmetadata=()=>{
  const dur=Number($('duration').value||0);
  if((!dur || dur<=0) && Number.isFinite($('video').duration) && $('video').duration>0){
    $('duration').value=$('video').duration.toFixed(3);
    if(current) current.duration=Number($('duration').value);
    renderTimeline();
  }
};
async function saveAnnotation(status){
  if(!current) return;
  const saliency=windows.map((_,i)=>Number(document.querySelector(`[data-score="${i}"]`).value||4));
  document.querySelectorAll('[data-clip]').forEach(el=>clipScores[el.dataset.clip]=Number(el.value||4));
  const body={qid:current.qid,user:$('user').value,annotator:activeAnnotator||$('user').value,duration:Number($('duration').value||$('video').duration||0),windows,saliency,clip_scores:clipScores,notes:$('notes').value,status};
  const data=await api('/api/annotation',{method:'POST',body:JSON.stringify(body)});
  current=data.task;
  if(currentGroup) await refreshCurrentGroup();
  renderAnnotations(); renderWindows(); renderQueryList(); renderSelectedQueryBox(); refreshGroups();
  alert(status==='submitted'?'已提交审核':'已保存草稿');
}
function renderAnnotations(){
  const anns=current.annotations||[];
  const queryHtml=`<div class="box" style="background:#f8fafc"><b>对应 Query</b><div>${escapeHtml(current.query||'未填写 query')}</div></div>`;
  const list=anns.length?anns.map(a=>{
    const rejected=a.status==='rejected';
    const approved=a.status==='approved';
    const submitted=a.status==='submitted';
    return `<div class="box"><div class="row"><b>${escapeHtml(a.annotator)}</b><span class="pill">${statusText(a.status)}</span><span class="muted">${a.updated_at}</span></div>${rejected?'<div class="box" style="border-color:#a16207;color:#7c2d12;background:#fff7ed">审核已退回，等待标注员修改后重新提交。</div>':''}${approved?'<div class="box" style="border-color:#0f766e;color:#065f46;background:#ecfdf5">审核已通过，已进入已完成内容。</div>':''}<div>片段: ${escapeHtml(JSON.stringify(a.windows))}</div><div>分数: ${escapeHtml(JSON.stringify(a.saliency))}</div><div class="muted">${escapeHtml(a.notes||'')}</div><div class="row"><button data-reviewer-only onclick="loadCorrection('${escapeJs(a.annotator)}')">载入修正</button>${submitted?`<button data-reviewer-only class="primary" onclick="review('${escapeJs(a.annotator)}','approved')">通过</button><button data-reviewer-only class="warn" onclick="review('${escapeJs(a.annotator)}','rejected')">退回</button>`:''}</div></div>`;
  }).join(''):'<span class="muted">暂无标注</span>';
  $('annotations').innerHTML=queryHtml+list;
  applyRoleUi();
}
async function review(annotator,decision){
  const data=await api('/api/review',{method:'POST',body:JSON.stringify({qid:current.qid,annotator,decision,user:$('user').value})});
  current=data.task;
  if(currentGroup) await refreshCurrentGroup();
  renderAnnotations(); renderQueryList(); renderSelectedQueryBox(); await refreshGroups();
  alert(decision==='approved'?'整个视频已审核通过，已放入已完成内容':'整个视频已退回，标注员可修改后重新提交');
}
async function reviewBatch(decision){
  if(!currentGroup) return;
  const vids=Array.from(document.querySelectorAll('.review-vid:checked')).map(x=>x.value);
  if(!vids.length) return alert('请先勾选要审核的视频');
  const result=await api('/api/review_batch',{method:'POST',body:JSON.stringify({user:$('user').value,group_id:currentGroup.id,vids,decision})});
  currentGroup=result.group; groupTasks=currentGroup.tasks||[];
  if(current) current=(await api(`/api/task/${current.qid}?user=${encodeURIComponent($('user').value)}`)).task;
  renderGroupPanel();
  if(current){renderQueryList(); renderSelectedQueryBox(); renderAnnotations();}
  await refreshGroups();
  alert(`${decision==='approved'?'已通过':'已退回'}选中视频；处理标注 ${result.reviewed||0} 条，跳过 ${result.skipped||0} 个视频`);
}

function queryConfirmed(){
  return !!(current && String(current.query||'').trim());
}
function canEditCurrentQuery(){
  if(!current) return false;
  if(isReviewer()) return true;
  const mine=currentGroup && currentGroup.claimed_by===$('user').value;
  const ann=selectedQueryAnnotation();
  const status=(ann&&ann.status)||current.my_status||current.status||'todo';
  return mine && !['submitted','approved'].includes(status);
}
function renderSelectedQueryBox(){
  if(!current) return;
  const ann=selectedQueryAnnotation();
  const status=(ann&&ann.status)||current.my_status||current.status||'todo';
  const holder=ensureSelectedQueryBox();
  const rejected=ann&&ann.status==='rejected';
  const editable=canEditCurrentQuery();
  holder.innerHTML=`<div class="row"><b>当前 Query #${current.qid}</b><span class="pill">${statusText(status)}</span><span class="pill">片段 ${windows.length}</span></div>
    <textarea id="queryDraft" class="query" placeholder="先填写 query，确认后再标注片段"${editable?'':' disabled'}>${escapeHtml(current.query||'')}</textarea>
    <div class="row">
      <button class="primary" onclick="confirmQuery()" ${editable?'':'disabled'}>确认 Query</button>
      <button onclick="addQueryForCurrentVideo()" ${currentGroup?'':'disabled'}>添加 Query</button>
      <button class="bad" onclick="deleteCurrentQuery()" ${editable?'':'disabled'}>删除 Query</button>
      ${queryConfirmed()?'<span class="pill">已确认，可标注片段</span>':'<span class="pill">未确认，暂不能标注片段</span>'}
    </div>
    ${!queryConfirmed()?'<div class="box" style="border-color:#a16207;color:#7c2d12;background:#fff7ed">请先填写并确认 query，然后再添加片段标注。</div>':''}
    ${rejected?'<div class="box" style="border-color:#a16207;color:#7c2d12;background:#fff7ed">该标注已被审核员退回，请修改 query 或片段后重新提交审核。</div>':''}`;
}
async function confirmQuery(){
  if(!current) return;
  const text=String(($('queryDraft')&&$('queryDraft').value)||'').trim();
  if(!text) return alert('请先填写 query');
  const body={qid:current.qid,user:$('user').value,query:text,vid:$('vid').value||current.vid,duration:Number($('duration').value||$('video').duration||current.duration||0),video_path:$('videoPath').value||current.video_path||'',group_id:currentGroup?currentGroup.id:current.group_id,split:current.split,status:current.status==='todo'?'draft':current.status};
  const data=await api('/api/task',{method:'POST',body:JSON.stringify(body)});
  current=data.task;
  if(currentGroup) await refreshCurrentGroup();
  await loadTask(current.qid);
  alert('Query 已确认，可以开始标注片段');
}
async function addQueryForCurrentVideo(){
  if(!current || !currentGroup) return alert('请先选择任务包中的一个视频');
  const body={qid:null,user:$('user').value,query:'',vid:current.vid,duration:Number($('duration').value||$('video').duration||current.duration||0),video_path:$('videoPath').value||current.video_path||'',group_id:currentGroup.id,split:current.split,status:'draft'};
  const data=await api('/api/task',{method:'POST',body:JSON.stringify(body)});
  current=data.task;
  if(currentGroup) await refreshCurrentGroup();
  await loadTask(current.qid);
}
async function deleteCurrentQuery(){
  if(!current) return;
  if(!confirm(`确认删除 Query #${current.qid}？对应片段标注也会删除。`)) return;
  const oldVid=current.vid;
  await api('/api/delete_task',{method:'POST',body:JSON.stringify({user:$('user').value,qid:current.qid})});
  current=null; windows=[]; windowScores=[]; clipScores={};
  if(currentGroup){
    await refreshCurrentGroup();
    const next=(groupTasks||[]).find(t=>t.vid===oldVid) || (groupTasks||[])[0];
    if(next) await loadTask(next.qid);
    else {$('editor').style.display='none'; $('empty').style.display='block';}
  } else {
    $('editor').style.display='none'; $('empty').style.display='block';
  }
  refreshGroups();
}
async function saveTask(){
  if($('queryDraft')) $('query').value=$('queryDraft').value;
  return confirmQuery();
}
async function deleteCurrentTask(){
  return deleteCurrentQuery();
}
function markStart(){
  if(!queryConfirmed()) return alert('请先填写并确认 query');
  $('segStart').value=$('video').currentTime.toFixed(3);
}
function markEnd(){
  if(!queryConfirmed()) return alert('请先填写并确认 query');
  $('segEnd').value=$('video').currentTime.toFixed(3);
  addWindow();
}
function addWindow(){
  if(!queryConfirmed()) return alert('请先填写并确认 query');
  let s=Number($('segStart').value),e=Number($('segEnd').value);
  if(!(e>s))return alert('end 必须大于 start');
  windows.push([s,e]); windowScores.push(4); sortWindows(); renderWindows();
}
async function loadTask(qid){
  const data=await api(`/api/task/${qid}?user=${encodeURIComponent($('user').value)}`);
  current=data.task; activeAnnotator=isReviewer()?null:$('user').value;
  windows=(current.my_annotation&&current.my_annotation.windows)||[];
  windowScores=(current.my_annotation&&current.my_annotation.saliency)||windows.map(()=>4);
  clipScores=(current.my_annotation&&current.my_annotation.clip_scores)||{};
  expandedWindow=null;
  $('empty').style.display='none'; $('editor').style.display='flex'; $('groupPanel').style.display=currentGroup?'flex':'none';
  $('title').textContent=`标注项 #${current.qid}`;
  $('taskMeta').textContent=`${statusText(current.status)} / ${current.split||'-'}`;
  $('query').value=current.query||''; $('query').style.display='none';
  $('vid').value=current.vid||''; $('duration').value=current.duration||''; $('videoPath').value=current.video_path||''; $('notes').value=current.my_annotation?current.my_annotation.notes:'';
  $('video').src=current.video_path?`/media/${current.qid}`:'';
  renderWindows(); renderAnnotations(); renderGroupPanel(); renderQueryList(); renderSelectedQueryBox(); applyRoleUi();
}
async function saveAnnotation(status){
  if(!current) return;
  if(!queryConfirmed()) return alert('请先填写并确认 query');
  const saliency=windows.map((_,i)=>Number(document.querySelector(`[data-score="${i}"]`).value||4));
  document.querySelectorAll('[data-clip]').forEach(el=>clipScores[el.dataset.clip]=Number(el.value||4));
  const body={qid:current.qid,user:$('user').value,annotator:activeAnnotator||$('user').value,duration:Number($('duration').value||$('video').duration||0),windows,saliency,clip_scores:clipScores,notes:$('notes').value,status};
  const data=await api('/api/annotation',{method:'POST',body:JSON.stringify(body)});
  current=data.task;
  if(currentGroup) await refreshCurrentGroup();
  renderAnnotations(); renderWindows(); renderQueryList(); renderSelectedQueryBox(); refreshGroups();
  alert(status==='submitted'?'已提交审核':'已保存草稿');
}

function visibleVideoGroups(){
  const vids=videoGroups();
  if(!isReviewer()) return vids;
  return vids.filter(v=>v.submitted>0 || v.approved>0);
}
function visibleQueriesForCurrentVideo(){
  return (groupTasks||[]).filter(t=>t.vid===current.vid);
}
function pickTaskForVideo(vid, preferredStatus){
  const items=(groupTasks||[]).filter(t=>(t.vid||`qid-${t.qid}`)===vid);
  const visible=isReviewer()?items.filter(t=>taskSubmitted(t)||taskApproved(t)):items;
  const list=visible.length?visible:items;
  if(!list.length) return null;
  if(isReviewer()){
    if(preferredStatus==='approved') return list.find(taskApproved) || list.find(taskSubmitted) || list[0];
    if(preferredStatus==='submitted') return list.find(taskSubmitted) || list.find(taskApproved) || list[0];
    return list.find(taskSubmitted) || list.find(taskApproved) || list[0];
  }
  return list.find(t=>!taskHasMyAnnotation(t)) || list.find(t=>t.my_status!=='submitted') || list[0];
}
async function deleteGroup(groupId, name){
  if(!confirm(`确认删除任务包 "${name}"（#G${groupId}）？\n\n视频任务和已有标注会保留，可重新发布到新的任务包。`))return;
  await api('/api/delete_group',{method:'POST',body:JSON.stringify({user:$('user').value,group_id:groupId})});
  if(currentGroup&&currentGroup.id===groupId){currentGroup=null; groupTasks=[]; current=null; $('groupPanel').style.display='none'; $('editor').style.display='none'; $('empty').style.display='block';}
  await refresh();
  clearPublishSelection();
}
async function openGroup(groupId){
  await autoSaveCurrentDraft();
  const data=await api(`/api/group/${groupId}?user=${encodeURIComponent($('user').value)}`);
  currentGroup=data.group; groupTasks=currentGroup.tasks||[];
  current=null; activeAnnotator=null; windows=[]; windowScores=[]; clipScores={};
  $('editor').style.display='none';
  $('empty').style.display='block';
  $('empty').textContent='请选择一个视频进入标注页面。';
  renderGroupPanel(); renderGroups();
  if(!visibleVideoGroups().length) $('empty').textContent=isReviewer()?'这个任务还没有标注员提交审核的视频。':'这个任务包暂无视频。';
}
function renderQueryList(){
  if(!current) return;
  let holder=$('queryList');
  if(!holder){
    const box=document.createElement('div');
    box.id='queryList';
    box.className='box stack';
    const queryBox=$('query').parentElement;
    queryBox.parentElement.insertBefore(box, queryBox.nextSibling);
    holder=box;
  }
  const same=visibleQueriesForCurrentVideo();
  if(!same.length){holder.style.display='none';return;}
  holder.style.display='flex';
  holder.innerHTML=`<div class="row"><b>Query 卡组</b><span class="pill">${escapeHtml(current.vid||'')}</span></div><div class="muted">${isReviewer()?'点击 query 卡片查看该 query 的标注片段；通过或退回会应用到整个视频。':'同一个视频可以创建多个 query；每个 query 对应自己的片段标注。'}</div><div class="group-grid">`+
    same.map((t,idx)=>{
       const status=t.my_status||t.status||'todo';
       const count=isReviewer()?Number(t.display_window_count||t.my_window_count||0):Number(t.my_window_count||0);
       const tags=[
         `<span class="pill">${statusText(status)}</span>`,
         `<span class="pill">片段 ${count}</span>`,
        taskSubmitted(t)?'<span class="pill">待审核</span>':'',
        taskApproved(t)?'<span class="pill">已完成</span>':'',
        taskRejected(t)?'<span class="pill">已退回</span>':''
      ].join('');
      return `<div class="task ${current&&current.qid===t.qid?'active':''}" onclick="loadTask(${t.qid})"><div class="row"><b>Query ${idx+1}</b><span class="pill">#${t.qid}</span>${tags}</div><div>${escapeHtml(t.query||'未填写 query')}</div></div>`;
    }).join('')+'</div>';
  applyRoleUi();
}
function renderGroupPanel(){
  const panel=$('groupPanel');
  if(!currentGroup){panel.style.display='none';return;}
  const all=videoGroups();
  const vids=visibleVideoGroups();
  const total=all.length;
  const annotated=all.filter(v=>v.annotated>0).length;
  const submitted=all.filter(v=>v.submitted>0).length;
  const approved=all.filter(v=>isReviewer() ? (v.approved>0 && v.submitted===0) : (v.approved===v.queryCount && v.queryCount>0)).length;
  const rejected=all.filter(v=>v.rejected>0).length;
  const remaining=Math.max(0,total-approved);
  const canSubmit=!isReviewer()&&currentGroup.claimed_by===$('user').value;
  const canReview=isReviewer()&&submitted>0;
  const makeCards=(list, preferredStatus)=>list.map(v=>{
    const active=current&&current.vid===v.vid;
    const checked=v.submitted>0?'checked':'';
    const first=v.tasks[0]||{};
    const submitCb=canSubmit?`<input type="checkbox" class="submit-vid" value="${escapeHtml(v.vid)}" ${checked} onclick="event.stopPropagation()">`:'';
    const reviewCb=canReview&&v.submitted>0?`<input type="checkbox" class="review-vid" value="${escapeHtml(v.vid)}" onclick="event.stopPropagation()">`:'';
    const cb=submitCb||reviewCb;
    return `<div class="group-video ${active?'active':''}" onclick="loadVideoTask('${escapeJs(v.vid)}','${preferredStatus||''}')"><div class="row">${cb}<b>${escapeHtml(v.vid)}</b></div><div class="row"><span class="pill">query ${v.queryCount}</span><span class="pill">已标注 ${v.annotated}</span><span class="pill">待审核 ${v.submitted}</span><span class="pill">已完成 ${v.approved}</span><span class="pill">退回 ${v.rejected}</span></div><small>${escapeHtml((first.query||'').slice(0,96))}</small></div>`;
  }).join('');
  const pending=vids.filter(v=>v.submitted>0 || (!isReviewer() && v.approved!==v.queryCount));
  const done=vids.filter(v=>isReviewer() ? (v.approved>0 && v.submitted===0) : (v.approved===v.queryCount && v.queryCount>0));
  const submitActions=canSubmit?`<div class="row"><button class="primary" onclick="submitGroupVideos('selected')">提交选中视频审核</button><button onclick="submitGroupVideos('all')">提交整个任务审核</button></div>`:'';
  const reviewActions=canReview?`<div class="row"><button class="primary" onclick="reviewBatch('approved')">通过选中视频</button><button class="warn" onclick="reviewBatch('rejected')">退回选中视频</button></div>`:'';
  panel.innerHTML=`<div class="row"><b>任务包 #G${currentGroup.id} ${escapeHtml(currentGroup.name||'')}</b><span class="pill">${statusText(currentGroup.status)}</span><span class="pill">${currentGroup.claimed_by?'已接取: '+escapeHtml(currentGroup.claimed_by):'未接取'}</span></div><div class="row"><span class="pill">总视频 ${total}</span><span class="pill">已完成 ${approved}</span><span class="pill">剩余 ${remaining}</span>${submitted?`<span class="pill">待审核 ${submitted}</span>`:''}${rejected?`<span class="pill">退回 ${rejected}</span>`:''}</div>${isReviewer()?'<div class="muted">这里只显示标注员已经提交审核的视频。点击视频后，可查看 query；审核按钮会处理整个视频。也可以勾选多个待审核视频后批量通过或退回。</div>':''}${submitActions}${reviewActions}<b>${isReviewer()?'待审核视频':'进行中 / 待处理'}</b><div class="group-grid">${makeCards(pending,'submitted')||'<span class="muted">暂无待处理视频</span>'}</div><b>已完成</b><div class="group-grid">${makeCards(done,'approved')||'<span class="muted">暂无已完成视频</span>'}</div>`;
  panel.style.display='flex';
}
async function submitGroupVideos(mode){
  let qids=[];
  if(mode==='selected'){
    const vids=new Set(Array.from(document.querySelectorAll('.submit-vid:checked')).map(x=>x.value));
    if(!vids.size) return alert('请先勾选要提交审核的视频');
    qids=(groupTasks||[]).filter(t=>vids.has(t.vid||`qid-${t.qid}`)).map(t=>Number(t.qid));
  }
  const result=await api('/api/submit_group',{method:'POST',body:JSON.stringify({user:$('user').value,group_id:currentGroup.id,qids})});
  currentGroup=result.group; groupTasks=currentGroup.tasks||[];
  renderGroupPanel(); renderQueryList(); refreshGroups();
  alert(mode==='selected'?'已提交选中的视频审核':'已提交整个任务中全部 query 审核');
}
async function submitCurrentVideoForReview(){
  clearTimeout(autoSaveTimer);
  if(!currentGroup||!current) return;
  const qids=Array.from(new Set((visibleQueriesForCurrentVideo().length?visibleQueriesForCurrentVideo():[current]).map(t=>Number(t.qid))));
  const result=await api('/api/submit_group',{method:'POST',body:JSON.stringify({user:$('user').value,group_id:currentGroup.id,qids})});
  currentGroup=result.group; groupTasks=currentGroup.tasks||[];
  renderGroupPanel(); renderQueryList(); refreshGroups();
}
async function saveAnnotation(status){
  if(!current) return;
  if(!queryConfirmed()) return alert('请先填写并确认 query');
  const saliency=windows.map((_,i)=>Number(document.querySelector(`[data-score="${i}"]`).value||4));
  document.querySelectorAll('[data-clip]').forEach(el=>clipScores[el.dataset.clip]=Number(el.value||4));
  const body={qid:current.qid,user:$('user').value,annotator:activeAnnotator||$('user').value,duration:Number($('duration').value||$('video').duration||0),windows,saliency,clip_scores:clipScores,notes:$('notes').value,status:'draft'};
  const data=await api('/api/annotation',{method:'POST',body:JSON.stringify(body)});
  current=data.task;
  if(status==='submitted'){
    await submitCurrentVideoForReview();
    current=(await api(`/api/task/${current.qid}?user=${encodeURIComponent($('user').value)}`)).task;
  } else if(currentGroup) {
    await refreshCurrentGroup();
  }
  renderAnnotations(); renderWindows(); renderQueryList(); renderSelectedQueryBox(); refreshGroups();
  alert(status==='submitted'?'已提交当前视频下全部 query 审核':'已保存草稿');
}
function renderAnnotations(){
  const anns=current.annotations||[];
  const queryHtml=`<div class="box" style="background:#f8fafc"><b>对应 Query</b><div>${escapeHtml(current.query||'未填写 query')}</div></div>`;
  const list=anns.length?anns.map(a=>{
    const rejected=a.status==='rejected';
    const approved=a.status==='approved';
    const submitted=a.status==='submitted';
    return `<div class="box"><div class="row"><b>${escapeHtml(a.annotator)}</b><span class="pill">${statusText(a.status)}</span><span class="muted">${a.updated_at}</span></div>${rejected?'<div class="box" style="border-color:#a16207;color:#7c2d12;background:#fff7ed">审核已退回，等待标注员修改后重新提交。</div>':''}${approved?'<div class="box" style="border-color:#0f766e;color:#065f46;background:#ecfdf5">审核已通过，已进入已完成内容。</div>':''}<div>片段: ${escapeHtml(JSON.stringify(a.windows))}</div><div>分数: ${escapeHtml(JSON.stringify(a.saliency))}</div><div class="muted">${escapeHtml(a.notes||'')}</div><div class="row">${submitted?`<button data-reviewer-only class="primary" onclick="review('${escapeJs(a.annotator)}','approved')">通过整个视频</button><button data-reviewer-only class="warn" onclick="review('${escapeJs(a.annotator)}','rejected')">退回整个视频</button>`:''}</div></div>`;
  }).join(''):'<span class="muted">暂无标注</span>';
  $('annotations').innerHTML=queryHtml+list;
  applyRoleUi();
}
async function loadTask(qid){
  const data=await api(`/api/task/${qid}?user=${encodeURIComponent($('user').value)}`);
  current=data.task; activeAnnotator=isReviewer()?null:$('user').value;
  const displayAnn=isReviewer()?selectedQueryAnnotation():current.my_annotation;
  windows=(displayAnn&&displayAnn.windows)||[];
  windowScores=(displayAnn&&displayAnn.saliency)||windows.map(()=>4);
  clipScores=(displayAnn&&displayAnn.clip_scores)||{};
  expandedWindow=null;
  $('empty').style.display='none'; $('editor').style.display='flex'; $('groupPanel').style.display=currentGroup?'flex':'none';
  $('title').textContent=`标注项 #${current.qid}`;
  $('taskMeta').textContent=`${statusText(current.status)} / ${current.split||'-'}`;
  $('query').value=current.query||''; $('query').style.display='none';
  $('vid').value=current.vid||''; $('duration').value=current.duration||''; $('videoPath').value=current.video_path||''; $('notes').value=displayAnn?displayAnn.notes:'';
  $('video').src=current.video_path?`/media/${current.qid}`:'';
  renderWindows(); renderAnnotations(); renderGroupPanel(); renderQueryList(); renderSelectedQueryBox(); applyRoleUi();
}

let autoSaveTimer=null, autoSaving=false;
function queryConfirmed(){
  const draft=$('queryDraft') ? $('queryDraft').value : '';
  return !!(current && String(current.query||draft||'').trim());
}
function collectScoresFromDom(){
  const saliency=windows.map((_,i)=>Number((document.querySelector(`[data-score="${i}"]`)||{}).value||windowScores[i]||4));
  document.querySelectorAll('[data-clip]').forEach(el=>clipScores[el.dataset.clip]=Number(el.value||4));
  windowScores=saliency;
  return saliency;
}
function canAutoSaveCurrent(){
  if(!current || isReviewer() || !canEditCurrentQuery()) return false;
  const ann=current.my_annotation;
  return !(ann && ['submitted','approved'].includes(ann.status));
}
async function autoSaveCurrentDraft(){
  if(!canAutoSaveCurrent() || autoSaving) return;
  autoSaving=true;
  try{
    const queryText=String(($('queryDraft')&&$('queryDraft').value)||current.query||'').trim();
    if(queryText){
      const body={qid:current.qid,user:$('user').value,query:queryText,vid:$('vid').value||current.vid,duration:Number($('duration').value||$('video').duration||current.duration||0),video_path:$('videoPath').value||current.video_path||'',group_id:currentGroup?currentGroup.id:current.group_id,split:current.split,status:current.status==='todo'?'draft':current.status};
      const data=await api('/api/task',{method:'POST',body:JSON.stringify(body)});
      current=data.task;
    }
    if(queryText && (windows.length || ($('notes')&&$('notes').value))){
      const body={qid:current.qid,user:$('user').value,annotator:$('user').value,duration:Number($('duration').value||$('video').duration||current.duration||0),windows,saliency:collectScoresFromDom(),clip_scores:clipScores,notes:($('notes')&&$('notes').value)||'',status:'draft'};
      const data=await api('/api/annotation',{method:'POST',body:JSON.stringify(body)});
      current=data.task;
    }
    if(currentGroup){
      const data=await api(`/api/group/${currentGroup.id}?user=${encodeURIComponent($('user').value)}`);
      currentGroup=data.group; groupTasks=currentGroup?(currentGroup.tasks||[]):[];
      renderGroupPanel(); renderQueryList();
    }
  } finally {
    autoSaving=false;
  }
}
function scheduleAutoSave(){
  if(!canAutoSaveCurrent()) return;
  clearTimeout(autoSaveTimer);
  autoSaveTimer=setTimeout(()=>autoSaveCurrentDraft().catch(err=>console.warn('autosave failed',err)),700);
}
function renderSelectedQueryBox(){
  if(!current) return;
  const ann=selectedQueryAnnotation();
  const status=(ann&&ann.status)||current.my_status||current.status||'todo';
  const holder=ensureSelectedQueryBox();
  const rejected=ann&&ann.status==='rejected';
  const editable=canEditCurrentQuery();
  holder.innerHTML=`<div class="row"><b>当前 Query #${current.qid}</b><span class="pill">${statusText(status)}</span><span class="pill">片段 ${windows.length}</span></div>
    <textarea id="queryDraft" class="query" placeholder="先填写 query，系统会自动保存"${editable?'':' disabled'} oninput="current.query=this.value;scheduleAutoSave()">${escapeHtml(current.query||'')}</textarea>
    <div class="row">
      <button class="primary" onclick="confirmQuery()" ${editable?'':'disabled'}>确认 Query</button>
      <button onclick="addQueryForCurrentVideo()" ${currentGroup&&!isReviewer()?'':'disabled'}>添加 Query</button>
      <button class="bad" onclick="deleteCurrentQuery()" ${editable?'':'disabled'}>删除 Query</button>
      ${queryConfirmed()?'<span class="pill">已填写，片段会自动保存</span>':'<span class="pill">请先填写 query</span>'}
    </div>
    ${!queryConfirmed()?'<div class="box" style="border-color:#a16207;color:#7c2d12;background:#fff7ed">请先填写 query；填写后可直接标注，系统会自动保存。</div>':''}
    ${rejected?'<div class="box" style="border-color:#a16207;color:#7c2d12;background:#fff7ed">该标注已被审核员退回，请修改 query 或片段后重新提交审核。</div>':''}`;
}
async function confirmQuery(){
  if(!current) return;
  const text=String(($('queryDraft')&&$('queryDraft').value)||current.query||'').trim();
  if(!text) return alert('请先填写 query');
  clearTimeout(autoSaveTimer);
  const body={qid:current.qid,user:$('user').value,query:text,vid:$('vid').value||current.vid,duration:Number($('duration').value||$('video').duration||current.duration||0),video_path:$('videoPath').value||current.video_path||'',group_id:currentGroup?currentGroup.id:current.group_id,split:current.split,status:current.status==='todo'?'draft':current.status};
  const data=await api('/api/task',{method:'POST',body:JSON.stringify(body)});
  current=data.task;
  if(currentGroup) await refreshCurrentGroup();
  await loadTask(current.qid);
}
async function addQueryForCurrentVideo(){
  if(!current || !currentGroup) return alert('请先选择任务包中的一个视频');
  await autoSaveCurrentDraft();
  const body={qid:null,user:$('user').value,query:'',vid:current.vid,duration:Number($('duration').value||$('video').duration||current.duration||0),video_path:$('videoPath').value||current.video_path||'',group_id:currentGroup.id,split:current.split,status:'draft'};
  const data=await api('/api/task',{method:'POST',body:JSON.stringify(body)});
  current=data.task;
  if(currentGroup) await refreshCurrentGroup();
  await loadTask(current.qid);
}
async function loadTask(qid){
  if(current && current.qid!==qid) await autoSaveCurrentDraft();
  const data=await api(`/api/task/${qid}?user=${encodeURIComponent($('user').value)}`);
  current=data.task; activeAnnotator=isReviewer()?null:$('user').value;
  const displayAnn=isReviewer()?selectedQueryAnnotation():current.my_annotation;
  windows=(displayAnn&&displayAnn.windows)||[];
  windowScores=(displayAnn&&displayAnn.saliency)||windows.map(()=>4);
  clipScores=(displayAnn&&displayAnn.clip_scores)||{};
  expandedWindow=null;
  $('empty').style.display='none'; $('editor').style.display='flex'; $('groupPanel').style.display=currentGroup?'flex':'none';
  $('title').textContent=`标注项 #${current.qid}`;
  $('taskMeta').textContent=`${statusText(current.status)} / ${current.split||'-'}`;
  $('query').value=current.query||''; $('query').style.display='none';
  $('vid').value=current.vid||''; $('duration').value=current.duration||''; $('videoPath').value=current.video_path||''; $('notes').value=displayAnn?displayAnn.notes:'';
  if($('notes')) $('notes').oninput=scheduleAutoSave;
  $('video').src=current.video_path?`/media/${current.qid}`:'';
  renderWindows(); renderAnnotations(); renderGroupPanel(); renderQueryList(); renderSelectedQueryBox(); applyRoleUi();
}
function addWindow(){
  if(!queryConfirmed()) return alert('请先填写 query');
  let s=Number($('segStart').value),e=Number($('segEnd').value);
  if(!(e>s))return alert('end 必须大于 start');
  windows.push([s,e]); windowScores.push(4); sortWindows(); renderWindows(); scheduleAutoSave();
}
function renderWindows(){
  const rows=windows.map((w,i)=>{const clips=expandedWindow===i?`<tr><td colspan="5"><div class="clip-grid">${clipIdsForWindow(w).map(cid=>`<div class="clip-cell">clip ${cid}<br><small>${(cid*CLIP_LEN).toFixed(1)}-${((cid+1)*CLIP_LEN).toFixed(1)}s</small><br><input type="number" min="0" max="4" value="${clipScores[cid]??windowScores[i]??4}" data-clip="${cid}" onchange="scheduleAutoSave()"></div>`).join('')}</div></td></tr>`:''; return `<tr><td>${w[0]}</td><td>${w[1]}</td><td><input type="number" min="0" max="4" value="${windowScores[i]??4}" data-score="${i}" onchange="scheduleAutoSave()"></td><td><button onclick="expandedWindow=expandedWindow===${i}?null:${i};renderWindows()">clip 评分</button></td><td><button class="bad" onclick="windows.splice(${i},1);windowScores.splice(${i},1);renderWindows();scheduleAutoSave()">删除</button></td></tr>${clips}`}).join('');
  $('windows').innerHTML=rows; renderTimeline();
}
async function loadVideoTask(vid, preferredStatus){
  await autoSaveCurrentDraft();
  const task=pickTaskForVideo(vid, preferredStatus);
  if(task) await loadTask(task.qid);
}
async function prevGroupVideo(){const vids=videoGroups(); const i=currentVideoIndex(); if(i>0) await loadVideoTask(vids[i-1].vid);}
async function nextGroupVideo(){const vids=videoGroups(); const i=currentVideoIndex(); if(i>=0&&i<vids.length-1) await loadVideoTask(vids[i+1].vid);}
async function saveAnnotation(status){
  if(!current) return;
  if(!queryConfirmed()) return alert('请先填写 query');
  clearTimeout(autoSaveTimer);
  if(status==='submitted'){
    const body={qid:current.qid,user:$('user').value,annotator:$('user').value,duration:Number($('duration').value||$('video').duration||current.duration||0),windows,saliency:collectScoresFromDom(),clip_scores:clipScores,notes:($('notes')&&$('notes').value)||'',status:'draft'};
    const data=await api('/api/annotation',{method:'POST',body:JSON.stringify(body)});
    current=data.task;
  } else {
    await autoSaveCurrentDraft();
  }
  if(status==='submitted'){
    await submitCurrentVideoForReview();
    current=(await api(`/api/task/${current.qid}?user=${encodeURIComponent($('user').value)}`)).task;
    clearTimeout(autoSaveTimer);
    renderAnnotations(); renderWindows(); renderQueryList(); renderSelectedQueryBox(); refreshGroups();
    alert('已提交当前视频下全部 query 审核');
  } else {
    renderAnnotations(); renderWindows(); renderQueryList(); renderSelectedQueryBox(); refreshGroups();
  }
}
showAuth();
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Lightweight multi-user video annotation app.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--import_jsonl", action="append", default=[])
    parser.add_argument("--video_root", default=None)
    args = parser.parse_args()

    store = Store(args.db)
    for item in args.import_jsonl:
        if ":" in item:
            split, path = item.split(":", 1)
        else:
            split, path = None, item
        result = store.import_jsonl(path, split=split, video_root=args.video_root)
        print(f"imported {path}: {result}")

    AppHandler.store = store
    AppHandler.args = args
    server = ThreadingHTTPServer((args.host, args.port), AppHandler)
    print(f"Annotation app: http://{args.host}:{args.port}")
    print(f"Database: {args.db}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("bye")


if __name__ == "__main__":
    main()

