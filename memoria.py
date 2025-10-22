# memoria.py
# =====================================================================================
# Capa de persistencia (SQLite) para el "Profesor Asesor" de ciencias básicas.
# Guarda usuarios, dudas, sesiones de quiz y respuestas pregunta a pregunta.
# Provee utilidades para estadísticas y progreso.
# =====================================================================================

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any


DEFAULT_DB_PATH = "asesor_memoria.sqlite3"


@dataclass
class QuizQuestionLog:
    idx: int
    question: str
    options: List[str]
    correct_index: int
    user_answer_index: Optional[int]
    is_correct: Optional[bool]
    explanation: str


class MemoryDB:
    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        self._init_db()

    # -------------------------- Inicialización de tablas --------------------------

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL
        );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS topics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject TEXT NOT NULL,
            topic TEXT NOT NULL,
            UNIQUE(subject, topic)
        );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS doubts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            topic_id INTEGER NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id),
            FOREIGN KEY(topic_id) REFERENCES topics(id)
        );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS quiz_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            topic_id INTEGER NOT NULL,
            difficulty TEXT NOT NULL,
            n_questions INTEGER NOT NULL,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id),
            FOREIGN KEY(topic_id) REFERENCES topics(id)
        );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS quiz_questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            q_index INTEGER NOT NULL,
            question TEXT NOT NULL,
            options_json TEXT NOT NULL,
            correct_index INTEGER NOT NULL,
            user_answer_index INTEGER,
            is_correct INTEGER,
            explanation TEXT,
            FOREIGN KEY(session_id) REFERENCES quiz_sessions(id)
        );
        """)

        # Índices útiles
        cur.execute("CREATE INDEX IF NOT EXISTS idx_doubts_user ON doubts(user_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_doubts_topic ON doubts(topic_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_quiz_sess_user_topic ON quiz_sessions(user_id, topic_id);")

        conn.commit()
        conn.close()

    # ------------------------------ Helpers genéricos ------------------------------

    def _conn(self):
        return sqlite3.connect(self.db_path)

    # -------------------------------- Usuarios/Temas --------------------------------

    def get_or_create_user(self, name: str) -> int:
        conn = self._conn()
        cur = conn.cursor()
        cur.execute("SELECT id FROM users WHERE name=?;", (name,))
        row = cur.fetchone()
        if row:
            uid = row[0]
        else:
            cur.execute("INSERT INTO users (name) VALUES (?);", (name,))
            conn.commit()
            uid = cur.lastrowid
        conn.close()
        return uid

    def get_user_id(self, name: str) -> Optional[int]:
        """Devuelve el id del usuario si existe; si no, None."""
        conn = self._conn()
        cur = conn.cursor()
        cur.execute("SELECT id FROM users WHERE name=?;", (name,))
        row = cur.fetchone()
        conn.close()
        return row[0] if row else None

    def list_users(self) -> List[str]:
        """Lista de nombres de usuario (orden alfabético, sin duplicados)."""
        conn = self._conn()
        cur = conn.cursor()
        cur.execute("SELECT name FROM users ORDER BY LOWER(name) ASC;")
        rows = [r[0] for r in cur.fetchall()]
        conn.close()
        return rows

    def rename_user(self, old_name: str, new_name: str) -> bool:
        """Renombra un usuario. Devuelve True si se actualizó."""
        conn = self._conn()
        cur = conn.cursor()
        try:
            cur.execute("UPDATE users SET name=? WHERE name=?;", (new_name, old_name))
            conn.commit()
            return cur.rowcount > 0
        except sqlite3.IntegrityError:
            # nuevo nombre ya existe (UNIQUE)
            return False
        finally:
            conn.close()

    def delete_user(self, name: str) -> bool:
        """
        Elimina un usuario por nombre. NO borra sus dudas o quizzes (conservar historial).
        Devuelve True si se eliminó alguna fila.
        """
        conn = self._conn()
        cur = conn.cursor()
        cur.execute("DELETE FROM users WHERE name=?;", (name,))
        conn.commit()
        deleted = cur.rowcount > 0
        conn.close()
        return deleted

    def get_or_create_topic(self, subject: str, topic: str) -> int:
        conn = self._conn()
        cur = conn.cursor()
        cur.execute("SELECT id FROM topics WHERE subject=? AND topic=?;", (subject, topic))
        row = cur.fetchone()
        if row:
            tid = row[0]
        else:
            cur.execute("INSERT INTO topics (subject, topic) VALUES (?, ?);", (subject, topic))
            conn.commit()
            tid = cur.lastrowid
        conn.close()
        return tid

    # ------------------------------------- Dudas -------------------------------------

    def log_doubt(self, user_id: int, topic_id: int, question: str, answer: str) -> int:
        conn = self._conn()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO doubts (user_id, topic_id, question, answer, timestamp)
            VALUES (?, ?, ?, ?, ?);
        """, (user_id, topic_id, question, answer, datetime.utcnow().isoformat()))
        conn.commit()
        did = cur.lastrowid
        conn.close()
        return did

    def recent_doubts(self, user_id: int, topic_id: int, limit: int = 10) -> List[Tuple[str, str, str]]:
        """Devuelve lista [(timestamp, question, answer), ...]"""
        conn = self._conn()
        cur = conn.cursor()
        cur.execute("""
            SELECT timestamp, question, answer
            FROM doubts
            WHERE user_id=? AND topic_id=?
            ORDER BY timestamp DESC
            LIMIT ?;
        """, (user_id, topic_id, limit))
        rows = cur.fetchall()
        conn.close()
        return rows

    def last_context_for_user(self, user_id: int) -> Optional[Tuple[str, str, str]]:
        """
        Último (Materia, Tema, ÚltimaPregunta) para el usuario, si lo hay.
        Devuelve (subject, topic, last_question) o None.
        """
        conn = self._conn()
        cur = conn.cursor()
        cur.execute("""
            SELECT t.subject, t.topic, d.question
            FROM doubts d
            JOIN topics t ON t.id = d.topic_id
            WHERE d.user_id=?
            ORDER BY d.timestamp DESC
            LIMIT 1;
        """, (user_id,))
        row = cur.fetchone()
        conn.close()
        if row:
            return row[0], row[1], row[2]
        return None

    # ------------------------------------ Quizzes ------------------------------------

    def start_quiz_session(self, user_id: int, topic_id: int, difficulty: str, n_questions: int) -> int:
        conn = self._conn()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO quiz_sessions (user_id, topic_id, difficulty, n_questions, started_at)
            VALUES (?, ?, ?, ?, ?);
        """, (user_id, topic_id, difficulty, n_questions, datetime.utcnow().isoformat()))
        conn.commit()
        sid = cur.lastrowid
        conn.close()
        return sid

    def finish_quiz_session(self, session_id: int):
        conn = self._conn()
        cur = conn.cursor()
        cur.execute("""
            UPDATE quiz_sessions SET finished_at=? WHERE id=?;
        """, (datetime.utcnow().isoformat(), session_id))
        conn.commit()
        conn.close()

    def log_quiz_question(self, session_id: int, q: QuizQuestionLog):
        conn = self._conn()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO quiz_questions
            (session_id, q_index, question, options_json, correct_index,
             user_answer_index, is_correct, explanation)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """, (
            session_id, q.idx, q.question, json.dumps(q.options, ensure_ascii=False),
            q.correct_index,
            q.user_answer_index if q.user_answer_index is not None else None,
            1 if q.is_correct else (0 if q.is_correct is not None else None),
            q.explanation
        ))
        conn.commit()
        conn.close()

    def update_quiz_answer(self, session_id: int, q_index: int,
                           user_answer_index: int, is_correct: bool, explanation: str = ""):
        conn = self._conn()
        cur = conn.cursor()
        cur.execute("""
            UPDATE quiz_questions
            SET user_answer_index=?, is_correct=?, explanation=COALESCE(NULLIF(?, ''), explanation)
            WHERE session_id=? AND q_index=?;
        """, (user_answer_index, 1 if is_correct else 0, explanation, session_id, q_index))
        conn.commit()
        conn.close()

    # ------------------------------- Estadísticas / Progreso -------------------------

    def topic_stats(self, user_id: int, topic_id: int) -> Dict[str, Any]:
        """Totales y accuracy histórico para el tema."""
        conn = self._conn()
        cur = conn.cursor()
        cur.execute("""
            SELECT COUNT(*), SUM(is_correct)
            FROM quiz_questions
            WHERE session_id IN (SELECT id FROM quiz_sessions WHERE user_id=? AND topic_id=?);
        """, (user_id, topic_id))
        total, correct = cur.fetchone()
        total = total or 0
        correct = correct or 0
        acc = (correct / total * 100.0) if total > 0 else 0.0
        conn.close()
        return {"total": total, "correct": correct, "accuracy": acc}

    def progress_blocks(self, user_id: int, topic_id: int, block_size: int = 5) -> List[Tuple[int, float]]:
        """
        Accuracy en bloques de 'block_size' preguntas, ordenadas cronológicamente.
        Devuelve [(bloque, accuracy%), ...]
        """
        conn = self._conn()
        cur = conn.cursor()
        cur.execute("""
            SELECT q.is_correct
            FROM quiz_questions q
            JOIN quiz_sessions s ON s.id = q.session_id
            WHERE s.user_id=? AND s.topic_id=?
            ORDER BY s.started_at ASC, q.q_index ASC;
        """, (user_id, topic_id))
        rows = cur.fetchall()
        conn.close()

        if not rows:
            return []

        values = [0 if r[0] is None else int(r[0]) for r in rows]
        blocks = []
        b = 1
        for i in range(0, len(values), block_size):
            chunk = values[i:i + block_size]
            acc = sum(chunk) / len(chunk) * 100.0
            blocks.append((b, acc))
            b += 1
        return blocks
