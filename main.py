# main.py
# =====================================================================================
# Chat con Gemini 1.5 Flash + Memoria SQLite (persistencia real)
# - UI:   interfaz.ChatWindow  (interfaz de chat + panel de usuario)
# - DB:   memoria.MemoryDB     (recuerda usuarios, temas y últimas Q&A)
# - LLM:  SOLO gemini-1.5-flash (o gemini-1.5-flash-latest si está disponible)
# - TOOLS: funciones en herramientas.py (cálculo, simbólico, unidades, química, etc.)
#
# Flujo:
#   • La app inicia sin sesión activa y muestra un mensaje de bienvenida.
#   • Desde el panel "Usuario" puedes iniciar sesión (o crear un usuario) y
#     cambiar entre usuarios sin cerrar la aplicación.
#   • Al cambiar de usuario, se recupera el último contexto y se muestra un saludo.
#   • Comandos /tools ejecutan utilidades especializadas (no requieren modificar UI/DB).
# =====================================================================================

from __future__ import annotations

import os
import sys
import re
from typing import Optional, Tuple, List

from dotenv import load_dotenv
load_dotenv()

from PyQt6 import QtCore, QtWidgets

# UI y DB
from interfaz import ChatWindow
from memoria import MemoryDB

# ======== HERRAMIENTAS (archivo externo herramientas.py) =============================
# Asegúrate de tener herramientas.py en el mismo directorio.
from herramientas import (
    calc, wiki,
    deriva, integra, limite, resuelve, simplifica,
    convierte, masa_molar, suvat,
    stats_lista, csv_head, csv_describe, csv_col_stats, csv_hist,
    plot, analizar_codigo
)

# Silenciar logs ruidosos gRPC/abseil
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("ABSL_LOGGING_MIN_LOG_LEVEL", "3")


# ============================== Utilidades de memoria ================================

def user_exists(db: MemoryDB, name: str) -> Tuple[bool, int]:
    """Devuelve (existe?, user_id). Crea el usuario si no existe."""
    conn = db._conn()
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE name=?;", (name,))
    row = cur.fetchone()
    conn.close()
    if row:
        return True, row[0]
    uid = db.get_or_create_user(name)
    return False, uid

def last_context_for_user(db: MemoryDB, user_id: int) -> Optional[Tuple[str, str, str]]:
    """
    Último (Materia, Tema, ÚltimaPregunta) para el usuario, si lo hay.
    Devuelve (subject, topic, last_question) o None si no hay historial.
    """
    conn = db._conn()
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

def list_users(db: MemoryDB) -> List[str]:
    conn = db._conn()
    cur = conn.cursor()
    cur.execute("SELECT name FROM users ORDER BY LOWER(name) ASC;")
    rows = [r[0] for r in cur.fetchall()]
    conn.close()
    return rows


# ============================== Backend en QThread ===================================

class Backend(QtCore.QObject):
    responseReady = QtCore.pyqtSignal(str)

    def __init__(self, db: MemoryDB,
                 user_name: str = "Invitado",
                 uid: Optional[int] = None,
                 initial_subject: str = "General",
                 initial_topic: str = "-"):
        super().__init__()
        self.db = db

        # Estado de usuario
        self.user_name = user_name
        self.uid = uid if uid is not None else self.db.get_or_create_user(user_name)

        # Contexto actual (memoria de sesión)
        self.current_subject: str = initial_subject or "General"
        self.current_topic: str = initial_topic or "-"

        # Gemini
        self.genai = None
        self.model_name = None
        self._configure_gemini_only_flash()

    # ------------------------- Configuración de Gemini (solo Flash) -------------------

    def _configure_gemini_only_flash(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            # Permitimos que la UI se inicie; avisaremos al usuario al recibir un mensaje.
            self.model_name = "gemini-2.5-flash"
            return

        import google.generativeai as genai
        self.genai = genai
        genai.configure(api_key=api_key)

        try:
            models = list(genai.list_models())
            supported = {m.name for m in models
                         if "generateContent" in (getattr(m, "supported_generation_methods", []) or [])}
            if "gemini-2.5-flash-latest" in supported:
                self.model_name = "gemini-2.5-flash-latest"
            elif "gemini-2.5-flash" in supported:
                self.model_name = "gemini-2.5-flash"
            else:
                self.model_name = next(iter(supported)) if supported else "gemini-2.5-flash"
        except Exception:
            self.model_name = "gemini-2.5-flash"

    # ------------------------------ Entrada desde la UI -------------------------------

    @QtCore.pyqtSlot(str)
    def handle_message(self, user_text: str):
        # 0) Validación de API
        if self.genai is None and not os.getenv("GOOGLE_API_KEY"):
            self.responseReady.emit(
                "⚠️ Falta configurar la clave de Google AI (variable `GOOGLE_API_KEY` en tu archivo .env). "
                "Sin eso no puedo generar respuestas."
            )
            return

        # 1) Comandos (materia/tema/herramientas)
        handled, tool_response, ctx_changed = self._handle_command(user_text)
        if handled:
            if tool_response:
                self.responseReady.emit(tool_response)
            if ctx_changed:
                self.responseReady.emit(
                    f"Contexto actualizado → **Materia:** {self.current_subject} · **Tema:** {self.current_topic}"
                )
            return

        # 2) Inferir materia/tema si no están fijados
        if self.current_subject == "General":
            self.current_subject = self._guess_subject(user_text)
        if self.current_topic in ("-", "", "general"):
            self.current_topic = self._guess_topic(user_text)

        # 3) Memoria: últimas 3 Q&A de este tema (persistidas en SQLite)
        mem_ctx = self._memory_context(self.current_subject, self.current_topic, k=3)

        # 4) Preguntar al modelo (temperatura baja)
        answer = self._ask_gemini(user_text, mem_ctx)

        # 5) Persistir
        tid = self.db.get_or_create_topic(self.current_subject, self.current_topic)
        self.db.log_doubt(self.uid, tid, user_text, answer)

        # 6) Responder
        self.responseReady.emit(answer)

    # Permite cambiar de usuario desde la UI
    @QtCore.pyqtSlot(str)
    def change_user(self, name: str):
        name = (name or "").strip() or "Invitado"
        existed, uid = user_exists(self.db, name)
        self.user_name = name
        self.uid = uid

        # Reestablecer contexto según último uso
        ctx = last_context_for_user(self.db, uid) if existed else None
        if ctx:
            subj, topic, last_q = ctx
            self.current_subject, self.current_topic = subj, topic
            greet = (f"Bienvenido de vuelta, **{name}**.\n"
                     f"Última sesión: **{subj} / {topic}**.\n"
                     f"Tu última pregunta fue: “{last_q}”.\n"
                     f"Puedes continuar o cambiar el contexto con:\n"
                     f"• `/materia NuevaMateria`\n"
                     f"• `/tema NuevoTema`")
        else:
            self.current_subject, self.current_topic = "General", "-"
            greet = (f"Hola, **{name}**. Ya puedes escribir tu consulta.\n"
                     f"Para fijar el contexto usa:\n"
                     f"• `/materia Calculo`\n"
                     f"• `/tema Limites laterales`")
        self.responseReady.emit(greet)

    # --------------------------- Lógica del LLM (solo Flash) --------------------------

    def _ask_gemini(self, user_text: str, memory_ctx: str) -> str:
        prompt = (
            "Eres profesor asesor de ciencias básicas de ingeniería en IBERO Puebla.\n"
            "Explica con claridad, de forma paso a paso y concisa. Usa notación simple (puedes usar ^).\n"
            "Si es un ejercicio numérico: muestra fórmula, sustitución con unidades y resultado con unidades.\n"
            "Si es teórico: da una definición breve, una analogía y un ejemplo mínimo.\n"
            "Si faltan datos, solicita solo los indispensables. No inventes datos.\n"
            "Evita paja y listas interminables.\n\n"
            f"Estudiante: {self.user_name}\n"
            f"Materia: {self.current_subject}\n"
            f"Tema: {self.current_topic}\n\n"
            "Memoria (últimas 3 interacciones en este tema):\n"
            f"{memory_ctx or '—'}\n\n"
            "Pregunta del estudiante:\n"
            f"{user_text}\n\n"
            "Responde en español y procura no exceder 10–15 líneas."
        )
        try:
            generation_config = {
                "temperature": 0.2,
                "top_p": 0.9,
                "top_k": 40,
                "max_output_tokens": 700,
            }
            resp = self.genai.GenerativeModel(self.model_name, generation_config=generation_config) \
                              .generate_content(prompt)
            if hasattr(resp, "text") and resp.text:
                return resp.text.strip()
            if getattr(resp, "candidates", None):
                for cand in resp.candidates:
                    parts = getattr(cand, "content", None)
                    if parts and getattr(parts, "parts", None):
                        txt = "".join(getattr(p, "text", "") or "" for p in parts.parts)
                        if txt.strip():
                            return txt.strip()
            return "No pude generar una respuesta clara. Intenta reformular tu pregunta o añade datos/unidades."
        except Exception as e:
            return (f"No pude consultar el modelo ({self.model_name}). ¿Está correcta tu clave de API?\n"
                    f"Detalle técnico: {e}")

    # ---------------------------- Memoria persistente ---------------------------------

    def _memory_context(self, subject: str, topic: str, k: int = 3) -> str:
        tid = self.db.get_or_create_topic(subject, topic or "-")
        rows = self.db.recent_doubts(self.uid, tid, limit=k)
        if not rows:
            return ""
        chunks = []
        for ts, q, a in rows:
            a = a.replace("\n", " ").strip()
            if len(a) > 300:
                a = a[:297] + "..."
            chunks.append(f"- [{ts}] P: {q}\n  R: {a}")
        return "\n".join(chunks)

    # ----------------------- Comandos / materia / tema / herramientas -----------------

    def _handle_command(self, text: str) -> Tuple[bool, Optional[str], bool]:
        """
        Devuelve (handled, response, ctx_changed)
        - handled: si se procesó algún comando
        - response: texto a mostrar (si aplica)
        - ctx_changed: si cambió materia/tema para avisar a la UI
        """
        t = text.strip()

        # /materia X
        m = re.match(r"^\s*/materia\s+(.+)$", t, flags=re.I)
        if m:
            self.current_subject = m.group(1).strip()
            return True, None, True

        # /tema X
        m = re.match(r"^\s*/tema\s+(.+)$", t, flags=re.I)
        if m:
            self.current_topic = m.group(1).strip()
            return True, None, True

        # ===================== Herramientas =====================

        # Calculadora segura
        m = re.match(r"^\s*/calc\s+(.+)$", t, flags=re.I)
        if m:
            return True, calc(m.group(1).strip()), False

        # Wikipedia
        m = re.match(r"^\s*/wiki\s+(.+)$", t, flags=re.I)
        if m:
            return True, wiki(m.group(1).strip()), False

        # Cálculo simbólico (SymPy)
        m = re.match(r"^\s*/deriva\s+(.+?)(?:\s+([a-zA-Z]))?$", t, flags=re.I)
        if m:
            expr, var = m.group(1), (m.group(2) or "x")
            return True, deriva(expr, var), False

        m = re.match(r"^\s*/integra\s+(.+?)(?:\s+([a-zA-Z]))?$", t, flags=re.I)
        if m:
            expr, var = m.group(1), (m.group(2) or "x")
            return True, integra(expr, var), False

        m = re.match(r"^\s*/limite\s+(.+?)\s+([a-zA-Z])\s*->\s*([^ ]+)(?:\s+([+-]))?$", t, flags=re.I)
        if m:
            expr, var, at, direction = m.group(1), m.group(2), m.group(3), m.group(4)
            return True, limite(expr, var, at, direction), False

        m = re.match(r"^\s*/resuelve\s+(.+)$", t, flags=re.I)
        if m:
            return True, resuelve(m.group(1)), False

        m = re.match(r"^\s*/simplifica\s+(.+)$", t, flags=re.I)
        if m:
            return True, simplifica(m.group(1)), False

        # Unidades (pint)
        m = re.match(r"^\s*/u\s+(.+)$", t, flags=re.I)
        if m:
            return True, convierte(m.group(1)), False

        # Química: masa molar
        m = re.match(r"^\s*/mm\s+([A-Za-z0-9()]+)$", t, flags=re.I)
        if m:
            return True, masa_molar(m.group(1)), False

        # Física: SUVAT (u,v,a,t,s como pares k=v; se necesitan >=3)
        if t.lower().startswith("/suvat"):
            pairs = re.findall(r"(u|v|a|t|s)\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", t)
            if pairs:
                vals = {k: float(v) for k, v in pairs}
                return True, suvat(**vals), False

        # Estadística/Datos
        m = re.match(r"^\s*/stats\s+(.+)$", t, flags=re.I)
        if m:
            return True, stats_lista(m.group(1)), False

        m = re.match(r"^\s*/csv\s+head\s+(.+?)(?:\s+(\d+))?$", t, flags=re.I)
        if m:
            path, n = m.group(1), int(m.group(2) or 5)
            return True, csv_head(path, n), False

        m = re.match(r"^\s*/csv\s+describe\s+(.+)$", t, flags=re.I)
        if m:
            return True, csv_describe(m.group(1)), False

        m = re.match(r"^\s*/csv\s+col\s+(.+?)\s+(.+)$", t, flags=re.I)
        if m:
            return True, csv_col_stats(m.group(1), m.group(2)), False

        m = re.match(r"^\s*/csv\s+hist\s+(.+?)\s+(.+?)(?:\s+(\d+))?$", t, flags=re.I)
        if m:
            path, col, bins = m.group(1), m.group(2), int(m.group(3) or 20)
            return True, csv_hist(path, col, bins), False

        # Graficador
        m = re.match(r"^\s*/plot\s+y=(.+?)\s+x:([^ ]+)$", t, flags=re.I)
        if m:
            expr, xspec = m.group(1), m.group(2)
            return True, plot(expr, xspec), False

        # Programación: análisis estático
        # Opción 1: /analiza ```python\n...\n```
        m = re.match(r"^\s*/analiza\s+```(?:python)?\n([\s\S]+?)\n```$", t, flags=re.I)
        if m:
            return True, analizar_codigo(m.group(1)), False
        # Opción 2: /analiza <código en una línea>
        m = re.match(r"^\s*/analiza\s+(.+)$", t, flags=re.I)
        if m:
            return True, analizar_codigo(m.group(1)), False

        return False, None, False

    def _guess_subject(self, text: str) -> str:
        t = text.lower()
        if any(w in t for w in ["deriv", "integral", "límite", "limite", "serie", "teorema fundamental"]):
            return "Cálculo"
        if any(w in t for w in ["matriz", "vector", "autovalor", "autovector", "diagonalizar"]):
            return "Álgebra Lineal"
        if any(w in t for w in ["fuerza", "velocidad", "aceleración", "newton", "circuito", "ohm", "voltaje", "campo"]):
            return "Física"
        if any(w in t for w in ["mol", "mole", "reacción", "estequiometría", "ácido", "base", "ph", "equilibrio"]):
            return "Química"
        if any(w in t for w in ["probabilidad", "estadística", "media", "varianza", "distribución"]):
            return "Probabilidad y Estadística"
        if any(w in t for w in ["programación", "código", "algoritmo", "complejidad", "python"]):
            return "Programación"
        return "General"

    def _guess_topic(self, text: str) -> str:
        keys = [
            "límite", "limite", "derivada", "integral", "teorema fundamental", "series",
            "matriz", "vector", "autovalor", "autovector",
            "ley de ohm", "campo eléctrico", "segunda ley de newton",
            "ph", "ácido", "base", "estequiometría",
            "probabilidad", "distribución normal", "varianza", "regresión",
            "complejidad", "algoritmo"
        ]
        t = text.lower()
        for k in keys:
            if k in t:
                return k
        return " ".join(text.split()[:5]).strip() or "general"


# ================================== Entrypoint =======================================

def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Profesor Asesor — Chat (Memoria + Herramientas)")

    # Instancias
    db = MemoryDB()
    win = ChatWindow()

    # Mensaje de bienvenida (sin sesión)
    win.append_assistant(
        "¡Bienvenido! Soy tu **Profesor Asesor** para Cálculo, Física, Química, "
        "Álgebra Lineal, Probabilidad/Estadística y Programación.\n\n"
        "1) Abre el panel **Usuario** (arriba derecha) para iniciar/cambiar sesión.\n"
        "2) Opcional: fija el contexto con:\n"
        "   • `/materia Calculo`\n"
        "   • `/tema Limites laterales`\n\n"
        "3) **Herramientas disponibles** (ejemplos):\n"
        "   • `/calc 2*(3+4)^2`\n"
        "   • `/wiki Transformada de Laplace`\n"
        "   • `/deriva sin(x)^2 x` · `/integra e^(2x) x` · `/limite (sin(x))/x x->0`\n"
        "   • `/resuelve x^2-5x+6=0` · `/simplifica (x^2-1)/(x-1)`\n"
        "   • `/u 60 km/h -> m/s` · `/mm Ca(OH)2`\n"
        "   • `/suvat u=0 a=2 t=10`\n"
        "   • `/stats 1,2,2,3,5,8`\n"
        "   • `/csv head datos.csv 10` · `/csv describe datos.csv`\n"
        "   • `/csv col datos.csv temperatura` · `/csv hist datos.csv temperatura 30`\n"
        "   • `/plot y=sin(x)+x^2 x:-2*pi:2*pi`\n"
        "   • `/analiza def f(x):\\n    return x*x`"
    )

    # Backend en hilo (inicia con usuario "Invitado")
    thread = QtCore.QThread()
    backend = Backend(db=db, user_name="Invitado")
    backend.moveToThread(thread)

    # Conexiones Chat
    win.sendMessage.connect(backend.handle_message)
    backend.responseReady.connect(win.append_assistant)

    # Conexiones de Usuario
    win.changeUser.connect(backend.change_user)
    win.requestUserList.connect(lambda: win.set_users(list_users(db)))

    # Estado inicial de la UI
    win.set_current_user("Invitado")
    win.set_users(list_users(db))

    app.aboutToQuit.connect(thread.quit)
    thread.start()

    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
