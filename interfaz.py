# interfaz.py
# --------------------------------------------------------------------------------------
# Interfaz de chat (solo UI) con PyQt6
# - Tema oscuro elegante
# - Burbujas para usuario y asistente
# - Envío con Enter (Shift+Enter = salto de línea)
# - Señales:
#       sendMessage(str)   -> para enviar el prompt al backend
#       changeUser(str)    -> para iniciar sesión / cambiar de usuario
#       requestUserList()  -> para pedir al backend refrescar la lista de usuarios
#
# NOTA: Esta UI no incluye lógica de LLM ni mensajes iniciales.
# --------------------------------------------------------------------------------------

from __future__ import annotations

from PyQt6 import QtCore, QtGui, QtWidgets

# ============================ Estilos (QSS) ==========================================

DARK_QSS = """
* { font-family: 'Segoe UI', 'Inter', 'Roboto', sans-serif; }
QMainWindow { background: #0f1115; }
QFrame#TopBar {
    background: #0c0f14; border-bottom: 1px solid #1b2130;
}
QLabel#Title { color: #EAF0FF; font-size: 16px; font-weight: 800; }
QLabel#UserBadge {
    color: #C7D3FF; background: #141927; border: 1px solid #2a3a6b;
    padding: 4px 10px; border-radius: 10px; font-weight: 700; font-size: 12px;
}
QScrollArea { border: none; }
QTextEdit {
    background: #0f141d; color: #EAF0FF; border: 1px solid #243047;
    border-radius: 10px; padding: 10px; selection-background-color:#274060;
}
QPushButton {
    background: #1b2540; color: #EAF0FF; border: 1px solid #2a3a6b;
    border-radius: 10px; padding: 10px 14px; font-weight: 700;
}
QPushButton:hover { background: #23315a; }
QPushButton#Primary {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #4C7EFF, stop:1 #8D60FF);
    border: 0; color: white;
}
QToolButton {
    background: transparent; color:#8aa0d6; border:1px dashed #2a3a6b; border-radius:8px; padding:6px 10px;
}
QFrame#UserBar {
    background: #0e131c; border-bottom: 1px solid #1b2130;
}
QLineEdit {
    background: #0f141d; color: #EAF0FF; border: 1px solid #243047;
    border-radius: 8px; padding: 8px 10px;
}
QComboBox {
    background: #0f141d; color: #EAF0FF; border: 1px solid #243047;
    border-radius: 8px; padding: 6px 8px;
}
QLabel#Hint { color:#8aa0d6; font-size:11px; }
"""

# ============================ Burbuja de chat ========================================

class ChatBubble(QtWidgets.QFrame):
    """Burbuja para 'assistant' o 'user'."""
    def __init__(self, text: str, role: str = "assistant", parent=None):
        super().__init__(parent)
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)

        is_assistant = (role == "assistant")
        bg = "#162032" if is_assistant else "#1a1f2b"
        border = "#31405e" if is_assistant else "#3a4a6d"

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)

        container = QtWidgets.QFrame()
        container.setStyleSheet(
            f"QFrame {{background:{bg}; border:1px solid {border}; border-radius:12px;}}"
            "QLabel {color:#EAF0FF; padding:8px;}"
        )
        v = QtWidgets.QVBoxLayout(container)
        v.setContentsMargins(10, 8, 10, 8)

        label = QtWidgets.QLabel(text)
        label.setWordWrap(True)
        v.addWidget(label)

        who = QtWidgets.QLabel("Profesor Asesor · IA" if is_assistant else "Tú")
        who.setStyleSheet("color:#8aa0d6; font-size:11px;")
        v.addWidget(who, alignment=QtCore.Qt.AlignmentFlag.AlignRight)

        if is_assistant:
            layout.addWidget(container, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
            layout.addStretch()
        else:
            layout.addStretch()
            layout.addWidget(container, 0, QtCore.Qt.AlignmentFlag.AlignRight)

# ============================ Ventana principal ======================================

class ChatWindow(QtWidgets.QMainWindow):
    """
    Interfaz de solo chat + barra de usuario (login/cambio de usuario).
    Conecta:
        - sendMessage(str)  al backend de LLM
        - changeUser(str)   a tu manejador de sesión/DB
        - requestUserList() para que el backend devuelva lista de usuarios con set_users(...)
    """

    sendMessage = QtCore.pyqtSignal(str)
    changeUser = QtCore.pyqtSignal(str)
    requestUserList = QtCore.pyqtSignal()

    def __init__(self, *, title: str = "Profesor Asesor — Chat"):
        super().__init__()
        self.setWindowTitle(title)
        self.resize(980, 690)
        self.setStyleSheet(DARK_QSS)
        self._build_ui()
        self._connect_signals()

        # Estado
        self._current_user: str = "Sin sesión"

    # ---------------- UI ----------------

    def _build_ui(self):
        central = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Top bar
        top = QtWidgets.QFrame()
        top.setObjectName("TopBar")
        ht = QtWidgets.QHBoxLayout(top)
        ht.setContentsMargins(16, 10, 16, 10)
        ht.setSpacing(8)

        self.lbl_title = QtWidgets.QLabel("Profesor Asesor – Chat")
        self.lbl_title.setObjectName("Title")
        ht.addWidget(self.lbl_title)

        ht.addStretch()

        self.lbl_user_badge = QtWidgets.QLabel("👤 Sin sesión")
        self.lbl_user_badge.setObjectName("UserBadge")
        ht.addWidget(self.lbl_user_badge)

        self.btn_toggle_user = QtWidgets.QToolButton()
        self.btn_toggle_user.setText("Usuario")
        self.btn_toggle_user.setToolTip("Mostrar/Ocultar panel de usuario")
        ht.addWidget(self.btn_toggle_user)

        self.btn_clear = QtWidgets.QToolButton()
        self.btn_clear.setText("Limpiar chat")
        ht.addWidget(self.btn_clear)

        root.addWidget(top)

        # ==== Barra de Usuario (colapsable) ==========================================
        self.user_bar = QtWidgets.QFrame()
        self.user_bar.setObjectName("UserBar")
        self.user_bar.setVisible(False)

        ub = QtWidgets.QGridLayout(self.user_bar)
        ub.setContentsMargins(16, 10, 16, 10)
        ub.setHorizontalSpacing(10)
        ub.setVerticalSpacing(6)

        # Columna 1: Crear / entrar con nombre
        lbl_new = QtWidgets.QLabel("Iniciar sesión / crear usuario")
        lbl_new.setStyleSheet("color:#EAF0FF; font-weight:700;")
        ub.addWidget(lbl_new, 0, 0, 1, 2)

        self.le_user = QtWidgets.QLineEdit()
        self.le_user.setPlaceholderText("Escribe tu nombre…")
        self.le_user.setClearButtonEnabled(True)
        ub.addWidget(self.le_user, 1, 0)

        self.btn_login = QtWidgets.QPushButton("Entrar")
        self.btn_login.setObjectName("Primary")
        ub.addWidget(self.btn_login, 1, 1)

        hint = QtWidgets.QLabel("Tip: Enter para entrar rápidamente")
        hint.setObjectName("Hint")
        ub.addWidget(hint, 2, 0, 1, 2)

        # Columna 2: Elegir de existentes
        lbl_exist = QtWidgets.QLabel("Cambiar a usuario existente")
        lbl_exist.setStyleSheet("color:#EAF0FF; font-weight:700;")
        ub.addWidget(lbl_exist, 0, 3, 1, 2)

        self.cb_users = QtWidgets.QComboBox()
        self.cb_users.setMinimumWidth(220)
        ub.addWidget(self.cb_users, 1, 3)

        self.btn_switch = QtWidgets.QPushButton("Cambiar")
        ub.addWidget(self.btn_switch, 1, 4)

        self.btn_refresh_users = QtWidgets.QToolButton()
        self.btn_refresh_users.setText("Actualizar lista")
        ub.addWidget(self.btn_refresh_users, 2, 3, 1, 2, alignment=QtCore.Qt.AlignmentFlag.AlignLeft)

        root.addWidget(self.user_bar)

        # ==== Área de chat (scroll) ===================================================
        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.chat_container = QtWidgets.QWidget()
        self.chat_layout = QtWidgets.QVBoxLayout(self.chat_container)
        self.chat_layout.setContentsMargins(12, 12, 12, 12)
        self.chat_layout.setSpacing(8)
        self.chat_layout.addStretch()
        self.scroll.setWidget(self.chat_container)
        root.addWidget(self.scroll, 1)

        # ==== Input ==================================================================
        bottom = QtWidgets.QFrame()
        hb = QtWidgets.QHBoxLayout(bottom)
        hb.setContentsMargins(12, 10, 12, 12)
        hb.setSpacing(8)

        self.txt_input = QtWidgets.QTextEdit()
        self.txt_input.setMinimumHeight(70)
        self.txt_input.setPlaceholderText("Escribe tu pregunta aquí…  (Shift+Enter = salto de línea)")
        hb.addWidget(self.txt_input, 1)

        self.btn_send = QtWidgets.QPushButton("Enviar")
        self.btn_send.setObjectName("Primary")
        self.btn_send.setFixedHeight(42)
        hb.addWidget(self.btn_send)

        root.addWidget(bottom, 0)
        self.setCentralWidget(central)

    def _connect_signals(self):
        # Chat
        self.btn_send.clicked.connect(self._on_send_clicked)
        self.btn_clear.clicked.connect(self.clear_chat)
        self.txt_input.installEventFilter(self)

        # User bar
        self.btn_toggle_user.clicked.connect(self._toggle_user_bar)
        self.btn_login.clicked.connect(self._on_login_clicked)
        self.btn_switch.clicked.connect(self._on_switch_clicked)
        self.btn_refresh_users.clicked.connect(self.requestUserList.emit)
        self.le_user.returnPressed.connect(self._on_login_clicked)

    # -------------- API pública --------------

    def append_user(self, text: str):
        self._append_bubble(text, role="user")

    def append_assistant(self, text: str):
        self._append_bubble(text, role="assistant")

    def clear_chat(self):
        # Elimina todas las burbujas (deja el stretch final)
        for i in reversed(range(self.chat_layout.count() - 1)):
            w = self.chat_layout.itemAt(i).widget()
            if w:
                w.setParent(None)

    def set_header_title(self, text: str):
        """Cambia el título visible en la barra superior (no el de la ventana)."""
        self.lbl_title.setText(text)

    def set_current_user(self, name: str):
        """Actualiza el badge del usuario actual."""
        self._current_user = name or "Sin sesión"
        self.lbl_user_badge.setText(f"👤 {self._current_user}")

    def set_users(self, users: list[str]):
        """Rellena la lista de usuarios existentes (no selecciona)."""
        self.cb_users.blockSignals(True)
        self.cb_users.clear()
        self.cb_users.addItems(users or [])
        self.cb_users.blockSignals(False)

    # -------------- Lógica interna de UI --------------

    def _append_bubble(self, text: str, role: str):
        # Inserta antes del stretch
        idx = self.chat_layout.count() - 1
        self.chat_layout.insertWidget(idx, ChatBubble(text, role))
        QtCore.QTimer.singleShot(
            0,
            lambda: self.scroll.verticalScrollBar().setValue(self.scroll.verticalScrollBar().maximum())
        )

    def _on_send_clicked(self):
        text = self.txt_input.toPlainText().strip()
        if not text:
            return
        self.append_user(text)
        self.txt_input.clear()
        self.sendMessage.emit(text)

    def _toggle_user_bar(self):
        self.user_bar.setVisible(not self.user_bar.isVisible())
        if self.user_bar.isVisible():
            # Al abrir, pedir lista de usuarios para tenerla fresca
            self.requestUserList.emit()
            QtCore.QTimer.singleShot(0, self.le_user.setFocus)

    def _on_login_clicked(self):
        name = self.le_user.text().strip()
        if not name:
            self.le_user.setFocus()
            return
        self.changeUser.emit(name)
        self.le_user.clear()
        self.set_current_user(name)

    def _on_switch_clicked(self):
        name = self.cb_users.currentText().strip()
        if not name:
            return
        self.changeUser.emit(name)
        self.set_current_user(name)

    # Enter para enviar (Shift+Enter = salto de línea)
    def eventFilter(self, obj, event):
        if obj is self.txt_input and event.type() == QtCore.QEvent.Type.KeyPress:
            if event.key() in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter):
                if event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier:
                    return super().eventFilter(obj, event)
                self._on_send_clicked()
                return True
        return super().eventFilter(obj, event)
