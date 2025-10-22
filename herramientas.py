# herramientas.py
# =============================================================================
# Herramientas para un "Profesor Asesor" de ciencias básicas.
# Incluye:
# - Calc segura (aritmética)
# - Wikipedia (resumen del primer resultado)
# - Cálculo simbólico (SymPy): derivar, integrar, límites, resolver, simplificar
# - Unidades (pint): conversión dimensional
# - Química: masa molar para fórmulas sencillas (H2O, Ca(OH)2, C6H12O6, etc.)
# - Física: solver de cinemática (SUVAT)
# - Estadística y datos (numpy/pandas): describe, head, stats de columna, histograma
# - Graficador simple (matplotlib) de y = f(x)
# - Programación: análisis estático básico (complejidad ciclomat., funciones, clases)
#
# Cada función devuelve texto amigable (y en algunos casos una ruta de imagen generada).
# =============================================================================

from __future__ import annotations

import os
import re
import uuid
import ast
import math
import operator as op
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import requests
import sympy as sp
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no necesita servidor de ventanas
import matplotlib.pyplot as plt
import pint

# ============================= CALCULADORA SEGURA ==============================

_ALLOWED = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
    ast.Mod: op.mod, ast.Pow: op.pow, ast.USub: op.neg, ast.UAdd: op.pos
}

def _eval_ast_numexpr(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Constante no numérica.")
    if isinstance(node, ast.BinOp):
        fn = _ALLOWED.get(type(node.op))
        if not fn: raise ValueError("Operador no permitido.")
        return fn(_eval_ast_numexpr(node.left), _eval_ast_numexpr(node.right))
    if isinstance(node, ast.UnaryOp):
        fn = _ALLOWED.get(type(node.op))
        if not fn: raise ValueError("Operador unario no permitido.")
        return fn(_eval_ast_numexpr(node.operand))
    if isinstance(node, ast.Expr):
        return _eval_ast_numexpr(node.value)
    raise ValueError("Expresión no permitida.")

def calc(expr: str) -> str:
    """
    Calculadora aritmética segura (+, -, *, /, **, %, paréntesis).
    Uso: /calc 2*(3+4)^2  -> usa ** en vez de ^
    """
    expr = expr.replace("^", "**")
    try:
        node = ast.parse(expr, mode="eval")
        val = _eval_ast_numexpr(node.body)
        if isinstance(val, float):
            # formato bonito
            val = float(f"{val:.10g}")
        return f" Resultado: **{val}**"
    except Exception as e:
        return f" No pude evaluar la expresión. Detalle: {e}"

# ================================ WIKIPEDIA ===================================

def wiki(query: str, lang: str = "es") -> str:
    """
    Devuelve el extracto introductorio del primer resultado relevante en Wikipedia.
    """
    try:
        s = requests.get(
            f"https://{lang}.wikipedia.org/w/api.php",
            params={"action": "query", "list": "search", "srsearch": query,
                    "utf8": "1", "format": "json", "srlimit": 1},
            timeout=10
        ).json()
        hits = s.get("query", {}).get("search", [])
        if not hits:
            return "No encontré resultados en Wikipedia."
        title = hits[0]["title"]
        p = requests.get(
            f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{title}",
            timeout=10
        ).json()
        extract = p.get("extract") or "(sin extracto)"
        url = p.get("content_urls", {}).get("desktop", {}).get("page", "")
        out = f" **{title}** — {extract}"
        if url:
            out += f"\n\nEnlace: {url}"
        return out
    except Exception as e:
        return f" Error al consultar Wikipedia: {e}"

# ============================== CÁLCULO SIMBÓLICO ==============================

def _sympify_expr(expr: str, var: str = "x"):
    x = sp.symbols(var)
    # Permitir ^ como potencia
    expr = expr.replace("^", "**")
    return sp.sympify(expr, convert_xor=True), x

def deriva(expr: str, var: str = "x") -> str:
    try:
        f, x = _sympify_expr(expr, var)
        df = sp.diff(f, x)
        return f" d/d{var} {sp.simplify(f)} = **{sp.simplify(df)}**"
    except Exception as e:
        return f" Error al derivar: {e}"

def integra(expr: str, var: str = "x") -> str:
    try:
        f, x = _sympify_expr(expr, var)
        F = sp.integrate(f, x)
        return f" ∫ {sp.simplify(f)} d{var} = **{sp.simplify(F)} + C**"
    except Exception as e:
        return f" Error al integrar: {e}"

def limite(expr: str, var: str = "x", at: str = "0", direction: Optional[str] = None) -> str:
    """
    direction: None (dos lados), '+' (derecha), '-' (izquierda)
    """
    try:
        f, x = _sympify_expr(expr, var)
        a = sp.sympify(at)
        lim = sp.limit(f, x, a, dir=direction if direction in ("+", "-") else None)
        dir_txt = {"+" : "⁺", "-" : "⁻"}.get(direction, "")
        return f" lim{dir_txt}_{{{var}→{a}}} {sp.simplify(f)} = **{sp.simplify(lim)}**"
    except Exception as e:
        return f" Error al calcular el límite: {e}"

def resuelve(eq_expr: str, var: str = "x") -> str:
    """
    Acepta 'x^2-4=0' o solo 'x^2-4' (asume =0). Devuelve raíces.
    """
    try:
        x = sp.symbols(var)
        s = eq_expr.replace("^", "**")
        if "=" in s:
            left, right = s.split("=", 1)
            sol = sp.solve(sp.Eq(sp.sympify(left), sp.sympify(right)), x)
        else:
            sol = sp.solve(sp.sympify(s), x)
        return f" Soluciones en {var}: **{sol}**"
    except Exception as e:
        return f" Error al resolver: {e}"

def simplifica(expr: str) -> str:
    try:
        f = sp.sympify(expr.replace("^", "**"))
        return f" Simplificado: **{sp.simplify(f)}**"
    except Exception as e:
        return f" Error al simplificar: {e}"

# ================================== UNIDADES ==================================

_ureg = pint.UnitRegistry()
_ureg.default_format = "~P"  # formato corto (Pa, N, etc.)

def convierte(qty_str: str) -> str:
    """
    Convierte unidades. Ej: '9.81 m/s^2 -> ft/s^2'  o  '1 atm -> Pa'
    """
    try:
        if "->" not in qty_str:
            return "Uso: /u 9.81 m/s^2 -> ft/s^2"
        left, right = qty_str.split("->", 1)
        q = _ureg(left.strip())
        q_to = q.to(right.strip())
        return f" {q} = **{q_to}**"
    except Exception as e:
        return f"Error en conversión de unidades: {e}"

# ================================== QUÍMICA ===================================

# Masa atómica aproximada (u). Puedes ampliar este diccionario si lo necesitas.
_ATOMIC_MASS: Dict[str, float] = {
    # H a Ne
    "H": 1.008, "He": 4.0026, "Li": 6.94, "Be": 9.0122, "B": 10.81,
    "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "Ne": 20.180,
    # Na a Ar
    "Na": 22.990, "Mg": 24.305, "Al": 26.982, "Si": 28.085, "P": 30.974,
    "S": 32.06, "Cl": 35.45, "Ar": 39.948,
    # K a Kr (algunos comunes)
    "K": 39.098, "Ca": 40.078, "Sc": 44.956, "Ti": 47.867, "V": 50.942,
    "Cr": 51.996, "Mn": 54.938, "Fe": 55.845, "Co": 58.933, "Ni": 58.693,
    "Cu": 63.546, "Zn": 65.38, "Br": 79.904, "Ag": 107.868, "I": 126.904, "Ba": 137.327,
    # Halógenos y otros útiles
    "Sr": 87.62, "Sn": 118.71, "Pb": 207.2, "Hg": 200.59
}

class _Tok:
    def __init__(self, type_: str, value: str):
        self.type = type_
        self.value = value
    def __repr__(self): return f"Tok({self.type},{self.value})"

def _tokenize_formula(s: str) -> List[_Tok]:
    tokens: List[_Tok] = []
    i = 0
    while i < len(s):
        if s[i].isspace():
            i += 1; continue
        if s[i] == '(':
            tokens.append(_Tok('LP', '(')); i += 1; continue
        if s[i] == ')':
            tokens.append(_Tok('RP', ')')); i += 1; continue
        if s[i].isdigit():
            j = i
            while j < len(s) and s[j].isdigit(): j += 1
            tokens.append(_Tok('NUM', s[i:j])); i = j; continue
        # Elemento: mayúscula + opcional minúscula
        if s[i].isalpha():
            j = i + 1
            if j < len(s) and s[j].islower():
                j += 1
            elem = s[i:j]
            tokens.append(_Tok('EL', elem)); i = j; continue
        raise ValueError(f"Símbolo no válido en fórmula: '{s[i]}'")
    return tokens

def _parse_group(tokens: List[_Tok], pos: int = 0) -> Tuple[Dict[str, int], int]:
    """
    Grammar simple:
      group := item+
      item  := element [num] | '(' group ')' [num]
    Devuelve (conteo_elementos, nueva_pos)
    """
    counts: Dict[str, int] = {}
    i = pos
    while i < len(tokens):
        t = tokens[i]
        if t.type == 'EL':
            elem = t.value
            i += 1
            mult = 1
            if i < len(tokens) and tokens[i].type == 'NUM':
                mult = int(tokens[i].value); i += 1
            counts[elem] = counts.get(elem, 0) + mult
        elif t.type == 'LP':
            sub, j = _parse_group(tokens, i + 1)
            i = j
            if i >= len(tokens) or tokens[i].type != 'RP':
                raise ValueError("Paréntesis no balanceados en fórmula.")
            i += 1  # consume ')'
            mult = 1
            if i < len(tokens) and tokens[i].type == 'NUM':
                mult = int(tokens[i].value); i += 1
            for k, v in sub.items():
                counts[k] = counts.get(k, 0) + v * mult
        elif t.type == 'RP':
            # fin del subgrupo; devolvemos al llamador
            return counts, i
        else:
            raise ValueError("Token inesperado en fórmula.")
    return counts, i

def _count_formula(formula: str) -> Dict[str, int]:
    tokens = _tokenize_formula(formula)
    counts, pos = _parse_group(tokens, 0)
    if pos != len(tokens):
        raise ValueError("Fórmula mal parseada.")
    return counts

def masa_molar(formula: str) -> str:
    """
    Calcula masa molar (g/mol) de una fórmula química sencilla.
    Soporta paréntesis y subíndices enteros: Ca(OH)2, Al2(SO4)3, etc.
    """
    try:
        counts = _count_formula(formula)
        total = 0.0
        missing: List[str] = []
        for elem, n in counts.items():
            if elem not in _ATOMIC_MASS:
                missing.append(elem)
                continue
            total += _ATOMIC_MASS[elem] * n
        if missing:
            return f" Elementos no soportados en la tabla local: {', '.join(sorted(missing))}."
        return f"⚗️ M_{formula} = **{total:.4f} g/mol**"
    except Exception as e:
        return f" Error al interpretar la fórmula: {e}"

# =================================== FÍSICA ===================================

def suvat(u: Optional[float] = None, v: Optional[float] = None,
          a: Optional[float] = None, t: Optional[float] = None,
          s: Optional[float] = None) -> str:
    """
    Solver básico de cinemática 1D (SUVAT).
    Proporciónale 3 valores (cualquiera) y resolverá los restantes si es posible.
    Variables:
        u: velocidad inicial
        v: velocidad final
        a: aceleración
        t: tiempo
        s: desplazamiento
    Fórmulas usadas:
        v = u + a t
        s = u t + 1/2 a t^2
        v^2 = u^2 + 2 a s
    """
    known = {k: v for k, v in {"u": u, "v": v, "a": a, "t": t, "s": s}.items() if v is not None}
    if len(known) < 3:
        return "Proporciona al menos 3 variables, p. ej.: /suvat u=0 v=20 a=5"
    # Intento por combinaciones
    try:
        U, V, A, T, S = u, v, a, t, s
        # 1) v = u + a t
        if V is None and U is not None and A is not None and T is not None:
            V = U + A * T
        if U is None and V is not None and A is not None and T is not None:
            U = V - A * T
        if A is None and V is not None and U is not None and T is not None:
            A = (V - U) / T
        if T is None and V is not None and U is not None and A is not None and A != 0:
            T = (V - U) / A

        # 2) s = u t + 1/2 a t^2
        if S is None and U is not None and T is not None and A is not None:
            S = U * T + 0.5 * A * T * T
        if U is None and S is not None and T is not None and A is not None:
            U = (S - 0.5 * A * T * T) / T
        if A is None and S is not None and U is not None and T is not None:
            A = 2 * (S - U * T) / (T * T)
        if T is None and S is not None and U is not None and A is not None:
            # ecuación cuadrática: 0.5 A T^2 + U T - S = 0
            if A == 0:
                if U != 0:
                    T = S / U
            else:
                a2, b2, c2 = 0.5 * A, U, -S
                disc = b2*b2 - 4*a2*c2
                if disc >= 0:
                    r1 = (-b2 + math.sqrt(disc)) / (2*a2)
                    r2 = (-b2 - math.sqrt(disc)) / (2*a2)
                    T = r1 if r1 >= 0 else r2

        # 3) v^2 = u^2 + 2 a s
        if V is None and U is not None and A is not None and S is not None:
            V = math.sqrt(max(U*U + 2*A*S, 0.0)) if U*U + 2*A*S >= 0 else None
        if U is None and V is not None and A is not None and S is not None:
            U = math.sqrt(max(V*V - 2*A*S, 0.0)) if V*V - 2*A*S >= 0 else None
        if A is None and V is not None and U is not None and S is not None:
            A = (V*V - U*U) / (2*S) if S != 0 else A
        if S is None and V is not None and U is not None and A is not None:
            S = (V*V - U*U) / (2*A) if A != 0 else S

        out = []
        for name, val, unit in (("u", U, "m/s"), ("v", V, "m/s"),
                                ("a", A, "m/s^2"), ("t", T, "s"), ("s", S, "m")):
            if val is not None:
                out.append(f"{name} = {val:.6g} {unit}")
        if out:
            return "🏎️ Cinemática (SUVAT):\n" + "\n".join(out)
        return "No se pudo resolver con los datos proporcionados."
    except Exception as e:
        return f" Error en solver SUVAT: {e}"

# =========================== ESTADÍSTICA / DATOS =================================

def stats_lista(numeros: str) -> str:
    """
    Stats rápidos de una lista separada por comas. Ej: '1, 2, 2, 3, 5'
    """
    try:
        vals = np.array([float(x) for x in re.split(r"[,\s]+", numeros.strip()) if x])
        if len(vals) == 0:
            return "Proporciona números separados por comas."
        msg = [
            f"n = {len(vals)}",
            f"media = {np.mean(vals):.6g}",
            f"mediana = {np.median(vals):.6g}",
            f"desv.est. = {np.std(vals, ddof=1):.6g}" if len(vals) > 1 else "desv.est. = N/A",
            f"min = {np.min(vals):.6g}",
            f"max = {np.max(vals):.6g}",
            f"Q1 = {np.quantile(vals, 0.25):.6g}",
            f"Q3 = {np.quantile(vals, 0.75):.6g}",
        ]
        return "📊 Estadísticos básicos:\n" + "\n".join(msg)
    except Exception as e:
        return f" Error al calcular estadísticas: {e}"

def csv_head(path: str, n: int = 5) -> str:
    try:
        df = pd.read_csv(path)
        return f"📄 Primeras {min(n, len(df))} filas de {os.path.basename(path)}:\n\n{df.head(n).to_string(index=False)}"
    except Exception as e:
        return f" Error al leer CSV: {e}"

def csv_describe(path: str) -> str:
    try:
        df = pd.read_csv(path)
        desc = df.describe(include="all", datetime_is_numeric=True).transpose()
        return "📈 Descripción del dataset:\n" + desc.to_string()
    except Exception as e:
        return f" Error en describe(): {e}"

def csv_col_stats(path: str, col: str) -> str:
    try:
        df = pd.read_csv(path)
        if col not in df.columns:
            return f" La columna '{col}' no existe. Columnas disponibles: {list(df.columns)}"
        s = df[col].dropna()
        if s.empty:
            return f"Columna '{col}' está vacía o todo es NaN."
        msg = [
            f"n = {s.size}",
            f"media = {s.mean():.6g}" if np.issubdtype(s.dtype, np.number) else "media = N/A",
            f"mediana = {s.median() if np.issubdtype(s.dtype, np.number) else 'N/A'}",
            f"desv.est. = {s.std(ddof=1):.6g}" if np.issubdtype(s.dtype, np.number) and s.size > 1 else "desv.est. = N/A",
            f"mín = {s.min()}",
            f"máx = {s.max()}",
        ]
        return f"📌 Stats de columna '{col}':\n" + "\n".join(msg)
    except Exception as e:
        return f" Error al calcular stats de columna: {e}"

def csv_hist(path: str, col: str, bins: int = 20) -> str:
    """
    Genera histograma de una columna numérica. Devuelve la ruta de la imagen.
    """
    try:
        df = pd.read_csv(path)
        if col not in df.columns:
            return f" La columna '{col}' no existe. Columnas disponibles: {list(df.columns)}"
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            return f" La columna '{col}' no tiene datos numéricos."
        os.makedirs("plots", exist_ok=True)
        fname = f"hist_{uuid.uuid4().hex[:8]}.png"
        fpath = os.path.join("plots", fname)
        plt.figure()
        plt.hist(s, bins=bins)
        plt.xlabel(col); plt.ylabel("frecuencia"); plt.title(f"Histograma: {col}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(fpath, dpi=120)
        plt.close()
        return f"🖼️ Histograma guardado en: {fpath}"
    except Exception as e:
        return f" Error al generar histograma: {e}"

# =================================== GRÁFICAS ==================================

def plot(expr: str, xspec: str = "x:-5:5") -> str:
    """
    Dibuja y = f(x) en el rango indicado. Acepta:
      x:<min>:<max>              (≈800 puntos)
      x:<min>:<step>:<max>       (paso explícito)
    Soporta alias: ^ -> **, ln() -> log(), sen() -> sin(), 'e' -> constante e.
    """
    try:
        # --- normaliza expresiones comunes ---
        expr_norm = (
            expr.replace("^", "**")
                .replace("ln(", "log(")
                .replace("sen(", "sin(")
        )

        # --- constantes/alias para sympify ---
        locals_map = {
            "pi": sp.pi, "e": sp.E,
            "sen": sp.sin, "tg": sp.tan, "ctg": sp.cot,  # por si acaso
            "log": sp.log, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
            "sqrt": sp.sqrt, "exp": sp.exp,
        }

        # --- parseo de xspec ---
        parts = [p.strip() for p in xspec.split(":")]
        if len(parts) not in (3, 4):
            return " Formato de x inválido. Usa x:min:max o x:min:step:max"

        var = parts[0]
        if len(parts) == 3:
            a_s, b_s = parts[1], parts[2]
            step = None
        else:
            a_s, step_s, b_s = parts[1], parts[2], parts[3]
            step = float(sp.N(sp.sympify(step_s, locals=locals_map)))

        a = float(sp.N(sp.sympify(a_s, locals=locals_map)))
        b = float(sp.N(sp.sympify(b_s, locals=locals_map)))
        if a == b:
            return " El rango en x no puede tener el mismo inicio y fin."

        # --- construir función numérica con numpy ---
        x = sp.symbols(var)
        f_sym = sp.sympify(expr_norm, locals=locals_map)
        f = sp.lambdify(x, f_sym, modules=["numpy"])

        # --- vector de x ---
        if step is None:
            n = 800
            xs = np.linspace(a, b, n)
        else:
            npts = max(2, int(abs((b - a) / step)) + 1)
            xs = np.linspace(a, b, npts)

        ys = f(xs)

        # Manejo de resultados complejos: si la parte imaginaria es pequeña, toma la real
        ys = np.asarray(ys)
        if np.iscomplexobj(ys):
            if np.max(np.abs(np.imag(ys))) < 1e-9:
                ys = np.real(ys)
            else:
                # si realmente es complejo, grafica el módulo
                ys = np.abs(ys)

        # --- guardar figura ---
        os.makedirs("plots", exist_ok=True)
        fname = f"plot_{uuid.uuid4().hex[:8]}.png"
        fpath = os.path.join("plots", fname)

        plt.figure()
        plt.plot(xs, ys)
        plt.xlabel(var); plt.ylabel("y"); plt.title(f"y = {expr}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(fpath, dpi=120)
        plt.close()

        return f"🖼️ Gráfica guardada en: {fpath}"
    except Exception as e:
        return f" Error al graficar: {e}"


# ====================== APOYO A PROGRAMACIÓN (SEGURO) =========================

@dataclass
class CodeAnalysis:
    lines: int
    functions: int
    classes: int
    imports: int
    cyclomatic: int

def _cyclomatic_complexity_from_ast(tree: ast.AST) -> int:
    """
    Estimación simple de complejidad ciclomatica:
    +1 por cada If, For, While, And/Or, Try, With, BoolOp, IfExp, Comprehension
    Base = 1
    """
    count = 1
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.IfExp)):
            count += 1
        if isinstance(node, ast.BoolOp):  # and/or
            count += len(getattr(node, "values", [])) - 1
        if isinstance(node, (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)):
            count += 1
    return count

def analizar_codigo(py_code: str) -> str:
    """
    Analiza el código (sin ejecutarlo): funciones, clases, imports y complejidad.
    """
    try:
        tree = ast.parse(py_code)
        lines = len([ln for ln in py_code.splitlines() if ln.strip() != ""])
        funcs = sum(isinstance(n, ast.FunctionDef) for n in ast.walk(tree))
        classes = sum(isinstance(n, ast.ClassDef) for n in ast.walk(tree))
        imports = sum(isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(tree))
        cyclo = _cyclomatic_complexity_from_ast(tree)
        rep = [
            "🧩 Análisis estático del código:",
            f"- líneas (no vacías): {lines}",
            f"- funciones: {funcs}",
            f"- clases: {classes}",
            f"- imports: {imports}",
            f"- complejidad ciclomatica (aprox): {cyclo}",
            "",
            "Tip: menor complejidad suele ser más fácil de probar y mantener."
        ]
        return "\n".join(rep)
    except Exception as e:
        return f" Error al analizar el código: {e}"
