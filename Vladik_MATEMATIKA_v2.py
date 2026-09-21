"""
VLADIK MATEMATIK v2.0 — Автономный математический движок
Без нейросети. Всё на SymPy + SciPy + NumPy + Matplotlib.
Один файл. Все улучшения применены.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog, simpledialog
import sympy as sp
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa

import re
import os
import json
import time
import signal
import platform
import threading
import logging
import gc
from logging.handlers import RotatingFileHandler
from dataclasses import dataclass, asdict, field
from typing import Optional, Tuple, List, Any, Dict, TypedDict, Callable
from contextlib import contextmanager
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# SciPy — опционально (для оптимизации)
try:
    from scipy import optimize as scipy_optimize
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False

# SymPy безопасный парсер
try:
    from sympy.parsing.sympy_parser import (
        parse_expr, standard_transformations,
        implicit_multiplication_application, convert_xor,
    )
    _TRANSFORMS = standard_transformations + (convert_xor, implicit_multiplication_application)
    _PARSER_OK = True
except ImportError:
    _PARSER_OK = False


# ════════════════════════════════════════════════════════════════════
# КОНСТАНТЫ
# ════════════════════════════════════════════════════════════════════

MAX_ABS_VALUE = 1e10
DEFAULT_POINTS_2D = 2000
DEFAULT_POINTS_3D = 60
MAX_CACHE_SIZE = 128
HISTORY_MAX = 200
SYMPY_TIMEOUT = 10
MAX_INPUT_LENGTH = 5000

PLOT_COLORS = ['#6c5ce7', '#00b894', '#fdcb6e', '#e17055',
               '#0984e3', '#fd79a8', '#a29bfe', '#55efc4']

CONFIG_DIR = os.path.expanduser("~/.vladik_math")
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")
HISTORY_FILE = os.path.join(CONFIG_DIR, "history.json")
LOG_FILE = os.path.join(CONFIG_DIR, "vladik.log")

THEMES = {
    'dark': {
        'bg': '#0a0a0a', 'bg_secondary': '#0d0d0d', 'bg_tertiary': '#141414',
        'fg': '#ffffff', 'fg_secondary': '#b0b0b0', 'text_secondary': '#666666',
        'accent': '#cc0000', 'accent_hover': '#ff0000', 'accent_light': '#ff3333',
        'success': '#2ecc71', 'warning': '#f1c40f', 'error': '#e74c3c',
        'border': '#1a1a1a', 'input_bg': '#0d0d0d', 'output_bg': '#0a0a0a',
        'tab_active': '#cc0000',
    },
    'light': {
        'bg': '#f5f5f5', 'bg_secondary': '#ffffff', 'bg_tertiary': '#ebebeb',
        'fg': '#1a1a1a', 'fg_secondary': '#555555', 'text_secondary': '#888888',
        'accent': '#cc0000', 'accent_hover': '#ff0000', 'accent_light': '#e63946',
        'success': '#27ae60', 'warning': '#f39c12', 'error': '#c0392b',
        'border': '#d0d0d0', 'input_bg': '#ffffff', 'output_bg': '#fafafa',
        'tab_active': '#cc0000',
    },
}


# ════════════════════════════════════════════════════════════════════
# ЛОГИРОВАНИЕ
# ════════════════════════════════════════════════════════════════════

def setup_logger() -> logging.Logger:
    os.makedirs(CONFIG_DIR, exist_ok=True)
    logger = logging.getLogger("vladik")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
    try:
        fh = RotatingFileHandler(LOG_FILE, maxBytes=2_000_000, backupCount=3, encoding='utf-8')
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except Exception:
        pass
    return logger

log = setup_logger()


# ════════════════════════════════════════════════════════════════════
# УТИЛИТЫ
# ════════════════════════════════════════════════════════════════════

class TimeoutError_(Exception):
    pass


@contextmanager
def timeout(seconds: int):
    """Таймаут: SIGALRM на Unix, без реального прерывания на Windows."""
    if platform.system() != "Windows" and hasattr(signal, 'SIGALRM'):
        def _h(signum, frame):
            raise TimeoutError_(f"Превышено время выполнения ({seconds} с)")
        old = signal.signal(signal.SIGALRM, _h)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)
    else:
        yield


_LATEX_PATTERNS = [
    (re.compile(r'\\frac\{([^{}]+)\}\{([^{}]+)\}'), r'(\1)/(\2)'),
    (re.compile(r'\\sqrt\{([^{}]+)\}'), r'√(\1)'),
    (re.compile(r'\\cdot'), '·'),
    (re.compile(r'\\times'), '×'),
    (re.compile(r'\\pm'), '±'),
    (re.compile(r'\\infty'), '∞'),
    (re.compile(r'\\pi'), 'π'),
    (re.compile(r'\\theta'), 'θ'),
    (re.compile(r'\\alpha'), 'α'),
    (re.compile(r'\\beta'), 'β'),
    (re.compile(r'\\lambda'), 'λ'),
    (re.compile(r'\\leq'), '≤'),
    (re.compile(r'\\geq'), '≥'),
    (re.compile(r'\\neq'), '≠'),
    (re.compile(r'\\approx'), '≈'),
    (re.compile(r'\\left|\\right'), ''),
    (re.compile(r'\\[a-zA-Z]+'), ''),
    (re.compile(r'\{|\}'), ''),
]


def latex_to_text(s: str) -> str:
    if not s:
        return s
    for pat, repl in _LATEX_PATTERNS:
        s = pat.sub(repl, s)
    return s


def clean_output(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'<\|.*?\|>', '', text)
    text = latex_to_text(text)
    return re.sub(r'\s+', ' ', text).strip()


def humanize_time(sec: float) -> str:
    if sec < 1:
        return f"{sec * 1000:.0f} мс"
    if sec < 60:
        return f"{sec:.2f} с"
    return f"{sec / 60:.1f} мин"


def truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n - 3] + "..."


# ════════════════════════════════════════════════════════════════════
# КОНФИГ И ИСТОРИЯ
# ════════════════════════════════════════════════════════════════════

@dataclass
class AppConfig:
    theme: str = 'dark'
    window_width: int = 1400
    window_height: int = 900
    window_x: int = 100
    window_y: int = 100
    complex_mode: bool = False
    last_tab: str = 'equations'
    x_min: float = -10.0
    x_max: float = 10.0
    y_min: float = -5.0
    y_max: float = 5.0

    @classmethod
    def load(cls) -> 'AppConfig':
        try:
            os.makedirs(CONFIG_DIR, exist_ok=True)
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return cls(**{k: v for k, v in data.items()
                              if k in cls.__dataclass_fields__})
        except Exception as e:
            log.warning("Не удалось загрузить конфиг: %s", e)
        return cls()

    def save(self):
        try:
            os.makedirs(CONFIG_DIR, exist_ok=True)
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(asdict(self), f, indent=2, ensure_ascii=False)
        except Exception as e:
            log.warning("Не удалось сохранить конфиг: %s", e)


@dataclass
class HistoryEntry:
    timestamp: str
    problem: str
    problem_type: str
    result: str
    success: bool
    elapsed: float
    engine: str = "SymPy"

    def to_dict(self) -> Dict:
        return asdict(self)


class History:
    def __init__(self, max_entries: int = HISTORY_MAX):
        self.max_entries = max_entries
        self.entries: List[HistoryEntry] = []
        self._load()

    def _load(self):
        try:
            if os.path.exists(HISTORY_FILE):
                with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.entries = [HistoryEntry(**e) for e in data]
        except Exception as e:
            log.warning("История не загружена: %s", e)
            self.entries = []

    def _save(self):
        try:
            os.makedirs(CONFIG_DIR, exist_ok=True)
            with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
                json.dump([e.to_dict() for e in self.entries],
                          f, indent=2, ensure_ascii=False)
        except Exception as e:
            log.warning("История не сохранена: %s", e)

    def add(self, entry: HistoryEntry):
        self.entries.insert(0, entry)
        self.entries = self.entries[:self.max_entries]
        self._save()

    def clear(self):
        self.entries = []
        self._save()


# ════════════════════════════════════════════════════════════════════
# ЯДРО ПАРСИНГА И ВЫЧИСЛЕНИЙ
# ════════════════════════════════════════════════════════════════════

class ParseError(ValueError):
    pass


class VladikCore:
    _FUNC_REPLACEMENTS = [
        (r'\bln\b', 'log'),
        (r'\babs\s*\(', 'Abs('),
        (r'\barcsin\b', 'asin'),
        (r'\barccos\b', 'acos'),
        (r'\barctan\b', 'atan'),
    ]

    def __init__(self):
        self._symbols: Dict[str, Any] = {}
        for name in ['x', 'y', 'z', 't', 'a', 'b', 'c', 'u', 'v', 'w']:
            self._symbols[name] = sp.Symbol(name)
        self._symbols['n'] = sp.Symbol('n', integer=True, positive=True)
        self._symbols['k'] = sp.Symbol('k', integer=True, positive=True)
        self._symbols.update({
            'pi': sp.pi, 'π': sp.pi, 'e': sp.E,
            'inf': sp.oo, '∞': sp.oo, 'I': sp.I,
        })
        self._cache: Dict[str, sp.Expr] = {}
        self._cache_lock = threading.Lock()

    def parse_expression(self, expr_str: str) -> sp.Expr:
        if not expr_str or not expr_str.strip():
            raise ParseError("Пустое выражение")
        if len(expr_str) > MAX_INPUT_LENGTH:
            raise ParseError(f"Слишком длинный ввод (> {MAX_INPUT_LENGTH} символов)")

        key = expr_str.strip()
        with self._cache_lock:
            if key in self._cache:
                return self._cache[key]

        s = key.replace('^', '**')
        for pat, repl in self._FUNC_REPLACEMENTS:
            s = re.sub(pat, repl, s)

        try:
            if _PARSER_OK:
                expr = parse_expr(s, local_dict=self._symbols,
                                  transformations=_TRANSFORMS, evaluate=True)
            else:
                expr = sp.sympify(s, locals=self._symbols)
        except Exception as e:
            log.warning("Parse error %r: %s", expr_str, e)
            raise ParseError(f"Ошибка парсинга: {e}")

        with self._cache_lock:
            if len(self._cache) >= MAX_CACHE_SIZE:
                keys = list(self._cache.keys())[:MAX_CACHE_SIZE // 2]
                for k in keys:
                    del self._cache[k]
            self._cache[key] = expr
        return expr

    @staticmethod
    def _safe_scalar(val) -> float:
        try:
            if np.iscomplexobj(val):
                val = np.real(val)
            fv = float(val)
            return np.nan if (not np.isfinite(fv) or abs(fv) > MAX_ABS_VALUE) else fv
        except Exception:
            return np.nan

    def compute_values(self, expr_str: str, var: str = 'x',
                       x_min: float = -10, x_max: float = 10,
                       n_points: int = DEFAULT_POINTS_2D) -> Tuple[np.ndarray, np.ndarray]:
        expr = self.parse_expression(expr_str)
        var_sym = self.get_symbol(var)
        f = sp.lambdify(var_sym, expr, modules=['numpy'])
        x_vals = np.linspace(x_min, x_max, n_points, dtype=np.float64)
        try:
            y_vals = f(x_vals)
        except Exception:
            y_vals = np.array([self._safe_scalar(f(x)) for x in x_vals])
        if np.iscomplexobj(y_vals):
            y_vals = np.real(y_vals)
        y_vals = np.asarray(y_vals, dtype=np.float64)
        with np.errstate(invalid='ignore', over='ignore'):
            y_vals = np.where(~np.isfinite(y_vals), np.nan, y_vals)
            y_vals = np.where(np.abs(y_vals) > MAX_ABS_VALUE, np.nan, y_vals)
        return x_vals, y_vals

    def compute_polar(self, expr_str: str,
                      theta_min: float = 0.0, theta_max: float = 2 * np.pi,
                      n_points: int = DEFAULT_POINTS_2D) -> Tuple[np.ndarray, np.ndarray]:
        theta_sym = sp.Symbol('theta')
        local = dict(self._symbols)
        local['theta'] = theta_sym
        local['θ'] = theta_sym
        s = expr_str.replace('^', '**')
        s = re.sub(r'\btheta\b|θ', 'theta', s)
        expr = parse_expr(s, local_dict=local, transformations=_TRANSFORMS) if _PARSER_OK \
            else sp.sympify(s, locals=local)
        f = sp.lambdify(theta_sym, expr, modules=['numpy'])
        theta = np.linspace(theta_min, theta_max, n_points)
        try:
            r = f(theta)
        except Exception:
            r = np.array([self._safe_scalar(f(t)) for t in theta])
        r = np.asarray(r, dtype=np.float64)
        with np.errstate(invalid='ignore'):
            r = np.where(~np.isfinite(r), np.nan, r)
        return r * np.cos(theta), r * np.sin(theta)

    def compute_parametric(self, x_expr: str, y_expr: str,
                           t_min: float = 0.0, t_max: float = 2 * np.pi,
                           n_points: int = DEFAULT_POINTS_2D) -> Tuple[np.ndarray, np.ndarray]:
        t_sym = self.get_symbol('t')
        fx = sp.lambdify(t_sym, self.parse_expression(x_expr), modules=['numpy'])
        fy = sp.lambdify(t_sym, self.parse_expression(y_expr), modules=['numpy'])
        t_vals = np.linspace(t_min, t_max, n_points)
        try:
            x_vals = np.asarray(fx(t_vals), dtype=np.float64)
        except Exception:
            x_vals = np.array([self._safe_scalar(fx(t)) for t in t_vals])
        try:
            y_vals = np.asarray(fy(t_vals), dtype=np.float64)
        except Exception:
            y_vals = np.array([self._safe_scalar(fy(t)) for t in t_vals])
        with np.errstate(invalid='ignore'):
            x_vals[~np.isfinite(x_vals)] = np.nan
            y_vals[~np.isfinite(y_vals)] = np.nan
        return x_vals, y_vals

    def compute_implicit(self, expr_str: str,
                         x_min: float = -5, x_max: float = 5,
                         y_min: float = -5, y_max: float = 5,
                         n_points: int = 200) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        expr = self.parse_expression(expr_str)
        f = sp.lambdify((self._symbols['x'], self._symbols['y']),
                        expr, modules=['numpy'])
        x_vals = np.linspace(x_min, x_max, n_points)
        y_vals = np.linspace(y_min, y_max, n_points)
        X, Y = np.meshgrid(x_vals, y_vals)
        try:
            Z = f(X, Y)
        except Exception:
            Z = np.zeros_like(X)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    try:
                        Z[i, j] = self._safe_scalar(f(X[i, j], Y[i, j]))
                    except Exception:
                        Z[i, j] = np.nan
        Z = np.asarray(Z, dtype=np.float64)
        with np.errstate(invalid='ignore'):
            Z = np.where(~np.isfinite(Z), np.nan, Z)
        return X, Y, Z

    def plot_3d(self, expr_str: str, var1: str = 'x', var2: str = 'y',
                x_min: float = -5, x_max: float = 5,
                y_min: float = -5, y_max: float = 5
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        expr = self.parse_expression(expr_str)
        f = sp.lambdify((self._symbols[var1], self._symbols[var2]),
                        expr, modules=['numpy'])
        n = DEFAULT_POINTS_3D
        x_vals = np.linspace(x_min, x_max, n)
        y_vals = np.linspace(y_min, y_max, n)
        X, Y = np.meshgrid(x_vals, y_vals)
        try:
            Z = f(X, Y)
        except Exception:
            Z = np.zeros_like(X)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    try:
                        Z[i, j] = self._safe_scalar(f(X[i, j], Y[i, j]))
                    except Exception:
                        Z[i, j] = np.nan
        Z = np.asarray(Z, dtype=np.float64)
        with np.errstate(invalid='ignore'):
            Z = np.where(~np.isfinite(Z), np.nan, Z)
            Z = np.where(np.abs(Z) > MAX_ABS_VALUE, np.nan, Z)
        return X, Y, Z

    @staticmethod
    def is_3d_expression(expr: sp.Expr) -> bool:
        try:
            return len(expr.free_symbols) >= 2
        except Exception:
            return False

    def get_symbol(self, name: str) -> sp.Symbol:
        if name not in self._symbols:
            self._symbols[name] = sp.Symbol(name)
        return self._symbols[name]

    @property
    def symbols(self) -> Dict[str, Any]:
        return dict(self._symbols)


# ════════════════════════════════════════════════════════════════════
# РЕШАТЕЛЬ
# ════════════════════════════════════════════════════════════════════

class SolveResult(TypedDict, total=False):
    success: bool
    solution: Any
    error: Optional[str]
    type: str
    formatted: str
    engine: str
    elapsed: float
    verified: Optional[bool]


class MathSolver:
    def __init__(self, core: VladikCore):
        self.core = core

    # ---- публичный API ----

    def solve(self, problem: str, problem_type: str = 'auto',
              complex_mode: bool = False) -> SolveResult:
        t0 = time.perf_counter()
        problem = problem.strip()
        if not problem:
            return self._err("Пустая задача", problem_type, t0)
        if problem_type == 'auto':
            problem_type = self._detect_type(problem)

        try:
            if problem_type == 'ode':
                result = self._solve_ode(problem)
            elif problem_type == 'system':
                result = self._solve_system(problem, complex_mode)
            elif problem_type == 'inequality':
                result = self._solve_inequality(problem)
            else:
                result = self._solve_equation(problem, complex_mode)
        except TimeoutError_ as e:
            return self._err(str(e), problem_type, t0)
        except ParseError as e:
            return self._err(str(e), problem_type, t0)
        except Exception as e:
            log.exception("solve error")
            return self._err(f"Внутренняя ошибка: {e}", problem_type, t0)

        if result.get('success') and result.get('solution') is not None:
            try:
                result['verified'] = self._verify(problem, problem_type, result['solution'])
            except Exception:
                result['verified'] = None

        result['elapsed'] = time.perf_counter() - t0
        result.setdefault('engine', 'SymPy')
        result.setdefault('type', problem_type)
        return result

    @staticmethod
    def _detect_type(problem: str) -> str:
        s = problem.lower()
        if "y''" in s or "y'" in s:
            return 'ode'
        if ';' in problem or '\n' in problem:
            return 'system'
        if any(op in s for op in ['<', '>', '≤', '≥', '<=', '>=']):
            return 'inequality'
        return 'equation'

    # ---- уравнения ----

    def _solve_equation(self, problem: str, complex_mode: bool) -> SolveResult:
        eq_obj, var = self._parse_equation(problem)
        with timeout(SYMPY_TIMEOUT):
            solutions = sp.solve(eq_obj, var)
        if not solutions:
            return {'success': True, 'solution': [],
                    'type': 'equation',
                    'formatted': 'Нет решений'}
        filtered = []
        for sol in solutions:
            try:
                if sol.is_infinite:
                    continue
            except Exception:
                pass
            if complex_mode:
                filtered.append(sol)
            else:
                try:
                    if sol.is_real:
                        filtered.append(sol)
                    elif sol.is_real is None:
                        filtered.append(sol)
                except Exception:
                    filtered.append(sol)
        if not filtered:
            return {'success': True, 'solution': [],
                    'type': 'equation',
                    'formatted': 'Нет действительных решений' +
                                 ('' if complex_mode else ' (попробуйте режим комплексных чисел)')}
        parts = [f"{var} = {self._fmt_sympy(s)}" for s in filtered]
        return {'success': True, 'solution': filtered,
                'type': 'equation',
                'formatted': '\n'.join(parts)}

    # ---- системы ----

    def _solve_system(self, problem: str, complex_mode: bool) -> SolveResult:
        parts = [p.strip() for p in re.split(r'[;\n]+', problem) if p.strip()]
        if len(parts) < 2:
            return self._err("Нужно минимум 2 уравнения через ';'", 'system', 0.0)
        eqs, syms = [], set()
        for part in parts:
            eq_obj, _ = self._parse_equation(part)
            eqs.append(eq_obj)
            syms.update(eq_obj.free_symbols)
        vars_sorted = sorted(syms, key=str)
        if not vars_sorted:
            return self._err("Нет неизвестных", 'system', 0.0)
        with timeout(SYMPY_TIMEOUT):
            sols = sp.solve(eqs, vars_sorted, dict=True)
        if not sols:
            return {'success': True, 'solution': [],
                    'type': 'system',
                    'formatted': 'Система не имеет решений'}
        if not complex_mode:
            real_sols = []
            for d in sols:
                if all(self._is_real(v) for v in d.values()):
                    real_sols.append(d)
            if real_sols:
                sols = real_sols
        lines = [', '.join(f"{v} = {self._fmt_sympy(d[v])}"
                           for v in vars_sorted if v in d)
                 for d in sols]
        return {'success': True, 'solution': sols,
                'type': 'system',
                'formatted': '\n'.join(lines)}

    @staticmethod
    def _is_real(v) -> bool:
        try:
            return bool(v.is_real)
        except Exception:
            return True

    # ---- неравенства ----

    def _solve_inequality(self, problem: str) -> SolveResult:
        op_map = [('<=', '<='), ('>=', '>='), ('≤', '<='), ('≥', '>='),
                  ('<', '<'), ('>', '>')]
        op, left, right = None, None, None
        for sym, canon in op_map:
            if sym in problem:
                op = canon
                left, right = problem.split(sym, 1)
                break
        if op is None:
            return self._err("Неизвестный оператор", 'inequality', 0.0)
        le = self.core.parse_expression(left)
        re_ = self.core.parse_expression(right)
        var = self._pick_var(le, re_)
        rel = {'<': sp.Lt, '<=': sp.Le, '>': sp.Gt, '>=': sp.Ge}[op]
        ineq = rel(le, re_)
        with timeout(SYMPY_TIMEOUT):
            sol_set = sp.solveset(ineq, var, domain=sp.S.Reals)
        return {'success': True, 'solution': sol_set,
                'type': 'inequality',
                'formatted': f"Решение: {self._fmt_sympy(sol_set)}"}

    # ---- ДУ ----

    def _solve_ode(self, problem: str) -> SolveResult:
        """Порядок: y'' -> маркер, y' -> маркер, y -> y(x), маркеры -> Derivative."""
        x = self.core.get_symbol('x')
        func = sp.Function('y')(x)
        local = dict(self.core.symbols)
        local['y'] = sp.Function('y')
        local['Derivative'] = sp.Derivative

        def _norm(part: str) -> sp.Expr:
            s = part.strip().replace('^', '**')
            s = re.sub(r"\by\s*''", '__D2__', s)
            s = re.sub(r"\by\s*'", '__D1__', s)
            s = re.sub(r'\by\b', 'y(x)', s)
            s = s.replace('__D2__', 'Derivative(y(x), x, 2)')
            s = s.replace('__D1__', 'Derivative(y(x), x)')
            s = re.sub(r'\bln\b', 'log', s)
            if _PARSER_OK:
                return parse_expr(s, local_dict=local,
                                  transformations=_TRANSFORMS, evaluate=True)
            return sp.sympify(s, locals=local)

        try:
            if '=' in problem:
                left, right = problem.split('=', 1)
                ode_eq = sp.Eq(_norm(left), _norm(right))
            else:
                ode_eq = sp.Eq(_norm(problem), 0)
        except Exception as e:
            return self._err(f"Ошибка разбора ДУ: {e}", 'ode', 0.0)

        try:
            with timeout(SYMPY_TIMEOUT):
                solution = sp.dsolve(ode_eq, func)
        except TimeoutError_ as e:
            return self._err(str(e), 'ode', 0.0)
        except Exception as e:
            return self._err(f"Ошибка решения ДУ: {e}", 'ode', 0.0)

        if solution is None:
            return {'success': True, 'solution': None,
                    'type': 'ode', 'formatted': 'Решение не найдено'}

        verified = None
        try:
            chk = sp.checkodesol(ode_eq, solution)
            verified = bool(chk[0]) if isinstance(chk, tuple) else None
        except Exception:
            pass

        return {'success': True, 'solution': solution,
                'type': 'ode',
                'formatted': self._fmt_sympy(solution),
                'verified': verified}

    # ---- парсинг ----

    def _parse_equation(self, text: str):
        text = text.strip()
        if '=' in text:
            l, r = text.split('=', 1)
            eq = sp.Eq(self.core.parse_expression(l), self.core.parse_expression(r))
        else:
            eq = sp.Eq(self.core.parse_expression(text), 0)
        var = self._pick_var(eq.lhs, eq.rhs)
        return eq, var

    @staticmethod
    def _pick_var(*exprs) -> sp.Symbol:
        syms = set()
        for e in exprs:
            if e is not None:
                syms.update(getattr(e, 'free_symbols', set()))
        if not syms:
            return sp.Symbol('x')
        # предпочитаем x, потом y, потом остальные
        for pref in ('x', 'y', 'z', 't'):
            for s in syms:
                if str(s) == pref:
                    return s
        return sorted(syms, key=str)[0]

    # ---- форматирование ----

    @staticmethod
    def _fmt_sympy(x) -> str:
        try:
            return clean_output(sp.sstr(x, order='none'))
        except Exception:
            return clean_output(str(x))

    # ---- верификация ----

    def _verify(self, problem: str, ptype: str, solution) -> Optional[bool]:
        try:
            if ptype == 'equation':
                eq_obj, var = self._parse_equation(problem)
                sols = solution if isinstance(solution, list) else [solution]
                for s in sols:
                    try:
                        val = eq_obj.subs(var, s)
                        diff = sp.simplify(val.lhs - val.rhs)
                        if abs(complex(diff.evalf())) > 1e-6:
                            return False
                    except Exception:
                        return None
                return True
            if ptype == 'system':
                parts = [p.strip() for p in re.split(r'[;\n]+', problem) if p.strip()]
                eqs = [self._parse_equation(p)[0] for p in parts]
                sols = solution if isinstance(solution, list) else [solution]
                for d in sols:
                    for eq in eqs:
                        try:
                            diff = sp.simplify((eq.lhs - eq.rhs).subs(d))
                            if abs(complex(diff.evalf())) > 1e-6:
                                return False
                        except Exception:
                            return None
                return True
            if ptype == 'ode':
                return None  # проверено через checkodesol
        except Exception:
            return None
        return None

    @staticmethod
    def _err(msg: str, ptype: str, t0: float) -> SolveResult:
        return {'success': False, 'solution': None,
                'error': msg, 'type': ptype,
                'formatted': msg, 'engine': 'SymPy',
                'elapsed': time.perf_counter() - t0}


# ════════════════════════════════════════════════════════════════════
# РАСШИРЕННЫЙ МАТАНАЛИЗ
# ════════════════════════════════════════════════════════════════════

class AdvancedMath:
    def __init__(self, core: VladikCore):
        self.core = core

    def derivative(self, expr_str: str, var: str = 'x', order: int = 1) -> str:
        expr = self.core.parse_expression(expr_str)
        v = self.core.get_symbol(var)
        d = sp.diff(expr, v, order)
        return sp.sstr(sp.simplify(d))

    def integral(self, expr_str: str, var: str = 'x',
                 a: Optional[float] = None, b: Optional[float] = None) -> str:
        expr = self.core.parse_expression(expr_str)
        v = self.core.get_symbol(var)
        if a is None or b is None:
            res = sp.integrate(expr, v)
            return f"{sp.sstr(res)} + C"
        res = sp.integrate(expr, (v, a, b))
        return sp.sstr(sp.simplify(res))

    def limit(self, expr_str: str, var: str = 'x',
              point: str = '0', direction: str = '+-') -> str:
        expr = self.core.parse_expression(expr_str)
        v = self.core.get_symbol(var)
        pt = self.core.parse_expression(point) if point not in ('oo', '-oo') \
            else (sp.oo if point == 'oo' else -sp.oo)
        d = {'+': '+', '-': '-', '+-': '+-'}.get(direction, '+-')
        res = sp.limit(expr, v, pt, dir=d)
        return sp.sstr(res)

    def series(self, expr_str: str, var: str = 'x',
               point: str = '0', order: int = 6) -> str:
        expr = self.core.parse_expression(expr_str)
        v = self.core.get_symbol(var)
        pt = self.core.parse_expression(point)
        res = sp.series(expr, v, pt, order)
        return sp.sstr(res)

    def simplify_expr(self, expr_str: str) -> str:
        return sp.sstr(sp.simplify(self.core.parse_expression(expr_str)))

    def expand_expr(self, expr_str: str) -> str:
        return sp.sstr(sp.expand(self.core.parse_expression(expr_str)))

    def factor_expr(self, expr_str: str) -> str:
        return sp.sstr(sp.factor(self.core.parse_expression(expr_str)))

    def solve_minimize(self, expr_str: str, var: str = 'x',
                       a: float = -10, b: float = 10) -> str:
        if not _SCIPY_OK:
            return "SciPy не установлен"
        expr = self.core.parse_expression(expr_str)
        v = self.core.get_symbol(var)
        f = sp.lambdify(v, expr, modules=['numpy'])

        def fn(x):
            try:
                val = float(f(x))
                return val if np.isfinite(val) else 1e10
            except Exception:
                return 1e10

        res = scipy_optimize.minimize_scalar(fn, bounds=(a, b), method='bounded')
        return (f"Минимум в x = {res.x:.6g}, f(x) = {res.fun:.6g}\n"
                f"Итераций: {res.nfev}")

    def solve_maximize(self, expr_str: str, var: str = 'x',
                       a: float = -10, b: float = 10) -> str:
        if not _SCIPY_OK:
            return "SciPy не установлен"
        expr = self.core.parse_expression(expr_str)
        v = self.core.get_symbol(var)
        f = sp.lambdify(v, expr, modules=['numpy'])

        def fn(x):
            try:
                val = float(f(x))
                return -val if np.isfinite(val) else 1e10
            except Exception:
                return 1e10

        res = scipy_optimize.minimize_scalar(fn, bounds=(a, b), method='bounded')
        return (f"Максимум в x = {res.x:.6g}, f(x) = {-res.fun:.6g}")

    def matrix_from_str(self, s: str) -> sp.Matrix:
        """'[[1,2],[3,4]]' -> Matrix."""
        rows = re.findall(r'\[([^\[\]]+)\]', s)
        data = []
        for row in rows:
            data.append([self.core.parse_expression(x.strip())
                         for x in row.split(',')])
        return sp.Matrix(data)

    def matrix_ops(self, A_str: str, B_str: Optional[str] = None) -> Dict[str, str]:
        A = self.matrix_from_str(A_str)
        out = {
            'Матрица A': sp.sstr(A),
            'Определитель': sp.sstr(A.det()),
            'Ранг': str(A.rank()),
            'Транспонированная': sp.sstr(A.T),
        }
        if A.is_square:
            try:
                out['Обратная'] = sp.sstr(A.inv())
            except Exception:
                out['Обратная'] = "не существует (det = 0)"
            try:
                ev = A.eigenvals()
                ev_str = ', '.join(f"{k}: кратность {v}" for k, v in ev.items())
                out['Собственные значения'] = ev_str
            except Exception:
                pass
        if B_str:
            B = self.matrix_from_str(B_str)
            try:
                out['A + B'] = sp.sstr(A + B)
                out['A * B'] = sp.sstr(A * B)
            except Exception as e:
                out['A * B'] = f"ошибка: {e}"
        return out


# ════════════════════════════════════════════════════════════════════
# ГРАФИКИ
# ════════════════════════════════════════════════════════════════════

@dataclass
class PlotConfig:
    x_min: float = -10.0
    x_max: float = 10.0
    y_min: float = -5.0
    y_max: float = 5.0
    n_points: int = DEFAULT_POINTS_2D


class PlotManager:
    def __init__(self, container: tk.Frame, theme: Dict[str, str]):
        self.container = container
        self.theme = theme
        self.figure: Optional[Figure] = None
        self.canvas: Optional[FigureCanvasTkAgg] = None
        self.axes = None
        self.toolbar = None
        self.is_3d = False
        self.is_rotating = False
        self.rotation_angle = 0
        self._rotation_job = None

    def clear(self):
        for w in self.container.winfo_children():
            w.destroy()
        self.figure = None
        self.canvas = None
        self.axes = None
        self.toolbar = None
        self.is_3d = False
        self.is_rotating = False
        if self._rotation_job:
            try:
                self.container.after_cancel(self._rotation_job)
            except Exception:
                pass
            self._rotation_job = None

    def _new_figure(self, width=6.5, height=5.0) -> Figure:
        fig = Figure(figsize=(width, height), dpi=110)
        fig.patch.set_facecolor(self.theme['bg'])
        return fig

    def _finalize(self, fig: Figure):
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, self.container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        toolbar = NavigationToolbar2Tk(canvas, self.container)
        toolbar.update()
        self.figure = fig
        self.canvas = canvas
        self.toolbar = toolbar
        self._bind_zoom(canvas)

    def _bind_zoom(self, canvas):
        def on_scroll(event):
            if event.inaxes is None:
                return
            ax = event.inaxes
            factor = 1.2 if event.button == 'up' else 1 / 1.2
            xl = ax.get_xlim()
            yl = ax.get_ylim()
            xc = event.xdata
            yc = event.ydata
            if xc is None or yc is None:
                return
            ax.set_xlim([xc - (xc - xl[0]) * factor, xc + (xl[1] - xc) * factor])
            ax.set_ylim([yc - (yc - yl[0]) * factor, yc + (yl[1] - yc) * factor])
            canvas.draw_idle()
        canvas.mpl_connect('scroll_event', on_scroll)

    def plot_2d(self, functions: List[str], cfg: PlotConfig, core: VladikCore):
        self.clear()
        fig = self._new_figure()
        ax = fig.add_subplot(111)
        ax.set_facecolor(self.theme['bg'])

        any_plotted = False
        for i, fn in enumerate(functions):
            try:
                xv, yv = core.compute_values(fn, 'x', cfg.x_min, cfg.x_max, cfg.n_points)
                ax.plot(xv, yv, color=PLOT_COLORS[i % len(PLOT_COLORS)],
                        linewidth=2.2, label=fn)
                any_plotted = True
            except Exception as e:
                log.warning("plot %s: %s", fn, e)
        if not any_plotted:
            raise ValueError("Не удалось построить ни одну функцию")

        ax.spines['left'].set_position(('outward', 10))
        ax.spines['bottom'].set_position(('outward', 10))
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.grid(True, alpha=0.15, color='#2d2d2d', linestyle='--')
        ax.set_xlabel('x', color=self.theme['fg_secondary'])
        ax.set_ylabel('f(x)', color=self.theme['fg_secondary'])
        ax.set_title('Графики функций', color=self.theme['fg'])
        ax.tick_params(colors=self.theme['fg_secondary'])
        ax.set_xlim(cfg.x_min, cfg.x_max)
        if cfg.y_min != cfg.y_max:
            ax.set_ylim(cfg.y_min, cfg.y_max)
        else:
            ax.autoscale()
        if len(functions) > 1:
            ax.legend(facecolor=self.theme['bg_tertiary'],
                      edgecolor=self.theme['border'],
                      labelcolor=self.theme['fg'])

        self.axes = ax
        self.is_3d = False
        self._finalize(fig)

    def plot_3d(self, func: str, cfg: PlotConfig, core: VladikCore):
        self.clear()
        fig = self._new_figure()
        X, Y, Z = core.plot_3d(func, 'x', 'y',
                               cfg.x_min, cfg.x_max, cfg.y_min, cfg.y_max)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor(self.theme['bg'])
        ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.9,
                        linewidth=0, antialiased=True)
        ax.set_xlabel('x', color=self.theme['fg_secondary'])
        ax.set_ylabel('y', color=self.theme['fg_secondary'])
        ax.set_zlabel('z', color=self.theme['fg_secondary'])
        ax.set_title('3D поверхность', color=self.theme['fg'])
        ax.tick_params(colors=self.theme['fg_secondary'])
        self.axes = ax
        self.is_3d = True
        self._finalize(fig)

    def plot_polar(self, func: str, cfg: PlotConfig, core: VladikCore):
        self.clear()
        fig = self._new_figure()
        ax = fig.add_subplot(111, projection='polar')
        ax.set_facecolor(self.theme['bg'])
        x, y = core.compute_polar(func, 0, 2 * np.pi, cfg.n_points)
        r = np.sqrt(x * x + y * y)
        theta = np.arctan2(y, x)
        ax.plot(theta, r, color=PLOT_COLORS[0], linewidth=2)
        ax.set_title(f"Полярный: r = {func}", color=self.theme['fg'])
        ax.tick_params(colors=self.theme['fg_secondary'])
        self.axes = ax
        self.is_3d = False
        self._finalize(fig)

    def plot_parametric(self, x_expr: str, y_expr: str,
                        cfg: PlotConfig, core: VladikCore):
        self.clear()
        fig = self._new_figure()
        ax = fig.add_subplot(111)
        ax.set_facecolor(self.theme['bg'])
        x, y = core.compute_parametric(x_expr, y_expr, 0, 2 * np.pi, cfg.n_points)
        ax.plot(x, y, color=PLOT_COLORS[0], linewidth=2)
        ax.grid(True, alpha=0.15, color='#2d2d2d', linestyle='--')
        ax.set_xlabel('x', color=self.theme['fg_secondary'])
        ax.set_ylabel('y', color=self.theme['fg_secondary'])
        ax.set_title(f"x = {x_expr}, y = {y_expr}", color=self.theme['fg'])
        ax.tick_params(colors=self.theme['fg_secondary'])
        self.axes = ax
        self.is_3d = False
        self._finalize(fig)

    def plot_implicit(self, expr_str: str, cfg: PlotConfig, core: VladikCore):
        self.clear()
        fig = self._new_figure()
        ax = fig.add_subplot(111)
        ax.set_facecolor(self.theme['bg'])
        X, Y, Z = core.compute_implicit(expr_str,
                                        cfg.x_min, cfg.x_max,
                                        cfg.y_min, cfg.y_max)
        ax.contour(X, Y, Z, levels=[0], colors=[PLOT_COLORS[0]], linewidths=2)
        ax.grid(True, alpha=0.15, color='#2d2d2d', linestyle='--')
        ax.set_xlabel('x', color=self.theme['fg_secondary'])
        ax.set_ylabel('y', color=self.theme['fg_secondary'])
        ax.set_title(f"Неявный: {expr_str} = 0", color=self.theme['fg'])
        ax.tick_params(colors=self.theme['fg_secondary'])
        self.axes = ax
        self.is_3d = False
        self._finalize(fig)

    def export(self, path: str):
        if self.figure is None:
            raise ValueError("Нет графика для экспорта")
        self.figure.savefig(path, dpi=300, bbox_inches='tight',
                            facecolor=self.theme['bg'])

    def toggle_rotation(self) -> bool:
        if not self.is_3d or self.axes is None:
            return False
        self.is_rotating = not self.is_rotating
        if self.is_rotating:
            self._rotate()
        else:
            if self._rotation_job:
                try:
                    self.container.after_cancel(self._rotation_job)
                except Exception:
                    pass
                self._rotation_job = None
        return self.is_rotating

    def _rotate(self):
        if not self.is_rotating or self.axes is None:
            return
        self.rotation_angle = (self.rotation_angle + 2) % 360
        try:
            self.axes.view_init(elev=25, azim=self.rotation_angle)
            if self.canvas:
                self.canvas.draw_idle()
        except Exception:
            self.is_rotating = False
            return
        if self.is_rotating:
            self._rotation_job = self.container.after(40, self._rotate)


# ════════════════════════════════════════════════════════════════════
# GUI
# ════════════════════════════════════════════════════════════════════

class VladikGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.config = AppConfig.load()
        self.theme = THEMES[self.config.theme]
        self.core = VladikCore()
        self.solver = MathSolver(self.core)
        self.advanced = AdvancedMath(self.core)
        self.history = History()

        self.root.title("Vladik MATEMATIK v2.0")
        self.root.geometry(f"{self.config.window_width}x{self.config.window_height}"
                           f"+{self.config.window_x}+{self.config.window_y}")
        self.root.minsize(1100, 700)
        self.root.configure(bg=self.theme['bg'])

        self._setup_fonts()
        self._build_ui()
        self._bind_hotkeys()

        self.status_var = tk.StringVar(value="✅ Готово")
        self.status_bar = tk.Label(self.root, textvariable=self.status_var,
                                   relief=tk.FLAT, anchor=tk.W,
                                   font=self.fonts['status'],
                                   bg=self.theme['bg'],
                                   fg=self.theme['text_secondary'])
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=3)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self._switch_tab(self.config.last_tab)

    # ------------------------------------------------------------------
    # FONTS
    # ------------------------------------------------------------------
    def _setup_fonts(self):
        self.fonts = {
            'title': ('Segoe UI', 17, 'bold'),
            'tab_active': ('Segoe UI', 11, 'bold'),
            'tab_inactive': ('Segoe UI', 11),
            'label_bold': ('Segoe UI', 11, 'bold'),
            'label': ('Segoe UI', 10),
            'button_bold': ('Segoe UI', 11, 'bold'),
            'button': ('Segoe UI', 10),
            'entry': ('Consolas', 12),
            'output': ('Consolas', 11),
            'status': ('Segoe UI', 9),
            'small': ('Segoe UI', 9),
        }

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self):
        main = tk.Frame(self.root, bg=self.theme['bg'])
        main.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        top = tk.Frame(main, bg=self.theme['bg_secondary'], height=70)
        top.pack(fill=tk.X, pady=(0, 6))
        top.pack_propagate(False)

        tk.Label(top, text="Vladik MATEMATIK v2.0",
                 font=self.fonts['title'],
                 bg=self.theme['bg_secondary'],
                 fg=self.theme['accent']).pack(side=tk.RIGHT, padx=18, pady=16)

        self.tab_buttons: Dict[str, tk.Button] = {}
        tabs = [
            ("Уравнения", "equations"),
            ("Математика", "calculus"),
            ("Матрицы", "matrix"),
            ("Графики", "plots"),
            ("История", "history"),
            ("📚 Справка", "help"),
        ]
        for text, name in tabs:
            btn = tk.Button(top, text=text,
                            bg=self.theme['bg_tertiary'],
                            fg=self.theme['fg_secondary'],
                            font=self.fonts['tab_inactive'],
                            relief=tk.FLAT, borderwidth=0,
                            padx=14, pady=11,
                            activebackground=self.theme['accent'],
                            activeforeground='white',
                            command=lambda n=name: self._switch_tab(n))
            btn.pack(side=tk.LEFT, padx=2)
            self.tab_buttons[name] = btn

        self.content = tk.Frame(main, bg=self.theme['bg'])
        self.content.pack(fill=tk.BOTH, expand=True)

        self.tabs: Dict[str, tk.Frame] = {}
        self._tab_equations()
        self._tab_calculus()
        self._tab_matrix()
        self._tab_plots()
        self._tab_history()
        self._tab_help()

    def _new_tab(self, name: str) -> tk.Frame:
        f = tk.Frame(self.content, bg=self.theme['bg_secondary'])
        self.tabs[name] = f
        return f

    def _switch_tab(self, name: str):
        if name not in self.tabs:
            name = 'equations'
        for _, fr in self.tabs.items():
            fr.pack_forget()
        self.tabs[name].pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.config.last_tab = name
        for n, btn in self.tab_buttons.items():
            if n == name:
                btn.config(bg=self.theme['accent'], fg='white',
                           font=self.fonts['tab_active'])
            else:
                btn.config(bg=self.theme['bg_tertiary'],
                           fg=self.theme['fg_secondary'],
                           font=self.fonts['tab_inactive'])

    # ------------------------------------------------------------------
    # TAB: EQUATIONS
    # ------------------------------------------------------------------
    def _tab_equations(self):
        f = self._new_tab('equations')

        header = tk.Frame(f, bg=self.theme['bg_secondary'])
        header.pack(fill=tk.X, padx=10, pady=8)
        tk.Label(header, text="Решение уравнений, систем, неравенств и ДУ",
                 font=self.fonts['label_bold'],
                 bg=self.theme['bg_secondary'],
                 fg=self.theme['accent_light']).pack(side=tk.LEFT)

        self.complex_var = tk.BooleanVar(value=self.config.complex_mode)
        tk.Checkbutton(header, text="Режим комплексных чисел",
                       variable=self.complex_var,
                       bg=self.theme['bg_secondary'],
                       fg=self.theme['fg_secondary'],
                       selectcolor=self.theme['bg_tertiary'],
                       activebackground=self.theme['bg_secondary'],
                       font=self.fonts['small']).pack(side=tk.RIGHT, padx=8)

        tk.Label(f, text="Задача:",
                 bg=self.theme['bg_secondary'],
                 fg=self.theme['fg_secondary'],
                 font=self.fonts['label']).pack(anchor='w', padx=12)

        self.eq_input = tk.Text(f, height=3,
                                bg=self.theme['input_bg'],
                                fg=self.theme['fg'],
                                font=self.fonts['entry'],
                                insertbackground=self.theme['accent'],
                                relief=tk.FLAT, padx=12, pady=8,
                                wrap=tk.WORD,
                                highlightthickness=1,
                                highlightcolor=self.theme['accent'],
                                highlightbackground=self.theme['border'])
        self.eq_input.pack(fill=tk.X, padx=12, pady=6)
        self.eq_input.insert('1.0', "x^2 - 5x + 6 = 0")

        btns = tk.Frame(f, bg=self.theme['bg_secondary'])
        btns.pack(fill=tk.X, padx=12, pady=4)

        self._make_primary_button(btns, "Решить", self._solve_equation).pack(side=tk.LEFT, padx=4)
        self._make_button(btns, "🗑 Очистить", self._clear_eq_output).pack(side=tk.LEFT, padx=4)
        self._make_button(btns, "📂 Пример", self._insert_example).pack(side=tk.LEFT, padx=4)

        tk.Label(f, text="Примеры: x^2-4=0  |  x+y=3; x-y=1  |  x^2+1=0 (компл.)  |  y''+y=0",
                 bg=self.theme['bg_secondary'],
                 fg=self.theme['text_secondary'],
                 font=self.fonts['small']).pack(anchor='w', padx=12, pady=(2, 4))

        self.eq_output = scrolledtext.ScrolledText(f,
                                                   bg=self.theme['output_bg'],
                                                   fg=self.theme['fg'],
                                                   font=self.fonts['output'],
                                                   insertbackground=self.theme['accent'],
                                                   relief=tk.FLAT,
                                                   padx=12, pady=8,
                                                   wrap=tk.WORD,
                                                   highlightthickness=0)
        self.eq_output.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)

    # ------------------------------------------------------------------
    # TAB: CALCULUS
    # ------------------------------------------------------------------
    def _tab_calculus(self):
        f = self._new_tab('calculus')

        tk.Label(f, text="Производные, интегралы, пределы, ряды, оптимизация",
                 font=self.fonts['label_bold'],
                 bg=self.theme['bg_secondary'],
                 fg=self.theme['accent_light']).pack(anchor='w', padx=12, pady=8)

        row = tk.Frame(f, bg=self.theme['bg_secondary'])
        row.pack(fill=tk.X, padx=12, pady=4)
        tk.Label(row, text="f(x) =",
                 bg=self.theme['bg_secondary'],
                 fg=self.theme['fg_secondary'],
                 font=self.fonts['label']).pack(side=tk.LEFT)
        self.calc_input = tk.Entry(row,
                                   bg=self.theme['input_bg'],
                                   fg=self.theme['fg'],
                                   font=self.fonts['entry'],
                                   insertbackground=self.theme['accent'],
                                   relief=tk.FLAT)
        self.calc_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        self.calc_input.insert(0, "sin(x)/x")

        row2 = tk.Frame(f, bg=self.theme['bg_secondary'])
        row2.pack(fill=tk.X, padx=12, pady=4)

        tk.Label(row2, text="Переменная:",
                 bg=self.theme['bg_secondary'],
                 fg=self.theme['fg_secondary'],
                 font=self.fonts['label']).pack(side=tk.LEFT)
        self.calc_var = tk.Entry(row2, width=6,
                                 bg=self.theme['input_bg'],
                                 fg=self.theme['fg'],
                                 font=self.fonts['entry'],
                                 insertbackground=self.theme['accent'],
                                 relief=tk.FLAT)
        self.calc_var.insert(0, 'x')
        self.calc_var.pack(side=tk.LEFT, padx=4)

        tk.Label(row2, text="Порядок/Точка:",
                 bg=self.theme['bg_secondary'],
                 fg=self.theme['fg_secondary'],
                 font=self.fonts['label']).pack(side=tk.LEFT, padx=(12, 2))
        self.calc_extra = tk.Entry(row2, width=10,
                                   bg=self.theme['input_bg'],
                                   fg=self.theme['fg'],
                                   font=self.fonts['entry'],
                                   insertbackground=self.theme['accent'],
                                   relief=tk.FLAT)
        self.calc_extra.insert(0, '0')
        self.calc_extra.pack(side=tk.LEFT, padx=4)

        tk.Label(row2, text="a:",
                 bg=self.theme['bg_secondary'],
                 fg=self.theme['fg_secondary'],
                 font=self.fonts['label']).pack(side=tk.LEFT, padx=(12, 2))
        self.calc_a = tk.Entry(row2, width=6,
                               bg=self.theme['input_bg'],
                               fg=self.theme['fg'],
                               font=self.fonts['entry'],
                               insertbackground=self.theme['accent'],
                               relief=tk.FLAT)
        self.calc_a.insert(0, '0')
        self.calc_a.pack(side=tk.LEFT, padx=2)

        tk.Label(row2, text="b:",
                 bg=self.theme['bg_secondary'],
                 fg=self.theme['fg_secondary'],
                 font=self.fonts['label']).pack(side=tk.LEFT, padx=(6, 2))
        self.calc_b = tk.Entry(row2, width=6,
                               bg=self.theme['input_bg'],
                               fg=self.theme['fg'],
                               font=self.fonts['entry'],
                               insertbackground=self.theme['accent'],
                               relief=tk.FLAT)
        self.calc_b.insert(0, '1')
        self.calc_b.pack(side=tk.LEFT, padx=2)

        actions = tk.Frame(f, bg=self.theme['bg_secondary'])
        actions.pack(fill=tk.X, padx=12, pady=6)

        ops = [
            ("d/dx", self._op_derivative),
            ("∫ dx", self._op_integral),
            ("∫ₐᵇ dx", self._op_definite_integral),
            ("lim", self._op_limit),
            ("Ряд", self._op_series),
            ("Упростить", self._op_simplify),
            ("Раскрыть", self._op_expand),
            ("Факторизовать", self._op_factor),
            ("min", self._op_min),
            ("max", self._op_max),
        ]
        for text, cmd in ops:
            self._make_button(actions, text, cmd).pack(side=tk.LEFT, padx=3, pady=2)

        self.calc_output = scrolledtext.ScrolledText(f,
                                                     bg=self.theme['output_bg'],
                                                     fg=self.theme['fg'],
                                                     font=self.fonts['output'],
                                                     insertbackground=self.theme['accent'],
                                                     relief=tk.FLAT,
                                                     padx=12, pady=8,
                                                     wrap=tk.WORD,
                                                     highlightthickness=0)
        self.calc_output.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)

    # ------------------------------------------------------------------
    # TAB: MATRIX
    # ------------------------------------------------------------------
    def _tab_matrix(self):
        f = self._new_tab('matrix')

        tk.Label(f, text="Матрицы: определитель, обратная, ранг, собственные значения",
                 font=self.fonts['label_bold'],
                 bg=self.theme['bg_secondary'],
                 fg=self.theme['accent_light']).pack(anchor='w', padx=12, pady=8)

        tk.Label(f, text="Матрица A (например [[1,2],[3,4]]):",
                 bg=self.theme['bg_secondary'],
                 fg=self.theme['fg_secondary'],
                 font=self.fonts['label']).pack(anchor='w', padx=12)
        self.mat_a = tk.Entry(f, bg=self.theme['input_bg'], fg=self.theme['fg'],
                              font=self.fonts['entry'],
                              insertbackground=self.theme['accent'],
                              relief=tk.FLAT)
        self.mat_a.pack(fill=tk.X, padx=12, pady=4)
        self.mat_a.insert(0, '[[1,2],[3,4]]')

        tk.Label(f, text="Матрица B (опционально):",
                 bg=self.theme['bg_secondary'],
                 fg=self.theme['fg_secondary'],
                 font=self.fonts['label']).pack(anchor='w', padx=12)
        self.mat_b = tk.Entry(f, bg=self.theme['input_bg'], fg=self.theme['fg'],
                              font=self.fonts['entry'],
                              insertbackground=self.theme['accent'],
                              relief=tk.FLAT)
        self.mat_b.pack(fill=tk.X, padx=12, pady=4)

        self._make_primary_button(f, "Вычислить", self._op_matrix).pack(pady=8)
        self._make_button(f, "🗑 Очистить", lambda: self.mat_output.delete('1.0', tk.END)).pack(pady=2)

        self.mat_output = scrolledtext.ScrolledText(f,
                                                    bg=self.theme['output_bg'],
                                                    fg=self.theme['fg'],
                                                    font=self.fonts['output'],
                                                    insertbackground=self.theme['accent'],
                                                    relief=tk.FLAT,
                                                    padx=12, pady=8,
                                                    wrap=tk.WORD,
                                                    highlightthickness=0)
        self.mat_output.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)

    # ------------------------------------------------------------------
    # TAB: PLOTS
    # ------------------------------------------------------------------
    def _tab_plots(self):
        f = self._new_tab('plots')

        paned = tk.PanedWindow(f, orient=tk.HORIZONTAL, bg=self.theme['bg'],
                               sashwidth=4, sashrelief='flat')
        paned.pack(fill=tk.BOTH, expand=True)

        left = tk.Frame(paned, bg=self.theme['bg_secondary'], width=360)
        paned.add(left, width=360)
        self.plot_container = tk.Frame(paned, bg=self.theme['bg'])
        paned.add(self.plot_container)

        tk.Label(left, text="Настройки графиков",
                 font=self.fonts['label_bold'],
                 bg=self.theme['bg_secondary'],
                 fg=self.theme['accent_light']).pack(anchor='w', padx=12, pady=(10, 6))

        # Диапазоны
        rng = tk.LabelFrame(left, text="Диапазоны",
                            bg=self.theme['bg_secondary'],
                            fg=self.theme['fg_secondary'],
                            font=self.fonts['label'])
        rng.pack(fill=tk.X, padx=12, pady=4)

        self.range_entries: Dict[str, tk.Entry] = {}
        for i, (lbl, default) in enumerate([
            ("x_min", str(self.config.x_min)), ("x_max", str(self.config.x_max)),
            ("y_min", str(self.config.y_min)), ("y_max", str(self.config.y_max)),
        ]):
            r, c = divmod(i, 2)
            tk.Label(rng, text=lbl + ":",
                     bg=self.theme['bg_secondary'],
                     fg=self.theme['fg_secondary'],
                     font=self.fonts['small']).grid(row=r, column=c * 2, padx=4, pady=3)
            e = tk.Entry(rng, width=8,
                         bg=self.theme['input_bg'],
                         fg=self.theme['fg'],
                         font=self.fonts['entry'],
                         insertbackground=self.theme['accent'],
                         relief=tk.FLAT)
            e.insert(0, default)
            e.grid(row=r, column=c * 2 + 1, padx=4, pady=3)
            self.range_entries[lbl] = e

        # Тип
        typef = tk.LabelFrame(left, text="Тип графика",
                              bg=self.theme['bg_secondary'],
                              fg=self.theme['fg_secondary'],
                              font=self.fonts['label'])
        typef.pack(fill=tk.X, padx=12, pady=4)
        self.plot_type = tk.StringVar(value='2d')
        for txt, val in [("2D (y=f(x))", '2d'), ("3D (z=f(x,y))", '3d'),
                         ("Полярный r(θ)", 'polar'), ("Параметрический", 'parametric'),
                         ("Неявный F(x,y)=0", 'implicit')]:
            tk.Radiobutton(typef, text=txt, variable=self.plot_type, value=val,
                           bg=self.theme['bg_secondary'],
                           fg=self.theme['fg_secondary'],
                           selectcolor=self.theme['bg_tertiary'],
                           activebackground=self.theme['bg_secondary'],
                           font=self.fonts['small']).pack(anchor='w', padx=4)

        # Функции
        funcf = tk.LabelFrame(left, text="Функции (для 2D — список, для 3D/поляр — первая)",
                              bg=self.theme['bg_secondary'],
                              fg=self.theme['fg_secondary'],
                              font=self.fonts['label'])
        funcf.pack(fill=tk.BOTH, expand=True, padx=12, pady=4)

        self.func_list = tk.Listbox(funcf,
                                    bg=self.theme['input_bg'],
                                    fg=self.theme['fg'],
                                    font=('Consolas', 11),
                                    relief=tk.FLAT, borderwidth=0,
                                    selectmode=tk.SINGLE, height=5,
                                    selectbackground=self.theme['accent'])
        self.func_list.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        for f_default in ["sin(x)", "cos(x)"]:
            self.func_list.insert(tk.END, f_default)

        addrow = tk.Frame(funcf, bg=self.theme['bg_secondary'])
        addrow.pack(fill=tk.X, padx=6, pady=4)
        self.new_func = tk.Entry(addrow,
                                 bg=self.theme['input_bg'],
                                 fg=self.theme['fg'],
                                 font=self.fonts['entry'],
                                 insertbackground=self.theme['accent'],
                                 relief=tk.FLAT)
        self.new_func.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        self.new_func.bind('<Return>', lambda e: self._add_function())
        self._make_button(addrow, "➕", self._add_function, padx=10, pady=3).pack(side=tk.LEFT)

        actionrow = tk.Frame(funcf, bg=self.theme['bg_secondary'])
        actionrow.pack(fill=tk.X, padx=6, pady=4)
        self._make_button(actionrow, "✖ Удалить", self._remove_function).pack(side=tk.LEFT, padx=2)
        self._make_button(actionrow, "🗑 Очистить", self._clear_functions).pack(side=tk.LEFT, padx=2)

        # Кнопки
        act = tk.Frame(left, bg=self.theme['bg_secondary'])
        act.pack(fill=tk.X, padx=12, pady=8)
        self._make_primary_button(act, "📊 Построить", self._plot).pack(fill=tk.X, pady=3)
        self.rotate_btn = self._make_button(act, "🔄 Вращать 3D", self._toggle_rotation)
        self.rotate_btn.pack(fill=tk.X, pady=3)
        self._make_button(act, "💾 Экспорт PNG", self._export_plot).pack(fill=tk.X, pady=3)
        self._make_button(act, "🗑 Очистить график", self._clear_plot).pack(fill=tk.X, pady=3)

        self.plot_manager = PlotManager(self.plot_container, self.theme)
        self.plot_label = tk.Label(self.plot_container,
                                   text="📈 Введите функции и нажмите 'Построить'",
                                   font=('Segoe UI', 13),
                                   bg=self.theme['bg'],
                                   fg=self.theme['fg_secondary'])
        self.plot_label.pack(expand=True)

    # ------------------------------------------------------------------
    # TAB: HISTORY
    # ------------------------------------------------------------------
    def _tab_history(self):
        f = self._new_tab('history')

        top = tk.Frame(f, bg=self.theme['bg_secondary'])
        top.pack(fill=tk.X, padx=12, pady=8)
        tk.Label(top, text="📜 История решений",
                 font=self.fonts['label_bold'],
                 bg=self.theme['bg_secondary'],
                 fg=self.theme['accent_light']).pack(side=tk.LEFT)
        self._make_button(top, "🗑 Очистить всю", self._clear_history).pack(side=tk.RIGHT)

        self.history_text = scrolledtext.ScrolledText(f,
                                                      bg=self.theme['output_bg'],
                                                      fg=self.theme['fg'],
                                                      font=self.fonts['output'],
                                                      insertbackground=self.theme['accent'],
                                                      relief=tk.FLAT,
                                                      padx=12, pady=8,
                                                      wrap=tk.WORD,
                                                      highlightthickness=0)
        self.history_text.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)
        self.history_text.config(state=tk.DISABLED)

    # ------------------------------------------------------------------
    # TAB: HELP
    # ------------------------------------------------------------------
    def _tab_help(self):
        f = self._new_tab('help')

        top = tk.Frame(f, bg=self.theme['bg_secondary'])
        top.pack(fill=tk.X, padx=12, pady=8)
        tk.Label(top, text="📚 Справка",
                 font=self.fonts['label_bold'],
                 bg=self.theme['bg_secondary'],
                 fg=self.theme['accent_light']).pack(side=tk.LEFT)
        self.theme_btn = self._make_button(top, "🌗 Сменить тему", self._toggle_theme)
        self.theme_btn.pack(side=tk.RIGHT)

        txt = scrolledtext.ScrolledText(f,
                                        bg=self.theme['output_bg'],
                                        fg=self.theme['fg_secondary'],
                                        font=self.fonts['label'],
                                        relief=tk.FLAT,
                                        padx=14, pady=12,
                                        wrap=tk.WORD,
                                        highlightthickness=0)
        txt.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)
        txt.insert('1.0', self._help_content())
        txt.config(state=tk.DISABLED)

    @staticmethod
    def _help_content() -> str:
        return """======================================================
              VLADIK MATEMATIK v2.0
======================================================

ПОЛНОСТЬЮ АВТОНОМНЫЙ МАТЕМАТИЧЕСКИЙ ДВИЖОК
Работает офлайн. Использует SymPy, NumPy, SciPy.

──────────────────────────────────────────────────────
📐 УРАВНЕНИЯ
──────────────────────────────────────────────────────
  • x^2 - 5x + 6 = 0
  • x^3 - 2x + 1 = 0
  • x^2 + 1 = 0              (включите «Режим комплексных чисел»)
  • x + y = 3; x - y = 1     (система через ';')
  • x^2 - 4 < 0              (неравенство)

──────────────────────────────────────────────────────
🎓 МАТЕМАТИКА
──────────────────────────────────────────────────────
  • Производная:        d/dx(sin(x^2)) → 2x·cos(x^2)
  • Интеграл:           ∫ x^2 dx → x^3/3 + C
  • Определённый:       ∫₀¹ x^2 dx → 1/3
  • Предел:             lim sin(x)/x при x→0 → 1
  • Ряд Тейлора:        series exp(x) в 0, порядок 6
  • Оптимизация:        min / max на отрезке [a, b]

──────────────────────────────────────────────────────
🔢 МАТРИЦЫ
──────────────────────────────────────────────────────
  Формат: [[1,2],[3,4]]
  • Определитель, ранг, транспонирование
  • Обратная матрица
  • Собственные значения и кратности
  • Сложение и умножение A + B, A · B

──────────────────────────────────────────────────────
📈 ГРАФИКИ
──────────────────────────────────────────────────────
  • 2D:      y = f(x) — несколько функций
  • 3D:      z = f(x, y) — с вращением
  • Полярные: r(θ)
  • Параметрические: x(t), y(t)
  • Неявные:  F(x, y) = 0
  • Экспорт:  PNG 300 dpi
  • Зум колёсиком мыши

──────────────────────────────────────────────────────
⌨️ ГОРЯЧИЕ КЛАВИШИ
──────────────────────────────────────────────────────
  Ctrl+Enter      Решить
  Ctrl+L          Очистить ввод
  Ctrl+S          Сохранить график
  Ctrl+H          История
  Ctrl+T          Сменить тему
  F5              Построить график
  Escape          Очистить вывод

──────────────────────────────────────────────────────
⚙️ ТРЕБОВАНИЯ
──────────────────────────────────────────────────────
  Python 3.8+
  sympy, numpy, matplotlib, scipy (опционально)

======================================================
  Дубна, 2026
"""

    # ------------------------------------------------------------------
    # BUTTONS
    # ------------------------------------------------------------------
    def _make_button(self, parent, text, cmd, padx=12, pady=6, **kw) -> tk.Button:
        btn = tk.Button(parent, text=text,
                        bg=self.theme['bg_tertiary'],
                        fg=self.theme['fg_secondary'],
                        font=self.fonts['button'],
                        relief=tk.FLAT, borderwidth=0,
                        activebackground=self.theme['accent'],
                        activeforeground='white',
                        padx=padx, pady=pady,
                        command=cmd, **kw)

        def on_enter(_):
            btn.config(bg=self.theme['border'], fg=self.theme['fg'])

        def on_leave(_):
            btn.config(bg=self.theme['bg_tertiary'], fg=self.theme['fg_secondary'])

        btn.bind('<Enter>', on_enter)
        btn.bind('<Leave>', on_leave)
        return btn

    def _make_primary_button(self, parent, text, cmd, **kw) -> tk.Button:
        return tk.Button(parent, text=text,
                         bg=self.theme['accent'],
                         fg='white',
                         font=self.fonts['button_bold'],
                         relief=tk.FLAT, borderwidth=0,
                         activebackground=self.theme['accent_hover'],
                         activeforeground='white',
                         padx=18, pady=8,
                         command=cmd, **kw)

    # ------------------------------------------------------------------
    # HOTKEYS
    # ------------------------------------------------------------------
    def _bind_hotkeys(self):
        self.root.bind('<Control-Return>', lambda e: self._solve_equation())
        self.root.bind('<Control-l>', lambda e: self._clear_eq_output())
        self.root.bind('<Control-s>', lambda e: self._export_plot())
        self.root.bind('<Control-h>', lambda e: self._switch_tab('history'))
        self.root.bind('<Control-t>', lambda e: self._toggle_theme())
        self.root.bind('<F5>', lambda e: self._plot())
        self.root.bind('<Escape>', lambda e: self._clear_all_outputs())

    # ------------------------------------------------------------------
    # SOLVE
    # ------------------------------------------------------------------
    def _solve_equation(self):
        problem = self.eq_input.get('1.0', tk.END).strip()
        if not problem:
            messagebox.showwarning("Внимание", "Введите задачу")
            return
        if len(problem) > MAX_INPUT_LENGTH:
            messagebox.showerror("Ошибка", f"Слишком длинный ввод (> {MAX_INPUT_LENGTH})")
            return

        self.status_var.set("⏳ Решение...")
        self.root.after(50, lambda: self._do_solve(problem))

    def _do_solve(self, problem: str):
        try:
            result = self.solver.solve(
                problem,
                problem_type='auto',
                complex_mode=self.complex_var.get(),
            )
        except Exception as e:
            log.exception("solve")
            self.status_var.set("❌ Ошибка")
            messagebox.showerror("Ошибка", str(e))
            return

        self.eq_output.insert(tk.END, f"\n{'─' * 60}\n")
        self.eq_output.insert(tk.END, f"▶ Задача: {problem}\n")
        self.eq_output.insert(tk.END, f"⏱ {humanize_time(result.get('elapsed', 0))}  "
                                      f"[{result.get('engine', 'SymPy')}]\n")

        if result.get('success'):
            verified = result.get('verified')
            if verified is True:
                mark = "✅ (проверено)"
            elif verified is False:
                mark = "⚠️ (проверка не пройдена)"
            else:
                mark = ""
            self.eq_output.insert(tk.END, f"Решение {mark}:\n")
            for line in str(result.get('formatted', '')).split('\n'):
                if line.strip():
                    self.eq_output.insert(tk.END, f"  {line}\n")
            self.status_var.set(f"✅ Готово ({humanize_time(result.get('elapsed', 0))})")
        else:
            self.eq_output.insert(tk.END, f"❌ Ошибка: {result.get('error')}\n")
            self.status_var.set("❌ Ошибка")

        self.eq_output.see(tk.END)

        self.history.add(HistoryEntry(
            timestamp=datetime.now().isoformat(timespec='seconds'),
            problem=problem,
            problem_type=result.get('type', 'equation'),
            result=result.get('formatted', ''),
            success=bool(result.get('success')),
            elapsed=result.get('elapsed', 0.0),
            engine=result.get('engine', 'SymPy'),
        ))
        self._refresh_history()

    def _clear_eq_output(self):
        self.eq_output.delete('1.0', tk.END)

    def _insert_example(self):
        examples = [
            "x^2 - 5x + 6 = 0",
            "x^2 + 1 = 0",
            "x + y = 3; x - y = 1",
            "x^2 - 4 < 0",
            "y'' + y = 0",
            "y' - y = e^x",
        ]
        idx = simpledialog.askinteger("Пример", "Номер (1-6):", minvalue=1, maxvalue=6) or 0
        if 1 <= idx <= 6:
            self.eq_input.delete('1.0', tk.END)
            self.eq_input.insert('1.0', examples[idx - 1])

    # ------------------------------------------------------------------
    # CALCULUS
    # ------------------------------------------------------------------
    def _calc_args(self):
        expr = self.calc_input.get().strip()
        var = self.calc_var.get().strip() or 'x'
        extra = self.calc_extra.get().strip() or '0'
        return expr, var, extra

    def _write_calc(self, text: str, title: str = ""):
        self.calc_output.insert(tk.END, f"\n{'─' * 60}\n")
        if title:
            self.calc_output.insert(tk.END, f"▶ {title}\n")
        self.calc_output.insert(tk.END, f"  {text}\n")
        self.calc_output.see(tk.END)

    def _op_derivative(self):
        expr, var, extra = self._calc_args()
        try:
            order = int(extra) if extra else 1
            res = self.advanced.derivative(expr, var, order)
            self._write_calc(f"d^{order}/d{var}^{order}({expr}) = {res}", "Производная")
            self.status_var.set("✅ Производная")
        except Exception as e:
            self._write_calc(f"Ошибка: {e}", "Производная")

    def _op_integral(self):
        expr, var, _ = self._calc_args()
        try:
            res = self.advanced.integral(expr, var)
            self._write_calc(f"∫ {expr} d{var} = {res}", "Неопределённый интеграл")
            self.status_var.set("✅ Интеграл")
        except Exception as e:
            self._write_calc(f"Ошибка: {e}", "Интеграл")

    def _op_definite_integral(self):
        expr, var, _ = self._calc_args()
        try:
            a = float(self.calc_a.get() or 0)
            b = float(self.calc_b.get() or 1)
            res = self.advanced.integral(expr, var, a, b)
            self._write_calc(f"∫_{a}^{b} {expr} d{var} = {res}", "Определённый интеграл")
            self.status_var.set("✅ Интеграл")
        except Exception as e:
            self._write_calc(f"Ошибка: {e}", "Интеграл")

    def _op_limit(self):
        expr, var, point = self._calc_args()
        try:
            res = self.advanced.limit(expr, var, point)
            self._write_calc(f"lim({expr}, {var} → {point}) = {res}", "Предел")
            self.status_var.set("✅ Предел")
        except Exception as e:
            self._write_calc(f"Ошибка: {e}", "Предел")

    def _op_series(self):
        expr, var, point = self._calc_args()
        try:
            order = int(self.calc_a.get() or 6)
            res = self.advanced.series(expr, var, point, order)
            self._write_calc(f"Ряд для {expr} в {var}={point}, порядок {order}:\n  {res}",
                             "Ряд Тейлора")
            self.status_var.set("✅ Ряд")
        except Exception as e:
            self._write_calc(f"Ошибка: {e}", "Ряд")

    def _op_simplify(self):
        expr, _, _ = self._calc_args()
        try:
            res = self.advanced.simplify_expr(expr)
            self._write_calc(f"simplify({expr}) = {res}", "Упрощение")
        except Exception as e:
            self._write_calc(f"Ошибка: {e}", "Упрощение")

    def _op_expand(self):
        expr, _, _ = self._calc_args()
        try:
            res = self.advanced.expand_expr(expr)
            self._write_calc(f"expand({expr}) = {res}", "Раскрытие")
        except Exception as e:
            self._write_calc(f"Ошибка: {e}", "Раскрытие")

    def _op_factor(self):
        expr, _, _ = self._calc_args()
        try:
            res = self.advanced.factor_expr(expr)
            self._write_calc(f"factor({expr}) = {res}", "Факторизация")
        except Exception as e:
            self._write_calc(f"Ошибка: {e}", "Факторизация")

    def _op_min(self):
        expr, var, _ = self._calc_args()
        try:
            a = float(self.calc_a.get() or -10)
            b = float(self.calc_b.get() or 10)
            res = self.advanced.solve_minimize(expr, var, a, b)
            self._write_calc(res, "Минимум")
        except Exception as e:
            self._write_calc(f"Ошибка: {e}", "Минимум")

    def _op_max(self):
        expr, var, _ = self._calc_args()
        try:
            a = float(self.calc_a.get() or -10)
            b = float(self.calc_b.get() or 10)
            res = self.advanced.solve_maximize(expr, var, a, b)
            self._write_calc(res, "Максимум")
        except Exception as e:
            self._write_calc(f"Ошибка: {e}", "Максимум")

    # ------------------------------------------------------------------
    # MATRIX
    # ------------------------------------------------------------------
    def _op_matrix(self):
        a_str = self.mat_a.get().strip()
        b_str = self.mat_b.get().strip() or None
        if not a_str:
            messagebox.showwarning("Внимание", "Введите матрицу A")
            return
        try:
            res = self.advanced.matrix_ops(a_str, b_str)
            self.mat_output.insert(tk.END, f"\n{'─' * 60}\n")
            for k, v in res.items():
                self.mat_output.insert(tk.END, f"{k}:\n  {v}\n")
            self.mat_output.see(tk.END)
            self.status_var.set("✅ Матрицы")
        except Exception as e:
            self.mat_output.insert(tk.END, f"\n❌ Ошибка: {e}\n")
            self.mat_output.see(tk.END)

    # ------------------------------------------------------------------
    # PLOTS
    # ------------------------------------------------------------------
    def _plot_cfg(self) -> PlotConfig:
        def get(k):
            try:
                return float(self.range_entries[k].get())
            except Exception:
                return 0.0
        cfg = PlotConfig(x_min=get('x_min'), x_max=get('x_max'),
                         y_min=get('y_min'), y_max=get('y_max'))
        self.config.x_min, self.config.x_max = cfg.x_min, cfg.x_max
        self.config.y_min, self.config.y_max = cfg.y_min, cfg.y_max
        return cfg

    def _add_function(self):
        fn = self.new_func.get().strip()
        if fn:
            self.func_list.insert(tk.END, fn)
            self.new_func.delete(0, tk.END)

    def _remove_function(self):
        sel = self.func_list.curselection()
        if sel:
            self.func_list.delete(sel[0])

    def _clear_functions(self):
        self.func_list.delete(0, tk.END)

    def _plot(self):
        functions = list(self.func_list.get(0, tk.END))
        if not functions:
            messagebox.showwarning("Внимание", "Добавьте хотя бы одну функцию")
            return

        cfg = self._plot_cfg()
        ptype = self.plot_type.get()

        if self.plot_label is not None:
            try:
                self.plot_label.destroy()
            except Exception:
                pass
            self.plot_label = None

        try:
            if ptype == '2d':
                self.plot_manager.plot_2d(functions, cfg, self.core)
            elif ptype == '3d':
                if len(functions) > 1:
                    messagebox.showinfo("Информация", "В 3D строится только первая функция")
                self.plot_manager.plot_3d(functions[0], cfg, self.core)
            elif ptype == 'polar':
                self.plot_manager.plot_polar(functions[0], cfg, self.core)
            elif ptype == 'parametric':
                if len(functions) < 2:
                    messagebox.showwarning("Внимание",
                                           "Для параметрического нужны 2 функции: x(t), y(t)")
                    return
                self.plot_manager.plot_parametric(functions[0], functions[1],
                                                  cfg, self.core)
            elif ptype == 'implicit':
                self.plot_manager.plot_implicit(functions[0], cfg, self.core)
            self.status_var.set("📊 Построено")
        except Exception as e:
            log.exception("plot")
            self.status_var.set(f"❌ {e}")
            messagebox.showerror("Ошибка графика", str(e))

    def _toggle_rotation(self):
        if not self.plot_manager.toggle_rotation():
            messagebox.showinfo("Информация", "Сначала постройте 3D график")
            return
        if self.plot_manager.is_rotating:
            self.rotate_btn.config(text="⏸ Остановить")
        else:
            self.rotate_btn.config(text="🔄 Вращать 3D")

    def _export_plot(self):
        if self.plot_manager.figure is None:
            messagebox.showinfo("Информация", "Сначала постройте график")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")],
            initialfile="vladik_plot.png",
        )
        if path:
            try:
                self.plot_manager.export(path)
                self.status_var.set(f"💾 Сохранено: {os.path.basename(path)}")
            except Exception as e:
                messagebox.showerror("Ошибка", str(e))

    def _clear_plot(self):
        self.plot_manager.clear()
        self.plot_label = tk.Label(self.plot_container,
                                   text="📈 Введите функции и нажмите 'Построить'",
                                   font=('Segoe UI', 13),
                                   bg=self.theme['bg'],
                                   fg=self.theme['fg_secondary'])
        self.plot_label.pack(expand=True)
        self.rotate_btn.config(text="🔄 Вращать 3D")
        self.status_var.set("🗑 График очищен")

    # ------------------------------------------------------------------
    # HISTORY
    # ------------------------------------------------------------------
    def _refresh_history(self):
        self.history_text.config(state=tk.NORMAL)
        self.history_text.delete('1.0', tk.END)
        for e in self.history.entries:
            mark = "✅" if e.success else "❌"
            self.history_text.insert(
                tk.END,
                f"{mark} [{e.timestamp}] ({e.problem_type}, "
                f"{humanize_time(e.elapsed)})\n"
                f"   {truncate(e.problem, 120)}\n"
                f"   → {truncate(e.result.replace(chr(10), ' | '), 200)}\n\n"
            )
        self.history_text.config(state=tk.DISABLED)

    def _clear_history(self):
        if messagebox.askyesno("Подтверждение", "Очистить всю историю?"):
            self.history.clear()
            self._refresh_history()

    # ------------------------------------------------------------------
    # THEME
    # ------------------------------------------------------------------
    def _toggle_theme(self):
        self.config.theme = 'light' if self.config.theme == 'dark' else 'dark'
        self.theme = THEMES[self.config.theme]
        messagebox.showinfo("Тема",
                            "Тема изменена. Перезапустите приложение для применения.")

    # ------------------------------------------------------------------
    # MISC
    # ------------------------------------------------------------------
    def _clear_all_outputs(self):
        for w in (self.eq_output, self.calc_output, self.mat_output):
            try:
                w.delete('1.0', tk.END)
            except Exception:
                pass

    def on_close(self):
        try:
            self.config.complex_mode = bool(self.complex_var.get())
            try:
                geo = self.root.geometry().split('+')
                size = geo[0].split('x')
                self.config.window_width = int(size[0])
                self.config.window_height = int(size[1])
                if len(geo) >= 3:
                    self.config.window_x = int(geo[1])
                    self.config.window_y = int(geo[2])
            except Exception:
                pass
            self.config.save()
            self.plot_manager.clear()
        finally:
            gc.collect()
            self.root.destroy()


# ════════════════════════════════════════════════════════════════════
# ЗАПУСК
# ════════════════════════════════════════════════════════════════════

def main():
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    root = tk.Tk()
    app = VladikGUI(root)  # noqa
    root.mainloop()


if __name__ == "__main__":
    main()