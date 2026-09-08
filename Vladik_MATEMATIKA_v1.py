"""
VLADIK MATEMATIKA v28.2 - Красные элементы на темном фоне
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import sympy as sp
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import re
import json
import os
from datetime import datetime
import threading
import gc
from typing import Optional, Tuple, List, Any, Dict
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# НАСТРОЙКИ ГРАФИКОВ
# ============================================================================

plt.rcParams['figure.dpi'] = 120
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['lines.linewidth'] = 2.5

try:
    from ctypes import windll
    windll.shcore.SetProcessDpiAwareness(1)
except:
    pass

# ============================================================================
# КОНВЕРТЕР LATEX В ТЕКСТ
# ============================================================================

class TextConverter:
    """Преобразование математических выражений в читаемый текст"""
    
    @staticmethod
    def clean_output(text: str) -> str:
        """Очистка и форматирование вывода"""
        if not text:
            return ""
        
        # Убираем специальные токены
        text = re.sub(r'<\|.*?\|>', '', text)
        text = re.sub(r'\[.*?\]', '', text)
        
        # Заменяем математические обозначения
        replacements = {
            '**': '^',
            '*': '·',
            'sin': 'sin',
            'cos': 'cos',
            'tan': 'tan',
            'cot': 'cot',
            'sec': 'sec',
            'csc': 'csc',
            'asin': 'arcsin',
            'acos': 'arccos',
            'atan': 'arctan',
            'sinh': 'sh',
            'cosh': 'ch',
            'tanh': 'th',
            'sqrt': '√',
            'exp': 'e^',
            'log': 'log',
            'ln': 'ln',
            'pi': 'π',
            'inf': '∞',
            'oo': '∞'
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Очищаем лишние пробелы
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text

# ============================================================================
# УЛУЧШЕННЫЙ РЕШАТЕЛЬ
# ============================================================================

class EnhancedMathSolver:
    """Улучшенный решатель с русским выводом"""
    
    def __init__(self):
        self.core = VladikCore()
        self._cache = {}
    
    def solve_equation(self, equation: str) -> Dict[str, Any]:
        """Решение уравнения"""
        result = {
            'success': False,
            'solution': None,
            'error': None,
            'type': 'unknown',
            'formatted': ''
        }
        
        # Проверяем на дифференциальное уравнение
        if "y''" in equation or "y'" in equation or "y(" in equation:
            try:
                var = sp.Symbol('x')
                func = sp.Function('y')(var)
                
                # Преобразуем строку
                expr_str = equation.replace('y', 'y(x)')
                expr_str = expr_str.replace("y''", "Derivative(y(x), x, x)")
                expr_str = expr_str.replace("y'", "Derivative(y(x), x)")
                
                # Заменяем функции
                expr_str = re.sub(r'sin\(', 'sp.sin(', expr_str)
                expr_str = re.sub(r'cos\(', 'sp.cos(', expr_str)
                expr_str = re.sub(r'tan\(', 'sp.tan(', expr_str)
                expr_str = re.sub(r'cot\(', 'sp.cot(', expr_str)
                expr_str = re.sub(r'sec\(', 'sp.sec(', expr_str)
                expr_str = re.sub(r'csc\(', 'sp.csc(', expr_str)
                expr_str = re.sub(r'exp\(', 'sp.exp(', expr_str)
                expr_str = re.sub(r'log\(', 'sp.log(', expr_str)
                expr_str = re.sub(r'ln\(', 'sp.log(', expr_str)
                expr_str = re.sub(r'sqrt\(', 'sp.sqrt(', expr_str)
                expr_str = re.sub(r'π', 'sp.pi', expr_str)
                expr_str = re.sub(r'pi', 'sp.pi', expr_str)
                expr_str = re.sub(r'e', 'sp.E', expr_str)
                
                # Парсим
                expr = eval(expr_str, {"sp": sp, "__builtins__": {}}, self.core._symbols)
                
                # Создаем уравнение
                eq = sp.Eq(expr, 0)
                
                # Решаем
                solution = sp.dsolve(eq, func)
                
                if solution:
                    result['success'] = True
                    result['solution'] = solution
                    result['type'] = 'ODE'
                    result['formatted'] = TextConverter.clean_output(str(solution))
                    return result
            except Exception as e:
                result['error'] = str(e)
        
        # Обычное уравнение
        try:
            eq = equation.strip()
            if '=' in eq:
                left, right = eq.split('=')
                left_expr = self.core.parse_expression(left)
                right_expr = self.core.parse_expression(right)
                eq_obj = sp.Eq(left_expr, right_expr)
            else:
                expr = self.core.parse_expression(eq)
                eq_obj = sp.Eq(expr, 0)
            
            symbols = list(eq_obj.free_symbols)
            if symbols:
                var = symbols[0]
                sympy_solution = sp.solve(eq_obj, var)
                
                if sympy_solution:
                    result['success'] = True
                    result['type'] = 'algebraic'
                    
                    # Форматируем решения
                    solutions = []
                    for sol in sympy_solution:
                        if not sol.is_infinite:
                            sol_str = TextConverter.clean_output(str(sol))
                            solutions.append(sol_str)
                    
                    if solutions:
                        result['formatted'] = 'x = ' + ', '.join(solutions)
                    else:
                        result['formatted'] = 'Нет действительных решений'
                    
                    return result
        except Exception as e:
            result['error'] = str(e)
        
        return result

# ============================================================================
# ОПТИМИЗИРОВАННЫЙ ЗАГРУЗЧИК МОДЕЛИ
# ============================================================================

class ModelLoader:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        self._initialized = True
        self._model = None
        self._tokenizer = None
        self._ready = False
        self._loading = False
        self._status = "Ожидание"
        self._error = None
        
    def load(self) -> bool:
        if self._ready:
            return True
        if self._loading:
            return False
            
        self._loading = True
        self._status = "Загрузка системы..."
        
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            model_name = "Qwen/Qwen2.5-Math-1.5B"
            
            self._status = "Настройка системы..."
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                use_fast=True
            )
            
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._status = f"Загрузка системы на {device}..."
            
            self._model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                use_cache=True
            )
            
            self._model.eval()
            
            if device == "cpu":
                try:
                    self._model = torch.compile(self._model, mode="reduce-overhead")
                except:
                    pass
            
            self._ready = True
            self._status = f"✅ Готово ({device})"
            return True
            
        except Exception as e:
            self._error = str(e)
            self._status = f"❌ Ошибка: {str(e)[:80]}"
            self._model = None
            self._tokenizer = None
            return False
        finally:
            self._loading = False
            gc.collect()
    
    def is_ready(self) -> bool:
        return self._ready
    
    def is_loading(self) -> bool:
        return self._loading
    
    def get_status(self) -> str:
        return self._status
    
    def solve(self, problem: str, max_length: int = 1024) -> str:
        """Решение с увеличенной длиной ответа"""
        if not self._ready or self._model is None:
            return "⚠️ Модель не загружена"
        
        try:
            import torch
            
            # Улучшенный промпт на русском
            if "y''" in problem or "y'" in problem:
                prompt = f"""Реши дифференциальное уравнение пошагово:

{problem}

Найди общее решение y(x).

Решение:"""
            else:
                prompt = f"""Реши уравнение пошагово:

{problem}

Найди все действительные решения для x.

Решение:"""
            
            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=False
            )
            
            device = next(self._model.parameters()).device
            if device.type == 'cuda':
                inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self._tokenizer.eos_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                    use_cache=True,
                    num_beams=1,
                    early_stopping=False
                )
            
            solution = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Извлекаем ответ
            if "Решение:" in solution:
                solution = solution.split("Решение:")[-1].strip()
            elif "Ответ:" in solution:
                solution = solution.split("Ответ:")[-1].strip()
            elif "Solution:" in solution:
                solution = solution.split("Solution:")[-1].strip()
            
            # Очищаем
            solution = TextConverter.clean_output(solution)
            
            return solution.strip() or "Решение не найдено"
            
        except Exception as e:
            return f"❌ Ошибка: {str(e)}"
    
    def cleanup(self):
        if self._model is not None:
            try:
                import torch
                if hasattr(self._model, 'to'):
                    self._model.to('cpu')
                del self._model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            except:
                pass
            self._model = None
            self._tokenizer = None
            self._ready = False
            gc.collect()

# ============================================================================
# ОПТИМИЗИРОВАННОЕ ЯДРО ВЫЧИСЛЕНИЙ
# ============================================================================

class VladikCore:
    def __init__(self):
        self._symbols = {}
        self._cache = {}
        self._max_cache_size = 100
        
        for name in ['x', 'y', 'z', 't', 'a', 'b', 'c', 'n']:
            if name == 'n':
                self._symbols[name] = sp.Symbol(name, integer=True)
            else:
                self._symbols[name] = sp.Symbol(name)
        
        self._symbols.update({
            'pi': sp.pi, 'e': sp.E, 'inf': sp.oo, 'I': sp.I
        })
    
    def parse_expression(self, expr_str: str) -> sp.Expr:
        cache_key = expr_str
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            expr_str = expr_str.replace('^', '**')
            
            replacements = {
                'sin': 'sp.sin', 'cos': 'sp.cos', 'tan': 'sp.tan',
                'cot': 'sp.cot', 'sec': 'sp.sec', 'csc': 'sp.csc',
                'asin': 'sp.asin', 'acos': 'sp.acos', 'atan': 'sp.atan',
                'sinh': 'sp.sinh', 'cosh': 'sp.cosh', 'tanh': 'sp.tanh',
                'sqrt': 'sp.sqrt', 'log': 'sp.log', 'ln': 'sp.log',
                'exp': 'sp.exp', 'abs': 'sp.Abs',
                'π': 'sp.pi', '∞': 'sp.oo'
            }
            
            for old, new in replacements.items():
                expr_str = re.sub(rf'{old}\s*\(', f'{new}(', expr_str)
            
            expr = eval(expr_str, {"sp": sp, "__builtins__": {}}, self._symbols)
            
            if len(self._cache) >= self._max_cache_size:
                keys = list(self._cache.keys())[:self._max_cache_size//2]
                for key in keys:
                    del self._cache[key]
            
            self._cache[cache_key] = expr
            return expr
            
        except Exception as e:
            raise ValueError(f"Ошибка парсинга: {str(e)}")
    
    def compute_values(self, expr_str: str, var: str = 'x', 
                      x_min: float = -10, x_max: float = 10, 
                      n_points: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
        try:
            expr = self.parse_expression(expr_str)
            var_symbol = self._symbols.get(var, sp.Symbol(var))
            
            f = sp.lambdify(var_symbol, expr, modules=['numpy'])
            
            x_vals = np.linspace(x_min, x_max, n_points, dtype=np.float64)
            
            try:
                y_vals = f(x_vals)
            except Exception:
                y_vals = np.array([self._safe_eval(f, x) for x in x_vals])
            
            if np.iscomplexobj(y_vals):
                y_vals = np.real(y_vals)
            
            y_vals = np.where(np.isinf(y_vals) | np.isnan(y_vals) | (np.abs(y_vals) > 1e10), 
                             np.nan, y_vals)
            
            return x_vals, y_vals
            
        except Exception as e:
            raise ValueError(f"Ошибка вычислений: {str(e)}")
    
    @staticmethod
    def _safe_eval(f, x):
        try:
            val = f(x)
            if np.iscomplex(val):
                val = np.real(val)
            if np.isinf(val) or np.isnan(val) or np.abs(val) > 1e10:
                return np.nan
            return float(val)
        except:
            return np.nan
    
    def plot_3d(self, expr_str: str, var1: str = 'x', var2: str = 'y',
                x_min: float = -5, x_max: float = 5,
                y_min: float = -5, y_max: float = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        try:
            expr = self.parse_expression(expr_str)
            var1_sym = self._symbols.get(var1, sp.Symbol(var1))
            var2_sym = self._symbols.get(var2, sp.Symbol(var2))
            
            f = sp.lambdify((var1_sym, var2_sym), expr, modules=['numpy'])
            
            n_points = 60
            x_vals = np.linspace(x_min, x_max, n_points)
            y_vals = np.linspace(y_min, y_max, n_points)
            X, Y = np.meshgrid(x_vals, y_vals)
            
            try:
                Z = f(X, Y)
            except:
                Z = np.zeros_like(X)
                for i in range(X.shape[0]):
                    for j in range(X.shape[1]):
                        Z[i,j] = self._safe_eval_2d(f, X[i,j], Y[i,j])
            
            if np.iscomplexobj(Z):
                Z = np.real(Z)
            
            Z = np.where(np.isinf(Z) | np.isnan(Z) | (np.abs(Z) > 1e10), np.nan, Z)
            
            return X, Y, Z
            
        except Exception as e:
            raise ValueError(f"Ошибка 3D: {str(e)}")
    
    @staticmethod
    def _safe_eval_2d(f, x, y):
        try:
            val = f(x, y)
            if np.iscomplex(val):
                val = np.real(val)
            if np.isinf(val) or np.isnan(val) or np.abs(val) > 1e10:
                return np.nan
            return float(val)
        except:
            return np.nan

# ============================================================================
# GUI - КРАСНЫЕ КНОПКИ, ВКЛАДКИ И НАДПИСИ НА ТЕМНОМ ФОНЕ
# ============================================================================

class VladikGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Vladik MATEMATIKA v1")
        self.root.geometry("1300x800")
        self.root.minsize(1100, 700)
        self.root.configure(bg='#0a0a0a')
        
        self._setup_dpi()
        
        self.core = VladikCore()
        self.model_loader = ModelLoader()
        self.enhanced_solver = EnhancedMathSolver()
        
        self.current_tab = 'neural'
        self.is_rotating = False
        self.rotation_angle = 0
        self.plot_figure = None
        self.plot_canvas = None
        self.plot_axes = None
        self.is_3d_plot = False
        
        self._setup_fonts()
        self._setup_colors()
        self._create_widgets()
        self._setup_bindings()
        
        self.status_var = tk.StringVar(value="✅ Готово")
        self.status_bar = tk.Label(
            self.root, 
            textvariable=self.status_var,
            relief=tk.FLAT, 
            anchor=tk.W,
            font=self.fonts['status'],
            bg='#0a0a0a', 
            fg='#666666'
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=3)
        
        self._start_model_loading()
        self._show_default_content()
    
    def _setup_dpi(self):
        try:
            from ctypes import windll
            windll.shcore.SetProcessDpiAwareness(1)
        except:
            pass
    
    def _setup_fonts(self):
        self.fonts = {
            'title': ('Segoe UI', 18, 'bold'),
            'tab_active': ('Segoe UI', 11, 'bold'),
            'tab_inactive': ('Segoe UI', 11),
            'label_bold': ('Segoe UI', 11, 'bold'),
            'label': ('Segoe UI', 10),
            'label_small': ('Segoe UI', 9),
            'button_bold': ('Segoe UI', 11, 'bold'),
            'button': ('Segoe UI', 10),
            'entry': ('Consolas', 12),
            'output': ('Consolas', 11),
            'status': ('Segoe UI', 9),
        }
    
    def _setup_colors(self):
        """КРАСНЫЕ КНОПКИ, ВКЛАДКИ И НАДПИСИ НА ТЕМНОМ ФОНЕ"""
        self.colors = {
            'bg': '#0a0a0a',           # Темный фон
            'bg_secondary': '#0d0d0d',  # Чуть светлее
            'bg_tertiary': '#141414',   # Ещё светлее
            'fg': '#ffffff',           # Белый текст
            'fg_secondary': '#b0b0b0',  # Светло-серый текст
            'text_secondary': '#666666', # Серый текст
            'accent': '#cc0000',        # Ярко-красный акцент
            'accent_hover': '#ff0000',  # Ярко-красный при наведении
            'accent_light': '#ff3333',  # Светлый красный
            'success': '#2ecc71',       # Зелёный для успеха
            'warning': '#f1c40f',       # Жёлтый для предупреждений
            'error': '#e74c3c',         # Красный для ошибок
            'border': '#1a1a1a',        # Граница
            'button_bg': '#0d0d0d',     # Фон кнопки
            'button_hover': '#1a1a1a',  # Фон кнопки при наведении
            'input_bg': '#0d0d0d',      # Фон ввода
            'output_bg': '#0a0a0a',     # Фон вывода
            'tab_bg': '#0d0d0d',        # Фон вкладки
            'tab_active': '#cc0000',    # Активная вкладка (КРАСНАЯ)
            'button_small': '#0d0d0d'   # Маленькая кнопка
        }
    
    def _create_widgets(self):
        main_frame = tk.Frame(self.root, bg='#0a0a0a')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        top_panel = tk.Frame(main_frame, bg='#0d0d0d', height=75)
        top_panel.pack(fill=tk.X, pady=(0, 6))
        top_panel.pack_propagate(False)
        
        # КРАСНЫЙ ЗАГОЛОВОК
        title = tk.Label(
            top_panel, 
            text="Vladik MATEMATIKA v1",
            font=self.fonts['title'],
            bg='#0d0d0d', 
            fg='#cc0000'  # КРАСНЫЙ
        )
        title.pack(side=tk.RIGHT, padx=18, pady=14)
        
        self.tab_buttons = {}
        tabs = [("Решение уравнений", 'neural'), ("📈 Графики", 'plots'), ("📚 Справка", 'help')]
        
        for text, name in tabs:
            btn = tk.Button(
                top_panel,
                text=text,
                bg=self.colors['tab_active'] if name == 'neural' else self.colors['tab_bg'],
                fg='white' if name == 'neural' else self.colors['fg_secondary'],
                font=self.fonts['tab_active'] if name == 'neural' else self.fonts['tab_inactive'],
                relief=tk.FLAT,
                borderwidth=0,
                padx=18,
                pady=11,
                command=lambda n=name: self._switch_tab(n)
            )
            btn.pack(side=tk.LEFT, padx=2)
            self.tab_buttons[name] = btn
        
        self.content = tk.Frame(main_frame, bg='#0a0a0a')
        self.content.pack(fill=tk.BOTH, expand=True)
        
        self.tabs = {}
        self._create_neural_tab()
        self._create_plots_tab()
        self._create_help_tab()
        
        self._switch_tab('neural')
    
    def _create_neural_tab(self):
        frame = tk.Frame(self.content, bg='#0d0d0d')
        frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.tabs['neural'] = frame
        
        top = tk.Frame(frame, bg='#0d0d0d')
        top.pack(fill=tk.X, padx=6, pady=6)
        
        # КРАСНАЯ НАДПИСЬ
        tk.Label(
            top,
            text="Vladik MATEMATIKA v1",
            font=self.fonts['label_bold'],
            bg='#0d0d0d',
            fg='#ff3333'  # КРАСНЫЙ
        ).pack(anchor='w')
        
        self.model_status = tk.Label(
            top,
            text="⏳ Загрузка системы...",
            font=self.fonts['label'],
            bg='#0d0d0d',
            fg='#f1c40f'
        )
        self.model_status.pack(anchor='w', pady=3)
        
        type_frame = tk.Frame(top, bg='#0d0d0d')
        type_frame.pack(fill=tk.X, pady=6)
        
        tk.Label(
            type_frame,
            text="Тип:",
            font=self.fonts['label'],
            bg='#0d0d0d',
            fg='#b0b0b0'
        ).pack(side=tk.LEFT, padx=6)
        
        self.problem_type = tk.StringVar(value="equation")
        for text, value in [("Уравнение", "equation"), ("Система", "system"), ("ДУ", "ode")]:
            tk.Radiobutton(
                type_frame,
                text=text,
                variable=self.problem_type,
                value=value,
                bg='#0d0d0d',
                fg='#b0b0b0',
                selectcolor='#0d0d0d',
                font=self.fonts['label']
            ).pack(side=tk.LEFT, padx=10)
        
        tk.Label(
            frame,
            text="Задача:",
            font=self.fonts['label'],
            bg='#0d0d0d',
            fg='#b0b0b0'
        ).pack(anchor='w', padx=6)
        
        self.input_text = tk.Text(
            frame,
            height=3,
            bg='#0d0d0d',
            fg='#ffffff',
            font=self.fonts['entry'],
            wrap=tk.WORD,
            relief=tk.FLAT,
            padx=12,
            pady=8,
            highlightthickness=1,
            highlightcolor='#cc0000',  # КРАСНЫЙ
            highlightbackground='#1a1a1a'
        )
        self.input_text.pack(fill=tk.X, padx=6, pady=6)
        self.input_text.insert('1.0', "x^2 - 5x + 6 = 0")
        
        btn_frame = tk.Frame(frame, bg='#0d0d0d')
        btn_frame.pack(fill=tk.X, padx=6, pady=6)
        
        # КРАСНАЯ КНОПКА
        solve_btn = tk.Button(
            btn_frame,
            text="Решить",
            bg='#cc0000',  # КРАСНЫЙ
            fg='white',
            font=self.fonts['button_bold'],
            relief=tk.FLAT,
            padx=18,
            pady=8,
            borderwidth=0,
            command=self._solve_neural
        )
        solve_btn.pack(side=tk.LEFT, padx=4)
        
        clear_btn = self._create_button(
            btn_frame,
            "🗑 Очистить",
            self._clear_neural_output,
            fg_color='#b0b0b0'
        )
        clear_btn.pack(side=tk.LEFT, padx=4)
        
        self.loading_label = tk.Label(
            frame,
            text="",
            font=self.fonts['label'],
            bg='#0d0d0d',
            fg='#f1c40f'
        )
        self.loading_label.pack(pady=3)
        
        self.output_text = scrolledtext.ScrolledText(
            frame,
            bg='#0a0a0a',
            fg='#ffffff',
            font=self.fonts['output'],
            wrap=tk.WORD,
            height=10,
            insertbackground='#cc0000',  # КРАСНЫЙ
            padx=12,
            pady=8,
            highlightthickness=0,
            borderwidth=0
        )
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
    
    def _create_plots_tab(self):
        frame = tk.Frame(self.content, bg='#0a0a0a')
        frame.pack(fill=tk.BOTH, expand=True)
        self.tabs['plots'] = frame
        
        paned = tk.PanedWindow(
            frame,
            orient=tk.HORIZONTAL,
            bg='#0a0a0a',
            sashwidth=4,
            sashrelief='flat'
        )
        paned.pack(fill=tk.BOTH, expand=True)
        
        left = tk.Frame(paned, bg='#0a0a0a', width=380)
        paned.add(left, width=380)
        
        self.plot_container = tk.Frame(paned, bg='#0a0a0a')
        paned.add(self.plot_container)
        
        # КРАСНАЯ НАДПИСЬ
        tk.Label(
            left,
            text="Настройки графиков",
            font=self.fonts['label_bold'],
            bg='#0d0d0d',
            fg='#ff3333'  # КРАСНЫЙ
        ).pack(anchor='w', padx=12, pady=(10, 6))
        
        range_frame = tk.LabelFrame(
            left,
            text="Диапазоны",
            bg='#0d0d0d',
            fg='#b0b0b0',
            font=self.fonts['label']
        )
        range_frame.pack(fill=tk.X, padx=12, pady=6)
        
        self.range_entries = {}
        for i, (label, default) in enumerate([
            ("x от:", "-10"), ("до:", "10"),
            ("y от:", "-5"), ("до:", "5")
        ]):
            row = i // 2
            col = (i % 2) * 2
            tk.Label(
                range_frame,
                text=label,
                bg='#0d0d0d',
                fg='#b0b0b0',
                font=self.fonts['label']
            ).grid(row=row, column=col, padx=6, pady=3)
            
            entry = tk.Entry(
                range_frame,
                width=8,
                bg='#0d0d0d',
                fg='#ffffff',
                font=self.fonts['entry'],
                relief=tk.FLAT,
                borderwidth=0
            )
            entry.insert(0, default)
            entry.grid(row=row, column=col+1, padx=6, pady=3)
            self.range_entries[label] = entry
        
        func_frame = tk.LabelFrame(
            left,
            text="Функции",
            bg='#0d0d0d',
            fg='#b0b0b0',
            font=self.fonts['label']
        )
        func_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=6)
        
        self.func_listbox = tk.Listbox(
            func_frame,
            bg='#0d0d0d',
            fg='#ffffff',
            font=('Consolas', 11),
            relief=tk.FLAT,
            borderwidth=0,
            selectmode=tk.SINGLE,
            height=5
        )
        self.func_listbox.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.func_listbox.insert(tk.END, "sin(x)")
        self.func_listbox.insert(tk.END, "cos(x)")
        
        add_frame = tk.Frame(func_frame, bg='#0d0d0d')
        add_frame.pack(fill=tk.X, padx=6, pady=6)
        
        self.new_func_entry = tk.Entry(
            add_frame,
            bg='#0d0d0d',
            fg='#ffffff',
            font=self.fonts['entry'],
            relief=tk.FLAT,
            borderwidth=0
        )
        self.new_func_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        self.new_func_entry.bind('<Return>', lambda e: self._add_function())
        
        add_btn = self._create_button(
            add_frame,
            "➕",
            self._add_function,
            padx=10,
            pady=3
        )
        add_btn.pack(side=tk.LEFT)
        
        btn_frame = tk.Frame(func_frame, bg='#0d0d0d')
        btn_frame.pack(fill=tk.X, padx=6, pady=6)
        
        remove_btn = self._create_button(
            btn_frame,
            "✖ Удалить",
            self._remove_function,
            fg_color='#b0b0b0'
        )
        remove_btn.pack(side=tk.LEFT, padx=3)
        
        clear_funcs_btn = self._create_button(
            btn_frame,
            "🗑 Очистить",
            self._clear_functions,
            fg_color='#b0b0b0'
        )
        clear_funcs_btn.pack(side=tk.LEFT, padx=3)
        
        action_frame = tk.Frame(left, bg='#0d0d0d')
        action_frame.pack(fill=tk.X, padx=12, pady=10)
        
        # КРАСНАЯ КНОПКА
        plot_btn = tk.Button(
            action_frame,
            text="📊 Построить",
            bg='#cc0000',  # КРАСНЫЙ
            fg='white',
            font=self.fonts['button_bold'],
            relief=tk.FLAT,
            padx=15,
            pady=8,
            borderwidth=0,
            command=self._plot_functions
        )
        plot_btn.pack(fill=tk.X, pady=4)
        
        # КРАСНАЯ КНОПКА
        self.rotate_btn = self._create_button(
            action_frame,
            "🔄 Вращать 3D",
            self._toggle_rotation,
            fg_color='#ff3333',  # КРАСНЫЙ
            pady=8
        )
        self.rotate_btn.pack(fill=tk.X, pady=4)
        
        clear_plot_btn = self._create_button(
            action_frame,
            "🗑 Очистить график",
            self._clear_plot,
            fg_color='#b0b0b0',
            pady=8
        )
        clear_plot_btn.pack(fill=tk.X, pady=4)
        
        self.plot_label = tk.Label(
            self.plot_container,
            text="📈 Введите функции и нажмите 'Построить'",
            font=('Segoe UI', 14),
            bg='#0a0a0a',
            fg='#b0b0b0'
        )
        self.plot_label.pack(expand=True)
    
    def _create_help_tab(self):
        frame = tk.Frame(self.content, bg='#0d0d0d')
        frame.pack(fill=tk.BOTH, expand=True)
        self.tabs['help'] = frame
        
        # КРАСНАЯ НАДПИСЬ
        tk.Label(
            frame,
            text="📚 Справка",
            font=self.fonts['label_bold'],
            bg='#0d0d0d',
            fg='#ff3333'  # КРАСНЫЙ
        ).pack(anchor='w', padx=15, pady=12)
        
        help_text = scrolledtext.ScrolledText(
            frame,
            bg='#0d0d0d',
            fg='#b0b0b0',
            font=self.fonts['label'],
            wrap=tk.WORD,
            padx=15,
            pady=12,
            highlightthickness=0,
            borderwidth=0
        )
        help_text.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)
        
        content = """
======================================================
                    VLADIK MATEMATIKA v28.2                     
======================================================
                                                                  
  Математический движок с интеграцией AI                                              
  • Работает ОФЛАЙН после первой загрузки системы                    
  • Решает уравнения, системы, ДУ                                                             
                                                                  
  📈 ГРАФИКИ:                                                   
  • Классические 2D графики                                     
  • 3D графики с вращением                                      
  • Несколько функций на одном графике                                              
                                                                  
  💡 ПРИМЕРЫ:                                                   
  АЛГЕБРАИЧЕСКИЕ:                                               
  • x^2 - 5x + 6 = 0 → x = 2, 3                               
  • x^2 - 4 = 0 → x = -2, 2                                   
  • x + y = 3, x - y = 1 → x=2, y=1                          
                                                                  
  ДИФФЕРЕНЦИАЛЬНЫЕ:                                             
  • y'' + y = 0 → y = C1*sin(x) + C2*cos(x)                   
  • y' - y = e^x → y = (C + x)*e^x                            
  • y'' - 3y' + 2y = 0 → y = C1*e^x + C2*e^(2x)              
                                                                  
  ⚙️ ТРЕБОВАНИЯ:                                                
  • Python 3.8+                                                
  • 4+ ГБ ОЗУ                                                  
  • ~3 ГБ свободного места                          

======================================================
  Дубна, 2026
        """
        help_text.insert('1.0', content)
        help_text.config(state=tk.DISABLED)
    
    def _setup_bindings(self):
        self.root.bind('<Control-Enter>', lambda e: self._solve_neural())
    
    def _create_button(self, parent, text, command, fg_color='white', **kwargs):
        btn = tk.Button(
            parent,
            text=text,
            bg='#0d0d0d',
            fg=fg_color,
            font=self.fonts['button'],
            relief=tk.FLAT,
            borderwidth=0,
            command=command,
            **kwargs
        )
        
        def on_enter(e):
            btn.config(bg='#1a1a1a')
        
        def on_leave(e):
            btn.config(bg='#0d0d0d', fg=fg_color)
        
        btn.bind('<Enter>', on_enter)
        btn.bind('<Leave>', on_leave)
        return btn
    
    def _switch_tab(self, tab_name: str):
        for name, frame in self.tabs.items():
            frame.pack_forget()
        
        self.tabs[tab_name].pack(fill=tk.BOTH, expand=True)
        self.current_tab = tab_name
        
        for name, btn in self.tab_buttons.items():
            if name == tab_name:
                btn.config(
                    bg='#cc0000',  # КРАСНЫЙ
                    fg='white',
                    font=self.fonts['tab_active']
                )
            else:
                btn.config(
                    bg='#0d0d0d',
                    fg='#b0b0b0',
                    font=self.fonts['tab_inactive']
                )
    
    def _start_model_loading(self):
        def load():
            success = self.model_loader.load()
            self.root.after(0, self._update_model_status)
        
        thread = threading.Thread(target=load, daemon=True)
        thread.start()
    
    def _update_model_status(self):
        status = self.model_loader.get_status()
        if self.model_loader.is_ready():
            self.model_status.config(text=f"✅ {status}", fg='#2ecc71')
            self.status_var.set("✅ Модель готова")
        elif self.model_loader.is_loading():
            self.model_status.config(text=f"⏳ {status}", fg='#f1c40f')
            self.status_var.set(f"⏳ {status}")
            self.root.after(2000, self._update_model_status)
        else:
            self.model_status.config(text=f"⚠️ {status}", fg='#e74c3c')
            self.status_var.set(f"⚠️ {status}")
    
    def _solve_neural(self):
        problem = self.input_text.get('1.0', tk.END).strip()
        if not problem:
            messagebox.showwarning("Предупреждение", "Введите задачу")
            return
        
        # Сначала пробуем SymPy
        try:
            result = self.enhanced_solver.solve_equation(problem)
            if result['success']:
                self.loading_label.config(text="")
                self.output_text.insert(tk.END, f"\n{'─'*60}\n")
                self.output_text.insert(tk.END, f"▶ Задача: {problem}\n")
                self.output_text.insert(tk.END, f"✅ Решение: {result['formatted']}\n")
                self.output_text.see(tk.END)
                self.status_var.set("✅ Решено!")
                return
        except:
            pass
        
        # Используем нейросеть
        if not self.model_loader.is_ready():
            self._display_result(problem, "⚠️ Система загружается... Пожалуйста, подождите.")
            return
        
        self.loading_label.config(text="⏳ Идёт процесс решения...")
        self.root.update()
        
        def solve():
            solution = self.model_loader.solve(problem)
            self.root.after(0, lambda: self._display_result(problem, solution))
        
        thread = threading.Thread(target=solve, daemon=True)
        thread.start()
    
    def _display_result(self, problem: str, solution: str):
        """Отображение результата"""
        self.loading_label.config(text="")
        self.output_text.insert(tk.END, f"\n{'─'*60}\n")
        self.output_text.insert(tk.END, f"▶ Задача: {problem}\n")
        
        lines = solution.split('\n')
        for line in lines:
            if line.strip():
                self.output_text.insert(tk.END, f"  {line.strip()}\n")
        
        self.output_text.see(tk.END)
        self.status_var.set("✅ Решено!")
    
    def _clear_neural_output(self):
        self.output_text.delete('1.0', tk.END)
        self.loading_label.config(text="")
        self._show_default_content()
    
    def _show_default_content(self):
        content = """VLADIK MATEMATIKA v1

📌 Введите задачу и нажмите "Решить"

📌 ПРИМЕРЫ:
  • x^2 - 5x + 6 = 0
  • x^2 - 4 = 0
  • y'' + y = 0
  • y' - y = e^x

💡 Скиньте пожалуста пожертвования разработчику на номер 89269781125. Он очень хочет кушать:(
        """
        self.output_text.insert('1.0', content)
    
    def _add_function(self):
        func = self.new_func_entry.get().strip()
        if func:
            self.func_listbox.insert(tk.END, func)
            self.new_func_entry.delete(0, tk.END)
            self.status_var.set(f"✅ Добавлено: {func}")
    
    def _remove_function(self):
        selection = self.func_listbox.curselection()
        if selection:
            self.func_listbox.delete(selection[0])
            self.status_var.set("🗑 Функция удалена")
    
    def _clear_functions(self):
        self.func_listbox.delete(0, tk.END)
        self.status_var.set("🗑 Все функции очищены")
    
    def _get_range(self, label: str) -> float:
        try:
            return float(self.range_entries[label].get())
        except:
            return 0.0
    
    def _plot_functions(self):
        """Построение графиков с ОРИГИНАЛЬНЫМИ цветами"""
        functions = list(self.func_listbox.get(0, tk.END))
        if not functions:
            messagebox.showwarning("Предупреждение", "Добавьте хотя бы одну функцию")
            return
        
        try:
            x_min = self._get_range("x от:")
            x_max = self._get_range("до:")
            y_min = self._get_range("y от:")
            y_max = self._get_range("до:")
        except:
            x_min, x_max, y_min, y_max = -10, 10, -5, 5
        
        for widget in self.plot_container.winfo_children():
            widget.destroy()
        
        try:
            is_3d = False
            first_func = functions[0]
            try:
                expr = self.core.parse_expression(first_func)
                is_3d = len(list(expr.free_symbols)) >= 2
            except:
                is_3d = 'y' in first_func and 'x' in first_func
            
            fig = Figure(figsize=(5.5, 4.5), dpi=120)
            fig.patch.set_facecolor('#0a0a0a')
            
            if is_3d:
                if len(functions) > 1:
                    messagebox.showinfo("Информация", "В 3D строится только первая функция")
                
                X, Y, Z = self.core.plot_3d(
                    functions[0], 'x', 'y',
                    x_min, x_max, y_min, y_max
                )
                
                ax = fig.add_subplot(111, projection='3d')
                ax.set_facecolor('#0a0a0a')
                
                # ОРИГИНАЛЬНАЯ ЦВЕТОВАЯ СХЕМА
                surf = ax.plot_surface(
                    X, Y, Z,
                    cmap='viridis',
                    alpha=0.85,
                    linewidth=0,
                    antialiased=True
                )
                
                ax.set_xlabel('x', color='#b0b0b0', fontsize=12)
                ax.set_ylabel('y', color='#b0b0b0', fontsize=12)
                ax.set_zlabel('z', color='#b0b0b0', fontsize=12)
                ax.set_title('3D График', color='#ffffff', pad=15, fontsize=14)
                ax.tick_params(colors='#b0b0b0', labelsize=10)
                
                self.is_3d_plot = True
                self.plot_axes = ax
                self.status_var.set("🌐 3D график готов")
                
            else:
                ax = fig.add_subplot(111)
                ax.set_facecolor('#0a0a0a')
                
                # ОРИГИНАЛЬНЫЕ ЦВЕТА ГРАФИКОВ
                colors = ['#6c5ce7', '#00b894', '#fdcb6e', '#e17055', '#0984e3', '#fd79a8']
                
                for i, func_str in enumerate(functions):
                    try:
                        x_vals, y_vals = self.core.compute_values(
                            func_str, 'x', x_min, x_max
                        )
                        color = colors[i % len(colors)]
                        ax.plot(
                            x_vals, y_vals,
                            color=color,
                            linewidth=2.5,
                            label=func_str
                        )
                    except Exception as e:
                        messagebox.showerror("Ошибка", f"Функция '{func_str}': {str(e)}")
                        continue
                
                # Классический график
                ax.spines['left'].set_position(('outward', 10))
                ax.spines['bottom'].set_position(('outward', 10))
                ax.spines['right'].set_color('none')
                ax.spines['top'].set_color('none')
                
                ax.grid(True, alpha=0.15, color='#2d2d2d', linestyle='--')
                
                ax.set_xlabel('x', color='#b0b0b0', fontsize=12)
                ax.set_ylabel('f(x)', color='#b0b0b0', fontsize=12)
                ax.set_title('Графики функций', color='#ffffff', pad=15, fontsize=14)
                ax.tick_params(colors='#b0b0b0', labelsize=10)
                
                ax.set_xlim(x_min, x_max)
                if y_min != y_max:
                    ax.set_ylim(y_min, y_max)
                else:
                    ax.autoscale()
                
                if len(functions) > 1:
                    ax.legend(
                        facecolor='#141414',
                        edgecolor='#2d2d2d',
                        labelcolor='#ffffff',
                        fontsize=10
                    )
                
                self.is_3d_plot = False
                self.plot_axes = ax
                self.status_var.set("📊 Графики готовы")
            
            fig.tight_layout()
            
            canvas = FigureCanvasTkAgg(fig, self.plot_container)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            toolbar = NavigationToolbar2Tk(canvas, self.plot_container)
            toolbar.update()
            
            self.plot_canvas = canvas
            self.plot_figure = fig
            
        except Exception as e:
            error_label = tk.Label(
                self.plot_container,
                text=f"❌ Ошибка: {str(e)}",
                font=('Segoe UI', 12),
                bg='#0a0a0a',
                fg='#e74c3c'
            )
            error_label.pack(expand=True)
            self.status_var.set(f"❌ Ошибка: {str(e)}")
    
    def _toggle_rotation(self):
        if not self.is_3d_plot or not self.plot_axes:
            messagebox.showinfo("Информация", "Сначала постройте 3D график")
            return
        
        self.is_rotating = not self.is_rotating
        if self.is_rotating:
            self.rotate_btn.config(text="⏸ Остановить", fg='#e74c3c')
            self._rotate_3d_plot()
        else:
            self.rotate_btn.config(text="🔄 Вращать 3D", fg='#ff3333')
    
    def _rotate_3d_plot(self):
        if not self.is_rotating or not self.plot_axes:
            return
        
        self.rotation_angle += 1
        self.plot_axes.view_init(elev=20, azim=self.rotation_angle % 360)
        if self.plot_canvas:
            self.plot_canvas.draw()
        
        if self.is_rotating:
            self.root.after(30, self._rotate_3d_plot)
    
    def _clear_plot(self):
        for widget in self.plot_container.winfo_children():
            widget.destroy()
        
        self.plot_label = tk.Label(
            self.plot_container,
            text="📈 Введите функции и нажмите 'Построить'",
            font=('Segoe UI', 14),
            bg='#0a0a0a',
            fg='#b0b0b0'
        )
        self.plot_label.pack(expand=True)
        
        self.plot_canvas = None
        self.plot_figure = None
        self.plot_axes = None
        self.is_3d_plot = False
        self.is_rotating = False
        self.rotate_btn.config(text="🔄 Вращать 3D", fg='#ff3333')
        self.status_var.set("🗑 График очищен")
        gc.collect()
    
    def __del__(self):
        if hasattr(self, 'model_loader'):
            self.model_loader.cleanup()

# ============================================================================
# ЗАПУСК
# ============================================================================

def main():
    root = tk.Tk()
    app = VladikGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()