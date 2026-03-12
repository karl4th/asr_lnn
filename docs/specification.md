# DREAM: Полная техническая спецификация

**Dynamic Recall and Elastic Adaptive Memory**

Версия: 0.2.1  
Дата: Март 2026

---

## Содержание

1. [Обзор архитектуры](#1-обзор-архитектуры)
2. [Структура проекта](#2-структура-проекта)
3. [Блок 1: Predictive Coding](#3-блок-1-predictive-coding)
4. [Блок 2: Surprise Gate](#4-блок-2-surprise-gate)
5. [Блок 3: Fast Weights](#5-блок-3-fast-weights)
6. [Блок 4: Liquid Time-Constants](#6-блок-4-liquid-time-constants)
7. [Блок 5: Sleep Consolidation](#7-блок-5-sleep-consolidation)
8. [DREAM Cell: интеграция блоков](#8-dream-cell-интеграция-блоков)
9. [Координация между слоями](#9-координация-между-слоями)
10. [Конфигурация](#10-конфигурация)
11. [Состояние (State)](#11-состояние-state)
12. [API и примеры](#12-api-и-примеры)

---

## 1. Обзор архитектуры

**DREAM** — это непрерывная рекуррентная сеть с пятью модульными блоками:

```
┌────────────────────────────────────────────────────────────────┐
│                       DREAM Cell                               │
│                                                                │
│  x ──→ ┌──────────────┐ ──→ e ──→ ┌──────────────┐ ──→ S     │
│        │ 1.Predictive │         │ 2.Surprise   │            │
│  h ──→ │   Coding     │         │   Gate       │            │
│        └──────────────┘         └──────────────┘            │
│                              │                               │
│                              ↓                               │
│                       ┌──────────────┐                       │
│                       │ 3.Fast       │ ──→ U                │
│                       │   Weights    │                       │
│                       └──────────────┘                       │
│                              │                               │
│                              ↓                               │
│  x ──→ ┌──────────────┐ ──→ u_eff ──→ ┌──────────────┐      │
│        │   Base +     │               │ 4.LTC        │ ──→ h'│
│        │   Fast       │               │              │      │
│        └──────────────┘               └──────────────┘      │
│                              │                               │
│                              ↓                               │
│                       ┌──────────────┐                       │
│                       │ 5.Sleep      │ ──→ U_target         │
│                       │ Consolidation│                       │
│                       └──────────────┘                       │
└────────────────────────────────────────────────────────────────┘
```

**Обязательные блоки:** Predictive Coding, Surprise Gate  
**Опциональные блоки:** Fast Weights, LTC, Sleep

---

## 2. Структура проекта

```
dream/
├── __init__.py              # Экспорт API
├── config.py                # DREAMConfig (параметры)
├── state.py                 # DREAMState (состояние)
├── cell.py                  # DREAMCell (интеграция)
├── layer.py                 # DREAM (высокоуровневый)
├── stack.py                 # DREAMStack (многослойный)
└── layers/                  # Отдельные блоки
    ├── __init__.py
    ├── predictive_coding.py   # Блок 1
    ├── surprise_gate.py       # Блок 2
    ├── fast_weights.py        # Блок 3
    ├── ltc.py                 # Блок 4
    ├── sleep_consolidation.py # Блок 5
    └── coordination.py        # Координация (опционально)
```

---

## 3. Блок 1: Predictive Coding

**Файл:** `dream/layers/predictive_coding.py`

**Назначение:** Предсказание входа и вычисление ошибки предсказания.

### 3.1. Параметры

| Параметр | Форма | Описание |
|----------|-------|----------|
| `C` | `(hidden_dim, input_dim)` | Декодирующая матрица |
| `W` | `(input_dim, hidden_dim)` | Матрица инъекции ошибки |
| `B_base` | `(input_dim, hidden_dim)` | Базовая проекция входа |

### 3.2. Формулы

**Предсказание:**
```
x_pred = tanh(C^T @ h)
```
где:
- `h ∈ ℝ^(batch × hidden_dim)` — скрытое состояние
- `C ∈ ℝ^(hidden_dim × input_dim)` — декодирующая матрица
- `x_pred ∈ ℝ^(batch × input_dim)` — предсказание

**Ошибка предсказания:**
```
e = x - x_pred
```
где:
- `x ∈ ℝ^(batch × input_dim)` — вход
- `e ∈ ℝ^(batch × input_dim)` — ошибка

**Проекция входа:**
```
base_effect = x @ B_base
```
где:
- `B_base ∈ ℝ^(input_dim × hidden_dim)`
- `base_effect ∈ ℝ^(batch × hidden_dim)`

**Инъекция ошибки:**
```
error_injection = e @ W
```
где:
- `W ∈ ℝ^(input_dim × hidden_dim)`
- `error_injection ∈ ℝ^(batch × hidden_dim)`

### 3.3. Код

```python
class PredictiveCoding(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        self.C = nn.Parameter(torch.randn(hidden_dim, input_dim) * 0.1)
        self.W = nn.Parameter(torch.randn(input_dim, hidden_dim) * 0.1)
        self.B_base = nn.Parameter(torch.randn(input_dim, hidden_dim) * 0.1)
    
    def forward(self, x, h):
        x_pred = torch.tanh(h @ self.C)
        error = x - x_pred
        return x_pred, error
    
    def project_input(self, x):
        return x @ self.B_base
    
    def inject_error(self, error):
        return error @ self.W
```

---

## 4. Блок 2: Surprise Gate

**Файл:** `dream/layers/surprise_gate.py`

**Назначение:** Вычисление surprise как статистической аномалии через z-оценку.

### 4.1. Параметры

| Параметр | Форма | По умолчанию | Описание |
|----------|-------|--------------|----------|
| `τ_base` | скаляр | 2.0 | Базовый порог (z-score при S=0.5) |
| `γ` | скаляр | 1.0 | Температура сигмоиды |
| `κ` | скаляр | 0.5 | Коэффициент gain модуляции |
| `β` | скаляр | 0.05 | Сглаживание статистики |

### 4.2. Формулы

**Шаг 1: Норма ошибки**
```
||e|| = √(Σ e_i²)  # L2 норма, форма: (batch,)
```

**Шаг 2: Clip выбросов**
```
clipped_error = clamp(||e||, min=μ, max=μ+3σ)
```
где:
- `μ` —_running mean_ ошибки
- `σ` —_running std_ ошибки
- Clip предотвращает коррупцию статистики выбросами

**Шаг 3: Обновление статистики**
```
μ_new = (1 - β) · μ + β · clipped_error
σ²_new = (1 - β) · σ² + β · (clipped_error - μ_new)²
```

**Шаг 4: Z-оценка**
```
z = (||e|| - μ_new) / (σ_new + ε)
```
где `ε = 1e-6` для численной стабильности

**Шаг 5: Surprise**
```
S = sigmoid((z - τ_base) / γ)
```
где:
- `S ∈ [0, 1]` — коэффициент удивления
- `τ_base = 2.0` — при z=2, S=0.5
- `γ = 1.0` — крутизна сигмоиды

**Шаг 6: Gain модуляция**
```
gain = 1 + κ · S
```
где:
- `gain ∈ ℝ^(batch × 1)` — модуляция входа

### 4.3. Код

```python
class SurpriseGate(nn.Module):
    def __init__(self, hidden_dim, base_threshold=2.0, 
                 surprise_temperature=1.0, kappa=0.5, 
                 error_smoothing=0.05):
        self.tau_base = nn.Parameter(torch.tensor(base_threshold))
        self.gamma = nn.Parameter(torch.tensor(surprise_temperature))
        self.kappa = nn.Parameter(torch.tensor(kappa))
        self.beta = error_smoothing
    
    def update_statistics(self, error_norm, mu, sigma_sq):
        sigma = torch.sqrt(sigma_sq + 1e-6)
        # Clip выбросов
        clipped = torch.clamp(error_norm, min=mu, max=mu + 3*sigma)
        # Экспоненциальное сглаживание
        mu_new = (1 - self.beta) * mu + self.beta * clipped
        sigma_sq_new = (1 - self.beta) * sigma_sq + self.beta * (clipped - mu_new)**2
        return mu_new, sigma_sq_new
    
    def forward(self, error, error_var, error_mean, state_mu, state_sigma):
        error_norm = error.norm(dim=-1)
        mu_new, sigma_sq_new = self.update_statistics(
            error_norm, state_mu, state_sigma**2
        )
        sigma_new = torch.sqrt(sigma_sq_new + 1e-6)
        
        # Z-оценка
        z_score = (error_norm - mu_new) / (sigma_new + 1e-6)
        
        # Surprise
        surprise = torch.sigmoid((z_score - self.tau_base) / self.gamma)
        gain = 1.0 + self.kappa * surprise.unsqueeze(1)
        
        return surprise, error_norm, gain, mu_new, sigma_new
```

---

## 5. Блок 3: Fast Weights

**Файл:** `dream/layers/fast_weights.py`

**Назначение:** Быстрые веса с Hebбовым обучением и surprise-модуляцией.

### 5.1. Параметры

| Параметр | Форма | Описание |
|----------|-------|----------|
| `V` | `(input_dim, rank)` | Фиксированный ортогональный фильтр |
| `η` | `(hidden_dim,)` | Вектор пластичности |
| `U` | `(batch, hidden_dim, rank)` | Быстрые веса (в состоянии) |
| `U_target` | `(batch, hidden_dim, rank)` | Целевые веса (в состоянии) |

### 5.2. Формулы

**Инициализация V (ортогонализация):**
```
V_init ~ N(0, 0.1)
Q, R = qr(V_init)
V = Q  # Ортогональная матрица
```

**Вклад быстрых весов:**
```
fast_effect = (U @ V^T) @ x = einsum('bhr,ir,bi→bh', U, V, x)
```
где:
- `U ∈ ℝ^(batch × hidden_dim × rank)`
- `V ∈ ℝ^(input_dim × rank)`
- `x ∈ ℝ^(batch × input_dim)`
- `fast_effect ∈ ℝ^(batch × hidden_dim)`

**Обновление U (STDP):**
```
# Hebbian: outer(h, e) @ V
eV = e @ V  # (batch, rank)
hebbian = h ⊗ eV = h.unsqueeze(2) * eV.unsqueeze(1)  # (batch, hidden, rank)

# Plasticity modulation
plasticity = η · S  # (batch, hidden)

# Forgetting
forgetting = -λ · (U - U_target)

# Полный update
dU = forgetting + (plasticity ⊗ 1) · hebbian
U_new = U + dU · dt
```

**Гомеостаз (нормализация):**
```
||U|| = √(Σ U_ijk²)  # норма по (hidden, rank)
scale = target_norm / (||U|| + ε)
scale = clamp(scale, max=2.0)
U_new = U_new · scale
```

### 5.3. Код

```python
class FastWeights(nn.Module):
    def __init__(self, hidden_dim, input_dim, rank=16,
                 forgetting_rate=0.005, base_plasticity=0.5,
                 target_norm=2.0, time_step=0.1,
                 freeze_fast_weights=False):
        # Ортогонализация V
        V_init = torch.randn(input_dim, rank)
        Q, _ = torch.linalg.qr(V_init)
        self.register_buffer('V', Q)
        
        self.eta = nn.Parameter(torch.ones(hidden_dim) * base_plasticity)
        self.forgetting_rate = forgetting_rate
        self.time_step = time_step
        self.freeze_fast_weights = freeze_fast_weights
    
    def compute_fast_effect(self, U, V, x):
        return torch.einsum('bhr,ir,bi->bh', U, V, x)
    
    def update(self, h, error, surprise, U, U_target):
        if self.freeze_fast_weights:
            return U
        
        # Hebbian
        eV = error @ self.V
        hebbian = h.unsqueeze(2) * eV.unsqueeze(1)
        
        # Plasticity
        plasticity = self.eta.unsqueeze(0) * surprise.unsqueeze(1)
        plasticity = plasticity.unsqueeze(2)
        
        # Forgetting
        forgetting = -self.forgetting_rate * (U - U_target)
        
        # Update
        dU = forgetting + plasticity * hebbian
        U_new = U + dU * self.time_step
        
        # Homeostasis
        U_norm = U_new.norm(dim=(1,2), keepdim=True)
        scale = (self.target_norm / (U_norm + 1e-6)).clamp(max=2.0)
        return U_new * scale
```

---

## 6. Блок 4: Liquid Time-Constants

**Файл:** `dream/layers/ltc.py`

**Назначение:** Адаптивная скорость интеграции на основе surprise.

### 6.1. Параметры

| Параметр | Форма | По умолчанию | Описание |
|----------|-------|--------------|----------|
| `τ_sys` | скаляр | 5.0 | Базовая системная постоянная |
| `scale` | скаляр | 10.0 | Масштаб модуляции surprise |
| `dt` | скаляр | 0.1 | Шаг времени |

### 6.2. Формулы

**Динамическая постоянная времени:**
```
τ = τ_sys / (1 + S · scale)
τ = clamp(τ, min=0.01, max=50.0)
```

**Поведение:**
- `S ≈ 0` (низкий surprise) → `τ ≈ τ_sys` → медленная интеграция
- `S ≈ 1` (высокий surprise) → `τ ≈ τ_sys/(1+scale)` → быстрая интеграция

**LTC обновление (Эйлер):**
```
h_target = tanh(u_eff)
dt/τ = dt / (τ + dt)
dt/τ = clamp(dt/τ, min=0.01, max=0.5)

h_new = (1 - dt/τ) · h_prev + (dt/τ) · h_target
```

**Без LTC (классическое обновление):**
```
if not ltc_enabled:
    h_new = tanh(u_eff)
```

### 6.3. Код

```python
class LiquidTimeConstants(nn.Module):
    def __init__(self, ltc_tau_sys=5.0, ltc_surprise_scale=10.0,
                 time_step=0.1, ltc_enabled=True):
        self.tau_sys = nn.Parameter(torch.tensor(ltc_tau_sys))
        self.tau_surprise_scale = nn.Parameter(torch.tensor(ltc_surprise_scale))
        self.time_step = time_step
        self.ltc_enabled = ltc_enabled
    
    def forward(self, h_prev, u_eff, surprise):
        if not self.ltc_enabled or self.tau_sys.item() < 0.01:
            return torch.tanh(u_eff)
        
        # Динамическая τ
        tau = self.tau_sys / (1.0 + surprise * self.tau_surprise_scale)
        tau = torch.clamp(tau, 0.01, 50.0)
        
        # Цель
        h_target = torch.tanh(u_eff)
        
        # Эйлер
        dt_over_tau = self.time_step / (tau.unsqueeze(1) + self.time_step)
        dt_over_tau = torch.clamp(dt_over_tau, 0.01, 0.5)
        
        h_new = (1 - dt_over_tau) * h_prev + dt_over_tau * h_target
        return h_new
```

---

## 7. Блок 5: Sleep Consolidation

**Файл:** `dream/layers/sleep_consolidation.py`

**Назначение:** Консолидация быстрых весов в долговременную память.

### 7.1. Параметры

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `ζ` (sleep_rate) | float | 0.1 | Скорость консолидации |
| `S_min` | float | 0.5 | Порог surprise для сна |
| `T_sleep` | int | 100 | Мин. шагов между снами |
| `error_threshold` | float | 5.0 | Порог деградации |

### 7.2. Триггеры сна

**Триггер 1: Surprise**
```
if mean(avg_surprise) > S_min:
    trigger_sleep = True
```

**Триггер 2: Таймер**
```
steps_since_sleep += 1
if steps_since_sleep > T_sleep:
    trigger_sleep = True
    steps_since_sleep = 0
```

**Триггер 3: Деградация**
```
error_norm = mean(||error_mean||)
if error_norm > error_threshold:
    trigger_sleep = True
```

### 7.3. Консолидация

**Агрегация опыта:**
```
U_agg = (1 - ζ) · U_target + ζ · U
```

**Гомеостаз:**
```
||U_agg|| = норма по (hidden, rank)
scale = target_norm / (||U_agg|| + ε)
scale = clamp(scale, max=2.0)
U_target_new = U_agg · scale
```

### 7.4. Код

```python
class SleepConsolidation(nn.Module):
    def __init__(self, sleep_rate=0.1, min_surprise_for_sleep=0.5,
                 min_steps_for_sleep=100, error_threshold=5.0,
                 target_norm=2.0):
        self.sleep_rate = sleep_rate
        self.S_min = min_surprise_for_sleep
        self.T_sleep = min_steps_for_sleep
        self.error_threshold = error_threshold
        self.steps_since_sleep = 0
    
    def should_trigger_sleep(self, avg_surprise, error_mean):
        self.steps_since_sleep += 1
        
        # 1. Surprise
        if avg_surprise.mean().item() > self.S_min:
            return True
        
        # 2. Timer
        if self.steps_since_sleep > self.T_sleep:
            return True
        
        # 3. Degradation
        error_norm = error_mean.norm(dim=-1).mean().item()
        if error_norm > self.error_threshold:
            return True
        
        return False
    
    def forward(self, U, U_target, avg_surprise, error_mean, force=False):
        triggered = force or self.should_trigger_sleep(avg_surprise, error_mean)
        
        if triggered:
            # Агрегация
            U_agg = (1 - self.sleep_rate) * U_target + self.sleep_rate * U
            
            # Гомеостаз
            norm = U_agg.norm(dim=(1,2), keepdim=True)
            scale = (self.target_norm / (norm + 1e-6)).clamp(max=2.0)
            U_target_new = U_agg * scale
            
            self.steps_since_sleep = 0
            return U_target_new, True
        
        return U_target, False
```

---

## 8. DREAM Cell: Интеграция блоков

**Файл:** `dream/cell.py`

### 8.1. Forward pass (пошагово)

```
Вход: x ∈ ℝ^(batch × input_dim), state
Выход: h_new ∈ ℝ^(batch × hidden_dim), new_state
```

**Шаг 1: Predictive Coding**
```
x_pred = tanh(C^T @ h)
e = x - x_pred
```

**Шаг 2: Surprise Gate**
```
||e|| = e.norm(dim=-1)
μ_new, σ_new = update_statistics(||e||, μ, σ)  # с clip
z = (||e|| - μ_new) / (σ_new + ε)
S = sigmoid((z - τ_base) / γ)
gain = 1 + κ · S
```

**Шаг 3: Fast Weights Update**
```
if use_fast_weights:
    eV = e @ V
    hebbian = h.unsqueeze(2) * eV.unsqueeze(1)
    plasticity = η · S
    forgetting = -λ · (U - U_target)
    dU = forgetting + plasticity · hebbian
    U = U + dU · dt
    U = homeostasis(U)
```

**Шаг 4: Effective Input**
```
base_effect = x @ B_base
u_eff = gain · base_effect

if use_fast_weights:
    fast_effect = einsum('bhr,ir,bi->bh', U, V, x)
    u_eff = u_eff + 0.1 · fast_effect
```

**Шаг 5: LTC Update**
```
if use_ltc:
    τ = τ_sys / (1 + S · scale)
    h_target = tanh(u_eff)
    dt_over_tau = dt / (τ + dt)
    h_ltc = (1 - dt_over_tau) · h + dt_over_tau · h_target
else:
    h_ltc = tanh(u_eff)
```

**Шаг 6: Error Injection**
```
error_injection = e @ W
h_combined = h_ltc + error_injection
```

**Шаг 7: Stability**
```
h_new = 0.99 · h_combined + 0.01 · h  # leaky integration
```

**Шаг 8: Statistics Update**
```
error_mean = (1-β)·error_mean + β·e
error_var = (1-β)·error_var + β·(e - error_mean)²
avg_surprise = (1-β_s)·avg_surprise + β_s·S
```

**Шаг 9: Sleep**
```
if use_sleep:
    U_target, triggered = sleep(U, U_target, avg_surprise, error_mean)
```

### 8.2. Код

```python
class DREAMCell(nn.Module):
    def forward(self, x, state):
        # 1. Predictive Coding
        x_pred, error = self.predictive_coding(x, state.h)
        
        # 2. Surprise Gate
        surprise, _, gain, state.surprise_mu, state.surprise_sigma = \
            self.surprise_gate(error, state.error_var, state.error_mean,
                               state.surprise_mu, state.surprise_sigma)
        
        # 3. Fast Weights
        if self.use_fast_weights:
            state.U = self.fast_weights.update(
                state.h, error, surprise, state.U, state.U_target
            )
        
        # 4. Effective Input
        base_effect = self.predictive_coding.project_input(x)
        u_eff = gain * base_effect
        if self.use_fast_weights:
            fast_effect = self.fast_weights.compute_fast_effect(
                state.U, self.fast_weights.V, x
            )
            u_eff = u_eff + fast_effect * 0.1
        
        # 5. LTC
        h_ltc = self.ltc(state.h, u_eff, surprise)
        
        # 6. Error Injection
        error_injection = self.predictive_coding.inject_error(error)
        h_new = h_ltc + error_injection
        
        # 7. Stability
        h_new = h_new * 0.99 + state.h * 0.01
        
        # 8. Statistics
        state.error_mean = (1-self.beta)*state.error_mean + self.beta*error
        state.error_var = (1-self.beta)*state.error_var + self.beta*(error-state.error_mean)**2
        state.avg_surprise = (1-self.beta_s)*state.avg_surprise + self.beta_s*surprise
        
        # 9. Sleep
        if self.use_sleep:
            state.U_target, _ = self.sleep(
                state.U, state.U_target, state.avg_surprise, state.error_mean
            )
        
        return h_new, state
```

---

## 9. Координация между слоями

**Файл:** `dream/layers/coordination.py`

### 9.1. CoordinatedDREAMCell

**Дополнения к базовому cell:**

**Иерархический tau:**
```
τ_factor = 1.0 + 0.5 · layer_idx
# Layer 0: 1.0, Layer 1: 1.5, Layer 2: 2.0, ...

τ = (τ_sys · τ_factor) / (1 + S · scale)
```

**Нисходящая модуляция:**
```
if modulation_from_above is not None:
    modulation_strength = modulation.mean(dim=-1)
    surprise = surprise * (1.0 + 0.2 · modulation_strength)
```

**Prediction head:**
```
prediction = prediction_head(h_new)  # для нижнего слоя
```

**Modulation head:**
```
modulation = sigmoid(modulation_head(h_new))  # для нижнего слоя
```

### 9.2. CoordinatedDREAMStack

**Архитектура:**
```
Input → [Cell 0] → h₀ → [Cell 1] → h₁ → [Cell 2] → h₂
          ↑  ↓         ↑  ↓         ↑  ↓
       pred₀  mod₁  pred₁  mod₂  pred₂  mod₃
          │              │              │
          └──── loss ────┴──── loss ────┘
```

**Inter-layer loss:**
```
for i in 1..num_layers-1:
    pred_lower = prediction[i]  # предсказание для слоя i-1
    actual_lower = h[i-1]       # реальная активность
    loss_inter += MSE(pred_lower, actual_lower) / hidden_dim[i]
```

**Reconstruction loss:**
```
recon = output_projection(h_top)
loss_recon = MSE(recon, x)
```

---

## 10. Конфигурация

**Файл:** `dream/config.py`

### 10.1. DREAMConfig

```python
@dataclass
class DREAMConfig:
    # Размеры
    input_dim: int = 39
    hidden_dim: int = 256
    rank: int = 16
    
    # Управление блоками
    use_fast_weights: bool = True
    use_ltc: bool = True
    use_sleep: bool = True
    
    # Время
    time_step: float = 0.1
    
    # Пластичность
    forgetting_rate: float = 0.005      # λ
    base_plasticity: float = 0.5        # η
    
    # Surprise
    base_threshold: float = 2.0         # τ_base (z-score при S=0.5)
    entropy_influence: float = 0.1      # α
    surprise_temperature: float = 1.0   # γ
    kappa: float = 0.5
    
    # Сглаживание
    error_smoothing: float = 0.05       # β
    surprise_smoothing: float = 0.05    # β_s
    
    # Гомеостаз
    target_norm: float = 2.0
    
    # LTC
    ltc_tau_sys: float = 5.0
    ltc_surprise_scale: float = 10.0
    
    # Sleep
    sleep_rate: float = 0.1             # ζ
    min_surprise_for_sleep: float = 0.5 # S_min
    min_steps_for_sleep: int = 100      # T_sleep
    error_threshold_for_sleep: float = 5.0
```

---

## 11. Состояние (State)

**Файл:** `dream/state.py`

### 11.1. DREAMState

```python
@dataclass
class DREAMState:
    h: torch.Tensor              # (batch, hidden_dim)
    U: torch.Tensor              # (batch, hidden_dim, rank)
    U_target: torch.Tensor       # (batch, hidden_dim, rank)
    adaptive_tau: torch.Tensor   # (batch,) или скаляр
    error_mean: torch.Tensor     # (batch, input_dim)
    error_var: torch.Tensor      # (batch, input_dim)
    avg_surprise: torch.Tensor   # (batch,) или скаляр
    surprise_mu: torch.Tensor    # (batch,) или скаляр
    surprise_sigma: torch.Tensor # (batch,) или скаляр
```

### 11.2. Инициализация

```python
@classmethod
def init_from_config(cls, config, batch_size, device, dtype):
    h = torch.randn(batch_size, config.hidden_dim, device=device, dtype=dtype) * 0.01
    U = torch.zeros(batch_size, config.hidden_dim, config.rank, device=device, dtype=dtype)
    U_target = torch.zeros(batch_size, config.hidden_dim, config.rank, device=device, dtype=dtype)
    
    surprise_shape = (batch_size,) if batch_size > 1 else ()
    surprise_mu = torch.ones(surprise_shape, device=device, dtype=dtype)
    surprise_sigma = torch.full(surprise_shape, 0.1, device=device, dtype=dtype)
    
    return cls(
        h=h, U=U, U_target=U_target,
        adaptive_tau=torch.full(surprise_shape, config.base_threshold, device=device, dtype=dtype),
        error_mean=torch.zeros(batch_size, config.input_dim, device=device, dtype=dtype),
        error_var=torch.ones(batch_size, config.input_dim, device=device, dtype=dtype),
        avg_surprise=torch.zeros(surprise_shape, device=device, dtype=dtype),
        surprise_mu=surprise_mu,
        surprise_sigma=surprise_sigma,
    )
```

---

## 12. API и примеры

### 12.1. Базовое использование

```python
import torch
from dream import DREAM, DREAMConfig, DREAMCell

# Конфигурация
config = DREAMConfig(input_dim=64, hidden_dim=128, rank=8)

# Cell (низкоуровневый)
cell = DREAMCell(config)
state = cell.init_state(batch_size=32)
h, state = cell(x, state)

# Layer (высокоуровневый)
model = DREAM(input_dim=64, hidden_dim=128, rank=8)
x = torch.randn(32, 50, 64)  # (batch, time, features)
output, state = model(x)  # output: (32, 50, 128)

# Stack (многослойный)
stack = DREAMStack(
    input_dim=64,
    hidden_dims=[128, 128, 64],
    rank=8,
    dropout=0.1
)
output, states = stack(x)
```

### 12.2. Отключение блоков

```python
# Только Predictive Coding + Surprise
config = DREAMConfig(
    input_dim=64,
    hidden_dim=128,
    use_fast_weights=False,
    use_ltc=False,
    use_sleep=False
)
cell = DREAMCell(config)

# Без LTC
config = DREAMConfig(use_ltc=False)

# Без Sleep
config = DREAMConfig(use_sleep=False)
```

### 12.3. Двухэтапное обучение

```python
model = DREAM(input_dim=64, hidden_dim=128, rank=8)

# Этап 1: Обучение базы (fast weights заморожены)
model.train()  # freeze_fast_weights = True
for x, y in train_loader:
    output, _ = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

# Этап 2: Адаптация (fast weights активны)
model.eval()  # freeze_fast_weights = False
with torch.no_grad():
    output, _ = model(x_adapt)
```

### 12.4. Координация

```python
from dream import CoordinatedDREAMStack

model = CoordinatedDREAMStack(
    input_dim=80,
    hidden_dims=[128, 128, 128],
    rank=16,
    use_hierarchical_tau=True,
    use_inter_layer_prediction=True
)

x = torch.randn(32, 100, 80)
output, states, losses = model(x, return_losses=True)

# losses: {'reconstruction': ..., 'inter_layer': ...}
total_loss = losses['reconstruction'] + 0.01 * losses['inter_layer']
```

---

## Приложение A: Сводная таблица формул

| Блок | Формула | Описание |
|------|---------|----------|
| **Predictive Coding** | `x_pred = tanh(C^T @ h)` | Предсказание |
| | `e = x - x_pred` | Ошибка |
| **Surprise Gate** | `clipped = clamp(\|\|e\|\|, μ, μ+3σ)` | Clip выбросов |
| | `μ_new = (1-β)μ + β·clipped` | Обновление статистики |
| | `z = (\|\|e\|\| - μ) / (σ + ε)` | Z-оценка |
| | `S = sigmoid((z - τ_base) / γ)` | Surprise |
| **Fast Weights** | `hebbian = h ⊗ (e @ V)` | Hebbian обучение |
| | `dU = -λ(U-U_target) + (η·S)·hebbian` | STDP update |
| | `fast_effect = (U @ V^T) @ x` | Вклад в u_eff |
| **LTC** | `τ = τ_sys / (1 + S·scale)` | Динамическая τ |
| | `h_new = (1-dt/τ)h + (dt/τ)tanh(u_eff)` | Интеграция |
| **Sleep** | `trigger = (S > S_min) ∨ (steps > T_sleep) ∨ (error > threshold)` | Триггеры |
| | `U_agg = (1-ζ)U_target + ζ·U` | Консолидация |

---

**Конец спецификации**
