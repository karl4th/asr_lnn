# DREAM: Technical Specification

**Dynamic Recall and Elastic Adaptive Memory**

Версия: 0.2.0  
Дата: Март 2026

---

## Содержание

1. [Обзор](#1-обзор)
2. [Архитектура](#2-архитектура)
3. [Конфигурация](#3-конфигурация)
4. [Блоки](#4-блоки)
   - [4.1 Predictive Coding](#41-predictive-coding)
   - [4.2 Surprise Gate](#42-surprise-gate)
   - [4.3 Fast Weights](#43-fast-weights)
   - [4.4 Liquid Time-Constants](#44-liquid-time-constants)
   - [4.5 Sleep Consolidation](#45-sleep-consolidation)
5. [Координация между слоями](#5-координация-между-слоями)
6. [API](#6-api)
7. [Примеры использования](#7-примеры-использования)

---

## 1. Обзор

**DREAM** (Dynamic Recall and Elastic Adaptive Memory) — это архитектура непрерывного рекуррентного нейрона с:

- **Предиктивным кодированием** — предсказание входа и вычисление ошибки
- **Surprise-драйвом** — адаптация на основе новизны
- **Быстрыми весами** — Hebбово обучение с низкоранговой декомпозицией
- **Жидкими постоянными времени (LTC)** — адаптивная скорость интеграции
- **Консолидацией памяти** — стабилизация во время "сна"

### Ключевые особенности

| Особенность | Описание |
|-------------|----------|
| Модульность | Каждый блок можно включить/выключить |
| Непрерывность | Непрерывное время через LTC |
| Адаптивность | Surprise модулирует пластичность |
| Иерархичность | Координация между слоями (опционально) |

---

## 2. Архитектура

```
┌─────────────────────────────────────────────────────────────┐
│                      DREAM Cell                             │
│                                                             │
│  Input (x) ──→ ┌─────────────────┐ ──→ Error (e)           │
│                │ 1. Predictive   │                          │
│  Hidden (h) ─→ │    Coding       │ ──→ x_pred               │
│                └─────────────────┘                          │
│                       │                                     │
│                       ↓                                     │
│                ┌─────────────────┐                          │
│                │ 2. Surprise     │ ──→ S (surprise)         │
│                │    Gate         │ ──→ gain                 │
│                └─────────────────┘                          │
│                       │                                     │
│                       ↓                                     │
│                ┌─────────────────┐                          │
│                │ 3. Fast Weights │ ──→ U (fast weights)     │
│                │    (Hebbian)    │                          │
│                └─────────────────┘                          │
│                       │                                     │
│                       ↓                                     │
│                ┌─────────────────┐                          │
│                │ 4. LTC          │ ──→ h_new (hidden)       │
│                │    (Integration)│                          │
│                └─────────────────┘                          │
│                       │                                     │
│                       ↓                                     │
│                ┌─────────────────┐                          │
│                │ 5. Sleep        │ ──→ U_target             │
│                │    Consolidation│                          │
│                └─────────────────┘                          │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Конфигурация

### DREAMConfig

```python
from dream import DREAMConfig

config = DREAMConfig(
    # Размеры
    input_dim=80,        # Размер входа
    hidden_dim=256,      # Размер скрытого состояния
    rank=16,             # Ранг быстрых весов
    
    # Управление блоками
    use_fast_weights=True,   # Быстрые веса
    use_ltc=True,            # LTC
    use_sleep=True,          # Консолидация
    
    # Время
    time_step=0.1,           # dt
    
    # Пластичность
    forgetting_rate=0.005,   # λ (забывание)
    base_plasticity=0.5,     # η (база)
    
    # Surprise
    base_threshold=0.3,      # τ₀
    entropy_influence=0.1,   # α
    surprise_temperature=0.05, # γ
    kappa=0.5,               # gain
    
    # Сглаживание
    error_smoothing=0.05,    # β
    surprise_smoothing=0.05, # β_s
    
    # Гомеостаз
    target_norm=2.0,         # норма весов
    
    # LTC
    ltc_tau_sys=5.0,         # базовая τ
    ltc_surprise_scale=10.0, # масштаб surprise
    
    # Сон
    sleep_rate=0.005,        # ζ
    min_surprise_for_sleep=0.2  # S_min
)
```

### Отключение блоков

```python
# Только предиктивное кодирование + surprise
config = DREAMConfig(
    use_fast_weights=False,
    use_ltc=False,
    use_sleep=False
)

# Без LTC (классическое обновление)
config = DREAMConfig(use_ltc=False)

# Без быстрых весов (статическая база)
config = DREAMConfig(use_fast_weights=False)
```

---

## 4. Блоки

### 4.1. Predictive Coding

**Назначение:** Предсказание входа и вычисление ошибки предсказания.

**Формулы:**
```
x_pred = tanh(C^T @ h)
e = x - x_pred
```

**Параметры:**
- `C` — декодирующая матрица (hidden_dim × input_dim)
- `W` — матрица инъекции ошибки (input_dim × hidden_dim)
- `B_base` — базовая проекция входа (input_dim × hidden_dim)

**Файл:** `dream/layers/predictive_coding.py`

---

### 4.2. Surprise Gate

**Назначение:** Вычисление surprise на основе ошибки предсказания.

**Формулы:**
```
||e|| = norm(e)                          # норма ошибки
H = 0.5 * log(2πe * var(e))             # энтропия
τ = 1 + α * H                            # адаптивный порог
relative_error = ||e|| / baseline        # относительная ошибка
S = sigmoid((relative_error - τ) / γ)   # surprise
gain = 1 + κ * S                         # модуляция
```

**Параметры:**
- `τ₀` — базовый порог
- `α` — влияние энтропии
- `γ` — температура surprise
- `κ` — коэффициент модуляции

**Файл:** `dream/layers/surprise_gate.py`

---

### 4.3. Fast Weights

**Назначение:** Быстрые веса с Hebбовым обучением и surprise-модуляцией.

**Формулы:**
```
V — фиксированный ортогональный фильтр (input_dim × rank)
U — быстрые веса (batch × hidden_dim × rank)

dU = -λ * (U - U_target) + (η * S) * (h ⊗ e) @ V
U_new = U + dU * dt
```

**Нормализация:**
```
scale = target_norm / ||U||
U = U * scale
```

**Параметры:**
- `λ` — скорость забывания
- `η` — пластичность
- `rank` — ранг декомпозиции
- `target_norm` — целевая норма

**Режимы:**
- `freeze_fast_weights=True` — веса заморожены (обучение базы)
- `freeze_fast_weights=False` — веса активны (адаптация)

**Файл:** `dream/layers/fast_weights.py`

---

### 4.4. Liquid Time-Constants (LTC)

**Назначение:** Адаптивная скорость интеграции на основе surprise.

**Формулы:**
```
τ = τ_sys / (1 + S * scale)              # динамическая τ
dh/dt = (-h + tanh(u_eff)) / τ           # LTC обновление
h_new = (1 - dt/τ) * h + (dt/τ) * tanh(u_eff)
```

**Поведение:**
- Высокий surprise → малая τ → быстрое обновление
- Низкий surprise → большая τ → медленная интеграция

**Параметры:**
- `τ_sys` — базовая системная постоянная
- `scale` — масштаб модуляции surprise

**Файл:** `dream/layers/ltc.py`

---

### 4.5. Sleep Consolidation

**Назначение:** Консолидация быстрых весов в долговременную память.

**Формулы:**
```
if avg_surprise > S_min:
    dU_target = ζ * avg_surprise * (U - U_target)
    U_target = U_target + dU_target
```

**Параметры:**
- `ζ` — скорость консолидации
- `S_min` — минимальный surprise для активации

**Файл:** `dream/layers/sleep_consolidation.py`

---

## 5. Координация между слоями

**Модуль:** `dream.layers.coordination`

### CoordinatedDREAMStack

Иерархическая архитектура с:

1. **Нисходящая модуляция** — верхние слои модулируют пластичность нижних
2. **Иерархический tau** — верхние слои медленнее (интегрируют дольше)
3. **Межслойное предсказание** — предсказания активности нижнего слоя + loss

```
Input → [Layer 0] → h₀ → [Layer 1] → h₁ → [Layer 2] → h₂
          ↑  ↓         ↑  ↓         ↑  ↓
       pred₀  mod₁  pred₁  mod₂  pred₂  mod₃
          │              │              │
          └──── error ───┴──── error ───┘
                    ↓
            inter_layer_loss
```

### Использование

```python
from dream import CoordinatedDREAMStack

model = CoordinatedDREAMStack(
    input_dim=80,
    hidden_dims=[128, 128, 128],  # 3 слоя
    rank=16,
    use_hierarchical_tau=True,        # верхние слои медленнее
    use_inter_layer_prediction=True,  # межслойные loss
    inter_layer_loss_weight=0.01
)

output, states, losses = model(x, return_losses=True)
```

### Иерархический Tau

```
Layer 0: τ_factor = 1.0
Layer 1: τ_factor = 1.5
Layer 2: τ_factor = 2.0
Layer 3: τ_factor = 2.5
```

### Нисходящая модуляция

```python
# Модуляция влияет на пластичность
plasticity_boost = 1.0 + 0.2 * (modulation - 0.5)

# Модуляция влияет на surprise
surprise = surprise * (1.0 + 0.2 * modulation_strength)
```

---

## 6. API

### Базовый API

```python
from dream import DREAM, DREAMConfig, DREAMCell

# Конфигурация
config = DREAMConfig(input_dim=64, hidden_dim=128, rank=8)

# Cell (низкоуровневый)
cell = DREAMCell(config)
state = cell.init_state(batch_size=32)
h, state = cell(x, state)

# Layer (высокоуровневый, nn.LSTM-like)
model = DREAM(input_dim=64, hidden_dim=128, rank=8)
output, state = model(x)  # x: (batch, time, input_dim)

# Stack (многослойный)
stack = DREAMStack(
    input_dim=64,
    hidden_dims=[128, 128, 64],
    rank=8,
    dropout=0.1
)
output, states = stack(x)
```

### Координация

```python
from dream import CoordinatedDREAMStack

model = CoordinatedDREAMStack(
    input_dim=80,
    hidden_dims=[128, 128, 128],
    rank=16
)

output, states, losses = model(x, return_losses=True)
```

### Управление быстрыми весами

```python
# Заморозить быстрые веса (обучение базы)
model.set_fast_weights_mode(freeze=True)
model.train()

# Разморозить (адаптация)
model.set_fast_weights_mode(freeze=False)
model.eval()

# Автоматически в train()/eval()
model.train()  # fast weights заморожены
model.eval()   # fast weights активны
```

---

## 7. Примеры использования

### 7.1. Базовое использование

```python
import torch
from dream import DREAM

model = DREAM(input_dim=64, hidden_dim=128, rank=8)
x = torch.randn(32, 50, 64)  # (batch, time, features)

output, state = model(x)
print(output.shape)  # (32, 50, 128)
```

### 7.2. Отключение блоков

```python
from dream import DREAMConfig, DREAMCell

# Только предиктивное кодирование + surprise
config = DREAMConfig(
    input_dim=64,
    hidden_dim=128,
    use_fast_weights=False,
    use_ltc=False,
    use_sleep=False
)

cell = DREAMCell(config)
```

### 7.3. Двухэтапное обучение

```python
from dream import DREAM

model = DREAM(input_dim=64, hidden_dim=128, rank=8)

# Этап 1: Обучение базы (быстрые веса заморожены)
model.train()  # freeze_fast_weights = True
for x, y in train_loader:
    output, _ = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

# Этап 2: Адаптация (быстрые веса активны)
model.eval()  # freeze_fast_weights = False
with torch.no_grad():
    output, _ = model(x_adapt)
```

### 7.4. Координированный стек

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

# Losses: reconstruction, inter_layer
total_loss = losses['reconstruction'] + 0.01 * losses['inter_layer']
```

### 7.5. Последовательная обработка

```python
from dream import DREAMCell

config = DREAMConfig(input_dim=64, hidden_dim=128)
cell = DREAMCell(config)
state = cell.init_state(batch_size=32)

# Пошаговая обработка
for t in range(time_steps):
    x_t = x[:, t, :]
    h, state = cell(x_t, state)

# Или вся последовательность
output, state = cell.forward_sequence(x_seq, state, return_all=True)
```

---

## Структура проекта

```
dream/
├── __init__.py              # Экспорт API
├── config.py                # DREAMConfig
├── state.py                 # DREAMState
├── cell.py                  # DREAMCell (модульный)
├── layer.py                 # DREAM (высокоуровневый)
├── stack.py                 # DREAMStack
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

## Приложения

### A. Математические обозначения

| Символ | Описание |
|--------|----------|
| `x` | Входной сигнал |
| `h` | Скрытое состояние |
| `e` | Ошибка предсказания |
| `S` | Surprise |
| `τ` | Постоянная времени |
| `η` | Пластичность |
| `λ` | Забывание |
| `U, V` | Быстрые веса |
| `γ` | Температура surprise |
| `κ` | Gain модуляция |

### B. Таблица параметров конфигурации

| Параметр | По умолчанию | Описание |
|----------|--------------|----------|
| `input_dim` | 39 | Размер входа |
| `hidden_dim` | 256 | Размер скрытого состояния |
| `rank` | 16 | Ранг быстрых весов |
| `use_fast_weights` | True | Включить быстрые веса |
| `use_ltc` | True | Включить LTC |
| `use_sleep` | True | Включить сон |
| `time_step` | 0.1 | Шаг времени (dt) |
| `forgetting_rate` | 0.005 | Скорость забывания (λ) |
| `base_plasticity` | 0.5 | Базовая пластичность (η) |
| `base_threshold` | 0.3 | Базовый порог (τ₀) |
| `entropy_influence` | 0.1 | Влияние энтропии (α) |
| `surprise_temperature` | 0.05 | Температура (γ) |
| `kappa` | 0.5 | Gain (κ) |
| `error_smoothing` | 0.05 | Сглаживание ошибки (β) |
| `surprise_smoothing` | 0.05 | Сглаживание surprise (β_s) |
| `target_norm` | 2.0 | Целевая норма весов |
| `ltc_tau_sys` | 5.0 | Базовая τ LTC |
| `ltc_surprise_scale` | 10.0 | Масштаб surprise |
| `sleep_rate` | 0.005 | Скорость сна (ζ) |
| `min_surprise_for_sleep` | 0.2 | Мин. surprise (S_min) |

---

**Конец спецификации**
