# LNN Phoneme Recognition (Liquid Neural Networks ASR)

Распознавание фонем на основе **Liquid Neural Networks** с использованием **CTC loss**.

## 📋 Описание

Этот проект реализует архитектуру **Liquid Time-Constant (LTC) networks** для задачи распознавания фонем. Модель принимает Mel-спектрограмму и выдаёт последовательность фонем без необходимости точных таймингов.

### Архитектура

```
Mel-спектрограмма [80, T]
        ↓
Input Projection (Linear + LayerNorm + GELU)
        ↓
LTC Encoder (2-3 слоя)
        ↓
CTC Head (Linear + GELU + Dropout → 72 класса)
        ↓
CTC Loss / Greedy Decoding
```

## 📁 Структура проекта

```
qwen_lnn/
├── ltc_cell.py       # LTC ячейка и многослойная LTC RNN
├── model.py          # LNN ASR модель (encoder + CTC head)
├── train.py          # Скрипт обучения
├── recognize.py      # Скрипт распознавания
├── dataset.py        # Dataset для загрузки аудио + split 80/10/10
├── vocab.txt         # Словарь фонем (72 класса)
└── README.md         # Этот файл
```

## 🚀 Быстрый старт

### 1. Установка зависимостей

```bash
pip install torch torchaudio pandas numpy scikit-learn
```

### 2. Подготовка данных

Структура папок:
```
data/
├── metadata.csv      # Таблица с колонками: filename|phonemes
└── wavs/             # WAV файлы (16kHz)
```

Формат `metadata.csv`:
```csv
filename|phonemes
file001|HH AH L OW W ER L D
file002|AY IH M S OW R IY
```

### 3. Обучение

```bash
# Базовое обучение
python train.py \
  --metadata data/metadata.csv \
  --vocab vocab.txt \
  --audio-dir data/wavs \
  --batch-size 16 \
  --epochs 50 \
  --hidden-size 256 \
  --num-layers 2

# С кастомными параметрами
python train.py \
  --metadata data/metadata.csv \
  --audio-dir data/wavs \
  --batch-size 32 \
  --epochs 100 \
  --hidden-size 512 \
  --num-layers 3 \
  --lr 5e-4 \
  --save-dir checkpoints/exp1
```

### 4. Распознавание

```bash
# Распознавание одного файла
python recognize.py \
  --model checkpoints/best_model.pt \
  --vocab vocab.txt \
  --audio test_audio.wav

# Пакетное распознавание
python recognize.py \
  --model checkpoints/best_model.pt \
  --vocab vocab.txt \
  --audio-dir test_wavs/
```

## 📊 Параметры модели

| Параметр | Значение по умолчанию | Описание |
|----------|----------------------|----------|
| `hidden_size` | 256 | Размер скрытого состояния LTC |
| `num_layers` | 2 | Количество LTC слоёв |
| `num_classes` | 72 | Количество фонем (из vocab.txt) |
| `batch_size` | 16 | Размер батча для обучения |
| `learning_rate` | 1e-3 | Learning rate оптимизатора |
| `grad_clip` | 5.0 | Gradient clipping |

## 🔬 Детали архитектуры

### LTC Cell

Liquid Time-Constant ячейка реализует дифференциальное уравнение:

```
dx/dt = -[1/τ(x,h)] ⊙ h + f(x,h)
```

где:
- `τ` — адаптивная временная константа (зависит от входа и состояния)
- `f(x,h)` — переходная функция (нейросеть)

### Преимущества LTC

1. **Адаптивность**: временные константы подстраиваются под входные данные
2. **Устойчивость**: лучше обрабатывает длинные последовательности чем LSTM
3. **Интерпретируемость**: можно анализировать динамику временных констант

## 📈 CTC Loss

**Connectionist Temporal Classification** позволяет обучаться без точных таймингов:

- **Blank токен** (`<blank>`, индекс 0): заполнитель между повторениями
- **Выравнивание**: модель сама учится выравнивать предсказания по времени
- **Decoding**: greedy или beam search для получения финальной последовательности

### Пример CTC decoding

```
Предсказания модели: [H, H, <b>, H, E, E, <b>, L, L, O]
После удаления повторов: [H, <b>, E, <b>, L, O]
После удаления blank: [H, E, L, L, O]
```

## 🎯 Метрики качества

Во время обучения отслеживаются:

| Метрика | Описание |
|---------|----------|
| **Train Loss** | CTC loss на тренировочных данных |
| **Train PER** | Phoneme Error Rate на train (доля ошибок) |
| **Val Loss** | CTC loss на валидационных данных |
| **Val PER** | Phoneme Error Rate на val (доля ошибок) |

### Формат вывода во время обучения

```
Epoch | Train Loss | Train PER | Val Loss | Val PER |       LR | Time
---------------------------------------------------------------------------
    1 |     2.3456 |     85.2% |   2.1234 |    82.1% |   1.00e-03 |   45s
    2 |     1.8765 |     72.3% |   1.7654 |    68.5% |   1.00e-03 |   44s
    3 |     1.4321 |     58.1% |   1.3456 |    55.2% |   1.00e-03 |   45s
  ...
```

**PER (Phoneme Error Rate)**: доля неправильно распознанных фонем (рассчитывается через расстояние Левенштейна).

### Разделение данных

Данные автоматически разделяются на **80/10/10** (train/val/test):
- **Train (80%)**: обучение модели
- **Val (10%)**: валидация и подбор гиперпараметров
- **Test (10%)**: финальная оценка качества

## 📊 Примеры предсказаний

### Во время обучения

На 1-й и каждой 10-й эпохе показываются примеры предсказаний:

```
  📋 Примеры предсказаний:
  ----------------------------------------------------------------------
  Файл: file001
    🔴 Pred: HH AH L OW W ER L D
    🟢 True: HH AH L OW W ER L D
    📊 Match: 100.0%

  Файл: file002
    🔴 Pred: AY IH M S OW R IY
    🟢 True: AY IH M S OW R IY
    📊 Match: 100.0%
```

### Детальный режим распознавания

```bash
python recognize.py --model checkpoints/best_model.pt --audio test.wav
```

Вывод:
```
======================================================================
📁 Файл: test.wav
======================================================================
Распознанные фонемы: HH AH L OW W ER L D
Средняя уверенность: 0.856

Детально по фонемам:
   1. HH    | ████████   (0.823)
   2. AH    | █████████  (0.912)
   3. L     | ████████   (0.845)
   4. OW    | █████████  (0.901)
   5. W     | ███████    (0.756)
   6. ER    | █████████  (0.889)
   7. L     | ████████   (0.834)
   8. D     | █████████  (0.923)
```

### Демонстрация на синтетических данных

```bash
python demo_predictions.py
```

## 🔧 Модификация архитектуры

### Изменение количества слоёв

```python
from model import create_model

model = create_model(
    num_classes=72,
    hidden_size=512,  # больше нейронов
    num_layers=3      # больше слоёв
)
```

### Добавление beam search

В `recognize.py` можно заменить greedy decoding на beam search для лучшей точности.

## 📝 Советы по обучению

1. **Нормализация**: убедитесь, что статистики (`mel_stats.pth`) вычислены корректно
2. **Batch size**: уменьшите если не хватает памяти GPU
3. **Learning rate**: используйте scheduler (ReduceLROnPlateau по умолчанию)
4. **Аугментации**: включите SpecAugment в `dataset.py` для лучшей генерализации
5. **Early stopping**: следите за val loss и останавливайте при переобучении

## 🐛 Troubleshooting

### CUDA Out of Memory
```bash
# Уменьшите batch size
python train.py --batch-size 8
```

### Долгое обучение
```bash
# Увеличьте num_workers для DataLoader
python train.py --workers 8
```

### Модель не сходится
- Проверьте правильность vocab (blank=0, есть `<unk>`)
- Уменьшите learning rate
- Увеличьте grad_clip

## 📚 Ссылки

- [Liquid Neural Networks (Hasani et al.)](https://arxiv.org/abs/2011.05412)
- [CTC Loss (Graves et al.)](https://arxiv.org/abs/1211.3711)
- [torchaudio documentation](https://pytorch.org/audio/stable/index.html)

## 📄 Лицензия

MIT License
