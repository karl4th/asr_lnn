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
├── dataset.py        # Dataset для загрузки аудио
├── vocab.txt         # Словарь фонем (72 класса)
└── README.md         # Этот файл
```

## 🚀 Быстрый старт

### 1. Установка зависимостей

```bash
pip install torch torchaudio pandas numpy
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
- **Train Loss**: CTC loss на тренировочных данных
- **Val Loss**: CTC loss на валидационных данных
- **CER/PER**: Character/Phoneme Error Rate (можно добавить)

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
