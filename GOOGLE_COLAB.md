# Google Colab: Обучение LNN ASR

## 🚀 Быстрый старт в Colab

### 1. Откройте новый ноутбук в Google Colab

Перейдите на https://colab.research.google.com и создайте новый ноутбук.

### 2. Настройте GPU (T4)

```
Runtime → Change runtime type → Hardware accelerator: GPU → GPU: T4
```

### 3. Склонируйте репозиторий

```python
!git clone <YOUR_REPO_URL>
%cd qwen_lnn
```

### 4. Установите зависимости

```python
!pip install torch torchaudio pandas numpy scikit-learn -q
```

### 5. Смонтируйте Google Drive (опционально, для данных)

```python
from google.colab import drive
drive.mount('/content/drive')

# Если данные на Drive
!ln -s /content/drive/MyDrive/your_data data
```

### 6. Загрузите данные

```python
# Проверка структуры
!ls -la data/
!head -5 data/metadata.csv
```

### 7. Запустите обучение

```python
!python train.py \
  --metadata data/metadata.csv \
  --vocab vocab.txt \
  --audio-dir data/wavs \
  --batch-size 16 \
  --epochs 50 \
  --hidden-size 256 \
  --num-layers 2
```

### 8. (Опционально) Используйте TensorBoard

```python
%load_ext tensorboard
%tensorboard --logdir runs/
```

### 9. Сохраните модель на Drive

```python
# После обучения
!cp checkpoints/best_model.pt /content/drive/MyDrive/lnn_asr/best_model.pt
```

## 📊 Мониторинг обучения

Во время обучения вы увидите таблицу с метриками:

```
Epoch | Train Loss | Train PER | Val Loss | Val PER |       LR | Time
---------------------------------------------------------------------------
    1 |     2.3456 |     85.2% |   2.1234 |    82.1% |   1.00e-03 |   45s
    2 |     1.8765 |     72.3% |   1.7654 |    68.5% |   1.00e-03 |   44s
    ...
```

## ⏱️ Время обучения на T4

Примерное время для разных конфигураций:

| Датасет | Hidden Size | Epochs | Время |
|---------|-------------|--------|-------|
| 10 часов | 256 | 50 | ~30 мин |
| 100 часов | 256 | 50 | ~5 часов |
| 100 часов | 512 | 100 | ~12 часов |

## 🔧 Оптимизации для Colab

### Используйте mixed precision (AMP)

Для ускорения обучения на T4:

```python
# Добавьте в train.py
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# В train_epoch:
with autocast():
    logits, output_lengths = self.model(mel_spec, mel_lengths=mel_lengths)
    loss = self.criterion(...)
```

### Увеличьте batch size

T4 имеет 16GB памяти:

```bash
!python train.py --batch-size 32  # или даже 64
```

### Используйте больше workers

```bash
!python train.py --workers 4
```

## 📦 Скачивание модели после обучения

```python
from google.colab import files
files.download('checkpoints/best_model.pt')
```

## 🐛 Частые проблемы

### "RuntimeError: CUDA out of memory"

```python
# Уменьшите batch size
!python train.py --batch-size 8
```

### "Worker died"

```python
# Уменьшите num_workers
!python train.py --workers 2
```

### Долгое обучение

- Убедитесь что используется GPU: `!nvidia-smi`
- Проверьте что данные не на Google Drive (медленный I/O)
- Используйте mixed precision
