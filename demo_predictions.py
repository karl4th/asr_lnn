"""
Демонстрация предсказаний модели
Генерирует тестовые примеры и показывает как модель их распознаёт
"""

import torch
from model import create_model
from train import decode_ctc_output


def demo_predictions():
    """Демонстрация работы модели на синтетических данных"""
    
    print("=" * 70)
    print("ДЕМОНСТРАЦИЯ ПРЕДСКАЗАНИЙ LNN ASR МОДЕЛИ")
    print("=" * 70)
    
    # Загружаем модель
    model = create_model(num_classes=72, hidden_size=256, num_layers=2)
    model.eval()
    
    # Создаём фиктивные данные (разные длины последовательностей)
    test_cases = [
        {"name": "Короткая фраза", "time": 50},
        {"name": "Средняя фраза", "time": 100},
        {"name": "Длинная фраза", "time": 200},
    ]
    
    # Загружаем vocab для декодирования
    try:
        with open('vocab.txt', 'r') as f:
            tokens = [line.strip() for line in f.readlines()]
        idx_to_char = {idx: token for idx, token in enumerate(tokens)}
        print(f"\n✅ Vocab загружен: {len(tokens)} токенов")
    except FileNotFoundError:
        print("\n⚠️ Vocab.txt не найден, используем индексы")
        idx_to_char = {i: f"<P{i}>" for i in range(72)}
    
    print("\n" + "-" * 70)
    
    for test_case in test_cases:
        print(f"\n📝 Тест: {test_case['name']} (время={test_case['time']})")
        print("-" * 50)
        
        # Генерируем случайную мел-спектрограмму
        batch_size = 1
        mel_spec = torch.randn(batch_size, test_case['time'], 80)
        mel_lengths = torch.tensor([test_case['time']], dtype=torch.long)
        
        # Forward pass
        with torch.no_grad():
            logits, output_lengths = model(mel_spec, mel_lengths=mel_lengths)
            
            # Greedy decoding
            predictions = torch.argmax(logits, dim=-1)
            
            # Confidence
            probs = torch.softmax(logits, dim=-1)
            confidences = probs.max(dim=-1).values
            avg_conf = confidences.mean().item()
            
            # Декодируем
            decoded = decode_ctc_output(predictions, idx_to_char)
            
            # Показываем результат
            print(f"  Предсказание: {decoded[0]}")
            print(f"  Средняя уверенность: {avg_conf:.3f}")
            print(f"  Длина выхода: {len(predictions[0])} → {len(decoded[0].split())} фонем")
            
            # Показываем топ-3 фонемы для нескольких фреймов
            print(f"\n  Топ-3 фонемы для нескольких фреймов:")
            sample_frames = [0, len(predictions[0])//4, len(predictions[0])//2, 3*len(predictions[0])//4, -1]
            for frame_idx in sample_frames:
                if frame_idx >= len(predictions[0]):
                    continue
                frame_probs = probs[0, frame_idx]
                top3_values, top3_indices = torch.topk(frame_probs, 3)
                top3_phones = [idx_to_char.get(idx.item(), f'<{idx.item()}>') for idx in top3_indices]
                print(f"    Фрейм {frame_idx:3d}: {' | '.join([f'{p}({v:.2f})' for p, v in zip(top3_phones, top3_values)])}")
    
    print("\n" + "=" * 70)
    print("✅ Демонстрация завершена!")
    print("=" * 70)
    
    # Пример CTC decoding
    print("\n📖 Пример CTC декодирования:")
    print("-" * 70)
    
    # Симулируем выходы модели
    example_sequence = [
        1, 1, 1, 0, 0, 2, 2, 2, 0, 3, 3, 0, 0, 4, 4, 4, 4, 5
    ]
    
    print(f"  Сырые предсказания: {example_sequence}")
    
    # Удаляем повторения
    no_repeats = []
    prev = -1
    for token in example_sequence:
        if token != prev:
            no_repeats.append(token)
        prev = token
    
    print(f"  После удаления повторов: {no_repeats}")
    
    # Удаляем blank (индекс 0)
    final = [t for t in no_repeats if t != 0]
    print(f"  После удаления blank: {final}")
    
    # Декодируем в фонемы
    decoded_example = decode_ctc_output(torch.tensor([example_sequence]), idx_to_char)
    print(f"  Финальные фонемы: {decoded_example[0]}")
    
    print("\n" + "=" * 70)


def demo_confidence_distribution():
    """Демонстрация распределения уверенности модели"""
    
    print("\n📊 РАСПРЕДЕЛЕНИЕ УВЕРЕННОСТИ МОДЕЛИ")
    print("=" * 70)
    
    model = create_model(num_classes=72, hidden_size=256, num_layers=2)
    model.eval()
    
    # Генерируем данные
    mel_spec = torch.randn(1, 100, 80)
    mel_lengths = torch.tensor([100], dtype=torch.long)
    
    with torch.no_grad():
        logits, _ = model(mel_spec, mel_lengths=mel_lengths)
        probs = torch.softmax(logits, dim=-1)
        confidences = probs.max(dim=-1).values[0]  # [time]
        
        # Статистики
        print(f"\n  Статистики уверенности:")
        print(f"    Мин: {confidences.min().item():.3f}")
        print(f"    Макс: {confidences.max().item():.3f}")
        print(f"    Среднее: {confidences.mean().item():.3f}")
        print(f"    Медиана: {confidences.median().item():.3f}")
        print(f"    Std: {confidences.std().item():.3f}")
        
        # Гистограмма
        print(f"\n  Гистограмма распределения:")
        bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        for i in range(len(bins)-1):
            count = ((confidences >= bins[i]) & (confidences < bins[i+1])).sum().item()
            bar = '█' * (count // 2)
            print(f"    [{bins[i]:.1f}-{bins[i+1]:.1f}]: {bar} ({count})")
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    demo_predictions()
    demo_confidence_distribution()
