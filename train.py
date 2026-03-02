"""
Скрипт обучения LNN модели для распознавания фонем
Использует CTC loss для обучения без точных таймингов
"""

import os
import sys
import time
import math
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from dataset import ASRDataset, create_data_splits
from model import create_model, LNNASR


class CTCCollator:
    """
    Collate function для CTC обучения
    Обрабатывает последовательности разной длины
    """
    
    def __init__(self, batch_first=True):
        self.batch_first = batch_first
    
    def __call__(self, batch):
        """
        Args:
            batch: список словарей от dataset.__getitem__
        
        Returns:
            mel_specs: padded мел-спектрограммы
            mel_lengths: длины аудио последовательностей
            phone_seqs: padded последовательности фонем
            phone_lengths: длины последовательностей фонем
        """
        # Извлекаем данные
        mel_specs = [item['mel_spec'] for item in batch]
        phone_seqs = [item['phones'] for item in batch]
        filenames = [item['filename'] for item in batch]
        
        # Получаем длины
        mel_lengths = torch.tensor([m.shape[1] for m in mel_specs], dtype=torch.long)
        phone_lengths = torch.tensor([len(p) for p in phone_seqs], dtype=torch.long)
        
        # Padding мел-спектрограмм [batch, 80, max_time]
        # Для CTC важно: время должно быть вторым измерением для pack_padded_sequence
        max_mel_len = max(m.shape[1] for m in mel_specs)
        mel_specs_padded = []
        for m in mel_specs:
            padding = max_mel_len - m.shape[1]
            if padding > 0:
                m = torch.nn.functional.pad(m, (0, padding))
            mel_specs_padded.append(m)
        mel_specs = torch.stack(mel_specs_padded)  # [batch, 80, max_time]
        
        # Переворачиваем к [batch, max_time, 80] для модели
        mel_specs = mel_specs.transpose(1, 2)
        
        # Padding последовательностей фонем
        phone_seqs_padded = pad_sequence(phone_seqs, batch_first=True, padding_value=0)
        
        return {
            'mel_spec': mel_specs,
            'mel_lengths': mel_lengths,
            'phones': phone_seqs_padded,
            'phone_lengths': phone_lengths,
            'filenames': filenames
        }


def decode_ctc_output(predictions, idx_to_char, blank_idx=0):
    """
    Декодирует CTC предсказания в читаемые фонемы
    
    Args:
        predictions: предсказания модели [batch, time]
        idx_to_char: словарь индекс→фонема
        blank_idx: индекс blank токена
    
    Returns:
        decoded: список строк с фонемами
    """
    decoded = []
    for pred in predictions:
        phones = []
        prev_token = -1
        for token in pred:
            if token != prev_token and token != blank_idx:
                phone = idx_to_char.get(token.item(), '<unk>')
                phones.append(phone)
            prev_token = token
        decoded.append(' '.join(phones))
    return decoded


def compute_per(predictions, targets, blank_idx=0):
    """
    Вычисляет Phoneme Error Rate (PER)
    
    Args:
        predictions: предсказанные последовательности [batch, time]
        targets: целевые последовательности [batch, target_len]
        blank_idx: индекс blank токена
    
    Returns:
        per: Phoneme Error Rate (0.0 - 1.0)
        num_errors: количество ошибок
        num_total: общее количество фонем
    """
    total_errors = 0
    total_phones = 0
    
    for pred, target in zip(predictions, targets):
        # CTC decoding: удаляем повторения и blank
        decoded = []
        prev_token = -1
        for token in pred:
            if token != prev_token and token != blank_idx:
                decoded.append(token)
            prev_token = token
        
        # Простой edit distance (Levenshtein)
        pred_list = decoded
        target_list = target.tolist() if hasattr(target, 'tolist') else list(target)
        
        # Вычисляем edit distance
        edits = edit_distance(pred_list, target_list)
        total_errors += edits
        total_phones += len(target_list)
    
    per = total_errors / max(total_phones, 1)
    return per, total_errors, total_phones


def edit_distance(s1, s2):
    """Вычисляет расстояние Левенштейна между двумя последовательностями"""
    m, n = len(s1), len(s2)
    
    # DP таблица
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Базовые случаи
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    # Заполнение таблицы
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],    # deletion
                    dp[i][j-1],    # insertion
                    dp[i-1][j-1]   # substitution
                )
    
    return dp[m][n]


class Trainer:
    """
    Тренер для LNN ASR модели
    """
    
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, 
                 scheduler=None, device='cuda', grad_clip=5.0, idx_to_char=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.grad_clip = grad_clip
        self.idx_to_char = idx_to_char or {}
        
        # Статистика
        self.best_val_loss = float('inf')
        self.best_val_per = float('inf')
        self.epoch = 0
        self.step = 0
    
    def train_epoch(self):
        """Обучение на один epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        # Для PER
        all_predictions = []
        all_targets = []

        for batch_idx, batch in enumerate(self.train_loader):
            # Переносим на устройство
            mel_spec = batch['mel_spec'].to(self.device)
            mel_lengths = batch['mel_lengths'].to(self.device)
            phones = batch['phones'].to(self.device)
            phone_lengths = batch['phone_lengths'].to(self.device)

            # Forward pass
            logits, output_lengths = self.model(mel_spec, mel_lengths=mel_lengths)

            # CTC loss требует [time, batch, num_classes]
            log_probs = torch.log_softmax(logits, dim=-1)
            log_probs = log_probs.transpose(0, 1)  # [time, batch, num_classes]

            # Вычисляем loss
            loss = self.criterion(
                log_probs,
                phones,
                output_lengths,
                phone_lengths
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()

            # Статистика
            total_loss += loss.item()
            num_batches += 1
            self.step += 1
            
            # Собираем предсказания для PER
            predictions = torch.argmax(logits, dim=-1)  # [batch, time]
            for pred, target, length in zip(predictions, phones, phone_lengths):
                all_predictions.append(pred[:output_lengths[all_predictions.__len__() % len(predictions)]].cpu())
                all_targets.append(target[:length].cpu())

            # Лог каждые 10 батчей
            if (batch_idx + 1) % 10 == 0:
                avg_loss = total_loss / num_batches
                print(f"  Batch {batch_idx + 1}/{len(self.train_loader)}: loss={avg_loss:.4f}")

        # Вычисляем PER
        train_per, _, _ = compute_per(
            [p[:ol].cpu().numpy() for p, ol in zip(predictions, output_lengths)],
            [t[:pl].cpu().numpy() for t, pl in zip(phones, phone_lengths)]
        )

        return total_loss / num_batches, train_per
    
    @torch.no_grad()
    def validate(self, show_examples=True, num_examples=5):
        """Валидация"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        # Для PER и примеров
        all_predictions = []
        all_targets = []
        example_data = []  # для хранения примеров

        for batch_idx, batch in enumerate(self.val_loader):
            mel_spec = batch['mel_spec'].to(self.device)
            mel_lengths = batch['mel_lengths'].to(self.device)
            phones = batch['phones'].to(self.device)
            phone_lengths = batch['phone_lengths'].to(self.device)
            filenames = batch.get('filenames', [f'_{i}' for i in range(len(phones))])

            logits, output_lengths = self.model(mel_spec, mel_lengths=mel_lengths)

            log_probs = torch.log_softmax(logits, dim=-1)
            log_probs = log_probs.transpose(0, 1)

            loss = self.criterion(
                log_probs,
                phones,
                output_lengths,
                phone_lengths
            )

            total_loss += loss.item()
            num_batches += 1
            
            # Собираем для PER
            predictions = torch.argmax(logits, dim=-1)  # [batch, time]
            all_predictions.append(predictions)
            all_targets.append(phones)
            
            # Сохраняем примеры (первые num_examples батчей)
            if show_examples and len(example_data) < num_examples:
                decoded_preds = decode_ctc_output(predictions, self.idx_to_char)
                decoded_targets = decode_ctc_output(phones, self.idx_to_char)
                
                for i in range(min(len(predictions), num_examples - len(example_data))):
                    example_data.append({
                        'filename': filenames[i] if i < len(filenames) else f'_{batch_idx}_{i}',
                        'prediction': decoded_preds[i],
                        'target': decoded_targets[i],
                        'phone_length': phone_lengths[i].item()
                    })

        # Вычисляем PER на валидации
        all_preds_cat = torch.cat(all_predictions, dim=0)
        all_targets_cat = torch.cat(all_targets, dim=0)
        
        val_per, _, _ = compute_per(
            all_preds_cat.cpu().numpy(),
            all_targets_cat.cpu().numpy()
        )

        # Показываем примеры
        if show_examples and example_data:
            print("\n  📋 Примеры предсказаний:")
            print("  " + "-" * 70)
            for ex in example_data[:num_examples]:
                print(f"  Файл: {ex['filename']}")
                print(f"    🔴 Pred: {ex['prediction']}")
                print(f"    🟢 True: {ex['target']}")
                
                # Вычисляем accuracy для примера
                pred_phones = ex['prediction'].split()
                true_phones = ex['target'].split()
                if len(true_phones) > 0:
                    matches = sum(1 for p in pred_phones if p in true_phones)
                    acc = matches / len(true_phones) * 100
                    print(f"    📊 Match: {acc:.1f}%")
                print()

        return total_loss / num_batches, val_per
    
    def train(self, num_epochs, save_dir='checkpoints', log_interval=1):
        """
        Полный цикл обучения

        Args:
            num_epochs: количество эпох
            save_dir: директория для сохранения чекпоинтов
            log_interval: интервал логгирования в эпохах
        """
        os.makedirs(save_dir, exist_ok=True)

        print(f"\n🚀 Начало обучения на {num_epochs} эпох")
        print(f"   Устройство: {self.device}")
        print(f"   Параметры модели: {self.model.get_num_params():,}")
        print(f"   Train batches: {len(self.train_loader)}")
        print(f"   Val batches: {len(self.val_loader)}")
        print("-" * 60)
        
        # Заголовок таблицы метрик
        print(f"\n{'Epoch':>5} | {'Train Loss':>10} | {'Train PER':>9} | {'Val Loss':>10} | {'Val PER':>9} | {'LR':>8} | {'Time':>6}")
        print("-" * 75)

        for epoch in range(num_epochs):
            self.epoch = epoch + 1
            start_time = time.time()

            # Обучение
            train_loss, train_per = self.train_epoch()

            # Валидация (показываем примеры только на 1, 10, 20... эпохах)
            show_examples = (epoch == 1) or (epoch % 10 == 0)
            val_loss, val_per = self.validate(show_examples=show_examples)

            # Scheduler step
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Лог
            elapsed = time.time() - start_time
            lr = self.optimizer.param_groups[0]['lr']
            print(f"{self.epoch:>5} | {train_loss:>10.4f} | {train_per:>9.2%} | {val_loss:>10.4f} | {val_per:>9.2%} | {lr:>8.2e} | {elapsed:>5.0f}s")

            # Сохранение лучшего чекпоинта по PER
            if val_per < self.best_val_per:
                self.best_val_per = val_per
                checkpoint_path = os.path.join(save_dir, 'best_model.pt')
                self.save_checkpoint(checkpoint_path, metric='PER', metric_value=val_per)
                print(f"  ✅ Лучшая модель по PER сохранена (val_per={val_per:.2%})")

            # Сохранение последнего чекпоинта
            if (epoch + 1) % log_interval == 0:
                checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{self.epoch}.pt')
                self.save_checkpoint(checkpoint_path)

        print("\n" + "=" * 60)
        print(f"✅ Обучение завершено! Лучший val_per: {self.best_val_per:.2%}")
    
    def save_checkpoint(self, path, metric='loss', metric_value=None):
        """Сохраняет чекпоинт"""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': self.best_val_loss,
            'val_per': self.best_val_per,
            'best_metric': metric,
            'best_metric_value': metric_value,
        }
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path):
        """Загружает чекпоинт"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['val_loss']
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"📦 Загружен чекпоинт из {path} (epoch {self.epoch})")


def train(config=None):
    """
    Основная функция обучения
    
    Args:
        config: словарь с конфигурацией
    """
    # Конфигурация по умолчанию
    if config is None:
        config = {
            # Данные
            'metadata_path': 'data/metadata.csv',
            'vocab_path': 'vocab.txt',
            'audio_dir': 'data/wavs',
            
            # Модель
            'hidden_size': 256,
            'num_layers': 2,
            'num_classes': 72,
            
            # Обучение
            'batch_size': 16,
            'num_epochs': 50,
            'learning_rate': 1e-3,
            'weight_decay': 1e-5,
            'grad_clip': 5.0,
            
            # Прочее
            'num_workers': 4,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'save_dir': 'checkpoints',
        }
    
    print("📋 Конфигурация:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    print()
    
    # Устройство
    device = config['device']
    print(f"🔧 Используемое устройство: {device}")

    # Разделение данных 80/10/10
    print("\n📊 Разделение данных (80/10/10)...")
    train_df, val_df, test_df = create_data_splits(
        config['metadata_path'],
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=42
    )

    # Датасеты
    print("\n📊 Создание датасетов...")
    train_dataset = ASRDataset(
        metadata_path=config['metadata_path'],
        vocab_path=config['vocab_path'],
        audio_dir=config['audio_dir'],
        training=True,
        metadata_df=train_df
    )

    val_dataset = ASRDataset(
        metadata_path=config['metadata_path'],
        vocab_path=config['vocab_path'],
        audio_dir=config['audio_dir'],
        training=False,
        metadata_df=val_df
    )

    test_dataset = ASRDataset(
        metadata_path=config['metadata_path'],
        vocab_path=config['vocab_path'],
        audio_dir=config['audio_dir'],
        training=False,
        metadata_df=test_df
    )

    # Вычисляем статистики нормализации на train
    print("\n📈 Вычисление статистик нормализации на train данных...")
    train_dataset.compute_global_stats()
    val_dataset.means = train_dataset.means
    val_dataset.stds = train_dataset.stds
    test_dataset.means = train_dataset.means
    test_dataset.stds = train_dataset.stds

    # DataLoaders
    collator = CTCCollator(batch_first=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collator,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collator,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    print(f"✅ Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Модель
    print("\n🏗️ Создание модели...")
    model = create_model(
        num_classes=config['num_classes'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers']
    )
    
    # Loss и оптимизатор
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
    )
    
    # Тренер
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=ctc_loss,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        grad_clip=config['grad_clip'],
        idx_to_char=train_dataset.idx_to_char
    )
    
    # Обучение
    trainer.train(
        num_epochs=config['num_epochs'],
        save_dir=config['save_dir']
    )
    
    return trainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Обучение LNN ASR модели')
    
    # Данные
    parser.add_argument('--metadata', type=str, default='data/metadata.csv',
                        help='Путь к metadata.csv')
    parser.add_argument('--vocab', type=str, default='vocab.txt',
                        help='Путь к vocab.txt')
    parser.add_argument('--audio-dir', type=str, default='data/wavs',
                        help='Папка с аудио файлами')
    
    # Модель
    parser.add_argument('--hidden-size', type=int, default=256,
                        help='Размер скрытого состояния')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='Количество LTC слоёв')
    
    # Обучение
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Размер батча')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Количество эпох')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--grad-clip', type=float, default=5.0,
                        help='Gradient clipping')
    
    # Прочее
    parser.add_argument('--workers', type=int, default=4,
                        help='Количество workers для DataLoader')
    parser.add_argument('--device', type=str, default=None,
                        help='Устройство (cuda/cpu)')
    parser.add_argument('--save-dir', type=str, default='checkpoints',
                        help='Папка для чекпоинтов')
    
    args = parser.parse_args()
    
    # Собираем конфиг
    config = {
        'metadata_path': args.metadata,
        'vocab_path': args.vocab,
        'audio_dir': args.audio_dir,
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'num_classes': 72,
        'batch_size': args.batch_size,
        'num_epochs': args.epochs,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'grad_clip': args.grad_clip,
        'num_workers': args.workers,
        'device': args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'),
        'save_dir': args.save_dir,
    }
    
    train(config)
