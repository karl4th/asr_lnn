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

from dataset import ASRDataset
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


class Trainer:
    """
    Тренер для LNN ASR модели
    """
    
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, 
                 scheduler=None, device='cuda', grad_clip=5.0):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.grad_clip = grad_clip
        
        # Статистика
        self.best_val_loss = float('inf')
        self.epoch = 0
        self.step = 0
    
    def train_epoch(self):
        """Обучение на один epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
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
            
            # Лог каждые 10 батчей
            if (batch_idx + 1) % 10 == 0:
                avg_loss = total_loss / num_batches
                print(f"  Batch {batch_idx + 1}/{len(self.train_loader)}: loss={avg_loss:.4f}")
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def validate(self):
        """Валидация"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        for batch in self.val_loader:
            mel_spec = batch['mel_spec'].to(self.device)
            mel_lengths = batch['mel_lengths'].to(self.device)
            phones = batch['phones'].to(self.device)
            phone_lengths = batch['phone_lengths'].to(self.device)
            
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
        
        return total_loss / num_batches
    
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
        
        for epoch in range(num_epochs):
            self.epoch = epoch + 1
            start_time = time.time()
            
            print(f"\nEpoch {self.epoch}/{num_epochs}")
            
            # Обучение
            train_loss = self.train_epoch()
            
            # Валидация
            val_loss = self.validate()
            
            # Scheduler step
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Лог
            elapsed = time.time() - start_time
            lr = self.optimizer.param_groups[0]['lr']
            print(f"  Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | LR: {lr:.2e} | Time: {elapsed:.1f}s")
            
            # Сохранение лучшего чекпоинта
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                checkpoint_path = os.path.join(save_dir, 'best_model.pt')
                self.save_checkpoint(checkpoint_path)
                print(f"  ✅ Лучшая модель сохранена (val_loss={val_loss:.4f})")
            
            # Сохранение последнего чекпоинта
            if (epoch + 1) % log_interval == 0:
                checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{self.epoch}.pt')
                self.save_checkpoint(checkpoint_path)
        
        print("\n" + "=" * 60)
        print(f"✅ Обучение завершено! Лучший val_loss: {self.best_val_loss:.4f}")
    
    def save_checkpoint(self, path):
        """Сохраняет чекпоинт"""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': self.best_val_loss,
        }
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path):
        """Загружает чекпоинт"""
        checkpoint = torch.load(path, map_location=self.device)
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
    
    # Датасеты
    print("\n📊 Загрузка датасетов...")
    train_dataset = ASRDataset(
        metadata_path=config['metadata_path'],
        vocab_path=config['vocab_path'],
        audio_dir=config['audio_dir'],
        training=True
    )
    
    val_dataset = ASRDataset(
        metadata_path=config['metadata_path'],
        vocab_path=config['vocab_path'],
        audio_dir=config['audio_dir'],
        training=False
    )
    
    # Вычисляем статистики нормализации
    print("\n📈 Вычисление статистик нормализации...")
    train_dataset.compute_global_stats()
    val_dataset.means = train_dataset.means
    val_dataset.stds = train_dataset.stds
    
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
    
    print(f"✅ Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
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
        verbose=True
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
        grad_clip=config['grad_clip']
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
