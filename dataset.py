import os
import torch
import torchaudio
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torchaudio.transforms as T
from torch.utils.data import Dataset
import re

class ASRDataset(Dataset):
    def __init__(self, metadata_path, vocab_path, audio_dir, target_length=None, training=True, use_time_stretch=False):
        """
        Args:
            metadata_path: путь к metadata.csv
            vocab_path: путь к vocab.txt
            audio_dir: путь к папке с аудио
            target_length: если указан, все мел-спектрограммы будут дополнены/обрезаны до этой длины.
                           Для CTC обучения рекомендуется target_length=None.
            training: флаг обучения для включения аугментаций
            use_time_stretch: включить ли аугментацию изменения скорости (опасно для CTC)
        """
        self.metadata = pd.read_csv(metadata_path, sep='|')
        self.audio_dir = audio_dir
        self.target_length = target_length
        self.training = training
        self.use_time_stretch = use_time_stretch

        # Загружаем vocab
        self.vocab, self.char_to_idx = self._load_vocab(vocab_path)
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}

        # Валидация vocab (для 72 классов в Sanday)
        assert len(self.char_to_idx) == 72, f"Vocab size {len(self.char_to_idx)} != 72"
        assert self.char_to_idx.get('<blank>', -1) == 0, "Blank токен должен быть индексом 0"
        assert '<unk>' in self.char_to_idx, "UNK токен обязателен"
        print(f"✅ Vocab валидирован: {len(self.char_to_idx)} классов, blank=0")

        # Параметры мел-спектрограммы
        self.sample_rate = 16000
        self.n_mels = 80
        self.win_length = int(0.025 * self.sample_rate)  # 25 мс
        self.hop_length = int(0.01 * self.sample_rate)   # 10 мс
        self.n_fft = 512  # стандартное значение

        # Создаем трансформацию
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            power=2.0  # спектрограмма мощности
        )

        # Аугментации
        # Примечание: TimeStretch лучше работает на комплексных данных,
        # но для Mel-шкалы здесь используем Frequency/Time Masking как основные.
        self.freq_mask = T.FrequencyMasking(freq_mask_param=15)
        self.time_mask = T.TimeMasking(time_mask_param=35)

        # Для нормализации будем собирать статистики
        self.means = None
        self.stds = None
        self.stats_path = "mel_stats.pth"

    def _load_vocab(self, vocab_path):
        """Загружает vocab.txt и создает отображения символ->индекс"""
        with open(vocab_path, 'r') as f:
            tokens = [line.strip() for line in f.readlines()]

        # Создаем mapping
        char_to_idx = {token: idx for idx, token in enumerate(tokens)}

        print(f"📚 Загружен vocab: {len(tokens)} токенов")
        print(f"   Специальные токены: {[t for t in tokens if t.startswith('<')]}")

        return tokens, char_to_idx

    def _text_to_sequence(self, text):
        """Преобразует строку фонем в последовательность индексов"""
        # Разбиваем по пробелам
        phones = text.strip().split()

        # Конвертируем в индексы
        sequence = []
        for phone in phones:
            if phone in self.char_to_idx:
                sequence.append(self.char_to_idx[phone])
            else:
                # Если фонема не найдена в словаре, используем <unk>
                sequence.append(self.char_to_idx.get('<unk>', 0))

        return torch.tensor(sequence, dtype=torch.long)

    def _load_audio(self, audio_path):
        """Загружает аудио и ресемплирует до 16kHz"""
        waveform, sr = torchaudio.load(audio_path)

        # Если нужно, ресемплируем
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        return waveform

    def _compute_mel_spec(self, waveform, is_training=False):
        """Вычисляет лог-мел-спектрограмму с опциональными аугментациями"""
        # waveform shape: [1, T]

        # Вычисляем мел-спектрограмму
        mel_spec = self.mel_spectrogram(waveform)  # [1, n_mels, time]

        # Аугментации во время обучения (SpecAugment)
        if is_training:
            # Frequency masking
            if np.random.random() < 0.5:
                mel_spec = self.freq_mask(mel_spec)
            # Time masking
            if np.random.random() < 0.5:
                mel_spec = self.time_mask(mel_spec)

            # Time stretching (через интерполяцию для Mel-шкалы)
            # Внимание: для CTC это может создать рассогласование с метками!
            if self.use_time_stretch and np.random.random() < 0.3:
                rate = np.random.uniform(0.9, 1.1)
                mel_spec = torch.nn.functional.interpolate(
                    mel_spec, size=int(mel_spec.shape[-1] / rate), mode='linear', align_corners=False
                )

        # Убираем канальное измерение
        mel_spec = mel_spec.squeeze(0)  # [n_mels, time]

        # Применяем логарифм
        mel_spec = torch.log(mel_spec + 1e-8)

        return mel_spec

    def _normalize(self, mel_spec):
        """Нормализует мел-спектрограмму глобальными статистиками"""
        if self.means is not None and self.stds is not None:
            # Используем глобальные статистики
            mel_spec = (mel_spec - self.means) / self.stds
        return mel_spec

    def _pad_or_truncate(self, mel_spec):
        """Дополняет или обрезает до target_length"""
        if self.target_length is None:
            return mel_spec

        current_length = mel_spec.shape[1]

        if current_length < self.target_length:
            # Дополняем нулями
            padding = self.target_length - current_length
            mel_spec = torch.nn.functional.pad(mel_spec, (0, padding))
        elif current_length > self.target_length:
            # Обрезаем
            mel_spec = mel_spec[:, :self.target_length]

        return mel_spec

    def compute_global_stats(self, force_recompute=False):
        """Вычисляет или загружает глобальные среднее и стандартное отклонение"""
        if os.path.exists(self.stats_path) and not force_recompute:
            print(f"📦 Загружаем статистики из {self.stats_path}")
            stats = torch.load(self.stats_path)
            self.means = stats['means']
            self.stds = stats['stds']
            return self.means, self.stds

        print("📊 Вычисляем глобальные статистики (это может занять время)...")
        all_mels = []

        for idx in range(len(self)):
            try:
                # Берем данные без нормализации и аугментаций (force_eval=True)
                item = self.__getitem__(idx, normalize=False, force_eval=True)
                all_mels.append(item['mel_spec'].numpy())
            except Exception as e:
                print(f"   Пропускаем {idx}: {e}")
                continue

        all_mels = np.concatenate(all_mels, axis=1)  # [80, total_time]

        self.means = torch.tensor(np.mean(all_mels, axis=1, keepdims=True), dtype=torch.float32)
        self.stds = torch.tensor(np.std(all_mels, axis=1, keepdims=True) + 1e-8, dtype=torch.float32)

        # Сохраняем кэш
        torch.save({'means': self.means, 'stds': self.stds}, self.stats_path)

        print(f"✅ Статистики вычислены и сохранены в {self.stats_path}:")
        print(f"   Среднее: {self.means.squeeze().numpy()[:5]}... (shape: {self.means.shape})")
        print(f"   Std: {self.stds.squeeze().numpy()[:5]}...")

        return self.means, self.stds

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx, normalize=True, force_eval=False):
        """
        Возвращает:
            'mel_spec': [80, T] тензор с мел-спектрограммой
            'phones': [L] тензор с индексами фонем
            'text': строка с фонемами (для информации)
            'filename': имя файла
        """
        row = self.metadata.iloc[idx]
        is_training = self.training and not force_eval

        # Загружаем аудио
        audio_path = os.path.join(self.audio_dir, f"{row['filename']}.wav")
        waveform = self._load_audio(audio_path)

        # Вычисляем мел-спектрограмму
        mel_spec = self._compute_mel_spec(waveform, is_training=is_training)

        # Нормализуем (если нужно)
        if normalize and self.means is not None:
            mel_spec = self._normalize(mel_spec)

        # Дополняем/обрезаем до target_length
        mel_spec = self._pad_or_truncate(mel_spec)

        # Конвертируем текст фонем в последовательность индексов
        phones_seq = self._text_to_sequence(row['phonemes'])

        return {
            'mel_spec': mel_spec,  # [80, T]
            'phones': phones_seq,   # [L]
            'text': row['phonemes'], # для информации
            'filename': row['filename']
        }

    def get_item_with_timings(self, idx):
        """
        Возвращает элемент с таймингами из TextGrid (если нужно)
        """
        row = self.metadata.iloc[idx]

        # Здесь можно загрузить TextGrid и извлечь тайминги
        # Но для ASR обычно не нужно

        return self.__getitem__(idx)


