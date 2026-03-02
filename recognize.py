"""
Скрипт для распознавания фонем с помощью обученной LNN модели
"""

import torch
import torchaudio
import argparse
from pathlib import Path

from dataset import ASRDataset
from model import LNNASR


class PhonemeRecognizer:
    """
    Распознаватель фонем на основе LNN модели
    """
    
    def __init__(self, model_path, vocab_path, device=None):
        """
        Args:
            model_path: путь к чекпоинту модели
            vocab_path: путь к vocab.txt
            device: устройство для инференса
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Загружаем vocab
        self.vocab, self.char_to_idx = self._load_vocab(vocab_path)
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}
        
        # Загружаем модель
        self.model = self._load_model(model_path)
        
        # Параметры
        self.sample_rate = 16000
        self.n_mels = 80
        self.win_length = int(0.025 * self.sample_rate)
        self.hop_length = int(0.01 * self.sample_rate)
        self.n_fft = 512
        
        # Mel-спектрограмма трансформ
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            power=2.0
        )
        
        # Статистики для нормализации (если есть)
        self.means = None
        self.stds = None
        self._try_load_stats()
    
    def _load_vocab(self, vocab_path):
        """Загружает vocab.txt"""
        with open(vocab_path, 'r') as f:
            tokens = [line.strip() for line in f.readlines()]
        
        char_to_idx = {token: idx for idx, token in enumerate(tokens)}
        return tokens, char_to_idx
    
    def _load_model(self, model_path):
        """Загружает модель из чекпоинта"""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Создаём модель
        model = LNNASR(num_classes=72)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        print(f"✅ Модель загружена из {model_path}")
        print(f"   Параметры: {model.get_num_params():,}")
        
        return model
    
    def _try_load_stats(self):
        """Пытается загрузить статистики нормализации"""
        stats_path = Path('mel_stats.pth')
        if stats_path.exists():
            stats = torch.load(stats_path, map_location=self.device, weights_only=False)
            self.means = stats['means'].to(self.device)
            self.stds = stats['stds'].to(self.device)
            print(f"📦 Статистики нормализации загружены")
    
    def _compute_mel_spec(self, waveform):
        """Вычисляет лог-мел-спектрограмму"""
        mel_spec = self.mel_spectrogram(waveform)  # [1, 80, time]
        mel_spec = mel_spec.squeeze(0)  # [80, time]
        mel_spec = torch.log(mel_spec + 1e-8)  # логарифм
        return mel_spec
    
    def _normalize(self, mel_spec):
        """Нормализует мел-спектрограмму"""
        if self.means is not None and self.stds is not None:
            mel_spec = (mel_spec - self.means) / self.stds
        return mel_spec
    
    def _decode_sequence(self, indices):
        """Декодирует последовательность индексов в фонемы"""
        phones = []
        prev_token = -1
        
        for idx in indices:
            if idx != prev_token and idx != 0:  # пропускаем повторения и blank
                phone = self.idx_to_char.get(idx, '<unk>')
                phones.append(phone)
            prev_token = idx
        
        return phones
    
    def recognize(self, audio_path, return_detailed=False):
        """
        Распознаёт фонемы из аудио файла

        Args:
            audio_path: путь к аудио файлу (.wav)
            return_detailed: если True, возвращает детальную информацию

        Returns:
            phones: список распознанных фонем
            confidence: уверенность модели (средняя по всем фреймам)
            detailed: (опционально) детальная информация по фреймам
        """
        # Загружаем аудио
        waveform, sr = torchaudio.load(audio_path)

        # Ресемплируем если нужно
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # Вычисляем мел-спектрограмму
        mel_spec = self._compute_mel_spec(waveform)

        # Нормализуем
        mel_spec = self._normalize(mel_spec)

        # Добавляем batch dimension
        mel_spec = mel_spec.unsqueeze(0).to(self.device)  # [1, 80, time]

        # Инференс
        with torch.no_grad():
            logits, _ = self.model.forward(mel_spec)

            # Greedy decoding
            predictions = torch.argmax(logits, dim=-1)  # [1, time]

            # Вычисляем confidence (softmax max probability)
            probs = torch.softmax(logits, dim=-1)
            confidences = probs.max(dim=-1).values
            avg_confidence = confidences.mean().item()

            # Декодируем последовательность
            phones = self._decode_sequence(predictions[0])

        if return_detailed:
            # Декодируем с confidence per phone
            detailed = []
            prev_token = -1
            for i, (token, conf) in enumerate(zip(predictions[0], confidences)):
                if token != prev_token and token != 0:
                    phone = self.idx_to_char.get(token.item(), '<unk>')
                    detailed.append({
                        'phone': phone,
                        'confidence': conf.item(),
                        'frame': i
                    })
                prev_token = token
            
            return phones, avg_confidence, detailed
        
        return phones, avg_confidence

    def recognize_batch(self, audio_paths, show_examples=True):
        """
        Распознаёт фонемы из нескольких аудио файлов

        Args:
            audio_paths: список путей к аудио файлам
            show_examples: показывать детальные примеры

        Returns:
            results: список кортежей (phones, confidence)
        """
        results = []
        for idx, path in enumerate(audio_paths):
            try:
                if show_examples and idx < 3:
                    # Детальный вывод для первых 3 файлов
                    phones, conf, detailed = self.recognize(path, return_detailed=True)
                    results.append((path, phones, conf))
                    
                    print(f"\n{'='*70}")
                    print(f"📁 Файл: {Path(path).name}")
                    print(f"{'='*70}")
                    print(f"Распознанные фонемы: {' '.join(phones)}")
                    print(f"Средняя уверенность: {conf:.3f}")
                    print(f"\nДетально по фонемам:")
                    for i, item in enumerate(detailed):
                        conf_bar = '█' * int(item['confidence'] * 10)
                        print(f"  {i+1:2d}. {item['phone']:5s} | {conf_bar:10s} ({item['confidence']:.3f})")
                else:
                    phones, conf = self.recognize(path)
                    results.append((path, phones, conf))
                    print(f"✅ {Path(path).name}: {' '.join(phones)} (conf={conf:.3f})")
                    
            except Exception as e:
                print(f"❌ Ошибка {path}: {e}")
                results.append((path, None, None))

        return results


def main():
    parser = argparse.ArgumentParser(description='Распознавание фонем LNN')
    parser.add_argument('--model', type=str, required=True,
                        help='Путь к чекпоинту модели')
    parser.add_argument('--vocab', type=str, default='vocab.txt',
                        help='Путь к vocab.txt')
    parser.add_argument('--audio', type=str, nargs='+',
                        help='Пути к аудио файлам')
    parser.add_argument('--audio-dir', type=str,
                        help='Папка с аудио файлами для пакетного распознавания')
    
    args = parser.parse_args()
    
    # Создаём распознаватель
    recognizer = PhonemeRecognizer(
        model_path=args.model,
        vocab_path=args.vocab
    )
    
    # Распознавание
    if args.audio:
        # Одиночные файлы
        results = recognizer.recognize_batch(args.audio)
        
        print("\n" + "=" * 60)
        print("РЕЗУЛЬТАТЫ:")
        print("=" * 60)
        for path, phones, conf in results:
            if phones is not None:
                print(f"\n{Path(path).name}:")
                print(f"  Фонемы: {' '.join(phones)}")
                print(f"  Уверенность: {conf:.3f}")
    
    elif args.audio_dir:
        # Пакетное распознавание из папки
        audio_dir = Path(args.audio_dir)
        audio_files = list(audio_dir.glob('*.wav')) + list(audio_dir.glob('*.flac'))
        
        if not audio_files:
            print(f"❌ Аудио файлы не найдены в {audio_dir}")
            return
        
        print(f"📂 Найдено {len(audio_files)} аудио файлов")
        results = recognizer.recognize_batch([str(f) for f in audio_files])
    
    else:
        print("❌ Укажите --audio или --audio-dir")


if __name__ == '__main__':
    main()
