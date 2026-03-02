"""
LNN модель для распознавания фонем (ASR)
Архитектура: Mel-спектрограмма → LTC Encoder → CTC Head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ltc_cell import LTCRNN


class LNNEncoder(nn.Module):
    """
    Encoder на основе Liquid Neural Networks
    
    Архитектура:
    1. Linear projection: 80 mel bins → hidden_size
    2. LTC слои (2-3 слоя) для временного контекста
    3. LayerNorm для стабильности
    """
    
    def __init__(self, input_size=80, hidden_size=256, num_layers=2, dropout=0.1):
        super(LNNEncoder, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Проекция входа к hidden_size
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # LTC слои
        self.ltc_rnn = LTCRNN(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        # Выходная нормализация
        self.output_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, x, hidden=None):
        """
        Args:
            x: Mel-спектрограмма [batch, time, 80]
            hidden: начальное скрытое состояние для LTC
        
        Returns:
            encoded: закодированная последовательность [batch, time, hidden_size]
            hidden: финальное скрытое состояние
        """
        # Проекция входа
        x = self.input_projection(x)  # [batch, time, hidden_size]
        
        # LTC encoding
        encoded, hidden = self.ltc_rnn(x, hidden=hidden)  # [batch, time, hidden_size]
        
        # Нормализация
        encoded = self.output_norm(encoded)
        
        return encoded, hidden


class CTCHead(nn.Module):
    """
    CTC классификатор для фонем
    """
    
    def __init__(self, hidden_size, num_classes):
        super(CTCHead, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: encoded последовательность [batch, time, hidden_size]
        
        Returns:
            logits: логиты для каждого класса [batch, time, num_classes]
        """
        return self.classifier(x)


class LNNASR(nn.Module):
    """
    Полная модель для распознавания фонем
    
    Mel-спектрограмма → LNN Encoder → CTC Head → Фонемы
    """
    
    def __init__(self, num_classes=72, input_size=80, hidden_size=256, num_layers=2):
        super(LNNASR, self).__init__()
        
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        
        self.encoder = LNNEncoder(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers
        )
        
        self.ctc_head = CTCHead(hidden_size, num_classes)
        
        # Инициализация весов
        self._init_weights()
    
    def _init_weights(self):
        """Инициализация весов в стиле Transformer"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, mel_spec, mel_lengths=None, hidden=None):
        """
        Args:
            mel_spec: Mel-спектрограмма [batch, 80, time] или [batch, time, 80]
            mel_lengths: длины последовательностей [batch] (для padding mask)
            hidden: начальное скрытое состояние для LTC
        
        Returns:
            logits: CTC логиты [batch, time, num_classes]
            output_lengths: длины выходных последовательностей
        """
        # Приводим к [batch, time, 80]
        if mel_spec.dim() == 3 and mel_spec.shape[1] == 80:
            mel_spec = mel_spec.transpose(1, 2)
        
        batch_size, time_steps, _ = mel_spec.shape
        
        # Вычисляем длины если не переданы
        if mel_lengths is None:
            mel_lengths = torch.full((batch_size,), time_steps, dtype=torch.long, device=mel_spec.device)
        
        # Encoder
        encoded, hidden = self.encoder(mel_spec, hidden=hidden)  # [batch, time, hidden_size]
        
        # CTC Head
        logits = self.ctc_head(encoded)  # [batch, time, num_classes]
        
        # Вычисляем длины выходов (с учетом downsampling если есть)
        output_lengths = mel_lengths
        
        return logits, output_lengths
    
    def predict(self, mel_spec, greedy=True):
        """
        Инференс модели
        
        Args:
            mel_spec: Mel-спектрограмма [batch, 80, time] или [1, 80, time]
            greedy: если True, использовать greedy decoding
        
        Returns:
            predictions: предсказанные последовательности фонем
        """
        self.eval()
        
        with torch.no_grad():
            logits, _ = self.forward(mel_spec)
            
            if greedy:
                # Greedy CTC decoding
                predictions = torch.argmax(logits, dim=-1)  # [batch, time]
                
                # Удаляем повторения и blank токены
                decoded = []
                for pred in predictions:
                    prev_token = -1
                    seq = []
                    for token in pred:
                        if token != prev_token and token != 0:  # 0 = <blank>
                            seq.append(token.item())
                        prev_token = token
                    decoded.append(seq)
                
                return decoded
            else:
                # Beam search (можно добавить позже)
                return logits
    
    def get_num_params(self):
        """Возвращает количество параметров модели"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(num_classes=72, hidden_size=256, num_layers=2):
    """
    Factory функция для создания модели
    
    Args:
        num_classes: количество классов (72 для твоего vocab)
        hidden_size: размер скрытого состояния
        num_layers: количество LTC слоёв
    
    Returns:
        model: LNNASR модель
    """
    model = LNNASR(
        num_classes=num_classes,
        input_size=80,
        hidden_size=hidden_size,
        num_layers=num_layers
    )
    
    print(f"📊 Создана LNN ASR модель:")
    print(f"   Параметры: {model.get_num_params():,}")
    print(f"   Hidden size: {hidden_size}")
    print(f"   LTC слоёв: {num_layers}")
    print(f"   Классов: {num_classes}")
    
    return model
