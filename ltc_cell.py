"""
Liquid Time-Constant (LTC) Cell для PyTorch
Реализация адаптивной RNN ячейки с изменяемыми временными константами
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LTCCell(nn.Module):
    """
    Liquid Time-Constant Cell
    
    Основная идея: вместо фиксированных весов RNN, мы используем
    адаптивные временные константы, которые зависят от входа и состояния.
    
    dx/dt = -[1/τ(x,h)] ⊙ h + f(x,h)
    
    где τ — адаптивная временная константа, f — переходная функция
    """
    
    def __init__(self, input_size, hidden_size, num_neurons=None):
        super(LTCCell, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_neurons = num_neurons or hidden_size
        
        # Веса для вычисления скорости изменения состояния
        self.W_input = nn.Linear(input_size, hidden_size, bias=True)
        self.W_recurrent = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Веса для вычисления адаптивных временных констант
        self.W_tau = nn.Linear(input_size + hidden_size, hidden_size, bias=True)
        
        # Веса для gating (как в LSTM/GRU)
        self.W_update = nn.Linear(input_size + hidden_size, hidden_size, bias=True)
        
        # Инициализация весов
        self._init_weights()
    
    def _init_weights(self):
        """Инициализация весов в стиле LTC"""
        for module in [self.W_input, self.W_recurrent, self.W_tau, self.W_update]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x, h, dt=1.0):
        """
        Один шаг LTC ячейки
        
        Args:
            x: вход [batch, input_size]
            h: скрытое состояние [batch, hidden_size]
            dt: шаг времени (для дискретизации)
        
        Returns:
            h_new: новое скрытое состояние
        """
        batch_size = x.shape[0]
        
        # Вычисляем候选ное новое состояние (transition function)
        input_contrib = self.W_input(x)
        recurrent_contrib = self.W_recurrent(h)
        h_candidate = torch.tanh(input_contrib + recurrent_contrib)
        
        # Вычисляем адаптивные временные константы
        tau_input = torch.cat([x, h], dim=-1)
        tau = F.softplus(self.W_tau(tau_input)) + 1.0  # τ > 0
        
        # Вычисляем update gate (сколько нового состояния смешать)
        update_input = torch.cat([x, h], dim=-1)
        update_gate = torch.sigmoid(self.W_update(update_input))
        
        # LTC обновление: dx/dt = -h/τ + f(x,h)
        # Дискретная аппроксимация: h_new = h + dt * (-h/τ + update * h_candidate)
        dh = -h / tau + update_gate * h_candidate
        h_new = h + dt * dh
        
        return h_new
    
    def init_hidden(self, batch_size, device):
        """Инициализирует скрытое состояние нулями"""
        return torch.zeros(batch_size, self.hidden_size, device=device)


class LTCRNN(nn.Module):
    """
    Многослойная LTC RNN
    
    Обрабатывает последовательности и возвращает:
    - output: скрытые состояния для каждого шага времени
    - hidden: финальное скрытое состояние
    """
    
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.0, batch_first=True):
        super(LTCRNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        
        # Создаем слои LTC ячеек
        self.layers = nn.ModuleList()
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            self.layers.append(LTCCell(layer_input_size, hidden_size))
        
        # Dropout между слоями
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
    
    def forward(self, x, hidden=None):
        """
        Args:
            x: входная последовательность [seq_len, batch, input_size] или [batch, seq_len, input_size]
            hidden: начальное скрытое состояние [num_layers, batch, hidden_size]
        
        Returns:
            output: выходы для каждого шага [seq_len, batch, hidden_size]
            hidden: финальные скрытые состояния [num_layers, batch, hidden_size]
        """
        # Приводим к формату [seq_len, batch, input_size]
        if self.batch_first:
            x = x.transpose(0, 1)
        
        seq_len, batch_size, _ = x.shape
        
        # Инициализируем скрытые состояния
        if hidden is None:
            device = x.device
            hidden = torch.stack([
                layer.init_hidden(batch_size, device) 
                for layer in self.layers
            ])
        
        # Проходим по всем шагам времени
        outputs = []
        for t in range(seq_len):
            x_t = x[t]
            
            # Создаем новые скрытые состояния (не in-place!)
            new_hidden = []

            # Проходим через все слои
            for layer_idx, layer in enumerate(self.layers):
                h_prev = hidden[layer_idx]
                h_new = layer(x_t, h_prev)
                new_hidden.append(h_new)

                # Выход слоя становится входом следующего
                x_t = h_new

                # Применяем dropout между слоями (кроме последнего)
                if self.dropout is not None and layer_idx < self.num_layers - 1:
                    x_t = self.dropout(x_t)
            
            # Обновляем hidden после всех слоёв
            hidden = torch.stack(new_hidden)

            outputs.append(x_t)
        
        # Склеиваем выходы
        output = torch.stack(outputs)  # [seq_len, batch, hidden_size]
        
        if self.batch_first:
            output = output.transpose(0, 1)
        
        return output, hidden
