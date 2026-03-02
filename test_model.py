"""
Тест для проверки архитектуры LNN модели
Запускает forward pass с фиктивными данными
"""

import torch
from model import create_model, LNNASR
from ltc_cell import LTCCell, LTCRNN


def test_ltc_cell():
    """Тест LTC ячейки"""
    print("🧪 Тест LTC ячейки...")
    
    batch_size = 4
    input_size = 80
    hidden_size = 128
    
    cell = LTCCell(input_size, hidden_size)
    x = torch.randn(batch_size, input_size)
    h = torch.zeros(batch_size, hidden_size)
    
    h_new = cell(x, h)
    
    assert h_new.shape == (batch_size, hidden_size), f"Unexpected shape: {h_new.shape}"
    print(f"   ✅ LTC Cell: {h_new.shape}")


def test_ltc_rnn():
    """Тест многослойной LTC RNN"""
    print("🧪 Тест LTC RNN...")
    
    batch_size = 4
    seq_len = 50
    input_size = 80
    hidden_size = 128
    num_layers = 2
    
    rnn = LTCRNN(input_size, hidden_size, num_layers=num_layers, batch_first=True)
    x = torch.randn(batch_size, seq_len, input_size)
    
    output, hidden = rnn(x)
    
    assert output.shape == (batch_size, seq_len, hidden_size), f"Unexpected output shape: {output.shape}"
    assert hidden.shape == (num_layers, batch_size, hidden_size), f"Unexpected hidden shape: {hidden.shape}"
    print(f"   ✅ LTC RNN: output={output.shape}, hidden={hidden.shape}")


def test_lnn_asr_model():
    """Тест полной ASR модели"""
    print("🧪 Тест LNN ASR модели...")
    
    batch_size = 2
    time_steps = 100
    num_classes = 72
    
    model = create_model(num_classes=num_classes, hidden_size=256, num_layers=2)
    
    # Фиктивная мел-спектрограмма [batch, 80, time]
    mel_spec = torch.randn(batch_size, 80, time_steps)
    mel_lengths = torch.tensor([time_steps, time_steps - 10], dtype=torch.long)
    
    # Forward pass
    logits, output_lengths = model(mel_spec, mel_lengths=mel_lengths)
    
    assert logits.shape == (batch_size, time_steps, num_classes), f"Unexpected logits shape: {logits.shape}"
    print(f"   ✅ LNN ASR: logits={logits.shape}, output_lengths={output_lengths}")


def test_ctc_loss():
    """Тест CTC loss"""
    print("🧪 Тест CTC loss...")
    
    batch_size = 2
    time_steps = 100
    num_classes = 72
    
    model = create_model(num_classes=num_classes, hidden_size=256, num_layers=2)
    ctc_loss = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    
    # Данные
    mel_spec = torch.randn(batch_size, 80, time_steps)
    mel_lengths = torch.tensor([time_steps, time_steps - 10], dtype=torch.long)
    
    # Таргеты (последовательности фонем)
    phones = torch.tensor([
        [1, 2, 3, 4, 5, 0, 0],  # первая последовательность
        [10, 20, 30, 0, 0, 0, 0]  # вторая последовательность
    ], dtype=torch.long)
    phone_lengths = torch.tensor([5, 3], dtype=torch.long)
    
    # Forward
    logits, output_lengths = model(mel_spec, mel_lengths=mel_lengths)
    
    # CTC loss требует [time, batch, num_classes]
    log_probs = torch.log_softmax(logits, dim=-1).transpose(0, 1)
    
    loss = ctc_loss(log_probs, phones, output_lengths, phone_lengths)
    
    assert loss.dim() == 0, "Loss should be scalar"
    assert loss.requires_grad, "Loss should require grad"
    print(f"   ✅ CTC Loss: {loss.item():.4f}")


def test_inference():
    """Тест инференса"""
    print("🧪 Тест инференса...")
    
    model = create_model(num_classes=72, hidden_size=256, num_layers=2)
    model.eval()
    
    # Фиктивная мел-спектрограмма
    mel_spec = torch.randn(1, 80, 50)
    
    with torch.no_grad():
        predictions = model.predict(mel_spec, greedy=True)
    
    assert isinstance(predictions, list), "Predictions should be list"
    assert len(predictions) == 1, "Should have 1 prediction"
    print(f"   ✅ Инференс: {len(predictions[0])} фонем")


def main():
    print("=" * 60)
    print("ТЕСТИРОВАНИЕ LNN АРХИТЕКТУРЫ")
    print("=" * 60)
    print()
    
    test_ltc_cell()
    test_ltc_rnn()
    test_lnn_asr_model()
    test_ctc_loss()
    test_inference()
    
    print()
    print("=" * 60)
    print("✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ!")
    print("=" * 60)


if __name__ == '__main__':
    main()
