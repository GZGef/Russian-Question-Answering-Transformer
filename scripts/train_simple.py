# -*- coding: utf-8 -*-
"""
Пример обучения трансформера на PyTorch с GPU
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.transformer import Transformer
from src.gpu_config import configure_gpu, print_device_info, print_memory_usage


class SimpleDialogueDataset(Dataset):
    """
    Простой датасет для демонстрации
    """
    
    def __init__(self, num_samples=1000, max_len=20):
        """
        Создание синтетического датасета
        
        Args:
            num_samples: Количество примеров
            max_len: Максимальная длина последовательности
        """
        self.num_samples = num_samples
        self.max_len = max_len
        
        # Создаем случайные данные для демонстрации
        self.questions = torch.randint(3, 100, (num_samples, max_len))
        self.answers = torch.randint(3, 100, (num_samples, max_len))
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.questions[idx], self.answers[idx]


def train_step(model, batch, criterion, optimizer, device):
    """
    Один шаг обучения
    
    Args:
        model: Модель
        batch: Батч данных
        criterion: Функция потерь
        optimizer: Оптимизатор
        device: Устройство
        
    Returns:
        float: Значение потерь
    """
    questions, answers = batch
    questions = questions.to(device)
    answers = answers.to(device)
    
    # Входы и цели для декодировщика
    decoder_input = answers[:, :-1]  # Убираем последний токен
    targets = answers[:, 1:]  # Убираем первый токен
    
    # Прямой проход
    optimizer.zero_grad()
    logits = model(questions, decoder_input)
    
    # Вычисление потерь
    loss = criterion(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1)
    )
    
    # Обратный проход
    loss.backward()
    optimizer.step()
    
    return loss.item()


def main():
    """
    Основная функция обучения
    """
    print("\n" + "="*70)
    print("ОБУЧЕНИЕ ТРАНСФОРМЕРА НА PYTORCH С GPU")
    print("="*70 + "\n")
    
    # 1. Вывод информации о системе
    print_device_info()
    
    # 2. Настройка GPU
    device = configure_gpu()
    
    # 3. Параметры модели
    print("\n" + "="*70)
    print("ПАРАМЕТРЫ МОДЕЛИ")
    print("="*70)
    
    num_layers = 2
    d_model = 128
    num_heads = 4
    dff = 256
    input_vocab_size = 1000
    target_vocab_size = 1000
    dropout_rate = 0.1
    
    print(f"Количество слоев: {num_layers}")
    print(f"Размерность модели: {d_model}")
    print(f"Количество голов внимания: {num_heads}")
    print(f"Размерность FFN: {dff}")
    print(f"Размер словаря: {input_vocab_size}")
    print(f"Dropout: {dropout_rate}")
    
    # 4. Создание модели
    print("\n" + "="*70)
    print("СОЗДАНИЕ МОДЕЛИ")
    print("="*70)
    
    model = Transformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        input_vocab_size=input_vocab_size,
        target_vocab_size=target_vocab_size,
        dropout_rate=dropout_rate
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Модель создана")
    print(f"   Параметров: {num_params:,}")
    print(f"   Устройство: {device}")
    
    # 5. Создание датасета
    print("\n" + "="*70)
    print("СОЗДАНИЕ ДАТАСЕТА")
    print("="*70)
    
    dataset = SimpleDialogueDataset(num_samples=1000, max_len=20)
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    print(f"✅ Датасет создан")
    print(f"   Примеров: {len(dataset)}")
    print(f"   Размер батча: 32")
    
    # 6. Оптимизатор и функция потерь
    print("\n" + "="*70)
    print("НАСТРОЙКА ОБУЧЕНИЯ")
    print("="*70)
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Игнорируем padding
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"✅ Оптимизатор: Adam (lr=0.001)")
    print(f"✅ Функция потерь: CrossEntropyLoss")
    
    # 7. Обучение
    print("\n" + "="*70)
    print("ОБУЧЕНИЕ")
    print("="*70 + "\n")
    
    num_epochs = 3
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(dataloader):
            loss = train_step(model, batch, criterion, optimizer, device)
            total_loss += loss
            
            if (batch_idx + 1) % 10 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f"Эпоха [{epoch+1}/{num_epochs}], "
                      f"Батч [{batch_idx+1}/{len(dataloader)}], "
                      f"Loss: {avg_loss:.4f}")
        
        avg_epoch_loss = total_loss / len(dataloader)
        print(f"\n✅ Эпоха {epoch+1} завершена. Средняя потеря: {avg_epoch_loss:.4f}\n")
    
    # 8. Проверка использования памяти GPU
    if device.type == 'cuda':
        print_memory_usage()
    
    # 9. Сохранение модели
    print("="*70)
    print("СОХРАНЕНИЕ МОДЕЛИ")
    print("="*70)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'num_layers': num_layers,
        'd_model': d_model,
        'num_heads': num_heads,
        'dff': dff,
        'input_vocab_size': input_vocab_size,
        'target_vocab_size': target_vocab_size,
    }, 'transformer_pytorch.pth')
    
    print("✅ Модель сохранена в transformer_pytorch.pth")
    
    print("\n" + "="*70)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print("="*70 + "\n")
    
    print("💡 СЛЕДУЮЩИЕ ШАГИ:")
    print("   1. Модель успешно обучена на GPU")
    print("   2. Для реального обучения нужно:")
    print("      - Загрузить реальный датасет диалогов")
    print("      - Создать токенизатор (можно использовать HuggingFace)")
    print("      - Настроить параметры обучения")
    print("   3. Пример использования модели для генерации:")
    print("      model.generate(question_tokens, device=device)")


if __name__ == "__main__":
    main()