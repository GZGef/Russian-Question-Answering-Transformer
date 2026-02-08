# -*- coding: utf-8 -*-
"""
Скрипт для обучения модели трансформера на PyTorch
"""

import os
import sys
import gc
import time
import logging
import json

# Добавляем путь к src для импорта модулей
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.config import MAX_TOKENS, BATCH_SIZE, EPOCHS, LOGS_PATH, PLOTS_PATH
from src.models.transformer import Transformer
from src.data.dataset_pytorch import load_russian_dialogues, create_dataloader, print_sample_dialogues
from src.data.tokenizer_pytorch import create_tokenizers, print_tokenizer_info, get_special_token_ids
from src.gpu_config import configure_gpu, print_device_info, print_memory_usage, set_seed


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, total_epochs):
    """
    Обучение одной эпохи
    
    Args:
        model: Модель
        dataloader: DataLoader
        criterion: Функция потерь
        optimizer: Оптимизатор
        device: Устройство
        epoch: Номер эпохи
        total_epochs: Всего эпох
        
    Returns:
        float: Средняя потеря за эпоху
    """
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc=f'Эпоха {epoch}/{total_epochs}')
    
    for batch_idx, batch in enumerate(progress_bar):
        # Получаем данные
        question_ids = batch['question_ids'].to(device)
        answer_ids = batch['answer_ids'].to(device)
        
        # Входы и цели для декодировщика
        decoder_input = answer_ids[:, :-1]  # Убираем последний токен
        targets = answer_ids[:, 1:]  # Убираем первый токен
        
        # Обнуление градиентов
        optimizer.zero_grad()
        
        # Прямой проход
        logits = model(question_ids, decoder_input)
        
        # Вычисление потерь
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1)
        )
        
        # Обратный проход
        loss.backward()
        
        # Gradient clipping для стабильности
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Обновление статистики
        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)
        
        # Обновление progress bar
        progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
    
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """
    Валидация модели
    
    Args:
        model: Модель
        dataloader: DataLoader
        criterion: Функция потерь
        device: Устройство
        
    Returns:
        float: Средняя потеря на валидации
    """
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            question_ids = batch['question_ids'].to(device)
            answer_ids = batch['answer_ids'].to(device)
            
            decoder_input = answer_ids[:, :-1]
            targets = answer_ids[:, 1:]
            
            logits = model(question_ids, decoder_input)
            
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """
    Сохранение чекпоинта
    
    Args:
        model: Модель
        optimizer: Оптимизатор
        epoch: Номер эпохи
        loss: Потеря
        filepath: Путь к файлу
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filepath)
    print(f"💾 Чекпоинт сохранен: {filepath}")


def main():
    """
    Основная функция обучения
    """
    print("\n" + "="*70)
    print("ОБУЧЕНИЕ ТРАНСФОРМЕРА НА PYTORCH С РЕАЛЬНЫМИ ДАННЫМИ")
    print("="*70 + "\n")
    
    # 1. Настройка логирования
    os.makedirs(LOGS_PATH, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(LOGS_PATH, 'training.log'), encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    # 2. Настройка окружения
    print_device_info()
    device = configure_gpu()
    set_seed(42)
    
    # Логирование информации о системе
    logger.info(f"Устройство: {device}")
    logger.info(f"Количество эпох: {EPOCHS}")
    logger.info(f"Размер батча: {BATCH_SIZE}")
    logger.info(f"Максимальная длина последовательности: {MAX_TOKENS}")
    
    # 2. Загрузка датасета
    dataset = load_russian_dialogues(max_samples=20000)  # Ограничиваем для быстрого обучения
    print_sample_dialogues(dataset, num_samples=3)
    
    # 3. Создание токенизаторов
    tokenizer_qs, tokenizer_an = create_tokenizers()
    print_tokenizer_info(tokenizer_qs, "Токенизатор вопросов")
    
    # Получаем размеры словарей
    vocab_size_qs = len(tokenizer_qs)
    vocab_size_an = len(tokenizer_an)
    
    # 4. Создание DataLoader
    print(f"\n{'='*70}")
    print("СОЗДАНИЕ DATALOADER")
    print(f"{'='*70}")
    
    # Разделяем на train и validation
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset = dataset.select(range(train_size))
    val_dataset = dataset.select(range(train_size, len(dataset)))
    
    train_loader = create_dataloader(
        train_dataset, 
        tokenizer_qs, 
        tokenizer_an,
        batch_size=BATCH_SIZE,
        max_length=MAX_TOKENS,
        shuffle=True
    )
    
    val_loader = create_dataloader(
        val_dataset,
        tokenizer_qs,
        tokenizer_an,
        batch_size=BATCH_SIZE,
        max_length=MAX_TOKENS,
        shuffle=False
    )
    
    print(f"✅ Train batches: {len(train_loader)}")
    print(f"✅ Validation batches: {len(val_loader)}")
    print(f"{'='*70}\n")
    
    # 5. Создание модели
    print(f"{'='*70}")
    print("СОЗДАНИЕ МОДЕЛИ")
    print(f"{'='*70}")
    
    model = Transformer(
        num_layers=3,
        d_model=128,
        num_heads=4,
        dff=256,
        input_vocab_size=vocab_size_qs,
        target_vocab_size=vocab_size_an,
        dropout_rate=0.1
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Модель создана")
    print(f"   Параметров: {num_params:,}")
    print(f"   Устройство: {device}")
    print(f"{'='*70}\n")
    
    # 6. Оптимизатор и функция потерь
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer_an.pad_token_id)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    # 7. Обучение
    print(f"{'='*70}")
    print("НАЧАЛО ОБУЧЕНИЯ")
    print(f"{'='*70}\n")
    
    # Сбор истории обучения
    history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'learning_rate': []
    }
    
    best_val_loss = float('inf')
    
    for epoch in range(1, EPOCHS + 1):
        # Обучение
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch, EPOCHS)
        
        # Валидация
        val_loss = validate(model, val_loader, criterion, device)
        
        # Обновление learning rate
        scheduler.step(val_loss)
        
        # Сохранение истории
        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        # Логирование
        logger.info(f"Эпоха {epoch}/{EPOCHS}:")
        logger.info(f"  Train Loss: {train_loss:.4f}")
        logger.info(f"  Val Loss: {val_loss:.4f}")
        logger.info(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        print(f"\nЭпоха {epoch}/{EPOCHS}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}\n")
        
        # Сохранение лучшей модели
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                'models/best_transformer_pytorch.pth'
            )
        
        # Сохранение чекпоинта каждые 5 эпох
        if epoch % 5 == 0:
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                f'models/checkpoint_epoch_{epoch}.pth'
            )
    
    # 8. Финальное сохранение
    print(f"\n{'='*70}")
    print("СОХРАНЕНИЕ ФИНАЛЬНОЙ МОДЕЛИ")
    print(f"{'='*70}")
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size_qs': vocab_size_qs,
        'vocab_size_an': vocab_size_an,
        'num_layers': 3,
        'd_model': 128,
        'num_heads': 4,
        'dff': 256,
    }, 'models/final_transformer_pytorch.pth')
    
    print("✅ Финальная модель сохранена в models/final_transformer_pytorch.pth")
    
    # 9. Сохранение истории обучения и графиков
    print(f"\n{'='*70}")
    print("СОХРАНЕНИЕ ИСТОРИИ ОБУЧЕНИЯ И ГРАФИКОВ")
    print(f"{'='*70}")
    
    # Сохраняем историю обучения в JSON
    history_file = os.path.join(LOGS_PATH, 'training_history.json')
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f"✅ История обучения сохранена в: {history_file}")
    
    # Создаем директорию для графиков
    os.makedirs(PLOTS_PATH, exist_ok=True)
    
    # График потерь обучения
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['epoch'], history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    plt.plot(history['epoch'], history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    plt.xlabel('Эпоха')
    plt.ylabel('Потери')
    plt.title('Функция потерь во время обучения')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['epoch'], history['learning_rate'], 'g-', linewidth=2)
    plt.xlabel('Эпоха')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate во время обучения')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    loss_plot_path = os.path.join(PLOTS_PATH, 'training_loss.png')
    plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ График потерь сохранен в: {loss_plot_path}")
    
    # График сходимости (только потери)
    plt.figure(figsize=(10, 6))
    plt.plot(history['epoch'], history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    plt.plot(history['epoch'], history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    plt.xlabel('Эпоха')
    plt.ylabel('Потери')
    plt.title('Сходимость модели')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    convergence_plot_path = os.path.join(PLOTS_PATH, 'convergence.png')
    plt.savefig(convergence_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ График сходимости сохранен в: {convergence_plot_path}")
    
    # 10. Проверка использования памяти GPU
    if device.type == 'cuda':
        print()
        print_memory_usage()
    
    print(f"\n{'='*70}")
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print(f"{'='*70}\n")
    
    print("💡 СЛЕДУЮЩИЕ ШАГИ:")
    print("   1. Модель обучена на реальных данных")
    print("   2. Лучшая модель сохранена в models/best_transformer_pytorch.pth")
    print("   3. Для использования модели:")
    print("      - Загрузите модель и токенизаторы")
    print("      - Используйте model.generate() для генерации ответов")
    print("   4. Для запуска чат-бота:")
    print("      python scripts/chat.py")


if __name__ == "__main__":
    # Создаем директорию для моделей
    os.makedirs('models', exist_ok=True)
    main()