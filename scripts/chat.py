# -*- coding: utf-8 -*-
"""
Скрипт для запуска чат-бота на PyTorch
"""

import os
import sys

# Добавляем путь к src для импорта модулей
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch

from src.gpu_config import configure_gpu, print_device_info
from src.data.tokenizer_pytorch import create_tokenizers
from src.inference.chat_bot import run_chat_bot


def load_model_and_tokenizers():
    """
    Загрузка обученной модели PyTorch и токенизаторов
    
    Returns:
        tuple: Кортеж из (токенизатор вопросов, токенизатор ответов, модель, устройство)
    """
    print("\n" + "="*60)
    print("Загрузка модели PyTorch и токенизаторов")
    print("="*60 + "\n")
    
    # Проверка наличия моделей
    model_path = 'models/best_transformer_pytorch.pth'
    if not os.path.exists(model_path):
        model_path = 'models/final_transformer_pytorch.pth'
        if not os.path.exists(model_path):
            print(f"Ошибка: Модель не найдена по пути: models/best_transformer_pytorch.pth или models/final_transformer_pytorch.pth")
            print("\nПожалуйста, сначала обучите модель, выполнив:")
            print("python train_pytorch_full.py")
            sys.exit(1)
    
    print(f"Найдена модель: {model_path}")
    
    # Создание токенизаторов
    print("1. Загрузка токенизаторов...")
    tokenizer_qs, tokenizer_an = create_tokenizers()
    print("   Токенизаторы загружены\n")
    
    # Настройка устройства
    print("2. Настройка устройства...")
    device = configure_gpu()
    print(f"   Используется: {device}\n")
    
    # Загрузка модели
    print("3. Загрузка модели трансформера...")
    
    # Загружаем чекпоинт
    checkpoint = torch.load(model_path, map_location=device)
    
    # Создаем модель с теми же параметрами
    from src.models.transformer import Transformer
    
    # Определяем параметры модели
    if 'vocab_size_qs' in checkpoint and 'vocab_size_an' in checkpoint:
        vocab_size_qs = checkpoint['vocab_size_qs']
        vocab_size_an = checkpoint['vocab_size_an']
        num_layers = checkpoint.get('num_layers', 3)
        d_model = checkpoint.get('d_model', 128)
        num_heads = checkpoint.get('num_heads', 4)
        dff = checkpoint.get('dff', 256)
        dropout_rate = checkpoint.get('dropout_rate', 0.1)
    else:
        # Если чекпоинт не содержит информацию о размерах словарей,
        # используем значения по умолчанию
        vocab_size_qs = len(tokenizer_qs)
        vocab_size_an = len(tokenizer_an)
        num_layers = 3
        d_model = 128
        num_heads = 4
        dff = 256
        dropout_rate = 0.1
    
    model = Transformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        input_vocab_size=vocab_size_qs,
        target_vocab_size=vocab_size_an,
        dropout_rate=dropout_rate
    ).to(device)
    
    # Загружаем веса модели
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("   Модель загружена\n")
    
    print(f"Параметры модели:")
    print(f"   Количество слоев: {num_layers}")
    print(f"   Размерность модели: {d_model}")
    print(f"   Количество голов внимания: {num_heads}")
    print(f"   Размерность FFN: {dff}")
    print(f"   Размер словаря вопросов: {vocab_size_qs}")
    print(f"   Размер словаря ответов: {vocab_size_an}\n")
    
    return tokenizer_qs, tokenizer_an, model, device


def main():
    """
    Основная функция для запуска чат-бота
    """
    print("\n" + "="*60)
    print("Чат-бот на основе трансформера (PyTorch)")
    print("="*60 + "\n")
    
    # Вывод информации о системе
    print_device_info()
    
    # Загрузка модели и токенизаторов
    tokenizer_qs, tokenizer_an, model, device = load_model_and_tokenizers()
    
    # Запуск чат-бота
    print("Запуск чат-бота...")
    run_chat_bot(tokenizer_qs, tokenizer_an, model, device)


if __name__ == "__main__":
    main()