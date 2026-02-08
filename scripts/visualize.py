# -*- coding: utf-8 -*-
"""
Скрипт для визуализации весов внимания и других графиков
"""

import os
import sys

# Добавляем путь к src для импорта модулей
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from src.config import MODEL_PATH, QUESTIONS_VOCAB_PATH, ANSWERS_VOCAB_PATH, ATTENTION_MAPS_PATH
from src.data.tokenizer import create_custom_tokenizers
from src.models.transformer import Transformer
from src.visualization.attention_viz import plot_attention_weights, save_attention_weights


def load_model_and_tokenizers():
    """
    Загрузка обученной модели и токенизаторов
    
    Returns:
        tuple: Кортеж из (токенизаторы, модель)
    """
    print("\n" + "="*60)
    print("Загрузка модели и токенизаторов")
    print("="*60 + "\n")
    
    # Проверка наличия модели
    if not os.path.exists(MODEL_PATH):
        print(f"Ошибка: Модель не найдена по пути: {MODEL_PATH}")
        print("\nПожалуйста, сначала обучите модель, выполнив:")
        print("python scripts/train.py")
        sys.exit(1)
    
    # Проверка наличия словарей
    if not os.path.exists(QUESTIONS_VOCAB_PATH):
        print(f"Ошибка: Словарь вопросов не найден по пути: {QUESTIONS_VOCAB_PATH}")
        print("\nПожалуйста, сначала обучите модель, выполнив:")
        print("python scripts/train.py")
        sys.exit(1)
    
    if not os.path.exists(ANSWERS_VOCAB_PATH):
        print(f"Ошибка: Словарь ответов не найден по пути: {ANSWERS_VOCAB_PATH}")
        print("\nПожалуйста, сначала обучите модель, выполнив:")
        print("python scripts/train.py")
        sys.exit(1)
    
    # Загрузка токенизаторов
    print("1. Загрузка токенизаторов...")
    tokenizers = create_custom_tokenizers()
    print("   Токенизаторы загружены\n")
    
    # Загрузка модели
    print("2. Загрузка модели трансформера...")
    transformer = tf.saved_model.load(MODEL_PATH)
    print("   Модель загружена\n")
    
    return tokenizers, transformer


def generate_attention_visualization(tokenizers, transformer, sentence, save_path=None):
    """
    Генерация визуализации весов внимания для предложения
    
    Args:
        tokenizers: Токенизаторы для вопросов и ответов
        transformer: Обученная модель трансформера
        sentence: Входное предложение
        save_path: Путь для сохранения графика (опционально)
    """
    from src.inference.chat_bot import ChatBot
    
    print(f"\nГенерация визуализации для предложения: '{sentence}'")
    print("-" * 60)
    
    # Создание чат-бота
    bot = ChatBot(tokenizers, transformer)
    
    # Генерация ответа и весов внимания
    answer, answer_tokens, attention_weights = bot(tf.constant(sentence))
    
    print(f"Ответ: {answer.numpy().decode('utf-8')}")
    print()
    
    # Визуализация весов внимания
    if save_path:
        save_attention_weights(sentence, answer_tokens, attention_weights[0], save_path)
    else:
        plot_attention_weights(sentence, answer_tokens, attention_weights[0])
    
    return answer, answer_tokens, attention_weights


def generate_training_history_plot(history, save_path=None):
    """
    Генерация графика истории обучения
    
    Args:
        history: История обучения (словарь с метриками)
        save_path: Путь для сохранения графика (опционально)
    """
    from src.visualization.attention_viz import plot_training_history
    
    print("\nГенерация графика истории обучения...")
    print("-" * 60)
    
    plot_training_history(history, save_path)
    
    if save_path:
        print(f"График сохранен в: {save_path}")


def interactive_visualization(tokenizers, transformer):
    """
    Интерактивный режим визуализации
    
    Args:
        tokenizers: Токенизаторы для вопросов и ответов
        transformer: Обученная модель трансформера
    """
    print("\n" + "="*60)
    print("Интерактивный режим визуализации")
    print("="*60 + "\n")
    print("Введите предложение для генерации визуализации весов внимания")
    print("Введите 'exit' или 'quit' для выхода")
    print("Введите 'history' для генерации графика истории обучения")
    print("-" * 60)
    
    while True:
        sentence = input("\nВведите предложение: ").strip()
        
        if not sentence:
            continue
        
        if sentence.lower() in ('exit', 'quit'):
            print("Выход.")
            break
        
        if sentence.lower() == 'history':
            # Запрос пути для сохранения графика
            save_path = input("Введите путь для сохранения графика (или нажмите Enter для отображения): ").strip()
            if not save_path:
                save_path = None
            
            # Генерация графика истории обучения
            # Для этого нужно загрузить историю обучения
            print("\nДля генерации графика истории обучения необходимо:")
            print("1. Обучить модель (python scripts/train.py)")
            print("2. Сохранить историю обучения в файл")
            print("3. Загрузить историю из файла")
            print("\nПока что эта функция недоступна в интерактивном режиме.")
            continue
        
        # Запрос пути для сохранения визуализации
        save_path = input("Введите путь для сохранения визуализации (или нажмите Enter для отображения): ").strip()
        if not save_path:
            save_path = None
        
        try:
            generate_attention_visualization(tokenizers, transformer, sentence, save_path)
        except Exception as e:
            print(f"Ошибка при генерации визуализации: {e}")


def main():
    """
    Основная функция для запуска визуализации
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Визуализация весов внимания трансформера')
    parser.add_argument('--sentence', type=str, help='Входное предложение для визуализации')
    parser.add_argument('--save-path', type=str, help='Путь для сохранения графика')
    parser.add_argument('--interactive', action='store_true', help='Интерактивный режим')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("Визуализация весов внимания трансформера")
    print("="*60 + "\n")
    
    # Загрузка модели и токенизаторов
    tokenizers, transformer = load_model_and_tokenizers()
    
    if args.interactive:
        # Интерактивный режим
        interactive_visualization(tokenizers, transformer)
    elif args.sentence:
        # Режим одной визуализации
        generate_attention_visualization(tokenizers, transformer, args.sentence, args.save_path)
    else:
        # Режим по умолчанию - интерактивный
        print("Использование:")
        print("  python scripts/visualize.py --sentence 'Ваше предложение'")
        print("  python scripts/visualize.py --sentence 'Ваше предложение' --save-path 'output.png'")
        print("  python scripts/visualize.py --interactive")
        print("\nЗапуск интерактивного режима...")
        interactive_visualization(tokenizers, transformer)


if __name__ == "__main__":
    # Проверка наличия GPU
    print("\nИнформация о системе:")
    print(f"TensorFlow версия: {tf.__version__}")
    print(f"Доступных GPU: {len(tf.config.list_physical_devices('GPU'))}")
    
    if tf.config.list_physical_devices('GPU'):
        print("Используется GPU для инференса")
    else:
        print("Используется CPU для инференса")
    
    # Запуск визуализации
    main()