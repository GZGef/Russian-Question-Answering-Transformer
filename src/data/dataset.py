# -*- coding: utf-8 -*-
"""
Модуль для загрузки и подготовки датасета
"""

import os
from datasets import load_dataset
import tensorflow as tf

from src.config import DATASET_NAME, BUFFER_SIZE


def load_russian_dialogues():
    """
    Загрузка русскоязычного диалогового датасета с HuggingFace
    
    Returns:
        tf.data.Dataset: Датасет с вопросами и ответами
    """
    # Загрузка датасета
    dataset = load_dataset(DATASET_NAME, split='train')
    dataset = dataset.to_tf_dataset(
        columns=['question', 'answer'],
    )
    
    # Ограничение размера датасета
    dataset = dataset.take(BUFFER_SIZE)
    
    return dataset


def split_dataset(dataset):
    """
    Разделение датасета на вопросы и ответы
    
    Args:
        dataset: Датасет с вопросами и ответами
        
    Returns:
        tuple: Кортеж из датасетов вопросов и ответов
    """
    train_questions = dataset.map(lambda x: x['question'])
    train_answers = dataset.map(lambda x: x['answer'])
    
    return train_questions, train_answers


def print_sample_dialogues(dataset, num_samples=3):
    """
    Вывод примеров диалогов из датасета
    
    Args:
        dataset: Датасет с вопросами и ответами
        num_samples: Количество примеров для вывода
    """
    train_questions = dataset.map(lambda x: x['question'])
    train_answers = dataset.map(lambda x: x['answer'])
    
    print(f"\n{'='*60}")
    print(f"Примеры диалогов из датасета")
    print(f"{'='*60}\n")
    
    for i, (q, a) in enumerate(zip(train_questions.take(num_samples), train_answers.take(num_samples))):
        q_text = q.numpy().decode('utf-8')
        a_text = a.numpy().decode('utf-8')
        
        print(f"Диалог #{i+1}:")
        print(f"\tВопрос: {q_text}")
        print(f"\tОтвет:  {a_text}")
        print()
    
    print(f"{'='*60}")
    print(f"Всего пар вопрос-ответ: {len(dataset)}")
    print(f"{'='*60}\n")


def prepare_dataset_for_training(dataset):
    """
    Подготовка датасета для обучения
    
    Args:
        dataset: Исходный датасет
        
    Returns:
        tf.data.Dataset: Подготовленный датасет для обучения
    """
    from src.config import BATCH_SIZE, BUFFER_SIZE
    
    def prepare_batch(batch):
        """Подготовка одного батча данных"""
        qs = batch["question"]
        an = batch["answer"]
        
        return (qs, an)
    
    # Подготовка датасета
    prepared_dataset = (
        dataset
        .shuffle(BUFFER_SIZE)                     # перемешиваем данные
        .batch(BATCH_SIZE)                        # делим датасет на пакеты
        .map(prepare_batch, tf.data.AUTOTUNE)     # применим функцию prepare_batch
        .prefetch(buffer_size=tf.data.AUTOTUNE)   # prefetch для ускорения обучения
    )
    
    return prepared_dataset