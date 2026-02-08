# -*- coding: utf-8 -*-
"""
Модуль для загрузки и подготовки датасета (PyTorch версия)
"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

from src.config import DATASET_NAME, BUFFER_SIZE, BATCH_SIZE


class DialogueDataset(Dataset):
    """
    PyTorch Dataset для диалогов
    """
    
    def __init__(self, hf_dataset, tokenizer_qs, tokenizer_an, max_length=128):
        """
        Инициализация датасета
        
        Args:
            hf_dataset: HuggingFace датасет
            tokenizer_qs: Токенизатор для вопросов
            tokenizer_an: Токенизатор для ответов
            max_length: Максимальная длина последовательности
        """
        self.data = hf_dataset
        self.tokenizer_qs = tokenizer_qs
        self.tokenizer_an = tokenizer_an
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Получение одного примера
        
        Args:
            idx: Индекс примера
            
        Returns:
            tuple: (question_ids, answer_ids)
        """
        item = self.data[idx]
        question = item['question']
        answer = item['answer']
        
        # Проверка на None или пустые строки
        if question is None or not isinstance(question, str):
            question = ""
        if answer is None or not isinstance(answer, str):
            answer = ""
        
        # Токенизация вопроса
        question_encoded = self.tokenizer_qs(
            text=question,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Токенизация ответа
        answer_encoded = self.tokenizer_an(
            text=answer,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'question_ids': question_encoded['input_ids'].squeeze(0),
            'answer_ids': answer_encoded['input_ids'].squeeze(0),
            'question_mask': question_encoded['attention_mask'].squeeze(0),
            'answer_mask': answer_encoded['attention_mask'].squeeze(0)
        }


def load_russian_dialogues(max_samples=None):
    """
    Загрузка русскоязычного диалогового датасета с HuggingFace
    
    Args:
        max_samples: Максимальное количество примеров (None = все)
        
    Returns:
        Dataset: HuggingFace датасет с вопросами и ответами
    """
    print(f"\n{'='*70}")
    print("ЗАГРУЗКА ДАТАСЕТА")
    print(f"{'='*70}")
    print(f"Датасет: {DATASET_NAME}")
    
    # Загрузка датасета
    dataset = load_dataset(DATASET_NAME, split='train')
    
    # Ограничение размера если указано
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    print(f"✅ Загружено примеров: {len(dataset)}")
    print(f"{'='*70}\n")
    
    return dataset


def create_dataloader(dataset, tokenizer_qs, tokenizer_an, batch_size=BATCH_SIZE, 
                      max_length=128, shuffle=True, num_workers=0):
    """
    Создание PyTorch DataLoader
    
    Args:
        dataset: HuggingFace датасет
        tokenizer_qs: Токенизатор для вопросов
        tokenizer_an: Токенизатор для ответов
        batch_size: Размер батча
        max_length: Максимальная длина последовательности
        shuffle: Перемешивать ли данные
        num_workers: Количество воркеров для загрузки данных
        
    Returns:
        DataLoader: PyTorch DataLoader
    """
    dialogue_dataset = DialogueDataset(
        dataset, 
        tokenizer_qs, 
        tokenizer_an, 
        max_length
    )
    
    dataloader = DataLoader(
        dialogue_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True  # Ускоряет перенос на GPU
    )
    
    return dataloader


def print_sample_dialogues(dataset, num_samples=3):
    """
    Вывод примеров диалогов из датасета
    
    Args:
        dataset: HuggingFace датасет
        num_samples: Количество примеров для вывода
    """
    print(f"\n{'='*70}")
    print(f"ПРИМЕРЫ ДИАЛОГОВ ИЗ ДАТАСЕТА")
    print(f"{'='*70}\n")
    
    for i in range(min(num_samples, len(dataset))):
        item = dataset[i]
        print(f"Диалог #{i+1}:")
        print(f"  Вопрос: {item['question']}")
        print(f"  Ответ:  {item['answer']}")
        print()
    
    print(f"{'='*70}")
    print(f"Всего пар вопрос-ответ: {len(dataset)}")
    print(f"{'='*70}\n")
