# -*- coding: utf-8 -*-
"""
Модуль для токенизации (PyTorch версия с HuggingFace)
"""

from transformers import AutoTokenizer
import os


def create_or_load_tokenizer(vocab_path=None, model_name='cointegrated/rubert-tiny2'):
    """
    Создание или загрузка токенизатора
    
    Args:
        vocab_path: Путь к сохраненному токенизатору (опционально)
        model_name: Название предобученной модели для токенизатора
        
    Returns:
        AutoTokenizer: Токенизатор HuggingFace
    """
    if vocab_path and os.path.exists(vocab_path):
        # Загружаем сохраненный токенизатор
        print(f"📂 Загрузка токенизатора из {vocab_path}")
        tokenizer = AutoTokenizer.from_pretrained(vocab_path)
    else:
        # Создаем новый токенизатор на основе предобученной модели
        print(f"🔧 Создание токенизатора на основе {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Добавляем специальные токены если их нет
        special_tokens = {
            'pad_token': '[PAD]',
            'unk_token': '[UNK]',
            'bos_token': '[START]',
            'eos_token': '[END]'
        }
        
        # Добавляем только те токены, которых нет
        tokens_to_add = {}
        for key, value in special_tokens.items():
            if getattr(tokenizer, key) is None:
                tokens_to_add[key] = value
        
        if tokens_to_add:
            tokenizer.add_special_tokens(tokens_to_add)
        
        # Сохраняем токенизатор если указан путь
        if vocab_path:
            os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
            tokenizer.save_pretrained(vocab_path)
            print(f"💾 Токенизатор сохранен в {vocab_path}")
    
    return tokenizer


def create_tokenizers(questions_vocab_path='data/vocab/tokenizer_qs', 
                     answers_vocab_path='data/vocab/tokenizer_an',
                     model_name='cointegrated/rubert-tiny2'):
    """
    Создание токенизаторов для вопросов и ответов
    
    Args:
        questions_vocab_path: Путь к токенизатору вопросов
        answers_vocab_path: Путь к токенизатору ответов
        model_name: Название предобученной модели
        
    Returns:
        tuple: (tokenizer_qs, tokenizer_an)
    """
    print(f"\n{'='*70}")
    print("СОЗДАНИЕ ТОКЕНИЗАТОРОВ")
    print(f"{'='*70}")
    
    # Создаем токенизатор для вопросов
    tokenizer_qs = create_or_load_tokenizer(questions_vocab_path, model_name)
    print(f"✅ Токенизатор вопросов готов (vocab size: {len(tokenizer_qs)})")
    
    # Создаем токенизатор для ответов (используем тот же)
    tokenizer_an = create_or_load_tokenizer(answers_vocab_path, model_name)
    print(f"✅ Токенизатор ответов готов (vocab size: {len(tokenizer_an)})")
    
    print(f"{'='*70}\n")
    
    return tokenizer_qs, tokenizer_an


def get_special_token_ids(tokenizer):
    """
    Получение ID специальных токенов
    
    Args:
        tokenizer: Токенизатор
        
    Returns:
        dict: Словарь с ID специальных токенов
    """
    return {
        'pad_id': tokenizer.pad_token_id,
        'unk_id': tokenizer.unk_token_id,
        'start_id': tokenizer.bos_token_id if tokenizer.bos_token_id else tokenizer.cls_token_id,
        'end_id': tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.sep_token_id,
    }


def decode_tokens(tokenizer, token_ids, skip_special_tokens=True):
    """
    Декодирование токенов обратно в текст
    
    Args:
        tokenizer: Токенизатор
        token_ids: ID токенов
        skip_special_tokens: Пропускать ли специальные токены
        
    Returns:
        str: Декодированный текст
    """
    return tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)


def print_tokenizer_info(tokenizer, name="Токенизатор"):
    """
    Вывод информации о токенизаторе
    
    Args:
        tokenizer: Токенизатор
        name: Название токенизатора
    """
    print(f"\n{'='*70}")
    print(f"ИНФОРМАЦИЯ О {name.upper()}")
    print(f"{'='*70}")
    print(f"Размер словаря: {len(tokenizer)}")
    print(f"PAD токен: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    print(f"UNK токен: {tokenizer.unk_token} (ID: {tokenizer.unk_token_id})")
    print(f"START токен: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")
    print(f"END токен: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    print(f"{'='*70}\n")


def test_tokenizer(tokenizer, text="Привет, как дела?"):
    """
    Тестирование токенизатора
    
    Args:
        tokenizer: Токенизатор
        text: Текст для тестирования
    """
    print(f"\n{'='*70}")
    print("ТЕСТ ТОКЕНИЗАТОРА")
    print(f"{'='*70}")
    print(f"Исходный текст: {text}")
    
    # Токенизация
    encoded = tokenizer(text, return_tensors='pt')
    print(f"Токены (IDs): {encoded['input_ids'][0].tolist()}")
    
    # Декодирование
    decoded = tokenizer.decode(encoded['input_ids'][0], skip_special_tokens=True)
    print(f"Декодированный текст: {decoded}")
    print(f"{'='*70}\n")
