# -*- coding: utf-8 -*-
"""
Модуль для чат-бота и генерации ответов на PyTorch
"""

import torch
from src.config import MAX_TOKENS


class ChatBotPyTorch:
    """
    Класс чат-бота для генерации ответов на вопросы (PyTorch версия)
    """
    
    def __init__(self, tokenizer_qs, tokenizer_an, transformer, device='cpu'):
        """
        Инициализация чат-бота
        
        Args:
            tokenizer_qs: Токенизатор для вопросов
            tokenizer_an: Токенизатор для ответов
            transformer: Обученная модель трансформера (PyTorch)
            device: Устройство (cpu/cuda)
        """
        self.tokenizer_qs = tokenizer_qs
        self.tokenizer_an = tokenizer_an
        self.transformer = transformer
        self.device = device
        self.transformer.eval()
        
    def generate(
        self, 
        question, 
        max_length=MAX_TOKENS, 
        temperature=1.0, 
        top_k=50,
        top_p=0.95
    ):
        """
        Генерация ответа на вопрос
        
        Args:
            question: Вопрос (строка)
            max_length: Максимальная длина ответа
            temperature: Температура для семплирования
            top_k: Top-k семплирование
            top_p: Top-p (nucleus) семплирование
            
        Returns:
            dict: Словарь с ответом и дополнительной информацией
        """
        self.transformer.eval()
        
        with torch.no_grad():
            # Токенизация вопроса
            question_tokens = self.tokenizer_qs(
                question, 
                truncation=True, 
                max_length=MAX_TOKENS,
                return_tensors='pt'
            )['input_ids'].to(self.device)
            
            # Начальные токены для генерации
            if hasattr(self.tokenizer_an, 'bos_token_id') and self.tokenizer_an.bos_token_id is not None:
                generated_tokens = torch.tensor([[self.tokenizer_an.bos_token_id]], device=self.device)
            elif hasattr(self.tokenizer_an, 'cls_token_id') and self.tokenizer_an.cls_token_id is not None:
                generated_tokens = torch.tensor([[self.tokenizer_an.cls_token_id]], device=self.device)
            else:
                # Если нет BOS токена, используем первый токен
                generated_tokens = torch.tensor([[self.tokenizer_an.encode('')[0]]], device=self.device)
            
            # Получаем ID паддинга для игнорирования
            pad_token_id = self.tokenizer_an.pad_token_id if hasattr(self.tokenizer_an, 'pad_token_id') else 0
            eos_token_id = self.tokenizer_an.eos_token_id if hasattr(self.tokenizer_an, 'eos_token_id') else None
            
            # Генерация пошагово
            for _ in range(max_length):
                # Получаем выход модели
                with torch.no_grad():
                    logits = self.transformer(question_tokens, generated_tokens)
                
                # Берем логиты последнего токена
                next_token_logits = logits[:, -1, :] / temperature
                
                # Применяем top-k и top-p фильтрацию
                if top_k > 0 or top_p > 0.0:
                    # Сортируем логиты
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    sorted_probs = torch.softmax(sorted_logits, dim=-1)
                    
                    # Top-p фильтрация
                    if top_p > 0.0:
                        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                        # Удаляем токены с кумулятивной вероятностью > top_p
                        sorted_indices_to_remove = cumulative_probs > top_p
                        # Оставляем хотя бы один токен
                        sorted_indices_to_remove = sorted_indices_to_remove.clone()
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        
                        # Фильтруем с помощью маски
                        sorted_probs = sorted_probs.masked_fill(sorted_indices_to_remove, 0)
                    
                    # Top-k фильтрация
                    if top_k > 0:
                        # Оставляем только top-k токенов
                        k = min(top_k, sorted_probs.size(-1))
                        mask = torch.zeros_like(sorted_probs)
                        mask[..., :k] = 1
                        sorted_probs = sorted_probs * mask
                    
                    # Нормализуем вероятности
                    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                    
                    # Выбираем следующий токен
                    next_token = torch.multinomial(sorted_probs, num_samples=1)
                    next_token = sorted_indices.gather(-1, next_token)
                else:
                    # Используем argmax без фильтрации
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Проверяем на EOS токен
                if eos_token_id is not None and next_token.item() == eos_token_id:
                    break
                
                # Проверяем на паддинг
                if next_token.item() == pad_token_id:
                    break
                
                # Добавляем новый токен к сгенерированной последовательности
                generated_tokens = torch.cat([generated_tokens, next_token], dim=-1)
                
                # Если сгенерировано достаточно токенов, прерываем
                if generated_tokens.shape[1] >= max_length:
                    break
            
            # Декодируем токены в текст
            generated_text = self.tokenizer_an.decode(
                generated_tokens[0], 
                skip_special_tokens=True
            )
            
            return {
                'answer': generated_text,
                'tokens': generated_tokens[0].tolist(),
                'question_tokens': question_tokens[0].tolist()
            }
    
    def __call__(self, question, **kwargs):
        """
        Алиас для generate
        """
        return self.generate(question, **kwargs)


def run_chat_bot(tokenizer_qs, tokenizer_an, transformer, device='cpu'):
    """
    Запуск чат-бота в интерактивном режиме (PyTorch версия)
    
    Args:
        tokenizer_qs: Токенизатор для вопросов
        tokenizer_an: Токенизатор для ответов
        transformer: Обученная модель трансформера (PyTorch)
        device: Устройство (cpu/cuda)
    """
    bot = ChatBotPyTorch(tokenizer_qs, tokenizer_an, transformer, device)
    
    print("\n" + "="*60)
    print("Чат-бот (PyTorch) готов к общению!")
    print("Введите 'exit' или 'quit' для выхода")
    print("="*60 + "\n")
    
    while True:
        q = input("Вы: ").strip()
        if not q:
            continue

        if q.lower() in ('exit', 'quit'):
            print("Выход.")
            break

        try:
            result = bot(q)
            print(f'{"Вопрос:":25s}: {q}')
            print(f'{"Бот:":25s}: {result["answer"]}')
            print()
        except Exception as e:
            print(f"Ошибка при генерации ответа: {e}")
            import traceback
            traceback.print_exc()
            print()
