# -*- coding: utf-8 -*-
"""
Модуль для декодировщика трансформера
"""

import tensorflow as tf
from keras.layers import Dropout

from src.models.attention import CausalSelfAttention, CrossAttention, FeedForward
from src.models.positional_encoding import PositionalEmbedding


class DecoderLayer(tf.keras.layers.Layer):
    """
    Слой декодировщика
    
    Состоит из:
    1. Causal Self-Attention (маскированное внимание)
    2. Cross-Attention (внимание к кодировщику)
    3. Feed Forward Network
    """
    
    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1):
        """
        Инициализация слоя декодировщика
        
        Args:
            d_model: Размерность эмбеддингов
            num_heads: Количество голов внимания
            dff: Размерность скрытого слоя FFN
            dropout_rate: Вероятность dropout
        """
        super().__init__()

        # Слой внимания с причинно-следственной связью
        self.causal_self_attention = CausalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)

        # Слой с кросс-вниманием
        self.cross_attention = CrossAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)

        # Слой прямого распространения
        self.ffn = FeedForward(d_model, dff)

    def call(self, x, context):
        """
        Применение слоя декодировщика
        
        Args:
            x: Входной тензор (от предыдущего слоя декодировщика)
            context: Контекстный тензор (от кодировщика)
            
        Returns:
            tf.Tensor: Результат слоя декодировщика
        """
        # Пропускаем последовательность токенов через:
        # Каузальный слой внимания
        x = self.causal_self_attention(x=x)
        # Слой кросс-внимания и контекстным вектором из кодировщика
        x = self.cross_attention(x=x, context=context)

        # Запомним оценки внимания на будущее
        self.last_attn_scores = self.cross_attention.last_attn_scores
        # Через слой прямого распространения
        x = self.ffn(x)  # Форма `(batch_size, seq_len, d_model)`.
        return x


class Decoder(tf.keras.layers.Layer):
    """
    Декодировщик трансформера
    
    Состоит из:
    1. Positional Embedding
    2. Dropout
    3. N слоев декодировщика
    """
    
    def __init__(self, *, num_layers, d_model, num_heads, dff, vocab_size,
                dropout_rate=0.1):
        """
        Инициализация декодировщика
        
        Args:
            num_layers: Количество слоев
            d_model: Размерность эмбеддингов
            num_heads: Количество голов внимания
            dff: Размерность скрытого слоя FFN
            vocab_size: Размер словаря
            dropout_rate: Вероятность dropout
        """
        super().__init__()

        # Инициируем переменные внутри класса
        self.d_model = d_model
        self.num_layers = num_layers

        # Создаем объект класса позиционного кодирования
        self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size,
                                                d_model=d_model)
        # Создаем объект класса для слоя регуляризации
        self.dropout = Dropout(dropout_rate)

        # Создаем сразу стек слоев декодировщиков с помощью генератора списков по числу слоев
        self.dec_layers = [
            DecoderLayer(d_model=d_model, num_heads=num_heads,
                        dff=dff, dropout_rate=dropout_rate)
            for _ in range(num_layers)]

        # Сбрасываем оценки внимания
        self.last_attn_scores = None

    def call(self, x, context):
        """
        Применение декодировщика
        
        Args:
            x: Входные токены (целевая последовательность)
            context: Контекстный тензор (от кодировщика)
            
        Returns:
            tf.Tensor: Выход декодировщика
        """
        # Подаем на вход последовательность токенов x формой (batch, target_seq_len)

        # Пропускаем через слой позиционного кодирования (и конечно же эмбеддинг)
        x = self.pos_embedding(x)  # форма на выходе (batch_size, target_seq_len, d_model)

        # Регуляризация
        x = self.dropout(x)

        # Прогоняем через num_layers слоев декодировщиков
        for i in range(self.num_layers):
            x  = self.dec_layers[i](x, context)

        # Сохраняем оценки внимания из последнего слоя
        self.last_attn_scores = self.dec_layers[-1].last_attn_scores

        # Форма x на выходе (batch_size, target_seq_len, d_model)
        return x