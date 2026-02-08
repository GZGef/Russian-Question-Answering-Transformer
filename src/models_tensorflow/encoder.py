# -*- coding: utf-8 -*-
"""
Модуль для кодировщика трансформера
"""

import tensorflow as tf
from keras.layers import Dropout

from src.models.attention import GlobalSelfAttention, FeedForward
from src.models.positional_encoding import PositionalEmbedding


class EncoderLayer(tf.keras.layers.Layer):
    """
    Слой кодировщика
    
    Состоит из:
    1. Multi-Head Self-Attention
    2. Feed Forward Network
    """
    
    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1):
        """
        Инициализация слоя кодировщика
        
        Args:
            d_model: Размерность эмбеддингов
            num_heads: Количество голов внимания
            dff: Размерность скрытого слоя FFN
            dropout_rate: Вероятность dropout
        """
        super().__init__()

        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads,  # число голов
            key_dim=d_model,      # размерность ключа
            dropout=dropout_rate) # уровень регуляризации

        self.ffn = FeedForward(d_model, dff) # число нейронов во втором и первом Dense слое, соответственно

    def call(self, x):
        """
        Применение слоя кодировщика
        
        Args:
            x: Входной тензор
            
        Returns:
            tf.Tensor: Результат слоя кодировщика
        """
        x = self.self_attention(x)
        x = self.ffn(x)
        return x


class Encoder(tf.keras.layers.Layer):
    """
    Кодировщик трансформера
    
    Состоит из:
    1. Positional Embedding
    2. Dropout
    3. N слоев кодировщика
    """
    
    def __init__(self, *, num_layers, d_model, num_heads,
                dff, vocab_size, dropout_rate=0.1):
        """
        Инициализация кодировщика
        
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
        self.pos_embedding = PositionalEmbedding(
            vocab_size=vocab_size, d_model=d_model)

        # Создаем объект класса для слоя кодировщика
        self.enc_layers = [
            EncoderLayer(d_model=d_model,
                        num_heads=num_heads,
                        dff=dff,
                        dropout_rate=dropout_rate)
            for _ in range(num_layers)]

        # Создаем объект класса для слоя регуляризации
        self.dropout = Dropout(dropout_rate)

    def call(self, x):
        """
        Применение кодировщика
        
        Args:
            x: Входные токены
            
        Returns:
            tf.Tensor: Контекстное представление последовательности
        """
        # Форма x токена: (batch, seq_len)
        # Прогоняем последовательность токенов через слой позиционного кодирования
        x = self.pos_embedding(x)  # форма на выходе (batch_size, seq_len, d_model)

        # Прогоняем последовательность токенов через слой регуляризации
        x = self.dropout(x)

        # Прогоняем последовательность токенов через num_layers слоев кодировщика
        for i in range(self.num_layers):
            x = self.enc_layers[i](x)

        return x  # форма на выходе (batch_size, seq_len, d_model)