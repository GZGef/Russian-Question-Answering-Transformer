# -*- coding: utf-8 -*-
"""
Модуль для слоев внимания трансформера
"""

import tensorflow as tf
from keras.layers import MultiHeadAttention, LayerNormalization, Dense, Dropout
from keras.models import Sequential


class BaseAttention(tf.keras.layers.Layer):
    """
    Базовый класс для слоев внимания
    """
    
    def __init__(self, **kwargs):
        """
        Инициализация базового слоя внимания
        
        Args:
            **kwargs: Параметры для MultiHeadAttention
        """
        super().__init__()
        self.mha = MultiHeadAttention(**kwargs)
        self.layernorm = LayerNormalization()
        self.add = tf.keras.layers.Add()


class CrossAttention(BaseAttention):
    """
    Слой кросс-внимания (внимание между кодировщиком и декодировщиком)
    """
    
    def call(self, x, context):
        """
        Применение кросс-внимания
        
        Args:
            x: Входной тензор (от декодировщика)
            context: Контекстный тензор (от кодировщика)
            
        Returns:
            tf.Tensor: Результат кросс-внимания
        """
        # Пропускаем сигнал через многоголовое внимание
        attn_output, attn_scores = self.mha(
            query=x,                        # запрос
            key=context,                    # ключ
            value=context,                  # значение
            return_attention_scores=True    # возвращаем оценки внимания
        )

        # Запоминаем оценки на будущее
        self.last_attn_scores = attn_scores

        # Добавляем остаточную связь и нормализацию
        x = self.add([x, attn_output])
        x = self.layernorm(x)

        return x


class GlobalSelfAttention(BaseAttention):
    """
    Слой глобального самовнимания (внимание между токенами внутри последовательности)
    """
    
    def call(self, x):
        """
        Применение глобального самовнимания
        
        Args:
            x: Входной тензор
            
        Returns:
            tf.Tensor: Результат глобального самовнимания
        """
        # Пропускаем сигнал через многоголовое внимание
        attn_output = self.mha(
            query=x,  # запрос
            value=x,  # ключ
            key=x     # значение
        )

        # Добавляем остаточную связь и нормализацию
        x = self.add([x, attn_output])
        x = self.layernorm(x)

        return x


class CausalSelfAttention(BaseAttention):
    """
    Слой каузального самовнимания (маскированное внимание для декодировщика)
    """
    
    def call(self, x):
        """
        Применение каузального самовнимания
        
        Args:
            x: Входной тензор
            
        Returns:
            tf.Tensor: Результат каузального самовнимания
        """
        attn_output = self.mha(
            query=x,
            value=x,
            key=x,
            use_causal_mask=True  # маскирование для предотвращения "подглядывания" в будущее
        )

        x = self.add([x, attn_output])
        x = self.layernorm(x)

        return x


class FeedForward(tf.keras.layers.Layer):
    """
    Слой прямого распространения (Feed Forward Network)
    """
    
    def __init__(self, d_model, dff, dropout_rate=0.1):
        """
        Инициализация слоя прямого распространения
        
        Args:
            d_model: Размерность эмбеддингов
            dff: Размерность скрытого слоя
            dropout_rate: Вероятность dropout
        """
        super().__init__()
        self.seq = Sequential([
            Dense(dff, activation='relu'),
            Dense(d_model),
            Dropout(dropout_rate)
        ])
        self.add = tf.keras.layers.Add()
        self.layer_norm = LayerNormalization()

    def call(self, x):
        """
        Применение слоя прямого распространения
        
        Args:
            x: Входной тензор
            
        Returns:
            tf.Tensor: Результат слоя прямого распространения
        """
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x)
        return x