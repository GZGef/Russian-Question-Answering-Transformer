# -*- coding: utf-8 -*-
"""
Модуль для позиционного кодирования
"""

import numpy as np
import tensorflow as tf


def positional_encoding(length, depth):
    """
    Создание позиционного кодирования
    
    Args:
        length: Максимальная длина последовательности
        depth: Размерность эмбеддингов
        
    Returns:
        tf.Tensor: Позиционное кодирование формы (length, depth)
    """
    depth = depth / 2

    positions = np.arange(length)[:, np.newaxis]     # форма (seq, 1)
    depths = np.arange(depth)[np.newaxis, :] / depth   # форма (1, depth)

    angle_rates = 1 / (10000**depths)         # форма (1, depth)
    angle_rads = positions * angle_rates      # форма (pos, depth)

    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1)

    return tf.cast(pos_encoding, dtype=tf.float32)  # указываем тип возвращаемых данных


class PositionalEmbedding(tf.keras.layers.Layer):
    """
    Слой позиционного кодирования
    
    Комбинирует эмбеддинги токенов с позиционным кодированием
    """
    
    def __init__(self, vocab_size, d_model):
        """
        Инициализация слоя позиционного кодирования
        
        Args:
            vocab_size: Размер словаря
            d_model: Размерность эмбеддингов
        """
        super().__init__()
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True)
        self.pos_encoding = positional_encoding(length=2048, depth=d_model)

    def compute_mask(self, *args, **kwargs):
        """
        Возвращает маску эмбеддинга
        
        Маска нужна для выравнивания последовательностей до одной длины
        с помощью pad_sequences. Метод возвращает True для ненулевых
        токенов и False для нулевых токенов (паддинг)
        
        Returns:
            tf.Tensor: Маска
        """
        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, x):
        """
        Применение позиционного кодирования
        
        Args:
            x: Входные токены
            
        Returns:
            tf.Tensor: Токены с позиционным кодированием
        """
        length = tf.shape(x)[1]
        x = self.embedding(x)

        # Этот коэффициент задает относительный масштаб встраивания и позиционного кодирования
        # C этим параметром можно и нужно играться!
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x