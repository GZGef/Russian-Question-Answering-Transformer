# -*- coding: utf-8 -*-
"""
Модуль для полной архитектуры трансформера
"""

import tensorflow as tf

from src.models.encoder import Encoder
from src.models.decoder import Decoder


class Transformer(tf.keras.Model):
    """
    Полная архитектура трансформера
    
    Состоит из:
    1. Кодировщика
    2. Декодировщика
    3. Финального слоя (линейный слой для классификации)
    """
    
    def __init__(
        self,
        *,
        num_layers,
        d_model,
        num_heads,
        dff,
        input_vocab_size,
        target_vocab_size,
        dropout_rate=0.1
    ):
        """
        Инициализация трансформера
        
        Args:
            num_layers: Количество слоев
            d_model: Размерность эмбеддингов
            num_heads: Количество голов внимания
            dff: Размерность скрытого слоя FFN
            input_vocab_size: Размер словаря входных данных
            target_vocab_size: Размер словаря выходных данных
            dropout_rate: Вероятность dropout
        """
        super().__init__()
        # Кодировщик
        self.encoder = Encoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            vocab_size=input_vocab_size,
            dropout_rate=dropout_rate
        )

        # Декодировщик
        self.decoder = Decoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            vocab_size=target_vocab_size,
            dropout_rate=dropout_rate
        )
        
        # Конечный слой
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inputs, training=False):
        """
        Применение трансформера
        
        Args:
            inputs: Кортеж из (context, x)
            training: Режим обучения
            
        Returns:
            tf.Tensor: Логиты предсказаний
        """
        # Чтобы использовать метод `.fit` для обучения модели, необходимо передать
        # все входные данные в первом аргументе
        context, x = inputs

        # Передаем контекст в кодировщик
        context = self.encoder(context)  # форма выходных данных (batch_size, context_len, d_model)

        # Передаем контекст и целевой вектор в декодировщик
        x = self.decoder(x, context, training=training)  # форма выходных данных (batch_size, target_len, d_model)

        # Прогоняем выходные данные через финальный слой
        logits = self.final_layer(x)  # форма выходных данных (batch_size, target_len, target_vocab_size)

        try:
            # После прохождения данных через все слои необходимо удалить
            # маску, чтобы она не масштабировала потери и метрики
            # Обработчик ошибок позволяет избежать исключений при повторной попытке удаления
            del logits._keras_mask
        except AttributeError:  # отлавливаем ошибку отсутствия аттрибута
            pass

        # Возвращаем наши логиты
        return logits