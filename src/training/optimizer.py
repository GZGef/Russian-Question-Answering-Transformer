# -*- coding: utf-8 -*-
"""
Модуль для оптимизатора и расписания learning rate
"""

import tensorflow as tf
from keras.optimizers import Adam

from src.config import d_model


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Кастомное расписание learning rate
    
    Используется для оптимизации обучения трансформера
    """
    
    def __init__(self, d_model, warmup_steps=2000):
        """
        Инициализация расписания
        
        Args:
            d_model: Размерность эмбеддингов
            warmup_steps: Количество шагов для разогрева
        """
        super().__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        """
        Вычисление learning rate для заданного шага
        
        Args:
            step: Номер шага обучения
            
        Returns:
            float: Learning rate
        """
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def create_optimizer():
    """
    Создание оптимизатора с кастомным расписанием learning rate
    
    Returns:
        Adam: Оптимизатор Adam с кастомным расписанием
    """
    learning_rate = CustomSchedule(d_model)

    optimizer = Adam(
        learning_rate,
        beta_1=0.9,
        beta_2=0.98,
        epsilon=1e-9
    )
    
    return optimizer