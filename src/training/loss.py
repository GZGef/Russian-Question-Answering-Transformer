# -*- coding: utf-8 -*-
"""
Модуль для функций потерь
"""

import tensorflow as tf


def masked_loss(label, pred):
    """
    Функция потерь с учетом маски
    
    Игнорирует паддинг-токены (нулевые токены) при вычислении потерь
    
    Args:
        label: Истинные метки
        pred: Предсказанные логиты
        
    Returns:
        float: Значение функции потерь
    """
    # Задаем маску, где метки не равны 0
    mask = label != 0
    
    # Определяем функцию потерь
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none'
    )
    loss = loss_object(label, pred)

    # Важно чтобы mask и loss имели одинаковый тип данных
    mask = tf.cast(mask, dtype=loss.dtype)
    # Наложение маски на loss
    loss *= mask

    # Масштабирование потерь на маску
    loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
    return loss


def masked_accuracy(label, pred):
    """
    Функция точности с учетом маски
    
    Игнорирует паддинг-токены (нулевые токены) при вычислении точности
    
    Args:
        label: Истинные метки
        pred: Предсказанные логиты
        
    Returns:
        float: Значение точности
    """
    pred = tf.argmax(pred, axis=2)
    label = tf.cast(label, pred.dtype)
    
    # Оценка совпадения метки и предсказания
    match_ = label == pred
    
    # Задаем маску, где метки не равны 0
    mask = label != 0

    # Логическое И
    match_ = match_ & mask

    # Преобразуем к одному типу и масштабирование совпадений на маску
    match_ = tf.cast(match_, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(match_) / tf.reduce_sum(mask)