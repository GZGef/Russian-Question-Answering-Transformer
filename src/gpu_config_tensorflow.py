# -*- coding: utf-8 -*-
"""
Модуль для настройки GPU для TensorFlow
"""

import tensorflow as tf
import os


def configure_gpu(memory_limit=None, allow_growth=True):
    """
    Настройка GPU для TensorFlow
    
    Args:
        memory_limit: Лимит памяти GPU в MB (None = без ограничений)
        allow_growth: Разрешить динамическое выделение памяти
    
    Returns:
        bool: True если GPU доступен, False если используется CPU
    """
    # Получаем список физических GPU устройств
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        try:
            # Настройка для каждого GPU
            for gpu in gpus:
                if allow_growth:
                    # Разрешаем динамическое выделение памяти
                    tf.config.experimental.set_memory_growth(gpu, True)
                
                if memory_limit:
                    # Устанавливаем лимит памяти
                    tf.config.set_logical_device_configuration(
                        gpu,
                        [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)]
                    )
            
            print(f"\n{'='*60}")
            print(f"GPU Configuration")
            print(f"{'='*60}")
            print(f"Найдено GPU устройств: {len(gpus)}")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu.name}")
            print(f"Memory growth: {allow_growth}")
            if memory_limit:
                print(f"Memory limit: {memory_limit} MB")
            print(f"{'='*60}\n")
            
            return True
            
        except RuntimeError as e:
            print(f"Ошибка настройки GPU: {e}")
            print("Будет использоваться CPU")
            return False
    else:
        print(f"\n{'='*60}")
        print("GPU не найден. Используется CPU для обучения.")
        print(f"{'='*60}\n")
        return False


def get_device_info():
    """
    Получение информации об устройствах
    
    Returns:
        dict: Информация об устройствах
    """
    info = {
        'tensorflow_version': tf.__version__,
        'gpu_available': len(tf.config.list_physical_devices('GPU')) > 0,
        'gpu_devices': tf.config.list_physical_devices('GPU'),
        'cpu_devices': tf.config.list_physical_devices('CPU'),
        'built_with_cuda': tf.test.is_built_with_cuda() if hasattr(tf.test, 'is_built_with_cuda') else False
    }
    
    return info


def print_device_info():
    """
    Вывод информации об устройствах
    """
    info = get_device_info()
    
    print(f"\n{'='*60}")
    print("Информация о системе")
    print(f"{'='*60}")
    print(f"TensorFlow версия: {info['tensorflow_version']}")
    print(f"Built with CUDA: {info['built_with_cuda']}")
    print(f"GPU доступен: {info['gpu_available']}")
    print(f"Количество GPU: {len(info['gpu_devices'])}")
    
    if info['gpu_devices']:
        for i, gpu in enumerate(info['gpu_devices']):
            print(f"  GPU {i}: {gpu.name}")
    
    print(f"Количество CPU: {len(info['cpu_devices'])}")
    print(f"{'='*60}\n")


def set_mixed_precision(enabled=True):
    """
    Включение смешанной точности (mixed precision) для ускорения на GPU
    
    Args:
        enabled: Включить смешанную точность
    """
    if enabled:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("Смешанная точность (mixed_float16) включена для ускорения на GPU")
    else:
        policy = tf.keras.mixed_precision.Policy('float32')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("Используется стандартная точность (float32)")
