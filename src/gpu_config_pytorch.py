# -*- coding: utf-8 -*-
"""
Модуль для настройки GPU для PyTorch
"""

import torch
import os


def configure_gpu(device_id=0, memory_fraction=None, allow_tf32=True):
    """
    Настройка GPU для PyTorch
    
    Args:
        device_id: ID GPU устройства (по умолчанию 0)
        memory_fraction: Доля памяти GPU для использования (None = без ограничений)
        allow_tf32: Разрешить TensorFloat-32 для ускорения на Ampere+ GPU
    
    Returns:
        torch.device: Устройство для вычислений (cuda или cpu)
    """
    if torch.cuda.is_available():
        try:
            # Устанавливаем устройство по умолчанию
            torch.cuda.set_device(device_id)
            
            # Настройка TensorFloat-32 (ускорение на GPU Ampere и новее)
            if allow_tf32:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            
            # Оптимизация cuDNN
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            
            # Ограничение памяти (если указано)
            if memory_fraction is not None:
                torch.cuda.set_per_process_memory_fraction(memory_fraction, device_id)
            
            device = torch.device(f"cuda:{device_id}")
            
            print(f"\n{'='*60}")
            print(f"GPU Configuration")
            print(f"{'='*60}")
            print(f"Устройство: {device}")
            print(f"GPU: {torch.cuda.get_device_name(device_id)}")
            print(f"CUDA версия: {torch.version.cuda}")
            print(f"cuDNN версия: {torch.backends.cudnn.version()}")
            print(f"Общая память GPU: {torch.cuda.get_device_properties(device_id).total_memory / (1024**3):.2f} GB")
            print(f"TensorFloat-32: {allow_tf32}")
            print(f"cuDNN benchmark: {torch.backends.cudnn.benchmark}")
            if memory_fraction:
                print(f"Ограничение памяти: {memory_fraction * 100}%")
            print(f"{'='*60}\n")
            
            return device
            
        except RuntimeError as e:
            print(f"Ошибка настройки GPU: {e}")
            print("Будет использоваться CPU")
            return torch.device("cpu")
    else:
        print(f"\n{'='*60}")
        print("GPU не найден. Используется CPU для вычислений.")
        print(f"{'='*60}\n")
        return torch.device("cpu")


def get_device(prefer_gpu=True):
    """
    Получить устройство для вычислений
    
    Args:
        prefer_gpu: Предпочитать GPU если доступен
    
    Returns:
        torch.device: Устройство для вычислений
    """
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def get_device_info():
    """
    Получение информации об устройствах
    
    Returns:
        dict: Информация об устройствах
    """
    info = {
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'cudnn_version': torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'gpu_devices': []
    }
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_info = {
                'id': i,
                'name': torch.cuda.get_device_name(i),
                'compute_capability': torch.cuda.get_device_capability(i),
                'total_memory_gb': torch.cuda.get_device_properties(i).total_memory / (1024**3)
            }
            info['gpu_devices'].append(gpu_info)
    
    return info


def print_device_info():
    """
    Вывод информации об устройствах
    """
    info = get_device_info()
    
    print(f"\n{'='*60}")
    print("Информация о системе")
    print(f"{'='*60}")
    print(f"PyTorch версия: {info['pytorch_version']}")
    print(f"CUDA доступна: {info['cuda_available']}")
    
    if info['cuda_available']:
        print(f"CUDA версия: {info['cuda_version']}")
        print(f"cuDNN версия: {info['cudnn_version']}")
        print(f"Количество GPU: {info['gpu_count']}")
        
        for gpu in info['gpu_devices']:
            print(f"\n  GPU {gpu['id']}: {gpu['name']}")
            print(f"    Compute Capability: {gpu['compute_capability']}")
            print(f"    Память: {gpu['total_memory_gb']:.2f} GB")
    else:
        print("GPU не доступен, используется CPU")
    
    print(f"{'='*60}\n")


def set_seed(seed=42):
    """
    Установка seed для воспроизводимости результатов
    
    Args:
        seed: Значение seed
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Для полной воспроизводимости (может снизить производительность)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def clear_gpu_memory():
    """
    Очистка кэша памяти GPU
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("✅ Кэш памяти GPU очищен")


def get_memory_usage(device_id=0):
    """
    Получить информацию об использовании памяти GPU
    
    Args:
        device_id: ID GPU устройства
    
    Returns:
        dict: Информация о памяти
    """
    if not torch.cuda.is_available():
        return None
    
    allocated = torch.cuda.memory_allocated(device_id) / (1024**3)
    reserved = torch.cuda.memory_reserved(device_id) / (1024**3)
    total = torch.cuda.get_device_properties(device_id).total_memory / (1024**3)
    
    return {
        'allocated_gb': allocated,
        'reserved_gb': reserved,
        'total_gb': total,
        'free_gb': total - allocated
    }


def print_memory_usage(device_id=0):
    """
    Вывести информацию об использовании памяти GPU
    
    Args:
        device_id: ID GPU устройства
    """
    memory = get_memory_usage(device_id)
    
    if memory:
        print(f"\n{'='*60}")
        print(f"Использование памяти GPU {device_id}")
        print(f"{'='*60}")
        print(f"Выделено: {memory['allocated_gb']:.2f} GB")
        print(f"Зарезервировано: {memory['reserved_gb']:.2f} GB")
        print(f"Свободно: {memory['free_gb']:.2f} GB")
        print(f"Всего: {memory['total_gb']:.2f} GB")
        print(f"{'='*60}\n")
    else:
        print("GPU не доступен")


# Автоматическая настройка при импорте модуля
if __name__ != "__main__":
    # Проверяем доступность CUDA при импорте
    if torch.cuda.is_available():
        # Включаем оптимизации по умолчанию
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
