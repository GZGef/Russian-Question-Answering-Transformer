# -*- coding: utf-8 -*-
"""
Скрипт для проверки доступности CUDA и GPU в PyTorch
"""

import torch
import sys


def check_cuda():
    """
    Проверка доступности CUDA и информации о GPU
    """
    print("=" * 70)
    print("ПРОВЕРКА CUDA И GPU ДЛЯ PYTORCH")
    print("=" * 70)
    
    # Версия PyTorch
    print(f"\n📦 PyTorch версия: {torch.__version__}")
    
    # Проверка CUDA
    cuda_available = torch.cuda.is_available()
    print(f"\n🔥 CUDA доступна: {cuda_available}")
    
    if cuda_available:
        # Версия CUDA
        print(f"🔧 CUDA версия: {torch.version.cuda}")
        
        # Количество GPU
        gpu_count = torch.cuda.device_count()
        print(f"🎮 Количество GPU устройств: {gpu_count}")
        
        # Информация о каждом GPU
        print(f"\n{'=' * 70}")
        print("ИНФОРМАЦИЯ О GPU УСТРОЙСТВАХ")
        print("=" * 70)
        
        for i in range(gpu_count):
            print(f"\n🖥️  GPU {i}:")
            print(f"   Название: {torch.cuda.get_device_name(i)}")
            print(f"   Compute Capability: {torch.cuda.get_device_capability(i)}")
            
            # Память GPU
            total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"   Общая память: {total_memory:.2f} GB")
            
            # Текущее использование памяти
            if torch.cuda.is_initialized():
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                reserved = torch.cuda.memory_reserved(i) / (1024**3)
                print(f"   Выделено памяти: {allocated:.2f} GB")
                print(f"   Зарезервировано памяти: {reserved:.2f} GB")
        
        # Текущее устройство по умолчанию
        current_device = torch.cuda.current_device()
        print(f"\n🎯 Текущее GPU устройство по умолчанию: {current_device}")
        print(f"   ({torch.cuda.get_device_name(current_device)})")
        
        # Тест создания тензора на GPU
        print(f"\n{'=' * 70}")
        print("ТЕСТ СОЗДАНИЯ ТЕНЗОРА НА GPU")
        print("=" * 70)
        
        try:
            # Создаем тензор на GPU
            test_tensor = torch.randn(1000, 1000).cuda()
            print(f"✅ Успешно создан тензор на GPU")
            print(f"   Размер: {test_tensor.shape}")
            print(f"   Устройство: {test_tensor.device}")
            print(f"   Тип данных: {test_tensor.dtype}")
            
            # Простая операция на GPU
            result = test_tensor @ test_tensor.T
            print(f"✅ Успешно выполнена операция умножения матриц на GPU")
            print(f"   Результат размер: {result.shape}")
            
            # Очистка памяти
            del test_tensor, result
            torch.cuda.empty_cache()
            print(f"✅ Память GPU очищена")
            
        except Exception as e:
            print(f"❌ Ошибка при работе с GPU: {e}")
            return False
        
        # cuDNN
        print(f"\n{'=' * 70}")
        cudnn_available = torch.backends.cudnn.is_available()
        print(f"🚀 cuDNN доступен: {cudnn_available}")
        if cudnn_available:
            print(f"   cuDNN версия: {torch.backends.cudnn.version()}")
            print(f"   cuDNN enabled: {torch.backends.cudnn.enabled}")
        
        print(f"\n{'=' * 70}")
        print("✅ ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ! GPU ГОТОВ К ИСПОЛЬЗОВАНИЮ")
        print("=" * 70)
        
        return True
        
    else:
        print(f"\n{'=' * 70}")
        print("⚠️  CUDA НЕ ДОСТУПНА")
        print("=" * 70)
        print("\nВозможные причины:")
        print("1. NVIDIA GPU драйверы не установлены")
        print("2. PyTorch установлен без поддержки CUDA")
        print("3. Несовместимая версия CUDA")
        print("\nДля установки PyTorch с CUDA 12.1:")
        print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        print("=" * 70)
        
        return False


def get_recommended_device():
    """
    Получить рекомендуемое устройство для вычислений
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"\n💡 Рекомендуемое устройство: {device} ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print(f"\n💡 Рекомендуемое устройство: {device}")
    
    return device


if __name__ == "__main__":
    success = check_cuda()
    device = get_recommended_device()
    
    if not success and torch.cuda.is_available() == False:
        sys.exit(1)
    
    sys.exit(0)
