# -*- coding: utf-8 -*-
"""
Пример использования GPU конфигурации в PyTorch
"""

import torch
import torch.nn as nn
from src.gpu_config import (
    configure_gpu, 
    get_device, 
    print_device_info,
    print_memory_usage,
    clear_gpu_memory,
    set_seed
)


def main():
    """
    Демонстрация использования GPU в PyTorch
    """
    
    # 1. Вывод информации о системе
    print("=" * 70)
    print("ПРИМЕР ИСПОЛЬЗОВАНИЯ GPU В PYTORCH")
    print("=" * 70)
    
    print_device_info()
    
    # 2. Настройка GPU с оптимизациями
    device = configure_gpu(
        device_id=0,           # Использовать первый GPU
        memory_fraction=None,  # Без ограничения памяти (или 0.8 для 80%)
        allow_tf32=True        # Включить TensorFloat-32 для ускорения
    )
    
    # Альтернативный способ получения устройства
    # device = get_device(prefer_gpu=True)
    
    # 3. Установка seed для воспроизводимости
    set_seed(42)
    
    # 4. Создание простой модели
    print("\n" + "=" * 70)
    print("СОЗДАНИЕ И ПЕРЕНОС МОДЕЛИ НА GPU")
    print("=" * 70)
    
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.fc1 = nn.Linear(512, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, 10)
            self.relu = nn.ReLU()
        
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    # Создаем модель и переносим на GPU
    model = SimpleModel().to(device)
    print(f"✅ Модель создана и перенесена на {device}")
    print(f"   Параметров в модели: {sum(p.numel() for p in model.parameters()):,}")
    
    # 5. Создание данных на GPU
    print("\n" + "=" * 70)
    print("РАБОТА С ДАННЫМИ НА GPU")
    print("=" * 70)
    
    # Создаем тензоры на GPU
    batch_size = 32
    input_data = torch.randn(batch_size, 512).to(device)
    print(f"✅ Входные данные созданы на {device}")
    print(f"   Размер: {input_data.shape}")
    print(f"   Тип данных: {input_data.dtype}")
    
    # 6. Прямой проход через модель
    with torch.no_grad():
        output = model(input_data)
    
    print(f"✅ Прямой проход выполнен на GPU")
    print(f"   Выход размер: {output.shape}")
    
    # 7. Проверка использования памяти
    print_memory_usage(device_id=0)
    
    # 8. Пример обучения (несколько итераций)
    print("=" * 70)
    print("ПРИМЕР ОБУЧЕНИЯ НА GPU")
    print("=" * 70)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Создаем фиктивные метки
    labels = torch.randint(0, 10, (batch_size,)).to(device)
    
    # Несколько итераций обучения
    model.train()
    for epoch in range(5):
        optimizer.zero_grad()
        
        # Прямой проход
        outputs = model(input_data)
        loss = criterion(outputs, labels)
        
        # Обратный проход
        loss.backward()
        optimizer.step()
        
        print(f"Эпоха {epoch + 1}/5, Loss: {loss.item():.4f}")
    
    print("✅ Обучение завершено")
    
    # 9. Проверка финального использования памяти
    print_memory_usage(device_id=0)
    
    # 10. Очистка памяти
    del model, input_data, output, labels
    clear_gpu_memory()
    
    print("\n" + "=" * 70)
    print("ПРИМЕР ЗАВЕРШЕН УСПЕШНО!")
    print("=" * 70)
    
    # Дополнительные советы
    print("\n💡 СОВЕТЫ ПО ИСПОЛЬЗОВАНИЮ GPU:")
    print("   1. Всегда переносите модель и данные на одно устройство: .to(device)")
    print("   2. Используйте torch.cuda.empty_cache() для очистки памяти")
    print("   3. Используйте with torch.no_grad() для инференса")
    print("   4. Включите torch.backends.cudnn.benchmark = True для оптимизации")
    print("   5. Для больших моделей используйте gradient accumulation")
    print("   6. Мониторьте использование памяти GPU")
    

if __name__ == "__main__":
    main()
