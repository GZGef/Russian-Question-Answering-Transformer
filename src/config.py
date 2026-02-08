# -*- coding: utf-8 -*-
"""
Конфигурация и гиперпараметры модели трансформера
"""

# Константы для токенизатора
RESERVED_TOKENS = ["[PAD]", "[UNK]", "[START]", "[END]"]

# Гиперпараметры модели
VOCAB_SIZE = 11000          # Размер словаря
BATCH_SIZE = 64             # Размер батча
BUFFER_SIZE = 20000         # Размер буфера для перемешивания
MAX_TOKENS = 128            # Максимальная длина последовательности

# Архитектура трансформера
num_layers = 3              # Количество слоев
d_model = 128               # Размерность эмбеддингов
dff = 256                   # Размерность скрытого слоя FFN
num_heads = 4               # Количество голов внимания
dropout_rate = 0.1          # Вероятность dropout

# Обучение
EPOCHS = 32                 # Количество эпох

# Пути к данным
DATASET_NAME = 'Den4ikAI/russian_dialogues'  # Название датасета на HuggingFace
QUESTIONS_VOCAB_PATH = 'data/vocab/questions_vocab.txt'  # Путь к словарю вопросов
ANSWERS_VOCAB_PATH = 'data/vocab/answers_vocab.txt'      # Путь к словарю ответов

# Пути к моделям
MODEL_PATH = 'models/transformer.keras'  # Путь к сохраненной модели
CHAT_BOT_PATH = 'models/chat_bot'        # Путь к сохраненному чат-боту

# Пути к выходным данным
LOGS_PATH = 'outputs/logs'               # Путь к логам обучения
PLOTS_PATH = 'outputs/plots'             # Путь к графикам
ATTENTION_MAPS_PATH = 'outputs/attention_maps'  # Путь к картам внимания