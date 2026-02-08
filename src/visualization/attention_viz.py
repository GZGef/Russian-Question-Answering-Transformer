# -*- coding: utf-8 -*-
"""
Модуль для визуализации весов внимания
"""

import matplotlib.pyplot as plt
import tensorflow as tf

from src.config import ATTENTION_MAPS_PATH


def plot_attention_head(in_tokens, translated_tokens, attention):
    """
    Визуализация весов внимания для одной головы
    
    Args:
        in_tokens: Входные токены
        translated_tokens: Выходные токены
        attention: Веса внимания
    """
    # Пропускаем токен `[START]`
    translated_tokens = translated_tokens[1:]

    ax = plt.gca()
    ax.matshow(attention)
    ax.set_xticks(range(len(in_tokens)))
    ax.set_yticks(range(len(translated_tokens)))

    labels = [label.decode('utf-8') for label in in_tokens.numpy()]
    ax.set_xticklabels(labels, rotation=90)

    labels = [label.decode('utf-8') for label in translated_tokens.numpy()]
    ax.set_yticklabels(labels)


def plot_attention_weights(sentence, translated_tokens, attention_heads):
    """
    Визуализация весов внимания для всех голов
    
    Args:
        sentence: Входное предложение
        translated_tokens: Выходные токены
        attention_heads: Веса внимания для всех голов
    """
    in_tokens = tf.convert_to_tensor([sentence])
    in_tokens = tokenizers.qs.tokenize(in_tokens).to_tensor()
    in_tokens = tokenizers.qs.lookup(in_tokens)[0]

    fig = plt.figure(figsize=(16, 8))

    for h, head in enumerate(attention_heads):
        ax = fig.add_subplot(2, 4, h+1)

        plot_attention_head(in_tokens, translated_tokens, head)

        ax.set_xlabel(f'Head {h+1}')

    plt.tight_layout()
    plt.show()


def save_attention_weights(sentence, translated_tokens, attention_heads, filename=None):
    """
    Сохранение весов внимания в файл
    
    Args:
        sentence: Входное предложение
        translated_tokens: Выходные токены
        attention_heads: Веса внимания для всех голов
        filename: Имя файла для сохранения
    """
    import os
    
    if filename is None:
        filename = f"attention_{hash(sentence)}.png"
    
    in_tokens = tf.convert_to_tensor([sentence])
    in_tokens = tokenizers.qs.tokenize(in_tokens).to_tensor()
    in_tokens = tokenizers.qs.lookup(in_tokens)[0]

    fig = plt.figure(figsize=(16, 8))

    for h, head in enumerate(attention_heads):
        ax = fig.add_subplot(2, 4, h+1)

        plot_attention_head(in_tokens, translated_tokens, head)

        ax.set_xlabel(f'Head {h+1}')

    plt.tight_layout()
    
    # Создаем директорию если её нет
    os.makedirs(ATTENTION_MAPS_PATH, exist_ok=True)
    
    # Сохраняем график
    filepath = os.path.join(ATTENTION_MAPS_PATH, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Карта внимания сохранена в: {filepath}")


def plot_training_history(history, save_path=None):
    """
    Визуализация истории обучения
    
    Args:
        history: История обучения (словарь с метриками)
        save_path: Путь для сохранения графика
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # График потерь
    axes[0].plot(history['loss'], label='Train Loss')
    if 'val_loss' in history:
        axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Функция потерь')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # График точности
    axes[1].plot(history['accuracy'], label='Train Accuracy')
    if 'val_accuracy' in history:
        axes[1].plot(history['val_accuracy'], label='Val Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Точность')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"График обучения сохранен в: {save_path}")
    
    plt.show()