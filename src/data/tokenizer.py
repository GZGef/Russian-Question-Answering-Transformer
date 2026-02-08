# -*- coding: utf-8 -*-
"""
Модуль для токенизации на базе BERT
"""

import pathlib
import re
import tensorflow as tf
import tensorflow_text as text
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab

from src.config import RESERVED_TOKENS, VOCAB_SIZE, QUESTIONS_VOCAB_PATH, ANSWERS_VOCAB_PATH


def write_vocab_file(filepath, vocab):
    """
    Запись словаря в файл
    
    Args:
        filepath: Путь к файлу
        vocab: Словарь токенов
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        for token in vocab:
            print(token, file=f)


def build_vocab_from_dataset(dataset, vocab_path, bert_tokenizer_params):
    """
    Построение словаря BERT из датасета
    
    Args:
        dataset: Датасет для построения словаря
        vocab_path: Путь к файлу словаря
        bert_tokenizer_params: Параметры токенизатора BERT
        
    Returns:
        list: Список токенов словаря
    """
    bert_vocab_args = dict(
        vocab_size=VOCAB_SIZE,
        reserved_tokens=RESERVED_TOKENS,
        bert_tokenizer_params=bert_tokenizer_params,
        learn_params={},
    )
    
    vocab = bert_vocab.bert_vocab_from_dataset(
        dataset.batch(1000).prefetch(2),
        **bert_vocab_args
    )
    
    write_vocab_file(vocab_path, vocab)
    return vocab


def create_tokenizers():
    """
    Создание токенизаторов для вопросов и ответов
    
    Returns:
        tuple: Кортеж из токенизаторов вопросов и ответов
    """
    # Параметры токенизатора (lower_case - приводим к нижнему регистру)
    bert_tokenizer_params = dict(lower_case=True)
    
    # Создание токенизаторов
    qs_tokenizer = text.BertTokenizer(QUESTIONS_VOCAB_PATH, **bert_tokenizer_params)
    an_tokenizer = text.BertTokenizer(ANSWERS_VOCAB_PATH, **bert_tokenizer_params)
    
    return qs_tokenizer, an_tokenizer


def add_start_end(ragged):
    """
    Добавление токенов [START] и [END] к последовательности
    
    Args:
        ragged: Входная последовательность
        
    Returns:
        tf.RaggedTensor: Последовательность с добавленными токенами
    """
    START = tf.argmax(tf.constant(RESERVED_TOKENS) == "[START]")
    END = tf.argmax(tf.constant(RESERVED_TOKENS) == "[END]")
    
    count = ragged.bounding_shape()[0]
    starts = tf.fill([count, 1], START)
    ends = tf.fill([count, 1], END)
    
    return tf.concat([starts, ragged, ends], axis=1)


def cleanup_text(reserved_tokens, token_txt):
    """
    Очистка текста от зарезервированных токенов
    
    Args:
        reserved_tokens: Список зарезервированных токенов
        token_txt: Текст с токенами
        
    Returns:
        tf.Tensor: Очищенный текст
    """
    # Удаление токенов, кроме "[UNK]"
    bad_tokens = [re.escape(tok) for tok in reserved_tokens if tok != "[UNK]"]
    bad_token_re = "|".join(bad_tokens)
    
    # Ищем в строке регулярку
    bad_cells = tf.strings.regex_full_match(token_txt, bad_token_re)
    
    # Отсеиваем из исходной строки все найденные включения "плохих" токенов
    result = tf.ragged.boolean_mask(token_txt, ~bad_cells)
    
    # Сцепление строк
    result = tf.strings.reduce_join(result, separator=' ', axis=-1)
    
    return result


class CustomTokenizer(tf.Module):
    """
    Кастомный токенизатор на базе BERT
    """
    
    def __init__(self, reserved_tokens, vocab_path):
        """
        Инициализация токенизатора
        
        Args:
            reserved_tokens: Список зарезервированных токенов
            vocab_path: Путь к файлу словаря
        """
        # Определяем токенизатор
        self.tokenizer = text.BertTokenizer(vocab_path, lower_case=True)
        # Зарезервированные токены
        self._reserved_tokens = reserved_tokens
        # Путь к файлу словаря
        self._vocab_path = tf.saved_model.Asset(vocab_path)
        # Читаем из файла словарь и делим по строкам
        vocab = pathlib.Path(vocab_path).read_text().splitlines()
        self.vocab = tf.Variable(vocab)

        # Для экспорта класса необходимо создать так называемые сигнатуры,
        # чтобы tensorflow понимал с какими данными он работает

        # Сигнатура для tokenize (работает с пакетами строк).
        self.tokenize.get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string))

        # Сигнатура для `detokenize` и `lookup`
        # Могут работать как с `Tensors`, так и `RaggedTensors`
        # с тензорами формы [batch, tokens]
        self.detokenize.get_concrete_function(
            tf.TensorSpec(shape=[None, None], dtype=tf.int64))
        self.detokenize.get_concrete_function(
            tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64))

        self.lookup.get_concrete_function(
            tf.TensorSpec(shape=[None, None], dtype=tf.int64))
        self.lookup.get_concrete_function(
            tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64))

        # Методы `get_*` не имеют аргументов
        self.get_vocab_size.get_concrete_function()
        self.get_vocab_path.get_concrete_function()
        self.get_reserved_tokens.get_concrete_function()

    @tf.function
    def tokenize(self, strings):
        """
        Токенизация строк
        
        Args:
            strings: Строки для токенизации
            
        Returns:
            tf.RaggedTensor: Токенизированные строки
        """
        enc = self.tokenizer.tokenize(strings)
        # Объединяем оси `word` и `word-piece`
        enc = enc.merge_dims(-2, -1)
        enc = add_start_end(enc)
        return enc

    @tf.function
    def detokenize(self, tokenized):
        """
        Детокенизация токенов
        
        Args:
            tokenized: Токены для детокенизации
            
        Returns:
            tf.Tensor: Детокенизированные строки
        """
        words = self.tokenizer.detokenize(tokenized)
        return cleanup_text(self._reserved_tokens, words)

    @tf.function
    def lookup(self, token_ids):
        """
        Поиск токенов по их ID
        
        Args:
            token_ids: ID токенов
            
        Returns:
            tf.Tensor: Токены
        """
        return tf.gather(self.vocab, token_ids)

    @tf.function
    def get_vocab_size(self):
        """
        Получение размера словаря
        
        Returns:
            int: Размер словаря
        """
        return tf.shape(self.vocab)[0]

    @tf.function
    def get_vocab_path(self):
        """
        Получение пути к файлу словаря
        
        Returns:
            str: Путь к файлу словаря
        """
        return self._vocab_path

    @tf.function
    def get_reserved_tokens(self):
        """
        Получение списка зарезервированных токенов
        
        Returns:
            tf.Tensor: Список зарезервированных токенов
        """
        return tf.constant(self._reserved_tokens)


def create_custom_tokenizers():
    """
    Создание кастомных токенизаторов
    
    Returns:
        tf.Module: Модуль с токенизаторами для вопросов и ответов
    """
    tokenizers = tf.Module()
    tokenizers.qs = CustomTokenizer(RESERVED_TOKENS, QUESTIONS_VOCAB_PATH)
    tokenizers.an = CustomTokenizer(RESERVED_TOKENS, ANSWERS_VOCAB_PATH)
    
    return tokenizers