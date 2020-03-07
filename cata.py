import jieba
from jieba import posseg
import re
import csv
import math
import time
import pickle
from gensim import corpora, models
import os


class WordNet:
    def __init__(self):
        self.nodes = []
        self.edges = {}
        self.docs = []
        self.get_node_by_str = {}


def word_cut(list_of_documents, if_output_docs=False):
    """
    :param list_of_documents:
    :param if_output_docs:
    :return: a list of cleaned documents contain only Chinese characters
    """
    cleaned_corpus = []
    stop_words_set = generate_stop_words_set_from_dir()
    for document in list_of_documents:
        # keep only Chinese characters
        document = re.compile(r"[^\u4e00-\u9fa5]").sub(' ', document)


def generate_stop_words_set_from_dir(dir_of_stop_words_files=os.path.join('data', 'stop_words')):
    """
    generate stop words set from a given directory, by iterating all the txt files
    :param dir_of_stop_words_files:
    :return: a complete set of stop words
    """
    stop_words_set = set()
    for filename in os.listdir(dir_of_stop_words_files):
        if filename.endswith(".txt"):
            file = os.path.join(dir_of_stop_words_files, filename)
            stop_words = [line.rstrip('\n') for line in open(file, encoding='utf-8')]
            for stop_word in stop_words:
                stop_words_set.add(stop_word)
    return stop_words_set




