import pickle
import sys
import os
import re
import jieba
from jieba import posseg


def save_obj(obj, name):
    with open(os.path.join(os.getcwd(), 'output', 'objects', name + '.pkl'), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(os.path.join(os.getcwd(), 'output', 'objects', name + '.pkl'), 'rb') as f:
        return pickle.load(f)


def display_progress(prompt, curr_progress, total):
    if curr_progress % 100 == 0:
        progress_percent = curr_progress/total
        sys.stdout.write('\r')
        sys.stdout.write('%s [%s%s]%3.1f%s' % (
            prompt, '█' * int(progress_percent * 50), ' ' * int(50 - int(progress_percent * 50)), progress_percent * 100, '%'))
    elif curr_progress == total:
        sys.stdout.write('\r')
        sys.stdout.write('%s [%s]100%s\n' % (prompt, '█' * 50, '%'))
    sys.stdout.flush()


def collect_mutual_words_to_set_from_dir(dir_of_txt_files):
    """
    generate stop words set from a given directory, by iterating all the txt files
    :param dir_of_txt_files: each txt file should contain words sperated by '\n'
    :return: a complete set of words
    """
    word_set = set()
    is_initialized = False
    for filename in os.listdir(dir_of_txt_files):
        curr_word_set = set()
        if filename.endswith(".txt"):
            file = os.path.join(dir_of_txt_files, filename)
            words = [line.rstrip('\n') for line in open(file, encoding='utf-8')]
            for word in words:
                curr_word_set.add(word)
            if is_initialized is True and len(curr_word_set) != 0:
                word_set = word_set.intersection(curr_word_set)
            else:
                word_set = curr_word_set
                is_initialized = True
    return word_set


def collect_all_words_to_set_from_dir(dir_of_txt_files):
    """
    generate stop words set from a given directory, by iterating all the txt files
    :param dir_of_txt_files: each txt file should contain words sperated by '\n'
    :return: a complete set of words
    """
    word_set = set()
    for filename in os.listdir(dir_of_txt_files):
        if filename.endswith(".txt"):
            file = os.path.join(dir_of_txt_files, filename)
            words = [line.rstrip('\n') for line in open(file, encoding='utf-8')]
            for word in words:
                word_set.add(word)
    return word_set


def word_cut(list_of_docs, stop_words_set, selected_words_set=None,
             selected_part_of_speech= None, if_output_tokens=False):
    """
    :param selected_words_set:
    :param stop_words_set:
    :param if_output_tokens:
    :param selected_part_of_speech:
    :param list_of_docs:
    :return: a list of cleaned documents contain only Chinese characters
    """
    if selected_part_of_speech is None:
        # set default selection
        selected_part_of_speech = {'an', 'n', 'nr', 'ns', 'nt', 'nz', 'vn'}

    corpus_after_cut = []
    keep_chinese_chars = re.compile(r"[^\u4e00-\u9fa5]")
    num_of_docs = len(list_of_docs)
    index = 0

    for doc in list_of_docs:
        list_of_cut_words = []
        # keep only Chinese characters
        doc = keep_chinese_chars.sub(' ', doc)
        for word_with_part_of_speech in jieba.posseg.dt.cut(doc):
            word, part_of_speech = str(word_with_part_of_speech).split('/')
            if selected_words_set is None:
                if word not in stop_words_set and part_of_speech in selected_part_of_speech and len(word) > 1:
                    list_of_cut_words.append(word)
            else:
                if word not in stop_words_set and part_of_speech in selected_part_of_speech and len(word) > 1 \
                        and word in selected_words_set:
                    list_of_cut_words.append(word)
        corpus_after_cut.append(list_of_cut_words)

        index += 1
        display_progress('word cut', index, num_of_docs)

    if if_output_tokens:
        save_obj(corpus_after_cut, 'list_of_docs')

    return corpus_after_cut


def count_word_in_doc(list_of_words):
    word_counter_for_separate_doc = {}
    for word in list_of_words:
        if word not in word_counter_for_separate_doc.keys():
            word_counter_for_separate_doc[word] = 1
        else:
            word_counter_for_separate_doc[word] += 1
    return word_counter_for_separate_doc
