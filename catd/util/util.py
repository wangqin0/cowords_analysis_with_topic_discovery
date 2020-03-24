import pickle
import sys
import os
from pathlib import Path
import urllib.request
import logging


logging.getLogger().setLevel(logging.INFO)


def set_up_current_dir_as_working_dir(download_test_set=True):
    Path(os.path.join('data', 'original_data')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join('data', 'selected_words')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join('data', 'stop_words')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join('output', 'objects')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join('output', 'cut_docs')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join('output', 'description')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join('output', 'extracted_words')).mkdir(parents=True, exist_ok=True)
    if download_test_set:
        logging.info('downloading dataset...')
        data_set_url = 'https://raw.githubusercontent.com/dqwerter/dataset/master/tianya_posts_test_set_100.txt'
        urllib.request.urlretrieve(data_set_url, os.path.join('data', 'original_data', 'tianya_posts_test_set_100.txt'))
        logging.info('download complete.')


def save_obj(obj, name):
    with open(os.path.join(os.getcwd(), 'output', 'objects', name + '.pkl'), 'wb+') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(os.path.join(os.getcwd(), 'output', 'objects', name + '.pkl'), 'rb+') as f:
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


def count_word_in_doc(list_of_words):
    word_counter_for_separate_doc = {}
    for word in list_of_words:
        if word not in word_counter_for_separate_doc.keys():
            word_counter_for_separate_doc[word] = 1
        else:
            word_counter_for_separate_doc[word] += 1
    return word_counter_for_separate_doc
