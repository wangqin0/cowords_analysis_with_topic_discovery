import WordNet
import jieba
from jieba import posseg
import re
import pandas as pd
from dumpObj import *
import pprint


def fresh_run():
    # Read posts from a spread sheet
    docs = data_to_post('data/original_data/tianya_posts_test_set_1000.txt', if_output_docs=False)
    token_list = post_to_token(docs, if_output_tokens=True)
    word_net = WordNet.WordNet()
    word_net.add_corpus(token_list)
    # for token_grouped_by_doc in word_net.nodes.keys():
    #     print(word_net.nodes[token_grouped_by_doc])
    word_net.describe()
    save_obj(word_net, 'word_with_property_net')


def run_from_token_list():
    token_list = load_obj('token_list')
    word_net = WordNet.WordNet()
    word_net.add_corpus(token_list)
    # for token_grouped_by_doc in word_net.nodes.keys():
    #     print(word_net.nodes[token_grouped_by_doc])
    word_net.describe()
    save_obj(word_net, 'word_with_property_net_selected_20')


def data_to_post(filename, if_output_docs=False):
    """
    :param if_output_docs:
    :return: a list of documents (posts)
    """
    # excel
    # post_data_df = pd.read_excel('./data/tianya_data.xls')
    # print("[reading csv data] completed")

    # txt
    post_data_df = pd.read_csv(filename, sep="\n", header=None, error_bad_lines=False)
    # print(post_data_df)

    combined_docs = ''
    separate_docs = []
    num_docs = 0
    for index, row in post_data_df.iterrows():
        str_1 = keep_only_chinese(str(row[0]))
        if str_1:
            num_docs += 1
            combined_docs += str_1
            separate_docs.append(str_1)
        if index % 1000 == 0:
            print(index)
    print("[generate documents] completed")

    # save docs on request
    if if_output_docs:
        save_obj(separate_docs, 'separate_docs')
        file = open('doc/combined_doc.txt', 'w')
        file.write(str(combined_docs))
        file.close()
        file = open('doc/separate_doc.txt', 'w')
        for token in separate_docs:
            file.write(str(token))
            file.write('\n')
        file.close()
        print("[output documents] completed")

    # print summary
    print("[data to docs] summary")
    print("\tnumber of total docs: " + str(len(post_data_df)))
    print("\tnumber of valid docs: " + str(num_docs) + "\n")

    return separate_docs


def post_to_token(doc_list, if_output_tokens=False):
    '''
    :param doc_list:
    :param if_output_tokens:
    :return: list of list, tokens grouped by documents
    '''
    stop_words_list = []
    update_stop_words_list('data/stop_words/stop_words_from_HIT.txt', stop_words_list)
    update_stop_words_list('data/stop_words/stop_words_from_baidu.txt', stop_words_list)
    print("[generate stopwords dict] complete")

    token_list = []
    # selected_flag = {'v', 'n', 'ns', 'vn', 'nt', 'nr', 's', 'nz', 'nrt', 'nrtg', 'token', 'an'}
    selected_flag = {'an', 'n', 'nr', 'ns', 'nt', 'nz', 'vn'}
    doc_list_len = len(doc_list)
    count = 1
    # selected_tokens = load_obj('selected_tokens')
    for doc in doc_list:
        curr_list = []
        # word_process_selected(doc, curr_list, stop_words_list, selected_flag, selected_tokens)
        word_process(doc, curr_list, stop_words_list, selected_flag)
        token_list.append(curr_list)

        count += 1
        if count % 1000 == 0:
            print("[generate tokens] progress: " + str(count) + "\t / " + str(doc_list_len))
    print("[generate tokens] completed")
    # pprint.pprint(token_list)

    if if_output_tokens:
        save_obj(token_list, "token_list")
        print("[output tokens] completed")

    print("[post_to_token] summary")
    print("\tnumber of stopwords: " + str(len(stop_words_list)))
    print()
    return token_list


def keep_only_chinese(curr_word):
    rule = re.compile(r"[^\u4e00-\u9fa5]")
    line = rule.sub(' ', curr_word)
    return line


def word_process_selected(curr_post, token_list, stop_words, selected_flags, selected_tokens):
    seg_counter = 0
    for word in jieba.posseg.dt.cut(curr_post):
        seg_counter += 1
        word_0, flag = str(word).split('/')
        # if current word_with_property is not a stop word_with_property and its word_with_property class is selected, append it to the token_list of current doc
        if word_0 not in stop_words and word_0 in selected_tokens and flag in selected_flags and len(word_0) > 1:
            token_list.append(word_0)


def word_process(curr_post, token_list, stop_words, selected_flags):
    seg_counter = 0
    for word in jieba.posseg.dt.cut(curr_post):
        seg_counter += 1
        word_0, flag = str(word).split('/')
        # if current word_with_property is not a stop word_with_property and its word_with_property class is selected, append it to the token_list of current doc
        if word_0 not in stop_words and flag in selected_flags and len(word_0) > 1:
            token_list.append(word_0)


def update_stop_words_list(file_name, stop_words_list):
    stop_words_in_file = [line.rstrip('\n') for line in open(file_name)]
    for stop_word in stop_words_in_file:
        if stop_word not in stop_words_list:
            stop_words_list.append(stop_word)


if __name__ == '__main__':
    # run_from_token_list()
    fresh_run()
