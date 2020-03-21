import jieba
from jieba import posseg
import re


def keep_only_chinese(curr_word):
    rule = re.compile(r"[^\u4e00-\u9fa5]")
    line = rule.sub(' ', curr_word)
    return line


docs = open('data/original_data/tianya_posts_test_set_100.txt')
cut_result = {}
for doc in docs:
    for word_with_property in jieba.posseg.dt.cut(keep_only_chinese(doc)):
        word, flag = str(word_with_property).split('/')
        if flag in cut_result.keys():
            cut_result[flag].add(word)
        else:
            cut_result[flag] = set(word)

for word_property in cut_result.keys():
    print('--> ' + word_property)
    for word in cut_result[word_property]:
        print('\t' + word)

print()
