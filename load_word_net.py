import os

from matplotlib.font_manager import FontProperties

import catd
from pprint import pprint
import matplotlib.pyplot as plt

word_net = catd.util.load_obj('7_reduced_weibo_COVID19_new_graph')
# word_net = catd.WordNet()

# word_net.export_topic_net_for_gephi()
# word_net.export_word_net_for_gephi()

# word_net.vis_top_k_words_by_doc_count()
# word_net.vis_word_cloud()
# word_net.vis_doc_count_dist()
# word_net.vis_topic_time_statistics_aggregated()

# word_net.top_k_words_by_doc_count_in_each_topic()
word_net.generate_topic_graph()

print(word_net.description())
word_net.export_topic_net_for_gephi()

catd.util.save_obj(word_net, '7_reduced_weibo_COVID19_new_graph')