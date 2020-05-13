import catd

word_net = catd.util.load_obj('reduced_weibo_COVID19_complete')
# word_net = catd.WordNet()
word_net.topic_time_statistics_aggregated_visualization()
word_net.show_word_cloud()
