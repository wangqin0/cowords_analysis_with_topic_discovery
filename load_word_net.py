import catd

word_net = catd.util.load_obj('testset_weibo_COVID19')
# word_net = catd.WordNet()

word_net.vis_topic_time_statistics_aggregated()
