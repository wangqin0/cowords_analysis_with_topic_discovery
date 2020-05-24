import catd


if __name__ == '__main__':
    # catd.util.vis_post_num_time_stats('weibo_COVID19.db')

    word_net = catd.util.load_obj('reduced_weibo_COVID19')
    # word_net = catd.WordNet()
    # print(word_net.description())

    # word_net.vis_word_cloud()
    # word_net.vis_doc_count_dist()
    word_net.vis_top_k_words_by_doc_count()
    # word_net.vis_topic_time_statistics_aggregated()

    # word_net.export_word_net_for_gephi()
    # word_net.export_topic_net_for_gephi()
