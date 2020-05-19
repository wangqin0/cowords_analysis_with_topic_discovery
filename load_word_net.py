import catd

word_net = catd.util.load_obj('reduced_weibo_COVID19_complete')
# word_net = catd.WordNet()

# word_net.topic_time_statistics_aggregated_visualization()
# word_net.show_word_cloud()

# word_net.output_d3_force_graph_json()
topic_num, coherence_model_list = word_net.batch_coherence_for_lda_models(5, 51, 5)
catd.util.save_obj(coherence_model_list, 'coherence_model_list')
