import catd
from gensim.models import LdaModel, CoherenceModel
from gensim import corpora


def main():
    word_net = catd.util.load_obj('reduced_weibo_COVID19')
    # word_net.vis_topic_time_statistics_aggregated()
    word_net.batch_coherence_for_lda_models(2, 21, 1)


if __name__ == '__main__':
    main()
