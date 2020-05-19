import catd
from gensim.models import LdaModel, CoherenceModel
from gensim import corpora


def main():
    word_net = catd.util.load_obj('reduced_weibo_COVID19')
    nums_of_topics, coherence_model_list = word_net.batch_coherence_for_lda_models(5, 51, 5)


if __name__ == '__main__':
    main()
