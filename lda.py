import catd
from gensim.models import LdaModel
from gensim import matutils
import os
from pprint import pprint

# Set training parameters.
num_topics = 20
chunksize = 100000
passes = 20
iterations = 400
eval_every = 1  # Don't evaluate lda_model perplexity, takes too much time.

# for extraction
num_words = 50


def get_grouped_words(gensim_lda_model, num_topics=-1, num_words=num_words, formatted=False):
    """Get a representation for selected topics.

    Returns
    -------
    list of {str, tuple of (str, float)}
        a list of topics, each represented either as a string (when `formatted` == True) or word-probability
        pairs.

    """
    if num_topics < 0 or num_topics >= gensim_lda_model.num_topics:
        num_topics = gensim_lda_model.num_topics
        chosen_topics = range(num_topics)
    else:
        num_topics = min(num_topics, gensim_lda_model.num_topics)

        # add a little random jitter, to randomize results around the same alpha
        sort_alpha = gensim_lda_model.alpha + 0.0001 * gensim_lda_model.random_state.rand(len(gensim_lda_model.alpha))
        # random_state.rand returns float64, but converting back to dtype won't speed up anything

        sorted_topics = list(matutils.argsort(sort_alpha))
        chosen_topics = sorted_topics[:num_topics // 2] + sorted_topics[-num_topics // 2:]

    shown = []

    topic = gensim_lda_model.state.get_lambda()
    for i in chosen_topics:
        topic_ = topic[i]
        topic_ = topic_ / topic_.sum()  # normalize to probability distribution
        bestn = matutils.argsort(topic_, num_words, reverse=True)
        topic_ = [(gensim_lda_model.id2word[id], topic_[id]) for id in bestn]
        if formatted:
            topic_ = ' + '.join('%.3f*"%s"' % (v, k) for k, v in topic_)
        shown.append((i, topic_))

    return shown


word_net = catd.util.load_obj('reduced_tianya_posts_test_set_1000')
# word_net = catd.WordNet()
print(word_net.description())

word2id = word_net.generate_word_to_id()
id2word = word_net.generate_id_to_word()

token_list = word_net.get_cut_corpus()

corpus = word_net.generate_docs_to_bag_of_words()

lda_model = LdaModel(
    corpus=corpus,
    id2word=id2word,
    chunksize=chunksize,
    alpha='auto',
    eta='auto',
    iterations=iterations,
    num_topics=num_topics,
    passes=passes,
    eval_every=eval_every
)

shown = get_grouped_words(gensim_lda_model=lda_model)
pprint(shown)

lda_selected_words = set()
for topic in shown:
    for word_with_contribution in topic[1]:
        word = word_with_contribution[0]
        lda_selected_words.add(word)
        word_net.get_node_by_str[word].group = topic[0]

print(len(lda_selected_words))

word_net.output_d3_force_graph_json()

catd.util.save_obj(word_net, 'lda_labled_word_net')


with open(os.path.join('output', 'extracted_words', 'lda_top_' + str(num_words) + '_words.txt'),
          'w+', encoding='utf-8') as f:
    for word in lda_selected_words:
        f.write(word + '\n')

