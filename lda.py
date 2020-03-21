from gensim import corpora, models
import catd
from gensim.models import LdaModel

# Set training parameters.
num_topics = 10
chunksize = 100000
passes = 20
iterations = 400
eval_every = 1  # Don't evaluate lda_model perplexity, takes too much time.

word_net = catd.util.load_obj('word_net')

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


for topic in lda_model.print_topics(num_words=20):
    print(topic)

print(lda_model.print_topics())
print(lda_model.inference(corpus))
print()
