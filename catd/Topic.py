class Topic:
    def __init__(self, topic_id, words, feature_words, doc_count_weighted, word_count_weighted, inverse_document_frequency_weighted, time_statistics_aggregated):
        self.topic_id = topic_id
        self.words = words
        self.feature_words = feature_words
        self.doc_count_weighted = doc_count_weighted
        self.word_count_weighted = word_count_weighted
        self.inverse_document_frequency_weighted = inverse_document_frequency_weighted
        self.time_statistics_aggregated = time_statistics_aggregated

    def __str__(self):
        return '[topic info] id: {:8} num of feature_words: {} doc_count_weighted: {:5}  word_count_weighted: {:8}  ' \
               'inverse_document_frequency_weighted: {:3.5}  word: {}'\
            .format(self.topic_id, len(self.feature_words), self.doc_count_weighted, self.word_count_weighted,
                    self.inverse_document_frequency_weighted, self.words)
