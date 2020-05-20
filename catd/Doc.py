class Doc:
    def __init__(self, doc_id, word_id_count_in_doc, number_of_words, time):
        self.doc_id = doc_id
        self.number_of_words = number_of_words
        self.word_id_count_in_doc = word_id_count_in_doc
        self.word_id_tf = {}
        self.word_id_tf_idf = {}
        self.time = time

    def __str__(self):
        return '\n[Doc info]\ndoc_id: {0}\tnumber_of_words: {1}'.format(self.doc_id, self.number_of_words)

    def __len__(self):
        return self.number_of_words

    def description(self, word_net):
        doc_info = '\n[Doc info] doc_id: {0}\tnumber_of_words: {1}\n\tword_id_count_in_doc:\n' \
                   '\t\t count |   tf   | tf_idf |  word ' \
            .format(self.doc_id, self.number_of_words)
        for word_id in self.word_id_count_in_doc.keys():
            doc_info += '\n\t\t  {0:>2}     {1:2.4f}   {2:2.4f}    {3:<4}' \
                            .format(self.word_id_count_in_doc[word_id], self.word_id_tf[word_id],
                                    self.word_id_tf_idf[word_id], word_net.nodes[word_id].word)
        return doc_info
