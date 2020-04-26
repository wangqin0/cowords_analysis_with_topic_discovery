class WordNode:
    def __init__(self, node_id, word, doc_count=0, word_count=0, inverse_document_frequency=-1, group=None):
        self.node_id = node_id
        self.word = word
        self.doc_count = doc_count
        self.word_count = word_count
        self.inverse_document_frequency = inverse_document_frequency
        self.group = group
        self.time_statistics = {}

    def __str__(self):
        return '[node info] id: {:8} doc_count: {:5}  word_count: {:8}  inverse_document_frequency: {:3.5}  word: {}'\
                   .format(self.node_id, self.doc_count, self.word_count, self.inverse_document_frequency, self.word)
