# note

This project is intend for processing Chinese corpus.

Please set wording dir to project root.

## data structure

'''
* WordNet
    * nodes   list[WordNode1, WordNode2, ...])
    * edges   dict[word][neighbors] -> weight)
    * docs    list[Doc1, Doc2, ...]
    * get_node_by_str dict[word] -> WordNode

* WordNode
    * id
    * name
    * doc_count
    * word_count
    * inverse_document_frequency

* Doc
    * id
    * word_count_in_doc
    * word_tf_in_doc
    * word_tf_idf
    * num_of_words
'''

## 停用词来源

https://github.com/goto456/stopwords


