# should take less than 2 minute to run all the tests

import WordNet

print('[test] build wordnet object')
WordNet.fresh_run()

print('[test] load wordnet object')
test_wordnet = WordNet.load_obj('wordnet')
test_wordnet.describe()

print('[test] run from token list')
WordNet.run_from_token_list()
