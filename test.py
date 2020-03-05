from dumpObj import *

selected_tokens = load_obj('selected_tokens')
f = open('selected_tokens_20.txt', 'w+')
for token in selected_tokens:
    f.write(token + '\n')
