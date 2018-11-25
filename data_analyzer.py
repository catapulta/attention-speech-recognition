import numpy as np
from collections import Counter

x = np.load(r'.\data\dev_transcripts.npy', encoding='latin1')
x = np.concatenate([x, np.load(r'.\data\train_transcripts.npy', encoding='latin1')])

# concat = ''
# word_count = Counter()
letter_count = Counter()
rows = []
for words in x:
#     word_count.update(words)
    rows.append(len(' '.join(words.astype(str))))
    letter_count.update(' '.join(words.astype(str)))
rows = np.array(rows)
print(np.percentile(rows, 95))
print(rows.max())
#
# print(word_count)
# print(letter_count)
# print(len(letter_count))
#
# print(list(letter_count.keys()))

# x = np.load(r'.\data\dev.npy', encoding='latin1')
# print(x[0].shape)