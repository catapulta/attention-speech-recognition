import numpy as np
from collections import Counter

x = np.load(r'.\data\dev_transcripts.npy', encoding='latin1')
x = np.concatenate([x, np.load(r'.\data\train_transcripts.npy', encoding='latin1')])

letter_count = Counter()
rows = []
for words in x:
    rows.append(len(' '.join(words.astype(str))))
    letter_count.update(' '.join(words.astype(str)))
rows = np.array(rows)
print(np.percentile(rows, 95))
print(rows.max())
