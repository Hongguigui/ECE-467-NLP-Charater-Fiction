import pathlib
import os
from keras.layers import Dense, Dropout, Activation, LSTM, Bidirectional, Embedding
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import keras.utils as ku
from keras.utils import pad_sequences
import tensorflow as tf
from numpy.random import seed
import pandas as pd
import numpy as np
# simplified chinese tokenizer
import jieba
import time
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import re
import nltk

seed(1)
tf.random.set_seed(2)
punc = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.《》（）+-=()""''/="

# skipped directories
# SKIP = ["0", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
SKIP = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v",
        "w", "x", "y", "z"]
# SKIP = []

t1 = time.time()


def get_all_items(root: pathlib.Path, exclude):
    itemList = []
    for item in root.iterdir():
        if item.name in exclude:
            continue
        if item.is_dir():
            itemList.append(get_all_items(item, []))
            continue
        itemList.append(item)
    return itemList


# begin preprocessing
largeDir = pathlib.Path("./Books")
BookList = get_all_items(largeDir, SKIP)
BookList = [item for sublist in BookList for item in sublist]

# clean the dataset
# for path in BookList:
#     print(path)
#     file = open(path, 'r')
#     try:
#         fileStr = file.read()
#     except UnicodeDecodeError as error:
#         file.close()
#         os.remove(path)
#     continue

bigString = ""

for path in BookList:
    with open(path, 'r') as fiction:
        bigString += fiction.read()

# methods to strip punctuation and symbols
# bigString = re.sub(r"[%s]+" %punc, "", bigString)
bigString = re.sub(r'[^\w\s]', '', bigString)

# list of the words in their original order
allTokens = jieba.lcut(bigString, cut_all=False)
t2 = time.time()
print("Runtime for this cell in seconds: ", t2 - t1)
print("Corpus length in words: ", len(allTokens))

os.remove('vocab.txt')
minFreq = 10
wordFreq = {}
for token in allTokens:
    wordFreq[token] = wordFreq.get(token, 0) + 1

rareWords = set()
for k, v in wordFreq.items():
    if wordFreq[k] < minFreq:
        rareWords.add(k)

words = set(allTokens)
print("Unique words before filter: ", len(words))
print("To reduce vocab size, neglect words with appearances < ", minFreq)
words = sorted(set(words) - rareWords)
print("Unique words after filter: ", len(words))

words_file_path = "vocab.txt"

words_file = open(words_file_path, 'w')
words_file.write(words)

for w in words:
    print(w)
    if w != "\n":
        words_file.write(w)
        words_file.write("\n")
    else:
        words_file.write(w)
words_file.close()

wordAsKey = dict((c, i) for i, c in enumerate(words))
intAsKey = dict((i, c) for i, c in enumerate(words))