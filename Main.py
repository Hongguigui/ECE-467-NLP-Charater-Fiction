import pathlib
import os
from keras.layers import Embedding, LSTM, Dense, Dropout
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
SKIP = ["0", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
# SKIP = []


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
freqDist = nltk.FreqDist(allTokens)
words = freqDist.most_common(50000)

afterRareTokens = [word for word in allTokens if word in words]

charSet = set(afterRareTokens)
chars = sorted(list(charSet))
char_to_int = dict((c, i) for i, c in enumerate(chars))

numWords = len(afterRareTokens)
numVocab = len(charSet)

print(afterRareTokens)

print("Number of words: ", numWords)
print("Vocab size: ", numVocab)

# seqLen = 100
# dataX = []
# dataY = []
# for i in range(0, numWords - seqLen, 1):
#     seqIn = allTokens[i:i + seqLen]
#     seqOut = allTokens[i + seqLen]
#     dataX.append([char_to_int[char] for char in seqIn])
#     dataY.append(char_to_int[seqOut])
# nPatterns = len(dataX)
# print("Total Patterns: ", nPatterns)
#
# # reshape X to be [samples, time steps, features]
# X = np.reshape(dataX, (nPatterns, seqLen, 1))
# # normalize
# X = X / float(numVocab)
# # one hot encode the output variable
# y = to_categorical(dataY)
# print(y)
# print(y.shape)



