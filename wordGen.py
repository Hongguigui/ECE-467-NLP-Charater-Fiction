import re
from keras.models import load_model
import numpy as np
import jieba

jieba.setLogLevel(20)


def validate_seed(vocabulary, seed):
    """Validate that all the words in the seed are part of the vocabulary"""
    print("\nValidating that all the words in the seed are part of the vocabulary: ")
    seed_words = jieba.lcut(seed, cut_all=False)
    valid = True
    for w in seed_words:
        print(w, end="")
        if w in vocabulary:
            print(" ✓ in vocabulary")
        else:
            print(" ✗ NOT in vocabulary")
            valid = False
    return valid


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


# fixed
sequence_length = 15

quantity = 50
diversity = 0.5


def generate_text(model, indices_word, word_indices, seed,
                  sequence_length, diversity, quantity):

    sentence = jieba.lcut(seed, cut_all=False)
    print("----- Generating text")
    print('----- Diversity:' + str(diversity))
    print('----- Generating with seed:\n"' + "".join(sentence[-sequence_length:]) + '"')

    for i in range(quantity):
        x_pred = np.zeros((1, sequence_length))

        for t, word in enumerate(sentence[-sequence_length:]):
            x_pred[0, t] = word_indices[word]

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_word = indices_word[next_index]

        sentence = sentence[1:]
        sentence.append(next_word)

    sentence = sentence[-quantity:]
    print("".join(sentence))
    print("\n")


vocab_file = "vocab.txt"
model_file = "LSTM_Fic_model.h5"
seed = "今天我来到了这个学校不知道任务完成情况怎么样"

model = load_model(model_file)
print("\nSummary of the Network: ")
model.summary()

vocabulary = open(vocab_file, "r").readlines()
vocabulary = [re.sub(r'(\S+)\s+', r'\1', w) for w in vocabulary]
vocabulary = set(vocabulary)
# vocabulary.remove(' \n')
vocabulary.add(' ')
vocabulary = sorted(set(vocabulary))

word_indices = dict((c, i) for i, c in enumerate(vocabulary))
indices_word = dict((i, c) for i, c in enumerate(vocabulary))


if validate_seed(vocabulary, seed):
    print("\nSeed is correct.\n")
    # repeat the seed in case is not long enough, and take only the last elements
    seed = "".join((((seed + "") * sequence_length) + seed).split(" ")[-sequence_length:])
    generate_text(model, indices_word, word_indices, seed, sequence_length, diversity, quantity)
else:
    print('\033[91mERROR: Please fix the seed string\033[0m')
    exit(0)
