from __future__ import print_function

import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Sequential


BASE_DIR = ''
GLOVE_DIR = BASE_DIR + '/home/dinara/word2vec/glove.6B/'
#ODP_1BILNEWS_W2V_DIR = BASE_DIR + '/home/dinara/word2vec/word2vec_gensim_ODP/ODP_word2vec/'
#GN_DIR = BASE_DIR + '/home/dinara/word2vec/word2vec_gensim_ODP/ODP_word2vec/'

TEXT_DATA_DIR = BASE_DIR + '/home/dinara/word2vec/20_newsgroup/'
MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
#EMBEDDING_DIM = 300

VALIDATION_SPLIT = 0.2

# first, build index mapping words in the embeddings set
# to their embedding vector

print('Indexing word vectors.')

embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
#f = open(os.path.join(ODP_1BILNEWS_W2V_DIR, 'OUTV_dmoz_pages_all_no_world_only_and_1bil_words_news_10context_300f_0mincount.txt'))
#f = open(os.path.join(GN_DIR, 'GoogleNews-vectors-negative300.bin'))

for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

# second, prepare text samples and their labels
print('Processing text dataset')

texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids
for name in sorted(os.listdir(TEXT_DATA_DIR)):
    path = os.path.join(TEXT_DATA_DIR, name)
    if os.path.isdir(path):
        label_id = len(labels_index)
        labels_index[name] = label_id
        for fname in sorted(os.listdir(path)):
            if fname.isdigit():
                fpath = os.path.join(path, fname)
                if sys.version_info < (3,):
                    f = open(fpath)
                else:
                    f = open(fpath, encoding='latin-1')
                t = f.read()
                i = t.find('\n\n')  # skip header
                if 0 < i:
                    t = t[i:]
                texts.append(t)
                f.close()
                labels.append(label_id)

print('Found %s texts.' % len(texts))

# finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]

print('Preparing embedding matrix.')

# prepare embedding matrix
num_words = min(MAX_NB_WORDS, len(word_index))
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector



embed1 = Sequential()
embed1.add(Embedding(len(word_index) + 1,
                          EMBED_DIM,
                          weights=[embedding_matrix],
                          input_length=MAX_SEQUENCE_LENGTH,
                          trainable=False))
embed2 = Sequential()
embed2.add(Embedding(len(self.word_index) + 1,
                          EMBED_DIM,
                          weights=[embedding_matrix],
                          input_length=MAX_SEQUENCE_LENGTH,
                          trainable=True))

model = Sequential()
model.add(Merge([embed1 ,embed2], mode='concat', concat_axis=-1))
model.add(Reshape((2, MAX_SEQUENCE_LENGTH, EMBED_DIM)))
model.add(Convolution2D(64, 5, EMBED_DIM, activation="relu", border_mode='valid'))
model.add(MaxPooling2D((MAX_SEQUENCE_LENGTH-5+1,1)))
model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(len(nb_labels), activation="softmax"))
model.compile(loss='categorical_crossentropy',
                optimizer='rmsprop',
                metrics=['acc'])
