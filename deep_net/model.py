import keras
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Activation
from keras.layers import Embedding
from keras.callbacks import ModelCheckpoint
import os
import sklearn.metrics
from sklearn.metrics import roc_auc_score
import pandas as pd
import matplotlib.pyplot as plt
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


output_dir = 'model_output/imdb_deep_net'
epochs = 4
batch_size = 128

n_dim = 64
n_unique_words = 5000
n_words_to_skip = 50
max_review_length = 100
pad_type = 'pre'
trunc_type = 'pre'

n_dense = 64
dropout = 0.5

(x_train, y_train), (x_valid, y_valid) = imdb.load_data(
	num_words=n_unique_words,
	skip_top=n_words_to_skip
	)

word_index = keras.datasets.imdb.get_word_index()

# v+3 so we push the words 3 positions.
word_index = {k : (v+3) for k,v in word_index.items()}
# Now we fill in some keywords for the first 3 indexes as seen below.
word_index['PAD'] = 0
word_index['START'] = 1
word_index['UNK'] = 2

index_word = {v: k for k, v in word_index.items()}

review = ' '.join(index_word[id] for id in x_train[0])

#Preprocess -- ensure that all reviews are of the same length
x_train = pad_sequences(
	x_train,
	maxlen=max_review_length,
	padding=pad_type,
	truncating=trunc_type, 
	value=0
	)

x_valid = pad_sequences(
	x_valid,
	maxlen=max_review_length,
	padding=pad_type,
	truncating=trunc_type,
	value=0
	)

model = Sequential()
model.add(Embedding(
	n_unique_words,
	n_dim,
	input_length=max_review_length
	))
model.add(Flatten())
model.add(Dense(
	n_dense,
	activation='relu'
	))
model.add(Dropout(dropout))
model.add(Dense(
	1,
	activation='sigmoid'
	))
model.summary()

modelcheckpoint = ModelCheckpoint(
	filepath=os.path.join(os.getcwd(), output_dir, 'weights{epoch:02d}.hdf5')
	)
if not os.path.exists(output_dir):
	os.makedirs(output_dir)


#Compile and Run
model.compile(
	loss='binary_crossentropy',
	optimizer='adam',
	metrics=['accuracy']
	)

model.fit(
	x_train, 
	y_train, 
	batch_size=batch_size,
	epochs=epochs,
	verbose=1,
	validation_split=0.2,
	callbacks=[modelcheckpoint]
	)


