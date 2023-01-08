import argparse
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
if len(tf.config.list_physical_devices('GPU')) == 0:
    exit()

data = pd.read_csv('reddit.csv',sep=',')
max_words = 5000
max_len = 200

labels = data['is_depression'].values
labels = tf.keras.utils.to_categorical(labels, 3, dtype="float32")
features = data['clean_text'].values
X = []
for i in range(len(features)):
    X.append(str(features[i]))

#Tokenizing data and making them sequences
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X)
sequences = tokenizer.texts_to_sequences(X)
features = pad_sequences(sequences, maxlen=max_len)

#Splitting the data
X_train, X_test, y_train, y_test = train_test_split(features,labels, random_state=0)
print (len(X_train),len(X_test),len(y_train),len(y_test))

# Building the model
model = Sequential()
model.add(layers.Embedding(max_words, 40, input_length=max_len))
model.add(layers.Bidirectional(layers.LSTM(20,dropout=0.6)))
model.add(layers.Dense(3,activation='sigmoid'))
model.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=5,validation_data=(X_test, y_test))

#Validating model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print('Model accuracy: ',test_acc)

def preprocess_texts(text):

    max_words = 5000
    max_len = 200

    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(X)
    sequences = tokenizer.texts_to_sequences(text)
    sequences = pad_sequences(sequences, maxlen=max_len)
    # saving tokenizer
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return sequences

test_text = preprocess_texts(["I want to kill myself"])
print("Length", len(test_text))
labels = ["Not Depressed", "Depressed"]
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
sequence = tokenizer.texts_to_sequences(['I want to kill myself'])
test = pad_sequences(sequence, maxlen=200)
test = model.predict(test)

print(test)
print(labels[np.around(test['predictions'], decimals=0).argmax(axis=1)[0]])
