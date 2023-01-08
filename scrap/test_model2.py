import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf

X_test = "I want to kill myself"
model = tf.saved_model.load("model2")

def preprocess_texts(text):

    max_words = 5000
    max_len = 200

    tokenizer = Tokenizer(num_words=max_words)
    #tokenizer.fit_on_texts(X)
    sequences = tokenizer.texts_to_sequences(text)
    sequences = pad_sequences(sequences, maxlen=max_len)
    # saving tokenizer
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return sequences

test_text = preprocess_texts(X_test)
print(len(test_text))

predictions = model.predict(test_text)
