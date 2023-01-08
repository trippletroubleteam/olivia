import os
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import re
import unicodedata
import nltk
from nltk.corpus import stopwords
import keras
from tqdm import tqdm
import pickle
from keras.models import Model
import keras.backend as K
from sklearn.metrics import confusion_matrix,f1_score,classification_report
from keras.callbacks import ModelCheckpoint
import itertools
from keras.models import load_model
from sklearn.utils import shuffle
from transformers import *
from transformers import BertTokenizer, TFBertModel, BertConfig

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5,epsilon=1e-08)
model_save_path='./model2/bert_model.h5'
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
labels = ["Not Depressed", "Depressed"]


#Cleaning text, removing stop words greatly increases accuracy
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def clean_stopwords_shortwords(w):
    stopwords_list=stopwords.words('english')
    words = w.split()
    clean_words = [word for word in words if (word not in stopwords_list) and len(word) > 2]
    return " ".join(clean_words)

def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())
    w = re.sub(r"([?.!,¿])", r" ", w)
    w = re.sub(r'[" "]+', " ", w)
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
    w=clean_stopwords_shortwords(w)
    w=re.sub(r'@\w+', '',w)
    return w


def train():
    data = pd.read_csv("reddit.csv")

    data = data.iloc[2:]

    print(data.head())

    data=data.reset_index(drop=True)                                             # Reset index after dropping the columns/rows with NaN values
    data = shuffle(data)                                                         # Shuffle the dataset
    print('Available labels: ',data.is_depression.unique())                              # Print all the unique labels in the dataset
    data['clean_text']=data['clean_text'].map(preprocess_sentence)
    num_classes=len(data.is_depression.unique())

    bert_model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased',num_labels=num_classes)

    sent= 'how to train the model, lets look at how a trained model calculates its prediction.'
    tokens=bert_tokenizer.tokenize(sent)
    print(tokens)

    tokenized_sequence= bert_tokenizer.encode_plus(sent,add_special_tokens = True,max_length =30,pad_to_max_length = True,
    return_attention_mask = True)

    print(bert_tokenizer.decode(tokenized_sequence['input_ids']))

    sentences = data['clean_text']
    labels = data['is_depression']


    input_ids=[]
    attention_masks=[]

    for sent in sentences:
        bert_inp=bert_tokenizer.encode_plus(sent,add_special_tokens = True,max_length =64,pad_to_max_length = True,return_attention_mask = True)
        input_ids.append(bert_inp['input_ids'])
        attention_masks.append(bert_inp['attention_mask'])

    input_ids=np.asarray(input_ids)
    attention_masks=np.array(attention_masks)
    labels=np.array(labels)

    train_inp,val_inp,train_label,val_label,train_mask,val_mask=train_test_split(input_ids,labels,attention_masks,test_size=0.2)
    print('Train inp shape {} Val input shape {}\nTrain label shape {} Val label shape {}\nTrain attention mask shape {} Val attention mask shape {}'.format(train_inp.shape,val_inp.shape,train_label.shape,val_label.shape,train_mask.shape,val_mask.shape))

    log_dir='tensorboard_data/tb_bert'

    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath=model_save_path,save_weights_only=True,monitor='val_loss',mode='min',save_best_only=True),keras.callbacks.TensorBoard(log_dir=log_dir)]

    print('\nBert Model',bert_model.summary())


    bert_model.compile(loss=loss,optimizer=optimizer,metrics=[metric])

    history=bert_model.fit([train_inp,train_mask],train_label,batch_size=32,epochs=4,validation_data=([val_inp,val_mask],val_label),callbacks=callbacks)


def load_model():
    trained_model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased',num_labels=2)
    trained_model.compile(loss=loss,optimizer=optimizer, metrics=[metric])
    trained_model.load_weights(model_save_path)
    return trained_model


def test(trained_model, sentence):
    #sentence = "Today was a great day, I love my friends and my family and everyone around me"
    sentence = preprocess_sentence(sentence)
    sentence = bert_tokenizer.encode_plus(sentence, add_special_tokens=True, max_length=256, padding="max_length", return_tensors="tf")
    #preprocessed_sentence = np.array(preprocessed_sentence).reshape(1, -1)

    # Create the attention mask
    #attention_mask = np.ones((1, 64))

    # Make the prediction
    prediction = trained_model.predict([tf.cast(sentence.input_ids, tf.int64), tf.cast(sentence.attention_mask, tf.int64)])
    # prediction = np.array(prediction[0])
    # index = prediction.argmax(axis=1)[0]
    # print(index)
    # print(prediction)
    # return labels[index], prediction[0][1] * 100
    score = max(0, prediction[0][0][1] * 100)
    if score > 200:
        score = 200

    return score

    
if __name__ == "__main__":
    #train()
     model = load_model()
     while True:
         text = input('Enter Sentence: ')
         print(test(model , text))