from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf

model = tf.keras.models.load_model("model")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

print("here")
pred_sentences = ['I love my life. Nothing could make me happier than waking up every morning and seeing the sun shine.',
                  'I hate everything. I want to end myself every moment of everyday. Violence is the only thing that fills my head and I want to shoot up the school.']



tf_batch = tokenizer(pred_sentences, max_length=128, padding=True, truncation=True, return_tensors='tf')

tf_outputs = model(tf_batch)
tf_predictions = tf.nn.softmax(tf_outputs[0], axis=-1)
labels = ['Not Depressed','Depressed']
label = tf.argmax(tf_predictions, axis=1)
label = label.numpy()
for i in range(len(pred_sentences)):
  print(pred_sentences[i], ": \n", labels[label[i]])
