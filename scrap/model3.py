import numpy as np
import os

from tflite_model_maker import model_spec
from tflite_model_maker import text_classifier
from tflite_model_maker.config import ExportFormat
from tflite_model_maker.text_classifier import AverageWordVecSpec
from tflite_model_maker.text_classifier import DataLoader

import tensorflow as tf
assert tf.__version__.startswith('2')
tf.get_logger().setLevel('ERROR')


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
if len(tf.config.list_physical_devices('GPU')) == 0:
    exit()

import pandas as pd

def replace_label(original_file, new_file):
  # Load the original file to pandas. We need to specify the separator as
  # '\t' as the training data is stored in TSV format
  df = pd.read_csv(original_file)

  # Define how we want to change the label name
  label_map = {0: 'negative', 1: 'positive'}

  # Excute the label change
  df.replace({'is_depression': label_map}, inplace=True)

  # Write the updated dataset to a new file
  df.to_csv(new_file)

# Replace the label name for both the training and test dataset. Then write the
# updated CSV dataset to the current folder.
replace_label(os.path.join(os.path.join(data_dir, 'train.tsv')), 'train.csv')
replace_label(os.path.join(os.path.join(data_dir, 'dev.tsv')), 'dev.csv')

spec = model_spec.get('average_word_vec')

train_data = DataLoader.from_csv(
      filename='train.csv',
      text_column='clean_text',
      label_column='is_depression',
      model_spec=spec,
      is_training=True)
test_data = DataLoader.from_csv(
      filename='dev.csv',
      text_column='clean_text',
      label_column='is_depression',
      model_spec=spec,
      is_training=False)

model = text_classifier.create(train_data, model_spec=spec, epochs=100, batch_size=128)

loss, acc = model.evaluate(test_data)

model.export(export_dir='average_word_vec')

classifier = text.NLClassifier.create_from_file("model.tflite")
tes = """
I love my life right now, I made new friends and I just cannot wait to wake up again tommorow morning

"""
# Run inference
text_classification_result = classifier.classify(tes)

print(text_classification_result.classifications[0].categories[0].score)

print(text_classification_result.classifications[0].categories[0].score*100, "percent not depressed")
print(text_classification_result.classifications[0].categories[1].score*100, "percent depressed")
