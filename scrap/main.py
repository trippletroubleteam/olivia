import tflite_support

# Initialization
classifier = tflite_support.task.text_classifier.TextClassifier("model.tflite")

# Run inference
text = "I love this life im so happy!"
text_classification_result = classifier.classify(text)
print(
    text_classification_result.classifications[0].categories[0].score * 100,
    "percent not depressed",
)
print(
    text_classification_result.classifications[0].categories[1].score * 100,
    "percent depressed",
)
