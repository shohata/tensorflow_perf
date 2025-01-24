import tensorflow as tf
import tensorflow_hub as hub


# Download the model from TensorFlow Hub.
hub_url = "https://www.kaggle.com/models/tensorflow/resnet-50/frameworks/TensorFlow2/variations/classification/versions/1"
model = tf.keras.Sequential([hub.KerasLayer(hub_url)])

# Batch input shape.
model.build([None, 224, 224, 3])

# Display the model's architecture.
model.summary()

# Save the entire model as a SavedModel.
model.save("/saved_model/resnet-50")
