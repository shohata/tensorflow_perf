import tensorflow as tf
import tensorflow_hub as hub

# Download the model from TensorFlow Hub.
hub_url = "https://www.kaggle.com/models/google/resnet-v2/frameworks/TensorFlow2/variations/101-classification/versions/2"
model = tf.keras.Sequential([hub.KerasLayer(hub_url)])

# Batch input shape.
model.build([None, 224, 224, 3])  # Batch input shape.

# Display the model's architecture.
model.summary()

# Save the entire model as a SavedModel.
model.save("/saved_model/resnet-v2")
