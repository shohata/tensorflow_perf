import tensorflow as tf
import tensorflow_hub as hub


# Download the model from TensorFlow Hub.
hub_url = "https://www.kaggle.com/models/google/resnet50/frameworks/TensorFlow1/variations/remote-sensing-bigearthnet-resnet50/versions/1"
model = tf.keras.Sequential([hub.KerasLayer(hub_url)])

# Batch input shape.
model.build([None, 224, 224, 3])

# Display the model's architecture.
model.summary()

# Save the entire model as a SavedModel.
model.save("/saved_model/resnet50")
