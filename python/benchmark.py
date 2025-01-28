import tensorflow as tf
import tensorflow_hub as hub
import time
from concurrent.futures import ThreadPoolExecutor
import sys

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <model_dir>")
    exit(1)

model_dir = sys.argv[1]

# Read TFRecord files
filenames = [f"/data/imagenet2012/train/train-{i:05}-of-01024" for i in range(0, 1)]
raw_image_dataset = tf.data.TFRecordDataset(filenames)

# Create a dictionary describing the features.
image_feature_description = {
    "image/encoded": tf.io.FixedLenFeature([], tf.string),
    "image/class/label": tf.io.FixedLenFeature([], tf.int64),
}


# Parse the input tf.train.Example proto using the dictionary above.
def _parse_image_function(example_proto):
    parsed_example = tf.io.parse_single_example(
        example_proto, image_feature_description
    )
    image = tf.image.decode_jpeg(parsed_example["image/encoded"], channels=3)
    image = tf.image.resize(image, [224, 224])
    image = tf.expand_dims(image, axis=0)
    image = tf.tile(image, tf.constant([32, 1, 1, 1], tf.int32))
    label = tf.cast(parsed_example["image/class/label"], tf.int64) - 1  # [0-999]
    return image, label


print("Loading images...")
parsed_image_dataset = raw_image_dataset.map(
    _parse_image_function, num_parallel_calls=tf.data.AUTOTUNE
)
num_images = len(list(parsed_image_dataset)) * 32
print(f"Number of images: {num_images}")

# Load the saved model
print("Loading SavedModel...")
model = tf.keras.saving.load_model(model_dir)

# Display the model's architecture.
model.summary()

# Save the entire model as a SavedModel.
# model.save("saved_model/resnet_50")

# Warming Up
print("Warming up GPU...")
for image, label in parsed_image_dataset:
    for i in range(100):
        model.predict(image)
    break

# Create instance for prediction
tpe = ThreadPoolExecutor(max_workers=4)

# Inferences
print("Running measurements...")
start = time.perf_counter_ns()
for image, label in parsed_image_dataset:
    tpe.submit(model, image, training=False)
tpe.shutdown(wait=True)
end = time.perf_counter_ns()

duration = (end - start) / (1000 * 1000)
throughput = num_images / duration * 1000
latency = duration / num_images

print("Successfully run the measurement.")
print(f"Total predictions:  {num_images:5}")
print(f"Duration time:      {duration:9.3f} ms")
print(f"Throughput:         {throughput:9.3f} FPS")
print(f"Latency:            {latency:9.3f} ms")
