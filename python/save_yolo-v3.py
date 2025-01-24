import tensorflow as tf

# Using the repository:
#   https://github.com/qqwweee/keras-yolo3
#
# Input Layer:
#   input_1: serving_default_input_1:0
# Output Layer:
#   add_23: StatefulPartitionedCall:0

model = tf.keras.models.load_model("/saved_model/h5/yolo-v3.h5")
model.summary()
model.save("/saved_model/yolo-v3")

model = tf.keras.models.load_model("/saved_model/h5/yolo-v3-darknet53.h5")
model.summary()
model.save("/saved_model/yolo-v3-darknet53")
