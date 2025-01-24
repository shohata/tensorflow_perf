import tensorflow as tf


def parse(serialized_example):
    return tf.io.parse_single_example(
        serialized_example,
        features={
            "image/encoded": tf.io.FixedLenFeature([], tf.string),
            "image/class/label": tf.io.FixedLenFeature([], tf.int64),
        },
    )


# Read TFRecord filesj
filenames = [f"/data/imagenet2012/train/train-{i:05}-of-01024" for i in range(0, 1023)]
raw_dataset = tf.data.TFRecordDataset(filenames)
parsed_dataset = raw_dataset.map(parse, num_parallel_calls=tf.data.AUTOTUNE)

id = 0
for record in parsed_dataset:
    dir_id = id // 1000
    image_id = id % 1000
    tf.io.write_file(
        f"/data/imagenet2012/image/image-{dir_id:04}/image-{image_id:04}.jpg",
        record["image/encoded"],
    )
    id += 1
