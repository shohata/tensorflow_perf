# TensorFlow Perf

GPU Performance Monitor using TensorFlow by C++.
This application requires NVIDIA Container Toolkit.

## ML Model

This application can run the benchmark of the following ML models.

-   [resnet-50](https://www.kaggle.com/models/tensorflow/resnet-50/frameworks/TensorFlow2/variations/classification/versions/1)
-   [resnet50](https://www.kaggle.com/models/google/resnet50/frameworks/TensorFlow1/variations/remote-sensing-bigearthnet-resnet50/versions/1)
-   [resnet-v2](https://www.kaggle.com/models/google/resnet-v2/frameworks/TensorFlow2/variations/101-classification/versions/2)
-   [yolo-v3](https://github.com/qqwweee/keras-yolo3)

Download the models from TensorFlow Hub.

```bash
cd ./python
python3 save_resnet-50.py
python3 save_resnet50.py
python3 save_resnet-v2.py
python3 save_yolo-v3.py
```

## Build

For building the benchmark, enter the Docker environment.

```bash
# Build the docker image which contains the TensorFlow C++ SDK
make build
# Enter the Docker environment
make run
```

In the Docker environment, build the benchmark.

```bash
# Create the build directory
cmake -G Ninja -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
# Build the benchmark
cmake --build build
```

## Run the benchmarks

Run the benchmark.

```bash
./build/src/benchmark
```

Following options are available.

-   `graph_path` : graph to be executed (`"/saved_model/resnet-50"`)
-   `batch_size` : batch size to be executed (`32`)
-   `num_threads` : number of threads to execute (`4`)
-   `num_dirs` : number of directories to load images from (`10`)
-   `num_files` : number of files to load images from (`1000`)
-   `input_width` : resize image to this width in pixels (`224`)
-   `input_height` resize image to this height in pixels (`224`)
-   `input_channels` : desired number of color channels for decoded images (`3`)
-   `input_mean` : scale pixel values to this mean (`0`)
-   `input_std` : scale pixel values to this std deviation (`255`)
-   `input_layer` : name of input layer (`"serving_default_keras_layer_input:0"`)
-   `output_layer` : name of output layer (`"StatefulPartitionedCall:0"`)

You can get the following output.

```
Total predictions: 10016
Duration time: 26271 ms
Throughput: 381.257 FPS
Latency: 2.6229 ms
```
