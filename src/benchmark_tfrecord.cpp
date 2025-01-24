#include <iostream>
#include <fstream>
#include <iterator>
#include <string>
#include <chrono>
#include <vector>
#include <tensorflow/core/example/example.pb.h>
#include <tensorflow/cc/framework/ops.h>
#include <tensorflow/cc/framework/scope.h>
#include <tensorflow/cc/saved_model/loader.h>
#include <tensorflow/cc/saved_model/tag_constants.h>
#include <tensorflow/cc/ops/image_ops.h>
#include <tensorflow/cc/ops/math_ops.h>
#include <tensorflow/cc/ops/array_ops.h>
#include <tensorflow/cc/ops/io_ops.h>
#include <tensorflow/cc/ops/parsing_ops.h>

int main(int argc, char **argv)
{
  tensorflow::Status status;

  // Load the TFRecord dataset
  std::vector<tensorflow::Example> examples;
  try
  {
    const std::string record_path("/data/imagenet2012/train/train-00000-of-01024");
    std::ifstream input_file(record_path, std::ios::in | std::ios::binary);

    if (!input_file)
    {
      std::cerr << "Could not open file: " << record_path << std::endl;
      return 1;
    }

    // repeatedly Read length
    uint64_t length = 0;
    uint64_t max_len = 1000;
    char *data_buffer = new char[max_len];

    while (input_file.read(reinterpret_cast<char *>(&length), 8))
    {
      if (length > max_len)
      {
        delete[] data_buffer;
        max_len = 2 * length;
        data_buffer = new char[max_len];
      }

      // the following fields: crc of length(4 bytes), data(length bytes), crc of data(4 bytes)
      // we directly read in (length+8) bytes, and skip the head/tail
      input_file.read(data_buffer, length + 8);

      // tensorflow::Example example;
      tensorflow::Example example;
      if (!example.ParseFromArray(reinterpret_cast<void *>(data_buffer + 4), length))
      {
        std::cerr << "Failed to parse file." << std::endl;
        return -1;
      }
      else
      {
        examples.push_back(example);
      }
    }

    delete[] data_buffer;

    std::cout << "Successfully load the record." << std::endl;
    std::cout << "Total examples: " << examples.size() << std::endl;
  }
  catch (const std::exception &e)
  {
    std::cerr << "Failed to open input file." << std::endl;
    std::cerr << e.what() << std::endl;
    return 1;
  }

  // Use the loaded TFRecord dataset
  std::vector<tensorflow::Tensor> dataset;
  std::vector<int64_t> labels;
  try
  {
    std::vector<tensorflow::Tensor> inputs;
    const std::string input_name = "input";
    const std::string output_name = "preprocessed";

    for (auto &example : examples)
    {

      const std::string &byte_image = example.features().feature().at("image/encoded").bytes_list().value(0);
      const int64_t label = example.features().feature().at("image/class/label").int64_list().value(0);

      auto input = tensorflow::Tensor(tensorflow::DT_STRING, tensorflow::TensorShape());
      input.scalar<tensorflow::tstring>()() = byte_image;
      inputs.push_back(input);
      labels.push_back(label);
    }

    const int32_t input_height = 224;
    const int32_t input_width = 224;
    const int64_t input_channels = 3;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope();

    // Use a placeholder to read input data
    auto raw_image = tensorflow::ops::Placeholder(root.WithOpName(input_name), tensorflow::DataType::DT_STRING);
    auto decode_image = tensorflow::ops::DecodeJpeg(root.WithOpName("decode_jpeg"), raw_image, tensorflow::ops::DecodeJpeg::Attrs().Channels(input_channels));
    auto expand_image = tensorflow::ops::ExpandDims(root.WithOpName("expand_dimension"), decode_image, 0);
    auto resize_image = tensorflow::ops::ResizeBilinear(root.WithOpName("resize_image"), expand_image, tensorflow::ops::GuaranteeConst(root.WithOpName("size"), {input_height, input_width}));
    auto cast_image = tensorflow::ops::Cast(root.WithOpName(output_name), resize_image, tensorflow::DT_FLOAT);

    tensorflow::GraphDef graph;
    status = root.ToGraphDef(&graph);
    if (!status.ok())
    {
      std::cerr << "Error creating graph" << std::endl;
      std::cerr << status.message() << std::endl;
      return 1;
    }

    // Run the TFRecord dataset
    std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));

    std::cout << "Running the TFRecord dataset." << std::endl;
    std::cout << "Input node name: " << input_name << std::endl;
    std::cout << "Output node name: " << output_name << std::endl;

    status = session->Create(graph);
    if (!status.ok())
    {
      std::cerr << "Failed to create session." << std::endl;
      std::cerr << status.message() << std::endl;
      return 1;
    }

    for (auto &input : inputs)
    {
      std::vector<tensorflow::Tensor> outputs;
      status = session->Run({{input_name, input}}, {output_name}, {}, &outputs);
      if (!status.ok())
      {
        std::cerr << "Failed to run session." << std::endl;
        std::cerr << status.message() << std::endl;
        return 1;
      }
      dataset.push_back(outputs[0]);
    }

    std::cout << "Successfully run the TFRecord dataset." << std::endl;
    std::cout << "Total outputs: " << dataset.size() << std::endl;
  }
  catch (const std::exception &e)
  {
    std::cerr << "Failed to run the TFRecord dataset." << std::endl;
    std::cerr << e.what() << std::endl;
    return 1;
  }

  // Load the Tensorflow model
  std::vector<tensorflow::Tensor> results;
  try
  {
    // Use the loaded Tensorflow model
    const std::string model_path("/saved_model/resnet_50");
    tensorflow::SavedModelBundle bundle;
    status = tensorflow::LoadSavedModel(
        tensorflow::SessionOptions(), tensorflow::RunOptions(), model_path,
        {tensorflow::kSavedModelTagServe}, &bundle);
    if (!status.ok())
    {
      std::cerr << "Failed to load the Tensorflow model." << std::endl;
      std::cerr << status.message() << std::endl;
      return 1;
    }

    auto sig_map = bundle.GetSignatures();
    auto model_def = sig_map.at("serving_default");
    const std::string input_name = model_def.inputs().at("keras_layer_input").name();
    const std::string output_name = model_def.outputs().at("keras_layer").name();

    std::cout << "Show model input node:" << std::endl;
    for (auto &p : model_def.inputs())
    {
      std::cout << "key: " << p.first << ", value: " << p.second.name() << std::endl;
    }

    std::cout << "Show model output node:" << std::endl;
    for (auto &p : model_def.outputs())
    {
      std::cout << "key: " << p.first << ", value: " << p.second.name() << std::endl;
    }

    // Run the Tensorflow model
    tensorflow::Session *session = bundle.GetSession();

    std::cout << "Running the Tensorflow model." << std::endl;
    std::cout << "Input node name: " << input_name << std::endl;
    std::cout << "Output node name: " << output_name << std::endl;

    // Warm up the GPU
    for (auto &data : dataset)
    {
      std::vector<tensorflow::Tensor> outputs;
      status = session->Run({{input_name, data}}, {output_name}, {}, &outputs);
      if (!status.ok())
      {
        std::cerr << "Failed to run the Tensorflow model." << std::endl;
        std::cerr << status.message() << std::endl;
        return 1;
      }
      results.push_back(outputs[0]);
    }

    // Set Number of iterations
    const int num_threads = 4;
    const int num_iterations = 5;
    const int total_predictions = num_threads * num_iterations * dataset.size();
    std::vector<tensorflow::Tensor> outputs[num_threads];
    std::vector<std::thread> threads;
    std::chrono::system_clock::time_point start, end;

    // Run Measurements
    start = std::chrono::system_clock::now();
    for (int id = 0; id < num_threads; id++)
    {
      threads.emplace_back([&, id]
                           {
                             for (int i = 0; i < num_iterations; i++)
                             {
                               for (auto &data : dataset)
                               {
                                 status = session->Run({{input_name, data}}, {output_name}, {}, &outputs[id]);
                               }
                             } });
    }
    for (auto &thread : threads)
    {
      thread.join();
    }
    end = std::chrono::system_clock::now();

    int64_t total_msec = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "Successfully run the Tensorflow model." << std::endl;
    std::cout << "Total predictions: " << total_predictions << std::endl;
    std::cout << "Duration time: " << total_msec << " ms" << std::endl;
    std::cout << "Throughput: " << static_cast<float>(total_predictions) / static_cast<float>(total_msec) * 1000.0 << " FPS" << std::endl;
    std::cout << "Latency: " << static_cast<float>(total_msec) / static_cast<float>(total_predictions) << " ms" << std::endl;
  }
  catch (const std::exception &e)
  {
    std::cerr << "Failed to run the Tensorflow model." << std::endl;
    std::cerr << e.what() << std::endl;
    return 1;
  }

  return 0;
}
