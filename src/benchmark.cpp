#include <iostream>
#include <iomanip>
#include <sstream>
#include <iterator>
#include <string>
#include <chrono>
#include <vector>
#include <tensorflow/cc/framework/ops.h>
#include <tensorflow/cc/framework/scope.h>
#include <tensorflow/cc/saved_model/loader.h>
#include <tensorflow/cc/saved_model/tag_constants.h>
#include <tensorflow/cc/ops/image_ops.h>
#include <tensorflow/cc/ops/math_ops.h>
#include <tensorflow/cc/ops/array_ops.h>
#include <tensorflow/cc/ops/io_ops.h>
#include <tensorflow/core/util/command_line_flags.h>

using tensorflow::Flag;
using tensorflow::Tensor;

const std::string image_input_layer = "filename";
const std::string image_output_layer = "processed_image";

tensorflow::Status CreateImageProcessingGraph(tensorflow::Session *session, const int32_t input_height, const int32_t input_width, const int64_t input_channels, const float input_mean, const float input_std)
{
  using namespace tensorflow::ops;

  tensorflow::Scope root = tensorflow::Scope::NewRootScope();

  auto filename = Placeholder(root.WithOpName(image_input_layer), tensorflow::DT_STRING);
  auto jpeg_image = ReadFile(root.WithOpName("read_file"), filename);
  auto decoded_image = DecodeJpeg(root.WithOpName("decode_jpeg"), jpeg_image, DecodeJpeg::Attrs().Channels(input_channels));
  auto float_image = Cast(root.WithOpName("cast_image"), decoded_image, tensorflow::DT_FLOAT);
  auto expanded_image = ExpandDims(root.WithOpName("expand_dimensions"), float_image, 0);
  auto resized_image = ResizeBilinear(root.WithOpName("resize_image"), expanded_image, GuaranteeConst(root.WithOpName("size"), {input_height, input_width}));
  auto output_image = Div(root.WithOpName(image_output_layer), Sub(root.WithOpName("sub"), resized_image, input_mean), input_std);

  // Create Image Processing Graph
  tensorflow::GraphDef graph;
  TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));
  TF_RETURN_IF_ERROR(session->Create(graph));
  return ::tensorflow::OkStatus();
}

tensorflow::Status LoadImage(tensorflow::Session *session, const std::string &image_path, Tensor *image_tensor)
{
  std::vector<Tensor> outputs;

  TF_RETURN_IF_ERROR(session->Run({{image_input_layer, Tensor(image_path)}}, {image_output_layer}, {}, &outputs));
  image_tensor->CopyFrom(outputs[0], outputs[0].shape());
  return ::tensorflow::OkStatus();
}

tensorflow::Status LoadSavedModel(const std::string &model_path, tensorflow::SavedModelBundle *bundle, tensorflow::Session **session)
{
  TF_RETURN_IF_ERROR(
      tensorflow::LoadSavedModel(
          tensorflow::SessionOptions(), tensorflow::RunOptions(), model_path,
          {tensorflow::kSavedModelTagServe}, bundle));

  *session = bundle->GetSession();
  auto model_def = bundle->GetSignatures().at("serving_default");

  std::cout << "SavedModel Input Layers:" << std::endl;
  for (const auto &input : model_def.inputs())
  {
    std::cout << "  " << input.first << ": " << input.second.name() << std::endl;
  }
  std::cout << "SavedModel Output Layers:" << std::endl;
  for (const auto &output : model_def.outputs())
  {
    std::cout << "  " << output.first << ": " << output.second.name() << std::endl;
  }

  return ::tensorflow::OkStatus();
}

int main(int argc, char **argv)
{
  std::string graph_path = "/saved_model/resnet-50";
  int32_t batch_size = 32;
  int32_t num_threads = 4;
  int32_t num_dirs = 10;
  int32_t num_files = 1000;
  int32_t input_width = 224;
  int32_t input_height = 224;
  int32_t input_channels = 3;
  float input_mean = 0;
  float input_std = 255;
  std::string input_layer = "serving_default_keras_layer_input:0";
  std::string output_layer = "StatefulPartitionedCall:0";
  std::vector<Flag> flag_list = {
      Flag("graph_path", &graph_path, "graph to be executed"),
      Flag("batch_size", &batch_size, "batch size to be executed"),
      Flag("num_threads", &num_threads, "number of threads to execute"),
      Flag("num_dirs", &num_dirs, "number of directories to load images from"),
      Flag("num_files", &num_files, "number of files to load images from"),
      Flag("input_width", &input_width, "resize image to this width in pixels"),
      Flag("input_height", &input_height, "resize image to this height in pixels"),
      Flag("input_channels", &input_channels, "desired number of color channels for decoded images"),
      Flag("input_mean", &input_mean, "scale pixel values to this mean"),
      Flag("input_std", &input_std, "scale pixel values to this std deviation"),
      Flag("input_layer", &input_layer, "name of input layer"),
      Flag("output_layer", &output_layer, "name of output layer"),
  };
  const std::string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if (!parse_result)
  {
    LOG(ERROR) << usage;
    return -1;
  }

  tensorflow::Status status;
  tensorflow::Session *image_session = tensorflow::NewSession(tensorflow::SessionOptions());

  // Create Image Processing Graph
  std::cout << "Creating image processing graph..." << std::endl;
  status = CreateImageProcessingGraph(image_session, input_height, input_width, input_channels, input_mean, input_std);
  if (!status.ok())
  {
    std::cout << "Error creating image processing graph:" << std::endl;
    std::cout << status.message() << std::endl;
    return -1;
  }

  std::vector<std::thread> image_threads;
  std::vector<tensorflow::Status> image_status(num_dirs);
  std::vector<Tensor> images(num_dirs * num_files);

  // Load Images
  std::cout << "Loading images..." << std::endl;
  for (int32_t id = 0; id < num_dirs; id++)
  {
    image_threads.emplace_back([&, id]
                               {
    for (int32_t i = 0; i < num_files; i++)
    {
      std::stringstream image_path;
      image_path << "/data/imagenet2012/image/image-";
      image_path << std::setfill('0') << std::right << std::setw(4) << id;
      image_path << "/image-";
      image_path << std::setfill('0') << std::right << std::setw(4) << i;
      image_path << ".jpg";
      image_status[id] = LoadImage(image_session, image_path.str(), &images[id * num_files + i]);
      if (!image_status[id].ok()) break;
    } });
  }
  for (auto &thread : image_threads)
  {
    thread.join();
  }
  for (auto &s : image_status)
  {
    if (!s.ok())
    {
      std::cout << "Error loading images:" << std::endl;
      std::cout << s.message() << std::endl;
      return -1;
    }
  }

  const int32_t num_inputs = images.size() / batch_size;
  std::vector<Tensor> inputs(num_inputs, Tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({batch_size, input_height, input_width, input_channels})));

  // Tile the images with batch size
  std::cout << "Tiling images with batch size..." << std::endl;
  for (int32_t i = 0; i < num_inputs; i++)
  {
    for (int32_t j = 0; j < batch_size; j++)
    {
      const auto &image = images[i * batch_size + j].SubSlice(0);
      inputs[i].SubSlice(j).CopyFrom(image, image.shape());
    }
  }

  // Load the SavedModel
  tensorflow::SavedModelBundle bundle;
  tensorflow::Session *graph_session;

  std::cout << "Loading SavedModel..." << std::endl;
  status = LoadSavedModel(graph_path, &bundle, &graph_session);
  if (!status.ok())
  {
    std::cout << "Error loading SavedModel:" << std::endl;
    std::cout << status.message() << std::endl;
    return -1;
  }

  // Warm up the GPU
  std::cout << "Warming up GPU..." << std::endl;
  for (auto &input : inputs)
  {
    std::vector<Tensor> outputs;
    status = graph_session->Run({{input_layer, input}}, {output_layer}, {}, &outputs);
    if (!status.ok())
    {
      std::cout << "Error predicting image:" << std::endl;
      std::cout << status.message() << std::endl;
      return -1;
    }
  }

  std::chrono::system_clock::time_point start, end;
  std::vector<std::thread> threads;
  const int total_predictions = num_threads * num_inputs * batch_size;

  // Run Measurements
  std::cout << "Running measurements..." << std::endl;
  start = std::chrono::system_clock::now();
  for (int id = 0; id < num_threads; id++)
  {
    threads.emplace_back([&]
                         {
    for (auto &input : inputs)
    {
      graph_session->Run({{input_layer, input}}, {output_layer}, {}, nullptr);
    } });
  }

  for (auto &thread : threads)
  {
    thread.join();
  }
  end = std::chrono::system_clock::now();

  int64_t total_msec = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

  // Output the Results
  std::cout << "Successfully run the measurement." << std::endl;
  std::cout << "Total predictions: " << total_predictions << std::endl;
  std::cout << "Duration time: " << total_msec << " ms" << std::endl;
  std::cout << "Throughput: " << static_cast<float>(total_predictions) / static_cast<float>(total_msec) * 1000.0 << " FPS" << std::endl;
  std::cout << "Latency: " << static_cast<float>(total_msec) / static_cast<float>(total_predictions) << " ms" << std::endl;

  return 0;
}
