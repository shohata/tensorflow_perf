#include <iostream>
#include <iomanip>
#include <sstream>
#include <iterator>
#include <string>
#include <chrono>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <atomic>
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

tensorflow::Status CreateImageProcessingGraph(tensorflow::Session *session, const int32_t batch_size, const int32_t input_height, const int32_t input_width, const int64_t input_channels, const float input_mean, const float input_std)
{
  using namespace tensorflow::ops;

  tensorflow::Scope root = tensorflow::Scope::NewRootScope();

  auto image_size = GuaranteeConst(root.WithOpName("image_size"), {input_height, input_width});
  auto multiples = GuaranteeConst(root.WithOpName("multiples"), {batch_size, 1, 1, 1});
  auto multiples_int32 = Cast(root.WithOpName("cast_multiples"), multiples, tensorflow::DT_INT32);

  auto filename = Placeholder(root.WithOpName(image_input_layer), tensorflow::DT_STRING);
  auto jpeg_image = ReadFile(root.WithOpName("read_file"), filename);
  auto decoded_image = DecodeJpeg(root.WithOpName("decode_jpeg"), jpeg_image, DecodeJpeg::Attrs().Channels(input_channels));
  auto float_image = Cast(root.WithOpName("cast_image"), decoded_image, tensorflow::DT_FLOAT);
  auto expanded_image = ExpandDims(root.WithOpName("expand_dimensions"), float_image, 0);
  auto resized_image = ResizeBilinear(root.WithOpName("resize_image"), expanded_image, image_size);
  auto tiled_image = Tile(root.WithOpName("tile_image"), resized_image, multiples_int32);
  auto output_image = Div(root.WithOpName(image_output_layer), Sub(root.WithOpName("sub"), tiled_image, input_mean), input_std);

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

void ShowResults(const int measurement, const int total_predictions, const int64_t total_msec)
{
  std::cout << "Successfully run the measurement #" << measurement << "." << std::endl;
  std::cout << "Total predictions: " << total_predictions << std::endl;
  std::cout << "Duration time: " << total_msec << " ms" << std::endl;
  std::cout << "Throughput: " << static_cast<float>(total_predictions) / static_cast<float>(total_msec) * 1000.0 << " FPS" << std::endl;
  std::cout << "Latency: " << static_cast<float>(total_msec) / static_cast<float>(total_predictions) << " ms" << std::endl
            << std::endl;
}

class JobManager
{
private:
  std::vector<std::thread> threads;
  std::condition_variable condition;
  std::queue<std::function<void()>> job_queue;
  std::mutex queue_mutex;
  std::atomic<int> job_counter;
  std::atomic<bool> complete_flag;
  std::atomic<bool> stop_flag;

public:
  JobManager(int32_t num_threads);
  ~JobManager();
  void Submit(std::function<void()> job);
  void Wait();

private:
  void Worker();
};

JobManager::JobManager(int32_t num_threads)
{
  job_counter.store(0);
  complete_flag.store(false);
  stop_flag.store(false);

  for (int id = 0; id < num_threads; id++)
  {
    threads.emplace_back([this]
                         { this->Worker(); });
  }
}

JobManager::~JobManager()
{
  stop_flag.store(true);
  condition.notify_all();
  for (auto &thread : threads)
  {
    thread.join();
  }
}

void JobManager::Worker()
{
  while (true)
  {
    std::function<void()> job;

    {
      std::unique_lock<std::mutex> lock(queue_mutex);
      condition.wait(lock, [&]
                     { return !job_queue.empty() || stop_flag; });
      if (stop_flag && job_queue.empty())
      {
        break;
      }

      job = std::move(job_queue.front());
      job_queue.pop();
    }

    job();

    if (job_counter.fetch_sub(1) == 1)
    {
      complete_flag.store(true);
      condition.notify_all();
    }
  }
}

void JobManager::Submit(std::function<void()> job)
{
  {
    std::lock_guard<std::mutex> lock(queue_mutex);
    job_counter.fetch_add(1);
    complete_flag.store(false);

    job_queue.push(job);
  }
  condition.notify_one();
}

void JobManager::Wait()
{
  std::unique_lock<std::mutex> lock(queue_mutex);
  condition.wait(lock, [&]
                 { return complete_flag == true; });
}

int main(int argc, char **argv)
{
  std::string graph_path = "/saved_model/resnet-50";
  int32_t batch_size = 32;
  int32_t num_threads = 4;
  int32_t num_dirs = 1;
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
  status = CreateImageProcessingGraph(image_session, batch_size, input_height, input_width, input_channels, input_mean, input_std);
  if (!status.ok())
  {
    std::cout << "Error creating image processing graph:" << std::endl;
    std::cout << status.message() << std::endl;
    return -1;
  }

  JobManager *job_manager = new JobManager(num_threads);
  std::vector<Tensor> images(num_dirs * num_files);

  // Load Images
  std::cout << "Loading images..." << std::endl;
  for (int32_t i = 0; i < num_dirs; i++)
  {
    for (int32_t j = 0; j < num_files; j++)
    {
      job_manager->Submit([=, &images]
                          {
        std::stringstream image_path;
        image_path << "/data/imagenet2012/image/image-";
        image_path << std::setfill('0') << std::right << std::setw(4) << i;
        image_path << "/image-";
        image_path << std::setfill('0') << std::right << std::setw(4) << j;
        image_path << ".jpg";
        tensorflow::Status image_status = LoadImage(image_session, image_path.str(), &images[i * num_files + j]);
        if (!image_status.ok())
        {
          std::cout << "Error loading images:" << std::endl;
          std::cout << image_status.message() << std::endl;
        } });
    }
  }
  job_manager->Wait();
  delete job_manager;

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
  job_manager = new JobManager(num_threads);
  for (auto &image : images)
  {
    Tensor *input = &image;
    job_manager->Submit([&, input]
                        {
      std::vector<Tensor> outputs;
      status = graph_session->Run({{input_layer, *input}}, {output_layer}, {}, &outputs);
      if (!status.ok())
      {
        std::cout << "Error predicting image:" << std::endl;
        std::cout << status.message() << std::endl;
      } });
  }
  job_manager->Wait();
  delete job_manager;

  std::chrono::system_clock::time_point start, end;
  int64_t total_msec;
  const int total_predictions = images.size() * batch_size;

  // Run Measurements #1
  std::cout << "Running measurements..." << std::endl;
  job_manager = new JobManager(num_threads);
  start = std::chrono::system_clock::now();
  for (auto &image : images)
  {
    Tensor *input = &image;
    job_manager->Submit([&, input]
                        { graph_session->Run({{input_layer, *input}}, {output_layer}, {}, nullptr); });
  }
  job_manager->Wait();
  end = std::chrono::system_clock::now();
  delete job_manager;

  total_msec = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

  // Output the Results
  ShowResults(1, total_predictions, total_msec);

  // Run Measurements #2
  std::cout << "Running measurements..." << std::endl;
  job_manager = new JobManager(1);
  start = std::chrono::system_clock::now();
  for (auto &image : images)
  {
    Tensor *input = &image;
    job_manager->Submit([&, input]
                        { graph_session->Run({{input_layer, *input}}, {output_layer}, {}, nullptr); });
  }
  job_manager->Wait();
  end = std::chrono::system_clock::now();
  delete job_manager;

  total_msec = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

  // Output the Results
  ShowResults(2, total_predictions, total_msec);

  // Run Measurements #3
  std::cout << "Running measurements..." << std::endl;
  start = std::chrono::system_clock::now();
  for (auto &image : images)
  {
    graph_session->Run({{input_layer, image}}, {output_layer}, {}, nullptr);
  }
  end = std::chrono::system_clock::now();

  total_msec = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

  // Output the Results
  ShowResults(3, total_predictions, total_msec);

  return 0;
}
