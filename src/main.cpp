#include <torch/torch.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <format>
#include <vector>
#include <numeric>
#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>
#include "cnnvisualizer/preprocess.hpp"
#include "cnnvisualizer/postprocess.hpp"
#include "cnnvisualizer/utils.hpp"
#include "cnnvisualizer/constants.hpp"

int main(int argc, char* argv[]) {
  try {
    if (argc < 4) {
      std::cerr << "Required args: model_path, input_path, output_path" << std::endl;
      return 1;
    }
    std::string model_path(argv[1]);
    std::string input_path(argv[2]);
    std::string output_path(argv[3]);
    cnnv::InferMode infer_mode{cnnv::InferMode::CPU};
    if (argc > 4) {
      std::string infer_mode_arg{argv[4]};
      std::string infer_mode_arg_unified(infer_mode_arg);
      std::transform(infer_mode_arg.begin(), infer_mode_arg.end(), infer_mode_arg_unified.begin(), ::toupper);

      auto search = cnnv::STRING_TO_INFER_MODE.find(infer_mode_arg_unified);
      if (search == cnnv::STRING_TO_INFER_MODE.end()) {
        std::vector<std::string> keys(cnnv::STRING_TO_INFER_MODE.size());
        std::transform(
          cnnv::STRING_TO_INFER_MODE.begin(),
          cnnv::STRING_TO_INFER_MODE.end(),
          keys.begin(), [](std::pair<std::string, cnnv::InferMode> a) {return a.first;});
        std::cerr << std::format(
          "Unknown infer mode: {}. Supported options: {}.\n",
          infer_mode_arg,
          std::accumulate(keys.begin() + 1, keys.end(), *keys.begin(), [](std::string a, std::string b) {return a + ", " + b;})
        );
        return 1;
      }

      infer_mode = search->second;
      if (infer_mode != cnnv::InferMode::CPU && !torch::cuda::is_available()) {
        std::cerr << "CUDA GPU computation is not available! Please select CPU infer mode or leave the parameter blank.\n";
        return 1;
      }
    }

    Ort::Env env;
    Ort::MemoryInfo memory_info{ 
      (infer_mode == cnnv::InferMode::CPU)
      ? Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)
      : Ort::MemoryInfo("Cuda", OrtArenaAllocator, /*device_id*/0, OrtMemTypeDefault)};
    Ort::SessionOptions options;

    auto cuda_opts = std::make_unique<OrtCUDAProviderOptions>();
    auto trt_opts = std::make_unique<OrtTensorRTProviderOptions>();
    switch (infer_mode) 
    {
      case cnnv::InferMode::CUDA:
        options.AppendExecutionProvider_CUDA(*cuda_opts.get());
        break;
      case cnnv::InferMode::TENSORRT:
        trt_opts->trt_max_partition_iterations = 1000;
        trt_opts->trt_min_subgraph_size = 1;
        options.AppendExecutionProvider_TensorRT(*trt_opts.get());
        break;
      default:
        break;
    }

    Ort::Session session(env, model_path.c_str(), options);
    Ort::AllocatorWithDefaultOptions ort_alloc;

    size_t input_count = session.GetInputCount();
    std::vector<const char*> input_names(input_count);
    for (size_t i = 0; i < input_count; ++i) {
      input_names[i] = strdup(session.GetInputNameAllocated(i, ort_alloc).get());
    }

    size_t output_count = session.GetOutputCount();
    std::vector<const char*> output_names(output_count);
    for (size_t i = 0; i < output_count; ++i) {
      output_names[i] = strdup(session.GetOutputNameAllocated(i, ort_alloc).get());
    }

    std::vector<int64_t> output_shape = cnnv::DEFAULT_OUTPUT_SHAPE;
    torch::Tensor output = torch::zeros(output_shape, torch::TensorOptions().dtype(torch::kFloat32));
    Ort::Value output_tensor = Ort::Value::CreateTensor<float>(memory_info, output.data_ptr<float>(), output.numel(), output_shape.data(), output_shape.size());

    // warmup
    std::vector<int64_t> input_shape = cnnv::DEFAULT_INPUT_SHAPE;
    torch::Tensor input = torch::zeros(input_shape, torch::TensorOptions().dtype(torch::kFloat32));
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input.data_ptr<float>(), input.numel(), input_shape.data(), input_shape.size());
    session.Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1, output_names.data(), &output_tensor, 1);

    std::string ffprobe_cmd{std::format(
      "ffprobe "
      "-show_streams "
      "-select_streams v "
      "-print_format json "
      "-loglevel error "
      "{}",
      input_path
    )};
    std::unique_ptr<FILE, decltype(&pclose)> ffprobe_pipe(popen(ffprobe_cmd.c_str(), "r"), pclose);
    if (!ffprobe_pipe) throw std::runtime_error("popen() failed!");
    std::stringstream ffprobe_ss;
    std::array<char, 1> ffprobe_buf;
    while (not std::feof(ffprobe_pipe.get())) {
      std::fread(ffprobe_buf.data(), 1, 1, ffprobe_pipe.get());
      ffprobe_ss << ffprobe_buf[0];
    }

    rapidjson::Document d;
    d.Parse(ffprobe_ss.str().c_str());
    rapidjson::Value& stream_info = d["streams"][0];

    std::string codec_name{stream_info["codec_name"].GetString()};
    int frame_width{stream_info["width"].GetInt()};
    int frame_height{stream_info["height"].GetInt()};
    std::string r_frame_rate{stream_info["r_frame_rate"].GetString()};

    int numerator, denominator;
    size_t fraction_pos{r_frame_rate.find('/')};
    std::string ns{r_frame_rate.substr(0, fraction_pos)};
    std::stringstream{ns} >> numerator;
    std::string ds{r_frame_rate.substr(fraction_pos + 1)};
    std::stringstream{ds} >> denominator;
    double frame_rate{static_cast<double>(numerator) / denominator};
    std::string video_decode_hwaccel_info;
    if (torch::cuda::is_available()) {
      std::string nv_codec;
      auto it = cnnv::CODEC_TO_NVCODEC.cbegin();
      while (it != cnnv::CODEC_TO_NVCODEC.cend()) {
        if (it->first == codec_name || it->second == codec_name) {
          nv_codec = it->second;
          break;
        }
        ++it;
      }
      if (!nv_codec.empty()) {
        video_decode_hwaccel_info = std::format("-hwaccel cuda -vcodec {} ", nv_codec);
      }
    }

    std::string ffmpeg_cmd{std::format(
      "ffmpeg "
      "-loglevel error "
      "{}"
      "-i {} "
      "-pix_fmt bgr24 "
      "-f rawvideo "
      "pipe:",
      video_decode_hwaccel_info,
      input_path
    )};
    std::unique_ptr<FILE, decltype(&pclose)> ffmpeg_pipe(popen(ffmpeg_cmd.c_str(), "r"), pclose);
    if (!ffmpeg_pipe) throw std::runtime_error("popen() failed!");

    cv::VideoWriter writer(output_path, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), frame_rate, cv::Size(frame_width, frame_height));

    int count{0};
    int frame_channels{3};
    int buff_size{frame_channels * frame_height * frame_width};
    std::vector<char> buffer(buff_size);
    Ort::RunOptions run_options;
    torch::Tensor classes_to_keep = torch::tensor({0});
    torch::Tensor min_score = torch::tensor(0.65);
    torch::Tensor boxes, scores, classes;
    cv::Scalar cv_color_red{0, 0, 225};
    while (not std::feof(ffmpeg_pipe.get())) {
      std::fread(buffer.data(), 1, buff_size, ffmpeg_pipe.get());
      torch::Tensor frame = torch::from_blob(buffer.data(), {frame_height, frame_width, frame_channels}, torch::TensorOptions().dtype(torch::kUInt8));
      torch::Tensor input = cnnv::preprocess_frames(frame.permute({2, 0, 1}).unsqueeze(0), input_shape[2], input_shape[3]);
      Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input.data_ptr<float>(), input.numel(), input_shape.data(), input_shape.size());
      session.Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1, output_names.data(), &output_tensor, 1);
      std::tie(boxes, scores, classes) = cnnv::postprocess(output[0], classes_to_keep, min_score, input_shape[2], input_shape[3]);
      cv::Mat mat = cv::Mat(cv::Size(frame_width, frame_height), CV_8UC3, frame.data_ptr<uchar>());
      for (size_t i = 0; i < boxes.sizes()[0]; ++i) {
        std::vector<int64_t> box = cnnv::box_rel2abs(boxes[i], frame_height, frame_width);
        cv::rectangle(mat, cv::Point2d(box[0], box[1]), cv::Point2d(box[2], box[3]), cv_color_red, 2);
      }
      writer.write(mat);
      count++;
    }
    writer.release();
    std::cout << std::format("{}", count) << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
