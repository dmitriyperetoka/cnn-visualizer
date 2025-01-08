#include <torch/torch.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <format>
#include <unordered_map>
#include <vector>
#include <numeric>
#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"

torch::Tensor bgr_to_rgb(const torch::Tensor& bgr) {
  torch::TensorOptions options = torch::TensorOptions().device(bgr.device()).dtype(bgr.dtype());
  torch::Tensor rgb = torch::zeros(bgr.sizes(), options);
  rgb.select(-3, 0) = bgr.select(-3, 2);
  rgb.select(-3, 1) = bgr.select(-3, 1);
  rgb.select(-3, 2) = bgr.select(-3, 0);
  return rgb;
}

torch::Tensor preprocess_frames(const torch::Tensor& frames, int d_height, int d_width) {
  return bgr_to_rgb(torch::upsample_bilinear2d(frames, {d_height, d_width}, false)).to(torch::kFloat32).div(255);
}

torch::Tensor non_maximum_suppression(const torch::Tensor& boxes, const torch::Tensor& scores, float_t iou_thresh) {
  torch::TensorOptions nms_tensot_options = torch::TensorOptions().device(boxes.device()).dtype(boxes.dtype());
  torch::Tensor iou_threshold = torch::tensor(iou_thresh, nms_tensot_options);
  torch::Tensor x1 = boxes.select(1, 0);
  torch::Tensor y1 = boxes.select(1, 1);
  torch::Tensor x2 = boxes.select(1, 2);
  torch::Tensor y2 = boxes.select(1, 3);

  torch::Tensor areas = (x2 - x1 + 1) * (y2 - y1 + 1);
  torch::Tensor order = scores.argsort().flip({0,});

  std::vector<int> keep_vec;
  for (int ord_len = order.sizes()[0]; ord_len > 0; ord_len = order.sizes()[0]) {
    int i = order[0].item<int>();
    keep_vec.push_back(i);
    torch::Tensor order_after_first = order.slice(0, 1);

    torch::Tensor xx1 = torch::maximum(x1[i], x1.index({order_after_first}));
    torch::Tensor yy1 = torch::maximum(y1[i], y1.index({order_after_first}));
    torch::Tensor xx2 = torch::minimum(x2[i], x2.index({order_after_first}));
    torch::Tensor yy2 = torch::minimum(y2[i], y2.index({order_after_first}));

    torch::Tensor zero_tensor = torch::tensor(0., nms_tensot_options);
    torch::Tensor w = torch::maximum(zero_tensor, xx2 - xx1 + 1);
    torch::Tensor h = torch::maximum(zero_tensor, yy2 - yy1 + 1);
    torch::Tensor inter = w * h;
    torch::Tensor ovr = inter / (areas[i] + areas.index({order_after_first}) - inter);

    torch::Tensor inds = torch::where(ovr <= iou_threshold)[0] + 1;
    order = order.index({inds});
  }
  return torch::tensor(keep_vec);
}

std::vector<int64_t>box_rel2abs(const torch::Tensor& rel_box, int64_t frame_h, int64_t frame_w) {
  std::vector<int64_t> frame_sizes = {frame_w, frame_h, frame_w, frame_h};
  std::vector<int64_t> abs_box = {0, 0, 0, 0};

  for (int64_t i = 0; i < rel_box.sizes()[0]; ++i) {
    int64_t frame_size = frame_sizes[i];
    float_t coord = rel_box[i].item<float_t>();
    if (coord < 0) {
      coord = 0;
    }
    if (coord > 1) {
      coord = 1;
    }
    abs_box[i] = (int64_t)(coord * frame_size);
  }

  if (abs_box[2] == abs_box[0]) {
    if (abs_box[0] == 0) {
      abs_box[2] += 1;
    } else {
      abs_box[0] -= 1;
    }
  }

  if (abs_box[3] == abs_box[1]) {
    if (abs_box[1] == 0) {
      abs_box[3] += 1;
    } else {
      abs_box[1] -=1;
    }
  }
  return abs_box;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> postprocess(
    const torch::Tensor& nn_output,
    const torch::Tensor& classes_to_keep,
    const torch::Tensor& min_score,
    int64_t det_input_h,
    int64_t det_input_w,
    float_t iou_thresh=0.9
) {
  torch::Tensor nn_output_permuted = nn_output.permute({1, 0});
  torch::Tensor boxes = nn_output_permuted.narrow(1, 0, 4);
  std::tuple<torch::Tensor, torch::Tensor> scores_and_classes = nn_output_permuted
    .narrow(1, 4, nn_output_permuted.sizes()[1] - 4)
    .max(1);
  torch::Tensor scores = std::get<0>(scores_and_classes);
  torch::Tensor classes = std::get<1>(scores_and_classes);

  torch::Tensor keep = torch::where(torch::isin(classes, classes_to_keep))[0];
  boxes = boxes.index({keep});
  scores = scores.index({keep});
  classes = classes.index({keep});

  keep = torch::where(scores >= min_score)[0];
  boxes = boxes.index({keep});
  scores = scores.index({keep});
  classes = classes.index({keep});

  torch::Tensor x = boxes.narrow_copy(1, 0, 1);
  torch::Tensor y = boxes.narrow_copy(1, 1, 1);
  torch::Tensor w = boxes.narrow_copy(1, 2, 1);
  torch::Tensor h = boxes.narrow_copy(1, 3, 1);
  boxes.slice(1, 0, 1) = (x - w / 2) / det_input_w;  // top left x
  boxes.slice(1, 1, 2) = (y - h / 2) / det_input_h;  // top left y
  boxes.slice(1, 2, 3) = (x + w / 2) / det_input_w;  // bottom right x
  boxes.slice(1, 3, 4) = (y + h / 2) / det_input_h;  // bottom right y

  keep = non_maximum_suppression(boxes, scores, iou_thresh);
  boxes = boxes.index({keep});
  scores = scores.index({keep});
  classes = classes.index({keep});

  return {boxes, scores, classes};
}


enum class InferMode : int {CPU, CUDA, TENSORRT};


int main(int argc, char *argv[]) {
  try {
    if (argc < 4) {
      std::cerr << "Required args: model_path, input_path, output_path" << std::endl;
      return 1;
    }
    std::string model_path(argv[1]);
    std::string input_path(argv[2]);
    std::string output_path(argv[3]);
    InferMode infer_mode{InferMode::CPU};
    if (argc > 4) {
      std::string infer_mode_arg{argv[4]};
      std::string infer_mode_arg_unified(infer_mode_arg);
      std::transform(infer_mode_arg.begin(), infer_mode_arg.end(), infer_mode_arg_unified.begin(), ::toupper);
      std::unordered_map<std::string, InferMode> infer_mode_map{
        {"CPU", InferMode::CPU},
        {"CUDA", InferMode::CUDA},
        {"TENSORRT", InferMode::TENSORRT}
      };

      auto search = infer_mode_map.find(infer_mode_arg_unified);
      if (search == infer_mode_map.end()) {
        std::vector<std::string> keys(infer_mode_map.size());
        std::transform(infer_mode_map.begin(), infer_mode_map.end(), keys.begin(), [](std::pair<std::string, InferMode> a) {return a.first;});
        std::cerr << std::format(
          "Unknown infer mode: {}. Supported options: {}.\n",
          infer_mode_arg,
          std::accumulate(keys.begin() + 1, keys.end(), *keys.begin(), [](std::string a, std::string b) {return a + ", " + b;})
        );
        return 1;
      }

      infer_mode = search->second;
      if (infer_mode != InferMode::CPU && !torch::cuda::is_available()) {
        std::cerr << "CUDA GPU computation is not available! Please select CPU infer mode or leave the parameter blank.\n";
        return 1;
      }
    }

    Ort::Env env;
    Ort::MemoryInfo memory_info{ 
      (infer_mode == InferMode::CPU)
      ? Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)
      : Ort::MemoryInfo("Cuda", OrtArenaAllocator, /*device_id*/0, OrtMemTypeDefault)};
    Ort::SessionOptions options;

    auto cuda_opts = std::make_unique<OrtCUDAProviderOptions>();
    auto trt_opts = std::make_unique<OrtTensorRTProviderOptions>();
    switch (infer_mode) 
    {
      case InferMode::CUDA:
        options.AppendExecutionProvider_CUDA(*cuda_opts.get());
        break;
      case InferMode::TENSORRT:
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

    std::vector<int64_t> input_shape = {1, 3, 640, 640};
    torch::Tensor input = torch::zeros(input_shape, torch::TensorOptions().dtype(torch::kFloat32));
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input.data_ptr<float>(), input.numel(), input_shape.data(), input_shape.size());

    std::vector<int64_t> output_shape = {1, 5, 8400};
    torch::Tensor output = torch::zeros(output_shape, torch::TensorOptions().dtype(torch::kFloat32));
    Ort::Value output_tensor = Ort::Value::CreateTensor<float>(memory_info, output.data_ptr<float>(), output.numel(), output_shape.data(), output_shape.size());

    // warmup
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

    std::unordered_map<std::string, std::string> codec2nvcodec{
        {"av1", "av1_cuvid"},
        {"h264", "h264_cuvid"},
        {"x264", "h264_cuvid"},
        {"hevc", "hevc_cuvid"},
        {"x265", "hevc_cuvid"},
        {"h265", "hevc_cuvid"},
        {"mjpeg", "mjpeg_cuvid"},
        {"mpeg1video", "mpeg1_cuvid"},
        {"mpeg2video", "mpeg2_cuvid"},
        {"mpeg4", "mpeg4_cuvid"},
        {"vc1", "vc1_cuvid"},
        {"vp8", "vp8_cuvid"},
        {"vp9", "vp9_cuvid"},
    };
    std::string video_decode_hwaccel_info;
    if (torch::cuda::is_available()) {
      std::string nv_codec;
      auto it = codec2nvcodec.cbegin();
      while (it != codec2nvcodec.cend()) {
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
      torch::Tensor input = preprocess_frames(frame.permute({2, 0, 1}).unsqueeze(0), 640, 640);
      Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input.data_ptr<float>(), input.numel(), input_shape.data(), 4);
      session.Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1, output_names.data(), &output_tensor, 1);
      std::tie(boxes, scores, classes) = postprocess(output[0], classes_to_keep, min_score, 640, 640);
      cv::Mat mat = cv::Mat(cv::Size(frame_width, frame_height), CV_8UC3, frame.data_ptr<uchar>());
      for (size_t i = 0; i < boxes.sizes()[0]; ++i) {
        std::vector<int64_t> box = box_rel2abs(boxes[i], frame_height, frame_width);
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
