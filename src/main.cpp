#include <torch/script.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <cstdlib>

torch::Tensor bgr_to_rgb(torch::Tensor bgr) {
  torch::TensorOptions options = torch::TensorOptions().device(bgr.device()).dtype(bgr.dtype());
  torch::Tensor rgb = torch::zeros(bgr.sizes(), options);
  rgb.select(-3, 0) = bgr.select(-3, 2);
  rgb.select(-3, 1) = bgr.select(-3, 1);
  rgb.select(-3, 2) = bgr.select(-3, 0);
  return rgb;
}

torch::Tensor preprocess_frames(torch::Tensor frames, int d_height, int d_width) {
  return bgr_to_rgb(torch::upsample_bilinear2d(frames, {d_height, d_width}, false)).to(torch::kFloat32).div(255);
}

torch::Tensor non_maximum_suppression(torch::Tensor* boxes, torch::Tensor* scores, float_t iou_thresh) {
  torch::TensorOptions nms_tensot_options = torch::TensorOptions().device(boxes->device()).dtype(boxes->dtype());
  torch::Tensor iou_threshold = torch::tensor(iou_thresh, nms_tensot_options);
  torch::Tensor x1 = boxes->select(1, 0);
  torch::Tensor y1 = boxes->select(1, 1);
  torch::Tensor x2 = boxes->select(1, 2);
  torch::Tensor y2 = boxes->select(1, 3);

  torch::Tensor areas = (x2 - x1 + 1) * (y2 - y1 + 1);
  torch::Tensor order = scores->argsort().flip({0,});

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

std::vector<int64_t>box_rel2abs(torch::Tensor rel_box, int64_t frame_h, int64_t frame_w) {
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
    torch::Tensor nn_output,
    torch::Tensor classes_to_keep,
    torch::Tensor min_score,
    int64_t det_input_h,
    int64_t det_input_w,
    float_t iou_thresh=0.9
) {
  nn_output = nn_output.permute({1, 0});

  torch::Tensor boxes = nn_output.narrow(1, 0, 4);
  std::tuple<torch::Tensor, torch::Tensor> scores_and_classes = nn_output
    .narrow(1, 4, nn_output.sizes()[1] - 4)
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

  keep = non_maximum_suppression(&boxes, &scores, iou_thresh);
  boxes = boxes.index({keep});
  scores = scores.index({keep});
  classes = classes.index({keep});

  return {boxes, scores, classes};
}

int main(int argc, char *argv[]) {
  try {
    std::string model_path(argv[1]);
    std::string input_path(argv[2]);
    std::string output_path(argv[3]);

    Ort::Env env;
    Ort::MemoryInfo memory_info_cuda("Cuda", OrtArenaAllocator, /*device_id*/0, OrtMemTypeDefault);
    Ort::SessionOptions options;
    OrtTensorRTProviderOptions trt_opts;
    trt_opts.trt_max_partition_iterations = 1000;
    trt_opts.trt_min_subgraph_size = 1;
    trt_opts.device_id = 0;
    options.AppendExecutionProvider_TensorRT(trt_opts);
    Ort::Session session(env, model_path.c_str(), options);
    cv::VideoWriter writer(output_path, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30.0, cv::Size(1280, 720));
    std::string cmd{
        "ffmpeg "
        "-loglevel quiet "
        "-hwaccel cuda "
        "-vcodec h264_cuvid "
        "-i "
        + input_path + " "
        "-pix_fmt bgr24 "
        "-f rawvideo "
        "pipe:"
    };
    std::array<char, 2764800> buffer;
    int count;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }

    torch::Tensor classes_to_keep = torch::tensor({0});
    torch::Tensor min_score = torch::tensor(0.65);
    while (not std::feof(pipe.get())) {
        std::fread(buffer.data(), 1, 2764800, pipe.get());
        torch::Tensor frame = torch::from_blob(buffer.data(), {720, 1280, 3}, torch::TensorOptions().dtype(torch::kUInt8));
        Ort::RunOptions run_options;
        torch::Tensor input = preprocess_frames(frame.permute({2, 0, 1}).unsqueeze(0), 640, 640);
        std::array<int64_t, 4> input_shape = {1, 3, 640, 640};
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info_cuda, input.data_ptr<float>(), input.numel(), input_shape.data(), 4);
        torch::Tensor output = torch::zeros({1, 5, 8400}, torch::TensorOptions().dtype(torch::kFloat32));
        std::array<int64_t, 4> output_shape = {1, 5, 8400};
        Ort::Value output_tensor = Ort::Value::CreateTensor<float>(memory_info_cuda, output.data_ptr<float>(), output.numel(), output_shape.data(), 3);
        const char* input_names[] = {"images"};
        const char* output_names[] = {"output0"};
        session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, &output_tensor, 1);
        torch::Tensor boxes;
        torch::Tensor scores;
        torch::Tensor classes;
        std::tie(boxes, scores, classes) = postprocess(output[0], classes_to_keep, min_score, 640, 640);
        //std::cout << boxes << std::endl << scores << std::endl << classes << std::endl << std::endl;

        cv::Mat mat = cv::Mat(cv::Size(1280, 720), CV_8UC3, frame.data_ptr<uchar>());
        for (int i = 0; i < boxes.sizes()[0]; ++i) {
          std::vector<int64_t> box = box_rel2abs(boxes[i], 720, 1280);
          cv::rectangle(mat, cv::Point2d(box[0], box[1]), cv::Point2d(box[2], box[3]), cv::Scalar(0, 0, 225), 2);
        }
        writer.write(mat);
        count++;
    }
    writer.release();
    std::cout << count << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
