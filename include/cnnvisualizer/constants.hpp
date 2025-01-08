#pragma once
#include <unordered_map>
#include <string>
#include <vector>

namespace cnnv {

enum class InferMode : int {CPU, CUDA, TENSORRT};
const std::unordered_map<std::string, InferMode> STRING_TO_INFER_MODE{
  {"CPU", InferMode::CPU},
  {"CUDA", InferMode::CUDA},
  {"TENSORRT", InferMode::TENSORRT}
};

const int64_t DEFAULT_BATCH_SIZE{1};

const int64_t DEFAULT_INPUT_CHANNELS_CNT{3};
const int64_t DEFAULT_INPUT_HEIGHT{640};
const int64_t DEFAULT_INPUT_WIDTH{640};
const std::vector<int64_t> DEFAULT_INPUT_SHAPE{
  DEFAULT_BATCH_SIZE,
  DEFAULT_INPUT_CHANNELS_CNT,
  DEFAULT_INPUT_HEIGHT,
  DEFAULT_INPUT_WIDTH
};

const int64_t DEFAULT_OUTPUT_FEATURES_CNT{5};
const int64_t DEFAULT_OUTPUT_DETECTIONS_CNT{8400};
const std::vector<int64_t> DEFAULT_OUTPUT_SHAPE{
  DEFAULT_BATCH_SIZE,
  DEFAULT_OUTPUT_FEATURES_CNT,
  DEFAULT_OUTPUT_DETECTIONS_CNT
};

const std::unordered_map<std::string, std::string> CODEC_TO_NVCODEC{
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

} // namespace cnnv
