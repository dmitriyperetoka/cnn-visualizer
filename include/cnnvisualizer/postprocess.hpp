#pragma once
#include <torch/torch.h>

namespace cnnv {

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> postprocess(
    const torch::Tensor& nn_output,
    const torch::Tensor& classes_to_keep,
    const torch::Tensor& min_score,
    int64_t det_input_h,
    int64_t det_input_w,
    float_t iou_thresh=0.9
);

} // namespace cnnv
