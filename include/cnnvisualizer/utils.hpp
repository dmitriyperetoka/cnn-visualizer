#pragma once
#include <torch/torch.h>

namespace cnnv {

std::vector<int64_t>box_rel2abs(const torch::Tensor& rel_box, int64_t frame_h, int64_t frame_w);

} // namespace cnnv
