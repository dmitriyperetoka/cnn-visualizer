#pragma once
#include <torch/torch.h>

namespace cnnv {

torch::Tensor preprocess_frames(const torch::Tensor& frames, int d_height, int d_width);

} // namespace cnnv
