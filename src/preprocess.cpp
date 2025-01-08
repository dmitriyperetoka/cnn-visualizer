#include <torch/torch.h>

namespace cnnv {

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

} // namespace cnnv
