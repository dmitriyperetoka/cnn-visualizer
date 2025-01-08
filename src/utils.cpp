#include <torch/torch.h>

namespace cnnv {

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

} // namespace cnnv
