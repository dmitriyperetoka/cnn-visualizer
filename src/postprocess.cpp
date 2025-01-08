#include <torch/torch.h>

namespace cnnv {

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

} // namespace cnnv
