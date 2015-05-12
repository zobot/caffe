#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void PermuteLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LOG(FATAL) << "Unimplemented";
}

template <typename Dtype>
void PermuteLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  LOG(FATAL) << "Unimplemented";
}

INSTANTIATE_LAYER_GPU_FUNCS(PermuteLayer);


}  // namespace caffe
