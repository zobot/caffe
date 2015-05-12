#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void PermuteLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_NE(top[0], bottom[0]) << this->type() << " Layer does not "
    "allow in-place computation.";
}

template <typename Dtype>
void PermuteLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const vector<int> shape = bottom[0]->shape();
  vector<int> newShape;
  newShape.push_back(shape[1]);
  newShape.push_back(shape[0]);
  for (int i = 0; i < bottom[0]->num_axes() - 2; ++i) {
    newShape.push_back(shape[i + 2]);
  }
  top[0]->Reshape(newShape);
}

template <typename Dtype>
void PermuteLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int num0 = bottom[0]->shape(0);
  const int num1 = bottom[0]->shape(1);
  const int endVolume = bottom[0]->count(2);
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  for (int i = 0; i < num0; ++i) {
    for (int j = 0; j < num1; ++j) {
      for (int k = 0; k < endVolume; ++k) {
        const int index_bot = i * (num1 * endVolume) + j * endVolume + k;
        const int index_top = j * (num0 * endVolume) + i * endVolume + k;
        top_data[index_top] = bottom_data[index_bot];
      }
    }
  }
}

template <typename Dtype>
void PermuteLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const int num0 = bottom[0]->shape(0);
  const int num1 = bottom[0]->shape(1);
  const int endVolume = bottom[0]->count(2);
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  for (int i = 0; i < num0; ++i) {
    for (int j = 0; j < num1; ++j) {
      for (int k = 0; k < endVolume; ++k) {
        const int index_bot = i * (num1 * endVolume) + j * endVolume + k;
        const int index_top = j * (num0 * endVolume) + i * endVolume + k;
        bottom_diff[index_bot] = top_diff[index_top];
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(PermuteLayer);
#endif

INSTANTIATE_CLASS(PermuteLayer);
REGISTER_LAYER_CLASS(Permute);

}  // namespace caffe
