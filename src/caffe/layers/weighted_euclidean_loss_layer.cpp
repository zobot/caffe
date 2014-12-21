#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

// bottom[0] is the output of the network, bottom[1] is the output_torque,
// bottom[2] is the precision matrix * lambda_t of weights for each time
template <typename Dtype>
void WeightedEuclideanLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());
  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[0]->width(), 1);
  CHECK_EQ(bottom[2]->width(), 1);
  CHECK_EQ(bottom[2]->num(), bottom[0]->num());
  CHECK_EQ(bottom[2]->channels(), bottom[2]->height());
  CHECK_EQ(bottom[2]->height(), bottom[0]->channels());
  diff_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  temp_.Reshape(bottom[0]->num(), bottom[2]->channels(), 1, 1);
}

template <typename Dtype>
void WeightedEuclideanLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  int channels = bottom[2]->channels();
  int height = bottom[2]->height();
  Dtype loss(0.0), dot(0.0), inner_dot(0.0);
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());
  for (int i = 0; i < bottom[0]->num(); ++i) {
    dot = 0.0;
    for (int row = 0; row < channels; ++row) {
      inner_dot = 0.0;
      for (int k = 0; k < height; ++k) {
        inner_dot += bottom[2]->data_at(i, row, k, 0) * diff_.data_at(i, k, 0, 0);
      }
      dot += diff_.data_at(i, row, 0, 0) * inner_dot;
    }
    loss += dot;
  }
  top[0]->mutable_cpu_data()[0] = loss / bottom[0]->num() / Dtype(2);
}

template <typename Dtype>
void WeightedEuclideanLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1. : -1.;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();

      for (int n = 0; n < bottom[i]->num(); ++n) {
        for (int row = 0; row < bottom[2]->channels(); ++row) {
          Dtype temp = 0;
          for (int k = 0; k < bottom[2]->height(); ++k) {
            temp += bottom[2]->data_at(n, row, k, 0) * diff_.data_at(n, k, 0, 0);
          }
          *(bottom_diff + bottom[i]->offset(n, row)) = temp;
        }
      }
      caffe_scal(bottom[i]->count(), alpha, bottom_diff);
    }
  }
}

INSTANTIATE_CLASS(WeightedEuclideanLossLayer);
REGISTER_LAYER_CLASS(WEIGHTED_EUCLIDEAN_LOSS, WeightedEuclideanLossLayer);
}  // namespace caffe
