#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void EuclideanDistanceLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
      << "The data and label should have the same number.";
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());
  top[0]->Reshape(1, 1, 1, 1);
}

template <typename Dtype>
void EuclideanDistanceLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int count = bottom[0]->count();
  const int num = bottom[0]->num();
  const int point_dim = 3;
  const int num_points = bottom[0]->channels() / point_dim;
  // TODO: Add check that bottom[0]->channels() is a multiple of point_dim.
  Blob<Dtype> diff_(num, bottom[0]->channels(), 1, 1);
  // diff_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());
  caffe_sqr(count, diff_.cpu_data(), diff_.mutable_cpu_data());
  Dtype total_dist(0.0), sum;
  for (int i = 0; i < num; ++i) {
    // Compute distance for this num
    for (int j = 0; j < num_points; ++j) {

      int point_offset = j * point_dim;
      sum = Dtype(0.0);
      for (int k = 0; k < point_dim; ++k) {
        sum += diff_.data_at(i, point_offset + k, 0, 0);
      }
      total_dist += sqrt(sum);
    }
  }

  top[0]->mutable_cpu_data()[0] = total_dist / (num * num_points);
  // Euclidean distance layer should not be used as a loss function.
}

INSTANTIATE_CLASS(EuclideanDistanceLayer);
REGISTER_LAYER_CLASS(EUCLIDEAN_DISTANCE, EuclideanDistanceLayer);
}  // namespace caffe
