#include <vector>

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
	int num = bottom[0]->num();
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());
  caffe_cpu_gemv<Dtype>(CblasNoTrans, channels, height, 0.5, bottom[2]->cpu_data(),
			                  diff_.cpu_data(), 0., temp_.mutable_cpu_data());
	//Dtype dot = caffe_cpu_dot(num, diff_.cpu_data(), temp_.cpu_data());
	// Dot product and add over time steps
	Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), temp_.cpu_data());
  Dtype loss = dot / num / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void WeightedEuclideanLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      // const Dtype sign = (i == 0) ? 1. : -1.;
      const Dtype alpha = top[0]->cpu_diff()[0] / bottom[i]->num();
      caffe_cpu_gemv<Dtype>(CblasNoTrans, bottom[2]->channels(), bottom[2]->height(),
                            alpha, bottom[2]->cpu_data(), diff_.cpu_data(), 0.,
                            bottom[i]->mutable_cpu_data());
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(WeightedEuclideanLossLayer);
#endif

INSTANTIATE_CLASS(WeightedEuclideanLossLayer);
REGISTER_LAYER_CLASS(WEIGHTED_EUCLIDEAN_LOSS, WeightedEuclideanLossLayer);
}  // namespace caffe
