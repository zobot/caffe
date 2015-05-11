#include <vector>

#include "caffe/layer.hpp"
#include "caffe/neuron_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void GradientClipLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  gradient_clip = this->layer_param_.gradient_clip_param().gradient_clip();
  batch_axis = this->layer_param_.gradient_clip_param().batch_axis();
  CHECK_NE(top[0], bottom[0]) << this->type() << " Layer does not "
    "allow in-place computation.";
}

template <typename Dtype>
void GradientClipLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int count = top[0]->count();
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_copy(count, bottom[0]->cpu_data(), top_data);
}

template <typename Dtype>
void GradientClipLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const int count = top[0]->count();
  const int startVolume = bottom[0]->count(0, batch_axis);
  const int numBatch = bottom[0]->shape(batch_axis);
  const int endVolume = bottom[0]->count(batch_axis + 1);
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_copy(count, top_diff, bottom_diff);
    for (int i = 0; i < numBatch; ++i) {
      Dtype sumsq_i = 0;
      for (int j = 0; j < startVolume; ++j) {
        for (int k = 0; k < endVolume; ++k) {
          const int index = i * endVolume + j * endVolume * numBatch + k;
          sumsq_i += top_diff[index] * top_diff[index];
        }
      }
      const Dtype norm = std::sqrt(sumsq_i);
      if (norm > gradient_clip) {
        const Dtype scale_factor = gradient_clip / norm;
        for (int j = 0; j < startVolume; ++j) {
          for (int k = 0; k < endVolume; ++k) {
            const int index = i * endVolume + j * endVolume * numBatch + k;
            bottom_diff[index] = scale_factor * top_diff[index];
          }
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(GradientClipLayer);
#endif

INSTANTIATE_CLASS(GradientClipLayer);
REGISTER_LAYER_CLASS(GradientClip);

}  // namespace caffe
