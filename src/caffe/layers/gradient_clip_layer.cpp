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
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_copy(count, top_diff, bottom_diff);
    const Dtype sumsq_diff = top[0]->sumsq_diff();
    const Dtype l2norm_diff = std::sqrt(sumsq_diff);
    if (l2norm_diff > gradient_clip) {
      caffe_scal(count, gradient_clip / l2norm_diff, bottom_diff);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(GradientClipLayer);
#endif

INSTANTIATE_CLASS(GradientClipLayer);
REGISTER_LAYER_CLASS(GradientClip);

}  // namespace caffe
