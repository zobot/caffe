#ifndef CAFFE_SEQUENCE_LAYERS_HPP_
#define CAFFE_SEQUENCE_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype> class RecurrentLayer;

/**
 * @brief An abstract class for implementing recurrent behavior inside of an
 *        unrolled network.  This Layer type cannot be instantiated -- instaed,
 *        you should use one of its implementations which defines the recurrent
 *        architecture, such as RNNLayer or LSTMLayer.
 */
template <typename Dtype>
class RecurrentLayer : public Layer<Dtype> {
 public:
  explicit RecurrentLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reset();

  virtual inline const char* type() const { return "Recurrent"; }
  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline int MaxBottomBlobs() const { return 3; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

  virtual inline bool AllowForceBackward(const int bottom_index) const {
    // Can't propagate to sequence continuation indicators.
    return bottom_index != 1;
  }

  virtual inline shared_ptr<Net<Dtype> > UnrolledNet(){ return unrolled_net_; }

 protected:
  /**
   * @brief Fills net_param with the recurrent network arcthiecture.  Subclasses
   *        should define this -- see RNNLayer and LSTMLayer for examples.
   */
  virtual void FillUnrolledNet(NetParameter* net_param) const = 0;

  /**
   * @brief Fills names with the names of the 0th timestep recurrent input
   *        Blob&s.  Subclasses should define this -- see RNNLayer and LSTMLayer
   *        for examples.
   */
  virtual void RecurrentInputBlobNames(vector<string>* names) const = 0;

  /**
   * @brief Fills names with the names of the Tth timestep recurrent output
   *        Blob&s.  Subclasses should define this -- see RNNLayer and LSTMLayer
   *        for examples.
   */
  virtual void RecurrentOutputBlobNames(vector<string>* names) const = 0;

  /**
   * @brief Fills names with the names of the output blobs, concatenated across
   *        all timesteps.  Should return a name for each top Blob.
   *        Subclasses should define this -- see RNNLayer and LSTMLayer for
   *        examples.
   */
  virtual void OutputBlobNames(vector<string>* names) const = 0;

  /**
   * @param bottom input Blob vector (length 2-3)
   *
   *   -# @f$ (T \times N \times ...) @f$
   *      the time-varying input @f$ x @f$.  After the first two axes, whose
   *      dimensions must correspond to the number of timesteps @f$ T @f$ and
   *      the number of independent streams @f$ N @f$, respectively, its
   *      dimensions may be arbitrary.  Note that the ordering of dimensions --
   *      @f$ (T \times N \times ...) @f$, rather than
   *      @f$ (N \times T \times ...) @f$ -- means that the @f$ N @f$
   *      independent input streams must be "interleaved".
   *
   *   -# @f$ (T \times N) @f$
   *      the sequence continuation indicators @f$ \delta @f$.
   *      These inputs should be binary (0 or 1) indicators, where
   *      @f$ \delta_{t,n} = 0 @f$ means that timestep @f$ t @f$ of stream
   *      @f$ n @f$ is the beginning of a new sequence, and hence the previous
   *      hidden state @f$ h_{t-1} @f$ is multiplied by @f$ \delta_t = 0 @f$
   *      and has no effect on the cell's output at timestep @f$ t @f$, and
   *      a value of @f$ \delta_{t,n} = 1 @f$ means that timestep @f$ t @f$ of
   *      stream @f$ n @f$ is a continuation from the previous timestep
   *      @f$ t-1 @f$, and the previous hidden state @f$ h_{t-1} @f$ affects the
   *      updated hidden state and output.
   *
   *   -# @f$ (N \times ...) @f$ (optional)
   *      the static (non-time-varying) input @f$ x_{static} @f$.
   *      After the first axis, whose dimension must be the number of
   *      independent streams, its dimensions may be arbitrary.
   *      This is mathematically equivalent to using a time-varying input of
   *      @f$ x'_t = [x_t; x_{static}] @f$ -- i.e., tiling the static input
   *      across the @f$ T @f$ timesteps and concatenating with the time-varying
   *      input.  Note that if this input is used, all timesteps in a single
   *      batch within a particular one of the @f$ N @f$ streams must share the
   *      same static input, even if the sequence continuation indicators
   *      suggest that difference sequences are ending and beginning within a
   *      single batch.  This may require padding and/or truncation for uniform
   *      length.
   *
   * @param top output Blob vector (length 1)
   *   -# @f$ (T \times N \times D) @f$
   *      the time-varying output @f$ y @f$, where @f$ D @f$ is
   *      <code>recurrent_param.num_output()</code>.
   *      Refer to documentation for particular RecurrentLayer implementations
   *      (such as RNNLayer and LSTMLayer) for the definition of @f$ y @f$.
   */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  /// @brief A helper function, useful for stringifying timestep indices.
  virtual string int_to_str(const int t) const;

  /// @brief A Net to implement the Recurrent functionality.
  shared_ptr<Net<Dtype> > unrolled_net_;

  /// @brief The number of independent streams to process simultaneously.
  int N_;

  /**
   * @brief The number of timesteps in the layer's input, and the number of
   *        timesteps over which to backpropagate through time.
   */
  int T_;

  /// @brief Whether the layer has a "static" input copied across all timesteps.
  bool static_input_;

  vector<Blob<Dtype>* > recur_input_blobs_;
  vector<Blob<Dtype>* > recur_output_blobs_;
  vector<Blob<Dtype>* > output_blobs_;
  Blob<Dtype>* x_input_blob_;
  Blob<Dtype>* x_static_input_blob_;
  Blob<Dtype>* cont_input_blob_;
};

/**
 * @brief Processes sequential inputs using a "Long Short-Term Memory" (LSTM)
 *        [1] style recurrent neural network (RNN). Implemented as a network
 *        unrolled the LSTM computation in time.
 *
 *
 * The specific architecture used in this implementation is as described in
 * "Learning to Execute" [2], reproduced below:
 *     i_t := \sigmoid[ W_{hi} * h_{t-1} + W_{xi} * x_t + b_i ]
 *     f_t := \sigmoid[ W_{hf} * h_{t-1} + W_{xf} * x_t + b_f ]
 *     o_t := \sigmoid[ W_{ho} * h_{t-1} + W_{xo} * x_t + b_o ]
 *     g_t :=    \tanh[ W_{hg} * h_{t-1} + W_{xg} * x_t + b_g ]
 *     c_t := (f_t .* c_{t-1}) + (i_t .* g_t)
 *     h_t := o_t .* \tanh[c_t]
 * In the implementation, the i, f, o, and g computations are performed as a
 * single inner product.
 *
 * Notably, this implementation lacks the "diagonal" gates, as used in the
 * LSTM architectures described by Alex Graves [3] and others.
 *
 * [1] Hochreiter, Sepp, and Schmidhuber, Jürgen. "Long short-term memory."
 *     Neural Computation 9, no. 8 (1997): 1735-1780.
 *
 * [2] Zaremba, Wojciech, and Sutskever, Ilya. "Learning to execute."
 *     arXiv preprint arXiv:1410.4615 (2014).
 *
 * [3] Graves, Alex. "Generating sequences with recurrent neural networks."
 *     arXiv preprint arXiv:1308.0850 (2013).
 */
template <typename Dtype>
class LSTMLayer : public RecurrentLayer<Dtype> {
 public:
  explicit LSTMLayer(const LayerParameter& param)
      : RecurrentLayer<Dtype>(param) {}

  virtual inline const char* type() const { return "LSTM"; }

 protected:
  virtual void FillUnrolledNet(NetParameter* net_param) const;
  virtual void RecurrentInputBlobNames(vector<string>* names) const;
  virtual void RecurrentOutputBlobNames(vector<string>* names) const;
  virtual void OutputBlobNames(vector<string>* names) const;
};

/**
 * @brief A helper for LSTMLayer: computes a single timestep of the
 *        non-linearity of the LSTM, producing the updated cell and hidden
 *        states.
 */
template <typename Dtype>
class LSTMUnitLayer : public Layer<Dtype> {
 public:
  explicit LSTMUnitLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "LSTMUnit"; }
  virtual inline int ExactNumBottomBlobs() const { return 3; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

  virtual inline bool AllowForceBackward(const int bottom_index) const {
    // Can't propagate to sequence continuation indicators.
    return bottom_index != 2;
  }

 protected:
  /**
   * @param bottom input Blob vector (length 3)
   *   -# @f$ (1 \times N \times D) @f$
   *      the previous timestep cell state @f$ c_{t-1} @f$
   *   -# @f$ (1 \times N \times 4D) @f$
   *      the "gate inputs" @f$ [i_t', f_t', o_t', g_t'] @f$
   *   -# @f$ (1 \times 1 \times N) @f$
   *      the sequence continuation indicators  @f$ \delta_t @f$
   * @param top output Blob vector (length 2)
   *   -# @f$ (1 \times N \times D) @f$
   *      the updated cell state @f$ c_t @f$, computed as:
   *          i_t := \sigmoid[i_t']
   *          f_t := \sigmoid[f_t']
   *          o_t := \sigmoid[o_t']
   *          g_t := \tanh[g_t']
   *          c_t := cont_t * (f_t .* c_{t-1}) + (i_t .* g_t)
   *   -# @f$ (1 \times N \times D) @f$
   *      the updated hidden state @f$ h_t @f$, computed as:
   *          h_t := o_t .* \tanh[c_t]
   */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  /**
   * @brief Computes the error gradient w.r.t. the LSTMUnit inputs.
   *
   * @param top output Blob vector (length 2), providing the error gradient with
   *        respect to the outputs
   *   -# @f$ (1 \times N \times D) @f$:
   *      containing error gradients @f$ \frac{\partial E}{\partial c_t} @f$
   *      with respect to the updated cell state @f$ c_t @f$
   *   -# @f$ (1 \times N \times D) @f$:
   *      containing error gradients @f$ \frac{\partial E}{\partial h_t} @f$
   *      with respect to the updated cell state @f$ h_t @f$
   * @param propagate_down see Layer::Backward.
   * @param bottom input Blob vector (length 3), into which the error gradients
   *        with respect to the LSTMUnit inputs @f$ c_{t-1} @f$ and the gate
   *        inputs are computed.  Computatation of the error gradients w.r.t.
   *        the sequence indicators is not implemented.
   *   -# @f$ (1 \times N \times D) @f$
   *      the error gradient w.r.t. the previous timestep cell state
   *      @f$ c_{t-1} @f$
   *   -# @f$ (1 \times N \times 4D) @f$
   *      the error gradient w.r.t. the "gate inputs"
   *      @f$ [
   *          \frac{\partial E}{\partial i_t}
   *          \frac{\partial E}{\partial f_t}
   *          \frac{\partial E}{\partial o_t}
   *          \frac{\partial E}{\partial g_t}
   *          ] @f$
   *   -# @f$ (1 \times 1 \times N) @f$
   *      the gradient w.r.t. the sequence continuation indicators
   *      @f$ \delta_t @f$ is currently not computed.
   */
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  /// @brief The hidden and output dimension.
  int hidden_dim_;
  Blob<Dtype> X_acts_;
};

/**
 * @brief Processes time-varying inputs using a simple recurrent neural network
 *        (RNN). Implemented as a network unrolling the RNN computation in time.
 *
 * Given time-varying inputs @f$ x_t @f$, computes hidden state @f$
 *     h_t := \tanh[ W_{hh} h_{t_1} + W_{xh} x_t + b_h ]
 * @f$, and outputs @f$
 *     o_t := \tanh[ W_{ho} h_t + b_o ]
 * @f$.
 */
template <typename Dtype>
class RNNLayer : public RecurrentLayer<Dtype> {
 public:
  explicit RNNLayer(const LayerParameter& param)
      : RecurrentLayer<Dtype>(param) {}

  virtual inline const char* type() const { return "RNN"; }

 protected:
  virtual void FillUnrolledNet(NetParameter* net_param) const;
  virtual void RecurrentInputBlobNames(vector<string>* names) const;
  virtual void RecurrentOutputBlobNames(vector<string>* names) const;
  virtual void OutputBlobNames(vector<string>* names) const;
};

template <typename Dtype>
vector<Blob<Dtype>*> LinearizeSequence(shared_ptr<Net<Dtype> > in_net, const string name, 
        const vector<Dtype*>& inputs, const int num_timesteps){
  LOG(INFO) << "Layer name: " << name;
  LOG(INFO) << "String size: " << name.length();
  LOG(INFO) << "Net name: " << in_net->name();
  vector<string> layer_names = in_net->layer_names();
  LOG(INFO) << "Net layers: ";
  int sequence_index = -1;
  for (int i = 0; i < layer_names.size(); i++){
    if (layer_names[i] == name){
      sequence_index = i;
    }
    LOG(INFO) << layer_names[i];
  }
  LOG(INFO) << "Sequence Layer found at index: " << sequence_index;
  shared_ptr<Layer<Dtype> > sequence_layer = in_net->layers()[sequence_index];

  vector<string> input_blob_names = in_net->blob_names();
  LOG(INFO) << "Input Net blobs: ";
  int x_index = -1;
  for (int i = 0; i < input_blob_names.size(); i++){
    if (input_blob_names[i] == "joint_input"){
      x_index = i;
    }
    LOG(INFO) << input_blob_names[i];
  }
  LOG(INFO) << "x blob found at index: " << x_index;

  shared_ptr<LSTMLayer<Dtype> > lstm_layer = boost::dynamic_pointer_cast<LSTMLayer<Dtype> >(sequence_layer);

  shared_ptr<Net<Dtype> > lstm_net = lstm_layer->UnrolledNet();
  vector<string> unrolled_layer_names = lstm_net->layer_names();
  LOG(INFO) << "Unrolled Net layers: ";
  int hidden_layer_index = -1;
  for (int i = 0; i < unrolled_layer_names.size(); i++){
    if (unrolled_layer_names[i] == "rnn1_unit_1"){
      hidden_layer_index = i;
    }
    LOG(INFO) << unrolled_layer_names[i];
  }
  LOG(INFO) << "Hidden Layer found at index: " << hidden_layer_index;

  vector<string> unrolled_blob_names = lstm_net->blob_names();
  LOG(INFO) << "Unrolled Net blobs: ";
  int c_0_index = -1;
  int c_1_index = -1;
  int h_1_index = -1;
  for (int i = 0; i < unrolled_blob_names.size(); i++){
    if (unrolled_blob_names[i] == "c_0"){
      c_0_index = i;
    }
    if (unrolled_blob_names[i] == "c_1"){
      c_1_index = i;
    }
    if (unrolled_blob_names[i] == "h_1"){
      h_1_index = i;
    }
    LOG(INFO) << unrolled_blob_names[i];
  }

  LOG(INFO) << "c_0 blob found at index: " << c_0_index;
  LOG(INFO) << "c_1 blob found at index: " << c_1_index;
  LOG(INFO) << "h_1 blob found at index: " << h_1_index;

  vector<shared_ptr<Blob<Dtype> > > sequence_net_blobs = lstm_net->blobs();

  const vector<int>& c_0_shape = sequence_net_blobs[c_0_index]->shape();
  const vector<int>& c_1_shape = sequence_net_blobs[c_1_index]->shape();

  stringstream c_0_shape_stringstream;
  c_0_shape_stringstream << "c_0_shape: ";
  for (int i = 0; i < c_0_shape.size(); i++){
      c_0_shape_stringstream << c_0_shape[i];
      if (i < c_0_shape.size() - 1){
        c_0_shape_stringstream << ", ";
      }
  }
  string c_0_shape_string = c_0_shape_stringstream.str();
  LOG(INFO) << c_0_shape_string;

  stringstream c_1_shape_stringstream;
  c_1_shape_stringstream << "c_1_shape: ";
  for (int i = 0; i < c_1_shape.size(); i++){
      c_1_shape_stringstream << c_1_shape[i];
      if (i < c_1_shape.size() - 1){
        c_1_shape_stringstream << ", ";
      }
  }
  string c_1_shape_string = c_1_shape_stringstream.str();
  LOG(INFO) << c_1_shape_string;

  // TODO: make this agnostic to data layer and float as Dtype
  shared_ptr<MemoryDataLayer<float> > md_layer =
    boost::dynamic_pointer_cast<MemoryDataLayer<float> >(in_net->layers()[0]);

  float initial_loss;
  Dtype* input_data = in_net->blobs()[0]->mutable_cpu_data();
  Dtype* c_1_diff = sequence_net_blobs[c_1_index]->mutable_cpu_diff();
  Dtype* h_1_diff = sequence_net_blobs[h_1_index]->mutable_cpu_diff();

  const int x_dim = in_net->blobs()[x_index]->shape()[2];
  const int state_dim = c_0_shape[2];

  LOG(INFO) << "x dim: " << x_dim;
  LOG(INFO) << "Hidden state dim: " << state_dim;
  vector<Dtype*> inputs_timestep;
  inputs_timestep.push_back(inputs[0]);
  inputs_timestep.push_back(inputs[1]);

  Blob<Dtype>* x_jac_blob = new Blob<Dtype>(num_timesteps, state_dim, x_dim, 1);
  Dtype* x_jac_blob_data = x_jac_blob->mutable_cpu_data();

  Blob<Dtype>* hidden_jac_blob = new Blob<Dtype>(num_timesteps, state_dim, state_dim, 1);
  Dtype* hidden_jac_blob_data = hidden_jac_blob->mutable_cpu_data();

  for (int t = 0; t < num_timesteps; t++) {
    inputs_timestep[0] = inputs[0] + x_dim * t;
    inputs_timestep[1] = inputs[1] + t; // one clip per iteration

    //LOG(INFO) << inputs_timestep[0].size();

    md_layer->Reset(inputs_timestep, 1);
    //LOG(INFO) << "Forward for step: " << t;
    in_net->ForwardPrefilled();
    for (int i = 0; i < state_dim; i++) {
      for (int j = 0; j < state_dim; j++) {
        if (i != j) {
          c_1_diff[j] = 0.0;
        }
        else {
          LOG(INFO) << "Setting c_1_diff[" << j << "] = 1.0";
          c_1_diff[j] = 1.0;
        }
      }

      for (int j = 0; j < state_dim; j++) {
        h_1_diff[j] = 0.0;
      }

      //LOG(INFO) << "Backward for step: " << t << " for index: " << i;
      lstm_net->BackwardFrom(hidden_layer_index);
      in_net->BackwardFrom(sequence_index - 1);

      const Dtype* input_diff = in_net->blobs()[x_index]->cpu_diff();
      const Dtype* c_0_diff = sequence_net_blobs[c_0_index]->cpu_diff();

      const int x_offset = t * state_dim * x_dim + i * x_dim;
      const int hidden_offset = t * state_dim * state_dim + i * state_dim;
      for (int j = 0; j < x_dim; j++) {
          x_jac_blob_data[x_offset + j] = input_diff[j];
      }

      for (int j = 0; j < state_dim; j++) {
          hidden_jac_blob_data[hidden_offset + j] = c_0_diff[j];
      }
      //caffe_copy(x_dim, input_diff, x_jac_blob->mutable_cpu_data() + x_offset);
      //caffe_copy(state_dim, c_0_diff, hidden_jac_blob->mutable_cpu_data() + hidden_offset);
    }
  }
  //const vector<Blob<float>*>& output_blobs = net_->ForwardPrefilled(&initial_loss);
  
  stringstream x_jac_shape_stringstream;
  const vector<int> x_jac_shape = x_jac_blob->shape();
  x_jac_shape_stringstream << "x_jac_shape: ";
  for (int i = 0; i < x_jac_shape.size(); i++){
      x_jac_shape_stringstream << x_jac_shape[i];
      if (i < x_jac_shape.size() - 1){
        x_jac_shape_stringstream << ", ";
      }
  }
  string x_jac_shape_string = x_jac_shape_stringstream.str();
  LOG(INFO) << x_jac_shape_string;

  stringstream hidden_jac_shape_stringstream;
  const vector<int> hidden_jac_shape = hidden_jac_blob->shape();
  hidden_jac_shape_stringstream << "hidden_jac_shape: ";
  for (int i = 0; i < hidden_jac_shape.size(); i++){
      hidden_jac_shape_stringstream << hidden_jac_shape[i];
      if (i < hidden_jac_shape.size() - 1){
        hidden_jac_shape_stringstream << ", ";
      }
  }
  string hidden_jac_shape_string = hidden_jac_shape_stringstream.str();
  LOG(INFO) << hidden_jac_shape_string;

  stringstream hidden_jac_stringstream;
  const Dtype* hidden_jac = hidden_jac_blob->cpu_data();
  hidden_jac_stringstream << "hidden_jac: ";
  for (int i = 0; i < hidden_jac_blob->count(); i++){
      hidden_jac_stringstream << hidden_jac[i];
      if (i < hidden_jac_blob->count() - 1){
        hidden_jac_stringstream << ", ";
      }
  }
  string hidden_jac_string = hidden_jac_stringstream.str();
  LOG(INFO) << hidden_jac_string;

  LOG(INFO) << "last hidden jac: " << hidden_jac[hidden_jac_blob->count() - 1];
  //LOG(INFO) << "last hidden jac plus 1: " << hidden_jac[hidden_jac_blob->count()];
  //
  stringstream x_jac_stringstream;
  const Dtype* x_jac = x_jac_blob->cpu_data();
  x_jac_stringstream << "x_jac: ";
  for (int i = 0; i < x_jac_blob->count(); i++){
      x_jac_stringstream << x_jac[i];
      if (i < x_jac_blob->count() - 1){
        x_jac_stringstream << ", ";
      }
  }
  string x_jac_string = x_jac_stringstream.str();
  LOG(INFO) << x_jac_string;

  vector<Blob<Dtype>*> ret_vec;
  ret_vec.push_back(x_jac_blob);
  ret_vec.push_back(hidden_jac_blob);
  return ret_vec;
}

}  // namespace caffe

#endif  // CAFFE_SEQUENCE_LAYERS_HPP_
