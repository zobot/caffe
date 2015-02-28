//
// matcaffe.cpp provides a wrapper of the caffe::Net class as well as some
// caffe::Caffe functions so that one could easily call it from matlab.
// Note that for matlab, we will simply use float as the data type.

#include <sstream>
#include <string>
#include <vector>

#include "mex.h"

#include "google/protobuf/text_format.h"

#include "caffe/caffe.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/util/upgrade_proto.hpp"

#define MEX_ARGS int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs

// Log and throw a Mex error
inline void mex_error(const std::string &msg) {
  LOG(ERROR) << msg;
  mexErrMsgTxt(msg.c_str());
}

using namespace caffe;  // NOLINT(build/namespaces)

// The pointer to the internal caffe::Net instance
static shared_ptr<Net<float> > net_;
static shared_ptr<Solver<float> > solver_;
static int init_key = -2;

// Five things to be aware of:
//   caffe uses row-major order
//   matlab uses column-major order
//   caffe uses BGR color channel order
//   matlab uses RGB color channel order
//   images need to have the data mean subtracted
//
// Data coming in from matlab needs to be in the order
//   [width, height, channels, images]
// where width is the fastest dimension.
// Here is the rough matlab for putting image data into the correct
// format:
//   % convert from uint8 to single
//   im = single(im);
//   % reshape to a fixed size (e.g., 240x240)
//   im = imresize(im, [IMAGE_DIM IMAGE_DIM], 'bilinear');
//   % permute from RGB to BGR and subtract the data mean (already in BGR)
//   im = im(:,:,[3 2 1]) - data_mean;
//   % flip width and height to make width the fastest dimension
//   im = permute(im, [2 1 3]);
//
// If you have multiple images, cat them with cat(4, ...)
//
// The actual forward function. It takes in a cell array of 4-D arrays as
// input and outputs a cell array.

static mxArray* do_forward(const mxArray* const bottom) {
  const vector<Blob<float>*>& input_blobs = net_->input_blobs();
  if (static_cast<unsigned int>(mxGetDimensions(bottom)[0]) !=
      input_blobs.size()) {
    mex_error("Invalid input size");
  }
  for (unsigned int i = 0; i < input_blobs.size(); ++i) {
    const mxArray* const elem = mxGetCell(bottom, i);
    if (!mxIsSingle(elem)) {
      mex_error("MatCaffe require single-precision float point data");
    }
    if (mxGetNumberOfElements(elem) != input_blobs[i]->count()) {
      std::string error_msg;
      error_msg += "MatCaffe input size does not match the input size ";
      error_msg += "of the network";
      mex_error(error_msg);
    }

    const float* const data_ptr =
        reinterpret_cast<const float* const>(mxGetPr(elem));
    switch (Caffe::mode()) {
    case Caffe::CPU:
      caffe_copy(input_blobs[i]->count(), data_ptr,
          input_blobs[i]->mutable_cpu_data());
      break;
    case Caffe::GPU:
      caffe_copy(input_blobs[i]->count(), data_ptr,
          input_blobs[i]->mutable_gpu_data());
      break;
    default:
      mex_error("Unknown Caffe mode");
    }  // switch (Caffe::mode())
  }
  const vector<Blob<float>*>& output_blobs = net_->ForwardPrefilled();
  mxArray* mx_out = mxCreateCellMatrix(output_blobs.size(), 1);
  for (unsigned int i = 0; i < output_blobs.size(); ++i) {
    // internally data is stored as (width, height, channels, num)
    // where width is the fastest dimension
    mwSize dims[4] = {output_blobs[i]->width(), output_blobs[i]->height(),
      output_blobs[i]->channels(), output_blobs[i]->num()};
    mxArray* mx_blob =  mxCreateNumericArray(4, dims, mxSINGLE_CLASS, mxREAL);
    mxSetCell(mx_out, i, mx_blob);
    float* data_ptr = reinterpret_cast<float*>(mxGetPr(mx_blob));
    switch (Caffe::mode()) {
    case Caffe::CPU:
      caffe_copy(output_blobs[i]->count(), output_blobs[i]->cpu_data(),
          data_ptr);
      break;
    case Caffe::GPU:
      caffe_copy(output_blobs[i]->count(), output_blobs[i]->gpu_data(),
          data_ptr);
      break;
    default:
      mex_error("Unknown Caffe mode");
    }  // switch (Caffe::mode())
  }

  return mx_out;
}

// Input is a cell array of 4 4-D arrays containing image and joint info
static void vgps_train(const mxArray* const bottom) {
  vector<shared_ptr<Blob<float> > > input_blobs;
  input_blobs.resize(4);

  const mxArray* const rgb = mxGetCell(bottom, 0);
  const float* const rgb_ptr = reinterpret_cast<const float* const>(mxGetPr(rgb));
  const mxArray* const joint = mxGetCell(bottom, 1);
  const float* const joint_ptr = reinterpret_cast<const float* const>(mxGetPr(joint));
  const mxArray* const action = mxGetCell(bottom, 2);
  const float* const action_ptr = reinterpret_cast<const float* const>(mxGetPr(action));
  const mxArray* const prec = mxGetCell(bottom, 3);
  const float* const prec_ptr = reinterpret_cast<const float* const>(mxGetPr(prec));
  CHECK(mxIsSingle(rgb))
      << "MatCaffe require single-precision float point data";
  CHECK(mxIsSingle(joint))
      << "MatCaffe require single-precision float point data";

  const int num_samples = mxGetDimensions(action)[1];
  int channels = 1;
  int height = 1;
  int width = 1;
  if (mxGetNumberOfDimensions(rgb) == 4) {
    channels = mxGetDimensions(rgb)[2];
    height = mxGetDimensions(rgb)[1];
    width = mxGetDimensions(rgb)[0];
  }
  else {
    channels = mxGetDimensions(rgb)[0];
  }
  const int dX = mxGetDimensions(joint)[0];
  const int dU = mxGetDimensions(action)[0];
  // TODO - add check for dimensions from memory data layer dimensions.
  //CHECK_EQ(channels, 3) << "Channel dimension incorrect";
  //CHECK_EQ(height, 240) << "Image height dimension incorrect";
  //CHECK_EQ(dX, 21) << "Joint state dimension incorrect: " << dX;
  CHECK_EQ(dU, 7) << "Action dimension incorrect: " << dU;

  input_blobs[0] = shared_ptr<Blob<float> >(new Blob<float>());
  input_blobs[1] = shared_ptr<Blob<float> >(new Blob<float>());
  input_blobs[2] = shared_ptr<Blob<float> >(new Blob<float>());
  input_blobs[3] = shared_ptr<Blob<float> >(new Blob<float>());

  LOG(INFO) << "Image has size: " << width << " x " << height << " x " << channels;
  input_blobs[0]->Reshape(num_samples, channels, height, width);
  input_blobs[1]->Reshape(num_samples, dX, 1, 1);
  input_blobs[2]->Reshape(num_samples, dU, 1, 1);
  input_blobs[3]->Reshape(num_samples, dU, dU, 1);

  caffe_copy(input_blobs[0]->count(), rgb_ptr, input_blobs[0]->mutable_cpu_data());
  caffe_copy(input_blobs[1]->count(), joint_ptr, input_blobs[1]->mutable_cpu_data());
  caffe_copy(input_blobs[2]->count(), action_ptr, input_blobs[2]->mutable_cpu_data());
  caffe_copy(input_blobs[3]->count(), prec_ptr, input_blobs[3]->mutable_cpu_data());

  shared_ptr<MemoryDataLayer<float> > md_layer =
    boost::dynamic_pointer_cast<MemoryDataLayer<float> >(solver_->net()->layers()[0]);
  md_layer->Reset(input_blobs[0]->mutable_cpu_data(),
                  input_blobs[1]->mutable_cpu_data(),
                  input_blobs[2]->mutable_cpu_data(),
                  input_blobs[3]->mutable_cpu_data(), num_samples);

  LOG(INFO) << "Starting Solve";
  solver_->Solve();
}

// Input is a cell array of 4 4-D arrays containing image and joint info
static void vgps_trainb(const mxArray* const bottom) {
  vector<shared_ptr<Blob<float> > > input_blobs;
  input_blobs.resize(2);

  const mxArray* const rgb = mxGetCell(bottom, 0);
  const float* const rgb_ptr = reinterpret_cast<const float* const>(mxGetPr(rgb));
  const mxArray* const joint = mxGetCell(bottom, 1);
  const float* const joint_ptr = reinterpret_cast<const float* const>(mxGetPr(joint));
  const mxArray* const action = mxGetCell(bottom, 2);
  const float* const action_ptr = reinterpret_cast<const float* const>(mxGetPr(action));
  const mxArray* const prec = mxGetCell(bottom, 3);
  const float* const prec_ptr = reinterpret_cast<const float* const>(mxGetPr(prec));
  CHECK(mxIsSingle(rgb))
      << "MatCaffe require single-precision float point data";
  CHECK(mxIsSingle(joint))
      << "MatCaffe require single-precision float point data";

  const int num_samples = mxGetDimensions(rgb)[3];
  const int channels = mxGetDimensions(rgb)[2];
  const int dX = mxGetDimensions(joint)[0];
  const int dU = mxGetDimensions(action)[0];
  //CHECK_EQ(dX, 21) << "Joint state dimension incorrect: " << dX;
  CHECK_EQ(dU, 7) << "Action dimension incorrect: " << dU;

  input_blobs[0] = shared_ptr<Blob<float> >(new Blob<float>());
  input_blobs[1] = shared_ptr<Blob<float> >(new Blob<float>());
  input_blobs[2] = shared_ptr<Blob<float> >(new Blob<float>());
  input_blobs[3] = shared_ptr<Blob<float> >(new Blob<float>());

  input_blobs[0]->Reshape(num_samples, channels, 1, 1);
  input_blobs[1]->Reshape(num_samples, dX, 1, 1);
  input_blobs[2]->Reshape(num_samples, dU, 1, 1);
  input_blobs[3]->Reshape(num_samples, dU, dU, 1);

  caffe_copy(input_blobs[0]->count(), rgb_ptr, input_blobs[0]->mutable_cpu_data());
  caffe_copy(input_blobs[1]->count(), joint_ptr, input_blobs[1]->mutable_cpu_data());
  caffe_copy(input_blobs[2]->count(), action_ptr, input_blobs[2]->mutable_cpu_data());
  caffe_copy(input_blobs[3]->count(), prec_ptr, input_blobs[3]->mutable_cpu_data());

  shared_ptr<MemoryDataLayer<float> > md_layer =
    boost::dynamic_pointer_cast<MemoryDataLayer<float> >(solver_->net()->layers()[0]);
  md_layer->Reset(input_blobs[0]->mutable_cpu_data(),
                  input_blobs[1]->mutable_cpu_data(),
                  input_blobs[2]->mutable_cpu_data(),
                  input_blobs[3]->mutable_cpu_data(), num_samples);

  LOG(INFO) << "Starting Solve";
  solver_->Solve();
}

// Input is a cell array of 4 4-D arrays containing image and joint info
static mxArray* vgps_forward(const mxArray* const bottom) {
  vector<shared_ptr<Blob<float> > > input_blobs;
  input_blobs.resize(4);

  const mxArray* const rgb = mxGetCell(bottom, 0);
  const float* const rgb_ptr = reinterpret_cast<const float* const>(mxGetPr(rgb));
  const mxArray* const joint = mxGetCell(bottom, 1);
  const float* const joint_ptr = reinterpret_cast<const float* const>(mxGetPr(joint));
  const mxArray* const action = mxGetCell(bottom, 2);
  const float* const action_ptr = reinterpret_cast<const float* const>(mxGetPr(action));
  const mxArray* const prec = mxGetCell(bottom, 3);
  const float* const prec_ptr = reinterpret_cast<const float* const>(mxGetPr(prec));
  CHECK(mxIsSingle(rgb))
      << "MatCaffe require single-precision float point data";
  CHECK(mxIsSingle(joint))
      << "MatCaffe require single-precision float point data";

  const int num_samples = mxGetDimensions(rgb)[3];
  const int channels = mxGetDimensions(rgb)[2];
  const int height = mxGetDimensions(rgb)[0];
  const int width = mxGetDimensions(rgb)[0];
  const int dX = mxGetDimensions(joint)[0];
  const int dU = mxGetDimensions(action)[0];
  CHECK_EQ(channels, 3) << "Channel dimension incorrect";
  CHECK_EQ(height, 240) << "Image height dimension incorrect";
  //CHECK_EQ(dX, 21) << "Joint state dimension incorrect: " << dX;
  CHECK_EQ(dU, 7) << "Action dimension incorrect: " << dU;

  input_blobs[0] = shared_ptr<Blob<float> >(new Blob<float>());
  input_blobs[1] = shared_ptr<Blob<float> >(new Blob<float>());
  input_blobs[2] = shared_ptr<Blob<float> >(new Blob<float>());
  input_blobs[3] = shared_ptr<Blob<float> >(new Blob<float>());

  input_blobs[0]->Reshape(num_samples, channels, height, width);
  input_blobs[1]->Reshape(num_samples, dX, 1, 1);
  input_blobs[2]->Reshape(num_samples, dU, 1, 1);
  input_blobs[3]->Reshape(num_samples, dU, dU, 1);

  caffe_copy(input_blobs[0]->count(), rgb_ptr, input_blobs[0]->mutable_cpu_data());
  caffe_copy(input_blobs[1]->count(), joint_ptr, input_blobs[1]->mutable_cpu_data());
  caffe_copy(input_blobs[2]->count(), action_ptr, input_blobs[2]->mutable_cpu_data());
  caffe_copy(input_blobs[3]->count(), prec_ptr, input_blobs[3]->mutable_cpu_data());

  shared_ptr<MemoryDataLayer<float> > md_layer =
    boost::dynamic_pointer_cast<MemoryDataLayer<float> >(net_->layers()[0]);
  md_layer->Reset(input_blobs[0]->mutable_cpu_data(),
                  input_blobs[1]->mutable_cpu_data(),
                  input_blobs[2]->mutable_cpu_data(),
                  input_blobs[3]->mutable_cpu_data(), num_samples);

  float initial_loss;
  LOG(INFO) << "Running forward pass";
  const vector<Blob<float>*>& output_blobs = net_->ForwardPrefilled(&initial_loss);
  LOG(INFO) << "Initial loss: " << initial_loss;

  // output of fc is the second output blob.
  mxArray* mx_out = mxCreateCellMatrix(1, 1);
  mwSize dims[4] = {output_blobs[1]->width(), output_blobs[1]->height(),
    output_blobs[1]->channels(), output_blobs[1]->num()};
  mxArray* mx_blob =  mxCreateNumericArray(4, dims, mxSINGLE_CLASS, mxREAL);
  mxSetCell(mx_out, 0, mx_blob);
  float* data_ptr = reinterpret_cast<float*>(mxGetPr(mx_blob));
  switch (Caffe::mode()) {
  case Caffe::CPU:
    caffe_copy(output_blobs[1]->count(), output_blobs[1]->cpu_data(),
        data_ptr);
    break;
  case Caffe::GPU:
    caffe_copy(output_blobs[1]->count(), output_blobs[1]->gpu_data(),
        data_ptr);
    break;
  default:
    LOG(FATAL) << "Unknown Caffe mode.";
  }  // switch (Caffe::mode())

  return mx_out;
}

// Input is a cell array of 2 4-D arrays containing image and joint info
static mxArray* vgps_forwarda_only(const mxArray* const bottom) {
  vector<shared_ptr<Blob<float> > > input_blobs;
  input_blobs.resize(1);

  const mxArray* const rgb = mxGetCell(bottom, 0);
  const float* const rgb_ptr = reinterpret_cast<const float* const>(mxGetPr(rgb));
  CHECK(mxIsSingle(rgb))
      << "MatCaffe require single-precision float point data";

  const int num_samples = mxGetDimensions(rgb)[3];
  const int channels = mxGetDimensions(rgb)[2];
  const int height = mxGetDimensions(rgb)[0];
  const int width = mxGetDimensions(rgb)[0];
  CHECK_EQ(channels, 3);
  CHECK_EQ(height, 240);

  input_blobs[0] = shared_ptr<Blob<float> >(new Blob<float>());

  input_blobs[0]->Reshape(num_samples, channels, height, width);

  caffe_copy(input_blobs[0]->count(), rgb_ptr, input_blobs[0]->mutable_cpu_data());

  shared_ptr<MemoryDataLayer<float> > md_layer =
    boost::dynamic_pointer_cast<MemoryDataLayer<float> >(net_->layers()[0]);
  md_layer->Reset(input_blobs[0]->mutable_cpu_data(),
                  num_samples);

  float initial_loss;
  LOG(INFO) << "Running forward pass";
  const vector<Blob<float>*>& output_blobs = net_->ForwardPrefilled(&initial_loss);
  CHECK_EQ(output_blobs.size(), 1);

  // output of fc is the only output blob.
  mxArray* mx_out = mxCreateCellMatrix(1, 1);
  mwSize dims[4] = {output_blobs[0]->width(), output_blobs[0]->height(),
    output_blobs[0]->channels(), output_blobs[0]->num()};
  mxArray* mx_blob =  mxCreateNumericArray(4, dims, mxSINGLE_CLASS, mxREAL);
  mxSetCell(mx_out, 0, mx_blob);
  float* data_ptr = reinterpret_cast<float*>(mxGetPr(mx_blob));
  switch (Caffe::mode()) {
  case Caffe::CPU:
    caffe_copy(output_blobs[0]->count(), output_blobs[0]->cpu_data(),
        data_ptr);
    break;
  case Caffe::GPU:
    caffe_copy(output_blobs[0]->count(), output_blobs[0]->gpu_data(),
        data_ptr);
    break;
  default:
    LOG(FATAL) << "Unknown Caffe mode.";
  }  // switch (Caffe::mode())

  return mx_out;
}
// Input is a cell array of 2 4-D arrays containing image and joint info
static mxArray* vgps_forward_only(const mxArray* const bottom) {
  vector<shared_ptr<Blob<float> > > input_blobs;
  input_blobs.resize(2);

  const mxArray* const rgb = mxGetCell(bottom, 0);
  const float* const rgb_ptr = reinterpret_cast<const float* const>(mxGetPr(rgb));
  const mxArray* const joint = mxGetCell(bottom, 1);
  const float* const joint_ptr = reinterpret_cast<const float* const>(mxGetPr(joint));
  CHECK(mxIsSingle(rgb))
      << "MatCaffe require single-precision float point data";
  CHECK(mxIsSingle(joint))
      << "MatCaffe require single-precision float point data";

  const int num_samples = mxGetDimensions(rgb)[3];
  const int channels = mxGetDimensions(rgb)[2];
  const int height = mxGetDimensions(rgb)[0];
  const int width = mxGetDimensions(rgb)[0];
  const int dX = mxGetDimensions(joint)[0];
  CHECK_EQ(channels, 3);
  CHECK_EQ(height, 240);
  //CHECK_EQ(dX, 21);

  input_blobs[0] = shared_ptr<Blob<float> >(new Blob<float>());
  input_blobs[1] = shared_ptr<Blob<float> >(new Blob<float>());

  input_blobs[0]->Reshape(num_samples, channels, height, width);
  input_blobs[1]->Reshape(num_samples, dX, 1, 1);

  caffe_copy(input_blobs[0]->count(), rgb_ptr, input_blobs[0]->mutable_cpu_data());
  caffe_copy(input_blobs[1]->count(), joint_ptr, input_blobs[1]->mutable_cpu_data());

  shared_ptr<MemoryDataLayer<float> > md_layer =
    boost::dynamic_pointer_cast<MemoryDataLayer<float> >(net_->layers()[0]);
  md_layer->Reset(input_blobs[0]->mutable_cpu_data(),
                  input_blobs[1]->mutable_cpu_data(),
                  num_samples);

  float initial_loss;
  LOG(INFO) << "Running forward pass";
  const vector<Blob<float>*>& output_blobs = net_->ForwardPrefilled(&initial_loss);
  CHECK_EQ(output_blobs.size(), 1);

  // output of fc is the only output blob.
  mxArray* mx_out = mxCreateCellMatrix(1, 1);
  mwSize dims[4] = {output_blobs[0]->width(), output_blobs[0]->height(),
    output_blobs[0]->channels(), output_blobs[0]->num()};
  mxArray* mx_blob =  mxCreateNumericArray(4, dims, mxSINGLE_CLASS, mxREAL);
  mxSetCell(mx_out, 0, mx_blob);
  float* data_ptr = reinterpret_cast<float*>(mxGetPr(mx_blob));
  switch (Caffe::mode()) {
  case Caffe::CPU:
    caffe_copy(output_blobs[0]->count(), output_blobs[0]->cpu_data(),
        data_ptr);
    break;
  case Caffe::GPU:
    caffe_copy(output_blobs[0]->count(), output_blobs[0]->gpu_data(),
        data_ptr);
    break;
  default:
    LOG(FATAL) << "Unknown Caffe mode.";
  }  // switch (Caffe::mode())

  return mx_out;
}

// Returns cell array of protobuf string of the weights. *MUST BE CALLED AFTER TRAIN*
static mxArray* get_weights_string() {
  NetParameter net_param;
  net_->ToProto(&net_param, false);
  vector<int> to_remove;
  for (int i = 0; i < net_param.layers_size(); ++i) {
    const LayerParameter& layer_param = net_param.layers(i);
    if (layer_param.type() != LayerParameter::INNER_PRODUCT) continue;
    const FillerParameter& filler_param = layer_param.inner_product_param().weight_filler();
    if (filler_param.type() == "imagexy") to_remove.push_back(i);
  }
  for (int i = to_remove.size()-1; i >= 0; --i) {
    int r = to_remove[i];
    // swap the element to the end and then remove it.
    for (int j = r+1; j < net_param.layers_size(); ++j) {
      net_param.mutable_layers()->SwapElements(j-1, j);
    }
    net_param.mutable_layers()->RemoveLast();
  }

  string proto_string;
  google::protobuf::TextFormat::PrintToString(net_param, &proto_string);
  mxArray* mx_out = mxCreateCellMatrix(1, 1);
  mwSize dims[1] = {proto_string.length()};
  mxArray* mx_proto_string =  mxCreateCharArray(1, dims);
  mxSetCell(mx_out, 0, mx_proto_string);
  char* data_ptr = reinterpret_cast<char*>(mxGetPr(mx_proto_string));
  strcpy(data_ptr, proto_string.c_str());
  return mx_out;
}

static void save_weights_to_file(const char* const filename) {

  NetParameter net_param;
  net_->ToProto(&net_param, false);
  vector<int> to_remove;
  for (int i = 0; i < net_param.layers_size(); ++i) {
    const LayerParameter& layer_param = net_param.layers(i);
    if (layer_param.type() != LayerParameter::INNER_PRODUCT) continue;
    const FillerParameter& filler_param = layer_param.inner_product_param().weight_filler();
    if (filler_param.type() == "imagexy") to_remove.push_back(i);
  }
  for (int i = 0; i < to_remove.size(); ++i) {
    int r = to_remove[i];
    // swap the element to the end and then remove it.
    for (int j = r+1; j < net_param.layers_size(); ++j) {
      net_param.mutable_layers()->SwapElements(j-1, j);
    }
    net_param.mutable_layers()->RemoveLast();
  }

  WriteProtoToBinaryFile(net_param, filename);
}

static void set_weights_from_string(const mxArray* const proto_string) {
  const mxArray* const proto = mxGetCell(proto_string, 0);
  const char* const proto_char = reinterpret_cast<const char* const>(mxGetPr(proto));
  NetParameter net_param;
  google::protobuf::TextFormat::ParseFromString(string(proto_char), &net_param);
  net_->CopyTrainedLayersFrom(net_param);
}

static mxArray* do_backward(const mxArray* const top_diff) {
  const vector<Blob<float>*>& output_blobs = net_->output_blobs();
  const vector<Blob<float>*>& input_blobs = net_->input_blobs();
  if (static_cast<unsigned int>(mxGetDimensions(top_diff)[0]) !=
      output_blobs.size()) {
    mex_error("Invalid input size");
  }
  // First, copy the output diff
  for (unsigned int i = 0; i < output_blobs.size(); ++i) {
    const mxArray* const elem = mxGetCell(top_diff, i);
    const float* const data_ptr =
        reinterpret_cast<const float* const>(mxGetPr(elem));
    switch (Caffe::mode()) {
    case Caffe::CPU:
      caffe_copy(output_blobs[i]->count(), data_ptr,
          output_blobs[i]->mutable_cpu_diff());
      break;
    case Caffe::GPU:
      caffe_copy(output_blobs[i]->count(), data_ptr,
          output_blobs[i]->mutable_gpu_diff());
      break;
    default:
        mex_error("Unknown Caffe mode");
    }  // switch (Caffe::mode())
  }
  // LOG(INFO) << "Start";
  net_->Backward();
  // LOG(INFO) << "End";
  mxArray* mx_out = mxCreateCellMatrix(input_blobs.size(), 1);
  for (unsigned int i = 0; i < input_blobs.size(); ++i) {
    // internally data is stored as (width, height, channels, num)
    // where width is the fastest dimension
    mwSize dims[4] = {input_blobs[i]->width(), input_blobs[i]->height(),
      input_blobs[i]->channels(), input_blobs[i]->num()};
    mxArray* mx_blob =  mxCreateNumericArray(4, dims, mxSINGLE_CLASS, mxREAL);
    mxSetCell(mx_out, i, mx_blob);
    float* data_ptr = reinterpret_cast<float*>(mxGetPr(mx_blob));
    switch (Caffe::mode()) {
    case Caffe::CPU:
      caffe_copy(input_blobs[i]->count(), input_blobs[i]->cpu_diff(), data_ptr);
      break;
    case Caffe::GPU:
      caffe_copy(input_blobs[i]->count(), input_blobs[i]->gpu_diff(), data_ptr);
      break;
    default:
        mex_error("Unknown Caffe mode");
    }  // switch (Caffe::mode())
  }

  return mx_out;
}

static mxArray* do_get_weights() {
  const vector<shared_ptr<Layer<float> > >& layers = net_->layers();
  const vector<string>& layer_names = net_->layer_names();

  // Step 1: count the number of layers with weights
  int num_layers = 0;
  {
    string prev_layer_name = "";
    for (unsigned int i = 0; i < layers.size(); ++i) {
      vector<shared_ptr<Blob<float> > >& layer_blobs = layers[i]->blobs();
      if (layer_blobs.size() == 0) {
        continue;
      }
      if (layer_names[i] != prev_layer_name) {
        prev_layer_name = layer_names[i];
        num_layers++;
      }
    }
  }

  // Step 2: prepare output array of structures
  mxArray* mx_layers;
  {
    const mwSize dims[2] = {num_layers, 1};
    const char* fnames[2] = {"weights", "layer_names"};
    mx_layers = mxCreateStructArray(2, dims, 2, fnames);
  }

  // Step 3: copy weights into output
  {
    string prev_layer_name = "";
    int mx_layer_index = 0;
    for (unsigned int i = 0; i < layers.size(); ++i) {
      vector<shared_ptr<Blob<float> > >& layer_blobs = layers[i]->blobs();
      if (layer_blobs.size() == 0) {
        continue;
      }

      mxArray* mx_layer_cells = NULL;
      if (layer_names[i] != prev_layer_name) {
        prev_layer_name = layer_names[i];
        const mwSize dims[2] = {static_cast<mwSize>(layer_blobs.size()), 1};
        mx_layer_cells = mxCreateCellArray(2, dims);
        mxSetField(mx_layers, mx_layer_index, "weights", mx_layer_cells);
        mxSetField(mx_layers, mx_layer_index, "layer_names",
            mxCreateString(layer_names[i].c_str()));
        mx_layer_index++;
      }

      for (unsigned int j = 0; j < layer_blobs.size(); ++j) {
        // internally data is stored as (width, height, channels, num)
        // where width is the fastest dimension
        mwSize dims[4] = {layer_blobs[j]->width(), layer_blobs[j]->height(),
            layer_blobs[j]->channels(), layer_blobs[j]->num()};

        mxArray* mx_weights =
          mxCreateNumericArray(4, dims, mxSINGLE_CLASS, mxREAL);
        mxSetCell(mx_layer_cells, j, mx_weights);
        float* weights_ptr = reinterpret_cast<float*>(mxGetPr(mx_weights));

        switch (Caffe::mode()) {
        case Caffe::CPU:
          caffe_copy(layer_blobs[j]->count(), layer_blobs[j]->cpu_data(),
              weights_ptr);
          break;
        case Caffe::GPU:
          caffe_copy(layer_blobs[j]->count(), layer_blobs[j]->gpu_data(),
              weights_ptr);
          break;
        default:
          mex_error("Unknown Caffe mode");
        }
      }
    }
  }

  return mx_layers;
}

static void get_weights_string(MEX_ARGS) {
  plhs[0] = get_weights_string();
}

static void get_weights(MEX_ARGS) {
  plhs[0] = do_get_weights();
}

static void set_weights(MEX_ARGS) {
  if (nrhs != 1) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("Wrong number of arguments");
  }

  set_weights_from_string(prhs[0]);
}

static void save_weights_to_file(MEX_ARGS) {
  if (nrhs != 1) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("Wrong number of arguments");
  }

  const char* const filename = mxArrayToString(prhs[0]);
  save_weights_to_file(filename);
}

static void set_mode_cpu(MEX_ARGS) {
  Caffe::set_mode(Caffe::CPU);
}

static void set_mode_gpu(MEX_ARGS) {
  Caffe::set_mode(Caffe::GPU);
}

static void set_phase_forwarda(MEX_ARGS) {
  Caffe::set_phase(Caffe::FORWARDA);
}

static void set_phase_forwardb(MEX_ARGS) {
  Caffe::set_phase(Caffe::FORWARDB);
}

static void set_phase_traina(MEX_ARGS) {
  Caffe::set_phase(Caffe::TRAINA);
}

static void set_phase_trainb(MEX_ARGS) {
  Caffe::set_phase(Caffe::TRAINB);
}

static void set_phase_train(MEX_ARGS) {
  Caffe::set_phase(Caffe::TRAIN);
}

static void set_phase_test(MEX_ARGS) {
  Caffe::set_phase(Caffe::TEST);
}

static void set_device(MEX_ARGS) {
  if (nrhs != 1) {
    ostringstream error_msg;
    error_msg << "Expected 1 argument, got " << nrhs;
    mex_error(error_msg.str());
  }

  int device_id = static_cast<int>(mxGetScalar(prhs[0]));
  Caffe::SetDevice(device_id);
}

static void get_init_key(MEX_ARGS) {
  plhs[0] = mxCreateDoubleScalar(init_key);
}

static void init_train(MEX_ARGS) {
  if (nrhs != 2 && nrhs != 1) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("Wrong number of arguments");
  }

  // Initialize solver
  char* solver_file = mxArrayToString(prhs[0]);
  SolverParameter solver_param;
  ReadProtoFromTextFileOrDie(string(solver_file), &solver_param);
  mxFree(solver_file);
  LOG(INFO) << "Read solver param from solver file";

  solver_.reset(GetSolver<float>(solver_param));
  net_ = solver_->net();

  if (nrhs == 2) {
    char* model_file = mxArrayToString(prhs[1]);
    solver_->net()->CopyTrainedLayersFrom(string(model_file));
    mxFree(model_file);
  }

  // Set network as initialized
  init_key = random();  // NOLINT(caffe/random_fn)
  if (nlhs == 1) {
    plhs[0] = mxCreateDoubleScalar(init_key);
  }
}

// MUST CALL set_phase before calling init_test!!!
// first arg is prototxt file, second optional arg is batch size, third optional arg is model weights file
static void init_forward_batch(MEX_ARGS) {
  if (nrhs != 2 && nrhs != 1 && nrhs != 3) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("Wrong number of arguments");
  }

  char* param_file = mxArrayToString(prhs[0]);
  if (solver_) {
    solver_.reset();
  }
  NetParameter net_param;
  ReadNetParamsFromTextFileOrDie(string(param_file), &net_param);

  // Alter batch size of memory data layer in net_param
  if (nrhs == 2) {
    const char* batch_size_string = mxArrayToString(prhs[1]);
    int batch_size = atoi(batch_size_string);

    for (int i = 0; i < net_param.layers_size(); ++i) {
      const LayerParameter& layer_param = net_param.layers(i);
      if (layer_param.type() != LayerParameter::MEMORY_DATA) continue;
      MemoryDataParameter* mem_param = net_param.mutable_layers(i)->mutable_memory_data_param();
      mem_param->set_batch_size(batch_size);
    }
  }
  net_.reset(new Net<float>(net_param));

  if (nrhs == 3) {
    char* model_file = mxArrayToString(prhs[2]);
    net_->CopyTrainedLayersFrom(string(model_file));
    mxFree(model_file);
  }

  mxFree(param_file);

  init_key = random();  // NOLINT(caffe/random_fn)

  if (nlhs == 1) {
    plhs[0] = mxCreateDoubleScalar(init_key);
  }
}
// MUST CALL set_phase before calling init_test!!!
static void init_test(MEX_ARGS) {
  if (nrhs != 2 && nrhs != 1) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("Wrong number of arguments");
  }

  char* param_file = mxArrayToString(prhs[0]);

  if (solver_) {
    solver_.reset();
  }
  //Caffe::set_phase(Caffe::TEST);
  net_.reset(new Net<float>(string(param_file)));
  if (nrhs == 2) {
    char* model_file = mxArrayToString(prhs[1]);
    net_->CopyTrainedLayersFrom(string(model_file));
    mxFree(model_file);
  }

  mxFree(param_file);

  init_key = random();  // NOLINT(caffe/random_fn)

  if (nlhs == 1) {
    plhs[0] = mxCreateDoubleScalar(init_key);
  }
}

static void init(MEX_ARGS) {
  if (nrhs != 2) {
    ostringstream error_msg;
    error_msg << "Expected 2 arguments, got " << nrhs;
    mex_error(error_msg.str());
  }

  char* param_file = mxArrayToString(prhs[0]);
  char* model_file = mxArrayToString(prhs[1]);

  net_.reset(new Net<float>(string(param_file)));
  net_->CopyTrainedLayersFrom(string(model_file));

  mxFree(param_file);
  mxFree(model_file);

  init_key = random();  // NOLINT(caffe/random_fn)

  if (nlhs == 1) {
    plhs[0] = mxCreateDoubleScalar(init_key);
  }
}

static void reset(MEX_ARGS) {
  if (solver_) {
    solver_.reset();
  }
  if (net_) {
    net_.reset();
    init_key = -2;
    LOG(INFO) << "Network reset, call init before use it again";
  }
}

static void vgps_train(MEX_ARGS) {
  if (nrhs != 1) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("Wrong number of arguments");
  }

  vgps_train(prhs[0]);
}

static void vgps_trainb(MEX_ARGS) {
  if (nrhs != 1) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("Wrong number of arguments");
  }

  Caffe::set_phase(Caffe::TRAINB);
  vgps_trainb(prhs[0]);
}

static void forward(MEX_ARGS) {
  if (nrhs != 1) {
    ostringstream error_msg;
    error_msg << "Expected 1 argument, got " << nrhs;
    mex_error(error_msg.str());
  }

  plhs[0] = do_forward(prhs[0]);
}

static void vgps_forward(MEX_ARGS) {
  if (nrhs != 1) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("Wrong number of arguments");
  }

  plhs[0] = vgps_forward(prhs[0]);
}


static void vgps_forward_only(MEX_ARGS) {
  if (nrhs != 1) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("Wrong number of arguments");
  }

  plhs[0] = vgps_forward_only(prhs[0]);
}

static void vgps_forwarda_only(MEX_ARGS) {
  if (nrhs != 1) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("Wrong number of arguments");
  }

  plhs[0] = vgps_forwarda_only(prhs[0]);
}

static void backward(MEX_ARGS) {
  if (nrhs != 1) {
    ostringstream error_msg;
    error_msg << "Expected 1 argument, got " << nrhs;
    mex_error(error_msg.str());
  }

  plhs[0] = do_backward(prhs[0]);
}

static void is_initialized(MEX_ARGS) {
  if (!net_) {
    plhs[0] = mxCreateDoubleScalar(0);
  } else {
    plhs[0] = mxCreateDoubleScalar(1);
  }
}

static void read_mean(MEX_ARGS) {
    if (nrhs != 1) {
        mexErrMsgTxt("Usage: caffe('read_mean', 'path_to_binary_mean_file'");
        return;
    }
    const string& mean_file = mxArrayToString(prhs[0]);
    Blob<float> data_mean;
    LOG(INFO) << "Loading mean file from: " << mean_file;
    BlobProto blob_proto;
    bool result = ReadProtoFromBinaryFile(mean_file.c_str(), &blob_proto);
    if (!result) {
        mexErrMsgTxt("Couldn't read the file");
        return;
    }
    data_mean.FromProto(blob_proto);
    mwSize dims[4] = {data_mean.width(), data_mean.height(),
                      data_mean.channels(), data_mean.num() };
    mxArray* mx_blob =  mxCreateNumericArray(4, dims, mxSINGLE_CLASS, mxREAL);
    float* data_ptr = reinterpret_cast<float*>(mxGetPr(mx_blob));
    caffe_copy(data_mean.count(), data_mean.cpu_data(), data_ptr);
    mexWarnMsgTxt("Remember that Caffe saves in [width, height, channels]"
                  " format and channels are also BGR!");
    plhs[0] = mx_blob;
}

static void exitFunction(void) {
  int nlhs, nrhs;
  const mxArray **prhs;
  mxArray **plhs;
  reset(nlhs, plhs, nrhs, prhs);
  mexUnlock();
}

/** -----------------------------------------------------------------
 ** Available commands.
 **/
struct handler_registry {
  string cmd;
  void (*func)(MEX_ARGS);
};

static handler_registry handlers[] = {
  // Public API functions
  { "forward",            vgps_forward    },
  { "forward_only",       vgps_forward_only    },
  { "forwarda_only",       vgps_forwarda_only    },
  { "backward",           backward        },
  { "init",               init            },
  { "init_test",          init_test       },
  { "init_forward_batch", init_forward_batch},
  { "init_train",         init_train      },
  { "is_initialized",     is_initialized  },
  { "set_mode_cpu",       set_mode_cpu    },
  { "set_mode_gpu",       set_mode_gpu    },
  { "set_phase_train",    set_phase_train },
  { "set_phase_traina",    set_phase_traina },
  { "set_phase_trainb",    set_phase_trainb },
  { "set_phase_forwarda",    set_phase_forwarda },
  { "set_phase_forwardb",    set_phase_forwardb },
  { "set_phase_test",     set_phase_test  },
  { "set_device",         set_device      },
  { "get_weights",        get_weights     },
  { "get_weights_string", get_weights_string     },
  { "set_weights",        set_weights     },
  { "save_weights",        save_weights_to_file     },
  { "get_init_key",       get_init_key    },
  { "reset",              reset           },
  { "read_mean",          read_mean       },
  { "train",              vgps_train      },
  { "trainb",              vgps_trainb      },
  // The end.
  { "END",                NULL            },
};


/** -----------------------------------------------------------------
 ** matlab entry point: caffe(api_command, arg1, arg2, ...)
 **/
void mexFunction(MEX_ARGS) {
  mexLock();  // Avoid clearing the mex file.
  if (nrhs == 0) {
    mex_error("No API command given");
    return;
  }

  mexAtExit(exitFunction);

  { // Handle input command
    char *cmd = mxArrayToString(prhs[0]);
    bool dispatched = false;
    // Dispatch to cmd handler
    for (int i = 0; handlers[i].func != NULL; i++) {
      if (handlers[i].cmd.compare(cmd) == 0) {
        handlers[i].func(nlhs, plhs, nrhs-1, prhs+1);
        dispatched = true;
        break;
      }
    }
    if (!dispatched) {
      ostringstream error_msg;
      error_msg << "Unknown command '" << cmd << "'";
      mex_error(error_msg.str());
    }
    mxFree(cmd);
  }
}
