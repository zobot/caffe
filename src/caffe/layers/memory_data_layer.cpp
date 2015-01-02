#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"

namespace caffe {

template <typename Dtype>
void MemoryDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
     const vector<Blob<Dtype>*>& top) {
  top_size_ = top.size();
  batch_size_ = this->layer_param_.memory_data_param().batch_size();
  channels1_ = this->layer_param_.memory_data_param().channels1();
  channels2_ = this->layer_param_.memory_data_param().channels2();
  channels3_ = this->layer_param_.memory_data_param().channels3();
  channels4_ = this->layer_param_.memory_data_param().channels4();
  height1_ = this->layer_param_.memory_data_param().height1();
  height2_ = this->layer_param_.memory_data_param().height2();
  height3_ = this->layer_param_.memory_data_param().height3();
  height4_ = this->layer_param_.memory_data_param().height4();
  width1_ = this->layer_param_.memory_data_param().width1();
  width2_ = this->layer_param_.memory_data_param().width2();
  width3_ = this->layer_param_.memory_data_param().width3();
  width4_ = this->layer_param_.memory_data_param().width4();
  size1_ = channels1_ * height1_ * width1_;
  size2_ = channels2_ * height2_ * width2_;
  size3_ = channels3_ * height3_ * width3_;
  size4_ = channels4_ * height4_ * width4_;
  CHECK_GT(batch_size_ * size1_, 0) <<
      "batch_size, channels, height, and width must be specified and"
      " positive in memory_data_param";
  top[0]->Reshape(batch_size_, channels1_, height1_, width1_);
  if (top_size_ > 1) top[1]->Reshape(batch_size_, channels2_, height2_, width2_);
  if (top_size_ > 2) top[2]->Reshape(batch_size_, channels3_, height3_, width3_);
  if (top_size_ > 3) top[3]->Reshape(batch_size_, channels4_, height4_, width4_);
  added_data1_.Reshape(batch_size_, channels1_, height1_, width1_);
  added_data2_.Reshape(batch_size_, channels2_, height2_, width2_);
  added_data3_.Reshape(batch_size_, channels3_, height3_, width3_);
  added_data4_.Reshape(batch_size_, channels4_, height4_, width4_);
  data1_ = NULL;
  data2_ = NULL;
  data3_ = NULL;
  data4_ = NULL;
  added_data1_.cpu_data();
  added_data2_.cpu_data();
  added_data3_.cpu_data();
  added_data4_.cpu_data();
}

// This could be useful later when we just want to add a few more samples.
/*
template <typename Dtype>
void MemoryDataLayer<Dtype>::AddDatumVector(const vector<Datum>& datum_vector, int index) {
  CHECK(!has_new_data_) <<
      "Can't add Datum when earlier ones haven't been consumed"
      << " by the upper layers";
  size_t num = datum_vector.size();
  CHECK_GT(num, 0) << "There is no datum to add";
  CHECK_LE(num, batch_size_) <<
      "The number of added datum must be no greater than the batch size";

  // Apply data transformations (mirror, scale, crop...)
  this->data_transformer_.Transform(datum_vector, &added_data_);
  // Copy Labels
  Dtype* top_label = added_label_.mutable_cpu_data();
  for (int item_id = 0; item_id < num; ++item_id) {
    top_label[item_id] = datum_vector[item_id].label();
  }
  // num_images == batch_size_
  Dtype* top_data = added_data_.mutable_cpu_data();
  Reset(top_data, top_label, batch_size_);
  has_new_data_ = true;
}
*/

template <typename Dtype>
void MemoryDataLayer<Dtype>::Reset(Dtype* data1, Dtype* data2, Dtype* data3, Dtype* data4, int n) {
  CHECK(data1);
  CHECK(data2);
  CHECK(data3);
  CHECK(data4);
  CHECK_EQ(top_size_, 4);
  CHECK_EQ(n % batch_size_, 0) << "n must be a multiple of batch size";
  data1_ = data1;
  data2_ = data2;
  data3_ = data3;
  data4_ = data4;
  n_ = n;
  pos_ = 0;
}

template <typename Dtype>
void MemoryDataLayer<Dtype>::Reset(Dtype* data1, Dtype* data2, Dtype* data3, int n) {
  CHECK(data1);
  CHECK(data2);
  CHECK(data3);
  CHECK_EQ(top_size_, 3);
  CHECK_EQ(n % batch_size_, 0) << "n must be a multiple of batch size";
  data1_ = data1;
  data2_ = data2;
  data3_ = data3;
  n_ = n;
  pos_ = 0;
}

template <typename Dtype>
void MemoryDataLayer<Dtype>::Reset(Dtype* data1, Dtype* data2, int n) {
  CHECK(data1);
  CHECK(data2);
  CHECK_EQ(top_size_, 2);
  CHECK_EQ(n % batch_size_, 0) << "n must be a multiple of batch size";
  data1_ = data1;
  data2_ = data2;
  n_ = n;
  pos_ = 0;
}

template <typename Dtype>
void MemoryDataLayer<Dtype>::Reset(Dtype* data1, int n) {
  CHECK(data1);
  CHECK_EQ(top_size_, 1);
  CHECK_EQ(n % batch_size_, 0) << "n must be a multiple of batch size";
  data1_ = data1;
  n_ = n;
  pos_ = 0;
}

template <typename Dtype>
void MemoryDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK(data1_) << "MemoryDataLayer needs to be initalized by calling Reset";
  top[0]->set_cpu_data(data1_ + pos_ * size1_);
  if (top_size_ > 1) top[1]->set_cpu_data(data2_ + pos_ * size2_);
  if (top_size_ > 2) top[2]->set_cpu_data(data3_ + pos_ * size3_);
  if (top_size_ > 3) top[3]->set_cpu_data(data4_ + pos_ * size4_);
  pos_ = (pos_ + batch_size_) % n_;
  has_new_data_ = false;
}

INSTANTIATE_CLASS(MemoryDataLayer);
REGISTER_LAYER_CLASS(MEMORY_DATA, MemoryDataLayer);
}  // namespace caffe
