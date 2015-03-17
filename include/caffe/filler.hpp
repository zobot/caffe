// Fillers are random number generators that fills a blob using the specified
// algorithm. The expectation is that they are only going to be used during
// initialization time and will not involve any GPUs.

#ifndef CAFFE_FILLER_HPP
#define CAFFE_FILLER_HPP

#include <string>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

/// @brief Fills a Blob with constant or randomly-generated data.
template <typename Dtype>
class Filler {
 public:
  explicit Filler(const FillerParameter& param) : filler_param_(param) {}
  virtual ~Filler() {}
  virtual void Fill(Blob<Dtype>* blob) = 0;
 protected:
  FillerParameter filler_param_;
};  // class Filler


/// @brief Fills a Blob with constant values @f$ x = 0 @f$.
template <typename Dtype>
class ConstantFiller : public Filler<Dtype> {
 public:
  explicit ConstantFiller(const FillerParameter& param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Blob<Dtype>* blob) {
    Dtype* data = blob->mutable_cpu_data();
    const int count = blob->count();
    const Dtype value = this->filler_param_.value();
    CHECK(count);
    for (int i = 0; i < count; ++i) {
      data[i] = value;
    }
    CHECK_EQ(this->filler_param_.sparse(), -1)
         << "Sparsity not supported by this Filler.";
  }
};

/// @brief Fills a Blob with uniformly distributed values @f$ x\sim U(a, b) @f$.
template <typename Dtype>
class UniformFiller : public Filler<Dtype> {
 public:
  explicit UniformFiller(const FillerParameter& param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Blob<Dtype>* blob) {
    CHECK(blob->count());
    caffe_rng_uniform<Dtype>(blob->count(), Dtype(this->filler_param_.min()),
        Dtype(this->filler_param_.max()), blob->mutable_cpu_data());
    CHECK_EQ(this->filler_param_.sparse(), -1)
         << "Sparsity not supported by this Filler.";
  }
};

/// @brief Fills a Blob with Gaussian-distributed values @f$ x = a @f$.
template <typename Dtype>
class GaussianFiller : public Filler<Dtype> {
 public:
  explicit GaussianFiller(const FillerParameter& param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Blob<Dtype>* blob) {
    Dtype* data = blob->mutable_cpu_data();
    CHECK(blob->count());
    caffe_rng_gaussian<Dtype>(blob->count(), Dtype(this->filler_param_.mean()),
        Dtype(this->filler_param_.std()), blob->mutable_cpu_data());
    int sparse = this->filler_param_.sparse();
    CHECK_GE(sparse, -1);
    if (sparse >= 0) {
      // Sparse initialization is implemented for "weight" blobs; i.e. matrices.
      // These have num == channels == 1; width is number of inputs; height is
      // number of outputs.  The 'sparse' variable specifies the mean number
      // of non-zero input weights for a given output.
      CHECK_EQ(blob->num(), 1);
      CHECK_EQ(blob->channels(), 1);
      int num_outputs = blob->height();
      Dtype non_zero_probability = Dtype(sparse) / Dtype(num_outputs);
      rand_vec_.reset(new SyncedMemory(blob->count() * sizeof(int)));
      int* mask = reinterpret_cast<int*>(rand_vec_->mutable_cpu_data());
      caffe_rng_bernoulli(blob->count(), non_zero_probability, mask);
      for (int i = 0; i < blob->count(); ++i) {
        data[i] *= mask[i];
      }
    }
  }

 protected:
  shared_ptr<SyncedMemory> rand_vec_;
};

/** @brief Fills a Blob with values @f$ x \in [0, 1] @f$
 *         such that @f$ \forall i \sum_j x_{ij} = 1 @f$.
 */
template <typename Dtype>
class PositiveUnitballFiller : public Filler<Dtype> {
 public:
  explicit PositiveUnitballFiller(const FillerParameter& param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Blob<Dtype>* blob) {
    Dtype* data = blob->mutable_cpu_data();
    DCHECK(blob->count());
    caffe_rng_uniform<Dtype>(blob->count(), 0, 1, blob->mutable_cpu_data());
    // We expect the filler to not be called very frequently, so we will
    // just use a simple implementation
    int dim = blob->count() / blob->num();
    CHECK(dim);
    for (int i = 0; i < blob->num(); ++i) {
      Dtype sum = 0;
      for (int j = 0; j < dim; ++j) {
        sum += data[i * dim + j];
      }
      for (int j = 0; j < dim; ++j) {
        data[i * dim + j] /= sum;
      }
    }
    CHECK_EQ(this->filler_param_.sparse(), -1)
         << "Sparsity not supported by this Filler.";
  }
};

/** @brief Fills a Blob with values @f$ x \in {0,x,y} @f$
 * such that the output of the layer is the weighted average
 * x and y coordinate for each input channel.
 */
template <typename Dtype>
class ImageXYFiller : public Filler<Dtype> {
 public:
  explicit ImageXYFiller(const FillerParameter& param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Blob<Dtype>* blob) {
    Dtype* data = blob->mutable_cpu_data();
    DCHECK(blob->count());
    int num_channels = this->filler_param_.channels();
    int num_X = this->filler_param_.width();
    int num_Y = this->filler_param_.height();
    const string& xy = this->filler_param_.xy();

    // paramater blob should be 1x1xhxw where h is determined by number of
    // channels of the input and w is the dim of the input.
    CHECK_EQ(blob->num(), 1);
    CHECK_EQ(blob->channels(), 1);
    CHECK_EQ(blob->width(), num_channels * num_X * num_Y) << blob->width() << " != " << num_channels*num_X*num_Y;
    if (xy == "both" || xy == "nboth2") {
      // Output dimension of fully connected layer should be twice the number of
      // channels, and the input should be the dimension of input images..
      CHECK_EQ(blob->height(), num_channels * 2) << "Blob height: " << blob->height();
    } else if (xy == "x" || xy == "y" || xy == "ones") {
      CHECK_EQ(blob->height(), num_channels) << "Blob height: " << blob->height();
    } else {
      LOG(FATAL) << "Unknown x/y filler policy: " << xy;
    }

    for (int c = 0; c < num_channels; ++c) {
      // One for each feature map.
      for (int x = 0; x < num_X; ++x) {
        for (int y = 0; y < num_Y; ++y) {
          // Iterature over all params for this input point.
          for (int k = 0; k < blob->height(); ++k) {
            int offset = c*num_X*num_Y + y*num_X + x;
            Dtype* weight_ptr = data + blob->offset(0,0,k, offset);
            if (xy == "both" || xy == "nboth2") {
              if (c*2 == k) {
                // x coordinate
                if (xy == "both") weight_ptr[0] = 2*(Dtype(x+1) / Dtype(num_X) - Dtype(0.5));
                if (xy == "nboth2") weight_ptr[0] = - pow(2*(Dtype(x+1) / Dtype(num_X) - Dtype(0.5)), 2);
              } else if (c*2+1 == k) {
                if (xy == "both") weight_ptr[0] = 2*(Dtype(y+1) / Dtype(num_Y) - Dtype(0.5));
                if (xy == "nboth2") weight_ptr[0] = - pow(2*(Dtype(y+1) / Dtype(num_Y) - Dtype(0.5)), 2);
              } else {
                weight_ptr[0] = Dtype(0);
              }
            } else {
              if (c == k) {
                if (xy == "x") weight_ptr[0] = 2*(Dtype(x+1) / Dtype(num_X) - Dtype(0.5));
                if (xy == "y") weight_ptr[0] = 2*(Dtype(y+1) / Dtype(num_Y) - Dtype(0.5));
                if (xy == "ones") weight_ptr[0] = 1;
              } else {
                weight_ptr[0] = 0;
              }
            }
          }
        }
      }
    }

    // We expect the filler to not be called very frequently, so we will
    // just use a simple implementation
    CHECK_EQ(this->filler_param_.sparse(), -1)
         << "Sparsity not supported by this Filler.";
  }
};

/**
 * @brief Fills a Blob with values @f$ x \sim U(-a, +a) @f$ where @f$ a @f$
 *        is set inversely proportional to the number of incoming nodes.
 *
 * A Filler based on the paper [Bengio and Glorot 2010]: Understanding
 * the difficulty of training deep feedforward neuralnetworks, but does not
 * use the fan_out value.
 *
 * It fills the incoming matrix by randomly sampling uniform data from
 * [-scale, scale] where scale = sqrt(3 / fan_in) where fan_in is the number
 * of input nodes. You should make sure the input blob has shape (num, a, b, c)
 * where a * b * c = fan_in.
 *
 * TODO(dox): make notation in above comment consistent with rest & use LaTeX.
 */
template <typename Dtype>
class XavierFiller : public Filler<Dtype> {
 public:
  explicit XavierFiller(const FillerParameter& param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Blob<Dtype>* blob) {
    CHECK(blob->count());
    int fan_in = blob->count() / blob->num();
    Dtype scale = sqrt(Dtype(3) / fan_in);
    caffe_rng_uniform<Dtype>(blob->count(), -scale, scale,
        blob->mutable_cpu_data());
    CHECK_EQ(this->filler_param_.sparse(), -1)
         << "Sparsity not supported by this Filler.";
  }
};


/**
 * @brief Get a specific filler from the specification given in FillerParameter.
 *
 * Ideally this would be replaced by a factory pattern, but we will leave it
 * this way for now.
 */
template <typename Dtype>
Filler<Dtype>* GetFiller(const FillerParameter& param) {
  const std::string& type = param.type();
  if (type == "constant") {
    return new ConstantFiller<Dtype>(param);
  } else if (type == "gaussian") {
    return new GaussianFiller<Dtype>(param);
  } else if (type == "positive_unitball") {
    return new PositiveUnitballFiller<Dtype>(param);
  } else if (type == "uniform") {
    return new UniformFiller<Dtype>(param);
  } else if (type == "xavier") {
    return new XavierFiller<Dtype>(param);
  } else if (type == "imagexy") {
    return new ImageXYFiller<Dtype>(param);
  } else {
    CHECK(false) << "Unknown filler name: " << param.type();
  }
  return (Filler<Dtype>*)(NULL);
}

}  // namespace caffe

#endif  // CAFFE_FILLER_HPP_
