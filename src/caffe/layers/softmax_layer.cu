#include <algorithm>
#include <cfloat>
#include <vector>

#include "thrust/device_vector.h"

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void kernel_all_max(const int num, const int channels,
    const int spatial_dim, const Dtype* data, Dtype* out) {
  CUDA_KERNEL_LOOP(n, num) {
    Dtype maxval = -FLT_MAX;
    for (int c = 0; c < channels; ++c) {
      for (int s = 0; s < spatial_dim; ++s) {
        maxval = max(data[(n * channels + c) * spatial_dim + s], maxval);
      }
    }
    out[n] = maxval;
  }
}
template <typename Dtype>
__global__ void kernel_channel_max(const int num, const int channels,
    const int spatial_dim, const Dtype* data, Dtype* out) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    Dtype maxval = -FLT_MAX;
    for (int c = 0; c < channels; ++c) {
      maxval = max(data[(n * channels + c) * spatial_dim + s], maxval);
    }
    out[index] = maxval;
  }
}

template <typename Dtype>
__global__ void kernel_pixel_max(const int num, const int channels,
    const int spatial_dim, const Dtype* data, Dtype* out) {
  CUDA_KERNEL_LOOP(index, num * channels) {
    int n = index / channels;
    int c = index % channels;
    Dtype maxval = -FLT_MAX;
    for (int s = 0; s < spatial_dim; ++s) {
      maxval = max(data[(n * channels + c) * spatial_dim + s], maxval);
    }
    out[index] = maxval;
  }
}

template <typename Dtype>
__global__ void kernel_all_subtract(const int num, const int channels,
    const int spatial_dim, Dtype* data, const Dtype* channel_max) {
  CUDA_KERNEL_LOOP(n, num) {
    for (int c = 0; c < channels; ++c) {
      for (int s = 0; s < spatial_dim; ++s) {
        data[(n * channels + c) * spatial_dim + s] -= channel_max[n];
      }
    }
  }
}

template <typename Dtype>
__global__ void kernel_channel_subtract(const int num, const int channels,
    const int spatial_dim, Dtype* data, const Dtype* channel_max) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    for (int c = 0; c < channels; ++c) {
      data[(n * channels + c) * spatial_dim + s] -= channel_max[index];
    }
  }
}

template <typename Dtype>
__global__ void kernel_pixel_subtract(const int num, const int channels,
    const int spatial_dim, Dtype* data, const Dtype* pixel_max) {
  CUDA_KERNEL_LOOP(index, num * channels) {
    int n = index / channels;
    int c = index % channels;
    for (int s = 0; s < spatial_dim; ++s) {
      data[(n * channels + c) * spatial_dim + s] -= pixel_max[index];
    }
  }
}

template <typename Dtype>
__global__ void kernel_exp(const int count, const Dtype* data, Dtype* out) {
  CUDA_KERNEL_LOOP(index, count) {
    out[index] = exp(data[index]);
  }
}

template <typename Dtype>
__global__ void kernel_channel_sum(const int num, const int channels,
    const int spatial_dim, const Dtype* data, Dtype* channel_sum) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    Dtype sum = 0;
    for (int c = 0; c < channels; ++c) {
      sum += data[(n * channels + c) * spatial_dim + s];
    }
    channel_sum[index] = sum;
  }
}

template <typename Dtype>
__global__ void kernel_all_sum(const int num, const int channels,
    const int spatial_dim, const Dtype* data, Dtype* all_sum) {
  CUDA_KERNEL_LOOP(n, num) {
    Dtype sum = 0;
    for (int c = 0; c < channels; ++c) {
      for (int s = 0; s < spatial_dim; ++s) {
        sum += data[(n * channels + c) * spatial_dim + s];
      }
    }
    all_sum[n] = sum;
  }
}

template <typename Dtype>
__global__ void kernel_pixel_sum(const int num, const int channels,
    const int spatial_dim, const Dtype* data, Dtype* pixel_sum) {
  CUDA_KERNEL_LOOP(index, num * channels) {
    int n = index / channels;
    int c = index % channels;
    Dtype sum = 0;
    for (int s = 0; s < spatial_dim; ++s) {
      sum += data[(n * channels + c) * spatial_dim + s];
    }
    pixel_sum[index] = sum;
  }
}

template <typename Dtype>
__global__ void kernel_channel_div(const int num, const int channels,
    const int spatial_dim, Dtype* data, const Dtype* channel_sum) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    for (int c = 0; c < channels; ++c) {
      data[(n * channels + c) * spatial_dim + s] /= channel_sum[index];
    }
  }
}

template <typename Dtype>
__global__ void kernel_all_div(const int num, const int channels,
    const int spatial_dim, Dtype* data, const Dtype* all_sum) {
  CUDA_KERNEL_LOOP(n, num) {
    for (int c = 0; c < channels; ++c) {
      for (int s = 0; s < spatial_dim; ++s) {
        data[(n * channels + c) * spatial_dim + s] /= all_sum[n];
      }
    }
  }
}

template <typename Dtype>
__global__ void kernel_pixel_div(const int num, const int channels,
    const int spatial_dim, Dtype* data, const Dtype* pixel_sum) {
  CUDA_KERNEL_LOOP(index, num * channels) {
    int n = index / channels;
    int c = index % channels;
    // int offset = (n*channels+c)*spatial_dim;
    // caffe_gpu_axpby(spatial_dim, Dtype(1.0)/pixel_sum[index], data+offset, Dtype(0.0), data+offset);
    for (int s = 0; s < spatial_dim; ++s) {
      data[(n * channels + c) * spatial_dim + s] /= pixel_sum[index];
    }
  }
}

template <typename Dtype>
__global__ void kernel_channel_dot(const int num, const int channels,
    const int spatial_dim, const Dtype* data_1, const Dtype* data_2,
    Dtype* channel_dot) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    Dtype dot = 0;
    for (int c = 0; c < channels; ++c) {
      dot += (data_1[(n * channels + c) * spatial_dim + s]
          * data_2[(n * channels + c) * spatial_dim + s]);
    }
    channel_dot[index] = dot;
  }
}

template <typename Dtype>
__global__ void kernel_all_dot(const int num, const int channels,
    const int spatial_dim, const Dtype* data_1, const Dtype* data_2,
    Dtype* all_dot) {
  CUDA_KERNEL_LOOP(n, num) {
    Dtype dot = 0;
    for (int c = 0; c < channels; ++c) {
      for (int s = 0; s < spatial_dim; ++s) {
        dot += (data_1[(n * channels + c) * spatial_dim + s]
            * data_2[(n * channels + c) * spatial_dim + s]);
      }
    }
    all_dot[n] = dot;
  }
}

template <typename Dtype>
__global__ void kernel_pixel_dot(const int num, const int channels,
    const int spatial_dim, const Dtype* data_1, const Dtype* data_2,
    Dtype* pixel_dot) {
  CUDA_KERNEL_LOOP(index, num * channels) {
    int n = index / channels;
    int c = index % channels;
    Dtype dot = 0;
    for (int s = 0; s < spatial_dim; ++s) {
      dot += (data_1[(n * channels + c) * spatial_dim + s]
          * data_2[(n * channels + c) * spatial_dim + s]);
    }
    pixel_dot[index] = dot;
  }
}
template <typename Dtype>
void SoftmaxLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  Dtype* scale_data = scale_.mutable_gpu_data();
  int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int spatial_dim = bottom[0]->height() * bottom[0]->width();
  // We need to subtract the max to avoid numerical issues, compute the exp,
  // and then normalize.
  caffe_copy(bottom[0]->count(), bottom_data, top_data);
  if (dimension_ == "spatial") {
    // compute max
    // NOLINT_NEXT_LINE(whitespace/operators)
    kernel_pixel_max<Dtype><<<CAFFE_GET_BLOCKS(num * channels),
        CAFFE_CUDA_NUM_THREADS>>>(num, channels, spatial_dim, top_data,
        scale_data);
    // subtract
    // NOLINT_NEXT_LINE(whitespace/operators)
    kernel_pixel_subtract<Dtype><<<CAFFE_GET_BLOCKS(num * channels),
        CAFFE_CUDA_NUM_THREADS>>>(num, channels, spatial_dim, top_data,
        scale_data);
    // divide by temperature
    caffe_gpu_scal<Dtype>(num*spatial_dim*channels, temp_, top_data);
    // exponentiate
    // NOLINT_NEXT_LINE(whitespace/operators)
    kernel_exp<Dtype><<<CAFFE_GET_BLOCKS(num * channels * spatial_dim),
        CAFFE_CUDA_NUM_THREADS>>>(num * channels * spatial_dim, top_data,
        top_data);
    // sum after exp
    // NOLINT_NEXT_LINE(whitespace/operators)
    kernel_pixel_sum<Dtype><<<CAFFE_GET_BLOCKS(num * channels),
        CAFFE_CUDA_NUM_THREADS>>>(num, channels, spatial_dim, top_data,
        scale_data);
    // divide
    // NOLINT_NEXT_LINE(whitespace/operators)
    kernel_pixel_div<Dtype><<<CAFFE_GET_BLOCKS(num * channels),
        CAFFE_CUDA_NUM_THREADS>>>(num, channels, spatial_dim, top_data,
        scale_data);
  } else if (dimension_ == "channel") {
    // compute max
    // NOLINT_NEXT_LINE(whitespace/operators)
    kernel_channel_max<Dtype><<<CAFFE_GET_BLOCKS(num * spatial_dim),
        CAFFE_CUDA_NUM_THREADS>>>(num, channels, spatial_dim, top_data,
        scale_data);
    // subtract
    // NOLINT_NEXT_LINE(whitespace/operators)
    kernel_channel_subtract<Dtype><<<CAFFE_GET_BLOCKS(num * spatial_dim),
        CAFFE_CUDA_NUM_THREADS>>>(num, channels, spatial_dim, top_data,
        scale_data);
    // divide by temperature
    caffe_gpu_scal<Dtype>(num*spatial_dim*channels, temp_, top_data);
    // exponentiate
    // NOLINT_NEXT_LINE(whitespace/operators)
    kernel_exp<Dtype><<<CAFFE_GET_BLOCKS(num * channels * spatial_dim),
        CAFFE_CUDA_NUM_THREADS>>>(num * channels * spatial_dim, top_data,
        top_data);
    // sum after exp
    // NOLINT_NEXT_LINE(whitespace/operators)
    kernel_channel_sum<Dtype><<<CAFFE_GET_BLOCKS(num * spatial_dim),
        CAFFE_CUDA_NUM_THREADS>>>(num, channels, spatial_dim, top_data,
        scale_data);
    // divide
    // NOLINT_NEXT_LINE(whitespace/operators)
    kernel_channel_div<Dtype><<<CAFFE_GET_BLOCKS(num * spatial_dim),
        CAFFE_CUDA_NUM_THREADS>>>(num, channels, spatial_dim, top_data,
        scale_data);
  } else if (dimension_ == "all") {
    // compute max
    // NOLINT_NEXT_LINE(whitespace/operators)
    kernel_all_max<Dtype><<<CAFFE_GET_BLOCKS(num),
        CAFFE_CUDA_NUM_THREADS>>>(num, channels, spatial_dim, top_data,
        scale_data);
    // subtract
    // NOLINT_NEXT_LINE(whitespace/operators)
    kernel_all_subtract<Dtype><<<CAFFE_GET_BLOCKS(num),
        CAFFE_CUDA_NUM_THREADS>>>(num, channels, spatial_dim, top_data,
        scale_data);
    // divide by temperature
    caffe_gpu_scal<Dtype>(num*spatial_dim*channels, temp_, top_data);
    // exponentiate
    // NOLINT_NEXT_LINE(whitespace/operators)
    kernel_exp<Dtype><<<CAFFE_GET_BLOCKS(num * channels * spatial_dim),
        CAFFE_CUDA_NUM_THREADS>>>(num * channels * spatial_dim, top_data,
        top_data);
    // sum after exp
    // NOLINT_NEXT_LINE(whitespace/operators)
    kernel_all_sum<Dtype><<<CAFFE_GET_BLOCKS(num),
        CAFFE_CUDA_NUM_THREADS>>>(num, channels, spatial_dim, top_data,
        scale_data);
    // divide
    // NOLINT_NEXT_LINE(whitespace/operators)
    kernel_all_div<Dtype><<<CAFFE_GET_BLOCKS(num),
        CAFFE_CUDA_NUM_THREADS>>>(num, channels, spatial_dim, top_data,
        scale_data);
  } else {
    LOG(FATAL) << "Unrecognized softmax dimension type";
  }
}

template <typename Dtype>
void SoftmaxLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* top_data = top[0]->gpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  Dtype* scale_data = scale_.mutable_gpu_data();
  int num = top[0]->num();
  int channels = top[0]->channels();
  int spatial_dim = top[0]->height() * top[0]->width();
  caffe_copy(top[0]->count(), top_diff, bottom_diff);
  // Compute inner1d(top_diff, top_data) and subtract them from the bottom diff.
  if (dimension_ == "spatial") {
    // NOLINT_NEXT_LINE(whitespace/operators)
    kernel_pixel_dot<Dtype><<<CAFFE_GET_BLOCKS(num * channels),
        CAFFE_CUDA_NUM_THREADS>>>(num, channels, spatial_dim, top_diff, top_data,
        scale_data);
    // NOLINT_NEXT_LINE(whitespace/operators)
    kernel_pixel_subtract<Dtype><<<CAFFE_GET_BLOCKS(num * channels),
        CAFFE_CUDA_NUM_THREADS>>>(num, channels, spatial_dim, bottom_diff,
        scale_data);
  } else if (dimension_ == "channel") {
    // NOLINT_NEXT_LINE(whitespace/operators)
    kernel_channel_dot<Dtype><<<CAFFE_GET_BLOCKS(num * spatial_dim),
        CAFFE_CUDA_NUM_THREADS>>>(num, channels, spatial_dim, top_diff, top_data,
        scale_data);
    // NOLINT_NEXT_LINE(whitespace/operators)
    kernel_channel_subtract<Dtype><<<CAFFE_GET_BLOCKS(num * spatial_dim),
        CAFFE_CUDA_NUM_THREADS>>>(num, channels, spatial_dim, bottom_diff,
        scale_data);
  } else if (dimension_ == "all") {
    // NOLINT_NEXT_LINE(whitespace/operators)
    kernel_all_dot<Dtype><<<CAFFE_GET_BLOCKS(num),
        CAFFE_CUDA_NUM_THREADS>>>(num, channels, spatial_dim, top_diff, top_data,
        scale_data);
    // NOLINT_NEXT_LINE(whitespace/operators)
    kernel_all_subtract<Dtype><<<CAFFE_GET_BLOCKS(num),
        CAFFE_CUDA_NUM_THREADS>>>(num, channels, spatial_dim, bottom_diff,
        scale_data);
  }
  // elementwise multiplication
  caffe_gpu_mul<Dtype>(top[0]->count(), bottom_diff, top_data, bottom_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(SoftmaxLayer);


}  // namespace caffe
