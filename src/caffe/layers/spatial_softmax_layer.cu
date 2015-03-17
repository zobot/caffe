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
    const int spatial_dim, const int p2numel, Dtype* data, Dtype* out, unsigned int step) {
    int scaled_numel = p2numel/(2*step);
    for (int index = blockIdx.x*blockDim.x + threadIdx.x;
         index < num * scaled_numel;
         index += blockDim.x * gridDim.x) {
        int n = index / scaled_numel;
        int idx = (index % scaled_numel)*step*2;
        int index_data = n*channels*spatial_dim + idx;
        
        // Add.
        if (idx + step < channels*spatial_dim) {
            data[index_data] = max(data[index_data],data[index_data + step]);
            if (step*2 >= channels*spatial_dim)
                out[n] = data[index_data];
        }
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
    const int spatial_dim, const int p2spatial_dim, Dtype* data, Dtype* out, unsigned int step) {
    int scaled_spatial_dim = p2spatial_dim/(2*step);
    for (int index = blockIdx.x*blockDim.x + threadIdx.x;
         index < num * channels * scaled_spatial_dim;
         index += blockDim.x * gridDim.x) {
        int n = index / (channels * scaled_spatial_dim);
        int idx = (index % (channels * scaled_spatial_dim));
        int c = idx / scaled_spatial_dim;
        int s = (idx % scaled_spatial_dim)*step*2;
        int index_s_only = n*channels + c;
        int index_data = (n * channels + c) * spatial_dim + s;
        
        // Add.
        if (s + step < spatial_dim) {
            data[index_data] = max(data[index_data],data[index_data + step]);
            if (step*2 >= spatial_dim)
                out[index_s_only] = data[index_data];
        }
    }
}

template <typename Dtype>
__global__ void kernel_all_subtract(const int num, const int channels,
    const int spatial_dim, Dtype* data, const Dtype* channel_max) {
  CUDA_KERNEL_LOOP(index, num * channels * spatial_dim) {
    int n = index / (channels * spatial_dim);
    int idx = (index % (channels * spatial_dim));
    int c = idx / spatial_dim;
    int s = idx % spatial_dim;
    data[(n * channels + c) * spatial_dim + s] -= channel_max[n];
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
  CUDA_KERNEL_LOOP(index, num * channels * spatial_dim) {
    int n = index / (channels * spatial_dim);
    int idx = (index % (channels * spatial_dim));
    int c = idx / spatial_dim;
    int s = idx % spatial_dim;
    int index_s_only = n*channels + c;
    data[(n * channels + c) * spatial_dim + s] -= pixel_max[index_s_only];
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
    const int spatial_dim, const int p2numel, Dtype* data, Dtype* all_sum, unsigned int step) {
    int scaled_numel = p2numel/(2*step);
    for (int index = blockIdx.x*blockDim.x + threadIdx.x;
         index < num * scaled_numel;
         index += blockDim.x * gridDim.x) {
        int n = index / scaled_numel;
        int idx = (index % scaled_numel)*step*2;
        int index_data = n*channels*spatial_dim + idx;
        
        // Add.
        if (idx + step < channels*spatial_dim) {
            data[index_data] += data[index_data + step];
            if (step*2 >= channels*spatial_dim)
                all_sum[n] = data[index_data];
        }
    }
}

template <typename Dtype>
__global__ void kernel_pixel_sum(const int num, const int channels,
    const int spatial_dim, const int p2spatial_dim, Dtype* data, Dtype* pixel_sum, unsigned int step) {
    int scaled_spatial_dim = p2spatial_dim/(2*step);
    for (int index = blockIdx.x*blockDim.x + threadIdx.x;
         index < num * channels * scaled_spatial_dim;
         index += blockDim.x * gridDim.x) {
        int n = index / (channels * scaled_spatial_dim);
        int idx = (index % (channels * scaled_spatial_dim));
        int c = idx / scaled_spatial_dim;
        int s = (idx % scaled_spatial_dim)*step*2;
        int index_s_only = n*channels + c;
        int index_data = (n * channels + c) * spatial_dim + s;
        
        // Add.
        if (s + step < spatial_dim) {
            data[index_data] += data[index_data + step];
            if (step*2 >= spatial_dim)
                pixel_sum[index_s_only] = data[index_data];
        }
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
  CUDA_KERNEL_LOOP(index, num * channels * spatial_dim) {
    int n = index / (channels * spatial_dim);
    int idx = (index % (channels * spatial_dim));
    int c = idx / spatial_dim;
    int s = idx % spatial_dim;
    data[(n * channels + c) * spatial_dim + s] /= all_sum[n];
  }
}

template <typename Dtype>
__global__ void kernel_pixel_div(const int num, const int channels,
    const int spatial_dim, Dtype* data, const Dtype* pixel_sum) {
  CUDA_KERNEL_LOOP(index, num * channels * spatial_dim) {
    int n = index / (channels * spatial_dim);
    int idx = (index % (channels * spatial_dim));
    int c = idx / spatial_dim;
    int s = idx % spatial_dim;
    int index_s_only = n*channels + c;
    data[(n * channels + c) * spatial_dim + s] /= pixel_sum[index_s_only];
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
void SpatialSoftmaxLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  Dtype* scale_data = scale_.mutable_gpu_data();
  int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int spatial_dim = bottom[0]->height() * bottom[0]->width();
  int p2numel = pow(2,(int)ceil(log2((float)(channels*spatial_dim))));
  int p2spatial_dim = pow(2,(int)ceil(log2((float)spatial_dim)));
  // We need to subtract the max to avoid numerical issues, compute the exp,
  // and then normalize.
  caffe_copy(bottom[0]->count(), bottom_data, top_data);
  if (dimension_ == "spatial") {
    Dtype* temp_data = temp_data_.mutable_gpu_data();
    // compute max
    // NOLINT_NEXT_LINE(whitespace/operators)
    caffe_copy(bottom[0]->count(), top_data, temp_data);
    for (unsigned int step=1; step < spatial_dim; step *= 2) {
        int scaled_spatial_dim = p2spatial_dim/(2*step);
        kernel_pixel_max<Dtype><<<CAFFE_GET_BLOCKS(num * channels * scaled_spatial_dim),
            CAFFE_CUDA_NUM_THREADS>>>(num, channels, spatial_dim, p2spatial_dim,
            temp_data, scale_data, step);
    }
    // subtract
    // NOLINT_NEXT_LINE(whitespace/operators)
    kernel_pixel_subtract<Dtype><<<CAFFE_GET_BLOCKS(num * channels * spatial_dim),
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
    caffe_copy(bottom[0]->count(), top_data, temp_data);
    for (unsigned int step=1; step < spatial_dim; step *= 2) {
        int scaled_spatial_dim = p2spatial_dim/(2*step);
        kernel_pixel_sum<Dtype><<<CAFFE_GET_BLOCKS(num * channels * scaled_spatial_dim),
            CAFFE_CUDA_NUM_THREADS>>>(num, channels, spatial_dim, p2spatial_dim,
            temp_data, scale_data, step);
    }
    // divide
    // NOLINT_NEXT_LINE(whitespace/operators)
    kernel_pixel_div<Dtype><<<CAFFE_GET_BLOCKS(num * channels * spatial_dim),
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
    Dtype* temp_data = temp_data_.mutable_gpu_data();
    // compute max
    // NOLINT_NEXT_LINE(whitespace/operators)
    caffe_copy(bottom[0]->count(), top_data, temp_data);
    for (unsigned int step=1; step < spatial_dim * channels; step *= 2) {
        int scaled_numel = p2numel/(2*step);
        kernel_all_max<Dtype><<<CAFFE_GET_BLOCKS(num * scaled_numel),
             CAFFE_CUDA_NUM_THREADS>>>(num, channels, spatial_dim, p2numel,
             temp_data, scale_data, step);
    }
    // subtract
    // NOLINT_NEXT_LINE(whitespace/operators)
    kernel_all_subtract<Dtype><<<CAFFE_GET_BLOCKS(num * channels * spatial_dim),
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
    caffe_copy(bottom[0]->count(), top_data, temp_data);
    for (unsigned int step=1; step < spatial_dim * channels; step *= 2) {
        int scaled_numel = p2numel/(2*step);
        kernel_all_sum<Dtype><<<CAFFE_GET_BLOCKS(num * scaled_numel),
            CAFFE_CUDA_NUM_THREADS>>>(num, channels, spatial_dim, p2numel,
            temp_data, scale_data, step);
    }
    // divide
    // NOLINT_NEXT_LINE(whitespace/operators)
    kernel_all_div<Dtype><<<CAFFE_GET_BLOCKS(num * channels * spatial_dim),
        CAFFE_CUDA_NUM_THREADS>>>(num, channels, spatial_dim, top_data,
        scale_data);
  } else {
    LOG(FATAL) << "Unrecognized softmax dimension type";
  }
}

template <typename Dtype>
void SpatialSoftmaxLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
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
    kernel_pixel_subtract<Dtype><<<CAFFE_GET_BLOCKS(num * channels * spatial_dim),
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
    kernel_all_subtract<Dtype><<<CAFFE_GET_BLOCKS(num * channels * spatial_dim),
        CAFFE_CUDA_NUM_THREADS>>>(num, channels, spatial_dim, bottom_diff,
        scale_data);
  }
  // scale by temperature and elementwise multiplication
  caffe_gpu_scal<Dtype>(top[0]->count(), temp_, bottom_diff);
  caffe_gpu_mul<Dtype>(top[0]->count(), bottom_diff, top_data, bottom_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(SpatialSoftmaxLayer);


}  // namespace caffe
