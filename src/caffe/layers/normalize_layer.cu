#include <algorithm>
#include <cfloat>
#include <vector>
#include <limits>

#include "thrust/device_vector.h"

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void NormalizeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  Dtype* squared_data = squared_.mutable_gpu_data();
  Dtype* norm_data = norm_.mutable_gpu_data();
  caffe_gpu_set(norm_.count(), Dtype(0), norm_data);
  Dtype normsqr;
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / num;
  int spatial_dim = bottom[0]->height() * bottom[0]->width();
  int channels = bottom[0]->channels();
  // add eps to avoid overflow
  Dtype eps = std::numeric_limits<Dtype>::epsilon();
  for (int n = 0; n < num; ++n) {
    caffe_gpu_powx<Dtype>(dim, bottom_data, Dtype(2), squared_data);
    if (across_spatial_) {
      caffe_gpu_asum<Dtype>(dim, squared_data, &normsqr);
      caffe_gpu_scale<Dtype>(dim, Dtype(1)/(pow(normsqr, Dtype(0.5))+eps),
          bottom_data, top_data);
    } else {
      for (int c = 0; c < channels; ++c) {
        caffe_gpu_add<Dtype>(spatial_dim, squared_data+c*spatial_dim, norm_data,
            norm_data);
      }
      caffe_gpu_powx<Dtype>(spatial_dim, norm_data, Dtype(0.5), norm_data);
      for (int c = 0; c < channels; ++c) {
        caffe_gpu_div<Dtype>(spatial_dim, bottom_data+c*spatial_dim, norm_data,
            top_data+c*spatial_dim);
      }
      norm_data += spatial_dim;
    }
    bottom_data += dim;
    top_data += dim;
  }
}

template <typename Dtype>
void NormalizeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* top_data = top[0]->gpu_data();
  const Dtype* bottom_data = bottom[0]->mutable_gpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const Dtype* norm_data = norm_.gpu_data();
  Dtype* squared_data = squared_.mutable_gpu_data();
  int num = top[0]->num();
  int dim = top[0]->count() / num;
  int spatial_dim = top[0]->height() * top[0]->width();
  int channels = top[0]->channels();
  Dtype eps = std::numeric_limits<Dtype>::epsilon();
  for (int n = 0; n < num; ++n) {
    if (across_spatial_) {
      Dtype a;
      caffe_gpu_dot<Dtype>(dim, top_data, top_diff, &a);
      caffe_gpu_scale<Dtype>(dim, a, top_data, bottom_diff);
      caffe_gpu_sub<Dtype>(dim, top_diff, bottom_diff, bottom_diff);
      caffe_gpu_dot<Dtype>(dim, bottom_data, bottom_data, &a);
      caffe_gpu_scale<Dtype>(dim, Dtype(1)/(pow(a,Dtype(0.5))+eps), bottom_diff,
          bottom_diff);
    } else {
      // use squared_data to store temp result
      // dot product between top_data and top_diff
      caffe_gpu_mul<Dtype>(dim, top_data, top_diff, squared_data);
      for (int c = 1; c < channels; ++c) {
        caffe_gpu_add<Dtype>(spatial_dim, squared_data+c*spatial_dim, squared_data,
            squared_data);
      }
      // scale bottom_diff
      for (int c = 0; c < channels; ++c) {
        caffe_gpu_mul<Dtype>(spatial_dim, top_data+c*spatial_dim, squared_data,
            bottom_diff+c*spatial_dim);
      }
      caffe_gpu_sub<Dtype>(dim, top_diff, bottom_diff, bottom_diff);
      // divide by norm
      for (int c = 0; c < channels; ++c) {
        caffe_gpu_div<Dtype>(spatial_dim, bottom_diff+c*spatial_dim, norm_data,
            bottom_diff+c*spatial_dim);
      }
      norm_data += spatial_dim;
    }
    top_data += dim;
    top_diff += dim;
    bottom_diff += dim;
    bottom_data += dim;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(NormalizeLayer);


}  // namespace caffe
