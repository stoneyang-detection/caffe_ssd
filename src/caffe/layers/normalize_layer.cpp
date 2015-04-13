#include <algorithm>
#include <vector>
#include <cmath>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void NormalizeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
  squared_.Reshape(1, bottom[0]->channels(),
    bottom[0]->height(), bottom[0]->width());
  across_spatial_ = this->layer_param_.norm_param().across_spatial();
  if (across_spatial_) {
    norm_.Reshape(bottom[0]->num(), 1, 1, 1);
  } else {
    norm_.Reshape(bottom[0]->num(), 1, bottom[0]->height(), bottom[0]->width());
  }
  scale_ = this->layer_param_.norm_param().scale();
  CHECK_GT(scale_, Dtype(0));
}

template <typename Dtype>
void NormalizeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* squared_data = squared_.mutable_cpu_data();
  Dtype* norm_data = norm_.mutable_cpu_data();
  caffe_set<Dtype>(norm_.count(), Dtype(0), norm_data);
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / num;
  int spatial_dim = bottom[0]->height() * bottom[0]->width();
  int channels = bottom[0]->channels();
  Dtype eps = 1e-10;
  for (int n = 0; n < num; ++n) {
    caffe_sqr<Dtype>(dim, bottom_data, squared_data);
    if (across_spatial_) {
      norm_data[n] = pow(caffe_cpu_asum<Dtype>(dim, squared_data), Dtype(0.5)) + eps;
      caffe_cpu_scale<Dtype>(dim, scale_ / norm_data[n], bottom_data, top_data);
    } else {
      for (int c = 0; c < channels; ++c) {
        caffe_add<Dtype>(spatial_dim, squared_data+c*spatial_dim, norm_data,
                         norm_data);
      }
      caffe_powx<Dtype>(spatial_dim, norm_data, Dtype(0.5), norm_data);
      // add eps to avoid overflow
      caffe_add_scalar<Dtype>(spatial_dim, eps, norm_data);
      for (int c = 0; c < channels; ++c) {
        caffe_div<Dtype>(spatial_dim, bottom_data+c*spatial_dim, norm_data,
                         top_data+c*spatial_dim);
      }
      if (scale_ != 1) {
        caffe_cpu_scale<Dtype>(dim, scale_, top_data, top_data);
      }
      norm_data += spatial_dim;
    }
    bottom_data += dim;
    top_data += dim;
  }
}

template <typename Dtype>
void NormalizeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* top_data = top[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const Dtype* norm_data = norm_.cpu_data();
  Dtype* squared_data = squared_.mutable_cpu_data();
  int num = top[0]->num();
  int dim = top[0]->count() / num;
  int spatial_dim = top[0]->height() * top[0]->width();
  int channels = top[0]->channels();
  for (int n = 0; n < num; ++n) {
    if (across_spatial_) {
      Dtype a = caffe_cpu_dot<Dtype>(dim, top_data, top_diff);
      caffe_cpu_scale<Dtype>(dim, a / scale_ / scale_, top_data, bottom_diff);
      caffe_sub<Dtype>(dim, top_diff, bottom_diff, bottom_diff);
      CHECK_GT(norm_data[n], 0) << "norm should larger than 0";
      caffe_cpu_scale<Dtype>(dim, scale_ / norm_data[n], bottom_diff,
                             bottom_diff);
    } else {
      // use squared_data to store temp result
      // dot product between top_data and top_diff
      caffe_mul<Dtype>(dim, top_data, top_diff, squared_data);
      for (int c = 1; c < channels; ++c) {
        caffe_add<Dtype>(spatial_dim, squared_data+c*spatial_dim, squared_data,
                         squared_data);
      }
      // scale bottom_diff
      for (int c = 0; c < channels; ++c) {
        caffe_mul<Dtype>(spatial_dim, top_data+c*spatial_dim, squared_data,
                         bottom_diff+c*spatial_dim);
      }
      if (scale_ != 1) {
        caffe_cpu_scale<Dtype>(dim, 1 / scale_ / scale_, bottom_diff, bottom_diff);
      }
      caffe_sub<Dtype>(dim, top_diff, bottom_diff, bottom_diff);
      // divide by norm
      for (int c = 0; c < channels; ++c) {
        caffe_div<Dtype>(spatial_dim, bottom_diff+c*spatial_dim, norm_data,
                         bottom_diff+c*spatial_dim);
      }
      if (scale_ != 1) {
        caffe_cpu_scale<Dtype>(dim, scale_, bottom_diff, bottom_diff);
      }
      norm_data += spatial_dim;
    }
    top_data += dim;
    top_diff += dim;
    bottom_diff += dim;
  }
}


#ifdef CPU_ONLY
STUB_GPU(NormalizeLayer);
#endif

INSTANTIATE_CLASS(NormalizeLayer);
REGISTER_LAYER_CLASS(Normalize);

}  // namespace caffe
