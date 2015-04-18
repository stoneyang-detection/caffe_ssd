#include <algorithm>
#include <vector>
#include <cmath>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void NormalizeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_GE(bottom[0]->num_axes(), 2)
      << "Number of axes of bottom blob must be >=2.";
  squared_.Reshape(1, bottom[0]->channels(),
                   bottom[0]->height(), bottom[0]->width());
  NormalizeParameter norm_param = this->layer_param().norm_param();
  across_spatial_ = norm_param.across_spatial();
  if (across_spatial_) {
    norm_.Reshape(bottom[0]->num(), 1, 1, 1);
  } else {
    norm_.Reshape(bottom[0]->num(), 1, bottom[0]->height(), bottom[0]->width());
  }
  eps_ = norm_param.eps();
  int channels = bottom[0]->channels();
  channel_shared_ = norm_param.channel_shared();
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(1);
    if (channel_shared_) {
      this->blobs_[0].reset(new Blob<Dtype>(vector<int>(0)));
    } else {
      this->blobs_[0].reset(new Blob<Dtype>(vector<int>(1, channels)));
    }
    shared_ptr<Filler<Dtype> > scale_filler;
    if (norm_param.has_scale_filler()) {
      scale_filler.reset(GetFiller<Dtype>(norm_param.scale_filler()));
    } else {
      FillerParameter filler_param;
      filler_param.set_type("constant");
      filler_param.set_value(1.0);
      scale_filler.reset(GetFiller<Dtype>(filler_param));
    }
    scale_filler->Fill(this->blobs_[0].get());
  }
  if (channel_shared_) {
    CHECK_EQ(this->blobs_[0]->count(), 1)
        << "Scale size is inconsistent with prototxt config";
  } else {
    CHECK_EQ(this->blobs_[0]->count(), channels)
        << "Scale size is inconsistent with prototxt config";
  }

  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void NormalizeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_GE(bottom[0]->num_axes(), 2)
      << "Number of axes of bottom blob must be >=2.";
  top[0]->ReshapeLike(*bottom[0]);
  squared_.Reshape(1, bottom[0]->channels(),
                   bottom[0]->height(), bottom[0]->width());
  if (!across_spatial_) {
    norm_.Reshape(bottom[0]->num(), 1, bottom[0]->height(), bottom[0]->width());
  }
}

template <typename Dtype>
void NormalizeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* scale = this->blobs_[0]->cpu_data();
  Dtype* squared_data = squared_.mutable_cpu_data();
  Dtype* norm_data = norm_.mutable_cpu_data();
  caffe_set<Dtype>(norm_.count(), Dtype(0), norm_data);
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / num;
  int spatial_dim = bottom[0]->height() * bottom[0]->width();
  int channels = bottom[0]->channels();
  for (int n = 0; n < num; ++n) {
    caffe_sqr<Dtype>(dim, bottom_data, squared_data);
    if (across_spatial_) {
      // add eps to avoid overflow
      norm_data[n] = pow(caffe_cpu_asum<Dtype>(dim, squared_data)+eps_, Dtype(0.5));
      caffe_cpu_scale<Dtype>(dim, Dtype(1.0 / norm_data[n]), bottom_data, top_data);
    } else {
      for (int c = 0; c < channels; ++c) {
        caffe_add<Dtype>(spatial_dim, squared_data+c*spatial_dim, norm_data,
                         norm_data);
      }
      // add eps to avoid overflow
      caffe_add_scalar<Dtype>(spatial_dim, eps_, norm_data);
      caffe_powx<Dtype>(spatial_dim, norm_data, Dtype(0.5), norm_data);
      for (int c = 0; c < channels; ++c) {
        caffe_div<Dtype>(spatial_dim, bottom_data+c*spatial_dim, norm_data,
                         top_data+c*spatial_dim);
      }
      norm_data += spatial_dim;
    }
    // scale the output
    if (channel_shared_) {
      caffe_scal<Dtype>(dim, scale[0], top_data);
    } else {
      for (int c = 0; c < channels; ++c) {
        caffe_scal<Dtype>(spatial_dim, scale[c], top_data+c*spatial_dim);
      }
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
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const Dtype* scale = this->blobs_[0]->cpu_data();
  const Dtype* norm_data = norm_.cpu_data();
  Dtype* squared_data = squared_.mutable_cpu_data();
  int count = top[0]->count();
  int num = top[0]->num();
  int dim = count / num;
  int spatial_dim = top[0]->height() * top[0]->width();
  int channels = top[0]->channels();
  
  // Propagate to param
  if (this->param_propagate_down_[0]) {
    Dtype* scale_diff = this->blobs_[0]->mutable_cpu_diff();
    if (channel_shared_) {
      scale_diff[0] = caffe_cpu_dot<Dtype>(count, top_data, top_diff) / scale[0];
    } else {
      caffe_set(this->blobs_[0]->count(), Dtype(0), scale_diff);
      for (int n = 0; n < num; ++n) {
        caffe_mul<Dtype>(dim, top_data+n*dim, top_diff+n*dim, squared_data);
        for (int c = 0; c < channels; ++c) {
          scale_diff[c] += caffe_cpu_asum<Dtype>(spatial_dim,
                                        squared_data+c*spatial_dim) / scale[c];
        }
      }
    }
  }

  // Propagate to bottom
  if (propagate_down[0]) {
    for (int n = 0; n < num; ++n) {
      if (across_spatial_) {
        Dtype a = caffe_cpu_dot<Dtype>(dim, bottom_data, top_diff);
        caffe_cpu_scale<Dtype>(dim, a / norm_data[n] / norm_data[n], bottom_data,
                               bottom_diff);
        caffe_sub<Dtype>(dim, top_diff, bottom_diff, bottom_diff);
        caffe_cpu_scale<Dtype>(dim, Dtype(1.0 / norm_data[n]), bottom_diff,
                               bottom_diff);
      } else {
        // use squared_data to store temp result
        // dot product between bottom_data and top_diff
        caffe_mul<Dtype>(dim, bottom_data, top_diff, squared_data);
        for (int c = 1; c < channels; ++c) {
          caffe_add<Dtype>(spatial_dim, squared_data+c*spatial_dim, squared_data,
                           squared_data);
        }
        // scale bottom_diff
        for (int c = 0; c < channels; ++c) {
          caffe_mul<Dtype>(spatial_dim, bottom_data+c*spatial_dim, squared_data,
                           bottom_diff+c*spatial_dim);
        }
        // divide by square of norm
        caffe_powx<Dtype>(spatial_dim, norm_data, Dtype(2), squared_data);
        for (int c = 0; c < channels; ++c) {
          caffe_div<Dtype>(spatial_dim, bottom_diff+c*spatial_dim, squared_data,
                           bottom_diff+c*spatial_dim);
        }
        caffe_sub<Dtype>(dim, top_diff, bottom_diff, bottom_diff);
        // divide by norm
        for (int c = 0; c < channels; ++c) {
          caffe_div<Dtype>(spatial_dim, bottom_diff+c*spatial_dim, norm_data,
                           bottom_diff+c*spatial_dim);
        }
        norm_data += spatial_dim;
      }
      // scale the diff
      if (channel_shared_) {
        caffe_scal<Dtype>(dim, scale[0], bottom_diff);
      } else {
        for (int c = 0; c < channels; ++c) {
          caffe_scal<Dtype>(spatial_dim, scale[c], bottom_diff+c*spatial_dim);
        }
      }
      bottom_data += dim;
      top_diff += dim;
      bottom_diff += dim;
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(NormalizeLayer);
#endif

INSTANTIATE_CLASS(NormalizeLayer);
REGISTER_LAYER_CLASS(Normalize);

}  // namespace caffe
