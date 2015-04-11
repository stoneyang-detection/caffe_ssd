#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void UnPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  UnPoolingParameter unpool_param = this->layer_param_.unpooling_param();
  if (unpool_param.unpool() == UnPoolingParameter_UnPoolMethod_GROUP) {
    CHECK_EQ(bottom.size(), 2);
    CHECK_EQ(bottom[0]->width(), bottom[1]->width());
    CHECK_EQ(bottom[0]->height(), bottom[1]->height());
    CHECK_EQ(bottom[0]->num(), bottom[1]->num());
    out_kernel_h_ = out_kernel_w_ = 1;
    out_stride_h_ = out_stride_w_ = 1;
    out_pad_h_ = out_pad_w_ = 0;
    return;
  }
  CHECK(!unpool_param.has_out_kernel_size() !=
      !(unpool_param.has_out_kernel_h() && unpool_param.has_out_kernel_w()))
      << "Out filter size is out_kernel_size OR out_kernel_h and out_kernel_w; "
      << "not both";
  CHECK(unpool_param.has_out_kernel_size() ||
      (unpool_param.has_out_kernel_h() && unpool_param.has_out_kernel_w()))
      << "For non-square filters both out_kernel_h and out_kernel_w are "
      << "required.";
  CHECK((!unpool_param.has_out_pad() && unpool_param.has_out_pad_h()
      && unpool_param.has_out_pad_w())
      || (!unpool_param.has_out_pad_h() && !unpool_param.has_out_pad_w()))
      << "Out pad is out_pad OR out_pad_h and out_pad_w are required.";
  CHECK((!unpool_param.has_out_stride() && unpool_param.has_out_stride_h()
      && unpool_param.has_out_stride_w())
      || (!unpool_param.has_out_stride_h() && !unpool_param.has_out_stride_w()))
      << "Out stride is out_stride OR out_stride_h and out_stride_w are "
      << "required.";
  if (bottom.size() == 1) {
    if (unpool_param.has_out_kernel_size()) {
      out_kernel_h_ = out_kernel_w_ = unpool_param.out_kernel_size();
    } else {
      out_kernel_h_ = unpool_param.out_kernel_h();
      out_kernel_w_ = unpool_param.out_kernel_w();
    }
    CHECK_GT(out_kernel_h_, 0) << "Out filter dimensions cannot be zero.";
    CHECK_GT(out_kernel_w_, 0) << "Out filter dimensions cannot be zero.";
    if (!unpool_param.has_out_stride_h()) {
      out_stride_h_ = out_stride_w_ = unpool_param.out_stride();
    } else {
      out_stride_h_ = unpool_param.out_stride_h();
      out_stride_w_ = unpool_param.out_stride_w();
    }
    if (!unpool_param.has_out_pad_h()) {
      out_pad_h_ = out_pad_w_ = unpool_param.out_pad();
    } else {
      out_pad_h_ = unpool_param.out_pad_h();
      out_pad_w_ = unpool_param.out_pad_w();
    }
  } else {
    // Compute out_kernel and out_stride automatically
    out_kernel_h_ = static_cast<int>(ceil(static_cast<float>(
                bottom[1]->height()) / bottom[0]->height()));
    out_kernel_w_ = static_cast<int>(ceil(static_cast<float>(
                bottom[1]->width()) / bottom[0]->width()));

    out_stride_h_ = static_cast<int>(ceil(static_cast<float>(
                bottom[1]->height()) / bottom[0]->height()));
    out_stride_w_ = static_cast<int>(ceil(static_cast<float>(
                bottom[1]->width()) / bottom[0]->width()));

    // In case either width or height of bottom[0] is 1, we set stride to 1
    if (out_stride_h_ == bottom[1]->height()) {
      out_stride_h_ = 1;
    }
    if (out_stride_w_ == bottom[1]->width()) {
      out_stride_w_ = 1;
    }

    out_pad_h_ = static_cast<int>(floor(static_cast<float>(
                (bottom[0]->height()-1)*out_stride_h_+out_kernel_h_
                -bottom[1]->height())/2));
    out_pad_w_ = static_cast<int>(floor(static_cast<float>(
                (bottom[0]->width()-1)*out_stride_w_+out_kernel_w_
                -bottom[1]->width())/2));
  }
  if (out_pad_h_ != 0 || out_pad_w_ != 0) {
    CHECK(this->layer_param_.unpooling_param().unpool()
        != UnPoolingParameter_UnPoolMethod_GROUP)
        << "Out padding is not implemented for GROUP unpooling.";
    CHECK_LT(out_pad_h_, out_kernel_h_);
    CHECK_LT(out_pad_w_, out_kernel_w_);
  }
}

template <typename Dtype>
void UnPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // reset the out_kernel_size and out_stride, etc.
  this->LayerSetUp(bottom, top);
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();

  // special operation for GROUP
  UnPoolingParameter unpool_param = this->layer_param_.unpooling_param();
  if (unpool_param.unpool() == UnPoolingParameter_UnPoolMethod_GROUP) {
    top[0]->Reshape(num_, channels_, height_, width_);
    unpooled_height_ = height_;
    unpooled_width_ = width_;
    // get the map from pixel index to group index
    group_channels_ = bottom[1]->channels();
    const Dtype* group_data = bottom[1]->cpu_data();
    // map group_data to [0, num_group - 1] and store it at group_blob_
    group_blob_.ReshapeLike(*bottom[1]);
    Dtype* group_blob_data = group_blob_.mutable_cpu_data();
    for (int n = 0; n < num_; ++n) {
      for (int c = 0; c < group_channels_; ++c) {
        map<int, int> group_id_map;
        group_id_map.clear();
        int count = -1;
        for (int h = 0; h < height_; ++h) {
          for (int w = 0; w < width_; ++w) {
            int index = ((n * group_channels_ + c) * height_ + h) * width_ + w;
            int group_id = group_data[index];
            if (group_id_map.find(group_id) == group_id_map.end()) {
              ++count;
              group_id_map[group_id] = count;
              group_blob_data[index] = count;
            } else {
              group_blob_data[index] = group_id_map[group_id];
            }
          }
        }
      }
    }
    // find map between group_id and index
    group_maps_vec_.clear();
    for (int n = 0; n < num_; ++n) {
      vector<map<int, vector<int> > > group_maps;
      group_maps.clear();
      for (int gc = 0; gc < group_channels_; ++gc) {
        map<int, vector<int> > group_map;
        group_map.clear();
        for (int h = 0; h < height_; ++h) {
          for (int w = 0; w < width_; ++w) {
            int index = h * width_ + w;
            int group_index = (int)group_blob_data[index];
            group_map[group_index].push_back(index);
          }
        }
        group_blob_data += group_blob_.offset(0, 1);
        group_maps.push_back(group_map);
      }
      group_maps_vec_.push_back(group_maps);
    }
  } else {
    // deal with other cases
    unpooled_height_ = (height_ - 1) * out_stride_h_ - 2 * out_pad_h_ +
        out_kernel_h_;
    unpooled_width_ = (width_ - 1) * out_stride_w_ - 2 * out_pad_w_ +
        out_kernel_w_;
    top[0]->Reshape(num_, channels_, unpooled_height_, unpooled_width_);
  }

  // fill the mask
  this->FillMask();
}

template <typename Dtype>
void UnPoolingLayer<Dtype>::FillMask() {
  // Different unpool method needs different mask, but they are same across
  // channels and samples
  mask_.Reshape(1, 1, unpooled_height_, unpooled_width_);
  int* mask = mask_.mutable_cpu_data();
  switch (this->layer_param_.unpooling_param().unpool()) {
  case UnPoolingParameter_UnPoolMethod_FIXED:
    // mask_ records map of positions from bottom to top
    caffe_set(mask_.count(), -1, mask);
    for (int h = 0; h < height_; ++h) {
      for (int w = 0; w < width_; ++w) {
        int uhstart = h * out_stride_h_ - out_pad_h_;
        int uwstart = w * out_stride_w_ - out_pad_w_;
        int uhend = uhstart + out_kernel_h_;
        int uwend = uwstart + out_kernel_w_;
        int uhmid = floor((uhstart + uhend - 1) / 2);
        int uwmid = floor((uwstart + uwend - 1) / 2);
        uhmid = min(max(uhmid, 0), unpooled_height_-1);
        uwmid = min(max(uwmid, 0), unpooled_width_-1);
        const int unpool_index = uhmid * unpooled_width_ + uwmid;
        const int index = h * width_ + w;
        mask[unpool_index] = index;
      }
    }
    break;
  case UnPoolingParameter_UnPoolMethod_DIV:
  case UnPoolingParameter_UnPoolMethod_REP:
    // mask_ records counts of contributions to each unpooled position
    // same for DIV and REP unpool operation
    caffe_set(mask_.count(), 0, mask);
    for (int h = 0; h < height_; ++h) {
      for (int w = 0; w < width_; ++w) {
        int uhstart = h * out_stride_h_ - out_pad_h_;
        int uwstart = w * out_stride_w_ - out_pad_w_;
        int uhend = min(uhstart + out_kernel_h_, unpooled_height_);
        int uwend = min(uwstart + out_kernel_w_, unpooled_width_);
        uhstart = max(uhstart, 0);
        uwstart = max(uwstart, 0);
        for (int uh = uhstart; uh < uhend; ++uh) {
          for (int uw = uwstart; uw < uwend; ++uw) {
            const int unpool_index = uh * unpooled_width_ + uw;
            mask[unpool_index] += 1;
          }
        }
      }
    }
    break;
  case UnPoolingParameter_UnPoolMethod_GROUP:
    caffe_set(mask_.count(), (int)group_channels_, mask);
    break;
  default:
    LOG(FATAL) << "Unknown unpooling method.";
  }
}

template <typename Dtype>
void UnPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int top_count = top[0]->count();
  caffe_set(top_count, Dtype(0), top_data);
  const int* mask = mask_.cpu_data();
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more code.
  switch (this->layer_param_.unpooling_param().unpool()) {
  case UnPoolingParameter_UnPoolMethod_FIXED:
    // The main loop
    for (int n = 0; n < num_; ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int h = 0; h < height_; ++h) {
          for (int w = 0; w < width_; ++w) {
            int uhstart = h * out_stride_h_ - out_pad_h_;
            int uwstart = w * out_stride_w_ - out_pad_w_;
            int uhend = uhstart + out_kernel_h_;
            int uwend = uwstart + out_kernel_w_;
            int uhmid = floor((uhstart + uhend - 1) / 2);
            int uwmid = floor((uwstart + uwend - 1) / 2);
            uhmid = min(max(uhmid, 0), unpooled_height_-1);
            uwmid = min(max(uwmid, 0), unpooled_width_-1);
            const int unpool_index = uhmid * unpooled_width_ + uwmid;
            const int index = h * width_ + w;
            top_data[unpool_index] = bottom_data[index];
          }
        }
        // compute offset
        bottom_data += bottom[0]->offset(0, 1);
        top_data += top[0]->offset(0, 1);
      }
    }
    break;
  case UnPoolingParameter_UnPoolMethod_DIV:
    // The main loop
    for (int n = 0; n < num_; ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int h = 0; h < height_; ++h) {
          for (int w = 0; w < width_; ++w) {
            int uhstart = h * out_stride_h_ - out_pad_h_;
            int uwstart = w * out_stride_w_ - out_pad_w_;
            int uhend = min(uhstart + out_kernel_h_,
                            unpooled_height_ + out_pad_h_);
            int uwend = min(uwstart + out_kernel_w_,
                            unpooled_width_ + out_pad_w_);
            int unpool_size = (uhend - uhstart) * (uwend - uwstart);
            uhstart = max(uhstart, 0);
            uwstart = max(uwstart, 0);
            uhend = min(uhend, unpooled_height_);
            uwend = min(uwend, unpooled_width_);
            Dtype div_data = bottom_data[h * width_ + w] / unpool_size;
            for (int uh = uhstart; uh < uhend; ++uh) {
              for (int uw = uwstart; uw < uwend; ++uw) {
                int unpool_index = uh * unpooled_width_ + uw;
                CHECK_GT(mask[unpool_index], 0);
                top_data[unpool_index] += div_data / mask[unpool_index];
              }
            }
          }
        }
        // compute offset
        bottom_data += bottom[0]->offset(0, 1);
        top_data += top[0]->offset(0, 1);
      }
    }
    break;
  case UnPoolingParameter_UnPoolMethod_REP:
    // The main loop
    for (int n = 0; n < num_; ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int h = 0; h < height_; ++h) {
          for (int w = 0; w < width_; ++w) {
            int uhstart = h * out_stride_h_ - out_pad_h_;
            int uwstart = w * out_stride_w_ - out_pad_w_;
            int uhend = min(uhstart + out_kernel_h_,
                            unpooled_height_ + out_pad_h_);
            int uwend = min(uwstart + out_kernel_w_,
                            unpooled_width_ + out_pad_w_);
            uhstart = max(uhstart, 0);
            uwstart = max(uwstart, 0);
            uhend = min(uhend, unpooled_height_);
            uwend = min(uwend, unpooled_width_);
            Dtype data = bottom_data[h * width_ + w];
            for (int uh = uhstart; uh < uhend; ++uh) {
              for (int uw = uwstart; uw < uwend; ++uw) {
                int unpool_index = uh * unpooled_width_ + uw;
                CHECK_GT(mask[unpool_index], 0);
                top_data[unpool_index] += data / mask[unpool_index];
              }
            }
          }
        }
        // compute offset
        bottom_data += bottom[0]->offset(0, 1);
        top_data += top[0]->offset(0, 1);
      }
    }
    break;
  case UnPoolingParameter_UnPoolMethod_GROUP:
    // the main loop
    for (int n = 0; n < num_; ++n) {
      vector<map<int, vector<int> > >& group_maps = group_maps_vec_[n];
      for (int gc = 0; gc < group_channels_; ++gc) {
        map<int, vector<int> >& group_map = group_maps[gc];
        group_mean_.Reshape(1, 1, 1, group_map.size());
        Dtype* group_mean_data = group_mean_.mutable_cpu_data();
        // reset bottom_data and top_data
        bottom_data = bottom[0]->cpu_data() + bottom[0]->offset(n);
        top_data = top[0]->mutable_cpu_data() + top[0]->offset(n);
        for (int c = 0; c < channels_; ++c) {
          caffe_set(group_mean_.count(), Dtype(0), group_mean_data);
          // compute the mean for each group
          int group_id = 0;
          for (map<int, vector<int> >::iterator it = group_map.begin();
               it != group_map.end(); ++it) {
            for (int s = 0; s < it->second.size(); ++s) {
              group_mean_data[group_id] += bottom_data[it->second[s]];
            }
            group_mean_data[group_id] /= it->second.size() * bottom[1]->channels();
            ++group_id;
          }
          // assign the mean to top_data
          group_id = 0;
          for (map<int, vector<int> >::iterator it = group_map.begin();
               it != group_map.end(); ++it) {
            for (int s = 0; s < it->second.size(); ++s) {
              top_data[it->second[s]] += group_mean_data[group_id];
            }
            ++group_id;
          }
          // compute offset
          bottom_data += bottom[0]->offset(0, 1);
          top_data += top[0]->offset(0, 1);
        }
      }
    }
    break;
  default:
    LOG(FATAL) << "Unknown unpooling method.";
  }
}

template <typename Dtype>
void UnPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
  // Different unpooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more codes.
  const int* mask = mask_.cpu_data();
  switch (this->layer_param_.unpooling_param().unpool()) {
  case UnPoolingParameter_UnPoolMethod_FIXED:
    // The main loop
    for (int n = 0; n < num_; ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int uh = 0; uh < unpooled_height_; ++uh) {
          for (int uw = 0; uw < unpooled_width_; ++uw) {
            const int unpool_index = uh * unpooled_width_ + uw;
            const int index = mask[unpool_index];
            if (index != -1) {
              bottom_diff[index] = top_diff[unpool_index];
            }
          }
        }
        bottom_diff += bottom[0]->offset(0, 1);
        top_diff += top[0]->offset(0, 1);
      }
    }
    break;
  case UnPoolingParameter_UnPoolMethod_DIV:
    // The main loop
    for (int n = 0; n < num_; ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int h = 0; h < height_; ++h) {
          for (int w = 0; w < width_; ++w) {
            int uhstart = h * out_stride_h_ - out_pad_h_;
            int uwstart = w * out_stride_w_ - out_pad_w_;
            int uhend = min(uhstart + out_kernel_h_, unpooled_height_ + out_pad_h_);
            int uwend = min(uwstart + out_kernel_w_, unpooled_width_ + out_pad_w_);
            int unpool_size = (uhend - uhstart) * (uwend - uwstart);
            uhstart = max(uhstart, 0);
            uwstart = max(uwstart, 0);
            uhend = min(uhend, unpooled_height_);
            uwend = min(uwend, unpooled_width_);
            for (int uh = uhstart; uh < uhend; ++uh) {
              for (int uw = uwstart; uw < uwend; ++uw) {
                const int unpool_index = uh * unpooled_width_ + uw;
                CHECK_GT(mask[unpool_index], 0);
                bottom_diff[h * width_ + w] +=
                  top_diff[unpool_index] / unpool_size / mask[unpool_index];
              }
            }
          }
        }
        // offset
        bottom_diff += bottom[0]->offset(0, 1);
        top_diff += top[0]->offset(0, 1);
      }
    }
    break;
  case UnPoolingParameter_UnPoolMethod_REP:
    // The main loop
    for (int n = 0; n < num_; ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int h = 0; h < height_; ++h) {
          for (int w = 0; w < width_; ++w) {
            int uhstart = h * out_stride_h_ - out_pad_h_;
            int uwstart = w * out_stride_w_ - out_pad_w_;
            int uhend = min(uhstart + out_kernel_h_,
                            unpooled_height_ + out_pad_h_);
            int uwend = min(uwstart + out_kernel_w_,
                            unpooled_width_ + out_pad_w_);
            uhstart = max(uhstart, 0);
            uwstart = max(uwstart, 0);
            uhend = min(uhend, unpooled_height_);
            uwend = min(uwend, unpooled_width_);
            for (int uh = uhstart; uh < uhend; ++uh) {
              for (int uw = uwstart; uw < uwend; ++uw) {
                const int unpool_index = uh * unpooled_width_ + uw;
                CHECK_GT(mask[unpool_index], 0);
                bottom_diff[h * width_ + w] +=
                  top_diff[unpool_index] / mask[unpool_index];
              }
            }
          }
        }
        // offset
        bottom_diff += bottom[0]->offset(0, 1);
        top_diff += top[0]->offset(0, 1);
      }
    }
    break;
  case UnPoolingParameter_UnPoolMethod_GROUP:
    // the main loop
    for (int n = 0; n < num_; ++n) {
      vector<map<int, vector<int> > >& group_maps = group_maps_vec_[n];
      for (int c = 0; c < channels_; ++c) {
        for (int gc = 0; gc < group_channels_; ++gc) {
          map<int, vector<int> >& group_map = group_maps[gc];
          for (map<int, vector<int> >::iterator it = group_map.begin();
               it != group_map.end(); ++it) {
            for (int s1 = 0; s1 < it->second.size(); ++s1) {
              int index1 = it->second[s1];
              for (int s2 = 0; s2 < it->second.size(); ++s2) {
                int index2 = it->second[s2];
                bottom_diff[index1] +=
                    top_diff[index2] / it->second.size() / group_channels_;
              }
            }
          }
        }
        // compute offset
        bottom_diff += bottom[0]->offset(0, 1);
        top_diff += top[0]->offset(0, 1);
      }
    }
    break;
  default:
    LOG(FATAL) << "Unknown unpooling method.";
  }
}


#ifdef CPU_ONLY
STUB_GPU(UnPoolingLayer);
#endif

INSTANTIATE_CLASS(UnPoolingLayer);
REGISTER_LAYER_CLASS(UnPooling);

}  // namespace caffe
