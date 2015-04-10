#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void FixedUnPoolForward(const int nthreads, const Dtype* bottom_data,
    const int num, const int channels, const int height, const int width,
    const int unpooled_height, const int unpooled_width, const int out_kernel_h,
    const int out_kernel_w, const int out_stride_h, const int out_stride_w,
    const int out_pad_h, const int out_pad_w, Dtype* top_data) {
  CUDA_KERNEL_LOOP(unpool_index, nthreads) {
    int uw = unpool_index % unpooled_width;
    int uh = (unpool_index / unpooled_width) % unpooled_height;
    int c = (unpool_index / unpooled_width / unpooled_height) % channels;
    int n = unpool_index / unpooled_width / unpooled_height / channels;
    int hstart = (uh + out_pad_h < out_kernel_h) ? 0 :
      (uh + out_pad_h - out_kernel_h) / out_stride_h + 1;
    int hend = min((uh + out_pad_h) / out_stride_h + 1, height);
    int wstart = (uw + out_pad_w < out_kernel_w) ? 0 :
      (uw + out_pad_w - out_kernel_w) / out_stride_w + 1;
    int wend = min((uw + out_pad_w) / out_stride_w + 1, width);
    int offset = (n * channels + c) * height * width;
    int unpool_offset = (n * channels + c) * unpooled_height * unpooled_width;
    bottom_data += offset;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        int uhstart = h * out_stride_h - out_pad_h;
        int uwstart = w * out_stride_w - out_pad_w;
        int uhend = uhstart + out_kernel_h;
        int uwend = uwstart + out_kernel_w;
        int uhmid = (uhstart + uhend - 1) / 2;
        int uwmid = (uwstart + uwend - 1) / 2;
        uhmid = min(max(uhmid, 0), unpooled_height);
        uwmid = min(max(uwmid, 0), unpooled_width);
        if (unpool_offset + uhmid * unpooled_width + uwmid == unpool_index) {
          // find the mapping, assign & return
          int index = h * width + w;
          top_data[unpool_index] = bottom_data[index];
          return;
        }
      }
    }
  }
}

template <typename Dtype>
__global__ void DivUnPoolForward(const int nthreads, const Dtype* bottom_data,
    const int* mask, const int num, const int channels, const int height,
    const int width, const int unpooled_height, const int unpooled_width,
    const int out_kernel_h, const int out_kernel_w, const int out_stride_h,
    const int out_stride_w, const int out_pad_h, const int out_pad_w,
    Dtype* top_data) {
  CUDA_KERNEL_LOOP(unpool_index, nthreads) {
    int uw = unpool_index % unpooled_width + out_pad_w;
    int uh = (unpool_index / unpooled_width) % unpooled_height + out_pad_h;
    int c = (unpool_index / unpooled_width / unpooled_height) % channels;
    int n = unpool_index / unpooled_width / unpooled_height / channels;
    int spatial_dim = unpooled_height * unpooled_width;
    int hstart = (uh < out_kernel_h) ? 0 :
      (uh - out_kernel_h) / out_stride_h + 1;
    int hend = min(uh / out_stride_h + 1, height);
    int wstart = (uw < out_kernel_w) ? 0 : 
      (uw - out_kernel_w) / out_stride_w + 1;
    int wend = min(uw / out_stride_w + 1, width);
    Dtype divval = 0;
    bottom_data += (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        int uhstart = h * out_stride_h - out_pad_h;
        int uwstart = w * out_stride_w - out_pad_w;
        int uhend = min(uhstart + out_kernel_h, unpooled_height + out_pad_h);
        int uwend = min(uwstart + out_kernel_w, unpooled_width + out_pad_w);
        int unpool_size = (uhend - uhstart) * (uwend - uwstart);
        divval += bottom_data[h * width + w] / unpool_size;
      }
    }
    top_data[unpool_index] = divval / mask[unpool_index % spatial_dim];
  }
}

template <typename Dtype>
__global__ void RepUnPoolForward(const int nthreads, const Dtype* bottom_data,
    const int* mask, const int num, const int channels, const int height,
    const int width, const int unpooled_height, const int unpooled_width,
    const int out_kernel_h, const int out_kernel_w, const int out_stride_h,
    const int out_stride_w, const int out_pad_h, const int out_pad_w,
    Dtype* top_data) {
  CUDA_KERNEL_LOOP(unpool_index, nthreads) {
    int uw = unpool_index % unpooled_width + out_pad_w;
    int uh = (unpool_index / unpooled_width) % unpooled_height + out_pad_h;
    int c = (unpool_index / unpooled_width / unpooled_height) % channels;
    int n = unpool_index / unpooled_width / unpooled_height / channels;
    int spatial_dim = unpooled_height * unpooled_width;
    int hstart = (uh < out_kernel_h) ? 0 :
      (uh - out_kernel_h) / out_stride_h + 1;
    int hend = min(uh / out_stride_h + 1, height);
    int wstart = (uw < out_kernel_w) ? 0 : 
      (uw - out_kernel_w) / out_stride_w + 1;
    int wend = min(uw / out_stride_w + 1, width);
    Dtype val = 0;
    bottom_data += (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        int uhstart = h * out_stride_h - out_pad_h;
        int uwstart = w * out_stride_w - out_pad_w;
        int uhend = min(uhstart + out_kernel_h, unpooled_height + out_pad_h);
        int uwend = min(uwstart + out_kernel_w, unpooled_width + out_pad_w);
        val += bottom_data[h * width + w];
      }
    }
    top_data[unpool_index] = val / mask[unpool_index % spatial_dim];
  }
}

// convert map to Blob because CUDA code cannot access map STL
void GetMapData(const map<int, vector<int> >& group_map,
    Blob<int>* group_map_range, Blob<int>* group_map_index = NULL) {
  // get the start and end index of all groups
  group_map_range->Reshape(1, 1, 1, group_map.size()+1);
  int* group_map_range_data = group_map_range->mutable_cpu_data();
  int total_count = 0;
  int count = 0;
  for (map<int, vector<int> >::const_iterator it = group_map.begin();
      it != group_map.end(); ++it) {
    group_map_range_data[count] = total_count;
    total_count += it->second.size();
    ++count;
  }
  group_map_range_data[count] = total_count;
  // get the group_map_index if necessary
  if (group_map_index != NULL) {
    group_map_index->Reshape(1, 1, 1, total_count);
    int* group_map_index_data = group_map_index->mutable_cpu_data();
    count = 0;
    for (map<int, vector<int> >::const_iterator it = group_map.begin();
        it != group_map.end(); ++it) {
      for (int s = 0; s < it->second.size(); ++s) {
        group_map_index_data[count] = it->second[s];
        ++count;
      }
    }
  }
}

template <typename Dtype>
__global__ void ComputeGroupMean(const int nthreads, const Dtype* data,
    const int channels, const int height, const int width, const int num_groups,
    const int* group_map_range, const int* group_map_index,
    const int group_channels, Dtype* group_mean_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int group_id = index % num_groups;
    int c = (index / num_groups) % channels;
    data += c * height * width;
    int start_idx = group_map_range[group_id];
    int end_idx = group_map_range[group_id + 1];
    Dtype sumval = 0;
    for (int i = start_idx; i < end_idx; ++i) {
      sumval += data[group_map_index[i]];
    }
    group_mean_data[index] = sumval / (end_idx - start_idx) / group_channels;
  }
}

template <typename Dtype>
__global__ void GroupUnPoolForward(const int nthreads, const Dtype* group_mean_data,
    const int channels, const int height, const int width, const Dtype* group_data,
    const int num_groups, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int group_id = (int)group_data[h * width + w];
    top_data[index] += group_mean_data[c * num_groups + group_id];
  }
}

template <typename Dtype>
void UnPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* group_data = NULL;
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = top[0]->count();
  caffe_gpu_set(count, Dtype(0), top_data);
  const int bottom_count = bottom[0]->count() / num_;
  const int* mask = mask_.gpu_data();
  switch (this->layer_param_.unpooling_param().unpool()) {
  case UnPoolingParameter_UnPoolMethod_FIXED:
    // NOLINT_NEXT_LINE(whitespace/operators)
    FixedUnPoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, num_, channels_, height_, width_,
        unpooled_height_, unpooled_width_, out_kernel_h_, out_kernel_w_,
        out_stride_h_, out_stride_w_, out_pad_h_, out_pad_w_, top_data);
    break;
  case UnPoolingParameter_UnPoolMethod_DIV:
    // NOLINT_NEXT_LINE(whitespace/operators)
    DivUnPoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, mask, num_, channels_, height_, width_,
        unpooled_height_, unpooled_width_, out_kernel_h_, out_kernel_w_,
        out_stride_h_, out_stride_w_, out_pad_h_, out_pad_w_, top_data);
    break;
  case UnPoolingParameter_UnPoolMethod_REP:
    // NOLINT_NEXT_LINE(whitespace/operators)
    RepUnPoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, mask, num_, channels_, height_, width_,
        unpooled_height_, unpooled_width_, out_kernel_h_, out_kernel_w_,
        out_stride_h_, out_stride_w_, out_pad_h_, out_pad_w_, top_data);
    break;
  case UnPoolingParameter_UnPoolMethod_GROUP:
    // NOLINT_NEXT_LINE(whitespace/operators)
    CHECK_EQ(bottom.size(), 2);
    // use the reordered internal group data
    group_data = group_blob_.gpu_data();
    for (int n = 0; n < num_; ++n) {
      const vector<map<int, vector<int> > >& group_maps = group_maps_vec_[n];
      for (int gc = 0; gc < group_channels_; ++gc) {
        const map<int, vector<int> >& group_map = group_maps[gc];
        const int num_groups = group_map.size();
        // cuda function cannot call STL function, convert it to Blob data
        GetMapData(group_map, &group_map_range_, &group_map_index_);
        const int* group_map_range = group_map_range_.gpu_data();
        const int* group_map_index = group_map_index_.gpu_data();
        // prepare group_mean_
        group_mean_.Reshape(1, channels_, 1, num_groups);
        Dtype* group_mean_data = group_mean_.mutable_gpu_data();
        int group_count = group_mean_.count();
        // compute group_mean_data
        ComputeGroupMean<Dtype><<<CAFFE_GET_BLOCKS(group_count), CAFFE_CUDA_NUM_THREADS>>>(
            group_count, bottom_data, channels_, height_, width_, num_groups,
            group_map_range, group_map_index, group_channels_, group_mean_data);
        // spread group_mean_data to top_data
        GroupUnPoolForward<Dtype><<<CAFFE_GET_BLOCKS(bottom_count), CAFFE_CUDA_NUM_THREADS>>>(
            bottom_count, group_mean_data, channels_, height_, width_,
            group_data, num_groups, top_data);
        group_data += bottom[1]->offset(0, 1);
      }
      bottom_data += bottom[0]->offset(1);
      top_data += top[0]->offset(1);
    }
    break;
  default:
    LOG(FATAL) << "Unknown unpooling method.";
  }
  CUDA_POST_KERNEL_CHECK;
}


template <typename Dtype>
__global__ void FixedUnPoolBackward(const int nthreads, const Dtype* top_diff,
    const int* mask, const int num, const int channels, const int height,
    const int width, const int unpooled_height, const int unpooled_width,
    const int out_kernel_h, const int out_kernel_w, const int out_stride_h,
    const int out_stride_w, const int out_pad_h, const int out_pad_w,
    Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;
    int uhstart = h * out_stride_h - out_pad_h;
    int uwstart = w * out_stride_w - out_pad_w;
    int uhend = uhstart + out_kernel_h;
    int uwend = uwstart + out_kernel_w;
    int uhmid = (uhstart + uhend - 1) / 2;
    int uwmid = (uwstart + uwend - 1) / 2;
    uhmid = min(max(uhmid, 0), unpooled_height-1);
    uwmid = min(max(uwmid, 0), unpooled_width-1);
    int offset = (n * channels + c) * unpooled_height * unpooled_width;
    int unpool_index = uhmid * unpooled_width + uwmid;
    Dtype gradient = 0;
    if (mask[unpool_index] == h * width + w) {
      gradient += top_diff[unpool_index + offset];
    }
    bottom_diff[index] = gradient;
  }
}

template <typename Dtype>
__global__ void DivUnPoolBackward(const int nthreads, const Dtype* top_diff,
    const int* mask, const int num, const int channels, const int height,
    const int width, const int unpooled_height, const int unpooled_width,
    const int out_kernel_h, const int out_kernel_w, const int out_stride_h,
    const int out_stride_w, const int out_pad_h, const int out_pad_w,
    Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;
    int uhstart = h * out_stride_h - out_pad_h;
    int uwstart = w * out_stride_w - out_pad_w;
    int uhend = min(uhstart + out_kernel_h, unpooled_height + out_pad_h);
    int uwend = min(uwstart + out_kernel_w, unpooled_width + out_pad_w);
    int unpool_size = (uhend - uhstart) * (uwend - uwstart);
    uhstart = max(uhstart, 0);
    uwstart = max(uwstart, 0);
    uhend = min(uhend, unpooled_height);
    uwend = min(uwend, unpooled_width);
    Dtype gradient = 0;
    int offset = (n * channels + c) * unpooled_height * unpooled_width;
    for (int uh = uhstart; uh < uhend; ++uh) {
      for (int uw = uwstart; uw < uwend; ++uw) {
        int unpool_index = uh * unpooled_width + uw;
        gradient += top_diff[unpool_index + offset] / mask[unpool_index];
      }
    }
    bottom_diff[index] = gradient / unpool_size;
  }
}

template <typename Dtype>
__global__ void RepUnPoolBackward(const int nthreads, const Dtype* top_diff,
    const int* mask, const int num, const int channels, const int height,
    const int width, const int unpooled_height, const int unpooled_width,
    const int out_kernel_h, const int out_kernel_w, const int out_stride_h,
    const int out_stride_w, const int out_pad_h, const int out_pad_w,
    Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;
    int uhstart = h * out_stride_h - out_pad_h;
    int uwstart = w * out_stride_w - out_pad_w;
    int uhend = min(uhstart + out_kernel_h, unpooled_height + out_pad_h);
    int uwend = min(uwstart + out_kernel_w, unpooled_width + out_pad_w);
    uhstart = max(uhstart, 0);
    uwstart = max(uwstart, 0);
    uhend = min(uhend, unpooled_height);
    uwend = min(uwend, unpooled_width);
    Dtype gradient = 0;
    int offset = (n * channels + c) * unpooled_height * unpooled_width;
    for (int uh = uhstart; uh < uhend; ++uh) {
      for (int uw = uwstart; uw < uwend; ++uw) {
        int unpool_index = uh * unpooled_width + uw;
        gradient += top_diff[unpool_index + offset] / mask[unpool_index];
      }
    }
    bottom_diff[index] = gradient;
  }
}

template <typename Dtype>
__global__ void GroupUnPoolBackward(const int nthreads, const Dtype* group_mean_diff,
    const int channels, const int height, const int width, const Dtype* group_data,
    const int num_groups, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int group_id = (int)group_data[h * width + w];
    bottom_diff[index] += group_mean_diff[c * num_groups + group_id];
  }
}

template <typename Dtype>
void UnPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const Dtype* group_data;
  const int count = bottom[0]->count();
  const int top_count = top[0]->count() / top[0]->num();
  caffe_gpu_set(count, Dtype(0.), bottom_diff);
  const int* mask = mask_.gpu_data();
  switch (this->layer_param_.unpooling_param().unpool()) {
  case UnPoolingParameter_UnPoolMethod_FIXED:
    // NOLINT_NEXT_LINE(whitespace/operators)
    FixedUnPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, mask, num_, channels_, height_, width_,
        unpooled_height_, unpooled_width_, out_kernel_h_, out_kernel_w_,
        out_stride_h_, out_stride_w_, out_pad_h_, out_pad_w_, bottom_diff);
    break;
  case UnPoolingParameter_UnPoolMethod_DIV:
    // NOLINT_NEXT_LINE(whitespace/operators)
    DivUnPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, mask, num_, channels_, height_, width_,
        unpooled_height_, unpooled_width_, out_kernel_h_, out_kernel_w_,
        out_stride_h_, out_stride_w_, out_pad_h_, out_pad_w_, bottom_diff);
    break;
  case UnPoolingParameter_UnPoolMethod_REP:
    // NOLINT_NEXT_LINE(whitespace/operators)
    RepUnPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, mask, num_, channels_, height_, width_,
        unpooled_height_, unpooled_width_, out_kernel_h_, out_kernel_w_,
        out_stride_h_, out_stride_w_, out_pad_h_, out_pad_w_, bottom_diff);
    break;
  case UnPoolingParameter_UnPoolMethod_GROUP:
    // NOLINT_NEXT_LINE(whitespace/operators)
    CHECK_EQ(bottom.size(), 2);
    // use the reordered internal group data
    group_data = group_blob_.gpu_data();
    for (int n = 0; n < num_; ++n) {
      const vector<map<int, vector<int> > >& group_maps = group_maps_vec_[n];
      for (int gc = 0; gc < group_channels_; ++gc) {
        const map<int, vector<int> >& group_map = group_maps[gc];
        const int num_groups = group_map.size();
        // cuda function cannot call STL function, convert it to Blob data
        GetMapData(group_map, &group_map_range_, &group_map_index_);
        const int* group_map_range = group_map_range_.gpu_data();
        const int* group_map_index = group_map_index_.gpu_data();
        // prepare group_mean_
        group_mean_.Reshape(1, channels_, 1, num_groups);
        Dtype* group_mean_diff = group_mean_.mutable_gpu_diff();
        int group_count = group_mean_.count();
        // compute group_mean_diff
        ComputeGroupMean<Dtype><<<CAFFE_GET_BLOCKS(group_count), CAFFE_CUDA_NUM_THREADS>>>(
            group_count, top_diff, channels_, height_, width_, num_groups,
            group_map_range, group_map_index, group_channels_, group_mean_diff);
        // spread group_mean_diff to bottom_diff
        GroupUnPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            top_count, group_mean_diff, channels_, height_, width_, group_data,
            num_groups, bottom_diff);
        group_data += bottom[1]->offset(0, 1);
      }
      bottom_diff += bottom[0]->offset(1);
      top_diff += top[0]->offset(1);
    }
    break;
  default:
    LOG(FATAL) << "Unknown unpooling method.";
  }
  CUDA_POST_KERNEL_CHECK;
}


INSTANTIATE_LAYER_GPU_FUNCS(UnPoolingLayer);


}  // namespace caffe
