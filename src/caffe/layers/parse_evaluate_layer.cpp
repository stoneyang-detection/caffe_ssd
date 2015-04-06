#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void ParseEvaluateLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const ParseEvaluateParameter& parse_evaluate_param =
      this->layer_param_.parse_evaluate_param();
  CHECK(parse_evaluate_param.has_num_labels()) << "Must have num_labels!!";
  num_labels_ = parse_evaluate_param.num_labels();
  ignore_labels_.clear();
  int num_ignore_label = parse_evaluate_param.ignore_label().size();
  for (int i = 0; i < num_ignore_label; ++i) {
    ignore_labels_.insert(parse_evaluate_param.ignore_label(i));
  }
}

template <typename Dtype>
void ParseEvaluateLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
      << "The data and label should have the same number.";
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->channels(), 1);
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_GE(bottom[0]->width(), bottom[1]->width());
  top[0]->Reshape(1, num_labels_, 1, 3);
}

template <typename Dtype>
void ParseEvaluateLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                            const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  CHECK_EQ(bottom[0]->count(), bottom[1]->count());
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_set(top[0]->count(), Dtype(0), top_data);
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / bottom[0]->num();
  int spatial_dim = bottom[0]->height() * bottom[0]->width();
  for (int i = 0; i < num; ++i) {
    // Find all unique labels in the ground truth
    std::set<Dtype> labels(bottom_label + i * spatial_dim,
                           bottom_label + (i + 1) * spatial_dim);
    // Find all unqiue labels in prediction
    std::set<Dtype> predict_labels(bottom_data + i * spatial_dim,
                                   bottom_data + (i + 1) * spatial_dim);
    labels.insert(predict_labels.begin(), predict_labels.end());
    // Find regions with ignore_labels_
    Blob<Dtype> ignore_mask(1, 1, bottom[0]->height(), bottom[0]->width());
    Dtype* ignore_mask_data = ignore_mask.mutable_cpu_data();
    caffe_set(ignore_mask.count(), Dtype(0), ignore_mask_data);
    for (typename std::set<Dtype>::iterator it = ignore_labels_.begin();
         it != ignore_labels_.end(); ++it) {
      int ignore_label = *it;
      for (int j = 0; j < spatial_dim; ++j) {
        if (static_cast<int>(bottom_label[i * spatial_dim + j]) == ignore_label) {
          ignore_mask_data[j] = 1;
        }
      }
    }
    // count the number of ground truth labels, the predicted labels, and
    // predicted labels happens to be ground truth labels
    for (typename std::set<Dtype>::iterator it = labels.begin();
         it != labels.end(); ++it) {
      int label = *it;
      int predict_pos_label = 0;
      int predict_label = 0;
      int num_pos_label = 0;
      for (int j = 0; j < spatial_dim; ++j) {
        if (ignore_mask_data[j] == 1) {
          continue;
        }
        if (static_cast<int>(bottom_label[i * spatial_dim + j]) == label) {
          ++num_pos_label;
          if (static_cast<int>(bottom_data[i * dim + j]) == label) {
            ++predict_pos_label;
          }
        }
        if (static_cast<int>(bottom_data[i * dim + j]) == label) {
          ++predict_label;
        }
      }
      top_data[label*3] += predict_pos_label;
      top_data[label*3 + 1] += num_pos_label;
      top_data[label*3 + 2] += predict_label;
    }
  }
  // ParseEvaluate layer should not be used as a loss function.
}

INSTANTIATE_CLASS(ParseEvaluateLayer);
REGISTER_LAYER_CLASS(ParseEvaluate);

}  // namespace caffe
