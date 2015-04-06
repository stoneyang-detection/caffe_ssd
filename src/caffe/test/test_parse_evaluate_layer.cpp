#include <cfloat>
#include <cmath>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename Dtype>
class ParseEvaluateLayerTest : public ::testing::Test {
 protected:
  ParseEvaluateLayerTest()
      : blob_bottom_prediction_(new Blob<Dtype>()),
        blob_bottom_label_(new Blob<Dtype>()),
        blob_buffer_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()),
        num_labels_(10) {
    vector<int> shape(4);
    shape[0] = 2;
    shape[1] = 1;
    shape[2] = 10;
    shape[3] = 10;
    blob_bottom_prediction_->Reshape(shape);
    blob_bottom_label_->Reshape(shape);
    FillBottoms();

    shape[0] = 1;
    shape[1] = num_labels_;
    shape[2] = 1;
    shape[3] = 3;
    blob_buffer_->Reshape(shape);

    blob_bottom_vec_.push_back(blob_bottom_prediction_);
    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual void FillBottoms() {
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    shared_ptr<Caffe::RNG> rng(new Caffe::RNG(prefetch_rng_seed));
    caffe::rng_t* prefetch_rng =
          static_cast<caffe::rng_t*>(rng->generator());
    Dtype* prediction_data = blob_bottom_prediction_->mutable_cpu_data();
    Dtype* label_data = blob_bottom_label_->mutable_cpu_data();
    for (int i = 0; i < blob_bottom_label_->count(); ++i) {
      prediction_data[i] = (*prefetch_rng)() % num_labels_;
      label_data[i] = (*prefetch_rng)() % num_labels_;
    }
  }

  virtual ~ParseEvaluateLayerTest() {
    delete blob_bottom_prediction_;
    delete blob_bottom_label_;
    delete blob_buffer_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_prediction_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_buffer_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  int num_labels_;
};

TYPED_TEST_CASE(ParseEvaluateLayerTest, TestDtypes);

TYPED_TEST(ParseEvaluateLayerTest, TestSetup) {
  LayerParameter layer_param;
  ParseEvaluateParameter* parse_evaluate_param = layer_param.mutable_parse_evaluate_param();
  parse_evaluate_param->set_num_labels(this->num_labels_);
  ParseEvaluateLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), this->num_labels_);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 3);
}

TYPED_TEST(ParseEvaluateLayerTest, TestForwardCPU) {
  LayerParameter layer_param;
  Caffe::set_mode(Caffe::CPU);
  ParseEvaluateParameter* parse_evaluate_param = layer_param.mutable_parse_evaluate_param();
  parse_evaluate_param->set_num_labels(this->num_labels_);
  ParseEvaluateLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  TypeParam predict_label, true_label;
  TypeParam* buffer_data = this->blob_buffer_->mutable_cpu_data();
  caffe_set(this->blob_buffer_->count(), TypeParam(0), buffer_data);
  int index;
  for (int i = 0; i < 2; ++i) {
    for (int h = 0; h < 10; ++h) {
      for (int w = 0; w < 10; ++w) {
        predict_label = this->blob_bottom_prediction_->data_at(i, 0, h, w);
        true_label = this->blob_bottom_label_->data_at(i, 0, h, w);
        if (predict_label == true_label) {
          index = this->blob_buffer_->offset(0, predict_label, 0, 0);
          ++buffer_data[index];
        }
        index = this->blob_buffer_->offset(0, true_label, 0, 1);
        ++buffer_data[index];
        index = this->blob_buffer_->offset(0, predict_label, 0, 2);
        ++buffer_data[index];
      }
    }
  }
  for (int c = 0; c < this->num_labels_; ++c) {
    for (int k = 0; k < 3; ++k) {
      EXPECT_NEAR(this->blob_top_->data_at(0, c, 0, k),
                  this->blob_buffer_->data_at(0, c, 0, k), 1e-4);
    }
  }
}

}  // namespace caffe
