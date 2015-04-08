#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class UnPoolingLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  UnPoolingLayerTest()
      : blob_bottom_(new Blob<Dtype>()),
      blob_top_(new Blob<Dtype>()),
      blob_top_mask_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    Caffe::set_random_seed(1029);
    blob_bottom_->Reshape(2, 3, 3, 2);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~UnPoolingLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
    delete blob_top_mask_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_top_mask_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(UnPoolingLayerTest, TestDtypesAndDevices);

TYPED_TEST(UnPoolingLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  UnPoolingParameter* unpooling_param = layer_param.mutable_unpooling_param();
  unpooling_param->set_out_kernel_size(3);
  UnPoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), 5);
  EXPECT_EQ(this->blob_top_->width(), 4);
}

TYPED_TEST(UnPoolingLayerTest, TestSetupStrided) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  UnPoolingParameter* unpooling_param = layer_param.mutable_unpooling_param();
  unpooling_param->set_out_kernel_size(3);
  unpooling_param->set_out_stride(2);
  UnPoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), 7);
  EXPECT_EQ(this->blob_top_->width(), 5);
}

TYPED_TEST(UnPoolingLayerTest, TestSetupPadded) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  UnPoolingParameter* unpooling_param = layer_param.mutable_unpooling_param();
  unpooling_param->set_out_kernel_size(3);
  unpooling_param->set_out_stride(2);
  unpooling_param->set_out_pad(1);
  unpooling_param->set_unpool(UnPoolingParameter_UnPoolMethod_DIV);
  UnPoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), 5);
  EXPECT_EQ(this->blob_top_->width(), 3);
}

TYPED_TEST(UnPoolingLayerTest, TestSetupAuto1) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  UnPoolingParameter* unpooling_param = layer_param.mutable_unpooling_param();
  unpooling_param->set_out_kernel_size(0);
  unpooling_param->set_out_stride(0);
  UnPoolingLayer<Dtype> layer(layer_param);

  Blob<Dtype>* unpool_blob(new Blob<Dtype>());
  unpool_blob->Reshape(2, 3, 6, 4);
  this->blob_bottom_vec_.push_back(unpool_blob);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), 6);
  EXPECT_EQ(this->blob_top_->width(), 4);
  this->blob_bottom_vec_.pop_back();
}

TYPED_TEST(UnPoolingLayerTest, TestSetupAuto2) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  UnPoolingParameter* unpooling_param = layer_param.mutable_unpooling_param();
  unpooling_param->set_out_kernel_size(0);
  unpooling_param->set_out_stride(0);
  UnPoolingLayer<Dtype> layer(layer_param);

  Blob<Dtype>* unpool_blob(new Blob<Dtype>());
  unpool_blob->Reshape(2, 3, 7, 5);
  this->blob_bottom_vec_.push_back(unpool_blob);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), 7);
  EXPECT_EQ(this->blob_top_->width(), 5);
  this->blob_bottom_vec_.pop_back();
}

TYPED_TEST(UnPoolingLayerTest, TestSetupAuto3) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  UnPoolingParameter* unpooling_param = layer_param.mutable_unpooling_param();
  unpooling_param->set_out_kernel_size(3);
  unpooling_param->set_out_stride(2);
  UnPoolingLayer<Dtype> layer(layer_param);

  Blob<Dtype>* unpool_blob(new Blob<Dtype>());
  unpool_blob->Reshape(2, 3, 7, 5);
  this->blob_bottom_vec_.push_back(unpool_blob);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), 7);
  EXPECT_EQ(this->blob_top_->width(), 5);
  this->blob_bottom_vec_.pop_back();
}

TYPED_TEST(UnPoolingLayerTest, TestSetupGroup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  UnPoolingParameter* unpooling_param = layer_param.mutable_unpooling_param();
  unpooling_param->set_unpool(UnPoolingParameter_UnPoolMethod_GROUP);
  UnPoolingLayer<Dtype> layer(layer_param);

  Blob<Dtype>* group_blob(new Blob<Dtype>());
  group_blob->Reshape(2, 1, 3, 2);
  this->blob_bottom_vec_.push_back(group_blob);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), 3);
  EXPECT_EQ(this->blob_top_->width(), 2);
  this->blob_bottom_vec_.pop_back();
}

// Test for 2 x 2 square unpooling layer
TYPED_TEST(UnPoolingLayerTest, TestForwardSquareFixed) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  UnPoolingParameter* unpooling_param = layer_param.mutable_unpooling_param();
  unpooling_param->set_out_kernel_size(2);
  unpooling_param->set_unpool(UnPoolingParameter_UnPoolMethod_FIXED);
  const int num = 2;
  const int channels = 2;
  this->blob_bottom_->Reshape(num, channels, 3, 2);
  // Input: 2 x 2 channels of:
  //     [1 2]
  //     [9 4]
  //     [5 3]
  for (int i = 0; i < 6 * num * channels; i += 6) {
    this->blob_bottom_->mutable_cpu_data()[i +  0] = 1;
    this->blob_bottom_->mutable_cpu_data()[i +  1] = 2;
    this->blob_bottom_->mutable_cpu_data()[i +  2] = 9;
    this->blob_bottom_->mutable_cpu_data()[i +  3] = 4;
    this->blob_bottom_->mutable_cpu_data()[i +  4] = 5;
    this->blob_bottom_->mutable_cpu_data()[i +  5] = 3;
  }
  UnPoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), num);
  EXPECT_EQ(this->blob_top_->channels(), channels);
  EXPECT_EQ(this->blob_top_->height(), 4);
  EXPECT_EQ(this->blob_top_->width(), 3);

  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Expected output: 2 x 2 channels of:
  //     [1 2 0]
  //     [9 4 0]
  //     [5 3 0]
  //     [0 0 0]
  for (int i = 0; i < 12 * num * channels; i += 12) {
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 0], 1);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 1], 2);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 2], 0);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 3], 9);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 4], 4);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 5], 0);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 6], 5);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 7], 3);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 8], 0);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 9], 0);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 10], 0);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 11], 0);
  }
}

// Test for 3 x 2 rectangular unpooling layer with out_kernel_h > out_kernel_w
TYPED_TEST(UnPoolingLayerTest, TestForwardHighFixed) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  UnPoolingParameter* unpooling_param = layer_param.mutable_unpooling_param();
  unpooling_param->set_out_kernel_h(3);
  unpooling_param->set_out_kernel_w(2);
  unpooling_param->set_unpool(UnPoolingParameter_UnPoolMethod_FIXED);
  const int num = 2;
  const int channels = 2;
  this->blob_bottom_->Reshape(num, channels, 3, 3);
  // Input: 2 x 2 channels of:
  //     [1 2 8]
  //     [9 4 6]
  //     [5 3 7]
  for (int i = 0; i < 9 * num * channels; i += 9) {
    this->blob_bottom_->mutable_cpu_data()[i +  0] = 1;
    this->blob_bottom_->mutable_cpu_data()[i +  1] = 2;
    this->blob_bottom_->mutable_cpu_data()[i +  2] = 8;
    this->blob_bottom_->mutable_cpu_data()[i +  3] = 9;
    this->blob_bottom_->mutable_cpu_data()[i +  4] = 4;
    this->blob_bottom_->mutable_cpu_data()[i +  5] = 6;
    this->blob_bottom_->mutable_cpu_data()[i +  6] = 5;
    this->blob_bottom_->mutable_cpu_data()[i +  7] = 3;
    this->blob_bottom_->mutable_cpu_data()[i +  8] = 7;
  }
  UnPoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), num);
  EXPECT_EQ(this->blob_top_->channels(), channels);
  EXPECT_EQ(this->blob_top_->height(), 5);
  EXPECT_EQ(this->blob_top_->width(), 4);

  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Expected output: 2 x 2 channels of:
  //     [0 0 0 0]
  //     [1 2 8 0]
  //     [9 4 6 0]
  //     [5 3 7 0]
  //     [0 0 0 0]
  for (int i = 0; i < 20 * num * channels; i += 20) {
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 0], 0);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 1], 0);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 2], 0);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 3], 0);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 4], 1);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 5], 2);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 6], 8);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 7], 0);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 8], 9);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 9], 4);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 10], 6);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 11], 0);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 12], 5);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 13], 3);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 14], 7);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 15], 0);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 16], 0);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 17], 0);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 18], 0);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 19], 0);
  }
}

// Test for 2 x 3 rectangular unpooling layer with out_kernel_w > out_kernel_h
TYPED_TEST(UnPoolingLayerTest, TestForwardWideFixed) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  UnPoolingParameter* unpooling_param = layer_param.mutable_unpooling_param();
  unpooling_param->set_out_kernel_h(2);
  unpooling_param->set_out_kernel_w(3);
  unpooling_param->set_unpool(UnPoolingParameter_UnPoolMethod_FIXED);
  const int num = 2;
  const int channels = 2;
  this->blob_bottom_->Reshape(num, channels, 3, 3);
  // Input: 2 x 2 channels of:
  //     [1 2 8]
  //     [9 4 6]
  //     [5 3 7]
  for (int i = 0; i < 9 * num * channels; i += 9) {
    this->blob_bottom_->mutable_cpu_data()[i +  0] = 1;
    this->blob_bottom_->mutable_cpu_data()[i +  1] = 2;
    this->blob_bottom_->mutable_cpu_data()[i +  2] = 8;
    this->blob_bottom_->mutable_cpu_data()[i +  3] = 9;
    this->blob_bottom_->mutable_cpu_data()[i +  4] = 4;
    this->blob_bottom_->mutable_cpu_data()[i +  5] = 6;
    this->blob_bottom_->mutable_cpu_data()[i +  6] = 5;
    this->blob_bottom_->mutable_cpu_data()[i +  7] = 3;
    this->blob_bottom_->mutable_cpu_data()[i +  8] = 7;
  }
  UnPoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), num);
  EXPECT_EQ(this->blob_top_->channels(), channels);
  EXPECT_EQ(this->blob_top_->height(), 4);
  EXPECT_EQ(this->blob_top_->width(), 5);

  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Expected output: 2 x 2 channels of:
  //     [0 1 2 8 0]
  //     [0 9 4 6 0]
  //     [0 5 3 7 0]
  //     [0 0 0 0 0]
  for (int i = 0; i < 20 * num * channels; i += 20) {
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 0], 0);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 1], 1);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 2], 2);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 3], 8);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 4], 0);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 5], 0);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 6], 9);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 7], 4);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 8], 6);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 9], 0);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 10], 0);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 11], 5);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 12], 3);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 13], 7);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 14], 0);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 15], 0);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 16], 0);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 17], 0);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 18], 0);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 19], 0);
  }
}

TYPED_TEST(UnPoolingLayerTest, TestForwardFixedStrided) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  UnPoolingParameter* unpooling_param = layer_param.mutable_unpooling_param();
  unpooling_param->set_out_kernel_size(3);
  unpooling_param->set_out_stride(2);
  unpooling_param->set_unpool(UnPoolingParameter_UnPoolMethod_FIXED);
  const int num = 2;
  const int channels = 2;
  this->blob_bottom_->Reshape(num, channels, 3, 3);
  // Input:
  //     [ 1 2 4 ]
  //     [ 2 3 2 ]
  //     [ 4 2 1 ]
  for (int i = 0; i < 9 * num * channels; i += 9) {
    this->blob_bottom_->mutable_cpu_data()[i+0] = 1;
    this->blob_bottom_->mutable_cpu_data()[i+1] = 2;
    this->blob_bottom_->mutable_cpu_data()[i+2] = 4;
    this->blob_bottom_->mutable_cpu_data()[i+3] = 2;
    this->blob_bottom_->mutable_cpu_data()[i+4] = 3;
    this->blob_bottom_->mutable_cpu_data()[i+5] = 2;
    this->blob_bottom_->mutable_cpu_data()[i+6] = 4;
    this->blob_bottom_->mutable_cpu_data()[i+7] = 2;
    this->blob_bottom_->mutable_cpu_data()[i+8] = 1;
  }
  UnPoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), num);
  EXPECT_EQ(this->blob_top_->channels(), channels);
  EXPECT_EQ(this->blob_top_->height(), 7);
  EXPECT_EQ(this->blob_top_->width(), 7);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Dtype epsilon = 1e-8;
  // Output:
  //     [ 0 0 0 0 0 0 0 ]
  //     [ 0 1 0 2 0 4 0 ]
  //     [ 0 0 0 0 0 0 0 ]
  //     [ 0 2 0 3 0 2 0 ]
  //     [ 0 0 0 0 0 0 0 ]
  //     [ 0 4 0 2 0 1 0 ]
  //     [ 0 0 0 0 0 0 0 ]
  for (int i = 0; i < 49 * num * channels; i += 49) {
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+0], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+1], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+2], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+3], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+4], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+5], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+6], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+7], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+8], 1, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+9], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+10], 2, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+11], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+12], 4, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+13], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+14], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+15], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+16], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+17], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+18], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+19], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+20], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+21], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+22], 2, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+23], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+24], 3, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+25], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+26], 2, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+27], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+28], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+29], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+30], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+31], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+32], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+33], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+34], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+35], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+36], 4, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+37], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+38], 2, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+39], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+40], 1, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+41], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+42], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+43], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+44], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+45], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+46], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+47], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+48], 0, epsilon);
  }
}

TYPED_TEST(UnPoolingLayerTest, TestForwardFixedPadded) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  UnPoolingParameter* unpooling_param = layer_param.mutable_unpooling_param();
  unpooling_param->set_out_kernel_size(3);
  unpooling_param->set_out_stride(2);
  unpooling_param->set_out_pad(1);
  unpooling_param->set_unpool(UnPoolingParameter_UnPoolMethod_FIXED);
  const int num = 2;
  const int channels = 2;
  this->blob_bottom_->Reshape(num, channels, 3, 3);
  // Input:
  //     [ 1 2 4 ]
  //     [ 2 3 2 ]
  //     [ 4 2 1 ]
  for (int i = 0; i < 9 * num * channels; i += 9) {
    this->blob_bottom_->mutable_cpu_data()[i+0] = 1;
    this->blob_bottom_->mutable_cpu_data()[i+1] = 2;
    this->blob_bottom_->mutable_cpu_data()[i+2] = 4;
    this->blob_bottom_->mutable_cpu_data()[i+3] = 2;
    this->blob_bottom_->mutable_cpu_data()[i+4] = 3;
    this->blob_bottom_->mutable_cpu_data()[i+5] = 2;
    this->blob_bottom_->mutable_cpu_data()[i+6] = 4;
    this->blob_bottom_->mutable_cpu_data()[i+7] = 2;
    this->blob_bottom_->mutable_cpu_data()[i+8] = 1;
  }
  UnPoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), num);
  EXPECT_EQ(this->blob_top_->channels(), channels);
  EXPECT_EQ(this->blob_top_->height(), 5);
  EXPECT_EQ(this->blob_top_->width(), 5);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Dtype epsilon = 1e-8;
  // Output:
  //     [ 1 0 2 0 4 ]
  //     [ 0 0 0 0 0 ]
  //     [ 2 0 3 0 2 ]
  //     [ 0 0 0 0 0 ]
  //     [ 4 0 2 0 1 ]
  for (int i = 0; i < 25 * num * channels; i += 25) {
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+0], 1, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+1], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+2], 2, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+3], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+4], 4, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+5], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+6], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+7], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+8], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+9], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+10], 2, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+11], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+12], 3, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+13], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+14], 2, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+15], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+16], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+17], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+18], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+19], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+20], 4, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+21], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+22], 2, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+23], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+24], 1, epsilon);
  }
}

TYPED_TEST(UnPoolingLayerTest, TestForwardDiv) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  UnPoolingParameter* unpooling_param = layer_param.mutable_unpooling_param();
  unpooling_param->set_out_kernel_size(3);
  unpooling_param->set_unpool(UnPoolingParameter_UnPoolMethod_DIV);
  const int num = 2;
  const int channels = 2;
  this->blob_bottom_->Reshape(num, channels, 3, 3);
  FillerParameter filler_param;
  filler_param.set_value(Dtype(2));
  ConstantFiller<Dtype> filler(filler_param);
  filler.Fill(this->blob_bottom_);
  UnPoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), num);
  EXPECT_EQ(this->blob_top_->channels(), channels);
  EXPECT_EQ(this->blob_top_->height(), 5);
  EXPECT_EQ(this->blob_top_->width(), 5);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Dtype epsilon = 1e-5;
  for (int i = 0; i < 25 * num * channels; i += 25) {
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+0], 2.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+1], 4.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+2], 6.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+3], 4.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+4], 2.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+5], 4.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+6], 8.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+7], 12.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+8], 8.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+9], 4.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+10], 6.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+11], 12.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+12], 18.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+13], 12.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+14], 6.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+15], 4.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+16], 8.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+17], 12.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+18], 8.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+19], 4.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+20], 2.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+21], 4.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+22], 6.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+23], 4.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+24], 2.0/9, epsilon);
  }
}

TYPED_TEST(UnPoolingLayerTest, TestForwardDivStrided) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  UnPoolingParameter* unpooling_param = layer_param.mutable_unpooling_param();
  unpooling_param->set_out_kernel_size(3);
  unpooling_param->set_out_stride(2);
  unpooling_param->set_unpool(UnPoolingParameter_UnPoolMethod_DIV);
  const int num = 2;
  const int channels = 2;
  this->blob_bottom_->Reshape(num, channels, 3, 3);
  FillerParameter filler_param;
  filler_param.set_value(Dtype(2));
  ConstantFiller<Dtype> filler(filler_param);
  filler.Fill(this->blob_bottom_);
  UnPoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), num);
  EXPECT_EQ(this->blob_top_->channels(), channels);
  EXPECT_EQ(this->blob_top_->height(), 7);
  EXPECT_EQ(this->blob_top_->width(), 7);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Dtype epsilon = 1e-5;
  for (int i = 0; i < 49 * num * channels; i += 49) {
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+0], 2.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+1], 2.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+2], 4.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+3], 2.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+4], 4.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+5], 2.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+6], 2.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+7], 2.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+8], 2.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+9], 4.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+10], 2.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+11], 4.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+12], 2.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+13], 2.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+14], 4.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+15], 4.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+16], 8.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+17], 4.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+18], 8.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+19], 4.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+20], 4.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+21], 2.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+22], 2.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+23], 4.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+24], 2.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+25], 4.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+26], 2.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+27], 2.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+28], 4.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+29], 4.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+30], 8.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+31], 4.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+32], 8.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+33], 4.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+34], 4.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+35], 2.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+36], 2.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+37], 4.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+38], 2.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+39], 4.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+40], 2.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+41], 2.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+42], 2.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+43], 2.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+44], 4.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+45], 2.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+46], 4.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+47], 2.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+48], 2.0/9, epsilon);
  }
}

TYPED_TEST(UnPoolingLayerTest, TestForwardDivPadded) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  UnPoolingParameter* unpooling_param = layer_param.mutable_unpooling_param();
  unpooling_param->set_out_kernel_size(3);
  unpooling_param->set_out_stride(2);
  unpooling_param->set_out_pad(1);
  unpooling_param->set_unpool(UnPoolingParameter_UnPoolMethod_DIV);
  const int num = 2;
  const int channels = 2;
  this->blob_bottom_->Reshape(num, channels, 3, 3);
  FillerParameter filler_param;
  filler_param.set_value(Dtype(2));
  ConstantFiller<Dtype> filler(filler_param);
  filler.Fill(this->blob_bottom_);
  UnPoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), num);
  EXPECT_EQ(this->blob_top_->channels(), channels);
  EXPECT_EQ(this->blob_top_->height(), 5);
  EXPECT_EQ(this->blob_top_->width(), 5);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Dtype epsilon = 1e-5;
  for (int i = 0; i < 25 * num * channels; i += 25) {
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+0], 2.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+1], 4.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+2], 2.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+3], 4.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+4], 2.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+5], 4.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+6], 8.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+7], 4.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+8], 8.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+9], 4.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+10], 2.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+11], 4.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+12], 2.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+13], 4.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+14], 2.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+15], 4.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+16], 8.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+17], 4.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+18], 8.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+19], 4.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+20], 2.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+21], 4.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+22], 2.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+23], 4.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+24], 2.0/9, epsilon);
  }
}

TYPED_TEST(UnPoolingLayerTest, TestForwardRep) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  UnPoolingParameter* unpooling_param = layer_param.mutable_unpooling_param();
  unpooling_param->set_out_kernel_size(3);
  unpooling_param->set_unpool(UnPoolingParameter_UnPoolMethod_REP);
  const int num = 2;
  const int channels = 2;
  this->blob_bottom_->Reshape(num, channels, 3, 3);
  FillerParameter filler_param;
  filler_param.set_value(Dtype(2));
  ConstantFiller<Dtype> filler(filler_param);
  filler.Fill(this->blob_bottom_);
  UnPoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), num);
  EXPECT_EQ(this->blob_top_->channels(), channels);
  EXPECT_EQ(this->blob_top_->height(), 5);
  EXPECT_EQ(this->blob_top_->width(), 5);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Dtype epsilon = 1e-5;
  for (int i = 0; i < 25 * num * channels; i += 25) {
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+0], 2, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+1], 4, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+2], 6, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+3], 4, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+4], 2, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+5], 4, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+6], 8, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+7], 12, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+8], 8, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+9], 4, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+10], 6, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+11], 12, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+12], 18, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+13], 12, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+14], 6, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+15], 4, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+16], 8, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+17], 12, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+18], 8, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+19], 4, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+20], 2, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+21], 4, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+22], 6, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+23], 4, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+24], 2, epsilon);
  }
}

TYPED_TEST(UnPoolingLayerTest, TestForwardRepStrided) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  UnPoolingParameter* unpooling_param = layer_param.mutable_unpooling_param();
  unpooling_param->set_out_kernel_size(3);
  unpooling_param->set_out_stride(2);
  unpooling_param->set_unpool(UnPoolingParameter_UnPoolMethod_REP);
  const int num = 2;
  const int channels = 2;
  this->blob_bottom_->Reshape(num, channels, 3, 3);
  FillerParameter filler_param;
  filler_param.set_value(Dtype(2));
  ConstantFiller<Dtype> filler(filler_param);
  filler.Fill(this->blob_bottom_);
  UnPoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), num);
  EXPECT_EQ(this->blob_top_->channels(), channels);
  EXPECT_EQ(this->blob_top_->height(), 7);
  EXPECT_EQ(this->blob_top_->width(), 7);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Dtype epsilon = 1e-5;
  for (int i = 0; i < 49 * num * channels; i += 49) {
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+0], 2, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+1], 2, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+2], 4, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+3], 2, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+4], 4, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+5], 2, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+6], 2, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+7], 2, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+8], 2, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+9], 4, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+10], 2, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+11], 4, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+12], 2, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+13], 2, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+14], 4, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+15], 4, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+16], 8, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+17], 4, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+18], 8, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+19], 4, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+20], 4, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+21], 2, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+22], 2, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+23], 4, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+24], 2, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+25], 4, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+26], 2, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+27], 2, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+28], 4, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+29], 4, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+30], 8, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+31], 4, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+32], 8, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+33], 4, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+34], 4, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+35], 2, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+36], 2, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+37], 4, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+38], 2, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+39], 4, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+40], 2, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+41], 2, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+42], 2, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+43], 2, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+44], 4, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+45], 2, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+46], 4, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+47], 2, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+48], 2, epsilon);
  }
}

TYPED_TEST(UnPoolingLayerTest, TestForwardRepPadded) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  UnPoolingParameter* unpooling_param = layer_param.mutable_unpooling_param();
  unpooling_param->set_out_kernel_size(3);
  unpooling_param->set_out_stride(2);
  unpooling_param->set_out_pad(1);
  unpooling_param->set_unpool(UnPoolingParameter_UnPoolMethod_REP);
  const int num = 2;
  const int channels = 2;
  this->blob_bottom_->Reshape(num, channels, 3, 3);
  FillerParameter filler_param;
  filler_param.set_value(Dtype(2));
  ConstantFiller<Dtype> filler(filler_param);
  filler.Fill(this->blob_bottom_);
  UnPoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), num);
  EXPECT_EQ(this->blob_top_->channels(), channels);
  EXPECT_EQ(this->blob_top_->height(), 5);
  EXPECT_EQ(this->blob_top_->width(), 5);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Dtype epsilon = 1e-5;
  for (int i = 0; i < 25 * num * channels; i += 25) {
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+0], 2, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+1], 4, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+2], 2, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+3], 4, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+4], 2, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+5], 4, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+6], 8, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+7], 4, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+8], 8, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+9], 4, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+10], 2, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+11], 4, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+12], 2, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+13], 4, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+14], 2, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+15], 4, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+16], 8, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+17], 4, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+18], 8, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+19], 4, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+20], 2, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+21], 4, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+22], 2, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+23], 4, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+24], 2, epsilon);
  }
}

TYPED_TEST(UnPoolingLayerTest, TestForwardGroup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  UnPoolingParameter* unpooling_param = layer_param.mutable_unpooling_param();
  unpooling_param->set_unpool(UnPoolingParameter_UnPoolMethod_GROUP);
  const int num = 2;
  const int channels = 2;
  this->blob_bottom_->Reshape(num, channels, 3, 3);
  // Input:
  //     [ 1 2 4 ]
  //     [ 2 3 2 ]
  //     [ 1 2 0 ]
  for (int i = 0; i < 9 * num * channels; i += 9) {
    this->blob_bottom_->mutable_cpu_data()[i+0] = 1;
    this->blob_bottom_->mutable_cpu_data()[i+1] = 2;
    this->blob_bottom_->mutable_cpu_data()[i+2] = 4;
    this->blob_bottom_->mutable_cpu_data()[i+3] = 2;
    this->blob_bottom_->mutable_cpu_data()[i+4] = 3;
    this->blob_bottom_->mutable_cpu_data()[i+5] = 2;
    this->blob_bottom_->mutable_cpu_data()[i+6] = 1;
    this->blob_bottom_->mutable_cpu_data()[i+7] = 2;
    this->blob_bottom_->mutable_cpu_data()[i+8] = 0;
  }
  Blob<Dtype>* group_blob(new Blob<Dtype>());
  group_blob->Reshape(num, 1, 3, 3);
  // Group:
  //     [ 1 1 2 ]
  //     [ 1 1 2 ]
  //     [ 4 4 4 ]
  for (int i = 0; i < 9 * num; i += 9) {
    group_blob->mutable_cpu_data()[i+0] = 1;
    group_blob->mutable_cpu_data()[i+1] = 1;
    group_blob->mutable_cpu_data()[i+2] = 2;
    group_blob->mutable_cpu_data()[i+3] = 1;
    group_blob->mutable_cpu_data()[i+4] = 1;
    group_blob->mutable_cpu_data()[i+5] = 2;
    group_blob->mutable_cpu_data()[i+6] = 4;
    group_blob->mutable_cpu_data()[i+7] = 4;
    group_blob->mutable_cpu_data()[i+8] = 4;
  }
  this->blob_bottom_vec_.push_back(group_blob);
  UnPoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), num);
  EXPECT_EQ(this->blob_top_->channels(), channels);
  EXPECT_EQ(this->blob_top_->height(), 3);
  EXPECT_EQ(this->blob_top_->width(), 3);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Dtype epsilon = 1e-8;
  // Output:
  //     [ 2 2 3 ]
  //     [ 2 2 3 ]
  //     [ 1 1 1 ]
  for (int i = 0; i < 9 * num * channels; i += 9) {
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+0], 2, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+1], 2, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+2], 3, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+3], 2, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+4], 2, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+5], 3, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+6], 1, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+7], 1, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+8], 1, epsilon);
  }
  this->blob_bottom_vec_.pop_back();
}

TYPED_TEST(UnPoolingLayerTest, TestForwardGroupMult) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  UnPoolingParameter* unpooling_param = layer_param.mutable_unpooling_param();
  unpooling_param->set_unpool(UnPoolingParameter_UnPoolMethod_GROUP);
  const int num = 2;
  const int channels = 2;
  this->blob_bottom_->Reshape(num, channels, 3, 3);
  // Input:
  //     [ 1 2 4 ]
  //     [ 2 3 6 ]
  //     [ 2 2 5 ]
  for (int i = 0; i < 9 * num * channels; i += 9) {
    this->blob_bottom_->mutable_cpu_data()[i+0] = 1;
    this->blob_bottom_->mutable_cpu_data()[i+1] = 2;
    this->blob_bottom_->mutable_cpu_data()[i+2] = 4;
    this->blob_bottom_->mutable_cpu_data()[i+3] = 2;
    this->blob_bottom_->mutable_cpu_data()[i+4] = 3;
    this->blob_bottom_->mutable_cpu_data()[i+5] = 6;
    this->blob_bottom_->mutable_cpu_data()[i+6] = 2;
    this->blob_bottom_->mutable_cpu_data()[i+7] = 2;
    this->blob_bottom_->mutable_cpu_data()[i+8] = 5;
  }
  Blob<Dtype>* group_blob(new Blob<Dtype>());
  group_blob->Reshape(num, 3, 3, 3);
  // Group:
  //     [ 1 1 2 ]
  //     [ 1 1 2 ]
  //     [ 4 4 4 ]
  //     =========
  //     [ 8 8 8 ]
  //     [ 8 8 8 ]
  //     [ 8 8 8 ]
  //     =========
  //     [ 1 1 1 ]
  //     [ 1 1 1 ]
  //     [ 2 2 2 ]
  for (int i = 0; i < 27 * num; i += 27) {
    group_blob->mutable_cpu_data()[i+0] = 1;
    group_blob->mutable_cpu_data()[i+1] = 1;
    group_blob->mutable_cpu_data()[i+2] = 2;
    group_blob->mutable_cpu_data()[i+3] = 1;
    group_blob->mutable_cpu_data()[i+4] = 1;
    group_blob->mutable_cpu_data()[i+5] = 2;
    group_blob->mutable_cpu_data()[i+6] = 4;
    group_blob->mutable_cpu_data()[i+7] = 4;
    group_blob->mutable_cpu_data()[i+8] = 4;
    group_blob->mutable_cpu_data()[i+9] = 8;
    group_blob->mutable_cpu_data()[i+10] = 8;
    group_blob->mutable_cpu_data()[i+11] = 8;
    group_blob->mutable_cpu_data()[i+12] = 8;
    group_blob->mutable_cpu_data()[i+13] = 8;
    group_blob->mutable_cpu_data()[i+14] = 8;
    group_blob->mutable_cpu_data()[i+15] = 8;
    group_blob->mutable_cpu_data()[i+16] = 8;
    group_blob->mutable_cpu_data()[i+17] = 8;
    group_blob->mutable_cpu_data()[i+18] = 1;
    group_blob->mutable_cpu_data()[i+19] = 1;
    group_blob->mutable_cpu_data()[i+20] = 1;
    group_blob->mutable_cpu_data()[i+21] = 1;
    group_blob->mutable_cpu_data()[i+22] = 1;
    group_blob->mutable_cpu_data()[i+23] = 1;
    group_blob->mutable_cpu_data()[i+24] = 2;
    group_blob->mutable_cpu_data()[i+25] = 2;
    group_blob->mutable_cpu_data()[i+26] = 2;
  }
  this->blob_bottom_vec_.push_back(group_blob);
  UnPoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), num);
  EXPECT_EQ(this->blob_top_->channels(), channels);
  EXPECT_EQ(this->blob_top_->height(), 3);
  EXPECT_EQ(this->blob_top_->width(), 3);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Dtype epsilon = 1e-6;
  // Output:
  //     [ 8 8 11 ]
  //     [ 8 8 11 ]
  //     [ 9 9  9 ]
  for (int i = 0; i < 9 * num * channels; i += 9) {
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+0], 8.0/3, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+1], 8.0/3, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+2], 11.0/3, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+3], 8.0/3, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+4], 8.0/3, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+5], 11.0/3, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+6], 3, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+7], 3, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+8], 3, epsilon);
  }
  this->blob_bottom_vec_.pop_back();
}

TYPED_TEST(UnPoolingLayerTest, TestGradientFixed) {
  typedef typename TypeParam::Dtype Dtype;
  for (int out_kernel_h = 3; out_kernel_h <= 4; out_kernel_h++) {
    for (int out_kernel_w = 3; out_kernel_w <= 4; out_kernel_w++) {
      LayerParameter layer_param;
      UnPoolingParameter* unpooling_param = layer_param.mutable_unpooling_param();
      unpooling_param->set_out_kernel_h(out_kernel_h);
      unpooling_param->set_out_kernel_w(out_kernel_w);
      unpooling_param->set_out_stride(2);
      unpooling_param->set_out_pad(1);
      unpooling_param->set_unpool(UnPoolingParameter_UnPoolMethod_FIXED);
      UnPoolingLayer<Dtype> layer(layer_param);
      GradientChecker<Dtype> checker(1e-4, 1e-2);
      checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
                                      this->blob_top_vec_);
    }
  }
}

TYPED_TEST(UnPoolingLayerTest, TestGradientDiv) {
  typedef typename TypeParam::Dtype Dtype;
  for (int out_kernel_h = 3; out_kernel_h <= 4; out_kernel_h++) {
    for (int out_kernel_w = 3; out_kernel_w <= 4; out_kernel_w++) {
      LayerParameter layer_param;
      UnPoolingParameter* unpooling_param = layer_param.mutable_unpooling_param();
      unpooling_param->set_out_kernel_h(out_kernel_h);
      unpooling_param->set_out_kernel_w(out_kernel_w);
      unpooling_param->set_out_stride(2);
      unpooling_param->set_out_pad(1);
      unpooling_param->set_unpool(UnPoolingParameter_UnPoolMethod_DIV);
      UnPoolingLayer<Dtype> layer(layer_param);
      GradientChecker<Dtype> checker(1e-4, 1e-2);
      checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
                                      this->blob_top_vec_);
    }
  }
}

TYPED_TEST(UnPoolingLayerTest, TestGradientRep) {
  typedef typename TypeParam::Dtype Dtype;
  for (int out_kernel_h = 3; out_kernel_h <= 4; out_kernel_h++) {
    for (int out_kernel_w = 3; out_kernel_w <= 4; out_kernel_w++) {
      LayerParameter layer_param;
      UnPoolingParameter* unpooling_param = layer_param.mutable_unpooling_param();
      unpooling_param->set_out_kernel_h(out_kernel_h);
      unpooling_param->set_out_kernel_w(out_kernel_w);
      unpooling_param->set_out_stride(2);
      unpooling_param->set_out_pad(1);
      unpooling_param->set_unpool(UnPoolingParameter_UnPoolMethod_REP);
      UnPoolingLayer<Dtype> layer(layer_param);
      GradientChecker<Dtype> checker(1e-4, 1e-2);
      checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
                                      this->blob_top_vec_);
    }
  }
}

TYPED_TEST(UnPoolingLayerTest, TestGradientGroup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  UnPoolingParameter* unpooling_param = layer_param.mutable_unpooling_param();
  unpooling_param->set_unpool(UnPoolingParameter_UnPoolMethod_GROUP);
  Blob<Dtype>* group_blob(new Blob<Dtype>());
  int num = 2;
  group_blob->Reshape(num, 3, 3, 2);
  // Group:
  //     [ 1 2 ]
  //     [ 1 2 ]
  //     [ 4 4 ]
  //     =========
  //     [ 8 8 ]
  //     [ 8 8 ]
  //     [ 8 8 ]
  //     =========
  //     [ 1 1 ]
  //     [ 1 1 ]
  //     [ 2 2 ]
  for (int i = 0; i < 18 * num; i += 18) {
    group_blob->mutable_cpu_data()[i+0] = 1;
    group_blob->mutable_cpu_data()[i+1] = 2;
    group_blob->mutable_cpu_data()[i+2] = 1;
    group_blob->mutable_cpu_data()[i+3] = 2;
    group_blob->mutable_cpu_data()[i+4] = 4;
    group_blob->mutable_cpu_data()[i+5] = 4;
    group_blob->mutable_cpu_data()[i+6] = 8;
    group_blob->mutable_cpu_data()[i+7] = 8;
    group_blob->mutable_cpu_data()[i+8] = 8;
    group_blob->mutable_cpu_data()[i+9] = 8;
    group_blob->mutable_cpu_data()[i+10] = 8;
    group_blob->mutable_cpu_data()[i+11] = 8;
    group_blob->mutable_cpu_data()[i+12] = 1;
    group_blob->mutable_cpu_data()[i+13] = 1;
    group_blob->mutable_cpu_data()[i+14] = 1;
    group_blob->mutable_cpu_data()[i+15] = 1;
    group_blob->mutable_cpu_data()[i+16] = 2;
    group_blob->mutable_cpu_data()[i+17] = 2;
  }
  this->blob_bottom_vec_.push_back(group_blob);
  UnPoolingLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-4, 1e-2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
                                  this->blob_top_vec_);
  this->blob_bottom_vec_.pop_back();
}

}  // namespace caffe
