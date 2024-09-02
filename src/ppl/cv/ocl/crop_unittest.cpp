/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements. See the NOTICE file distributed with this
 * work for additional information regarding copyright ownership. The ASF
 * licenses this file to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance with the
 * License. You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */

#include "ppl/cv/ocl/crop.h"

#include <tuple>
#include <sstream>

#include "opencv2/core.hpp"
#include "gtest/gtest.h"

#include "ppl/common/ocl/pplopencl.h"
#include "utility/infrastructure.h"

using Parameters = std::tuple<int, int, int, cv::Size>;
inline std::string convertToStringCrop(const Parameters& parameters) {
  std::ostringstream formatted;

  int left = std::get<0>(parameters);
  formatted << "Left" << left << "_";

  int top = std::get<1>(parameters);
  formatted << "Top" << top << "_";

  int int_scale = std::get<2>(parameters);
  formatted << "IntScale" << int_scale << "_";

  cv::Size size = std::get<3>(parameters);
  formatted << size.width << "x";
  formatted << size.height;

  return formatted.str();
}

template <typename T, int channels>
class PplCvOclCropTest : public ::testing::TestWithParam<Parameters> {
 public:
  PplCvOclCropTest() {
    const Parameters& parameters = GetParam();
    left  = std::get<0>(parameters);
    top   = std::get<1>(parameters);
    scale = std::get<2>(parameters) / 10.f;
    size  = std::get<3>(parameters);

    ppl::common::ocl::createSharedFrameChain(false);
    context = ppl::common::ocl::getSharedFrameChain()->getContext();
    queue   = ppl::common::ocl::getSharedFrameChain()->getQueue();

    bool status = ppl::common::ocl::initializeKernelBinariesManager(
                      ppl::common::ocl::BINARIES_RETRIEVE);
    if (status) {
      ppl::common::ocl::FrameChain* frame_chain =
          ppl::common::ocl::getSharedFrameChain();
      frame_chain->setCreatingProgramType(ppl::common::ocl::WITH_BINARIES);
    }
  }

  ~PplCvOclCropTest() {
    ppl::common::ocl::shutDownKernelBinariesManager(
        ppl::common::ocl::BINARIES_RETRIEVE);
  }

  bool apply();

 private:
  int left;
  int top;
  float scale;
  cv::Size size;
  cl_context context;
  cl_command_queue queue;
};

template <typename T, int channels>
bool PplCvOclCropTest<T, channels>::apply() {
  int src_height = size.height * 2;
  int src_width  = size.width * 2;
  cv::Mat src;
  src = createSourceImage(src_height, src_width,
                          CV_MAKETYPE(cv::DataType<T>::depth, channels));
  cv::Mat dst(size.height, size.width,
              CV_MAKETYPE(cv::DataType<T>::depth, channels));
  cv::Mat cv_dst(size.height, size.width,
                 CV_MAKETYPE(cv::DataType<T>::depth, channels));

  int src_bytes0 = src.rows * src.step;
  int dst_bytes0 = dst.rows * dst.step;
  cl_int error_code = 0;
  cl_mem gpu_src = clCreateBuffer(context,
                                  CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
                                  src_bytes0, NULL, &error_code);
  CHECK_ERROR(error_code, clCreateBuffer);
  cl_mem gpu_dst = clCreateBuffer(context,
                                  CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY,
                                  dst_bytes0, NULL, &error_code);
  CHECK_ERROR(error_code, clCreateBuffer);
  error_code = clEnqueueWriteBuffer(queue, gpu_src, CL_FALSE, 0, src_bytes0,
                                    src.data, 0, NULL, NULL);
  CHECK_ERROR(error_code, clEnqueueWriteBuffer);

  int src_bytes1 = src_height * src_width * channels * sizeof(T);
  int dst_bytes1 = size.height * size.width * channels * sizeof(T);

  cl_mem gpu_input = clCreateBuffer(context,
                                    CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
                                    src_bytes1, NULL, &error_code);
  CHECK_ERROR(error_code, clCreateBuffer);
  cl_mem gpu_output = clCreateBuffer(context,
                                     CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY,
                                     dst_bytes1, NULL, &error_code);
  CHECK_ERROR(error_code, clCreateBuffer);
  
  T* input = (T*)clEnqueueMapBuffer(queue, gpu_input, CL_TRUE, CL_MAP_WRITE,
                                    0, src_bytes1, 0, NULL, NULL, &error_code);
  CHECK_ERROR(error_code, clEnqueueMapBuffer);
  copyMatToArray(src, input);
  error_code = clEnqueueUnmapMemObject(queue, gpu_input, input, 0, NULL, NULL);
  CHECK_ERROR(error_code, clEnqueueUnmapMemObject);

  cv::Rect roi(left, top, size.width, size.height);
  cv::Mat croppedImage = src(roi);
  croppedImage.copyTo(cv_dst);
  cv_dst = cv_dst * scale;

  ppl::cv::ocl::Crop<T, channels>(queue, src.rows, src.cols,
      src.step / sizeof(T), gpu_src, dst.rows, dst.cols,
      dst.step / sizeof(T), gpu_dst, left, top, scale);
  error_code = clEnqueueReadBuffer(queue, gpu_dst, CL_TRUE, 0, dst_bytes0,
                                   dst.data, 0, NULL, NULL);
  CHECK_ERROR(error_code, clEnqueueReadBuffer);
  ppl::cv::ocl::Crop<T, channels>(queue, src_height, src_width,
      src_width * channels, gpu_input, size.height, size.width, 
      size.width * channels, gpu_output, left, top, scale);
  T* output = (T*)clEnqueueMapBuffer(queue, gpu_output, CL_TRUE, CL_MAP_READ,
                                     0, dst_bytes1, 0, NULL, NULL, &error_code);
  CHECK_ERROR(error_code, clEnqueueMapBuffer);

  float epsilon;
  if (sizeof(T) == 1) {
    epsilon = EPSILON_1F;
  }
  else {
    epsilon = EPSILON_E6;
  }

  bool identity0 = checkMatricesIdentity<T>((const T*)cv_dst.data, cv_dst.rows,
      cv_dst.cols, cv_dst.channels(), cv_dst.step, (const T*)dst.data, dst.step,
      epsilon);
  bool identity1 = checkMatricesIdentity<T>((const T*)cv_dst.data, cv_dst.rows,
      cv_dst.cols, cv_dst.channels(), cv_dst.step, output,
      size.width * channels * sizeof(T), epsilon);
  error_code = clEnqueueUnmapMemObject(queue, gpu_output, output, 0, NULL,
                                       NULL);
  CHECK_ERROR(error_code, clEnqueueUnmapMemObject);

  clReleaseMemObject(gpu_src);
  clReleaseMemObject(gpu_dst);
  clReleaseMemObject(gpu_input);
  clReleaseMemObject(gpu_output);

  return (identity0 && identity1);
}

#define UNITTEST(T, channels)                                                   \
using PplCvOclCropTest ## T ## channels = PplCvOclCropTest<T, channels>;        \
TEST_P(PplCvOclCropTest ## T ## channels, Standard) {                           \
  bool identity = this->apply();                                                \
  EXPECT_TRUE(identity);                                                        \
}                                                                               \
                                                                                \
INSTANTIATE_TEST_CASE_P(IsEqual, PplCvOclCropTest ## T ## channels,             \
  ::testing::Combine(                                                           \
    ::testing::Values(0, 11, 187),                                              \
    ::testing::Values(0, 11, 187),                                              \
    ::testing::Values(0, 10, 15),                                               \
    ::testing::Values(cv::Size{321, 240}, cv::Size{642, 480},                   \
                      cv::Size{1283, 720}, cv::Size{1934, 1080},                \
                      cv::Size{320, 240}, cv::Size{640, 480},                   \
                      cv::Size{1280, 720}, cv::Size{1920, 1080})),              \
  [](const testing::TestParamInfo<                                              \
      PplCvOclCropTest ## T ## channels::ParamType>& info) {                    \
    return convertToStringCrop(info.param);                                     \
  }                                                                             \
);

UNITTEST(uchar, 1)
UNITTEST(uchar, 3)
UNITTEST(uchar, 4)
UNITTEST(float, 1)
UNITTEST(float, 3)
UNITTEST(float, 4)
