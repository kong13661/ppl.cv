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

#include "ppl/cv/ocl/sepfilter2d.h"

#include <tuple>
#include <sstream>

#include "opencv2/imgproc.hpp"
#include "gtest/gtest.h"

#include "ppl/common/ocl/pplopencl.h"
#include "utility/infrastructure.h"
#include "kerneltypes.h"


using Parameters = std::tuple<int, int, BorderType, cv::Size>;
inline std::string sepfilter2dToString(const Parameters& parameters) {
  std::ostringstream formatted;

  int ksize = std::get<0>(parameters);
  formatted << "Ksize" << ksize << "_";

  int int_delta = std::get<1>(parameters);
  formatted << "Delta" << int_delta << "_";

  BorderType border_type = std::get<2>(parameters);
  if (border_type == BORDER_REPLICATE) {
    formatted << "BORDER_REPLICATE" << "_";
  }
  else if (border_type == BORDER_REFLECT) {
    formatted << "BORDER_REFLECT" << "_";
  }
  else if (border_type == BORDER_REFLECT_101) {
    formatted << "BORDER_REFLECT_101" << "_";
  }
  else {  // border_type == BORDER_DEFAULT
    formatted << "BORDER_DEFAULT" << "_";
  }

  cv::Size size = std::get<3>(parameters);
  formatted << size.width << "x";
  formatted << size.height;

  return formatted.str();
}

template <typename Tsrc, typename Tdst, int channels>
class PplCvOclSepfilter2dToTest: public ::testing::TestWithParam<Parameters> {
 public:
  PplCvOclSepfilter2dToTest() {
    const Parameters& parameters = GetParam();
    ksize       = std::get<0>(parameters);
    delta       = std::get<1>(parameters) / 10.f;
    border_type = std::get<2>(parameters);
    size        = std::get<3>(parameters);

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

  ~PplCvOclSepfilter2dToTest() {
  }

  bool apply();

 private:
  int ksize;
  float delta;
  BorderType border_type;
  cv::Size size;
  cl_context context;
  cl_command_queue queue;
};

template <typename Tsrc, typename Tdst, int channels>
bool PplCvOclSepfilter2dToTest<Tsrc, Tdst, channels>::apply() {
  cv::Mat src, kernel0;
  src = createSourceImage(size.height, size.width,
                          CV_MAKETYPE(cv::DataType<Tsrc>::depth, channels));
  kernel0 = createSourceImage(1, ksize,
                             CV_MAKETYPE(cv::DataType<float>::depth, 1));
  cv::Mat dst(size.height, size.width,
              CV_MAKETYPE(cv::DataType<Tdst>::depth, channels));
  cv::Mat cv_dst(size.height, size.width,
                 CV_MAKETYPE(cv::DataType<Tdst>::depth, channels));

  int src_bytes0 = src.rows * src.step;
  int kernel_bytes0 = kernel0.rows * kernel0.step;
  int dst_bytes0 = dst.rows * dst.step;

  cl_int error_code = 0;
  cl_mem gpu_src = clCreateBuffer(context,
                                   CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
                                   src_bytes0, NULL, &error_code);
  CHECK_ERROR(error_code, clCreateBuffer);

  cl_mem gpu_kernel = clCreateBuffer(context,
                                     CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
                                     kernel_bytes0, NULL, &error_code);
  CHECK_ERROR(error_code, clCreateBuffer);

  cl_mem gpu_dst = clCreateBuffer(context,
                                  CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY,
                                  dst_bytes0, NULL, &error_code);
  CHECK_ERROR(error_code, clCreateBuffer);

  error_code = clEnqueueWriteBuffer(queue, gpu_src, CL_FALSE, 0, src_bytes0,
                                    src.data, 0, NULL, NULL);
  CHECK_ERROR(error_code, clEnqueueWriteBuffer);

  error_code = clEnqueueWriteBuffer(queue, gpu_kernel, CL_FALSE, 0, kernel_bytes0,
                                    kernel0.data, 0, NULL, NULL);
  CHECK_ERROR(error_code, clEnqueueWriteBuffer);

  cl_mem buffer = clCreateBuffer(                                            
      context, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,                    
      size.height * (int)sizeof(float) * size.width * (int)sizeof(float) * channels,      
      NULL, &error_code);                                                    
  CHECK_ERROR(error_code, clCreateBuffer);                                   

  int src_bytes1 = size.height * size.width * channels * sizeof(Tsrc);
  int kernel_bytes1 = ksize * sizeof(float);
  int dst_bytes1 = (size.height) * (size.width) * channels * sizeof(Tdst);
  cl_mem gpu_input = clCreateBuffer(context,
                                    CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                                    src_bytes1, NULL, &error_code);
  CHECK_ERROR(error_code, clCreateBuffer);

  cl_mem gpu_input_kernel = clCreateBuffer(context,
                                          CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                                          kernel_bytes1, NULL, &error_code);
  CHECK_ERROR(error_code, clCreateBuffer);

  cl_mem gpu_output = clCreateBuffer(context,
                                     CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                                     dst_bytes1, NULL, &error_code);
  CHECK_ERROR(error_code, clCreateBuffer);

  Tsrc* input = (Tsrc*)clEnqueueMapBuffer(queue, gpu_input, CL_TRUE, CL_MAP_WRITE,
                                    0, src_bytes1, 0, NULL, NULL, &error_code);
  CHECK_ERROR(error_code, clEnqueueMapBuffer);
  copyMatToArray(src, input);
  error_code = clEnqueueUnmapMemObject(queue, gpu_input, input, 0, NULL, NULL);
  CHECK_ERROR(error_code, clEnqueueUnmapMemObject);

  float* input_kernel = (float*)clEnqueueMapBuffer(queue, gpu_input_kernel, CL_TRUE,
                                    CL_MAP_WRITE, 0, kernel_bytes1, 0, NULL, NULL,
                                    &error_code);
  CHECK_ERROR(error_code, clEnqueueMapBuffer);
  copyMatToArray(kernel0, input_kernel);
  error_code = clEnqueueUnmapMemObject(queue, gpu_input_kernel, input_kernel, 0,
                                       NULL, NULL);
  CHECK_ERROR(error_code, clEnqueueUnmapMemObject);

  cv::BorderTypes cv_border = cv::BORDER_DEFAULT;
  if (border_type == BORDER_REPLICATE) {
    cv_border = cv::BORDER_REPLICATE;
  }
  else if (border_type == BORDER_REFLECT) {
    cv_border = cv::BORDER_REFLECT;
  }
  else if (border_type == BORDER_REFLECT_101) {
    cv_border = cv::BORDER_REFLECT_101;
  }
  else {
  }
  cv::sepFilter2D(src, cv_dst, cv_dst.depth(), kernel0, kernel0,
                  cv::Point(-1, -1), delta, cv_border);

  ppl::cv::ocl::SepFilter2D<Tsrc, Tdst, channels>(
      queue, src.rows, src.cols, src.step / sizeof(Tsrc), gpu_src, ksize,
      gpu_kernel, gpu_kernel, dst.step / sizeof(Tdst), gpu_dst, buffer, delta,
      border_type);
  ppl::cv::ocl::SepFilter2D<Tsrc, Tdst, channels>(
      queue, size.height, size.width, size.width * channels, gpu_input, ksize,
      gpu_input_kernel, gpu_input_kernel, size.width * channels, gpu_output, buffer,
      delta, border_type);
  error_code = clEnqueueReadBuffer(queue, gpu_dst, CL_TRUE, 0, dst_bytes0,
                                   dst.data, 0, NULL, NULL);
  CHECK_ERROR(error_code, clEnqueueReadBuffer);
  Tdst* output = (Tdst*)clEnqueueMapBuffer(queue, gpu_output, CL_TRUE, CL_MAP_READ,
                                     0, dst_bytes1, 0, NULL, NULL, &error_code);
  CHECK_ERROR(error_code, clEnqueueMapBuffer);

  float epsilon;
  if (sizeof(Tdst) == 1) {
    epsilon = EPSILON_1F;
  }
  else {
    epsilon = EPSILON_E4;
  }

  bool identity0 = checkMatricesIdentity<Tdst>((const Tdst*)cv_dst.data, cv_dst.rows,
      cv_dst.cols, cv_dst.channels(), cv_dst.step, (const Tdst*)dst.data, dst.step,
      epsilon);
  bool identity1 = checkMatricesIdentity<Tdst>((const Tdst*)cv_dst.data, cv_dst.rows,
      cv_dst.cols, cv_dst.channels(), cv_dst.step, output,
      dst.cols * channels * sizeof(Tdst), epsilon);
  error_code = clEnqueueUnmapMemObject(queue, gpu_output, output, 0, NULL,
                                       NULL);
  CHECK_ERROR(error_code, clEnqueueUnmapMemObject);

  clReleaseMemObject(gpu_src);
  clReleaseMemObject(gpu_kernel);
  clReleaseMemObject(gpu_dst);
  clReleaseMemObject(gpu_input);
  clReleaseMemObject(gpu_input_kernel);
  clReleaseMemObject(gpu_output);
  clReleaseMemObject(buffer);

  return (identity0 && identity1);
}

#define UNITTEST(Tsrc, Tdst, channels)                                         \
using PplCvOclSepfilter2dToTest ## Tsrc ## Tdst ## channels =                  \
        PplCvOclSepfilter2dToTest<Tsrc, Tdst, channels>;                       \
TEST_P(PplCvOclSepfilter2dToTest ## Tsrc ## Tdst ## channels, Standard) {      \
  bool identity = this->apply();                                               \
  EXPECT_TRUE(identity);                                                       \
}                                                                              \
                                                                               \
INSTANTIATE_TEST_CASE_P(IsEqual,                                               \
  PplCvOclSepfilter2dToTest ## Tsrc ## Tdst ## channels,                       \
  ::testing::Combine(                                                          \
    ::testing::Values(5),                                    \
    ::testing::Values(0),                                              \
    ::testing::Values(BORDER_REFLECT, BORDER_REPLICATE, BORDER_REFLECT_101),                                     \
    ::testing::Values(cv::Size{321, 240})),             \
  [](const testing::TestParamInfo<                                             \
      PplCvOclSepfilter2dToTest ## Tsrc ## Tdst ## channels::ParamType>&       \
        info) {                                                                \
    return sepfilter2dToString(info.param);                                    \
  }                                                                            \
);



UNITTEST(uchar, uchar, 1)
UNITTEST(uchar, uchar, 3)
UNITTEST(uchar, uchar, 4)
// UNITTEST(uchar, short, 1)
// UNITTEST(uchar, short, 3)
// UNITTEST(uchar, short, 4)
UNITTEST(float, float, 1)
UNITTEST(float, float, 3)
UNITTEST(float, float, 4)
