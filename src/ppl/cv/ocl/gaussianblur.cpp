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

#include "ppl/cv/ocl/gaussianblur.h"

#include "ppl/common/ocl/pplopencl.h"
#include "utility/utility.hpp"

#include "kernels/gaussianblur.cl"
#include "kerneltypes.h"
#include <math.h>

using namespace ppl::common;
using namespace ppl::common::ocl;

#define F32DIV 2
#define F32OFFSET 1
#define U8DIV 4
#define U8OFFSET 2

#define F32DIV_CN 1
#define F32OFFSET_CN 0
#define U8DIV_CN 4
#define U8OFFSET_CN 2

namespace ppl {
namespace cv {
namespace ocl {

void getGaussianKernel(float sigma, int ksize, float* coefficients) {
  bool fixed_kernel = false;
  if ((ksize & 1) == 1 && ksize <= 7 && sigma <= 0) {
    if (ksize == 1) {
      coefficients[0] = 1.f;
    }
    else if (ksize == 3) {
      coefficients[0] = 0.25f;
      coefficients[1] = 0.5f;
      coefficients[2] = 0.25f;
    }
    else if (ksize == 5) {
      coefficients[0] = 0.0625f;
      coefficients[1] = 0.25f;
      coefficients[2] = 0.375f;
      coefficients[3] = 0.25f;
      coefficients[4] = 0.0625f;
    }
    else {
      coefficients[0] = 0.03125f;
      coefficients[1] = 0.109375f;
      coefficients[2] = 0.21875f;
      coefficients[3] = 0.28125f;
      coefficients[4] = 0.21875f;
      coefficients[5] = 0.109375f;
      coefficients[6] = 0.03125f;
    }
    fixed_kernel = true;
  }

  double value = sigma > 0 ? sigma : ((ksize - 1) * 0.5f - 1) * 0.3f + 0.8f;
  double scale_2x = -0.5f / (value * value);
  double sum = 0.f;

  int i;
  double x;
  for (i = 0; i < ksize; i++) {
    x = i - (ksize - 1) * 0.5f;
    value = fixed_kernel ? coefficients[i] : std::exp(scale_2x * x * x);
    if (!fixed_kernel) {
      coefficients[i] = value;
    }
    sum += value;
  }

  sum = 1.f / sum;
  for (i = 0; i < ksize; i++) {
    coefficients[i] *= sum;
  }
}

#define RUN_KERNEL(interpolate, base_type)                                    \
  {                                                                           \
    float* kernel_cpu = new float[ksize];                                     \
    getGaussianKernel(sigma, ksize, kernel_cpu);                              \
    error_code = clEnqueueWriteBuffer(queue, kernel, CL_FALSE, 0,             \
                                      ksize * (int)sizeof(float), kernel_cpu, \
                                      0, NULL, NULL);                         \
    CHECK_ERROR(error_code, clEnqueueWriteBuffer);                            \
    delete[] kernel_cpu;                                                      \
                                                                              \
    ksize = ksize >> 1;                                                       \
    size_t local_size[] = {kBlockDimX0, kBlockDimY0};                         \
    global_cols = divideUp(cols, F32##DIV, F32##OFFSET);                      \
    global_rows = divideUp(rows, F32##DIV, F32##OFFSET);                      \
    global_size[0] = (size_t)global_cols;                                     \
    global_size[1] = (size_t)global_rows;                                     \
    frame_chain->setCompileOptions("-D GAUSSIANBLUR_" #base_type "1C");       \
    runOclKernel(frame_chain,                                                 \
                 "gaussianblur" #base_type "F32" #interpolate "C1Kernel", 2,  \
                 global_size, local_size, src, rows, cols, kernel, ksize,     \
                 src_stride, buffer, rows*(int)sizeof(float));  \
                                                                              \
    global_cols = divideUp(cols, base_type##DIV, base_type##OFFSET);          \
    global_rows = divideUp(rows, base_type##DIV, base_type##OFFSET);          \
    global_size[0] = (size_t)global_rows;                                     \
    global_size[1] = (size_t)global_cols;                                     \
    runOclKernel(frame_chain,                                                 \
                 "gaussianblur"                                               \
                 "F32" #base_type #interpolate "C1Kernel",                    \
                 2, global_size, local_size, buffer, cols, rows, kernel,      \
                 ksize, rows*(int)sizeof(float), dst, dst_stride);            \
    clReleaseMemObject(buffer);                                               \
    clReleaseMemObject(kernel);                                               \
  }

#define GAUSSIANBLUR_C1_TYPE(base_type, T)                                    \
  RetCode gaussianblurC1##base_type(                                          \
      const cl_mem src, int rows, int cols, int src_stride, int ksize,        \
      float sigma, cl_mem dst, int dst_stride, BorderType border_type,        \
      cl_context context, cl_command_queue queue) {                           \
    PPL_ASSERT(src != nullptr);                                               \
    PPL_ASSERT(dst != nullptr);                                               \
    PPL_ASSERT(rows >= 1 && cols >= 1);                                       \
  PPL_ASSERT(ksize > 0);\
  PPL_ASSERT((ksize & 1) == 1);\
    PPL_ASSERT(src_stride >= cols * (int)sizeof(T));                          \
    PPL_ASSERT(dst_stride >= cols * (int)sizeof(T));                          \
    PPL_ASSERT(border_type == BORDER_REPLICATE ||                             \
               border_type == BORDER_REFLECT ||                               \
               border_type == BORDER_REFLECT_101)                             \
    cl_int error_code;                                                        \
    cl_mem buffer =                                                           \
        clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,    \
                       cols * (int)sizeof(float) * rows * (int)sizeof(float), \
                       NULL, &error_code);                                    \
    CHECK_ERROR(error_code, clCreateBuffer);                                  \
                                                                              \
    cl_mem kernel =                                                           \
        clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,    \
                       ksize * (int)sizeof(float), NULL, &error_code);        \
    CHECK_ERROR(error_code, clCreateBuffer);                                  \
                                                                              \
    FrameChain* frame_chain = getSharedFrameChain();                          \
    frame_chain->setProjectName("cv");                                        \
    SET_PROGRAM_SOURCE(frame_chain, gaussianblur);                            \
                                                                              \
    int global_cols, global_rows;                                             \
    size_t global_size[2];                                                    \
                                                                              \
    if (border_type == BORDER_REPLICATE)                                      \
      RUN_KERNEL(interpolateReplicateBorder, base_type)                       \
    else if (border_type == BORDER_REFLECT)                                   \
      RUN_KERNEL(interpolateReflectBorder, base_type)                         \
    else if (border_type == BORDER_REFLECT_101)                               \
      RUN_KERNEL(interpolateReflect101Border, base_type)                      \
    return RC_SUCCESS;                                                        \
  }

#define RUN_KERNEL_CN(interpolate, base_type, channels)                      \
  {                                                                          \
    float* kernel_cpu = new float[ksize];                                     \
    getGaussianKernel(sigma, ksize, kernel_cpu);                              \
    error_code = clEnqueueWriteBuffer(queue, kernel, CL_FALSE, 0,             \
                                      ksize * (int)sizeof(float), kernel_cpu, \
                                      0, NULL, NULL);                         \
    CHECK_ERROR(error_code, clEnqueueWriteBuffer);                            \
    delete[] kernel_cpu;                                                      \
                                                                              \
    ksize = ksize >> 1;                                                       \
    size_t local_size[] = {kBlockDimX0, kBlockDimY0};                         \
    global_cols = cols;                                                      \
    global_rows = divideUp(rows, F32##DIV_CN, F32##OFFSET_CN);               \
    global_size[0] = (size_t)global_cols;                                    \
    global_size[1] = (size_t)global_rows;                                    \
    frame_chain->setCompileOptions("-D GAUSSIANBLUR_" #base_type             \
                                   "C" #channels);                           \
    runOclKernel(frame_chain,                                                \
                 "gaussianblur" #base_type "F32" #interpolate "C" #channels  \
                 "Kernel",                                                   \
                 2, global_size, local_size, src, rows, cols, kernel, ksize, \
                 src_stride, buffer, rows*(int)sizeof(float) * channels);                                         \
                                                                             \
    global_cols = divideUp(cols, base_type##DIV_CN, base_type##OFFSET_CN);   \
    global_rows = rows;                                                      \
    global_size[0] = (size_t)global_rows;                                    \
    global_size[1] = (size_t)global_cols;                                    \
    runOclKernel(frame_chain,                                                \
                 "gaussianblur"                                              \
                 "F32" #base_type #interpolate "C" #channels "Kernel",       \
                 2, global_size, local_size, buffer, cols, rows, kernel,     \
                 ksize, rows*(int)sizeof(float) * channels, dst, dst_stride);                                       \
    clReleaseMemObject(buffer);                                              \
  }

#define GAUSSIANBLUR_CN_TYPE(base_type, T, channels)                       \
  RetCode gaussianblurC##channels##base_type(                              \
      const cl_mem src, int rows, int cols, int src_stride, int ksize,     \
      float sigma, cl_mem dst, int dst_stride, BorderType border_type,     \
      cl_context context, cl_command_queue queue) {                        \
    PPL_ASSERT(src != nullptr);                                            \
    PPL_ASSERT(dst != nullptr);                                            \
    PPL_ASSERT(rows >= 1 && cols >= 1);                                    \
    PPL_ASSERT(src_stride >= cols * (int)sizeof(T) * channels);            \
    PPL_ASSERT(dst_stride >= cols * (int)sizeof(T) * channels);            \
  PPL_ASSERT(ksize > 0);\
  PPL_ASSERT((ksize & 1) == 1);\
    PPL_ASSERT(border_type == BORDER_REPLICATE ||                          \
               border_type == BORDER_REFLECT ||                            \
               border_type == BORDER_REFLECT_101)                          \
    cl_int error_code;                                                     \
    cl_mem buffer = clCreateBuffer(                                        \
        context, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,                \
        cols * (int)sizeof(float) * rows * (int)sizeof(float) * channels,  \
        NULL, &error_code);                                                \
    CHECK_ERROR(error_code, clCreateBuffer);                               \
                                                                           \
    cl_mem kernel =                                                        \
        clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,  \
                       ksize * (int)sizeof(float), NULL, &error_code);     \
    CHECK_ERROR(error_code, clCreateBuffer);                               \
                                                                           \
    FrameChain* frame_chain = getSharedFrameChain();                       \
    frame_chain->setProjectName("cv");                                     \
    SET_PROGRAM_SOURCE(frame_chain, gaussianblur);                         \
                                                                           \
    int global_cols, global_rows;                                          \
    size_t global_size[2];                                                 \
                                                                           \
    if (border_type == BORDER_REPLICATE)                                   \
      RUN_KERNEL_CN(interpolateReplicateBorder, base_type, channels)       \
    else if (border_type == BORDER_REFLECT)                                \
      RUN_KERNEL_CN(interpolateReflectBorder, base_type, channels)         \
    else if (border_type == BORDER_REFLECT_101)                            \
      RUN_KERNEL_CN(interpolateReflect101Border, base_type, channels)      \
    return RC_SUCCESS;                                                     \
  }

GAUSSIANBLUR_C1_TYPE(U8, uchar)
GAUSSIANBLUR_C1_TYPE(F32, float)
GAUSSIANBLUR_CN_TYPE(U8, uchar, 3)
GAUSSIANBLUR_CN_TYPE(F32, float, 3)
GAUSSIANBLUR_CN_TYPE(U8, uchar, 4)
GAUSSIANBLUR_CN_TYPE(F32, float, 4)
// GAUSSIANBLUR_CN_TYPE(F32, float, 3)
// GAUSSIANBLUR_CN_TYPE(F32, float, 4)

#define GAUSSIANBLUR_TYPE_C1_TEMPLATE(base_type, T)                           \
  template <>                                                                 \
  RetCode GaussianBlur<T, 1>(cl_context context, cl_command_queue queue,      \
                             int height, int width, int inWidthStride,        \
                             const cl_mem inData, int ksize, float sigma,     \
                             int outWidthStride, cl_mem outData, \
                             BorderType border_type) {                        \
    inWidthStride *= sizeof(T);                                               \
    outWidthStride *= sizeof(T);                                              \
    RetCode code = gaussianblurC1##base_type(                                 \
        inData, height, width, inWidthStride, ksize, sigma, outData,          \
        outWidthStride, border_type, context, queue);                  \
                                                                              \
    return code;                                                              \
  }

#define GAUSSIANBLUR_TYPE_CN_TEMPLATE(base_type, T, channels)            \
  template <>                                                            \
  RetCode GaussianBlur<T, channels>(                                     \
      cl_context context, cl_command_queue queue, int height, int width, \
      int inWidthStride, const cl_mem inData, int ksize, float sigma,    \
      int outWidthStride, cl_mem outData,                   \
      BorderType border_type) {                                          \
    inWidthStride *= sizeof(T);                                          \
    outWidthStride *= sizeof(T);                                         \
    RetCode code = gaussianblurC##channels##base_type(                   \
        inData, height, width, inWidthStride, ksize, sigma, outData,     \
        outWidthStride, border_type, context, queue);             \
                                                                         \
    return code;                                                         \
  }

GAUSSIANBLUR_TYPE_C1_TEMPLATE(U8, uchar)
GAUSSIANBLUR_TYPE_C1_TEMPLATE(F32, float)
GAUSSIANBLUR_TYPE_CN_TEMPLATE(U8, uchar, 3)
GAUSSIANBLUR_TYPE_CN_TEMPLATE(F32, float, 3)
GAUSSIANBLUR_TYPE_CN_TEMPLATE(U8, uchar, 4)
GAUSSIANBLUR_TYPE_CN_TEMPLATE(F32, float, 4)
// GAUSSIANBLUR_TYPE_CN_TEMPLATE(F32, float, 3)
// GAUSSIANBLUR_TYPE_CN_TEMPLATE(F32, float, 4)


}  // namespace ocl
}  // namespace cv
}  // namespace ppl
