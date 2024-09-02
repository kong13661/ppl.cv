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

#include "ppl/cv/ocl/adaptivethreshold.h"

#include "ppl/common/ocl/pplopencl.h"
#include "utility/utility.hpp"

#include "kernels/adaptivethreshold.cl"
#include "kerneltypes.h"
#include <cmath>

using namespace ppl::common;
using namespace ppl::common::ocl;
#define LARGE_MAX_KSIZE 256

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
void getGaussianKernel(float sigma, int ksize, float* coefficients);

#define RUN_KERNEL_MEAN(interpolate, base_type)                                \
  {                                                                            \
    uchar setted_value = 0;                                                    \
    if (maxValue < 255.f) {                                                   \
      setted_value = rintf(maxValue);                                         \
    }                                                                          \
    else {                                                                     \
      setted_value = 255;                                                      \
    }                                                                          \
                                                                               \
    int int_delta = 0;                                                         \
    if (threshold_type == THRESH_BINARY) {                                     \
      int_delta = std::ceil(delta);                                            \
    }                                                                          \
    else {                                                                     \
      int_delta = std::floor(delta);                                           \
    }                                                                          \
    float weight = 1.0f / (float)(ksize * ksize);                              \
                                                                               \
    size_t local_size[] = {kBlockDimX0, kBlockDimY0};                          \
    global_cols = divideUp(cols, F32##DIV, F32##OFFSET);                       \
    global_rows = divideUp(rows, F32##DIV, F32##OFFSET);                       \
    global_size[0] = (size_t)global_cols;                                      \
    global_size[1] = (size_t)global_rows;                                      \
    frame_chain->setCompileOptions("-D BOXFILTER_" #base_type "1C");           \
    runOclKernel(frame_chain,                                                  \
                 "adaptivethreshold_mean" #base_type "F32" #interpolate           \
                 "TRANSPOSEC1Kernel",                                                   \
                 2, global_size, local_size, src, src_stride, src, rows, cols, \
                 ksize >> 1, src_stride, buffer, rows * (int)sizeof(float),    \
                 threshold_type, weight, int_delta, setted_value);              \
                                                                               \
    global_cols = divideUp(cols, base_type##DIV, base_type##OFFSET);           \
    global_rows = divideUp(rows, base_type##DIV, base_type##OFFSET);           \
    global_size[0] = (size_t)global_rows;                                      \
    global_size[1] = (size_t)global_cols;                                      \
    runOclKernel(frame_chain,                                                  \
                 "adaptivethreshold_mean"                                         \
                 "F32" #base_type #interpolate "THRESHOLD_TRANSPOSEC1Kernel",                     \
                 2, global_size, local_size, src, src_stride, buffer, cols,    \
                 rows, ksize >> 1, rows * (int)sizeof(float), dst, dst_stride, \
                 threshold_type, weight, int_delta, setted_value);              \
    clReleaseMemObject(buffer);                                                \
  }

#define ADAPTIVEFILTER_MEAN_BOXFILTER_C1_TYPE(base_type, T)                   \
  RetCode adaptivethreshold_meanC1##base_type(                                   \
      const cl_mem src, int rows, int cols, int src_stride, cl_mem dst,       \
      int dst_stride, float maxValue, int threshold_type, int ksize, float delta,        \
      BorderType border_type, cl_context context, cl_command_queue queue) {   \
    PPL_ASSERT(src != nullptr);                                               \
    PPL_ASSERT(dst != nullptr);                                               \
    PPL_ASSERT(rows >= 1 && cols >= 1);                                       \
    PPL_ASSERT(ksize > 0);                                                    \
    PPL_ASSERT((ksize & 1) == 1);                                             \
    PPL_ASSERT(src_stride >= cols * (int)sizeof(T));                          \
    PPL_ASSERT(dst_stride >= cols * (int)sizeof(T));                          \
    PPL_ASSERT(threshold_type == THRESH_BINARY ||                             \
               threshold_type == THRESH_BINARY_INV);                          \
    PPL_ASSERT((ksize & 1) == 1 && ksize > 1 && ksize < LARGE_MAX_KSIZE);     \
    PPL_ASSERT(border_type == BORDER_REPLICATE ||                             \
               border_type == BORDER_REFLECT ||                               \
               border_type == BORDER_REFLECT_101);                            \
    cl_int error_code;                                                        \
    cl_mem buffer =                                                           \
        clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,    \
                       cols * (int)sizeof(float) * rows * (int)sizeof(float), \
                       NULL, &error_code);                                    \
    CHECK_ERROR(error_code, clCreateBuffer);                                  \
                                                                              \
    FrameChain* frame_chain = getSharedFrameChain();                          \
    frame_chain->setProjectName("cv");                                        \
    SET_PROGRAM_SOURCE(frame_chain, adaptivethreshold);                     \
                                                                              \
    int global_cols, global_rows;                                             \
    size_t global_size[2];                                                    \
                                                                              \
    if (border_type == BORDER_REPLICATE)                                      \
      RUN_KERNEL_MEAN(interpolateReplicateBorder, base_type)                  \
    else if (border_type == BORDER_REFLECT)                                   \
      RUN_KERNEL_MEAN(interpolateReflectBorder, base_type)                    \
    else if (border_type == BORDER_REFLECT_101)                               \
      RUN_KERNEL_MEAN(interpolateReflect101Border, base_type)                 \
    return RC_SUCCESS;                                                        \
  }
ADAPTIVEFILTER_MEAN_BOXFILTER_C1_TYPE(U8, uchar)

// TODO: set zeros if maxValue <= 0
#define RUN_KERNEL_GAUSSIANBLUR(interpolate, base_type)                        \
  {                                                                            \
                                                                               \
    uchar setted_value = 0;                                                    \
    if (maxValue < 255.f) {                                                   \
      setted_value = rintf(maxValue);                                         \
    }                                                                          \
    else {                                                                     \
      setted_value = 255;                                                      \
    }                                                                          \
                                                                               \
    int int_delta = 0;                                                         \
    if (threshold_type == THRESH_BINARY) {                                     \
      int_delta = std::ceil(delta);                                            \
    }                                                                          \
    else {                                                                     \
      int_delta = std::floor(delta);                                           \
    }                                                                          \
                                                                               \
    float* kernel_cpu = new float[ksize];                                      \
    getGaussianKernel(0, ksize, kernel_cpu);                               \
    error_code = clEnqueueWriteBuffer(queue, kernel, CL_FALSE, 0,              \
                                      ksize * (int)sizeof(float), kernel_cpu,  \
                                      0, NULL, NULL);                          \
    CHECK_ERROR(error_code, clEnqueueWriteBuffer);                             \
    delete[] kernel_cpu;                                                       \
                                                                               \
    ksize = ksize >> 1;                                                        \
    size_t local_size[] = {kBlockDimX0, kBlockDimY0};                          \
    global_cols = divideUp(cols, F32##DIV, F32##OFFSET);                       \
    global_rows = divideUp(rows, F32##DIV, F32##OFFSET);                       \
    global_size[0] = (size_t)global_cols;                                      \
    global_size[1] = (size_t)global_rows;                                      \
    frame_chain->setCompileOptions("-D GAUSSIANBLUR_" #base_type "1C");        \
    runOclKernel(frame_chain,                                                  \
                 "adaptivethreshold_gaussianblur" #base_type                   \
                 "F32" #interpolate "TRANSPOSEC1Kernel",                                \
                 2, global_size, local_size, src, src_stride, src, rows, cols, \
                 kernel, ksize, src_stride, buffer, rows*(int)sizeof(float),   \
                 threshold_type, int_delta, setted_value);                      \
                                                                               \
    global_cols = divideUp(cols, base_type##DIV, base_type##OFFSET);           \
    global_rows = divideUp(rows, base_type##DIV, base_type##OFFSET);           \
    global_size[0] = (size_t)global_rows;                                      \
    global_size[1] = (size_t)global_cols;                                      \
    runOclKernel(frame_chain,                                                  \
                 "adaptivethreshold_gaussianblur"                              \
                 "F32" #base_type #interpolate "THRESHOLD_TRANSPOSEC1Kernel",                     \
                 2, global_size, local_size, src, src_stride, buffer, cols,    \
                 rows, kernel, ksize, rows*(int)sizeof(float), dst,            \
                 dst_stride, threshold_type, int_delta, setted_value);          \
    clReleaseMemObject(buffer);                                                \
    clReleaseMemObject(kernel);                                                \
  }

#define ADAPTIVETHRESHOLD_GAUSSIANBLUR_C1_TYPE(base_type, T)                  \
  RetCode adaptivethreshold_gaussianblurC1##base_type(                        \
      const cl_mem src, int rows, int cols, int src_stride, cl_mem dst,       \
      int dst_stride, float maxValue, int threshold_type, int ksize, float delta,    \
      BorderType border_type, cl_context context, cl_command_queue queue) {   \
    PPL_ASSERT(src != nullptr);                                               \
    PPL_ASSERT(dst != nullptr);                                               \
    PPL_ASSERT(rows >= 1 && cols >= 1);                                       \
    PPL_ASSERT(ksize > 0);                                                    \
    PPL_ASSERT((ksize & 1) == 1);                                             \
    PPL_ASSERT(src_stride >= cols * (int)sizeof(T));                          \
    PPL_ASSERT(dst_stride >= cols * (int)sizeof(T));                          \
    PPL_ASSERT(threshold_type == THRESH_BINARY ||                             \
               threshold_type == THRESH_BINARY_INV);                          \
    PPL_ASSERT((ksize & 1) == 1 && ksize > 1 && ksize < LARGE_MAX_KSIZE);     \
    PPL_ASSERT(border_type == BORDER_REPLICATE ||                             \
               border_type == BORDER_REFLECT ||                               \
               border_type == BORDER_REFLECT_101);                            \
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
    SET_PROGRAM_SOURCE(frame_chain, adaptivethreshold);          \
                                                                              \
    int global_cols, global_rows;                                             \
    size_t global_size[2];                                                    \
                                                                              \
    if (border_type == BORDER_REPLICATE)                                      \
      RUN_KERNEL_GAUSSIANBLUR(interpolateReplicateBorder, base_type)          \
    else if (border_type == BORDER_REFLECT)                                   \
      RUN_KERNEL_GAUSSIANBLUR(interpolateReflectBorder, base_type)            \
    else if (border_type == BORDER_REFLECT_101)                               \
      RUN_KERNEL_GAUSSIANBLUR(interpolateReflect101Border, base_type)         \
    return RC_SUCCESS;                                                        \
  }

ADAPTIVETHRESHOLD_GAUSSIANBLUR_C1_TYPE(U8, uchar)

#define AdaptiveThreshold_TYPE_C1_TEMPLATE(base_type, T)                      \
  RetCode AdaptiveThreshold(                                                  \
      cl_context context, cl_command_queue queue, int height, int width,      \
      int inWidthStride, const cl_mem inData, int outWidthStride,             \
      cl_mem outData, float maxValue, int adaptiveMethod, int threshold_type, \
      int blockSize, float delta, BorderType border_type) {                                \
    PPL_ASSERT(adaptiveMethod == ADAPTIVE_THRESH_MEAN_C ||                   \
               adaptiveMethod == ADAPTIVE_THRESH_GAUSSIAN_C);                \
    inWidthStride *= sizeof(T);                                               \
    outWidthStride *= sizeof(T);                                              \
    RetCode code;\
    if (adaptiveMethod == ADAPTIVE_THRESH_MEAN_C) {                          \
      code = adaptivethreshold_meanC1##base_type(                     \
          inData, height, width, inWidthStride, outData, outWidthStride,      \
          maxValue, threshold_type, blockSize, delta, border_type, context, queue);  \
      return code;                                                            \
    }                                                                         \
    else {                                                                    \
      code = adaptivethreshold_gaussianblurC1##base_type(             \
          inData, height, width, inWidthStride, outData, outWidthStride,      \
          maxValue, threshold_type, blockSize, delta, border_type, context, queue);  \
    }                                                                         \
                                                                              \
    return code;                                                              \
  }

AdaptiveThreshold_TYPE_C1_TEMPLATE(U8, uchar)


}  // namespace ocl
}  // namespace cv
}  // namespace ppl
