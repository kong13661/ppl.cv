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
#include "utility/use_memory_pool.h"

#include "ppl/common/ocl/pplopencl.h"
#include "utility/utility.hpp"

#include "kernels/sepfilter2d.cl"
#include "kerneltypes.h"

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

#define RUN_KERNEL(interpolate, base_type_src, base_type_dst)                  \
  {                                                                            \
    global_cols = divideUp(cols, F32##DIV, F32##OFFSET);                       \
    global_rows = divideUp(rows, F32##DIV, F32##OFFSET);                       \
    global_size[0] = (size_t)global_cols;                                      \
    global_size[1] = (size_t)global_rows;                                      \
    frame_chain->setCompileOptions("-D " #interpolate "_SEP_C1");     \
    cl_mem buffer;                                                                         \
    GpuMemoryBlock buffer_block;                                                           \
    buffer_block.offset = 0;\
    if (memoryPoolUsed()) {                                                                \
      pplOclMalloc(buffer_block, rows * (int)sizeof(float) *                               \
                   cols);                                          \
      buffer = buffer_block.data;                                                          \
    }                                                                                      \
    else {                                                                                 \
      cl_int error_code = 0;                                                               \
      buffer = clCreateBuffer(                                                             \
          frame_chain->getContext(), CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,            \
          rows * (int)sizeof(float) * cols,                                                \
          NULL, &error_code);                                                              \
      CHECK_ERROR(error_code, clCreateBuffer);                                             \
    }                                                                                      \
    runOclKernel(frame_chain,                                                  \
                 "sepfilter2d" #base_type_src "F32" #interpolate "C1Kernel",   \
                 2, global_size, local_size, src, 0, rows, cols, kernel_x, ksize, \
                 src_stride, buffer, rows*(int)sizeof(float), is_symmetric,    \
                 0.f, (int)buffer_block.offset);                                                       \
                                                                               \
    global_cols = divideUp(cols, base_type_dst##DIV, base_type_dst##OFFSET);   \
    global_rows = divideUp(rows, base_type_dst##DIV, base_type_dst##OFFSET);   \
    global_size[0] = (size_t)global_rows;                                      \
    global_size[1] = (size_t)global_cols;                                      \
    runOclKernel(frame_chain,                                                  \
                 "sepfilter2d"                                                 \
                 "F32" #base_type_dst #interpolate "C1Kernel",                 \
                 2, global_size, local_size, buffer, (int)buffer_block.offset, cols, rows, kernel_y,     \
                 ksize, rows*(int)sizeof(float), dst, dst_stride,              \
                 is_symmetric, delta, 0);                                         \
    if (memoryPoolUsed()) {                                                     \
      pplOclFree(buffer_block);                                                 \
    }                                                                           \
    else {                                                                      \
      clReleaseMemObject(buffer);                                               \
    }                                                                           \
  }

#define SEPFILTER2D_C1_TYPE(base_type_src, Tsrc, base_type_dst, Tdst)          \
  RetCode sepfilter2dC1##base_type_src##base_type_dst(                         \
      const cl_mem src, int rows, int cols, int src_stride,                    \
      const cl_mem kernel_x, const cl_mem kernel_y, int ksize, cl_mem dst,     \
      int dst_stride, float delta, BorderType border_type, \
      cl_command_queue queue) {                                                \
    PPL_ASSERT(src != nullptr);                                                \
    PPL_ASSERT(dst != nullptr);                                                \
    PPL_ASSERT(rows >= 1 && cols >= 1);                                        \
    PPL_ASSERT(src_stride >= cols * (int)sizeof(Tsrc));                        \
    PPL_ASSERT(dst_stride >= cols * (int)sizeof(Tdst));                        \
    PPL_ASSERT(border_type == BORDER_REPLICATE ||                              \
               border_type == BORDER_REFLECT ||                                \
               border_type == BORDER_REFLECT_101)                              \
    FrameChain* frame_chain = getSharedFrameChain();                           \
    frame_chain->setProjectName("cv");                                         \
    SET_PROGRAM_SOURCE(frame_chain, sepfilter2d);                              \
    \
    int global_cols, global_rows;                                              \
    size_t local_size[] = {kBlockDimX0, kBlockDimY0};                          \
    size_t global_size[2];                                                     \
    int is_symmetric = ksize & 1;                                             \
    ksize = ksize >> 1;                                                        \
                                                                               \
    if (border_type == BORDER_REPLICATE)                                       \
      RUN_KERNEL(interpolateReplicateBorder, base_type_src, base_type_dst)     \
    else if (border_type == BORDER_REFLECT)                                    \
      RUN_KERNEL(interpolateReflectBorder, base_type_src, base_type_dst)       \
    else if (border_type == BORDER_REFLECT_101)                                \
      RUN_KERNEL(interpolateReflect101Border, base_type_src, base_type_dst)    \
    return RC_SUCCESS;                                                         \
  }

#define RUN_KERNEL_CN(interpolate, base_type_src, base_type_dst, channels)     \
  {                                                                            \
    global_cols = cols;                                                        \
    global_rows = divideUp(rows, F32##DIV_CN, F32##OFFSET_CN);                 \
    global_size[0] = (size_t)global_cols;                                      \
    global_size[1] = (size_t)global_rows;                                      \
    frame_chain->setCompileOptions("-D " #interpolate "_SEP_C"            \
                                   #channels);                             \
    runOclKernel(frame_chain,                                                  \
                 "sepfilter2d" #base_type_src "F32" #interpolate "C" #channels \
                 "Kernel",                                                     \
                 2, global_size, local_size, src, 0, rows, cols, kernel_x, ksize, \
                 src_stride, buffer, rows*(int)sizeof(float) * channels, is_symmetric,    \
                 0.f, (int)buffer_block.offset);                                                         \
                                                                               \
    global_cols =                                                              \
        divideUp(cols, base_type_dst##DIV_CN, base_type_dst##OFFSET_CN);       \
    global_rows = rows;                                                        \
    global_size[0] = (size_t)global_rows;                                      \
    global_size[1] = (size_t)global_cols;                                      \
    runOclKernel(frame_chain,                                                  \
                 "sepfilter2d"                                                 \
                 "F32" #base_type_dst #interpolate "C" #channels "Kernel",     \
                 2, global_size, local_size, buffer, (int)buffer_block.offset, cols, rows, kernel_y,     \
                 ksize, rows*(int)sizeof(float) * channels, dst, dst_stride,              \
                 is_symmetric, delta, 0);                                         \
    if (memoryPoolUsed()) {                                                     \
      pplOclFree(buffer_block);                                                 \
    }                                                                           \
    else {                                                                      \
      clReleaseMemObject(buffer);                                               \
    }                                                                           \
  }

#define SEPFILTER2D_CN_TYPE(base_type_src, Tsrc, base_type_dst, Tdst,          \
                            channels)                                          \
  RetCode sepfilter2dC##channels##base_type_src##base_type_dst(                \
      const cl_mem src, int rows, int cols, int src_stride,                    \
      const cl_mem kernel_x, const cl_mem kernel_y, int ksize, cl_mem dst,     \
      int dst_stride, float delta, BorderType border_type, \
      cl_command_queue queue) {                                                \
    PPL_ASSERT(src != nullptr);                                                \
    PPL_ASSERT(dst != nullptr);                                                \
    PPL_ASSERT(rows >= 1 && cols >= 1);                                        \
    PPL_ASSERT(src_stride >= cols * (int)sizeof(Tsrc) * channels);             \
    PPL_ASSERT(dst_stride >= cols * (int)sizeof(Tdst) * channels);             \
    PPL_ASSERT(border_type == BORDER_REPLICATE ||                              \
               border_type == BORDER_REFLECT ||                                \
               border_type == BORDER_REFLECT_101)                              \
    FrameChain* frame_chain = getSharedFrameChain();                           \
    frame_chain->setProjectName("cv");                                         \
    SET_PROGRAM_SOURCE(frame_chain, sepfilter2d);                              \
                                                                               \
    int global_cols, global_rows;                                              \
    size_t local_size[] = {kBlockDimX0, kBlockDimY0};                          \
    size_t global_size[2];                                                     \
    int is_symmetric = ksize & 1;                                              \
    ksize = ksize >> 1;                                                        \
\
    cl_mem buffer;                                                                    \
    GpuMemoryBlock buffer_block;                                                      \
    buffer_block.offset = 0;\
    if (memoryPoolUsed()) {                                                           \
      pplOclMalloc(buffer_block, rows * (int)sizeof(float) *                          \
                   cols * channels);                                     \
      buffer = buffer_block.data;                                                     \
    }                                                                                 \
    else {                                                                            \
      cl_int error_code = 0;                                                          \
      buffer = clCreateBuffer(                                                        \
          frame_chain->getContext(), CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,       \
          rows * (int)sizeof(float) * cols * channels,           \
          NULL, &error_code);                                                         \
      CHECK_ERROR(error_code, clCreateBuffer);                                        \
    }                                                                                 \
                                                                               \
    if (border_type == BORDER_REPLICATE)                                       \
      RUN_KERNEL_CN(interpolateReplicateBorder, base_type_src, base_type_dst,  \
                    channels)                                                  \
    else if (border_type == BORDER_REFLECT)                                    \
      RUN_KERNEL_CN(interpolateReflectBorder, base_type_src, base_type_dst,    \
                    channels)                                                  \
    else if (border_type == BORDER_REFLECT_101)                                \
      RUN_KERNEL_CN(interpolateReflect101Border, base_type_src, base_type_dst, \
                    channels)                                                  \
    return RC_SUCCESS;                                                         \
  }

SEPFILTER2D_C1_TYPE(U8, uchar, U8, uchar)
SEPFILTER2D_C1_TYPE(F32, float, F32, float)
SEPFILTER2D_CN_TYPE(U8, uchar, U8, uchar, 3)
SEPFILTER2D_CN_TYPE(F32, float, F32, float, 3)
SEPFILTER2D_CN_TYPE(U8, uchar, U8, uchar, 4)
SEPFILTER2D_CN_TYPE(F32, float, F32, float, 4)
// SEPFILTER2D_CN_TYPE(F32, float, 3)
// SEPFILTER2D_CN_TYPE(F32, float, 4)

#define SEPFILTER2D_TYPE_C1_TEMPLATE(base_type_src, Tsrc, base_type_dst, Tdst) \
  template <>                                                                  \
  RetCode SepFilter2D<Tsrc, Tdst, 1>(                                          \
      cl_command_queue queue, int height, int width,       \
      int inWidthStride, const cl_mem inData, int ksize, const cl_mem kernelX, \
      const cl_mem kernelY, int outWidthStride, cl_mem outData, float delta,   \
      BorderType border_type) {                                                \
    inWidthStride *= sizeof(Tsrc);                                             \
    outWidthStride *= sizeof(Tdst);                                            \
    RetCode code = sepfilter2dC1##base_type_src##base_type_dst(                \
        inData, height, width, inWidthStride, kernelX, kernelY, ksize,         \
        outData, outWidthStride, delta, border_type, queue);          \
                                                                               \
    return code;                                                               \
  }

#define SEPFILTER2D_TYPE_CN_TEMPLATE(base_type_src, Tsrc, base_type_dst, Tdst, \
                                     channels)                                 \
  template <>                                                                  \
  RetCode SepFilter2D<Tsrc, Tdst, channels>(                                          \
      cl_command_queue queue, int height, int width,       \
      int inWidthStride, const cl_mem inData, int ksize, const cl_mem kernelX, \
      const cl_mem kernelY, int outWidthStride, cl_mem outData, float delta,   \
      BorderType border_type) {                                                \
    inWidthStride *= sizeof(Tsrc);                                             \
    outWidthStride *= sizeof(Tdst);                                            \
    RetCode code = sepfilter2dC##channels##base_type_src##base_type_dst(       \
        inData, height, width, inWidthStride, kernelX, kernelY, ksize,         \
        outData, outWidthStride, delta, border_type, queue);          \
                                                                               \
    return code;                                                               \
  }

SEPFILTER2D_TYPE_C1_TEMPLATE(U8, uchar, U8, uchar)
SEPFILTER2D_TYPE_C1_TEMPLATE(F32, float, F32, float)
SEPFILTER2D_TYPE_CN_TEMPLATE(U8, uchar, U8, uchar, 3)
SEPFILTER2D_TYPE_CN_TEMPLATE(F32, float, F32, float, 3)
SEPFILTER2D_TYPE_CN_TEMPLATE(U8, uchar, U8, uchar, 4)
SEPFILTER2D_TYPE_CN_TEMPLATE(F32, float, F32, float, 4)
// SEPFILTER2D_TYPE_CN_TEMPLATE(F32, float, 3)
// SEPFILTER2D_TYPE_CN_TEMPLATE(F32, float, 4)


}  // namespace ocl
}  // namespace cv
}  // namespace ppl
