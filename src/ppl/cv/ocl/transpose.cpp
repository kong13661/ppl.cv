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

#include "ppl/cv/ocl/transpose.h"

#include "ppl/common/ocl/pplopencl.h"
#include "utility/utility.hpp"

#include "kernels/transpose.cl"

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

#define TRANSPOSE_C1_TYPE(base_type, T)                                      \
  RetCode transposeC1##base_type(const cl_mem src, int rows, int cols,       \
                                 int src_stride, cl_mem dst, int dst_stride, \
                                 cl_command_queue queue) {                   \
    PPL_ASSERT(src != nullptr);                                              \
    PPL_ASSERT(dst != nullptr);                                              \
    PPL_ASSERT(rows >= 1 && cols >= 1);                                      \
    PPL_ASSERT(src_stride >= cols * (int)sizeof(T));                         \
    PPL_ASSERT(dst_stride >= rows * (int)sizeof(T));                         \
                                                                             \
    FrameChain* frame_chain = getSharedFrameChain();                         \
    frame_chain->setProjectName("cv");                                       \
    SET_PROGRAM_SOURCE(frame_chain, transpose);                              \
                                                                             \
    int global_cols, global_rows;                                            \
    global_cols = divideUp(cols, base_type##DIV, base_type##OFFSET);         \
    global_rows = divideUp(rows, base_type##DIV, base_type##OFFSET);         \
    size_t local_size[] = {kBlockDimX0, kBlockDimY0};                        \
    size_t global_size[] = {(size_t)global_cols, (size_t)global_rows};       \
                                                                             \
    frame_chain->setCompileOptions("-D TRANSPOSE_" #base_type "1C");         \
    runOclKernel(frame_chain, "transpose" #base_type "C1Kernel", 2,          \
                 global_size, local_size, src, rows, cols, src_stride, dst,  \
                 dst_stride);                                                \
                                                                             \
    return RC_SUCCESS;                                                       \
  }

#define TRANSPOSE_CN_TYPE(base_type, T, channels)                              \
  RetCode transposeC##channels##base_type(                                     \
      const cl_mem src, int rows, int cols, int src_stride, cl_mem dst,        \
      int dst_stride, cl_command_queue queue) {                                \
    PPL_ASSERT(src != nullptr);                                                \
    PPL_ASSERT(dst != nullptr);                                                \
    PPL_ASSERT(rows >= 1 && cols >= 1);                                        \
    PPL_ASSERT(src_stride >= cols * (int)sizeof(T) * channels);                \
    PPL_ASSERT(dst_stride >= rows * (int)sizeof(T) * channels);                \
                                                                               \
    FrameChain* frame_chain = getSharedFrameChain();                           \
    frame_chain->setProjectName("cv");                                         \
    SET_PROGRAM_SOURCE(frame_chain, transpose);                                \
                                                                               \
    int global_cols, global_rows;                                              \
    global_cols = cols;                                                        \
    global_rows = divideUp(rows, base_type##DIV_CN, base_type##OFFSET_CN);     \
    size_t local_size[] = {kBlockDimX0, kBlockDimY0};                          \
    size_t global_size[] = {(size_t)global_cols, (size_t)global_rows};         \
                                                                               \
    frame_chain->setCompileOptions("-D TRANSPOSE_" #base_type "C" #channels);  \
    runOclKernel(frame_chain, "transpose" #base_type "C" #channels "Kernel",   \
                 2, global_size, local_size, src, rows, cols, src_stride, dst, \
                 dst_stride);                                                  \
                                                                               \
    return RC_SUCCESS;                                                         \
  }

TRANSPOSE_C1_TYPE(U8, uchar)
TRANSPOSE_C1_TYPE(F32, float)
TRANSPOSE_CN_TYPE(U8, uchar, 3)
TRANSPOSE_CN_TYPE(U8, uchar, 4)
TRANSPOSE_CN_TYPE(F32, float, 3)
TRANSPOSE_CN_TYPE(F32, float, 4)


#define TRANSPOSE_TYPE_C1_TEMPLATE(base_type, T)                               \
  template <>                                                                  \
  RetCode Transpose<T, 1>(cl_command_queue queue, int height, int width,       \
                          int inWidthStride, const cl_mem inData,              \
                          int outWidthStride, cl_mem outData) {                \
    inWidthStride *= sizeof(T);                                                \
    outWidthStride *= sizeof(T);                                               \
    RetCode code = transposeC1##base_type(                                     \
        inData, height, width, inWidthStride, outData, outWidthStride, queue); \
                                                                               \
    return code;                                                               \
  }

#define TRANSPOSE_TYPE_CN_TEMPLATE(base_type, T, channels)                     \
  template <>                                                                  \
  RetCode Transpose<T, channels>(cl_command_queue queue, int height, int width,\
                          int inWidthStride, const cl_mem inData,              \
                          int outWidthStride, cl_mem outData) {                \
    inWidthStride *= sizeof(T);                                                \
    outWidthStride *= sizeof(T);                                               \
    RetCode code = transposeC##channels##base_type(                            \
        inData, height, width, inWidthStride, outData, outWidthStride, queue); \
                                                                               \
    return code;                                                               \
  }

TRANSPOSE_TYPE_C1_TEMPLATE(U8, uchar)
TRANSPOSE_TYPE_C1_TEMPLATE(F32, float)
TRANSPOSE_TYPE_CN_TEMPLATE(U8, uchar, 3)
TRANSPOSE_TYPE_CN_TEMPLATE(U8, uchar, 4)
TRANSPOSE_TYPE_CN_TEMPLATE(F32, float, 3)
TRANSPOSE_TYPE_CN_TEMPLATE(F32, float, 4)


}  // namespace ocl
}  // namespace cv
}  // namespace ppl
