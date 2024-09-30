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

#include "ppl/cv/ocl/rotate.h"

#include "ppl/common/ocl/pplopencl.h"
#include "utility/utility.hpp"

#include "kernels/rotate.cl"

using namespace ppl::common;
using namespace ppl::common::ocl;

#define F32DIV 2
#define F32OFFSET 1
#define U8DIV 4
#define U8OFFSET 2

#define F32DIV_CN 2
#define F32OFFSET_CN 1
#define U8DIV_CN 4
#define U8OFFSET_CN 2

#define DEGREE_SRC_STRIDE_CHECK_0(T)   src_stride >= src_rows * (int)sizeof(T)
#define DEGREE_SRC_STRIDE_CHECK_90(T)  src_stride >= src_cols * (int)sizeof(T)
#define DEGREE_SRC_STRIDE_CHECK_180(T) src_stride >= src_rows * (int)sizeof(T)
#define DEGREE_SRC_STRIDE_CHECK_270(T) src_stride >= src_cols * (int)sizeof(T)

#define DEGREE_DST_STRIDE_CHECK_0(T)   dst_stride >= src_cols * (int)sizeof(T)
#define DEGREE_DST_STRIDE_CHECK_90(T)  dst_stride >= src_rows * (int)sizeof(T)
#define DEGREE_DST_STRIDE_CHECK_180(T) dst_stride >= src_cols * (int)sizeof(T)
#define DEGREE_DST_STRIDE_CHECK_270(T) dst_stride >= src_rows * (int)sizeof(T)

namespace ppl {
namespace cv {
namespace ocl {

#define DEGREE_C1_TYPE(base_type, T, degree)                                   \
  RetCode rotateC1##degree##base_type(                                         \
      const cl_mem src, int src_rows, int src_cols, int src_stride,            \
      cl_mem dst, int dst_rows, int dst_cols, int dst_stride,                  \
      cl_command_queue queue) {                                                \
    PPL_ASSERT(src != nullptr);                                                \
    PPL_ASSERT(dst != nullptr);                                                \
    PPL_ASSERT(src_rows >= 1 && src_cols >= 1);                                \
    PPL_ASSERT(DEGREE_SRC_STRIDE_CHECK_##degree(T));                           \
    PPL_ASSERT(DEGREE_DST_STRIDE_CHECK_##degree(T));                           \
    PPL_ASSERT((src_rows == dst_rows && src_cols == dst_cols) ||               \
               (src_rows == dst_cols && src_cols == dst_rows));                \
                                                                               \
    FrameChain* frame_chain = getSharedFrameChain();                           \
    frame_chain->setProjectName("cv");                                         \
    SET_PROGRAM_SOURCE(frame_chain, rotate);                                   \
                                                                               \
    int global_cols, global_rows;                                              \
    global_cols = divideUp(src_cols, base_type##DIV, base_type##OFFSET);       \
    global_rows = divideUp(src_rows, base_type##DIV, base_type##OFFSET);       \
    size_t local_size[] = {kBlockDimX0, kBlockDimY0};                          \
    size_t global_size[] = {(size_t)global_cols, (size_t)global_rows};         \
                                                                               \
    frame_chain->setCompileOptions("-D ROTATE" #degree "_" #base_type "C1");   \
    runOclKernel(frame_chain, "rotateC1" #degree #base_type "Kernel", 2,       \
                 global_size, local_size, src, src_rows, src_cols, src_stride, \
                 dst, dst_stride);                                             \
                                                                               \
    return RC_SUCCESS;                                                         \
  }

#define DEGREE_CN_TYPE(base_type, T, degree, channels)                                   \
  RetCode rotateC##channels##degree##base_type(                                         \
      const cl_mem src, int src_rows, int src_cols, int src_stride,            \
      cl_mem dst, int dst_rows, int dst_cols, int dst_stride,                  \
      cl_command_queue queue) {                                                \
    PPL_ASSERT(src != nullptr);                                                \
    PPL_ASSERT(dst != nullptr);                                                \
    PPL_ASSERT(src_rows >= 1 && src_cols >= 1);                                \
    PPL_ASSERT(DEGREE_SRC_STRIDE_CHECK_##degree(T));                           \
    PPL_ASSERT(DEGREE_DST_STRIDE_CHECK_##degree(T));                           \
    PPL_ASSERT((src_rows == dst_rows && src_cols == dst_cols) ||               \
               (src_rows == dst_cols && src_cols == dst_rows));                \
                                                                               \
    FrameChain* frame_chain = getSharedFrameChain();                           \
    frame_chain->setProjectName("cv");                                         \
    SET_PROGRAM_SOURCE(frame_chain, rotate);                                   \
                                                                               \
    int global_cols, global_rows;                                              \
    global_cols = src_cols;       \
    global_rows = divideUp(src_rows, base_type##DIV_CN, base_type##OFFSET_CN);       \
    size_t local_size[] = {kBlockDimX0, kBlockDimY0};                          \
    size_t global_size[] = {(size_t)global_cols, (size_t)global_rows};         \
                                                                               \
    frame_chain->setCompileOptions("-D ROTATE" #degree "_" #base_type "C" #channels);   \
    runOclKernel(frame_chain, "rotateC"#channels #degree #base_type "Kernel", 2,       \
                 global_size, local_size, src, src_rows, src_cols, src_stride, \
                 dst, dst_stride);                                             \
                                                                               \
    return RC_SUCCESS;                                                         \
  }

DEGREE_C1_TYPE(U8, uchar, 90)
DEGREE_C1_TYPE(F32, float, 90)
DEGREE_C1_TYPE(U8, uchar, 180)
DEGREE_C1_TYPE(F32, float, 180)
DEGREE_C1_TYPE(U8, uchar, 270)
DEGREE_C1_TYPE(F32, float, 270)

DEGREE_CN_TYPE(U8, uchar, 90, 3)
DEGREE_CN_TYPE(F32, float, 90, 3)
DEGREE_CN_TYPE(U8, uchar, 180, 3)
DEGREE_CN_TYPE(F32, float, 180, 3)
DEGREE_CN_TYPE(U8, uchar, 270, 3)
DEGREE_CN_TYPE(F32, float, 270, 3)

DEGREE_CN_TYPE(U8, uchar, 90, 4)
DEGREE_CN_TYPE(F32, float, 90, 4)
DEGREE_CN_TYPE(U8, uchar, 180, 4)
DEGREE_CN_TYPE(F32, float, 180, 4)
DEGREE_CN_TYPE(U8, uchar, 270, 4)
DEGREE_CN_TYPE(F32, float, 270, 4)

#define DEGREE_TYPE_TEMPLATE(base_type, channels, T)                        \
  template <>                                                               \
  RetCode Rotate<T, channels>(                                              \
      cl_command_queue queue, int inHeight, int inWidth, int inWidthStride, \
      const cl_mem inData, int outHeight, int outWidth, int outWidthStride, \
      cl_mem outData, int degree) {                                         \
    inWidthStride *= sizeof(T);                                             \
    outWidthStride *= sizeof(T);                                            \
    PPL_ASSERT(degree == 90 || degree == 180 || degree == 270);             \
    RetCode code;                                                           \
    if (degree == 90)                                                       \
      code = rotateC##channels##90##base_type(                              \
          inData, inHeight, inWidth, inWidthStride, outData, outHeight,     \
          outWidth, outWidthStride, queue);                                 \
    else if (degree == 180)                                                 \
      code = rotateC##channels##180##base_type(                             \
          inData, inHeight, inWidth, inWidthStride, outData, outHeight,     \
          outWidth, outWidthStride, queue);                                 \
    else if (degree == 270)                                                 \
      code = rotateC##channels##270##base_type(                             \
          inData, inHeight, inWidth, inWidthStride, outData, outHeight,     \
          outWidth, outWidthStride, queue);                                 \
                                                                            \
    return code;                                                            \
  }

DEGREE_TYPE_TEMPLATE(U8, 1, uchar)
DEGREE_TYPE_TEMPLATE(F32, 1, float)
DEGREE_TYPE_TEMPLATE(U8, 3, uchar)
DEGREE_TYPE_TEMPLATE(F32, 3, float)
DEGREE_TYPE_TEMPLATE(U8, 4, uchar)
DEGREE_TYPE_TEMPLATE(F32, 4, float)


}  // namespace ocl
}  // namespace cv
}  // namespace ppl
