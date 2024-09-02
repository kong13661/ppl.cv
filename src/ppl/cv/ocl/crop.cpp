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

#include "ppl/common/ocl/pplopencl.h"
#include "utility/utility.hpp"

#include "kernels/crop.cl"

using namespace ppl::common;
using namespace ppl::common::ocl;

#define F32DIV 2
#define F32OFFSET 1
#define U8DIV 4
#define U8OFFSET 2

namespace ppl {
namespace cv {
namespace ocl {

#define CROP_TYPE(base_type, T)                                                \
  RetCode crop##base_type(                                                     \
      const cl_mem src, int src_rows, int src_cols, int channels,              \
      int src_stride, cl_mem dst, int dst_rows, int dst_cols, int dst_stride,  \
      int left, int top, float scale, cl_command_queue queue) {                \
    PPL_ASSERT(src != nullptr);                                                \
    PPL_ASSERT(dst != nullptr);                                                \
    PPL_ASSERT(src_rows >= 1 && src_cols >= 1);                                \
    PPL_ASSERT(dst_rows >= 1 && dst_cols >= 1);                                \
    PPL_ASSERT(src_rows >= dst_rows && src_cols >= dst_cols);                  \
    PPL_ASSERT(left >= 0 && left < src_cols - dst_cols);                       \
    PPL_ASSERT(top >= 0 && top < src_rows - dst_rows);                         \
    PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);               \
    PPL_ASSERT(src_stride >= src_cols * channels * (int)sizeof(T));            \
    PPL_ASSERT(dst_stride >= dst_cols * channels * (int)sizeof(T));            \
                                                                               \
    FrameChain* frame_chain = getSharedFrameChain();                           \
    frame_chain->setProjectName("cv");                                         \
    SET_PROGRAM_SOURCE(frame_chain, crop);                                     \
                                                                               \
    int columns = dst_cols * channels;                                         \
    dst_cols = divideUp(columns, base_type##DIV, base_type##OFFSET);           \
    size_t local_size[] = {kBlockDimX0, kBlockDimY0};                          \
    size_t global_size[] = {(size_t)dst_cols, (size_t)dst_rows};               \
                                                                               \
    frame_chain->setCompileOptions("-D CROP_" #base_type);                     \
    runOclKernel(frame_chain, "crop" #base_type "Kernel", 2, global_size,      \
                 local_size, src, src_stride, top, left* channels, scale, dst, \
                 dst_rows, columns, dst_stride);                               \
                                                                               \
    return RC_SUCCESS;                                                         \
  }

CROP_TYPE(U8, uchar)
CROP_TYPE(F32, float)

#define CROP_TYPE_TEMPLATE(base_type, T, channels)                                       \
template <>                                                                              \
RetCode Crop<T, channels>(cl_command_queue queue,                                        \
                      int inHeight,                                                      \
                       int inWidth,                                                      \
                       int inWidthStride,                                                \
                       const cl_mem inData,                                              \
                       int outHeight,                                                    \
                       int outWidth,                                                     \
                       int outWidthStride,                                               \
                       cl_mem outData,                                                   \
                       const int left,                                                   \
                       const int top,                                                    \
                       const float scale) {                                              \
  inWidthStride *= sizeof(T);                                                            \
  outWidthStride *= sizeof(T);                                                           \
  RetCode code = crop##base_type(inData, inHeight, inWidth, channels, inWidthStride,     \
                outData, outHeight, outWidth, outWidthStride, left, top, scale, queue);  \
                                                                                         \
  return code;                                                                           \
}

CROP_TYPE_TEMPLATE(U8, uchar, 1)
CROP_TYPE_TEMPLATE(U8, uchar, 3)
CROP_TYPE_TEMPLATE(U8, uchar, 4)
CROP_TYPE_TEMPLATE(F32, float, 1)
CROP_TYPE_TEMPLATE(F32, float, 3)
CROP_TYPE_TEMPLATE(F32, float, 4)

}  // namespace ocl
}  // namespace cv
}  // namespace ppl
