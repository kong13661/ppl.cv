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

#include "ppl/cv/ocl/convertto.h"

#include "ppl/common/ocl/pplopencl.h"
#include "utility/utility.hpp"

#include "kernels/convertto.cl"

using namespace ppl::common;
using namespace ppl::common::ocl;

namespace ppl {
namespace cv {
namespace ocl {

#define float_aligned 7
#define uchar_aligned 1

#define CONVERTTOTYPE_F32(base_type, Tsrc)                                    \
  RetCode convertto##base_type##_2_F32(                                       \
      const cl_mem src0, int rows, int cols, int channels, int src0_stride,   \
      cl_mem dst, int dst_stride, float scale, float delta,                   \
      cl_command_queue queue) {                                               \
    PPL_ASSERT(src0 != nullptr);                                              \
    PPL_ASSERT(dst != nullptr);                                               \
    PPL_ASSERT(rows >= 1 && cols >= 1);                                       \
    PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);              \
    PPL_ASSERT(src0_stride >= cols * channels * (int)sizeof(Tsrc));           \
    PPL_ASSERT(dst_stride >= cols * channels * (int)sizeof(float));           \
                                                                              \
    FrameChain* frame_chain = getSharedFrameChain();                          \
    frame_chain->setProjectName("cv");                                        \
    SET_PROGRAM_SOURCE(frame_chain, convertto);                               \
                                                                              \
    int columns = cols * channels;                                            \
    size_t local_size[] = {kBlockDimX0, kBlockDimY0};                         \
    size_t global_size[] = {(size_t)divideUp(columns, 2, 1), (size_t)rows};   \
                                                                              \
    if ((src0_stride & Tsrc##_aligned) == 0 && (dst_stride & 7) == 0) {       \
      cols = divideUp(columns, 2, 1);                                         \
      frame_chain->setCompileOptions("-D CONVERTTO_" #base_type               \
                                     "_2_F32ALIGNED");                        \
      runOclKernel(frame_chain, "convertto" #base_type "_2_F32Kernel0", 2,    \
                   global_size, local_size, src0, rows, cols, src0_stride,    \
                   dst, dst_stride, scale, delta);                            \
    }                                                                         \
    else {                                                                    \
      frame_chain->setCompileOptions("-D CONVERTTO_" #base_type               \
                                     "_2_F32UNALIGNED");                      \
      runOclKernel(frame_chain, "convertto" #base_type "_2_F32Kernel1", 2,    \
                   global_size, local_size, src0, rows, columns, src0_stride, \
                   dst, dst_stride, scale, delta);                            \
    }                                                                         \
                                                                              \
    return RC_SUCCESS;                                                        \
  }

#define CONVERTTOTYPE_U8(base_type, Tsrc)                                      \
  RetCode convertto##base_type##_2_U8(const cl_mem src0, int rows, int cols,   \
                                      int channels, int src0_stride,           \
                                      cl_mem dst, int dst_stride, float scale, \
                                      float delta, cl_command_queue queue) {   \
    PPL_ASSERT(src0 != nullptr);                                               \
    PPL_ASSERT(dst != nullptr);                                                \
    PPL_ASSERT(rows >= 1 && cols >= 1);                                        \
    PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);               \
    PPL_ASSERT(src0_stride >= cols * channels * (int)sizeof(Tsrc));            \
    PPL_ASSERT(dst_stride >= cols * channels * (int)sizeof(uchar));            \
                                                                               \
    FrameChain* frame_chain = getSharedFrameChain();                           \
    frame_chain->setProjectName("cv");                                         \
    SET_PROGRAM_SOURCE(frame_chain, convertto);                                \
                                                                               \
    int columns = cols * channels;                                             \
    if (src0_stride * (int)sizeof(Tsrc) == columns && dst_stride == columns) { \
      columns *= rows;                                                         \
      cols = divideUp(columns, 4, 2);                                          \
      size_t local_size[] = {512, 1};                                          \
      size_t global_size[] = {(size_t)roundUp(cols, 512, 9), 1};               \
                                                                               \
      frame_chain->setCompileOptions("-D CONVERTTO_" #base_type "_2_U81D");    \
      runOclKernel(frame_chain, "convertto" #base_type "_2_U8Kernel0", 2,      \
                   global_size, local_size, src0, columns, dst, scale, delta); \
    }                                                                          \
    else {                                                                     \
      columns = cols * channels;                                               \
      cols = divideUp(columns, 4, 2);                                          \
      size_t local_size[] = {kBlockDimX0, kBlockDimY0};                        \
      size_t global_size[] = {(size_t)cols, (size_t)rows};                     \
                                                                               \
      frame_chain->setCompileOptions("-D CONVERTTO_" #base_type "_2_U82D");    \
      runOclKernel(frame_chain, "convertto" #base_type "_2_U8Kernel1", 2,      \
                   global_size, local_size, src0, rows, columns, src0_stride,  \
                   dst, dst_stride, scale, delta);                             \
    }                                                                          \
                                                                               \
    return RC_SUCCESS;                                                         \
  }

#define CONVERTTO_TEMPLATE(src_type, dst_type, Tsrc, Tdst, channels)         \
  template <>                                                                \
  RetCode Convertto<Tsrc, Tdst, channels>(                                   \
      cl_command_queue queue, int height, int width, int inWidthStride0,     \
      const cl_mem inData0, int outWidthStride, cl_mem outData, float scale, \
      float delta) {                                                         \
    inWidthStride0 *= sizeof(Tsrc);                                          \
    outWidthStride *= sizeof(Tdst);                                          \
    RetCode code = convertto##src_type##_2_##dst_type(                       \
        inData0, height, width, channels, inWidthStride0, outData,           \
        outWidthStride, scale, delta, queue);                                \
    return code;                                                             \
  }

#define CONVERTTO_TEMPLATE_TYPE(src_type, dst_type, Tsrc, Tdst) \
  CONVERTTO_TEMPLATE(src_type, dst_type, Tsrc, Tdst, 1)         \
  CONVERTTO_TEMPLATE(src_type, dst_type, Tsrc, Tdst, 3)         \
  CONVERTTO_TEMPLATE(src_type, dst_type, Tsrc, Tdst, 4)

CONVERTTOTYPE_F32(F32, float)
CONVERTTOTYPE_F32(U8, uchar)
CONVERTTOTYPE_U8(F32, float)
CONVERTTOTYPE_U8(U8, uchar)

CONVERTTO_TEMPLATE_TYPE(F32, F32, float, float)
CONVERTTO_TEMPLATE_TYPE(U8, F32, uchar, float)
CONVERTTO_TEMPLATE_TYPE(F32, U8, float, uchar)
CONVERTTO_TEMPLATE_TYPE(U8, U8, uchar, uchar)

}  // namespace ocl
}  // namespace cv
}  // namespace ppl
