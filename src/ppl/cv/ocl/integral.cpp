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

#include "ppl/cv/ocl/integral.h"

#include "ppl/common/ocl/pplopencl.h"
#include "utility/utility.hpp"

#include "kernels/integral.cl"
#include <string.h>

using namespace ppl::common;
using namespace ppl::common::ocl;

#define F32DIV 2
#define F32OFFSET 1
#define I32DIV 2
#define I32OFFSET 1

#define BLOCK_X 128

namespace ppl {
namespace cv {
namespace ocl {

#define INTEGRAL_TYPE(src_base_type, Tsrc, dst_base_type, Tdst)                          \
  RetCode integral##src_base_type##dst_base_type(                                        \
      const cl_mem src, int src_rows, int src_cols,                                      \
      int src_stride, cl_mem dst, cl_mem integral_tmp, int dst_rows, int dst_cols, int dst_stride,            \
      cl_command_queue queue) {                                      \
    PPL_ASSERT(src != nullptr);                                                          \
    PPL_ASSERT(dst != nullptr);                                                          \
    PPL_ASSERT(src_rows >= 1 && src_cols >= 1);                                          \
    PPL_ASSERT(dst_rows >= 1 && dst_cols >= 1);                                          \
    PPL_ASSERT((dst_rows == src_rows && dst_cols == src_cols) ||                         \
               (dst_rows == src_rows + 1 && dst_cols == src_cols + 1));                  \
    PPL_ASSERT(src_stride >= src_cols * (int)sizeof(Tsrc));                              \
    PPL_ASSERT(dst_stride >= dst_cols * (int)sizeof(Tdst));                              \
                                                                                         \
                                                                                         \
    size_t local_size[] = {BLOCK_X};                                                     \
    size_t global_size[] = {(size_t)(divideUp(src_rows,                                  \
                                              dst_base_type##DIV,                        \
                                              dst_base_type##OFFSET) * BLOCK_X)};        \
    FrameChain* frame_chain = getSharedFrameChain();                                     \
    frame_chain->setProjectName("cv");                                                   \
    SET_PROGRAM_SOURCE(frame_chain, integral);                                           \
    global_size[0] = divideUp(dst_rows * dst_cols, 2, 1);\
    local_size[0] = 128;\
    runOclKernel(frame_chain, "setZero" #dst_base_type, 1,      \
                 global_size, local_size, integral_tmp, dst_rows * dst_rows);        \
                                                                                         \
                                                                                         \
    local_size[0] = BLOCK_X;\
    global_size[0] = (size_t)(divideUp(src_rows,                                         \
                                       dst_base_type##DIV,                               \
                                       dst_base_type##OFFSET) * BLOCK_X);                \
    frame_chain->setCompileOptions("-D CROP_" #src_base_type);                           \
    runOclKernel(frame_chain, "integral" #src_base_type #dst_base_type "Kernel", 1,      \
                 global_size, local_size, src, src_rows, src_cols, src_stride,           \
                 integral_tmp, dst_cols, dst_rows, dst_rows * (int)sizeof(Tdst));        \
                                                                                         \
    global_size[0] = (size_t)(divideUp(dst_cols,                                         \
                                       dst_base_type##DIV,                               \
                                       dst_base_type##OFFSET) * BLOCK_X);                \
    runOclKernel(frame_chain, "integral" #dst_base_type #dst_base_type "Kernel", 1,      \
                 global_size, local_size, integral_tmp, dst_cols, dst_rows,              \
                 dst_rows * (int)sizeof(Tdst), dst, dst_rows, dst_cols, dst_stride);     \
                                                                                         \
    return RC_SUCCESS;                                                                   \
  }

INTEGRAL_TYPE(U8, uchar, I32, int)
INTEGRAL_TYPE(F32, float, F32, float)

#define INTEGRAL_TYPE_TEMPLATE(src_base_type, Tsrc, dst_base_type, Tdst)                 \
template <>                                                                              \
RetCode Integral<Tsrc, Tdst>(cl_command_queue queue,                                     \
                             int inHeight,                                               \
                             int inWidth,                                                \
                             int inWidthStride,                                          \
                             const cl_mem inData,                                        \
                             int outHeight,                                              \
                             int outWidth,                                               \
                             int outWidthStride,                                         \
                             cl_mem outData,                                             \
                             cl_mem buffer) {                                       \
  inWidthStride *= sizeof(Tsrc);                                                         \
  outWidthStride *= sizeof(Tdst);                                                        \
  RetCode code = integral##src_base_type##dst_base_type(inData, inHeight, inWidth,       \
   inWidthStride, outData, buffer, outHeight, outWidth, outWidthStride, queue);         \
                                                                                         \
  return code;                                                                           \
}

INTEGRAL_TYPE_TEMPLATE(U8, uchar, I32, int)
INTEGRAL_TYPE_TEMPLATE(F32, float, F32, float)

}  // namespace ocl
}  // namespace cv
}  // namespace ppl
