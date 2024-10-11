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

#define BLOCK_X 128

namespace ppl {
namespace cv {
namespace ocl {

RetCode integralU8I32(const cl_mem src, int src_rows, int src_cols,
                      int src_stride, cl_mem dst, cl_mem integral_tmp,
                      int dst_rows, int dst_cols, int dst_stride,
                      cl_command_queue queue) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(src_rows >= 1 && src_cols >= 1);
  PPL_ASSERT(dst_rows >= 1 && dst_cols >= 1);
  PPL_ASSERT((dst_rows == src_rows && dst_cols == src_cols) ||
             (dst_rows == src_rows + 1 && dst_cols == src_cols + 1));
  PPL_ASSERT(src_stride >= src_cols * (int)sizeof(uchar));
  PPL_ASSERT(dst_stride >= dst_cols * (int)sizeof(int));
  size_t local_size[] = {BLOCK_X};
  size_t global_size[] = {(size_t)(divideUp(src_rows, 2, 1) * BLOCK_X)};

  FrameChain* frame_chain = getSharedFrameChain();
  frame_chain->setProjectName("cv");
  SET_PROGRAM_SOURCE(frame_chain, integral);

  global_size[0] = divideUp(dst_rows * dst_cols, 2, 1);
  local_size[0] = 128;
  frame_chain->setCompileOptions("-D INTEGRAL_U8");
  runOclKernel(frame_chain, "setZeroI32", 1, global_size, local_size,
               integral_tmp, dst_rows * dst_rows);
  local_size[0] = BLOCK_X;
  global_size[0] = (size_t)(divideUp(src_rows, 2, 1) * BLOCK_X);
  runOclKernel(frame_chain, "integralU8I32Kernel", 1, global_size, local_size,
               src, src_rows, src_cols, src_stride, integral_tmp, dst_cols,
               dst_rows, dst_rows * (int)sizeof(int));
  global_size[0] = (size_t)(divideUp(dst_cols, 2, 1) * BLOCK_X);
  runOclKernel(frame_chain, "integralI32I32Kernel", 1, global_size, local_size,
               integral_tmp, dst_cols, dst_rows, dst_rows * (int)sizeof(int),
               dst, dst_rows, dst_cols, dst_stride);
  return RC_SUCCESS;
}

RetCode integralF32F32(const cl_mem src, int src_rows, int src_cols,
                       int src_stride, cl_mem dst, cl_mem integral_tmp,
                       int dst_rows, int dst_cols, int dst_stride,
                       cl_command_queue queue) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(src_rows >= 1 && src_cols >= 1);
  PPL_ASSERT(dst_rows >= 1 && dst_cols >= 1);
  PPL_ASSERT((dst_rows == src_rows && dst_cols == src_cols) ||
             (dst_rows == src_rows + 1 && dst_cols == src_cols + 1));
  PPL_ASSERT(src_stride >= src_cols * (int)sizeof(float));
  PPL_ASSERT(dst_stride >= dst_cols * (int)sizeof(float));
  size_t local_size[] = {BLOCK_X};
  size_t global_size[] = {(size_t)(divideUp(src_rows, 2, 1) * BLOCK_X)};

  FrameChain* frame_chain = getSharedFrameChain();
  frame_chain->setProjectName("cv");
  SET_PROGRAM_SOURCE(frame_chain, integral);

  frame_chain->setCompileOptions("-D INTEGRAL_F32");
  global_size[0] = divideUp(dst_rows * dst_cols, 2, 1);
  local_size[0] = 128;
  runOclKernel(frame_chain, "setZeroF32", 1, global_size, local_size,
               integral_tmp, dst_rows * dst_rows);
  local_size[0] = BLOCK_X;
  global_size[0] = (size_t)(divideUp(src_rows, 2, 1) * BLOCK_X);
  runOclKernel(frame_chain, "integralF32F32Kernel", 1, global_size, local_size,
               src, src_rows, src_cols, src_stride, integral_tmp, dst_cols,
               dst_rows, dst_rows * (int)sizeof(float));
  global_size[0] = (size_t)(divideUp(dst_cols, 2, 1) * BLOCK_X);
  runOclKernel(frame_chain, "integralF32F32Kernel", 1, global_size, local_size,
               integral_tmp, dst_cols, dst_rows, dst_rows * (int)sizeof(float),
               dst, dst_rows, dst_cols, dst_stride);
  return RC_SUCCESS;
}


template <>
RetCode Integral<uchar, int>(cl_command_queue queue,
                             int inHeight,
                             int inWidth,
                             int inWidthStride,
                             const cl_mem inData,
                             int outHeight,
                             int outWidth,
                             int outWidthStride,
                             cl_mem outData,
                             cl_mem buffer) {
  inWidthStride *= sizeof(uchar);
  outWidthStride *= sizeof(int);
  RetCode code =
      integralU8I32(inData, inHeight, inWidth, inWidthStride, outData, buffer,
                    outHeight, outWidth, outWidthStride, queue);
  return code;
}

template <>
RetCode Integral<float, float>(cl_command_queue queue,
                               int inHeight,
                               int inWidth,
                               int inWidthStride,
                               const cl_mem inData,
                               int outHeight,
                               int outWidth,
                               int outWidthStride,
                               cl_mem outData,
                               cl_mem buffer) {
  inWidthStride *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code =
      integralF32F32(inData, inHeight, inWidth, inWidthStride, outData, buffer,
                     outHeight, outWidth, outWidthStride, queue);
  return code;
}

}  // namespace ocl
}  // namespace cv
}  // namespace ppl
