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

#include "ppl/cv/ocl/split.h"

#include "ppl/common/ocl/pplopencl.h"
#include "utility/utility.hpp"

#include "kernels/split.cl"

using namespace ppl::common;
using namespace ppl::common::ocl;

namespace ppl {
namespace cv {
namespace ocl {

#define SPLIT3CHANNELS_TYPE(base_type, T)                                      \
  RetCode split3##base_type(                                                   \
      const cl_mem src, int rows, int cols, int src_stride, cl_mem dst0,       \
      cl_mem dst1, cl_mem dst2, int dst_stride, cl_command_queue queue) {      \
    PPL_ASSERT(src != nullptr);                                                \
    PPL_ASSERT(dst0 != nullptr);                                               \
    PPL_ASSERT(dst1 != nullptr);                                               \
    PPL_ASSERT(dst2 != nullptr);                                               \
    PPL_ASSERT(rows >= 1 && cols >= 1);                                        \
    PPL_ASSERT(src_stride >= cols * 3 * (int)sizeof(T));                       \
    PPL_ASSERT(dst_stride >= cols * (int)sizeof(T));                           \
                                                                               \
    FrameChain* frame_chain = getSharedFrameChain();                           \
    frame_chain->setProjectName("cv");                                         \
    SET_PROGRAM_SOURCE(frame_chain, split);                                    \
                                                                               \
    int columns = cols;                                                        \
    size_t local_size[] = {kBlockDimX0, kBlockDimY0};                          \
    size_t global_size[] = {(size_t)cols, (size_t)rows};                       \
                                                                               \
    if (src_stride == 3 * columns * (int)sizeof(T) &&                          \
        dst_stride == columns * (int)sizeof(T)) {                              \
      columns *= rows;                                                         \
      cols = columns;                                                          \
      local_size[0] = 512;                                                     \
      local_size[1] = 1;                                                       \
      global_size[0] = (size_t)roundUp(cols, 512, 9);                          \
      global_size[1] = 1;                                                      \
      frame_chain->setCompileOptions("-D SPLIT3_" #base_type "1D");            \
      runOclKernel(frame_chain, "split3" #base_type "Kernel0", 2, global_size, \
                   local_size, src, columns, dst0, dst1, dst2);                \
    }                                                                          \
    else {                                                                     \
      frame_chain->setCompileOptions("-D SPLIT3_" #base_type "2D");            \
      runOclKernel(frame_chain, "split3" #base_type "Kernel1", 2, global_size, \
                   local_size, src, rows, columns, src_stride, dst0, dst1,     \
                   dst2, dst_stride);                                          \
    }                                                                          \
                                                                               \
    return RC_SUCCESS;                                                         \
  }

#define SPLIT4CHANNELS_TYPE(base_type, T)                                      \
  RetCode split4##base_type(const cl_mem src, int rows, int cols,              \
                            int src_stride, cl_mem dst0, cl_mem dst1,          \
                            cl_mem dst2, cl_mem dst3, int dst_stride,          \
                            cl_command_queue queue) {                          \
    PPL_ASSERT(src != nullptr);                                                \
    PPL_ASSERT(dst0 != nullptr);                                               \
    PPL_ASSERT(dst1 != nullptr);                                               \
    PPL_ASSERT(dst2 != nullptr);                                               \
    PPL_ASSERT(dst3 != nullptr);                                               \
    PPL_ASSERT(rows >= 1 && cols >= 1);                                        \
    PPL_ASSERT(src_stride >= cols * 4 * (int)sizeof(T));                       \
    PPL_ASSERT(dst_stride >= cols * (int)sizeof(T));                           \
                                                                               \
    FrameChain* frame_chain = getSharedFrameChain();                           \
    frame_chain->setProjectName("cv");                                         \
    SET_PROGRAM_SOURCE(frame_chain, split);                                    \
                                                                               \
    int columns = cols;                                                        \
    cols = columns;                                                            \
    size_t local_size[] = {kBlockDimX0, kBlockDimY0};                          \
    size_t global_size[] = {(size_t)cols, (size_t)rows};                       \
                                                                               \
    if (src_stride == 4 * columns * (int)sizeof(T) &&                          \
        dst_stride == columns * (int)sizeof(T)) {                              \
      columns *= rows;                                                         \
      cols = columns;                                                          \
      local_size[0] = 512;                                                     \
      local_size[1] = 1;                                                       \
      global_size[0] = (size_t)roundUp(cols, 512, 9);                          \
      global_size[1] = 1;                                                      \
      frame_chain->setCompileOptions("-D SPLIT4_" #base_type "1D");            \
      runOclKernel(frame_chain, "split4" #base_type "Kernel0", 2, global_size, \
                   local_size, src, columns, dst0, dst1, dst2, dst3);          \
    }                                                                          \
    else {                                                                     \
      frame_chain->setCompileOptions("-D SPLIT4_" #base_type "2D");            \
      runOclKernel(frame_chain, "split4" #base_type "Kernel1", 2, global_size, \
                   local_size, src, rows, columns, src_stride, dst0, dst1,     \
                   dst2, dst3, dst_stride);                                    \
    }                                                                          \
                                                                               \
    return RC_SUCCESS;                                                         \
  }

SPLIT3CHANNELS_TYPE(U8 , uchar)
SPLIT3CHANNELS_TYPE(F32, float)
SPLIT4CHANNELS_TYPE(U8 , uchar)
SPLIT4CHANNELS_TYPE(F32, float)

#define SPLIT3CHANNELS_TEMPLATE(base_type, T)                              \
  template <>                                                              \
  RetCode Split3Channels<T>(cl_command_queue queue, int height, int width, \
                            int inWidthStride, const cl_mem inData,        \
                            int outWidthStride, cl_mem outData0,           \
                            cl_mem outData1, cl_mem outData2) {            \
    inWidthStride *= sizeof(T);                                            \
    outWidthStride *= sizeof(T);                                           \
    RetCode code =                                                         \
        split3##base_type(inData, height, width, inWidthStride, outData0,  \
                          outData1, outData2, outWidthStride, queue);      \
                                                                           \
    return code;                                                           \
  }

#define SPLIT4CHANNELS_TEMPLATE(base_type, T)                                \
  template <>                                                                \
  RetCode Split4Channels<T>(                                                 \
      cl_command_queue queue, int height, int width, int inWidthStride,      \
      const cl_mem inData, int outWidthStride, cl_mem outData0,              \
      cl_mem outData1, cl_mem outData2, cl_mem outData3) {                   \
    inWidthStride *= sizeof(T);                                              \
    outWidthStride *= sizeof(T);                                             \
    RetCode code = split4##base_type(inData, height, width, inWidthStride,   \
                                     outData0, outData1, outData2, outData3, \
                                     outWidthStride, queue);                 \
                                                                             \
    return code;                                                             \
  }


SPLIT3CHANNELS_TEMPLATE(U8 , uchar)
SPLIT3CHANNELS_TEMPLATE(F32, float)
SPLIT4CHANNELS_TEMPLATE(U8 , uchar)
SPLIT4CHANNELS_TEMPLATE(F32, float)

}  // namespace ocl
}  // namespace cv
}  // namespace ppl
