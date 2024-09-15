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

#include "ppl/cv/ocl/filter2d.h"

#include "ppl/common/ocl/pplopencl.h"
#include "utility/utility.hpp"

#include "kernels/filter2d.cl"
#include "kerneltypes.h"

using namespace ppl::common;
using namespace ppl::common::ocl;

namespace ppl {
namespace cv {
namespace ocl {

#define RUN_KERNEL(interpolate, base_type)                                           \
  {                                                                                  \
    if (channels == 1){                                                              \
      frame_chain->setCompileOptions("-D SEPFILTER2D_" #base_type "1C");             \
      runOclKernel(frame_chain,                                                      \
                  "filter2D" #base_type "C1" #interpolate "Kernel",                  \
                  2, global_size, local_size, src, rows, cols, src_stride, kernel,   \
                  radius, dst, dst_stride, delta);                                   \
    }                                                                                \
    else{                                                                            \
      if (channels == 3){                                                            \
        frame_chain->setCompileOptions("-D SEPFILTER2D_" #base_type "1C");           \
        runOclKernel(frame_chain,                                                    \
                    "filter2D" #base_type "C3" #interpolate "Kernel",                \
                    2, global_size, local_size, src, rows, cols, src_stride, kernel, \
                    radius, dst, dst_stride, delta);                                 \
      }                                                                              \
      else {                                                                         \
        frame_chain->setCompileOptions("-D SEPFILTER2D_" #base_type "1C");           \
        runOclKernel(frame_chain,                                                    \
                    "filter2D" #base_type "C4" #interpolate "Kernel",                \
                    2, global_size, local_size, src, rows, cols, src_stride, kernel, \
                    radius, dst, dst_stride, delta);                                 \
      }                                                                              \
    }                                                                                \
  }

#define FILTER2D_TYPE(base_type, T)                                                \
  RetCode filter2D##base_type(                                                     \
    const cl_mem src, int rows, int cols, int channels,                            \
    int src_stride, const cl_mem kernel, int ksize, cl_mem dst,                    \
    int dst_stride, float delta, BorderType border_type, cl_command_queue queue) { \
    PPL_ASSERT(src != nullptr);                                                    \
    PPL_ASSERT(kernel != nullptr);                                                 \
    PPL_ASSERT(dst != nullptr);                                                    \
    PPL_ASSERT(rows >= 1 && cols >= 1);                                            \
    PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);                   \
    PPL_ASSERT(src_stride >= cols * channels * (int)sizeof(T));                \
    PPL_ASSERT(dst_stride >= cols * channels * (int)sizeof(T));                \
    PPL_ASSERT(ksize > 0);                                                         \
    PPL_ASSERT((ksize & 1) == 1);                                                  \
    PPL_ASSERT(border_type == BORDER_REPLICATE ||                                  \
              border_type == BORDER_REFLECT ||                                     \
              border_type == BORDER_REFLECT_101);                                  \
  int radius = ksize >> 1;                                                         \
                                                                                   \
    FrameChain* frame_chain = getSharedFrameChain();                               \
    frame_chain->setProjectName("cv");                                             \
    SET_PROGRAM_SOURCE(frame_chain, filter2d);                                     \
                                                                                   \
    int global_cols;                                                  \
    size_t local_size[] = {kBlockDimX0, kBlockDimY0};                              \
                                                                                   \
    if (channels == 1)                                                             \
      global_cols = divideUp(cols, 4, 2);                                          \
    else                                                                           \
      global_cols = cols;                                                          \
    size_t global_size[] = {(size_t)global_cols, (size_t)rows};                    \
    ksize = ksize >> 1;                                                            \
                                                                                   \
    if (border_type == BORDER_REPLICATE) {                                         \
      RUN_KERNEL(interpolateReplicateBorder, base_type)                            \
    } else if (border_type == BORDER_REFLECT) {                                    \
      RUN_KERNEL(interpolateReflectBorder, base_type)                              \
    } else if (border_type == BORDER_REFLECT_101) {                                \
      RUN_KERNEL(interpolateReflect101Border, base_type)                           \
    }                                                                              \
                                                                                   \
    return RC_SUCCESS;                                                             \
  }

FILTER2D_TYPE(U8, uchar)
FILTER2D_TYPE(F32, float)

#define FILTER2D_TYPE_TEMPLATE(base_type, T, channels)                          \
  template <>                                                                  \
  RetCode Filter2D<T, channels>(                                                \
      cl_command_queue queue, int height, int width,                           \
      int inWidthStride, const cl_mem inData, int ksize, const cl_mem kernel,  \
      int outWidthStride, cl_mem outData, float delta,                         \
      BorderType border_type) {                                                \
    inWidthStride *= sizeof(T);                                                \
    outWidthStride *= sizeof(T);                                               \
    RetCode code = filter2D##base_type(                                        \
        inData, height, width, channels, inWidthStride, kernel, ksize,         \
        outData, outWidthStride, delta, border_type, queue);                   \
                                                                               \
    return code;                                                               \
  }

FILTER2D_TYPE_TEMPLATE(U8, uchar, 1)
FILTER2D_TYPE_TEMPLATE(U8, uchar, 3)
FILTER2D_TYPE_TEMPLATE(U8, uchar, 4)
FILTER2D_TYPE_TEMPLATE(F32, float, 1)
FILTER2D_TYPE_TEMPLATE(F32, float, 3)
FILTER2D_TYPE_TEMPLATE(F32, float, 4)

// SEPFILTER2D_TYPE_CN_TEMPLATE(F32, float, 3)
// SEPFILTER2D_TYPE_CN_TEMPLATE(F32, float, 4)


}  // namespace ocl
}  // namespace cv
}  // namespace ppl
