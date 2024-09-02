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

#include "ppl/cv/ocl/equalizehist.h"

#include "ppl/common/ocl/pplopencl.h"
#include "utility/utility.hpp"

#include "kernels/equalizehist.cl"
#include <string.h>

using namespace ppl::common;
using namespace ppl::common::ocl;

namespace ppl {
namespace cv {
namespace ocl {

#define MAX_BLOCKS 128

RetCode equalizehist(const cl_mem src, int rows, int cols,
              int src_stride, cl_mem dst, int dst_stride,
              cl_command_queue queue, cl_context context) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(src_stride >= cols * (int)sizeof(uchar));
  PPL_ASSERT(dst_stride >= cols * (int)sizeof(uchar));

  FrameChain* frame_chain = getSharedFrameChain();
  frame_chain->setProjectName("cv");
  SET_PROGRAM_SOURCE(frame_chain, equalizehist);

  int columns = cols;
  cols = divideUp(columns, 4, 2);
  size_t local_size[2];
  size_t global_size[2];
  cl_int error_code = 0;
  cl_mem hist = clCreateBuffer(context,
                               CL_MEM_READ_WRITE | CL_MEM_HOST_WRITE_ONLY,
                               256 * sizeof(int),
                               NULL,
                               &error_code);
  CHECK_ERROR(error_code, clCreateBuffer);
  int *hist_host = (int *)clEnqueueMapBuffer(queue, hist, CL_TRUE, CL_MAP_WRITE, 0, 256 * sizeof(int),
                                      0, NULL, NULL, &error_code);
  CHECK_ERROR(error_code, clEnqueueMapBuffer);
  memset(hist_host, 0, 256 * sizeof(int));
  error_code = clEnqueueUnmapMemObject(queue, hist, hist_host, 0, NULL, NULL);
  CHECK_ERROR(error_code, clEnqueueUnmapMemObject);

  cl_mem group_count = clCreateBuffer(context,
                               CL_MEM_READ_WRITE | CL_MEM_HOST_WRITE_ONLY,
                               sizeof(int),
                               NULL,
                               &error_code);
  CHECK_ERROR(error_code, clCreateBuffer);
  int zero = 0;
  error_code = clEnqueueWriteBuffer(queue, group_count, CL_FALSE, 0, sizeof(int),
                                    &zero, 0, NULL, NULL);
  CHECK_ERROR(error_code, clEnqueueWriteBuffer);

  if (src_stride == columns && dst_stride == columns) {
    columns *= rows;
    local_size[0]  = 256;
    local_size[1]  = 1;
    global_size[0] = std::min((size_t)(MAX_BLOCKS * 256), (size_t)roundUp(columns, 256, 8));
    global_size[1] = 1;

    frame_chain->setCompileOptions("-D U8 -D U81D");
    runOclKernel(frame_chain, "equalizeHistKernel0", 2, global_size, local_size, src,
                columns, hist, group_count);
    runOclKernel(frame_chain, "equalizeHistKernel00", 2, global_size, local_size, src,
                columns, dst, hist);
  }
  else {
    local_size[0]  = kBlockDimX1;
    local_size[1]  = kBlockDimY1;
    global_size[0] = roundUp(cols, kBlockDimX1, kBlockShiftX1);
    global_size[1] = roundUp(rows, kBlockDimY1, kBlockShiftY1);
    global_size[1] = std::min((size_t)(MAX_BLOCKS * kBlockDimX1 * kBlockDimY1 / global_size[0]), 
                         global_size[1]);

    frame_chain->setCompileOptions("-D U8 -D U8UNALIGNED");
    runOclKernel(frame_chain, "equalizeHistKernel1", 2, global_size, local_size, src,
                src_stride, rows, columns, hist, group_count);
    runOclKernel(frame_chain, "equalizeHistKernel11", 2, global_size, local_size, src,
                src_stride, rows, columns, dst, dst_stride, hist);
  }
  clReleaseMemObject(hist);
  clReleaseMemObject(group_count);
  return RC_SUCCESS;
}


RetCode equalizeHist(cl_command_queue queue,
                int height,
                int width,
                int inWidthStride,
                const cl_mem inData,
                int outWidthStride,
                cl_mem outData,
                cl_context context
                ) {
  RetCode code = equalizehist(inData, height, width, inWidthStride, 
                          outData, outWidthStride, queue, context);
  return code;
}

}  // namespace ocl
}  // namespace cv
}  // namespace ppl
