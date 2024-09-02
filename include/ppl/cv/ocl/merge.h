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

#ifndef _ST_HPC_PPL_CV_OCL_MERGE_H_
#define _ST_HPC_PPL_CV_OCL_MERGE_H_

#include "CL/cl.h"

#include "ppl/common/retcode.h"

namespace ppl {
namespace cv {
namespace ocl {

template <typename T>
ppl::common::RetCode Merge3Channels(cl_command_queue queue,
                         int height,
                         int width,
                         int inWidthStride,
                         const cl_mem inData0,
                         const cl_mem inData1,
                         const cl_mem inData2,
                         int outWidthStride,
                         cl_mem outData);

template <typename T>
ppl::common::RetCode Merge4Channels(cl_command_queue queue,
                         int height,
                         int width,
                         int inWidthStride,
                         const cl_mem inData0,
                         const cl_mem inData1,
                         const cl_mem inData2,
                         const cl_mem inData3,
                         int outWidthStride,
                         cl_mem outData);

}  // namespace ocl
}  // namespace cv
}  // namespace ppl

#endif  // _ST_HPC_PPL_CV_OCL_MERGE_H_
