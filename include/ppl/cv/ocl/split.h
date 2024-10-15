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

#ifndef _ST_HPC_PPL_CV_OCL_SPLIT_H_
#define _ST_HPC_PPL_CV_OCL_SPLIT_H_

#include "CL/cl.h"

#include "ppl/common/retcode.h"

namespace ppl {
namespace cv {
namespace ocl {

/**
 * @brief Divides a multi-channel array into 3 single-channel arrays.
 * @tparam T The data type, used for both source image and destination image,
 *         currently only uint8_t(uchar) and float are supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less than
 *                       `width * 3 * sizeof(T)`.
 * @param inData         input image data, it should be a buffer object.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride.
 * @param outData0       the first single-channel output image data, it should be a buffer object.
 * @param outData1       the second single-channel output image data, it should be a buffer object.
 * @param outData2       the third single-channel output image data, it should be a buffer object.
 * @return The execution status, succeeds or fails with an error code.
 * @warning All parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>channels
 * <tr><td>uint8_t(uchar)<td>3
 * <tr><td>float<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files <td>#include "ppl/cv/ocl/split.h";
 * <tr><td>Project      <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/split.h"
 * #include "ppl/common/oclcommon.h"
 * using namespace ppl::cv::ocl;
 * using namespace ppl::common::ocl;
 *
 * int main(int argc, char** argv) {
 *   int width    = 640;
 *   int height   = 480;
 *   int channels = 3;
 *
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int data_size = height * width * sizeof(float);
 *   float* input = (float*)malloc(data_size) * channels;
 *   float* output = (float*)malloc(data_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, data_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_output0 = clCreateBuffer(context, CL_MEM_WRITE_ONLY, data_size,
 *                                      NULL, &error_code);
 *   cl_mem gpu_output1 = clCreateBuffer(context, CL_MEM_WRITE_ONLY, data_size,
 *                                      NULL, &error_code);
 *   cl_mem gpu_output2 = clCreateBuffer(context, CL_MEM_WRITE_ONLY, data_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     data_size, input, 0, NULL, NULL);
 *
 *   Split3Channels<float>(queue, height, width,
 *       width * channels, gpu_input, width, 
 *       gpu_output0, gpu_output1, gpu_output2);
 *   error_code = clEnqueueReadBuffer(queue, gpu_output, CL_TRUE, 0, data_size,
 *                                    output, 0, NULL, NULL);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_output0);
 *   clReleaseMemObject(gpu_output1);
 *   clReleaseMemObject(gpu_output2);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode Split3Channels(cl_command_queue queue,
                         int height,
                         int width,
                         int inWidthStride,
                         const cl_mem inData,
                         int outWidthStride,
                         cl_mem outData0,
                         cl_mem outData1,
                         cl_mem outData2);

/**
 * @brief Divides a multi-channel array into 4 single-channel arrays.
 * @tparam T The data type, used for both source image and destination image,
 *         currently only uint8_t(uchar) and float are supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less than
 *                       `width * 4 * sizeof(T)`.
 * @param inData         input image data, it should be a buffer object.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride.
 * @param outData0       the first single-channel output image data, it should be a buffer object.
 * @param outData1       the second single-channel output image data, it should be a buffer object.
 * @param outData2       the third single-channel output image data, it should be a buffer object.
 * @param outData3       the fourth single-channel output image data, it should be a buffer object.
 * @return The execution status, succeeds or fails with an error code, it should be a buffer object.
 * @note 1 For best performance, a 2D array allocated by cudaMallocPitch() is
 *         recommended.
 * @warning All parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>channels
 * <tr><td>uint8_t(uchar)<td>4
 * <tr><td>float<td>4
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files <td>#include "ppl/cv/ocl/split.h";
 * <tr><td>Project      <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/split.h"
 * #include "ppl/common/oclcommon.h"
 * using namespace ppl::cv::ocl;
 * using namespace ppl::common::ocl;
 *
 * int main(int argc, char** argv) {
 *   int width    = 640;
 *   int height   = 480;
 *   int channels = 4;
 *
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int data_size = height * width * sizeof(float);
 *   float* input = (float*)malloc(data_size) * channels;
 *   float* output = (float*)malloc(data_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, data_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_output0 = clCreateBuffer(context, CL_MEM_WRITE_ONLY, data_size,
 *                                      NULL, &error_code);
 *   cl_mem gpu_output1 = clCreateBuffer(context, CL_MEM_WRITE_ONLY, data_size,
 *                                      NULL, &error_code);
 *   cl_mem gpu_output2 = clCreateBuffer(context, CL_MEM_WRITE_ONLY, data_size,
 *                                      NULL, &error_code);
 *   cl_mem gpu_output3 = clCreateBuffer(context, CL_MEM_WRITE_ONLY, data_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     data_size, input, 0, NULL, NULL);
 *
 *   Split4Channels<float>(queue, height, width,
 *       width * channels, gpu_input, width, 
 *       gpu_output0, gpu_output1, gpu_output2, gpu_output3);
 *   error_code = clEnqueueReadBuffer(queue, gpu_output, CL_TRUE, 0, data_size,
 *                                    output, 0, NULL, NULL);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_output0);
 *   clReleaseMemObject(gpu_output1);
 *   clReleaseMemObject(gpu_output2);
 *   clReleaseMemObject(gpu_output3);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode Split4Channels(cl_command_queue queue,
                         int height,
                         int width,
                         int inWidthStride,
                         const cl_mem inData,
                         int outWidthStride,
                         cl_mem outData0,
                         cl_mem outData1,
                         cl_mem outData2,
                         cl_mem outData3);

}  // namespace ocl
}  // namespace cv
}  // namespace ppl

#endif  // _ST_HPC_PPL_CV_OCL_SPLIT_H_
