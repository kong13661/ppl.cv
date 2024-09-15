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

#include "kerneltypes.h"

#define convert_float_sat convert_float

#define FILTER2DC1KERNEL(base_type, T, interpolation)                             \
__kernel void filter2D##base_type##C1##interpolation##Kernel(                     \
  const global T* src, int rows, int cols, int src_stride,                        \
  const global float* src_kernel, int radius, global T* dst,                      \
  int dst_stride, float delta){                                                   \
  int element_x, element_y;                                                       \
  element_x = get_global_id(0) * 4;                                               \
  element_y = get_global_id(1);                                                   \
  if (element_x >= cols || element_y >= rows)                                     \
    return;                                                                       \
                                                                                  \
  int origin_x = element_x - radius;                                              \
  int origin_y = element_y - radius;                                              \
  int top_x    = element_x + radius;                                              \
  int top_y    = element_y + radius;                                              \
                                                                                  \
  int data_index, kernel_index = 0;                                               \
  const global T* input;                                                          \
  T##4 value;                                                                     \
  float4 sum = (float4)(0.0f);                                                    \
                                                                                  \
  bool isnt_border_block = true;                                                  \
  data_index = radius / (get_local_size(0) * 4);                                  \
  if (get_group_id(0) <= data_index) isnt_border_block = false;                   \
  data_index = (cols - radius) / (get_local_size(0) * 4);                         \
  if (get_group_id(0) >= data_index) isnt_border_block = false;                   \
                                                                                  \
  if (isnt_border_block) {                                                        \
    for (int i = origin_y; i <= top_y; i++) {                                     \
      data_index = interpolation(rows, radius, i);                                \
      input = (const global T*)((uchar*)src + data_index * src_stride);             \
      input = input + origin_x;                                                   \
      for (int j = origin_x; j <= top_x; j++) {                                   \
        value = vload4(0, input);                                                   \
        sum = sum + convert_float(value) * src_kernel[kernel_index];              \
        kernel_index++;                                                           \
        input++;                                                          \
      }                                                                           \
    }                                                                             \
  }                                                                               \
  else {                                                                          \
    for (int i = origin_y; i <= top_y; i++) {                                     \
      data_index = interpolation(rows, radius, i);                                \
      input = (const global T*)((uchar*)src + data_index * src_stride);             \
      for (int j = origin_x; j <= top_x; j++) {                                   \
        data_index = interpolation(cols, radius, j);                              \
        value.x = input[data_index];                                                \
        data_index = interpolation(cols, radius, j + 1);                          \
        value.y = input[data_index];                                                \
        data_index = interpolation(cols, radius, j + 2);                          \
        value.z = input[data_index];                                                \
        data_index = interpolation(cols, radius, j + 3);                          \
        value.w = input[data_index];                                                \
        sum = sum + convert_float(value) * src_kernel[kernel_index];              \
        kernel_index++;                                                           \
      }                                                                           \
    }                                                                             \
  }                                                                               \
                                                                                  \
  if (delta != 0.f) {                                                             \
    sum += (float4)(delta);                                                       \
  }                                                                               \
  dst = (global T*)((uchar*)dst + element_y * dst_stride);                        \
  if (element_x < cols - 3) {                                                     \
    dst = dst + element_x;                                                        \
    vstore4(convert_##T##_sat(sum), 0, dst);                                      \
  }                                                                               \
  else {                                                                          \
    dst[element_x] = convert_##T##_sat(sum.x);                                    \
    if (element_x < cols - 1) {                                                   \
      dst[element_x + 1] = convert_##T##_sat(sum.y);                              \
    }                                                                             \
    if (element_x < cols - 2) {                                                   \
      dst[element_x + 2] = convert_##T##_sat(sum.z);                              \
    }                                                                             \
  }                                                                               \
}


#define FILTER2DCNKERNEL(base_type, T, interpolation, channels)                   \
__kernel void filter2D##base_type##C##channels##interpolation##Kernel(            \
  const global T* src, int rows, int cols, int src_stride,                        \
  const global float* src_kernel, int radius, global T* dst,                      \
  int dst_stride, float delta){                                                   \
  int element_x, element_y;                                                       \
  element_x = get_global_id(0);                                                   \
  element_y = get_global_id(1);                                                   \
  if (element_x >= cols || element_y >= rows) {                                   \
    return;                                                                       \
  }                                                                               \
                                                                                  \
  int origin_x = element_x - radius;                                              \
  int origin_y = element_y - radius;                                              \
  int top_x    = element_x + radius;                                              \
  int top_y    = element_y + radius;                                              \
                                                                                  \
  int data_index, kernel_index = 0;                                               \
  const global T* input;                                                          \
  T##channels value;                                                              \
  float##channels output = (float##channels)(0.0f);                               \
                                                                                  \
  bool isnt_border_block = true;                                                  \
  data_index = radius / (get_local_size(0));                                      \
  if (get_group_id(0) <= data_index) isnt_border_block = false;                   \
  data_index = (cols - radius) / (get_local_size(0));                             \
  if (get_group_id(0) >= data_index) isnt_border_block = false;                   \
                                                                                  \
  float##channels sum = (float##channels)(0);                                     \
  if (isnt_border_block) {                                                        \
    for (int i = origin_y; i <= top_y; i++) {                                     \
      data_index = interpolation(rows, radius, i);                                \
      input = (const global T*)((uchar*)src + data_index * src_stride);             \
      for (int j = origin_x; j <= top_x; j++) {                                   \
        value = vload##channels(j, input);                                        \
        sum = sum + convert_float(value) * src_kernel[kernel_index];              \
        kernel_index++;                                                           \
      }                                                                           \
    }                                                                             \
  }                                                                               \
  else {                                                                          \
    for (int i = origin_y; i <= top_y; i++) {                                     \
      data_index = interpolation(rows, radius, i);                                \
      input = (const global T*)((uchar*)src + data_index * src_stride);             \
      for (int j = origin_x; j <= top_x; j++) {                                   \
        data_index = interpolation(cols, radius, j);                              \
        value = vload##channels(data_index, input);                                        \
        sum = sum + convert_float(value) * src_kernel[kernel_index];              \
        kernel_index++;                                                           \
      }                                                                           \
    }                                                                             \
  }                                                                               \
  if (delta != 0.f) {                                                             \
    sum += (float##channels)delta;                                                                 \
  }                                                                               \
  dst = (global T*)((uchar*)dst + element_y * dst_stride);                        \
  vstore##channels(convert_##T##_sat(sum), element_x, dst);                                \
}



FILTER2DC1KERNEL(U8, uchar, interpolateReplicateBorder)
FILTER2DC1KERNEL(U8, uchar, interpolateReflectBorder)
FILTER2DC1KERNEL(U8, uchar, interpolateReflect101Border)
FILTER2DC1KERNEL(F32, float, interpolateReplicateBorder)
FILTER2DC1KERNEL(F32, float, interpolateReflectBorder)
FILTER2DC1KERNEL(F32, float, interpolateReflect101Border)


FILTER2DCNKERNEL(U8, uchar, interpolateReplicateBorder, 3)
FILTER2DCNKERNEL(U8, uchar, interpolateReflectBorder, 3)
FILTER2DCNKERNEL(U8, uchar, interpolateReflect101Border, 3)
FILTER2DCNKERNEL(F32, float, interpolateReplicateBorder, 3)
FILTER2DCNKERNEL(F32, float, interpolateReflectBorder, 3)
FILTER2DCNKERNEL(F32, float, interpolateReflect101Border, 3)


FILTER2DCNKERNEL(U8, uchar, interpolateReplicateBorder, 4)
FILTER2DCNKERNEL(U8, uchar, interpolateReflectBorder, 4)
FILTER2DCNKERNEL(U8, uchar, interpolateReflect101Border, 4)
FILTER2DCNKERNEL(F32, float, interpolateReplicateBorder, 4)
FILTER2DCNKERNEL(F32, float, interpolateReflectBorder, 4)
FILTER2DCNKERNEL(F32, float, interpolateReflect101Border, 4)