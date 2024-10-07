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

#define INDEX_CONVERT1 x
#define INDEX_CONVERT2 y
#define INDEX_CONVERT3 z
#define INDEX_CONVERT4 w
#define convert_float1 convert_float
#define convert_uchar1 convert_uchar_sat
#define convert_uchar2 convert_uchar2_sat
#define convert_uchar3 convert_uchar3_sat
#define convert_uchar4 convert_uchar4_sat

#define GAUSSIANBLUR_VEC1(src_array, i) \
  src_array[0].INDEX_CONVERT##i
#define GAUSSIANBLUR_VEC2(src_array, i) \
  GAUSSIANBLUR_VEC1(src_array, i), src_array[1].INDEX_CONVERT##i
#define GAUSSIANBLUR_VEC3(src_array, i) \
  GAUSSIANBLUR_VEC2(src_array, i), src_array[2].INDEX_CONVERT##i
#define GAUSSIANBLUR_VEC4(src_array, i) \
  GAUSSIANBLUR_VEC3(src_array, i), src_array[3].INDEX_CONVERT##i

#define vstore1 vstore
#define uchar1 uchar
#define float1 float

#define GAUSSIANBLUR_ELEMENT1(T, src_array, output, j)              \
  output[0] = convert_##T##j((float##j)(GAUSSIANBLUR_VEC##j(src_array, 1)));
#define GAUSSIANBLUR_ELEMENT2(T, src_array, output, j)              \
  GAUSSIANBLUR_ELEMENT1(T, src_array, output, j)                    \
  output[1] = convert_##T##j((float##j)(GAUSSIANBLUR_VEC##j(src_array, 2)));
#define GAUSSIANBLUR_ELEMENT3(T, src_array, output, j)              \
  GAUSSIANBLUR_ELEMENT2(T, src_array, output, j)                    \
  output[2] = convert_##T##j((float##j)(GAUSSIANBLUR_VEC##j(src_array, 3)));
#define GAUSSIANBLUR_ELEMENT4(T, src_array, output, j)              \
  GAUSSIANBLUR_ELEMENT3(T, src_array, output, j)                    \
  output[3] = convert_##T##j((float##j)(GAUSSIANBLUR_VEC##j(src_array, 4)));

#define VSTORE_REMAIN_ROWS1(input, rows_load)                    \
  int offset = element_y * rows_load;                            \
  dst[offset] = input;

#define VSTORE_REMAIN_ROWS1_VEC(input, rows_load)                \
  int offset = element_y * rows_load;                            \
  dst[offset] = input.INDEX_CONVERT##1;

#define VSTORE_REMAIN_ROWS2(input, rows_load)                    \
  VSTORE_REMAIN_ROWS1_VEC(input, rows_load)                      \
  dst[offset + 1] = input.INDEX_CONVERT##2;

#define VSTORE_REMAIN_ROWS3(input, rows_load)                    \
  VSTORE_REMAIN_ROWS2(input, rows_load)                          \
  dst[offset + 2] = input.INDEX_CONVERT##3;

#define GAUSSIANBLUR_AND_SAVE_VEC(T, src_array, output, rows_load, i, j) \
  {                                                                     \
    T##j output_value[i];                                               \
    GAUSSIANBLUR_ELEMENT##i(T, src_array, output, j)                     \
    for (int k = 0; k < i; k++) {                                       \
      vstore##j(output[k], element_y, dst);                             \
      dst = (global T*)((global uchar*)dst + dst_stride);                      \
    }                                                                   \
  }

#define GAUSSIANBLUR_AND_SAVE_SCALE(T, src_array, output, rows_load, i, j) \
  {                                                                       \
    T##j output_value[i];                                                 \
    GAUSSIANBLUR_ELEMENT##i(T, src_array, output, j)                       \
    for (int k = 0; k < i; k++) {                                         \
      VSTORE_REMAIN_ROWS##j(output[k], rows_load)                         \
      dst = (global T*)((global uchar*)dst + dst_stride);                        \
    }                                                                     \
  }

#define GAUSSIANBLUR_C1_RAMAIN_ROW1(T, cols_load, rows_load, rows_load_global, \
                                   save_name)                                 \
  if (remain_cols >= cols_load)                                               \
  GAUSSIANBLUR_AND_SAVE_##save_name(T, input_value, output_value,              \
                                   rows_load_global, cols_load, rows_load)

#define GAUSSIANBLUR_C1_RAMAIN_ROW2(T, cols_load, rows_load, rows_load_global, \
                                   save_name)                                 \
  GAUSSIANBLUR_C1_RAMAIN_ROW1(T, cols_load, rows_load, rows_load_global,       \
                             save_name)                                       \
  else if (remain_cols == 1) GAUSSIANBLUR_AND_SAVE_##save_name(                \
      T, input_value, output_value, rows_load_global, 1, rows_load)

#define GAUSSIANBLUR_C1_RAMAIN_ROW3(T, cols_load, rows_load, rows_load_global, \
                                   save_name)                                 \
  GAUSSIANBLUR_C1_RAMAIN_ROW2(T, cols_load, rows_load, rows_load_global,       \
                             save_name)                                       \
  else if (remain_cols == 2) GAUSSIANBLUR_AND_SAVE_##save_name(                \
      T, input_value, output_value, rows_load_global, 2, rows_load)

#define GAUSSIANBLUR_C1_RAMAIN_ROW4(T, cols_load, rows_load, rows_load_global, \
                                   save_name)                                 \
  GAUSSIANBLUR_C1_RAMAIN_ROW3(T, cols_load, rows_load, rows_load_global,       \
                             save_name)                                       \
  else if (remain_cols == 3) GAUSSIANBLUR_AND_SAVE_##save_name(                \
      T, input_value, output_value, rows_load_global, 3, rows_load)

#define GAUSSIANBLUR_C1_RAMAIN_COL1(T, cols_load, rows_load)                  \
  if (remain_rows >= rows_load) {                                            \
    GAUSSIANBLUR_C1_RAMAIN_ROW##cols_load(T, cols_load, rows_load, rows_load, \
                                         VEC)                                \
  }

#define GAUSSIANBLUR_C1_RAMAIN_COL2(T, cols_load, rows_load)                 \
  GAUSSIANBLUR_C1_RAMAIN_COL1(T, cols_load, rows_load)                       \
  else if (remain_rows == 1) {                                              \
    GAUSSIANBLUR_C1_RAMAIN_ROW##cols_load(T, cols_load, 1, rows_load, SCALE) \
  }

#define GAUSSIANBLUR_C1_RAMAIN_COL3(T, cols_load, rows_load)                 \
  GAUSSIANBLUR_C1_RAMAIN_COL2(T, cols_load, rows_load)                       \
  else if (remain_rows == 2) {                                              \
    GAUSSIANBLUR_C1_RAMAIN_ROW##cols_load(T, cols_load, 2, rows_load, SCALE) \
  }

#define GAUSSIANBLUR_C1_RAMAIN_COL4(T, cols_load, rows_load)                 \
  GAUSSIANBLUR_C1_RAMAIN_COL3(T, cols_load, rows_load)                       \
  else if (remain_rows == 3) {                                              \
    GAUSSIANBLUR_C1_RAMAIN_ROW##cols_load(T, cols_load, 3, rows_load, SCALE) \
  }

#define READ_BOARDER1(interpolation, value)           \
  data_index = interpolation(cols, radius, j); \
  value.x = convert_float(src_temp[data_index]);

#define READ_BOARDER2(interpolation, value)               \
  READ_BOARDER1(interpolation, value)                     \
  data_index = interpolation(cols, radius, j + 1); \
  value.y = convert_float(src_temp[data_index]);

#define READ_BOARDER3(interpolation, value)               \
  READ_BOARDER2(interpolation, value)                     \
  data_index = interpolation(cols, radius, j + 2); \
  value.z = convert_float(src_temp[data_index]);

#define READ_BOARDER4(interpolation, value)               \
  READ_BOARDER3(interpolation, value)                     \
  data_index = interpolation(cols, radius, j + 3); \
  value.w = convert_float(src_temp[data_index]);

__kernel
void getGaussianKernel(float sigma, int ksize, global float* coefficients, int offset) {
  float value = sigma > 0 ? sigma : ((ksize - 1) * 0.5f - 1) * 0.3f + 0.8f;
  float scale_2x = -0.5f / (value * value);
  float sum = 0.f;
  coefficients = (global float*)((global uchar*)coefficients + offset);

  int i;
  float x;
  for (i = 0; i < ksize; i++) {
    x = i - (ksize - 1) * 0.5f;
    value = exp(scale_2x * x * x);
    coefficients[i] = value;
    sum +=value;
  }

  sum = 1.f / sum;
  for (i = 0; i < ksize; i++) {
    coefficients[i] *= sum;
  }
}

// #if defined(GAUSSIANBLUR_U81C) || defined(GAUSSIANBLUR_F321C) || \
//     defined(ALL_KERNELS)
#define GAUSSIANBLUR_KERNEL_C1_TYPE(base_type_src, Tsrc, base_type_dst, Tdst,                             \
                                   cols_load, rows_load, interpolation)                                   \
  __kernel void                                                                                           \
      gaussianblur##base_type_src##base_type_dst##interpolation##C1Kernel(                                \
          global const Tsrc* src, int src_offset, int rows, int cols,                                     \
          global const float* filter_kernel, int kernel_offset, int radius, int src_stride,               \
          global Tdst* dst, int dst_stride, int dst_offset) {                                             \
    int element_x = get_global_id(0);                                                                     \
    int element_y = get_global_id(1);                                                                     \
    int group_x = get_group_id(0);                                                                        \
    int group_y = get_group_id(1);                                                                        \
    int index_x = element_x * cols_load, index_y = element_y * rows_load;                                 \
    if (index_x >= cols || index_y >= rows) {                                                             \
      return;                                                                                             \
    }                                                                                                     \
    src = (global const Tsrc*)((uchar*)src + src_offset);                                                 \
    dst = (global Tdst*)((uchar*)dst + dst_offset);                                                       \
    filter_kernel = (global const float*)((global uchar*)filter_kernel + kernel_offset);                  \
                                                                                                          \
    src = (global const Tsrc*)((global uchar*)src + index_y * src_stride);                                \
    int remain_cols = cols - index_x, remain_rows = rows - index_y;                                       \
    int bottom = index_x - radius;                                                                        \
    int top = index_x + radius;                                                                           \
    int filter_kernel_index;                                                                              \
    int data_index;                                                                                       \
                                                                                                          \
    float##cols_load input_value[rows_load];                                                              \
    bool isnt_border_block = true;                                                                        \
    data_index = radius / (get_local_size(0)) / cols_load;                                                \
    if (group_x <= data_index)                                                                            \
      isnt_border_block = false;                                                                          \
    data_index = (cols - radius) / (get_local_size(0)) / cols_load;                                       \
    if (group_x >= data_index)                                                                            \
      isnt_border_block = false;                                                                          \
    int data_index0;                                                                                      \
                                                                                                          \
    global const Tsrc* src_temp;                                                                          \
    for (int i = 0; i < min(remain_rows, rows_load); i++) {                                               \
      ((float##cols_load*)input_value+i)[0] = (float##cols_load)(0);                                      \
      src_temp = src;                                                                                     \
      filter_kernel_index = 0;                                                                            \
      if (isnt_border_block) {                                                                            \
        src_temp += index_x;                                                                              \
        for (int j = radius; j > 0; j--) {                                                                \
          ((float##cols_load*)input_value+i)[0] +=                                                        \
              (convert_float##cols_load(vload##cols_load(0, src_temp - j)) +                              \
               convert_float##cols_load(vload##cols_load(0, src_temp + j))) *                             \
              filter_kernel[filter_kernel_index];                                                         \
          filter_kernel_index++;                                                                          \
        }                                                                                                 \
        ((float##cols_load*)input_value+i)[0] += convert_float##cols_load(vload##cols_load(0, src_temp)) *\
                            filter_kernel[filter_kernel_index];                                           \
      }                                                                                                   \
      else {                                                                                              \
        float##cols_load value;                                                                           \
        int j;                                                                                            \
        float##cols_load value1;                                                                          \
        for (int j_radius = radius; j_radius > 0; j_radius--) {                                           \
          j = index_x - j_radius;                                                                         \
          READ_BOARDER##cols_load(interpolation, value);                                                  \
          j = index_x + j_radius;                                                                         \
          READ_BOARDER##cols_load(interpolation, value1);                                                 \
          ((float##cols_load*)input_value+i)[0] += (value + value1) * filter_kernel[filter_kernel_index]; \
          filter_kernel_index++;                                                                          \
        }                                                                                                 \
        j = index_x;                                                                                      \
        READ_BOARDER##cols_load(interpolation, value);                                                    \
        ((float##cols_load*)input_value+i)[0] += (value) * filter_kernel[filter_kernel_index];            \
      }                                                                                                   \
      src = (global const Tsrc*)((global uchar*)src + src_stride);                                        \
    }                                                                                                     \
                                                                                                          \
    dst = (global Tdst*)((global uchar*)dst + dst_stride * index_x);                                      \
    GAUSSIANBLUR_C1_RAMAIN_COL##rows_load(Tdst, cols_load, rows_load)                                     \
  }
// #endif

#define GAUSSIANBLUR_SAVE_CHANNEL1(T, rows_load, channels)                                                   \
  if (remain_rows >= rows_load) {                                                                            \
    for (int i = 0; i < rows_load; i++) {                                                                    \
      vstore##channels(convert_##T##channels(input_value[i]), index_y + i,                                   \
                       dst);                                                                                 \
    }                                                                                                        \
  }

#define GAUSSIANBLUR_SAVE_CHANNEL2(T, rows_load, channels)                                                   \
  GAUSSIANBLUR_SAVE_CHANNEL1(T, rows_load, channels)                                                         \
  else if (remain_rows == 1) {                                                                               \
    vstore##channels(convert_##T##channels(input_value[0]), index_y, dst);                                   \
  }

#define GAUSSIANBLUR_SAVE_CHANNEL3(T, rows_load, channels)                                                   \
  GAUSSIANBLUR_SAVE_CHANNEL2(T, rows_load, channels)                                                         \
  else if (remain_rows == 2) {                                                                               \
    vstore##channels(convert_##T##channels(input_value[0]), index_y, dst);                                   \
    vstore##channels(convert_##T##channels(input_value[1]), index_y + 1, dst);                               \
  }

#define GAUSSIANBLUR_SAVE_CHANNEL4(T, rows_load, channels)                                                   \
  GAUSSIANBLUR_SAVE_CHANNEL3(T, rows_load, channels)                                                         \
  else if (remain_rows == 3) {                                                                               \
    vstore##channels(convert_##T##channels(input_value[0]), index_y, dst);                                   \
    vstore##channels(convert_##T##channels(input_value[1]), index_y + 1, dst);                               \
    vstore##channels(convert_##T##channels(input_value[2]), index_y + 2, dst);                               \
  }

// #if defined(GAUSSIANBLUR_U8C3) || defined(GAUSSIANBLUR_F32C3) ||                                          \
//     defined(GAUSSIANBLUR_U8C4) || defined(GAUSSIANBLUR_F32C4) ||                                          \
//     defined(ALL_KERNELS)
#define GAUSSIANBLUR_KERNEL_CN_TYPE(base_type_src, Tsrc, base_type_dst, Tdst,                                \
                                   channels, rows_load, interpolation)                                       \
  __kernel void                                                                                              \
      gaussianblur##base_type_src##base_type_dst##interpolation##C##channels##Kernel(                        \
          global const Tsrc* src, int src_offset, int rows, int cols,                                        \
          global const float* filter_kernel, int kernel_offset, int radius, int src_stride,                  \
          global Tdst* dst, int dst_stride, int dst_offset) {                                                \
    int element_x = get_global_id(0);                                                                        \
    int element_y = get_global_id(1);                                                                        \
    int group_x = get_group_id(0);                                                                           \
    int group_y = get_group_id(1);                                                                           \
    int index_x = element_x, index_y = element_y * rows_load;                                                \
    if (index_x >= cols || index_y >= rows) {                                                                \
      return;                                                                                                \
    }                                                                                                        \
                                                                                                             \
    src = (global const Tsrc*)((uchar*)src + src_offset);                                                    \
    dst = (global Tdst*)((uchar*)dst + dst_offset);                                                          \
    src = (global const Tsrc*)((global uchar*)src + index_y * src_stride);                                   \
    filter_kernel = (global const float*)((global uchar*)filter_kernel + kernel_offset);                     \
    int remain_rows = rows - index_y;                                                                        \
    int bottom = index_x - radius;                                                                           \
    int top = index_x + radius;                                                                              \
    int filter_kernel_index;                                                                                 \
    int data_index;                                                                                          \
                                                                                                             \
                                                                                                             \
    float##channels input_value[rows_load];                                                                  \
                                                                                                             \
    bool isnt_border_block = true;                                                                           \
    data_index = radius / (get_local_size(0));                                                               \
    if (group_x <= data_index)                                                                               \
      isnt_border_block = false;                                                                             \
    data_index = (cols - radius) / (get_local_size(0));                                                      \
    if (group_x >= data_index)                                                                               \
      isnt_border_block = false;                                                                             \
    int data_index0;                                                                                         \
                                                                                                             \
    for (int i = 0; i < min(remain_rows, rows_load); i++) {                                                  \
      filter_kernel_index = 0;                                                                               \
      ((float##channels*)input_value+i)[0] = (float##channels)(0);                                           \
      if (isnt_border_block) {                                                                               \
        for (int j = radius; j > 0; j--) {                                                                   \
          ((float##channels*)input_value+i)[0] += (convert_float##channels(vload##channels(index_x - j, src))\
                                + convert_float##channels(vload##channels(index_x + j, src))) *              \
                            filter_kernel[filter_kernel_index];                                              \
          filter_kernel_index++;                                                                             \
        }                                                                                                    \
        ((float##channels*)input_value+i)[0] += convert_float##channels(vload##channels(index_x, src)) *     \
                            filter_kernel[filter_kernel_index];                                              \
      }                                                                                                      \
      else {                                                                                                 \
        for (int j = radius; j > 0; j--) {                                                                   \
          data_index = interpolation(cols, radius, index_x - j);                                             \
          data_index0 = interpolation(cols, radius, index_x + j);                                            \
          ((float##channels*)input_value+i)[0] +=                                                            \
              (convert_float##channels(vload##channels(data_index, src)) +                                   \
              convert_float##channels(vload##channels(data_index0, src))) *                                  \
              filter_kernel[filter_kernel_index];                                                            \
          filter_kernel_index++;                                                                             \
        }                                                                                                    \
        data_index = interpolation(cols, radius, index_x);                                                   \
        ((float##channels*)input_value+i)[0] += convert_float##channels(vload##channels(data_index, src)) *  \
                            filter_kernel[filter_kernel_index];                                              \
      }                                                                                                      \
      src = (global const Tsrc*)((global uchar*)src + src_stride);                                           \
    }                                                                                                        \
                                                                                                             \
                                                                                                             \
    dst = (global Tdst*)((global uchar*)dst + dst_stride * index_x);                                         \
    GAUSSIANBLUR_SAVE_CHANNEL##rows_load(Tdst, rows_load, channels)                                          \
  }
// #endif


GAUSSIANBLUR_KERNEL_C1_TYPE(F32, float, U8, uchar, 4, 4, interpolateReplicateBorder)
GAUSSIANBLUR_KERNEL_C1_TYPE(U8, uchar, F32, float, 2, 2, interpolateReplicateBorder)
GAUSSIANBLUR_KERNEL_C1_TYPE(F32, float, F32, float, 2, 2, interpolateReplicateBorder)

GAUSSIANBLUR_KERNEL_C1_TYPE(F32, float, U8, uchar, 4, 4, interpolateReflectBorder)
GAUSSIANBLUR_KERNEL_C1_TYPE(U8, uchar, F32, float, 2, 2, interpolateReflectBorder)
GAUSSIANBLUR_KERNEL_C1_TYPE(F32, float, F32, float, 2, 2, interpolateReflectBorder)

GAUSSIANBLUR_KERNEL_C1_TYPE(F32, float, U8, uchar, 4, 4, interpolateReflect101Border)
GAUSSIANBLUR_KERNEL_C1_TYPE(U8, uchar, F32, float, 2, 2, interpolateReflect101Border)
GAUSSIANBLUR_KERNEL_C1_TYPE(F32, float, F32, float, 2, 2, interpolateReflect101Border)

GAUSSIANBLUR_KERNEL_CN_TYPE(U8, uchar, F32, float, 3, 1, interpolateReplicateBorder)
GAUSSIANBLUR_KERNEL_CN_TYPE(F32, float, U8, uchar, 3, 4, interpolateReplicateBorder)
GAUSSIANBLUR_KERNEL_CN_TYPE(F32, float, F32, float, 3, 1, interpolateReplicateBorder)


GAUSSIANBLUR_KERNEL_CN_TYPE(U8, uchar, F32, float, 3, 1, interpolateReflectBorder)
GAUSSIANBLUR_KERNEL_CN_TYPE(F32, float, U8, uchar, 3, 4, interpolateReflectBorder)
GAUSSIANBLUR_KERNEL_CN_TYPE(F32, float, F32, float, 3, 1, interpolateReflectBorder)

GAUSSIANBLUR_KERNEL_CN_TYPE(U8, uchar, F32, float, 3, 1, interpolateReflect101Border)
GAUSSIANBLUR_KERNEL_CN_TYPE(F32, float, U8, uchar, 3, 4, interpolateReflect101Border)
GAUSSIANBLUR_KERNEL_CN_TYPE(F32, float, F32, float, 3, 1, interpolateReflect101Border)

GAUSSIANBLUR_KERNEL_CN_TYPE(U8, uchar, F32, float, 4, 1, interpolateReplicateBorder)
GAUSSIANBLUR_KERNEL_CN_TYPE(F32, float, U8, uchar, 4, 4, interpolateReplicateBorder)
GAUSSIANBLUR_KERNEL_CN_TYPE(F32, float, F32, float, 4, 1, interpolateReplicateBorder)

GAUSSIANBLUR_KERNEL_CN_TYPE(U8, uchar, F32, float, 4, 1, interpolateReflectBorder)
GAUSSIANBLUR_KERNEL_CN_TYPE(F32, float, U8, uchar, 4, 4, interpolateReflectBorder)
GAUSSIANBLUR_KERNEL_CN_TYPE(F32, float, F32, float, 4, 1, interpolateReflectBorder)

GAUSSIANBLUR_KERNEL_CN_TYPE(U8, uchar, F32, float, 4, 1, interpolateReflect101Border)
GAUSSIANBLUR_KERNEL_CN_TYPE(F32, float, U8, uchar, 4, 4, interpolateReflect101Border)
GAUSSIANBLUR_KERNEL_CN_TYPE(F32, float, F32, float, 4, 1, interpolateReflect101Border)
