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

#define TRANSPOSE_VEC1(src_array, i) \
  src_array[0].INDEX_CONVERT##i
#define TRANSPOSE_VEC2(src_array, i) \
  TRANSPOSE_VEC1(src_array, i), src_array[1].INDEX_CONVERT##i
#define TRANSPOSE_VEC3(src_array, i) \
  TRANSPOSE_VEC2(src_array, i), src_array[2].INDEX_CONVERT##i
#define TRANSPOSE_VEC4(src_array, i) \
  TRANSPOSE_VEC3(src_array, i), src_array[3].INDEX_CONVERT##i

#define vstore1 vstore
#define uchar1 uchar
#define float1 float

#define TRANSPOSE_ELEMENT1(T, src_array, output, j)              \
  output[0] = convert_##T##j((float##j)(TRANSPOSE_VEC##j(src_array, 1)));
#define TRANSPOSE_ELEMENT2(T, src_array, output, j)              \
  TRANSPOSE_ELEMENT1(T, src_array, output, j)                    \
  output[1] = convert_##T##j((float##j)(TRANSPOSE_VEC##j(src_array, 2)));
#define TRANSPOSE_ELEMENT3(T, src_array, output, j)              \
  TRANSPOSE_ELEMENT2(T, src_array, output, j)                    \
  output[2] = convert_##T##j((float##j)(TRANSPOSE_VEC##j(src_array, 3)));
#define TRANSPOSE_ELEMENT4(T, src_array, output, j)              \
  TRANSPOSE_ELEMENT3(T, src_array, output, j)                    \
  output[3] = convert_##T##j((float##j)(TRANSPOSE_VEC##j(src_array, 4)));

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

#define TRANSPOSE_AND_SAVE_VEC(T, src_array, output, rows_load, i, j)   \
  {                                                                     \
    T##j output_value[i];                                               \
    TRANSPOSE_ELEMENT##i(T, src_array, output, j)                       \
    for (int k = 0; k < i; k++) {                                       \
      vstore##j(output[k], element_y, dst);                             \
      dst = (global T*)((global uchar*)dst + dst_stride);                      \
    }                                                                   \
  }

#define TRANSPOSE_AND_SAVE_SCALE(T, src_array, output, rows_load, i, j)   \
  {                                                                       \
    T##j output_value[i];                                                 \
    TRANSPOSE_ELEMENT##i(T, src_array, output, j)                         \
    for (int k = 0; k < i; k++) {                                         \
      VSTORE_REMAIN_ROWS##j(output[k], rows_load)                         \
      dst = (global T*)((global uchar*)dst + dst_stride);                        \
    }                                                                     \
  }

#define TRANSPOSE_C1_RAMAIN_ROW1(T, cols_load, rows_load, rows_load_global,   \
                                   save_name)                                 \
  if (remain_cols >= cols_load)                                               \
  TRANSPOSE_AND_SAVE_##save_name(T, input_value, output_value,                \
                                   rows_load_global, cols_load, rows_load)

#define TRANSPOSE_C1_RAMAIN_ROW2(T, cols_load, rows_load, rows_load_global,   \
                                   save_name)                                 \
  TRANSPOSE_C1_RAMAIN_ROW1(T, cols_load, rows_load, rows_load_global,         \
                             save_name)                                       \
  else if (remain_cols == 1) TRANSPOSE_AND_SAVE_##save_name(                  \
      T, input_value, output_value, rows_load_global, 1, rows_load)

#define TRANSPOSE_C1_RAMAIN_ROW3(T, cols_load, rows_load, rows_load_global, \
                                   save_name)                                 \
  TRANSPOSE_C1_RAMAIN_ROW2(T, cols_load, rows_load, rows_load_global,       \
                             save_name)                                       \
  else if (remain_cols == 2) TRANSPOSE_AND_SAVE_##save_name(                \
      T, input_value, output_value, rows_load_global, 2, rows_load)

#define TRANSPOSE_C1_RAMAIN_ROW4(T, cols_load, rows_load, rows_load_global,   \
                                   save_name)                                 \
  TRANSPOSE_C1_RAMAIN_ROW3(T, cols_load, rows_load, rows_load_global,         \
                             save_name)                                       \
  else if (remain_cols == 3) TRANSPOSE_AND_SAVE_##save_name(                  \
      T, input_value, output_value, rows_load_global, 3, rows_load)

#define TRANSPOSE_C1_RAMAIN_COL1(T, cols_load, rows_load)                    \
  if (remain_rows >= rows_load) {                                            \
    TRANSPOSE_C1_RAMAIN_ROW##cols_load(T, cols_load, rows_load, rows_load,   \
                                         VEC)                                \
  }

#define TRANSPOSE_C1_RAMAIN_COL2(T, cols_load, rows_load)                   \
  TRANSPOSE_C1_RAMAIN_COL1(T, cols_load, rows_load)                         \
  else if (remain_rows == 1) {                                              \
    TRANSPOSE_C1_RAMAIN_ROW##cols_load(T, cols_load, 1, rows_load, SCALE)   \
  }

#define TRANSPOSE_C1_RAMAIN_COL3(T, cols_load, rows_load)                   \
  TRANSPOSE_C1_RAMAIN_COL2(T, cols_load, rows_load)                         \
  else if (remain_rows == 2) {                                              \
    TRANSPOSE_C1_RAMAIN_ROW##cols_load(T, cols_load, 2, rows_load, SCALE)   \
  }

#define TRANSPOSE_C1_RAMAIN_COL4(T, cols_load, rows_load)                   \
  TRANSPOSE_C1_RAMAIN_COL3(T, cols_load, rows_load)                         \
  else if (remain_rows == 3) {                                              \
    TRANSPOSE_C1_RAMAIN_ROW##cols_load(T, cols_load, 3, rows_load, SCALE)   \
  }

// ============================================================================================

#define THRESHOLD_BINARY_VSTORE_REMAIN_ROWS1(input, rows_load)                              \
  int offset = element_y * rows_load;                                                \
  dst[offset] = convert_int(src_input[offset]) > convert_int(convert_uchar1(input) - delta) ? setted_value : 0;

#define THRESHOLD_BINARY_VSTORE_REMAIN_ROWS1_VEC(input, rows_load)                          \
  int offset = element_y * rows_load;                                                \
  dst[offset] = convert_int(src_input[offset]) > convert_int(convert_uchar1(input.INDEX_CONVERT##1) - delta) ? setted_value : 0;

#define THRESHOLD_BINARY_VSTORE_REMAIN_ROWS2(input, rows_load)                              \
  THRESHOLD_BINARY_VSTORE_REMAIN_ROWS1_VEC(input, rows_load)                                \
  dst[offset + 1] = convert_int(src_input[offset + 1]) > convert_int(convert_uchar1(input.INDEX_CONVERT##2) - delta) ? setted_value : 0;

#define THRESHOLD_BINARY_VSTORE_REMAIN_ROWS3(input, rows_load)                              \
  THRESHOLD_BINARY_VSTORE_REMAIN_ROWS2(input, rows_load)                                    \
  dst[offset + 2] = convert_int(src_input[offset + 2]) > convert_int(convert_uchar1(input.INDEX_CONVERT##3) - delta) ? setted_value : 0;

#define THRESHOLD_BINARY_INV_VSTORE_REMAIN_ROWS1(input, rows_load)                              \
  int offset = element_y * rows_load;                                                \
  dst[offset] = convert_int(src_input[offset]) > convert_int(convert_uchar1(input) - delta) ? 0 : setted_value;

#define THRESHOLD_BINARY_INV_VSTORE_REMAIN_ROWS1_VEC(input, rows_load)                          \
  int offset = element_y * rows_load;                                                \
  dst[offset] = convert_int(src_input[offset]) > convert_int(convert_uchar1(input.INDEX_CONVERT##1) - delta) ? 0 : setted_value;

#define THRESHOLD_BINARY_INV_VSTORE_REMAIN_ROWS2(input, rows_load)                              \
  THRESHOLD_BINARY_INV_VSTORE_REMAIN_ROWS1_VEC(input, rows_load)                                \
  dst[offset + 1] = convert_int(src_input[offset + 1]) > convert_int(convert_uchar1(input.INDEX_CONVERT##2) - delta) ? 0 : setted_value;

#define THRESHOLD_BINARY_INV_VSTORE_REMAIN_ROWS3(input, rows_load)                              \
  THRESHOLD_BINARY_INV_VSTORE_REMAIN_ROWS2(input, rows_load)                                    \
  dst[offset + 2] = convert_int(src_input[offset + 2]) > convert_int(convert_uchar1(input.INDEX_CONVERT##3) - delta) ? 0 : setted_value;

#define THRESHOLD_TRANSPOSE_AND_SAVE_VEC(T, src_array, output, rows_load, i, j)\
  {                                                                          \
    T##j output_value[i];                                                    \
    TRANSPOSE_ELEMENT##i(T, src_array, output, j);                           \
    if (threshold_type == THRESH_BINARY) {                                   \
      for (int k = 0; k < i; k++) {                                          \
        output[k] = convert_uchar##j(convert_int##j(vload##j(element_y, src_input)) > (convert_int##j(output[k]) - delta)     \
                        ? (int##j)(setted_value)                                       \
                        : (int##j)(0));                                                 \
        vstore##j(output[k], element_y, dst);                                \
        src_input = (global T*)((global uchar*)src_input + src_input_stride);       \
        dst = (global T*)((global uchar*)dst + dst_stride);                         \
      }                                                                      \
    }                                                                        \
    else {                                                                   \
      for (int k = 0; k < i; k++) {                                          \
        output[k] = convert_uchar##j(convert_int##j(vload##j(element_y, src_input)) > (convert_int##j(output[k]) - delta)     \
                        ? (int##j)(0)                                                  \
                        : (int##j)(setted_value));                                      \
        vstore##j(output[k], element_y, dst);                                \
        src_input = (global T*)((global uchar*)src_input + src_input_stride);       \
        dst = (global T*)((global uchar*)dst + dst_stride);                         \
      }                                                                      \
    }                                                                        \
  }

#define THRESHOLD_TRANSPOSE_AND_SAVE_SCALE(T, src_array, output, rows_load, i, j)\
  {                                                                            \
    T##j output_value[i];                                                      \
    TRANSPOSE_ELEMENT##i(T, src_array, output, j)                              \
    if (threshold_type == THRESH_BINARY) {                                     \
      for (int k = 0; k < i; k++) {                                            \
        THRESHOLD_BINARY_VSTORE_REMAIN_ROWS##j(output[k], rows_load)           \
        src_input = (global T*)((global uchar*)src_input + src_input_stride);     \
        dst = (global T*)((global uchar*)dst + dst_stride);                           \
      }                                                                        \
    }                                                                          \
    else {                                                                     \
      for (int k = 0; k < i; k++) {                                            \
        THRESHOLD_BINARY_INV_VSTORE_REMAIN_ROWS##j(output[k], rows_load)       \
        src_input = (global T*)((global uchar*)src_input + src_input_stride);     \
        dst = (global T*)((global uchar*)dst + dst_stride);                           \
      }                                                                        \
    }                                                                          \
  }

#define THRESHOLD_TRANSPOSE_C1_RAMAIN_ROW1(T, cols_load, rows_load, rows_load_global,   \
                                   save_name)                                           \
  if (remain_cols >= cols_load)                                                         \
  THRESHOLD_TRANSPOSE_AND_SAVE_##save_name(T, input_value, output_value,                \
                                   rows_load_global, cols_load, rows_load)

#define THRESHOLD_TRANSPOSE_C1_RAMAIN_ROW2(T, cols_load, rows_load, rows_load_global,   \
                                   save_name)                                           \
  THRESHOLD_TRANSPOSE_C1_RAMAIN_ROW1(T, cols_load, rows_load, rows_load_global,         \
                             save_name)                                                 \
  else if (remain_cols == 1) THRESHOLD_TRANSPOSE_AND_SAVE_##save_name(                  \
      T, input_value, output_value, rows_load_global, 1, rows_load)

#define THRESHOLD_TRANSPOSE_C1_RAMAIN_ROW3(T, cols_load, rows_load, rows_load_global, \
                                   save_name)                                         \
  THRESHOLD_TRANSPOSE_C1_RAMAIN_ROW2(T, cols_load, rows_load, rows_load_global,       \
                             save_name)                                               \
  else if (remain_cols == 2) THRESHOLD_TRANSPOSE_AND_SAVE_##save_name(                \
      T, input_value, output_value, rows_load_global, 2, rows_load)

#define THRESHOLD_TRANSPOSE_C1_RAMAIN_ROW4(T, cols_load, rows_load, rows_load_global,   \
                                   save_name)                                           \
  THRESHOLD_TRANSPOSE_C1_RAMAIN_ROW3(T, cols_load, rows_load, rows_load_global,         \
                             save_name)                                                 \
  else if (remain_cols == 3) THRESHOLD_TRANSPOSE_AND_SAVE_##save_name(                  \
      T, input_value, output_value, rows_load_global, 3, rows_load)

#define THRESHOLD_TRANSPOSE_C1_RAMAIN_COL1(T, cols_load, rows_load)                    \
  if (remain_rows >= rows_load) {                                                      \
    THRESHOLD_TRANSPOSE_C1_RAMAIN_ROW##cols_load(T, cols_load, rows_load, rows_load,   \
                                         VEC)                                          \
  }

#define THRESHOLD_TRANSPOSE_C1_RAMAIN_COL2(T, cols_load, rows_load)                   \
  THRESHOLD_TRANSPOSE_C1_RAMAIN_COL1(T, cols_load, rows_load)                         \
  else if (remain_rows == 1) {                                                        \
    THRESHOLD_TRANSPOSE_C1_RAMAIN_ROW##cols_load(T, cols_load, 1, rows_load, SCALE)   \
  }

#define THRESHOLD_TRANSPOSE_C1_RAMAIN_COL3(T, cols_load, rows_load)                   \
  THRESHOLD_TRANSPOSE_C1_RAMAIN_COL2(T, cols_load, rows_load)                         \
  else if (remain_rows == 2) {                                                        \
    THRESHOLD_TRANSPOSE_C1_RAMAIN_ROW##cols_load(T, cols_load, 2, rows_load, SCALE)   \
  }

#define THRESHOLD_TRANSPOSE_C1_RAMAIN_COL4(T, cols_load, rows_load)                   \
  THRESHOLD_TRANSPOSE_C1_RAMAIN_COL3(T, cols_load, rows_load)                         \
  else if (remain_rows == 3) {                                                        \
    THRESHOLD_TRANSPOSE_C1_RAMAIN_ROW##cols_load(T, cols_load, 3, rows_load, SCALE)   \
  }

#define THRESHOLD_TRANSPOSE_NORMALIZA(rows_load, cols_load)          \
  for (int i = 0; i < min(remain_rows, rows_load); i++) { \
    ((float##cols_load*)input_value+i)[0] *= weight;                             \
  }

#define TRANSPOSE_NORMALIZA(rows_load, cols_load) 

// ============================================================================================

#define READ_BOARDER1(interpolation)           \
  data_index = interpolation(cols, radius, j); \
  value.x = convert_float(src_temp[data_index]);

#define READ_BOARDER2(interpolation)               \
  READ_BOARDER1(interpolation)                     \
  data_index = interpolation(cols, radius, j + 1); \
  value.y = convert_float(src_temp[data_index]);

#define READ_BOARDER3(interpolation)               \
  READ_BOARDER2(interpolation)                     \
  data_index = interpolation(cols, radius, j + 2); \
  value.z = convert_float(src_temp[data_index]);

#define READ_BOARDER4(interpolation)               \
  READ_BOARDER3(interpolation)                     \
  data_index = interpolation(cols, radius, j + 3); \
  value.w = convert_float(src_temp[data_index]);

__kernel
void getGaussianKernel(float sigma, int ksize, global float* coefficients,
                       int offset) {
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
    sum += value;
  }

  sum = 1.f / sum;
  for (i = 0; i < ksize; i++) {
    coefficients[i] *= sum;
  }
}

// #if defined(TRANSPOSE_U81C) || defined(TRANSPOSE_F321C) || \
//     defined(ALL_KERNELS)
#define ADAPTIVETHRESHOLD_MEAN_KERNEL_C1_TYPE(base_type_src, Tsrc,                \
                                              base_type_dst, Tdst, cols_load,     \
                                              rows_load, interpolation, is_set_value)           \
  __kernel void                                                                   \
      adaptivethreshold_mean##base_type_src##base_type_dst##interpolation##is_set_value##C1Kernel( \
          global const uchar* src_input, int src_input_stride,                     \
          global const Tsrc* src, int rows, int cols, int radius,                 \
          int src_stride, global Tdst* dst, int dst_stride,     \
          int threshold_type, float weight, int delta,          \
          uchar setted_value) {                                                   \
    int element_x = get_global_id(0);                                             \
    int element_y = get_global_id(1);                                             \
    int group_x = get_group_id(0);                                                \
    int group_y = get_group_id(1);                                                \
    int index_x = element_x * cols_load, index_y = element_y * rows_load;         \
    if (index_x >= cols || index_y >= rows) {                                     \
      return;                                                                     \
    }                                                                             \
                                                                                  \
    src = (global const Tsrc*)((global uchar*)src + index_y * src_stride);               \
    int remain_cols = cols - index_x, remain_rows = rows - index_y;               \
    int bottom = index_x - radius;                                                \
    int top = index_x + radius;                                                   \
    int data_index;                                                               \
                                                                                  \
    float##cols_load input_value[rows_load];                                      \
    bool isnt_border_block = true;                                                \
    data_index = radius / (get_local_size(0) * cols_load);                        \
    if (group_x <= data_index)                                                    \
      isnt_border_block = false;                                                  \
    data_index = (cols - radius) / (get_local_size(0) * cols_load);               \
    if (group_x >= data_index)                                                    \
      isnt_border_block = false;                                                  \
                                                                                  \
    global const Tsrc* src_temp;                                                  \
    for (int i = 0; i < min(remain_rows, rows_load); i++) {                       \
      ((float##cols_load*)input_value+i)[0] = (float##cols_load)(0);                                     \
      src_temp = src;                                                             \
      if (isnt_border_block) {                                                    \
        src_temp += bottom;                                                       \
        for (int j = bottom; j <= top; j++) {                                     \
          ((float##cols_load*)input_value+i)[0] +=                                                       \
              convert_float##cols_load(vload##cols_load(0, src_temp));            \
          src_temp += 1;                                                          \
        }                                                                         \
      }                                                                           \
      else {                                                                      \
        float##cols_load value;                                                   \
        for (int j = bottom; j <= top; j++) {                                     \
          READ_BOARDER##cols_load(interpolation);                                 \
          ((float##cols_load*)input_value+i)[0] += value;                                                \
        }                                                                         \
      }                                                                           \
      src = (global const Tsrc*)((global uchar*)src + src_stride);                       \
    }                                                                             \
    is_set_value##_NORMALIZA(rows_load, cols_load)\
                                                                                  \
    dst = (global Tdst*)((global uchar*)dst + dst_stride * index_x);                     \
    src_input = src_input + index_x * src_input_stride;           \
    is_set_value ## _C1_RAMAIN_COL ## rows_load(Tdst, cols_load, rows_load)              \
  }
// #endif

#define ADAPTIVETHRESHOLD_GAUSSIANBLUR_KERNEL_C1_TYPE(                                       \
    base_type_src, Tsrc, base_type_dst, Tdst, cols_load, rows_load,                          \
    interpolation, is_set_value)                                                                           \
  __kernel void                                                                              \
      adaptivethreshold_gaussianblur##base_type_src##base_type_dst##interpolation##is_set_value##C1Kernel( \
          global const uchar* src_input, int src_input_stride,\
          global const Tsrc* src, int rows, int cols,                                        \
          global const float* filter_kernel, int kernel_offset, int radius, int src_stride,                     \
          global Tdst* dst, int dst_stride, int threshold_type,                              \
          int delta, uchar setted_value) {                                 \
    int element_x = get_global_id(0);                                                        \
    int element_y = get_global_id(1);                                                        \
    int group_x = get_group_id(0);                                                           \
    int group_y = get_group_id(1);                                                           \
    int index_x = element_x * cols_load, index_y = element_y * rows_load;                    \
    if (index_x >= cols || index_y >= rows) {                                                \
      return;                                                                                \
    }                                                                                        \
                                                                                             \
    src = (global const Tsrc*)((global uchar*)src + index_y * src_stride);                          \
    filter_kernel = (global const float*)((global uchar*)filter_kernel + kernel_offset);     \
    int remain_cols = cols - index_x, remain_rows = rows - index_y;                          \
    int bottom = index_x - radius;                                                           \
    int top = index_x + radius;                                                              \
    int filter_kernel_index;                                                                 \
    int data_index;                                                                          \
                                                                                             \
    float##cols_load input_value[rows_load];                                                 \
    bool isnt_border_block = true;                                                           \
    data_index = radius / (get_local_size(0) * cols_load);                                   \
    if (group_x <= data_index)                                                               \
      isnt_border_block = false;                                                             \
    data_index = (cols - radius) / (get_local_size(0) * cols_load);                          \
    if (group_x >= data_index)                                                               \
      isnt_border_block = false;                                                             \
                                                                                             \
    global const Tsrc* src_temp;                                                             \
    for (int i = 0; i < min(remain_rows, rows_load); i++) {                                  \
      ((float##cols_load*)input_value+i)[0] = (float##cols_load)(0);                                                \
      src_temp = src;                                                                        \
      filter_kernel_index = 0;                                                               \
      if (isnt_border_block) {                                                               \
        src_temp += bottom;                                                                  \
        for (int j = bottom; j <= top; j++) {                                                \
          ((float##cols_load*)input_value+i)[0] +=                                                                  \
              convert_float##cols_load(vload##cols_load(0, src_temp)) *                      \
              filter_kernel[filter_kernel_index];                                            \
          src_temp += 1;                                                                     \
          filter_kernel_index++;                                                             \
        }                                                                                    \
      }                                                                                      \
      else {                                                                                 \
        float##cols_load value;                                                              \
        for (int j = bottom; j <= top; j++) {                                                \
          READ_BOARDER##cols_load(interpolation);                                            \
          ((float##cols_load*)input_value+i)[0] += value * filter_kernel[filter_kernel_index];                      \
          filter_kernel_index++;                                                             \
        }                                                                                    \
      }                                                                                      \
      src = (global const Tsrc*)((global uchar*)src + src_stride);                                  \
    }                                                                                        \
                                                                                             \
    dst = (global Tdst*)((global uchar*)dst + dst_stride * index_x);                                \
    src_input = src_input + index_x * src_input_stride;                      \
    is_set_value##_C1_RAMAIN_COL##rows_load(Tdst, cols_load, rows_load)                         \
  }

ADAPTIVETHRESHOLD_MEAN_KERNEL_C1_TYPE(F32, float, U8, uchar, 4, 4, interpolateReplicateBorder, THRESHOLD_TRANSPOSE)
ADAPTIVETHRESHOLD_MEAN_KERNEL_C1_TYPE(U8, uchar, F32, float, 2, 2, interpolateReplicateBorder, TRANSPOSE)
ADAPTIVETHRESHOLD_MEAN_KERNEL_C1_TYPE(F32, float, U8, uchar, 4, 4, interpolateReflectBorder, THRESHOLD_TRANSPOSE)
ADAPTIVETHRESHOLD_MEAN_KERNEL_C1_TYPE(U8, uchar, F32, float, 2, 2, interpolateReflectBorder, TRANSPOSE)
ADAPTIVETHRESHOLD_MEAN_KERNEL_C1_TYPE(F32, float, U8, uchar, 4, 4, interpolateReflect101Border, THRESHOLD_TRANSPOSE)
ADAPTIVETHRESHOLD_MEAN_KERNEL_C1_TYPE(U8, uchar, F32, float, 2, 2, interpolateReflect101Border, TRANSPOSE)

ADAPTIVETHRESHOLD_GAUSSIANBLUR_KERNEL_C1_TYPE(F32, float, U8, uchar, 4, 4, interpolateReplicateBorder, THRESHOLD_TRANSPOSE)
ADAPTIVETHRESHOLD_GAUSSIANBLUR_KERNEL_C1_TYPE(U8, uchar, F32, float, 2, 2, interpolateReplicateBorder, TRANSPOSE)
ADAPTIVETHRESHOLD_GAUSSIANBLUR_KERNEL_C1_TYPE(F32, float, U8, uchar, 4, 4, interpolateReflectBorder, THRESHOLD_TRANSPOSE)
ADAPTIVETHRESHOLD_GAUSSIANBLUR_KERNEL_C1_TYPE(U8, uchar, F32, float, 2, 2, interpolateReflectBorder, TRANSPOSE)
ADAPTIVETHRESHOLD_GAUSSIANBLUR_KERNEL_C1_TYPE(F32, float, U8, uchar, 4, 4, interpolateReflect101Border, THRESHOLD_TRANSPOSE)
ADAPTIVETHRESHOLD_GAUSSIANBLUR_KERNEL_C1_TYPE(U8, uchar, F32, float, 2, 2, interpolateReflect101Border, TRANSPOSE)


