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

#define BOXFILTER_VEC1(src_array, i) \
  src_array[0].INDEX_CONVERT##i
#define BOXFILTER_VEC2(src_array, i) \
  BOXFILTER_VEC1(src_array, i), src_array[1].INDEX_CONVERT##i
#define BOXFILTER_VEC3(src_array, i) \
  BOXFILTER_VEC2(src_array, i), src_array[2].INDEX_CONVERT##i
#define BOXFILTER_VEC4(src_array, i) \
  BOXFILTER_VEC3(src_array, i), src_array[3].INDEX_CONVERT##i

#define vstore1 vstore
#define uchar1 uchar
#define float1 float

#define BOXFILTER_ELEMENT1(T, src_array, output, j)              \
  output[0] = convert_##T##j((float##j)(BOXFILTER_VEC##j(src_array, 1)));
#define BOXFILTER_ELEMENT2(T, src_array, output, j)              \
  BOXFILTER_ELEMENT1(T, src_array, output, j)                    \
  output[1] = convert_##T##j((float##j)(BOXFILTER_VEC##j(src_array, 2)));
#define BOXFILTER_ELEMENT3(T, src_array, output, j)              \
  BOXFILTER_ELEMENT2(T, src_array, output, j)                    \
  output[2] = convert_##T##j((float##j)(BOXFILTER_VEC##j(src_array, 3)));
#define BOXFILTER_ELEMENT4(T, src_array, output, j)              \
  BOXFILTER_ELEMENT3(T, src_array, output, j)                    \
  output[3] = convert_##T##j((float##j)(BOXFILTER_VEC##j(src_array, 4)));

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

#define BOXFILTER_AND_SAVE_VEC(T, src_array, output, rows_load, i, j) \
  {                                                                     \
    T##j output_value[i];                                               \
    BOXFILTER_ELEMENT##i(T, src_array, output, j)                     \
    for (int k = 0; k < i; k++) {                                       \
      vstore##j(output[k], element_y, dst);                             \
      dst = (global T*)((global uchar*)dst + dst_stride);                      \
    }                                                                   \
  }

#define BOXFILTER_AND_SAVE_SCALE(T, src_array, output, rows_load, i, j) \
  {                                                                       \
    T##j output_value[i];                                                 \
    BOXFILTER_ELEMENT##i(T, src_array, output, j)                       \
    for (int k = 0; k < i; k++) {                                         \
      VSTORE_REMAIN_ROWS##j(output[k], rows_load)                         \
      dst = (global T*)((global uchar*)dst + dst_stride);                        \
    }                                                                     \
  }

#define BOXFILTER_C1_RAMAIN_ROW1(T, cols_load, rows_load, rows_load_global, \
                                   save_name)                                 \
  if (remain_cols >= cols_load)                                               \
  BOXFILTER_AND_SAVE_##save_name(T, input_value, output_value,              \
                                   rows_load_global, cols_load, rows_load)

#define BOXFILTER_C1_RAMAIN_ROW2(T, cols_load, rows_load, rows_load_global, \
                                   save_name)                                 \
  BOXFILTER_C1_RAMAIN_ROW1(T, cols_load, rows_load, rows_load_global,       \
                             save_name)                                       \
  else if (remain_cols == 1) BOXFILTER_AND_SAVE_##save_name(                \
      T, input_value, output_value, rows_load_global, 1, rows_load)

#define BOXFILTER_C1_RAMAIN_ROW3(T, cols_load, rows_load, rows_load_global, \
                                   save_name)                                 \
  BOXFILTER_C1_RAMAIN_ROW2(T, cols_load, rows_load, rows_load_global,       \
                             save_name)                                       \
  else if (remain_cols == 2) BOXFILTER_AND_SAVE_##save_name(                \
      T, input_value, output_value, rows_load_global, 2, rows_load)

#define BOXFILTER_C1_RAMAIN_ROW4(T, cols_load, rows_load, rows_load_global, \
                                   save_name)                                 \
  BOXFILTER_C1_RAMAIN_ROW3(T, cols_load, rows_load, rows_load_global,       \
                             save_name)                                       \
  else if (remain_cols == 3) BOXFILTER_AND_SAVE_##save_name(                \
      T, input_value, output_value, rows_load_global, 3, rows_load)

#define BOXFILTER_C1_RAMAIN_COL1(T, cols_load, rows_load)                  \
  if (remain_rows >= rows_load) {                                            \
    BOXFILTER_C1_RAMAIN_ROW##cols_load(T, cols_load, rows_load, rows_load, \
                                         VEC)                                \
  }

#define BOXFILTER_C1_RAMAIN_COL2(T, cols_load, rows_load)                 \
  BOXFILTER_C1_RAMAIN_COL1(T, cols_load, rows_load)                       \
  else if (remain_rows == 1) {                                              \
    BOXFILTER_C1_RAMAIN_ROW##cols_load(T, cols_load, 1, rows_load, SCALE) \
  }

#define BOXFILTER_C1_RAMAIN_COL3(T, cols_load, rows_load)                 \
  BOXFILTER_C1_RAMAIN_COL2(T, cols_load, rows_load)                       \
  else if (remain_rows == 2) {                                              \
    BOXFILTER_C1_RAMAIN_ROW##cols_load(T, cols_load, 2, rows_load, SCALE) \
  }

#define BOXFILTER_C1_RAMAIN_COL4(T, cols_load, rows_load)                 \
  BOXFILTER_C1_RAMAIN_COL3(T, cols_load, rows_load)                       \
  else if (remain_rows == 3) {                                              \
    BOXFILTER_C1_RAMAIN_ROW##cols_load(T, cols_load, 3, rows_load, SCALE) \
  }

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

// #if defined(BOXFILTER_U81C) || defined(BOXFILTER_F321C) || \
//     defined(ALL_KERNELS)
#define BOXFILTER_KERNEL_C1_TYPE(base_type_src, Tsrc, base_type_dst, Tdst,    \
                                 cols_load, rows_load, interpolation)         \
  __kernel void                                                               \
      boxfilter##base_type_src##base_type_dst##interpolation##C1Kernel(       \
          global const Tsrc* src, int src_offset, int rows, int cols, int radius,             \
          int src_stride, global Tdst* dst, int dst_stride, int is_symmetric, \
          int normalize, float weight, int dst_offset) {                                             \
    int element_x = get_global_id(0);                                         \
    int element_y = get_global_id(1);                                         \
    int group_x = get_group_id(0);                                            \
    int group_y = get_group_id(1);                                            \
    int index_x = element_x * cols_load, index_y = element_y * rows_load;     \
    src = (global const Tsrc*)((global uchar*)src + src_offset);                            \
    dst = (global Tdst*)((global uchar*)dst + dst_offset);                                  \
    if (index_x >= cols || index_y >= rows) {                                 \
      return;                                                                 \
    }                                                                         \
                                                                              \
    src = (global const Tsrc*)((global uchar*)src + index_y * src_stride);           \
    int remain_cols = cols - index_x, remain_rows = rows - index_y;           \
    int bottom = index_x - radius;                                            \
    int top = index_x + radius;                                               \
    int data_index;                                                           \
                                                                              \
    if (!is_symmetric) {                                                      \
      top -= 1;                                                               \
    }                                                                         \
                                                                              \
    float##cols_load input_value[rows_load];                                  \
    bool isnt_border_block = true;                                            \
    data_index = radius / (get_local_size(0) * cols_load);                    \
    if (group_x <= data_index)                                                \
      isnt_border_block = false;                                              \
    data_index = (cols - radius) / (get_local_size(0) * cols_load);           \
    if (group_x >= data_index)                                                \
      isnt_border_block = false;                                              \
                                                                              \
    global const Tsrc* src_temp;                                              \
    for (int i = 0; i < min(remain_rows, rows_load); i++) {                   \
      ((float##cols_load*)input_value+i)[0] = (float##cols_load)(0);                                 \
      src_temp = src;                                                         \
      if (isnt_border_block) {                                                \
        src_temp += bottom;                                                   \
        for (int j = bottom; j <= top; j++) {                                 \
          ((float##cols_load*)input_value+i)[0] +=                                                   \
              convert_float##cols_load(vload##cols_load(0, src_temp));        \
          src_temp += 1;                                                      \
        }                                                                     \
      }                                                                       \
      else {                                                                  \
        float##cols_load value;                                               \
        for (int j = bottom; j <= top; j++) {                                 \
          READ_BOARDER##cols_load(interpolation);                             \
          ((float##cols_load*)input_value+i)[0] += value;                                            \
        }                                                                     \
      }                                                                       \
      src = (global const Tsrc*)((global uchar*)src + src_stride);                   \
    }                                                                         \
    if (normalize) {                                                   \
      for (int i = 0; i < min(remain_rows, rows_load); i++) {                 \
        ((float##cols_load*)input_value+i)[0] *= weight;                  \
      }                                                                       \
    }                                                                         \
                                                                              \
    dst = (global Tdst*)((global uchar*)dst + dst_stride * index_x);                 \
    BOXFILTER_C1_RAMAIN_COL##rows_load(Tdst, cols_load, rows_load)            \
  }
// #endif

#define BOXFILTER_SAVE_CHANNEL1(T, rows_load, channels)                    \
  if (remain_rows >= rows_load) {                                          \
    for (int i = 0; i < rows_load; i++) {                                  \
      vstore##channels(convert_##T##channels(input_value[i]), index_y + i, \
                       dst);                                               \
    }                                                                      \
  }

#define BOXFILTER_SAVE_CHANNEL2(T, rows_load, channels)                    \
  BOXFILTER_SAVE_CHANNEL1(T, rows_load, channels)                          \
  else if (remain_rows == 1) {                                             \
    vstore##channels(convert_##T##channels(input_value[0]), index_y, dst); \
  }

#define BOXFILTER_SAVE_CHANNEL3(T, rows_load, channels)                        \
  BOXFILTER_SAVE_CHANNEL2(T, rows_load, channels)                              \
  else if (remain_rows == 2) {                                                 \
    vstore##channels(convert_##T##channels(input_value[0]), index_y, dst);     \
    vstore##channels(convert_##T##channels(input_value[1]), index_y + 1, dst); \
  }

#define BOXFILTER_SAVE_CHANNEL4(T, rows_load, channels)                        \
  BOXFILTER_SAVE_CHANNEL3(T, rows_load, channels)                              \
  else if (remain_rows == 3) {                                                 \
    vstore##channels(convert_##T##channels(input_value[0]), index_y, dst);     \
    vstore##channels(convert_##T##channels(input_value[1]), index_y + 1, dst); \
    vstore##channels(convert_##T##channels(input_value[2]), index_y + 2, dst); \
  }

// #if defined(BOXFILTER_U8C3) || defined(BOXFILTER_F32C3) || \
//     defined(BOXFILTER_U8C4) || defined(BOXFILTER_F32C4) || \
//     defined(ALL_KERNELS)
#define BOXFILTER_KERNEL_CN_TYPE(base_type_src, Tsrc, base_type_dst, Tdst,         \
                                 channels, rows_load, interpolation)               \
  __kernel void                                                                    \
      boxfilter##base_type_src##base_type_dst##interpolation##C##channels##Kernel( \
          global const Tsrc* src, int src_offset, int rows, int cols, int radius,                  \
          int src_stride, global Tdst* dst, int dst_stride, int is_symmetric,      \
          int normalize, float weight, int dst_offset) {                                                  \
    int element_x = get_global_id(0);                                              \
    int element_y = get_global_id(1);                                              \
    int group_x = get_group_id(0);                                                 \
    int group_y = get_group_id(1);                                                 \
    int index_x = element_x, index_y = element_y * rows_load;                      \
    src = (global const Tsrc*)((global uchar*)src + src_offset);                            \
    dst = (global Tdst*)((global uchar*)dst + dst_offset);                                  \
    if (index_x >= cols || index_y >= rows) {                                      \
      return;                                                                      \
    }                                                                              \
                                                                                   \
    src = (global const Tsrc*)((global uchar*)src + index_y * src_stride);                \
    int remain_rows = rows - index_y;                                              \
    int bottom = index_x - radius;                                                 \
    int top = index_x + radius;                                                    \
    int data_index;                                                                \
                                                                                   \
    if (!is_symmetric) {                                                           \
      top -= 1;                                                                    \
    }                                                                              \
                                                                                   \
    float##channels input_value[rows_load];                                        \
                                                                                   \
    bool isnt_border_block = true;                                                 \
    data_index = radius / (get_local_size(0));                                     \
    if (group_x <= data_index)                                                     \
      isnt_border_block = false;                                                   \
    data_index = (cols - radius) / (get_local_size(0));                            \
    if (group_x >= data_index)                                                     \
      isnt_border_block = false;                                                   \
                                                                                   \
    for (int i = 0; i < min(remain_rows, rows_load); i++) {                        \
      ((float##channels*)input_value+i)[0] = (float##channels)(0);                                       \
      if (isnt_border_block) {                                                     \
        for (int j = bottom; j <= top; j++) {                                      \
          ((float##channels*)input_value+i)[0] += convert_float##channels(vload##channels(j, src));      \
        }                                                                          \
      }                                                                            \
      else {                                                                       \
        for (int j = bottom; j <= top; j++) {                                      \
          data_index = interpolation(cols, radius, j);                             \
          ((float##channels*)input_value+i)[0] +=                                                        \
              convert_float##channels(vload##channels(data_index, src));           \
        }                                                                          \
      }                                                                            \
      src = (global const Tsrc*)((global uchar*)src + src_stride);                        \
    }                                                                              \
    if (normalize) {                                                        \
      for (int i = 0; i < min(remain_rows, rows_load); i++) {                      \
        ((float##channels*)input_value+i)[0] *= weight;                       \
      }                                                                            \
    }                                                                              \
                                                                                   \
    dst = (global Tdst*)((global uchar*)dst + dst_stride * index_x);                      \
    BOXFILTER_SAVE_CHANNEL##rows_load(Tdst, rows_load, channels)                   \
  }
// #endif

BOXFILTER_KERNEL_C1_TYPE(F32, float, U8, uchar, 2, 2, interpolateReplicateBorder)
BOXFILTER_KERNEL_C1_TYPE(U8, uchar, F32, float, 2, 2, interpolateReplicateBorder)
BOXFILTER_KERNEL_C1_TYPE(F32, float, F32, float, 2, 2, interpolateReplicateBorder)
BOXFILTER_KERNEL_C1_TYPE(F32, float, U8, uchar, 2, 2, interpolateReflectBorder)
BOXFILTER_KERNEL_C1_TYPE(U8, uchar, F32, float, 2, 2, interpolateReflectBorder)
BOXFILTER_KERNEL_C1_TYPE(F32, float, F32, float, 2, 2, interpolateReflectBorder)
BOXFILTER_KERNEL_C1_TYPE(F32, float, U8, uchar, 2, 2, interpolateReflect101Border)
BOXFILTER_KERNEL_C1_TYPE(U8, uchar, F32, float, 2, 2, interpolateReflect101Border)
BOXFILTER_KERNEL_C1_TYPE(F32, float, F32, float, 2, 2, interpolateReflect101Border)

BOXFILTER_KERNEL_CN_TYPE(U8, uchar, F32, float, 3, 1, interpolateReplicateBorder)
BOXFILTER_KERNEL_CN_TYPE(F32, float, U8, uchar, 3, 1, interpolateReplicateBorder)
BOXFILTER_KERNEL_CN_TYPE(F32, float, F32, float, 3, 1, interpolateReplicateBorder)
BOXFILTER_KERNEL_CN_TYPE(U8, uchar, F32, float, 3, 1, interpolateReflectBorder)
BOXFILTER_KERNEL_CN_TYPE(F32, float, U8, uchar, 3, 1, interpolateReflectBorder)
BOXFILTER_KERNEL_CN_TYPE(F32, float, F32, float, 3, 1, interpolateReflectBorder)
BOXFILTER_KERNEL_CN_TYPE(U8, uchar, F32, float, 3, 1, interpolateReflect101Border)
BOXFILTER_KERNEL_CN_TYPE(F32, float, U8, uchar, 3, 1, interpolateReflect101Border)
BOXFILTER_KERNEL_CN_TYPE(F32, float, F32, float, 3, 1, interpolateReflect101Border)

BOXFILTER_KERNEL_CN_TYPE(U8, uchar, F32, float, 4, 1, interpolateReplicateBorder)
BOXFILTER_KERNEL_CN_TYPE(F32, float, U8, uchar, 4, 1, interpolateReplicateBorder)
BOXFILTER_KERNEL_CN_TYPE(F32, float, F32, float, 4, 1, interpolateReplicateBorder)
BOXFILTER_KERNEL_CN_TYPE(U8, uchar, F32, float, 4, 1, interpolateReflectBorder)
BOXFILTER_KERNEL_CN_TYPE(F32, float, U8, uchar, 4, 1, interpolateReflectBorder)
BOXFILTER_KERNEL_CN_TYPE(F32, float, F32, float, 4, 1, interpolateReflectBorder)
BOXFILTER_KERNEL_CN_TYPE(U8, uchar, F32, float, 4, 1, interpolateReflect101Border)
BOXFILTER_KERNEL_CN_TYPE(F32, float, U8, uchar, 4, 1, interpolateReflect101Border)
BOXFILTER_KERNEL_CN_TYPE(F32, float, F32, float, 4, 1, interpolateReflect101Border)


// #if defined(BOXFILTER_F32C3) || defined(ALL_KERNELS)
// BOXFILTER_KERNEL_CN_TYPE(F32, float, 3, 1)
// #endif

// #if defined(BOXFILTER_U8C4) || defined(ALL_KERNELS)
// BOXFILTER_KERNEL_CN_TYPE(U8, uchar, 4, 4)
// #endif

// #if defined(BOXFILTER_F32C4) || defined(ALL_KERNELS)
// BOXFILTER_KERNEL_CN_TYPE(F32, float, 4, 1)
// #endif
