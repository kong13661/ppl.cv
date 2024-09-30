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

#define INDEX_CONVERT1 x
#define INDEX_CONVERT2 y
#define INDEX_CONVERT3 z
#define INDEX_CONVERT4 w

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
  output[0] = (T##j)(TRANSPOSE_VEC##j(src_array, 1));
#define TRANSPOSE_ELEMENT2(T, src_array, output, j)              \
  TRANSPOSE_ELEMENT1(T, src_array, output, j)                    \
  output[1] = (T##j)(TRANSPOSE_VEC##j(src_array, 2));
#define TRANSPOSE_ELEMENT3(T, src_array, output, j)              \
  TRANSPOSE_ELEMENT2(T, src_array, output, j)                    \
  output[2] = (T##j)(TRANSPOSE_VEC##j(src_array, 3));
#define TRANSPOSE_ELEMENT4(T, src_array, output, j)              \
  TRANSPOSE_ELEMENT3(T, src_array, output, j)                    \
  output[3] = (T##j)(TRANSPOSE_VEC##j(src_array, 4));

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

#define TRANSPOSE_AND_SAVE_VEC(T, src_array, output, rows_load, i, j) \
  {                                                                   \
    T##j output_value[i];                                             \
    TRANSPOSE_ELEMENT##i(T, src_array, output, j)                     \
    for (int k = 0; k < i; k++) {                                     \
      vstore##j(output[k], element_y, dst);                           \
      dst = (global T*)((uchar*)dst + dst_stride);                    \
    }                                                                 \
  }

#define TRANSPOSE_AND_SAVE_SCALE(T, src_array, output, rows_load, i, j) \
  {                                                                     \
    T##j output_value[i];                                               \
    TRANSPOSE_ELEMENT##i(T, src_array, output, j)                       \
    for (int k = 0; k < i; k++) {                                       \
      VSTORE_REMAIN_ROWS##j(output[k], rows_load)                       \
      dst = (global T*)((uchar*)dst + dst_stride);                      \
    }                                                                   \
  }

#define TRANSPOSE_C1_RAMAIN_ROW1(T, cols_load, rows_load, rows_load_global, \
                                 save_name)                                 \
  if (remain_cols >= cols_load)                                             \
  TRANSPOSE_AND_SAVE_##save_name(T, input_value, output_value,              \
                                 rows_load_global, cols_load, rows_load)

#define TRANSPOSE_C1_RAMAIN_ROW2(T, cols_load, rows_load, rows_load_global, \
                                 save_name)                                 \
  TRANSPOSE_C1_RAMAIN_ROW1(T, cols_load, rows_load, rows_load_global,       \
                           save_name)                                       \
  else if (remain_cols == 1) TRANSPOSE_AND_SAVE_##save_name(                \
      T, input_value, output_value, rows_load_global, 1, rows_load)

#define TRANSPOSE_C1_RAMAIN_ROW3(T, cols_load, rows_load, rows_load_global, \
                                 save_name)                                 \
  TRANSPOSE_C1_RAMAIN_ROW2(T, cols_load, rows_load, rows_load_global,       \
                           save_name)                                       \
  else if (remain_cols == 2) TRANSPOSE_AND_SAVE_##save_name(                \
      T, input_value, output_value, rows_load_global, 2, rows_load)

#define TRANSPOSE_C1_RAMAIN_ROW4(T, cols_load, rows_load, rows_load_global, \
                                 save_name)                                 \
  TRANSPOSE_C1_RAMAIN_ROW3(T, cols_load, rows_load, rows_load_global,       \
                           save_name)                                       \
  else if (remain_cols == 3) TRANSPOSE_AND_SAVE_##save_name(                \
      T, input_value, output_value, rows_load_global, 3, rows_load)

#define TRANSPOSE_C1_RAMAIN_COL1(T, cols_load, rows_load)                  \
  if (remain_rows >= rows_load) {                                          \
    TRANSPOSE_C1_RAMAIN_ROW##cols_load(T, cols_load, rows_load, rows_load, \
                                       VEC)                                \
  }

#define TRANSPOSE_C1_RAMAIN_COL2(T, cols_load, rows_load)                 \
  TRANSPOSE_C1_RAMAIN_COL1(T, cols_load, rows_load)                       \
  else if (remain_rows == 1) {                                            \
    TRANSPOSE_C1_RAMAIN_ROW##cols_load(T, cols_load, 1, rows_load, SCALE) \
  }

#define TRANSPOSE_C1_RAMAIN_COL3(T, cols_load, rows_load)                 \
  TRANSPOSE_C1_RAMAIN_COL2(T, cols_load, rows_load)                       \
  else if (remain_rows == 2) {                                            \
    TRANSPOSE_C1_RAMAIN_ROW##cols_load(T, cols_load, 2, rows_load, SCALE) \
  }

#define TRANSPOSE_C1_RAMAIN_COL4(T, cols_load, rows_load)                 \
  TRANSPOSE_C1_RAMAIN_COL3(T, cols_load, rows_load)                       \
  else if (remain_rows == 3) {                                            \
    TRANSPOSE_C1_RAMAIN_ROW##cols_load(T, cols_load, 3, rows_load, SCALE) \
  }

// #if defined(TRANSPOSE_U81C) || defined(TRANSPOSE_F321C) || defined(ALL_KERNELS)
#define TRANSPOSE_KERNEL_C1_TYPE(base_type, T, rows_load, cols_load)          \
  __kernel void transpose##base_type##C1Kernel(                               \
      global const T* src, int rows, int cols, int src_stride, global T* dst, \
      int dst_stride) {                                                       \
    int element_x = get_global_id(0);                                         \
    int element_y = get_global_id(1);                                         \
    int index_x = element_x * cols_load, index_y = element_y * rows_load;     \
    if (index_x >= cols || index_y >= rows) {                                 \
      return;                                                                 \
    }                                                                         \
                                                                              \
    src = (global const T*)((uchar*)src + index_y * src_stride);              \
    int remain_cols = cols - index_x, remain_rows = rows - index_y;           \
                                                                              \
    T##cols_load input_value[rows_load];                                      \
    for (int i = 0; i < min(remain_rows, rows_load); i++) {                   \
      input_value[i] = vload##cols_load(element_x, src);                      \
      src = (global const T*)((uchar*)src + src_stride);                      \
    }                                                                         \
                                                                              \
    dst = (global T*)((uchar*)dst + dst_stride * index_x);                    \
    TRANSPOSE_C1_RAMAIN_COL##rows_load(T, cols_load, rows_load)               \
  }
// #endif

#define TRANSPOSE_SAVE_CHANNEL1(T, rows_load, channels)   \
  if (remain_rows >= rows_load) {                         \
    for (int i = 0; i < rows_load; i++) {                 \
      vstore##channels(input_value[i], index_y + i, dst); \
    }                                                     \
  }

#define TRANSPOSE_SAVE_CHANNEL2(T, rows_load, channels) \
  TRANSPOSE_SAVE_CHANNEL1(T, rows_load, channels)       \
  else if (remain_rows == 1) {                          \
    vstore##channels(input_value[0], index_y, dst);     \
  }

#define TRANSPOSE_SAVE_CHANNEL3(T, rows_load, channels) \
  TRANSPOSE_SAVE_CHANNEL2(T, rows_load, channels)       \
  else if (remain_rows == 2) {                          \
    vstore##channels(input_value[0], index_y, dst);     \
    vstore##channels(input_value[1], index_y + 1, dst); \
  }

#define TRANSPOSE_SAVE_CHANNEL4(T, rows_load, channels) \
  TRANSPOSE_SAVE_CHANNEL3(T, rows_load, channels)       \
  else if (remain_rows == 3) {                          \
    vstore##channels(input_value[0], index_y, dst);     \
    vstore##channels(input_value[1], index_y + 1, dst); \
    vstore##channels(input_value[2], index_y + 2, dst); \
  }

#if defined(TRANSPOSE_U8C3) || defined(TRANSPOSE_F32C3) || \
    defined(TRANSPOSE_U8C4) || defined(TRANSPOSE_F32C4) || defined(ALL_KERNELS)
#define TRANSPOSE_KERNEL_CN_TYPE(base_type, T, channels, rows_load)           \
  __kernel void transpose##base_type##C##channels##Kernel(                    \
      global const T* src, int rows, int cols, int src_stride, global T* dst, \
      int dst_stride) {                                                       \
    int element_x = get_global_id(0);                                         \
    int element_y = get_global_id(1);                                         \
    int index_x = element_x, index_y = element_y * rows_load;                 \
    if (index_x >= cols || index_y >= rows) {                                 \
      return;                                                                 \
    }                                                                         \
                                                                              \
    src = (global const T*)((uchar*)src + index_y * src_stride);              \
    int remain_rows = rows - index_y;                                         \
                                                                              \
    T##channels input_value[rows_load];                                       \
    for (int i = 0; i < min(remain_rows, rows_load); i++) {                   \
      input_value[i] = vload##channels(element_x, src);                       \
      src = (global const T*)((uchar*)src + src_stride);                      \
    }                                                                         \
                                                                              \
    dst = (global T*)((uchar*)dst + dst_stride * index_x);                    \
    TRANSPOSE_SAVE_CHANNEL##rows_load(T, rows_load, channels)                 \
  }
#endif
TRANSPOSE_KERNEL_C1_TYPE(F32, float, 2, 2)

#if defined(TRANSPOSE_U81C) || defined(ALL_KERNELS)
TRANSPOSE_KERNEL_C1_TYPE(U8, uchar, 4, 4)
#endif

#if defined(TRANSPOSE_F321C) || defined(ALL_KERNELS)
TRANSPOSE_KERNEL_C1_TYPE(F32, float, 2, 2)
#endif

#if defined(TRANSPOSE_U8C3) || defined(ALL_KERNELS)
TRANSPOSE_KERNEL_CN_TYPE(U8, uchar, 3, 4)
#endif

#if defined(TRANSPOSE_F32C3) || defined(ALL_KERNELS)
TRANSPOSE_KERNEL_CN_TYPE(F32, float, 3, 1)
#endif

#if defined(TRANSPOSE_U8C4) || defined(ALL_KERNELS)
TRANSPOSE_KERNEL_CN_TYPE(U8, uchar, 4, 4)
#endif

#if defined(TRANSPOSE_F32C4) || defined(ALL_KERNELS)
TRANSPOSE_KERNEL_CN_TYPE(F32, float, 4, 1)
#endif
