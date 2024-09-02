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

#define vstore1 vstore
#define uchar1 uchar
#define float1 float

#define ROTATE_SAVE_90_OUTPUT_RIGHTVALUE_1(T, col_index) \
  input_value[0].INDEX_CONVERT##col_index

#define ROTATE_SAVE_90_OUTPUT_RIGHTVALUE_2(T, col_index) \
  input_value[1].INDEX_CONVERT##col_index,               \
      input_value[0].INDEX_CONVERT##col_index

#define ROTATE_SAVE_90_OUTPUT_RIGHTVALUE_3(T, col_index) \
  input_value[2].INDEX_CONVERT##col_index,               \
      input_value[1].INDEX_CONVERT##col_index,           \
      input_value[0].INDEX_CONVERT##col_index

#define ROTATE_SAVE_90_OUTPUT_RIGHTVALUE_4(T, col_index) \
  input_value[3].INDEX_CONVERT##col_index,               \
      input_value[2].INDEX_CONVERT##col_index,           \
      input_value[1].INDEX_CONVERT##col_index,           \
      input_value[0].INDEX_CONVERT##col_index

#define ROTATE_SAVE_90_OUTPUT_1(T, cols_remained, rows_remained) \
  T##rows_remained output_value;                                 \
  output_value = (T##rows_remained)(                             \
      ROTATE_SAVE_90_OUTPUT_RIGHTVALUE_##rows_remained(T, 1));   \
  vstore##rows_remained(output_value, 0, dst);                   \
  dst = (global T*)((uchar*)dst + dst_stride);

#define ROTATE_SAVE_90_OUTPUT_2(T, cols_remained, rows_remained) \
  ROTATE_SAVE_90_OUTPUT_1(T, cols_remained, rows_remained)       \
  output_value = (T##rows_remained)(                             \
      ROTATE_SAVE_90_OUTPUT_RIGHTVALUE_##rows_remained(T, 2));   \
  vstore##rows_remained(output_value, 0, dst);                   \
  dst = (global T*)((uchar*)dst + dst_stride);

#define ROTATE_SAVE_90_OUTPUT_3(T, cols_remained, rows_remained) \
  ROTATE_SAVE_90_OUTPUT_2(T, cols_remained, rows_remained)       \
  output_value = (T##rows_remained)(                             \
      ROTATE_SAVE_90_OUTPUT_RIGHTVALUE_##rows_remained(T, 3));   \
  vstore##rows_remained(output_value, 0, dst);                   \
  dst = (global T*)((uchar*)dst + dst_stride);

#define ROTATE_SAVE_90_OUTPUT_4(T, cols_remained, rows_remained) \
  ROTATE_SAVE_90_OUTPUT_3(T, cols_remained, rows_remained)       \
  output_value = (T##rows_remained)(                             \
      ROTATE_SAVE_90_OUTPUT_RIGHTVALUE_##rows_remained(T, 4));   \
  vstore##rows_remained(output_value, 0, dst);

#define ROTATE_SAVE_IF_ROW1(T, cols_remained, rows_remained, cols_load,    \
                            rows_load, degree)                             \
  if (rows_remained >= rows_load) {                                        \
    ROTATE_SAVE_##degree##_OUTPUT_##cols_load(T, cols_remained, rows_load) \
  }

#define ROTATE_SAVE_IF_ROW2(T, cols_remained, rows_remained, cols_load,      \
                            rows_load, degree)                               \
  ROTATE_SAVE_IF_ROW1(T, cols_remained, rows_remained, cols_load, rows_load, \
                      degree)                                                \
  else if (rows_remained == 1) {                                             \
    ROTATE_SAVE_##degree##_OUTPUT_##cols_load(T, cols_remained, 1)           \
  }

#define ROTATE_SAVE_IF_ROW3(T, cols_remained, rows_remained, cols_load,      \
                            rows_load, degree)                               \
  ROTATE_SAVE_IF_ROW2(T, cols_remained, rows_remained, cols_load, rows_load, \
                      degree)                                                \
  else if (rows_remained == 2) {                                             \
    ROTATE_SAVE_##degree##_OUTPUT_##cols_load(T, cols_remained, 2)           \
  }

#define ROTATE_SAVE_IF_ROW4(T, cols_remained, rows_remained, cols_load,      \
                            rows_load, degree)                               \
  ROTATE_SAVE_IF_ROW3(T, cols_remained, rows_remained, cols_load, rows_load, \
                      degree)                                                \
  else if (rows_remained == 3) {                                             \
    ROTATE_SAVE_##degree##_OUTPUT_##cols_load(T, cols_remained, 3)           \
  }

#define ROTATE_SAVE_IF_COL1(T, rows_load, cols_remained, rows_remained,   \
                            cols_load, dst, degree)                       \
  if (cols_remained >= cols_load) {                                       \
    ROTATE_SAVE_IF_ROW##rows_load(T, cols_load, rows_remained, cols_load, \
                                  rows_load, degree)                      \
  }

#define ROTATE_SAVE_IF_COL2(T, rows_load, cols_remained, rows_remained,      \
                            cols_load, dst, degree)                          \
  ROTATE_SAVE_IF_COL1(T, rows_load, cols_remained, rows_remained, cols_load, \
                      dst, degree)                                           \
  else if (cols_remained == 1) {                                             \
    ROTATE_SAVE_IF_ROW##rows_load(T, 1, rows_remained, 1, rows_load, degree) \
  }

#define ROTATE_SAVE_IF_COL3(T, rows_load, cols_remained, rows_remained,      \
                            cols_load, dst, degree)                          \
  ROTATE_SAVE_IF_COL2(T, rows_load, cols_remained, rows_remained, cols_load, \
                      dst, degree)                                           \
  else if (cols_remained == 2) {                                             \
    ROTATE_SAVE_IF_ROW##rows_load(T, 2, rows_remained, 2, rows_load, degree) \
  }

#define ROTATE_SAVE_IF_COL4(T, rows_load, cols_remained, rows_remained,      \
                            cols_load, dst, degree)                          \
  ROTATE_SAVE_IF_COL3(T, rows_load, cols_remained, rows_remained, cols_load, \
                      dst, degree)                                           \
  else if (cols_remained == 3) {                                             \
    ROTATE_SAVE_IF_ROW##rows_load(T, 3, rows_remained, 3, rows_load, degree) \
  }

#define ROTATE_DST_90(T, rows_load, cols_load)            \
  dst = (global T*)((uchar*)dst + dst_stride * index_x) + \
        max(rows - index_y - rows_load, 0);

// =======================================================================

#define ROTATE_SAVE_180_OUTPUT_RIGHTVALUE_1(T, row_index) \
  input_value[row_index].x

#define ROTATE_SAVE_180_OUTPUT_RIGHTVALUE_2(T, row_index) \
  input_value[row_index].y, input_value[row_index].x

#define ROTATE_SAVE_180_OUTPUT_RIGHTVALUE_3(T, row_index) \
  input_value[row_index].z, input_value[row_index].y,     \
  input_value[row_index].x

#define ROTATE_SAVE_180_OUTPUT_RIGHTVALUE_4(T, row_index) \
  input_value[row_index].w, input_value[row_index].z,     \
      input_value[row_index].y, input_value[row_index].x

#define ROTATE_SAVE_180_OUTPUT_1_(T, cols_remained, rows_remained) \
  T##cols_remained output_value;                                   \
  output_value = (T##cols_remained)(                               \
      ROTATE_SAVE_180_OUTPUT_RIGHTVALUE_##cols_remained(T, 0));    \
  vstore##cols_remained(output_value, 0, dst);                     \
  dst = (global T*)((uchar*)dst + dst_stride);

#define ROTATE_SAVE_180_OUTPUT_2_(T, cols_remained, rows_remained) \
  T##cols_remained output_value;                                   \
  output_value = (T##cols_remained)(                               \
      ROTATE_SAVE_180_OUTPUT_RIGHTVALUE_##cols_remained(T, 1));    \
  vstore##cols_remained(output_value, 0, dst);                     \
  dst = (global T*)((uchar*)dst + dst_stride);                     \
                                                                   \
  output_value = (T##cols_remained)(                               \
      ROTATE_SAVE_180_OUTPUT_RIGHTVALUE_##cols_remained(T, 0));    \
  vstore##cols_remained(output_value, 0, dst);                     \
  dst = (global T*)((uchar*)dst + dst_stride);

#define ROTATE_SAVE_180_OUTPUT_3_(T, cols_remained, rows_remained) \
  T##cols_remained output_value;                                   \
  output_value = (T##cols_remained)(                               \
      ROTATE_SAVE_180_OUTPUT_RIGHTVALUE_##cols_remained(T, 2));    \
  vstore##cols_remained(output_value, 0, dst);                     \
  dst = (global T*)((uchar*)dst + dst_stride);                     \
                                                                   \
  output_value = (T##cols_remained)(                               \
      ROTATE_SAVE_180_OUTPUT_RIGHTVALUE_##cols_remained(T, 1));    \
  vstore##cols_remained(output_value, 0, dst);                     \
  dst = (global T*)((uchar*)dst + dst_stride);                     \
                                                                   \
  output_value = (T##cols_remained)(                               \
      ROTATE_SAVE_180_OUTPUT_RIGHTVALUE_##cols_remained(T, 0));    \
  vstore##cols_remained(output_value, 0, dst);                     \
  dst = (global T*)((uchar*)dst + dst_stride);

#define ROTATE_SAVE_180_OUTPUT_4_(T, cols_remained, rows_remained) \
  T##cols_remained output_value;                                   \
  output_value = (T##cols_remained)(                               \
      ROTATE_SAVE_180_OUTPUT_RIGHTVALUE_##cols_remained(T, 3));    \
  vstore##cols_remained(output_value, 0, dst);                     \
  dst = (global T*)((uchar*)dst + dst_stride);                     \
                                                                   \
  output_value = (T##cols_remained)(                               \
      ROTATE_SAVE_180_OUTPUT_RIGHTVALUE_##cols_remained(T, 2));    \
  vstore##cols_remained(output_value, 0, dst);                     \
  dst = (global T*)((uchar*)dst + dst_stride);                     \
                                                                   \
  output_value = (T##cols_remained)(                               \
      ROTATE_SAVE_180_OUTPUT_RIGHTVALUE_##cols_remained(T, 1));    \
  vstore##cols_remained(output_value, 0, dst);                     \
  dst = (global T*)((uchar*)dst + dst_stride);                     \
                                                                   \
  output_value = (T##cols_remained)(                               \
      ROTATE_SAVE_180_OUTPUT_RIGHTVALUE_##cols_remained(T, 0));    \
  vstore##cols_remained(output_value, 0, dst);                     \
  dst = (global T*)((uchar*)dst + dst_stride);

#define ROTATE_SAVE_180_OUTPUT_1(T, cols_remained, rows_remained) \
  ROTATE_SAVE_180_OUTPUT_##rows_remained##_(T, cols_remained, rows_remained)

#define ROTATE_SAVE_180_OUTPUT_2(T, cols_remained, rows_remained) \
  ROTATE_SAVE_180_OUTPUT_##rows_remained##_(T, cols_remained, rows_remained)

#define ROTATE_SAVE_180_OUTPUT_3(T, cols_remained, rows_remained) \
  ROTATE_SAVE_180_OUTPUT_##rows_remained##_(T, cols_remained, rows_remained)

#define ROTATE_SAVE_180_OUTPUT_4(T, cols_remained, rows_remained) \
  ROTATE_SAVE_180_OUTPUT_##rows_remained##_(T, cols_remained, rows_remained)

#define ROTATE_DST_180(T, rows_load, cols_load)                        \
  dst = (global T*)((uchar*)dst +                                      \
                    dst_stride * max(rows - index_y - rows_load, 0)) + \
        max(cols - index_x - cols_load, 0);

// ===============================================================================

#define ROTATE_SAVE_270_OUTPUT_RIGHTVALUE_1(T, col_index) \
  input_value[0].INDEX_CONVERT##col_index

#define ROTATE_SAVE_270_OUTPUT_RIGHTVALUE_2(T, col_index) \
  input_value[0].INDEX_CONVERT##col_index,                \
      input_value[1].INDEX_CONVERT##col_index

#define ROTATE_SAVE_270_OUTPUT_RIGHTVALUE_3(T, col_index) \
  input_value[0].INDEX_CONVERT##col_index,                \
      input_value[1].INDEX_CONVERT##col_index,            \
      input_value[2].INDEX_CONVERT##col_index

#define ROTATE_SAVE_270_OUTPUT_RIGHTVALUE_4(T, col_index) \
  input_value[0].INDEX_CONVERT##col_index,                \
      input_value[1].INDEX_CONVERT##col_index,            \
      input_value[2].INDEX_CONVERT##col_index,            \
      input_value[3].INDEX_CONVERT##col_index

#define ROTATE_SAVE_270_OUTPUT_1(T, cols_remained, rows_remained) \
  T##rows_remained output_value;                                  \
  output_value = (T##rows_remained)(                              \
      ROTATE_SAVE_270_OUTPUT_RIGHTVALUE_##rows_remained(T, 1));   \
  vstore##rows_remained(output_value, 0, dst);                    \
  dst = (global T*)((uchar*)dst + dst_stride);

#define ROTATE_SAVE_270_OUTPUT_2(T, cols_remained, rows_remained) \
  T##rows_remained output_value;                                  \
  output_value = (T##rows_remained)(                              \
      ROTATE_SAVE_270_OUTPUT_RIGHTVALUE_##rows_remained(T, 2));   \
  vstore##rows_remained(output_value, 0, dst);                    \
  dst = (global T*)((uchar*)dst + dst_stride);                    \
                                                                  \
  output_value = (T##rows_remained)(                              \
      ROTATE_SAVE_270_OUTPUT_RIGHTVALUE_##rows_remained(T, 1));   \
  vstore##rows_remained(output_value, 0, dst);                    \
  dst = (global T*)((uchar*)dst + dst_stride);

#define ROTATE_SAVE_270_OUTPUT_3(T, cols_remained, rows_remained) \
  T##rows_remained output_value;                                  \
  output_value = (T##rows_remained)(                              \
      ROTATE_SAVE_270_OUTPUT_RIGHTVALUE_##rows_remained(T, 3));   \
  vstore##rows_remained(output_value, 0, dst);                    \
  dst = (global T*)((uchar*)dst + dst_stride);                    \
                                                                  \
  output_value = (T##rows_remained)(                              \
      ROTATE_SAVE_270_OUTPUT_RIGHTVALUE_##rows_remained(T, 2));   \
  vstore##rows_remained(output_value, 0, dst);                    \
  dst = (global T*)((uchar*)dst + dst_stride);                    \
                                                                  \
  output_value = (T##rows_remained)(                              \
      ROTATE_SAVE_270_OUTPUT_RIGHTVALUE_##rows_remained(T, 1));   \
  vstore##rows_remained(output_value, 0, dst);                    \
  dst = (global T*)((uchar*)dst + dst_stride);

#define ROTATE_SAVE_270_OUTPUT_4(T, cols_remained, rows_remained) \
  T##rows_remained output_value;                                  \
  output_value = (T##rows_remained)(                              \
      ROTATE_SAVE_270_OUTPUT_RIGHTVALUE_##rows_remained(T, 4));   \
  vstore##rows_remained(output_value, 0, dst);                    \
  dst = (global T*)((uchar*)dst + dst_stride);                    \
                                                                  \
  output_value = (T##rows_remained)(                              \
      ROTATE_SAVE_270_OUTPUT_RIGHTVALUE_##rows_remained(T, 3));   \
  vstore##rows_remained(output_value, 0, dst);                    \
  dst = (global T*)((uchar*)dst + dst_stride);                    \
                                                                  \
  output_value = (T##rows_remained)(                              \
      ROTATE_SAVE_270_OUTPUT_RIGHTVALUE_##rows_remained(T, 2));   \
  vstore##rows_remained(output_value, 0, dst);                    \
  dst = (global T*)((uchar*)dst + dst_stride);                    \
                                                                  \
  output_value = (T##rows_remained)(                              \
      ROTATE_SAVE_270_OUTPUT_RIGHTVALUE_##rows_remained(T, 1));   \
  vstore##rows_remained(output_value, 0, dst);                    \
  dst = (global T*)((uchar*)dst + dst_stride);

#define ROTATE_DST_270(T, rows_load, cols_load)                        \
  dst = (global T*)((uchar*)dst +                                      \
                    dst_stride * max(cols - index_x - cols_load, 0)) + \
        index_y;

// ==============================================================================

// #if defined(TOTATE90_U81C) || defined(RAOTATE90_F321C) || defined(ALL_KERNELS)
#define ROTATE_KERNEL_C1_TYPE(base_type, T, rows_load, cols_load, degree)     \
  __kernel void rotateC1##degree##base_type##Kernel(                          \
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
    int cols_remained = cols - index_x, rows_remained = rows - index_y;       \
                                                                              \
    T##cols_load input_value[rows_load];                                      \
    for (int i = 0; i < min(rows_remained, rows_load); i++) {                 \
      input_value[i] = vload##cols_load(element_x, src);                      \
      src = (global const T*)((uchar*)src + src_stride);                      \
    }                                                                         \
                                                                              \
    ROTATE_DST_##degree(T, rows_load, cols_load)                              \
        ROTATE_SAVE_IF_COL##cols_load(T, rows_load, cols_remained,            \
                                      rows_remained, cols_load, dst, degree)  \
  }
// #endif


// =================================================================

#define ROTATE_SAVE_90_OUTPUT_CN_1(T, channels) \
  vstore##channels(input_value[0], 0, dst);

#define ROTATE_SAVE_90_OUTPUT_CN_2(T, channels) \
  vstore##channels(input_value[1], 0, dst);     \
  dst += channels;                              \
  vstore##channels(input_value[0], 0, dst);

#define ROTATE_SAVE_90_OUTPUT_CN_3(T, channels) \
  vstore##channels(input_value[2], 0, dst);     \
  dst += channels;                              \
  vstore##channels(input_value[1], 0, dst);     \
  dst += channels;                              \
  vstore##channels(input_value[0], 0, dst);

#define ROTATE_SAVE_90_OUTPUT_CN_4(T, channels) \
  vstore##channels(input_value[3], 0, dst);     \
  dst += channels;                              \
  vstore##channels(input_value[2], 0, dst);     \
  dst += channels;                              \
  vstore##channels(input_value[1], 0, dst);     \
  dst += channels;                              \
  vstore##channels(input_value[0], 0, dst);

#define ROTATE_DST_CN_90(T, rows_load, cols_load, channels) \
  dst = (global T*)((uchar*)dst + dst_stride * index_x) +   \
        max(rows - index_y - rows_load, 0) * channels;

// =================================================================

#define ROTATE_SAVE_180_OUTPUT_CN_1(T, channels) \
  vstore##channels(input_value[0], 0, dst);

#define ROTATE_SAVE_180_OUTPUT_CN_2(T, channels) \
  vstore##channels(input_value[1], 0, dst);     \
  dst += dst_stride;                              \
  vstore##channels(input_value[0], 0, dst);

#define ROTATE_SAVE_180_OUTPUT_CN_3(T, channels) \
  vstore##channels(input_value[2], 0, dst);     \
  dst += dst_stride;                              \
  vstore##channels(input_value[1], 0, dst);     \
  dst += dst_stride;                              \
  vstore##channels(input_value[0], 0, dst);

#define ROTATE_SAVE_180_OUTPUT_CN_4(T, channels) \
  vstore##channels(input_value[3], 0, dst);     \
  dst += dst_stride;                              \
  vstore##channels(input_value[2], 0, dst);     \
  dst += dst_stride;                              \
  vstore##channels(input_value[1], 0, dst);     \
  dst += dst_stride;                              \
  vstore##channels(input_value[0], 0, dst);

#define ROTATE_DST_CN_180(T, rows_load, cols_load, channels) \
  dst = (global T*)((uchar*)dst + dst_stride * max(rows - index_y - rows_load, 0)) +   \
        max(cols - index_x - cols_load, 0) * channels;


// =================================================================

#define ROTATE_SAVE_270_OUTPUT_CN_1(T, channels) \
  vstore##channels(input_value[0], 0, dst);

#define ROTATE_SAVE_270_OUTPUT_CN_2(T, channels) \
  ROTATE_SAVE_270_OUTPUT_CN_1(T, channels)       \
  dst += channels;                               \
  vstore##channels(input_value[1], 0, dst);

#define ROTATE_SAVE_270_OUTPUT_CN_3(T, channels) \
  ROTATE_SAVE_270_OUTPUT_CN_2(T, channels)       \
  dst += channels;                               \
  vstore##channels(input_value[2], 0, dst);

#define ROTATE_SAVE_270_OUTPUT_CN_4(T, channels) \
  ROTATE_SAVE_270_OUTPUT_CN_3(T, channels)       \
  dst += channels;                               \
  vstore##channels(input_value[3], 0, dst);

#define ROTATE_DST_CN_270(T, rows_load, cols_load, channels)           \
  dst = (global T*)((uchar*)dst +                                      \
                    dst_stride * max(cols - index_x - cols_load, 0)) + \
        index_y * channels;

#define ROTATE_SAVE_ROW_CN_IF_1(T, rows_load, channels, degree) \
  if (rows_remained >= rows_load) {                             \
    ROTATE_SAVE_##degree##_OUTPUT_CN_##rows_load(T, channels)   \
  }

#define ROTATE_SAVE_ROW_CN_IF_2(T, rows_load, channels, degree) \
  ROTATE_SAVE_ROW_CN_IF_1(T, rows_load, channels, degree)       \
  if (rows_remained == 1) {                                     \
    ROTATE_SAVE_##degree##_OUTPUT_CN_##1(T, channels)           \
  }

#define ROTATE_SAVE_ROW_CN_IF_3(T, rows_load, channels, degree) \
  ROTATE_SAVE_ROW_CN_IF_2(T, rows_load, channels, degree)       \
  if (rows_remained == 2) {                                     \
    ROTATE_SAVE_##degree##_OUTPUT_CN_##2(T, channels)           \
  }

#define ROTATE_SAVE_ROW_CN_IF_4(T, rows_load, channels, degree) \
  ROTATE_SAVE_ROW_CN_IF_3(T, rows_load, channels, degree)       \
  if (rows_remained == 3) {                                     \
    ROTATE_SAVE_##degree##_OUTPUT_CN_##3(T, channels)           \
  }

// =================================================================

#define ROTATE_KERNEL_CN_TYPE(base_type, T, channels, rows_load, degree)      \
  __kernel void rotateC##channels##degree##base_type##Kernel(                 \
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
    int rows_remained = rows - index_y;                                       \
                                                                              \
    T##channels input_value[rows_load];                                       \
    for (int i = 0; i < min(rows_remained, rows_load); i++) {                 \
      input_value[i] = vload##channels(element_x, src);                       \
      src = (global const T*)((uchar*)src + src_stride);                      \
    }                                                                         \
                                                                              \
    ROTATE_DST_CN_##degree(T, rows_load, 1, channels)                         \
        ROTATE_SAVE_ROW_CN_IF_##rows_load(T, rows_load, channels, degree)     \
  }

ROTATE_KERNEL_C1_TYPE(F32, float, 2, 2, 90)
ROTATE_KERNEL_C1_TYPE(U8, uchar, 4, 4, 90)

ROTATE_KERNEL_C1_TYPE(U8, uchar, 4, 4, 180)
ROTATE_KERNEL_C1_TYPE(F32, float, 2, 2, 180)

ROTATE_KERNEL_C1_TYPE(F32, float, 2, 2, 270)
ROTATE_KERNEL_C1_TYPE(U8, uchar, 4, 4, 270)

ROTATE_KERNEL_CN_TYPE(U8, uchar, 3, 4, 90)
ROTATE_KERNEL_CN_TYPE(F32, float, 3, 1, 90)

ROTATE_KERNEL_CN_TYPE(U8, uchar, 4, 4, 90)
ROTATE_KERNEL_CN_TYPE(F32, float, 4, 1, 90)

ROTATE_KERNEL_CN_TYPE(U8, uchar, 3, 4, 180)
ROTATE_KERNEL_CN_TYPE(F32, float, 3, 1, 180)

ROTATE_KERNEL_CN_TYPE(U8, uchar, 4, 4, 180)
ROTATE_KERNEL_CN_TYPE(F32, float, 4, 1, 180)

ROTATE_KERNEL_CN_TYPE(U8, uchar, 3, 4, 270)
ROTATE_KERNEL_CN_TYPE(F32, float, 3, 1, 270)

ROTATE_KERNEL_CN_TYPE(U8, uchar, 4, 4, 270)
ROTATE_KERNEL_CN_TYPE(F32, float, 4, 1, 270)
