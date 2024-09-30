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

#define TRANSPOSE_VEC1(src_array, i)                                                         \
  src_array[0].INDEX_CONVERT##i
#define TRANSPOSE_VEC2(src_array, i)                                                         \
  TRANSPOSE_VEC1(src_array, i), src_array[1].INDEX_CONVERT##i
#define TRANSPOSE_VEC3(src_array, i)                                                         \
  TRANSPOSE_VEC2(src_array, i), src_array[2].INDEX_CONVERT##i
#define TRANSPOSE_VEC4(src_array, i)                                                         \
  TRANSPOSE_VEC3(src_array, i), src_array[3].INDEX_CONVERT##i

#define vstore1 vstore
#define uchar1 uchar
#define float1 float
#define int1 int

#define TRANSPOSE_ELEMENT1(Tsrc, Tdst, src_array, output, j) \
  output[0] = (Tdst##j)(TRANSPOSE_VEC##j(src_array, 1));
#define TRANSPOSE_ELEMENT2(Tsrc, Tdst, src_array, output, j) \
  TRANSPOSE_ELEMENT1(Tsrc, Tdst, src_array, output, j)       \
  output[1] = (Tdst##j)(TRANSPOSE_VEC##j(src_array, 2));
#define TRANSPOSE_ELEMENT3(Tsrc, Tdst, src_array, output, j) \
  TRANSPOSE_ELEMENT2(Tsrc, Tdst, src_array, output, j)       \
  output[2] = (Tdst##j)(TRANSPOSE_VEC##j(src_array, 3));
#define TRANSPOSE_ELEMENT4(Tsrc, Tdst, src_array, output, j) \
  TRANSPOSE_ELEMENT3(Tsrc, Tdst, src_array, output, j)       \
  output[3] = (Tdst##j)(TRANSPOSE_VEC##j(src_array, 4));

#define VSTORE_REMAIN_ROWS1(input, rows_load)     \
  int offset = element_y * rows_load;             \
  dst_tmp[offset] = input;

#define VSTORE_REMAIN_ROWS1_VEC(input, rows_load) \
  int offset = element_y * rows_load;             \
  dst_tmp[offset] = input.INDEX_CONVERT##1;

#define VSTORE_REMAIN_ROWS2(input, rows_load)     \
  VSTORE_REMAIN_ROWS1_VEC(input, rows_load)       \
  dst_tmp[offset + 1] = input.INDEX_CONVERT##2;

#define VSTORE_REMAIN_ROWS3(input, rows_load)     \
  VSTORE_REMAIN_ROWS2(input, rows_load)           \
  dst_tmp[offset + 2] = input.INDEX_CONVERT##3;

#define TRANSPOSE_AND_SAVE_VEC(Tsrc, Tdst, src_array, output, rows_load, i, j) \
    {                                                                          \
        Tdst##j output_value[i];                                               \
        TRANSPOSE_ELEMENT##i(Tsrc, Tdst, src_array, output, j)                 \
        for (int k = 0; k < i; k++)                                            \
        {                                                                      \
            vstore##j(output[k], element_y, dst_tmp);                          \
            dst_tmp = (global Tdst*)((uchar*)dst_tmp + dst_stride);            \
        }                                                                      \
    }

#define TRANSPOSE_AND_SAVE_SCALE(Tsrc, Tdst, src_array, output, rows_load, i, j)           \
    {                                                                                      \
        Tdst##j output_value[i];                                                           \
        TRANSPOSE_ELEMENT##i(Tsrc, Tdst, src_array, output, j) for (int k = 0; k < i; k++) \
        {                                                                                  \
            VSTORE_REMAIN_ROWS##j(output[k], rows_load)                                    \
            dst_tmp = (global Tdst*)((uchar*)dst_tmp + dst_stride);                        \
        }                                                                                  \
    }

#define TRANSPOSE_RAMAIN_ROW1(Tsrc, Tdst, cols_load, rows_load, rows_load_global, save_name) \
    if (remain_cols >= cols_load)                                                            \
    TRANSPOSE_AND_SAVE_##save_name(Tsrc, Tdst, input_value, output_value, rows_load_global,  \
                                   cols_load, rows_load)

#define TRANSPOSE_RAMAIN_ROW2(Tsrc, Tdst, cols_load, rows_load, rows_load_global, save_name) \
    TRANSPOSE_RAMAIN_ROW1(Tsrc, Tdst, cols_load, rows_load, rows_load_global, save_name)     \
    else if (remain_cols == 1) TRANSPOSE_AND_SAVE_##save_name(                               \
        Tsrc, Tdst, input_value, output_value, rows_load_global, 1, rows_load)

#define TRANSPOSE_RAMAIN_ROW3(Tsrc, Tdst, cols_load, rows_load, rows_load_global, save_name) \
    TRANSPOSE_RAMAIN_ROW2(Tsrc, Tdst, cols_load, rows_load, rows_load_global, save_name)     \
    else if (remain_cols == 2) TRANSPOSE_AND_SAVE_##save_name(                               \
        Tsrc, Tdst, input_value, output_value, rows_load_global, 2, rows_load)

#define TRANSPOSE_RAMAIN_ROW4(Tsrc, Tdst, cols_load, rows_load, rows_load_global, save_name) \
    TRANSPOSE_RAMAIN_ROW3(Tsrc, Tdst, cols_load, rows_load, rows_load_global, save_name)     \
    else if (remain_cols == 3) TRANSPOSE_AND_SAVE_##save_name(                               \
        Tsrc, Tdst, input_value, output_value, rows_load_global, 3, rows_load)

#define TRANSPOSE_RAMAIN_COL1(Tsrc, Tdst, cols_load, rows_load)                              \
    if (remain_rows >= rows_load) {                                                          \
        TRANSPOSE_RAMAIN_ROW##cols_load(Tsrc, Tdst, cols_load, rows_load, rows_load, VEC)    \
    }

#define TRANSPOSE_RAMAIN_COL2(Tsrc, Tdst, cols_load, rows_load)                              \
  TRANSPOSE_RAMAIN_COL1(Tsrc, Tdst, cols_load, rows_load)                                    \
  else if (remain_rows == 1) {                                                               \
    TRANSPOSE_RAMAIN_ROW##cols_load(Tsrc, Tdst, cols_load, 1, rows_load, SCALE)              \
  }

#define TRANSPOSE_RAMAIN_COL3(Tsrc, Tdst, cols_load, rows_load)                              \
  TRANSPOSE_RAMAIN_COL2(Tsrc, Tdst, cols_load, rows_load)                                    \
  else if (remain_rows == 2) {                                                               \
    TRANSPOSE_RAMAIN_ROW##cols_load(Tsrc, Tdst, cols_load, 2, rows_load, SCALE)              \
  }

#define TRANSPOSE_RAMAIN_COL4(Tsrc, Tdst, cols_load, rows_load)                              \
  TRANSPOSE_RAMAIN_COL3(Tsrc, Tdst, cols_load, rows_load)                                    \
  else if (remain_rows == 3) {                                                               \
    TRANSPOSE_RAMAIN_ROW##cols_load(Tsrc, Tdst, cols_load, 3, rows_load, SCALE)              \
  }

#define VECTOR_SUM1(input_sum, input_value, load_size, Tdst) \
  ((Tdst*)input_sum+j)[0] += ((Tdst##load_size*)input_value+j)[0].x;

#define VECTOR_SUM2(input_sum, input_value, load_size, Tdst) \
  VECTOR_SUM1(input_sum, input_value, load_size, Tdst)       \
  ((Tdst*)input_sum+j)[0] += ((Tdst##load_size*)input_value+j)[0].y;

#define VECTOR_SUM3(input_sum, input_value, load_size, Tdst) \
  VECTOR_SUM2(input_sum, input_value, load_size, Tdst)       \
  ((Tdst*)input_sum+j)[0] += ((Tdst##load_size*)input_value+j)[0].z;

#define VECTOR_SUM4(input_sum, input_value, load_size, Tdst) \
  VECTOR_SUM3(input_sum, input_value, load_size, Tdst)       \
  ((Tdst*)input_sum+j)[0] += ((Tdst##load_size*)input_value+j)[0].w;

#define CUM_SUM1(input_sum, input_value, load_size, Tdst)    \
  ((Tdst##load_size*)input_value+j)[0].x += ((Tdst*)input_sum+j)[0];

#define CUM_SUM2(input_sum, input_value, load_size, Tdst)    \
  CUM_SUM1(input_sum, input_value, load_size, Tdst)          \
  ((Tdst##load_size*)input_value+j)[0].y += ((Tdst##load_size*)input_value+j)[0].x;

#define CUM_SUM3(input_sum, input_value, load_size, Tdst)    \
  CUM_SUM2(input_sum, input_value, load_size, Tdst)          \
  ((Tdst##load_size*)input_value+j)[0].z += ((Tdst##load_size*)input_value+j)[0].y;

#define CUM_SUM4(input_sum, input_value, load_size, Tdst)    \
  CUM_SUM3(input_sum, input_value, load_size, Tdst)          \
  ((Tdst##load_size*)input_value+j)[0].w += ((Tdst##load_size*)input_value+j)[0].z;

__kernel void setZeroF32(global float* dst, int cols){
  int index_x = get_global_id(0);
  index_x <<= 1;
  if (index_x >= cols){
    return;
  }
  if (cols - index_x >= 2){
    vstore2((float2)(0.0f, 0.0f), get_global_id(0), dst);
  } else {
    dst[index_x] = 0;
  }
}

__kernel void setZeroI32(global int* dst, int cols){
  int index_x = get_global_id(0);
  index_x <<= 1;
  if (index_x >= cols){
    return;
  }
  if (cols - index_x >= 2){
    vstore2((int2)(0.0f, 0.0f), get_global_id(0), dst);
  } else {
    dst[index_x] = 0;
  }
}


#define INTEGRAL_VERTICAL_KERNEL(src_base_type, Tsrc, dst_base_type, Tdst, load_size)   \
__kernel void integral##src_base_type##dst_base_type##Kernel(                           \
                    global const Tsrc* src, int src_rows, int src_cols, int src_stride, \
                    global Tdst* dst, int dst_rows, int dst_cols, int dst_stride) {     \
    int element_y = get_group_id(0);                                                    \
    int local_x = get_local_id(0);                                                      \
    int local_size = get_local_size(0);                                                 \
    int index_x = local_x * load_size, index_y = element_y * load_size;                 \
    if (index_x >= src_cols || index_y >= src_rows) {                                   \
      return;                                                                           \
    }                                                                                   \
    int dst_offset;                                                                     \
                                                                                        \
    if (src_rows == dst_cols){                                                          \
      dst_offset = 0;                                                                   \
    }                                                                                   \
    else {                                                                              \
      dst_offset = 1;                                                                   \
    }                                                                                   \
                                                                                        \
    global const Tsrc* src_tmp;                                                         \
    int remain_cols = src_cols - index_x, remain_rows = src_rows - index_y;             \
    Tdst##load_size input_value[load_size];                                             \
    global Tdst* dst_tmp = dst + dst_offset;                                            \
    dst_tmp = (global Tdst*)((uchar*)dst_tmp + dst_stride * (index_x + dst_offset));    \
    __local Tdst prev_sum[load_size];                                                   \
    if (get_global_id(0) == 0){                                                         \
      for (int i = 0; i < min(remain_rows, load_size); i++){                            \
        dst_tmp[0] = 0;                                                                 \
        dst_tmp = (global Tdst*)((uchar*)dst_tmp + dst_stride);                         \
      }                                                                                 \
    }                                                                                   \
    if (local_x == local_size - 1 || index_x + load_size >= src_cols){                  \
      for (int i = 0; i < load_size; i++) {                                             \
        prev_sum[i] = 0;                                                                \
      }                                                                                 \
    }                                                                                   \
    barrier(CLK_LOCAL_MEM_FENCE);                                                       \
                                                                                        \
    src = (global const Tsrc*)((uchar*)src + src_stride * index_y);                     \
    Tdst input_sum[load_size] = {0};                                                    \
    int x_offset = local_size * load_size;                                              \
    dst_tmp = dst + dst_offset;                                                         \
    dst_tmp = (global Tdst*)((uchar*)dst_tmp + dst_stride * (index_x + dst_offset));    \
    global Tdst* dst_tmp_prev;                                                          \
    while (remain_cols > 0){                                                            \
      barrier(CLK_LOCAL_MEM_FENCE);                                                     \
      for (int i = 0; i < load_size; i++) {                                             \
        input_sum[i] = prev_sum[i];                                                     \
      }                                                                                 \
  {                                                                                 \
    int i;                                                                          \
    for (i = 0; i <= index_x - 1; i = i + load_size) {                              \
      src_tmp = src + i;                                                            \
      for (int j = 0; j < min(remain_rows, load_size); j++) {                       \
        ((Tdst##load_size*)input_value+j)[0] = convert_##Tdst##load_size(vload##load_size(0, src_tmp));              \
        src_tmp = (global const Tsrc*)((uchar*)src_tmp + src_stride);               \
        VECTOR_SUM##load_size(input_sum, input_value, load_size, Tdst)                               \
      }                                                                             \
    }                                                                               \
    i = index_x;                                                                    \
    src_tmp = src + i;                                                              \
      for (int j = 0; j < min(remain_rows, load_size); j++) {                       \
        ((Tdst##load_size*)input_value+j)[0] = convert_##Tdst##load_size(vload##load_size(0, src_tmp));              \
        src_tmp = (global const Tsrc*)((uchar*)src_tmp + src_stride);               \
        CUM_SUM##load_size(input_sum, input_value, load_size, Tdst)                                  \
      }                                                                             \
  }                                                                                 \
                                                                                        \
      dst_tmp_prev = dst_tmp;                                                           \
      TRANSPOSE_RAMAIN_COL##load_size(Tsrc, Tdst, load_size, load_size)                 \
      dst_tmp = (global Tdst*)((uchar*)dst_tmp_prev + dst_stride * (x_offset));         \
      src += x_offset;                                                                  \
      remain_cols = remain_cols - x_offset;                                             \
                                                                                        \
      if (local_x == local_size - 1 || index_x + load_size >= src_cols){                \
        for (int i = 0; i < min(remain_rows, load_size); i++) {                         \
          prev_sum[i] = input_value[i].INDEX_CONVERT##load_size;                        \
        }                                                                               \
      }                                                                                 \
    }                                                                                   \
  }


INTEGRAL_VERTICAL_KERNEL(U8, uchar, I32, int, 2)
INTEGRAL_VERTICAL_KERNEL(I32, int, I32, int, 2)

INTEGRAL_VERTICAL_KERNEL(F32, float, F32, float, 2)
