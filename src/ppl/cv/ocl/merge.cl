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

#if defined(MERGE3_U81D) || defined(MERGE3_F321D) || defined(ALL_KERNELS)
#define MERGE3KERNEL_TYPE0(base_type, T)                                \
  __kernel void merge3##base_type##Kernel0(                             \
      global const T* src0, global const T* src1, global const T* src2, \
      int cols, global T* dst) {                                    \
    int element_x = get_global_id(0);                                   \
    if (element_x >= cols) {                                            \
      return;                                                           \
    }                                                                   \
                                                                        \
    T input_value0 = src0[element_x];                                   \
    T input_value1 = src1[element_x];                                   \
    T input_value2 = src2[element_x];                                   \
    T##3 output_value = (T##3)(input_value0, input_value1, input_value2);     \
    vstore3(output_value, element_x, dst);                              \
  }
#endif

#if defined(MERGE3_U82D) || defined(MERGE3_F322D) || defined(ALL_KERNELS)
#define MERGE3KERNEL_TYPE1(base_type, T)                                       \
  __kernel void merge3##base_type##Kernel1(                                    \
      global const T* src0, global const T* src1, global const T* src2,        \
      int rows, int cols, int src_stride, global T* dst, int dst_stride) { \
    int element_x = get_global_id(0);                                          \
    int element_y = get_global_id(1);                                          \
    int offset = element_y * src_stride;                                       \
    if (element_x >= cols || element_y >= rows) {                              \
      return;                                                                  \
    }                                                                          \
                                                                               \
    src0 = (global const T*)((uchar*)src0 + offset);                                        \
    src1 = (global const T*)((uchar*)src1 + offset);                                        \
    src2 = (global const T*)((uchar*)src2 + offset);                                        \
    dst = (global T*)((uchar*)dst + element_y * dst_stride);                          \
    T input_value0 = src0[element_x];                                          \
    T input_value1 = src1[element_x];                                          \
    T input_value2 = src2[element_x];                                          \
    T##3 output_value = (T##3)(input_value0, input_value1, input_value2);            \
    vstore3(output_value, element_x, dst);                                     \
  }
#endif

#if defined(MERGE4_U81D) || defined(MERGE4_F321D) || defined(ALL_KERNELS)
#define MERGE4KERNEL_TYPE0(base_type, T)                                \
  __kernel void merge4##base_type##Kernel0(                             \
      global const T* src0, global const T* src1, global const T* src2, \
      global const T* src3, int cols, global T* dst) {              \
    int element_x = get_global_id(0);                                   \
    int index_x = element_x;                                            \
    if (index_x >= cols) {                                              \
      return;                                                           \
    }                                                                   \
                                                                        \
    T input_value0 = src0[index_x];                                     \
    T input_value1 = src1[index_x];                                     \
    T input_value2 = src2[index_x];                                     \
    T input_value3 = src3[index_x];                                     \
    T##4 output_value = (T##4)(input_value0, input_value1, input_value2,      \
                         input_value3);                                 \
    vstore4(output_value, element_x, dst);                              \
  }
#endif

#if defined(MERGE4_U82D) || defined(MERGE4_F322D) || defined(ALL_KERNELS)
#define MERGE4KERNEL_TYPE1(base_type, T)                                \
  __kernel void merge4##base_type##Kernel1(                             \
      global const T* src0, global const T* src1, global const T* src2, \
      global const T* src3, int rows, int cols, int src_stride,         \
      global T* dst, int dst_stride) {                              \
    int element_x = get_global_id(0);                                   \
    int element_y = get_global_id(1);                                   \
    int offset = element_y * src_stride;                                \
    if (element_x >= cols || element_y >= rows) {                       \
      return;                                                           \
    }                                                                   \
                                                                        \
    src0 = (global const T*)((uchar*)src0 + offset);                                 \
    src1 = (global const T*)((uchar*)src1 + offset);                                 \
    src2 = (global const T*)((uchar*)src2 + offset);                                 \
    src3 = (global const T*)((uchar*)src3 + offset);                                 \
    dst = (global T*)((uchar*)dst + element_y * dst_stride);                   \
    T input_value0 = src0[element_x];                                   \
    T input_value1 = src1[element_x];                                   \
    T input_value2 = src2[element_x];                                   \
    T input_value3 = src3[element_x];                                   \
    T##4 output_value = (T##4)(input_value0, input_value1, input_value2,      \
                         input_value3);                                 \
    vstore4(output_value, element_x, dst);                              \
  }
#endif

#if defined(MERGE3_U81D) || defined(ALL_KERNELS)
MERGE3KERNEL_TYPE0(U8, uchar)
#endif

#if defined(MERGE3_F321D) || defined(ALL_KERNELS)
MERGE3KERNEL_TYPE0(F32, float)
#endif

#if defined(MERGE3_U82D) || defined(ALL_KERNELS)
MERGE3KERNEL_TYPE1(U8, uchar)
#endif

#if defined(MERGE3_F322D) || defined(ALL_KERNELS)
MERGE3KERNEL_TYPE1(F32, float)
#endif

#if defined(MERGE4_U81D) || defined(ALL_KERNELS)
MERGE4KERNEL_TYPE0(U8, uchar)
#endif

#if defined(MERGE4_F321D) || defined(ALL_KERNELS)
MERGE4KERNEL_TYPE0(F32, float)
#endif

#if defined(MERGE4_U82D) || defined(ALL_KERNELS)
MERGE4KERNEL_TYPE1(U8, uchar)
#endif

#if defined(MERGE4_F322D) || defined(ALL_KERNELS)
MERGE4KERNEL_TYPE1(F32, float)
#endif