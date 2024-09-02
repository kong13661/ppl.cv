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

#if defined(SPLIT3_U81D) || defined(SPLIT3_F321D) || defined(ALL_KERNELS)
#define SPLIT3KERNEL_TYPE0(base_type, T)                                   \
  __kernel void split3##base_type##Kernel0(global const T* src, int cols,  \
                                           global T* dst0, global T* dst1, \
                                           global T* dst2) {               \
    int element_x = get_global_id(0);                                      \
    if (element_x >= cols) {                                           \
      return;                                                              \
    }                                                                      \
                                                                           \
    T##3 input_value;                                                      \
    input_value = vload3(element_x, src);                                  \
    dst0[element_x] = input_value.x;                                       \
    dst1[element_x] = input_value.y;                                       \
    dst2[element_x] = input_value.z;                                       \
  }
#endif

#if defined(SPLIT3_U82D) || defined(SPLIT3_F322D) || defined(ALL_KERNELS)
#define SPLIT3KERNEL_TYPE1(base_type, T)                                       \
  __kernel void split3##base_type##Kernel1(                                    \
      global const T* src, int rows, int cols, int src_stride, global T* dst0, \
      global T* dst1, global T* dst2, int dst_stride) {                        \
    int element_x = get_global_id(0);                                          \
    int element_y = get_global_id(1);                                          \
    if (element_x >= cols || element_y >= rows) {                          \
      return;                                                                  \
    }                                                                          \
    T##3 input_value;                                                          \
    input_value =                                                              \
        vload3(element_x, (global T*)((uchar*)src + element_y * src_stride));  \
    int offset = element_y * dst_stride;                                       \
    dst0 = (global T*)((uchar*)dst0 + offset);                                 \
    dst1 = (global T*)((uchar*)dst1 + offset);                                 \
    dst2 = (global T*)((uchar*)dst2 + offset);                                 \
    dst0[element_x] = input_value.x;                                           \
    dst1[element_x] = input_value.y;                                           \
    dst2[element_x] = input_value.z;                                           \
  }
#endif

#if defined(SPLIT4_U81D) || defined(SPLIT4_F321D) || defined(ALL_KERNELS)
#define SPLIT4KERNEL_TYPE0(base_type, T)                                     \
  __kernel void split4##base_type##Kernel0(global const T* src, int cols,    \
                                           global T* dst0, global T* dst1,   \
                                           global T* dst2, global T* dst3) { \
    int element_x = get_global_id(0);                                        \
    if (element_x >= cols) {                                            \
      return;                                                                \
    }                                                                        \
    T##4 input_value;                                                        \
    input_value = vload4(element_x, src);                                    \
    dst0[element_x] = input_value.x;                                         \
    dst1[element_x] = input_value.y;                                         \
    dst2[element_x] = input_value.z;                                         \
    dst3[element_x] = input_value.w;                                         \
  }
#endif

#if defined(SPLIT4_U82D) || defined(SPLIT4_F322D) || defined(ALL_KERNELS)
#define SPLIT4KERNEL_TYPE1(base_type, T)                                       \
  __kernel void split4##base_type##Kernel1(                                    \
      global const T* src, int rows, int cols, int src_stride, global T* dst0, \
      global T* dst1, global T* dst2, global T* dst3, int dst_stride) {        \
    int element_x = get_global_id(0);                                          \
    int element_y = get_global_id(1);                                          \
    if (element_x >= cols || element_y >= rows) {                         \
      return;                                                                  \
    }                                                                          \
                                                                               \
    T##4 input_value;                                                          \
    input_value =                                                              \
        vload4(element_x, (global T*)((uchar*)src + element_y * src_stride));  \
    int offset = element_y * dst_stride;                                       \
    dst0 = (global T*)((uchar*)dst0 + offset);                                 \
    dst1 = (global T*)((uchar*)dst1 + offset);                                 \
    dst2 = (global T*)((uchar*)dst2 + offset);                                 \
    dst3 = (global T*)((uchar*)dst3 + offset);                                 \
    dst0[element_x] = input_value.x;                                           \
    dst1[element_x] = input_value.y;                                           \
    dst2[element_x] = input_value.z;                                           \
    dst3[element_x] = input_value.w;                                           \
  }
#endif

#if defined(SPLIT3_U81D) || defined(ALL_KERNELS)
SPLIT3KERNEL_TYPE0(U8, uchar)
#endif

#if defined(SPLIT3_F321D) || defined(ALL_KERNELS)
SPLIT3KERNEL_TYPE0(F32, float)
#endif

#if defined(SPLIT3_U82D) || defined(ALL_KERNELS)
SPLIT3KERNEL_TYPE1(U8, uchar)
#endif

#if defined(SPLIT3_F322D) || defined(ALL_KERNELS)
SPLIT3KERNEL_TYPE1(F32, float)
#endif

#if defined(SPLIT4_U81D) || defined(ALL_KERNELS)
SPLIT4KERNEL_TYPE0(U8, uchar)
#endif

#if defined(SPLIT4_F321D) || defined(ALL_KERNELS)
SPLIT4KERNEL_TYPE0(F32, float)
#endif

#if defined(SPLIT4_U82D) || defined(ALL_KERNELS)
SPLIT4KERNEL_TYPE1(U8, uchar)
#endif

#if defined(SPLIT4_F322D) || defined(ALL_KERNELS)
SPLIT4KERNEL_TYPE1(F32, float)
#endif