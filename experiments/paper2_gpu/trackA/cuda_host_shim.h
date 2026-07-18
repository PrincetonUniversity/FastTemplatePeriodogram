/* cuda_host_shim.h -- minimal CUDA-C++ shim so the RawKernel sources in
 * kernels.py can be SYNTAX-checked with a host C++ compiler on a machine
 * with no CUDA toolkit (validate_local.py runs `clang++ -fsyntax-only
 * -include cuda_host_shim.h` over the dumped source). Never used on the
 * pod; catches typos before GPU time is spent. NOT a semantic emulation.
 */
#pragma once
#include <cstddef>
#include <cmath>

#define __global__
#define __device__
#define __forceinline__ inline
#define __shared__
#ifndef __restrict__
#define __restrict__
#endif

struct ftp_dim3 { unsigned int x, y, z; };
extern ftp_dim3 threadIdx, blockIdx, blockDim, gridDim;

struct double2 { double x, y; };
inline double2 make_double2(double x, double y)
{ double2 z; z.x = x; z.y = y; return z; }

inline void __syncthreads() {}
inline void sincos(double a, double* s, double* c)
{ *s = ::sin(a); *c = ::cos(a); }
inline double __longlong_as_double(long long v)
{ (void)v; return 0.0; }
