#ifndef CUDA_MATH_CUH
#define CUDA_MATH_CUH
/* This file is for experimental result reproduction, do not use it in any real program.
 * This file is to generate the comparision between provided by cuda 10 and our fast_math library
 * Purpose: the code we converted will call "fast_h2exp" but actually fast_h2exp is the h2exp provided by cuda, so no need to change the code when
 * measuring the speed & error of our fast_math library and CUDA math calls. Just replace this header file in the place of "fast_math.cuh"
 * then measure the new error + speed for comparison.
*/
#include <stdint.h>
#include <cuda_fp16.h>
#include "half2_operator_overload.cuh"
//important macroes in cuda_fp16.hpp for corectness (without my messy type & pointer casting
/*
#define __HALF_TO_US(var) *(reinterpret_cast<unsigned short *>(&(var)))
#define __HALF_TO_CUS(var) *(reinterpret_cast<const unsigned short *>(&(var)))
#define __HALF_TO_VUS(var) *(reinterpret_cast<volatile unsigned short *>(&(var)))
#define __HALF_TO_CVUS(var) *(reinterpret_cast<const volatile unsigned short *>(&(var)))
#define __HALF2_TO_UI(var) *(reinterpret_cast<unsigned int *>(&(var)))
#define __HALF2_TO_CUI(var) *(reinterpret_cast<const unsigned int *>(&(var)))
*/
//79 cycles half2 division on Nvidia V100
__device__ half2 fast_h2rcp(half2 input){
  return h2rcp(input);
}

__device__ half2 exp_half2_saturated(half2 input) {
    return h2exp(input);
}

__device__ half2 fast_h2exp(half2 input){
/*
    half2 result = __float2half2_rn(1477)*input;
   result.x += __float2half_rn(15293.36);
   result.y += __float2half_rn(15293.36);
   short2 result_short2;
    result_short2.x = (short)(result.x);
    result_short2.y = (short)(result.y);
    return *(half2*)(&result_short2);
*/
    return h2exp(input);
}

//100 cycles
__device__ half2 fast_h2log2(half2 input){
	return h2log2(input);
}

//50 cycles, no cvt (type conversion instructions)
__device__  half2 fastest_h2log2(half2 x){
  return h2log2(x);
}

__device__ half2 fast_h2log(half2 input){
    //0.6931 = ln(2)
    return h2log(input);
}

__device__ half2 fast_h2log10(half2 input){

    return h2log10(input);
}



__device__ half2 fast_h2rsqrt( half2 input) {
	return h2rsqrt(input);
}

// magic number 0x1DE9
__device__ half2 fast_h2sqrt(half2 input){
	return h2sqrt(input);

}

__device__ half2 sin_poly_half2(half2 x){
  half2 coeff1 = __float2half2_rn(-0.166667);
  half2 coeff2 = __float2half2_rn(0.008333);
  half2 coeff3 = __float2half2_rn(-0.00019841269);
  return x +  x*x*x*coeff1 + x*x*x*x*x*coeff2 + x*x*x*x*x*x*x*coeff3 ;
}

__device__ half2 cos_poly_half2(half2 x){
  half2 coeff1 = __float2half2_rn(-0.5);
  half2 coeff2 = __float2half2_rn(0.041666667);
  half2 coeff3 = __float2half2_rn(-0.0013888889);
  return 1 +  x*x*coeff1 + x*x*x*x*coeff2 + x*x*x*x*x*x*coeff3 ;
}


__device__ half2 fast_h2sin(half2 x){

	return h2sin(x);
}

__device__ half2 fast_h2cos(half2 x){
	return h2cos(x);
}

typedef struct __device_builtin__ half2_2
{
    half2 x, y;
} half2_2 ;

typedef struct __device_builtin__ half2_3
{
    half2 x, y,z;
} half2_3 ;

typedef struct __device_builtin__ half2_4
{
    half2 x, y, z, w;
} half2_4 ;

__device__ half2 fast_h2asin(half2 x){

	float2 input = __half22float2(x);
	float2 result;
	result.x = asinf(input.x);
	result.y = asinf(input.y);
	return __float22half2_rn(result);

}

__device__ half2 fast_h2acos(half2 input){

  float2 input_float = __half22float2(input);
	float2 result;
	result.x = acosf(input.x);
	result.y = acosf(input.y);
	return __float22half2_rn(result);
}


#endif
