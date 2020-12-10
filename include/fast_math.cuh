#ifndef FAST_MATH_CUH
#define FAST_MATH_CUH
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
  half2 c0 = __float2half2_rn(2.823529);
  half2 c1 = __float2half2_rn(-1.8823529);
  half2 four = __float2half2_rn(4.0);
  half2 one = __float2half2_rn(1.0);

  half2 temp_2 = input*four;
  int c = *((int*) (&temp_2));
  c= (c ^ 0x7c007c00 ) & 0xfc00fc00 ;
  half2 factor = (*(half2*)(&c));

  half2 d = input * factor;

  half2 x =  d*c1;
  x.x = __hadd(x.x, c0.x);
  x.y = __hadd(x.y, c0.y);
  x = x + x * one - x * x * d;

  return x*factor;
}

__device__ half2 exp_half2_saturated(half2 input) {
     half2 result = __float2half2_rn(1477)*input;
    result.x += __float2half_rn(15293.36);
   result.y += __float2half_rn(15293.36);
   short2 result_short2;
    result_short2.x = (short)(result.x);
    result_short2.y = (short)(result.y);

    if(input.x < __float2half_rn(-10))
        result_short2.x  = 0;
    if(input.x > __float2half_rn(10))
        //31743 = 7BFF (65504 - largest normal number)
        result_short2.x  = 31743;
    if(input.y < __float2half_rn(-10))
        result_short2.y = 0;
    if(input.y > __float2half_rn(10))
        result_short2.x  = 31743;

  return *(half2*)(&result_short2);
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
    return exp_half2_saturated(input);
}

//100 cycles
__device__ half2 fast_h2log2(half2 input){
    uint32_t input_int = *(uint32_t*)&input;
    uint32_t exp_raw = (input_int>>10) & 0x001f001f; //get 5 bit exponent only
    half2 exp_raw_half2 = *(half2*)&exp_raw;
    short exp1 = __half_as_ushort(exp_raw_half2.x) - 15;
    short exp2 = __half_as_ushort(exp_raw_half2.y) - 15;
    __half2_raw exp_converted ;
    exp_converted.x = __short2half_rn(exp1);
    exp_converted.y = __short2half_rn(exp2);
    uint32_t mantissa = input_int & 0x03ff03ff;
    uint32_t normalized_int = mantissa | 0x3c003c00;
    half2 normalized = *(half2*)&normalized_int;
    return exp_converted  + normalized - 1.0 + 0.045;
}

//50 cycles, no cvt (type conversion instructions)
__device__  half2 fastest_h2log2(half2 x){
  // exp range from -15 -> 15;
  //uint32_t x_half2 = floats2half2(x,x*4); //e and e+2
  uint32_t x_half2 = *(reinterpret_cast<const unsigned int *>(&(x)));
  uint32_t exp_raw = (x_half2>>10) & 0x001f001f;

  //uint32_t magic_num = __float2half2_rn(32.0);
  uint32_t  magic_num = 0x50005000 | exp_raw;
  half2 exp_converted = *(half2*)&magic_num;
  exp_converted = (exp_converted - 32.0)*32.0 -15.0;
  uint32_t normalized_int = (x_half2 & 0x03ff03ff) | 0x3c003c00;
  half2 normalized = *(half2*)&normalized_int;
  return exp_converted  + normalized - __float2half2_rn(0.955);
}

__device__ half2 fast_h2log(half2 input){
    //0.6931 = ln(2)
    return fastest_h2log2(input)*0.6931;
}

__device__ half2 fast_h2log10(half2 input){

    //0.301 = log10(2);
    return fastest_h2log2(input)*0.301;
}



//54 cycles half2 rsqrt trick on Nvidia V100
__device__ half2 fast_h2rsqrt( half2 input) {

  uint32_t nosign  = (*(uint32_t*)(&input)) & 0x7fff7fff;
//  float xhalf = half(0.5f * abs_x);
  half2 abs_x = *((half2*)&nosign);
  half2 xhalf = __float2half2_rn(0.5f) * abs_x;
  uint32_t sign_only = 0x80008000 &  (*((uint32_t*)&input));
  //sign can multiply;
  //3c00 = 1 ; Bc00 = -1;
  uint32_t multiply_fact = 0x3c003c00 ^ sign_only;

 // uint32_t nosign = *((uint32_t*)&abs_x);

  nosign = 0x59BB59BB - (nosign >> 1);  //
  nosign = nosign & 0x7fff7fff;
  half2 nosign_half2 = *((half2*)& nosign);
//  half2 result = nosign_half2*(__float2half2_rn(1.5f)-(xhalf*nosign_half2*nosign_half2));
  half2 result = nosign_half2*__float2half2_rn(1.5f);

  result -=xhalf*nosign_half2*nosign_half2*nosign_half2;

  return result;
}

// magic number 0x1DE9
__device__ half2 fast_h2sqrt(half2 input){
	return input*fast_h2rsqrt(input);

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


__device__ half2 cos_poly_half2_accurate(half2 x,uint32_t sign_int){
  half2 coeff1 = __float2half2_rn(-0.5);
  half2 coeff2 = __float2half2_rn(0.041666667);
  half2 coeff3 = __float2half2_rn(-0.0013888889);
  half2 coeff4 = __float2half2_rn(0.00002480158);
  half2 result =  1 +  x*x*coeff1 + x*x*x*x*coeff2 + x*x*x*x*x*x*coeff3 ;//+ x*x*x*x*x*x*x*x*coeff4 ;

  uint32_t result_int = *(uint32_t*)&result;
  result_int ^= sign_int;
  return *(half2*)&result_int;
}

__device__ half2 cos_poly_half2_accurate_chebyshev(half2 x_new,uint32_t sign_int){


  //0.0287138*x^4 + 0.0231667*x^3 -0.514296*x^2 + 0.00292027*x + 0.999908

  half2 coeff1 = __float2half2_rn(0.999908);
  half2 coeff2 = __float2half2_rn(0.00292027);

  half2 coeff3 = __float2half2_rn(-0.514296);

  half2 coeff4 = __float2half2_rn(0.0231667);
  half2 coeff5 = __float2half2_rn(0.0287138);

  half2 result =  coeff1 +  x_new*coeff2 + x_new*x_new*coeff3 + x_new*x_new*x_new*coeff4 + x_new*x_new*x_new*x_new*coeff5
  ;//+ x_new* x_new*x_new*x_new*x_new*coeff6;
   ;//+ x*x*x*x*x*x*x*x*coeff4 ;

  uint32_t result_int = *(uint32_t*)&result;
  result_int ^= sign_int;
  return *(half2*)&result_int;
}
__device__ half2 cos_poly_half2_accurate_chebyshev_6(half2 x_new,uint32_t sign_int){


  //-0.0057639854, 0.051200377, -0.0074740962, -0.49730893, -0.00035824697, 1.0000078

  half2 coeff1 = __float2half2_rn( 1.0000078);
  half2 coeff2 = __float2half2_rn(-0.00035824697);

  half2 coeff3 = __float2half2_rn(-0.49730893);

  half2 coeff4 = __float2half2_rn(-0.0074740962);
  half2 coeff5 = __float2half2_rn(0.051200377);
  half2 coeff6 = __float2half2_rn(-0.0057639854);
  half2 result =  coeff1 +  x_new*coeff2 + x_new*x_new*coeff3 + x_new*x_new*x_new*coeff4 + x_new*x_new*x_new*x_new*coeff5
  + x_new* x_new*x_new*x_new*x_new*coeff6;
   ;//+ x*x*x*x*x*x*x*x*coeff4 ;

  uint32_t result_int = *(uint32_t*)&result;
  result_int ^= sign_int;
  return *(half2*)&result_int;
}

__device__ half2 fast_h2cos_accurate(half2 x){
  x = h2absf(x);
  half2 pi = __float2half2_rn( 3.14159265359f);
  half2 invpi = __float2half2_rn(0.31830988618f);
  half2 k = x*invpi;
  half2 k_short_half;
  short2 k_short;
  k_short.x = __half2short_rn(k.x);
  k_short.y = __half2short_rn(k.y);

  k_short_half.x = __short2half_rn(k_short.x);
  k_short_half.y = __short2half_rn(k_short.y);

  half2 normalized =  x - k_short_half*pi;

  uint32_t normalized_abs_int = *(uint32_t*)&normalized;
  normalized_abs_int &=  0x7fff7fff;
  half2 normalized_abs = *(half2*)&normalized_abs_int;
  uint32_t k_int = *(uint32_t*)&k_short;
  k_int = (k_int & 0x00010001 )<<15;
/*  if(threadIdx.x == 0 && blockIdx.x == 0){
    printf("normed val %f k_short %d, k_int %x  %x\n", __low2float(normalized_abs), k_short.x, k_int,k_int <<15);
  }
*/
//  printf("normed val %f k_short %d, k_short %d  %x\n", __low2float(normalized_abs), k_short.x, k_short.y,k_int );

  return cos_poly_half2_accurate(normalized_abs, k_int);
}

__device__ half2 fast_h2sin_accurate(half2 x){

  return fast_h2cos_accurate(x - __float2half2_rn(1.57079632679f));
  //return fast_h2cos_accurate(x + __float2half2_rn(4.71238898038));

}
//#define OLDPOLY
__device__ half2 fast_h2sin(half2 x){
#ifdef OLDPOLY
// range reduction to -2pi;2pi
  half2 twopi = __float2half2_rn( 6.2831853071795865f);
  half2 invtwopi = __float2half2_rn(0.15915494309189534f);
  half2 k = x*invtwopi;
  half2 k_short_half;
  k_short_half.x = __short2half_rn(__half2short_rn(k.x));
  k_short_half.y = __short2half_rn(__half2short_rn(k.y));

  half2 normalized =  x - k_short_half*twopi;
  return sin_poly_half2(normalized);
#else
  return fast_h2sin_accurate(x);
#endif
}


__device__ half2 fast_h2cos(half2 x){
#ifdef OLDPOLY
  half2 twopi = __float2half2_rn( 6.2831853071795865f);
  half2 invtwopi = __float2half2_rn(0.15915494309189534f);
  half2 k = x*invtwopi;
  half2 k_short_half;
  k_short_half.x = __short2half_rn(__half2short_rn(k.x));
  k_short_half.y = __short2half_rn(__half2short_rn(k.y));

  half2 normalized =  x - k_short_half*twopi;
  return cos_poly_half2(normalized);
#else
 return fast_h2cos_accurate(x);
#endif
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
__device__ half2_2 fast_h2sincos(half2 x){
  half2 twopi = __float2half2_rn( 6.2831853071795865f);
  half2 invtwopi = __float2half2_rn(0.15915494309189534f);
  half2 k = x*invtwopi;
  half2 k_short_half;
  k_short_half.x = __short2half_rn(__half2short_rn(k.x));
  k_short_half.y = __short2half_rn(__half2short_rn(k.y));

  half2 normalized =  x - k_short_half*twopi;

  half2 result_sin;
  half2 result_cos;
  half2 coeff1 = __float2half2_rn(-0.166667);
  half2 coeff2 = __float2half2_rn(0.008333);
  half2 coeff3 = __float2half2_rn(-0.00019841269);
  half2 x_2 =  normalized*normalized;
  half2 x_4 = x_2 * x_2;
  half2 x_6 = x_2 * x_4;
  half2 coeff_cos1 = __float2half2_rn(-0.5);
  half2 coeff_cos2 = __float2half2_rn(0.041666667);
  half2 coeff_cos3 = __float2half2_rn(-0.0013888889);
  result_cos =  1 +  x_2*coeff_cos1 + x_4*coeff_cos2 + x_6*coeff_cos3 ;
  result_sin = normalized +  x_2*normalized*coeff1 + x_4*x*coeff2 + x_6*x*coeff3 ;

  half2_2 result;
  result.x = result_sin;
  result.y = result_cos;
  return result;

}
__device__ half2 fast_h2asin(half2 x){

    half2 coeff_3 = __float2half2_rn(0.16667);
    half2 coeff_5 = __float2half2_rn(0.075);
    half2 coeff_7 = __float2half2_rn(0.04464);
    half2 x_2 = x*x;
    half2 x_3 = x_2*x;
    return x + coeff_3*x_3 + coeff_5*x_2*x_3 + coeff_7*x_3*x_3*x;
}

__device__ half2 fast_h2acos(half2 input){

    return __float2half2_rn(1.5708) - fast_h2asin(input);;
}

/*
double RandNum (double min, double max)
{
   return min + (max - min) * ((double)rand()
                            / (double)RAND_MAX);
}
*/

//log2(​x), x-​1+​0.045 in [1,2]

/*
float fastlog2(float input){
  //assume input > 0;
  int input_int = *(int*)&input;
  int exp = (input_int>>23)-127;
  //3f80 0000 1*2^0;
  //007f ffff mantissa only
  int mantissa = input_int & 0x007fffff;
  int normalized_int = mantissa | 0x3f800000;
  float normalized = *(float*)&(normalized_int);

  //half
  short exp = (input_short>>10)-15;
  //3c00 1*2^0;
  //03ff mantissa only
  short mantissa = input_short & 0x03ff;
  short normalized_short = mantissa | 0x3c00;

  printf("normalized %f  exp : %d ", normalized, exp);
  //return exp * 0.30 ; //wrong +-0.3;
  float result = exp  + normalized - 1 + 0.045;
  return result;
}
float fastlog10(float input ){
  //0.301 = log10(2);
  return fastlog2(input)*0.301;
}
float fastlog(float input ){
  //0.6931 = ln(2)
  return fastlog2(input)*0.6931;
}
*/
#endif
