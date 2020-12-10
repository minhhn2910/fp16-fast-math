#ifndef HALF2_OVERLOAD_CUH
#define HALF2_OVERLOAD_CUH
#include "cuda_fp16.h"
//#define __h2div h2div
#define __h2exp h2exp

inline __device__ __half2 h2absf(__half2 d)
{
	(*((unsigned int*)(&d))) &= 0x7fff7fff; //set 2 sign bits to 0
	return d;
}
inline __device__  __half2 h2negf(__half2 d)
{
	(*((unsigned int*)(&d))) ^= 0x80008000; //flip 2 sign bits
	return d;
}
/*
inline __device__ __half2 operator + (const __half2 a, const __half2 b) {
   return __hadd2(a, b);
 }
inline __device__ __half2 operator * (const __half2 a, const __half2 b) {
   return __hmul2(a, b);
 }
inline __device__ __half2 operator - (const __half2 a, const __half2 b) {
   return __hsub2(a, b);
 }
inline __device__ __half2 operator / (const __half2 a, const __half2 b) {
	return __h2div(a,b);
 }

inline __device__ __half2 operator - (const __half2 a) {
   return h2negf(a);
 }
inline __device__ __half2& operator += (__half2& a, const __half2 b) {
   a = a + b;
   return a;
 }
inline __device__ __half2& operator *= (__half2& a, const __half2 b) {
   a = a * b;
   return a;
 }
inline __device__ __half2& operator -= (__half2& a, const __half2 b) {
   a = a - b;
   return a;
 }
inline __device__ __half2& operator /= (__half2& a, const __half2 b) {
   a = a / b;
   return a;
 }
//~ inline __device__ __half2& operator = (__half2& a, const __half2 b) {
   //~ a =  __float2half2_rn(b);
   //~ return a;
 //~ }

 
inline __device__ __half2 operator == (const __half2 a, const __half2 b) {
   return __heq2(a, b);
 }
inline __device__ __half2 operator != (const __half2 a, const __half2 b) {
   return __hne2(a, b);
 }
inline __device__ __half2 operator < (const __half2 a, const __half2 b) {
   return __hlt2(a, b);
 }
inline __device__ __half2 operator <= (const __half2 a, const __half2 b) {
   return __hle2(a, b);
 }
inline __device__ __half2 operator > (const __half2 a, const __half2 b) {
   return __hgt2(a, b);
 }
inline __device__ __half2 operator >= (const __half2 a, const __half2 b) {
   return __hge2(a, b);
 }
*/

inline __device__ __half2 operator + (const float a, const __half2 b) {
   return __hadd2(__float2half2_rn(a), b);
 }
inline __device__ __half2 operator * (const float a, const __half2 b) {
   return __hmul2(__float2half2_rn(a), b);
 }
inline __device__ __half2 operator - (const float a, const __half2 b) {
   return __hsub2(__float2half2_rn(a), b);
 }
inline __device__ __half2 operator / (const float a, const __half2 b) {
	return __h2div(__float2half2_rn(a),b);
 }
inline __device__ __half2 operator + (const __half2 a, const float b) {
   return __hadd2(a, __float2half2_rn(b));
 }
inline __device__ __half2 operator * (const __half2 a, const float b) {
   return __hmul2(a, __float2half2_rn(b));
 }
inline __device__ __half2 operator - (const __half2 a, const float b) {
   return __hsub2(a, __float2half2_rn(b));
 }
inline __device__ __half2 operator / (const __half2 a, const float b) {
	return __h2div(a,__float2half2_rn(b));
 }




inline __device__ __half operator + (const float a, const __half b) {
   return __hadd(__float2half_rn(a), b);
 }
inline __device__ __half operator * (const float a, const __half b) {
   return __hmul(__float2half_rn(a), b);
 }
inline __device__ __half operator - (const float a, const __half b) {
   return __hsub(__float2half_rn(a), b);
 }
inline __device__ __half operator / (const float a, const __half b) {
	return __hdiv(__float2half_rn(a),b);
 }
inline __device__ __half operator + (const __half a, const float b) {
   return __hadd(a, __float2half_rn(b));
 }
inline __device__ __half operator * (const __half a, const float b) {
   return __hmul(a, __float2half_rn(b));
 }
inline __device__ __half operator - (const __half a, const float b) {
   return __hsub(a, __float2half_rn(b));
 }

//mathfunc float
inline __device__ __half2 __fdividef(const __half2 a, const __half2 b){
	return a/b;
}
//mathfunc float
inline __device__ __half2 __fdividef(const float a, const __half2 b){
	return __float2half2_rn(a)/b;
}
//mathfunc float
inline __device__ __half2 __fdividef(const __half2 a, const float b){
	return a/__float2half2_rn(b);
}
inline __device__ __half2 cosf(const __half2 a){
	return h2cos(a);
}
inline __device__ __half2 sinf(const __half2 a){
	return h2sin(a);
	}
inline __device__ __half2 expf(const __half2 a){
	return h2exp(a);
	}
inline __device__ __half2 __expf(const __half2 a){
	return h2exp(a);
	}
//~ #define FLOAT_EXP10 "exp10f("
//~ #define FLOAT_EXP2 "exp2f("
inline __device__ __half2 logf(const __half2 a){
	return h2log(a);
	}
inline __device__ __half2 __logf(const __half2 a){
	return h2log(a);
	}
//~ #define FLOAT_LOG10 "log10f("
//~ #define FLOAT_LOG2 "log2f("
inline __device__ __half2 rsqrtf(const __half2 a){
	return h2rsqrt(a);
	}
inline __device__ __half2 sqrtf(const __half2 a){
	return h2sqrt(a);
	}
inline __device__ __half2 fabsf(const __half2 a){
	return h2absf(a);
	}


#endif
// math funcs
//~ #define HALF2_COS "h2cos("
//~ #define HALF2_SIN "h2sin("
//~ #define HALF2_EXP "h2exp("
//~ #define HALF2_EXP10 "h2exp10("
//~ #define HALF2_EXP2 "h2exp2("
//~ #define HALF2_LOG "h2log("
//~ #define HALF2_LOG10 "h2log10("
//~ #define HALF2_LOG2 "h2log2("
//~ #define HALF2_RSQRT "h2rsqrt("
//~ #define HALF2_SQRT "h2sqrt("
//~ #define HALF2_ABS "h2abs("
