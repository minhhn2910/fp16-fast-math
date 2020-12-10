#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <cuda_fp16.h>
#include "newhalf.hpp"
#include "../../include/fast_math.cuh"
#define FP_TYPE float
#define FP_DEV_TYPE float
#define FP_TYPE_REF double
#define LOOP 500
#define INC_DIFF 0.001
#define HOST_FUNC asin
#define DEV_FUNC asinf
//#define PERFORMANCE

/* Kernel for vector addition */

double RandNum (double min, double max)
{
   return min + (max - min) * ((double)rand()
                            / (double)RAND_MAX);
}


__global__ void Vec_add(FP_DEV_TYPE x[], FP_DEV_TYPE y[], FP_DEV_TYPE z[], int n) {
   /* blockDim.x = threads_per_block                            */
   /* First block gets first threads_per_block components.      */
   /* Second block gets next threads_per_block components, etc. */
   int tid = blockDim.x * blockIdx.x + threadIdx.x;
   /* block_count*threads_per_block may be >= n */
   // a register to avoid compiler optimization
   float sink = 0.0;
//   float sink2 = 0.0;
   if (tid < n) {
      float temp = x[tid];
//      float temp2 = x[2*tid+1];
#ifdef PERFORMANCE //measure performance
      for (int i = 0; i < LOOP; i++){
        temp += INC_DIFF;
//	temp2 += INC_DIFF;
	      sink += DEV_FUNC(temp);
//	sink2 += DEV_FUNC(temp2);
	}
#else //measure accuracy
	sink = DEV_FUNC(temp);
#endif
      z[tid]=  sink;
//      z[2*tid+1]=  sink2;

   }
}  /* Vec_add */


/* Host code */
int main(int argc, char* argv[]) {

   int n, i;
   FP_TYPE *h_x, *h_y, *h_z, *h_lookup;
   FP_TYPE_REF *h_z_ref;
   FP_DEV_TYPE *d_x, *d_y, *d_z, *d_lookup ;
   uint32_t *h_startClk, *h_stopClk;
   uint32_t *d_startClk, *d_stopClk;
   int threads_per_block;
   int block_count;
   size_t size, size_clock;
	cudaEvent_t start, stop;
  float elapsedTime;
  srand(1234);
   /* Get number of components in vector */
   if (argc != 2) {
      fprintf(stderr, "usage: %s <vector order>\n", argv[0]);
      exit(0);
   }
   n = strtol(argv[1], NULL, 10); // half2 = 2x half , reduce size
   size = n*sizeof(FP_TYPE);
   size_clock = n*sizeof(uint32_t);
   /* Allocate input vectors in host memory */
   h_x = (FP_TYPE*) malloc(size);
   h_y = (FP_TYPE*) malloc(size);
   h_z = (FP_TYPE*) malloc(size);


   h_z_ref = (FP_TYPE_REF*) malloc(n*sizeof(FP_TYPE_REF));
   // declare and allocate memory

   /* Initialize input vectors */
   for (i = 0; i < n; i++) {
     double temp = RandNum(-1,1);
     h_x[i] = temp;
     h_z_ref[i] = HOST_FUNC(temp);
   }


   /* Allocate vectors in device memory */
   cudaMalloc(&d_x, size);
   cudaMalloc(&d_y, size);
   cudaMalloc(&d_z, size);
     /* Copy vectors from host memory to device memory */
   cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);
   cudaMemcpy(d_y, h_y, size, cudaMemcpyHostToDevice);

//   cudaMemcpy(buffer, h_buffer, MAX_TEXTURE_SIZE*sizeof(float), cudaMemcpyHostToDevice); //copy data to texture

   /* Define block size */
   threads_per_block = 256;

   block_count = (n + threads_per_block - 1)/threads_per_block;
	cudaEventCreate(&start);
	cudaEventRecord(start,0);

   Vec_add<<<block_count, threads_per_block>>>(d_x, d_y, d_z, n);

   cudaDeviceSynchronize();
	cudaEventCreate(&stop);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
 cudaEventElapsedTime(&elapsedTime, start,stop);
 printf("Elapsed time : %f ms\n" ,elapsedTime);
  cudaMemcpy(h_z, d_z, size, cudaMemcpyDeviceToHost);
#ifndef PERFORMANCE
  double err = 0;
   for (i = 0; i < n; i++) {
     if(h_z_ref[i]!=0)
        err += fabs((h_z[i]-h_z_ref[i])/h_z_ref[i]);
   }

   printf("err %.8f \n", err/n);
#endif
   /* Free device memory */
   cudaFree(d_x);
   cudaFree(d_y);
   cudaFree(d_z);

   /* Free host memory */
   free(h_x);
   free(h_y);
   free(h_z);

   return 0;
}  /* main */
