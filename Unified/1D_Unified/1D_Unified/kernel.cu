#include "cuda_runtime.h"
#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#define blockx 8
void Input(int*,int*);
__global__ void add(int N, float *orig,float *jaco) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if((i > 0) && (i < N-1) && (j > 0) && (j < N-1))
  {
		jaco[i * N + j] = 0.25 * (orig[(i-1) * N + j]
					  +orig[(i+1) * N + j]
					  +orig[i * N + (j-1)]
					  +orig[i * N + (j+1)]);
  }
}
__global__ void transfer(int N, float *a, float *b) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  b[i * N + j] = a[i * N + j];
}
int main()
{
  int i,j,N,limit,iteration;
  int fire_start,fire_end;
  Input(&N,&limit);
  float *orig,*jaco,*temp,elapsedTime;
  float *dev_orig, *dev_jaco;
  clock_t start_t,end_t;
  double total_t;
  int size = N * N * sizeof(float);
  orig = (float *)malloc(sizeof(float) * (N * N));
  jaco = (float *)malloc(sizeof(float) * (N * N));
  cudaMallocManaged(&orig, N*N*sizeof(float));
  cudaMallocManaged(&jaco, N*N*sizeof(float));
//initial 
  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
    orig[i * N + j] = 0.0;
    }
  }
  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      orig[i * N] = 20.0;
      orig[i] = 20.0;
      orig[i * N + (N - 1)] = 20.0;
      orig[(N - 1) * N + i] = 20.0;
    }
  }
  for (i = int(N / 2) - 2;i < int(N / 2) + 2;i++)
  {
	orig[int(i)] = 100.0;
  }
  memcpy((void *)jaco, (void *)orig, N*N *sizeof(float));
  printf("Initial Temperatures: \n");
  for (i = 0; i < N; i += N/10) {
    for (j = 0; j < N; j += N/10) {
      printf("%-.2f\t",orig[i * N + j]);
      }
    printf("\n");
  }
  printf("\n");
  cudaEvent_t e_start, e_stop;
  cudaEventCreate(&e_start);
  cudaEventCreate(&e_stop);
  cudaEventRecord(e_start, 0);
  cudaEventRecord(e_stop, 0);  
  
  dim3 dimBlock(blockx,blockx);
  dim3 dimGrid (N/dimBlock.x , N/dimBlock.y) ;
  start_t = clock();
// it's fine
  for(iteration = 0; iteration < limit;iteration++)
  {
    add<<<dimGrid,dimBlock>>>( N,orig,jaco);
    transfer<<<dimGrid,dimBlock>>>(N, jaco,orig);
  }
  cudaDeviceSynchronize();
  end_t = clock();
  total_t = (double)(end_t - start_t) / CLOCKS_PER_SEC;
  cudaEventSynchronize(e_stop);
  cudaEventElapsedTime(&elapsedTime, e_start, e_stop);
  printf("N = %d, execute time : %f clock_t: %f\n",N,elapsedTime,total_t);
  printf("After firing: \n");
  for (i = 0; i < N; i += N/10) {
    for (j = 0; j < N; j += N/10) {
      printf("%-.2f\t",orig[i * N + j]);
      }
    printf("\n");
  }
  cudaFree(orig);
  cudaFree(jaco);
  system("PAUSE");
  return 0;
}
void Input(int* N,int* limit) {
  printf("plz input N ft\n");
  scanf("%d", N);
  printf("plz input iter num\n");
  scanf("%d", limit);

}
