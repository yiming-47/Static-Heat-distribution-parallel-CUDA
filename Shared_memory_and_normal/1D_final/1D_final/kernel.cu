#include "cuda_runtime.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <ctime>

constexpr int kBlockSize = 16;

#define CUDA_CHECK(call) do { \
  const cudaError_t error = (call); \
  if (error != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d: %s\\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
    std::exit(EXIT_FAILURE); \
  } \
} while (0)

__global__ void jacobi_step(const float* input, float* output, int n) {
  // x advances through contiguous columns, so a warp performs coalesced accesses.
  const int column = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= n || column >= n) return;

  const int index = row * n + column;
  if (row == 0 || row == n - 1 || column == 0 || column == n - 1) {
    output[index] = input[index];
    return;
  }

  output[index] = 0.25f * (input[index - n] + input[index + n]
                          + input[index - 1] + input[index + 1]);
}

static bool read_input(int* n, int* iterations) {
  std::printf("Grid size N (>= 2): ");
  if (std::scanf("%d", n) != 1 || *n < 2) return false;
  std::printf("Iteration count (>= 0): ");
  return std::scanf("%d", iterations) == 1 && *iterations >= 0;
}

int main() {
  int n = 0;
  int iterations = 0;
  if (!read_input(&n, &iterations)) {
    std::fprintf(stderr, "Invalid input.\\n");
    return EXIT_FAILURE;
  }

  const size_t element_count = static_cast<size_t>(n) * static_cast<size_t>(n);
  const size_t bytes = element_count * sizeof(float);
  float* host_current = static_cast<float*>(std::malloc(bytes));
  float* host_next = static_cast<float*>(std::malloc(bytes));
  if (!host_current || !host_next) {
    std::fprintf(stderr, "Host allocation failed.\\n");
    std::free(host_current);
    std::free(host_next);
    return EXIT_FAILURE;
  }

  std::fill(host_current, host_current + element_count, 0.0f);
  for (int i = 0; i < n; ++i) {
    host_current[i * n] = 20.0f;
    host_current[i * n + n - 1] = 20.0f;
    host_current[i] = 20.0f;
    host_current[(n - 1) * n + i] = 20.0f;
  }
  const int source_begin = std::max(0, n / 2 - 2);
  const int source_end = std::min(n, n / 2 + 2);
  for (int column = source_begin; column < source_end; ++column) host_current[column] = 100.0f;
  std::copy(host_current, host_current + element_count, host_next);

  float* device_current = nullptr;
  float* device_next = nullptr;
  CUDA_CHECK(cudaMalloc(&device_current, bytes));
  CUDA_CHECK(cudaMalloc(&device_next, bytes));
  CUDA_CHECK(cudaMemcpy(device_current, host_current, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_next, host_next, bytes, cudaMemcpyHostToDevice));

  const dim3 block(kBlockSize, kBlockSize);
  const dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);
  cudaEvent_t start_event, stop_event;
  CUDA_CHECK(cudaEventCreate(&start_event));
  CUDA_CHECK(cudaEventCreate(&stop_event));

  const clock_t cpu_start = std::clock();
  CUDA_CHECK(cudaEventRecord(start_event));
  for (int iteration = 0; iteration < iterations; ++iteration) {
    jacobi_step<<<grid, block>>>(device_current, device_next, n);
    CUDA_CHECK(cudaGetLastError());
    float* swap = device_current;
    device_current = device_next;
    device_next = swap;
  }
  CUDA_CHECK(cudaEventRecord(stop_event));
  CUDA_CHECK(cudaEventSynchronize(stop_event));
  const clock_t cpu_stop = std::clock();

  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event));
  CUDA_CHECK(cudaMemcpy(host_current, device_current, bytes, cudaMemcpyDeviceToHost));
  std::printf("N = %d, GPU time: %.3f ms, CPU wall-clock: %.3f s\\n", n, elapsed_ms,
              static_cast<double>(cpu_stop - cpu_start) / CLOCKS_PER_SEC);

  const int sample_step = std::max(1, n / 10);
  std::printf("After firing:\\n");
  for (int row = 0; row < n; row += sample_step) {
    for (int column = 0; column < n; column += sample_step)
      std::printf("%.2f\\t", host_current[row * n + column]);
    std::printf("\\n");
  }

  CUDA_CHECK(cudaEventDestroy(start_event));
  CUDA_CHECK(cudaEventDestroy(stop_event));
  CUDA_CHECK(cudaFree(device_current));
  CUDA_CHECK(cudaFree(device_next));
  std::free(host_current);
  std::free(host_next);
  return EXIT_SUCCESS;
}
