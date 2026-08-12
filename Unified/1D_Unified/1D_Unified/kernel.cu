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
  const int column = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= n || column >= n) return;

  const int index = row * n + column;
  output[index] = (row == 0 || row == n - 1 || column == 0 || column == n - 1)
      ? input[index]
      : 0.25f * (input[index - n] + input[index + n] + input[index - 1] + input[index + 1]);
}

static bool read_input(int* n, int* iterations) {
  std::printf("Grid size N (>= 2): ");
  if (std::scanf("%d", n) != 1 || *n < 2) return false;
  std::printf("Iteration count (>= 0): ");
  return std::scanf("%d", iterations) == 1 && *iterations >= 0;
}

int main() {
  int n = 0, iterations = 0;
  if (!read_input(&n, &iterations)) {
    std::fprintf(stderr, "Invalid input.\\n");
    return EXIT_FAILURE;
  }

  const size_t element_count = static_cast<size_t>(n) * static_cast<size_t>(n);
  const size_t bytes = element_count * sizeof(float);
  float* current = nullptr;
  float* next = nullptr;
  CUDA_CHECK(cudaMallocManaged(&current, bytes));
  CUDA_CHECK(cudaMallocManaged(&next, bytes));

  std::fill(current, current + element_count, 0.0f);
  for (int i = 0; i < n; ++i) {
    current[i * n] = 20.0f;
    current[i * n + n - 1] = 20.0f;
    current[i] = 20.0f;
    current[(n - 1) * n + i] = 20.0f;
  }
  for (int column = std::max(0, n / 2 - 2); column < std::min(n, n / 2 + 2); ++column)
    current[column] = 100.0f;
  std::copy(current, current + element_count, next);

  const dim3 block(kBlockSize, kBlockSize);
  const dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);
  cudaEvent_t start_event, stop_event;
  CUDA_CHECK(cudaEventCreate(&start_event));
  CUDA_CHECK(cudaEventCreate(&stop_event));

  const clock_t cpu_start = std::clock();
  CUDA_CHECK(cudaEventRecord(start_event));
  for (int iteration = 0; iteration < iterations; ++iteration) {
    jacobi_step<<<grid, block>>>(current, next, n);
    CUDA_CHECK(cudaGetLastError());
    float* swap = current;
    current = next;
    next = swap;
  }
  CUDA_CHECK(cudaEventRecord(stop_event));
  CUDA_CHECK(cudaEventSynchronize(stop_event));
  const clock_t cpu_stop = std::clock();

  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event));
  std::printf("N = %d, GPU time: %.3f ms, CPU wall-clock: %.3f s\\n", n, elapsed_ms,
              static_cast<double>(cpu_stop - cpu_start) / CLOCKS_PER_SEC);

  const int sample_step = std::max(1, n / 10);
  std::printf("After firing:\\n");
  for (int row = 0; row < n; row += sample_step) {
    for (int column = 0; column < n; column += sample_step)
      std::printf("%.2f\\t", current[row * n + column]);
    std::printf("\\n");
  }

  CUDA_CHECK(cudaEventDestroy(start_event));
  CUDA_CHECK(cudaEventDestroy(stop_event));
  CUDA_CHECK(cudaFree(current));
  CUDA_CHECK(cudaFree(next));
  return EXIT_SUCCESS;
}
