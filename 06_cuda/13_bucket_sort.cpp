#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void init_bucket(int *bucket) {
  bucket[threadIdx.x] = 0;
}

__global__ void make_bucket(int *bucket, int *key) {
  atomicAdd(&bucket[key[threadIdx.x]], 1);
}

__global__ void exe_bucket(int *bucket, int *key, int *offset) {
  int i = threadIdx.x;
  for (int j = offset[i]; j < offset[i] + bucket[i]; j++) {
      key[j] = i;
  }
}

int main() {
  int n = 50;
  int range = 5;

  int *key;
  int *bucket;
  int *offset;
  cudaMallocManaged(&key, n * sizeof(int));
  cudaMallocManaged(&bucket, n * sizeof(int));
  cudaMallocManaged(&offset, range * sizeof(int));

  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  init_bucket<<<1, range>>>(bucket);
  make_bucket<<<1, n>>>(bucket, key);
  cudaDeviceSynchronize();

  for (int i=1; i<range; i++) 
    offset[i] = offset[i-1] + bucket[i-1];

  exe_bucket<<<1, range>>>(bucket, key, offset);
  cudaDeviceSynchronize();

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
}
