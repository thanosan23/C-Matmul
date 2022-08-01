#define _GNU_SOURCE
// clang -mfma -O2 -ffast-math -g main.c -Rpass=vector && ./a.out
#include <stdio.h>
#include <stdint.h>
#include <immintrin.h>
#include <unistd.h>
#include <pthread.h>
#include <math.h>

#include "profile.h"

#define N 1024
#define BLOCK_SIZE 8

float A[N*N] __attribute__ ((aligned(32)));
float B[N*N] __attribute__ ((aligned(32)));
float C[N*N] __attribute__ ((aligned(32)));

__m256* A_avx = (__m256*) A;
__m256* B_avx = (__m256*) B;

int main() {
  struct timespec start, end;
  start = current_time();
  for(int by = 0; by < N; by+=BLOCK_SIZE) {
    for(int bx = 0; bx < N; bx+=BLOCK_SIZE) {
      __m256 acc[BLOCK_SIZE*BLOCK_SIZE] = {};
      for(int y = 0; y < BLOCK_SIZE; ++y) {
        for(int k = 0; k < N; k+=8) {
          for(int x = 0; x < BLOCK_SIZE; ++x) {
            acc[(y*BLOCK_SIZE)+x] = _mm256_fmadd_ps(A_avx[((by+y)*N+k)/8], B_avx[((bx+x)*N+k)/8], acc[(y*BLOCK_SIZE)+x]);
          }
        }
      }
      float res;
      for(int y = 0; y < BLOCK_SIZE; ++y) {
        for(int x = 0; x < BLOCK_SIZE; ++x) {
          res = 0.0;
          for(int b = 0; b < 8; b++) res+=acc[(y*BLOCK_SIZE)+x][b];
          C[(by+y)*N+(bx+x)] = res;
        }
      }
    }
  } 
  end = current_time();
  double fma = N*N*N; // fused multiply adds
  double tflop = (2*fma);
  double s = diff(&end, &start);
  printf("%f TFLOP/S\n", (tflop/s)*1e-12);
  return 0;
}