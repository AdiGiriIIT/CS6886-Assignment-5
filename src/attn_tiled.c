/*
 * attn_tiled.c -- Q3b: cache and register tiling.
 *
 * Implements the corresponding Q3 optimisation in attention().
 * Keep the driver (main) below unchanged. See attn_naive.c for the input and
 * output format.
 */
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* ------------------------------------------------------------------------- */
/* attention(): write your implementation here.                               */
/* ------------------------------------------------------------------------- */
/* C = A B: 32x64 output cache tiles, reduction panels of 64,
 * and a 4x16 accumulator tile (eight AVX2 vectors) inside each panel.
 * A compiler vector type makes the register tile explicit; no math library.
 */
typedef float float8 __attribute__((vector_size(32)));

static void blocked_matmul(const float *A, const float *B, float *C,
                           int M, int N, int K)
{
    enum { BM = 32, BN = 64, BK = 64, MR = 4, NR = 16 };
    memset(C, 0, (size_t)M * N * sizeof(float));
    for (int ii = 0; ii < M; ii += BM) {
        const int ie = ii + BM < M ? ii + BM : M;
        for (int jj = 0; jj < N; jj += BN) {
            const int je = jj + BN < N ? jj + BN : N;
            for (int kk = 0; kk < K; kk += BK) {
                const int ke = kk + BK < K ? kk + BK : K;
                for (int i = ii; i < ie; i += MR) {
                    const int rows = i + MR <= ie ? MR : ie - i;
                    for (int j = jj; j < je; j += NR) {
                        const int cols = j + NR <= je ? NR : je - j;
                        /* Fixed bounds expose eight independent vector accumulators.
                         * Keep the general path below for partial edge tiles. */
                        if (rows == MR && cols == NR) {
                            float8 a00;
                            memcpy(&a00, C + (size_t)(i + 0) * N + j + 0, sizeof(float8));
                            float8 a01;
                            memcpy(&a01, C + (size_t)(i + 0) * N + j + 8, sizeof(float8));
                            float8 a10;
                            memcpy(&a10, C + (size_t)(i + 1) * N + j + 0, sizeof(float8));
                            float8 a11;
                            memcpy(&a11, C + (size_t)(i + 1) * N + j + 8, sizeof(float8));
                            float8 a20;
                            memcpy(&a20, C + (size_t)(i + 2) * N + j + 0, sizeof(float8));
                            float8 a21;
                            memcpy(&a21, C + (size_t)(i + 2) * N + j + 8, sizeof(float8));
                            float8 a30;
                            memcpy(&a30, C + (size_t)(i + 3) * N + j + 0, sizeof(float8));
                            float8 a31;
                            memcpy(&a31, C + (size_t)(i + 3) * N + j + 8, sizeof(float8));
                            for (int k = kk; k < ke; ++k) {
                                float8 b0, b1;
                                memcpy(&b0, B + (size_t)k * N + j, sizeof(float8));
                                memcpy(&b1, B + (size_t)k * N + j + 8, sizeof(float8));
                                const float q0 = A[(size_t)(i + 0) * K + k];
                                a00 += q0 * b0;
                                a01 += q0 * b1;
                                const float q1 = A[(size_t)(i + 1) * K + k];
                                a10 += q1 * b0;
                                a11 += q1 * b1;
                                const float q2 = A[(size_t)(i + 2) * K + k];
                                a20 += q2 * b0;
                                a21 += q2 * b1;
                                const float q3 = A[(size_t)(i + 3) * K + k];
                                a30 += q3 * b0;
                                a31 += q3 * b1;
                            }
                            memcpy(C + (size_t)(i + 0) * N + j + 0, &a00, sizeof(float8));
                            memcpy(C + (size_t)(i + 0) * N + j + 8, &a01, sizeof(float8));
                            memcpy(C + (size_t)(i + 1) * N + j + 0, &a10, sizeof(float8));
                            memcpy(C + (size_t)(i + 1) * N + j + 8, &a11, sizeof(float8));
                            memcpy(C + (size_t)(i + 2) * N + j + 0, &a20, sizeof(float8));
                            memcpy(C + (size_t)(i + 2) * N + j + 8, &a21, sizeof(float8));
                            memcpy(C + (size_t)(i + 3) * N + j + 0, &a30, sizeof(float8));
                            memcpy(C + (size_t)(i + 3) * N + j + 8, &a31, sizeof(float8));
                            continue;
                        }
                        float acc[MR][NR];
                        for (int r = 0; r < rows; ++r)
                            for (int c = 0; c < cols; ++c)
                                acc[r][c] = C[(size_t)(i + r) * N + j + c];
                        for (int k = kk; k < ke; ++k) {
                            const float *b = B + (size_t)k * N + j;
                            for (int r = 0; r < rows; ++r) {
                                const float a = A[(size_t)(i + r) * K + k];
                                for (int c = 0; c < cols; ++c)
                                    acc[r][c] += a * b[c];
                            }
                        }
                        for (int r = 0; r < rows; ++r)
                            for (int c = 0; c < cols; ++c)
                                C[(size_t)(i + r) * N + j + c] = acc[r][c];
                    }
                }
            }
        }
    }
}

void attention(const float *Q, const float *K, const float *V, float *O, int N, int d)
{
    const float scale = 1.0f / sqrtf((float)d);
    /* Transpose K so the score update has independent, contiguous j lanes. */
    float *KT = malloc((size_t)d * N * sizeof(float));
    float *S = malloc((size_t)N * N * sizeof(float));
    if (!KT || !S) {
        fprintf(stderr, "attention: allocation failed\n");
        exit(EXIT_FAILURE);
    }
    for (int j = 0; j < N; ++j)
        for (int k = 0; k < d; ++k)
            KT[(size_t)k * N + j] = K[(size_t)j * d + k];
    blocked_matmul(Q, KT, S, N, N, d);
    for (size_t x = 0; x < (size_t)N * N; ++x) S[x] *= scale;
    /* Stable row softmax: preserve the baseline's reduction order. */
    for (int i = 0; i < N; ++i) {
        float *row = S + (size_t)i * N;
        float max = row[0];
        for (int j = 1; j < N; ++j)
            if (row[j] > max) max = row[j];
        float sum = 0.0f;
        for (int j = 0; j < N; ++j) {
            row[j] = expf(row[j] - max);
            sum += row[j];
        }
        for (int j = 0; j < N; ++j) row[j] /= sum;
    }
    blocked_matmul(S, V, O, N, d, N);
    free(S);
    free(KT);
}

/* ------------------------------------------------------------------------- */
/* Driver: do not change below this line.                                     */
/* ------------------------------------------------------------------------- */
#define REPS 5

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: %s <input.bin> <output.bin>\n", argv[0]);
        return 1;
    }
    FILE *in = fopen(argv[1], "rb");
    if (!in) {
        perror(argv[1]);
        return 1;
    }
    int32_t hdr[2];
    if (fread(hdr, sizeof hdr, 1, in) != 1) {
        fprintf(stderr, "%s: bad header\n", argv[1]);
        return 1;
    }
    int N = hdr[0], d = hdr[1];
    size_t n = (size_t)N * d;
    float *Q = malloc(n * sizeof(float)), *K = malloc(n * sizeof(float));
    float *V = malloc(n * sizeof(float)), *O = malloc(n * sizeof(float));
    if (!Q || !K || !V || !O || fread(Q, sizeof(float), n, in) != n ||
        fread(K, sizeof(float), n, in) != n || fread(V, sizeof(float), n, in) != n) {
        fprintf(stderr, "%s: could not read Q, K, V\n", argv[1]);
        return 1;
    }
    fclose(in);

    attention(Q, K, V, O, N, d); /* warm-up, not timed */
    double t[REPS];
    for (int r = 0; r < REPS; r++) {
        double t0 = now();
        attention(Q, K, V, O, N, d);
        t[r] = now() - t0;
    }
    qsort(t, REPS, sizeof t[0], cmp_double);
    double median = t[REPS / 2];

    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    const char *name = strrchr(argv[0], '/') ? strrchr(argv[0], '/') + 1 : argv[0];
    printf("%-18s N=%-5d d=%-4d threads=%-3d time_ms=%10.3f  GFLOP/s=%8.2f\n", name, N, d,
           threads, median * 1e3, 4.0 * N * (double)N * d / median * 1e-9);

    FILE *out = fopen(argv[2], "wb");
    if (!out || fwrite(hdr, sizeof hdr, 1, out) != 1 || fwrite(O, sizeof(float), n, out) != n) {
        perror(argv[2]);
        return 1;
    }
    fclose(out);
    free(Q);
    free(K);
    free(V);
    free(O);
    return 0;
}
