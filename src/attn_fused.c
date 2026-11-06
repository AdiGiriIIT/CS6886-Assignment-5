/*
 * attn_fused.c -- Q3c: row-wise fusion (the N x N matrix is never stored).
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
void attention(const float *Q, const float *K, const float *V, float *O, int N, int d)
{
    const float scale = 1.0f / sqrtf((float)d);
    /* Transpose K so the score update has independent, contiguous j lanes. */
    float *KT = malloc((size_t)d * N * sizeof(float));
    float *S = malloc((size_t)N * sizeof(float));
    if (!KT || !S) {
        fprintf(stderr, "attention: allocation failed\n");
        exit(EXIT_FAILURE);
    }
    for (int j = 0; j < N; ++j)
        for (int k = 0; k < d; ++k)
            KT[(size_t)k * N + j] = K[(size_t)j * d + k];
    /* Finish each query row before moving on: no N x N allocation. */
    for (int i = 0; i < N; ++i) {
        float *row = S;
        memset(row, 0, (size_t)N * sizeof(float));
        for (int k = 0; k < d; ++k) {
            const float q = Q[(size_t)i * d + k];
            const float *key = KT + (size_t)k * N;
            for (int j = 0; j < N; ++j) row[j] += q * key[j];
        }
        for (int j = 0; j < N; ++j) row[j] *= scale;
        float max = row[0];
        for (int j = 1; j < N; ++j)
            if (row[j] > max) max = row[j];
        float sum = 0.0f;
        for (int j = 0; j < N; ++j) {
            row[j] = expf(row[j] - max);
            sum += row[j];
        }
        for (int j = 0; j < N; ++j) row[j] /= sum;
        float *out = O + (size_t)i * d;
        memset(out, 0, (size_t)d * sizeof(float));
        for (int j = 0; j < N; ++j) {
            const float weight = row[j];
            const float *value = V + (size_t)j * d;
            for (int k = 0; k < d; ++k) out[k] += weight * value[k];
        }
    }
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
