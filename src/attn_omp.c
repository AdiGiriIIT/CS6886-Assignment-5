/*
 * attn_omp.c -- Q4: FlashAttention-style attention with OpenMP.
 *
 * CPU FlashAttention-style tiles with stable online softmax.
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
/* CPU FlashAttention-style tiling and online softmax. No N x N scores.
 * The local AVX2/FMA kernel and exp approximation match the first candidate. */
#ifndef _OPENMP
#error "Compile attn_omp.c with OpenMP enabled (-fopenmp)"
#endif
#ifndef ATTN_STATIC_SCHEDULE
#define ATTN_STATIC_SCHEDULE 0
#endif
#include <immintrin.h>
#ifndef FLASH_BQ
#define FLASH_BQ 24
#endif
#ifndef FLASH_BK
#define FLASH_BK 128
#endif
#ifndef FLASH_MR
#define FLASH_MR 6
#endif
#ifndef FLASH_NR
#define FLASH_NR 16
#endif
#ifndef FLASH_EXP
#define FLASH_EXP 1
#endif
#if FLASH_BQ < 1 || FLASH_BK < 1 || FLASH_MR < 1 || FLASH_NR < 8 || FLASH_NR % 8 != 0 || FLASH_BK % FLASH_NR != 0
#error "Invalid FlashAttention tile configuration"
#endif

/* exp(x) for softmax's nonpositive arguments. Range reduction followed by
 * a degree-7 Taylor polynomial on [-ln(2)/2, ln(2)/2]. Values below -87
 * are flushed to zero (absolute error <= exp(-87)); exp(0) is exactly one.
 * This is an FP32 approximation, validated separately and end-to-end. */
static inline __m256 flash_exp8(__m256 x)
{
    const __m256 lower = _mm256_set1_ps(-87.0f);
    __m256 keep = _mm256_cmp_ps(x, lower, _CMP_GE_OQ);
    x = _mm256_max_ps(lower, _mm256_min_ps(x, _mm256_setzero_ps()));
    __m256 n = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(1.4426950408889634f)),
                              _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 r = _mm256_fnmadd_ps(n, _mm256_set1_ps(0.693145751953125f), x);
    r = _mm256_fnmadd_ps(n, _mm256_set1_ps(1.428606765330187e-6f), r);
    __m256 p = _mm256_set1_ps(1.0f / 5040.0f);
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f / 720.0f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f / 120.0f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f / 24.0f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f / 6.0f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.5f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f));
    __m256i exponent = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(n),
                                                       _mm256_set1_epi32(127)), 23);
    return _mm256_and_ps(keep, _mm256_mul_ps(p, _mm256_castsi256_ps(exponent)));
}

/* C = A B. Packed B panels hold K successive FLASH_NR-element rows.
 * The full reduction stays in registers; partial output tiles use scalar C. */
static void flash_product(const float *A, const float *B, float *C,
                         int M, int N, int K, int panel_k, int add)
{
    enum { MR = FLASH_MR, NR = FLASH_NR, NV = FLASH_NR / 8 };
    for (int j = 0; j < N; j += NR) {
        const int cols = N - j < NR ? N - j : NR;
        for (int i = 0; i < M; i += MR) {
            const int rows = M - i < MR ? M - i : MR;
            if (rows == MR && cols == NR) {
                __m256 acc[MR][NV];
                #pragma unroll
                for (int r = 0; r < MR; ++r) {
                    #pragma unroll
                    for (int v = 0; v < NV; ++v)
                        acc[r][v] = add
                            ? _mm256_loadu_ps(C + (size_t)(i + r) * N + j + v * 8)
                            : _mm256_setzero_ps();
                }
                for (int k = 0; k < K; ++k) {
                    const float *b = B + (size_t)(j / NR) * panel_k * NR + (size_t)k * NR;
                    __m256 bv[NV];
                    #pragma unroll
                    for (int v = 0; v < NV; ++v)
                        bv[v] = _mm256_loadu_ps(b + v * 8);
                    #pragma unroll
                    for (int r = 0; r < MR; ++r) {
                        __m256 a = _mm256_set1_ps(A[(size_t)(i + r) * K + k]);
                        #pragma unroll
                        for (int v = 0; v < NV; ++v)
                            acc[r][v] = _mm256_fmadd_ps(a, bv[v], acc[r][v]);
                    }
                }
                #pragma unroll
                for (int r = 0; r < MR; ++r) {
                    #pragma unroll
                    for (int v = 0; v < NV; ++v)
                        _mm256_storeu_ps(C + (size_t)(i + r) * N + j + v * 8,
                                         acc[r][v]);
                }
            } else {
                for (int r = 0; r < rows; ++r)
                    for (int c = 0; c < cols; ++c) {
                        float sum = add ? C[(size_t)(i + r) * N + j + c] : 0.0f;
                        for (int k = 0; k < K; ++k) {
                            size_t index = (size_t)(j / NR) * panel_k * NR + (size_t)k * NR + c;
                            sum += A[(size_t)(i + r) * K + k] * B[index];
                        }
                        C[(size_t)(i + r) * N + j + c] = sum;
                    }
            }
        }
    }
}

void attention(const float *Q, const float *K, const float *V, float *O, int N, int d)
{
    enum { NR = FLASH_NR };
    const int qrows = N < FLASH_BQ ? N : FLASH_BQ;
    const int krows = N < FLASH_BK ? N : FLASH_BK;
    const size_t npad = ((size_t)N + NR - 1) / NR * NR;
    const size_t dpad = ((size_t)d + NR - 1) / NR * NR;
    float *keys = calloc(npad * d, sizeof(float));
    float *values = calloc(dpad * N, sizeof(float));
    if (!keys || !values) {
        fprintf(stderr, "attention: allocation failed\n");
        exit(EXIT_FAILURE);
    }
    const float scale = 1.0f / sqrtf((float)d);
    /* Query blocks own disjoint output rows. Only packed K/V are shared;
     * score tiles and online softmax state are private to each worker. */
    #pragma omp parallel default(none) shared(Q, K, V, O, keys, values, N, d, qrows, krows, scale, stderr)
    {
        float *scores = malloc((size_t)qrows * krows * sizeof(float));
        float *maximum = malloc((size_t)qrows * sizeof(float));
        float *normalizer = malloc((size_t)qrows * sizeof(float));
        if (!scores || !maximum || !normalizer) {
            fprintf(stderr, "attention: worker allocation failed\n");
            exit(EXIT_FAILURE);
        }
        /* Assign whole packed panels, avoiding interleaved writers. */
        #pragma omp for schedule(static) nowait
        for (int panel = 0; panel < N; panel += NR) {
            const int width = N - panel < NR ? N - panel : NR;
            for (int k = 0; k < d; ++k)
                for (int c = 0; c < width; ++c) {
                    size_t index = (size_t)(panel / NR) * d * NR + (size_t)k * NR + c;
                    keys[index] = K[(size_t)(panel + c) * d + k];
                }
        }
        /* Split even the 4/8 feature panels across key-row chunks so that
         * packing can use more than 4/8 workers on the supplied inputs. */
        #pragma omp for collapse(2) schedule(static) nowait
        for (int panel = 0; panel < d; panel += NR)
            for (int block = 0; block < N; block += 128) {
                const int width = d - panel < NR ? d - panel : NR;
                const int end = N - block < 128 ? N : block + 128;
                for (int j = block; j < end; ++j)
                    for (int c = 0; c < width; ++c)
                        values[(size_t)(panel / NR) * N * NR + (size_t)j * NR + c]
                            = V[(size_t)j * d + panel + c];
            }
        /* Finish both packing loops before any worker reads packed operands. */
        #pragma omp barrier
#if ATTN_STATIC_SCHEDULE
        #pragma omp for schedule(static) nowait
#else
        #pragma omp for schedule(dynamic, 1) nowait
#endif
        for (int first = 0; first < N; first += qrows) {
            const int rows = N - first < qrows ? N - first : qrows;
            float *out = O + (size_t)first * d;
            memset(out, 0, (size_t)rows * d * sizeof(float));
            for (int i = 0; i < rows; ++i) {
                maximum[i] = -INFINITY;
                normalizer[i] = 0.0f;
            }
            for (int key_first = 0; key_first < N; key_first += FLASH_BK) {
                const int cols = N - key_first < FLASH_BK ? N - key_first : FLASH_BK;
                /* key_first is aligned to a packed NR-wide panel. */
                flash_product(Q + (size_t)first * d,
                              keys + (size_t)(key_first / NR) * d * NR,
                              scores, rows, cols, d, d, 0);
                for (int i = 0; i < rows; ++i) {
                    float *row = scores + (size_t)i * cols;
                    for (int j = 0; j < cols; ++j) row[j] *= scale;
                    float tile_max = row[0];
                    for (int j = 1; j < cols; ++j)
                        if (row[j] > tile_max) tile_max = row[j];
                    const float new_max = fmaxf(maximum[i], tile_max);
                    /* Rescale past contributions when a later tile raises the max.
                     * First tile: alpha=0; unchanged max: alpha=1 exactly. */
                    const float alpha = key_first == 0 ? 0.0f
                        : (new_max == maximum[i] ? 1.0f : expf(maximum[i] - new_max));
                    float sum = 0.0f;
                    int j = 0;
                    if (FLASH_EXP) {
                        const __m256 m = _mm256_set1_ps(new_max);
                        for (; j + 8 <= cols; j += 8) {
                            __m256 e = flash_exp8(_mm256_sub_ps(_mm256_loadu_ps(row + j), m));
                            _mm256_storeu_ps(row + j, e);
                            for (int lane = 0; lane < 8; ++lane) sum += row[j + lane];
                        }
                    }
                    for (; j < cols; ++j) {
                        row[j] = expf(row[j] - new_max);
                        sum += row[j];
                    }
                    normalizer[i] = alpha * normalizer[i] + sum;
                    maximum[i] = new_max;
                    float *o = out + (size_t)i * d;
                    for (int k = 0; k < d; ++k) o[k] *= alpha;
                }
                /* U <- alpha*U + exp(scores-new_max) V_tile.
                 * Packed V has global panel stride N even for a short last tile. */
                flash_product(scores, values + (size_t)key_first * NR,
                              out, rows, d, cols, N, 1);
            }
            for (int i = 0; i < rows; ++i) {
                float *o = out + (size_t)i * d;
                const float inverse = 1.0f / normalizer[i];
                for (int k = 0; k < d; ++k) o[k] *= inverse;
            }
        }
        free(normalizer);
        free(maximum);
        free(scores);
    } /* Join before releasing the shared packed arrays. */
    free(values);
    free(keys);
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
