#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); return 1; \
} } while (0)

enum { Q_N = 1024, KV_MAX = 576, RAW_CAP = 2 };

int main(void) {
    const size_t page = (size_t)getpagesize();
    const size_t model_bytes = page * 2;
    void *model = NULL;
    CHECK(!posix_memalign(&model, page, model_bytes));
    float *q_weight = model;
    float *kv_weight = (float *)((char *)model + page);
    for (int i = 0; i < Q_N; i++) q_weight[i] = 1.0f;
    CHECK(ds4_gpu_set_model_map(model, model_bytes));

    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc_managed(Q_N * sizeof(float));
    ds4_gpu_tensor *q_out = ds4_gpu_tensor_alloc_managed(Q_N * sizeof(float));
    ds4_gpu_tensor *kv = ds4_gpu_tensor_alloc_managed(KV_MAX * sizeof(float));
    ds4_gpu_tensor *kv_out = ds4_gpu_tensor_alloc_managed(KV_MAX * sizeof(float));
    ds4_gpu_tensor *raw = ds4_gpu_tensor_alloc_managed(RAW_CAP * KV_MAX * sizeof(float));
    CHECK(q && q_out && kv && kv_out && raw);

    float q_input[Q_N], kv_input[KV_MAX], raw_input[RAW_CAP * KV_MAX];
    float q_ref[Q_N], kv_ref[KV_MAX], raw_ref[RAW_CAP * KV_MAX];
    float q_new[Q_N], kv_new[KV_MAX], raw_new[RAW_CAP * KV_MAX];
    for (int i = 0; i < Q_N; i++) q_input[i] = (float)(i % 31 - 15) / 32.0f;
    const int shapes[][2] = {{64, 64}, {68, 64}, {128, 64}, {128, 66},
                             {132, 64}, {256, 64}, {512, 64}, {576, 64}};
    unsigned finite_cases = 0, nonfinite_matches = 0;
    for (unsigned shape = 0; shape < sizeof(shapes) / sizeof(*shapes); shape++) {
        const int n = shapes[shape][0], rot = shapes[shape][1];
        for (int mode = 0; mode < 8; mode++) {
            for (int i = 0; i < n; i++) {
                const float boundary[] = {0x1p-20f, 0x1p-12f, 0x1p-4f,
                                          0.9999f, 1.0f, 1.0001f, 448.0f};
                const float v = mode == 0 ? 0.0f : mode == 1 ? boundary[i % 7] :
                    mode == 2 ? (float)((i * 7919) % 4093) / 123.0f :
                    mode == 3 ? (i % 3 ? 0x1p-18f : 0x1p-25f) :
                    (float)((i * 17) % 63 + 1) / 16.0f;
                kv_input[i] = i & 1 ? -v : v;
                kv_weight[i] = 1.0f;
            }
            // Norm's input stays finite, but its output has one non-finite
            // non-RoPE value for the reduction's NaN/Inf characterization.
            if (mode >= 4 && n > rot) {
                int lane = mode < 6 ? 0 : mode == 6 ? 33 : 63;
                if (lane >= n - rot) lane = n - rot - 1;
                kv_weight[lane] = (mode & 1) ? INFINITY : NAN;
            }
            for (int i = 0; i < RAW_CAP * n; i++) raw_input[i] = -123.0f;
            for (int variant = 0; variant < 2; variant++) {
                if (variant) setenv("DS4_METAL_FP8_KV_SIMD_MAX", "1", 1);
                else unsetenv("DS4_METAL_FP8_KV_SIMD_MAX");
                CHECK(ds4_gpu_tensor_write(q, 0, q_input, sizeof(q_input)));
                CHECK(ds4_gpu_tensor_write(kv, 0, kv_input, n * sizeof(float)));
                CHECK(ds4_gpu_tensor_write(raw, 0, raw_input, RAW_CAP * n * sizeof(float)));
                CHECK(ds4_gpu_dsv4_qkv_rms_norm_kv_rope_fp8_store_tensor(
                    q_out, q, model, model_bytes, 0, Q_N,
                    kv_out, kv, page, n, raw, RAW_CAP, 1, rot,
                    97, 4096, 10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f, 1e-6f));
                float *qo = variant ? q_new : q_ref;
                float *ko = variant ? kv_new : kv_ref;
                float *ro = variant ? raw_new : raw_ref;
                CHECK(ds4_gpu_tensor_read(q_out, 0, qo, sizeof(q_ref)));
                CHECK(ds4_gpu_tensor_read(kv_out, 0, ko, n * sizeof(float)));
                CHECK(ds4_gpu_tensor_read(raw, 0, ro, RAW_CAP * n * sizeof(float)));
            }
            const int same = !memcmp(q_ref, q_new, sizeof(q_ref)) &&
                             !memcmp(kv_ref, kv_new, n * sizeof(float)) &&
                             !memcmp(raw_ref, raw_new, RAW_CAP * n * sizeof(float));
            if (mode < 4) {
                if (!same) fprintf(stderr, "finite mismatch: head=%d rot=%d mode=%d\n", n, rot, mode);
                CHECK(same);
                finite_cases++;
            } else {
                nonfinite_matches += same;
                const int lane = mode < 6 ? 0 : mode == 6 ? 33 : 63;
                const int active_lane = lane < n - rot ? lane : n - rot - 1;
                fprintf(stderr, "nonfinite head=%d rot=%d %s: %s (KV=%a raw=%a)\n",
                        n, rot, mode & 1 ? "Inf" : "NaN",
                        same ? "bitwise match" : "differs",
                        active_lane >= 0 ? kv_ref[active_lane] : 0.0f,
                        active_lane >= 0 ? raw_ref[n + active_lane] : 0.0f);
            }
        }
    }
    unsetenv("DS4_METAL_FP8_KV_SIMD_MAX");
    fprintf(stderr, "FP8 fused max: %u finite byte-exact cases; %u/32 nonfinite bitwise matches\n",
            finite_cases, nonfinite_matches);
    ds4_gpu_tensor_free(q); ds4_gpu_tensor_free(q_out);
    ds4_gpu_tensor_free(kv); ds4_gpu_tensor_free(kv_out); ds4_gpu_tensor_free(raw);
    ds4_gpu_cleanup();
    free(model);
    return 0;
}
