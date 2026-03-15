/***************************************************************************
--      test_stats.c - MVE vectorised array statistics: sum, min, max
--
--           Copyright (C) 2026 By Ulrik Hørlyk Hjort
--
-- Permission is hereby granted, free of charge, to any person obtaining
-- a copy of this software and associated documentation files (the
-- "Software"), to deal in the Software without restriction, including
-- without limitation the rights to use, copy, modify, merge, publish,
-- distribute, sublicense, and/or sell copies of the Software, and to
-- permit persons to whom the Software is furnished to do so, subject to
-- the following conditions:
--
-- The above copyright notice and this permission notice shall be
-- included in all copies or substantial portions of the Software.
--
-- THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
-- EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
-- MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
-- NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
-- LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
-- OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
-- WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
-- ***************************************************************************/
/*
 Demonstrates:
   vaddvq_f32  - horizontal sum
   vminnmvq_f32 / vmaxnmvq_f32 - horizontal min/max with running accumulator
   vminnmq_f32 / vmaxnmq_f32   - element-wise min/max
   vcvtq_f32_s32 / vcvtq_s32_f32 - type conversion
   tail predication throughout
*/
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <inttypes.h>
#include "mve.h"

// Sum of array
static float array_sum(const float *a, int n) {
    float32x4_t acc = vdupq_n_f32(0.0f);
    int i = 0;
    while (i < n) {
        mve_pred16_t tail = vctp32q((uint32_t)(n - i));
        float32x4_t v = vpselq_f32(vldrwq_f32(a + i), vdupq_n_f32(0.0f), tail);
        acc = vaddq_f32(acc, v);
        i += 4;
    }
    return vaddvq_f32(acc);
}

// Min of array
static float array_min(const float *a, int n) {
    float32x4_t acc = vdupq_n_f32(FLT_MAX);
    int i = 0;
    while (i < n) {
        mve_pred16_t tail = vctp32q((uint32_t)(n - i));
        float32x4_t v = vpselq_f32(vldrwq_f32(a + i), vdupq_n_f32(FLT_MAX), tail);
        acc = vminnmq_f32(acc, v);
        i += 4;
    }
    return vminnmvq_f32(FLT_MAX, acc);
}

// Max of array
static float array_max(const float *a, int n) {
    float32x4_t acc = vdupq_n_f32(-FLT_MAX);
    int i = 0;
    while (i < n) {
        mve_pred16_t tail = vctp32q((uint32_t)(n - i));
        float32x4_t v = vpselq_f32(vldrwq_f32(a + i), vdupq_n_f32(-FLT_MAX), tail);
        acc = vmaxnmq_f32(acc, v);
        i += 4;
    }
    return vmaxnmvq_f32(-FLT_MAX, acc);
}

// Element-wise clamp: result[i] = clamp(a[i], lo, hi)
static void array_clamp(const float *in, float *out, int n, float lo, float hi) {
    float32x4_t vlo = vdupq_n_f32(lo);
    float32x4_t vhi = vdupq_n_f32(hi);
    int i = 0;
    while (i < n) {
        mve_pred16_t tail = vctp32q((uint32_t)(n - i));
        float32x4_t v = vpselq_f32(vldrwq_f32(in + i), vlo, tail);  // mask inactive -> lo (harmless)
        v = vmaxnmq_f32(v, vlo);
        v = vminnmq_f32(v, vhi);
        // write back active lanes only
        float32x4_t prev = vldrwq_f32(out + i);
        vstrwq_f32(out + i, vpselq_f32(v, prev, tail));
        i += 4;
    }
}

// int32 -> float -> scale -> int32 round-trip
static void int_scale(const int32_t *in, int32_t *out, int n, float scale) {
    int i = 0;
    while (i < n) {
        mve_pred16_t tail = vctp32q((uint32_t)(n - i));
        int32x4_t  vi  = vpselq_s32(vldrwq_s32(in + i), vdupq_n_s32(0), tail);
        float32x4_t vf = vcvtq_f32_s32(vi);
        vf = vmulq_f32(vf, vdupq_n_f32(scale));
        int32x4_t  res = vcvtq_s32_f32(vf);
        int32x4_t  prev = vldrwq_s32(out + i);
        vstrwq_s32(out + i, vpselq_s32(res, prev, tail));
        i += 4;
    }
}

// -------------------------------------------------------

static int g_pass = 0, g_fail = 0;

static void checkf(const char *name, float got, float expected, float tol) {
    float diff = got - expected; if (diff < 0) diff = -diff;
    int ok = (diff <= tol);
    printf("  %-40s  got=%.4f  exp=%.4f  %s\n",
           name, (double)got, (double)expected, ok ? "OK" : "FAIL");
    if (ok) g_pass++; else g_fail++;
}


static void checki(const char *name, int32_t got, int32_t expected) {
    int ok = (got == expected);
    printf("  %-40s  got=%" PRId32 "  exp=%" PRId32 "  %s\n", name, got, expected, ok ? "OK" : "FAIL");
    if (ok) g_pass++; else g_fail++;
}

int main(void) {
    puts("=== array_sum ===");
    {
        float a[12] = {1,2,3,4,5,6,7,8,9,  0,0,0};
        checkf("sum(1..9)", array_sum(a, 9), 45.0f, 1e-3f);
        float b[4] = {-1,-2,-3,-4};
        checkf("sum(-1,-2,-3,-4)", array_sum(b, 4), -10.0f, 1e-5f);
        float c[4] = {0,0,0,0};
        checkf("sum(zeros)", array_sum(c, 4), 0.0f, 1e-5f);
    }

    puts("\n=== array_min / array_max ===");
    {
        float a[8] = {3,1,4,1,5,9,2,6};
        checkf("min(3,1,4,1,5,9,2,6)", array_min(a, 8), 1.0f, 1e-5f);
        checkf("max(3,1,4,1,5,9,2,6)", array_max(a, 8), 9.0f, 1e-5f);
    }
    {
        float b[5] = {-3,0,7,-10,2};  // padded to 8
        float bpad[8] = {-3,0,7,-10,2, 0,0,0};
        checkf("min(n=5 with negatives)", array_min(bpad, 5), -10.0f, 1e-5f);
        checkf("max(n=5 with negatives)", array_max(bpad, 5),   7.0f, 1e-5f);
        (void)b;
    }

    puts("\n=== array_clamp ===");
    {
        float in[8]  = {-5, 0, 3, 7, 10, 15, 2, 8};
        float out[8] = { 0, 0, 0, 0,  0,  0, 0, 0};
        array_clamp(in, out, 8, 0.0f, 8.0f);
        //            0, 0, 3, 7,  8,  8, 2, 8
        checkf("clamp(-5→0)",   out[0], 0.0f, 1e-5f);
        checkf("clamp(0→0)",    out[1], 0.0f, 1e-5f);
        checkf("clamp(3→3)",    out[2], 3.0f, 1e-5f);
        checkf("clamp(7→7)",    out[3], 7.0f, 1e-5f);
        checkf("clamp(10→8)",   out[4], 8.0f, 1e-5f);
        checkf("clamp(15→8)",   out[5], 8.0f, 1e-5f);
    }

    puts("\n=== int32 scale via vcvtq (int→float→int) ===");
    {
        int32_t in[8]  = {1,2,3,4,5,6,7,8};
        int32_t out[8] = {0,0,0,0,0,0,0,0};
        int_scale(in, out, 8, 10.0f);
        // out[i] = round(in[i]*10)
        for (int i = 0; i < 8; i++) {
            char name[32];
            snprintf(name, sizeof(name), "out[%d]=%d*10", i, i+1);
            checki(name, out[i], (i+1)*10);
        }
    }

    puts("\n=== partial chunk: n=6 ===");
    {
        float a[8] = {1,2,3,4,5,6, 0,0};
        checkf("sum(n=6)", array_sum(a, 6), 21.0f, 1e-4f);
        checkf("min(n=6)", array_min(a, 6),  1.0f, 1e-5f);
        checkf("max(n=6)", array_max(a, 6),  6.0f, 1e-5f);
    }

    puts("");
    printf("Result: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
