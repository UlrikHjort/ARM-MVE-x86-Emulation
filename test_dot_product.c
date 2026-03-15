/***************************************************************************
--  test_dot_product.c - MVE vectorised dot product with tail predication      
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
 Demonstrates the classic MVE tail-predicated loop pattern:
   while (n > 0) { tail = vctp32q(n); ...; n -= 4; }
 The tail predicate handles the final partial chunk without scalar fallback.
*/
#include <stdio.h>
#include <math.h>
#include "mve.h"

// Arrays are padded to next multiple of 4 (extra elements = 0)
// so reads past 'n' are safe and don't affect the result.
#define PADDED(n) (((n) + 3) & ~3)

static float dot_product(const float *a, const float *b, int n) {
    float32x4_t acc = vdupq_n_f32(0.0f);
    int i = 0;
    while (i < n) {
        mve_pred16_t tail = vctp32q((uint32_t)(n - i));
        float32x4_t va = vpselq_f32(vldrwq_f32(a + i), vdupq_n_f32(0.0f), tail);
        float32x4_t vb = vpselq_f32(vldrwq_f32(b + i), vdupq_n_f32(0.0f), tail);
        acc = vfmaq_f32(acc, va, vb);
        i += 4;
    }
    return vaddvq_f32(acc);
}

static int check(const char *name, float got, float expected, float tol) {
    float diff = got - expected;
    if (diff < 0.0f) diff = -diff;
    int ok = (diff <= tol);
    printf("  %-36s  got=%.4f  exp=%.4f  %s\n",
           name, (double)got, (double)expected, ok ? "OK" : "FAIL");
    return ok;
}

int main(void) {
    int pass = 0, fail = 0;

    puts("=== dot product, n=4 (exact multiple) ===");
    {
        float a[4] = {1,2,3,4};
        float b[4] = {1,1,1,1};
        // 1+2+3+4 = 10
        if (check("dot([1,2,3,4],[1,1,1,1])", dot_product(a,b,4), 10.0f, 1e-5f)) pass++; else fail++;
    }

    puts("\n=== dot product, n=9 (partial last chunk) ===");
    {
        // Padded to 12
        float a[12] = {1,2,3,4,5,6,7,8,9,  0,0,0};
        float b[12] = {1,1,1,1,1,1,1,1,1,  0,0,0};
        // sum 1..9 = 45
        if (check("dot(1..9, ones)", dot_product(a,b,9), 45.0f, 1e-4f)) pass++; else fail++;
    }

    puts("\n=== dot product, n=1 (single element) ===");
    {
        float a[4] = {7.0f, 0,0,0};
        float b[4] = {3.0f, 0,0,0};
        if (check("dot([7],[3])", dot_product(a,b,1), 21.0f, 1e-5f)) pass++; else fail++;
    }

    puts("\n=== dot product, n=7 (two chunks: 4+3) ===");
    {
        float a[8] = {1,0,0,0, 0,0,1,0};  // a=[1,0,0,0,0,0,1,0]
        float b[8] = {5,0,0,0, 0,0,3,0};  // b=[5,0,0,0,0,0,3,0]
        // dot = 1*5 + 0+0+0+0+0 + 1*3 = 8
        if (check("dot (n=7, sparse)", dot_product(a,b,7), 8.0f, 1e-5f)) pass++; else fail++;
    }

    puts("\n=== dot product, n=16 (4 full chunks) ===");
    {
        float a[16], b[16];
        float expected = 0.0f;
        for (int i = 0; i < 16; i++) {
            a[i] = (float)(i + 1);
            b[i] = 1.0f;
            expected += a[i];
        }
        // sum 1..16 = 136
        if (check("dot(1..16, ones)", dot_product(a,b,16), expected, 1e-3f)) pass++; else fail++;
    }

    puts("");
    printf("Result: %d passed, %d failed\n", pass, fail);
    return fail ? 1 : 0;
}
