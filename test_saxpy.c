/***************************************************************************
--      test_saxpy.c - MVE vectorised SAXPY: y[i] = alpha * x[i] + y[i]   
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
 Shows predicated in-place FMA across an arbitrarily-sized array.
 Uses vfmaq_n_f32 (multiply by scalar) + tail predication.
*/
#include <stdio.h>
#include <math.h>
#include "mve.h"

#define PADDED(n) (((n) + 3) & ~3)

static void saxpy(float alpha, const float *x, float *y, int n) {
    int i = 0;
    while (i < n) {
        mve_pred16_t tail = vctp32q((uint32_t)(n - i));

        float32x4_t vx = vpselq_f32(vldrwq_f32(x + i), vdupq_n_f32(0.0f), tail);
        float32x4_t vy = vpselq_f32(vldrwq_f32(y + i), vdupq_n_f32(0.0f), tail);

        // vy = vy + alpha * vx
        vy = vfmaq_n_f32(vy, vx, alpha);

        // Write back only active lanes (others keep original y value)
        float32x4_t orig = vldrwq_f32(y + i);
        float32x4_t out  = vpselq_f32(vy, orig, tail);
        vstrwq_f32(y + i, out);

        i += 4;
    }
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

    puts("=== SAXPY n=4, alpha=2 ===");
    {
        float x[4] = {1,2,3,4};
        float y[4] = {10,10,10,10};
        saxpy(2.0f, x, y, 4);
        // y = [10+2,10+4,10+6,10+8] = [12,14,16,18]
        if (check("y[0]", y[0], 12.0f, 1e-5f)) pass++; else fail++;
        if (check("y[1]", y[1], 14.0f, 1e-5f)) pass++; else fail++;
        if (check("y[2]", y[2], 16.0f, 1e-5f)) pass++; else fail++;
        if (check("y[3]", y[3], 18.0f, 1e-5f)) pass++; else fail++;
    }

    puts("\n=== SAXPY n=5 (partial last chunk), alpha=3 ===");
    {
        // Padded to 8
        float x[8] = {1,2,3,4,5,  0,0,0};
        float y[8] = {0,0,0,0,0,  9,9,9};  // y[5..7]=9 must not be touched
        saxpy(3.0f, x, y, 5);
        // y[0..4] = 3*1,3*2,3*3,3*4,3*5 = 3,6,9,12,15
        if (check("y[0]",  y[0],  3.0f, 1e-5f)) pass++; else fail++;
        if (check("y[1]",  y[1],  6.0f, 1e-5f)) pass++; else fail++;
        if (check("y[2]",  y[2],  9.0f, 1e-5f)) pass++; else fail++;
        if (check("y[3]",  y[3], 12.0f, 1e-5f)) pass++; else fail++;
        if (check("y[4]",  y[4], 15.0f, 1e-5f)) pass++; else fail++;
        if (check("y[5] untouched", y[5], 9.0f, 1e-5f)) pass++; else fail++;
    }

    puts("\n=== SAXPY n=11, alpha=0.5 ===");
    {
        float x[12] = {2,4,6,8,10,12,14,16,18,20,22, 0};
        float y[12] = {1,1,1,1, 1, 1, 1, 1, 1, 1, 1, 0};
        saxpy(0.5f, x, y, 11);
        // y[i] = 1 + 0.5*(2,4,...,22) = 2,3,4,5,6,7,8,9,10,11,12
        int ok = 1;
        for (int i = 0; i < 11; i++) {
            float expected = 1.0f + 0.5f * (float)(2*(i+1));
            if (fabsf(y[i] - expected) > 1e-4f) ok = 0;
        }
        if (check("y[0..10] = 2..12", ok ? 0.0f : 1.0f, 0.0f, 0.5f)) pass++; else fail++;
    }

    puts("");
    printf("Result: %d passed, %d failed\n", pass, fail);
    return fail ? 1 : 0;
}
