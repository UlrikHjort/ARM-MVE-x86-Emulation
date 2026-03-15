/***************************************************************************
-- mandelbrot.c - Cross-platform SIMD Mandelbrot using real MVE ACLE intrinsics
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
#include <stdio.h>
#include "mve.h"

#define WIDTH    80
#define HEIGHT   40
#define MAX_ITER 50

char palette[] = " .,:;irsXA253hMHGS#9B&@";

// Mandelbrot iteration for 4 pixels at once using MVE intrinsics
int mandel4(float32x4_t cr, float32x4_t ci) {
    float32x4_t zr   = vdupq_n_f32(0.0f);
    float32x4_t zi   = vdupq_n_f32(0.0f);
    int32x4_t   iter = vdupq_n_s32(0);

    for (int i = 0; i < MAX_ITER; i++) {
        float32x4_t zr2 = vmulq_f32(zr, zr);
        float32x4_t zi2 = vmulq_f32(zi, zi);
        float32x4_t mag = vaddq_f32(zr2, zi2);

        // MVE: comparison returns mve_pred16_t (not a vector)
        mve_pred16_t active = vcmpleq_f32(mag, vdupq_n_f32(4.0f));

        // All lanes escaped -> done
        if (active == 0) break;

        float32x4_t new_zr = vaddq_f32(vsubq_f32(zr2, zi2), cr);
        float32x4_t new_zi = vfmaq_n_f32(ci, vmulq_f32(zr, zi), 2.0f);

        zr = new_zr;
        zi = new_zi;

        // Add 1 to still-active lanes using vpselq (no type-pun needed)
        iter = vaddq_s32(iter, vpselq_s32(vdupq_n_s32(1), vdupq_n_s32(0), active));
    }

    return (iter[0] & 0xFF)        |
           ((iter[1] & 0xFF) << 8) |
           ((iter[2] & 0xFF) << 16)|
           ((iter[3] & 0xFF) << 24);
}

int main() {
    for (int y = 0; y < HEIGHT; y++) {
        float ci = (y - HEIGHT / 2.0f) * 2.5f / HEIGHT;

        for (int x = 0; x < WIDTH; x += 4) {
            // Tail predication: handle last chunk even if < 4 pixels wide
            mve_pred16_t tail = vctp32q((uint32_t)(WIDTH - x));

            float32x4_t cr = {
                (x+0 - WIDTH/2.0f) * 3.5f / WIDTH,
                (x+1 - WIDTH/2.0f) * 3.5f / WIDTH,
                (x+2 - WIDTH/2.0f) * 3.5f / WIDTH,
                (x+3 - WIDTH/2.0f) * 3.5f / WIDTH,
            };
            float32x4_t civ = vdupq_n_f32(ci);

            int packed = mandel4(cr, civ);

            for (int i = 0; i < 4; i++) {
                // Skip inactive tail lanes
                if (!_MVE_PRED32_ACTIVE(tail, i)) continue;

                int it  = (packed >> (i * 8)) & 0xFF;
                int idx = it * (int)(sizeof(palette) - 2) / MAX_ITER;
                putchar(palette[idx]);
            }
        }
        putchar('\n');
    }
}
