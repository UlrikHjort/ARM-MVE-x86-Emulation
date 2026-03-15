/***************************************************************************
--                       misc test
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
#include <inttypes.h>
#include "mve.h"

int main() {
    float32x4_t a = {1,2,3,4};
    float32x4_t b = {5,6,7,8};

    // c = a + b*2
    float32x4_t c = vfmaq_f32(a, b, vdupq_n_f32(2.0f));

    // MVE: comparisons return mve_pred16_t, not a vector
    mve_pred16_t pred = vcmpgtq_f32(c, vdupq_n_f32(10.0f));

    // MVE: predicated select via vpselq (true=c, false=a)
    float32x4_t result = vpselq_f32(c, a, pred);

    printf("result = (%f, %f, %f, %f)\n",
        result[0], result[1], result[2], result[3]);

    int32x4_t x = {1,2,3,4};
    int32x4_t y = {5,6,7,8};

    // MVE: vpselq_s32 with mve_pred16_t from vcmpgtq_s32
    int32x4_t z = vpselq_s32(x, y, vcmpgtq_s32(x, y));
    printf("int result = (%" PRId32 ", %" PRId32 ", %" PRId32 ", %" PRId32 ")\n",
        z[0], z[1], z[2], z[3]);
}
