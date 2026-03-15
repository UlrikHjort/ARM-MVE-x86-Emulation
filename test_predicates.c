/***************************************************************************
-- test_predicates.c - Verify MVE predicate encoding, vctp32q, vpselq, predicate logic 
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
#include <stdint.h>
#include "mve.h"

static int g_pass = 0, g_fail = 0;

static void check16(const char *name, mve_pred16_t got, mve_pred16_t expected) {
    int ok = (got == expected);
    printf("  %-42s  got=0x%04X  exp=0x%04X  %s\n",
           name, (unsigned)got, (unsigned)expected, ok ? "OK" : "FAIL");
    if (ok) g_pass++; else g_fail++;
}

static void checkf(const char *name, float got, float expected) {
    int ok = (got == expected);
    printf("  %-42s  got=%.1f  exp=%.1f  %s\n", name, (double)got, (double)expected, ok ? "OK" : "FAIL");
    if (ok) g_pass++; else g_fail++;
}

// -------------------------------------------------------
int main(void) {
    puts("=== vctp32q (32-bit tail predication) ===");
    // nibble per lane: vctp32q(n) activates lanes 0..n-1
    check16("vctp32q(0)", vctp32q(0), 0x0000);
    check16("vctp32q(1)", vctp32q(1), 0x000F);  // lane 0 only
    check16("vctp32q(2)", vctp32q(2), 0x00FF);  // lanes 0-1
    check16("vctp32q(3)", vctp32q(3), 0x0FFF);  // lanes 0-2
    check16("vctp32q(4)", vctp32q(4), 0xFFFF);  // all 4 lanes
    check16("vctp32q(5)", vctp32q(5), 0xFFFF);  // saturates at 0xFFFF

    puts("\n=== vctp16q (16-bit tail predication) ===");
    // pair per lane: vctp16q(n) activates lanes 0..n-1
    check16("vctp16q(0)", vctp16q(0), 0x0000);
    check16("vctp16q(1)", vctp16q(1), 0x0003);  // lane 0 only
    check16("vctp16q(4)", vctp16q(4), 0x00FF);  // lanes 0-3
    check16("vctp16q(8)", vctp16q(8), 0xFFFF);  // all 8 lanes

    puts("\n=== vcmpgtq_f32 -> mve_pred16_t encoding ===");
    {
        float32x4_t a = {1.0f, 5.0f, 3.0f, 7.0f};
        float32x4_t b = {2.0f, 4.0f, 3.0f, 6.0f};
        // lane0: 1>2=F, lane1: 5>4=T, lane2: 3>3=F, lane3: 7>6=T
        // active lanes 1,3 -> nibbles 1,3 set -> 0xF0F0
        check16("vcmpgtq_f32 {1,5,3,7}>{2,4,3,6}", vcmpgtq_f32(a, b), 0xF0F0);
    }
    {
        float32x4_t a = {10.0f, 10.0f, 10.0f, 10.0f};
        float32x4_t b = {5.0f,  5.0f,  5.0f,  5.0f};
        check16("vcmpgtq_f32 all true",  vcmpgtq_f32(a, b), 0xFFFF);
        check16("vcmpgtq_f32 all false", vcmpgtq_f32(b, a), 0x0000);
    }

    puts("\n=== vcmpleq_f32 ===");
    {
        float32x4_t mag  = {1.0f, 5.0f, 3.0f, 4.0f};
        float32x4_t four = {4.0f, 4.0f, 4.0f, 4.0f};
        // lane0: 1<=4=T, lane1: 5<=4=F, lane2: 3<=4=T, lane3: 4<=4=T
        // active lanes 0,2,3: 0x000F | 0x0F00 | 0xF000 = 0xFF0F
        check16("vcmpleq_f32 lanes 0,2,3 active", vcmpleq_f32(mag, four), 0xFF0F);
    }

    puts("\n=== vpselq_f32 ===");
    {
        float32x4_t x = {10.0f, 20.0f, 30.0f, 40.0f};
        float32x4_t y = {1.0f,   2.0f,  3.0f,  4.0f};

        float32x4_t r;

        r = vpselq_f32(x, y, 0xFFFF);  // all active -> all from x
        checkf("vpselq all-active [0]", r[0], 10.0f);
        checkf("vpselq all-active [3]", r[3], 40.0f);

        r = vpselq_f32(x, y, 0x0000);  // none active -> all from y
        checkf("vpselq none-active [0]", r[0], 1.0f);
        checkf("vpselq none-active [3]", r[3], 4.0f);

        // lanes 0,2 active (nibbles 0,2 set) -> 0x0F0F
        r = vpselq_f32(x, y, 0x0F0F);
        checkf("vpselq 0x0F0F [0] -> x", r[0], 10.0f);
        checkf("vpselq 0x0F0F [1] -> y", r[1],  2.0f);
        checkf("vpselq 0x0F0F [2] -> x", r[2], 30.0f);
        checkf("vpselq 0x0F0F [3] -> y", r[3],  4.0f);
    }

    puts("\n=== vpselq_f32 driven by vcmpgtq_f32 ===");
    {
        float32x4_t a   = {1.0f, 5.0f, 3.0f, 7.0f};
        float32x4_t b   = {2.0f, 4.0f, 3.0f, 6.0f};
        mve_pred16_t p  = vcmpgtq_f32(a, b);      // lanes 1,3 (a>b)
        float32x4_t res = vpselq_f32(a, b, p);    // a where a>b, else b
        checkf("max[0] = b[0]=2", res[0], 2.0f);
        checkf("max[1] = a[1]=5", res[1], 5.0f);
        checkf("max[2] = b[2]=3", res[2], 3.0f);
        checkf("max[3] = a[3]=7", res[3], 7.0f);
    }

    puts("\n=== predicate logic (vpand / vpor / vpnot) ===");
    {
        mve_pred16_t p0 = vctp32q(2);  // lanes 0,1 -> 0x00FF
        mve_pred16_t p1 = vctp32q(3);  // lanes 0,1,2 -> 0x0FFF
        check16("vpand(0x00FF, 0x0FFF)", vpand(p0, p1), 0x00FF);
        check16("vpor (0x00FF, 0x0FFF)", vpor (p0, p1), 0x0FFF);
        check16("vpnot(0x00FF)        ", (mve_pred16_t)(vpnot(p0) & 0xFFFF), 0xFF00);
    }

    puts("\n=== vcmpgtq_s32 ===");
    {
        int32x4_t a = {1, 10, 3, 10};
        int32x4_t b = {5,  5, 3,  5};
        // lane0: 1>5=F, lane1: 10>5=T, lane2: 3>3=F, lane3: 10>5=T -> 0xF0F0
        check16("vcmpgtq_s32 lanes 1,3 active", vcmpgtq_s32(a, b), 0xF0F0);
    }

    puts("");
    printf("Result: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
