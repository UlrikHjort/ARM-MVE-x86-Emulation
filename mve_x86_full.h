/***************************************************************************
--  mve_x86_full.h - ARM MVE (Helium) ACLE intrinsic emulation on x86
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
 Implements the real MVE API: mve_pred16_t comparisons, vpselq, vctp32q, etc.
 Backed by GCC/Clang __attribute__((vector_size)) -> SSE on x86.
*/
#ifndef MVE_X86_FULL_H
#define MVE_X86_FULL_H

#include <stdint.h>

// -------------------------------------------------------
// Vector types  (same names as ARM MVE ACLE)
// -------------------------------------------------------
typedef float    float32x4_t  __attribute__((vector_size(16)));
typedef int32_t  int32x4_t    __attribute__((vector_size(16)));
typedef int16_t  int16x8_t    __attribute__((vector_size(16)));
typedef uint16_t uint16x8_t   __attribute__((vector_size(16)));
typedef int8_t   int8x16_t    __attribute__((vector_size(16)));
typedef uint8_t  uint8x16_t   __attribute__((vector_size(16)));

// -------------------------------------------------------
// MVE predicate type
// VPR.P0 is 16 bits; lane encoding by element width:
//   32-bit (float32x4 / int32x4): nibble per lane
//       lane 0 -> bits [3:0],  lane 1 -> bits [7:4]
//       lane 2 -> bits [11:8], lane 3 -> bits [15:12]
//   16-bit (int16x8):  pair per lane  (bits [1:0]..[15:14])
//    8-bit (int8x16):  one bit per lane (bits [0]..[15])
// A lane is active when bit 0 of its field is 1.
// -------------------------------------------------------
typedef uint16_t mve_pred16_t;

// Internal helpers: build a mve_pred16_t from per-lane boolean conditions.
//
// _MVE_P32 — 4 lanes, 32-bit elements (float32x4_t / int32x4_t)
//   The 16-bit predicate is divided into four 4-bit nibbles (one per lane).
//   A lane is active when its nibble is 0xF (all bits set); inactive = 0x0.
//
//   bit layout:  15..12  11..8   7..4    3..0
//                lane3   lane2   lane1   lane0
//                0xF000  0x0F00  0x00F0  0x000F
//
//   Each term:  (lN) ? 0xNNNNu : 0u
//     - true  → OR in 0xF for that nibble  (all 4 bits set = lane active)
//     - false → OR in 0x0 for that nibble  (all 4 bits clear = lane inactive)
//
//   Example: lanes 1 and 3 active  →  0x0000|0x00F0|0x0000|0xF000 = 0xF0F0
//
#define _MVE_P32(l0,l1,l2,l3) \
    ((mve_pred16_t)(((l0)?0x000Fu:0u)|((l1)?0x00F0u:0u)| \
                    ((l2)?0x0F00u:0u)|((l3)?0xF000u:0u)))

// _MVE_P16 - 8 lanes, 16-bit elements (int16x8_t)
//   The 16-bit predicate is divided into eight 2-bit pairs (one per lane).
//   A lane is active when its pair is 0x3 (both bits set); inactive = 0x0.
//
//   bit layout:  15..14  13..12  11..10  9..8    7..6    5..4    3..2    1..0
//                lane7   lane6   lane5   lane4   lane3   lane2   lane1   lane0
//                0xC000  0x3000  0x0C00  0x0300  0x00C0  0x0030  0x000C  0x0003
//
//   The constants follow the pattern 0x0003 << (lane * 2) - each lane's
//   2-bit mask is shifted 2 positions further left than the previous one.
//
//   Example: lanes 0 and 2 active  ->  0x0003|0x0030 = 0x0033
//
#define _MVE_P16(l0,l1,l2,l3,l4,l5,l6,l7) \
    ((mve_pred16_t)(((l0)?0x0003u:0u)|((l1)?0x000Cu:0u)| \
                    ((l2)?0x0030u:0u)|((l3)?0x00C0u:0u)| \
                    ((l4)?0x0300u:0u)|((l5)?0x0C00u:0u)| \
                    ((l6)?0x3000u:0u)|((l7)?0xC000u:0u)))

// Test whether lane n is active in a 32-bit predicate
#define _MVE_PRED32_ACTIVE(p,n) (((p) >> ((n)*4)) & 1u)
// Test whether lane n is active in a 16-bit predicate
#define _MVE_PRED16_ACTIVE(p,n) (((p) >> ((n)*2)) & 1u)

// -------------------------------------------------------
// Tail predication  (vctp - Vector Create Tail Predicate)
// -------------------------------------------------------
static inline mve_pred16_t vctp32q(uint32_t n) {
    static const uint16_t t[5] = {0x0000u, 0x000Fu, 0x00FFu, 0x0FFFu, 0xFFFFu};
    return (n >= 4) ? 0xFFFFu : t[n];
}

static inline mve_pred16_t vctp16q(uint32_t n)
{
    static const uint16_t t[9] = {
        0x0000u, 0x0003u, 0x000Fu, 0x003Fu, 0x00FFu,
        0x03FFu, 0x0FFFu, 0x3FFFu, 0xFFFFu
    };
    return (n >= 8) ? 0xFFFFu : t[n];
}

static inline mve_pred16_t vctp8q(uint32_t n)
{
    return (n >= 16) ? 0xFFFFu : (uint16_t)((1u << n) - 1u);
}

// -------------------------------------------------------
// Load / Store
// -------------------------------------------------------
#define vldrwq_f32(ptr)         (*(const float32x4_t *)(ptr))
#define vstrwq_f32(ptr, val)    (*(float32x4_t *)(ptr) = (val))
#define vldrwq_s32(ptr)         (*(const int32x4_t *)(ptr))
#define vstrwq_s32(ptr, val)    (*(int32x4_t *)(ptr) = (val))
#define vldrwq_u16(ptr)         (*(const uint16x8_t *)(ptr))
#define vstrwq_u16(ptr, val)    (*(uint16x8_t *)(ptr) = (val))
// MVE ACLE also keeps vld1q/vst1q names
#define vld1q_f32   vldrwq_f32
#define vst1q_f32   vstrwq_f32
#define vld1q_s32   vldrwq_s32
#define vst1q_s32   vstrwq_s32

// -------------------------------------------------------
// Broadcast / duplication
// -------------------------------------------------------
static inline float32x4_t vdupq_n_f32(float v)      { return (float32x4_t){v,v,v,v}; }
static inline int32x4_t   vdupq_n_s32(int32_t v)    { return (int32x4_t){v,v,v,v}; }
static inline int16x8_t   vdupq_n_s16(int16_t v)    { return (int16x8_t){v,v,v,v,v,v,v,v}; }
static inline uint16x8_t  vdupq_n_u16(uint16_t v)   { return (uint16x8_t){v,v,v,v,v,v,v,v}; }

// -------------------------------------------------------
// Lane access
// -------------------------------------------------------
#define vgetq_lane_f32(v, i)        ((v)[i])
#define vgetq_lane_s32(v, i)        ((v)[i])
#define vsetq_lane_f32(val, v, i)   ((v)[i] = (val), (v))
#define vsetq_lane_s32(val, v, i)   ((v)[i] = (val), (v))

// -------------------------------------------------------
// Unpredicated arithmetic
// -------------------------------------------------------
#define vaddq_f32(a, b)      ((a) + (b))
#define vsubq_f32(a, b)      ((a) - (b))
#define vmulq_f32(a, b)      ((a) * (b))
#define vaddq_s32(a, b)      ((a) + (b))
#define vsubq_s32(a, b)      ((a) - (b))
#define vandq_s32(a, b)      ((a) & (b))
#define vorrq_s32(a, b)      ((a) | (b))
#define veorq_s32(a, b)      ((a) ^ (b))
#define vshlq_n_s32(a, n)    ((a) << (n))
#define vshrq_n_s32(a, n)    ((a) >> (n))
#define vshlq_n_s16(a, n)    ((a) << (n))
#define vshrq_n_s16(a, n)    ((a) >> (n))

// Fused multiply-add: a + b*c
static inline float32x4_t vfmaq_f32(float32x4_t a, float32x4_t b, float32x4_t c)
{
    return a + b * c;
}
// Scalar multiply-add: a + b * scalar
static inline float32x4_t vfmaq_n_f32(float32x4_t a, float32x4_t b, float s)
{
    return a + b * vdupq_n_f32(s);
}

// -------------------------------------------------------
// Comparisons  ->  mve_pred16_t   (NOT a vector mask)
// -------------------------------------------------------

// float32x4
static inline mve_pred16_t vcmpgtq_f32(float32x4_t a, float32x4_t b)
{ return _MVE_P32(a[0]>b[0], a[1]>b[1], a[2]>b[2], a[3]>b[3]); }

static inline mve_pred16_t vcmpltq_f32(float32x4_t a, float32x4_t b)
{ return _MVE_P32(a[0]<b[0], a[1]<b[1], a[2]<b[2], a[3]<b[3]); }

static inline mve_pred16_t vcmpgeq_f32(float32x4_t a, float32x4_t b)
{ return _MVE_P32(a[0]>=b[0], a[1]>=b[1], a[2]>=b[2], a[3]>=b[3]); }

static inline mve_pred16_t vcmpleq_f32(float32x4_t a, float32x4_t b)
{ return _MVE_P32(a[0]<=b[0], a[1]<=b[1], a[2]<=b[2], a[3]<=b[3]); }

static inline mve_pred16_t vcmpeqq_f32(float32x4_t a, float32x4_t b)
{ return _MVE_P32(a[0]==b[0], a[1]==b[1], a[2]==b[2], a[3]==b[3]); }

static inline mve_pred16_t vcmpneq_f32(float32x4_t a, float32x4_t b)
{ return _MVE_P32(a[0]!=b[0], a[1]!=b[1], a[2]!=b[2], a[3]!=b[3]); }

// float32x4 vs scalar
static inline mve_pred16_t vcmpgtq_n_f32(float32x4_t a, float s)
{ return vcmpgtq_f32(a, vdupq_n_f32(s)); }
static inline mve_pred16_t vcmpltq_n_f32(float32x4_t a, float s)
{ return vcmpltq_f32(a, vdupq_n_f32(s)); }
static inline mve_pred16_t vcmpleq_n_f32(float32x4_t a, float s)
{ return vcmpleq_f32(a, vdupq_n_f32(s)); }
static inline mve_pred16_t vcmpgeq_n_f32(float32x4_t a, float s)
{ return vcmpgeq_f32(a, vdupq_n_f32(s)); }

// int32x4
static inline mve_pred16_t vcmpgtq_s32(int32x4_t a, int32x4_t b)
{ return _MVE_P32(a[0]>b[0], a[1]>b[1], a[2]>b[2], a[3]>b[3]); }

static inline mve_pred16_t vcmpltq_s32(int32x4_t a, int32x4_t b)
{ return _MVE_P32(a[0]<b[0], a[1]<b[1], a[2]<b[2], a[3]<b[3]); }

static inline mve_pred16_t vcmpgeq_s32(int32x4_t a, int32x4_t b)
{ return _MVE_P32(a[0]>=b[0], a[1]>=b[1], a[2]>=b[2], a[3]>=b[3]); }

static inline mve_pred16_t vcmpleq_s32(int32x4_t a, int32x4_t b)
{ return _MVE_P32(a[0]<=b[0], a[1]<=b[1], a[2]<=b[2], a[3]<=b[3]); }

static inline mve_pred16_t vcmpeqq_s32(int32x4_t a, int32x4_t b)
{ return _MVE_P32(a[0]==b[0], a[1]==b[1], a[2]==b[2], a[3]==b[3]); }

// int16x8
static inline mve_pred16_t vcmpgtq_s16(int16x8_t a, int16x8_t b) {
	return _MVE_P16(a[0]>b[0],a[1]>b[1],a[2]>b[2],a[3]>b[3],
                  a[4]>b[4],a[5]>b[5],a[6]>b[6],a[7]>b[7]);
}

static inline mve_pred16_t vcmpeqq_s16(int16x8_t a, int16x8_t b) {
	return _MVE_P16(a[0]==b[0],a[1]==b[1],a[2]==b[2],a[3]==b[3],
                  a[4]==b[4],a[5]==b[5],a[6]==b[6],a[7]==b[7]);
}

// -------------------------------------------------------
// Predicate logic
// -------------------------------------------------------
#define vpand(a, b)   ((mve_pred16_t)((a) & (b)))
#define vpor(a, b)    ((mve_pred16_t)((a) | (b)))
#define vpnot(a)      ((mve_pred16_t)(~(a)))

// -------------------------------------------------------
// Predicated select: vpselq
// Selects from 'a' where pred lane is active, else from 'b'.
// (Mirrors real MVE: vpselq(trueVal, falseVal, pred))
// -------------------------------------------------------
static inline float32x4_t vpselq_f32(float32x4_t a, float32x4_t b, mve_pred16_t pred) {
    return (float32x4_t){
        _MVE_PRED32_ACTIVE(pred, 0) ? a[0] : b[0],
        _MVE_PRED32_ACTIVE(pred, 1) ? a[1] : b[1],
        _MVE_PRED32_ACTIVE(pred, 2) ? a[2] : b[2],
        _MVE_PRED32_ACTIVE(pred, 3) ? a[3] : b[3],
    };
}

static inline int32x4_t vpselq_s32(int32x4_t a, int32x4_t b, mve_pred16_t pred) {
    return (int32x4_t){
        _MVE_PRED32_ACTIVE(pred, 0) ? a[0] : b[0],
        _MVE_PRED32_ACTIVE(pred, 1) ? a[1] : b[1],
        _MVE_PRED32_ACTIVE(pred, 2) ? a[2] : b[2],
        _MVE_PRED32_ACTIVE(pred, 3) ? a[3] : b[3],
    };
}

static inline int16x8_t vpselq_s16(int16x8_t a, int16x8_t b, mve_pred16_t pred) {
    return (int16x8_t){
        _MVE_PRED16_ACTIVE(pred, 0) ? a[0] : b[0],
        _MVE_PRED16_ACTIVE(pred, 1) ? a[1] : b[1],
        _MVE_PRED16_ACTIVE(pred, 2) ? a[2] : b[2],
        _MVE_PRED16_ACTIVE(pred, 3) ? a[3] : b[3],
        _MVE_PRED16_ACTIVE(pred, 4) ? a[4] : b[4],
        _MVE_PRED16_ACTIVE(pred, 5) ? a[5] : b[5],
        _MVE_PRED16_ACTIVE(pred, 6) ? a[6] : b[6],
        _MVE_PRED16_ACTIVE(pred, 7) ? a[7] : b[7],
    };
}

// -------------------------------------------------------
// Predicated merge arithmetic  (_m suffix = merge)
// Inactive lanes keep value from 'inactive'.
// -------------------------------------------------------
static inline float32x4_t vaddq_m_f32(float32x4_t inactive,
                                       float32x4_t a, float32x4_t b,
                                       mve_pred16_t pred)
{ return vpselq_f32(vaddq_f32(a, b), inactive, pred); }

static inline float32x4_t vsubq_m_f32(float32x4_t inactive,
                                       float32x4_t a, float32x4_t b,
                                       mve_pred16_t pred)
{ return vpselq_f32(vsubq_f32(a, b), inactive, pred); }

static inline float32x4_t vmulq_m_f32(float32x4_t inactive,
                                       float32x4_t a, float32x4_t b,
                                       mve_pred16_t pred)
{ return vpselq_f32(vmulq_f32(a, b), inactive, pred); }

static inline float32x4_t vfmaq_m_f32(float32x4_t inactive,
                                       float32x4_t a, float32x4_t b,
                                       float32x4_t c, mve_pred16_t pred)
{ return vpselq_f32(vfmaq_f32(a, b, c), inactive, pred); }

static inline int32x4_t vaddq_m_s32(int32x4_t inactive,
                                     int32x4_t a, int32x4_t b,
                                     mve_pred16_t pred)
{ return vpselq_s32(vaddq_s32(a, b), inactive, pred); }

static inline int32x4_t vsubq_m_s32(int32x4_t inactive,
                                     int32x4_t a, int32x4_t b,
                                     mve_pred16_t pred)
{ return vpselq_s32(vsubq_s32(a, b), inactive, pred); }

// -------------------------------------------------------
// Type conversion
// -------------------------------------------------------
static inline float32x4_t vcvtq_f32_s32(int32x4_t v) {
    return (float32x4_t){(float)v[0], (float)v[1], (float)v[2], (float)v[3]};
}

static inline int32x4_t vcvtq_s32_f32(float32x4_t v) {
    return (int32x4_t){(int32_t)v[0], (int32_t)v[1], (int32_t)v[2], (int32_t)v[3]};
}

// -------------------------------------------------------
// Horizontal reductions
// -------------------------------------------------------
// Sum all lanes -> scalar
static inline float    vaddvq_f32(float32x4_t v) { return v[0]+v[1]+v[2]+v[3]; }
static inline int32_t  vaddvq_s32(int32x4_t v)   { return v[0]+v[1]+v[2]+v[3]; }

// Min/max across all lanes (init = starting value, e.g. +inf / -inf)
static inline float vminnmvq_f32(float init, float32x4_t v) {
    float r = init;
    for (int i = 0; i < 4; i++) if (v[i] < r) r = v[i];
    return r;
}

static inline float vmaxnmvq_f32(float init, float32x4_t v) {
    float r = init;
    for (int i = 0; i < 4; i++) if (v[i] > r) r = v[i];
    return r;
}

// -------------------------------------------------------
// Element-wise min/max  (float: NaN-propagating "nm" variants)
// -------------------------------------------------------
static inline float32x4_t vminnmq_f32(float32x4_t a, float32x4_t b)
{ return vpselq_f32(a, b, vcmpltq_f32(a, b)); }

static inline float32x4_t vmaxnmq_f32(float32x4_t a, float32x4_t b)
{ return vpselq_f32(a, b, vcmpgtq_f32(a, b)); }

static inline int32x4_t vminq_s32(int32x4_t a, int32x4_t b)
{ return vpselq_s32(a, b, vcmpltq_s32(a, b)); }

static inline int32x4_t vmaxq_s32(int32x4_t a, int32x4_t b)
{ return vpselq_s32(a, b, vcmpgtq_s32(a, b)); }

#endif // MVE_X86_FULL_H
