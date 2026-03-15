/***************************************************************************
--   mve.h - Portable MVE header: real arm_mve.h on ARM, emulation on x86 
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
 Include this in all programs instead of arm_mve.h or mve_x86_full.h directly.
 On ARM Cortex-M55/M85 (__ARM_FEATURE_MVE is defined by the compiler):
   -> includes the real <arm_mve.h>
 On x86 / anything else:
   -> includes our GCC-vector-extension emulation
*/
#pragma once

#if defined(__ARM_FEATURE_MVE)
#  include <arm_mve.h>
#else
#  include "mve_x86_full.h"
#endif

// -------------------------------------------------------
// Compatibility additions (not in GCC's arm_mve.h but
// standard in the MVE ACLE spec, or trivially derived).
// Defined here so all platforms get them.
// -------------------------------------------------------

// Predicate logic - mve_pred16_t is uint16_t, so & / | / ~ work
// natively on both real ARM and x86.  Give them friendly names.
#ifndef vpand
#  define vpand(a, b)  ((mve_pred16_t)((a) & (b)))
#endif
#ifndef vpor
#  define vpor(a, b)   ((mve_pred16_t)((a) | (b)))
#endif
#ifndef vpnot
#  define vpnot(a)     ((mve_pred16_t)(~(uint16_t)(a)))
#endif

// The following are missing from GCC 14's arm_mve.h for floats.
// Only add them on the ARM path; the x86 emulation header already has them.
#if defined(__ARM_FEATURE_MVE)

// Horizontal float sum via lane extraction -> VADDV on real MVE.
static inline float vaddvq_f32(float32x4_t v) {
    return vgetq_lane_f32(v, 0) + vgetq_lane_f32(v, 1)
         + vgetq_lane_f32(v, 2) + vgetq_lane_f32(v, 3);
}

// Horizontal float min/max: vminnmavq/vmaxnmavq are the generic forms in
// arm_mve.h that accept float accumulators.
static inline float vminnmvq_f32(float init, float32x4_t v)
{ return vminnmavq(init, v); }

static inline float vmaxnmvq_f32(float init, float32x4_t v)
{ return vmaxnmavq(init, v); }

#endif // __ARM_FEATURE_MVE

// Helper macros not in the standard ACLE but used by our demos.
// Defined here so they work on both platforms.
//
// Test whether lane n is active in a 32-bit-element predicate.
// MVE nibble encoding: lane n occupies bits [(n*4)+3 : n*4]; bit n*4 is the active bit.
#ifndef _MVE_PRED32_ACTIVE
#  define _MVE_PRED32_ACTIVE(p, n)  (((p) >> ((n) * 4)) & 1u)
#endif

// Test whether lane n is active in a 16-bit-element predicate.
// MVE pair encoding: lane n occupies bits [(n*2)+1 : n*2]; bit n*2 is the active bit.
#ifndef _MVE_PRED16_ACTIVE
#  define _MVE_PRED16_ACTIVE(p, n)  (((p) >> ((n) * 2)) & 1u)
#endif
