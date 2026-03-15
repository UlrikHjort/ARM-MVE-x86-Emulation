# MVE x86 Emulation

Run ARM Cortex-M55 **MVE (Helium)** intrinsic code on x86 without modification.

## What it is

`mve_x86_full.h` emulates the ARM MVE ACLE intrinsic API using GCC/Clang
`__attribute__((vector_size))`, which compiles to SSE on x86.

`mve.h` is the portable include: on real ARM (`-mcpu-cortex-m55`) it routes
to the compiler's `<arm_mve.h>`; on x86 it uses the emulation.

The predicate bit layout follows the MVE spec - nibble per 32-bit lane, pair per 16-bit lane.

## Files

| File | Purpose |
|---|---|
| `mve.h` | Portable include - real MVE on ARM, emulation on x86 |
| `mve_x86_full.h` | x86 emulation of the MVE ACLE intrinsic API |
| `mandelbrot.c` | Mandelbrot renderer using MVE tail predication |
| `test_predicates.c` | Predicate encoding, `vctp32q/16q`, `vpselq`, logic ops |
| `test_dot_product.c` | Vectorised dot product with tail predication |
| `test_saxpy.c` | SAXPY with partial-chunk handling |
| `test_stats.c` | Array sum, min, max, clamp, int↔float conversion |


## Build

```
make          # x86 (gcc + clang) + ARM cross-compile verification
make run      # run all tests (gcc)
make run-all  # gcc vs clang side-by-side summary
make arm      # cross-compile only (object files, no link)
make clean

If arm cross compilation set ARM_TOOLCHAIN in the Makefile to point to the location og the arm toolchain
```

Requires `gcc`, `clang`, and `arm-none-eabi-gcc` with Cortex-M55 support.
The ARM target compiles with `-mcpu-cortex-m55 -mfloat-abi-hard` and uses
the real `<arm_mve.h>`, confirming the intrinsic names are MVE-correct.

## Predicate encoding

For 32-bit elements (`float32x4_t`, `int32x4_t`), the 16-bit predicate
encodes one nibble per lane:

```
bits [3:0]   -> lane 0
bits [7:4]   -> lane 1
bits [11:8]  -> lane 2
bits [15:12] -> lane 3
```

So `vctp32q(3)` - `0x0FFF` (lanes 0–2 active), and a comparison where
only lanes 1 and 3 match returns `0xF0F0`.
