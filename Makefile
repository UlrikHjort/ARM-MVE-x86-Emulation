# -----------------------------------------------------------------------
# Makefile - MVE emulation on x86 + cross-compile verification for ARM
#
# Targets:
#   make         -> build all x86 binaries (gcc + clang) and ARM objects
#   make x86     -> build x86 binaries only
#   make arm     -> cross-compile to ARM Cortex-M55 object files (.o)
#   make run     -> run all x86 tests (gcc build)
#   make run-all -> run all x86 tests with both gcc and clang
#   make clean   -> remove build/
#
# 2026 By Ulrik Hørlyk Hjort
# -----------------------------------------------------------------------

# ---- x86 compilers ----
GCC   = gcc
CLANG = /usr/bin/clang

X86_CFLAGS = -O2 -Wall -Wextra -msse4.2 -lm

# ---- ARM cross-compiler (Cortex-M55 = MVE float+int, __ARM_FEATURE_MVE=3) ----
ARM_TOOLCHAIN = <PATH TO ARM GNU TOOLCHAIN> 
ARM_CC        = $(ARM_TOOLCHAIN)/arm-none-eabi-gcc
ARM_CFLAGS    = -mcpu=cortex-m55 -mfloat-abi=hard -O2 -Wall -Wextra -c

# ---- Sources ----
# Test programs (have a main, print PASS/FAIL, return exit code)
TESTS = test_predicates test_dot_product test_saxpy test_stats

# Demo programs (visual output, no pass/fail)
DEMOS = main mandelbrot

ALL_PROGS = $(DEMOS) $(TESTS)

# ---- Output directories ----
BUILD_GCC   = build/gcc
BUILD_CLANG = build/clang
BUILD_ARM   = build/arm

DIRS = $(BUILD_GCC) $(BUILD_CLANG) $(BUILD_ARM)

# ---- Common headers (any change rebuilds everything) ----
HEADERS = mve.h mve_x86_full.h

# ======================================================================
.PHONY: all x86 arm run run-all clean

all: x86 arm

# ---- x86 builds ----
x86: $(addprefix $(BUILD_GCC)/,  $(ALL_PROGS)) \
     $(addprefix $(BUILD_CLANG)/,$(ALL_PROGS))

$(BUILD_GCC)/% : %.c $(HEADERS) | $(BUILD_GCC)
	$(GCC)   $(X86_CFLAGS) -o $@ $<

$(BUILD_CLANG)/% : %.c $(HEADERS) | $(BUILD_CLANG)
	$(CLANG) $(X86_CFLAGS) -o $@ $<

# ---- ARM cross-compile (object files only - no linking for bare metal) ----
arm: $(addprefix $(BUILD_ARM)/,$(addsuffix .o,$(ALL_PROGS)))
	@echo ""
	@echo "ARM MVE compilation: all OK (real arm_mve.h intrinsics verified)"

$(BUILD_ARM)/%.o : %.c $(HEADERS) | $(BUILD_ARM)
	@printf "  ARM  %-30s  " "$<"
	@$(ARM_CC) $(ARM_CFLAGS) -o $@ $< && echo "OK" || echo "FAIL"

# ---- Run x86 tests ----
run: $(addprefix $(BUILD_GCC)/,$(TESTS))
	@echo "========================================================"
	@echo " Running tests (GCC x86)"
	@echo "========================================================"
	@overall=0; \
	for t in $(TESTS); do \
		echo ""; \
		echo "--- $$t ---"; \
		$(BUILD_GCC)/$$t || overall=1; \
	done; \
	echo ""; \
	echo "--- demos (visual) ---"; \
	$(BUILD_GCC)/main; \
	$(BUILD_GCC)/mandelbrot; \
	exit $$overall

run-all: $(addprefix $(BUILD_GCC)/,$(ALL_PROGS)) \
         $(addprefix $(BUILD_CLANG)/,$(ALL_PROGS))
	@echo "========================================================"
	@echo " Running tests: GCC vs Clang"
	@echo "========================================================"
	@for t in $(TESTS); do \
		echo ""; \
		echo "=== $$t ==="; \
		printf "  [gcc  ] "; $(BUILD_GCC)/$$t   | tail -1; \
		printf "  [clang] "; $(BUILD_CLANG)/$$t | tail -1; \
	done

# ---- Utility ----
$(DIRS):
	mkdir -p $@

clean:
	rm -rf build/
