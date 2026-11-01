# build/libqdengine.dylib: the inference engine (Apple clang, arm64 macOS).
# -ffp-contract=off: the compiler must not fuse a * b + c where that would round differently
# from the reference; step.c fuses explicitly where it is exact (bf16 x bf16 products).
ROOT := ../..
BUILD := $(ROOT)/build
CC ?= cc
CFLAGS ?= -O3 -mcpu=native
CFLAGS += -std=c11 -Wall -Wextra -Wno-unused-parameter -ffp-contract=off -fno-fast-math -fPIC
SRCS := model.c step.c json.c
HDRS := qd_engine.h internal.h json.h
# OUT: another place for the library (a build under test; QD_ENGINE_LIB points Python at it)
OUT ?= $(BUILD)/libqdengine.dylib

.PHONY: all clean
all: $(OUT)

$(OUT): $(SRCS) $(HDRS) Makefile
	@mkdir -p $(dir $(OUT))
	$(CC) $(CFLAGS) -dynamiclib -install_name @rpath/libqdengine.dylib -o $@.tmp $(SRCS) -lpthread
	mv -f $@.tmp $@  # (a new file: processes that have the old one loaded keep it)

clean:
	rm -f $(OUT)
