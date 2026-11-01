# build/libqdtrace.dylib: the reference traced machine, the trace decoder and the lockstep checker.
#   make -C native/trace           build the library
#   make -C native/trace tokens    regenerate qd_tokens.h from chipformer.rv.spec
#   make -C native/trace sanitize  build/libqdtrace-san.dylib (ASan + UBSan); run the tests on it:
#     DYLD_INSERT_LIBRARIES=$(cc -print-resource-dir)/lib/darwin/libclang_rt.asan_osx_dynamic.dylib \
#     ASAN_OPTIONS=detect_leaks=0 QD_TRACE_LIB=build/libqdtrace-san.dylib \
#     .venv/bin/python -m pytest tests/test_trace_*.py
ROOT := $(abspath ../..)
BUILD := $(ROOT)/build
LIB := $(BUILD)/libqdtrace.dylib
SAN := $(BUILD)/libqdtrace-san.dylib
PYTHON ?= $(if $(wildcard $(ROOT)/.venv/bin/python),$(ROOT)/.venv/bin/python,uv run python)

CC ?= cc
CFLAGS ?= -O3 -std=c11 -Wall -Wextra -Wno-unused-function -fPIC
SRCS := qd_ref.c qd_decoder.c qd_check.c
HDRS := qd_trace.h qd_isa.h qd_tokens.h ../qd_mem.h

.PHONY: all tokens check-tokens sanitize clean
all: $(LIB)

$(LIB): $(SRCS) $(HDRS) Makefile | $(BUILD)
	$(CC) $(CFLAGS) -I.. -dynamiclib -install_name @rpath/libqdtrace.dylib -o $@.tmp $(SRCS)
	mv -f $@.tmp $@  # (a new file: processes that have the old one loaded keep it)

sanitize: $(SAN)
$(SAN): $(SRCS) $(HDRS) Makefile | $(BUILD)
	$(CC) -O1 -g -std=c11 -Wall -Wextra -Wno-unused-function -fPIC -fno-omit-frame-pointer \
	    -fsanitize=address,undefined -fno-sanitize-recover=undefined -I.. -dynamiclib \
	    -install_name @rpath/libqdtrace-san.dylib -o $@ $(SRCS)

$(BUILD):
	mkdir -p $@

tokens:
	$(PYTHON) gen_tokens.py

check-tokens:
	$(PYTHON) gen_tokens.py --check

clean:
	rm -f $(LIB) $(SAN)
