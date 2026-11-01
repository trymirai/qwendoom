# build/libqdrun.dylib: the model-mode hot loop (one sim tick in C; qwendoom/runner.py loads it).
# It links against build/libqdengine.dylib and build/libqdtrace.dylib through their public
# headers; both are found next to it at run time (@rpath = @loader_path).
ROOT := $(abspath ../..)
BUILD := $(ROOT)/build
LIB := $(BUILD)/libqdrun.dylib
DEPS := $(BUILD)/libqdengine.dylib $(BUILD)/libqdtrace.dylib

CC ?= cc
CFLAGS ?= -O2
CFLAGS += -std=gnu11 -Wall -Wextra -Wno-unused-function -Wno-unused-variable -fPIC \
          -I.. -I../engine -I../trace

.PHONY: all clean
all: $(LIB)

$(LIB): qd_run.c qd_run.h ../engine/qd_engine.h ../trace/qd_trace.h ../trace/qd_tokens.h \
        ../qd_mem.h Makefile | $(DEPS)
	$(CC) $(CFLAGS) -dynamiclib -install_name @rpath/libqdrun.dylib -o $@.tmp qd_run.c \
	    -L$(BUILD) -lqdengine -lqdtrace -Wl,-rpath,@loader_path -lpthread
	mv -f $@.tmp $@  # (a new file: processes that have the old one loaded keep it)

# the libraries it links against (their own Makefiles keep them current: `make native`)
$(BUILD)/libqdengine.dylib:
	$(MAKE) -C ../engine
$(BUILD)/libqdtrace.dylib:
	$(MAKE) -C ../trace

clean:
	rm -f $(LIB)
