# The host interpreter: build/libqdhost.dylib (loaded by qwendoom/host.py) and build/qdhost (a
# small CLI for timing and smoke runs).
ROOT  := $(abspath ../..)
BUILD := $(ROOT)/build
CC    := cc
# -enable-tail-merge=false keeps one indirect branch per handler (direct threading): merged
# dispatch tails cost ~30% (measured 966 -> 1305 host MIPS, best of 6, on an M5 Max)
CFLAGS := -O2 -std=gnu11 -Wall -Wextra -Wno-unused-parameter -Wno-unused-label -I. -I.. -fPIC \
          -mllvm -enable-tail-merge=false

all: $(BUILD)/libqdhost.dylib $(BUILD)/qdhost

$(BUILD)/libqdhost.dylib: qd_host.c qd_interp.inc qd_host.h qd_host_abi.h ../qd_mem.h
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) -dynamiclib -install_name @rpath/libqdhost.dylib -o $@.tmp qd_host.c
	mv -f $@.tmp $@  # (a new file: processes that have the old one loaded keep it)

$(BUILD)/qdhost: qdhost_cli.c qd_host.c qd_interp.inc qd_host.h qd_host_abi.h ../qd_mem.h
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) -o $@ qdhost_cli.c qd_host.c

# riscv-tests (rv32ui, rv32um, rv32uzba) built for the host computer twice: as sim code in ROM
# and as host code (isa/riscv_test.h); tests/test_host_isa.py runs them
ISA    := $(ROOT)/third_party/riscv-tests/isa
ISAOUT := $(BUILD)/host-isa
RVCC   := /opt/homebrew/opt/llvm/bin/clang --target=riscv32-unknown-elf -march=rv32im_zba \
          -mabi=ilp32 -nostdlib -fuse-ld=lld -Wl,--no-relax -T isa/isa.ld -Iisa -I$(ISA)/macros/scalar
ISA_TESTS := $(addprefix rv32ui/,simple add addi and andi auipc beq bge bgeu blt bltu bne jal jalr \
               lb lbu lh lhu lw ld_st lui or ori sb sh sw st_ld sll slli slt slti sltiu sltu sra \
               srai srl srli sub xor xori) \
             $(addprefix rv32um/,div divu mul mulh mulhsu mulhu rem remu) \
             $(addprefix rv32uzba/,sh1add sh2add sh3add)
ISA_ELFS := $(foreach t,$(ISA_TESTS),$(ISAOUT)/sim-$(subst /,-,$(t)).elf $(ISAOUT)/host-$(subst /,-,$(t)).elf)

isa-tests: $(ISA_ELFS) $(ISAOUT)/sim-selftest_fail.elf $(ISAOUT)/host-selftest_fail.elf

$(ISAOUT)/%-selftest_fail.elf: isa/selftest_fail.S isa/riscv_test.h isa/isa.ld
	@mkdir -p $(@D)
	@$(RVCC) -DQD_ISA_SIM=$(if $(filter sim,$*),1,0) -o $@ $<

$(ISAOUT)/sim-%.elf: isa/riscv_test.h isa/isa.ld
	@mkdir -p $(@D)
	@$(RVCC) -DQD_ISA_SIM=1 -o $@ $(ISA)/$(subst -,/,$*).S

$(ISAOUT)/host-%.elf: isa/riscv_test.h isa/isa.ld
	@mkdir -p $(@D)
	@$(RVCC) -DQD_ISA_SIM=0 -o $@ $(ISA)/$(subst -,/,$*).S

clean:
	rm -rf $(BUILD)/libqdhost.dylib $(BUILD)/qdhost $(ISAOUT)

.PHONY: all clean isa-tests
