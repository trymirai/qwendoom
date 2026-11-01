# DOOM for the qwendoom machine: doomgeneric (third_party, GPL-2.0) + doom/patches + our platform
# layer and libc, built into one RV32IM+Zba ELF (build/doom.elf). doom/split.py reads the objects,
# finds everything sim_main / P_Ticker can reach and writes the linker script that puts it in
# ROM / RAM / DROM; the rest goes to HOST (see ARCHITECTURE.md and docs/doom.md).
#
#   make -C doom            build/doom.elf (+ build/doom.map, build/doom-split.json)
#   make -C doom patches    regenerate doom/patches/ from an edited tree in build/doom-edit/

ROOT  := $(abspath ..)
BUILD := $(ROOT)/build
SRC   := $(BUILD)/doom-src
OBJ   := $(BUILD)/doom-obj
EDIT  := $(BUILD)/doom-edit
DG    := $(ROOT)/third_party/doomgeneric/doomgeneric
LLVM  := /opt/homebrew/opt/llvm/bin
CC    := $(LLVM)/clang
LD    := /opt/homebrew/bin/ld.lld
PY    := $(ROOT)/.venv/bin/python

# no compressed instructions; relaxation with a global pointer (small data within 2 KB of gp
# takes one instruction); per-function / per-object sections so the split can place each one;
# no global merging (it would fuse host and sim variables into one section)
ARCH   := --target=riscv32-unknown-elf -march=rv32im_zba -mabi=ilp32
CFLAGS := $(ARCH) -O2 -ffunction-sections -fdata-sections -ffreestanding -nostdlibinc -fno-common \
          -mrelax -msmall-data-limit=8 -mllvm -enable-global-merge=false -g0 -MMD \
          -isystem libc/include -I. -I$(ROOT)/native -I$(ROOT)/native/host
DOOMDEFS := -DNORMALUNIX -DLINUX -D_DEFAULT_SOURCE -DDOOMGENERIC_RESX=320 -DDOOMGENERIC_RESY=200

# doomgeneric's objects (its Makefile's list) without the platform files we replace: i_system,
# i_video, s_sound, doomgeneric.c and the X11 frontend (qd_platform.c / qd_sim.c instead)
DOOM_SRCS := dummy am_map doomdef doomstat dstrings d_event d_items d_iwad d_loop d_main d_mode \
  d_net f_finale f_wipe g_game hu_lib hu_stuff info i_cdmus i_endoom i_joystick i_scale i_sound \
  i_timer memio m_argv m_bbox m_cheat m_config m_controls m_fixed m_menu m_misc m_random p_ceilng \
  p_doors p_enemy p_floor p_inter p_lights p_map p_maputl p_mobj p_plats p_pspr p_saveg p_setup \
  p_sight p_spec p_switch p_telept p_tick p_user r_bsp r_data r_draw r_main r_plane r_segs r_sky \
  r_things sha1 sounds statdump st_lib st_stuff tables v_video wi_stuff w_checksum w_file w_main \
  w_wad z_zone w_file_stdc i_input
QD_SRCS   := qd_platform qd_sim
LIBC_SRCS := string stdlib stdio libgcc

DOOM_OBJS := $(DOOM_SRCS:%=$(OBJ)/doom/%.o)
QD_OBJS   := $(QD_SRCS:%=$(OBJ)/qd/%.o)
LIBC_OBJS := $(LIBC_SRCS:%=$(OBJ)/libc/%.o) $(OBJ)/libc/crt0.o
OBJS      := $(DOOM_OBJS) $(QD_OBJS) $(LIBC_OBJS)
PATCHES   := $(sort $(wildcard patches/*.patch))

MAKEFLAGS += --no-builtin-rules
.SUFFIXES:
.PHONY: all clean patches
all: $(BUILD)/doom.elf

# the patched source tree: a fresh copy of doomgeneric with every patch applied
$(SRC)/.stamp: $(PATCHES) $(wildcard $(DG)/*.c $(DG)/*.h)
	rm -rf $(SRC) && mkdir -p $(SRC)
	cp $(DG)/*.c $(DG)/*.h $(SRC)/
	for p in $(PATCHES); do patch -s -p1 -d $(SRC) < $$p || exit 1; done
	touch $@

$(OBJ)/doom/%.o: $(SRC)/.stamp
	@mkdir -p $(@D)
	@echo cc $*.c; $(CC) $(CFLAGS) $(DOOMDEFS) -w -I$(SRC) -c $(SRC)/$*.c -o $@

$(OBJ)/qd/%.o: %.c qd_doom.h $(SRC)/.stamp
	@mkdir -p $(@D)
	@echo cc $<; $(CC) $(CFLAGS) $(DOOMDEFS) -Wall -I$(SRC) -c $< -o $@

$(OBJ)/libc/%.o: libc/src/%.c
	@mkdir -p $(@D)
	@echo cc $<; $(CC) $(CFLAGS) -Wall -c $< -o $@

$(OBJ)/libc/crt0.o: libc/src/crt0.S
	@mkdir -p $(@D)
	$(CC) $(ARCH) -mrelax -c $< -o $@

# the split: sim reachability over the objects' relocations -> the linker script
$(OBJ)/doom.ld: $(OBJS) split.py
	@echo split.py analyze; $(PY) split.py analyze --out-ld $@ --out-json $(BUILD)/doom-split.json $(OBJS)

$(BUILD)/doom.elf: $(OBJ)/doom.ld $(OBJS)
	@echo ld.lld $@; $(LD) -m elf32lriscv -T $(OBJ)/doom.ld --relax --relax-gp --gc-sections -e _start -u sim_main \
	  -Map $(BUILD)/doom.map -o $@ $(OBJS)
	$(PY) split.py check $@ $(BUILD)/doom-split.json

-include $(OBJS:.o=.d)

# doom/patches/<file>.patch from build/doom-edit/ (a copy of doomgeneric you edited)
patches:
	@for f in $(EDIT)/*.c $(EDIT)/*.h; do b=$$(basename $$f); \
	  if ! cmp -s $$f $(DG)/$$b; then \
	    diff -u --label a/$$b --label b/$$b $(DG)/$$b $$f > patches/$$b.patch; \
	    echo "patches/$$b.patch"; fi; done

clean:
	rm -rf $(SRC) $(OBJ) $(BUILD)/doom.elf $(BUILD)/doom.map $(BUILD)/doom-split.json
