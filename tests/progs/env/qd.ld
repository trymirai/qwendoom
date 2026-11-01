/* The qwendoom address map (ARCHITECTURE.md) for test programs: code in the ROM at 0x100000
   (256 KB), read-only data in the DROM at 0x80000 (512 KB), data and bss in the RAM from 0x3000
   (as the sim's globals), the stack below 0x3000 (crt0 sets sp = __stack_top; a program may
   define its own, e.g. the RAM's top for deep recursion). */
ENTRY(_start)
MEMORY
{
  RAM (rw) : ORIGIN = 0x3000, LENGTH = 0x40000 - 0x3000
  DROM (r) : ORIGIN = 0x80000, LENGTH = 512K
  ROM (rx) : ORIGIN = 0x100000, LENGTH = 256K
}
SECTIONS
{
  .text : { *(.text.init) *(.text .text.*) } > ROM
  .rodata : { *(.rodata .rodata.* .srodata .srodata.*) } > DROM
  .data : { *(.data .data.* .sdata .sdata.*) } > RAM
  .bss (NOLOAD) : { *(.bss .bss.* .sbss .sbss.*) *(COMMON) . = ALIGN(16); __bss_end = .; } > RAM
  PROVIDE(__stack_top = 0x3000);
}
