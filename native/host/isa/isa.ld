/* riscv-tests on the qwendoom host computer (see riscv_test.h): sim code in ROM, host code in
   HOST, test data in sim RAM. */
ENTRY(_start)
MEMORY
{
  RAM  (rw)  : ORIGIN = 0x3000, LENGTH = 0x3d000
  ROM  (rx)  : ORIGIN = 0x100000, LENGTH = 0x40000
  HOST (rwx) : ORIGIN = 0x1000000, LENGTH = 0x100000
}
SECTIONS
{
  .rom : { *(.text.sim) } > ROM
  .text : { *(.text.host) *(.text .text.*) } > HOST
  .data : { *(.data .data.* .sdata .sdata.* .bss .bss.* .sbss .sbss.*) } > RAM
  __global_pointer$ = 0x3800;
}
