/* The qwendoom address map (see ARCHITECTURE.md), shared by every native component. Must agree
   with qwendoom/spec.py. */
#ifndef QD_MEM_H
#define QD_MEM_H

#include <stdint.h>

#define QD_RAM_BASE   0x00000000u
#define QD_RAM_SIZE   0x00040000u /* 256 KB: model RAM (load/store) */
#define QD_DROM_BASE  0x00080000u
#define QD_DROM_SIZE  0x00080000u /* 512 KB: model DROM (load only) */
#define QD_ROM_BASE   0x00100000u
#define QD_ROM_SIZE   0x00040000u /* 256 KB: model ROM (code, fetch only) */
#define QD_HOST_BASE  0x01000000u
#define QD_HOST_SIZE  0x04000000u /* 64 MB: host interpreter only */

#define QD_SIM_STACK_TOP 0x3000u  /* sim sp starts here and grows down to 0x100 */
#define QD_YIELD_BYTE    256u     /* ecall a7=1, a0=256 -> BYTE(256): the tick is done */
#define QD_CSR_INPUT     0xC01u   /* csrr rd, time: the host-injected input port */

/* Backing stores for the four regions; one instance is shared by the host interpreter, the
   trace machine and the trace decoder. */
typedef struct {
    uint8_t *ram;  /* QD_RAM_SIZE bytes */
    uint8_t *drom; /* QD_DROM_SIZE bytes */
    uint8_t *rom;  /* QD_ROM_SIZE bytes (raw RISC-V instructions) */
    uint8_t *host; /* QD_HOST_SIZE bytes */
} qd_mem;

#endif
