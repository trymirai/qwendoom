// Switch statements compiled to jump tables (in the DROM): a small stack-machine bytecode
// interpreter, and a dense switch on character classes.
#include "env/rt.h"

enum { PUSH, ADD, SUB, MUL, DUP, SWAP, JNZ, DEC, PRINT, HALT, AND, XOR, SHL, OVER, DROP, NOP };
static const uint8_t prog[] = {
    PUSH, 1, PUSH, 10,                            // acc, counter
    DUP, PRINT,                                   // loop (pc 4): print the counter
    SWAP, PUSH, 3, MUL, PUSH, 7, XOR, OVER, ADD,  // acc = (acc * 3 ^ 7) + counter
    PUSH, 1, SHL, PUSH, 1, SUB, PUSH, 127, AND,   // acc = ((acc << 1) - 1) & 127
    NOP, SWAP, DEC, DUP, JNZ, 4,                  // until the counter is 0
    DROP, DUP, PRINT, HALT,
};

static int32_t run(void) {
    int32_t st[16];
    int sp = 0, pc = 0;
    for (;;) {
        uint8_t op = prog[pc++];
        switch (op) {
        case PUSH: st[sp++] = prog[pc++]; break;
        case ADD: sp--; st[sp - 1] += st[sp]; break;
        case SUB: sp--; st[sp - 1] -= st[sp]; break;
        case MUL: sp--; st[sp - 1] *= st[sp]; break;
        case DUP: st[sp] = st[sp - 1]; sp++; break;
        case SWAP: { int32_t t = st[sp - 1]; st[sp - 1] = st[sp - 2]; st[sp - 2] = t; break; }
        case JNZ: { uint8_t t = prog[pc++]; if (st[--sp]) pc = t; break; }
        case DEC: st[sp - 1]--; break;
        case PRINT: qd_putdec(st[--sp]); qd_putchar(' '); break;
        case HALT: return sp ? st[sp - 1] : 0;
        case AND: sp--; st[sp - 1] &= st[sp]; break;
        case XOR: sp--; st[sp - 1] ^= st[sp]; break;
        case SHL: sp--; st[sp - 1] <<= st[sp]; break;
        case OVER: st[sp] = st[sp - 2]; sp++; break;
        case DROP: sp--; break;
        default: break;
        }
    }
}

static int klass(char c) {
    switch (c) {
    case 'a': case 'e': case 'i': case 'o': case 'u': return 1;
    case '0': case '1': case '2': case '3': case '4':
    case '5': case '6': case '7': case '8': case '9': return 2;
    case ' ': return 3;
    case '.': case ',': case '!': case '?': return 4;
    case 'q': return 5;
    case 'x': return 6;
    case 'z': return 7;
    default: return 0;
    }
}

int main(void) {
    int32_t r = run();
    const char *s = "the quick brown fox, 1984 zebras! exit?";
    uint32_t counts[8] = {0};
    for (const char *p = s; *p; p++) counts[klass(*p)]++;
    qd_puts("| switch ");
    qd_putdec(r);
    for (int i = 0; i < 8; i++) {
        qd_putchar(' ');
        qd_putdec((int32_t)counts[i]);
    }
    qd_putchar('\n');
    return 0;
}
