/* The RV32IM+Zba interpreter body, included twice by qd_host.c:
     QD_SIM 0  run_host: the host context, unrestricted over the four regions, dirty-tracks the
               RAM it writes, flags DROM writes, serves the host ecalls;
     QD_SIM 1  run_sim:  the sim context: fetch only from ROM, loads from RAM / DROM, stores to
               RAM (above the null guard), aligned accesses, the YIELD / exit ecalls, csrr time
               pops the input queue; it also counts the circuit's trace tokens.
   Direct-threaded: each decoded instruction (dinst) holds its handler's label; the PC lives as a
   pointer into the current region's decoded array (dbase / pbase) and is rebuilt only when needed
   (links, errors, events). Instruction counting is exact; the budget is checked on taken jumps
   (one shared exit, L_BUDGET, so that every handler keeps its own dispatch branch). */

#if QD_SIM
#define FN run_sim
#else
#define FN run_host
#endif

static int FN(qd_host *h, cpu_t *c, uint64_t limit, const void *const **labels_out)
{
    static const void *const labels[U_NUM] = {
        [U_ILLEGAL] = &&L_ILLEGAL, [U_LUI] = &&L_LI, [U_AUIPC] = &&L_LI, [U_JAL] = &&L_JAL,
        [U_JALR] = &&L_JALR, [U_BEQ] = &&L_BEQ, [U_BNE] = &&L_BNE, [U_BLT] = &&L_BLT,
        [U_BGE] = &&L_BGE, [U_BLTU] = &&L_BLTU, [U_BGEU] = &&L_BGEU, [U_LB] = &&L_LB,
        [U_LH] = &&L_LH, [U_LW] = &&L_LW, [U_LBU] = &&L_LBU, [U_LHU] = &&L_LHU, [U_SB] = &&L_SB,
        [U_SH] = &&L_SH, [U_SW] = &&L_SW, [U_ADDI] = &&L_ADDI, [U_SLTI] = &&L_SLTI,
        [U_SLTIU] = &&L_SLTIU, [U_XORI] = &&L_XORI, [U_ORI] = &&L_ORI, [U_ANDI] = &&L_ANDI,
        [U_SLLI] = &&L_SLLI, [U_SRLI] = &&L_SRLI, [U_SRAI] = &&L_SRAI, [U_ADD] = &&L_ADD,
        [U_SUB] = &&L_SUB, [U_SLL] = &&L_SLL, [U_SLT] = &&L_SLT, [U_SLTU] = &&L_SLTU,
        [U_XOR] = &&L_XOR, [U_SRL] = &&L_SRL, [U_SRA] = &&L_SRA, [U_OR] = &&L_OR,
        [U_AND] = &&L_AND, [U_MUL] = &&L_MUL, [U_MULH] = &&L_MULH, [U_MULHSU] = &&L_MULHSU,
        [U_MULHU] = &&L_MULHU, [U_DIV] = &&L_DIV, [U_DIVU] = &&L_DIVU, [U_REM] = &&L_REM,
        [U_REMU] = &&L_REMU, [U_SH1ADD] = &&L_SH1ADD, [U_SH2ADD] = &&L_SH2ADD,
        [U_SH3ADD] = &&L_SH3ADD, [U_FENCE] = &&L_FENCE, [U_ECALL] = &&L_ECALL,
        [U_EBREAK] = &&L_EBREAK, [U_CSR_TIME] = &&L_CSR_TIME, [U_CSR_ZERO] = &&L_CSR_ZERO,
        [U_CSR_MISA] = &&L_CSR_MISA, [U_BADJUMP] = &&L_BADJUMP, [U_END] = &&L_END,
    };
    if (labels_out) {
        *labels_out = labels;
        return 0;
    }

    uint32_t *const x = c->x;
    uint8_t *const M = h->m;
    uint64_t ic = c->icount;
    const dinst *ip, *dbase;
    uint32_t pbase, ea, t;
#if QD_SIM
    uint64_t tok = c->tokens;
    uint32_t n63 = c->n63;  /* commits since the last J / R (the R refresh comes at 63) */
    qd_sim_stats *st = &c->stats;
#endif

#define PC_OF(p) (pbase + (uint32_t)((p) - dbase) * 4u)
#define SAVE(p) do { c->pc = PC_OF(p); c->icount = ic; } while (0)
#if QD_SIM
#define SAVE_SIM() do { c->tokens = tok; c->n63 = n63; } while (0)
#define DISPATCH() do { if (n63 == 63) { tok++; n63 = 0; } n63++; st->ops[ip->uop]++; ic++; goto *ip->op; } while (0)
#define VTOK(v) (tok += ip->rd == 32 ? 1 : vlen(v))
#define JTOK() (tok++, n63 = 0)
#else
#define SAVE_SIM() do { } while (0)
#define DISPATCH() do { ic++; goto *ip->op; } while (0)
#define VTOK(v) ((void)0)
#define JTOK() ((void)0)
#endif
#define NEXT() do { ip++; DISPATCH(); } while (0)
#define WR(v) do { uint32_t v_ = (v); x[ip->rd] = v_; VTOK(v_); NEXT(); } while (0)
#define FAIL(...) do { SAVE(ip); SAVE_SIM(); fail(h, PC_OF(ip), __VA_ARGS__); return QD_EV_ERROR; } while (0)
    /* jump to address t (any region this context may fetch from) */
#define JUMP_TO(tgt) do {                                                                         \
        uint32_t t_ = (tgt);                                                                      \
        if (!locate(h, QD_SIM, t_, &ip, &dbase, &pbase))                                          \
            FAIL("%s jump to %s", QD_SIM ? "sim" : "host", describe(h, t_));                      \
        JTOK();                                                                                   \
        if (__builtin_expect(ic >= limit, 0)) goto L_BUDGET;                                      \
        DISPATCH();                                                                               \
    } while (0)
#define BRANCH(cond) do {                                                                         \
        if (cond) {                                                                               \
            ip += ip->imm; JTOK();                                                                \
            if (__builtin_expect(ic >= limit, 0)) goto L_BUDGET;                                  \
            DISPATCH();                                                                           \
        }                                                                                         \
        VTOK_NT(); NEXT();                                                                        \
    } while (0)
#if QD_SIM
#define VTOK_NT() (tok++)
#else
#define VTOK_NT() ((void)0)
#endif

#if QD_SIM
    /* loads: aligned, from RAM above the null guard or DROM; stores: aligned, RAM above the guard */
#define LD_CHECK(a, n) do {                                                                       \
        if (((a) & ((n) - 1)) || !((a) - QD_NULL_GUARD < QD_RAM_SIZE - QD_NULL_GUARD              \
                                   || ((a) - QD_DROM_BASE < QD_DROM_SIZE && ++st->drom_loads)))  \
            FAIL("sim load (%d bytes) from %s", (n), describe(h, (a)));                           \
        st->loads++;                                                                              \
    } while (0)
#define ST_CHECK(a, n) do {                                                                       \
        if (((a) & ((n) - 1)) || (a) - QD_NULL_GUARD >= QD_RAM_SIZE - QD_NULL_GUARD)              \
            FAIL("sim store (%d bytes) to %s", (n), describe(h, (a)));                            \
        st->stores++;                                                                             \
    } while (0)
#define ST_TOK(n) (tok += (n) == 4 ? 1 : 4)
#else
#define LD_CHECK(a, n) do {                                                                       \
        if ((a) - QD_HOST_BASE >= QD_HOST_SIZE && !host_load_ok(h, (a)))                         \
            FAIL("host load (%d bytes) from %s", (n), describe(h, (a)));                          \
    } while (0)
#define ST_CHECK(a, n) do {                                                                       \
        if ((a) - QD_HOST_BASE >= QD_HOST_SIZE && !host_store_slow(h, (a), (n)))                 \
            FAIL("host store (%d bytes) to %s", (n), describe(h, (a)));                           \
    } while (0)
#define ST_TOK(n) ((void)0)
#endif

    if (!locate(h, QD_SIM, c->pc, &ip, &dbase, &pbase)) {
        fail(h, c->pc, "%s context resumed at a bad pc %#x", QD_SIM ? "sim" : "host", c->pc);
        return QD_EV_ERROR;
    }
    DISPATCH();

L_LI:     WR((uint32_t)ip->imm);
L_ADDI:
#if QD_SIM
    if (ip->rd == 2 && x[ip->rs1] + ip->imm < st->min_sp)
        st->min_sp = x[ip->rs1] + ip->imm;
#endif
          WR(x[ip->rs1] + ip->imm);
L_SLTI:   WR((int32_t)x[ip->rs1] < ip->imm);
L_SLTIU:  WR(x[ip->rs1] < (uint32_t)ip->imm);
L_XORI:   WR(x[ip->rs1] ^ ip->imm);
L_ORI:    WR(x[ip->rs1] | ip->imm);
L_ANDI:   WR(x[ip->rs1] & ip->imm);
L_SLLI:   WR(x[ip->rs1] << ip->imm);
L_SRLI:   WR(x[ip->rs1] >> ip->imm);
L_SRAI:   WR((uint32_t)((int32_t)x[ip->rs1] >> ip->imm));
L_ADD:    WR(x[ip->rs1] + x[ip->rs2]);
L_SUB:    WR(x[ip->rs1] - x[ip->rs2]);
L_SLL:    WR(x[ip->rs1] << (x[ip->rs2] & 31));
L_SLT:    WR((int32_t)x[ip->rs1] < (int32_t)x[ip->rs2]);
L_SLTU:   WR(x[ip->rs1] < x[ip->rs2]);
L_XOR:    WR(x[ip->rs1] ^ x[ip->rs2]);
L_SRL:    WR(x[ip->rs1] >> (x[ip->rs2] & 31));
L_SRA:    WR((uint32_t)((int32_t)x[ip->rs1] >> (x[ip->rs2] & 31)));
L_OR:     WR(x[ip->rs1] | x[ip->rs2]);
L_AND:    WR(x[ip->rs1] & x[ip->rs2]);
L_SH1ADD: WR((x[ip->rs1] << 1) + x[ip->rs2]);
L_SH2ADD: WR((x[ip->rs1] << 2) + x[ip->rs2]);
L_SH3ADD: WR((x[ip->rs1] << 3) + x[ip->rs2]);
L_MUL:    WR(x[ip->rs1] * x[ip->rs2]);
L_MULH:   WR((uint32_t)(((int64_t)(int32_t)x[ip->rs1] * (int32_t)x[ip->rs2]) >> 32));
L_MULHSU: WR((uint32_t)(((int64_t)(int32_t)x[ip->rs1] * (int64_t)(uint64_t)x[ip->rs2]) >> 32));
L_MULHU:  WR((uint32_t)(((uint64_t)x[ip->rs1] * x[ip->rs2]) >> 32));
L_DIV:
L_DIVU:
L_REM:
L_REMU: {
        uint32_t a = x[ip->rs1], b = x[ip->rs2], r = rv_div(ip->uop, a, b);
        x[ip->rd] = r;
#if QD_SIM
        tok += ip->rd == 32 ? 1 : 1 + (8 - div_start(ip->uop, a, b)) + vlen(r);
#endif
        NEXT();
    }

L_LB:  ea = x[ip->rs1] + ip->imm; LD_CHECK(ea, 1); WR((uint32_t)(int32_t)(int8_t)M[ea]);
L_LBU: ea = x[ip->rs1] + ip->imm; LD_CHECK(ea, 1); WR(M[ea]);
L_LH:  ea = x[ip->rs1] + ip->imm; LD_CHECK(ea, 2); WR((uint32_t)(int32_t)ld16s(M + ea));
L_LHU: ea = x[ip->rs1] + ip->imm; LD_CHECK(ea, 2); WR(ld16(M + ea));
L_LW:  ea = x[ip->rs1] + ip->imm; LD_CHECK(ea, 4); WR(ld32(M + ea));
L_SB:  ea = x[ip->rs1] + ip->imm; ST_CHECK(ea, 1); M[ea] = (uint8_t)x[ip->rd]; ST_TOK(1); NEXT();
L_SH:  ea = x[ip->rs1] + ip->imm; ST_CHECK(ea, 2); st16(M + ea, (uint16_t)x[ip->rd]); ST_TOK(2); NEXT();
L_SW:  ea = x[ip->rs1] + ip->imm; ST_CHECK(ea, 4); st32(M + ea, x[ip->rd]); ST_TOK(4); NEXT();

L_BEQ:  BRANCH(x[ip->rs1] == x[ip->rs2]);
L_BNE:  BRANCH(x[ip->rs1] != x[ip->rs2]);
L_BLT:  BRANCH((int32_t)x[ip->rs1] < (int32_t)x[ip->rs2]);
L_BGE:  BRANCH((int32_t)x[ip->rs1] >= (int32_t)x[ip->rs2]);
L_BLTU: BRANCH(x[ip->rs1] < x[ip->rs2]);
L_BGEU: BRANCH(x[ip->rs1] >= x[ip->rs2]);

L_JAL:
    x[ip->rd] = PC_OF(ip) + 4;
    ip += ip->imm;
    JTOK();
    if (__builtin_expect(ic >= limit, 0)) goto L_BUDGET;
    DISPATCH();
L_JALR:
    t = (x[ip->rs1] + ip->imm) & ~1u;
    x[ip->rd] = PC_OF(ip) + 4;
    JUMP_TO(t);

L_FENCE: VTOK_NT(); NEXT();
L_CSR_ZERO: WR(0);
L_CSR_MISA: WR(QD_MISA);
L_CSR_TIME:
#if QD_SIM
    {   /* the input port: the next queued word (pokes, then 0); its value record is injected */
        uint32_t v = c->inq_pos < c->inq_len ? c->inq[c->inq_pos++] : 0;
        st->input_words++;
        WR(v);
    }
#else
    WR(0);
#endif

L_ECALL: {
        SAVE(ip);
        SAVE_SIM();
#if QD_SIM
        uint32_t a7 = x[17], a0 = x[10];
        if (a7 == 1 && a0 == QD_YIELD_BYTE) {
            c->tokens++;  /* BYTE(256) */
            c->pc += 4;
            return QD_EV_YIELD;
        }
        if (a7 == 1 && (a0 & 511) < 256) {  /* putchar: a BYTE token, allowed in the circuit */
            sim_putchar(h, a0 & 0xFF);
            tok = ++c->tokens;
            NEXT();
        }
        if (a7 == 93) {
            c->tokens++;  /* HALT */
            guest_exit_error(h, 1, x[10], x[11], x[12]);
            return QD_EV_ERROR;
        }
        fail(h, c->pc, "sim ecall a7=%u a0=%u (the circuit has only YIELD and exit)", a7, a0);
        return QD_EV_ERROR;
#else
        int ev = host_ecall(h, c);
        ic = c->icount;
        if (ev == QD_EV_CONTINUE)
            NEXT();
        c->pc += 4;
        return ev;
#endif
    }

L_BUDGET:
    SAVE(ip);
    SAVE_SIM();
    return QD_EV_BUDGET;

L_EBREAK:  FAIL("ebreak");
L_ILLEGAL: FAIL("illegal instruction %#010x", ld32(M + PC_OF(ip)));
L_BADJUMP: FAIL("branch / jump out of %s", QD_SIM ? "ROM" : "the code");
L_END:     FAIL("ran off the end of the code");

#undef FN
#undef PC_OF
#undef SAVE
#undef SAVE_SIM
#undef DISPATCH
#undef VTOK
#undef VTOK_NT
#undef JTOK
#undef NEXT
#undef WR
#undef FAIL
#undef JUMP_TO
#undef BRANCH
#undef LD_CHECK
#undef ST_CHECK
#undef ST_TOK
}
