// +build amd64

#include "textflag.h"

// func crc32CLMUL(data []byte) uint32
TEXT ·crc32CLMUL(SB), NOSPLIT, $0-28
    MOVQ data_base+0(FP), SI   // SI = data pointer
    MOVQ data_len+8(FP), CX    // CX = length
    MOVL $0xFFFFFFFF, AX       // Initial CRC value

    // Check if we have enough data for CLMUL
    CMPQ CX, $256
    JL   fallback

    // Load polynomial constant
    MOVQ $0x1DB710641, BX
    MOVQ BX, X15

    // Process 128 bytes at a time
clmul_loop:
    CMPQ CX, $128
    JL   clmul_remainder

    // Load 4 x 32-byte chunks
    VMOVDQU 0(SI), Y0
    VMOVDQU 32(SI), Y1
    VMOVDQU 64(SI), Y2
    VMOVDQU 96(SI), Y3

    // Fold into running CRC
    MOVD    AX, X4
    VPCLMULQDQ $0x00, X15, X4, X5
    VPCLMULQDQ $0x11, X15, X4, X6
    VPXOR   X5, X0, X0
    VPXOR   X6, X0, X0

    // Continue folding
    VPCLMULQDQ $0x00, X15, X0, X5
    VPCLMULQDQ $0x11, X15, X0, X6
    VPXOR   X5, X1, X1
    VPXOR   X6, X1, X1

    VPCLMULQDQ $0x00, X15, X1, X5
    VPCLMULQDQ $0x11, X15, X1, X6
    VPXOR   X5, X2, X2
    VPXOR   X6, X2, X2

    VPCLMULQDQ $0x00, X15, X2, X5
    VPCLMULQDQ $0x11, X15, X2, X6
    VPXOR   X5, X3, X3
    VPXOR   X6, X3, X3

    // Extract final CRC for this iteration
    MOVD    X3, AX

    ADDQ $128, SI
    SUBQ $128, CX
    JMP  clmul_loop

clmul_remainder:
    // Process 16-byte chunks
    CMPQ CX, $16
    JL   clmul_small

clmul_16:
    MOVOU  (SI), X0
    MOVD   AX, X1
    VPCLMULQDQ $0x00, X15, X1, X2
    VPCLMULQDQ $0x11, X15, X1, X3
    VPXOR  X2, X0, X0
    VPXOR  X3, X0, X0
    MOVD   X0, AX

    ADDQ $16, SI
    SUBQ $16, CX
    CMPQ CX, $16
    JGE  clmul_16

clmul_small:
    // Fall through to scalar processing
    JMP scalar_crc32

fallback:
scalar_crc32:
    // Scalar CRC32 for remaining bytes
    CMPQ CX, $0
    JE   done

scalar_loop:
    MOVBLZX (SI), BX
    XORL    BX, AX

    // CRC32 table lookup would go here
    // For simplicity, using direct computation
    MOVL $8, DX
bit_loop:
    SHRL $1, AX
    JNC  no_poly
    XORL $0xEDB88320, AX
no_poly:
    DECL DX
    JNZ  bit_loop

    INCQ SI
    DECQ CX
    JNZ  scalar_loop

done:
    NOTL AX
    MOVL AX, ret+24(FP)
    VZEROUPPER
    RET

// func crc32cCLMUL(data []byte) uint32
TEXT ·crc32cCLMUL(SB), NOSPLIT, $0-28
    MOVQ data_base+0(FP), SI
    MOVQ data_len+8(FP), CX
    MOVL $0xFFFFFFFF, AX

    // Castagnoli polynomial: 0x82F63B78
    MOVQ $0x82F63B78, BX
    MOVQ BX, X15

    // Similar to crc32CLMUL but with Castagnoli polynomial
    CMPQ CX, $256
    JL   crc32c_scalar

crc32c_loop:
    CMPQ CX, $128
    JL   crc32c_remainder

    VMOVDQU 0(SI), Y0
    VMOVDQU 32(SI), Y1
    VMOVDQU 64(SI), Y2
    VMOVDQU 96(SI), Y3

    MOVD    AX, X4
    VPCLMULQDQ $0x00, X15, X4, X5
    VPCLMULQDQ $0x11, X15, X4, X6
    VPXOR   X5, X0, X0
    VPXOR   X6, X0, X0

    VPCLMULQDQ $0x00, X15, X0, X5
    VPCLMULQDQ $0x11, X15, X0, X6
    VPXOR   X5, X1, X1
    VPXOR   X6, X1, X1

    VPCLMULQDQ $0x00, X15, X1, X5
    VPCLMULQDQ $0x11, X15, X1, X6
    VPXOR   X5, X2, X2
    VPXOR   X6, X2, X2

    VPCLMULQDQ $0x00, X15, X2, X5
    VPCLMULQDQ $0x11, X15, X2, X6
    VPXOR   X5, X3, X3
    VPXOR   X6, X3, X3

    MOVD    X3, AX

    ADDQ $128, SI
    SUBQ $128, CX
    JMP  crc32c_loop

crc32c_remainder:
    CMPQ CX, $16
    JL   crc32c_scalar

    MOVOU  (SI), X0
    MOVD   AX, X1
    VPCLMULQDQ $0x00, X15, X1, X2
    VPCLMULQDQ $0x11, X15, X1, X3
    VPXOR  X2, X0, X0
    VPXOR  X3, X0, X0
    MOVD   X0, AX

    ADDQ $16, SI
    SUBQ $16, CX
    CMPQ CX, $16
    JGE  crc32c_remainder

crc32c_scalar:
    CMPQ CX, $0
    JE   crc32c_done

crc32c_scalar_loop:
    MOVBLZX (SI), BX
    XORL    BX, AX

    MOVL $8, DX
crc32c_bit_loop:
    SHRL $1, AX
    JNC  crc32c_no_poly
    XORL $0x82F63B78, AX
crc32c_no_poly:
    DECL DX
    JNZ  crc32c_bit_loop

    INCQ SI
    DECQ CX
    JNZ  crc32c_scalar_loop

crc32c_done:
    NOTL AX
    MOVL AX, ret+24(FP)
    VZEROUPPER
    RET
