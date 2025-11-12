// +build amd64

#include "textflag.h"

// func xorBytesAVX2(dst, src1, src2 []byte)
TEXT ·xorBytesAVX2(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DI    // DI = dst pointer
    MOVQ src1_base+24(FP), SI  // SI = src1 pointer
    MOVQ src2_base+48(FP), DX  // DX = src2 pointer
    MOVQ dst_len+8(FP), CX     // CX = length

    // Check if length >= 256 for AVX2 processing
    CMPQ CX, $256
    JL   avx2_remainder

avx2_loop:
    // Process 256 bytes (8 x 32 bytes) per iteration
    CMPQ CX, $256
    JL   avx2_remainder

    // Load and XOR 8 x 32-byte chunks
    VMOVDQU 0(SI), Y0
    VMOVDQU 0(DX), Y1
    VPXOR   Y0, Y1, Y2
    VMOVDQU Y2, 0(DI)

    VMOVDQU 32(SI), Y0
    VMOVDQU 32(DX), Y1
    VPXOR   Y0, Y1, Y2
    VMOVDQU Y2, 32(DI)

    VMOVDQU 64(SI), Y0
    VMOVDQU 64(DX), Y1
    VPXOR   Y0, Y1, Y2
    VMOVDQU Y2, 64(DI)

    VMOVDQU 96(SI), Y0
    VMOVDQU 96(DX), Y1
    VPXOR   Y0, Y1, Y2
    VMOVDQU Y2, 96(DI)

    VMOVDQU 128(SI), Y0
    VMOVDQU 128(DX), Y1
    VPXOR   Y0, Y1, Y2
    VMOVDQU Y2, 128(DI)

    VMOVDQU 160(SI), Y0
    VMOVDQU 160(DX), Y1
    VPXOR   Y0, Y1, Y2
    VMOVDQU Y2, 160(DI)

    VMOVDQU 192(SI), Y0
    VMOVDQU 192(DX), Y1
    VPXOR   Y0, Y1, Y2
    VMOVDQU Y2, 192(DI)

    VMOVDQU 224(SI), Y0
    VMOVDQU 224(DX), Y1
    VPXOR   Y0, Y1, Y2
    VMOVDQU Y2, 224(DI)

    ADDQ $256, SI
    ADDQ $256, DX
    ADDQ $256, DI
    SUBQ $256, CX
    JMP  avx2_loop

avx2_remainder:
    // Process 32-byte chunks
    CMPQ CX, $32
    JL   avx2_small

avx2_32:
    VMOVDQU (SI), Y0
    VMOVDQU (DX), Y1
    VPXOR   Y0, Y1, Y2
    VMOVDQU Y2, (DI)

    ADDQ $32, SI
    ADDQ $32, DX
    ADDQ $32, DI
    SUBQ $32, CX
    CMPQ CX, $32
    JGE  avx2_32

avx2_small:
    // Process 8-byte chunks with SSE
    CMPQ CX, $8
    JL   scalar_remainder

sse_8:
    MOVQ    (SI), X0
    MOVQ    (DX), X1
    PXOR    X1, X0
    MOVQ    X0, (DI)

    ADDQ $8, SI
    ADDQ $8, DX
    ADDQ $8, DI
    SUBQ $8, CX
    CMPQ CX, $8
    JGE  sse_8

scalar_remainder:
    // Process remaining bytes
    CMPQ CX, $0
    JE   done

scalar_loop:
    MOVB (SI), AX
    XORB (DX), AX
    MOVB AX, (DI)

    INCQ SI
    INCQ DX
    INCQ DI
    DECQ CX
    JNZ  scalar_loop

done:
    VZEROUPPER  // Clear upper bits of YMM registers
    RET

// func xorBytesSSSE3(dst, src1, src2 []byte)
TEXT ·xorBytesSSSE3(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DI
    MOVQ src1_base+24(FP), SI
    MOVQ src2_base+48(FP), DX
    MOVQ dst_len+8(FP), CX

ssse3_loop:
    // Process 16 bytes at a time with SSE
    CMPQ CX, $16
    JL   ssse3_remainder

    MOVOU  (SI), X0
    MOVOU  (DX), X1
    PXOR   X1, X0
    MOVOU  X0, (DI)

    ADDQ $16, SI
    ADDQ $16, DX
    ADDQ $16, DI
    SUBQ $16, CX
    JMP  ssse3_loop

ssse3_remainder:
    // Process 8 bytes
    CMPQ CX, $8
    JL   ssse3_scalar

    MOVQ    (SI), X0
    MOVQ    (DX), X1
    PXOR    X1, X0
    MOVQ    X0, (DI)

    ADDQ $8, SI
    ADDQ $8, DX
    ADDQ $8, DI
    SUBQ $8, CX

ssse3_scalar:
    // Process remaining bytes
    CMPQ CX, $0
    JE   ssse3_done

ssse3_scalar_loop:
    MOVB (SI), AX
    XORB (DX), AX
    MOVB AX, (DI)

    INCQ SI
    INCQ DX
    INCQ DI
    DECQ CX
    JNZ  ssse3_scalar_loop

ssse3_done:
    RET
