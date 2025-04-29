import math
import torch
import triton
import triton.language as tl

@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
    }
)
@triton.jit
def _fwd_kernel_d_2(
    #even_part,
    #stride_evenb,
    #stride_evenh,
    #stride_evenm, 
    Q,
    K,
    V,
    Bias,
    Out,
    #Lse,
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_bb,
    stride_bh,
    stride_bm,
    stride_ob,
    stride_oh,
    stride_om,
    nheads,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    headdim,
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    BIAS_TYPE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    # off_b = tl.program_id(1)
    # off_h = tl.program_id(2)
    # off_hb = off_b * nheads + off_h
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_m_hash = start_m * 1 + tl.arange(0, 1)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, 128)
    offs_odd = tl.arange(0, 64)
    offs_even = tl.arange(64, 128)
    #odd_ptr = even_part + off_b * stride_evenb + off_h * stride_evenh + (offs_m_hash[:, None] * stride_evenm +offs_odd[None, :])
    #even_ptr = even_part + off_b * stride_evenb + off_h * stride_evenh + (offs_m_hash[:, None] * stride_evenm +offs_even[None, :])
    odd_v = tl.arange(0, 64)#tl.load(odd_ptr)
    even_v = tl.arange(64, 128)#tl.load(even_ptr)
    #print(odd_v)
    # Initialize pointers to Q, K, V
    # Adding parenthesis around indexing might use int32 math instead of int64 math?
    # https://github.com/openai/triton/issues/741
    # I'm seeing a tiny bit of difference (5-7us)
    '''
    q_ptrs = (
        Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qm + odd_v)
    )
    simq_ptrs = (
        Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qm + even_v)
    )
    k_ptrs = (
        K + off_b * stride_kb + off_h * stride_kh + (offs_n[:, None] * stride_kn + even_v)
    )
    simk_ptrs = (
        K + off_b * stride_kb + off_h * stride_kh + (offs_n[:, None] * stride_kn + odd_v)
    )
    v_ptrs = (
        V + off_b * stride_vb + off_h * stride_vh + (offs_n[:, None] * stride_vn + offs_d[None, :])
    )
    if BIAS_TYPE == "vector":
        b_ptrs = Bias + off_b * stride_bb + off_h * stride_bh + offs_n
    elif BIAS_TYPE == "matrix":
        b_ptrs = (
            Bias
            + off_b * stride_bb
            + off_h * stride_bh
            + (offs_m[:, None] * stride_bm + offs_n[None, :])
        )
    # initialize pointer to m and l
    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)
    # load q: it will stay in SRAM throughout
    # [2022-10-30] TD: Triton bug - in the case of EVEN_M=True and EVEN_N=False, if we just call
    # tl.load(q_ptrs), we get the wrong output!
    if EVEN_M & EVEN_N:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs)
            simq = tl.load(simq_ptrs)
        else:
            q = tl.load(q_ptrs, mask=offs_d[None, :] < 64, other=0.0)
            simq = tl.load(simq_ptrs, mask=offs_d[None, :] < 64, other=0.0)
    else:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
            simq = tl.load(simq_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
        else:
            q = tl.load(
                q_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < 64), other=0.0
            )
            simq = tl.load(
                simq_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < 64), other=0.0
            )
    q = (q/2 + simq/2).to(dtype=tl.float16)
    # loop over k, v and update accumulator
    end_n = seqlen_k if not IS_CAUSAL else tl.minimum((start_m + 1) * BLOCK_M, seqlen_k)
    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        if EVEN_N & EVEN_M:  # If we just do "if EVEN_N", there seems to be some race condition
            if EVEN_HEADDIM:
                k = tl.load(k_ptrs + start_n * stride_kn)
                simk = tl.load(simk_ptrs + start_n * stride_kn)
            else:
                k = tl.load(k_ptrs + start_n * stride_kn, mask=offs_d[None, :] < 64, other=0.0)
                simk = tl.load(simk_ptrs + start_n * stride_kn, mask=offs_d[None, :] < 64, other=0.0)
        else:
            if EVEN_HEADDIM:
                k = tl.load(
                    k_ptrs + start_n * stride_kn,
                    mask=(start_n + offs_n)[:, None] < seqlen_k,
                    other=0.0,
                )
                simk = tl.load(
                    simk_ptrs + start_n * stride_kn,
                    mask=(start_n + offs_n)[:, None] < seqlen_k,
                    other=0.0,
                )
            else:
                k = tl.load(
                    k_ptrs + start_n * stride_kn,
                    mask=(start_n + offs_n)[:, None] < seqlen_k,
                    other=0.0,
                )
                simk = tl.load(
                    simk_ptrs + start_n * stride_kn,
                    mask=(start_n + offs_n)[:, None] < seqlen_k,
                    other=0.0,
                )
        k = k + simk
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))
        # Trying to combine the two masks seem to make the result wrong
        if not EVEN_N:  # Need to mask out otherwise the softmax is wrong
            qk += tl.where((start_n + offs_n)[None, :] < seqlen_k, 0, float("-inf"))
        if IS_CAUSAL:
            qk += tl.where(offs_m[:, None] >= (start_n + offs_n)[None, :], 0, float("-inf"))
        if BIAS_TYPE != "none":
            if BIAS_TYPE == "vector":
                if EVEN_N:
                    bias = tl.load(b_ptrs + start_n).to(tl.float32)
                else:
                    bias = tl.load(
                        b_ptrs + start_n, mask=(start_n + offs_n) < seqlen_k, other=0.0
                    ).to(tl.float32)
                bias = bias[None, :]
            elif BIAS_TYPE == "matrix":
                if EVEN_M & EVEN_N:
                    bias = tl.load(b_ptrs + start_n).to(tl.float32)
                else:
                    bias = tl.load(
                        b_ptrs + start_n,
                        mask=(offs_m[:, None] < seqlen_q)
                        & ((start_n + offs_n)[None, :] < seqlen_k),
                        other=0.0,
                    ).to(tl.float32)
            # Slightly faster to multiply the softmax_scale in the tl.exp below since the compiler
            # can then fuse the mult and add into an fma instruction. But if we have bias we need to
            # to multiply with softmax_scale here.
            qk = qk * softmax_scale + bias
            m_ij = tl.maximum(tl.max(qk, 1), lse_i)
            p = tl.exp(qk - m_ij[:, None])
        else:
            m_ij = tl.maximum(tl.max(qk, 1) * softmax_scale, lse_i)
            p = tl.exp(qk * softmax_scale - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        # scale acc_o
        acc_o_scale = tl.exp(m_i - m_ij)

        # # -- update output accumulator acc_o --
        acc_o = acc_o * acc_o_scale[:, None]

        if EVEN_N & EVEN_M:  # If we just do "if EVEN_N", there seems to be some race condition
            if EVEN_HEADDIM:
                v = tl.load(v_ptrs + start_n * stride_vn)
            else:
                v = tl.load(v_ptrs + start_n * stride_vn, mask=offs_d[None, :] < headdim, other=0.0)
        else:
            if EVEN_HEADDIM:
                v = tl.load(
                    v_ptrs + start_n * stride_vn,
                    mask=(start_n + offs_n)[:, None] < seqlen_k,
                    other=0.0,
                )
            else:
                v = tl.load(
                    v_ptrs + start_n * stride_vn,
                    mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                    other=0.0,
                )
        p = p.to(v.dtype)
        acc_o += tl.dot(p, v)

        # -- update statistics
        m_i = m_ij
        l_i_new = tl.exp(lse_i - m_ij) + l_ij
        lse_i = m_ij + tl.log(l_i_new)

    o_scale = tl.exp(m_i - lse_i)
    acc_o = acc_o * o_scale[:, None]
    # rematerialize offsets to save registers
    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # write back l and m
    #lse_ptrs = Lse + off_hb * seqlen_q_rounded + offs_m
    #tl.store(lse_ptrs, lse_i)
    # initialize pointers to output
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    out_ptrs = (
        Out
        + off_b * stride_ob
        + off_h * stride_oh
        + (offs_m[:, None] * stride_om + offs_d[None, :])
    )
    if EVEN_M:
        if EVEN_HEADDIM:
            tl.store(out_ptrs, acc_o)
        else:
            tl.store(out_ptrs, acc_o, mask=offs_d[None, :] < headdim)
    else:
        if EVEN_HEADDIM:
            tl.store(out_ptrs, acc_o, mask=offs_m[:, None] < seqlen_q)
        else:
            tl.store(
                out_ptrs, acc_o, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim)
            )
'''
@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
    }
)
@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    Bias,
    Out,
    Lse,
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_bb,
    stride_bh,
    stride_bm,
    stride_ob,
    stride_oh,
    stride_om,
    nheads,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    headdim,
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    BIAS_TYPE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    # off_b = tl.program_id(1)
    # off_h = tl.program_id(2)
    # off_hb = off_b * nheads + off_h
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, 64)
    offs_d1 = tl.arange(0, 64)
    # Initialize pointers to Q, K, V
    # Adding parenthesis around indexing might use int32 math instead of int64 math?
    # https://github.com/openai/triton/issues/741
    # I'm seeing a tiny bit of difference (5-7us)
    q_ptrs = (
        Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qm + offs_d1[None, :])
    )
    k_ptrs = (
        K + off_b * stride_kb + off_h * stride_kh + (offs_n[:, None] * stride_kn + offs_d1[None, :])
    )
    v_ptrs = (
        V + off_b * stride_vb + off_h * stride_vh + (offs_n[:, None] * stride_vn + offs_d[None, :])
    )
    if BIAS_TYPE == "vector":
        b_ptrs = Bias + off_b * stride_bb + off_h * stride_bh + offs_n
    elif BIAS_TYPE == "matrix":
        b_ptrs = (
            Bias
            + off_b * stride_bb
            + off_h * stride_bh
            + (offs_m[:, None] * stride_bm + offs_n[None, :])
        )
    # initialize pointer to m and l
    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)
    # load q: it will stay in SRAM throughout
    # [2022-10-30] TD: Triton bug - in the case of EVEN_M=True and EVEN_N=False, if we just call
    # tl.load(q_ptrs), we get the wrong output!
    if EVEN_M & EVEN_N:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs)
        else:
            q = tl.load(q_ptrs, mask=offs_d1[None, :] < headdim, other=0.0)
    else:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
        else:
            q = tl.load(
                q_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d1[None, :] < headdim), other=0.0
            )
    # loop over k, v and update accumulator
    end_n = seqlen_k if not IS_CAUSAL else tl.minimum((start_m + 1) * BLOCK_M, seqlen_k)
    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        if EVEN_N & EVEN_M:  # If we just do "if EVEN_N", there seems to be some race condition
            if EVEN_HEADDIM:
                k = tl.load(k_ptrs + start_n * stride_kn)
            else:
                k = tl.load(k_ptrs + start_n * stride_kn, mask=offs_d1[None, :] < headdim, other=0.0)
        else:
            if EVEN_HEADDIM:
                k = tl.load(
                    k_ptrs + start_n * stride_kn,
                    mask=(start_n + offs_n)[:, None] < seqlen_k,
                    other=0.0,
                )
            else:
                k = tl.load(
                    k_ptrs + start_n * stride_kn,
                    mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d1[None, :] < headdim),
                    other=0.0,
                )
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))
        # Trying to combine the two masks seem to make the result wrong
        if not EVEN_N:  # Need to mask out otherwise the softmax is wrong
            qk += tl.where((start_n + offs_n)[None, :] < seqlen_k, 0, float("-inf"))
        if IS_CAUSAL:
            qk += tl.where(offs_m[:, None] >= (start_n + offs_n)[None, :], 0, float("-inf"))
        if BIAS_TYPE != "none":
            if BIAS_TYPE == "vector":
                if EVEN_N:
                    bias = tl.load(b_ptrs + start_n).to(tl.float32)
                else:
                    bias = tl.load(
                        b_ptrs + start_n, mask=(start_n + offs_n) < seqlen_k, other=0.0
                    ).to(tl.float32)
                bias = bias[None, :]
            elif BIAS_TYPE == "matrix":
                if EVEN_M & EVEN_N:
                    bias = tl.load(b_ptrs + start_n).to(tl.float32)
                else:
                    bias = tl.load(
                        b_ptrs + start_n,
                        mask=(offs_m[:, None] < seqlen_q)
                        & ((start_n + offs_n)[None, :] < seqlen_k),
                        other=0.0,
                    ).to(tl.float32)
            # Slightly faster to multiply the softmax_scale in the tl.exp below since the compiler
            # can then fuse the mult and add into an fma instruction. But if we have bias we need to
            # to multiply with softmax_scale here.
            qk = qk * softmax_scale + bias
            m_ij = tl.maximum(tl.max(qk, 1), lse_i)
            p = tl.exp(qk - m_ij[:, None])
        else:
            m_ij = tl.maximum(tl.max(qk, 1) * softmax_scale, lse_i)
            p = tl.exp(qk * softmax_scale - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        # scale acc_o
        acc_o_scale = tl.exp(m_i - m_ij)

        # # -- update output accumulator acc_o --
        acc_o = acc_o * acc_o_scale[:, None]

        if EVEN_N & EVEN_M:  # If we just do "if EVEN_N", there seems to be some race condition
            if EVEN_HEADDIM:
                v = tl.load(v_ptrs + start_n * stride_vn)
            else:
                v = tl.load(v_ptrs + start_n * stride_vn, mask=offs_d[None, :] < headdim, other=0.0)
        else:
            if EVEN_HEADDIM:
                v = tl.load(
                    v_ptrs + start_n * stride_vn,
                    mask=(start_n + offs_n)[:, None] < seqlen_k,
                    other=0.0,
                )
            else:
                v = tl.load(
                    v_ptrs + start_n * stride_vn,
                    mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                    other=0.0,
                )
        p = p.to(v.dtype)
        #print(v)
        acc_o += tl.dot(p, v)

        # -- update statistics
        m_i = m_ij
        l_i_new = tl.exp(lse_i - m_ij) + l_ij
        lse_i = m_ij + tl.log(l_i_new)

    o_scale = tl.exp(m_i - lse_i)
    acc_o = acc_o * o_scale[:, None]
    # rematerialize offsets to save registers
    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # write back l and m
    lse_ptrs = Lse + off_hb * seqlen_q_rounded + offs_m
    tl.store(lse_ptrs, lse_i)
    # initialize pointers to output
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    out_ptrs = (
        Out
        + off_b * stride_ob
        + off_h * stride_oh
        + (offs_m[:, None] * stride_om + offs_d[None, :])
    )
    
    if EVEN_M:
        if EVEN_HEADDIM:
            tl.store(out_ptrs, acc_o)
        else:
            tl.store(out_ptrs, acc_o, mask=offs_d[None, :] < headdim)
    else:
        if EVEN_HEADDIM:
            tl.store(out_ptrs, acc_o, mask=offs_m[:, None] < seqlen_q)
        else:
            tl.store(
                out_ptrs, acc_o, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim)
            )

@triton.jit
def _fwd_kernel_flashattn(
    #odd,
    #even,
    #stride_odd_b,
    #stride_odd_h,
    #stride_odd_n,
    #stride_even_b,
    #stride_even_h,
    #stride_even_n,
    even_part,
    Q,
    K,
    V,
    Bias,
    Out,
    #Lse,
    #TMP,  # NOTE: TMP is a scratchpad buffer to workaround a compiler bug
    #even,
    #odd,
    softmax_scale,
    stride_evenb,
    stride_evenh,
    stride_evenm,    
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_bb,
    stride_bh,
    stride_bm,
    stride_ob,
    stride_oh,
    stride_om,
    nheads,
    seqlen_q,
    seqlen_k,
    #seqlen_q_rounded,
    #headdim,
    #CACHE_KEY_SEQLEN_Q,
    #CACHE_KEY_SEQLEN_K,
    BIAS_TYPE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    offs_m_hash = start_m * 1 + tl.arange(0, 1)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_t = tl.arange(0, BLOCK_T)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    offs_e = tl.arange(0, 64)
    #offs_odd = tl.arange(0, 32)
    #offs_even = tl.arange(32, 64)
    #odd_ptr = even_part + off_b * stride_evenb + off_h * stride_evenh + offs_m_hash * stride_evenm +offs_odd
    #even_ptr = even_part + off_b * stride_evenb + off_h * stride_evenh + offs_m_hash * stride_evenm +offs_even
    #oddout_ptr = even + off_b * stride_odd_b + off_h * stride_odd_h + offs_m_hash[:, None] * stride_odd_n +offs_odd[None, :]
    #evenout_ptr = odd + off_b * stride_even_b + off_h * stride_even_h + offs_m_hash[:, None] * stride_even_n +offs_odd[None, :]
    #odd_ptr = odd + offs_d
    #even_ptr = even + offs_d
    #odd_v = tl.load(odd_ptr)
    #even_v = tl.load(even_ptr)
    #tl.store(oddout_ptr,odd_v)
    #tl.store(evenout_ptr,even_v)
    
    #print(oddout_ptr)
    #print(odd_v)
    q_ptrs = (
        Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qm + offs_e[None, :])
    )
    #simq_ptrs = (
    #    Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qm + even_v[None, :])
    #)
    k_ptrs = (
        K + off_b * stride_kb + off_h * stride_kh + (offs_n[:, None] * stride_kn + offs_e[None, :])
    )
    v_ptrs = (
        V + off_b * stride_vb + off_h * stride_vh + (offs_n[:, None] * stride_vn + offs_e[None, :])
    )
    if BIAS_TYPE == "vector":
        b_ptrs = Bias + off_b * stride_bb + off_h * stride_bh + offs_n
    elif BIAS_TYPE == "matrix":
        b_ptrs = (
            Bias
            + off_b * stride_bb
            + off_h * stride_bh
            + (offs_m[:, None] * stride_bm + offs_n[None, :])
        )
    # initialize pointer to m and l
    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    acc_o = tl.zeros([BLOCK_M, 64], dtype=tl.float32)
    # load q: it will stay in SRAM throughout
    # [2022-10-30] TD: Triton bug - in the case of EVEN_M=True and EVEN_N=False, if we just call
    # tl.load(q_ptrs), we get the wrong output!
    if EVEN_M & EVEN_N:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs)
            #simq = tl.load(simq_ptrs)
        else:
            q = tl.load(q_ptrs, mask=offs_d[None, :] < BLOCK_HEADDIM, other=0.0)
            #simq = tl.load(simq_ptrs, mask=offs_d[None, :] < BLOCK_HEADDIM, other=0.0)
    else:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
            #simq = tl.load(simq_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
        else:
            q = tl.load(
                q_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < BLOCK_HEADDIM), other=0.0
            )
            #simq = tl.load(
            #    simq_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < BLOCK_HEADDIM), other=0.0
            #)
    # loop over k, v and update accumulator
    #q = q + simq
    end_n = seqlen_k if not IS_CAUSAL else tl.minimum((start_m + 1) * BLOCK_M, seqlen_k)
    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        if EVEN_N & EVEN_M:  # If we just do "if EVEN_N", there seems to be some race condition
            if EVEN_HEADDIM:
                k = tl.load(k_ptrs + start_n * stride_kn)
            else:
                k = tl.load(k_ptrs + start_n * stride_kn, mask=offs_d[None, :] < BLOCK_HEADDIM, other=0.0)
        else:
            if EVEN_HEADDIM:
                k = tl.load(
                    k_ptrs + start_n * stride_kn,
                    mask=(start_n + offs_n)[:, None] < seqlen_k,
                    other=0.0,
                )
            else:
                k = tl.load(
                    k_ptrs + start_n * stride_kn,
                    mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < BLOCK_HEADDIM),
                    other=0.0,
                )
        #print(k)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))
        # Trying to combine the two masks seem to make the result wrong
        if not EVEN_N:  # Need to mask out otherwise the softmax is wrong
            qk += tl.where((start_n + offs_n)[None, :] < seqlen_k, 0, float("-inf"))
        if IS_CAUSAL:
            qk += tl.where(offs_m[:, None] >= (start_n + offs_n)[None, :], 0, float("-inf"))
        if BIAS_TYPE != "none":
            if BIAS_TYPE == "vector":
                if EVEN_N:
                    bias = tl.load(b_ptrs + start_n).to(tl.float32)
                else:
                    bias = tl.load(
                        b_ptrs + start_n, mask=(start_n + offs_n) < seqlen_k, other=0.0
                    ).to(tl.float32)
                bias = bias[None, :]
            elif BIAS_TYPE == "matrix":
                if EVEN_M & EVEN_N:
                    bias = tl.load(b_ptrs + start_n).to(tl.float32)
                else:
                    bias = tl.load(
                        b_ptrs + start_n,
                        mask=(offs_m[:, None] < seqlen_q)
                        & ((start_n + offs_n)[None, :] < seqlen_k),
                        other=0.0,
                    ).to(tl.float32)
            # Slightly faster to multiply the softmax_scale in the tl.exp below since the compiler
            # can then fuse the mult and add into an fma instruction. But if we have bias we need to
            # to multiply with softmax_scale here.
            qk = qk * softmax_scale + bias
            m_ij = tl.maximum(tl.max(qk, 1), lse_i)
            p = tl.exp(qk - m_ij[:, None])
        else:
            m_ij = tl.maximum(tl.max(qk, 1) * softmax_scale, lse_i)
            p = tl.exp(qk * softmax_scale - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        # scale acc_o
        acc_o_scale = tl.exp(m_i - m_ij)

        # # -- update output accumulator acc_o --
        acc_o = acc_o * acc_o_scale[:, None]

        if EVEN_N & EVEN_M:  # If we just do "if EVEN_N", there seems to be some race condition
            if EVEN_HEADDIM:
                v = tl.load(v_ptrs + start_n * stride_vn)
            else:
                v = tl.load(v_ptrs + start_n * stride_vn, mask=offs_d[None, :] < BLOCK_HEADDIM, other=0.0)
        else:
            if EVEN_HEADDIM:
                v = tl.load(
                    v_ptrs + start_n * stride_vn,
                    mask=(start_n + offs_n)[:, None] < seqlen_k,
                    other=0.0,
                )
            else:
                v = tl.load(
                    v_ptrs + start_n * stride_vn,
                    mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_e[None, :] < BLOCK_HEADDIM),
                    other=0.0,
                )
        p = p.to(v.dtype)
        acc_o += tl.dot(p, v)

        # -- update statistics
        m_i = m_ij
        l_i_new = tl.exp(lse_i - m_ij) + l_ij
        lse_i = m_ij + tl.log(l_i_new)
    
    o_scale = tl.exp(m_i - lse_i)
    acc_o = acc_o * o_scale[:, None]
    start_m = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    out_ptrs = (
        Out
        + off_b * stride_ob
        + off_h * stride_oh
        + (offs_m[:, None] * stride_om + offs_e[None, :])
    )
    tl.store(out_ptrs, acc_o)

@triton.jit
def _fwd_kernel_flash_4_64(
    #odd,
    #even,
    #stride_odd_b,
    #stride_odd_h,
    #stride_odd_n,
    #stride_even_b,
    #stride_even_h,
    #stride_even_n,
    even_part,
    Q,
    K,
    V,
    Bias,
    Out,
    #Lse,
    #TMP,  # NOTE: TMP is a scratchpad buffer to workaround a compiler bug
    #even,
    #odd,
    softmax_scale,
    stride_evenb,
    stride_evenh,
    stride_evenm,    
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_bb,
    stride_bh,
    stride_bm,
    stride_ob,
    stride_oh,
    stride_om,
    nheads,
    seqlen_q,
    seqlen_k,
    #seqlen_q_rounded,
    #headdim,
    #CACHE_KEY_SEQLEN_Q,
    #CACHE_KEY_SEQLEN_K,
    BIAS_TYPE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    offs_m_hash = start_m * 1 + tl.arange(0, 1)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_t = tl.arange(0, BLOCK_T)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    offs_e = tl.arange(0, 64)
    offs_odd = tl.arange(0, 16)
    offs_odd_1 = tl.arange(32, 48)
    offs_even = tl.arange(0, 16)
    offs_even_1 = tl.arange(48, 64)
    odd_ptr = even_part + off_b * stride_evenb + off_h * stride_evenh + offs_m_hash * stride_evenm +offs_odd
    even_ptr = even_part + off_b * stride_evenb + off_h * stride_evenh + offs_m_hash * stride_evenm +offs_even
    odd_ptr_1 = even_part + off_b * stride_evenb + off_h * stride_evenh + offs_m_hash * stride_evenm +offs_odd_1
    even_ptr_1 = even_part + off_b * stride_evenb + off_h * stride_evenh + offs_m_hash * stride_evenm +offs_even_1
    #oddout_ptr = even + off_b * stride_odd_b + off_h * stride_odd_h + offs_m_hash[:, None] * stride_odd_n +offs_odd[None, :]
    #evenout_ptr = odd + off_b * stride_even_b + off_h * stride_even_h + offs_m_hash[:, None] * stride_even_n +offs_odd[None, :]
    #odd_ptr = odd + offs_d
    #even_ptr = even + offs_d
    odd_v = tl.load(odd_ptr)
    even_v = tl.load(even_ptr)
    odd_v_1 = tl.load(odd_ptr_1)
    even_v_1 = tl.load(even_ptr_1)
    #tl.store(oddout_ptr,odd_v)
    #tl.store(evenout_ptr,even_v)
    
    #print(oddout_ptr)
    #print(odd_v)
    q_ptrs = (
        Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qm + odd_v[None, :])
    )
    simq_ptrs = (
        Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qm + even_v[None, :])
    )
    q_ptrs_1 = (
        Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qm + odd_v_1[None, :])
    )
    simq_ptrs_1 = (
        Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qm + even_v_1[None, :])
    )
    k_ptrs = (
        K + off_b * stride_kb + off_h * stride_kh + (offs_n[:, None] * stride_kn + even_v[None, :])
    )
    
    v_ptrs = (
        V + off_b * stride_vb + off_h * stride_vh + (offs_n[:, None] * stride_vn + offs_e[None, :])
    )
    if BIAS_TYPE == "vector":
        b_ptrs = Bias + off_b * stride_bb + off_h * stride_bh + offs_n
    elif BIAS_TYPE == "matrix":
        b_ptrs = (
            Bias
            + off_b * stride_bb
            + off_h * stride_bh
            + (offs_m[:, None] * stride_bm + offs_n[None, :])
        )
    # initialize pointer to m and l
    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    acc_o = tl.zeros([BLOCK_M, 64], dtype=tl.float32)
    # load q: it will stay in SRAM throughout
    # [2022-10-30] TD: Triton bug - in the case of EVEN_M=True and EVEN_N=False, if we just call
    # tl.load(q_ptrs), we get the wrong output!
    if EVEN_M & EVEN_N:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs)
            simq = tl.load(simq_ptrs)
            q_1 = tl.load(q_ptrs_1)
            simq_1 = tl.load(simq_ptrs_1)
        else:
            q_1 = tl.load(q_ptrs_1, mask=offs_d[None, :] < BLOCK_HEADDIM, other=0.0)
            simq_1 = tl.load(simq_ptrs_1, mask=offs_d[None, :] < BLOCK_HEADDIM, other=0.0)
            q = tl.load(q_ptrs, mask=offs_d[None, :] < BLOCK_HEADDIM, other=0.0)
            simq = tl.load(simq_ptrs, mask=offs_d[None, :] < BLOCK_HEADDIM, other=0.0)
    else:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
            simq = tl.load(simq_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
            q_1 = tl.load(q_ptrs_1, mask=offs_m[:, None] < seqlen_q, other=0.0)
            simq_1 = tl.load(simq_ptrs_1, mask=offs_m[:, None] < seqlen_q, other=0.0)
        else:
            q = tl.load(
                q_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < BLOCK_HEADDIM), other=0.0
            )
            simq = tl.load(
                simq_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < BLOCK_HEADDIM), other=0.0
            )
            q_1 = tl.load(
                q_ptrs_1, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < BLOCK_HEADDIM), other=0.0
            )
            simq_1 = tl.load(
                simq_ptrs_1, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < BLOCK_HEADDIM), other=0.0
            )
    # loop over k, v and update accumulator
    q = q + simq + q_1 + simq_1
    end_n = seqlen_k if not IS_CAUSAL else tl.minimum((start_m + 1) * BLOCK_M, seqlen_k)
    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        if EVEN_N & EVEN_M:  # If we just do "if EVEN_N", there seems to be some race condition
            if EVEN_HEADDIM:
                k = tl.load(k_ptrs + start_n * stride_kn)
            else:
                k = tl.load(k_ptrs + start_n * stride_kn, mask=offs_d[None, :] < BLOCK_HEADDIM, other=0.0)
        else:
            if EVEN_HEADDIM:
                k = tl.load(
                    k_ptrs + start_n * stride_kn,
                    mask=(start_n + offs_n)[:, None] < seqlen_k,
                    other=0.0,
                )
            else:
                k = tl.load(
                    k_ptrs + start_n * stride_kn,
                    mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < BLOCK_HEADDIM),
                    other=0.0,
                )
        #print(k)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))
        # Trying to combine the two masks seem to make the result wrong
        if not EVEN_N:  # Need to mask out otherwise the softmax is wrong
            qk += tl.where((start_n + offs_n)[None, :] < seqlen_k, 0, float("-inf"))
        if IS_CAUSAL:
            qk += tl.where(offs_m[:, None] >= (start_n + offs_n)[None, :], 0, float("-inf"))
        if BIAS_TYPE != "none":
            if BIAS_TYPE == "vector":
                if EVEN_N:
                    bias = tl.load(b_ptrs + start_n).to(tl.float32)
                else:
                    bias = tl.load(
                        b_ptrs + start_n, mask=(start_n + offs_n) < seqlen_k, other=0.0
                    ).to(tl.float32)
                bias = bias[None, :]
            elif BIAS_TYPE == "matrix":
                if EVEN_M & EVEN_N:
                    bias = tl.load(b_ptrs + start_n).to(tl.float32)
                else:
                    bias = tl.load(
                        b_ptrs + start_n,
                        mask=(offs_m[:, None] < seqlen_q)
                        & ((start_n + offs_n)[None, :] < seqlen_k),
                        other=0.0,
                    ).to(tl.float32)
            # Slightly faster to multiply the softmax_scale in the tl.exp below since the compiler
            # can then fuse the mult and add into an fma instruction. But if we have bias we need to
            # to multiply with softmax_scale here.
            qk = qk * softmax_scale + bias
            m_ij = tl.maximum(tl.max(qk, 1), lse_i)
            p = tl.exp(qk - m_ij[:, None])
        else:
            m_ij = tl.maximum(tl.max(qk, 1) * softmax_scale, lse_i)
            p = tl.exp(qk * softmax_scale - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        # scale acc_o
        acc_o_scale = tl.exp(m_i - m_ij)

        # # -- update output accumulator acc_o --
        acc_o = acc_o * acc_o_scale[:, None]

        if EVEN_N & EVEN_M:  # If we just do "if EVEN_N", there seems to be some race condition
            if EVEN_HEADDIM:
                v = tl.load(v_ptrs + start_n * stride_vn)
            else:
                v = tl.load(v_ptrs + start_n * stride_vn, mask=offs_d[None, :] < BLOCK_HEADDIM, other=0.0)
        else:
            if EVEN_HEADDIM:
                v = tl.load(
                    v_ptrs + start_n * stride_vn,
                    mask=(start_n + offs_n)[:, None] < seqlen_k,
                    other=0.0,
                )
            else:
                v = tl.load(
                    v_ptrs + start_n * stride_vn,
                    mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_e[None, :] < BLOCK_HEADDIM),
                    other=0.0,
                )
        p = p.to(v.dtype)
        acc_o += tl.dot(p, v)

        # -- update statistics
        m_i = m_ij
        l_i_new = tl.exp(lse_i - m_ij) + l_ij
        lse_i = m_ij + tl.log(l_i_new)
    
    o_scale = tl.exp(m_i - lse_i)
    acc_o = acc_o * o_scale[:, None]
    start_m = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    out_ptrs = (
        Out
        + off_b * stride_ob
        + off_h * stride_oh
        + (offs_m[:, None] * stride_om + offs_e[None, :])
    )
    tl.store(out_ptrs, acc_o)

@triton.jit
def _fwd_kernel_flash_4_128(
    #odd,
    #even,
    #stride_odd_b,
    #stride_odd_h,
    #stride_odd_n,
    #stride_even_b,
    #stride_even_h,
    #stride_even_n,
    even_part,
    Q,
    K,
    V,
    Bias,
    Out,
    #Lse,
    #TMP,  # NOTE: TMP is a scratchpad buffer to workaround a compiler bug
    #even,
    #odd,
    softmax_scale,
    stride_evenb,
    stride_evenh,
    stride_evenm,    
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_bb,
    stride_bh,
    stride_bm,
    stride_ob,
    stride_oh,
    stride_om,
    nheads,
    seqlen_q,
    seqlen_k,
    #seqlen_q_rounded,
    #headdim,
    #CACHE_KEY_SEQLEN_Q,
    #CACHE_KEY_SEQLEN_K,
    BIAS_TYPE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    offs_m_hash = start_m * 1 + tl.arange(0, 1)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_t = tl.arange(0, BLOCK_T)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    offs_e = tl.arange(0, 128)
    offs_odd = tl.arange(0, 32)
    offs_odd_1 = tl.arange(64, 96)
    offs_even = tl.arange(32, 64)
    offs_even_1 = tl.arange(96, 128)
    odd_ptr = even_part + off_b * stride_evenb + off_h * stride_evenh + offs_m_hash * stride_evenm +offs_odd
    even_ptr = even_part + off_b * stride_evenb + off_h * stride_evenh + offs_m_hash * stride_evenm +offs_even
    odd_ptr_1 = even_part + off_b * stride_evenb + off_h * stride_evenh + offs_m_hash * stride_evenm +offs_odd_1
    even_ptr_1 = even_part + off_b * stride_evenb + off_h * stride_evenh + offs_m_hash * stride_evenm +offs_even_1
    #oddout_ptr = even + off_b * stride_odd_b + off_h * stride_odd_h + offs_m_hash[:, None] * stride_odd_n +offs_odd[None, :]
    #evenout_ptr = odd + off_b * stride_even_b + off_h * stride_even_h + offs_m_hash[:, None] * stride_even_n +offs_odd[None, :]
    #odd_ptr = odd + offs_d
    #even_ptr = even + offs_d
    odd_v = tl.load(odd_ptr)
    even_v = tl.load(even_ptr)
    odd_v_1 = tl.load(odd_ptr_1)
    even_v_1 = tl.load(even_ptr_1)
    #tl.store(oddout_ptr,odd_v)
    #tl.store(evenout_ptr,even_v)
    
    #print(oddout_ptr)
    #print(odd_v)
    q_ptrs = (
        Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qm + odd_v[None, :])
    )
    simq_ptrs = (
        Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qm + even_v[None, :])
    )
    q_ptrs_1 = (
        Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qm + odd_v_1[None, :])
    )
    simq_ptrs_1 = (
        Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qm + even_v_1[None, :])
    )
    k_ptrs = (
        K + off_b * stride_kb + off_h * stride_kh + (offs_n[:, None] * stride_kn + even_v[None, :])
    )
    
    v_ptrs = (
        V + off_b * stride_vb + off_h * stride_vh + (offs_n[:, None] * stride_vn + offs_e[None, :])
    )
    if BIAS_TYPE == "vector":
        b_ptrs = Bias + off_b * stride_bb + off_h * stride_bh + offs_n
    elif BIAS_TYPE == "matrix":
        b_ptrs = (
            Bias
            + off_b * stride_bb
            + off_h * stride_bh
            + (offs_m[:, None] * stride_bm + offs_n[None, :])
        )
    # initialize pointer to m and l
    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    acc_o = tl.zeros([BLOCK_M, 128], dtype=tl.float32)
    # load q: it will stay in SRAM throughout
    # [2022-10-30] TD: Triton bug - in the case of EVEN_M=True and EVEN_N=False, if we just call
    # tl.load(q_ptrs), we get the wrong output!
    if EVEN_M & EVEN_N:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs)
            simq = tl.load(simq_ptrs)
            q_1 = tl.load(q_ptrs_1)
            simq_1 = tl.load(simq_ptrs_1)
        else:
            q_1 = tl.load(q_ptrs_1, mask=offs_d[None, :] < BLOCK_HEADDIM, other=0.0)
            simq_1 = tl.load(simq_ptrs_1, mask=offs_d[None, :] < BLOCK_HEADDIM, other=0.0)
            q = tl.load(q_ptrs, mask=offs_d[None, :] < BLOCK_HEADDIM, other=0.0)
            simq = tl.load(simq_ptrs, mask=offs_d[None, :] < BLOCK_HEADDIM, other=0.0)
    else:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
            simq = tl.load(simq_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
            q_1 = tl.load(q_ptrs_1, mask=offs_m[:, None] < seqlen_q, other=0.0)
            simq_1 = tl.load(simq_ptrs_1, mask=offs_m[:, None] < seqlen_q, other=0.0)
        else:
            q = tl.load(
                q_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < BLOCK_HEADDIM), other=0.0
            )
            simq = tl.load(
                simq_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < BLOCK_HEADDIM), other=0.0
            )
            q_1 = tl.load(
                q_ptrs_1, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < BLOCK_HEADDIM), other=0.0
            )
            simq_1 = tl.load(
                simq_ptrs_1, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < BLOCK_HEADDIM), other=0.0
            )
    # loop over k, v and update accumulator
    q = q + simq + q_1 + simq_1
    end_n = seqlen_k if not IS_CAUSAL else tl.minimum((start_m + 1) * BLOCK_M, seqlen_k)
    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        if EVEN_N & EVEN_M:  # If we just do "if EVEN_N", there seems to be some race condition
            if EVEN_HEADDIM:
                k = tl.load(k_ptrs + start_n * stride_kn)
            else:
                k = tl.load(k_ptrs + start_n * stride_kn, mask=offs_d[None, :] < BLOCK_HEADDIM, other=0.0)
        else:
            if EVEN_HEADDIM:
                k = tl.load(
                    k_ptrs + start_n * stride_kn,
                    mask=(start_n + offs_n)[:, None] < seqlen_k,
                    other=0.0,
                )
            else:
                k = tl.load(
                    k_ptrs + start_n * stride_kn,
                    mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < BLOCK_HEADDIM),
                    other=0.0,
                )
        #print(k)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))
        # Trying to combine the two masks seem to make the result wrong
        if not EVEN_N:  # Need to mask out otherwise the softmax is wrong
            qk += tl.where((start_n + offs_n)[None, :] < seqlen_k, 0, float("-inf"))
        if IS_CAUSAL:
            qk += tl.where(offs_m[:, None] >= (start_n + offs_n)[None, :], 0, float("-inf"))
        if BIAS_TYPE != "none":
            if BIAS_TYPE == "vector":
                if EVEN_N:
                    bias = tl.load(b_ptrs + start_n).to(tl.float32)
                else:
                    bias = tl.load(
                        b_ptrs + start_n, mask=(start_n + offs_n) < seqlen_k, other=0.0
                    ).to(tl.float32)
                bias = bias[None, :]
            elif BIAS_TYPE == "matrix":
                if EVEN_M & EVEN_N:
                    bias = tl.load(b_ptrs + start_n).to(tl.float32)
                else:
                    bias = tl.load(
                        b_ptrs + start_n,
                        mask=(offs_m[:, None] < seqlen_q)
                        & ((start_n + offs_n)[None, :] < seqlen_k),
                        other=0.0,
                    ).to(tl.float32)
            # Slightly faster to multiply the softmax_scale in the tl.exp below since the compiler
            # can then fuse the mult and add into an fma instruction. But if we have bias we need to
            # to multiply with softmax_scale here.
            qk = qk * softmax_scale + bias
            m_ij = tl.maximum(tl.max(qk, 1), lse_i)
            p = tl.exp(qk - m_ij[:, None])
        else:
            m_ij = tl.maximum(tl.max(qk, 1) * softmax_scale, lse_i)
            p = tl.exp(qk * softmax_scale - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        # scale acc_o
        acc_o_scale = tl.exp(m_i - m_ij)

        # # -- update output accumulator acc_o --
        acc_o = acc_o * acc_o_scale[:, None]

        if EVEN_N & EVEN_M:  # If we just do "if EVEN_N", there seems to be some race condition
            if EVEN_HEADDIM:
                v = tl.load(v_ptrs + start_n * stride_vn)
            else:
                v = tl.load(v_ptrs + start_n * stride_vn, mask=offs_d[None, :] < BLOCK_HEADDIM, other=0.0)
        else:
            if EVEN_HEADDIM:
                v = tl.load(
                    v_ptrs + start_n * stride_vn,
                    mask=(start_n + offs_n)[:, None] < seqlen_k,
                    other=0.0,
                )
            else:
                v = tl.load(
                    v_ptrs + start_n * stride_vn,
                    mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_e[None, :] < BLOCK_HEADDIM),
                    other=0.0,
                )
        p = p.to(v.dtype)
        acc_o += tl.dot(p, v)

        # -- update statistics
        m_i = m_ij
        l_i_new = tl.exp(lse_i - m_ij) + l_ij
        lse_i = m_ij + tl.log(l_i_new)
    
    o_scale = tl.exp(m_i - lse_i)
    acc_o = acc_o * o_scale[:, None]
    start_m = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    out_ptrs = (
        Out
        + off_b * stride_ob
        + off_h * stride_oh
        + (offs_m[:, None] * stride_om + offs_e[None, :])
    )
    tl.store(out_ptrs, acc_o)

@triton.jit
def _fwd_kernel_flash_2_64(
    #odd,
    #even,
    #stride_odd_b,
    #stride_odd_h,
    #stride_odd_n,
    #stride_even_b,
    #stride_even_h,
    #stride_even_n,
    even_part,
    Q,
    K,
    V,
    Bias,
    Out,
    #Lse,
    #TMP,  # NOTE: TMP is a scratchpad buffer to workaround a compiler bug
    #even,
    #odd,
    softmax_scale,
    stride_evenb,
    stride_evenh,
    stride_evenm,    
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_bb,
    stride_bh,
    stride_bm,
    stride_ob,
    stride_oh,
    stride_om,
    nheads,
    seqlen_q,
    seqlen_k,
    #seqlen_q_rounded,
    #headdim,
    #CACHE_KEY_SEQLEN_Q,
    #CACHE_KEY_SEQLEN_K,
    BIAS_TYPE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    offs_m_hash = start_m * 1 + tl.arange(0, 1)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_t = tl.arange(0, BLOCK_T)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    offs_e = tl.arange(0, 64)
    offs_odd = tl.arange(0, 32)
    offs_even = tl.arange(32, 64)
    odd_ptr = even_part + off_b * stride_evenb + off_h * stride_evenh + offs_m_hash * stride_evenm +offs_odd
    even_ptr = even_part + off_b * stride_evenb + off_h * stride_evenh + offs_m_hash * stride_evenm +offs_even
    #oddout_ptr = even + off_b * stride_odd_b + off_h * stride_odd_h + offs_m_hash[:, None] * stride_odd_n +offs_odd[None, :]
    #evenout_ptr = odd + off_b * stride_even_b + off_h * stride_even_h + offs_m_hash[:, None] * stride_even_n +offs_odd[None, :]
    #odd_ptr = odd + offs_d
    #even_ptr = even + offs_d
    odd_v = tl.load(odd_ptr)
    even_v = tl.load(even_ptr)
    #tl.store(oddout_ptr,odd_v)
    #tl.store(evenout_ptr,even_v)
    
    #print(oddout_ptr)
    #print(odd_v)
    q_ptrs = (
        Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qm + odd_v[None, :])
    )
    simq_ptrs = (
        Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qm + even_v[None, :])
    )
    k_ptrs = (
        K + off_b * stride_kb + off_h * stride_kh + (offs_n[:, None] * stride_kn + even_v[None, :])
    )
    v_ptrs = (
        V + off_b * stride_vb + off_h * stride_vh + (offs_n[:, None] * stride_vn + offs_e[None, :])
    )
    if BIAS_TYPE == "vector":
        b_ptrs = Bias + off_b * stride_bb + off_h * stride_bh + offs_n
    elif BIAS_TYPE == "matrix":
        b_ptrs = (
            Bias
            + off_b * stride_bb
            + off_h * stride_bh
            + (offs_m[:, None] * stride_bm + offs_n[None, :])
        )
    # initialize pointer to m and l
    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    acc_o = tl.zeros([BLOCK_M, 64], dtype=tl.float32)
    # load q: it will stay in SRAM throughout
    # [2022-10-30] TD: Triton bug - in the case of EVEN_M=True and EVEN_N=False, if we just call
    # tl.load(q_ptrs), we get the wrong output!
    if EVEN_M & EVEN_N:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs)
            simq = tl.load(simq_ptrs)
        else:
            q = tl.load(q_ptrs, mask=offs_d[None, :] < BLOCK_HEADDIM, other=0.0)
            simq = tl.load(simq_ptrs, mask=offs_d[None, :] < BLOCK_HEADDIM, other=0.0)
    else:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
            simq = tl.load(simq_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
        else:
            q = tl.load(
                q_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < BLOCK_HEADDIM), other=0.0
            )
            simq = tl.load(
                simq_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < BLOCK_HEADDIM), other=0.0
            )
    # loop over k, v and update accumulator
    q = q + simq
    end_n = seqlen_k if not IS_CAUSAL else tl.minimum((start_m + 1) * BLOCK_M, seqlen_k)
    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        if EVEN_N & EVEN_M:  # If we just do "if EVEN_N", there seems to be some race condition
            if EVEN_HEADDIM:
                k = tl.load(k_ptrs + start_n * stride_kn)
            else:
                k = tl.load(k_ptrs + start_n * stride_kn, mask=offs_d[None, :] < BLOCK_HEADDIM, other=0.0)
        else:
            if EVEN_HEADDIM:
                k = tl.load(
                    k_ptrs + start_n * stride_kn,
                    mask=(start_n + offs_n)[:, None] < seqlen_k,
                    other=0.0,
                )
            else:
                k = tl.load(
                    k_ptrs + start_n * stride_kn,
                    mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < BLOCK_HEADDIM),
                    other=0.0,
                )
        #print(k)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))
        # Trying to combine the two masks seem to make the result wrong
        if not EVEN_N:  # Need to mask out otherwise the softmax is wrong
            qk += tl.where((start_n + offs_n)[None, :] < seqlen_k, 0, float("-inf"))
        if IS_CAUSAL:
            qk += tl.where(offs_m[:, None] >= (start_n + offs_n)[None, :], 0, float("-inf"))
        if BIAS_TYPE != "none":
            if BIAS_TYPE == "vector":
                if EVEN_N:
                    bias = tl.load(b_ptrs + start_n).to(tl.float32)
                else:
                    bias = tl.load(
                        b_ptrs + start_n, mask=(start_n + offs_n) < seqlen_k, other=0.0
                    ).to(tl.float32)
                bias = bias[None, :]
            elif BIAS_TYPE == "matrix":
                if EVEN_M & EVEN_N:
                    bias = tl.load(b_ptrs + start_n).to(tl.float32)
                else:
                    bias = tl.load(
                        b_ptrs + start_n,
                        mask=(offs_m[:, None] < seqlen_q)
                        & ((start_n + offs_n)[None, :] < seqlen_k),
                        other=0.0,
                    ).to(tl.float32)
            # Slightly faster to multiply the softmax_scale in the tl.exp below since the compiler
            # can then fuse the mult and add into an fma instruction. But if we have bias we need to
            # to multiply with softmax_scale here.
            qk = qk * softmax_scale + bias
            m_ij = tl.maximum(tl.max(qk, 1), lse_i)
            p = tl.exp(qk - m_ij[:, None])
        else:
            m_ij = tl.maximum(tl.max(qk, 1) * softmax_scale, lse_i)
            p = tl.exp(qk * softmax_scale - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        # scale acc_o
        acc_o_scale = tl.exp(m_i - m_ij)

        # # -- update output accumulator acc_o --
        acc_o = acc_o * acc_o_scale[:, None]

        if EVEN_N & EVEN_M:  # If we just do "if EVEN_N", there seems to be some race condition
            if EVEN_HEADDIM:
                v = tl.load(v_ptrs + start_n * stride_vn)
            else:
                v = tl.load(v_ptrs + start_n * stride_vn, mask=offs_d[None, :] < BLOCK_HEADDIM, other=0.0)
        else:
            if EVEN_HEADDIM:
                v = tl.load(
                    v_ptrs + start_n * stride_vn,
                    mask=(start_n + offs_n)[:, None] < seqlen_k,
                    other=0.0,
                )
            else:
                v = tl.load(
                    v_ptrs + start_n * stride_vn,
                    mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_e[None, :] < BLOCK_HEADDIM),
                    other=0.0,
                )
        p = p.to(v.dtype)
        acc_o += tl.dot(p, v)

        # -- update statistics
        m_i = m_ij
        l_i_new = tl.exp(lse_i - m_ij) + l_ij
        lse_i = m_ij + tl.log(l_i_new)
    
    o_scale = tl.exp(m_i - lse_i)
    acc_o = acc_o * o_scale[:, None]
    start_m = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    out_ptrs = (
        Out
        + off_b * stride_ob
        + off_h * stride_oh
        + (offs_m[:, None] * stride_om + offs_e[None, :])
    )
    tl.store(out_ptrs, acc_o)

@triton.jit
def _fwd_kernel_flash_2_32(
    #odd,
    #even,
    #stride_odd_b,
    #stride_odd_h,
    #stride_odd_n,
    #stride_even_b,
    #stride_even_h,
    #stride_even_n,
    even_part,
    Q,
    K,
    V,
    Bias,
    Out,
    #Lse,
    #TMP,  # NOTE: TMP is a scratchpad buffer to workaround a compiler bug
    #even,
    #odd,
    softmax_scale,
    stride_evenb,
    stride_evenh,
    stride_evenm,    
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_bb,
    stride_bh,
    stride_bm,
    stride_ob,
    stride_oh,
    stride_om,
    nheads,
    seqlen_q,
    seqlen_k,
    #seqlen_q_rounded,
    #headdim,
    #CACHE_KEY_SEQLEN_Q,
    #CACHE_KEY_SEQLEN_K,
    BIAS_TYPE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    offs_m_hash = start_m * 1 + tl.arange(0, 1)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_t = tl.arange(0, BLOCK_T)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    offs_e = tl.arange(0, 32)
    offs_odd = tl.arange(0, 16)
    offs_even = tl.arange(16, 32)
    odd_ptr = even_part + off_b * stride_evenb + off_h * stride_evenh + offs_m_hash * stride_evenm +offs_odd
    even_ptr = even_part + off_b * stride_evenb + off_h * stride_evenh + offs_m_hash * stride_evenm +offs_even
    #oddout_ptr = even + off_b * stride_odd_b + off_h * stride_odd_h + offs_m_hash[:, None] * stride_odd_n +offs_odd[None, :]
    #evenout_ptr = odd + off_b * stride_even_b + off_h * stride_even_h + offs_m_hash[:, None] * stride_even_n +offs_odd[None, :]
    #odd_ptr = odd + offs_d
    #even_ptr = even + offs_d
    odd_v = tl.load(odd_ptr)
    even_v = tl.load(even_ptr)
    #tl.store(oddout_ptr,odd_v)
    #tl.store(evenout_ptr,even_v)
    
    #print(oddout_ptr)
    #print(odd_v)
    q_ptrs = (
        Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qm + odd_v[None, :])
    )
    simq_ptrs = (
        Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qm + even_v[None, :])
    )
    k_ptrs = (
        K + off_b * stride_kb + off_h * stride_kh + (offs_n[:, None] * stride_kn + even_v[None, :])
    )
    v_ptrs = (
        V + off_b * stride_vb + off_h * stride_vh + (offs_n[:, None] * stride_vn + offs_e[None, :])
    )
    if BIAS_TYPE == "vector":
        b_ptrs = Bias + off_b * stride_bb + off_h * stride_bh + offs_n
    elif BIAS_TYPE == "matrix":
        b_ptrs = (
            Bias
            + off_b * stride_bb
            + off_h * stride_bh
            + (offs_m[:, None] * stride_bm + offs_n[None, :])
        )
    # initialize pointer to m and l
    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    acc_o = tl.zeros([BLOCK_M, 32], dtype=tl.float32)
    # load q: it will stay in SRAM throughout
    # [2022-10-30] TD: Triton bug - in the case of EVEN_M=True and EVEN_N=False, if we just call
    # tl.load(q_ptrs), we get the wrong output!
    if EVEN_M & EVEN_N:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs)
            simq = tl.load(simq_ptrs)
        else:
            q = tl.load(q_ptrs, mask=offs_d[None, :] < BLOCK_HEADDIM, other=0.0)
            simq = tl.load(simq_ptrs, mask=offs_d[None, :] < BLOCK_HEADDIM, other=0.0)
    else:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
            simq = tl.load(simq_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
        else:
            q = tl.load(
                q_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < BLOCK_HEADDIM), other=0.0
            )
            simq = tl.load(
                simq_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < BLOCK_HEADDIM), other=0.0
            )
    # loop over k, v and update accumulator
    q = q + simq
    end_n = seqlen_k if not IS_CAUSAL else tl.minimum((start_m + 1) * BLOCK_M, seqlen_k)
    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        if EVEN_N & EVEN_M:  # If we just do "if EVEN_N", there seems to be some race condition
            if EVEN_HEADDIM:
                k = tl.load(k_ptrs + start_n * stride_kn)
            else:
                k = tl.load(k_ptrs + start_n * stride_kn, mask=offs_d[None, :] < BLOCK_HEADDIM, other=0.0)
        else:
            if EVEN_HEADDIM:
                k = tl.load(
                    k_ptrs + start_n * stride_kn,
                    mask=(start_n + offs_n)[:, None] < seqlen_k,
                    other=0.0,
                )
            else:
                k = tl.load(
                    k_ptrs + start_n * stride_kn,
                    mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < BLOCK_HEADDIM),
                    other=0.0,
                )
        #print(k)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))
        # Trying to combine the two masks seem to make the result wrong
        if not EVEN_N:  # Need to mask out otherwise the softmax is wrong
            qk += tl.where((start_n + offs_n)[None, :] < seqlen_k, 0, float("-inf"))
        if IS_CAUSAL:
            qk += tl.where(offs_m[:, None] >= (start_n + offs_n)[None, :], 0, float("-inf"))
        if BIAS_TYPE != "none":
            if BIAS_TYPE == "vector":
                if EVEN_N:
                    bias = tl.load(b_ptrs + start_n).to(tl.float32)
                else:
                    bias = tl.load(
                        b_ptrs + start_n, mask=(start_n + offs_n) < seqlen_k, other=0.0
                    ).to(tl.float32)
                bias = bias[None, :]
            elif BIAS_TYPE == "matrix":
                if EVEN_M & EVEN_N:
                    bias = tl.load(b_ptrs + start_n).to(tl.float32)
                else:
                    bias = tl.load(
                        b_ptrs + start_n,
                        mask=(offs_m[:, None] < seqlen_q)
                        & ((start_n + offs_n)[None, :] < seqlen_k),
                        other=0.0,
                    ).to(tl.float32)
            # Slightly faster to multiply the softmax_scale in the tl.exp below since the compiler
            # can then fuse the mult and add into an fma instruction. But if we have bias we need to
            # to multiply with softmax_scale here.
            qk = qk * softmax_scale + bias
            m_ij = tl.maximum(tl.max(qk, 1), lse_i)
            p = tl.exp(qk - m_ij[:, None])
        else:
            m_ij = tl.maximum(tl.max(qk, 1) * softmax_scale, lse_i)
            p = tl.exp(qk * softmax_scale - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        # scale acc_o
        acc_o_scale = tl.exp(m_i - m_ij)

        # # -- update output accumulator acc_o --
        acc_o = acc_o * acc_o_scale[:, None]

        if EVEN_N & EVEN_M:  # If we just do "if EVEN_N", there seems to be some race condition
            if EVEN_HEADDIM:
                v = tl.load(v_ptrs + start_n * stride_vn)
            else:
                v = tl.load(v_ptrs + start_n * stride_vn, mask=offs_d[None, :] < BLOCK_HEADDIM, other=0.0)
        else:
            if EVEN_HEADDIM:
                v = tl.load(
                    v_ptrs + start_n * stride_vn,
                    mask=(start_n + offs_n)[:, None] < seqlen_k,
                    other=0.0,
                )
            else:
                v = tl.load(
                    v_ptrs + start_n * stride_vn,
                    mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_e[None, :] < BLOCK_HEADDIM),
                    other=0.0,
                )
        p = p.to(v.dtype)
        acc_o += tl.dot(p, v)

        # -- update statistics
        m_i = m_ij
        l_i_new = tl.exp(lse_i - m_ij) + l_ij
        lse_i = m_ij + tl.log(l_i_new)
    
    o_scale = tl.exp(m_i - lse_i)
    acc_o = acc_o * o_scale[:, None]
    start_m = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    out_ptrs = (
        Out
        + off_b * stride_ob
        + off_h * stride_oh
        + (offs_m[:, None] * stride_om + offs_e[None, :])
    )
    tl.store(out_ptrs, acc_o)

@triton.jit
def _fwd_kernel_flash_2_128(
    #odd,
    #even,
    #stride_odd_b,
    #stride_odd_h,
    #stride_odd_n,
    #stride_even_b,
    #stride_even_h,
    #stride_even_n,
    even_part,
    Q,
    K,
    V,
    Bias,
    Out,
    #Lse,
    #TMP,  # NOTE: TMP is a scratchpad buffer to workaround a compiler bug
    #even,
    #odd,
    softmax_scale,
    stride_evenb,
    stride_evenh,
    stride_evenm,    
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_bb,
    stride_bh,
    stride_bm,
    stride_ob,
    stride_oh,
    stride_om,
    nheads,
    seqlen_q,
    seqlen_k,
    #seqlen_q_rounded,
    #headdim,
    #CACHE_KEY_SEQLEN_Q,
    #CACHE_KEY_SEQLEN_K,
    BIAS_TYPE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    offs_m_hash = start_m * 1 + tl.arange(0, 1)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_t = tl.arange(0, BLOCK_T)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    offs_e = tl.arange(0, 128)
    offs_odd = tl.arange(0, 64)
    offs_even = tl.arange(64, 128)
    odd_ptr = even_part + off_b * stride_evenb + off_h * stride_evenh + offs_m_hash * stride_evenm +offs_odd
    even_ptr = even_part + off_b * stride_evenb + off_h * stride_evenh + offs_m_hash * stride_evenm +offs_even
    #oddout_ptr = even + off_b * stride_odd_b + off_h * stride_odd_h + offs_m_hash[:, None] * stride_odd_n +offs_odd[None, :]
    #evenout_ptr = odd + off_b * stride_even_b + off_h * stride_even_h + offs_m_hash[:, None] * stride_even_n +offs_odd[None, :]
    #odd_ptr = odd + offs_d
    #even_ptr = even + offs_d
    odd_v = tl.load(odd_ptr)
    even_v = tl.load(even_ptr)
    #tl.store(oddout_ptr,odd_v)
    #tl.store(evenout_ptr,even_v)
    
    #print(oddout_ptr)
    #print(odd_v)
    q_ptrs = (
        Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qm + odd_v[None, :])
    )
    simq_ptrs = (
        Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qm + even_v[None, :])
    )
    k_ptrs = (
        K + off_b * stride_kb + off_h * stride_kh + (offs_n[:, None] * stride_kn + even_v[None, :])
    )
    v_ptrs = (
        V + off_b * stride_vb + off_h * stride_vh + (offs_n[:, None] * stride_vn + offs_e[None, :])
    )
    if BIAS_TYPE == "vector":
        b_ptrs = Bias + off_b * stride_bb + off_h * stride_bh + offs_n
    elif BIAS_TYPE == "matrix":
        b_ptrs = (
            Bias
            + off_b * stride_bb
            + off_h * stride_bh
            + (offs_m[:, None] * stride_bm + offs_n[None, :])
        )
    # initialize pointer to m and l
    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    acc_o = tl.zeros([BLOCK_M, 128], dtype=tl.float32)
    # load q: it will stay in SRAM throughout
    # [2022-10-30] TD: Triton bug - in the case of EVEN_M=True and EVEN_N=False, if we just call
    # tl.load(q_ptrs), we get the wrong output!
    if EVEN_M & EVEN_N:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs)
            simq = tl.load(simq_ptrs)
        else:
            q = tl.load(q_ptrs, mask=offs_d[None, :] < BLOCK_HEADDIM, other=0.0)
            simq = tl.load(simq_ptrs, mask=offs_d[None, :] < BLOCK_HEADDIM, other=0.0)
    else:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
            simq = tl.load(simq_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
        else:
            q = tl.load(
                q_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < BLOCK_HEADDIM), other=0.0
            )
            simq = tl.load(
                simq_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < BLOCK_HEADDIM), other=0.0
            )
    # loop over k, v and update accumulator
    q = q + simq
    end_n = seqlen_k if not IS_CAUSAL else tl.minimum((start_m + 1) * BLOCK_M, seqlen_k)
    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        if EVEN_N & EVEN_M:  # If we just do "if EVEN_N", there seems to be some race condition
            if EVEN_HEADDIM:
                k = tl.load(k_ptrs + start_n * stride_kn)
            else:
                k = tl.load(k_ptrs + start_n * stride_kn, mask=offs_d[None, :] < BLOCK_HEADDIM, other=0.0)
        else:
            if EVEN_HEADDIM:
                k = tl.load(
                    k_ptrs + start_n * stride_kn,
                    mask=(start_n + offs_n)[:, None] < seqlen_k,
                    other=0.0,
                )
            else:
                k = tl.load(
                    k_ptrs + start_n * stride_kn,
                    mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < BLOCK_HEADDIM),
                    other=0.0,
                )
        #print(k)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))
        # Trying to combine the two masks seem to make the result wrong
        if not EVEN_N:  # Need to mask out otherwise the softmax is wrong
            qk += tl.where((start_n + offs_n)[None, :] < seqlen_k, 0, float("-inf"))
        if IS_CAUSAL:
            qk += tl.where(offs_m[:, None] >= (start_n + offs_n)[None, :], 0, float("-inf"))
        if BIAS_TYPE != "none":
            if BIAS_TYPE == "vector":
                if EVEN_N:
                    bias = tl.load(b_ptrs + start_n).to(tl.float32)
                else:
                    bias = tl.load(
                        b_ptrs + start_n, mask=(start_n + offs_n) < seqlen_k, other=0.0
                    ).to(tl.float32)
                bias = bias[None, :]
            elif BIAS_TYPE == "matrix":
                if EVEN_M & EVEN_N:
                    bias = tl.load(b_ptrs + start_n).to(tl.float32)
                else:
                    bias = tl.load(
                        b_ptrs + start_n,
                        mask=(offs_m[:, None] < seqlen_q)
                        & ((start_n + offs_n)[None, :] < seqlen_k),
                        other=0.0,
                    ).to(tl.float32)
            # Slightly faster to multiply the softmax_scale in the tl.exp below since the compiler
            # can then fuse the mult and add into an fma instruction. But if we have bias we need to
            # to multiply with softmax_scale here.
            qk = qk * softmax_scale + bias
            m_ij = tl.maximum(tl.max(qk, 1), lse_i)
            p = tl.exp(qk - m_ij[:, None])
        else:
            m_ij = tl.maximum(tl.max(qk, 1) * softmax_scale, lse_i)
            p = tl.exp(qk * softmax_scale - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        # scale acc_o
        acc_o_scale = tl.exp(m_i - m_ij)

        # # -- update output accumulator acc_o --
        acc_o = acc_o * acc_o_scale[:, None]

        if EVEN_N & EVEN_M:  # If we just do "if EVEN_N", there seems to be some race condition
            if EVEN_HEADDIM:
                v = tl.load(v_ptrs + start_n * stride_vn)
            else:
                v = tl.load(v_ptrs + start_n * stride_vn, mask=offs_d[None, :] < BLOCK_HEADDIM, other=0.0)
        else:
            if EVEN_HEADDIM:
                v = tl.load(
                    v_ptrs + start_n * stride_vn,
                    mask=(start_n + offs_n)[:, None] < seqlen_k,
                    other=0.0,
                )
            else:
                v = tl.load(
                    v_ptrs + start_n * stride_vn,
                    mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_e[None, :] < BLOCK_HEADDIM),
                    other=0.0,
                )
        p = p.to(v.dtype)
        acc_o += tl.dot(p, v)

        # -- update statistics
        m_i = m_ij
        l_i_new = tl.exp(lse_i - m_ij) + l_ij
        lse_i = m_ij + tl.log(l_i_new)
    
    o_scale = tl.exp(m_i - lse_i)
    acc_o = acc_o * o_scale[:, None]
    start_m = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    out_ptrs = (
        Out
        + off_b * stride_ob
        + off_h * stride_oh
        + (offs_m[:, None] * stride_om + offs_e[None, :])
    )
    tl.store(out_ptrs, acc_o)
@triton.jit
def _angular_lsh_kernel_even_odd(
        even_part,
        #odd_part,
        #in_mat,
        #proj_dir,
        #perm,
        #enc_vec,
        buckets,
        stride_evenb,
        stride_evenh,
        stride_evenm,
        #stride_oddb,
        #stride_oddh,
        #stride_oddm,
        #stride_in_matb,
        #stride_in_math,
        #stride_in_matm,
        #stride_proj_dirb,
        #stride_proj_dirh,
        #stride_proj_dird,
        stride_bucketsb,
        stride_bucketsh,
        stride_bucketsh_blockn,
        nheads,
        seqlen,
        seqlen_rounded,
        headdim,
        NUM_PROJ_ROUNDED: tl.constexpr,
        num_projs: tl.constexpr,
        BLOCK_HEADDIM: tl.constexpr,
        #EVEN_M: tl.constexpr,
        #EVEN_HEADDIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    #offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_m_hash = start_m * 1 + tl.arange(0, 1)
    #offs_n = tl.arange(0, NUM_PROJ_ROUNDED)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    #offs_p = tl.arange(0, BLOCK_M)
    '''
    in_mat_ptrs = (
            in_mat + off_b * stride_in_matb + off_h * stride_in_math + (offs_m[:, None] * stride_in_matm +
                                                                        offs_d[None, :])
    )
    proj_dir_ptrs = (
        proj_dir + off_b * stride_proj_dirb + off_h * stride_proj_dirh + (offs_p[:, None] * stride_proj_dird +
                                                                          offs_n[None, :])
    )

    # load in_mat block
    if EVEN_M:
        if EVEN_HEADDIM:
            mat = tl.load(in_mat_ptrs)
        else:
            mat = tl.load(in_mat_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
    else:
        if EVEN_HEADDIM:
            mat = tl.load(in_mat_ptrs, mask=offs_m[:, None] < seqlen, other=0.0)
        else:
            mat = tl.load(in_mat_ptrs, mask=(offs_m[:, None] < seqlen) & (offs_d[None, :] < headdim), other=0.0)

    # load proj_dir block, need to mask out out of bound offsets
    if EVEN_HEADDIM:
        proj_dir_block = tl.load(proj_dir_ptrs, mask=offs_n[None, :] < num_projs, other=0.0)
    else:
        proj_dir_block = tl.load(proj_dir_ptrs,
                                 mask=(offs_n[None, :] < num_projs) & (offs_d[:, None] * stride_proj_dird < headdim),
                                 other=0.0)

    # multiply the in_mat block with proj_dir block to get the mask
    mask = tl.dot(tl.trans(mat), proj_dir_block)
    mask = tl.where(mask > 0.0, 1.0, 0.0)

    # form enc_vec
    encoding_vectors = tl.load(enc_vec+offs_n, mask=offs_n < num_projs, other=0.0)

    # multiply mask by enc_vec
    bin_ids = tl.sum(mask * encoding_vectors[None, :], 1).to(tl.int32)
    # bin_ids = tl.ravel(bin_ids)  # flatten the bin_ids into a 1d tensor

    # read hash buckets from look up table
    hash_buckets = tl.load(perm+bin_ids)
    '''
    # write back bin_ids
    # initialize pointers to output
    #buckets_ptrs = buckets + off_b * stride_bucketsb + off_h * stride_bucketsh + offs_m
    buckets_ptrs = buckets + off_b * stride_bucketsb + off_h * stride_bucketsh + offs_m_hash[:, None] * stride_bucketsh_blockn +offs_d[None, :]
    even_ptrs = even_part + off_b * stride_evenb + off_h * stride_evenh + offs_m_hash[:, None] * stride_evenm +offs_d[None, :]
    buckets_sort = tl.load(buckets_ptrs)
    #buckets_sort = tl.view(buckets_sort,[1,64])
    buckets_sort = tl.sort(buckets_sort,dim=1)
    buckets_sort = tl.view(buckets_sort,[1,headdim])
    #print(buckets_sort)
    tl.store(even_ptrs, buckets_sort)
    #print(buckets)
    #hash_buckets = tl.sort(hash_buckets,dim=0)
    #print(hash_buckets)
    #hash_buckets = tl.view(hash_buckets,[1,64])
    #if EVEN_M:
    #    tl.store(buckets_ptrs, hash_buckets)
    #else:
    #    tl.store(buckets_ptrs, hash_buckets, mask=offs_m < seqlen)

@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen"] % args["BLOCK_M"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
    }
)
@triton.jit
def _angular_lsh_kernel(
        #even_part,
        #odd_part,
        in_mat,
        proj_dir,
        perm,
        enc_vec,
        buckets,
        #stride_evenb,
        #stride_evenh,
        #stride_evenm,
        #stride_oddb,
        #stride_oddh,
        #stride_oddm,
        stride_in_matb,
        stride_in_math,
        stride_in_matm,
        stride_proj_dirb,
        stride_proj_dirh,
        stride_proj_dird,
        stride_bucketsb,
        stride_bucketsh,
        stride_bucketsh_blockn,
        nheads,
        seqlen,
        seqlen_rounded,
        headdim,
        NUM_PROJ_ROUNDED: tl.constexpr,
        num_projs: tl.constexpr,
        BLOCK_HEADDIM: tl.constexpr,
        EVEN_M: tl.constexpr,
        EVEN_HEADDIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_m_hash = start_m * 1 + tl.arange(0, 1)
    offs_n = tl.arange(0, NUM_PROJ_ROUNDED)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    offs_p = tl.arange(0, BLOCK_M)

    in_mat_ptrs = (
            in_mat + off_b * stride_in_matb + off_h * stride_in_math + (offs_m[:, None] * stride_in_matm +
                                                                        offs_d[None, :])
    )
    proj_dir_ptrs = (
        proj_dir + off_b * stride_proj_dirb + off_h * stride_proj_dirh + (offs_p[:, None] * stride_proj_dird +
                                                                          offs_n[None, :])
    )

    # load in_mat block
    if EVEN_M:
        if EVEN_HEADDIM:
            mat = tl.load(in_mat_ptrs)
        else:
            mat = tl.load(in_mat_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
    else:
        if EVEN_HEADDIM:
            mat = tl.load(in_mat_ptrs, mask=offs_m[:, None] < seqlen, other=0.0)
        else:
            mat = tl.load(in_mat_ptrs, mask=(offs_m[:, None] < seqlen) & (offs_d[None, :] < headdim), other=0.0)

    # load proj_dir block, need to mask out out of bound offsets
    if EVEN_HEADDIM:
        proj_dir_block = tl.load(proj_dir_ptrs, mask=offs_n[None, :] < num_projs, other=0.0)
    else:
        proj_dir_block = tl.load(proj_dir_ptrs,
                                 mask=(offs_n[None, :] < num_projs) & (offs_d[:, None] * stride_proj_dird < headdim),
                                 other=0.0)

    # multiply the in_mat block with proj_dir block to get the mask
    mask = tl.dot(tl.trans(mat), proj_dir_block)
    mask = tl.where(mask > 0.0, 1.0, 0.0)

    # form enc_vec
    encoding_vectors = tl.load(enc_vec+offs_n, mask=offs_n < num_projs, other=0.0)

    # multiply mask by enc_vec
    bin_ids = tl.sum(mask * encoding_vectors[None, :], 1).to(tl.int32)
    # bin_ids = tl.ravel(bin_ids)  # flatten the bin_ids into a 1d tensor

    # read hash buckets from look up table
    hash_buckets = tl.load(perm+bin_ids)

    # write back bin_ids
    # initialize pointers to output
    #buckets_ptrs = buckets + off_b * stride_bucketsb + off_h * stride_bucketsh + offs_m
    buckets_ptrs = buckets + off_b * stride_bucketsb + off_h * stride_bucketsh + offs_m_hash[:, None] * stride_bucketsh_blockn +offs_d[None, :]


    #hash_buckets = tl.sort(hash_buckets,dim=0)
    #print(hash_buckets)
    hash_buckets = tl.view(hash_buckets,[1,headdim])
    #print(buckets_ptrs)
    if EVEN_M:
        tl.store(buckets_ptrs, hash_buckets)
    else:
        tl.store(buckets_ptrs, hash_buckets)

def _flash_attn(q, proj_dir, perm, enc_vec,k,v):
    bias=None
    causal=False
    softmax_scale=None
    batch, nheads, seqlen_q,d = q.shape
    _, _, seqlen_k, _ = k.shape
    #assert k.shape == (batch, seqlen_k, nheads, d)
    #assert v.shape == (batch, seqlen_k, nheads, d)
    assert d <= 128, "FlashAttention only support head dimensions up to 128"
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same type"
    assert q.dtype in [torch.float16, torch.bfloat16], "Only support fp16 and bf16"
    assert q.is_cuda and k.is_cuda and v.is_cuda
    d = 64
    softmax_scale = softmax_scale or 1.0 / math.sqrt(d)

    has_bias = bias is not None
    bias_type = "none"
    if has_bias:
        assert bias.dtype in [q.dtype, torch.float]
        assert bias.is_cuda
        assert bias.dim() == 4
        if bias.stride(-1) != 1:
            bias = bias.contiguous()
        if bias.shape[2:] == (1, seqlen_k):
            bias_type = "vector"
        elif bias.shape[2:] == (seqlen_q, seqlen_k):
            bias_type = "matrix"
        else:
            raise RuntimeError(
                "Last 2 dimensions of bias must be (1, seqlen_k)" " or (seqlen_q, seqlen_k)"
            )
        bias = bias.expand(batch, nheads, seqlen_q, seqlen_k)
    bias_strides = (bias.stride(0), bias.stride(1), bias.stride(2)) if has_bias else (0, 0, 0)
    num_projs = proj_dir.shape[-1]
    batch, nheads, seqlen, d = q.shape
    #assert (proj_dir.shape == (batch, nheads, d, num_projs)) or (proj_dir.shape == (1, 1, d, num_projs))
    assert q.dtype == proj_dir.dtype, "All three tensors must have the same type"
    assert q.dtype in [torch.float16, torch.bfloat16], "Only support fp16 and bf16"
    assert q.is_cuda and proj_dir.is_cuda and perm.is_cuda and enc_vec.is_cuda
    if proj_dir.shape[:2] == (1, 1):
        stride_proj_dirb, stride_proj_dirh = 0, 0
    else:
        stride_proj_dirb, stride_proj_dirh = proj_dir.stride()[:2]

    seqlen_rounded = math.ceil(seqlen / 128) * 128
    num_projs_rounded = 16

    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    BLOCK = 128
    block_n = triton.cdiv(seqlen,BLOCK)
    #buckets = torch.empty((batch, nheads, seqlen), device=in_mat.device, dtype=torch.int32)
    buckets = torch.empty((batch, nheads, block_n,d), device=q.device, dtype=torch.int32)
    even_part =  torch.empty((batch, nheads, block_n,64), device=q.device, dtype=torch.int32)
    #odd_part =  torch.empty((batch, nheads, block_n,32), device=in_mat.device, dtype=torch.int32)
    odd =  torch.empty((batch, nheads, block_n,32), device=q.device, dtype=torch.int32)
    even =  torch.empty((batch, nheads, block_n,32), device=q.device, dtype=torch.int32)
    num_warps = 4 if d <= 64 else 8
    grid = lambda META: (triton.cdiv(seqlen, META["BLOCK_M"]), batch * nheads)
    o = torch.empty_like(q)

    _fwd_kernel_flashattn[grid](
        #odd = odd,
        #even = even,
        #stride_odd_b=odd.stride(0),
        #stride_odd_h=odd.stride(1),
        #stride_odd_n=odd.stride(2),
        #stride_even_b=even.stride(0),
        #stride_even_h=even.stride(1),
        #stride_even_n=even.stride(2),
        even_part = even_part,
        Q=q,
        K=k,
        V=v,
        Bias=bias,
        Out=o,
        #lse,
        #tmp,
        #even,
        #odd,
        softmax_scale = softmax_scale,
        stride_evenb=even_part.stride(0),
        stride_evenh=even_part.stride(1),
        stride_evenm=even_part.stride(2),
        stride_qb=q.stride(0),
        stride_qh=q.stride(1),
        stride_qm =q.stride(2),
        stride_kb =k.stride(0),
        stride_kh=k.stride(1),
        stride_kn=k.stride(2),
        stride_vb=v.stride(0),
        stride_vh=v.stride(1),
        stride_vn=v.stride(2),
        #*bias_strides,
        stride_bb=bias_strides[0],
        stride_bh=bias_strides[1],
        stride_bm=bias_strides[2],
        stride_ob=o.stride(0),
        stride_oh=o.stride(1),
        stride_om=o.stride(2),
        nheads=nheads,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        #seqlen_q_rounded,
        #headdim =d,
        #seqlen_q // 32,
        #seqlen_k // 32,  # key for triton cache (limit number of compilations)
        # Can't use kwargs here because triton autotune expects key to be args, not kwargs
        IS_CAUSAL=causal, BLOCK_HEADDIM=BLOCK_HEADDIM,
        BIAS_TYPE=bias_type,
        #causal,
        #BLOCK_HEADDIM,
        EVEN_M =True,
        EVEN_N =True,
        EVEN_HEADDIM = True,
        BLOCK_M=128,
        BLOCK_N=128,
        BLOCK_E=16,
        BLOCK_T=32,
        num_warps=num_warps,
        num_stages=1,
    )
    #print(even_part[0][0][0],even[0][0][0],odd[0][0][0])
    #print(o)
    return o

def _angular_lsh(q, proj_dir, perm, enc_vec,k,v,ty):
    # shape constraints
    bias=None
    causal=False
    softmax_scale=None
    batch, nheads, seqlen_q,d = q.shape
    _, _, seqlen_k, _ = k.shape
    #assert k.shape == (batch, seqlen_k, nheads, d)
    #assert v.shape == (batch, seqlen_k, nheads, d)
    assert d <= 128, "FlashAttention only support head dimensions up to 128"
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same type"
    assert q.dtype in [torch.float16, torch.bfloat16], "Only support fp16 and bf16"
    assert q.is_cuda and k.is_cuda and v.is_cuda
    #d = 128
    softmax_scale = softmax_scale or 1.0 / math.sqrt(d)

    has_bias = bias is not None
    bias_type = "none"
    if has_bias:
        assert bias.dtype in [q.dtype, torch.float]
        assert bias.is_cuda
        assert bias.dim() == 4
        if bias.stride(-1) != 1:
            bias = bias.contiguous()
        if bias.shape[2:] == (1, seqlen_k):
            bias_type = "vector"
        elif bias.shape[2:] == (seqlen_q, seqlen_k):
            bias_type = "matrix"
        else:
            raise RuntimeError(
                "Last 2 dimensions of bias must be (1, seqlen_k)" " or (seqlen_q, seqlen_k)"
            )
        bias = bias.expand(batch, nheads, seqlen_q, seqlen_k)
    bias_strides = (bias.stride(0), bias.stride(1), bias.stride(2)) if has_bias else (0, 0, 0)
    num_projs = proj_dir.shape[-1]
    batch, nheads, seqlen, d = q.shape
    #assert (proj_dir.shape == (batch, nheads, d, num_projs)) or (proj_dir.shape == (1, 1, d, num_projs))
    assert q.dtype == proj_dir.dtype, "All three tensors must have the same type"
    assert q.dtype in [torch.float16, torch.bfloat16], "Only support fp16 and bf16"
    assert q.is_cuda and proj_dir.is_cuda and perm.is_cuda and enc_vec.is_cuda
    if proj_dir.shape[:2] == (1, 1):
        stride_proj_dirb, stride_proj_dirh = 0, 0
    else:
        stride_proj_dirb, stride_proj_dirh = proj_dir.stride()[:2]

    seqlen_rounded = math.ceil(seqlen / 128) * 128
    num_projs_rounded = 16

    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    BLOCK = 128
    block_n = triton.cdiv(seqlen,BLOCK)
    #buckets = torch.empty((batch, nheads, seqlen), device=in_mat.device, dtype=torch.int32)
    buckets = torch.empty((batch, nheads, block_n,d), device=q.device, dtype=torch.int32)
    even_part =  torch.empty((batch, nheads, block_n,d), device=q.device, dtype=torch.int32)
    #odd_part =  torch.empty((batch, nheads, block_n,32), device=in_mat.device, dtype=torch.int32)
    odd =  torch.empty((batch, nheads, block_n,32), device=q.device, dtype=torch.int32)
    even =  torch.empty((batch, nheads, block_n,32), device=q.device, dtype=torch.int32)
    num_warps = 4 if d <= 64 else 8
    grid = lambda META: (triton.cdiv(seqlen, META["BLOCK_M"]), batch * nheads)
    o = torch.empty_like(q)
    
    _angular_lsh_kernel[grid](
        #even_part=even_part,
        #odd_part=odd_part,
        in_mat=q,
        proj_dir=proj_dir,
        perm=perm,
        enc_vec=enc_vec,
        buckets=buckets,
        #stride_evenb=even_part.stride(0),
        #stride_evenh=even_part.stride(1),
        #stride_evenm=even_part.stride(2),
        #stride_oddb=odd_part.stride(0),
        #stride_oddh=odd_part.stride(1),
        #stride_oddm=odd_part.stride(2),
        stride_in_matb=q.stride(0),
        stride_in_math=q.stride(1),
        stride_in_matm=q.stride(2),
        stride_proj_dirb=stride_proj_dirb,
        stride_proj_dirh=stride_proj_dirh,
        stride_proj_dird=proj_dir.stride(2),
        stride_bucketsb=buckets.stride(0),
        stride_bucketsh=buckets.stride(1),
        stride_bucketsh_blockn=buckets.stride(2),
        nheads=nheads,
        seqlen=seqlen,
        seqlen_rounded=seqlen_rounded,
        headdim=d,
        NUM_PROJ_ROUNDED=num_projs_rounded,
        num_projs=num_projs,
        BLOCK_HEADDIM=BLOCK_HEADDIM,
        BLOCK_M=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
    
    _angular_lsh_kernel_even_odd[grid](
        even_part=even_part,
        #odd_part=odd_part,
        #in_mat=in_mat,
        #proj_dir=proj_dir,
        #perm=perm,
        #enc_vec=enc_vec,
        buckets=buckets,
        stride_evenb=even_part.stride(0),
        stride_evenh=even_part.stride(1),
        stride_evenm=even_part.stride(2),
        #stride_oddb=odd_part.stride(0),
        #stride_oddh=odd_part.stride(1),
        #stride_oddm=odd_part.stride(2),
        #stride_in_matb=in_mat.stride(0),
        #stride_in_math=in_mat.stride(1),
        #stride_in_matm=in_mat.stride(2),
        #stride_proj_dirb=stride_proj_dirb,
        #stride_proj_dirh=stride_proj_dirh,
        #stride_proj_dird=proj_dir.stride(2),
        stride_bucketsb=buckets.stride(0),
        stride_bucketsh=buckets.stride(1),
        stride_bucketsh_blockn=buckets.stride(2),
        nheads=nheads,
        seqlen=seqlen,
        seqlen_rounded=seqlen_rounded,
        headdim=d,
        NUM_PROJ_ROUNDED=num_projs_rounded,
        num_projs=num_projs,
        BLOCK_HEADDIM=BLOCK_HEADDIM,
        BLOCK_M=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
    
    
    #even_part = buckets
    if ty=='4_128':
        _fwd_kernel_flash_4_128[grid](
            #odd = odd,
            #even = even,
            #stride_odd_b=odd.stride(0),
            #stride_odd_h=odd.stride(1),
            #stride_odd_n=odd.stride(2),
            #stride_even_b=even.stride(0),
            #stride_even_h=even.stride(1),
            #stride_even_n=even.stride(2),
            even_part = even_part,
            Q=q,
            K=k,
            V=v,
            Bias=bias,
            Out=o,
            #lse,
            #tmp,
            #even,
            #odd,
            softmax_scale = softmax_scale,
            stride_evenb=even_part.stride(0),
            stride_evenh=even_part.stride(1),
            stride_evenm=even_part.stride(2),
            stride_qb=q.stride(0),
            stride_qh=q.stride(1),
            stride_qm =q.stride(2),
            stride_kb =k.stride(0),
            stride_kh=k.stride(1),
            stride_kn=k.stride(2),
            stride_vb=v.stride(0),
            stride_vh=v.stride(1),
            stride_vn=v.stride(2),
            #*bias_strides,
            stride_bb=bias_strides[0],
            stride_bh=bias_strides[1],
            stride_bm=bias_strides[2],
            stride_ob=o.stride(0),
            stride_oh=o.stride(1),
            stride_om=o.stride(2),
            nheads=nheads,
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            #seqlen_q_rounded,
            #headdim =d,
            #seqlen_q // 32,
            #seqlen_k // 32,  # key for triton cache (limit number of compilations)
            # Can't use kwargs here because triton autotune expects key to be args, not kwargs
            IS_CAUSAL=causal, BLOCK_HEADDIM=BLOCK_HEADDIM,
            BIAS_TYPE=bias_type,
            #causal,
            #BLOCK_HEADDIM,
            EVEN_M =True,
            EVEN_N =True,
            EVEN_HEADDIM = True,
            BLOCK_M=128,
            BLOCK_N=64,
            BLOCK_E=16,
            BLOCK_T=32,
            num_warps=num_warps,
            num_stages=1,
        )
    elif ty=='2_128':
        _fwd_kernel_flash_2_128[grid](
            #odd = odd,
            #even = even,
            #stride_odd_b=odd.stride(0),
            #stride_odd_h=odd.stride(1),
            #stride_odd_n=odd.stride(2),
            #stride_even_b=even.stride(0),
            #stride_even_h=even.stride(1),
            #stride_even_n=even.stride(2),
            even_part = even_part,
            Q=q,
            K=k,
            V=v,
            Bias=bias,
            Out=o,
            #lse,
            #tmp,
            #even,
            #odd,
            softmax_scale = softmax_scale,
            stride_evenb=even_part.stride(0),
            stride_evenh=even_part.stride(1),
            stride_evenm=even_part.stride(2),
            stride_qb=q.stride(0),
            stride_qh=q.stride(1),
            stride_qm =q.stride(2),
            stride_kb =k.stride(0),
            stride_kh=k.stride(1),
            stride_kn=k.stride(2),
            stride_vb=v.stride(0),
            stride_vh=v.stride(1),
            stride_vn=v.stride(2),
            #*bias_strides,
            stride_bb=bias_strides[0],
            stride_bh=bias_strides[1],
            stride_bm=bias_strides[2],
            stride_ob=o.stride(0),
            stride_oh=o.stride(1),
            stride_om=o.stride(2),
            nheads=nheads,
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            #seqlen_q_rounded,
            #headdim =d,
            #seqlen_q // 32,
            #seqlen_k // 32,  # key for triton cache (limit number of compilations)
            # Can't use kwargs here because triton autotune expects key to be args, not kwargs
            IS_CAUSAL=causal, BLOCK_HEADDIM=BLOCK_HEADDIM,
            BIAS_TYPE=bias_type,
            #causal,
            #BLOCK_HEADDIM,
            EVEN_M =True,
            EVEN_N =True,
            EVEN_HEADDIM = True,
            BLOCK_M=128,
            BLOCK_N=64,
            BLOCK_E=16,
            BLOCK_T=32,
            num_warps=num_warps,
            num_stages=1,
        )
    elif ty=='2_64':
        _fwd_kernel_flash_2_64[grid](
            #odd = odd,
            #even = even,
            #stride_odd_b=odd.stride(0),
            #stride_odd_h=odd.stride(1),
            #stride_odd_n=odd.stride(2),
            #stride_even_b=even.stride(0),
            #stride_even_h=even.stride(1),
            #stride_even_n=even.stride(2),
            even_part = even_part,
            Q=q,
            K=k,
            V=v,
            Bias=bias,
            Out=o,
            #lse,
            #tmp,
            #even,
            #odd,
            softmax_scale = softmax_scale,
            stride_evenb=even_part.stride(0),
            stride_evenh=even_part.stride(1),
            stride_evenm=even_part.stride(2),
            stride_qb=q.stride(0),
            stride_qh=q.stride(1),
            stride_qm =q.stride(2),
            stride_kb =k.stride(0),
            stride_kh=k.stride(1),
            stride_kn=k.stride(2),
            stride_vb=v.stride(0),
            stride_vh=v.stride(1),
            stride_vn=v.stride(2),
            #*bias_strides,
            stride_bb=bias_strides[0],
            stride_bh=bias_strides[1],
            stride_bm=bias_strides[2],
            stride_ob=o.stride(0),
            stride_oh=o.stride(1),
            stride_om=o.stride(2),
            nheads=nheads,
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            #seqlen_q_rounded,
            #headdim =d,
            #seqlen_q // 32,
            #seqlen_k // 32,  # key for triton cache (limit number of compilations)
            # Can't use kwargs here because triton autotune expects key to be args, not kwargs
            IS_CAUSAL=causal, BLOCK_HEADDIM=BLOCK_HEADDIM,
            BIAS_TYPE=bias_type,
            #causal,
            #BLOCK_HEADDIM,
            EVEN_M =True,
            EVEN_N =True,
            EVEN_HEADDIM = True,
            BLOCK_M=128,
            BLOCK_N=64,
            BLOCK_E=16,
            BLOCK_T=32,
            num_warps=num_warps,
            num_stages=1,
        )
    elif ty=='2_128':
        _fwd_kernel_flash_2_128[grid](
            #odd = odd,
            #even = even,
            #stride_odd_b=odd.stride(0),
            #stride_odd_h=odd.stride(1),
            #stride_odd_n=odd.stride(2),
            #stride_even_b=even.stride(0),
            #stride_even_h=even.stride(1),
            #stride_even_n=even.stride(2),
            even_part = even_part,
            Q=q,
            K=k,
            V=v,
            Bias=bias,
            Out=o,
            #lse,
            #tmp,
            #even,
            #odd,
            softmax_scale = softmax_scale,
            stride_evenb=even_part.stride(0),
            stride_evenh=even_part.stride(1),
            stride_evenm=even_part.stride(2),
            stride_qb=q.stride(0),
            stride_qh=q.stride(1),
            stride_qm =q.stride(2),
            stride_kb =k.stride(0),
            stride_kh=k.stride(1),
            stride_kn=k.stride(2),
            stride_vb=v.stride(0),
            stride_vh=v.stride(1),
            stride_vn=v.stride(2),
            #*bias_strides,
            stride_bb=bias_strides[0],
            stride_bh=bias_strides[1],
            stride_bm=bias_strides[2],
            stride_ob=o.stride(0),
            stride_oh=o.stride(1),
            stride_om=o.stride(2),
            nheads=nheads,
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            #seqlen_q_rounded,
            #headdim =d,
            #seqlen_q // 32,
            #seqlen_k // 32,  # key for triton cache (limit number of compilations)
            # Can't use kwargs here because triton autotune expects key to be args, not kwargs
            IS_CAUSAL=causal, BLOCK_HEADDIM=BLOCK_HEADDIM,
            BIAS_TYPE=bias_type,
            #causal,
            #BLOCK_HEADDIM,
            EVEN_M =True,
            EVEN_N =True,
            EVEN_HEADDIM = True,
            BLOCK_M=128,
            BLOCK_N=64,
            BLOCK_E=16,
            BLOCK_T=32,
            num_warps=num_warps,
            num_stages=1,
        )
    elif ty=='2_32':
        _fwd_kernel_flash_2_32[grid](
            #odd = odd,
            #even = even,
            #stride_odd_b=odd.stride(0),
            #stride_odd_h=odd.stride(1),
            #stride_odd_n=odd.stride(2),
            #stride_even_b=even.stride(0),
            #stride_even_h=even.stride(1),
            #stride_even_n=even.stride(2),
            even_part = even_part,
            Q=q,
            K=k,
            V=v,
            Bias=bias,
            Out=o,
            #lse,
            #tmp,
            #even,
            #odd,
            softmax_scale = softmax_scale,
            stride_evenb=even_part.stride(0),
            stride_evenh=even_part.stride(1),
            stride_evenm=even_part.stride(2),
            stride_qb=q.stride(0),
            stride_qh=q.stride(1),
            stride_qm =q.stride(2),
            stride_kb =k.stride(0),
            stride_kh=k.stride(1),
            stride_kn=k.stride(2),
            stride_vb=v.stride(0),
            stride_vh=v.stride(1),
            stride_vn=v.stride(2),
            #*bias_strides,
            stride_bb=bias_strides[0],
            stride_bh=bias_strides[1],
            stride_bm=bias_strides[2],
            stride_ob=o.stride(0),
            stride_oh=o.stride(1),
            stride_om=o.stride(2),
            nheads=nheads,
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            #seqlen_q_rounded,
            #headdim =d,
            #seqlen_q // 32,
            #seqlen_k // 32,  # key for triton cache (limit number of compilations)
            # Can't use kwargs here because triton autotune expects key to be args, not kwargs
            IS_CAUSAL=causal, BLOCK_HEADDIM=BLOCK_HEADDIM,
            BIAS_TYPE=bias_type,
            #causal,
            #BLOCK_HEADDIM,
            EVEN_M =True,
            EVEN_N =True,
            EVEN_HEADDIM = True,
            BLOCK_M=128,
            BLOCK_N=64,
            BLOCK_E=16,
            BLOCK_T=32,
            num_warps=num_warps,
            num_stages=1,
        )
    '''
    seqlen_q_rounded = math.ceil(seqlen_q / 128) * 128
    #lse = torch.empty((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)
    _fwd_kernel_d_2[grid](
        #even_part,
        #even_part.stride(0),
        #even_part.stride(1),
        #even_part.stride(2),
        q,
        k,
        v,
        bias,
        o,
        #lse,
        softmax_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        *bias_strides,
        o.stride(0),
        o.stride(1),
        o.stride(2),
        nheads,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        d,
        seqlen_q // 32,
        seqlen_k // 32,  # key for triton cache (limit number of compilations)
        # Can't use kwargs here because triton autotune expects key to be args, not kwargs
        # IS_CAUSAL=causal, BLOCK_HEADDIM=d,
        bias_type,
        causal,
        BLOCK_HEADDIM,
        #EVEN_M =True,
        #EVEN_N =True,
        #EVEN_HEADDIM = True,
        BLOCK_M=BLOCK,
        BLOCK_N=128,
        num_warps=num_warps,
        num_stages=1,
    )
    '''
    #print(even_part[0][0][0],even[0][0][0],odd[0][0][0])
    #print("-------------------------------------------------")
    return o

def _flash_attn_forward(q, k, v, bias=None, causal=False, softmax_scale=None):
    '''
    # shape constraints
    #q = q.transpose(1, 2)
    #k = k.transpose(1, 2)
    #v = v.transpose(1, 2)
    #batch, seqlen_q, nheads, d = k.shape
    batch, nheads,seqlen_q, d = k.shape
    #_, seqlen_k, _, _ = k.shape
    _, _,seqlen_k, _ = k.shape
    #assert k.shape == (batch, seqlen_k, nheads, d)
    #assert v.shape == (batch, seqlen_k, nheads, d)
    assert d <= 128, "FlashAttention only support head dimensions up to 128"
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same type"
    assert q.dtype in [torch.float16, torch.bfloat16], "Only support fp16 and bf16"
    assert q.is_cuda and k.is_cuda and v.is_cuda
    softmax_scale = softmax_scale or 1.0 / math.sqrt(d)

    has_bias = bias is not None
    bias_type = "none"
    if has_bias:
        assert bias.dtype in [q.dtype, torch.float]
        assert bias.is_cuda
        assert bias.dim() == 4
        if bias.stride(-1) != 1:
            bias = bias.contiguous()
        if bias.shape[2:] == (1, seqlen_k):
            bias_type = "vector"
        elif bias.shape[2:] == (seqlen_q, seqlen_k):
            bias_type = "matrix"
        else:
            raise RuntimeError(
                "Last 2 dimensions of bias must be (1, seqlen_k)" " or (seqlen_q, seqlen_k)"
            )
        bias = bias.expand(batch, nheads, seqlen_q, seqlen_k)
    bias_strides = (bias.stride(0), bias.stride(1), bias.stride(2)) if has_bias else (0, 0, 0)

    seqlen_q_rounded = math.ceil(seqlen_q / 128) * 128
    lse = torch.empty((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)
    o = torch.empty_like(q)

    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    BLOCK = 128
    num_warps = 4 if d <= 64 else 8
    grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), batch * nheads)
    '''
    bias=None
    causal=False
    softmax_scale=None
    batch, nheads, seqlen_q,d = q.shape
    _, _, seqlen_k, _ = k.shape
    #assert k.shape == (batch, seqlen_k, nheads, d)
    #assert v.shape == (batch, seqlen_k, nheads, d)
    assert d <= 128, "FlashAttention only support head dimensions up to 128"
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same type"
    assert q.dtype in [torch.float16, torch.bfloat16], "Only support fp16 and bf16"
    assert q.is_cuda and k.is_cuda and v.is_cuda
    d = 64
    softmax_scale = softmax_scale or 1.0 / math.sqrt(d)

    has_bias = bias is not None
    bias_type = "none"
    if has_bias:
        assert bias.dtype in [q.dtype, torch.float]
        assert bias.is_cuda
        assert bias.dim() == 4
        if bias.stride(-1) != 1:
            bias = bias.contiguous()
        if bias.shape[2:] == (1, seqlen_k):
            bias_type = "vector"
        elif bias.shape[2:] == (seqlen_q, seqlen_k):
            bias_type = "matrix"
        else:
            raise RuntimeError(
                "Last 2 dimensions of bias must be (1, seqlen_k)" " or (seqlen_q, seqlen_k)"
            )
        bias = bias.expand(batch, nheads, seqlen_q, seqlen_k)
    bias_strides = (bias.stride(0), bias.stride(1), bias.stride(2)) if has_bias else (0, 0, 0)
    batch, nheads, seqlen, d = q.shape
    #assert (proj_dir.shape == (batch, nheads, d, num_projs)) or (proj_dir.shape == (1, 1, d, num_projs))


    seqlen_rounded = math.ceil(seqlen / 128) * 128
    num_projs_rounded = 16

    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    BLOCK = 128
    block_n = triton.cdiv(seqlen,BLOCK)
    #buckets = torch.empty((batch, nheads, seqlen), device=in_mat.device, dtype=torch.int32)
    buckets = torch.empty((batch, nheads, block_n,d), device=q.device, dtype=torch.int32)
    even_part =  torch.empty((batch, nheads, block_n,64), device=q.device, dtype=torch.int32)
    #odd_part =  torch.empty((batch, nheads, block_n,32), device=in_mat.device, dtype=torch.int32)
    odd =  torch.empty((batch, nheads, block_n,32), device=q.device, dtype=torch.int32)
    even =  torch.empty((batch, nheads, block_n,32), device=q.device, dtype=torch.int32)
    num_warps = 4 if d <= 64 else 8
    grid = lambda META: (triton.cdiv(seqlen, META["BLOCK_M"]), batch * nheads)
    o = torch.empty_like(q)
    seqlen_q_rounded = math.ceil(seqlen_q / 128) * 128
    lse = torch.empty((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)
    _fwd_kernel[grid](
        q,
        k,
        v,
        bias,
        o,
        lse,
        softmax_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        *bias_strides,
        o.stride(0),
        o.stride(1),
        o.stride(2),
        nheads,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        d,
        seqlen_q // 32,
        seqlen_k // 32,  # key for triton cache (limit number of compilations)
        # Can't use kwargs here because triton autotune expects key to be args, not kwargs
        # IS_CAUSAL=causal, BLOCK_HEADDIM=d,
        bias_type,
        causal,
        BLOCK_HEADDIM,
        #EVEN_M =True,
        #EVEN_N =True,
        #EVEN_HEADDIM = True,
        BLOCK_M=BLOCK,
        BLOCK_N=64,
        num_warps=num_warps,
        num_stages=1,
    )
    return o  # softmax_scale could have been updated

class AngularLSHTriton(torch.nn.Module):
    """
    inputs:
        - num_projs: a positive integer that determines the number of random projections used by hash function
        - dim: positive integer that determines the dimension of input vectors
        - mat: a tensor whose last shape is equal to dim and gets hashed by the lsh function
    output:
        - buckets: a tensor with shape mat.shape[:-1] and each entry is an integer in [0, 2^num_proj - 1]
    """
    def __init__(self, num_projs, dim, rng=None):
        super().__init__()
        self.num_projs = num_projs

        if num_projs > 0:
            self.register_buffer('perm', self._unit_hamming_distance_array(self.num_projs), persistent=False)
            self.register_buffer('proj_dir', torch.randn(dim + (num_projs,), generator=rng), persistent=False)
            self.register_buffer('enc_vec', 2 ** torch.arange(self.num_projs).view(1, 1, 1, -1), persistent=False)
        else:
            raise ValueError("Invalid value for num_projs")

    def _unit_hamming_distance_array(self, size_n):
        if size_n == 1:
            return torch.tensor([0, 1], dtype=torch.int32)
        a = self._unit_hamming_distance_array(size_n - 1)
        b = torch.concat([a, torch.flip(a, dims=[0]) + 2 ** (size_n - 1)], 0)
        return b if b.stride(-1) == 1 else b.contiguous()

    def hash_torch(self, mat):
        print(mat.shape,self.proj_dir.shape)
        mask = torch.einsum('...nd,...dr -> ...nr', mat, self.proj_dir)
        mask = mask > 0
        bin_ids = (mask * self.enc_vec).sum(-1)
        return self.perm[bin_ids]

    def hash_triton(self, q,k,v,ty):
        return _angular_lsh(q, self.proj_dir, self.perm, self.enc_vec,k,v,ty)
    
    def flashattn_triton(self, q,k,v):
        return _flash_attn(q, self.proj_dir, self.perm, self.enc_vec,k,v)
    def flashattn_v2_triton(self, q,k,v):
        return _flash_attn_forward(q,k,v)

    def __repr__(self):
        return f"AngularLSH(num_proj={self.num_projs}, proj_dir.shape={self.proj_dir.shape})"
