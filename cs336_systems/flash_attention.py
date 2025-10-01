import torch
import math
from einops import einsum
import triton
import triton.language as tl

def flash_attention_backward_recomputation(Q, K, V, O, grad_O, L, is_causal):
        d_model = Q.size(-1)
        scale = 1.0 / math.sqrt(d_model)

        D = (O * grad_O).sum(dim=-1) #
        S = einsum(Q, K, "... q d, ... k d -> ... q k") * scale
        if is_causal:
            mask = torch.tril(torch.ones(S.shape[-2], S.shape[-1], device=S.device))
            S = S.masked_fill(mask==0, - torch.inf)

        P = torch.exp(S - L[..., None]) # q x k
        grad_V = einsum(P, grad_O, "... q k, ... q d-> ... k d")
        grad_P = einsum(grad_O, V, "... q d, ... k d -> ... q k")
        grad_S = P * (grad_P - D[..., None])
        grad_Q = einsum(grad_S, K, "... q k, ... k d -> ... q d") * scale
        grad_K = einsum(grad_S, Q, "... q k, ... q d -> ... k d") * scale
        return grad_Q, grad_K, grad_V


class FlashAttention2(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        is_causal: bool = False
    ) -> torch.Tensor:

        batch_size, n_queries, d_model = Q.shape
        sqrt_d = math.sqrt(d_model)
        n_keys = K.size(1)

        tile_size_q, tile_size_k = 16, 16

        O = torch.empty(batch_size, n_queries, d_model)
        L = torch.empty(batch_size, n_queries)


        for q_index in range(n_queries // tile_size_q):
            O_tile = torch.zeros(batch_size, tile_size_q, d_model)
            Q_tile = Q[:, q_index * tile_size_q : (q_index + 1) * tile_size_q]

            l = torch.zeros(batch_size, tile_size_q)
            m = torch.full((batch_size, tile_size_q,), float('-inf'))

            for k_index in range(n_keys // tile_size_k):
                m_prev = m
                K_tile = K[:, k_index * tile_size_k : (k_index + 1) * tile_size_k]
                V_tile = V[:, k_index * tile_size_k : (k_index + 1) * tile_size_k]
                S = einsum(Q_tile, K_tile, "... q d, ... k d -> ... q k") / sqrt_d
                m = torch.max(m_prev, S.max(dim=-1).values)
                P = torch.exp(S - m[:, :, None])
                l = torch.exp(m_prev - m) * l + P.sum(dim=-1)

                O_tile = einsum(torch.diag_embed(torch.exp(m_prev - m)), O_tile, "... q q, ... q d -> ... q d") + \
                        einsum(P, V_tile, "... q k, ... k d -> ... q d")


            O_tile = einsum(torch.diag_embed(1.0 / l), O_tile, "... q q, ... q d -> ... q d")
            O[:, q_index * tile_size_q : (q_index + 1) * tile_size_q] = O_tile
            L[:, q_index * tile_size_q : (q_index + 1) * tile_size_q] = (m + torch.log(l))

        ctx.save_for_backward(O, L, Q, K, V)
        ctx.is_causal = is_causal
        ctx.scale = 1.0 /sqrt_d
        return O


    @staticmethod
    @torch.compile
    def backward(ctx, grad_O):
        O, L, Q, K, V = ctx.saved_tensors
        grad_Q, grad_K, grad_V = flash_attention_backward_recomputation(Q, K, V, O, grad_O, L, ctx.is_causal)
        return grad_Q, grad_K, grad_V, None


@triton.jit
def _flash_fwd_inner_kernel(
    output, l, m,
    queries, Kt_ptr_base, V_ptr_base,
    stride_kd, stride_kk,
    stride_vk, stride_vd,
    N_KEYS, D,
    scale,
    query_tile_index: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    stage: tl.constexpr
):

    # causal: handles K/V blocks to the left of the diagonal blocks
    if stage == 1:
        lo, hi = 0, query_tile_index * Q_TILE_SIZE
    # causal: handles diagonal blocks
    elif stage == 2:
        lo, hi = query_tile_index * Q_TILE_SIZE, (query_tile_index + 1) * Q_TILE_SIZE
    # non causal
    else:
        lo, hi = 0, N_KEYS

    # making K/V block ptrs in the inner kernek to avoid passing block ptrs between triton kernels (causing compiler error)
    Kt_block_ptr = tl.make_block_ptr(
        Kt_ptr_base,
        shape=(D, N_KEYS),
        strides=(stride_kd, stride_kk),
        offsets=(0, 0),
        block_shape=(D, K_TILE_SIZE),
        order=(0, 1),
    )
    Kt_block_ptr = Kt_block_ptr.advance((0, K_TILE_SIZE * (lo // K_TILE_SIZE)))


    V_block_ptr = tl.make_block_ptr(
        V_ptr_base,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = V_block_ptr.advance((K_TILE_SIZE * (lo // K_TILE_SIZE), 0))

    q_offsets = Q_TILE_SIZE * query_tile_index + tl.arange(0, Q_TILE_SIZE)

    for k_index in range(lo // K_TILE_SIZE, tl.cdiv(hi, K_TILE_SIZE)):
        m_prev = m
        keys_t = tl.load(Kt_block_ptr, boundary_check=(0, 1), padding_option="zero") # (D, K_TILE_SIZE)
        values = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero") # (K_TILE_SIZE, D)

        # S = tl.zeros((Q_TILE_SIZE, K_TILE_SIZE), dtype=tl.float32)
        S = tl.dot(queries, keys_t) * scale # (Q_TILE_SIZE, K_TILE_SIZE)

        if stage == 2:
            k_offsets = k_index * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            mask = q_offsets[:, None] < k_offsets[None, :]
            S = tl.where(mask, - 1e6, S)


        m = tl.maximum(tl.max(S, axis=-1), m_prev)
        P = tl.math.exp(S - m[:, None])

        corrector = tl.math.exp(m_prev - m)
        l = corrector * l + tl.sum(P, axis=-1)

        output = corrector[:, None] * output + tl.dot(P.to(values.dtype), values)

        Kt_block_ptr = Kt_block_ptr.advance((0, K_TILE_SIZE))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))


    return output, l, m


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr
):
    stage = 3 if is_causal else 1

    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )


    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    output = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    l = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    m = tl.full((Q_TILE_SIZE,), - float("inf"), dtype=tl.float32)
    queries = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero") # (Q_TILE_SIZE, D)


    K_ptr_base = K_ptr + batch_index * stride_kb
    V_ptr_base = V_ptr + batch_index * stride_vb

    if stage == 1 or stage == 3:
        output, l, m = _flash_fwd_inner_kernel(
            output, l, m, queries, K_ptr_base, V_ptr_base,
            stride_kd, stride_kk,
            stride_vk, stride_vd,
            N_KEYS, D,
            scale,
            query_tile_index, Q_TILE_SIZE, K_TILE_SIZE, 4 - stage
        )

    if stage == 3:
        output, l, m = _flash_fwd_inner_kernel(
            output, l, m, queries, K_ptr_base, V_ptr_base,
            stride_kd, stride_kk,
            stride_vk, stride_vd,
            N_KEYS, D,
            scale,
            query_tile_index, Q_TILE_SIZE, K_TILE_SIZE, 2
        )


    output /= l[:, None]
    logsum = tl.log(l) + m

    tl.store(O_block_ptr, output.to(O_block_ptr.type.element_ty), boundary_check=(0, 1))
    tl.store(L_block_ptr, logsum.to(L_block_ptr.type.element_ty), boundary_check=(0, ))

@triton.jit
def flash_bwd_D(
    O_ptr, grad_O_ptr, D_ptr,
    stride_ob, stride_oq, stride_od,
    stride_dob, stride_doq, stride_dod,
    stride_db, stride_dq,
    N_QUERIES,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr
):
    # Program indices
    output_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(output_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    grad_O_block_ptr = tl.make_block_ptr(
        grad_O_ptr + batch_index * stride_dob,
        shape=(N_QUERIES, D),
        strides=(stride_doq, stride_dod),
        offsets=(output_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )


    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db,
        shape=(N_QUERIES,),
        strides=(stride_dq,),
        offsets=(output_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    d = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    o = tl.load(O_block_ptr, boundary_check=(0, 1), padding_option="zero")
    d_o = tl.load(grad_O_block_ptr, boundary_check=(0, 1), padding_option="zero")
    d = tl.sum(o * d_o, axis=-1)
    tl.store(D_block_ptr, d.to(D_block_ptr.type.element_ty), boundary_check=(0, ))



@triton.jit
def flash_bwd_dQ(
    Q_ptr, K_ptr, V_ptr, grad_O_ptr, L_ptr,
    grad_Q_ptr, D_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_dob, stride_doq, stride_dod,
    stride_lb, stride_lq,
    stride_dqb, stride_dqq, stride_dqd,
    stride_db, stride_dq,
    scale,
    stage,
    N_QUERIES, N_KEYS,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)


    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    Kt_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(D, N_KEYS),
        strides=(stride_kd, stride_kk),
        offsets=(0, 0),
        block_shape=(D, K_TILE_SIZE),
        order=(0, 1),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )



    grad_O_block_ptr = tl.make_block_ptr(
        grad_O_ptr + batch_index * stride_dob,
        shape=(N_QUERIES, D),
        strides=(stride_doq, stride_dod),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )


    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db,
        shape=(N_QUERIES,),
        strides=(stride_dq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    grad_Q_block_ptr = tl.make_block_ptr(
        grad_Q_ptr + batch_index * stride_dqb,
        shape=(N_QUERIES, D),
        strides=(stride_dqq, stride_dqd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    queries = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero") # (Q_TILE_SIZE, D)
    grad_outputs = tl.load(grad_O_block_ptr, boundary_check=(0, 1), padding_option="zero") # (Q_TILE_SIZE, D)
    d = tl.load(D_block_ptr, boundary_check=(0,), padding_option="zero") # (Q_TILE_SIZE,)
    l = tl.load(L_block_ptr,  boundary_check=(0,), padding_option="zero") # (Q_TILE_SIZE,)
    grad_queries = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    q_offsets = Q_TILE_SIZE * query_tile_index + tl.arange(0, Q_TILE_SIZE)

    for k_index in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        keys_t = tl.load(Kt_block_ptr, boundary_check=(0, 1), padding_option="zero") # (D, K_TILE_SIZE)
        values = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero") # (K_TILE_SIZE, D)

        S = tl.zeros((Q_TILE_SIZE, K_TILE_SIZE), dtype=tl.float32)
        S = tl.dot(queries, keys_t, acc=S) * scale # (Q_TILE_SIZE, K_TILE_SIZE)
        P = tl.exp(S - l[:,None])

        if stage == 3:
            k_offsets = k_index * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            mask = q_offsets[:, None] < k_offsets[None, :]
            P = tl.where(mask, 0.0, P)

        grad_P = tl.zeros((Q_TILE_SIZE, K_TILE_SIZE), dtype=tl.float32)
        grad_P = tl.dot(grad_outputs, tl.trans(values), acc=grad_P)
        grad_S = P * (grad_P - d[:, None])
        grad_queries = tl.dot(grad_S, tl.trans(keys_t), acc=grad_queries)

        Kt_block_ptr = Kt_block_ptr.advance((0, K_TILE_SIZE))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    grad_queries = grad_queries * scale
    tl.store(grad_Q_block_ptr, grad_queries.to(grad_Q_block_ptr.type.element_ty), boundary_check=(0, 1))






@triton.jit
def flash_bwd_dKdV(
    Q_ptr, K_ptr, V_ptr, grad_O_ptr, L_ptr,
    D_ptr, grad_K_ptr, grad_V_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_dob, stride_doq, stride_dod,
    stride_lb, stride_lq,
    stride_db, stride_dq,
    stride_dkb, stride_dkk, stride_dkd,
    stride_dvb, stride_dvk, stride_dvd,
    scale,
    stage,
    N_QUERIES, N_KEYS,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr
):
    # Program indices
    key_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)


    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    Kt_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(D, N_KEYS),
        strides=(stride_kd, stride_kk),
        offsets=(0, key_tile_index * K_TILE_SIZE),
        block_shape=(D, K_TILE_SIZE),
        order=(0, 1),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )


    grad_O_block_ptr = tl.make_block_ptr(
        grad_O_ptr + batch_index * stride_dob,
        shape=(N_QUERIES, D),
        strides=(stride_doq, stride_dod),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )


    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db,
        shape=(N_QUERIES,),
        strides=(stride_dq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )



    grad_V_block_ptr = tl.make_block_ptr(
        grad_V_ptr + batch_index * stride_dvb,
        shape=(N_KEYS, D),
        strides=(stride_dvk, stride_dvd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    grad_K_block_ptr = tl.make_block_ptr(
        grad_K_ptr + batch_index * stride_dkb,
        shape=(N_KEYS, D),
        strides=(stride_dkk, stride_dkd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )


    keys_t = tl.load(Kt_block_ptr, boundary_check=(0, 1), padding_option="zero") # (D, K_TILE_SIZE)
    values = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero") # (K_TILE_SIZE, D)

    grad_keys = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)
    grad_values = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)

    for q_index in range(tl.cdiv(N_QUERIES, Q_TILE_SIZE)):
        queries = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero") # (Q_TILE_SIZE, D)
        grad_outputs = tl.load(grad_O_block_ptr, boundary_check=(0, 1), padding_option="zero") # (Q_TILE_SIZE, D)
        d = tl.load(D_block_ptr, boundary_check=(0,), padding_option="zero") # (Q_TILE_SIZE,)

        q_offsets = Q_TILE_SIZE * q_index + tl.arange(0, Q_TILE_SIZE)
        l = tl.load(L_block_ptr,  boundary_check=(0,), padding_option="zero") # (Q_TILE_SIZE,)

        S = tl.zeros((Q_TILE_SIZE, K_TILE_SIZE), dtype=tl.float32)
        S = tl.dot(queries, keys_t, acc=S) * scale # (Q_TILE_SIZE, K_TILE_SIZE)
        P = tl.exp(S - l[:,None])

        if stage == 3:
            k_offsets = key_tile_index * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            mask = q_offsets[:, None] < k_offsets[None, :]
            P = tl.where(mask, 0.0, P)

        grad_values = tl.dot(tl.trans(P), grad_outputs, acc=grad_values)

        grad_P = tl.zeros((Q_TILE_SIZE, K_TILE_SIZE), dtype=tl.float32)
        grad_P = tl.dot(grad_outputs, tl.trans(values), acc=grad_P)
        grad_S = P * (grad_P - d[:, None])
        grad_keys = tl.dot(tl.trans(grad_S), queries, acc=grad_keys)

        Q_block_ptr = Q_block_ptr.advance((Q_TILE_SIZE, 0))
        grad_O_block_ptr = grad_O_block_ptr.advance((Q_TILE_SIZE, 0))
        D_block_ptr = D_block_ptr.advance((Q_TILE_SIZE,))
        L_block_ptr = L_block_ptr.advance((Q_TILE_SIZE,))

    grad_keys = grad_keys * scale
    tl.store(grad_K_block_ptr, grad_keys.to(grad_K_block_ptr.type.element_ty), boundary_check=(0, 1))
    tl.store(grad_V_block_ptr, grad_values.to(grad_V_block_ptr.type.element_ty), boundary_check=(0, 1))
















class TritonFlashAttention2(torch.autograd.Function):
    @staticmethod
    @torch.compile
    def forward(
        ctx,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        is_causal: bool = False
    ) -> torch.Tensor:


        batch_size, n_queries, d_model = Q.shape
        scale = 1. / d_model ** 0.5
        n_keys = K.size(1)


        O = torch.empty(batch_size, n_queries, d_model, device=Q.device, dtype=Q.dtype)
        L = torch.empty(batch_size, n_queries, device=Q.device, dtype=Q.dtype)

        assert Q.is_cuda and K.is_cuda and V.is_cuda, "Expected CUDA tensors"
        assert Q.is_contiguous() and K.is_contiguous() and V.is_contiguous(), "Our pointer arithmetic will assume contiguous Q, K, V"
        assert K.shape[-1] == V.shape[-1] == d_model, "Dimension mismatch"

        ctx.Q_TILE_SIZE = 16
        ctx.K_TILE_SIZE = 16
        ctx.is_causal = is_causal
        ctx.scale = scale

        flash_fwd_kernel[(math.ceil(n_queries / ctx.Q_TILE_SIZE), batch_size,)](
            Q, K, V,
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            n_queries, n_keys,
            scale,
            d_model,
            ctx.Q_TILE_SIZE,
            ctx.K_TILE_SIZE,
            is_causal
        )

        ctx.save_for_backward(O, L, Q, K, V)

        return O



    @staticmethod
    def backward(ctx, grad_O):
        # O, L, Q, K, V = ctx.saved_tensors
        # grad_Q, grad_K, grad_V = flash_attention_backward_recomputation(Q, K, V, O, grad_O, L, ctx.is_causal)
        # return grad_Q, grad_K, grad_V, None
        O, L, Q, K, V = ctx.saved_tensors

        batch_size, n_queries, d_model = O.shape
        n_keys = K.size(1)
        stage = 3 if ctx.is_causal else 1

        D = torch.empty_like(grad_O, device=grad_O.device, dtype=grad_O.dtype) # batch_size x n_queries

        flash_bwd_D[(math.ceil(n_queries / ctx.Q_TILE_SIZE), batch_size)](
            O, grad_O, D,
            O.stride(0), O.stride(1), O.stride(2),
            grad_O.stride(0), grad_O.stride(1), grad_O.stride(2),
            D.stride(0), D.stride(1),
            n_queries,
            d_model,
            ctx.Q_TILE_SIZE
        )


        grad_Q = torch.empty_like(Q, device=grad_O.device, dtype=grad_O.dtype) # batch_size x n_queries
        flash_bwd_dQ[(math.ceil(n_queries / ctx.Q_TILE_SIZE), batch_size)](
            Q, K, V, grad_O, L,
            grad_Q, D,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            grad_O.stride(0), grad_O.stride(1), grad_O.stride(2),
            L.stride(0), L.stride(1),
            grad_Q.stride(0), grad_Q.stride(1), grad_Q.stride(2),
            D.stride(0), D.stride(1),
            ctx.scale,
            stage,
            n_queries, n_keys,
            d_model,
            ctx.Q_TILE_SIZE,
            ctx.K_TILE_SIZE
        )


        grad_V = torch.empty_like(V, device=grad_O.device, dtype=grad_O.dtype)
        grad_K = torch.empty_like(K, device=grad_O.device, dtype=grad_O.dtype)
        flash_bwd_dKdV[(math.ceil(n_keys / ctx.K_TILE_SIZE), batch_size)](
            Q, K, V, grad_O, L,
            D, grad_K, grad_V,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            grad_O.stride(0), grad_O.stride(1), grad_O.stride(2),
            L.stride(0), L.stride(1),
            D.stride(0), D.stride(1),
            grad_K.stride(0), grad_K.stride(1), grad_K.stride(2),
            grad_V.stride(0), grad_V.stride(1), grad_V.stride(2),
            ctx.scale,
            stage,
            n_queries, n_keys,
            d_model,
            ctx.Q_TILE_SIZE,
            ctx.K_TILE_SIZE
        )



        return grad_Q, grad_K, grad_V, None







