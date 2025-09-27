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

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
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

    queries = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero") # (Q_TILE_SIZE, D)
    q_offsets = Q_TILE_SIZE * query_tile_index + tl.arange(0, Q_TILE_SIZE)

    output = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    l = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    m = tl.full((Q_TILE_SIZE,), - float("inf"), dtype=tl.float32)


    for k_index in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        m_prev = m
        keys = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero") # (K_TILE_SIZE, D)
        values = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero") # (K_TILE_SIZE, D)

        # S = tl.zeros((Q_TILE_SIZE, K_TILE_SIZE), dtype=tl.float32)
        S = tl.dot(queries, tl.trans(keys)) * scale # (Q_TILE_SIZE, K_TILE_SIZE)

        if is_causal:
            k_offsets = k_index * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            mask = q_offsets[:, None] < k_offsets[None, :]
            S = tl.where(mask, - 1e6, S)


        m = tl.maximum(tl.max(S, axis=-1), m_prev)
        P = tl.math.exp(S - m[:, None])

        corrector = tl.math.exp(m_prev - m)
        l = corrector * l + tl.sum(P, axis=-1)

        output = corrector[:, None] * output + tl.dot(P.to(values.dtype), values)

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))


    output /= l[:, None]
    logsum = tl.log(l) + m

    tl.store(O_block_ptr, output.to(O_block_ptr.type.element_ty), boundary_check=(0, 1))
    tl.store(L_block_ptr, logsum.to(L_block_ptr.type.element_ty), boundary_check=(0, ))



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


        O = torch.empty(batch_size, n_queries, d_model, device=Q.device)
        L = torch.empty(batch_size, n_queries, device=Q.device)

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
        O, L, Q, K, V = ctx.saved_tensors
        grad_Q, grad_K, grad_V = flash_attention_backward_recomputation(Q, K, V, O, grad_O, L, ctx.is_causal)
        return grad_Q, grad_K, grad_V, None










