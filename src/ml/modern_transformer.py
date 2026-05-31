"""Modern decoder-only Transformer blocks (RoPE + RMSNorm + SwiGLU + pre-norm).

Trained from scratch on the Infineon step vocabulary — same inference API as
`ProcessTransformerNet`, but uses Llama/Qwen-style components instead of the
2017-vintage absolute-positional post-norm encoder stack.
"""
from __future__ import annotations

import math

from src.ml.transformer_model import TransformerConfig, require_torch

try:
    import torch
    from torch import Tensor, nn
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    Tensor = object  # type: ignore[misc,assignment]
    nn = object  # type: ignore[assignment]


if torch is not None:

    class RMSNorm(nn.Module):
        def __init__(self, dim: int, eps: float = 1e-6) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))
            self.eps = eps

        def forward(self, x: Tensor) -> Tensor:
            var = x.pow(2).mean(dim=-1, keepdim=True)
            return x * torch.rsqrt(var + self.eps) * self.weight

    class RotaryEmbedding(nn.Module):
        def __init__(self, head_dim: int, max_len: int) -> None:
            super().__init__()
            inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self.head_dim = head_dim
            self.max_len = max_len

        def _build(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
            t = torch.arange(seq_len, device=device, dtype=dtype)
            freqs = torch.outer(t, self.inv_freq.to(device=device, dtype=dtype))
            emb = torch.repeat_interleave(freqs, 2, dim=-1)
            cos = emb.cos()[None, None, :, :]
            sin = emb.sin()[None, None, :, :]
            return cos, sin

        @staticmethod
        def _rotate_half(x: Tensor) -> Tensor:
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        def apply(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
            cos, sin = self._build(q.size(2), q.device, q.dtype)
            q_rot = (q * cos) + (self._rotate_half(q) * sin)
            k_rot = (k * cos) + (self._rotate_half(k) * sin)
            return q_rot, k_rot

    class SwiGLU(nn.Module):
        def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
            super().__init__()
            self.w_gate = nn.Linear(d_model, d_ff, bias=False)
            self.w_up = nn.Linear(d_model, d_ff, bias=False)
            self.w_down = nn.Linear(d_ff, d_model, bias=False)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: Tensor) -> Tensor:
            return self.dropout(self.w_down(torch.nn.functional.silu(self.w_gate(x)) * self.w_up(x)))

    class ModernDecoderBlock(nn.Module):
        def __init__(self, config: TransformerConfig, rope: RotaryEmbedding) -> None:
            super().__init__()
            self.rope = rope
            self.attn_norm = RMSNorm(config.d_model)
            self.ffn_norm = RMSNorm(config.d_model)
            self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
            self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
            self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
            self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)
            self.ffn = SwiGLU(config.d_model, config.resolved_dim_feedforward(), config.dropout)
            self.n_heads = config.n_heads
            self.head_dim = config.d_model // config.n_heads
            self.dropout = nn.Dropout(config.dropout)

        def forward(self, x: Tensor, attn_mask: Tensor, key_padding_mask: Tensor | None) -> Tensor:
            b, t, _ = x.shape
            h = self.n_heads
            hd = self.head_dim

            residual = x
            x = self.attn_norm(x)
            q = self.q_proj(x).view(b, t, h, hd).transpose(1, 2)
            k = self.k_proj(x).view(b, t, h, hd).transpose(1, 2)
            v = self.v_proj(x).view(b, t, h, hd).transpose(1, 2)
            q, k = self.rope.apply(q, k)
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(hd)
            scores = scores.masked_fill(attn_mask, float("-inf"))
            if key_padding_mask is not None:
                scores = scores.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
            attn = torch.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            y = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, t, h * hd)
            x = residual + self.o_proj(y)

            residual = x
            x = residual + self.ffn(self.ffn_norm(x))
            return x

    class ModernProcessTransformerNet(nn.Module):
        """Decoder-only LM with RoPE, RMSNorm, SwiGLU, pre-norm blocks."""

        def __init__(self, vocab_size: int, config: TransformerConfig, pad_id: int) -> None:
            super().__init__()
            if config.d_model % config.n_heads != 0:
                raise ValueError("d_model must be divisible by n_heads")
            self.config = config
            self.pad_id = pad_id
            self.token_embedding = nn.Embedding(vocab_size, config.d_model, padding_idx=pad_id)
            self.rope = RotaryEmbedding(config.d_model // config.n_heads, config.max_len)
            self.layers = nn.ModuleList(
                [ModernDecoderBlock(config, self.rope) for _ in range(config.n_layers)]
            )
            self.final_norm = RMSNorm(config.d_model)
            self.lm_head = nn.Linear(config.d_model, vocab_size, bias=False)
            self.lm_head.weight = self.token_embedding.weight

        def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
            batch_size, seq_len = input_ids.shape
            if seq_len > self.config.max_len:
                raise ValueError(
                    f"input length {seq_len} exceeds max_len={self.config.max_len}"
                )
            x = self.token_embedding(input_ids)
            causal = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_ids.device),
                diagonal=1,
            )
            key_padding = None
            if attention_mask is not None:
                key_padding = attention_mask == 0
            for layer in self.layers:
                x = layer(x, causal, key_padding)
            x = self.final_norm(x)
            return self.lm_head(x)

else:  # pragma: no cover

    class ModernProcessTransformerNet:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            require_torch()


def build_process_net(
    arch: str,
    vocab_size: int,
    config: TransformerConfig,
    pad_id: int,
) -> nn.Module:
    """Factory: vanilla (2017 stack) or modern (RoPE/RMSNorm/SwiGLU)."""
    require_torch()
    if arch == "modern":
        return ModernProcessTransformerNet(vocab_size, config, pad_id)
    if arch in ("vanilla", "legacy", ""):
        from src.ml.transformer_model import ProcessTransformerNet
        return ProcessTransformerNet(vocab_size, config, pad_id)
    raise ValueError(f"Unknown arch {arch!r}; expected 'vanilla' or 'modern'")
