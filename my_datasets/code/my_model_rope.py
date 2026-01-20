# my_model_rope.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Rotary embedding helper
# -------------------------
class RotaryEmbedding:
    """
    Simple RoPE implementation. Caches sin/cos for given max_seq_len/head_dim.
    Works for applying to tensors shaped (B, T, H, head_dim).
    """
    def __init__(self, head_dim, base=10000, max_seq_len=2048):
        self.head_dim = head_dim
        self.base = base
        self.max_seq_len_cached = None
        self.register_cache(max_seq_len)

    def register_cache(self, max_seq_len):
        if self.max_seq_len_cached == max_seq_len:
            return
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)  # [T, head_dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [T, head_dim]
        cos = torch.cos(emb)
        sin = torch.sin(emb)
        # store as float32 (converted later on device)
        self._cos = cos
        self._sin = sin
        self.max_seq_len_cached = max_seq_len

    def get_cos_sin(self, seq_len, device, dtype):
        if seq_len > self.max_seq_len_cached:
            # extend cache
            self.register_cache(seq_len * 2)
        return self._cos[:seq_len].to(device=device, dtype=dtype), self._sin[:seq_len].to(device=device, dtype=dtype)

def apply_rope(x, cos, sin):
    # x: (B, T, H, head_dim)
    # split even/odd
    B, T, H, D = x.shape
    # reshape last dim into [D/2, 2] to perform rotation
    x1 = x[..., ::2]  # (B,T,H,D/2)
    x2 = x[..., 1::2]
    x_rot = torch.cat([x1 * cos.unsqueeze(1).unsqueeze(2) - x2 * sin.unsqueeze(1).unsqueeze(2),
                       x1 * sin.unsqueeze(1).unsqueeze(2) + x2 * cos.unsqueeze(1).unsqueeze(2)], dim=-1)
    return x_rot

# -------------------------
# Model components
# -------------------------
class FeedForward(nn.Module):
    def __init__(self, d_model, mlp_ratio, dropout):
        super().__init__()
        hidden = d_model * mlp_ratio
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout)
        )
    def forward(self, x): return self.net(x)

class Block(nn.Module):
    def __init__(self, d_model, n_heads, mlp_ratio, dropout, max_seq_len):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, mlp_ratio, dropout)
        self.dropout = nn.Dropout(dropout)

        # RoPE helper per block (can be shared across blocks too)
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len)

    def forward(self, x):
        B, T, D = x.shape
        h = self.ln1(x)  # (B, T, D)
        qkv = self.qkv(h)  # (B, T, 3*D)
        qkv = qkv.view(B, T, 3, self.n_heads, self.head_dim)  # (B, T, 3, H, head_dim)
        q = qkv[:, :, 0]  # (B, T, H, head_dim)
        k = qkv[:, :, 1]
        v = qkv[:, :, 2]

        # prepare RoPE
        cos, sin = self.rope.get_cos_sin(T, device=x.device, dtype=x.dtype)  # cos: (T, head_dim)
        # cos/sin shape align: expand to (T, head_dim) -> then inside apply_rope we align dims
        # rearrange cos/sin to (T, head_dim) used in apply_rope signature via broadcasting
        # apply_rope accepts cos,sin shaped (T, head_dim) and expects broadcast over head dim axis
        # implement apply_rope that uses cos/sin with broadcasting:
        # But our apply_rope expects cos,sin shaped (T, head_dim) and will be broadcasted.

        # apply RoPE to q,k
        # cos/sin need to be shaped (T, head_dim); apply_rope expects cos and sin of shape (T, head_dim)
        # Create cos/sin with shape (T, head_dim)
        cos_t = cos  # (T, head_dim)
        sin_t = sin
        # Broadcast cos/sin inside function
        # reshape q/k: (B,T,H,head_dim)
        q = apply_rope(q, cos_t, sin_t)
        k = apply_rope(k, cos_t, sin_t)

        # scaled dot-product attention (causal)
        # compute q @ k^T per head
        # reshape to (B, H, T, head_dim) for matmul
        q_ = q.permute(0,2,1,3)  # (B,H,T,hd)
        k_ = k.permute(0,2,3,1)  # (B,H,hd,T)
        attn_scores = torch.matmul(q_, k_) / (self.head_dim ** 0.5)  # (B,H,T,T)

        # causal mask: upper triangular (T,T) True where masked
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        attn_scores = attn_scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn = torch.softmax(attn_scores, dim=-1)
        attn = self.dropout(attn)
        v_ = v.permute(0,2,1,3)  # (B,H,T,hd)
        out = torch.matmul(attn, v_)  # (B,H,T,hd)
        out = out.permute(0,2,1,3).contiguous().view(B, T, D)  # (B,T,D)

        out = self.out_proj(out)
        x = x + out
        # feed-forward
        x = x + self.ff(self.ln2(x))
        return x

class TinyGPT_RoPE(nn.Module):
    def __init__(self, vocab_size, d_model=384, n_layers=8, n_heads=24, mlp_ratio=4, max_len=4096, dropout=0.1, pad_token_id=0):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        # No absolute positional embedding
        self.blocks = nn.ModuleList([Block(d_model, n_heads, mlp_ratio, dropout, max_seq_len=max_len) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok.weight
        self.max_len = max_len
        self.pad_token_id = pad_token_id

    def forward(self, input_ids, labels=None):
        B, T = input_ids.shape
        if T > self.max_len:
            input_ids = input_ids[:, -self.max_len:]
            T = input_ids.shape[1]
        x = self.tok(input_ids)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.size(-1)),
                                   labels[:, 1:T].reshape(-1),
                                   ignore_index=self.pad_token_id)
        return logits, loss
