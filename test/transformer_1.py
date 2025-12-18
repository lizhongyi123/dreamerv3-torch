import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # [T, D]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [T,1]
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe)  # [T,D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,D]
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)  # [1,T,D] broadcast


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_model = d_model
        self.nhead = nhead
        self.d_head = d_model // nhead

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x: [B,T,D]
        attn_mask: [T,T] bool, True表示要mask（常用于causal）
        key_padding_mask: [B,T] bool, True表示padding位置要mask
        """
        B, T, D = x.shape

        qkv = self.qkv(x)  # [B,T,3D]
        q, k, v = qkv.chunk(3, dim=-1)

        # [B,T,D] -> [B,H,T,d_head]
        q = q.view(B, T, self.nhead, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.nhead, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.nhead, self.d_head).transpose(1, 2)

        # attention scores: [B,H,T,T]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)

        # causal / custom mask
        if attn_mask is not None:
            # attn_mask True => mask out
            scores = scores.masked_fill(attn_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        # padding mask
        if key_padding_mask is not None:
            # key_padding_mask True => mask out keys (最后一维T是key位置)
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn = F.softmax(scores, dim=-1)  # [B,H,T,T]
        attn = self.dropout(attn)

        out = attn @ v  # [B,H,T,d_head]
        out = out.transpose(1, 2).contiguous().view(B, T, D)  # [B,T,D]
        return self.out(out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, dim_ff: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, dim_ff)
        self.fc2 = nn.Linear(dim_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, nhead, dropout)
        self.drop1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, dim_ff, dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask=None, key_padding_mask=None) -> torch.Tensor:
        # Pre-LN 结构：更稳定
        h = self.attn(self.ln1(x), attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = x + self.drop1(h)

        h = self.ffn(self.ln2(x))
        x = x + self.drop2(h)
        return x


def causal_mask(T: int, device) -> torch.Tensor:
    # True 表示要mask掉上三角（未来）
    return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)


class TinyTransformerLM(nn.Module):
    """
    Encoder-only 的最小语言模型示意：
    tokens -> embedding+pos -> N层EncoderBlock -> lm_head logits
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_ff: int = 256,
        dropout: float = 0.0,
        pad_id: int = 0,
        max_len: int = 2048,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model

        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = PositionalEncoding(d_model, max_len=max_len)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, nhead, dim_ff, dropout)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, ids: torch.Tensor, use_causal: bool = True) -> torch.Tensor:
        """
        ids: [B,T]
        return logits: [B,T,V]
        """
        B, T = ids.shape
        device = ids.device

        x = self.emb(ids) * math.sqrt(self.d_model)  # [B,T,D]
        x = self.pos(x)
        x = self.drop(x)

        attn_mask = causal_mask(T, device) if use_causal else None
        key_padding_mask = (ids == self.pad_id)  # [B,T]

        for blk in self.blocks:
            x = blk(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # [B,T,V]
        return logits


if __name__ == "__main__":
    torch.manual_seed(0)

    vocab = 5000
    pad_id = 0
    model = TinyTransformerLM(vocab_size=vocab, pad_id=pad_id, num_layers=2)

    B, T = 2, 8
    ids = torch.randint(1, vocab, (B, T))
    ids[0, -2:] = pad_id  # 造点padding看看mask是否可用

    logits = model(ids, use_causal=True)
    print("logits:", logits.shape)  # [B,T,V]
