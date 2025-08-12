import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple
from einops import rearrange, einsum
from dataclasses import dataclass

# Helper functions
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def l2norm(t, dim=-1):
    return F.normalize(t, dim=dim)

# BitLinear module
class BitLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, num_groups=1):
        super().__init__(in_features, out_features, bias)
        self.num_groups = num_groups
        
    def forward(self, x):
        weight = self.weight
        return F.linear(x, weight, self.bias)

# RMSNorm
class RMSNorm(nn.Module):
    def __init__(self, dim, affine=True):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim)) if affine else 1.0

    def forward(self, x):
        return l2norm(x) * self.gamma * self.scale

# BitMGQA
class BitMGQA(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        query_heads: int = 8,
        kv_heads: int = 4,
        dropout: float = 0.1,
        bias: bool = True,
        layer_norm: bool = True,
        layer_norm_eps: float = 1e-5,
        gamma_init: float = 1.0,
        linear_groups: int = 1,
    ):
        super().__init__()
        self.embed_dim = embed_dim  # 存储主特征维度[11](@ref)
        self.query_heads = query_heads
        self.kv_heads = kv_heads
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.gamma_init = gamma_init

        if self.query_heads % self.kv_heads != 0:
            raise ValueError(f"query_heads ({query_heads}) must be divisible by kv_heads ({kv_heads})")
        elif (embed_dim % self.query_heads != 0) or (embed_dim % self.kv_heads != 0):
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by query_heads ({query_heads}) and kv_heads ({kv_heads})")

        head_dim = embed_dim // query_heads
        if not head_dim % 8 == 0:
            raise ValueError(f"head_dim (embed_dim / num_heads = {head_dim}) must be divisible by 8")
        if not head_dim <= 128:
            raise ValueError(f"head_dim (embed_dim / num_heads = {head_dim}) must be <= 128")

        #  投影层 Projection layers using BitLinear
        self.q_proj = BitLinear(embed_dim, embed_dim, bias=bias)
        kv_embed_dim = embed_dim // query_heads * kv_heads
        self.k_proj = BitLinear(embed_dim, kv_embed_dim, bias=bias)
        self.v_proj = BitLinear(embed_dim, kv_embed_dim, bias=bias)
        
        self.norm = nn.LayerNorm(embed_dim, eps=layer_norm_eps) if layer_norm else None
        self.out_proj = BitLinear(embed_dim, embed_dim, bias=bias)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight)
        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0)
        nn.init.xavier_normal_(self.k_proj.weight)
        if self.k_proj.bias is not None:
            nn.init.constant_(self.k_proj.bias, 0)
        nn.init.xavier_normal_(self.v_proj.weight, gain=self.gamma_init)
        if self.v_proj.bias is not None:
            nn.init.constant_(self.v_proj.bias, 0)
        nn.init.xavier_normal_(self.out_proj.weight, gain=self.gamma_init)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, need_weights: bool = False, 
                is_causal: bool = False, average_attn_weights: bool = False):
        # 简化的注意力实现
        B, T, C = query.size()
        
        # 查询投影
        q = self.q_proj(query).view(B, T, self.query_heads, C // self.query_heads).transpose(1, 2)
        # 键值投影
        k = self.k_proj(key).view(B, T, self.kv_heads, C // self.query_heads).transpose(1, 2)
        v = self.v_proj(value).view(B, T, self.kv_heads, C // self.query_heads).transpose(1, 2)
        
        #防止kv头数不同，扩展头数
        if self.kv_heads != self.query_heads:
            expand_ratio = self.query_heads // self.kv_heads
            k = k.repeat_interleave(expand_ratio, dim=1)
            v = v.repeat_interleave(expand_ratio, dim=1)
        
        # 注意力计算
        att = (q @ k.transpose(-2, -1)) * (1.0 / k.size(-1)**0.5)
        
        if is_causal:
            mask = torch.tril(torch.ones(T, T, device=query.device)).view(1, 1, T, T)
            att = att.masked_fill(mask == 0, float('-inf'))
        
        att = F.softmax(att, dim=-1)
        att = F.dropout(att, p=self.dropout, training=self.training)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, -1)

        # 形状验证
        if y.size(-1) != self.embed_dim:
            # 自动调整特征维度
            y = y[:, :, :self.embed_dim]  # 裁剪或使用投影
        
        if self.norm:
            y = self.norm(y)
        
        y = self.out_proj(y)
        return y, att if need_weights else None

# BitFeedForward
class BitFeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        glu: bool = False,
        glu_mult_bias: bool = False,
        swish: bool = False,
        post_act_ln: bool = False,
        dropout: float = 0.0,
        no_bias: bool = False,
        zero_init_output: bool = False,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        activation = nn.SiLU() if swish else nn.GELU()

        if glu:
            project_in = nn.Sequential(
                BitLinear(dim, inner_dim * 2, bias=not no_bias),
                nn.GLU(dim=-1)
            )
        else:
            project_in = nn.Sequential(
                BitLinear(dim, inner_dim, bias=not no_bias),
                activation
            )
        
        ff_layers = [project_in]
        
        if post_act_ln:
            ff_layers.append(nn.LayerNorm(inner_dim))
        
        ff_layers.append(nn.Dropout(dropout))
        ff_layers.append(BitLinear(inner_dim, dim_out, bias=not no_bias))
        
        self.ff = nn.Sequential(*ff_layers)
        
        if zero_init_output:
            nn.init.zeros_(self.ff[-1].weight)
            if self.ff[-1].bias is not None:
                nn.init.zeros_(self.ff[-1].bias)

    def forward(self, x):
        return self.ff(x)

# 定义 BitNet 配置类
@dataclass
class BitNetConfig:
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    block_size: int = 1024
    bias: bool = True
    vocab_size: int = 50304
    dropout: float = 0.1

# Transformer Block
class BitBlock(nn.Module):
    def __init__(self, config: BitNetConfig):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = BitMGQA(
            embed_dim=config.n_embd,
            query_heads=config.n_head,
            kv_heads=config.n_head // 2 if config.n_head > 1 else 1,
            dropout=config.dropout,
            bias=config.bias,
        )
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = BitFeedForward(
            dim=config.n_embd,
            mult=4,
            swish=True,
            post_act_ln=True,
            dropout=config.dropout,
            no_bias=not config.bias,
        )

    def forward(self, x):
        # 残差连接
        attn_out, _ = self.attn(self.ln_1(x), self.ln_1(x), self.ln_1(x))
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x

# BitNetTransformer
class BitNetTransformer(nn.Module):
    def __init__(self, config: BitNetConfig):
        super().__init__()
        self.config = config
        
        # 词嵌入层
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        # 位置嵌入层
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        
        # Dropout
        self.drop = nn.Dropout(config.dropout)
        
        # Transformer 块
        self.h = nn.ModuleList([
            BitBlock(config) for _ in range(config.n_layer)
        ])
        
        # 最终层归一化
        self.ln_f = RMSNorm(config.n_embd)
        
        # 语言模型头
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # 权重绑定
        self.lm_head.weight = self.wte.weight
        
        # 初始化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, BitLinear)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        
        # 位置编码
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        
        # 嵌入层
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb)
        
        # Transformer 块
        for block in self.h:
            x = block(x)
        
        x = self.ln_f(x)
        
        # 语言模型头
        logits = self.lm_head(x)
        
        # 计算损失
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss

    def crop_block_size(self, block_size):
        # 裁剪块大小
        self.config.block_size = block_size
        self.wpe.weight = nn.Parameter(self.wpe.weight[:block_size])
        
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        # 生成文本
        for _ in range(max_new_tokens):
            # 如果序列太长则裁剪
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            # 获取预测
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Top-k 过滤
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # 采样
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # 添加到序列
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx