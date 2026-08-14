import torch
import torch.nn as nn


## GPT configs
GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # qkv  bias
} 

### Layer Normalization
class LayerNormalization(torch.nn.Module):
    def __init__(self,emb_dim):
        super().__init__()
        self.eps=1e-5
        self.scale=torch.nn.Parameter(torch.ones(emb_dim))
        self.shift=torch.nn.Parameter(torch.zeros(emb_dim))
    def forward(self,x):
        self.mean=x.mean(dim=-1,keepdim=True)    
        self.var=x.var(dim=-1,keepdim=True,unbiased=False)
        norm=(x-self.mean)/torch.sqrt(self.var+self.eps)   
        out= norm*self.scale + self.shift 
        return out
    def get_norm_params(self):
        print(f"Means is {self.mean}, \n Variance is {self.var}")

### Generalization of gelu

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))

## Feed forward NN
class FeedForward(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.layer=nn.Sequential(
            nn.Linear(cfg["emb_dim"],4*cfg["emb_dim"]),
            GELU(),
            nn.Linear(4*cfg["emb_dim"],cfg["emb_dim"])
        )
    def forward(self,x):
        out=self.layer(x)
        return out  
    
### Multihead Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, n_heads, qkv_bias=False):
        super().__init__()
        assert d_out % n_heads == 0, "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.n_heads = n_heads
        self.context_length = context_length
        self.head_dim = d_out // n_heads

        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer("mask", torch.triu(
            torch.ones(context_length, context_length), diagonal=1))
        self.out_proj = nn.Linear(d_out, d_out)

        self.register_buffer("k_cache", None, persistent=False)
        self.register_buffer("v_cache", None, persistent=False)

    def forward(self, x, use_cache=False):
        b, num_token, d_in = x.shape

        new_key = self.W_k(x)
        query = self.W_q(x)
        new_value = self.W_v(x)

        new_key = new_key.view(b, num_token, self.n_heads, self.head_dim).transpose(1, 2)
        query = query.view(b, num_token, self.n_heads, self.head_dim).transpose(1, 2)
        new_value = new_value.view(b, num_token, self.n_heads, self.head_dim).transpose(1, 2)

        if use_cache:
            if self.k_cache is None:
                self.k_cache, self.v_cache = new_key, new_value
            else:
                self.k_cache = torch.cat([self.k_cache, new_key], dim=2)
                self.v_cache = torch.cat([self.v_cache, new_value], dim=2)
            new_key, new_value = self.k_cache, self.v_cache

        q_len = query.size(2)
        k_len = new_key.size(2)
        assert k_len <= self.context_length, (
            f"KV cache length {k_len} exceeds context_length "
            f"{self.context_length}; call reset_cache() or truncate."
        )

        attn_scores = query @ new_key.transpose(2, 3)
        offset = k_len - q_len
        mask_bool = self.mask[offset:offset + q_len, :k_len].bool()
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(
            attn_scores / torch.sqrt(torch.tensor(self.head_dim, dtype=attn_scores.dtype)),
            dim=-1
        )
        attn_weights = self.dropout(attn_weights)

        context_vector = (attn_weights @ new_value).transpose(1, 2)
        context_vector = context_vector.contiguous().view(b, num_token, self.d_out)
        context_vector = self.out_proj(context_vector)
        return context_vector

    def reset_cache(self):
        self.k_cache = None
        self.v_cache = None

## Multihead latent attention
class MHLA(nn.Module):
    def __init__(self, d_model, d_cache, context_length, dropout, n_heads, qkv_bias=False):
        super().__init__()
        assert d_model % n_heads ==0, "Wrong n_head size"
        self.d_model=d_model
        self.d_cache=d_cache
        self.n_heads = n_heads
        self.context_length = context_length
        self.head_dim = d_model // n_heads
        self.W_dkv=nn.Linear(d_model,d_cache,bias=qkv_bias)
        self.W_q=nn.Linear(d_model,d_model,bias=qkv_bias)
        self.W_uk=nn.Linear(d_cache,d_model,bias=qkv_bias) 
        self.W_uv=nn.Linear(d_cache,d_model,bias=qkv_bias) 
        self.register_buffer("mask", torch.triu(
            torch.ones(context_length, context_length), diagonal=1))
        self.out_proj = nn.Linear(d_model,d_model)
        self.register_buffer("cache_kv",None)
        self.dropout=nn.Dropout(p=dropout)
        
    def forward(self,x,use_cache=False):
        b,n_tokens,dim=x.shape
        q=self.W_q(x) # [b,n_tokens,d_model]
        C_kv=self.W_dkv(x)  # [B, n_token, d_cache]
        past_len= 0 if self.cache_kv is None else self.cache_kv.shape[1]
        if use_cache:
            if self.cache_kv is None:
                self.cache_kv=C_kv
            else:
                self.cache_kv=torch.cat([self.cache_kv,C_kv],dim=1)
                C_kv=self.cache_kv

        kv_len=C_kv.shape[1]
        k=   self.W_uk (C_kv)
        v=    self.W_uv(C_kv)  
        k=k.view(b,-1,self.n_heads,self.head_dim).transpose(1,2)
        v=v.view(b,-1,self.n_heads,self.head_dim).transpose(1,2) ## -1 means figure out cureent dim as it varies depending on cache_kv size
        q=q.view(b,n_tokens,self.n_heads,self.head_dim).transpose(1,2)  

        attention_score= q @ k.transpose(2,3)
        mask_bool=self.mask.bool()[past_len:past_len + n_tokens, :kv_len]
        
        attention_score=attention_score.masked_fill(mask_bool,-torch.inf)

        attention_weights=torch.softmax(attention_score / torch.sqrt(torch.tensor(self.head_dim)),dim=-1)
        attention_weights = self.dropout(attention_weights)

        context_vector= (attention_weights @ v).transpose(1,2)
        context_vector = context_vector.contiguous().view(b, n_tokens, self.d_model)
        context_vector=self.out_proj(context_vector)
        return context_vector
    def reset_cache(self):
        self.cache_kv=None
            
        
             




class Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            dropout=cfg["drop_rate"],
            n_heads=cfg["n_heads"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNormalization(cfg["emb_dim"])
        self.norm2 = LayerNormalization(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x, use_cache=False):
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x, use_cache)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x

    def reset_cache(self):
        self.attn.reset_cache()
    

if __name__ == "__main__":
    x = torch.rand(2, 4, 768)
    block = Transformer(GPT_CONFIG_124M)
    output = block(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
