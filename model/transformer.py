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
    def __init__(self,d_in,d_out,context_length,dropout,n_heads,qkv_bias=False):
        super().__init__()
        assert (d_out % n_heads == 0), \
            "d_out must be divisible by num_heads"
        self.d_out=d_out
        self.n_heads=n_heads
        self.context_length=context_length
        self.head_dim = d_out // n_heads
        self.W_q=nn.Linear(d_in,d_out,bias=qkv_bias)
        self.W_k=nn.Linear(d_in,d_out,bias=qkv_bias)
        self.W_v=nn.Linear(d_in,d_out,bias=qkv_bias)
        self.dropout=nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length),
                       diagonal=1))
        self.out_proj = nn.Linear(d_out, d_out)  
        self.register_buffer("k_cache",None)
        self.register_buffer("v_cache",None)


    def forward(self,x,use_cache=False):  
        b,num_token,d_in=x.shape 
        new_new_key=self.W_k(x)  
        query=self.W_q(x)
        new_value=self.W_v(x)

        new_new_key=new_key.view(b,num_token,self.n_heads,self.head_dim)
        query=query.view(b,num_token,self.n_heads,self.head_dim)
        new_value=new_value.view(b,num_token,self.n_heads,self.head_dim)

        new_key = new_key.transpose(1, 2)
        query = query.transpose(1, 2)
        new_value = new_value.transpose(1, 2)

        if self.k_cache is None:
            self.k_cache,self.v_cache= new_key,new_value
        else:
            self.k_cache= torch.cat([self.k_cache,new_key],dim=2)
            self.v_cache= torch.cat([self.v_cache,new_value],dim=2)   

        new_key,new_value=self.k_cache,self.v_cache
        q_len= new_key.size(2)
        k_len=new_value.size(2)

        assert q_len <= self.context_length (
            f"KV cache length {k_len} exceeds context_length"
            f"{self.context_length}; call reset_cache() or truncate."
        )

        attn_scores = query @ new_key.transpose(2, 3)
        offset=  k_len - q_len
        mask_bool = self.mask[offset:offset + q_len, :k_len].bool()
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        attn_weights = torch.softmax(attn_scores / torch.sqrt(torch.tensor(self.head_dim)), dim=-1)
        
        attn_weights = self.dropout(attn_weights)
        context_vector= (attn_weights @ new_value).transpose(1, 2) 
        context_vector = context_vector.contiguous().view(b, num_token, self.d_out)
        context_vector=self.out_proj(context_vector)
        return context_vector
    
    ### Single block of transformer
class Transformer(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.attn=MultiHeadAttention(
            d_in=GPT_CONFIG_124M["emb_dim"],
            d_out=GPT_CONFIG_124M["emb_dim"],
            context_length=GPT_CONFIG_124M["context_length"],
            dropout=GPT_CONFIG_124M["drop_rate"],
            n_heads=GPT_CONFIG_124M["n_heads"],
            qkv_bias=GPT_CONFIG_124M["qkv_bias"]
        )
        self.ff=FeedForward(cfg)
        self.norm1 = LayerNormalization(cfg["emb_dim"])
        self.norm2 = LayerNormalization(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])
        
    def forward(self,x):

        shortcut = x
        x=self.norm1(x)
        x=self.attn(x)
        x=self.drop_shortcut(x)
        x=x+shortcut
         ## 2nd shortcut connection
        shortcut=x

        x=self.norm2(x)
        x=self.ff(x)
        x=self.drop_shortcut(x)
        x=x+shortcut
        return x
  

if __name__ == "__main__":
    x = torch.rand(2, 4, 768)
    block = Transformer(GPT_CONFIG_124M)
    output = block(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
