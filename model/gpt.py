from transformer import Transformer,LayerNormalization,GELU,FeedForward,MultiHeadAttention
import torch 
import torch.nn as nn

GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
} 


class GPT_124(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.token_emd=nn.Embedding(cfg["vocab_size"],cfg["emb_dim"])
        self.pos_emb=nn.Embedding(cfg["context_length"],cfg["emb_dim"])
        self.drop_out=nn.Dropout(cfg["drop_rate"])
        self.trf=nn.Sequential(
            *[Transformer(cfg) for _ in range(cfg["n_layers"])]
        )
        self.Layer_norm=LayerNormalization(cfg["emb_dim"])
        self.output_head=nn.Linear(cfg["emb_dim"],cfg["vocab_size"],bias=False)
        
    def forward(self,input_ids,use_cache=False): ## takes tokens as input
        batch_size,seq_length=input_ids.shape
        tok_embd=self.token_emd(input_ids)
        pos_embd=self.pos_emb(torch.arange(seq_length,device=input_ids.device))
        x=tok_embd+pos_embd
        x=self.drop_out(x)
        x=self.trf(x,use_cache)
        x=self.Layer_norm(x)
        out=self.output_head(x)
        return out


        
