import torch
import torch.nn as nn
from collections import deque
import torch.nn.functional as F
from model.dit_components import RMSNorm
import matplotlib.pyplot as plt
import os

# TODO: REVIST THE PAGIN LOGIC
class PagedJointAttention(nn.Module):
    def __init__(self, heads: int, embedding_size: int, dropout: float = 0.1, page_size: int = 512, max_pages: int = 16,
                 add_q_context: bool = None, add_kv_proj: bool = False):
        """
        Creates a Paged KV Cache Attention Mechanisim
        """
        super(PagedJointAttention, self).__init__()

        self.heads = heads
        self.embedding_size = embedding_size
        self.add_q_context = add_q_context
        self.add_kv_proj = add_kv_proj
        self.page_size = page_size
        self.max_pages = max_pages
        self.head_dim = embedding_size // heads
        self.max_tokens = max_pages * page_size

        # normalization
        self.norm_q = RMSNorm(self.head_dim)
        self.norm_k = RMSNorm(self.head_dim)

        if add_kv_proj:
            self.norm_added_q = RMSNorm(self.head_dim)
            self.norm_added_k = RMSNorm(self.head_dim)

        self.to_out = nn.ModuleList([
            nn.Linear(embedding_size, embedding_size),
            nn.Dropout(dropout)
        ])

        # avoid error
        self.batch_first = False

        assert embedding_size % heads == 0, "embedding size must be divisible by heads"

        self.to_q = nn.Linear(embedding_size, embedding_size)
        self.to_k = nn.Linear(embedding_size, embedding_size)
        self.to_v = nn.Linear(embedding_size, embedding_size)

        # for joint attention
        if add_kv_proj:

            if add_q_context is not None:
                self.add_q_proj = nn.Linear(embedding_size, embedding_size)

            self.add_k_proj = nn.Linear(embedding_size, embedding_size)
            self.add_v_proj = nn.Linear(embedding_size, embedding_size)

        if add_q_context is not None and not add_q_context:
            self.to_add_out = nn.Linear(embedding_size, embedding_size)
        else: self.to_add_out = None

        self.k_cache = deque(maxlen = self.max_pages)
        self.v_cache = deque(maxlen = self.max_pages)

    def forward(self, x: torch.Tensor, use_cache: bool = True, encoder_hidden_state: torch.Tensor = None): 
        # reference attention equation
        # https://pbs.twimg.com/profile_images/1624054272676532224/UNv4ONME_400x400.jpg

        residual = x

        batch_size = x.size(0)

        Q = self.to_q(x)
        new_K = self.to_k(x)
        new_V = self.to_v(x)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2)
        new_K = new_K.view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2)
        new_V = new_V.view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2)

        # store in cache and get the full cache
        if use_cache:
            self.store_in_cache(new_K, new_V)

            # get the previous K and V for complete context
            K, V = self.get_cached_kv()
        else:
            K, V = new_K, new_V 

        # normalize query and key
        Q = self.norm_q(Q)
        K = self.norm_k(K)

        # the Joint Attention part
        if encoder_hidden_state is not None:
            
            # forward pass with reshaping 
            encoder_query = self.add_q_proj(encoder_hidden_state)\
                .view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2) # (B, seq_len, heads, heads_dim)
            
            encoder_key = self.add_k_proj(encoder_hidden_state)\
                .view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2) # //
            
            encoder_value = self.add_v_proj(encoder_hidden_state)\
                .view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2) # //
            
            # normalize
            if self.add_kv_proj:
                encoder_query = self.norm_added_q(encoder_query)
                encoder_key = self.norm_added_k(encoder_key)

            # concat the attentions
            Q = torch.concat([Q, encoder_query], dim = 2)
            K = torch.concat([K, encoder_key], dim = 2)
            V = torch.concat([V, encoder_value], dim = 2)

        hidden_states = F.scaled_dot_product_attention(Q, K, V, dropout_p = 0.0, is_causal = False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.embedding_size).to(Q.dtype)

        # split hidden_states and encoder hidden states
        if encoder_hidden_state is not None:
            hidden_states, encoder_hidden_states = (
                hidden_states[:, :residual.size(1)],
                hidden_states[:, residual.size(1):]
            )

            if not self.add_q_context:
                encoder_hidden_states = self.to_add_out(encoder_hidden_states)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)
        # dropout
        hidden_states = self.to_out[1](hidden_states)

        if encoder_hidden_state is None: return hidden_states

        else: return hidden_states, encoder_hidden_states

    def store_in_cache(self, K_new: torch.Tensor, V_new: torch.Tensor):
        """
        Splits incoming K/V by pages of size `page_size` and appends each chunk to deques.
        """

        k_chunks = torch.split(K_new, self.page_size, dim = 2)
        v_chunks = torch.split(V_new, self.page_size, dim = 2)

        for k_chunk, v_chunk in zip(k_chunks, v_chunks):
            self.k_cache.append(k_chunk)
            self.v_cache.append(v_chunk)

    def get_cached_kv(self):
        """
        Concatenate all cached pages along sequence dim to form full K and V.
        """
        if not self.k_cache:
            return None, None
        K = torch.cat(list(self.k_cache), dim=2)
        V = torch.cat(list(self.v_cache), dim=2)
        return K, V

    def reset_cache(self):
        """Clear all cached pages."""
        self.k_cache.clear()
        self.v_cache.clear()

    def __del__(self):
        try:
            self.reset_cache()
        except AttributeError:
            pass

    def compute_attention_weights(self, Q, K):
        """
        Q, K: (B, heads, T, head_dim)
        Return: attention weights (B, heads, T, T')
        """
        scale = Q.shape[-1] ** -0.5
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        return attn_weights


def test_atten(embedding_size = 512):
    
    x = torch.rand(77, embedding_size)

    attention = PagedJointAttention(embedding_size = embedding_size, heads = 8)
    output = attention(x.unsqueeze(0))

    print(output)
    del attention

def test_atten_with_visualization(embedding_size=512, save_dir="attention_vis"):
    B, T, H = 1, 16, 8
    attention = PagedJointAttention(embedding_size=embedding_size, heads=H)

    x = torch.rand(B, T, embedding_size)
    q = attention.to_q(x).view(B, T, H, -1).transpose(1, 2)
    k = attention.to_k(x).view(B, T, H, -1).transpose(1, 2)
    v = attention.to_v(x).view(B, T, H, -1).transpose(1, 2)

    # attention weight 계산
    attn_weights = attention.compute_attention_weights(q, k)  # [B, heads, T, T]
    attn_output = torch.matmul(attn_weights, v)               # [B, heads, T, D]
    output = attn_output.transpose(1, 2).reshape(B, T, embedding_size)

    # 출력
    print("Output shape:", output.shape)
    print("Attention weights shape:", attn_weights.shape)

    # 저장 디렉토리 생성
    os.makedirs(save_dir, exist_ok=True)

    # 첫 번째 head의 attention map 저장
    head_idx = 0
    attn_map = attn_weights[0, head_idx].detach().cpu().numpy()

    plt.figure(figsize=(6, 5))
    plt.imshow(attn_map, cmap='viridis')
    plt.title(f"Attention Map - Head {head_idx}")
    plt.xlabel("Key Position")
    plt.ylabel("Query Position")
    plt.colorbar()
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"attention_map_head{head_idx}.png")
    plt.savefig(save_path)
    plt.close()

    print(f"[Saved] attention map saved to {save_path}")

if __name__ == "__main__":
    #test_atten()
    test_atten_with_visualization()