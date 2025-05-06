# attention_block.py
import torch
import torch.nn as nn
import torch.nn.functional as F
class EfficientMHA(nn.Module):
    def __init__(self, d_model: int = 128, n_heads: int = 8, dropout: float = 0.2):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)
        self.dp  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: [B,L,D], mask: [B,L,L] (True=keep)
        B, L, D = x.size()
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)
        scores = scores.masked_fill(~mask.unsqueeze(1), -1e4)
        scores = torch.nan_to_num(scores, neginf=-1e4, posinf=1e4)
        attn   = self.dp(F.softmax(scores, dim=-1))
        ctx    = attn @ v
        ctx    = ctx.transpose(1, 2).reshape(B, L, D)
        out    = self.out(ctx)
        valid  = mask.any(-1).float().unsqueeze(-1)
        return out * valid
class MaskedLayerNorm(nn.Module):
    """
    LayerNorm that ignores positions where mask == 0.
    mask: (B, S, I) or (B, L) broadcastable to x.shape[:-1]
    """
    def __init__(self, hidden_dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.bias   = nn.Parameter(torch.zeros(hidden_dim))
        self.eps    = eps

    def forward(self, x, mask):
        # mask → (..., 1) for broadcasting
        while mask.dim() < x.dim():
            mask = mask.unsqueeze(-1)
        masked_x   = x * mask
        denom      = mask.sum(dim=(-2, -3), keepdim=True).clamp_min(1.0)
        mean       = masked_x.sum(dim=(-2, -3), keepdim=True) / denom
        var        = ((masked_x - mean) ** 2 * mask).sum(dim=(-2,-3), keepdim=True) / denom
        normed     = (x - mean) / torch.sqrt(var + self.eps)
        return normed * self.weight + self.bias
    
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embedding_dim=384, num_heads=8, dropout=0.2, max_interactions=None):
        super(MultiHeadSelfAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.max_interactions = max_interactions  # 새 인자: 최대 interaction 수 (예: 100 등)
        
        self.attention_heads = nn.ModuleList([
            nn.ModuleDict({
                'query': nn.Linear(embedding_dim, embedding_dim),
                'key': nn.Linear(embedding_dim, embedding_dim),
                'value': nn.Linear(embedding_dim, embedding_dim)
            }) for _ in range(num_heads)
        ])

        self.dropout = nn.Dropout(dropout)
        self.out_linear = nn.Linear(num_heads * embedding_dim, embedding_dim)

    def forward(self, x, mask):
        # x: [batch_size, max_interaction, embedding_dim]
        batch_size, max_inter, embed_dim = x.shape
        if self.max_interactions is not None and max_inter > self.max_interactions:
            x = x[:, :self.max_interactions, :]
            mask = mask[:, :self.max_interactions]
            max_inter = self.max_interactions

        head_outputs = []
        for head in self.attention_heads:
            Q = head['query'](x)
            K = head['key'](x)
            V = head['value'](x)

            scores = torch.matmul(Q, K.transpose(-2, -1)) / (embed_dim ** 0.5)

            causal_mask = torch.tril(torch.ones((max_inter, max_inter), device=x.device)).bool().unsqueeze(0)
            scores = scores.masked_fill(~causal_mask, -1e4)

            padding_mask = mask.unsqueeze(1)
            scores = scores.masked_fill(padding_mask == 0, -1e4)

            # 만약 해당 row의 모든 값이 -inf라면, softmax 계산 전에 임시로 0으로 변경
            scores = torch.nan_to_num(
                scores,
                nan=0.0,
                neginf=-1e4,
                posinf=+1e4
            )
            attn_weights = torch.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            head_output = torch.matmul(attn_weights, V)
            head_outputs.append(head_output)

        concat_output = torch.cat(head_outputs, dim=-1)
        output = self.out_linear(concat_output)
        output = output.view(batch_size, max_inter, embed_dim)
        output = output * mask.unsqueeze(-1)
        return output
    
def preprocess_inputs(attention_output, time_gaps, user_embedding, valid_mask=None):
    """
    Attention 결과, Time Gap, 사용자 임베딩을 받아서 전처리합니다.

    Parameters:
    - attention_output: Tensor of shape [batch_size, max_session, max_interaction, 384]
    - time_gaps: Tensor of shape [batch_size, max_interaction, 384]
    - user_embedding: Tensor of shape [384]

    Returns:
    - combined_input: Tensor of shape [batch_size, max_session, max_interaction, 384]
    """
    # Time Gap을 세션 내에서 왼쪽으로 시프트
    #shifted_time_gaps = torch.roll(time_gaps, shifts=-1, dims=1)
    #shifted_time_gaps[:, -1, :] = 0  # 마지막 interaction의 TG는 0으로 설정

    # 사용자 임베딩을 Attention 결과와 Time Gap에 더하기
    combined_input = attention_output + time_gaps + user_embedding.view(user_embedding.shape[0], 1, -1)

    if valid_mask is not None:
        combined_input = combined_input * valid_mask.unsqueeze(-1)
    return combined_input

class FFN(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=512, output_dim=384, dropout=0.1):
        super(FFN, self).__init__()
        self.ln = MaskedLayerNorm(input_dim)
        self.fc1  = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2  = nn.Linear(hidden_dim, output_dim)
        self.dp   = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        h = self.ln(x, mask)
        h = self.dp(self.relu(self.fc1(h)))
        h = self.fc2(h)
        return (x + self.dp(h)) * mask.unsqueeze(-1)         # residual

def create_ffn_model(input_dim=384, hidden_dim=512, output_dim=384):
    """
    FFN 모델을 생성합니다.

    Parameters:
    - input_dim: 입력 차원 (기본값: 384)
    - hidden_dim: 은닉층 차원 (기본값: 512)
    - output_dim: 출력 차원 (기본값: 384)

    Returns:
    - model: FFN 모델 객체
    """
    model = FFN(input_dim, hidden_dim, output_dim)
    return model
