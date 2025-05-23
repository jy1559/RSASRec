# attention_block.py
import torch
import torch.nn as nn
import torch.nn.functional as F

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
            scores = scores.masked_fill(~causal_mask, float('-inf'))

            padding_mask = mask.unsqueeze(1)
            scores = scores.masked_fill(padding_mask == 0, float('-inf'))

            # 만약 해당 row의 모든 값이 -inf라면, softmax 계산 전에 임시로 0으로 변경
            all_inf = torch.isinf(scores).all(dim=-1, keepdim=True)
            scores_safe = scores.masked_fill(all_inf, 0.0)
            attn_weights = torch.softmax(scores_safe, dim=-1)
            # 원래 모든 값이 -inf였던 row는 0으로 설정
            attn_weights = torch.where(all_inf, torch.zeros_like(attn_weights), attn_weights)

            attn_weights = self.dropout(attn_weights)
            head_output = torch.matmul(attn_weights, V)
            head_outputs.append(head_output)

        concat_output = torch.cat(head_outputs, dim=-1)
        output = self.out_linear(concat_output)
        output = output.view(batch_size, max_inter, embed_dim)
        output = output * mask.unsqueeze(-1)
        return output