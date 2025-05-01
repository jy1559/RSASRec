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
    shifted_time_gaps = torch.roll(time_gaps, shifts=-1, dims=1)
    shifted_time_gaps[:, -1, :] = 0  # 마지막 interaction의 TG는 0으로 설정

    # 사용자 임베딩을 Attention 결과와 Time Gap에 더하기
    combined_input = attention_output + shifted_time_gaps + user_embedding.view(user_embedding.shape[0], 1, -1)

    if valid_mask is not None:
        combined_input = combined_input * valid_mask.unsqueeze(-1)
    return combined_input

class FFN(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=512, output_dim=384):
        super(FFN, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

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
