import torch
import torch.nn as nn

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
