import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def sinusoidal_encoding(x, d_model, max_timescale=1000000000):
    """
    x: tensor of shape (..., 1), representing a scalar input (e.g., timestamp, log-scaled time gap)
    d_model: desired encoding dimension (should be even)
    Returns: tensor of shape (..., d_model) with sinusoidal encoding
    """

    device = x.device
    half_dim = d_model // 2
    exponent = torch.arange(0, half_dim, dtype=torch.float32, device=device) * (-math.log(max_timescale) / half_dim)
    exponent = exponent.unsqueeze(0)  # shape: (1, half_dim)
    if torch.isnan(exponent).any():
        print(f"[WARNING] NaN detected in exponent.")
    scaled = x * torch.exp(exponent)    # shape: (N, half_dim) by broadcasting

    if torch.isnan(x).any():
        print(f"[WARNING] NaN detected in x.")
    nan_mask = torch.isnan(scaled)
    if nan_mask.any():
        nan_indices = torch.nonzero(nan_mask)
        print(f"NaN 발견: 총 {nan_indices.size(0)}개")
        # 3개만 출력
        for idx in nan_indices[:3]:
            # idx: 예를 들어 (i, j) -> i: x의 인덱스, j: exponent의 인덱스
            a, b, c, j = idx.tolist()
            print(f"인덱스 (i={a},{b},{c}, j={j}): x = {x[a, b, c].item()}, exponent = {exponent[0, j].item()}, scaled = {scaled[a,b,c, j].item()}")
    else:
        pass#print("NaN 없음")
    sin_enc = torch.sin(scaled)           # (N, half_dim)
    cos_enc = torch.cos(scaled)           # (N, half_dim)
    if torch.isnan(sin_enc).any():
        print(f"[WARNING] NaN detected in sin_enc.")
    out = torch.cat([sin_enc, cos_enc], dim=-1)  # (N, d_model)
    # 명시적으로 첫번째 차원은 그대로 두고 d_model 차원으로 reshape (예: (1000, 512))
    out = out.view(*x.shape[:-1], d_model)
    return out

class TimeGapEmbedding(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=128, embedding_type='hybrid', sinusoidal_dim=512, sinusoidal_max=10000):
        """
        out_dim: 최종 출력 차원 (예: 128)
        hidden_dim: FFN 내부 은닉 차원 (ffn branch에 사용)
        embedding_type: 'ffn', 'sinusoidal', 또는 'hybrid' 중 선택
        sinusoidal_dim: Sinusoidal 인코딩 차원 (time gap의 경우 기본 512)
        """
        super(TimeGapEmbedding, self).__init__()
        self.embedding_type = embedding_type
        self.sinusoidal_dim = sinusoidal_dim
        self.sinusoidal_max = sinusoidal_max
        self.out_dim = embedding_dim

        if embedding_type in 'ffn':
            # FFN branch: 입력은 스칼라 (1), 중간 차원 hidden_dim, 최종 sinusoidal_dim으로 매핑한 후 out_dim 투사
            self.fc1 = nn.Linear(1, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, sinusoidal_dim)
            self.fc3 = nn.Linear(sinusoidal_dim, embedding_dim)
        if embedding_type == 'hybrid':
            # hybrid: FFN + Sinusoidal
            self.fc1 = nn.Linear(sinusoidal_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, embedding_dim)
        if embedding_type == 'sinusoidal':
            # 만약 최종 차원이 sinusoidal_dim과 다르다면 간단한 선형 투사를 후처리로 추가할 수 있음
            if embedding_dim != sinusoidal_dim:
                self.proj = nn.Linear(sinusoidal_dim, embedding_dim)
            else:
                self.proj = None

    def forward(self, time_gaps):
        """
        time_gaps: [batch_size, max_session, max_interaction] (원본 값, 예: 밀리초 차이 등)
        """
        # 로그 스케일 적용 (정의역을 양수로)
        x = time_gaps#torch.log1p(time_gaps)  # shape: [B, S, I]
        x_exp = x.unsqueeze(-1)     # shape: [B, S, I, 1]

        if self.embedding_type == 'sinusoidal':
            # 단순 Sinusoidal 인코딩
            sin_enc = sinusoidal_encoding(x_exp, self.sinusoidal_dim, self.sinusoidal_max)  # [B, S, I, sinusoidal_dim]
            if self.proj is not None:
                sin_enc = self.proj(sin_enc)  # [B, S, I, out_dim]
            return sin_enc

        elif self.embedding_type == 'ffn':
            # FFN만 사용하는 경우
            out = F.relu(self.fc1(x_exp))
            out = F.relu(self.fc2(out))
            out = self.fc3(out)  # [B, S, I, out_dim]
            return out

        elif self.embedding_type == 'hybrid':
            # hybrid: Sinusoidal + FFN, 예를 들어 평균을 취함.
            sin_enc = sinusoidal_encoding(x_exp, self.sinusoidal_dim, self.sinusoidal_max)  # [B, S, I, sinusoidal_dim]
            if torch.isnan(sin_enc).any():
                print(f"[WARNING] NaN detected in sin_enc.")
            ffn_out = F.relu(self.fc1(sin_enc))
            ffn_out = self.fc2(ffn_out)  # [B, S, I, out_dim]
            return ffn_out
        else:
            raise ValueError("embedding_type must be 'ffn', 'sinusoidal', or 'hybrid'.")
