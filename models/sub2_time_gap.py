import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeGapEmbedding(nn.Module):
    def __init__(self, embedding_dim=384, hidden_dim=128):
        super(TimeGapEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # MLP 정의
        self.fc1 = nn.Linear(1, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.embedding_dim)

    def forward(self, time_gaps):
        # time_gaps: [batch_size, max_session, max_interaction]
        # 로그 스케일 적용 (1을 더하여 로그의 정의역을 양수로 유지)
        time_gaps = torch.log1p(time_gaps)

        # MLP를 통해 임베딩
        x = F.relu(self.fc1(time_gaps.unsqueeze(-1)))  # [batch_size, max_session, max_interaction, hidden_dim]
        x = self.fc2(x).squeeze(-1)  # [batch_size, max_session, max_interaction, embedding_dim]

        return x
