import torch
import torch.nn as nn
import torch.nn.functional as F

class UserEmbeddingUpdater(nn.Module):
    def __init__(self, embedding_dim=384):
        super(UserEmbeddingUpdater, self).__init__()
        
        # 모든 사용자가 동일하게 시작하는 initial embedding (학습 가능)
        self.initial_embedding = nn.Parameter(torch.randn(embedding_dim))
        
        # 게이팅 메커니즘을 위한 선형 레이어
        self.gating_layer = nn.Linear(embedding_dim * 2, embedding_dim)

    def aggregate_attention(self, attention_output, interaction_mask):
        """
        Attention 결과를 interaction 수를 고려한 평균으로 aggregate
        
        Parameters:
        - attention_output: [batch_size, max_session, max_interaction, embedding_dim]
        - interaction_mask: [batch_size, max_session, max_interaction]
        
        Returns:
        - session_emb: [batch_size, max_session, embedding_dim]
        """
        # 마스크를 차원 확장하여 attention_output과 동일하게 맞추기
        mask_expanded = interaction_mask.unsqueeze(-1)  # [batch_size, max_session, max_interaction, 1]
        
        # 패딩된 interaction 무시하고 합산
        masked_attention = attention_output * mask_expanded  # 패딩 부분은 0이 됨
        
        # 실제 interaction 개수 계산 (padding 제외)
        valid_counts = interaction_mask.sum(dim=2).unsqueeze(-1).clamp(min=1)  # 0 나누기 방지
        
        # interaction 수에 따라 평균내어 session embedding 생성
        session_emb = masked_attention.sum(dim=2) / valid_counts  # [batch_size, max_session, embedding_dim]

        return session_emb

    def forward(self, attention_output, interaction_mask, session_mask, prev_user_embedding=None):
        """
        이전의 user embedding을 받아 session 결과를 바탕으로 gating하여 업데이트
        
        Parameters:
        - attention_output: [batch_size, max_session, max_interaction, embedding_dim]
        - interaction_mask: [batch_size, max_session, max_interaction] (interaction 존재 여부)
        - session_mask: [batch_size, max_session] (세션 존재 여부)
        - prev_user_embedding: [batch_size, embedding_dim] or None
        
        Returns:
        - user_embedding: [batch_size, embedding_dim] (업데이트된 사용자 embedding)
        """
        batch_size, max_session, _, embedding_dim = attention_output.shape

        # Aggregate session-level embedding (interaction 수 고려한 가중 평균)
        session_emb = self.aggregate_attention(attention_output, interaction_mask)  # [batch_size, max_session, embedding_dim]

        # 초기 user embedding을 모든 사용자에 대해 동일하게 설정하거나 이전 embedding 사용
        if prev_user_embedding is None:
            user_emb = self.initial_embedding.unsqueeze(0).expand(batch_size, -1)  # [batch_size, embedding_dim]
        else:
            user_emb = prev_user_embedding  # [batch_size, embedding_dim]

        # 세션을 순차적으로 돌며 user embedding 업데이트
        for sess_idx in range(max_session):
            current_session_emb = session_emb[:, sess_idx, :]  # [batch_size, embedding_dim]
            sess_exist = session_mask[:, sess_idx].unsqueeze(-1)  # [batch_size, 1]

            combined = torch.cat([user_emb, current_session_emb], dim=-1)  # [batch_size, 2*embedding_dim]
            gate = torch.sigmoid(self.gating_layer(combined))  # [batch_size, embedding_dim]

            # gating을 통한 user embedding 업데이트
            updated_emb = gate * user_emb + (1 - gate) * current_session_emb

            # 세션이 없는 경우 (padding된 세션) 이전 embedding 유지
            user_emb = updated_emb * sess_exist + user_emb * (1 - sess_exist)

        return user_emb
