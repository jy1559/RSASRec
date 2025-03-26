# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .sub1_sequence_embedding import sentence_embedder, get_embeddings, AddInfoEncoder, TimestampEncoder, ProjectionFFN
from .sub2_time_gap import TimeGapEmbedding
from .sub3_attention import MultiHeadSelfAttention
from .sub4_user_embedding import UserEmbeddingUpdater
from .sub5_FFN import preprocess_inputs, create_ffn_model

from time import time
from tqdm.auto import tqdm

# --- LoRA 관련 코드 ---
class LoRALinear(nn.Module):
    """
    nn.Linear에 LoRA adapter를 적용하는 모듈.
    주어진 linear layer의 weight에 대해 low-rank 업데이트를 수행합니다.
    """
    def __init__(self, orig_linear, r=4, alpha=32):
        super(LoRALinear, self).__init__()
        self.orig_linear = orig_linear
        self.r = r
        self.alpha = alpha
        # A: [r, in_features], B: [out_features, r]
        self.lora_A = nn.Parameter(torch.randn(r, orig_linear.in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.randn(orig_linear.out_features, r) * 0.01)
        self.scaling = self.alpha / self.r
        self.lora_enabled = True

    def forward(self, x):
        # 원래 linear 연산
        out = self.orig_linear(x)
        if self.lora_enabled:
            # x: [*, in_features]
            # x * A^T -> [*, r], then [*, r] * B^T -> [*, out_features]
            lora_update = torch.matmul(x, self.lora_A.t())
            lora_update = torch.matmul(lora_update, self.lora_B.t())
            out = out + self.scaling * lora_update
        return out
    
def apply_lora(module, r=4, alpha=32):
    """
    재귀적으로 module 내부의 모든 nn.Linear를 LoRALinear로 교체합니다.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, r, alpha))
        else:
            apply_lora(child, r, alpha)

def pad_features(tensor_list, pad_value=0.0):
    """
    tensor_list: list of [L_i, D] tensors.
    Returns: [B, L_max, D] tensor padded with pad_value.
    """
    B = len(tensor_list)
    D = tensor_list[0].shape[1]
    L_max = max(t.shape[0] for t in tensor_list)
    out = torch.full((B, L_max, D), pad_value, device=tensor_list[0].device)
    for i, t in enumerate(tensor_list):
        L = t.shape[0]
        out[i, :L, :] = t
    return out

def select_target_features(ffn_out, interaction_mask, strategy):
    """
    ffn_out: [B, I, D] tensor from FFN for a single session.
    interaction_mask: [B, I] tensor.
    strategy: 
        - 'EachSession_LastInter': return last valid interaction feature, shape [B, D]
        - 'Global_LastInter': return last valid interaction feature, shape [B, D]
        - 'LastSession_AllInter': return all valid interactions (padded), shape [B, L_max, D]
    """
    B, I, D = ffn_out.shape
    if strategy in ['EachSession_LastInter', 'Global_LastInter']:
        valid_mask = (interaction_mask != 0)  # [B, I]
        # reverse along interaction dimension
        rev_mask = valid_mask.flip(dims=[1])
        # 각 배치별로 뒤에서부터 첫번째 valid index
        rev_first_valid_idx = torch.argmax(rev_mask.to(torch.int32), dim=1)  # [B]
        last_valid_idx = I - 1 - rev_first_valid_idx  # [B]
        # gather: 각 배치의 마지막 valid interaction 선택
        idx = last_valid_idx.unsqueeze(1).unsqueeze(2).expand(B, 1, D)
        target_features = torch.gather(ffn_out, dim=1, index=idx).squeeze(1)  # [B, D]
        return target_features
    elif strategy == 'LastSession_AllInter':
        # 각 배치별로 모든 valid interaction을 모아서 pad 처리
        feature_list = []
        for b in range(B):
            valid_features = ffn_out[b][interaction_mask[b].bool()]  # [L_b, D]
            feature_list.append(valid_features)
        target_features = pad_features(feature_list, pad_value=0.0)  # [B, L_max, D]
        return target_features
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


class SeqRecModel(nn.Module):
    def __init__(self, 
                 embed_dim=128,
                 strategy='EachSession_LastInter',
                 update={'llm': False, 'tg':True, 'attention':True, 'ffn':True, 'user_emb':True, 'init_emb':True},
                 use_llm = False,
                 lora = {'use': False, 'r':4, 'alpha':32},
                 ffn_hidden_dim=256,
                 time_gap_hidden_dim=32,
                 num_attention_heads=8,
                 num_add_info = 0,
                 dropout=0.2,
                 item_embedding_dict = None,
                 hf_model_path='sentence-transformers/all-MiniLM-L6-v2',
                 device='cpu'):
        super(SeqRecModel, self).__init__()
        
        self.strategy = strategy
        
        self.update_llm = update['llm']
        self.update_time_gap = update['tg']
        self.update_attention = update['attention']
        self.update_ffn = update['ffn']
        self.item_embedding_dict = item_embedding_dict
        self.use_llm = use_llm
        self.update_user_embedding = update['user_emb']
        self.update_initial_embedding = update['init_emb']
        self.use_lora = lora['use']
        self.lora_r = lora['r']
        self.lora_alpha = lora['alpha']

        if self.use_llm: 
            self.tokenizer, self.sentence_model = sentence_embedder(hf_model_path)
            self.projection_ffn = None
        else: 
            self.add_info_ffn = nn.ModuleList([AddInfoEncoder(384) for _ in range(num_add_info)])
            self.projection_ffn = ProjectionFFN(384, ffn_hidden_dim, embed_dim, device)
            self.timestamp_encoder = TimestampEncoder(384)

        if self.use_lora:
            from models.model import apply_lora  # assume apply_lora is defined in model.py
            apply_lora(self.sentence_model, r=self.lora_r, alpha=self.lora_alpha)
        
        self.time_gap_embed = TimeGapEmbedding(embedding_dim=embed_dim, hidden_dim=time_gap_hidden_dim)
        self.attention = MultiHeadSelfAttention(embedding_dim=embed_dim, num_heads=num_attention_heads, dropout=dropout)
        self.ffn = create_ffn_model(input_dim=embed_dim, hidden_dim=ffn_hidden_dim, output_dim=embed_dim)
        self.user_emb_updater = UserEmbeddingUpdater(embedding_dim=embed_dim)
        
    def forward(self, batch, prev_user_embedding=None):
        """
        batch: {
          'embedding_sentences': [B, S, I] 텍스트 문자열 텐서,
          'delta_ts': [B, S, I] 텐서,
          'interaction_mask': [B, S, I] 텐서,
          'session_mask': [B, S] 텐서
        }
        prev_user_embedding: [B, embed_dim] (없으면 updater 내부의 initial embedding 사용)
        
        Returns:
          - output_features: fixed shape tensor depending on strategy:
              * 'EachSession_LastInter': [B, S, embed_dim]
              * 'Global_LastInter': [B, embed_dim]
              * 'LastSession_AllInter': [B, L_max, embed_dim]
          - updated_user_embedding: [B, embed_dim]
        """
        #sentences = batch['embedding_sentences']
        interaction_mask = batch['interaction_mask']
        delta_ts = batch['delta_ts']
        session_mask = batch['session_mask']
        B, S, I = interaction_mask.shape
        
        st = time()
        #Sentence Embedding 구하기
        if self.use_llm:
            if self.update_llm:
                seq_emb = get_embeddings(batch, self.use_llm, tokenizer=self.tokenizer, llm_model=self.sentence_model, item_embeddings_dict=self.item_embedding_dict)
            else:
                with torch.no_grad():
                    seq_emb = get_embeddings(batch, self.use_llm, tokenizer=self.tokenizer, llm_model=self.sentence_model, item_embeddings_dict=self.item_embedding_dict)
        else:
            seq_emb = get_embeddings(batch, self.use_llm, item_embeddings_dict=self.item_embedding_dict, projection_ffn=self.projection_ffn, add_info_encoder=self.add_info_ffn, timestamp_encoder=self.timestamp_encoder, valid_mask=interaction_mask)
        
        #print(f"Sentence embedding 생성 완료. 소요 시간: {time() - st:.2f}")
        st = time()


        #Time gap embedding 얻기
        time_gap_emb = self.time_gap_embed(delta_ts)
        if not self.update_time_gap:
            time_gap_emb = time_gap_emb.detach()
        
        # Input Embedding 얻기
        combined_emb = seq_emb + time_gap_emb
        #print(f"Time gap embedding 생성 완료. 소요 시간: {time() - st:.2f}")

        #User embedding 구하기. prev_user_embedding은 웬만하면 None임
        if prev_user_embedding is not None:     user_emb_current = prev_user_embedding
        else:
            if self.update_initial_embedding:   user_emb_current = self.user_emb_updater.initial_embedding.unsqueeze(0).expand(B, -1)           #Initial embedding 업데이트 O
            else:                               user_emb_current = self.user_emb_updater.initial_embedding.detach().unsqueeze(0).expand(B, -1)  #Initial embedding 업데이트 X

        
        if self.strategy in ['EachSession_LastInter', 'Global_LastInter']:
            session_target_features = []  # 각 session의 결과 저장 ([B, D] per session)
            for s in range(S):
                # 현재 session의 embedding: [B, I, D] / mask: [B, I]
                emb_s = combined_emb[:, s, :, :]
                mask_s = interaction_mask[:, s, :]

                attention_out_s = self.attention(emb_s, mask_s)  # [B, I, D]
                ffn_input_s = preprocess_inputs(attention_out_s, 
                                                time_gap_emb[:, s, :, :], 
                                                user_emb_current.view(B, 1, -1))
                ffn_out_s = self.ffn(ffn_input_s)  # [B, I, D]

                # 단일 session에 대해 target feature 선택 (select_target_features는 [B, I, D] 입력으로 작동)
                target_feature_s = select_target_features(ffn_out_s, mask_s, self.strategy)  # [B, D]
                session_target_features.append(target_feature_s.unsqueeze(1))  # [B, 1, D]

                # 각 session별 user embedding 업데이트
                user_emb_current = self.user_emb_updater(
                    attention_out_s.unsqueeze(1),  # [B, 1, I, D]
                    mask_s.unsqueeze(1),           # [B, 1, I]
                    session_mask[:, s].unsqueeze(-1),  # [B, 1]
                    prev_user_embedding=user_emb_current
                )
            if self.strategy == 'EachSession_LastInter':
                # 모든 session의 결과를 [B, S, D]로 결합
                output_features = torch.cat(session_target_features, dim=1)
            else:  # Global_LastInter: 마지막 session의 결과만 사용
                output_features = session_target_features[-1].squeeze(1)  # [B, D]
            return output_features, user_emb_current

        elif self.strategy == 'LastSession_AllInter':
            # 모든 session의 FFN 결과는 저장하지 않고, 마지막 valid session만 선택
            # 각 배치별로 마지막 valid session index를 구합니다.
            last_session_indices = []
            for b in range(B):
                valid_sessions = (session_mask[b] != 0)  # [S]
                if valid_sessions.sum() == 0:
                    last_session_indices.append(0)
                else:
                    # 뒤집어서 첫 번째 True의 인덱스를 이용
                    rev_valid = valid_sessions.flip(0)
                    last_idx = S - 1 - torch.argmax(rev_valid.to(torch.int32)).item()
                    last_session_indices.append(last_idx)

            # 각 배치별로 해당 session의 FFN 결과 계산
            output_features_list = []
            for b in range(B):
                s_idx = last_session_indices[b]
                emb_s = combined_emb[b, s_idx, :, :].unsqueeze(0)  # [1, I, D]
                mask_s = interaction_mask[b, s_idx, :].unsqueeze(0)  # [1, I]
                attention_out_s = self.attention(emb_s, mask_s)  # [1, I, D]
                ffn_input_s = preprocess_inputs(attention_out_s, 
                                                time_gap_emb[b, s_idx, :, :].unsqueeze(0), 
                                                user_emb_current[b].view(1, 1, -1))
                ffn_out_s = self.ffn(ffn_input_s)  # [1, I, D]
                # select_target_features는 단일 session에 대해 [1, L_max, D] 반환 (padding 처리)
                target_feature_b = select_target_features(ffn_out_s, mask_s, self.strategy)  # [1, L_max, D]
                output_features_list.append(target_feature_b)
            # [B, L_max, D]
            output_features = torch.cat(output_features_list, dim=0)
            return output_features, user_emb_current

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")