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
import time
import wandb

def check_nan(tensor, name='Tensor'):
    # tensor 내 NaN의 개수 구하기
    nan_count = torch.isnan(tensor).sum().item()
    total_count = tensor.numel()
    nan_ratio = 100.0 * nan_count / total_count
    if nan_count > 0:
        print(f"{name}: Total elements: {total_count}, NaN elements: {nan_count} ({nan_ratio:.2f}%)")
    return nan_count > 0

class Timer:
    def __init__(self, name, wandb_logging):
        self.name = name
        self.logging = wandb_logging
    def __enter__(self):
        self.start = time.time()
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start
        if self.logging:
            wandb.log({f"Timing/{self.name}": elapsed})
        
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

def select_target_features_flat(ffn_out, valid_mask, strategy):
    """
    Args:
      ffn_out: [B, S, I, d] tensor (각 토큰의 FFN 출력)
      valid_mask: [B, S, I] Boolean tensor (True: valid, False: invalid)
      strategy: 선택 전략 문자열. 아래 7가지 중 하나:
          - 'EachSession_LastInter'
          - 'Global_LastInter'
          - 'LastSession_AllInter'
          - 'AllInter_ExceptFirst'
          - 'AllInter'
          - 'EachSession_First_and_Last_Inter'
          - 'EachSession_Except_First'
                
    Returns:
      selected_target_features: [B, L_max, d] tensor, flatten한 후 선택된 feature들 (padding된 부분은 0)
      loss_mask: [B, L_max] Boolean tensor, 각 위치가 유효하면 True, padding이면 False
      session_ids: [B, L_max] tensor, 각 선택 토큰이 속한 세션 id (padding: -1)
    """
    B, S, I, d = ffn_out.shape
    # 결과를 저장할 리스트들 (각 배치 sample별)
    all_selected = []
    all_loss_mask = []
    all_session_ids = []
    
    for b in range(B):
        selected = []      # 선택된 feature (tensor shape: [*, d])
        valid_mask_list = []  # 각 선택된 feature에 대해 True (나중에 모두 True, padding은 False)
        sess_ids_list = []    # 각 선택 토큰의 session id
        
        if strategy in ['Global_LastInter', 'AllInter', 'AllInter_ExceptFirst']:
            # 전체 flatten 처리: ignore 세션 구분
            flat_feat = ffn_out[b].reshape(-1, d)          # [S*I, d]
            flat_valid = valid_mask[b].reshape(-1)           # [S*I]
            valid_idxs = (flat_valid == True).nonzero(as_tuple=False).squeeze(-1)
            if valid_idxs.dim() == 0:
                valid_idxs = valid_idxs.unsqueeze(0)
            if strategy == 'Global_LastInter':
                # 전체에서 마지막 valid token만 선택
                if valid_idxs.numel() > 0:
                    valid_idxs = valid_idxs[-1:].clone()  # keep as 1 element
                else:
                    valid_idxs = valid_idxs  # empty
            elif strategy == 'AllInter_ExceptFirst':
                # flatten 순서대로 첫 번째 valid token 제외
                if valid_idxs.numel() > 0:
                    valid_idxs = valid_idxs[1:]
            elif strategy == 'AllInter':
                # 모든 valid token 선택 → 그대로 valid_idxs
                pass
            # 선택된 인덱스에 해당하는 feature와 session id 계산
            for idx in valid_idxs:
                selected.append(flat_feat[idx])
                valid_mask_list.append(True)
                # 세션 id: flatten index // I (정수 나눗셈)
                sess_ids_list.append(int(idx // I))
                
        else:
            # 세션별 처리: 각 세션마다 따로 valid token을 선택
            # (각 세션의 처리 결과들을 순서대로 concatenate함)
            for s in range(S):
                session_feat = ffn_out[b, s, :, :]       # [I, d]
                session_valid = valid_mask[b, s, :]        # [I]
                valid_idxs = (session_valid == True).nonzero(as_tuple=False).squeeze(-1)
                if valid_idxs.dim() == 0:
                    valid_idxs = valid_idxs.unsqueeze(0)
                if strategy == 'EachSession_LastInter':
                    if valid_idxs.numel() > 0:
                        # 마지막 valid token만 선택
                        sel_idx = valid_idxs[-1]
                        selected.append(session_feat[sel_idx])
                        valid_mask_list.append(True)
                        sess_ids_list.append(s)
                elif strategy == 'EachSession_First_and_Last_Inter':
                    if valid_idxs.numel() > 0:
                        # 첫 번째와 마지막 토큰 선택
                        first_idx = valid_idxs[0]
                        selected.append(session_feat[first_idx])
                        valid_mask_list.append(True)
                        sess_ids_list.append(s)
                        if valid_idxs.numel() > 1:
                            last_idx = valid_idxs[-1]
                            selected.append(session_feat[last_idx])
                            valid_mask_list.append(True)
                            sess_ids_list.append(s)
                elif strategy == 'EachSession_Except_First':
                    if valid_idxs.numel() > 0:
                        # 첫 번째 토큰 제외한 나머지 선택 (모든 valid token이 하나만 있으면 그 토큰을 선택)
                        if valid_idxs.numel() > 1:
                            for idx in valid_idxs[1:]:
                                selected.append(session_feat[idx])
                                valid_mask_list.append(True)
                                sess_ids_list.append(s)
                        else:
                            selected.append(session_feat[valid_idxs[0]])
                            valid_mask_list.append(True)
                            sess_ids_list.append(s)
                elif strategy == 'LastSession_AllInter':
                    # 이 전략은 전체 세션 중 마지막 valid session에서만 선택
                    pass  # 아래에서 처리한 후 loop를 빠져나가서 한 번만 선택하도록 함
                else:
                    # 기본: 각 세션에서 마지막 valid token 선택 (default)
                    if valid_idxs.numel() > 0:
                        sel_idx = valid_idxs[-1]
                        selected.append(session_feat[sel_idx])
                        valid_mask_list.append(True)
                        sess_ids_list.append(s)
            # 만약 strategy가 'LastSession_AllInter'이면,
            if strategy == 'LastSession_AllInter':
                # 첫째, 각 세션이 valid했는지 판단하여 마지막 valid session index 구하기
                valid_sessions = []
                for s in range(S):
                    if valid_mask[b, s, :].any():
                        valid_sessions.append(s)
                if valid_sessions:
                    last_s = valid_sessions[-1]
                    session_feat = ffn_out[b, last_s, :, :]
                    session_valid = valid_mask[b, last_s, :]
                    valid_idxs = (session_valid == True).nonzero(as_tuple=False).squeeze(-1)
                    if valid_idxs.dim() == 0:
                        valid_idxs = valid_idxs.unsqueeze(0)
                    # 선택: 모든 valid token in that session
                    selected = []  # overwrite previous selection for this sample
                    valid_mask_list = []
                    sess_ids_list = []
                    for idx in valid_idxs:
                        selected.append(session_feat[idx])
                        valid_mask_list.append(True)
                        sess_ids_list.append(last_s)
                else:
                    selected = []
                    valid_mask_list = []
                    sess_ids_list = []
        
        # selected: list of tensors with shape [d] for sample b
        if len(selected) == 0:
            selected_tensor = torch.empty(0, d, device=ffn_out.device)
        else:
            selected_tensor = torch.stack(selected, dim=0)  # [L_b, d]
        all_selected.append(selected_tensor)
        # loss mask: boolean vector of length L_b (all True for selected tokens)
        all_loss_mask.append(torch.tensor(valid_mask_list, dtype=torch.bool, device=ffn_out.device))
        # session ids: tensor of shape [L_b]
        all_session_ids.append(torch.tensor(sess_ids_list, dtype=torch.long, device=ffn_out.device))
    
    # 배치 내 최대 길이(L_max)
    L_max = max(selected.shape[0] for selected in all_selected) if all_selected else 0
    
    padded_features = []
    padded_loss_mask = []
    padded_session_ids = []
    for feat, mask, sess_ids in zip(all_selected, all_loss_mask, all_session_ids):
        L = feat.shape[0]
        if L < L_max:
            pad_feat = torch.zeros(L_max - L, d, device=ffn_out.device)
            feat = torch.cat([feat, pad_feat], dim=0)
            pad_mask = torch.zeros(L_max - L, dtype=torch.bool, device=ffn_out.device)
            mask = torch.cat([mask, pad_mask], dim=0)
            pad_sess = torch.full((L_max - L,), -1, dtype=torch.long, device=ffn_out.device)
            sess_ids = torch.cat([sess_ids, pad_sess], dim=0)
        padded_features.append(feat)
        padded_loss_mask.append(mask)
        padded_session_ids.append(sess_ids)
    
    selected_target_features = torch.stack(padded_features, dim=0)  # [B, L_max, d]
    
    return selected_target_features


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
                 item_embedding_tensor = None,
                 hf_model_path='sentence-transformers/all-MiniLM-L6-v2',
                 device='cpu'):
        super(SeqRecModel, self).__init__()
        
        self.strategy = strategy
        
        self.update_llm = update['llm']
        self.update_time_gap = update['tg']
        self.update_attention = update['attention']
        self.update_ffn = update['ffn']
        self.item_embedding_tensor = item_embedding_tensor
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

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        if self.use_lora:
            from models.model import apply_lora  # assume apply_lora is defined in model.py
            apply_lora(self.sentence_model, r=self.lora_r, alpha=self.lora_alpha)
        
        self.time_gap_embed = TimeGapEmbedding(embedding_dim=embed_dim, hidden_dim=time_gap_hidden_dim, embedding_type='hybrid', sinusoidal_dim=512)
        self.attention = MultiHeadSelfAttention(embedding_dim=embed_dim, num_heads=num_attention_heads, dropout=dropout)
        self.ffn = create_ffn_model(input_dim=embed_dim, hidden_dim=ffn_hidden_dim, output_dim=embed_dim)
        self.user_emb_updater = UserEmbeddingUpdater(embedding_dim=embed_dim)

    def forward(self, batch, prev_user_embedding=None, chunk=False):
        """
        batch: {
        'delta_ts': [B, S, I] tensor,
        'item_id': [B, S, I] tensor (padding: -1),
        'interaction_mask': [B, S, I] tensor (1 for valid, 0 for pad),
        'session_mask': [B, S] tensor (1 for valid, 0 for pad)
        }
        prev_user_embedding: [B, d] (없으면 updater의 initial embedding 사용)
        chunk: bool, True이면 chunking을 사용하여 배치 처리 
            만약 True면 strategy에 따른 슬라이싱 하지 않고 raw output 그대로 출력력(default: False)

        이 함수는 각 세션별 입력에 대해 learnable [CLS] 토큰을 prepend하여,
        예를 들어 세션 [a, b, c, d, -1] → [[CLS], a, b, c, d, -1] 로 구성합니다.
        네트워크는 이 시퀀스를 처리하여 [B, S, I+1, d]의 출력을 내고,
        예측은 output[:, :, :-1, :] (즉, 인덱스 0 ~ I-1)을 사용하여
        각 세션의 타겟인 [a, b, c, d, -1] ([B, S, I])와 비교합니다.

        
        Returns:
        predictions: [B, S, I, d] tensor – 각 세션의 예측 결과
        targets: [B, S, I] tensor – 원래의 item_id (타겟)
        updated_user_embedding: [B, d]
        """
        # --- Embedding 계산 (get_embeddings는 그대로 사용) ---
        if self.use_llm:
            if self.update.get('llm', False):
                seq_emb = get_embeddings(batch, self.use_llm, tokenizer=self.tokenizer, llm_model=self.sentence_model, item_embeddings_tensor=self.item_embedding_tensor)
            else:
                with torch.no_grad():
                    seq_emb = get_embeddings(batch, self.use_llm, tokenizer=self.tokenizer, llm_model=self.sentence_model, item_embeddings_tensor=self.item_embedding_tensor)
        else:
            seq_emb = get_embeddings(batch, self.use_llm,
                                item_embeddings_tensor=self.item_embedding_tensor,
                                projection_ffn=self.projection_ffn,
                                add_info_encoder=self.add_info_ffn,
                                timestamp_encoder=self.timestamp_encoder,
                                valid_mask=batch['interaction_mask'])
            
        
        time_gap_emb = self.time_gap_embed(batch['delta_ts'])  # [B, S, I, d]
        if not self.update_time_gap:
            time_gap_emb = time_gap_emb.detach()
        combined_emb = seq_emb + time_gap_emb  # [B, S, I, d]
        
        # --- User Embedding 초기화 ---
        B, S, I, d = combined_emb.shape
        if prev_user_embedding is not None:
            user_emb_current = prev_user_embedding
        else:
            if self.update_initial_embedding:
                user_emb_current = self.user_emb_updater.initial_embedding.unsqueeze(0).expand(B, -1)
            else:
                user_emb_current = self.user_emb_updater.initial_embedding.detach().unsqueeze(0).expand(B, -1)
        
        # --- [CLS] 토큰 추가 ---
        # [CLS] 토큰 역할의 learnable embedding (self.cls_emb: [d])
        # cls_token: [1, 1, d] → expand to [B, S, 1, d]
        cls_tokens = self.cls_token.expand(B, S, 1, -1)
        # 새 session input: [B, S, I+1, d]
        session_input = torch.cat([cls_tokens, combined_emb], dim=2)
        # 새 interaction mask: [B, S, I+1] (CLS 토큰은 항상 valid → 1)
        cls_mask = torch.ones(B, S, 1, device=batch['interaction_mask'].device)
        session_mask_extended = torch.cat([cls_mask, batch['interaction_mask']], dim=2)
        
         # 5. 각 세션별 처리: 기존에는 loop 내에서 처리했으나,
        #    user embedding 업데이트은 그대로 loop으로 진행하되, FFN 출력만 따로 저장하여 나중에 flatten 처리
        ffn_out_list = []
        for s in range(S):
            sess_input = session_input[:, s, :, :]   # [B, I+1, d]
            sess_mask = session_mask_extended[:, s, :] # [B, I+1]
            attn_out = self.attention(sess_input, sess_mask)  # [B, I+1, d]
            ffn_input = attn_out[:, 1:, :]  # 실제 예측 대상, [B, I, d]
            # 시간 gap 처리: [CLS] 자리에는 0 벡터
            zero_time = torch.zeros(B, 1, attn_out.size(-1), device=attn_out.device)
            sess_time_gap = torch.cat([zero_time, time_gap_emb[:, s, :, :]], dim=1)  # [B, I+1, d]
            ffn_time_gap = sess_time_gap[:, 1:, :]  # [B, I, d]
            # preprocess_inputs 함수를 이용하여 FFN 입력 준비 (user embedding 더함)
            # 여기서는 user embedding 업데이트도 진행 (원래 코드처럼)
            ffn_in = preprocess_inputs(ffn_input, ffn_time_gap, user_emb_current.unsqueeze(1), valid_mask=sess_mask[:, 1:])
            ffn_out = self.ffn(ffn_in)  # [B, I, d]
            ffn_out_list.append(ffn_out.unsqueeze(1))  # [B, 1, I, d]
            # Update user embedding per session (기존 코드 그대로)
            prev_user_embedding = self.user_emb_updater(
                attn_out.unsqueeze(1),           # [B, 1, I+1, d]
                sess_mask.unsqueeze(1),            # [B, 1, I+1]
                batch['session_mask'][:, s].unsqueeze(-1),  # [B, 1]
                prev_user_embedding=user_emb_current
            )
        # Stack FFN 출력: [B, S, I, d]
        ffn_out_all = torch.cat(ffn_out_list, dim=1)
        # valid mask for FFN output는 원래의 interaction_mask: [B, S, I]
        valid_mask_all = batch['interaction_mask']
        
        if chunk:
            return ffn_out_all, valid_mask_all, prev_user_embedding
        else:
            # 6. Flatten하여 target feature 선택 (get_batch_item_ids와 유사한 방식)
            selected_target_features = select_target_features_flat(ffn_out_all, valid_mask_all, self.strategy)
            # selected_target_features: [B, L, d], loss_mask: [B, L], session_ids: [B, L]
            
            return selected_target_features, prev_user_embedding