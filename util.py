# util.py
import json
import pickle
import random
import torch
from tqdm.auto import tqdm
import torch.nn.functional as F
import numpy as np
from models.Part1_Embedding import sentence_embedder, mean_pooling
import time
import wandb

class Timer:
    def __init__(self, name, wandb_logging, batch_counter):
        self.name = name
        self.logging = wandb_logging
        self.batch_counter = batch_counter
    def __enter__(self):
        self.start = time.time()
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start
        if self.logging:
            wandb.log({f"Timing/{self.name}": elapsed}, step=self.batch_counter)

def compute_and_save_item_embeddings(metadata_path, output_path, hf_model_path='sentence-transformers/all-MiniLM-L6-v2', batch_size=128, device='cuda:0'):
    """
    item_metadata.json 파일을 읽어서, 각 아이템의 정보를 LLM을 통해 embedding으로 변환한 후,
    결과를 pickle 파일로 저장합니다.
    
    Parameters:
    - metadata_path: str, item_metadata.json 파일 경로.
    - output_path: str, 결과 embedding을 저장할 pickle 파일 경로.
    - hf_model_path: str, HuggingFace 문장 임베딩 모델 경로.
    - batch_size: int, 배치 처리 크기.
    - device: str, 모델 실행 디바이스 ('cpu' 또는 'cuda')
    
    Returns:
    - embeddings: dict, {item_id: embedding (numpy array)} 형태.
    """
    # 아이템 메타데이터 로드
    with open(metadata_path, 'r', encoding='utf-8') as f:
        item_metadata = json.load(f)
    
    # 문장 임베딩 모델 초기화
    tokenizer, model = sentence_embedder(hf_model_path)
    model.to(device)
    model.eval()
    
    item_ids = list(item_metadata.keys())
    # 각 아이템의 텍스트 정보를 문자열로 변환 (필요시 전처리 가능)
    sentences = [str(item_metadata[item_id]) for item_id in item_ids]
    embeddings = {}
    for i in tqdm(range(0, len(sentences), batch_size), total=len(sentences) // batch_size):
        batch_sentences = sentences[i:i+batch_size]
        encoded_input = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors='pt').to(device)
        with torch.no_grad():
            model_output = model(**encoded_input)
        # Mean pooling을 이용해 문장 embedding 계산
        batch_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
        batch_embeddings = batch_embeddings.cpu().numpy()
        for j, item_id in enumerate(item_ids[i:i+batch_size]):
            embeddings[item_id] = batch_embeddings[j]
    
    # embedding 결과를 pickle 파일로 저장
    with open(output_path, 'wb') as f:
        pickle.dump(embeddings, f)
    
    return embeddings


def load_item_embeddings(embedding_file_path, device):
    """
    전처리된 _revised item_embedding_normalized 파일을 로드하여, 
    [N+1, embedding_dim] tensor를 반환합니다.
    (index 0은 padding용 0 벡터, 나머지는 연속된 new index 순서대로 저장됨)
    
    Args:
      file_path: str, 전처리된 pickle 파일 경로 (예: item_embedding_normalized_revised.pickle)
      device: torch.device
      
    Returns:
      item_embeddings_tensor: [N+1, embedding_dim] tensor, device로 이동됨.
    """
    with open(embedding_file_path, 'rb') as f:
        data = pickle.load(f)
    # data는 {'embedding_tensor': tensor, 'id2index': mapping} 형태이나,
    # 여기서는 id2index가 필요 없으므로 embedding_tensor만 반환합니다.
    embedding_tensor = data['embedding_tensor']
    return embedding_tensor.to(device)

def load_candidate_tensor(candidate_file, candidate_size, device):
    """
    전처리된 _revised candidate_sets 파일을 로드하여,
    [N+1, candidate_size-1] tensor를 반환합니다.
    
    Args:
      file_path: str, 전처리된 npz 파일 경로 (예: candidate_sets_revised.npz)
      device: torch.device
      
    Returns:
      candidate_tensor: [N+1, candidate_size-1] long tensor, device로 이동됨.
    """
    data = np.load(candidate_file)
    # 저장된 candidate_tensor가 "candidate_tensor" 키에 저장됨.
    candidate_tensor_np = data['candidate_tensor']
    # candidate_size 열만 slice합니다.
    candidate_tensor_np = candidate_tensor_np[:, :candidate_size]
    candidate_tensor = torch.tensor(candidate_tensor_np, dtype=torch.long, device=device)
    return candidate_tensor

def build_candidate_for_item(item_id, projection_ffn, candidate_tensor, item_embeddings_tensor, candidate_size, device):
    """
    주어진 item_id (이미 new index로 변환된 값)에 대해 후보 임베딩들을 반환합니다.
    이전 dict 기반 방식 대신, 전처리된 candidate_tensor (shape: [N+1, candidate_size])와
    item_embeddings_tensor (shape: [N+1, base_dim])를 사용합니다.
    
    처리 로직:
      - 만약 item_id가 -1 (패딩)인 경우, candidate 크기(candidate_size)만큼 0 벡터를 반환.
      - 그렇지 않은 경우, positive 후보는 item_id 자신이며,
        negative 후보는 candidate_tensor[item_id]의 처음 candidate_size-1 값을 사용합니다.
      - 따라서 최종 후보 리스트는 torch.cat([ [item_id], candidate_tensor[item_id][:candidate_size-1] ])
        형태가 되어 총 candidate_size개가 됩니다.
      - 이 candidate id 리스트로부터 item_embeddings_tensor에서 임베딩을 lookup하고,
        단층 projection (projection_ffn)을 적용하여 최종 후보 임베딩 ([candidate_size, d])를 리턴합니다.
    
    Returns:
      candidate_embeddings_tensor: [candidate_size, d] tensor (projection_ffn 적용 후)
      pos_index: int, positive candidate의 index (항상 0)
    """
    if item_id == -1:
        d = projection_ffn(item_embeddings_tensor[0:1]).shape[-1]
        zero_emb = torch.zeros(d, dtype=torch.float32, device=device)
        candidate_embeddings_tensor = torch.stack([zero_emb for _ in range(candidate_size)], dim=0)
        pos_index = -1
        return candidate_embeddings_tensor, pos_index

    # positive candidate: item_id 자신
    pos_tensor = torch.tensor([item_id], dtype=torch.long, device=device)
    # negative candidates: candidate_tensor[item_id][:candidate_size-1]
    neg_tensor = candidate_tensor[item_id, :candidate_size-1]  # shape: [candidate_size-1]
    # 전체 candidate_ids: [candidate_size]
    candidate_ids = torch.cat([pos_tensor, neg_tensor], dim=0)
    
    # Lookup base embeddings: item_embeddings_tensor는 [N+1, base_dim]
    base_embs = item_embeddings_tensor[candidate_ids]  # [candidate_size, base_dim]
    # Apply projection_ffn vectorized: 결과 [candidate_size, d]
    candidate_embeddings_tensor = projection_ffn(base_embs)
    
    pos_index = 0
    return candidate_embeddings_tensor, pos_index

def build_candidate_set(batch_item_ids, candidate_size, item_embeddings, projection_ffn, candidate_dict):
    """
    배치 내 각 positive 아이템에 대해, 전체 아이템 embedding에서 negative sample을
    추가하여 candidate set을 만듭니다.
    
    각 candidate set은 positive embedding (label 1)과 negative embedding들 (label 0)으로 구성됩니다.
    무작위로 섞은 후, 정답(positive)의 인덱스를 기록합니다.
    
    Parameters:
      - batch_item_ids: 리스트, 배치 내 각 샘플의 positive item id.
          * 각 원소가 str인 1차원 리스트이거나, 2차원 리스트 (예: 각 user의 session별 positive id 리스트)
      - candidate_size: int, 각 candidate set의 총 크기 (positive 1개 + negative samples).
      - item_embeddings: dict, {item_id: embedding (numpy array)}.
      - projection_ffn: nn.Module, 384차원 embedding을 128차원으로 projection.
    
    Returns:
      - candidate_set_tensor: torch.Tensor, shape이
            [batch, candidate_size, embed_dim] 또는 [batch, num_sessions, candidate_size, embed_dim]
      - correct_indices_tensor: torch.Tensor, 정답(positive)이 위치하는 인덱스 (shape에 맞게)
    """
    device = next(projection_ffn.parameters()).device if projection_ffn is not None else torch.device('cpu')
    all_item_ids = list(item_embeddings.keys())

    # 배치가 2차원 (각 user의 session별)인지 1차원인지를 확인
    if isinstance(batch_item_ids[0], list):
        # 2D case: 각 원소가 session별 item id 리스트
        candidate_sets = []
        correct_indices = []
        for user_sessions in batch_item_ids:
            candidate_set_user = []
            correct_indices_user = []
            for item_id in user_sessions:
                cand, pos_idx = build_candidate_for_item(item_id, projection_ffn, candidate_dict, item_embeddings, candidate_size, device)
                candidate_set_user.append(cand)         # cand: [candidate_size, embed_dim]
                correct_indices_user.append(pos_idx)
            # candidate_set_user: [num_sessions, candidate_size, embed_dim]
            candidate_set_user_tensor = torch.stack(candidate_set_user, dim=0)
            candidate_sets.append(candidate_set_user_tensor)
            correct_indices.append(correct_indices_user)
        # 최종: candidate_set_tensor: [batch, num_sessions, candidate_size, embed_dim]
        candidate_set_tensor = torch.stack(candidate_sets, dim=0)
        # correct_indices_tensor: [batch, num_sessions]
        correct_indices_tensor = torch.tensor(correct_indices, dtype=torch.long, device=device)
    else:
        # 1D case: 각 원소가 단일 item id
        candidate_sets = []
        correct_indices = []
        for item_id in tqdm(batch_item_ids, total=len(batch_item_ids), desc='Candidate구하기', leave=False):
            cand, pos_idx = build_candidate_for_item(item_id, projection_ffn, candidate_dict, item_embeddings, candidate_size, device)
            candidate_sets.append(cand)  # cand: [candidate_size, embed_dim]
            correct_indices.append(pos_idx)
        # 최종: candidate_set_tensor: [batch, candidate_size, embed_dim]
        candidate_set_tensor = torch.stack(candidate_sets, dim=0)
        correct_indices_tensor = torch.tensor(correct_indices, dtype=torch.long, device=device)
    
    return candidate_set_tensor, correct_indices_tensor

def get_candidate_set_for_batch(batch_item_ids, candidate_size, item_embeddings_tensor, projection_ffn, candidate_tensor=None, global_candidate=False):
    """
    batch_item_ids: [B, L] tensor (already padded from get_batch_item_ids)
    Returns:
      candidate_set_tensor: [B, L_valid, candidate_size, d] where L_valid is the padded valid length for each sample
      correct_indices_tensor: [B, L_valid] tensor
    """
    device = next(projection_ffn.parameters()).device if projection_ffn is not None else torch.device('cpu')
    B, L = batch_item_ids.shape

    if not global_candidate:
        # 기존 로직 유지
        candidate_sets = []
        correct_indices = []
        valid_lengths = []  # 각 샘플별 valid 길이 (패딩 제외)
        for b in range(B):
            candidate_set_sample = []
            correct_indices_sample = []
            valid_len = (batch_item_ids[b] != -1).sum().item()
            valid_lengths.append(valid_len)
            for l in range(valid_len):
                item_val = int(batch_item_ids[b, l].item())
                cand, pos_idx = build_candidate_for_item(item_val, projection_ffn, candidate_tensor, item_embeddings_tensor, candidate_size, device)
                candidate_set_sample.append(cand)
                correct_indices_sample.append(pos_idx)
            if valid_len > 0:
                candidate_set_sample = torch.stack(candidate_set_sample, dim=0)  # [valid_len, candidate_size, d]
                correct_indices_sample = torch.tensor(correct_indices_sample, dtype=torch.long, device=device)
            else:
                candidate_set_sample = torch.zeros((0, candidate_size, projection_ffn.fc1.out_features), device=device)
                correct_indices_sample = torch.zeros((0,), dtype=torch.long, device=device)
            candidate_sets.append(candidate_set_sample)
            correct_indices.append(correct_indices_sample)
        max_len = max(valid_lengths) if valid_lengths else 0
        padded_candidate_sets = []
        padded_correct_indices = []
        for cs, ci, v_len in zip(candidate_sets, correct_indices, valid_lengths):
            if v_len < max_len:
                pad_tensor = torch.zeros((max_len - v_len, cs.shape[1], cs.shape[2]), device=cs.device)
                cs_padded = torch.cat([cs, pad_tensor], dim=0)
                ci_pad = torch.full((max_len - v_len,), -1, dtype=torch.long, device=ci.device)
                ci_padded = torch.cat([ci, ci_pad], dim=0)
            else:
                cs_padded = cs
                ci_padded = ci
            padded_candidate_sets.append(cs_padded)
            padded_correct_indices.append(ci_padded)
        candidate_set_tensor = torch.stack(padded_candidate_sets, dim=0)  # [B, max_len, candidate_size, d]
        correct_indices_tensor = torch.stack(padded_correct_indices, dim=0)  # [B, max_len]
    else:
        # --- Global Candidate Mode (Tensor 기반) ---
        # 전체 candidate set은 이미 preprocessed된 item_embeddings_tensor에서 index 0(패딩)을 제외한 모든 항목입니다.
        # 즉, global candidate set = projection_ffn(item_embeddings_tensor[1:])
        N = item_embeddings_tensor.shape[0]
        global_candidate_ids = torch.arange(1, N, device=device)  # [C] (C = N-1)
        global_base_embs = item_embeddings_tensor[global_candidate_ids]  # [C, base_dim]
        candidate_set_tensor = projection_ffn(global_base_embs)  # [C, d]
        
        # 정답 index 계산:
        # batch_item_ids: [B, L] (new index, padding=-1), 정답 index = item_id - 1
        valid_mask = (batch_item_ids != -1)
        correct_indices_tensor = torch.full((B, L), -1, dtype=torch.long, device=device)
        # 에러 수정: batch_item_ids[valid_mask]를 long 형으로 변환하여 연산
        correct_indices_tensor[valid_mask] = batch_item_ids[valid_mask].long() - 1
    return candidate_set_tensor, correct_indices_tensor



def get_batch_item_ids(item_ids, strategy):
    """
    item_ids: [B, S, I] tensor (padding은 -1)
    strategy: 선택 전략 (예: 'EachSession_LastInter', 'Global_LastInter', ...)
    
    Returns:
      selected_ids: [B, max_len] tensor (각 sample에서 선택된 item id 목록, padding=-1)
      loss_mask: [B, max_len] Boolean tensor (True이면 해당 위치를 loss 계산에 포함)
      session_ids: [B, max_len] tensor (각 위치의 item이 어느 세션에서 선택되었는지; padding=-1)
    """
    B, S, I = item_ids.shape
    results = []         # 각 sample별 selected item id list
    mask_results = []    # 각 sample별 valid 여부 (True: valid)
    session_results = [] # 각 sample별, 해당 item이 나온 session id
    
    if strategy == 'EachSession_LastInter':
        for b in range(B):
            chosen = []
            sess_ids = []
            for s in range(S):
                session = item_ids[b, s, :]  # [I]
                valid_indices = (session != -1).nonzero(as_tuple=False).squeeze(-1)
                if valid_indices.numel() > 0:
                    last_index = valid_indices[-1].item()
                    chosen.append(session[last_index].item())
                    sess_ids.append(s)
            results.append(chosen)
            mask_results.append([True] * len(chosen))
            session_results.append(sess_ids)
            
    elif strategy == 'Global_LastInter':
        for b in range(B):
            flat = item_ids[b].view(-1)  # [S*I]
            valid_indices = (flat != -1).nonzero(as_tuple=False).squeeze(-1)
            if valid_indices.numel() > 0:
                last_index = valid_indices[-1].item()
                chosen_item = flat[last_index].item()
                # session id 계산: last_index // I
                sess_id = last_index // I
                results.append([chosen_item])
                mask_results.append([True])
                session_results.append([sess_id])
            else:
                results.append([])
                mask_results.append([])
                session_results.append([])
                
    elif strategy == 'LastSession_AllInter':
        for b in range(B):
            valid_sessions = [ (item_ids[b, s, :] != -1).any().item() for s in range(S) ]
            valid_session_indices = [s for s, valid in enumerate(valid_sessions) if valid]
            if valid_session_indices:
                last_session = valid_session_indices[-1]
                session_vec = item_ids[b, last_session, :]
                valid_items = session_vec[session_vec != -1]
                chosen = valid_items.tolist()
                sess_ids = [last_session] * len(chosen)
            else:
                chosen = []
                sess_ids = []
            results.append(chosen)
            mask_results.append([True] * len(chosen))
            session_results.append(sess_ids)
            
    elif strategy == 'AllInter_ExceptFirst':
        for b in range(B):
            flat = item_ids[b].view(-1)
            valid = flat[flat != -1]
            indices = (flat != -1).nonzero(as_tuple=False).squeeze(-1)
            if indices.numel() > 0:
                indices = indices[1:]  # 첫 번째 제외
                chosen = flat[indices].tolist()
                sess_ids = (indices // I).tolist()
            else:
                chosen = []
                sess_ids = []
            results.append(chosen)
            mask_results.append([True] * len(chosen))
            session_results.append(sess_ids)
            
    elif strategy == 'AllInter':
        for b in range(B):
            flat = item_ids[b].view(-1)
            indices = (flat != -1).nonzero(as_tuple=False).squeeze(-1)
            chosen = flat[indices].tolist()
            sess_ids = (indices // I).tolist()
            results.append(chosen)
            mask_results.append([True] * len(chosen))
            session_results.append(sess_ids)
            
    elif strategy == 'EachSession_First_and_Last_Inter':
        for b in range(B):
            chosen = []
            sess_ids = []
            for s in range(S):
                session = item_ids[b, s, :]
                valid_indices = (session != -1).nonzero(as_tuple=False).squeeze(-1)
                if valid_indices.numel() > 0:
                    first_index = valid_indices[0].item()
                    last_index = valid_indices[-1].item()
                    chosen.extend([session[first_index].item(), session[last_index].item()])
                    sess_ids.extend([s, s])
            results.append(chosen)
            mask_results.append([True] * len(chosen))
            session_results.append(sess_ids)
            
    elif strategy == 'EachSession_Except_First':
        for b in range(B):
            chosen = []
            sess_ids = []
            for s in range(S):
                session = item_ids[b, s, :]
                valid = session[session != -1]
                if valid.numel() > 0:
                    remainder = valid[1:]
                    chosen.extend(remainder.tolist())
                    sess_ids.extend([s] * remainder.numel())
            results.append(chosen)
            mask_results.append([True] * len(chosen))
            session_results.append(sess_ids)
            
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # 배치 내 최대 길이(max_len)로 padding (-1 for item id, False for mask, -1 for session id)
    max_len = max(len(r) for r in results) if results else 0
    padded_ids = []
    padded_mask = []
    padded_session_ids = []
    for r, m, s in zip(results, mask_results, session_results):
        pad_len = max_len - len(r)
        padded_ids.append(r + [-1] * pad_len)
        padded_mask.append(m + [False] * pad_len)
        padded_session_ids.append(s + [-1] * pad_len)
    
    selected_ids = torch.tensor(padded_ids, dtype=item_ids.dtype, device=item_ids.device)
    loss_mask = torch.tensor(padded_mask, dtype=torch.bool, device=item_ids.device)
    session_ids = torch.tensor(padded_session_ids, dtype=torch.long, device=item_ids.device)
    return selected_ids, loss_mask, session_ids


def get_candidate_set_for_batch_tensorized(batch_item_ids, candidate_size, 
                                             item_embeddings_t, projection_ffn, 
                                             candidate_dict_t=None, 
                                             global_candidate=False,
                                             item_embeddings_dict=None):
    """
    Args:
      batch_item_ids: [B, L] tensor (padding=-1)
      candidate_size: int, Non-global 모드에서만 사용 (positive+negatives)
      item_embeddings_t: [num_items, base_dim] tensor (GPU에 올린 아이템 임베딩)
      projection_ffn: nn.Module, 입력: base_dim, 출력: d (최종 임베딩 차원)
      candidate_dict_t: [num_items, candidate_size-1] tensor (Non-global 모드용)
      global_candidate: bool, 글로벌 후보 집합 여부. 
         - False: 출력 candidate_set_tensor는 [B, L, candidate_size, d]
         - True:  출력 candidate_set_tensor는 [C, d] (C는 전체 후보 수, 매우 클 수 있음)
      item_embeddings_dict: Global 모드 사용 시, 원본 item 임베딩 dictionary (키: str 혹은 int)
                           필수 (전체 item 집합을 구성하기 위해 사용)
    Returns:
      candidate_set_tensor, correct_indices_tensor
      - Non-global mode:
          candidate_set_tensor: [B, L, candidate_size, d]
          correct_indices_tensor: [B, L] (padding은 -1)
      - Global mode:
          candidate_set_tensor: [C, d]  (C: 전체 후보 수)
          correct_indices_tensor: [B, L] (각 valid 위치는 global 후보 집합 내 positive item의 index; padding은 -1)
    """
    device = item_embeddings_t.device
    B, L = batch_item_ids.shape

    if not global_candidate:
        # --- Non-Global Candidate Mode (벡터화) ---
        valid_mask = (batch_item_ids != -1)  # [B, L]
        valid_indices = valid_mask.nonzero(as_tuple=False)  # [N_valid, 2]
        N_valid = valid_indices.shape[0]
        
        d = projection_ffn(item_embeddings_t[0:1]).shape[-1]
        candidate_set_tensor = torch.zeros((B, L, candidate_size, d), dtype=torch.float32, device=device)
        correct_indices_tensor = torch.full((B, L), -1, dtype=torch.long, device=device)
        
        if N_valid == 0:
            return candidate_set_tensor, correct_indices_tensor

        flat_item_ids = batch_item_ids[valid_mask]  # [N_valid]
        # 각 유효 아이템에 대해, positive는 자기 자신, negatives는 candidate_dict_t lookup
        negatives = candidate_dict_t[flat_item_ids]   # [N_valid, candidate_size - 1]
        candidate_ids = torch.empty((N_valid, candidate_size), dtype=torch.long, device=device)
        candidate_ids[:, 0] = flat_item_ids
        candidate_ids[:, 1:] = negatives

        base_embs = item_embeddings_t[candidate_ids]  # [N_valid, candidate_size, base_dim]
        base_embs_flat = base_embs.view(-1, base_embs.shape[-1])  # [N_valid*candidate_size, base_dim]
        projected_flat = projection_ffn(base_embs_flat)  # [N_valid*candidate_size, d]
        projected = projected_flat.view(N_valid, candidate_size, -1)  # [N_valid, candidate_size, d]
        
        valid_correct = torch.zeros((N_valid,), dtype=torch.long, device=device)
        candidate_set_tensor[valid_mask] = projected
        correct_indices_tensor[valid_mask] = valid_correct

        return candidate_set_tensor, correct_indices_tensor
    else:
        # --- Global Candidate Mode (벡터화) ---
        if item_embeddings_dict is None:
            raise ValueError("Global candidate 모드에서는 item_embeddings_dict가 필요합니다.")

        # 1. 전체 후보 집합 구성: item_embeddings_dict의 모든 키 (또는 필요한 경우 전처리된 전체 후보 리스트)
        # 여기서 후보 수 C는 매우 클 수 있음.
        try:
            all_candidate_ids = [int(key) for key in item_embeddings_dict.keys()]
        except Exception:
            all_candidate_ids = [key for key in item_embeddings_dict.keys()]
        # 전체 후보 집합을 tensor로 만들기
        candidate_ids_global = torch.tensor(all_candidate_ids, dtype=torch.long, device=device)  # [C]
        
        # 2. 후보 임베딩 lookup + projection
        global_base_embs = item_embeddings_t[candidate_ids_global]  # [C, base_dim]
        global_projected = projection_ffn(global_base_embs)  # [C, d]
        candidate_set_tensor = global_projected  # [C, d]

        # 3. 배치 내 각 valid 아이템에 대해, global 후보 집합 내에서 positive item의 인덱스 찾기.
        #    batch_item_ids: [B, L] (padding=-1)
        #    -> valid_mask_global: [B, L]
        valid_mask_global = (batch_item_ids != -1)
        # 초기 correct indices tensor: [B, L]
        correct_indices_tensor = torch.full((B, L), -1, dtype=torch.long, device=device)
        
        if valid_mask_global.sum() == 0:
            return candidate_set_tensor, correct_indices_tensor
        
        # 4. vectorized 방식: 
        #    - candidate_ids_global: [C] → 확장: [1, 1, C]
        ic = candidate_ids_global.view(1, 1, -1)
        #    - batch_item_ids: [B, L] → 확장: [B, L, 1]
        expanded_items = batch_item_ids.unsqueeze(-1)  # [B, L, 1]
        # 비교: 두 tensor를 비교 -> [B, L, C] boolean tensor
        match = (expanded_items == ic)
        # valid한 위치에 대해, 후보가 여러 개 일 경우 첫번째 True의 index를 가져옴.
        # argmax는 첫번째 최대값(여기서는 True가 1로 인식됨)을 반환합니다.
        match_float = match.float()  # [B, L, C]
        correct_idx = match_float.argmax(dim=-1)  # [B, L]
        # 단, padding 위치는 여전히 -1로 유지
        correct_indices_tensor[valid_mask_global] = correct_idx[valid_mask_global]
        print(f"candidate_set_tensor.shape: {candidate_set_tensor.shape}, correct_indices_tensor.shape: {correct_indices_tensor.shape}")
        print(f"candidate_set_tensor.example: {candidate_set_tensor[0]}")
        print(f"correct_indices_tensor.example: {correct_indices_tensor[0]}")
        return candidate_set_tensor, correct_indices_tensor
