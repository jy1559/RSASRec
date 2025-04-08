# util.py
import json
import pickle
import random
import torch
from tqdm.auto import tqdm
import torch.nn.functional as F
import numpy as np
from models.sub1_sequence_embedding import sentence_embedder, mean_pooling

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


def load_item_embeddings(embedding_file_path):
    """
    저장된 item embedding pickle 파일을 로드합니다.
    
    Parameters:
    - embedding_file_path: str, embedding pickle 파일 경로.
    
    Returns:
    - embeddings: dict, {item_id: embedding (numpy array)}
    """
    with open(embedding_file_path, 'rb') as f:
        embeddings = pickle.load(f)
    return embeddings

def build_candidate_for_item(item_id, projection_ffn, candidate_dict, item_embeddings, candidate_size, device):
    """
    주어진 item_id에 대해 후보 임베딩들을 반환합니다.
    기존 랜덤 샘플링 대신, 미리 저장된 candidate_dict(npz에서 불러온 후보 세트)를 사용합니다.
    
    Parameters:
    - item_id: int, 처리할 아이템 id (padding은 -1)
    - projection_ffn: nn.Module, 단층 projector (예: nn.Linear(384, 128))
    - candidate_dict: dict, {item_id (int): list of candidate item ids (int)} precomputed 후보 세트
    - item_embeddings: dict, {str(item_id): numpy array} 형태의 precomputed item 임베딩
    - candidate_size: int, 최종 후보 세트 크기 (예: 128)
    
    Returns:
    - candidate_embeddings_shuffled: list of [emb_dim] torch.Tensor (단층 projector 적용 후)
    - pos_index: int, positive candidate의 인덱스 (여기서는 항상 0)
    """
    # 패딩된 경우: item_id가 -1이면, 동일한 크기의 0 벡터 후보 세트를 반환
    if item_id == -1:
        emb_dim = projection_ffn.fc2.out_features if hasattr(projection_ffn, "fc2") else projection_ffn.fc1.out_features
        zero_emb = torch.zeros(emb_dim, dtype=torch.float32, device=device)
        candidate_embeddings_shuffled = [zero_emb for _ in range(candidate_size)]
        pos_index = -1
        return torch.stack(candidate_embeddings_shuffled, dim=0), pos_index

    # Positive candidate: 자기 자신의 임베딩 (반드시 후보 세트의 첫 번째 위치로)
    positive_emb = torch.tensor(item_embeddings[str(item_id)], dtype=torch.float32, device=device)
    
    # 미리 저장된 후보 세트에서 해당 아이템의 후보들을 조회 (candidate_dict는 int:item id -> list[int])
    if item_id in candidate_dict:
        candidate_list = [item_id] + candidate_dict[item_id]
    else:
        candidate_list = [item_id]
    
    if len(candidate_list) != candidate_size:
        raise ValueError(f"Candidate set length for item {item_id} is {len(candidate_list)}, expected {candidate_size}.")
    
    # 각 후보 item id에 대해, item_embeddings에서 base 임베딩을 조회합니다.
    candidate_embeddings = []
    for cid in candidate_list:
        if cid == -1:
            emb = torch.zeros(positive_emb.shape, dtype=torch.float32, device=device)
        else:
            emb = torch.tensor(item_embeddings[str(cid)], dtype=torch.float32, device = device)
        candidate_embeddings.append(emb)
    
    # 단층 projector 적용: 각 candidate embedding에 대해 projection_ffn (입력 384 -> 출력 128)을 적용합니다.
    if projection_ffn is not None:
        candidate_embeddings = [projection_ffn(emb.unsqueeze(0)).squeeze(0) for emb in candidate_embeddings]
    
    # positive candidate는 item_id 자신이므로 index 0로 둡니다.
    pos_index = 0
    if isinstance(candidate_embeddings, list):
        candidate_embeddings_tensor = torch.stack(candidate_embeddings, dim=0)
    else:
        candidate_embeddings_tensor = candidate_embeddings
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

def get_candidate_set_for_batch(batch_item_ids, candidate_size, item_embeddings, projection_ffn, candidate_dict=None, global_candidate=False):
    """
    batch_item_ids: [B, L] tensor (already padded from get_batch_item_ids)
    Returns:
      candidate_set_tensor: [B, L_valid, candidate_size, d] where L_valid is the padded valid length for each sample
      correct_indices_tensor: [B, L_valid] tensor
    """
    device = next(projection_ffn.parameters()).device if projection_ffn is not None else torch.device('cpu')
    B, L = batch_item_ids.shape
    candidate_sets = []
    correct_indices = []
    valid_lengths = []  # 각 샘플별 valid 길이 (패딩 제외)
    # 각 샘플별로 valid item (padding이 아닌 값)에 대해서 후보 생성
    for b in range(B):
        candidate_set_sample = []
        correct_indices_sample = []
        # valid 길이: -1이 아닌 값의 개수
        valid_len = (batch_item_ids[b] != -1).sum().item()
        valid_lengths.append(valid_len)
        for l in range(valid_len):
            item_val = int(batch_item_ids[b, l].item())
            cand, pos_idx = build_candidate_for_item(item_val, projection_ffn, candidate_dict, item_embeddings, candidate_size, device)
            candidate_set_sample.append(cand)  # cand: [candidate_size, d]
            correct_indices_sample.append(pos_idx)
        if valid_len > 0:
            candidate_set_sample = torch.stack(candidate_set_sample, dim=0)  # [valid_len, candidate_size, d]
            correct_indices_sample = torch.tensor(correct_indices_sample, dtype=torch.long, device=device)  # [valid_len]
        else:
            # 만약 valid item이 없다면 (드물겠지만) 0 길이 tensor 생성
            candidate_set_sample = torch.zeros((0, candidate_size, projection_ffn.fc1.out_features), device=device)
            correct_indices_sample = torch.zeros((0,), dtype=torch.long, device=device)
        candidate_sets.append(candidate_set_sample)
        correct_indices.append(correct_indices_sample)
    # 배치 내 최대 valid 길이
    max_len = max(valid_lengths) if valid_lengths else 0
    # 각 샘플별로 max_len으로 pad (pad: candidate embedding은 0, 정답 인덱스는 -1)
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
    
    if global_candidate:
        all_candidate_ids = list(item_embeddings.keys())[:candidate_size]
        candidate_set_tensor = torch.tensor(all_candidate_ids, dtype=torch.long, device=device)
        correct_indices_tensor = torch.zeros(B, L, dtype=torch.long, device=device)
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