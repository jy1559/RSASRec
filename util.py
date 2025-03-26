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
    - device: str, 모델 실행 디바이스 ('cpu' 또는 'cuda').
    
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
    print(len(sentences))
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
        if i == 0: print(batch_embeddings.shape)
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
        zero_emb = torch.zeros(emb_dim, dtype=torch.float32)
        candidate_embeddings_shuffled = [zero_emb for _ in range(candidate_size)]
        pos_index = -1
        return candidate_embeddings_shuffled, pos_index

    # Positive candidate: 자기 자신의 임베딩 (반드시 후보 세트의 첫 번째 위치로)
    positive_emb = torch.tensor(item_embeddings[str(item_id)], dtype=torch.float32)
    
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
    if isinstance(candidate_embeddings_shuffled, list):
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
            # 모든 요소가 텐서인지 확인
            for idx, elem in enumerate(candidate_set_user):
                if not isinstance(elem, torch.Tensor):
                    raise TypeError(f"Expected Tensor but got {type(elem)} at index {idx} in candidate_set_user")
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

def get_candidate_set_for_batch(batch_item_ids, candidate_size, item_embeddings, projection_ffn, candidate_dict, global_candidate=False):
    """
    주어진 배치의 positive item id들을 기반으로 candidate set과 정답 인덱스를 생성합니다.
    
    Parameters:
      - batch_item_ids: 리스트 (예: [item_id1, item_id2, ...]) — 각 샘플의 positive item id.
      - candidate_size: int, candidate set의 크기.
      - item_embeddings: dict, {item_id: embedding (numpy array)}.
      - global_candidate: bool, True이면 모든 샘플에 대해 동일한 candidate set을 사용.
    
    Returns:
      - candidate_set_tensor: 
            * global_candidate=False: shape은 [batch_size, candidate_size, embed_dim] 또는 
              [batch_size, num_sessions, candidate_size, embed_dim] (2차원 리스트인 경우)
            * global_candidate=True: [batch_size, candidate_size, embed_dim] (모든 샘플에 동일 candidate set)
      - correct_indices_tensor: 각 샘플에 대해 정답(positive)의 인덱스를 나타내는 텐서.
    """
    if not global_candidate:
        return build_candidate_set(batch_item_ids, candidate_size, item_embeddings, projection_ffn, candidate_dict)
    else:
        # global candidate: candidate set은 전역적으로 하나 생성하고 모든 샘플에 동일하게 사용.
        all_item_ids = list(item_embeddings.keys())
        if len(all_item_ids) < candidate_size:
            candidate_ids = all_item_ids.copy()
        else:
            candidate_ids = random.sample(all_item_ids, candidate_size)
        
        # 배치의 각 positive id가 candidate_ids에 포함되어 있는지 확인.
        # 만약 포함되어 있지 않다면, 강제로 candidate_ids[0]을 해당 positive id로 교체.
        correct_indices = []
        for pos_id in batch_item_ids:
            pos_id_str = str(pos_id)
            if pos_id_str in candidate_ids:
                correct_indices.append(candidate_ids.index(pos_id_str))
            else:
                candidate_ids[0] = pos_id_str
                correct_indices.append(0)
        
        # candidate_ids에 해당하는 embedding을 한 번 계산.
        candidate_set = [item_embeddings[iid] for iid in candidate_ids]
        candidate_set = np.array(candidate_set, dtype=np.float32)  # [candidate_size, embed_dim]
        # 배치 크기만큼 복제하여 [batch_size, candidate_size, embed_dim] 형태로 만듦.
        candidate_set_tensor = torch.tensor(np.repeat(candidate_set[np.newaxis, ...], len(batch_item_ids), axis=0))
        correct_indices_tensor = torch.tensor(correct_indices, dtype=torch.long)
        return candidate_set_tensor, correct_indices_tensor


def get_batch_item_ids(item_ids, strategy='EachSession_LastInter'):
    """
    item_ids: [B, S, I] 텐서 또는 리스트 (패딩은 -1)
    strategy:
      - 'EachSession_LastInter': 각 세션별 마지막 valid item id → [B, S] 텐서
      - 'Global_LastInter': 각 user 전체에서 마지막 valid item id → [B] 텐서
      - 'LastSession_AllInter': 각 user의 마지막 valid session의 valid item id들을
                                 최대 길이(max_valid)로 padding하여 [B, max_valid] 텐서로 반환.
    
    Returns:
      batch_item_ids: 고정 shape의 텐서 (정수형, -1은 padding)
    """
    # 만약 tensor라면 리스트로 변환하지 않고 텐서 연산으로 처리합니다.
    # 우선, ensure tensor type
    if not torch.is_tensor(item_ids):
        item_ids = torch.tensor(item_ids, dtype=torch.int32)
    
    B, S, I = item_ids.shape
    
    if strategy == 'EachSession_LastInter':
        # valid mask: 1이면 valid, 0이면 padding
        valid_mask = (item_ids != -1)  # [B, S, I]
        rev_mask = valid_mask.flip(dims=[-1])
        rev_first_valid_idx = torch.argmax(rev_mask.to(torch.int32), dim=-1)  # [B, S]
        last_valid_idx = I - 1 - rev_first_valid_idx  # [B, S]
        last_items = torch.gather(item_ids, dim=2, index=last_valid_idx.unsqueeze(-1)).squeeze(-1)
        return last_items  # [B, S]
    
    elif strategy == 'Global_LastInter':
        # Flatten [B, S, I] → [B, S*I]
        flat_ids = item_ids.view(B, S * I)
        valid_mask = (flat_ids != -1)
        rev_mask = valid_mask.flip(dims=[-1])
        rev_first_valid_idx = torch.argmax(rev_mask.to(torch.int32), dim=-1)  # [B]
        last_valid_idx = S * I - 1 - rev_first_valid_idx  # [B]
        last_items = torch.gather(flat_ids, dim=1, index=last_valid_idx.unsqueeze(-1)).squeeze(-1)
        return last_items  # [B]    
    
    elif strategy == 'LastSession_AllInter':
        # 각 user의 마지막 valid session에서 valid item id들을 추출하여 padding
        # session_valid: [B, S]
        session_valid = (item_ids != -1).any(dim=-1)
        rev_session_valid = session_valid.flip(dims=[1])
        last_session_idx = S - 1 - torch.argmax(rev_session_valid.to(torch.int32), dim=1)  # [B]
        
        # For each sample, extract the valid item ids from that session.
        batch_list = []
        max_len = 0
        for b in range(B):
            sess_ids = item_ids[b, last_session_idx[b], :]  # [I]
            valid_ids = sess_ids[sess_ids != -1]
            max_len = max(max_len, valid_ids.numel())
            batch_list.append(valid_ids)
        # Pad each valid_ids to length max_len with -1
        padded = []
        for valid_ids in batch_list:
            pad_length = max_len - valid_ids.numel()
            if pad_length > 0:
                pad_tensor = torch.full((pad_length,), -1, dtype=valid_ids.dtype)
                padded_ids = torch.cat([valid_ids, pad_tensor], dim=0)
            else:
                padded_ids = valid_ids
            padded.append(padded_ids.unsqueeze(0))
        return torch.cat(padded, dim=0)  # [B, max_len]
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
