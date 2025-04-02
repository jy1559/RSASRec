from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import torch.nn as nn
from .sub2_time_gap import sinusoidal_encoding
from tqdm.auto import tqdm

########################
# 1. Mean Pooling 함수 #
########################
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # 모든 토큰 임베딩
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask

###############################
# 2. LLM 초기화 및 임베딩 함수 #
###############################
def sentence_embedder(hf_model_path='sentence-transformers/all-MiniLM-L6-v2'):
    tokenizer = AutoTokenizer.from_pretrained(hf_model_path)
    model = AutoModel.from_pretrained(hf_model_path)
    return tokenizer, model

def get_llm_embeddings(tokenizer, model, embedding_sentences, interaction_mask, sub_batch_size=32):
    """
    LLM을 사용하여 문장 임베딩을 얻는 함수.
    - embedding_sentences: list of list of list of str, shape [B, S, I]
    - interaction_mask: [B, S, I] tensor
    반환: [B, S, I, hidden_size] tensor
    """
    device = next(model.parameters()).device
    B, S, I = interaction_mask.shape

    # valid 문장 모으기
    valid_sentences = []
    indices = []  # 각 valid 문장의 위치 (b, s, i)
    for b in range(B):
        for s in range(S):
            for i in range(I):
                if interaction_mask[b, s, i] == 1:
                    valid_sentences.append(embedding_sentences[b][s][i])
                    indices.append((b, s, i))
    
    hidden_size = model.config.hidden_size
    output = torch.zeros(B, S, I, hidden_size, device=device)

    if len(valid_sentences) == 0:
        return output

    all_embeds = []
    for start in range(0, len(valid_sentences), sub_batch_size):
        batch_sents = valid_sentences[start:start+sub_batch_size]
        encoded_input = tokenizer(batch_sents, padding=True, truncation=True, return_tensors='pt')
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        with torch.no_grad():
            model_output = model(**encoded_input)
        batch_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
        all_embeds.append(batch_embeddings)
    all_embeds = torch.cat(all_embeds, dim=0)
    
    indices_tensor = torch.tensor(indices, dtype=torch.long, device=device).t()  # shape: [3, num_valid]
    output[indices_tensor[0], indices_tensor[1], indices_tensor[2]] = all_embeds
    return output

####################################
# 3. Projection 및 Positional Encoding
####################################

class ProjectionFFN(nn.Module):
    """
    2층 FFN projection: 입력 차원(input_dim) -> hidden_dim -> output_dim (emb_dim)
    """
    def __init__(self, input_dim, hidden_dim, output_dim, device, single_layer = True):
        super(ProjectionFFN, self).__init__()
        self.device = device
        self.single_layer = single_layer
        if self.single_layer:
            self.fc1 = nn.Linear(input_dim, output_dim) 
        else:
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        return self.fc1(x) if self.single_layer else self.fc2(self.relu(self.fc1(x)))

class TimestampEncoder(nn.Module):
    """
    Timestamp를 인코딩하는 모듈.
    간단히 1차원 스칼라를 받아 output_dim 차원 벡터로 변환.
    """
    def __init__(self, output_dim, sinusoidal_dim=512, sinusoidal_max=10000):
        super(TimestampEncoder, self).__init__()
        self.linear = nn.Linear(sinusoidal_dim, output_dim)
        self.sinusoidal_dim = sinusoidal_dim
        self.sinusoidal_max = sinusoidal_max

    def forward(self, x):
        # x: [batch] 또는 [1] 형태의 tensor
        sin_enc = sinusoidal_encoding(x, self.sinusoidal_dim, self.sinusoidal_max)
        return self.linear(sin_enc)  # 결과: [output_dim]
        

class AddInfoEncoder(nn.Module):
    """
    add_info를 인코딩하는 모듈.
    add_info가 단순 스칼라인 경우 사용.
    """
    def __init__(self, output_dim):
        super(AddInfoEncoder, self).__init__()
        self.linear = nn.Linear(1, output_dim)
    
    def forward(self, x):
        # x: [batch] 또는 [1] tensor
        return self.linear(x.unsqueeze(-1)).squeeze(-1)  # 결과: [output_dim]

#########################################
# 4. Item+Positional 임베딩 함수 (LLM 미사용)
#########################################
def get_item_embeddings_old(item_ids, add_info, timestamps, item_embeddings_dict,
                        projection_ffn, timestamp_encoder, add_info_encoders, valid_mask):
    """
    LLM 미사용 시, precomputed item embedding과 timestamp, add_info를 결합하여
    2층 FFN projection으로 emb_dim 차원으로 매핑합니다.
    
    - item_ids: [B, S, I] tensor (padding: -1)
    - add_info: list of list of list of add_info (numeric) [B, S, I]
       * 각 add_info[b][s][i]는 add_info 값들의 리스트이며, 
         만약 None이면 그 세션의 이후 interaction은 더 이상 valid하지 않음을 의미.
    - timestamps: [B, S, I] tensor (float)
    - item_embeddings_dict: { str(item_id): np.array([...]) } (예: base embedding은 384차원)
    - projection_ffn: ProjectionFFN 모듈, 입력 차원은 각 구성요소가 같다고 가정 (ex. 384)
    - timestamp_encoder: TimestampEncoder 모듈 (출력 차원: ts_enc_dim == base_dim)
    - add_info_encoders: list of AddInfoEncoder 모듈 (각 출력 차원: add_enc_dim == base_dim)
    
    반환: [B, S, I, emb_dim] tensor
    """
    device = item_ids.device
    B, S, I = item_ids.shape
    base_dim = 384  # base embedding 차원 (예시)
    emb_dim = projection_ffn.fc2.out_features if hasattr(projection_ffn, "fc2") else projection_ffn.fc1.out_features # projection_ffn의 출력 차원
    
    # 2. base embedding 채우기: [B, S, I, base_dim]
    base_embs = torch.zeros(B, S, I, base_dim, device=device)
    # 유효한 item id들만 골라내어 unique id 리스트 생성
    valid_mask = (valid_mask != 0)
    valid_item_ids = item_ids[valid_mask]
    if valid_item_ids.numel() > 0:
        unique_ids = torch.unique(valid_item_ids)
    else:
        unique_ids = torch.tensor([], device=device)
    # 각 unique id에 대해 dictionary에 있으면 embedding을, 없으면 0 벡터를 할당
    mapping = {}
    for iid in unique_ids.tolist():
        iid_str = str(iid)
        if iid_str in item_embeddings_dict:
            mapping[iid] = torch.tensor(item_embeddings_dict[iid_str], dtype=torch.float32, device=device)
        else:
            mapping[iid] = torch.zeros(base_dim, device=device)
    # item_ids 텐서에서 각 원소에 대해 mapping을 적용 (유효한 위치만)
    for uid in unique_ids.tolist():
        mask_uid = (item_ids == uid) & valid_mask  # [B, S, I]
        if mask_uid.any():
            base_embs[mask_uid] = mapping[uid]
    
    # 3. Timestamp encoding: timestamps [B,S,I] -> [B,S,I, base_dim]
    ts_flat = timestamps.view(-1, 1)  # [B*S*I, 1]
    ts_enc_flat = timestamp_encoder(ts_flat)  # [B*S*I, base_dim]
    ts_enc = ts_enc_flat.view(B, S, I, -1)
    # 4. Add_info encoding:
    # add_info는 list 형태이므로, 먼저 [B, S, I, num_add_info] 텐서로 변환 (유효한 경우에만 값 사용)
    num_add_info = len(add_info_encoders)
    add_info_tensor = torch.zeros(B, S, I, num_add_info, dtype=torch.float32, device=device)
    for b in range(B):
        for s in range(S):
            for i in range(I):
                if valid_mask[b, s, i]:
                    # add_info[b][s][i]는 add_info 값 리스트라 가정 (길이 == num_add_info)
                    ai_vals = add_info[b][s][i]
                    add_info_tensor[b, s, i] = torch.tensor(ai_vals, dtype=torch.float32, device=device)
    
    # 각 add_info feature에 대해 인코딩 후 합산 (출력 shape: [B,S,I, base_dim])
    add_enc_sum = torch.zeros(B, S, I, base_dim, device=device)
    for j, encoder in enumerate(add_info_encoders):
        # j번째 add_info: [B,S,I]
        ai_j = add_info_tensor[..., j].unsqueeze(-1)  # [B,S,I,1]
        ai_j_flat = ai_j.view(-1, 1)  # [B*S*I, 1]
        ai_enc_flat = encoder(ai_j_flat)  # [B*S*I, base_dim]
        ai_enc = ai_enc_flat.view(B, S, I, -1)
        add_enc_sum = add_enc_sum + ai_enc

    # 5. 각 구성요소 합산: base_embs, ts_enc, add_enc_sum (모두 [B,S,I, base_dim])
    combined = base_embs + ts_enc + add_enc_sum

    # 6. Projection FFN 적용: flatten 후 재구성
    combined_flat = combined.view(-1, combined.shape[-1])  # [B*S*I, base_dim]
    projected_flat = projection_ffn(combined_flat)  # [B*S*I, emb_dim]
    projected = projected_flat.view(B, S, I, -1)

    # 7. invalid한 interaction은 0으로 만들기 (masking)
    projected = projected * valid_mask.unsqueeze(-1).float()
    
    return projected

def get_item_embeddings(item_ids, add_info, timestamps, item_embeddings_dict,
                        projection_ffn, timestamp_encoder, add_info_encoders, valid_mask):
    """
    LLM 미사용 시, precomputed item embedding과 timestamp, add_info를 결합하여
    2층 FFN projection으로 emb_dim 차원으로 매핑합니다.
    
    - item_ids: [B, S, I] tensor (padding: -1)
    - add_info: list of list of list of add_info 값 (numeric), shape [B, S, I, num_add_info]
      * padding된 부분은 반드시 add_info 값이 존재하지 않으므로, valid_mask=0
    - timestamps: [B, S, I] tensor (float)
    - item_embeddings_dict: dict, { str(item_id): np.array([...]) } (예: base embedding은 384차원)
    - projection_ffn: ProjectionFFN 모듈, 입력 차원은 base_dim(384) → 출력 emb_dim (예: 128)
    - timestamp_encoder: TimestampEncoder 모듈 (출력 차원: base_dim)
    - add_info_encoders: list of AddInfoEncoder 모듈 (각 출력 차원: base_dim)
    - valid_mask: [B, S, I] tensor (1: valid, 0: invalid)
    
    반환: [B, S, I, emb_dim] tensor (padding 위치는 0)
    """
    device = item_ids.device
    B, S, I = item_ids.shape
    base_dim = 384  # 원래 임베딩 차원
    emb_dim = projection_ffn.fc2.out_features if hasattr(projection_ffn, "fc2") else projection_ffn.fc1.out_features  # ProjectionFFN 출력 차원 (예: 128)
    
    # --- 1. 사전: item_embeddings_dict를 lookup table로 전환 (한번만 수행) ---
    if not hasattr(get_item_embeddings, 'embedding_matrix'):
        # item_ids는 -1이 패딩이므로, 인덱스 0은 패딩용(0 벡터)으로 예약합니다.
        keys = sorted(item_embeddings_dict.keys(), key=lambda x: int(x))
        mapping = {int(k): idx + 1 for idx, k in enumerate(keys)}
        embedding_list = [torch.zeros(base_dim, dtype=torch.float32, device=device)]  # index 0: padding
        for k in keys:
            emb = torch.tensor(item_embeddings_dict[k], dtype=torch.float32, device=device)
            embedding_list.append(emb)
        embedding_matrix = torch.stack(embedding_list, dim=0)  # [num_items+1, base_dim]
        get_item_embeddings.embedding_matrix = embedding_matrix
        get_item_embeddings.mapping = mapping

    # --- 2. Base Embedding Lookup (벡터화) ---
    # item_ids: [B,S,I] → flatten
    item_ids_flat = item_ids.view(-1)  # [B*S*I]
    indices = []
    mapping = get_item_embeddings.mapping
    for id in item_ids_flat.tolist():
        if id == -1:
            indices.append(0)
        else:
            indices.append(mapping.get(id, 0))
    indices_tensor = torch.tensor(indices, dtype=torch.long, device=device).view(B, S, I)
    # Lookup: [B,S,I, base_dim]
    base_embs = get_item_embeddings.embedding_matrix[indices_tensor]

    # --- 3. Timestamp 인코딩 일괄 처리 ---
    ts_flat = timestamps.view(-1, 1)         # [B*S*I, 1]
    ts_enc_flat = timestamp_encoder(ts_flat) # [B*S*I, base_dim]
    ts_enc = ts_enc_flat.view(B, S, I, -1)

    # --- 4. Add_info 인코딩 일괄 처리 ---
    num_add_info = len(add_info_encoders)
    if not isinstance(add_info, list) or not all(isinstance(sublist, list) for sublist in add_info):
        raise ValueError("add_info는 [B, S, I, num_add_info] 형태의 리스트여야 합니다.")
    add_info_tensor = torch.zeros(B, S, I, num_add_info, dtype=torch.float32, device=device)
    for b in range(B):
        for s in range(S):
            for i in range(I):
                if valid_mask[b, s, i]:
                    ai_vals = add_info[b][s][i]
                    if not isinstance(ai_vals, list):
                        raise ValueError(f"add_info[{b}][{s}][{i}]는 리스트여야 합니다.")
                    add_info_tensor[b, s, i] = torch.tensor(ai_vals, dtype=torch.float32, device=device)

    add_enc_sum = torch.zeros(B, S, I, base_dim, device=device)
    # 각 부가정보 피처별로 encoder 적용 (배치 연산)
    for j, encoder in enumerate(add_info_encoders):
        ai_j = add_info_tensor[..., j].view(-1, 1)  # [B*S*I, 1]
        ai_enc_flat = encoder(ai_j)                # [B*S*I, base_dim]
        ai_enc = ai_enc_flat.view(B, S, I, -1)
        add_enc_sum = add_enc_sum + ai_enc

    # --- 5. 요소별 결합 ---
    # 모두 [B,S,I, base_dim]
    combined = base_embs + 0.1*ts_enc + 0.1*add_enc_sum
    # valid_mask: [B,S,I] → unsqueeze to [B,S,I,1] and multiply: invalid 위치는 0
    combined = combined * valid_mask.unsqueeze(-1).float()

    # --- 6. Projection FFN 적용 ---
    combined_flat = combined.view(-1, base_dim)       # [B*S*I, base_dim]
    projected_flat = projection_ffn(combined_flat)      # [B*S*I, emb_dim]
    projected = projected_flat.view(B, S, I, -1)         # [B,S,I, emb_dim]
    
    return projected


#########################################
# 5. 단일 진입점 함수: get_embeddings
#########################################
def get_embeddings(batch_dict, use_llm,
                   tokenizer=None, llm_model=None,
                   item_embeddings_dict=None,
                   projection_ffn=None,
                   timestamp_encoder=None,
                   add_info_encoder=None,
                   valid_mask=None):
    """
    batch_dict: collate_fn 결과 (dict)
      - use_llm=True: {'embedding_sentences': ..., 'interaction_mask': ...}
      - use_llm=False: {'item_id': ..., 'add_info': ..., 'delta_ts': ...} 여기서 delta_ts를 timestamps로 사용
    """
    if use_llm:
        sentences = batch_dict['embedding_sentences']
        mask = batch_dict['interaction_mask']
        emb_output = get_llm_embeddings(tokenizer, llm_model, sentences, mask)
    else:
        item_ids = batch_dict['item_id']
        add_info = batch_dict['add_info']
        timestamps = batch_dict['delta_ts']  # 또는 별도의 timestamps key가 있다면 그걸 사용
        emb_output = get_item_embeddings(item_ids, add_info, timestamps,
                                         item_embeddings_dict,
                                         projection_ffn,
                                         timestamp_encoder,
                                         add_info_encoder,
                                         valid_mask)
    return emb_output