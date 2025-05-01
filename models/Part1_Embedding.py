from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm.auto import tqdm
import math

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

##############################
# 2. Sinusoidal Encoding 함수 #
##############################
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


def get_item_embeddings(item_ids, add_info, timestamps, item_embeddings_tensor,
                        projection_ffn, timestamp_encoder, add_info_encoders, valid_mask):
    """
    LLM 미사용 시, 미리 tensor화된 item embedding과 timestamp, add_info를 결합하여
    2층 FFN projection으로 emb_dim 차원으로 매핑합니다.
    
    - item_ids: [B, S, I] tensor (padding: -1), 이미 new index로 변환되어 있음.
    - add_info: [B, S, I, num_add_info] 형태의 리스트 (각 원소는 numeric list)
                * padding된 부분은 valid_mask가 0
    - timestamps: [B, S, I] tensor (float)
    - item_embeddings_tensor: [N+1, base_dim] tensor, 여기서 index 0은 패딩 전용 0 벡터
    - projection_ffn: ProjectionFFN 모듈 (입력: base_dim, 출력: emb_dim)
    - timestamp_encoder: TimestampEncoder 모듈 (출력: base_dim)
    - add_info_encoders: list of AddInfoEncoder 모듈 (각 출력: base_dim)
    - valid_mask: [B, S, I] tensor (1: valid, 0: padding)
    
    반환: [B, S, I, emb_dim] tensor (padding 위치는 0)
    """
    device = item_ids.device
    B, S, I = item_ids.shape
    base_dim = item_embeddings_tensor.shape[1]  # 예: 384
    # ProjectionFFN 출력 차원 (예: 128) - fc2 있으면 fc2, 아니면 fc1 사용
    emb_dim = projection_ffn.fc2.out_features if hasattr(projection_ffn, "fc2") else projection_ffn.fc1.out_features  

    # --- 1. Base Embedding Lookup (new index 사용) ---
    # item_ids는 이미 new index로 변환되어 있으므로, padding 값(-1)은 0으로 바꿔 lookup합니다.
    # 벡터화: torch.where()를 사용하여 -1이면 0, 나머지는 그대로 사용.
    indices_tensor = torch.where(item_ids == -1, torch.zeros_like(item_ids), item_ids)  # [B, S, I]
    # lookup: item_embeddings_tensor의 shape: [N+1, base_dim]
    base_embs = item_embeddings_tensor[indices_tensor]  # [B, S, I, base_dim]

    # --- 2. Timestamp 인코딩 (벡터화) ---
    ts_flat = timestamps.reshape(-1, 1)         # [B*S*I, 1]
    ts_enc_flat = timestamp_encoder(ts_flat) # [B*S*I, base_dim]
    ts_enc = ts_enc_flat.reshape(B, S, I, -1)     # [B, S, I, base_dim]

    # --- 3. Add_info 인코딩 (리스트를 tensor로 변환) ---
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
    for j, encoder in enumerate(add_info_encoders):
        ai_j = add_info_tensor[..., j].view(-1, 1)      # [B*S*I, 1]
        ai_enc_flat = encoder(ai_j)                     # [B*S*I, base_dim]
        ai_enc = ai_enc_flat.view(B, S, I, -1)            # [B, S, I, base_dim]
        add_enc_sum = add_enc_sum + ai_enc

    # --- 4. 요소별 결합 ---
    # base_embs, ts_enc, add_enc_sum 모두 [B, S, I, base_dim]
    combined = base_embs + 0.1 * ts_enc + 0.1 * add_enc_sum
    combined = combined * valid_mask.unsqueeze(-1).float()  # padding 위치 0으로 처리

    # --- 5. Projection FFN 적용 ---
    combined_flat = combined.view(-1, base_dim)     # [B*S*I, base_dim]
    projected_flat = projection_ffn(combined_flat)    # [B*S*I, emb_dim]
    projected = projected_flat.view(B, S, I, -1)       # [B, S, I, emb_dim]
    
    return projected


#########################################
# 5. 단일 진입점 함수: get_embeddings
#########################################
def get_embeddings(batch_dict, use_llm,
                   tokenizer=None, llm_model=None,
                   item_embeddings_tensor=None,
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
                                         item_embeddings_tensor,
                                         projection_ffn,
                                         timestamp_encoder,
                                         add_info_encoder,
                                         valid_mask)
    return emb_output



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