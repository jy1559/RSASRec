# loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def pad_and_stack(list_of_tensors, pad_value=0.0):
    """
    길이가 다른 2D 텐서 리스트(list of [L_i, D])를 동일한 길이(L_max)로 padding하여 [B, L_max, D] 텐서로 만듭니다.
    """
    device = list_of_tensors[0].device
    D = list_of_tensors[0].shape[1]
    lengths = [t.shape[0] for t in list_of_tensors]
    L_max = max(lengths)
    B = len(list_of_tensors)
    out = torch.full((B, L_max, D), pad_value, device=device)
    for i, t in enumerate(list_of_tensors):
        L = t.shape[0]
        out[i, :L, :] = t
    return out, torch.tensor(lengths, device=device)

def compute_seqrec_metrics(logits_flat, targets_flat, session_ids=None, valid_mask=None, k=[1, 3, 5, 10], chunk_size=1024):
    if valid_mask is None:
        valid_mask = (targets_flat != -1)

    valid_logits = logits_flat[valid_mask]        # [N_valid, C]
    valid_targets = targets_flat[valid_mask]      # [N_valid]
    valid_session_ids = session_ids[valid_mask]       # [N_valid]

    if valid_targets.numel() == 0:
        return {"accuracy": 0.0, "MRR": 0.0, "HitRate@1": 0.0, "NDCG@1": 0.0}

    N, C = valid_logits.shape
    device = valid_logits.device

    # 각 row마다 target logit만 뽑기
    target_scores = valid_logits[torch.arange(N), valid_targets]  # [N]

    # Accuracy
    preds = torch.argmax(valid_logits, dim=1)
    accuracy = (preds == valid_targets).float().mean().item()

    # 메모리 절약 rank 계산
    rank = torch.zeros(N, device=device, dtype=torch.long)
    for i in range(0, C, chunk_size):
        chunk = valid_logits[:, i:i+chunk_size]  # [N, chunk]
        # target_scores: [N, 1] vs chunk: [N, chunk]
        rank += (chunk > target_scores.unsqueeze(1)).sum(dim=1)

    # MRR
    mrr = (1.0 / (rank.float() + 1)).mean().item()

    hitrate, ndcg_k = {}, {}
    for k_val in k:
        hit_at_k = (rank < k_val).float().mean().item()
        dcg = torch.where(rank < k_val, 1.0 / torch.log2(rank.float() + 2), torch.zeros_like(rank.float()))
        ndcg = dcg.mean().item()
        hitrate[k_val] = hit_at_k
        ndcg_k[k_val] = ndcg

    metrics = {
        "accuracy": accuracy,
        "MRR": mrr,
    }
    for k_val in k:
        metrics[f"HitRate@{k_val}"] = hitrate[k_val]
        metrics[f"NDCG@{k_val}"] = ndcg_k[k_val]

    # Compute NSU per session
    # Assumes that within each session, the order in the flattened tensors is the ideal order.
    unique_sessions = torch.unique(valid_session_ids)
    nsu_list = []
    for s in unique_sessions:
        # Indices for session s
        sess_idx = (valid_session_ids == s).nonzero(as_tuple=True)[0]
        if sess_idx.numel() == 0:
            continue
        # For session s, get predicted ranks
        sess_ranks = rank[sess_idx]  # [m]
        m = sess_ranks.numel()
        # Create ideal order indices: 1,2,...,m
        ideal_order = torch.arange(1, m+1, device=device).float()
        # Ideal weights: 1 / log2(j+1)
        ideal_weights = 1.0 / torch.log2(ideal_order + 1)
        # Predicted discount: 1 / log2(r_j+2)
        predicted_discount = 1.0 / torch.log2(sess_ranks.float() + 2)
        # Session utility: sum_{j=1}^m w_j * predicted_discount_j
        U_actual = torch.sum(ideal_weights * predicted_discount)
        # Ideal utility: sum_{j=1}^m (w_j)^2
        U_ideal = torch.sum(ideal_weights ** 2)
        nsu_s = U_actual / U_ideal if U_ideal > 0 else 0.0
        nsu_list.append(nsu_s.item())

    NSU_metric = np.mean(nsu_list) if nsu_list else 0.0

    # Combine all metrics in a dictionary
    metrics = {
        "accuracy": accuracy,
        "MRR": mrr,
        "NSU": NSU_metric,
    }
    for k_val in k:
        metrics[f"HitRate@{k_val}"] = hitrate[k_val]
        metrics[f"NDCG@{k_val}"] = ndcg_k[k_val]

    return metrics


def compute_loss(target_features, candidate_set, correct_indices, 
                 strategy='EachSession_LastInter', global_candidate=False, 
                 loss_type='cross_entropy', similarity_type='cosine', custom_loss=None):
    """
    target_features: 
       - 각 전략에 따라 shape이 달라집니다.
         * EachSession_LastInter: [B, S, D]
         * Global_LastInter: [B, D]
         * LastSession_AllInter: 원래는 variable-length list, but 여기서는 [B, L, D] (zero padding 적용)
    candidate_set:
       - global_candidate == True: [num_candidates, D] (전역 candidate set)
       - global_candidate == False: 
             * For EachSession_LastInter: [B, S, candidate_size, D]
             * For LastSession_AllInter: [B, candidate_size, D] (예: per-sample candidate set)
    correct_indices:
       - 각 샘플에 대해 정답(positive)의 인덱스.
         * For EachSession_LastInter: if target_features is [B, S, D] then correct_indices should be [B, S]
         * For Global_LastInter: [B]
         * For LastSession_AllInter: [B] (candidate set per sample)
         - padding된 항목은 -1로 표시되어 있어야 합니다.
    
    이 함수는 지정한 similarity 계산 방식(기본: cosine)으로 각 candidate와 target feature 간의 유사도를 구하고,
    CrossEntropyLoss를 계산합니다.
    
    loss_type: 'cross_entropy' (기본)
    similarity_type: 'cosine' 또는 'dot'
    custom_loss: 만약 사용자 정의 loss function이 있으면 사용
    
    Returns:
       loss: scalar tensor (loss 값)
    """
    # similarity 계산 함수
    def compute_logits(target, cand):
        # target: [..., D], cand: [..., D]
        if similarity_type == 'cosine':
            target_norm = F.normalize(target, p=2, dim=-1, eps=1e-8)
            cand_norm = F.normalize(cand, p=2, dim=-1, eps=1e-8)
            return torch.sum(target_norm * cand_norm, dim=-1)
        elif similarity_type == 'dot':
            return torch.sum(target * cand, dim=-1)
        else:
            raise ValueError(f"Unknown similarity type: {similarity_type}")
    
    # global candidate인 경우, candidate_set: [num_candidates, D]
    # 각 target feature에 대해 같은 candidate_set 사용
    if global_candidate:
        C = candidate_set.shape[0]
        chunk_size = 1024

        if target_features.dim() == 3:
            B, S, D = target_features.shape
            target_exp = target_features.unsqueeze(2)  # [B, S, 1, D]
            logits_chunks = []

            for i in range(0, C, chunk_size):
                cand_chunk = candidate_set[i:i+chunk_size]  # [chunk_size, D]
                cand_exp = cand_chunk.unsqueeze(0).unsqueeze(0)  # [1, 1, chunk_size, D]
                logits_chunk = compute_logits(target_exp, cand_exp)  # [B, S, chunk_size]
                logits_chunks.append(logits_chunk)

            logits = torch.cat(logits_chunks, dim=-1)  # [B, S, C]
            logits_flat = logits.view(B * S, C)
            targets_flat = correct_indices.view(B * S)
            session_ids = torch.arange(B, device=target_features.device).unsqueeze(1).expand(B, S).reshape(-1)

        elif target_features.dim() == 2:
            B, D = target_features.shape
            target_exp = target_features.unsqueeze(1)  # [B, 1, D]
            logits_chunks = []

            for i in range(0, C, chunk_size):
                cand_chunk = candidate_set[i:i+chunk_size]  # [chunk_size, D]
                cand_exp = cand_chunk.unsqueeze(0)  # [1, chunk_size, D]
                logits_chunk = compute_logits(target_exp, cand_exp)  # [B, chunk_size]
                logits_chunks.append(logits_chunk)

            logits = torch.cat(logits_chunks, dim=-1)  # [B, C]
            logits_flat = logits
            targets_flat = correct_indices
            session_ids = torch.arange(B, device=target_features.device)

        else:
            raise ValueError("target_features의 차원은 2 또는 3이어야 합니다.")
    else:
        # Per-sample candidate set: candidate_set shape: 
        # For EachSession_LastInter: [B, S, candidate_size, D]
        # For LastSession_AllInter: [B, candidate_size, D]
        if target_features.dim() == 3:
            # Assume EachSession_LastInter case: target_features: [B, S, D]
            B, S, D = target_features.shape
            x = -1 if S <= 3 else 3
            """print(f"target_features shape: {target_features.shape}")
            print(f"candidate_set shape: {candidate_set.shape}")
            
            print(f"target_features: {target_features[0, :x, :5]}")
            print(f"candidate_set: {candidate_set[0, :, :x, :5]}")"""
            _, _, candidate_size, _ = candidate_set.shape
            # Compute logits for each sample and session: [B, S, candidate_size]
            logits = compute_logits(target_features.unsqueeze(2).expand(-1, -1, candidate_size, -1), candidate_set)
            logits_flat = logits.view(B * S, candidate_size)
            targets_flat = correct_indices.view(B * S)
            session_ids = torch.arange(B, device=target_features.device).unsqueeze(1).expand(B, S).reshape(-1)

        elif target_features.dim() == 2:
            # LastSession_AllInter case: target_features: [B, D], candidate_set: [B, candidate_size, D]
            B, D = target_features.shape
            candidate_size = candidate_set.shape[1]
            logits = compute_logits(target_features.unsqueeze(1).expand(-1, candidate_size, -1), candidate_set)
            logits_flat = logits
            targets_flat = correct_indices
            session_ids = torch.arange(B, device=target_features.device)

        else:
            raise ValueError("target_features의 차원은 2 또는 3이어야 합니다.")
    
    # loss 계산 시, correct_indices가 -1인 부분은 무시합니다.
    mask = (targets_flat != -1)
    if custom_loss is not None:
        if mask.sum() == 0:
            return torch.tensor(0.0, device=logits_flat.device)
        loss = custom_loss(logits_flat[mask], targets_flat[mask])
    else:
        loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        loss = loss_fn(logits_flat, targets_flat)
    metrics = compute_seqrec_metrics(logits_flat, targets_flat, session_ids, mask)
    return loss, metrics

if __name__ == '__main__':
    # 간단한 테스트: 각 경우에 대해 임의의 텐서를 만들어 계산
    B, S, D = 2, 3, 384
    candidate_size = 5
    # 예: strategy EachSession_LastInter, global_candidate=True
    target_features = torch.randn(B, S, D)
    candidate_set = torch.randn(100, D)  # global candidate: 100개 item
    correct_indices = torch.randint(0, 100, (B, S))
    loss = compute_loss(target_features, candidate_set, correct_indices, 
                        strategy='EachSession_LastInter', global_candidate=True)
    print("Loss (global_candidate=True, EachSession_LastInter):", loss.item())
    
    # 예: strategy LastSession_AllInter, global_candidate=False
    target_features = torch.randn(B, D)
    candidate_set = torch.randn(B, candidate_size, D)
    correct_indices = torch.randint(0, candidate_size, (B,))
    loss = compute_loss(target_features, candidate_set, correct_indices, 
                        strategy='LastSession_AllInter', global_candidate=False)
    print("Loss (global_candidate=False, LastSession_AllInter):", loss.item())
