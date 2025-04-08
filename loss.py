# loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
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


def compute_seqrec_metrics(logits_flat, targets_flat, session_ids, valid_mask, k=[1, 3, 5, 10], chunk_size=1024):
    """
    Args:
      logits_flat:  [N, C] tensor, N = B*L, 각 valid target에 대한 candidate 점수
      targets_flat: [N] tensor, 각 위치의 정답 candidate 인덱스
      session_ids:  [N] tensor, 각 위치의 세션 식별자
      valid_mask:   [N] bool tensor, valid 위치 mask
      k: list, HitRate@k, NDCG@k 계산용
      chunk_size: int, rank 계산 시 사용
      
    Returns:
      metrics: dict containing "accuracy", "MRR", "HitRate@k", "NDCG@k", "SRA", "WSRA"
    """
    device = logits_flat.device
    N, C = logits_flat.shape

    if valid_mask is None:
        valid_mask = (targets_flat != -1)

    valid_logits = logits_flat[valid_mask]       # [N_valid, C]
    valid_targets = targets_flat[valid_mask]       # [N_valid]
    valid_session_ids = session_ids[valid_mask]     # [N_valid]
    
    # 1. pos_in_session 계산: 각 valid target에 대해, 동일한 session 내에서 등장 순서를 결정
    pos_in_session = torch.zeros(valid_session_ids.shape, device=device, dtype=torch.long)
    sess_counter = defaultdict(int)
    valid_indices = valid_session_ids.tolist()
    for i, s_id in enumerate(valid_indices):
        pos_in_session[i] = sess_counter[s_id]
        sess_counter[s_id] += 1

    if valid_targets.numel() == 0:
        base_metrics = {"accuracy": 0.0, "MRR": 0.0}
        hit_ndcg = {f"HitRate@{k_val}": 0.0 for k_val in k}
        hit_ndcg.update({f"NDCG@{k_val}": 0.0 for k_val in k})
        return {**base_metrics, **hit_ndcg, "SRA": 0.0, "WSRA": 0.0}

    N_valid = valid_logits.shape[0]

    # Accuracy
    preds = torch.argmax(valid_logits, dim=1)
    accuracy = (preds == valid_targets).float().mean().item()

    # Rank 계산 (0-based)
    target_scores = valid_logits[torch.arange(N_valid), valid_targets]  # [N_valid]
    rank = torch.zeros(N_valid, device=device, dtype=torch.long)
    for start in range(0, C, chunk_size):
        chunk = valid_logits[:, start:start+chunk_size]
        rank += (chunk > target_scores.unsqueeze(1)).sum(dim=1)
    mrr = (1.0 / (rank.float() + 1)).mean().item()

    # HitRate@k, NDCG@k
    hitrate = {}
    ndcg_k = {}
    for k_val in k:
        hit_at_k = (rank < k_val).float().mean().item()
        dcg = torch.where(
            rank < k_val,
            1.0 / torch.log2(rank.float() + 2),
            torch.zeros_like(rank, dtype=torch.float)
        )
        ndcg = dcg.mean().item()
        hitrate[k_val] = hit_at_k
        ndcg_k[k_val] = ndcg

    # SRA 및 WSRA 계산
    sess_dict = defaultdict(list)
    for i in range(N_valid):
        s_id = valid_session_ids[i].item()
        p = pos_in_session[i].item()
        r_val = rank[i].item()
        sess_dict[s_id].append((p, r_val))
    
    sra_list = []
    wsra_list = []
    for s_id, items in sess_dict.items():
        items_sorted = sorted(items, key=lambda x: x[0])
        m = len(items_sorted)
        if m < 2:
            continue
        sum_s = 0.0
        weighted_s = 0.0
        sum_weights = 0.0
        for idx in range(m - 1):
            _, r1 = items_sorted[idx]
            _, r2 = items_sorted[idx + 1]
            S_val = 1.0 if (r1 < r2) else 0.0
            sum_s += S_val
            w = 1.0 / np.log2(idx + 2)
            weighted_s += w * S_val
            sum_weights += w
        sra_list.append(sum_s / (m - 1))
        wsra_list.append(weighted_s / sum_weights if sum_weights > 0 else 0.0)
    
    SRA_metric = float(np.mean(sra_list)) if len(sra_list) > 0 else 0.0
    WSRA_metric = float(np.mean(wsra_list)) if len(wsra_list) > 0 else 0.0

    metrics = {
        "accuracy": accuracy,
        "MRR": mrr,
        "SRA": SRA_metric,
        "WSRA": WSRA_metric,
    }
    for k_val in k:
        metrics[f"HitRate@{k_val}"] = hitrate[k_val]
        metrics[f"NDCG@{k_val}"] = ndcg_k[k_val]

    return metrics

def compute_loss(target_features, candidate_set, correct_indices,
                 strategy=None, global_candidate=False,
                 loss_mask=None, session_ids=None,
                 similarity_type='cosine', k=[1, 3, 5, 10], chunk_size=1024):
    """
    Args:
      target_features: [B, L, d] tensor (forward에서 추출된 최종 target feature)
      candidate_set:
         - if global_candidate is False: [B, L, candidate_size, d] tensor,
         - else: [num_candidates, d] tensor (global candidate set)
      correct_indices: [B, L] tensor, 각 위치에 대해 정답 candidate 인덱스 (padding은 -1)
      loss_mask: [B, L] tensor, valid 위치는 1, 패딩은 0
      session_ids: [B, L] tensor, 각 위치의 세션 식별자 (get_batch_item_ids에서 리턴)
      similarity_type: 'cosine' 또는 'dot'
      k: list, HitRate@k, NDCG@k 계산용
      chunk_size: int, rank 계산 시 chunk size
      
    Returns:
      loss: scalar tensor (valid 위치에 대해 cross entropy loss 계산)
      metrics: dict, {"accuracy", "MRR", "HitRate@k", "NDCG@k", "SRA", "WSRA"}
    """

    B, L, d = target_features.shape
    #print(f"target_features: {target_features.shape}, candidate_set: {candidate_set.shape}, correct_indices: {correct_indices.shape}, loss_mask: {loss_mask.shape}, session_ids: {session_ids.shape}")
    # 1. candidate set에 따른 logits 계산
    if not global_candidate:
        candidate_size = candidate_set.shape[2]
        logits = torch.zeros(B, L, candidate_size, device=target_features.device)
        for b in range(B):
            for l in range(L):
                if loss_mask[b, l] == 1:
                    if similarity_type == 'cosine':
                        tgt = F.normalize(target_features[b, l], p=2, dim=-1, eps=1e-8)
                        cand = F.normalize(candidate_set[b, l], p=2, dim=-1, eps=1e-8)
                        logits[b, l] = torch.sum(tgt * cand, dim=-1)
                    elif similarity_type == 'dot':
                        logits[b, l] = torch.sum(target_features[b, l] * candidate_set[b, l], dim=-1)
                    else:
                        raise ValueError(f"Unknown similarity type: {similarity_type}")
    else:
        # Global candidate: candidate_set shape is [C, d]
        candidate_size = candidate_set.shape[0]
        cand_exp = candidate_set.unsqueeze(0).unsqueeze(0)  # [1, 1, C, d]
        tgt_exp = target_features.unsqueeze(2)  # [B, L, 1, d]
        if similarity_type == 'cosine':
            tgt_norm = F.normalize(tgt_exp, p=2, dim=-1, eps=1e-8)
            cand_norm = F.normalize(cand_exp, p=2, dim=-1, eps=1e-8)
            logits = torch.sum(tgt_norm * cand_norm, dim=-1)  # [B, L, C]
        elif similarity_type == 'dot':
            logits = torch.sum(tgt_exp * cand_exp, dim=-1)
        else:
            raise ValueError(f"Unknown similarity type: {similarity_type}")
    
    # 2. Flatten 및 valid 영역 선택해서 loss 계산
    logits_flat = logits.view(-1, candidate_size)       # [B*L, candidate_size]
    correct_flat = correct_indices.view(-1)             # [B*L]
    loss_mask_flat = loss_mask.view(-1).bool()            # [B*L]
    
    valid_logits = logits_flat[loss_mask_flat]            # [N_valid, candidate_size]
    valid_targets = correct_flat[loss_mask_flat]          # [N_valid]
    
    if valid_logits.shape[0] == 0:
        loss = torch.tensor(0.0, device=target_features.device)
    else:
        loss = F.cross_entropy(valid_logits, valid_targets)
    
    # 3. Metrics 계산: compute_seqrec_metrics 내부에서 pos_in_session을 계산
    metrics = compute_seqrec_metrics(
        logits_flat,
        correct_flat,
        session_ids.view(-1),
        loss_mask_flat,
        k=k,
        chunk_size=chunk_size
    )
    
    return loss, metrics