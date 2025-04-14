import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

def check_nan(tensor):
    # tensor 내 NaN의 개수 구하기
    nan_count = torch.isnan(tensor).sum().item()
    total_count = tensor.numel()
    nan_ratio = 100.0 * nan_count / total_count
    print(f"Total elements: {total_count}, NaN elements: {nan_count} ({nan_ratio:.2f}%)")
    return nan_count > 0

def pad_and_stack(list_of_tensors, pad_value=0.0):
    """
    길이가 다른 2D 텐서 목록을 패딩하여 [B, L_max, D] 텐서로 반환합니다.
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

def _compute_global_loss(valid_target, valid_targets, candidate_set, similarity_type, chunk_size):
    """
    Global candidate 모드에서, valid_target ([N_valid, d])와 candidate_set ([C, d])
    간의 유사도를 청크 단위로 계산하여, 각 valid target에 대해 전체 후보의 log-sum-exp와 
    정답(candidate) logit을 구합니다.
    """
    device = valid_target.device
    N_valid = valid_target.shape[0]
    C = candidate_set.shape[0]
    total_logsum = torch.full((N_valid,), float('-inf'), device=device)
    correct_logit = torch.full((N_valid,), float('-inf'), device=device)
    
    for start in range(0, C, chunk_size):
        end = min(C, start + chunk_size)
        chunk_candidates = candidate_set[start:end]  # [chunk, d]
        if similarity_type == 'cosine':
            tgt_norm = F.normalize(valid_target, p=2, dim=-1, eps=1e-8)
            chunk_norm = F.normalize(chunk_candidates, p=2, dim=-1, eps=1e-8)
            logits_chunk = torch.matmul(tgt_norm, chunk_norm.t())  # [N_valid, chunk]
        elif similarity_type == 'dot':
            logits_chunk = torch.matmul(valid_target, chunk_candidates.t())  # [N_valid, chunk]
        else:
            raise ValueError(f"Unknown similarity type: {similarity_type}")
            
        chunk_lse = torch.logsumexp(logits_chunk, dim=1)  # [N_valid]
        total_logsum = torch.logaddexp(total_logsum, chunk_lse)
        
        in_chunk = (valid_targets >= start) & (valid_targets < end)
        if in_chunk.any():
            rel_idx = valid_targets[in_chunk] - start  # relative index within chunk
            correct_logits_chunk = logits_chunk[in_chunk, rel_idx]
            correct_logit[in_chunk] = correct_logits_chunk
    return total_logsum, correct_logit

def _compute_global_rank(valid_target, candidate_set, similarity_type, chunk_size, correct_logit):
    """
    Global candidate 모드에서, valid_target ([N_valid, d])와 candidate_set ([C, d])
    간의 유사도를 청크 단위로 계산하여, 각 valid target에 대해 정답 후보보다 높은 후보 수를 누적합니다.
    """
    device = valid_target.device
    N_valid = valid_target.shape[0]
    C = candidate_set.shape[0]
    rank_count = torch.zeros(N_valid, device=device, dtype=torch.long)
    
    for start in range(0, C, chunk_size):
        end = min(C, start + chunk_size)
        chunk_candidates = candidate_set[start:end]
        if similarity_type == 'cosine':
            tgt_norm = F.normalize(valid_target, p=2, dim=-1, eps=1e-8)
            chunk_norm = F.normalize(chunk_candidates, p=2, dim=-1, eps=1e-8)
            logits_chunk = torch.matmul(tgt_norm, chunk_norm.t())
        elif similarity_type == 'dot':
            logits_chunk = torch.matmul(valid_target, chunk_candidates.t())
        else:
            raise ValueError(f"Unknown similarity type: {similarity_type}")
            
        greater = (logits_chunk > correct_logit.unsqueeze(1)).long()  # [N_valid, chunk]
        rank_count += greater.sum(dim=1)
    return rank_count

def compute_loss_and_metrics_global(target_features, candidate_set, correct_indices,
                                    loss_mask, session_ids, similarity_type='cosine', 
                                    k=[1,3,5,10], chunk_size=1024):
    """
    Global candidate 모드에서, global candidate set ([C, d])에 대해,
    청크 단위 연산을 사용하여 loss와 다양한 metric을 계산합니다.
    
    Args:
      target_features: [B, L, d] tensor
      candidate_set: [C, d] tensor (global candidate set)
      correct_indices: [B, L] tensor, 각 유효 위치에서 정답 인덱스 = (item_id - 1)
      loss_mask: [B, L] tensor, valid=1, padding=0
      session_ids: [B, L] tensor
      similarity_type: 'cosine' 또는 'dot'
      k: list for HitRate@k and NDCG@k
      chunk_size: int, 청크 사이즈
      
    Returns:
      loss: scalar tensor (평균 cross entropy loss)
      metrics: dict containing "accuracy", "MRR", "HitRate@k", "NDCG@k", "SRA", "WSRA"
    """
    device = target_features.device
    B, L, d = target_features.shape
    total_entries = B * L
    target_flat = target_features.view(total_entries, d)        # [B*L, d]
    targets_flat = correct_indices.view(total_entries)          # [B*L]
    loss_mask_flat = loss_mask.view(total_entries).bool()         # [B*L]
    session_ids_flat = session_ids.view(total_entries)           # [B*L]
    
    valid_target = target_flat[loss_mask_flat]                   # [N_valid, d]
    valid_targets = targets_flat[loss_mask_flat].long()          # [N_valid]
    valid_sessions = session_ids_flat[loss_mask_flat]            # [N_valid]
    N_valid = valid_target.shape[0]
    if N_valid == 0:
        return torch.tensor(0.0, device=device), {}
    
    total_logsum, correct_logit = _compute_global_loss(valid_target, valid_targets, candidate_set, similarity_type, chunk_size)
    loss_per_sample = -(correct_logit - total_logsum)
    loss = loss_per_sample.mean()
    
    rank_count = _compute_global_rank(valid_target, candidate_set, similarity_type, chunk_size, correct_logit)
    accuracy = (rank_count == 0).float().mean().item()
    mrr = (1.0 / (rank_count.float() + 1)).mean().item()
    
    hitrate = {}
    ndcg_k = {}
    for k_val in k:
        hit_at_k = (rank_count < k_val).float().mean().item()
        dcg = torch.where(
            rank_count < k_val,
            1.0 / torch.log2(rank_count.float() + 2),
            torch.zeros_like(rank_count, dtype=torch.float)
        )
        ndcg = dcg.mean().item()
        hitrate[k_val] = hit_at_k
        ndcg_k[k_val] = ndcg

    pos_in_session = torch.zeros(N_valid, device=device, dtype=torch.long)
    sess_counter = defaultdict(int)
    for i, s in enumerate(valid_sessions.tolist()):
        pos_in_session[i] = sess_counter[s]
        sess_counter[s] += 1
    sess_dict = defaultdict(list)
    for i in range(N_valid):
        s_id = valid_sessions[i].item()
        p = pos_in_session[i].item()
        r_val = rank_count[i].item()
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
        
    return loss, metrics

def compute_seqrec_metrics(logits_flat, targets_flat, session_ids, valid_mask, k=[1,3,5,10], chunk_size=1024):
    """
    Non-global candidate 모드에서, flatten 된 logits를 가지고 metric을 계산합니다.
    
    Args:
      logits_flat: [N, C] tensor, N = B*L
      targets_flat: [N] tensor, 정답 후보 인덱스 (padding은 -1)
      session_ids: [N] tensor, 세션 id
      valid_mask: [N] bool tensor, valid 위치의 mask
      k: list, HitRate@k, NDCG@k 계산용
      chunk_size: int, rank 계산 시 사용
    Returns:
      metrics: dict containing "accuracy", "MRR", "HitRate@k", "NDCG@k", "SRA", "WSRA"
    """
    device = logits_flat.device
    N, C = logits_flat.shape

    if valid_mask is None:
        valid_mask = (targets_flat != -1)

    valid_logits = logits_flat[valid_mask]
    valid_targets = targets_flat[valid_mask]
    valid_session_ids = session_ids[valid_mask]
    
    pos_in_session = torch.zeros(valid_session_ids.shape, device=device, dtype=torch.long)
    sess_counter = defaultdict(int)
    for i, s_id in enumerate(valid_session_ids.tolist()):
        pos_in_session[i] = sess_counter[s_id]
        sess_counter[s_id] += 1

    if valid_targets.numel() == 0:
        base_metrics = {"accuracy": 0.0, "MRR": 0.0}
        hit_ndcg = {f"HitRate@{k_val}": 0.0 for k_val in k}
        hit_ndcg.update({f"NDCG@{k_val}": 0.0 for k_val in k})
        return {**base_metrics, **hit_ndcg, "SRA": 0.0, "WSRA": 0.0}

    N_valid = valid_logits.shape[0]
    preds = torch.argmax(valid_logits, dim=1)
    accuracy = (preds == valid_targets).float().mean().item()

    target_scores = valid_logits[torch.arange(N_valid, device=device), valid_targets]
    rank = torch.zeros(N_valid, device=device, dtype=torch.long)
    for start in range(0, C, chunk_size):
        chunk = valid_logits[:, start:start+chunk_size]
        rank += (chunk > target_scores.unsqueeze(1)).sum(dim=1)
    mrr = (1.0 / (rank.float() + 1)).mean().item()

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

def compute_loss_and_metrics(target_features, candidate_set, correct_indices,
                             strategy=None, global_candidate=False,
                             loss_mask=None, session_ids=None,
                             similarity_type='cosine', k=[1,3,5,10], chunk_size=1024):
    """
    Top-level unified function for computing loss and metrics.
    For non-global candidate mode, it uses the nested loops approach.
    For global candidate mode, it calls compute_loss_and_metrics_global.
    """
    #print(f'target_features.shape: {target_features.shape}')
    ###rint(f'candidate_set.shape: {candidate_set.shape}')
    #print(f'candidate_set.example(): {candidate_set[0, 0, 0, :20]}')
    #check_nan(target_features)
    #check_nan(candidate_set)
    if not global_candidate:
        B, L, d = target_features.shape
        candidate_size = candidate_set.shape[2]
        logits = torch.zeros((B, L, candidate_size), device=target_features.device)
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
        logits_flat = logits.view(-1, candidate_size)
        targets_flat = correct_indices.view(-1)
        loss_mask_flat = loss_mask.view(-1).bool()
        valid_logits = logits_flat[loss_mask_flat]
        valid_targets = targets_flat[loss_mask_flat]
        if valid_logits.shape[0] == 0:
            loss = torch.tensor(0.0, device=target_features.device)
        else:
            loss = F.cross_entropy(valid_logits, valid_targets)
        metrics = compute_seqrec_metrics(logits_flat, targets_flat, session_ids.view(-1), loss_mask_flat, k=k, chunk_size=chunk_size)
        return loss, metrics
    else:
        return compute_loss_and_metrics_global(target_features, candidate_set, correct_indices, loss_mask, session_ids, similarity_type, k, chunk_size)
