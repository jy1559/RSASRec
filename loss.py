# loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

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

def compute_seqrec_metrics(logits_flat, targets_flat, valid_mask=None, k=[1, 3, 5]):
    """
    logits_flat: [N, candidate_size] 텐서, 각 row는 후보에 대한 점수
    targets_flat: [N] 텐서, 각 원소는 정답 후보의 인덱스 (패딩된 경우 -1)
    valid_mask: optional, [N] boolean 텐서. 제공하지 않으면 (targets_flat != -1)로 설정.
    k: HitRate와 NDCG를 계산할 때 사용할 cutoff
    
    반환:
      metrics: dict, {"accuracy": ..., "MRR": ..., "HitRate@k": ..., "NDCG@k": ...}
    """
    if valid_mask is None:
        valid_mask = (targets_flat != -1)
    
    # 유효 샘플만 필터링
    valid_logits = logits_flat[valid_mask]
    valid_targets = targets_flat[valid_mask]
    
    if valid_targets.numel() == 0:
        return {"accuracy": 0.0, "MRR": 0.0, f"HitRate": 0.0, f"NDCG": 0.0}
    
    # Accuracy: argmax한 값이 정답과 일치하는지
    preds = torch.argmax(valid_logits, dim=-1)
    accuracy = (preds == valid_targets).float().mean().item()
    
    # 순위 계산: 각 샘플에 대해, 정답의 rank (0-indexed)
    # argsort 내림차순 -> 각 row에 대해 정답이 몇 번째에 있는지 계산
    sorted_indices = torch.argsort(valid_logits, dim=-1, descending=True)  # [N_valid, candidate_size]
    # 각 row에 대해, 정답이 있는 위치(0-indexed)를 찾습니다.
    eq = (sorted_indices == valid_targets.unsqueeze(1))  # [N_valid, candidate_size]
    ranks = torch.argmax(eq.to(torch.int64), dim=1)  # [N_valid]
    
    # MRR: 평균 reciprocal rank
    mrr = (1.0 / (ranks.to(torch.float32) + 1)).mean().item()
    
    hitrate, ndcg_k = {}, {}
    for k_val in k:

        hit_at_k = (ranks < k_val).float().mean().item()
        dcg = torch.where(ranks < k_val, 1.0 / torch.log2(ranks.to(torch.float32) + 2), torch.zeros_like(ranks.to(torch.float32)))
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
        # global_candidate: candidate_set shape [C, D]
        # target_features may be [B, S, D] or [B, D]
        if target_features.dim() == 3:
            B, S, D = target_features.shape
            C = candidate_set.shape[0]
            # Expand candidate_set: [1, 1, C, D]
            cand_exp = candidate_set.unsqueeze(0).unsqueeze(0)
            # target_features: [B, S, 1, D]
            target_exp = target_features.unsqueeze(2)
            # Compute logits: [B, S, C])
            logits = compute_logits(target_exp, cand_exp)
            # correct_indices should be [B, S]
            # Flatten: [B*S, C] and target: [B*S]
            logits_flat = logits.view(B * S, C)
            targets_flat = correct_indices.view(B * S)
        elif target_features.dim() == 2:
            # target_features: [B, D]
            B, D = target_features.shape
            C = candidate_set.shape[0]
            # Expand candidate_set: [1, C, D]
            cand_exp = candidate_set.unsqueeze(0)
            # target_features: [B, 1, D]
            target_exp = target_features.unsqueeze(1)
            logits = compute_logits(target_exp, cand_exp)  # [B, C]
            logits_flat = logits
            targets_flat = correct_indices
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
        elif target_features.dim() == 2:
            # LastSession_AllInter case: target_features: [B, D], candidate_set: [B, candidate_size, D]
            B, D = target_features.shape
            candidate_size = candidate_set.shape[1]
            logits = compute_logits(target_features.unsqueeze(1).expand(-1, candidate_size, -1), candidate_set)
            logits_flat = logits
            targets_flat = correct_indices
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
    metrics = compute_seqrec_metrics(logits_flat, targets_flat, mask)
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
