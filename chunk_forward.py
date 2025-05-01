import torch
import torch.nn as nn
import os
import argparse
import numpy as np
import torchviz
from torch.amp import autocast
from contextlib import nullcontext
import wandb
from tqdm.auto import tqdm
from loss import compute_loss_and_metrics


def select_target_features_flat_chunk(chunk_raw_output, valid_mask_all, strategy, chunk_position=None):
    """
    Args:
      chunk_raw_output: torch.Tensor, shape [B, S_chunk, I, d], model forward의 원시 출력.
      valid_mask_all: torch.Tensor, shape [B, S_chunk, I] (bool), padding 아닌 부분은 True.
      strategy: string, 예: "EachSession_LastInter", "Global_LastInter", "AllInter_ExceptFirst"
      chunk_position: optional str, 기본값 None. (로그용 정보 정도로 활용)
      
    Returns:
      selected_output: torch.Tensor, shape [B, L_chunk, d] (loss 계산에 사용할 target features)
      selected_loss_mask: torch.Tensor, shape [B, L_chunk] (bool)
      selected_session_ids: torch.Tensor, shape [B, L_chunk] (int), 각 선택 토큰이 속한 session index
       
      만약 어떤 sample에서도 valid token이 하나도 선택되지 않으면 (L_chunk == 0) None, None, None을 반환합니다.
    """
    B, S_chunk, I, d = chunk_raw_output.shape
    
    selected_all = []  # 각 sample별 선택한 feature (list of lists)
    sess_id_all = []   # 각 sample별 선택한 session id (list of ints)
    
    for b in range(B):
        selected_b = []
        sess_ids_b = []
        if strategy == "EachSession_LastInter":
            for s in range(S_chunk):
                valid_idx = (valid_mask_all[b, s, :] == True).nonzero(as_tuple=False).squeeze(-1)
                if valid_idx.numel() > 0:
                    sel_idx = valid_idx[-1].item()
                    selected_b.append(chunk_raw_output[b, s, sel_idx, :])
                    sess_ids_b.append(s)
        elif strategy == "Global_LastInter":
            flat_valid = valid_mask_all[b].view(-1)
            valid_idx = (flat_valid == True).nonzero(as_tuple=False).squeeze(-1)
            if valid_idx.numel() > 0:
                sel_idx = valid_idx[-1].item()
                s_idx = sel_idx // I
                token_idx = sel_idx % I
                selected_b.append(chunk_raw_output[b, s_idx, token_idx, :])
                sess_ids_b.append(s_idx)
        elif strategy == "AllInter_ExceptFirst":
            flat_valid = valid_mask_all[b].view(-1)
            valid_idx = (flat_valid == True).nonzero(as_tuple=False).squeeze(-1)

            if chunk_position == "first" and valid_idx.numel() > 1:
                use_idxs = valid_idx[1:]
            else:
                use_idxs = valid_idx       # 이후 청크는 전부 선택
            
            for idx in use_idxs:
                    idx = idx.item()
                    s_idx     = idx // I
                    token_idx = idx % I
                    selected_b.append(chunk_raw_output[b, s_idx, token_idx, :])
                    sess_ids_b.append(s_idx)
        else:
            # 기본적으로 EachSession_LastInter 처리
            for s in range(S_chunk):
                valid_idx = (valid_mask_all[b, s, :] == True).nonzero(as_tuple=False).squeeze(-1)
                if valid_idx.numel() > 0:
                    sel_idx = valid_idx[-1].item()
                    selected_b.append(chunk_raw_output[b, s, sel_idx, :])
                    sess_ids_b.append(s)
        selected_all.append(selected_b)
        sess_id_all.append(sess_ids_b)
    
    max_len = max(len(lst) for lst in selected_all) if selected_all else 0
    if max_len == 0:
        return None, None, None

    padded_features = []
    padded_loss_mask = []
    padded_sess_ids = []
    for b in range(B):
        curr_len = len(selected_all[b])
        if curr_len < max_len:
            pad_size = max_len - curr_len
            feat_tensor = torch.stack(selected_all[b]) if curr_len > 0 else torch.empty((0, d), device=chunk_raw_output.device)
            pad_feat = torch.zeros(pad_size, d, device=chunk_raw_output.device)
            padded_features.append(torch.cat([feat_tensor, pad_feat], dim=0))
            mask_tensor = torch.ones(curr_len, dtype=torch.bool, device=chunk_raw_output.device)
            pad_mask = torch.zeros(pad_size, dtype=torch.bool, device=chunk_raw_output.device)
            padded_loss_mask.append(torch.cat([mask_tensor, pad_mask], dim=0))
            sess_tensor = torch.tensor(sess_id_all[b], dtype=torch.long, device=chunk_raw_output.device) if curr_len > 0 else torch.empty((0,), dtype=torch.long, device=chunk_raw_output.device)
            pad_sess = torch.full((pad_size,), -1, dtype=torch.long, device=chunk_raw_output.device)
            padded_sess_ids.append(torch.cat([sess_tensor, pad_sess], dim=0))
        else:
            padded_features.append(torch.stack(selected_all[b]))
            padded_loss_mask.append(torch.ones(max_len, dtype=torch.bool, device=chunk_raw_output.device))
            padded_sess_ids.append(torch.tensor(sess_id_all[b], dtype=torch.long, device=chunk_raw_output.device))
    
    selected_output = torch.stack(padded_features, dim=0)     # [B, L_chunk, d]
    selected_loss_mask = torch.stack(padded_loss_mask, dim=0)   # [B, L_chunk]
    selected_session_ids = torch.stack(padded_sess_ids, dim=0)    # [B, L_chunk]
    return selected_output, selected_loss_mask, selected_session_ids

def select_targets_and_candidates_chunk(chunk_raw_output, valid_mask_all, 
                                          full_candidate_set, full_correct_indices, 
                                          full_loss_mask, full_session_ids, 
                                          strategy, global_candidate, s_start, s_end, chunk_position=None):
    """
    Args:
      chunk_raw_output: torch.Tensor, shape [B, S_chunk, I, d] – 모델 forward의 원시 출력.
      valid_mask_all: torch.Tensor, shape [B, S_chunk, I] (bool).
      full_candidate_set: 
          if global_candidate is False: [B, S, candidate_size, d]
          else: [C, d]
      full_correct_indices:
          if global_candidate is False: [B, S]
          else: [B, L]
      full_loss_mask: torch.Tensor, shape [B, S] – get_batch_item_ids 결과.
      full_session_ids: torch.Tensor, shape [B, S] – get_batch_item_ids 결과.
      strategy: string, e.g., "EachSession_LastInter", "Global_LastInter", "AllInter_ExceptFirst"
      global_candidate: bool.
      s_start, s_end: int, 현재 청크에 해당하는 session index 범위 (전체 기준)
      chunk_position: optional str, 기본값 None (로그용)
      
    Returns:
      selected_output: torch.Tensor, [B, L_chunk, d]
      selected_loss_mask: torch.Tensor, [B, L_chunk]
      selected_session_ids: torch.Tensor, [B, L_chunk] – 여기서는 전체 session index (즉, s_start을 더한 값)
      candidate_chunk: 
           if global_candidate is False: [B, L_chunk, candidate_size, d]
           else: [C, d]
      candidate_correct:
           if global_candidate is False: [B, L_chunk]
           else: [B, L]
    """
    # 1. Target selection
    selected_output, selected_loss_mask, selected_session_ids_rel = select_target_features_flat_chunk(
        chunk_raw_output, valid_mask_all, strategy, chunk_position
    )
    # 원래의 전체 session index는 현재 청크의 시작(s_start) 값을 더해줍니다.
    if selected_session_ids_rel is None:
        return None, None, None, None, None
    selected_session_ids = selected_session_ids_rel + s_start  # [B, L_chunk]

    # 2. Candidate set extraction
    if global_candidate:
        candidate_chunk = full_candidate_set
        corr_list = []
        for b in range(full_correct_indices.size(0)):
            # 세션 범위 & loss_mask 둘 다 만족하는 위치만 취함
            mask = ((full_session_ids[b] >= s_start) &
                    (full_session_ids[b] <  s_end)  &
                     full_loss_mask[b])              # [L_full] bool
            corr_list.append(full_correct_indices[b, mask])  # [L_chunk_b]

        # 배치마다 길이가 다르므로 padding
        L_chunk = selected_output.shape[1]           # 패딩 목표 길이
        for i, t in enumerate(corr_list):
            if t.numel() < L_chunk:
                pad = torch.full((L_chunk - t.numel(),),
                                 -1, dtype=torch.long,
                                 device=full_correct_indices.device)
                corr_list[i] = torch.cat([t, pad], dim=0)
        candidate_correct = torch.stack(corr_list, dim=0)      # [B, L_chunk]
    else:
        # full_candidate_set: [B, S, candidate_size, d]; full_correct_indices: [B, S]
        B_local = full_candidate_set.shape[0]
        candidate_chunk_list = []
        candidate_correct_list = []
        for b in range(B_local):
            # 각 sample의 전체 candidate set에서, target session index에 해당하는 candidate만 선택.
            # selected_session_ids[b]는 [L_chunk] (전체 session index)
            tokens_sess = selected_session_ids[b]  # list or tensor of length L_chunk
            cand_list = []
            corr_list = []
            for s in tokens_sess.tolist():
                if s < 0 or s >= full_candidate_set.shape[1]:
                    cand_list.append(torch.zeros(full_candidate_set.shape[2], full_candidate_set.shape[3], device=full_candidate_set.device))
                    corr_list.append(-1)
                else:
                    cand_list.append(full_candidate_set[b, s, :, :])  # [candidate_size, d]
                    corr_list.append(full_correct_indices[b, s])
            cand_tensor = torch.stack(cand_list, dim=0)  # [L_chunk, candidate_size, d]
            corr_tensor = torch.tensor(corr_list, dtype=torch.long, device=full_candidate_set.device)  # [L_chunk]
            candidate_chunk_list.append(cand_tensor)
            candidate_correct_list.append(corr_tensor)
        candidate_chunk = torch.stack(candidate_chunk_list, dim=0)  # [B, L_chunk, candidate_size, d]
        candidate_correct = torch.stack(candidate_correct_list, dim=0)  # [B, L_chunk]

    return selected_output, selected_loss_mask, selected_session_ids, candidate_chunk, candidate_correct

def batch_by_chunk(batch, full_candidate_set, full_correct_indices, full_loss_mask, full_session_ids,
                   model, strategy, global_candidate, is_train, use_amp, scaler, accumulation_steps,
                   device, batch_th, wandb_logging, base_batch_counter):
    """
    TBPTT 방식을 적용하여, 대형 배치를 session 단위 청크로 나눠 처리합니다.
    각 청크마다 model forward를 수행하고, chunk 모드에서는 forward(..., chunk=True)로 raw output과 valid_mask_all을 얻습니다.
    그런 다음 helper 함수 select_targets_and_candidates_chunk를 호출하여,
    strategy에 맞게 target features와 candidate set까지 모두 처리한 결과를 반환받아 loss 계산 및 backward를 수행합니다.
    
    full_candidate_set, full_correct_indices, full_loss_mask, full_session_ids는 미리 no_grad()로 계산된 값을 사용합니다.
    """
    B, S, I = batch['item_id'].shape
    total_interactions = S * I
    if batch_th <= 0:
        num_chunks = 1
    else:
        num_chunks = int((total_interactions * 2) // batch_th) + 3
        num_chunks = min(num_chunks, S)
    
    # 세션 분할 경계 계산
    chunk_boundaries = []
    base = 0
    for idx in range(num_chunks):
        end = base + (S - base) // (num_chunks - idx)
        chunk_boundaries.append((base, end))
        base = end
    
    total_loss_sum = 0.0
    total_valid_count = 0
    metric_acc = {
        "accuracy": 0.0,
        "MRR": 0.0,
        "SRA": 0.0,
        "WSRA": 0.0,
        "HitRate@1": 0.0,
        "NDCG@1": 0.0,
        "HitRate@3": 0.0,
        "NDCG@3": 0.0,
        "HitRate@5": 0.0,
        "NDCG@5": 0.0,
        "HitRate@10": 0.0,
        "NDCG@10": 0.0,
    }
    prev_user_embedding = None
    for chunk_idx, (s_start, s_end) in enumerate(chunk_boundaries):
        current_batch_counter = base_batch_counter + chunk_idx

        # 각 key별로 session slicing (tensor와 list 대응)
        sub_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                if key in ['item_id', 'delta_ts', 'interaction_mask']:
                    sub_batch[key] = value[:, s_start:s_end, :]
                else:
                    sub_batch[key] = value[:, s_start:s_end]
            else:
                new_val = []
                for elem in value:
                    new_val.append(elem[s_start:s_end])
                sub_batch[key] = new_val

        context = autocast(device) if (use_amp and is_train and scaler is not None) else nullcontext()
        with context:
            chunk_raw_output, valid_mask_all, updated_user_embedding = model(sub_batch, prev_user_embedding, chunk=True)
        prev_user_embedding = updated_user_embedding.detach()
        
        # helper 함수를 호출해서 target features 및 candidate set 등 모두 처리
        chunk_position = 'last' if (chunk_idx == len(chunk_boundaries) - 1) else ('first' if (chunk_idx == 0) else None)
        helper_outputs = select_targets_and_candidates_chunk(
            chunk_raw_output, valid_mask_all, full_candidate_set, full_correct_indices,
            full_loss_mask, full_session_ids, strategy, global_candidate, s_start, s_end, chunk_position
        )
        selected_output, selected_loss_mask, selected_session_ids, candidate_chunk, candidate_correct = helper_outputs
        
        if selected_output is None:
            continue
        """if global_candidate:
            print(f"full_correct_indices shape: {full_correct_indices.shape}")
            print(f"seleted_output shape: {selected_output.shape}")
            print(f"selected_loss_mask shape: {selected_loss_mask.shape}")
            print(f"candidate_chunk shape: {candidate_chunk.shape}")
            print(f"candidate_correct shape: {candidate_correct.shape}")"""
        loss_chunk, metrics_chunk = compute_loss_and_metrics(
            selected_output, candidate_chunk, candidate_correct,
            strategy=strategy, global_candidate=global_candidate,
            loss_mask=selected_loss_mask, session_ids=selected_session_ids
        )
        valid_count = int(selected_loss_mask.sum().item())
        if valid_count > 0:
            total_loss_sum += loss_chunk.item() * valid_count
            total_valid_count += valid_count
            for key in metric_acc.keys():
                metric_acc[key] += metrics_chunk.get(key, 0.0) * valid_count
        if is_train and valid_count > 0:
            loss_chunk_scaled = loss_chunk / accumulation_steps
            if scaler is not None:
                if chunk_idx < len(chunk_boundaries) - 1:
                    scaler.scale(loss_chunk_scaled).backward(retain_graph=True)
                else:
                    scaler.scale(loss_chunk_scaled).backward()
            else:
                if chunk_idx < len(chunk_boundaries) - 1:
                    loss_chunk_scaled.backward(retain_graph=True)
                else:
                    loss_chunk_scaled.backward()
        torch.cuda.empty_cache()
    if total_valid_count > 0:
        avg_loss = total_loss_sum / total_valid_count
        avg_metrics = {key: metric_acc[key] / total_valid_count for key in metric_acc}
    else:
        avg_loss = 0.0
        avg_metrics = {key: 0.0 for key in metric_acc}
    return avg_loss, avg_metrics