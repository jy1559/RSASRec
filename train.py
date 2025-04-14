import torch
import torch.profiler
import torch.nn as nn
import os
from time import time
import argparse
import numpy as np
from torchviz import make_dot
import torchviz
from torch.amp import autocast, GradScaler
from contextlib import nullcontext
import wandb
from tqdm.auto import tqdm
# dataset.py 내 get_dataloaders 함수 사용 (경로에 맞게 import 수정)
from Datasets.dataset import get_dataloaders
# 모델 정의
from models.model import SeqRecModel
# optimizer 설정
from optimizer import get_optimizer
# loss 계산 모듈
from loss import compute_loss_and_metrics
# candidate set 및 정답 인덱스, 배치 아이디 생성, Timer 등 util 함수들
from util import get_candidate_set_for_batch, get_batch_item_ids, load_item_embeddings, Timer, load_candidate_tensor

def parse_args():
    parser = argparse.ArgumentParser(description="Train SeqRecModel")
    parser.add_argument("--dataset_folder", type=str, default="./Datasets", help="Dataset 폴더 경로")
    parser.add_argument("--dataset_name", type=str, default="Globo", help="Dataset 이름")
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--val_batch_size", type=int, default=4)
    parser.add_argument("--test_batch_size", type=int, default=4)
    parser.add_argument("--train_batch_th", type=int, default=300000)
    parser.add_argument("--val_batch_th", type=int, default=500000)
    parser.add_argument("--test_batch_th", type=int, default=500000)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--use_llm", type=bool, default=False)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--use_bucket_batching", type=bool, default=True)
    parser.add_argument("--optimizer", type=str, default="AdamW")
    parser.add_argument("--train_strategy", type=str, default="EachSession_LastInter")
    parser.add_argument("--test_strategy", type=str, default="Global_LastInter")
    parser.add_argument("--candidate_size", type=int, default=32, help="Candidate set 크기 (positive + negatives)")
    parser.add_argument("--global_candidate", action='store_true', help="글로벌 candidate set 사용 여부")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use, e.g., 'cuda:0', 'cuda:1', etc.")
    parser.add_argument("--wandb_off", action="store_true", help="Turn off wandb logging")
    parser.add_argument("--use_amp", action="store_true", help="Turn on FP16 training")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    return parser.parse_args()

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
            if valid_idx.numel() > 1:
                for idx in valid_idx[1:]:
                    idx = idx.item()
                    s_idx = idx // I
                    token_idx = idx % I
                    selected_b.append(chunk_raw_output[b, s_idx, token_idx, :])
                    sess_ids_b.append(s_idx)
            elif valid_idx.numel() == 1:
                idx = valid_idx.item()
                s_idx = idx // I
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
        candidate_correct = full_correct_indices
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
        num_chunks = int(total_interactions // batch_th) + 5
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

def train_one_epoch(model, accumulated_step, dataloader, optimizer, device, 
                    candidate_size, global_candidate, item_embeddings_tensor, 
                    candidate_tensor, wandb_logging, train_strategy='EachSession_LastInter',
                    accumulation_steps=1, scaler=None, batch_th = 50000):
    model.train()
    model.strategy = train_strategy
    total_loss = 0.0
    batch_counter = accumulated_step
    pbar = tqdm(dataloader, desc="epoch", leave=False, total=len(dataloader))
    loss_accum_cpu = 0.0  
    optimizer.zero_grad()
    
    for batch in pbar:
        batch_counter += 1
        with Timer("batch_to_device", wandb_logging, batch_counter):
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) 
                     for k, v in batch.items()}
        if wandb_logging:
            wandb.log({
                "Batch_size/batch_total_size": batch['item_id'].shape[0] * batch['item_id'].shape[1] * batch['item_id'].shape[2],
                "Batch_size/num_user": batch['item_id'].shape[0],
                "Batch_size/max_sessions": batch['item_id'].shape[1],
                "Batch_size/max_interactions": batch['item_id'].shape[2],
                "Batch_size/attention_overhead": batch['item_id'].shape[0] * batch['item_id'].shape[1] * batch['item_id'].shape[2] * batch['item_id'].shape[2],
            }, step=batch_counter)
        # 판단: 배치 내 총 interaction 수가 threshold의 1.25배 이상이면 chunk 방식 실행
        session_size = batch['item_id'].shape[1]
        interaction_size = batch['item_id'].shape[2]
        use_chunk = (session_size * interaction_size > 1.25 *  batch_th)
        if use_chunk:
            # 미리 candidate 및 loss 관련 정보를 full batch로 계산
            with Timer("get_batch_item_ids", wandb_logging, batch_counter):
                with torch.no_grad():
                    full_batch_item_ids, full_loss_mask, full_session_ids = get_batch_item_ids(batch['item_id'], strategy=train_strategy)
            with Timer("get_candidate_set_for_batch", wandb_logging, batch_counter):
                with torch.no_grad():
                    full_candidate_set, full_correct_indices = get_candidate_set_for_batch(
                        full_batch_item_ids,
                        candidate_size,
                        item_embeddings_tensor=item_embeddings_tensor,
                        projection_ffn=model.projection_ffn,
                        candidate_tensor=candidate_tensor,
                        global_candidate=global_candidate
                    )
            with Timer("batch_by_chunk", wandb_logging, batch_counter):
                loss_value, metric = batch_by_chunk(batch, full_candidate_set, full_correct_indices, full_loss_mask, full_session_ids,
                                                    model, strategy=train_strategy, global_candidate=global_candidate,
                                                    is_train=True, use_amp=(scaler is not None), scaler=scaler,
                                                    accumulation_steps=accumulation_steps, device=device,
                                                    batch_th=batch_th, wandb_logging=wandb_logging,
                                                    base_batch_counter=batch_counter)
        else:
            # 일반 방식: forward 전체 배치
            context = autocast(device) if scaler is not None else nullcontext()
            with context, Timer("model_forward", wandb_logging, batch_counter):
                output_features, updated_user_embedding = model(batch)
            with Timer("get_batch_item_ids", wandb_logging, batch_counter):
                with torch.no_grad():
                    batch_item_ids, loss_mask, session_ids = get_batch_item_ids(batch['item_id'], strategy=train_strategy)
            if wandb_logging:
                wandb.log({
                    "Batch_size/loss_total_size": batch_item_ids.shape[0] * batch_item_ids.shape[1],
                    "Batch_size/loss_maxlen": batch_item_ids.shape[1]
                }, step=batch_counter)
            with Timer("get_candidate_set_for_batch", wandb_logging, batch_counter):
                with torch.no_grad():
                    candidate_set, correct_indices = get_candidate_set_for_batch(
                        batch_item_ids,
                        candidate_size,
                        item_embeddings_tensor=item_embeddings_tensor,
                        projection_ffn=model.projection_ffn,
                        candidate_tensor=candidate_tensor,
                        global_candidate=global_candidate
                    )
            candidate_set = candidate_set.to(device)
            correct_indices = correct_indices.to(device)
            with Timer("compute_loss_and_metrics", wandb_logging, batch_counter):
                loss, metric = compute_loss_and_metrics(
                    output_features, candidate_set, correct_indices,
                    strategy=train_strategy,
                    global_candidate=global_candidate,
                    loss_mask=loss_mask,
                    session_ids=session_ids
                )
            loss = loss / accumulation_steps
            with Timer("backward", wandb_logging, batch_counter):
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            loss_value = loss.item() * accumulation_steps
        loss_accum_cpu += loss_value
        total_loss += loss_value
        if batch_counter % accumulation_steps == 0:
            with Timer("optimizer_step", wandb_logging, batch_counter):
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
            loss_accum_cpu = 0.0
        if wandb_logging:
            wandb.log({
                "train/loss": loss_value / accumulation_steps,
                "train/accuracy": metric["accuracy"],
                "train/MRR": metric["MRR"],
                "train/HitRate@1": metric["HitRate@1"],
                "train/NDCG@1": metric["NDCG@1"],
                "train/HitRate@3": metric["HitRate@3"],
                "train/NDCG@3": metric["NDCG@3"],
                "train/HitRate@5": metric["HitRate@5"],
                "train/NDCG@5": metric["NDCG@5"],
                "train/HitRate@10": metric["HitRate@10"],
                "train/NDCG@10": metric["NDCG@10"],
                "train/SRA": metric["SRA"],
                "train/WSRA": metric["WSRA"]
            }, step=batch_counter)
        pbar.set_postfix(loss=loss_value, acc=f'{metric["accuracy"]*100:.2f}%', 
                         HR_3=f'{metric["HitRate@3"]*100:.2f}%', HR_10=f'{metric["HitRate@10"]*100:.2f}%',
                         MRR=f'{metric["MRR"]*100:.2f}%', NDCG3=f'{metric["NDCG@3"]*100:.2f}%',
                         NDCG5=f'{metric["NDCG@5"]*100:.2f}%', SRA=f'{metric["SRA"]*100:.2f}%',
                         WSRA=f'{metric["WSRA"]*100:.2f}%')
        torch.cuda.empty_cache()
    return total_loss / len(dataloader), batch_counter

def evaluate(model, dataloader, device, item_embeddings_tensor, candidate_size, candidate_tensor, loss_strategy='Global_LastInter', global_candidate=True, batch_th = 100000):
    model.eval()
    model.strategy = loss_strategy
    total_loss = 0.0
    metrics = {}
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            session_size = batch['item_id'].shape[1]
            interaction_size = batch['item_id'].shape[2]
            use_chunk = (session_size * interaction_size > 2.5 * batch_th)
            if use_chunk:
                full_batch_item_ids, full_loss_mask, full_session_ids = get_batch_item_ids(batch['item_id'], strategy=loss_strategy)
                full_candidate_set, full_correct_indices = get_candidate_set_for_batch(
                    full_batch_item_ids,
                    candidate_size,
                    item_embeddings_tensor=item_embeddings_tensor,
                    projection_ffn=model.projection_ffn,
                    candidate_tensor=candidate_tensor,
                    global_candidate=global_candidate
                )
                loss_value, metrics = batch_by_chunk(batch, full_candidate_set, full_correct_indices, full_loss_mask, full_session_ids,
                                                     model, strategy=loss_strategy, global_candidate=global_candidate,
                                                     is_train=False, use_amp=False, scaler=None, accumulation_steps=1,
                                                     device=device, batch_th=batch_th, wandb_logging=False,
                                                     base_batch_counter=0)
                total_loss += loss_value
            else:
                output_features, updated_user_embedding = model(batch)
                batch_item_ids, loss_mask, session_ids = get_batch_item_ids(batch['item_id'], strategy=loss_strategy)
                candidate_set, correct_indices = get_candidate_set_for_batch(
                    batch_item_ids,
                    candidate_size,
                    item_embeddings_tensor=item_embeddings_tensor,
                    projection_ffn=model.projection_ffn,
                    candidate_tensor=candidate_tensor,
                    global_candidate=global_candidate
                )
                candidate_set = candidate_set.to(device)
                correct_indices = correct_indices.to(device)
                loss, metrics = compute_loss_and_metrics(
                    output_features, candidate_set, correct_indices,
                    strategy=loss_strategy,
                    global_candidate=global_candidate,
                    loss_mask=loss_mask,
                    session_ids=session_ids
                )
                total_loss += loss.item()
    return total_loss / len(dataloader), metrics

def main():
    args = parse_args()
    device = args.device
    if not args.wandb_off:
        wandb.init(project="RSASRec_25April", config=vars(args))
        wandb.define_metric("val/*", step_metric="val_epoch")
    # 데이터셋 로더 생성
    sc = time()
    train_loader, val_loader, test_loader = get_dataloaders(args)
    #print(f"Dataset loader 생성 완료. 소요 시간: {(time() - sc):.2f}초")
    sc = time()
    
    item_embedding_file = os.path.join(args.dataset_folder, args.dataset_name, "item_embedding_normalized_revised.pickle")
    item_embeddings_tensor = load_item_embeddings(item_embedding_file, args.device)
    #print(f"Item embedding load 완료. 소요 시간: {(time() - sc):.2f}초")
    sc = time()
    
    candidate_file = os.path.join(args.dataset_folder, args.dataset_name, "candidate_sets_revised.npz")
    candidate_tensor = load_candidate_tensor(candidate_file, args.candidate_size, args.device)
    #print(f"Candidate set load 완료. 소요 시간: {(time() - sc):.2f}초")
    sc = time()
    
    if args.dataset_name == "Globo":
        num_add_info = 8
    elif args.dataset_name == "LFM-BeyMS":
        num_add_info = 0
    elif args.dataset_name == "Retail_Rocket":
        num_add_info = 2
    
    model = SeqRecModel(num_add_info=num_add_info, item_embedding_tensor=item_embeddings_tensor if not args.use_llm else None, device=device)
    model.to(device)
    use_amp = args.use_amp
    accumulation_steps = args.accumulation_steps
    scaler = GradScaler() if use_amp else None
    if not args.wandb_off:
        pass  # wandb.watch(model, log="all")
    #print(f"Model 초기화 완료. 소요 시간: {(time() - sc):.2f}초")
    sc = time()
    
    optimizer_config = {
        "optimizer": args.optimizer,
        "lr": args.lr,
        "weight_decay": args.weight_decay
    }
    optimizer = get_optimizer(model, optimizer_config)
    
    accumulated_step = 0
    pbar = tqdm(range(1, args.num_epochs + 1), desc="Train", total=args.num_epochs)
    for epoch in pbar:
        train_loss, accumulated_step = train_one_epoch(
            model, accumulated_step, train_loader, optimizer, device,
            candidate_size=args.candidate_size,
            global_candidate=args.global_candidate,
            item_embeddings_tensor=item_embeddings_tensor,
            candidate_tensor=candidate_tensor,
            wandb_logging=not args.wandb_off,
            train_strategy=args.train_strategy,
            accumulation_steps=accumulation_steps,
            scaler=scaler,
            batch_th=args.train_batch_th
        )
        val_loss, metrics = evaluate(
            model,
            val_loader,
            device,
            item_embeddings_tensor,
            candidate_size=args.candidate_size,
            candidate_tensor=candidate_tensor,
            global_candidate=True,
            loss_strategy=args.test_strategy,
            batch_th = args.val_batch_th
        )
        pbar.set_postfix(acc=f'{metrics["accuracy"]*100:.2f}%', train_loss=f'{train_loss:.4f}',val_loss = f'{val_loss:.4f}',
                         HR_5=f'{metrics["HitRate@5"]*100:.2f}%', MRR=f'{metrics["MRR"]*100:.2f}%', NDCG3=f'{metrics["NDCG@3"]*100:.2f}%',
                         WSRA=f'{metrics["WSRA"]*100:.2f}%')
        if not args.wandb_off:
            wandb.log({"train/epoch": epoch}, step=accumulated_step)
            wandb.log({
                "val_epoch": epoch,
                "val/loss": val_loss,
                "val/accuracy": metrics["accuracy"],
                "val/MRR": metrics["MRR"],
                "val/HitRate@1": metrics["HitRate@1"],
                "val/NDCG@1": metrics["NDCG@1"],
                "val/HitRate@3": metrics["HitRate@3"],
                "val/NDCG@3": metrics["NDCG@3"],
                "val/HitRate@5": metrics["HitRate@5"],
                "val/NDCG@5": metrics["NDCG@5"],
                "val/HitRate@10": metrics["HitRate@10"],
                "val/NDCG@10": metrics["NDCG@10"],
                "val/SRA": metrics["SRA"],
                "val/WSRA": metrics["WSRA"]
            })
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pt")
        torch.save(model.state_dict(), checkpoint_path)
    
    test_loss, test_metrics = evaluate(
        model,
        test_loader,
        device,
        item_embeddings_tensor,
        candidate_size=args.candidate_size,
        candidate_tensor=candidate_tensor,
        global_candidate=True,
        loss_strategy=args.test_strategy
    )
    if not args.wandb_off:
        wandb.log({
            "test/loss": test_loss,
            "test/accuracy": test_metrics["accuracy"],
            "test/MRR": test_metrics["MRR"],
            "test/HitRate@1": test_metrics["HitRate@1"],
            "test/NDCG@1": test_metrics["NDCG@1"],
            "test/HitRate@3": test_metrics["HitRate@3"],
            "test/NDCG@3": test_metrics["NDCG@3"],
            "test/HitRate@5": test_metrics["HitRate@5"],
            "test/NDCG@5": test_metrics["NDCG@5"],
            "test/HitRate@10": test_metrics["HitRate@10"],
            "test/NDCG@10": test_metrics["NDCG@10"],
            "test/SRA": test_metrics["SRA"],
            "test/WSRA": test_metrics["WSRA"]
        })
    print(f"Test Loss = {test_loss:.4f}")

if __name__ == "__main__":
    main()
