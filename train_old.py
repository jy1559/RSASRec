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
# candidate set 및 정답 인덱스 생성을 위한 util 함수
from util import get_candidate_set_for_batch, get_batch_item_ids, load_item_embeddings, Timer, load_candidate_tensor
# (실제 item 임베딩은 사전 계산 후 로드하여 사용해야 함. 여기서는 placeholder로 처리)

def parse_args():
    parser = argparse.ArgumentParser(description="Train SeqRecModel")
    parser.add_argument("--dataset_folder", type=str, default="./Datasets", help="Dataset 폴더 경로")
    parser.add_argument("--dataset_name", type=str, default="Globo", help="Dataset 이름")
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--val_batch_size", type=int, default=4)
    parser.add_argument("--test_batch_size", type=int, default=4)
    parser.add_argument("--train_batch_th", type=int, default=50000)
    parser.add_argument("--val_batch_th", type=int, default=200000)
    parser.add_argument("--test_batch_th", type=int, default=200000)
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

def train_one_epoch(model, accumulated_step, dataloader, optimizer, device, 
                    candidate_size, global_candidate, item_embeddings_tensor, 
                    candidate_tensor, wandb_logging, train_strategy='EachSession_LastInter',
                    accumulation_steps=1, scaler=None):
    model.train()
    model.strategy = train_strategy
    total_loss = 0.0
    batch_counter = accumulated_step
    pbar = tqdm(dataloader, desc="epoch", leave=False, total=len(dataloader))
    
    # logging용 변수 (CPU)
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
            }, step=batch_counter)
        # forward pass를 AMP autocat 블록(또는 그대로 실행)으로 감싸기
        context = autocast(device) if scaler is not None else nullcontext()
        with context, Timer("model_forward", wandb_logging, batch_counter):
            output_features, updated_user_embedding = model(batch)
        
        with Timer("get_batch_item_ids", wandb_logging, batch_counter):
            batch_item_ids, loss_mask, session_ids = get_batch_item_ids(batch['item_id'], strategy=train_strategy)
        
        if wandb_logging:
            wandb.log({
                "Batch_size/loss_total_size": batch_item_ids.shape[0] * batch_item_ids.shape[1],
                "Batch_size/loss_maxlen": batch_item_ids.shape[1]
            }, step=batch_counter)
        
        with Timer("get_candidate_set_for_batch", wandb_logging, batch_counter):
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
        
        # Gradient accumulation: loss를 accumulation_steps로 나눈 후, backward를 호출하여 gradient가 누적됩니다.
        loss = loss / accumulation_steps
        with Timer("backward", wandb_logging, batch_counter):
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
        
        # logging: 실제 loss 값(미니배치의 loss * accumulation_steps)을 CPU에 누적하여 기록
        loss_accum_cpu += loss.item() * accumulation_steps
        total_loss += loss.item() * accumulation_steps
        
        if batch_counter % accumulation_steps == 0:
            with Timer("optimizer_step", wandb_logging, batch_counter):
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
            
            if wandb_logging:
                wandb.log({
                    "train/loss": loss_accum_cpu / accumulation_steps,
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
            loss_accum_cpu = 0.0
        
        pbar.set_postfix(loss=loss.item() * accumulation_steps, acc=f'{metric["accuracy"]*100:.2f}%',
                         HR_3=f'{metric["HitRate@3"]*100:.2f}%',
                         HR_10=f'{metric["HitRate@10"]*100:.2f}%',
                         MRR=f'{metric["MRR"]*100:.2f}%',
                         NDCG3=f'{metric["NDCG@3"]*100:.2f}%',
                         NDCG5=f'{metric["NDCG@5"]*100:.2f}%',
                         SRA=f'{metric["SRA"]*100:.2f}%',
                         WSRA=f'{metric["WSRA"]*100:.2f}%')
        torch.cuda.empty_cache()
    return total_loss / len(dataloader), batch_counter


def evaluate(model, dataloader, device, item_embeddings_tensor, candidate_size, candidate_tensor, loss_strategy='Global_LastInter', global_candidate = True):
    model.eval()
    model.strategy = loss_strategy
    total_loss = 0.0
    metrics = {}
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            output_features, updated_user_embedding = model(batch)
            
            batch_item_ids, loss_mask, session_ids = get_batch_item_ids(batch['item_id'], strategy=loss_strategy)
            
            candidate_set, correct_indices = get_candidate_set_for_batch(
                batch_item_ids,
                candidate_size = -1,
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
    print(f"Dataset loader 생성 완료. 소요 시간: {(time() -sc):.2f}초")
    sc = time()
    
    item_embedding_file = os.path.join(args.dataset_folder, args.dataset_name, "item_embedding_normalized_revised.pickle")
    item_embeddings_tensor = load_item_embeddings(item_embedding_file, args.device)
    print(f"Item embedding load 완료. 소요 시간: {(time() -sc):.2f}초")
    sc = time()
    
    # candidate 세트 파일 경로 (예: candidate_sets.npz로 저장한 경우)
    candidate_file = os.path.join(args.dataset_folder, args.dataset_name, "candidate_sets_revised.npz")
    candidate_tensor = load_candidate_tensor(candidate_file, args.candidate_size, args.device)
    print(f"Candidate set load 완료. 소요 시간: {(time() -sc):.2f}초")
    sc = time()

    # 모델 초기화 및 device 설정
    if args.dataset_name == "Globo": num_add_info = 8
    elif args.dataset_name == "LFM-BeyMS": num_add_info = 0
    elif args.dataset_name == "Retail_Rocket": num_add_info = 2
    
    model = SeqRecModel(num_add_info=num_add_info, item_embedding_tensor = item_embeddings_tensor if not args.use_llm else None, device=device)
    model.to(device)
    use_amp = args.use_amp
    accumulation_steps = args.accumulation_steps  # 예: 4 (만약 1이면 accumulation 없이)
    scaler = GradScaler() if use_amp else None
    if not args.wandb_off:
        pass#wandb.watch(model, log="all")
    print(f"Model 초기화 완료. 소요 시간: {(time() -sc):.2f}초")
    sc = time()


    # optimizer 설정 (optimizer.py 활용)
    optimizer_config = {
        "optimizer": args.optimizer,
        "lr": args.lr,
        "weight_decay": args.weight_decay
    }
    optimizer = get_optimizer(model, optimizer_config)
    
    accumulated_step = 0
    # 학습 loop
    for epoch in tqdm(range(1, args.num_epochs + 1), desc="Train", total = args.num_epochs):
        train_loss, accumulated_step = train_one_epoch(
            model, accumulated_step, train_loader, optimizer, device,
            candidate_size=args.candidate_size,
            global_candidate=args.global_candidate,
            item_embeddings_tensor=item_embeddings_tensor,
            candidate_tensor=candidate_tensor,
            wandb_logging=not args.wandb_off,
            train_strategy=args.train_strategy,
            accumulation_steps=accumulation_steps,
            scaler=scaler
        )
        val_loss, metrics = evaluate(model, 
                                    val_loader, 
                                    device, 
                                    item_embeddings_tensor, 
                                    candidate_size=args.candidate_size, 
                                    candidate_tensor = candidate_tensor, 
                                    global_candidate = True, #args.global_candidate,
                                    loss_strategy = args.test_strategy)
        print(f"Epoch {epoch}/{args.num_epochs}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val metrics: {metrics}")
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
        # 체크포인트 저장
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pt")
        torch.save(model.state_dict(), checkpoint_path)
    
    # 최종 테스트 평가
    test_loss, test_metrics = evaluate(model, 
                                       test_loader, 
                                       device, 
                                       item_embeddings_tensor, 
                                       candidate_size=args.candidate_size, 
                                       candidate_tensor= candidate_tensor, 
                                       global_candidate = True, #args.global_candidate,
                                       loss_strategy = args.test_strategy)
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
