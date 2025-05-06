import torch
import torch.profiler
import torch.nn as nn
import os
from time import time
import argparse
from torch.amp import autocast, GradScaler
from collections import defaultdict
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

from chunk_forward import batch_by_chunk

def parse_args():
    parser = argparse.ArgumentParser(description="Train SeqRecModel")
    parser.add_argument("--dataset_folder", type=str, default="./Datasets", help="Dataset 폴더 경로")
    parser.add_argument("--dataset_name", type=str, default="Globo", help="Dataset 이름")
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--val_batch_size", type=int, default=4)
    parser.add_argument("--test_batch_size", type=int, default=4)
    parser.add_argument("--train_batch_th", type=int, default=70000)
    parser.add_argument("--val_batch_th", type=int, default=1000000)
    parser.add_argument("--test_batch_th", type=int, default=1000000)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--use_llm", type=bool, default=False)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--use_bucket_batching", type=bool, default=True)
    parser.add_argument("--optimizer", type=str, default="AdamW")
    parser.add_argument("--train_strategy", type=str, default="EachSession_LastInter")
    parser.add_argument("--test_strategy", type=str, default="Global_LastInter")
    parser.add_argument("--candidate_size", type=int, default=64, help="Candidate set 크기 (positive + negatives)")
    parser.add_argument("--global_candidate", action='store_true', help="글로벌 candidate set 사용 여부")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use, e.g., 'cuda:0', 'cuda:1', etc.")
    parser.add_argument("--wandb_off", action="store_true", help="Turn off wandb logging")
    parser.add_argument("--use_amp", action="store_true", help="Turn on FP16 training")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--projection_before_summation", type=bool, default=True, help="Embedding 만들 때 add_info랑 time gap을 384로 더한 후 projection할지")
    return parser.parse_args()



def train_one_epoch(model, accumulated_step, dataloader, optimizer, device, 
                    candidate_size, global_candidate, item_embeddings_tensor, 
                    candidate_tensor, wandb_logging, train_strategy='EachSession_LastInter',
                    accumulation_steps=1, epoch_num = 0, scaler=None, batch_th = 50000):
    """for name, p in model.named_parameters():
        if not p.requires_grad:
            print(f"'{name}' has requires_grad=False")"""
    model.train()
    model.strategy = train_strategy
    total_loss = 0.0
    batch_counter = accumulated_step
    total_batches = len(dataloader)
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
                "Batch_size/user_total_size": batch['item_id'].shape[1] * batch['item_id'].shape[2],
                "Batch_size/num_user": batch['item_id'].shape[0],
                "Batch_size/max_sessions": batch['item_id'].shape[1],
                "Batch_size/max_interactions": batch['item_id'].shape[2],
                "Batch_size/attention_overhead": batch['item_id'].shape[0] * batch['item_id'].shape[1] * batch['item_id'].shape[2] * batch['item_id'].shape[2],
            }, step=batch_counter)
        # 판단: 배치 내 총 interaction 수가 threshold의 1.1배 이상이면 chunk 방식 실행
        batch_size = batch['item_id'].shape[0]
        session_size = batch['item_id'].shape[1]
        interaction_size = batch['item_id'].shape[2]
        use_chunk = False#(session_size * interaction_size > 0.4 *  batch_th) or (interaction_size > 1000)
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
                loss, metric, _ = compute_loss_and_metrics(
                    output_features, candidate_set, correct_indices,
                    strategy=train_strategy,
                    global_candidate=global_candidate,
                    loss_mask=loss_mask,
                    session_ids=session_ids
                )
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
        #after = param.detach()
        #print("max |Δparam|:", (before - after).abs().max().item())
        if wandb_logging:
            log_bucket = 200
            progress = (batch_counter / total_batches)
            progress = int(round(progress * log_bucket)) / log_bucket
            log_dict = {"epoch_progress": progress,
                        "train/loss": loss_value / accumulation_steps}
            for k in metric.keys():
                val = metric[k].item() if torch.is_tensor(metric[k]) else metric[k]
                log_dict[f"train/{k}"] = val
            wandb.log(log_dict, step=batch_counter)
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
    metric_sum = defaultdict(float)
    valid_total = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="val", leave=False, total=len(dataloader)):
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            session_size = batch['item_id'].shape[1]
            interaction_size = batch['item_id'].shape[2]
            use_chunk = False#(session_size * interaction_size > 1.2 * batch_th) or (interaction_size > 1000)
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
                loss, metrics, valid_cnt = compute_loss_and_metrics(
                    output_features, candidate_set, correct_indices,
                    strategy=loss_strategy,
                    global_candidate=global_candidate,
                    loss_mask=loss_mask,
                    session_ids=session_ids
                )
                total_loss += loss.item()
                for k, v in metrics.items():
                    metric_sum[k]   += v * valid_cnt    # 가중 합
                valid_total  += valid_cnt        # 가중치 누적
    epoch_metrics = {k: metric_sum[k] / valid_total for k in metric_sum}
    return total_loss / len(dataloader), epoch_metrics

def main():
    args = parse_args()
    device = args.device
    if not args.wandb_off:
        wandb.init(project="RSASRec_25April", config=vars(args))
        wandb.define_metric("epoch_progress")
        wandb.define_metric("train/*", step_metric="epoch_progress", overwrite=True)
        wandb.define_metric("val_epoch")
        wandb.define_metric("val*", step_metric="val_epoch")
        metric_names = [
            "loss", "accuracy", "MRR", "HitRate@1", "NDCG@1",
            "HitRate@3", "NDCG@3", "HitRate@5", "NDCG@5",
            "HitRate@10", "NDCG@10", "SRA", "WSRA"]
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
    
    if args.dataset_name == "Globo": num_add_info = 8
    elif args.dataset_name == "LFM-BeyMS": num_add_info = 0
    elif args.dataset_name == "Retail_Rocket": num_add_info = 2
    
    model = SeqRecModel(num_add_info=num_add_info, 
                        item_embedding_tensor=item_embeddings_tensor if not args.use_llm else None, 
                        device=device,
                        projection_before_summation = args.projection_before_summation)
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
            epoch_num=epoch,
            scaler=scaler,
            batch_th=args.train_batch_th
        )
        if not args.wandb_off:
            # 0) 공통 메타
            log_dict = {"val_epoch": epoch}

            # 2) 주 실험 strategy: args.test_strategy
            strategies = [args.test_strategy] \
                        if args.test_strategy == "Global_LastInter" else \
                        [args.test_strategy, "Global_LastInter"]          # 필요시 확장 가능
            #  ──  Local / Global 두 번 돌림 ──────────────────────────
            for strat in strategies:
                for gc_flag in [False]:          # Global, Local
                    loss, metrics = evaluate(
                        model, val_loader, device,
                        item_embeddings_tensor,
                        candidate_size=args.candidate_size,
                        candidate_tensor=candidate_tensor,
                        global_candidate=gc_flag,
                        loss_strategy=strat,
                        batch_th=args.val_batch_th
                    )
                    cand_tag = "Global_candidate" if gc_flag else "Local_candidate"
                    head = f"val({strat})_{cand_tag}"
                    log_dict[f"{head}/loss"] = loss
                    for k, v in metrics.items():
                        log_dict[f"{head}/{k}"] = float(v)

            # 3) WANDB 업로드 (step=None → wandb 내부 step, x축은 val_epoch로 정의해 둠)
            wandb.log(log_dict)
            pbar.set_postfix(acc=f'{metrics["accuracy"]*100:.2f}%', train_loss=f'{train_loss:.4f}',val_loss = f'{loss:.4f}',
                            HR_5=f'{metrics["HitRate@5"]*100:.2f}%', MRR=f'{metrics["MRR"]*100:.2f}%', NDCG3=f'{metrics["NDCG@3"]*100:.2f}%',
                            WSRA=f'{metrics["WSRA"]*100:.2f}%')
        #checkpoint_dir = "checkpoints"
        #os.makedirs(checkpoint_dir, exist_ok=True)
        #checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pt"
        #torch.save(model.state_dict(), checkpoint_path)
    
    test_loss, test_metrics = evaluate(
        model,
        test_loader,
        device,
        item_embeddings_tensor,
        candidate_size=args.candidate_size,
        candidate_tensor=candidate_tensor,
        global_candidate=False,
        loss_strategy=args.test_strategy
    )
    log_dict = {f"test({args.test_strategy})": test_loss}
    for k in test_metrics.keys():
        val = test_metrics[k].item() if torch.is_tensor(test_metrics[k]) else test_metrics[k]          # tensor → float
        log_dict[f"test({args.test_strategy})/{k}"] = val
    if not args.wandb_off:
        wandb.log(log_dict)
        if args.test_strategy != 'Global_LastInter':
            test_loss, test_metrics = evaluate(
                model,
                test_loader,
                device,
                item_embeddings_tensor,
                candidate_size=args.candidate_size,
                candidate_tensor=candidate_tensor,
                global_candidate=False,
                loss_strategy="Global_LastInter"
            )
            log_dict[f"test(Global_LastInter)"] = test_loss
            for k in test_metrics.keys():
                val = test_metrics[k].item() if torch.is_tensor(test_metrics[k]) else test_metrics[k]          # tensor → float
                log_dict[f"test(Global_LastInter)/{k}"] = val
            wandb.log(log_dict)
    print(f"Test Loss = {test_loss:.4f}")

if __name__ == "__main__":
    main()
