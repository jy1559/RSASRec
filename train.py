import torch
import torch.nn as nn
import os
from time import time
import argparse
import numpy as np
from torchviz import make_dot
import torchviz
from tqdm.auto import tqdm
# dataset.py 내 get_dataloaders 함수 사용 (경로에 맞게 import 수정)
from Datasets.dataset import get_dataloaders
# 모델 정의
from models.model import SeqRecModel
# optimizer 설정
from optimizer import get_optimizer
# loss 계산 모듈
from loss import compute_loss
# candidate set 및 정답 인덱스 생성을 위한 util 함수
from util import get_candidate_set_for_batch, get_batch_item_ids, load_item_embeddings
# (실제 item 임베딩은 사전 계산 후 로드하여 사용해야 함. 여기서는 placeholder로 처리)

def parse_args():
    parser = argparse.ArgumentParser(description="Train SeqRecModel")
    parser.add_argument("--dataset_folder", type=str, default="./Datasets", help="Dataset 폴더 경로")
    parser.add_argument("--dataset_name", type=str, default="Globo", help="Dataset 이름")
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--val_batch_size", type=int, default=4)
    parser.add_argument("--test_batch_size", type=int, default=4)
    parser.add_argument("--train_batch_th", type=int, default=50000)
    parser.add_argument("--val_batch_th", type=int, default=100000)
    parser.add_argument("--test_batch_th", type=int, default=100000)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--use_llm", type=bool, default=False)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--use_bucket_batching", type=bool, default=True)
    parser.add_argument("--optimizer", type=str, default="AdamW")
    parser.add_argument("--candidate_size", type=int, default=32, help="Candidate set 크기 (positive + negatives)")
    parser.add_argument("--global_candidate", action='store_true', help="글로벌 candidate set 사용 여부")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use, e.g., 'cuda:0', 'cuda:1', etc.")
    return parser.parse_args()

def train_one_epoch(model, dataloader, optimizer, device, candidate_size, global_candidate, item_embeddings, candidate_dict):
    model.train()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc="epoch", leave=False, total=len(dataloader))
    for batch in pbar:
        # batch 내 tensor들은 device로 이동 (문자열은 그대로 유지)
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        optimizer.zero_grad()
        
        # 모델 Forward Pass: 모델은 output_features와 업데이트된 user embedding을 반환
        output_features, updated_user_embedding = model(batch)
        #print(output_features)
        #dot = make_dot(output_features, params=dict(model.named_parameters()))
        #dot.render("model_graph", format="png")
        # 배치에서 정답 아이템 id 추출 (예: EachSession_LastInter 전략)
        batch_item_ids = get_batch_item_ids(batch['item_id'], strategy='EachSession_LastInter')
        # 실제 학습에서는 사전 계산된 item_embeddings를 로드하여 사용해야 합니다.
        # 여기서는 예시를 위해 빈 dict로 전달한 후 내부에서 적절한 처리를 하거나 dummy candidate set을 사용하도록 합니다.
        candidate_set, correct_indices = get_candidate_set_for_batch(batch_item_ids.tolist(),
                                                                     candidate_size,
                                                                     item_embeddings=item_embeddings,  # 실제 item_embeddings 로드 필요
                                                                     projection_ffn=model.projection_ffn,
                                                                     candidate_dict=candidate_dict,
                                                                     global_candidate=global_candidate)
        # candidate_set과 correct_indices를 device로 이동
        candidate_set = candidate_set.to(device)
        correct_indices = correct_indices.to(device)
        # Loss 계산: 전략에 따라 output_features의 shape와 candidate_set shape가 달라질 수 있음.
        loss, metric = compute_loss(output_features, candidate_set, correct_indices,
                            strategy='EachSession_LastInter',
                            global_candidate=global_candidate)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item(), acc=f'{metric["accuracy"]*100:.2f}%', 
                         HR_3=f'{metric["HitRate@3"]*100:.2f}%',
                         MRR=f'{metric["MRR"]*100:.2f}%', 
                         NDCG3=f'{metric["NDCG@3"]*100:.2f}%')
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device, item_embeddings, candidate_size):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            output_features, updated_user_embedding = model(batch)
            
            batch_item_ids = get_batch_item_ids(batch['item_id'], strategy='EachSession_LastInter')
            candidate_set, correct_indices = get_candidate_set_for_batch(batch_item_ids.tolist(),
                                                                         candidate_size,
                                                                         item_embeddings=item_embeddings,  # 실제 item_embeddings 로드 필요
                                                                         projection_ffn=model.projection_ffn,
                                                                         global_candidate=True)
            candidate_set = candidate_set.to(device)
            correct_indices = correct_indices.to(device)
            
            loss, metrics = compute_loss(output_features, candidate_set, correct_indices,
                                strategy='EachSession_LastInter',
                                global_candidate=True)
            total_loss += loss.item()
    return total_loss / len(dataloader), metrics

def main():
    args = parse_args()
    device = args.device
    
    # 데이터셋 로더 생성
    sc = time()
    train_loader, val_loader, test_loader = get_dataloaders(args)
    print(f"Dataset loader 생성 완료. 소요 시간: {(time() -sc):.2f}초")
    sc = time()
    
    item_embedding_file = os.path.join(args.dataset_folder, args.dataset_name, "item_embedding_normalized.pickle")
    item_embeddings = load_item_embeddings(item_embedding_file)
    print(f"Item embedding load 완료. 소요 시간: {(time() -sc):.2f}초")
    sc = time()
    
    # candidate 세트 파일 경로 (예: candidate_sets.npz로 저장한 경우)
    candidate_file = os.path.join(args.dataset_folder, args.dataset_name, "candidate_sets.npz")
    # np.load로 npz 파일 불러오기
    data = np.load(candidate_file)
    candidate_keys = data["keys"]
    candidate_values = data["values"]
    candidate_dict = {}
    for i, k in enumerate(candidate_keys):
        cand_list = candidate_values[i].tolist()
        if len(cand_list) < args.candidate_size:
            raise ValueError(f"Candidate set for item {k} has length {len(cand_list)} greater than candidate_size {args.candidate_size}.")
        candidate_dict[int(k)] = cand_list[:args.candidate_size - 1]
    print(f"Candidate set load 완료. 소요 시간: {(time() -sc):.2f}초")
    sc = time()

    # 모델 초기화 및 device 설정
    if args.dataset_name == "Globo": num_add_info = 8
    elif args.dataset_name == "LFM-BeyMS": num_add_info = 0
    elif args.dataset_name == "Retail_Rocket": num_add_info = 2
    
    model = SeqRecModel(num_add_info=num_add_info, item_embedding_dict = item_embeddings if not args.use_llm else None, device=device)
    model.to(device)
    print(f"Model 초기화 완료. 소요 시간: {(time() -sc):.2f}초")
    sc = time()
    

    # optimizer 설정 (optimizer.py 활용)
    optimizer_config = {
        "optimizer": args.optimizer,
        "lr": args.lr,
        "weight_decay": args.weight_decay
    }
    optimizer = get_optimizer(model, optimizer_config)
    
    # 학습 loop
    for epoch in tqdm(range(1, args.num_epochs + 1), desc="Train", total = args.num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device,
                                     candidate_size=args.candidate_size,
                                     global_candidate=args.global_candidate,
                                     item_embeddings =item_embeddings,
                                     candidate_dict = candidate_dict)
        val_loss, metrics = evaluate(model, val_loader, device, item_embeddings, candidate_size=args.candidate_size)
        print(f"Epoch {epoch}/{args.num_epochs}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Vale metrics: {metrics}")
        
        # 체크포인트 저장
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pt")
        torch.save(model.state_dict(), checkpoint_path)
    
    # 최종 테스트 평가
    test_loss = evaluate(model, test_loader, device, candidate_size=args.candidate_size)
    print(f"Test Loss = {test_loss:.4f}")

if __name__ == "__main__":
    main()
