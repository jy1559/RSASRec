#!/usr/bin/env python
import os
import sys
import pickle
import numpy as np
import torch
from tqdm import tqdm

# 입력 인자로 파티션 번호 (0~3) 받기 (이전 코드와 동일)
if len(sys.argv) < 2:
    print("Usage: python precompute_candidates_gpu.py <partition_index (0~3)>")
    sys.exit(1)
try:
    part_index = int(sys.argv[1])
    if part_index < 0 or part_index > 3:
        raise ValueError
except ValueError:
    print("Partition index must be an integer between 0 and 3.")
    sys.exit(1)

# -------------------
# 후보 후보 설정
candidate_size = 128   # 최종 후보 수 128개로 변경
ratio_hard = 0.4
ratio_easy = 0.2
n_hard = int(candidate_size * ratio_hard)
n_easy = int(candidate_size * ratio_easy)
n_rand = candidate_size - n_hard - n_easy

# 그룹별 offset
offsets = {
    'hard': 0.2,
    'rand': 0.0,
    'easy': -0.2
}
# 랜덤 노이즈 범위 (uniform)
noise_range = 0.05

# 블록 사이즈 (메모리와 속도 균형, 조정 가능)
block_size = 256

def precompute_candidates_for_folder(folder_path, part_index, gpu_id):
    print(f"Processing folder: {folder_path} for partition {part_index} using cuda:{gpu_id}")
    
    embedding_pickle_path = os.path.join(folder_path, "item_embedding.pickle")
    if not os.path.exists(embedding_pickle_path):
        print("File not found:", embedding_pickle_path)
        return
    
    with open(embedding_pickle_path, "rb") as f:
        item_embeddings_dict = pickle.load(f)  # { item_id (str): np.array([...]) }
    
    # 아이템 ID 정렬 (문자열이지만 비교 시 int로)
    item_ids = sorted(item_embeddings_dict.keys(), key=lambda x: int(x))
    num_items = len(item_ids)
    
    base_dim = 384  # 원래 임베딩 차원
    # index 0은 패딩용 0 벡터로 예약
    embedding_list = [np.zeros((base_dim,), dtype=np.float32)]
    for item_id in item_ids:
        embedding_list.append(item_embeddings_dict[item_id])
    # embedding_matrix: [num_items+1, base_dim]
    embedding_matrix = np.stack(embedding_list, axis=0)
    
    # embedding_matrix는 CPU에 로드
    embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float32, device="cpu")
    
    # 지정한 GPU 사용
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    # 실제 아이템 임베딩: index 1~end, shape: [num_items, base_dim]
    real_embeddings = embedding_matrix[1:].to(device)
    norms = torch.norm(real_embeddings, dim=1, keepdim=True)
    embedding_norm = real_embeddings / (norms + 1e-8)
    # 이후 block 처리 위해 CPU로 옮김
    embedding_norm = embedding_norm.cpu()
    
    # Partition: 전체 아이템 목록을 4등분하여, 입력 받은 파티션만 처리
    partition_size = num_items // 4
    start_idx = part_index * partition_size
    end_idx = num_items if part_index == 3 else (part_index + 1) * partition_size
    print(f"Processing query items from index {start_idx} to {end_idx} (total {num_items} items)")
    
    candidate_set_dict = {}
    
    # Block processing: 파티션 내 아이템들을 block_size 단위로 처리
    for block_start in tqdm(range(start_idx, end_idx, block_size), desc="Block processing"):
        block_end = min(block_start + block_size, end_idx)
        block_embeddings = embedding_norm[block_start:block_end]  # [B, base_dim]
        sim_block = torch.matmul(block_embeddings, embedding_norm.t())  # [B, num_items]
        sim_block = sim_block.cpu().numpy()
        
        for i, row in enumerate(sim_block):
            global_index = block_start + i
            # 자기 자신 제외
            row[global_index] = -np.inf
            
            # Negative sampling using partial sorting
            hard_idx = np.argpartition(-row, n_hard)[:n_hard]
            easy_idx = np.argpartition(row, n_easy)[:n_easy]
            all_indices = np.arange(num_items)
            mask = np.ones(num_items, dtype=bool)
            mask[hard_idx] = False
            mask[easy_idx] = False
            middle_idx = all_indices[mask]
            if len(middle_idx) < n_rand:
                rand_idx = middle_idx
            else:
                rand_idx = np.random.choice(middle_idx, size=n_rand, replace=False)
            
            candidate_tuples = []
            for idx in hard_idx:
                candidate_tuples.append((idx, row[idx], 'hard'))
            for idx in rand_idx:
                candidate_tuples.append((idx, row[idx], 'rand'))
            for idx in easy_idx:
                candidate_tuples.append((idx, row[idx], 'easy'))
            
            scored_candidates = []
            for cand in candidate_tuples:
                idx_val, sim_val, cat = cand
                noise = np.random.uniform(0, noise_range)
                score = sim_val + noise
                scored_candidates.append((idx_val, score))
            scored_candidates.sort(key=lambda x: -x[1])
            final_indices = [sc[0] for sc in scored_candidates][:candidate_size]
            candidate_item_ids = [int(item_ids[idx]) for idx in final_indices]
            candidate_set_dict[int(item_ids[global_index])] = candidate_item_ids
    
    # 저장: npz 파일로 저장 (압축하여 용량 절감)
    output_candidate_file = os.path.join(folder_path, f"precomputed_candidate_set_{part_index}.npz")
    # 딕셔너리를 두 개의 배열로 변환: keys와 values (values는 2D array, shape: [num_queries, candidate_size])
    keys = np.array(list(candidate_set_dict.keys()), dtype=np.int32)
    # values 배열: 각 row는 candidate_set_dict[key] (각각 길이 candidate_size)
    values = np.array([candidate_set_dict[k] for k in keys], dtype=np.int32)
    np.savez_compressed(output_candidate_file, keys=keys, values=values)
    print("Candidate set saved to", output_candidate_file)

def main():
    parent_folder = os.getcwd()
    subfolders = ["Globo", "LFM-BeyMS", "Retail_Rocket"]
    for subfolder in tqdm(subfolders, desc="Processing folders"):
        folder_path = os.path.join(parent_folder, subfolder)
        if os.path.isdir(folder_path):
            precompute_candidates_for_folder(folder_path, part_index, part_index)  # GPU id를 파티션 번호로 사용
        else:
            print("Folder not found:", folder_path)

if __name__ == "__main__":
    main()
