# dataset.py
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Sampler
import random
import math
from datetime import datetime

class SeqRecDataset(Dataset):
    def __init__(self, interactions_path, item_metadata_path, use_llm):
        with open(interactions_path, 'r', encoding='utf-8') as f:
            interactions_json = json.load(f)
            self.interaction_data = interactions_json["data"]
            self.index = interactions_json["index"]

        with open(item_metadata_path, 'r', encoding='utf-8') as f:
            self.item_metadata = json.load(f)
        self.use_llm = use_llm
        self.users = list(self.interaction_data.keys())

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user_id = self.users[idx]
        sessions = self.interaction_data[user_id]

        user_sessions = []
        for session in sessions:
            session_interactions = []
            prev_timestamp = None
            for interaction in session:
                item_id, timestamp, add_info = interaction

                if self.use_llm:    
                    sentence = self._create_embedding_sentence(item_id, timestamp, add_info)
                    add_info = None
                else:               
                    sentence = None
                delta_t = None if prev_timestamp is None else math.log(timestamp - prev_timestamp + 1e-7)
                prev_timestamp = timestamp

                session_interactions.append({
                    'item_id': item_id,
                    'embedding_sentence': sentence,  # use_llm=False인 경우 None
                    'delta_t': delta_t,
                    'add_info': add_info,  # 필요 시 별도 key
                    'timestamp': timestamp
                })
            user_sessions.append(session_interactions)

        return {'user_id': user_id, 'sessions': user_sessions}

    def _create_embedding_sentence(self, item_id, timestamp, additional_info):
        item_info = self.item_metadata.get(item_id, "No metadata available")
        readable_time = datetime.fromtimestamp(timestamp / 1000).strftime('%Y-%m-%d %H:%M:%S')
        sentence_parts = [f"Item Info: {item_info}", f"Interaction Time: {readable_time}"]

        if additional_info:
            add_info_parts = [f"{name}: {additional_info[idx]}" for name, idx in self.index["add_index"].items()]
            sentence_parts.append(f"Additional Info: {', '.join(add_info_parts)}")

        return " | ".join(sentence_parts)

def seq_collate_fn(batch, use_llm=True):
    max_sessions = max(len(user['sessions']) for user in batch)
    max_interactions = max(len(sess) for user in batch for sess in user['sessions'])

    # 공통 필드
    delta_t_batch, session_mask, interaction_mask = [], [], []
    item_id_batch = []
    # LLM 사용 시 문장 배치
    sentences_batch = [] if use_llm else None
    # add_info 사용 시 add_info 배치
    add_info_batch = [] if not use_llm else None

    for user in batch:
        user_sess_mask, user_inter_mask = [], []
        user_delta_t, user_item_id = [], []
        # 문장 / add_info
        if use_llm:
            user_sentences = []
        else:
            user_add_info = []

        for session in user['sessions']:
            sess_len = len(session)
            # delta_ts, item_ids, inter_presence
            delta_ts = [0 if inter['delta_t'] is None else inter['delta_t'] for inter in session]
            items_id = [inter['item_id'] for inter in session]
            inter_presence = [1]*sess_len
            # LLM 문장
            if use_llm:
                sentences = [inter['embedding_sentence'] for inter in session]
            else:
                add_infos = [inter['add_info'] for inter in session]

            # padding
            pad_len = max_interactions - sess_len
            delta_ts += [0]*pad_len
            items_id += [-1]*pad_len
            inter_presence += [0]*pad_len

            if use_llm:
                sentences += ["[PAD]"]*pad_len
            else:
                add_infos += [None]*pad_len  # or dict(...) empty

            # 세션단 결과 저장
            user_delta_t.append(delta_ts)
            user_item_id.append(items_id)
            user_inter_mask.append(inter_presence)

            if use_llm:
                user_sentences.append(sentences)
            else:
                user_add_info.append(add_infos)

            user_sess_mask.append(1)

        # 세션 패딩
        pad_sessions = max_sessions - len(user['sessions'])
        for _ in range(pad_sessions):
            user_delta_t.append([0]*max_interactions)
            user_item_id.append([-1]*max_interactions)
            user_inter_mask.append([0]*max_interactions)
            user_sess_mask.append(0)
            if use_llm: user_sentences.append(["[PAD]"]*max_interactions)
            else:       user_add_info.append([None]*max_interactions)
                
        # 유저별 결과 누적
        delta_t_batch.append(user_delta_t)
        item_id_batch.append(user_item_id)
        interaction_mask.append(user_inter_mask)
        session_mask.append(user_sess_mask)

        if use_llm:
            sentences_batch.append(user_sentences)
        else:
            add_info_batch.append(user_add_info)

    # 최종 반환 딕셔너리
    collate_dict = {
        'delta_ts': torch.tensor(delta_t_batch, dtype=torch.float32),
        'item_id': torch.tensor(item_id_batch, dtype=torch.int32),
        'interaction_mask': torch.tensor(interaction_mask, dtype=torch.float32),
        'session_mask': torch.tensor(session_mask, dtype=torch.float32)
    }
    if use_llm:
        collate_dict['embedding_sentences'] = sentences_batch  # list of list of list of str
    else:
        collate_dict['add_info'] = add_info_batch  # list of list of list of add_info

    return collate_dict

class BucketBatchSampler(Sampler):
    """
    전체 dataset을 (세션 수, 각 세션의 최대 interaction 개수) 기준으로 정렬한 뒤,
    batch_size 단위로 인덱스를 묶어 배치를 구성합니다.
    배치 자체의 순서는 옵션에 따라 섞을 수 있습니다.
    """
    def __init__(self, dataset, batch_size, batch_th=0, shuffle_batches=True, sort_order='descending'):
        self.dataset = dataset
        self.batch_size = batch_size
        self.batch_th = batch_th
        self.shuffle_batches = shuffle_batches
        
        # 전체 인덱스를 구한 뒤, 각 샘플의 (num_sessions, max_interactions)로 정렬
        indices = list(range(len(dataset)))
        def sort_key(idx):
            sample = dataset[idx]
            num_sessions = len(sample['sessions'])
            max_interactions = max([len(sess) for sess in sample['sessions']]) if sample['sessions'] else 0
            return (num_sessions, max_interactions)
        indices = sorted(indices, key=sort_key, reverse=(sort_order=='descending'))
        
        if self.batch_th is None or self.batch_th == 0:
            # 고정 batch_size로 그룹화
            self.batches = [indices[i:i+batch_size] for i in range(0, len(indices), batch_size)]
        else:
            # 동적 배치 구성: 현재 배치에 샘플을 추가했을 때 cost = (현재 배치 길이) * (배치 내 최대 session 수) * (배치 내 최대 interaction 수)가 batch_th 이하가 되어야 함
            self.batches = []
            current_batch = []
            current_max_sessions = 0
            current_max_interactions = 0

            for idx in indices:
                sample = dataset[idx]
                num_sessions = len(sample['sessions'])
                max_interactions = max([len(sess) for sess in sample['sessions']]) if sample['sessions'] else 0

                # 만약 이 샘플을 추가한다면 업데이트될 최대값들
                candidate_max_sessions = max(current_max_sessions, num_sessions)
                candidate_max_interactions = max(current_max_interactions, max_interactions)
                candidate_batch_size = len(current_batch) + 1
                candidate_cost = candidate_batch_size * candidate_max_sessions * candidate_max_interactions

                # 현재 배치에 최소 한 개 샘플이 있는 경우에 한해 cost 초과 시 배치를 마감
                if candidate_cost > self.batch_th and current_batch:
                    self.batches.append(current_batch)
                    current_batch = [idx]
                    current_max_sessions = num_sessions
                    current_max_interactions = max_interactions
                else:
                    current_batch.append(idx)
                    current_max_sessions = candidate_max_sessions
                    current_max_interactions = candidate_max_interactions

            if current_batch:
                self.batches.append(current_batch)
        
        if self.shuffle_batches:
            random.shuffle(self.batches)

    def __iter__(self):
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)
    

def get_dataloaders(args):
    interaction_path = os.path.join(args.dataset_folder, args.dataset_name, "interactions_revised.json")
    metadata_path = os.path.join(args.dataset_folder, args.dataset_name, "item_metadata.json")

    dataset = SeqRecDataset(interaction_path, metadata_path, args.use_llm)

    total_len = len(dataset)
    train_len = int(total_len * args.train_ratio)
    val_len = int(total_len * args.val_ratio)
    test_len = total_len - train_len - val_len

    train_set, val_set, test_set = random_split(
        dataset, [train_len, val_len, test_len], generator=torch.Generator().manual_seed(42)
    )

    if getattr(args, 'use_bucket_batching', True):
        train_sampler = BucketBatchSampler(train_set, batch_th=args.train_batch_th, batch_size=args.train_batch_size, shuffle_batches=True)
        val_sampler = BucketBatchSampler(val_set, batch_th=args.val_batch_th, batch_size=args.val_batch_size, shuffle_batches=False)
        test_sampler = BucketBatchSampler(test_set, batch_th=args.test_batch_th, batch_size=args.test_batch_size, shuffle_batches=False)
        
        train_loader = DataLoader(train_set, batch_sampler=train_sampler, collate_fn=lambda x: seq_collate_fn(x, use_llm=args.use_llm))
        val_loader = DataLoader(val_set, batch_sampler=val_sampler, collate_fn=lambda x: seq_collate_fn(x, use_llm=args.use_llm))
        test_loader = DataLoader(test_set, batch_sampler=test_sampler, collate_fn=lambda x: seq_collate_fn(x, use_llm=args.use_llm))
    else:
        train_loader = DataLoader(train_set, batch_size=args.train_batch_size, shuffle=True, collate_fn=lambda x: seq_collate_fn(x, use_llm=args.use_llm))
        val_loader = DataLoader(val_set, batch_size=args.val_batch_size, shuffle=False, collate_fn=lambda x: seq_collate_fn(x, use_llm=args.use_llm))
        test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, collate_fn=lambda x: seq_collate_fn(x, use_llm=args.use_llm))

    return train_loader, val_loader, test_loader
