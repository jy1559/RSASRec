import argparse
import time
from Mar2025_Module.Datasets.dataset import get_dataloaders
from Mar2025_Module.models.model import SeqRecModel
import pickle
from util import get_batch_item_ids, get_candidate_set_for_batch
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_folder", default="/home/jy1559/Mar2025_Module/Datasets")
parser.add_argument("--dataset_name", default="Globo")
parser.add_argument("--train_ratio", default=0.8, type=float)
parser.add_argument("--val_ratio", default=0.1, type=float)
parser.add_argument("--train_batch_size", default=4, type=int)
parser.add_argument("--val_batch_size", default=8, type=int)
parser.add_argument("--test_batch_size", default=8, type=int)
args = parser.parse_args()
args.use_bucket_batching = True

st = time.time()
train_loader, val_loader, test_loader = get_dataloaders(args)
print(f"Data load: {st-time.time()}초")
st = time.time()
model = SeqRecModel()
with open(f"/home/jy1559/Mar2025_Module/Datasets/{args.dataset_name}/item_emedding.pickle","rb") as fr:
    data = pickle.load(fr)
print(f"Pickle load: {st-time.time()}초")
st = time.time()
# 사용 예시
print(f"model: {model}")
for batch in train_loader:
    print(batch['item_id'])
    labels = get_batch_item_ids(batch['item_id'])
    candidate = get_candidate_set_for_batch(labels, data)
    print(len(candidate))
    for x in candidate:
        print(x[0])
    
    
    #print(batch['embedding_sentences'][0])  # 첫 user 데이터 예시
    #print(batch['interaction_mask'])     # mask 확인
    break
