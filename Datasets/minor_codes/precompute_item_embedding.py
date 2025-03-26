import json
import pickle
import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

def mean_pooling(model_output, attention_mask):
    """
    모델 출력의 token 임베딩에 대해 attention mask를 고려한 평균 풀링.
    """
    token_embeddings = model_output[0]  # 모든 토큰 임베딩
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, dim=1) / torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

def compute_and_save_item_embeddings(metadata_path, output_path, hf_model_path, batch_size, device):
    # item_metadata.json 파일 로드
    with open(metadata_path, 'r', encoding='utf-8') as f:
        item_metadata = json.load(f)
    
    # 모델과 토크나이저 초기화
    tokenizer = AutoTokenizer.from_pretrained(hf_model_path)
    model = AutoModel.from_pretrained(hf_model_path)
    model.to(device)
    model.eval()
    
    item_ids = list(item_metadata.keys())
    # 각 item의 정보를 문자열로 변환 (필요 시 전처리 가능)
    sentences = [str(item_metadata[item_id]) for item_id in item_ids]
    
    embeddings = {}
    for i in tqdm(range(0, len(sentences), batch_size), desc="Computing embeddings"):
        batch_sentences = sentences[i:i+batch_size]
        encoded_input = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors='pt')
        # 모델과 동일한 device로 이동
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        with torch.no_grad():
            model_output = model(**encoded_input)
        batch_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
        batch_embeddings = batch_embeddings.cpu().numpy()
        for j, item_id in enumerate(item_ids[i:i+batch_size]):
            embeddings[item_id] = batch_embeddings[j]
    
    # pickle로 저장 (효율적)
    with open(output_path, 'wb') as f:
        pickle.dump(embeddings, f)
    
    return embeddings

def main():
    parser = argparse.ArgumentParser(description="Precompute item embeddings using a sentence-transformer model")
    #parser.add_argument('--dataset_name', type=str, required=True, help='Dataset name (folder under ./Datasets)')
    parser.add_argument('--hf_model_path', type=str, default='sentence-transformers/all-MiniLM-L6-v2', help='HuggingFace model path')
    parser.add_argument('--batch_size', type=int, default=4096, help='Batch size for embedding computation')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use (e.g., "cuda:0", "cuda:1", etc.)')
    args = parser.parse_args()
    
    dataset_name = 'Ratail_Rocket'
    metadata_path = f'./{dataset_name}/item_metadata.json'
    output_path = f'./{dataset_name}/item_embedding_LLM.pickle'
    
    print(f"Loading metadata from {metadata_path}")
    print(f"Saving embeddings to {output_path}")
    
    compute_and_save_item_embeddings(metadata_path, output_path, args.hf_model_path, args.batch_size, args.device)
    print("Item embeddings computation and saving completed.")
if __name__ == '__main__':
    main()
