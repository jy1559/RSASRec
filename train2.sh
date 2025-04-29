#!/bin/bash
# run_experiments.sh
# 여러 실험을 순차적으로 실행하는 스크립트입니다.
# 첫 번째 인자가 --wandb_off이면 wandb_flag에 넣음
wandb_flag=""
if [ "$2" = "off" ]; then
    wandb_flag="--wandb_off"
    echo "WandB logging turned off for experiments."
fi

echo "실험 시작..."


# 실험 2: 기본 데이터셋(예: Globo), candidate_size 64, use_amp, accumulation_steps 4, device cuda:1
echo "실험 1 실행: Globo에서 train_strategy EachSession_Except_First"
python train.py --device $1 --test_strategy AllInter_ExceptFirst --use_amp --candidate_size 64 --accumulation_steps 1 $wandb_flag --train_strategy EachSession_Except_First --train_batch_th 100000

echo "실험 3 실행: Globo에서 test때 global_candidate 미사용"
python train.py --device $1 --test_strategy AllInter_ExceptFirst --use_amp --candidate_size 64 --accumulation_steps 1 $wandb_flag --test_global_candidate False  --train_batch_th 750000

echo "실험 5 실행: Globo에서 train_strategy AllInter_ExceptFirst"
python train.py --device $1 --test_strategy AllInter_ExceptFirst --use_amp --candidate_size 64 --accumulation_steps 1 $wandb_flag --train_strategy AllInter_ExceptFirst  --train_batch_th 100000



# 실험 1: LFM-BeyMS 데이터셋, candidate_size 64, use_amp, accumulation_steps 4, device cuda:1
#echo "실험 2 실행: LFM-BeyMS 데이터셋 lr 2e-4, test_strategy AllInter_ExceptFirst"
#python train.py --device $1 --candidate_size 64 --dataset_name LFM-BeyMS --use_amp --accumulation_steps 16 $wandb_flag --test_strategy AllInter_ExceptFirst

# 실험 3: Retail rocket 데이터셋, candidate_size 64, use_amp, accumulation_steps 4, device cuda:1
#echo "실험 3 실행: Retail rocket 데이터셋"
#python train.py --device $1 --candidate_size 64 --dataset_name Retail_Rocket --use_amp --accumulation_steps 4 $wandb_flag --test_strategy AllInter_ExceptFirst

# 실험 4: 기본 데이터셋(예: Globo), candidate_size 64, use_amp, accumulation_steps 4, device cuda:1 모든거거
#echo "실험 4 실행: 기본 데이터셋 (Globo)"
#python train.py --device $1 --candidate_size 64 --train_strategy AllInter_ExceptFirst --use_amp --accumulation_steps 4 $wandb_flag --test_strategy AllInter_ExceptFirst

# 실험 5: LFM-BeyMS 데이터셋, candidate_size 64, use_amp, accumulation_steps 4, device cuda:1
#echo "실험 5 실행: LFM-BeyMS 데이터셋"
#python train.py --device $1 --candidate_size 64 --train_strategy AllInter_ExceptFirst --dataset_name LFM-BeyMS --use_amp --accumulation_steps 16 $wandb_flag --test_strategy AllInter_ExceptFirst

# 실험 6: Retail rocket 데이터셋, candidate_size 64, use_amp, accumulation_steps 4, device cuda:1
#echo "실험 6 실행: Retail rocket 데이터셋"
#python train.py --device $1 --candidate_size 64 --train_strategy AllInter_ExceptFirst --dataset_name Retail_Rocket --use_amp --accumulation_steps 4 $wandb_flag --test_strategy AllInter_ExceptFirst
echo "모든 실험 완료!"