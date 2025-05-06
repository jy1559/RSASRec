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

:<<'END'
echo "실험 1 실행: 기본 (Globo, global_candidate, train_strategy EachSession_Except_First)"
python train.py --device $1 --test_strategy EachSession_LastInter --use_amp --candidate_size 64 --accumulation_steps 1 $wandb_flag  --train_batch_th 750000

echo "실험 2 실행: Learning rate 0.005로 증가가"
python train.py --device $1 --test_strategy EachSession_LastInter --use_amp --candidate_size 64 --accumulation_steps 1 $wandb_flag --train_batch_th 750000 --lr 0.005

echo "실험 3 실행: Learning rate 0.01로 증가가"
python train.py --device $1 --test_strategy EachSession_LastInter --use_amp --candidate_size 64 --accumulation_steps 1 $wandb_flag --train_batch_th 750000 --lr 0.01

echo "실험 4 실행: train_strategy EachSession_Except_First"
python train.py --device $1 --test_strategy EachSession_LastInter --use_amp --candidate_size 64 --accumulation_steps 1 $wandb_flag --train_strategy EachSession_Except_First --train_batch_th 100000

echo "실험 5 실행: train_strategy EachSession_First_and_Last_Inter"
python train.py --device $1 --test_strategy EachSession_LastInter --use_amp --candidate_size 64 --accumulation_steps 1 $wandb_flag --train_strategy EachSession_First_and_Last_Inter --train_batch_th 300000

echo "실험 6 실행: train_strategy AllInter_ExceptFirst"
python train.py --device $1 --test_strategy EachSession_LastInter --use_amp --candidate_size 64 --accumulation_steps 1 $wandb_flag --train_strategy AllInter_ExceptFirst --train_batch_th 100000

echo "실험 7 실행: accumulateion_steps 4"
python train.py --device $1 --test_strategy EachSession_LastInter --use_amp --candidate_size 64 --accumulation_steps 1 $wandb_flag --train_batch_th 750000


echo "실험 3 실행: EachSession_Except_First Candidate 100 accumulation_steps 4 lr 4e-3"
python train.py --device $1 --test_strategy EachSession_LastInter --use_amp --candidate_size 100 --accumulation_steps 1 $wandb_flag --train_strategy EachSession_Except_First --train_batch_th 200000 --lr 1e-3

echo "실험 4 실행: EachSession_Except_First Candidate 100 batch_th 30000"
python train.py --device $1 --test_strategy EachSession_LastInter --use_amp --candidate_size 100 --accumulation_steps 1 $wandb_flag --train_strategy EachSession_Except_First --train_batch_th 30000 --weight_decay 1e-3 --lr 1e-4

echo "실험 5 실행: EachSession_Except_First Candidate 100 lr 5e-4"
python train.py --device $1 --test_strategy EachSession_LastInter --use_amp --candidate_size 100 --accumulation_steps 1 $wandb_flag --train_strategy EachSession_Except_First --train_batch_th 200000 --lr 2e-3


echo "실험 1 실행: EachSession_Except_First Candidate 100 lr 1e-4"
python train.py --device $1 --test_strategy EachSession_LastInter --use_amp --candidate_size 100 --accumulation_steps 1 $wandb_flag --train_strategy EachSession_Except_First --train_batch_th 200000 --lr 1e-4

echo "실험 2 실행: EachSession_Except_First Candidate 100 lr 5e-5"
python train.py --device $1 --test_strategy EachSession_LastInter --use_amp --candidate_size 100 --accumulation_steps 1 $wandb_flag --train_strategy EachSession_Except_First --train_batch_th 200000 --lr 5e-5
END
echo "실험 1 실행: EachSession_Except_First Candidate 100 lr 1e-3"
python train.py --device $1 --test_strategy EachSession_LastInter --use_amp --candidate_size 100 --accumulation_steps 1 $wandb_flag --train_strategy EachSession_Except_First --train_batch_th 10000 --weight_decay 1e-3 --lr 1e-3

echo "실험 2 실행: EachSession_Except_First Candidate 100 lr 5e-4"
python train.py --device $1 --test_strategy EachSession_LastInter --use_amp --candidate_size 100 --accumulation_steps 1 $wandb_flag --train_strategy EachSession_Except_First --train_batch_th 10000 --weight_decay 1e-3 --lr 2e-3

echo "실험 3 실행: EachSession_Except_First Candidate 100 lr 1e-5"
python train.py --device $1 --test_strategy EachSession_LastInter --use_amp --candidate_size 100 --accumulation_steps 1 $wandb_flag --train_strategy EachSession_Except_First --train_batch_th 30000 --lr 5e-4

echo "실험 4 실행: EachSession_Except_First Candidate 100 wd 1e-3"
python train.py --device $1 --test_strategy EachSession_LastInter --use_amp --candidate_size 100 --accumulation_steps 1 $wandb_flag --train_strategy EachSession_Except_First --train_batch_th 30000 --weight_decay 1e-3 --lr 1e-3

echo "실험 5 실행: EachSession_Except_First Candidate 100 lr 1e-4"
python train.py --device $1 --test_strategy EachSession_LastInter --use_amp --candidate_size 100 --accumulation_steps 1 $wandb_flag --train_strategy EachSession_Except_First --train_batch_th 10000 --weight_decay 1e-3 --lr 1e-3
echo "모든 실험 완료!"