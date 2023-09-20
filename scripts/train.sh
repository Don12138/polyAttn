#!/bin/bash
SEED=12138
LOAD_FROM=""
# MODEL=polygnn
# TASK=reaction_prediction
EXP_NO=polygnn_single_ea_no_ho_add_weight
DATASET=copolymer_4w
MPN_TYPE=polygnn
# MAX_REL_POS=4
ACCUM_COUNT=4
# ENC_PE=none
# ENC_H=256
BATCH_SIZE=64
# ENC_EMB_SCALE=sqrt
MAX_STEP=3000
# ENC_LAYER=4
# BATCH_TYPE=tokens
# REL_BUCKETS=11


# REL_POS=emb_only
# ATTN_LAYER=6
LR=0.1
# DROPOUT=0.3

# REPR_START=smiles
# REPR_END=smiles

PREFIX=${DATASET}_${MODEL}


/home/chenlidong/.conda/envs/py_38_torch_113_pyg/bin/python train.py \
  --mpn_type="$MPN_TYPE" \
  --exp_no=$EXP_NO\
  --data_name="$DATASET" \
  --load_from="$LOAD_FROM" \
  --log_file="$PREFIX.train.$EXP_NO.log" \
  --save_dir="./checkpoints/$PREFIX.$EXP_NO" \
  --seed=$SEED \
  --epoch=2000 \
  --max_steps="$MAX_STEP" \
  --warmup_steps=10 \
  --lr="$LR" \
  --weight_decay=0.0 \
  --clip_norm=20.0 \
  --train_batch_size="$BATCH_SIZE" \
  --valid_batch_size="$BATCH_SIZE" \
  --predict_batch_size="$BATCH_SIZE" \
  --accumulation_count="$ACCUM_COUNT" \
  --num_workers=0 \
  --log_iter=20 \
  --eval_iter=200 \
  --save_iter=1000 \
