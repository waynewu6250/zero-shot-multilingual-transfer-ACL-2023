######### baseline #########
# training #
python train_t5.py \
                   --sub_model='-cont-fs' \
                   --datause en \
                   --fewshot_data id \
                   --fewshot 100 \
                   --postfix _fewshot_id_rg200 \
                   --GPU 1 \
                   --n_epochs 5 \
                   --mode train \
                   --model_checkpoint 'facebook/mbart-large-50-many-to-many-mmt' \
                   --model_name "mbart-largemmt" \
                   --nmt_pretrain \
                   --seed 200

# testing #
python train_t5.py \
                   --sub_model='-cont-fs' \
                   --datause id \
                   --postfix _fewshot_id_rg200 \
                   --mode test \
                   --model_checkpoint 'facebook/mbart-large-50-many-to-many-mmt' \
                   --model_name "mbart-largemmt"



######### XDFusion #########
# first-round adapter #
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_t5.py \
                   --sub_model='-adp-fs' \
                   --datause en \
                   --fewshot_data id \
                   --fewshot 100 \
                   --postfix _fewshot_id_rg \
                   --GPU 4 \
                   --n_epochs 5 \
                   --mode train \
                   --model_checkpoint 'facebook/mbart-large-50-many-to-many-mmt' \
                   --model_name "mbart-largemmt" \
                   --nmt_pretrain \
                   --pretrain_seq2seq \
                   --seed 100

# second-round adapter #
CUDA_VISIBLE_DEVICES=0,1 python train_t5.py \
                   --sub_model='-adp-fs' \
                   --datause en \
                   --fewshot_data id \
                   --fewshot 100 \
                   --postfix _expert_id \
                   --GPU 2 \
                   --n_epochs 60 \
                   --mode train \
                   --model_checkpoint 'facebook/mbart-large-50-many-to-many-mmt' \
                   --model_name "mbart-largemmt" \
                   --nmt_pretrain \
                   --seed 100

# testing #
CUDA_VISIBLE_DEVICES=0 python train_t5.py \
                   --sub_model='-adp-fs' \
                   --datause id \
                   --postfix _expert_id \
                   --mode test \
                   --model_checkpoint 'facebook/mbart-large-50-many-to-many-mmt' \
                   --model_name "mbart-largemmt"