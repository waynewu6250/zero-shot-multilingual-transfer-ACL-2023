# template for testing different models
CUDA_VISIBLE_DEVICES=1 python train_t5.py --task dst \
                   --sub_model='-cont' \
                   --datause es \
                   --postfix _normal_frz \
                   --mode test \
                   --model_checkpoint 'google/mt5-small' \
                   --model_name "mt5-small"

CUDA_VISIBLE_DEVICES=1 python train_t5.py --task dst \
                   --sub_model='-zh-dst-100' \
                   --datause zh \
                   --mode test \
                   --model_checkpoint 'google/mt5-base' \
                   --model_name "mt5-base"
                   
CUDA_VISIBLE_DEVICES=4 python train_t5.py --task dst \
                   --sub_model='-adp' \
                   --datause es \
                   --postfix _fewshot_es_no_dst \
                   --mode test \
                   --model_checkpoint 'facebook/mbart-large-50' \
                   --model_name "mbart-large"

CUDA_VISIBLE_DEVICES=1 python train_t5.py --task dst \
                   --sub_model='-en-dst' \
                   --datause es \
                   --postfix _normal \
                   --mode test \
                   --model_checkpoint 'facebook/mbart-large-50-many-to-many-mmt' \
                   --model_name "mbart-largemmt"

CUDA_VISIBLE_DEVICES=1 python train_t5.py --task dst \
                   --sub_model='-adp-fs' \
                   --datause es \
                   --postfix _fewshot_es_second_rg_cont \
                   --mode test \
                   --model_checkpoint 'facebook/mbart-large-50-many-to-many-mmt' \
                   --model_name "mbart-largemmt"

CUDA_VISIBLE_DEVICES=1 python train_t5.py --task dst \
                   --sub_model='-cont-fs' \
                   --datause es \
                   --postfix _normal_es_rg \
                   --mode test \
                   --model_checkpoint 'facebook/mbart-large-50-many-to-many-mmt' \
                   --model_name "mbart-largemmt"

CUDA_VISIBLE_DEVICES=4 python train_t5.py --task dst \
                   --sub_model='-mem-fs' \
                   --datause es \
                   --postfix _mem_sp_imp_es \
                   --mode test \
                   --model_checkpoint 'facebook/mbart-large-50-many-to-many-mmt' \
                   --model_name "mbart-largemmt" \
                   --fe