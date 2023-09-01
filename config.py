"""Configuration file"""
import argparse
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--model_checkpoint", type=str, default="google/mt5-small", help="Path, url or short name of the model")
    parser.add_argument("--model_name", type=str, default="mt5-small", help="use t5 or bart?")
    parser.add_argument("--saving_dir", type=str, default="./experiments/", help="Path for saving")
    parser.add_argument("--sub_model", type=str, default="")
    parser.add_argument("--load_checkpoint", action="store_true")
    parser.add_argument('-p', '--postfix', default='', dest='postfix', help='postfix of checkpoint path')
    parser.add_argument('--task', type=str , default='rg', help="use rg or dst?")
    parser.set_defaults(load_checkpoint=False)

    parser.add_argument("--train_batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--meta_batch_size", type=int, default=1, help="Batch size for meta training")
    parser.add_argument("--dev_batch_size", type=int, default=16, help="Batch size for validation")
    parser.add_argument("--test_batch_size", type=int, default=16, help="Batch size for test")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Accumulate gradients on several steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--n_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=557, help="Random seed")
    parser.add_argument("--verbose", action='store_true', help="continual baseline")
    parser.add_argument("--length", type=int, default=50, help="Batch size for validation")
    parser.add_argument("--GPU", type=int, default=2, help="number of gpu to use")

    # data related
    parser.add_argument('--datause', type=str , default='en')
    parser.add_argument("--fe", action='store_true') # run F&E dataset
    parser.add_argument("--fewshot_data", type=str, default='es')
    parser.add_argument("--fewshot", type=int, default=0)
    parser.add_argument("--test_ratio", type=float, default=0.993)
    parser.add_argument("--max_history", type=int, default=10, help="max number of turns in the dialogue") # 5

    # special setting
    parser.add_argument("--t5_span_pretrain", action='store_true')
    parser.add_argument("--translate_train", action='store_true')
    parser.add_argument("--nmt_pretrain", action='store_true')
    parser.add_argument("--pretrain_seq2seq", action='store_true')
    

    # contrastive learning
    parser.add_argument("--adv", action="store_true")
    parser.set_defaults(adv=False)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--hidden_size", type=int, default=1024) # mt5-base: 768, mt5-small: 512
    parser.add_argument("--devices", default='0', type=str,
                        help="gpu device ids to use, concat with '_', ex) '0_1_2_3'")
    parser.add_argument("--world_size", default=1,
                        help="Number of total workers. Initial value should be set to the number of nodes."
                             "Final value will be Num.nodes * Num.devices")

    args = parser.parse_args()
    opt2 = Config()
    
    return args, opt2
    
class Config:

    # Woz data
    paths = {'en': 'raw_dataset/GlobalWOZ/globalwoz_v2/en-source/multiwoz.json',
             'es': 'raw_dataset/GlobalWOZ/globalwoz_v2/es/F&F_es.json',
             'id': 'raw_dataset/GlobalWOZ/globalwoz_v2/id/F&F_id.json',
             'zh': 'raw_dataset/GlobalWOZ/globalwoz_v2/zh/F&F_zh.json'}
    FE_paths = {'es': 'raw_dataset/GlobalWOZ/globalwoz_v2/es/FE/F&E_es.json',
                'id': 'raw_dataset/GlobalWOZ/globalwoz_v2/id/FE/F&E_id.json',
                'zh': 'raw_dataset/GlobalWOZ/globalwoz_v2/zh/FE/F&E_zh.json'}
    translate_paths = {'es': 'raw_dataset/GlobalWOZ/globalwoz_v2/es/translate_es',
                       'id': 'raw_dataset/GlobalWOZ/globalwoz_v2/id/translate_id',
                       'zh': 'raw_dataset/GlobalWOZ/globalwoz_v2/zh/translate_zh'}
    entities_paths = {'es': 'raw_dataset/GlobalWOZ/globalwoz_v2/es/dialogue_act_F&F_es.json',
                      'id': 'raw_dataset/GlobalWOZ/globalwoz_v2/id/dialogue_act_F&F_id.json',
                      'zh': 'raw_dataset/GlobalWOZ/globalwoz_v2/zh/dialogue_act_F&F_zh.json'}
    adapter_paths = {'en': "experiments/pretrain-clm-mbart-largemmt-en/adapter/",
                     'es': "experiments/pretrain-clm-mbart-largemmt-es/adapter/",
                     'id': "experiments/pretrain-clm-mbart-largemmt-id/adapter/",
                     'zh': "experiments/pretrain-clm-mbart-largemmt-zh/adapter/"}
    
    # language mapping
    lang_map = {'en': 'en_XX',
                'es': 'es_XX',
                'id': 'id_ID',
                'zh': 'zh_CN'}

opt = Config()

