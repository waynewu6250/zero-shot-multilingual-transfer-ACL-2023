"""Data pipeline for GlobalWOZ dataset and CCMatrix parallel data"""
import json
from lib2to3.pgen2.pgen import DFAState
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset, Sampler
from transformers import T5Tokenizer, MBart50Tokenizer
# from transformers.models.t5.modeling_flax_t5 import shift_tokens_right
import random
from functools import partial
import numpy as np
from collections import defaultdict

from data.gwoz_preprocess import preprocessGWOZ
from data.gwoz_preprocess_template import preprocessGWOZ_template
from datasets import load_dataset
from utils import *

# 1. iterate over all dialogs
# 2. iterate over all turns, add dialog history
# 3.        iterate over all slots and add value to output
# 4.        save into a dictionary for each data

class GWOZDataset(Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, args, opt, raw_data, tokenizer, batch_size, type, cc_data=[], translate_data=None):
        """Reads source and target sequences from txt files."""
        self.args = args
        self.opt = opt
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.n_cc = 2

        gwoz_data = self.get_e2e_from_dial(args, opt, raw_data, tokenizer, type)
        if translate_data:
            gwoz_data_trans = self.get_e2e_from_dial(args, opt, translate_data, tokenizer, type)
            gwoz_data = list(zip(gwoz_data, gwoz_data_trans))
        
        # Simple concatenation
        self.data = gwoz_data + cc_data
        
        # Sampling strategy: make sure every batch will have at least n cc samples
        # if type == 'train':
        #     random.Random(557).shuffle(gwoz_data)
        
        #     self.data = []
        #     gwoz_iter = iter(gwoz_data)
        #     cc_iter = iter(cc_data)
            
        #     while True:
        #         try:
        #             batch = [next(gwoz_iter) for _ in range(batch_size-self.n_cc)] + [next(cc_iter) for _ in range(self.n_cc)] # create a batch
        #             self.data.extend(batch)
        #         except StopIteration:
        #             break
        # else:
        #     self.data = gwoz_data

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        item_info = self.data[index]
        return item_info

    def __len__(self):
        return len(self.data)

    def get_e2e_from_dial(self, args, opt, raw_data, tokenizer, type):
        data = []

        # read data
        # if type=="train" and args["fewshot"]>0:
        #     random.Random(args["seed"]).shuffle(raw_data)
        #     raw_data = raw_data[:int(len(raw_data)*args["fewshot"])]
        
        for dial_dict in raw_data:
            plain_history = []
            data_detail = {}
            latest_API_OUT = "API-OUT: "

            # Reading data
            for turn_id, t in enumerate(dial_dict["dialogue"]):
                
                # Turn user
                if t['spk']=="USER":
                    plain_history.append(f"{t['spk']}: {t['utt'].strip()}")
                elif t['spk']=="API-OUT":
                    latest_API_OUT = f"{t['spk']}: {t['utt'].strip()}"
                elif((t['spk'] == "SYSTEM") and turn_id!=0 and t["utt"]!= ""):
                    data_detail = {"dial_id":t["id"],
                                   "turn_id":t["turn_id"],
                                   "spk":t["spk"],
                                   "dataset":t["dataset"],
                                   "dialog_history": " ".join(plain_history[-args["max_history"]:]),
                                   "turn_belief": latest_API_OUT,
                                   "intput_text": " ".join(plain_history[-args["max_history"]:] + [latest_API_OUT]),
                                   "output_text": f'{t["utt"].strip()} {tokenizer.eos_token}',
                                   "type": "gwoz"
                                   }
                    data.append(data_detail)
                    plain_history.append(f"{t['spk']}: {t['utt'].strip()}")
                    latest_API_OUT = "API-OUT: "
                elif((t['spk'] == "API") and turn_id!=0 and t["utt"]!= ""):
                    if args['task'] == 'dst':
                        data_detail = {"dial_id":t["id"],
                                    "turn_id":str(t["turn_id"])+"API",
                                    "spk":t["spk"],
                                    "dataset":t["dataset"],
                                    "dialog_history": " ".join(plain_history[-args["max_history"]:]),
                                    "turn_belief": latest_API_OUT,
                                    "intput_text": " ".join(plain_history[-args["max_history"]:]),
                                    "output_text": f'{t["utt"].strip()} {tokenizer.eos_token}',
                                    "type": "gwoz"
                                    }
                        data.append(data_detail)
                    else:
                        pass
                
                # sanity check
                # for k,v in data_detail.items():
                #     print(k,': ',v)
                # print()
                # print('-------')
                
        return data

def filter_services(data,serv):
    """Filter out services not existing in training data."""
    filtered_dialogue = []
    for dial in data:
        flag_temp = True
        for turn in dial['dialogue']:
            if(turn["spk"]=="API"):
                for s in turn["service"]:
                    if s not in serv:
                        flag_temp = False
        if(flag_temp):
            filtered_dialogue.append(dial)
    return filtered_dialogue

def get_domains_slots(data):
    """Get ontology of domain, slots and services from data."""
    services = set()
    intent = set()
    len_dialogue = []
    for dial in data:
        for s in dial["services"]:
            services.add(s)
        len_dialogue.append(len([0 for t in dial['dialogue'] if t["spk"] in ["USER","SYSTEM"]]))
        for turn in dial['dialogue']:
            if(turn["spk"]=="API"):
                for s in turn["service"]:
                    if(" " in s or len(s)==1):
                        print(s) 
                        print(turn)
                        input()
                    intent.add(s)
    print("Domain",len(services))
    print("Intent",len(intent))
    print("Avg. Turns",np.mean(len_dialogue))
    return len(services), len(intent), np.mean(len_dialogue), intent
        
def collate_fn(data, tokenizer, tokenizer2, fewshot_data_type):
    """Normal collate function to batchify data."""
    
    batch_data_gwoz = defaultdict(list)
    batch_data_cc = defaultdict(list)
    for d in data:
        if d['type'] == 'gwoz':
            for key in d:
                batch_data_gwoz[key].append(d[key])
        else:
            for key in d:
                batch_data_cc[key].append(d[key])
    
    ########### gwoz ###########
    if 'intput_text' in batch_data_gwoz and len(batch_data_gwoz['intput_text']) != 0:
        input_batch = tokenizer(batch_data_gwoz["intput_text"], padding=True, truncation=True, max_length=300, return_tensors="pt", add_special_tokens=True, verbose=False)
        
        batch_data_gwoz["encoder_input"] = input_batch["input_ids"]
        batch_data_gwoz["attention_mask"] = input_batch["attention_mask"]
        output_batch = tokenizer(batch_data_gwoz["output_text"], padding=True, return_tensors="pt", add_special_tokens=True, return_attention_mask=False)
        # replace the padding id to -100 for cross-entropy
        output_batch['input_ids'].masked_fill_(output_batch['input_ids']==tokenizer.pad_token_id, -100)
        batch_data_gwoz["decoder_output"] = output_batch['input_ids']
    
    ########### mc4 ###########
    if 'en' in batch_data_cc and len(batch_data_cc['en']) != 0:
        
        en_batch = tokenizer(batch_data_cc["en"], padding='max_length', truncation=True, max_length=50, return_tensors="pt", verbose=False)
        with tokenizer.as_target_tokenizer():
            fo_batch = tokenizer(batch_data_cc[fewshot_data_type], padding='max_length', truncation=True, max_length=50, return_tensors="pt", verbose=False)
        en_input_ids = en_batch["input_ids"]
        en_attention_mask = en_batch["attention_mask"]
        fo_input_ids = fo_batch["input_ids"]
        fo_attention_mask = fo_batch["attention_mask"]
        
    
    ########### merge ###########
    batch_clean = {}
    batch_clean["encoder_input_gwoz"] = batch_data_gwoz["encoder_input"] if "encoder_input" in batch_data_gwoz else None
    batch_clean["attention_mask_gwoz"] = batch_data_gwoz["attention_mask"] if "attention_mask" in batch_data_gwoz else None
    batch_clean["decoder_output_gwoz"] = batch_data_gwoz["decoder_output"] if "decoder_output" in batch_data_gwoz else None
    # batch_clean["encoder_input_cc"] = batch_data_cc["encoder_input"] if "encoder_input" in batch_data_cc else None
    # batch_clean["attention_mask_cc"] = batch_data_cc["attention_mask"] if "attention_mask" in batch_data_cc else None
    # batch_clean["decoder_input_ids_cc"] = batch_data_cc["decoder_input_ids"] if "decoder_input_ids" in batch_data_cc else None
    # batch_clean["decoder_output_cc"] = batch_data_cc["decoder_output"] if "decoder_output" in batch_data_cc else None
    
    batch_clean["en_encoder_input_cc"] = en_input_ids if 'en' in batch_data_cc else None
    batch_clean["en_attention_mask_cc"] = en_attention_mask if 'en' in batch_data_cc else None
    batch_clean["fo_encoder_input_cc"] = fo_input_ids if 'en' in batch_data_cc else None
    batch_clean["fo_attention_mask_cc"] = fo_attention_mask if 'en' in batch_data_cc else None
    
    if "encoder_input" in batch_data_gwoz:
        # only for testing use
        for key in batch_data_gwoz:
            batch_clean[key] = batch_data_gwoz[key]
            
    return batch_clean

def collate_fn_translate_train(data, tokenizer, tokenizer_trans, fewshot_data_type):
    """Collate function for translation train setting."""
    
    batch_data_gwoz = defaultdict(list)
    batch_data_gwoz_trans = defaultdict(list)
    batch_data_cc = defaultdict(list)
    for d in data:
        if type(d) == tuple:
            d_en, d_fo = d
            for key in d_en:
                batch_data_gwoz[key].append(d_en[key])
                batch_data_gwoz_trans[key].append(d_fo[key])
        else:
            for key in d:
                batch_data_cc[key].append(d[key])
    
    ########### gwoz ###########
    if 'intput_text' in batch_data_gwoz and len(batch_data_gwoz['intput_text']) != 0:
        input_batch = tokenizer(batch_data_gwoz["intput_text"], padding=True, truncation=True, max_length=300, return_tensors="pt", add_special_tokens=True, verbose=False)
        
        batch_data_gwoz["encoder_input"] = input_batch["input_ids"]
        batch_data_gwoz["attention_mask"] = input_batch["attention_mask"]
        output_batch = tokenizer(batch_data_gwoz["output_text"], padding=True, return_tensors="pt", add_special_tokens=True, return_attention_mask=False)
        # replace the padding id to -100 for cross-entropy
        output_batch['input_ids'].masked_fill_(output_batch['input_ids']==tokenizer.pad_token_id, -100)
        batch_data_gwoz["decoder_output"] = output_batch['input_ids']
        
        # translate data
        input_batch = tokenizer_trans(batch_data_gwoz_trans["intput_text"], padding=True, truncation=True, max_length=300, return_tensors="pt", add_special_tokens=True, verbose=False)
        
        batch_data_gwoz_trans["encoder_input"] = input_batch["input_ids"]
        batch_data_gwoz_trans["attention_mask"] = input_batch["attention_mask"]
        output_batch = tokenizer_trans(batch_data_gwoz_trans["output_text"], padding=True, return_tensors="pt", add_special_tokens=True, return_attention_mask=False)
        # replace the padding id to -100 for cross-entropy
        output_batch['input_ids'].masked_fill_(output_batch['input_ids']==tokenizer_trans.pad_token_id, -100)
        batch_data_gwoz_trans["decoder_output"] = output_batch['input_ids']
        
        # sanity check
        # print(tokenizer.batch_decode(batch_data_gwoz["encoder_input"], skip_special_tokens=False)[0])
        # print(tokenizer.batch_decode(batch_data_gwoz_trans["encoder_input"], skip_special_tokens=False)[0])
   
    ########### mc4 ###########
    if 'en' in batch_data_cc and len(batch_data_cc['en']) != 0:
        
        en_batch = tokenizer(batch_data_cc["en"], padding='max_length', truncation=True, max_length=50, return_tensors="pt", verbose=False)
        with tokenizer.as_target_tokenizer():
            fo_batch = tokenizer(batch_data_cc[fewshot_data_type], padding='max_length', truncation=True, max_length=50, return_tensors="pt", verbose=False)
        en_input_ids = en_batch["input_ids"]
        en_attention_mask = en_batch["attention_mask"]
        fo_input_ids = fo_batch["input_ids"]
        fo_attention_mask = fo_batch["attention_mask"]
    
    ########### merge ###########
    batch_clean = {}
    batch_clean["encoder_input_gwoz"] = batch_data_gwoz["encoder_input"] if "encoder_input" in batch_data_gwoz else None
    batch_clean["attention_mask_gwoz"] = batch_data_gwoz["attention_mask"] if "attention_mask" in batch_data_gwoz else None
    batch_clean["decoder_output_gwoz"] = batch_data_gwoz["decoder_output"] if "decoder_output" in batch_data_gwoz else None
    
    batch_clean["encoder_input_gwoz_trans"] = batch_data_gwoz_trans["encoder_input"] if "encoder_input" in batch_data_gwoz else None
    batch_clean["attention_mask_gwoz_trans"] = batch_data_gwoz_trans["attention_mask"] if "attention_mask" in batch_data_gwoz else None
    batch_clean["decoder_output_gwoz_trans"] = batch_data_gwoz_trans["decoder_output"] if "decoder_output" in batch_data_gwoz else None
    
    batch_clean["en_encoder_input_cc"] = en_input_ids if 'en' in batch_data_cc else None
    batch_clean["en_attention_mask_cc"] = en_attention_mask if 'en' in batch_data_cc else None
    batch_clean["fo_encoder_input_cc"] = fo_input_ids if 'en' in batch_data_cc else None
    batch_clean["fo_attention_mask_cc"] = fo_attention_mask if 'en' in batch_data_cc else None
    
    if "encoder_input" in batch_data_gwoz:
        # only for testing use
        for key in batch_data_gwoz:
            batch_clean[key] = batch_data_gwoz[key]
            
    return batch_clean

def prepare_data_nmt(args, opt, data_type, fewshot_data_type=None):
    """Main data preparation function."""

    # Get tokenizer
    if "t5" in args["model_name"]:
        print('Use t5 tokenizer...')
        tokenizer = T5Tokenizer.from_pretrained(args["model_checkpoint"], bos_token="[bos]", eos_token="[eos]", sep_token="[sep]")
        tokenizer_trans = tokenizer
    elif "bart" in args["model_name"]:
        print('Use bart tokenizer...')
        if args['translate_train']:
            tokenizer = MBart50Tokenizer.from_pretrained(args["model_checkpoint"], src_lang=opt.lang_map[args['datause']], tgt_lang=opt.lang_map[args['fewshot_data']])
            tokenizer_trans = MBart50Tokenizer.from_pretrained(args["model_checkpoint"], src_lang=opt.lang_map[args['fewshot_data']], tgt_lang=opt.lang_map[args['fewshot_data']])
        else:
            tokenizer = MBart50Tokenizer.from_pretrained(args["model_checkpoint"], src_lang=opt.lang_map[args['datause']], tgt_lang=opt.lang_map[args['fewshot_data']])
            tokenizer2 = MBart50Tokenizer.from_pretrained(args["model_checkpoint"], src_lang=opt.lang_map[args['fewshot_data']], tgt_lang=opt.lang_map[args['datause']])
            # special_tokens_dict = {'additional_special_tokens': ['[DST]','[RG]','[NMT]']}
            # num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
            # num_added_toks = tokenizer2.add_special_tokens(special_tokens_dict)

    # Get global woz data
    if args['translate_train']:
        print('Translate data into {}...'.format(fewshot_data_type))
        train_trans, dev_trans, test_trans = preprocessGWOZ_template(opt.paths[data_type], 
                                                   opt.translate_paths[fewshot_data_type], 
                                                   data_type, 
                                                   fewshot_data_type, 
                                                   args["test_ratio"], 
                                                   preload=False, 
                                                   seed=args["seed"], 
                                                   enable_fe=args["fe"])
        train, dev, test = preprocessGWOZ(opt.paths[data_type], args["test_ratio"], args["seed"])
    else:
        test_path = opt.FE_paths[data_type] if args["mode"] == 'test' and args["fe"] else opt.paths[data_type]
        train, dev, test = preprocessGWOZ(test_path, args["test_ratio"], args["seed"])
        train_trans, dev_trans, test_trans = None, None, None
        # train = train+train_tmp
        # dev = dev+dev_tmp
        # test = test+test_tmp
        
    if args['fewshot'] > 0:
        fewshot_train, fewshot_dev, _ = preprocessGWOZ(opt.paths[fewshot_data_type], args["test_ratio"], args["seed"])
        random.Random(557).shuffle(fewshot_train) # Fix fewshot data to train for reproducibility
        random.Random(557).shuffle(fewshot_dev) # Fix fewshot data to train for reproducibility
        train = train + fewshot_train[:args['fewshot']]
        dev = dev + fewshot_dev[:args['fewshot']]

    n_domain, n_intent, n_turns, services = get_domains_slots(train)
    dev = filter_services(dev,services) ## Remove dialogue with API not present in the train set
    test = filter_services(test,services) ## Remove dialogue with API not present in the train set
    if args['translate_train']:
        dev_trans = filter_services(dev_trans,services)
        test_trans = filter_services(test_trans,services)
    
    # Get ccmatrix data
    amount = 123225 * 0.01 # total dialogue data
    cc = load_dataset("yhavinga/ccmatrix", name=data_type+"-"+fewshot_data_type, split='train', streaming=True)
    # mc4 = load_dataset("mc4", fewshot_data_type, split='train', streaming=True)
    cc_data = defaultdict(list)
    for i, instance in enumerate(cc):
        if i <= int(amount*0.9):
            cc_data['train'].append({'type': 'ccmatrix', data_type: instance['translation'][data_type], fewshot_data_type: instance['translation'][fewshot_data_type]})
        elif i <= amount:
            cc_data['dev'].append({'type': 'ccmatrix', data_type: instance['translation'][data_type], fewshot_data_type: instance['translation'][fewshot_data_type]})
        else:
            break

    train_dataset = GWOZDataset(args, opt, train, tokenizer, args["train_batch_size"], "train", cc_data['train'], translate_data=train_trans)
    dev_dataset = GWOZDataset(args, opt, dev, tokenizer, args["dev_batch_size"], "dev", cc_data['dev'], translate_data=dev_trans)
    test_dataset = GWOZDataset(args, opt, test, tokenizer, args["test_batch_size"], "test")

    print('Train: {}, Dev: {}, Test: {}'.format(len(train_dataset),len(dev_dataset),len(test_dataset)))

    if args['translate_train']:
        train_loader = DataLoader(train_dataset, batch_size=args["train_batch_size"], shuffle=True, collate_fn=partial(collate_fn_translate_train, tokenizer=tokenizer, tokenizer_trans=tokenizer_trans, fewshot_data_type=fewshot_data_type), num_workers=4)
        dev_loader = DataLoader(dev_dataset, batch_size=args["dev_batch_size"], shuffle=False, collate_fn=partial(collate_fn_translate_train, tokenizer=tokenizer, tokenizer_trans=tokenizer_trans, fewshot_data_type=fewshot_data_type), num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=args["test_batch_size"], shuffle=False, collate_fn=partial(collate_fn, tokenizer=tokenizer, tokenizer2=tokenizer_trans, fewshot_data_type=fewshot_data_type), num_workers=4)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args["train_batch_size"], shuffle=True, collate_fn=partial(collate_fn, tokenizer=tokenizer, tokenizer2=tokenizer2, fewshot_data_type=fewshot_data_type), num_workers=4)
        dev_loader = DataLoader(dev_dataset, batch_size=args["dev_batch_size"], shuffle=False, collate_fn=partial(collate_fn, tokenizer=tokenizer, tokenizer2=tokenizer2, fewshot_data_type=fewshot_data_type), num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=args["test_batch_size"], shuffle=False, collate_fn=partial(collate_fn, tokenizer=tokenizer, tokenizer2=tokenizer2, fewshot_data_type=fewshot_data_type), num_workers=4)

    return train_loader, dev_loader, test_loader


def prepare_data_test_train(args, opt, data_type):
    """Main data preparation function for test-time training."""
    
    tokenizer = MBart50Tokenizer.from_pretrained(args["model_checkpoint"], src_lang=opt.lang_map['en'], tgt_lang=opt.lang_map[args['datause']])
    tokenizer_trans = MBart50Tokenizer.from_pretrained(args["model_checkpoint"], src_lang=opt.lang_map[args['datause']], tgt_lang=opt.lang_map[args['datause']])
    
    test_path = opt.FE_paths[data_type] if args["mode"] == 'test' and args["fe"] else opt.paths[data_type]
    train, dev, test = preprocessGWOZ(test_path, args["test_ratio"], args["seed"])
    
    n_domain, n_intent, n_turns, services = get_domains_slots(train)
    dev = filter_services(dev,services) ## Remove dialogue with API not present in the train set
    test = filter_services(test,services) ## Remove dialogue with API not present in the train set
    
    test_train = test[:int(0.9*len(test))]
    test_dev = test[int(0.9*len(test)):]
    
    # Get ccmatrix data
    amount = 123225 * 0.1 # total dialogue data
    cc = load_dataset("yhavinga/ccmatrix", name="en-"+data_type, split='train', streaming=True)
    # mc4 = load_dataset("mc4", fewshot_data_type, split='train', streaming=True)
    cc_data = defaultdict(list)
    for i, instance in enumerate(cc):
        if i <= int(amount*0.5):
            cc_data['train'].append({'type': 'ccmatrix', 'en': instance['translation']['en'], data_type: instance['translation'][data_type]})
        elif i <= amount:
            cc_data['dev'].append({'type': 'ccmatrix', 'en': instance['translation']['en'], data_type: instance['translation'][data_type]})
        else:
            break
    
    test_train_dataset = GWOZDataset(args, opt, test_train, tokenizer, "train", cc_data['train'])
    test_dev_dataset = GWOZDataset(args, opt, test_dev, tokenizer, "dev", cc_data['dev'])
    
    print('Train: {}, Dev: {}'.format(len(test_train_dataset),len(test_dev_dataset)))
    
    test_train_loader = DataLoader(test_train_dataset, batch_size=args["train_batch_size"], shuffle=False, collate_fn=partial(collate_fn, tokenizer=tokenizer, tokenizer2=tokenizer_trans, fewshot_data_type=data_type), num_workers=4)
    test_dev_loader = DataLoader(test_dev_dataset, batch_size=args["dev_batch_size"], shuffle=False, collate_fn=partial(collate_fn, tokenizer=tokenizer, tokenizer2=tokenizer_trans, fewshot_data_type=data_type), num_workers=4)
    
    return test_train_loader, test_dev_loader
    