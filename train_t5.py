"""Main training pipeline"""
import os, random
from tabnanny import check
import torch
from tqdm import tqdm
import numpy as np
import json
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.cloud_io import load as pl_load
from transformers import MT5ForConditionalGeneration, T5Tokenizer, MBartForConditionalGeneration, MBart50Tokenizer
import re
import math
import gzip

from dataset_woz import prepare_data
from dataset_nmt import prepare_data_nmt, prepare_data_test_train
from dataset_nmt_fs import prepare_data_nmt_fs
from model.Seq2seq import Dialog_Seq2seq
from model.Seq2seq_adapter import Dialog_Seq2seq_adp
from model.Seq2seq_mem import Dialog_Seq2seq_mem

########### Contrastive Learning ###########
# from model.Seq2seq_cont import Dialog_Seq2seq_cont
# from model.Seq2seq_cont_dec import Dialog_Seq2seq_cont
# from model.Seq2seq_cont_dec_test_train import Dialog_Seq2seq_cont
from model.Seq2seq_cont_fs import Dialog_Seq2seq_cont
# from model.Seq2seq_cont_fs_dec import Dialog_Seq2seq_cont

from evaluate import cal_entropy, cal_kl, cal_rg
from config import get_args

# import torch.multiprocessing as mp
# import torch.distributed as dist
# >>> import torch
# >>> random_seed = 1 # or any of your favorite number 
# >>> torch.manual_seed(random_seed)
# >>> torch.cuda.manual_seed(random_seed)
# >>> torch.backends.cudnn.deterministic = True
# >>> torch.backends.cudnn.benchmark = False
# >>> import numpy as np
# >>> np.random.seed(random_seed)

def train(args, opt):
    """Main train function

    Args:
        args: arguments
        opt: configurations
    """

    print(torch.distributed.is_available())
    print(torch.cuda.is_available())

    # Set up arguments and seed
    args = vars(args)
    seed_everything(args["seed"])

    # data
    print('Data use for main training: ', args['datause'])
    if args['fewshot'] > 0:
        print('Data use for few-shot training: ', args['fewshot_data'])
    if args['t5_span_pretrain']:
        print('Data use for t5 span pretraining: ', args['fewshot_data'])
    if args['translate_train']:
        print('Data use for translate train: ', args['fewshot_data'])
    
    # select data loader
    if args['nmt_pretrain'] and args['fewshot'] > 0:
        print('Few-shot data setting...')
        train_loader, val_loader, test_loader = prepare_data_nmt_fs(args, opt, args['datause'], fewshot_data_type=args['fewshot_data'])
    elif args['nmt_pretrain']:
        print('Contrastive learning setting...')
        print('Data use for nmt train: {}-{}'.format(args['datause'], args['fewshot_data']))
        train_loader, val_loader, test_loader = prepare_data_nmt(args, opt, args['datause'], fewshot_data_type=args['fewshot_data'])
    else:
        print('Normal setting...')
        train_loader, val_loader, test_loader = prepare_data(args, opt, args['datause'], fewshot_data_type=args['fewshot_data'])

    # model
    checkpoint_path = args["model_checkpoint"]
    # save model path
    save_path = os.path.join(args["saving_dir"],args["model_name"]+args["sub_model"])
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    load_model_path = checkpoint_path
    
    if "t5" in args["model_name"]:
        from transformers import MT5Config
        config = MT5Config.from_pretrained(load_model_path)
        config.attention_probs_dropout_prob = 0.1
        model = MT5ForConditionalGeneration.from_pretrained(load_model_path, config=config)
        tokenizer = T5Tokenizer.from_pretrained(load_model_path, bos_token="[bos]", eos_token="[eos]", sep_token="[sep]")
        model.resize_token_embeddings(new_num_tokens=len(tokenizer))
    elif "bart" in args["model_name"]:
        from transformers import MBartConfig
        config = MBartConfig.from_pretrained(load_model_path)
        # config.encoder_layers=12
        # config.decoder_layers=1
        model = MBartForConditionalGeneration.from_pretrained(load_model_path, config=config)
        if args['translate_train']:
            tokenizer = MBart50Tokenizer.from_pretrained(args["model_checkpoint"], src_lang=opt.lang_map[args['fewshot_data']], tgt_lang=opt.lang_map[args['fewshot_data']])
        else:
            tokenizer = MBart50Tokenizer.from_pretrained(args["model_checkpoint"], src_lang=opt.lang_map[args['datause']], tgt_lang=opt.lang_map[args['datause']])

    if args["load_checkpoint"]:
        # find the last checkpoint path
        logs_path = os.listdir(os.path.join(save_path, 'lightning_logs'))
        ver_num = max([folder.strip('version_') for folder in logs_path if 'version' in folder])
        path = os.path.join(save_path, 'lightning_logs')+'/version_'+ver_num+'/checkpoints'
        checkpoint_path = os.path.join(path, os.listdir(path)[-1])
        print('Load checkpoint from path: ', checkpoint_path)
    else:
        checkpoint_path = None
    
    # T5 model with frequency and contrastive loss
    if "-adp" in args["sub_model"] or args["sub_model"] == "-test":
        # first time on English dialog training
        if args["pretrain_seq2seq"]:
            task = Dialog_Seq2seq_adp(args, opt, tokenizer, model)
        # second time on Foreign dialog training
        else:
            checkpoint_path = 'experiments/{}-adp-fs/pytorch_model_fewshot_{}_rg.ckpt'.format(args['model_name'], args['fewshot_data'])
            task = Dialog_Seq2seq_adp.load_from_checkpoint(checkpoint_path, args=args, opt=opt, tokenizer=tokenizer, model=model, strict=False) # strict=False
        
    elif "-cont" in args["sub_model"]:
        task = Dialog_Seq2seq_cont(args, tokenizer, model)
    elif "-mem" in args["sub_model"]:
        task = Dialog_Seq2seq_mem(args, tokenizer, model)
    else:
        task = Dialog_Seq2seq(args, tokenizer, model)
    
    # Add Callbacks
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=30)
    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(save_path, args["postfix"]),
                                          filename="best_model",
                                          save_top_k=1,
                                          verbose=True,
                                          monitor="val_loss",
                                          mode="min")
    # checkpoint_path = os.path.join(save_path, "pytorch_model{}.ckpt".format(args["postfix"]))
    trainer = Trainer(
                default_root_dir=save_path,
                accumulate_grad_batches=args["gradient_accumulation_steps"],
                gradient_clip_val=args["max_norm"],
                max_epochs=args["n_epochs"],
                # callbacks=[early_stopping_callback, checkpoint_callback],
                gpus=args["GPU"],
                deterministic=True,
                num_nodes=1,
                #precision=16,
                accelerator="ddp",
                # resume_from_checkpoint=checkpoint_path
                # automatic_optimization=False
                )
    trainer.fit(task, train_loader, val_loader)
    print('save checkpoint...')
    checkpoint_path = os.path.join(save_path, "pytorch_model{}.ckpt".format(args["postfix"]))
    trainer.save_checkpoint(checkpoint_path)
    # task.model.save_pretrained(save_path)
    task.tokenizer.save_pretrained(save_path)

    tokenizer_to_evaluate = task.tokenizer
    model_to_evaluate = task.model

    print("Complete saving model.")
    
    # Uncomment to test directly after training the model
    # print("test start...")
    # evaluate model
    # _ = evaluate_model(args, tokenizer_to_evaluate, model_to_evaluate, test_loader, save_path)

def test_train(args, opt):
    """Test-time training: https://arxiv.org/pdf/1909.13231.pdf
    (Deprecated; not useful)

    Args:
        args: arguments
        opt: configurations
    """
    
    print(torch.distributed.is_available())
    print(torch.cuda.is_available())

    # Set up arguments and seed
    args = vars(args)
    seed_everything(args["seed"])
    
    # data
    print('Data use for main training: ', args['datause'])
    print('Data use for nmt train: en-{}'.format(args['datause']))
    test_train_loader, test_dev_loader = prepare_data_test_train(args, opt, args['datause'])
    
    # model
    save_path = os.path.join(args["saving_dir"],args["model_name"]+args["sub_model"])
    checkpoint_path = os.path.join(save_path, "pytorch_model{}.ckpt".format(args["postfix"]))
    model = MBartForConditionalGeneration.from_pretrained(args["model_checkpoint"])
    tokenizer = MBart50Tokenizer.from_pretrained(save_path, src_lang=opt.lang_map[args['datause']], tgt_lang=opt.lang_map[args['datause']])
    
    task = Dialog_Seq2seq_cont.load_from_checkpoint(checkpoint_path, args=args, tokenizer=tokenizer, model=model)
    
    trainer = Trainer(
                default_root_dir=save_path,
                accumulate_grad_batches=args["gradient_accumulation_steps"],
                gradient_clip_val=args["max_norm"],
                max_epochs=args["n_epochs"],
                # callbacks=[pl.callbacks.EarlyStopping(monitor='val_loss',min_delta=0.00, patience=5,verbose=False, mode='min')],
                gpus=args["GPU"],
                deterministic=True,
                num_nodes=1,
                #precision=16,
                accelerator="ddp"
                )
    trainer.fit(task, test_train_loader, test_dev_loader)
    print('save checkpoint...')
    checkpoint_path = os.path.join(save_path, "pytorch_model{}_test_train.ckpt".format(args["postfix"]))
    trainer.save_checkpoint(checkpoint_path)
    # task.model.save_pretrained(save_path)
    task.tokenizer.save_pretrained(save_path)
    

def test(args, opt):
    """Main test function

    Args:
        args: arguments
        opt: configurations
    """

    print(torch.distributed.is_available())
    print(torch.cuda.is_available())

    # Set up arguments and seed
    args = vars(args)
    seed_everything(args["seed"])

    # data
    print('Data use for testing: ', args['datause'])
    train_loader, val_loader, test_loader = prepare_data(args, opt, args['datause'])
    
    # model
    save_path = os.path.join(args["saving_dir"],args["model_name"]+args["sub_model"])
    
    # Use which checkpoint to test 1) last model 2) new best model
    checkpoint_path = os.path.join(save_path, "pytorch_model{}.ckpt".format(args["postfix"]))
    # checkpoint_path = os.path.join(save_path, args["postfix"], 'best_model-v0.ckpt')
    
    if "t5" in args["model_name"]:
        from transformers import MT5Config
        config = MT5Config.from_pretrained(args["model_checkpoint"])
        config.attention_probs_dropout_prob = 0.1
        model = MT5ForConditionalGeneration.from_pretrained(args["model_checkpoint"], config=config)
        tokenizer = T5Tokenizer.from_pretrained(save_path, bos_token="[bos]", eos_token="[eos]", sep_token="[sep]")
        model.resize_token_embeddings(new_num_tokens=len(tokenizer))
    elif "bart" in args["model_name"]:
        from transformers import MBartConfig
        config = MBartConfig.from_pretrained(args["model_checkpoint"])
        # config.encoder_layers=12
        # config.decoder_layers=1
        model = MBartForConditionalGeneration.from_pretrained(args["model_checkpoint"], config=config)
        tokenizer = MBart50Tokenizer.from_pretrained(save_path, src_lang=opt.lang_map[args['datause']], tgt_lang=opt.lang_map[args['datause']])
        ######## translate-test baseline ########
        # prev_model = MBartForConditionalGeneration.from_pretrained(args["model_checkpoint"])
        # prev_tokenizer = MBart50Tokenizer.from_pretrained(save_path, src_lang=opt.lang_map['en'], tgt_lang=opt.lang_map['en'])
    
    if "-adp" in args["sub_model"] or args["sub_model"] == "-test":
        print('Model: adapter...')
        task = Dialog_Seq2seq_adp.load_from_checkpoint(checkpoint_path, args=args, opt=opt, tokenizer=tokenizer, model=model)
        model = task.model
    elif "-cont" in args["sub_model"]:
        print('Model: contrastive...')
        task = Dialog_Seq2seq_cont.load_from_checkpoint(checkpoint_path, args=args, tokenizer=tokenizer, model=model)
        model = task.model
    elif "-mem" in args["sub_model"]:
        task = Dialog_Seq2seq_mem.load_from_checkpoint(checkpoint_path, args=args, tokenizer=tokenizer, model=model)
        model = task.model
    else:
        task = Dialog_Seq2seq.load_from_checkpoint(checkpoint_path, args=args, tokenizer=tokenizer, model=model)
        model = task.model

    # save model path
    print("test start...")
    
    #evalutask
    _ = evaluate_model(args, tokenizer, task, model, test_loader, save_path)#, prev_model=prev_model, prev_tokenizer=prev_tokenizer)

def evaluate_model(args, tokenizer, task, model, test_loader, save_path, prefix="zeroshot"):#, prev_model=None, prev_tokenizer=None):
    """Main evaluation function

    Args:
        args: arguments
        tokenizer: tokenizer to use
        model: model to use
        test_loader: test dataloader
        save_path: where to save model and results
        vocab: vocabulary of HypRank model
        prefix (str, optional): . Defaults to "zeroshot".
    """
    
    # save path
    save_path = os.path.join(save_path,"results/")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    task.to(device)
    model.to(device)
    model.eval()
    # prev_model.to(device)
    # prev_model.eval()

    results = []
    total_words = 0
    total_eval_loss = 0

    for batch in tqdm(test_loader):
        ######## translate-test baseline ########
        # en_input = prev_model.generate(input_ids=batch["encoder_input_gwoz"].to(device),
        #                             attention_mask=batch["attention_mask_gwoz"].to(device),
        #                             eos_token_id=tokenizer.eos_token_id,
        #                             forced_bos_token_id=tokenizer.lang_code_to_id[opt.lang_map['en']],
        #                             do_sample=False,
        #                             num_beams=1,
        #                             max_length=200,
        #                             )
        # en_input_batch = prev_tokenizer.batch_decode(en_input, skip_special_tokens=True)
        # input_batch = prev_tokenizer(en_input_batch, padding=True, truncation=True, max_length=300, return_tensors="pt", add_special_tokens=True, verbose=False)
        # responses = model.generate(input_ids=input_batch["input_ids"].to(device),
        #                             attention_mask=input_batch["attention_mask"].to(device),
        #                             eos_token_id=tokenizer.eos_token_id,
        #                             forced_bos_token_id=tokenizer.lang_code_to_id[opt.lang_map['en']],
        #                             max_length=200
        #                             )
        # value_batch = tokenizer.batch_decode(responses, skip_special_tokens=True)
        
        if "t5" in args["model_name"]:
            responses = model.generate(input_ids=batch["encoder_input_gwoz"].to(device),
                                    attention_mask=batch["attention_mask_gwoz"].to(device),
                                    eos_token_id=tokenizer.eos_token_id,
                                    # do_sample=True,
                                    max_length=200,
                                    )
        else:
            responses = model.generate(input_ids=batch["encoder_input_gwoz"].to(device),
                                    attention_mask=batch["attention_mask_gwoz"].to(device),
                                    eos_token_id=tokenizer.eos_token_id,
                                    forced_bos_token_id=tokenizer.lang_code_to_id[opt.lang_map[args['datause']]],
                                    # do_sample=False,
                                    # num_beams=1,
                                    max_length=200,
                                    # task=task
                                    )
        # batch wise
        value_batch = tokenizer.batch_decode(responses, skip_special_tokens=True)
            # rerun the pretrained model
            # input_batch = prev_tokenizer(value_batch, padding=True, truncation=True, max_length=300, return_tensors="pt", add_special_tokens=True, verbose=False)
            # responses = prev_model.generate(input_ids=input_batch["input_ids"].to(device),
            #                         attention_mask=input_batch["attention_mask"].to(device),
            #                         eos_token_id=tokenizer.eos_token_id,
            #                         forced_bos_token_id=tokenizer.lang_code_to_id[opt.lang_map[args['datause']]],
            #                         # do_sample=True,
            #                         max_length=200,
            #                         )
            # value_batch = prev_tokenizer.batch_decode(responses, skip_special_tokens=True)
            
        for idx, resp in enumerate(value_batch):
            # print(resp)
            # print(batch["output_text"][idx])
            # print('-------')
            results.append({"dial_id":batch["dial_id"][idx],
                            "turn_id":batch["turn_id"][idx],
                            "spk":batch["spk"][idx],
                            "dataset":batch["dataset"][idx],
                            "input_text":batch["intput_text"][idx],
                            "turn_belief":batch["turn_belief"][idx],
                            "gold":batch["output_text"][idx].replace("</s>","").replace("[eos]",""),
                            "genr":resp})
        # de
        # perplexity
        eval_loss = model(input_ids=batch["encoder_input"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                        labels=batch["decoder_output"].to(device)
                        ).loss

        label_size = torch.sum(torch.sum(batch["decoder_output"] != -100, dim=1).type(eval_loss.type())).item()
        total_eval_loss += eval_loss.item() * label_size
        total_words += label_size
    
    # save results
    with open(save_path+'/generated_responses_{}{}.json'.format(args['datause'], args['postfix']), 'w') as fp:
        json.dump(results, fp, indent=4, ensure_ascii=False)
    
    # evaluate
    evaluate(args, results, np.exp(total_eval_loss / total_words))

def score(args, opt):
    """Score the saved json file directly without running the model inference again.

    Args:
        args: arguments
        opt: configurations
    """

    args = vars(args)

    save_path = os.path.join(args["saving_dir"],args["model_name"]+args["sub_model"])
    save_json = os.path.join(save_path,"results/")+'generated_responses_{}{}.json'.format(args['datause'], args['postfix'])
    if not os.path.exists(save_json):
        raise OSError('File not found: {}'.format(save_json))
    results = json.load(open(save_json))

    # evaluate
    evaluate(args, results)


def evaluate(args, results, perplexity=None):
    """Main evaluate function.

    Args:
        args: arguments
        results: generation outputs
        perplexity: perplexity score of given model. Defaults to None.
    """

    results = cal_rg(results, args["datause"])
    print('SacreBLEU: {:2f}; BLEU1: {:.2f}; BLEU2: {:.2f}'.format(results['BLEU_score']*100, results['BLEU1']*100, results['BLEU2']*100))
    print('Slot error rate: {:.2f}'.format(results['SER']*100))
    if args['mode'] != 'score':
        print('Perplexity: {:.2f}'.format(perplexity))
    if args['task'] == 'dst':
        print('Joint goal accuracy: {:.2f}'.format(results['turn_level_joint_acc']*100))

def representation(args, opt):
    """Show the representation from the encoder and decoder.

    Args:
        args: arguments
        opt: configurations
    """
    
    print(torch.distributed.is_available())
    print(torch.cuda.is_available())

    # Set up arguments and seed
    args = vars(args)
    seed_everything(args["seed"])

    # data
    print('Data use for testing: ', args['datause'])
    train_loader, val_loader, test_loader = prepare_data(args, opt, args['datause'])
    
    # model
    save_path = os.path.join(args["saving_dir"],args["model_name"]+args["sub_model"])
    
    # Use which checkpoint to test 1) last model 2) new best model
    checkpoint_path = os.path.join(save_path, "final/pytorch_model{}.ckpt".format(args["postfix"]))
    # checkpoint_path = os.path.join(save_path, args["postfix"], 'best_model-v0.ckpt')
    
    if "t5" in args["model_name"]:
        model = MT5ForConditionalGeneration.from_pretrained(args["model_checkpoint"])
        tokenizer = T5Tokenizer.from_pretrained(save_path, bos_token="[bos]", eos_token="[eos]", sep_token="[sep]")
        model.resize_token_embeddings(new_num_tokens=len(tokenizer))
    elif "bart" in args["model_name"]:
        model = MBartForConditionalGeneration.from_pretrained(args["model_checkpoint"])
        tokenizer = MBart50Tokenizer.from_pretrained(save_path, src_lang=opt.lang_map[args['datause']], tgt_lang=opt.lang_map[args['datause']])
    
    if "-adp" in args["sub_model"] or args["sub_model"] == "-test":
        print('Model: adapter...')
        task = Dialog_Seq2seq_adp.load_from_checkpoint(checkpoint_path, args=args, opt=opt, tokenizer=tokenizer, model=model)
        model = task.model
    elif "-cont" in args["sub_model"]:
        print('Model: contrastive...')
        task = Dialog_Seq2seq_cont.load_from_checkpoint(checkpoint_path, args=args, tokenizer=tokenizer, model=model)
        model = task.model
    elif "-mem" in args["sub_model"]:
        task = Dialog_Seq2seq_mem.load_from_checkpoint(checkpoint_path, args=args, tokenizer=tokenizer, model=model)
        model = task.model
    else:
        task = Dialog_Seq2seq.load_from_checkpoint(checkpoint_path, args=args, tokenizer=tokenizer, model=model)
        model = task.model
    
    # save path
    save_path = os.path.join(save_path,"results/")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.config.output_hidden_states = True
    task.to(device)
    model.to(device)
    model.eval()
    
    hidden = []
    for batch in tqdm(test_loader):
        hidden_states = model(input_ids=batch["encoder_input"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                        labels=batch["decoder_output"].to(device)
                        ).decoder_hidden_states[-1].mean(dim=1)
        hidden.append(hidden_states.detach().cpu().numpy())
    final_hidden = np.vstack(hidden)
    torch.save(final_hidden, './experiments/repr/{}_dual.pth'.format(args['datause']))



if __name__ == "__main__":
    args, opt = get_args()
    if args.mode=="train":
        train(args, opt)
    elif args.mode =="test_train":
        test_train(args, opt)
    elif args.mode=="test":
        test(args, opt)
    elif args.mode=="score":
        score(args, opt)
    elif args.mode=="repr":
        representation(args, opt)