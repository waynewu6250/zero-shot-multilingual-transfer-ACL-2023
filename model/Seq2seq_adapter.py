"""T5 baseline model"""
import pytorch_lightning as pl
import torch
from transformers import T5ForConditionalGeneration, AdamW
from transformers import AdapterConfig
from transformers.adapters.composition import Stack
import transformers.adapters.composition as ac
import torch.nn as nn
import os

class Dialog_Seq2seq_adp(pl.LightningModule):

    def __init__(self, args, opt, tokenizer, model):
        super().__init__()
        self.args = args
        self.opt = opt
        self.tokenizer = tokenizer
        self.model = model
        self.model.config.output_hidden_states = True
        # self.load_adapter()
        # self.load_task_adapter()
        if not args["pretrain_seq2seq"]:
            self.load_mixture_of_experts()
        self.lr = args["lr"]
        self.tau = args['tau']
    
    def forward(self, input_ids, attention_mask, labels, decoder_input_ids=None):
        outputs = self.model(input_ids=input_ids,
                          attention_mask=attention_mask,
                          labels=labels,
                          decoder_input_ids=decoder_input_ids)
        return outputs.loss
        
    
    def load_adapter(self):
        # Load the language adapters
        lang_adapter_config = AdapterConfig.load("pfeiffer", reduction_factor=2)
        self.model.load_adapter(self.opt.adapter_paths["en"], config=lang_adapter_config)
        
        # Add a new task adapter
        self.model.add_adapter("dialog")
        
        # Train dialog adapter only
        if self.args['mode'] == 'train':
            self.model.load_adapter(self.opt.adapter_paths[self.args['fewshot_data']], config=lang_adapter_config)
            self.model.train_adapter(["dialog"])
            self.model.active_adapters = Stack("en", "dialog")
        else:
            self.model.load_adapter(self.opt.adapter_paths[self.args['datause']], config=lang_adapter_config)
            # # Unfreeze and activate stack setup
            # self.model.active_adapters = "dialog"
            self.model.active_adapters = Stack(self.args['datause'], "dialog")
    
    def load_task_adapter(self):
        lang_adapter_config = AdapterConfig.load("pfeiffer", reduction_factor=2)
    
    def load_mixture_of_experts(self):
        ######## Baseline setting with no AdapterFusion ########
        # self.model.add_adapter("dialog")
        # self.model.train_adapter(['dialog'])
        # self.model.active_adapters = 'dialog'
        
        # Training
        if self.args['mode'] == 'train':
            lang_adapter_config = AdapterConfig.load("pfeiffer", reduction_factor=2)
            self.model.load_adapter(self.opt.adapter_paths[self.args['datause']], config=lang_adapter_config)
            self.model.load_adapter(self.opt.adapter_paths[self.args['fewshot_data']], config=lang_adapter_config)
            self.model.add_adapter_fusion([self.args['datause'], self.args['fewshot_data']])
            self.model.active_adapters = ac.Fuse(self.args['datause'], self.args['fewshot_data'])
            
            self.model.train_adapter_fusion(ac.Fuse(self.args['datause'], self.args['fewshot_data']))
        
        # Testing
        else:
            lang_adapter_config = AdapterConfig.load("pfeiffer", reduction_factor=2)
            self.model.load_adapter(self.opt.adapter_paths['en'], config=lang_adapter_config)
            self.model.load_adapter(self.opt.adapter_paths[self.args['datause']], config=lang_adapter_config)
            self.model.add_adapter_fusion(['en', self.args['datause']])
            self.model.active_adapters = ac.Fuse('en', self.args['datause'])
        

    def training_step(self, batch, batch_idx):
        
        opt = self.optimizers()
        
        self.model.train()
        loss = 0
        if self.args['translate_train']:
            loss += self.forward(input_ids=batch["encoder_input_gwoz_trans"],
                            attention_mask=batch["attention_mask_gwoz_trans"],
                            labels=batch["decoder_output_gwoz_trans"]
                            ).loss
        else:
                
            if "encoder_input_gwoz" in batch and batch["encoder_input_gwoz"] != None:
                loss += self.forward(input_ids=batch["encoder_input_gwoz"],
                                attention_mask=batch["attention_mask_gwoz"],
                                labels=batch["decoder_output_gwoz"]
                                )
            if "encoder_input_mc4" in batch and batch["encoder_input_mc4"] != None:
                loss += self.forward(input_ids=batch["encoder_input_mc4"],
                                attention_mask=batch["attention_mask_mc4"],
                                labels=batch["decoder_output_mc4"]
                            ).loss
            if "encoder_input_cc" in batch and batch["encoder_input_cc"] != None:
                loss += self.forward(input_ids=batch["encoder_input_cc"],
                                attention_mask=batch["attention_mask_cc"],
                                decoder_input_ids=batch["decoder_input_ids_cc"],
                                labels=batch["decoder_output_cc"]
                            ).loss
        
        return {'loss': loss, 'log': {'train_loss': loss}}

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        loss = 0
        if self.args['translate_train']:
            loss += self.forward(input_ids=batch["encoder_input_gwoz_trans"],
                            attention_mask=batch["attention_mask_gwoz_trans"],
                            labels=batch["decoder_output_gwoz_trans"]
                            ).loss
        else:
            if "encoder_input_gwoz" in batch and batch["encoder_input_gwoz"] != None:
                loss += self.forward(input_ids=batch["encoder_input_gwoz"],
                                attention_mask=batch["attention_mask_gwoz"],
                                labels=batch["decoder_output_gwoz"]
                                )
            if "encoder_input_mc4" in batch and batch["encoder_input_mc4"] != None:
                loss += self.forward(input_ids=batch["encoder_input_mc4"],
                                attention_mask=batch["attention_mask_mc4"],
                                labels=batch["decoder_output_mc4"]
                            ).loss
            if "encoder_input_cc" in batch and batch["encoder_input_cc"] != None:
                loss += self.forward(input_ids=batch["encoder_input_cc"],
                                attention_mask=batch["attention_mask_cc"],
                                decoder_input_ids=batch["decoder_input_ids_cc"],
                                labels=batch["decoder_output_cc"]
                            ).loss


        return {'val_loss': loss, 'log': {'val_loss': loss}}
        # return result

    def validation_epoch_end(self, outputs):
        val_loss_mean = sum([o['val_loss'] for o in outputs]) / len(outputs)
        # show val_loss in progress bar but only log val_loss
        results = {'progress_bar': {'val_loss': val_loss_mean.item()}, 'log': {'val_loss': val_loss_mean.item()},
                   'val_loss': val_loss_mean.item()}
        return results
    
    def predict_step(self, batch, batch_idx):
        dst_outputs = self.model.generate(input_ids=batch["encoder_input"],
                                    attention_mask=batch["attention_mask"],
                                    eos_token_id=self.tokenizer.eos_token_id,
                                    do_sample=True,
                                    max_length=200,
                                    )
        dst_outputs = self.tokenizer.batch_decode(dst_outputs, skip_special_tokens=True)
        return dst_outputs, batch['uid']

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr, correct_bias=True)

    ##########################################################################################
    
    def _shift_right(self, input_ids):
        decoder_start_token_id = self.tokenizer.bos_token_id
        pad_token_id = self.tokenizer.pad_token_id

        assert decoder_start_token_id is not None, (
            "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id."
            " See T5 docs for more information"
        )

        # shift inputs to the right
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id

        assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        assert torch.all(shifted_input_ids >= 0).item(), "Verify that `shifted_input_ids` has only positive values"

        return shifted_input_ids

    def avg_pool(self, hidden_states, mask):
        length = torch.sum(mask, 1, keepdim=True).float()
        mask = mask.unsqueeze(2)
        hidden = hidden_states.masked_fill(mask == 0, 0.0)
        avg_hidden = torch.sum(hidden, 1) / length

        return avg_hidden
   




