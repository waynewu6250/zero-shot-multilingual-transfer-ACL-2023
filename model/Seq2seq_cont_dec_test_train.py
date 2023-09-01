"""T5 baseline model"""
import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, AdamW
import os

class Dialog_Seq2seq_cont(pl.LightningModule):

    def __init__(self, args, tokenizer, model):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.model = model
        # for param in self.model.decoder.parameters():
        #     param.requires_grad = False
        self.lr = args["lr"]
        self.tau = args['tau']

        self.projection = nn.Sequential(nn.Linear(self.args['hidden_size'], self.args['hidden_size']),
                                        nn.ReLU())
        self.projection2 = nn.Sequential(nn.Linear(self.args['hidden_size'], self.args['hidden_size']),
                                        nn.ReLU())
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.cont_crit = nn.CrossEntropyLoss()
    
    def forward(self, batch):
        
        cont2_loss = 0
        
        if "encoder_input_gwoz" in batch and batch["encoder_input_gwoz"] != None:
            input_ids=batch["encoder_input_gwoz"]
            attention_mask=batch["attention_mask_gwoz"]
            labels=batch["decoder_output_gwoz"]
            
            # Get encoder outputs
            encoder = self.model.get_encoder()
            decoder = self.model.get_decoder()
            encoder_outputs_fo = encoder(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        )[0]
            decoder_outputs = decoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_outputs_fo,
                encoder_attention_mask=attention_mask
            )
            sequence_output = decoder_outputs[0]
        
        if "en_encoder_input_cc" in batch and batch["en_encoder_input_cc"] != None:
            en_ids_cc = batch["en_encoder_input_cc"] 
            en_attention_mask_cc = batch["en_attention_mask_cc"] 
            fo_ids_cc = batch["fo_encoder_input_cc"] 
            fo_attention_mask_cc = batch["fo_attention_mask_cc"] 
        
            # External data
            encoder_outputs_en_cc = encoder(input_ids=en_ids_cc,
                                        attention_mask=en_attention_mask_cc,
                                        )[0]

            encoder_outputs_fo_cc = encoder(input_ids=fo_ids_cc,
                                        attention_mask=fo_attention_mask_cc,
                                        )[0]
            
            # Contrastive Learning on decoder side
            proj_neg_h = self.projection2(encoder_outputs_en_cc)
            proj_cc_h = self.projection2(encoder_outputs_fo_cc)
            proj_dg_h = self.projection2(sequence_output)
            avg_neg = self.avg_pool(proj_neg_h, en_attention_mask_cc) # (bt, h)
            avg_cc = self.avg_pool(proj_cc_h, fo_attention_mask_cc) # (bt, h)
            avg_dg = self.avg_pool(proj_dg_h, attention_mask) # (bd, h)
            
            new_vec = torch.cat([avg_cc[0, :].unsqueeze(0), avg_neg[1:, :]], dim=0) # Positive sample is the first.
            
            cos = nn.CosineSimilarity(dim=-1)
            sim_matrix = cos(avg_dg.unsqueeze(1), new_vec.unsqueeze(0)) # (bd, bt)
            logits = sim_matrix / self.tau
            batch_size = sequence_output.size(0)
            labels = torch.zeros(batch_size, device=sequence_output.device).long()
            cont2_loss = self.cont_crit(logits, labels)
        
        return cont2_loss
        

    def training_step(self, batch, batch_idx):
        self.model.train()
        if "encoder_input_gwoz" in batch and batch["encoder_input_gwoz"] != None:
            loss = self.forward(batch)

        # result = pl.TrainResult(loss)
        # result.log('train_loss', loss, on_epoch=True)
        return {'loss': loss, 'log': {'train_loss': loss}}
        # return result

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        if "encoder_input_gwoz" in batch and batch["encoder_input_gwoz"] != None:
            loss = self.forward(batch)

        return {'val_loss': loss, 'log': {'val_loss': loss}}
        # return result

    def validation_epoch_end(self, outputs):
        val_loss_mean = sum([o['val_loss'] for o in outputs if o['val_loss']]) / len(outputs)
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
   




