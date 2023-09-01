"""T5 baseline model"""
import pytorch_lightning as pl
import torch
from transformers import T5ForConditionalGeneration, AdamW
import os

class Dialog_Seq2seq(pl.LightningModule):

    def __init__(self, args, tokenizer, model):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.model = model
        # for name, param in self.model.named_parameters():
        #     if 'decoder' in name or 'embed_positions' in name or 'shared' in name:
        #         param.requires_grad = False
        self.lr = args["lr"]


    def training_step(self, batch, batch_idx):
        self.model.train()
        loss = 0
        if "encoder_input_gwoz" in batch and batch["encoder_input_gwoz"] != None:
            if self.args['translate_train']:
                input_ids = batch["encoder_input_gwoz_trans"]
                attention_mask = batch["attention_mask_gwoz_trans"]
                labels = batch["decoder_output_gwoz_trans"]
            else:
                input_ids = batch["encoder_input_gwoz"]
                attention_mask = batch["attention_mask_gwoz"]
                labels = batch["decoder_output_gwoz"]
            loss += self.model(input_ids=input_ids,
                               attention_mask=attention_mask,
                               labels=labels
                               ).loss
        if "encoder_input_mc4" in batch and batch["encoder_input_mc4"] != None:
            loss += self.model(input_ids=batch["encoder_input_mc4"],
                            attention_mask=batch["attention_mask_mc4"],
                            labels=batch["decoder_output_mc4"]
                          ).loss
        if "encoder_input_cc" in batch and batch["encoder_input_cc"] != None:
            loss += self.model(input_ids=batch["encoder_input_cc"],
                            attention_mask=batch["attention_mask_cc"],
                            decoder_input_ids=batch["decoder_input_ids_cc"],
                            labels=batch["decoder_output_cc"]
                          ).loss

        # result = pl.TrainResult(loss)
        # result.log('train_loss', loss, on_epoch=True)
        return {'loss': loss, 'log': {'train_loss': loss}}
        # return result

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        loss = 0
        if "encoder_input_gwoz" in batch and batch["encoder_input_gwoz"] != None:
            if self.args['translate_train']:
                input_ids = batch["encoder_input_gwoz_trans"]
                attention_mask = batch["attention_mask_gwoz_trans"]
                labels = batch["decoder_output_gwoz_trans"]
            else:
                input_ids = batch["encoder_input_gwoz"]
                attention_mask = batch["attention_mask_gwoz"]
                labels = batch["decoder_output_gwoz"]
            loss += self.model(input_ids=input_ids,
                               attention_mask=attention_mask,
                               labels=labels
                               ).loss
        if "encoder_input_mc4" in batch and batch["encoder_input_mc4"] != None:
            loss += self.model(input_ids=batch["encoder_input_mc4"],
                            attention_mask=batch["attention_mask_mc4"],
                            labels=batch["decoder_output_mc4"]
                          ).loss
        if "encoder_input_cc" in batch and batch["encoder_input_cc"] != None:
            loss += self.model(input_ids=batch["encoder_input_cc"],
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
   




