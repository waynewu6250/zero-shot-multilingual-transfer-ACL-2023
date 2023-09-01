"""T5 baseline model"""
import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, AdamW
import os

class Dialog_Seq2seq_mem(pl.LightningModule):

    def __init__(self, args, tokenizer, model):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.model = model
        # for param in self.model.decoder.parameters():
        #     param.requires_grad = False
        self.lr = args["lr"]
        self.tau = args['tau']

        # self.projection = nn.Sequential(nn.Linear(self.args['hidden_size'], self.args['hidden_size']),
        #                                 nn.ReLU())
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.cont_crit = nn.CrossEntropyLoss()
        
        # MemSeq2seq
        # self.random_range = 0.5
        # self.M = 1024
        # self.keys = nn.parameter.Parameter(torch.FloatTensor(self.M, model.config.hidden_size).uniform_(-self.random_range, self.random_range)) # (M, H)
        # self.values = nn.parameter.Parameter(torch.FloatTensor(self.M, model.config.hidden_size).uniform_(-self.random_range, self.random_range)) # (M, H)
        
        # ImpMem
        # self.random_range = 0.5
        # self.M = 1024
        # self.num_heads = 32
        # self.dim_per_head = model.config.hidden_size // self.num_heads
        # self.keys = nn.parameter.Parameter(torch.FloatTensor(self.num_heads, self.M, self.dim_per_head).uniform_(-self.random_range, self.random_range)) # (N, M, NH)
        # self.values = nn.parameter.Parameter(torch.FloatTensor(self.num_heads, self.M, self.dim_per_head).uniform_(-self.random_range, self.random_range)) # (N, M, NH)
        
        # SPMem
        # self.random_range = 0.5
        # self.M = 1024
        # self.keys = nn.ParameterList([nn.parameter.Parameter(torch.FloatTensor(self.M, model.config.hidden_size).uniform_(-self.random_range, self.random_range)) for i in range(3)]) # (3, M, H) (shared, en, fo)
        # self.values = nn.ParameterList([nn.parameter.Parameter(torch.FloatTensor(self.M, model.config.hidden_size).uniform_(-self.random_range, self.random_range)) for i in range(3)]) # (3, M, H) (shared, en, fo)
        
        # SPImpMem
        self.random_range = 0.5
        self.M = 1024
        self.num_heads = 32
        self.dim_per_head = model.config.hidden_size // self.num_heads
        self.keys = nn.ParameterList([nn.parameter.Parameter(torch.FloatTensor(self.num_heads, self.M, self.dim_per_head).uniform_(-self.random_range, self.random_range)) for i in range(3)]) # (N, M, NH)
        self.values = nn.ParameterList([nn.parameter.Parameter(torch.FloatTensor(self.num_heads, self.M, self.dim_per_head).uniform_(-self.random_range, self.random_range)) for i in range(3)]) # (N, M, NH)
        
    def memory_collect_spmem(self, encoder_outputs, shape, choice):
        
        # SPMem
        # weights = torch.mm(encoder_outputs, self.keys[choice].transpose(1,0)) # (B*T, M)
        # weights = nn.Softmax(dim=1)(weights)
        # new_encoder_outputs = torch.mm(weights, self.values[choice])
        # new_encoder_outputs = new_encoder_outputs.view(*shape)
        
        # SPImpMem
        b, t, h = shape
        encoder_outputs = encoder_outputs.view(self.num_heads, b*t, self.dim_per_head) # (N, B*T, NH)
        weights = torch.bmm(encoder_outputs, self.keys[choice].transpose(2,1)) # (N, B*T, M)
        weights = nn.Softmax(dim=2)(weights)
        new_encoder_outputs = torch.bmm(weights, self.values[choice])
        new_encoder_outputs = new_encoder_outputs.view(b, t, h)
        
        return new_encoder_outputs
        
    def get_encoder_outputs(self, input_ids, attention_mask, choice=None):
        
        # Get encoder outputs
        encoder = self.model.get_encoder()
        outputs = encoder(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    )#[0] # (B, T, H)
        encoder_outputs = outputs.last_hidden_state
        
        encoder_shape = encoder_outputs.shape
        b, t, h = encoder_outputs.shape
        encoder_outputs = encoder_outputs.view(b*t, h)
        
        # MemSeq2seq
        # weights = torch.mm(encoder_outputs, self.keys.transpose(1,0)) # (B*T, M)
        # weights = nn.Softmax(dim=1)(weights)
        # new_encoder_outputs = torch.mm(weights, self.values)
        # new_encoder_outputs = new_encoder_outputs.view(b, t, h)
        
        # ImpMemSeq2seq
        # encoder_outputs = encoder_outputs.view(self.num_heads, b*t, self.dim_per_head) # (N, B*T, NH)
        # weights = torch.bmm(encoder_outputs, self.keys.transpose(2,1)) # (N, B*T, M)
        # weights = nn.Softmax(dim=2)(weights)
        # new_encoder_outputs = torch.bmm(weights, self.values)
        # new_encoder_outputs = new_encoder_outputs.view(b, t, h)
        
        # SPMem/SPImpMem
        output_shared = self.memory_collect_spmem(encoder_outputs, encoder_shape, choice=0)
        output_lang = self.memory_collect_spmem(encoder_outputs, encoder_shape, choice=choice)
        new_encoder_outputs = output_shared + output_lang
        
        return new_encoder_outputs
        
    
    def main_forward(self, input_ids, attention_mask, labels, choice=1):
        
        new_encoder_outputs = self.get_encoder_outputs(input_ids, attention_mask, choice)
        decoder = self.model.get_decoder()
            
        # Get fo logits
        decoder_input_ids = self._shift_right(labels)
        decoder_attention_mask = torch.sign(decoder_input_ids)
        # since bos id is 0, change it into 1 (should be attended)
        decoder_attention_mask[:, 0] = 1
        decoder_outputs = decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=new_encoder_outputs,
            encoder_attention_mask=attention_mask
        )
        sequence_output = decoder_outputs[0]
        # sequence_output = sequence_output * (self.model.model_dim ** -0.5)
        lm_logits = self.model.lm_head(sequence_output)
        vocab_size = lm_logits.size(-1)
        nll = self.criterion(lm_logits.view(-1, vocab_size), labels.view(-1))
        
        return nll, sequence_output, decoder_attention_mask
    
    def forward(self, batch):
        
        nll = 0
        
        if "encoder_input_gwoz" in batch and batch["encoder_input_gwoz"] != None:
            loss_en, sequence_output_en, decoder_attention_mask_en = self.main_forward(batch["encoder_input_gwoz"], batch["attention_mask_gwoz"], batch["decoder_output_gwoz"], choice=1)
            nll += loss_en
            
        if "encoder_input_gwoz_fs" in batch and batch["encoder_input_gwoz_fs"] != None:
            loss_fo, sequence_output_fo, decoder_attention_mask_fo = self.main_forward(batch["encoder_input_gwoz_fs"], batch["attention_mask_gwoz_fs"], batch["decoder_output_gwoz_fs"], choice=2)
            nll += loss_fo
        
        # if "en_encoder_input_cc" in batch and batch["en_encoder_input_cc"] != None:
        #     en_ids_cc = batch["en_encoder_input_cc"] 
        #     en_attention_mask_cc = batch["en_attention_mask_cc"] 
        #     fo_ids_cc = batch["fo_encoder_input_cc"] 
        #     fo_attention_mask_cc = batch["fo_attention_mask_cc"] 
        
        #     # External data
        #     encoder = self.model.get_encoder()
        #     encoder_outputs_en_cc = encoder(input_ids=en_ids_cc,
        #                                 attention_mask=en_attention_mask_cc,
        #                                 )[0]

        #     encoder_outputs_fo_cc = encoder(input_ids=fo_ids_cc,
        #                                 attention_mask=fo_attention_mask_cc,
        #                                 )[0]
        #     # Contrastive Learning
        #     proj_en_h = self.projection(encoder_outputs_en_cc)
        #     proj_fo_h = self.projection(encoder_outputs_fo_cc)
        #     avg_en = self.avg_pool(proj_en_h, en_attention_mask_cc)
        #     avg_fo = self.avg_pool(proj_fo_h, fo_attention_mask_cc)

        #     cos = nn.CosineSimilarity(dim=-1)
        #     sim_matrix = cos(avg_en.unsqueeze(1),
        #                     avg_fo.unsqueeze(0))

        #     logits = sim_matrix / self.tau
        #     batch_size = en_ids_cc.size(0)
        #     labels = torch.arange(batch_size, device=en_ids_cc.device)
        #     cont_loss = self.cont_crit(logits, labels)
        
        return nll
        

    def training_step(self, batch, batch_idx):
        self.model.train()
        loss = 0
        loss += self.forward(batch)
            
        # result = pl.TrainResult(loss)
        # result.log('train_loss', loss, on_epoch=True)
        return {'loss': loss, 'log': {'train_loss': loss}}
        # return result

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        loss = 0
        loss += self.forward(batch)

        return {'val_loss': loss, 'log': {'val_loss': loss}}
        # return result

    def validation_epoch_end(self, outputs):
        val_loss_mean = sum([o['val_loss'] for o in outputs]) / len(outputs)
        # show val_loss in progress bar but only log val_loss
        results = {'progress_bar': {'val_loss': val_loss_mean.item()}, 'log': {'val_loss': val_loss_mean.item()},
                   'val_loss': val_loss_mean.item()}
        return results

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
   




