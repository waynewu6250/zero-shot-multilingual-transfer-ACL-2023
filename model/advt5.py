"""T5 with frequency and contrastive loss"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import T5ForConditionalGeneration, AdamW
from torch.autograd import Variable
import numpy as np
from collections import Counter


class AdvContrastiveNMT(nn.Module):
    def __init__(self, model, tokenizer, args):
        super(AdvContrastiveNMT, self).__init__()

        self.tau = args['tau']
        self.pos_eps = args['pos_eps']
        self.neg_eps = args['neg_eps']

        self.t5_model = model
        self.t5_tokenizer = tokenizer
        self.projection = nn.Sequential(nn.Linear(args['hidden_size'], args['hidden_size']),
                                        nn.ReLU())
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.word_freq = np.zeros(len(self.t5_tokenizer))
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def forward(self, input_ids, attention_mask, lm_labels, adv=True):
        # input_ids: ids of article tokens
        # attention_mask: mask for input_ids 0 for PAD 1 o.w
        # lm_labels: shift decoder_input_ids left

        encoder = self.t5_model.get_encoder()
        decoder = self.t5_model.get_decoder()

        encoder_outputs = encoder(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  inputs_embeds=None,
                                  head_mask=None
                                  )

        hidden_states = encoder_outputs[0]

        decoder_input_ids = self._shift_right(lm_labels)
        decoder_attention_mask = torch.sign(decoder_input_ids)
        # since bos id is 0, change it into 1 (should be attended)
        decoder_attention_mask[:, 0] = 1

        decoder_outputs = decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=None,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=None,
            use_cache=None,
        )
        sequence_output = decoder_outputs[0]
        # Rescale output before projecting on vocab
        # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
        sequence_output = sequence_output * (self.t5_model.model_dim ** -0.5)
        lm_logits = self.t5_model.lm_head(sequence_output)

        # Add hidden states and attention if they are here
        decoder_outputs = (lm_logits,) + decoder_outputs[1:]

        vocab_size = lm_logits.size(-1)

        self.update_frequency(lm_labels)
        # self.criterion.weight = self.loss_weight()

        # nll = self.criterion(lm_logits.view(-1, vocab_size),
        #                      lm_labels.view(-1))

        self.criterion.reduction = 'none'
        preds = lm_logits.argmax(-1)
        nll = self.criterion(lm_logits.view(-1, vocab_size),
                             lm_labels.view(-1))
        freq_pred = self.word_freq[preds.view(-1).cpu().numpy()]
        freq_pred = torch.FloatTensor(freq_pred).to(self.device)
        freq_GT = self.word_freq[lm_labels.view(-1).cpu().numpy()]
        freq_GT = torch.FloatTensor(freq_GT).to(self.device)
        total_freq = self.word_freq.sum()
        weight = 1 + F.relu(freq_pred - freq_GT) / total_freq
        nll = torch.matmul(nll, weight)

        if adv:
            """
            avg_doc: encoder hidden
            avg_abs: decoder hidden
            avg_pert: small noise
            avg_pos_dec: large noise
            """
            proj_enc_h = self.projection(hidden_states)
            proj_dec_h = self.projection(sequence_output)
            avg_doc = self.avg_pool(proj_enc_h, attention_mask)
            avg_abs = self.avg_pool(proj_dec_h, decoder_attention_mask)

            cos = nn.CosineSimilarity(dim=-1)
            cont_crit = nn.CrossEntropyLoss()
            sim_matrix = cos(avg_doc.unsqueeze(1),
                             avg_abs.unsqueeze(0))
            
            # train only negative samples
            input_check = (input_ids[None, :, :] - input_ids[:, None, :]).sum(-1)
            attn_mask = (input_check == 0).long() - torch.eye(len(input_check), device=input_check.device)
            sim_matrix = sim_matrix.masked_fill_(attn_mask.bool(), -float("Inf"))

            logits = sim_matrix / self.tau
            batch_size = input_ids.size(0)
            labels = torch.arange(batch_size, device=input_ids.device)
            cont_loss = cont_crit(logits, labels)

            # train only adversarial samples
            # batch_size = input_ids.size(0)
            # perturbed_dec = self.generate_adv(sequence_output, lm_labels)  # [n,b,t,d] or [b,t,d]
            # proj_pert_dec_h = self.projection(perturbed_dec)
            # avg_pert = self.avg_pool(proj_pert_dec_h, decoder_attention_mask)
            # adv_sim = cos(avg_doc, avg_pert).unsqueeze(1)  # [b,1]
            # pos_sim_1 = cos(avg_doc, avg_abs).unsqueeze(-1) # [b,1]
            # logits = torch.cat([pos_sim_1, adv_sim], dim=1) / self.tau # [b,2]

            # pos_dec_hidden = self.generate_cont_adv(hidden_states, attention_mask,
            #                                         sequence_output, decoder_attention_mask,
            #                                         lm_logits,
            #                                         self.tau, self.pos_eps)
            # avg_pos_dec = self.avg_pool(self.projection(pos_dec_hidden),
            #                             decoder_attention_mask)
            # pos_sim_2 = cos(avg_doc, avg_pos_dec).unsqueeze(-1)  # [b,1]
            # new_logits = torch.cat([pos_sim_2, adv_sim], dim=1) / self.tau # [b,2]

            # labels = torch.zeros(batch_size, device=input_ids.device).long()
            # cont_loss = cont_crit(logits, labels)
            # new_cont_loss = cont_crit(new_logits, labels)
            # cont_loss = 0.5 * (cont_loss + new_cont_loss)

            
            # # adversarial 
            # # part 1
            # batch_size = input_ids.size(0)
            # perturbed_dec = self.generate_adv(sequence_output,
            #                                   lm_labels)  # [n,b,t,d] or [b,t,d]
            # proj_pert_dec_h = self.projection(perturbed_dec)
            # avg_pert = self.avg_pool(proj_pert_dec_h,
            #                          decoder_attention_mask)

            # adv_sim = cos(avg_doc, avg_pert).unsqueeze(1)  # [b,1]
            # logits = torch.cat([sim_matrix, adv_sim], 1) / self.tau

            # # part 2
            # pos_dec_hidden = self.generate_cont_adv(hidden_states, attention_mask,
            #                                         sequence_output, decoder_attention_mask,
            #                                         lm_logits,
            #                                         self.tau, self.pos_eps)
            # avg_pos_dec = self.avg_pool(self.projection(pos_dec_hidden),
            #                             decoder_attention_mask)

            # pos_sim = cos(avg_doc, avg_pos_dec).unsqueeze(-1)  # [b,1]

            # identity = torch.eye(batch_size, device=input_ids.device)
            # pos_sim = identity * pos_sim
            # neg_sim = sim_matrix.masked_fill(identity == 1, 0)
            # new_sim_matrix = pos_sim + neg_sim # replace original self case as augmented sample with large noise
            # new_logits = torch.cat([new_sim_matrix, adv_sim], 1)

            # labels = torch.arange(batch_size,
            #                       device=input_ids.device)

            # cont_loss = cont_crit(logits, labels)
            # new_cont_loss = cont_crit(new_logits, labels)

            # cont_loss = 0.5 * (cont_loss + new_cont_loss)

            return nll, cont_loss

        else:
            return nll
    
    # def clean_preds(self, preds):
    #     res = []
    #     preds = preds.cpu().tolist()
    #     for pred in preds:
    #         ind = pred.index(self.t5_tokenizer.sep_token_id)+1
    #         for token in pred[ind:]:
    #              if token not in self.t5_tokenizer.all_special_ids and token != -100:
    #                 res.append(token)
    #         # if self.tokenizer.end_token in pred:
    #         #     ind = pred.index(self.tokenizer.eos_token) + 1 # end_idx included
    #         #     pred = pred[:ind]
    #         # if len(pred) == 0:
    #         #     continue
    #         # if pred[0] == self.tokenizer.start_token:
    #         #     pred = pred[1:]
    #         # res.append([p for p in pred if p not in self.t5_tokenizer.all_special_ids and p != -100])
    #     return res
    
    def update_frequency(self, preds):
        preds = preds.cpu().tolist()
        for pred in preds:
            ind = pred.index(self.t5_tokenizer.sep_token_id)+1
            for token in pred[ind:]:
                 if token not in self.t5_tokenizer.all_special_ids and token != -100:
                    self.word_freq[token] += 1

        # curr = Counter()
        # curr.update(preds)

        # # self.word_freq *= self.opt['decay_factor']
        # for k, v in curr.items():
        #     self.word_freq[k] += v
    
    def loss_weight(self):
        RF = self.word_freq / self.word_freq.sum() # relative frequency
        a = -1 / RF.max()
        weight = a * RF + 1
        weight = weight / weight.sum() * len(weight) # normalization
        return torch.FloatTensor(weight).to(self.device)

    def generate_adv(self, dec_hiddens, lm_labels):
        dec_hiddens = dec_hiddens.detach()

        dec_hiddens.requires_grad = True

        lm_logits = self.t5_model.lm_head(dec_hiddens)
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        loss = criterion(lm_logits.view(-1, lm_logits.size(-1)),
                         lm_labels.view(-1))

        loss.backward()
        dec_grad = dec_hiddens.grad.detach()
        l2_norm = torch.norm(dec_grad, dim=-1)

        dec_grad /= (l2_norm.unsqueeze(-1) + 1e-12)

        perturbed_dec = dec_hiddens + self.neg_eps * dec_grad.detach()
        perturbed_dec = perturbed_dec  # [b,t,d]

        self.zero_grad()

        return perturbed_dec

    def generate_cont_adv(self, enc_hiddens, enc_mask,
                          dec_hiddens, dec_mask, lm_logits,
                          tau, eps):
        enc_hiddens = enc_hiddens.detach()
        dec_hiddens = dec_hiddens.detach()
        lm_logits = lm_logits.detach()
        dec_hiddens.requires_grad = True

        avg_enc = self.avg_pool(self.projection(enc_hiddens),
                                enc_mask)

        avg_dec = self.avg_pool(self.projection(dec_hiddens),
                                dec_mask)

        cos = nn.CosineSimilarity(dim=-1)
        logits = cos(avg_enc.unsqueeze(1), avg_dec.unsqueeze(0)) / tau

        cont_crit = nn.CrossEntropyLoss()
        labels = torch.arange(avg_enc.size(0),
                              device=enc_hiddens.device)
        loss = cont_crit(logits, labels)
        loss.backward()

        dec_grad = dec_hiddens.grad.detach()
        l2_norm = torch.norm(dec_grad, dim=-1)
        dec_grad /= (l2_norm.unsqueeze(-1) + 1e-12)

        perturb_dec_hidden = dec_hiddens + eps * dec_grad
        perturb_dec_hidden = perturb_dec_hidden.detach()
        perturb_dec_hidden.requires_grad = True
        perturb_logits = self.t5_model.lm_head(perturb_dec_hidden)

        true_probs = F.softmax(lm_logits, -1)
        true_probs = true_probs * dec_mask.unsqueeze(-1).float()

        perturb_log_probs = F.log_softmax(perturb_logits, -1)

        kl_crit = nn.KLDivLoss(reduction="sum")
        vocab_size = lm_logits.size(-1)

        kl = kl_crit(perturb_log_probs.view(-1, vocab_size),
                     true_probs.view(-1, vocab_size))
        kl = kl / torch.sum(dec_mask).float() 
        kl.backward()

        kl_grad = perturb_dec_hidden.grad.detach()

        l2_norm = torch.norm(kl_grad, dim=-1)

        kl_grad /= (l2_norm.unsqueeze(-1) + 1e-12)

        perturb_dec_hidden = perturb_dec_hidden - eps * kl_grad

        
        return perturb_dec_hidden

    def avg_pool(self, hidden_states, mask):
        length = torch.sum(mask, 1, keepdim=True).float()
        mask = mask.unsqueeze(2)
        hidden = hidden_states.masked_fill(mask == 0, 0.0)
        avg_hidden = torch.sum(hidden, 1) / length

        return avg_hidden
    
    def _shift_right(self, input_ids):
        decoder_start_token_id = self.t5_tokenizer.bos_token_id
        pad_token_id = self.t5_tokenizer.pad_token_id

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


class Dialog_Seq2seq_adv(pl.LightningModule):

    def __init__(self, args, tokenizer, model):
        super().__init__()
        self.t5_tokenizer = tokenizer
        self.t5_model = model
        self.args = args
        self.main_model = AdvContrastiveNMT(self.t5_model, self.t5_tokenizer, self.args)
        self.lr = args["lr"]

    def training_step(self, batch, batch_idx):
        self.main_model.train()
        if self.args['adv']:
            loss, cont_loss = self.main_model(input_ids=batch["encoder_input"],
                            attention_mask=batch["attention_mask"],
                            lm_labels=batch["decoder_output"],
                            adv=True
                            )
            total_loss = loss + cont_loss
        else:
            loss = self.main_model(input_ids=batch["encoder_input"],
                            attention_mask=batch["attention_mask"],
                            lm_labels=batch["decoder_output"],
                            adv=False
                            )
            total_loss = loss

        # result = pl.TrainResult(loss)
        # result.log('train_loss', loss, on_epoch=True)
        return {'loss': total_loss, 'log': {'train_loss': loss}}
        # return result

    def validation_step(self, batch, batch_idx):
        self.main_model.eval()
        loss = self.main_model(input_ids=batch["encoder_input"],
                          attention_mask=batch["attention_mask"],
                          lm_labels=batch["decoder_output"],
                          adv=False
                          )


        return {'val_loss': loss, 'log': {'val_loss': loss}}
        # return result

    def validation_epoch_end(self, outputs):
        val_loss_mean = sum([o['val_loss'] for o in outputs]) / len(outputs)
        # show val_loss in progress bar but only log val_loss
        results = {'progress_bar': {'val_loss': val_loss_mean.item()}, 'log': {'val_loss': val_loss_mean.item()},
                   'val_loss': val_loss_mean.item()}
        return results
    
    def predict_step(self, batch, batch_idx):
        dst_outputs = self.main_model.t5_model.generate(input_ids=batch["encoder_input"],
                                    attention_mask=batch["attention_mask"],
                                    eos_token_id=self.tokenizer.eos_token_id,
                                    do_sample=True,
                                    max_length=200,
                                    )
        return dst_outputs

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr, correct_bias=True)
    
    # save model
    def save_model(self, save_path):
        model_to_save = self.main_model.module if hasattr(
            self.main_model, "module") else self.main_model
        ckpt = {
            "args": self.args,
            "state_dict": model_to_save.t5_model.state_dict(),
        }
        
        torch.save(ckpt, save_path)
