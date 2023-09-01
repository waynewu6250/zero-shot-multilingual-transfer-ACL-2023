#encoding=utf-8

import torch
from torch.utils.data import random_split
from datasets import load_dataset
import os


# ## Initiating model and trainer for training
from transformers import (
    MBartForConditionalGeneration, MBart50Tokenizer,
    Seq2SeqTrainingArguments, Seq2SeqTrainer
  )
from transformers import BartModel, BartConfig, BartTokenizerFast
from transformers import DataCollatorForLanguageModeling
from transformers import AdapterConfig
from data_collator import DataCollatorForDenoisingTasks

lang_map = {'en': 'en_XX',
            'es': 'es_XX',
            'id': 'id_ID',
            'zh': 'zh_CN'}

configuration = BartConfig(
    vocab_size=52000,
    max_position_embeddings=258,
    d_model=256,
    encoder_layers=3,
    decoder_layers=3,
    encoder_attention_heads=4,
    decoder_attention_heads=4,
    decoder_ffn_dim=1024,
    encoder_ffn_dim=1024,
)
lang = 'id'
model_path = 'facebook/mbart-large-50-many-to-many-mmt'
save_dir = "/fsx/waynewu/experiments/pretrain-clm-mbart-largemmt-"+lang
# model_path = 'facebook/mbart-large-cc25'

# Model & Tokenizer
model = MBartForConditionalGeneration.from_pretrained(model_path)
tokenizer = MBart50Tokenizer.from_pretrained(model_path, src_lang=lang_map[lang], tgt_lang=lang_map[lang])

# Add language adapter
lang_adapter_config = AdapterConfig.load("pfeiffer", reduction_factor=2)
# model.load_adapter("zh/wiki@ukp", config=lang_adapter_config)
model.add_adapter(lang, config=lang_adapter_config)
model.train_adapter([lang])
model.set_active_adapters(lang)

# Get datasets
mc4 = load_dataset('mc4', lang, split='train', streaming=True)
amount = 1232250 * 0.1
data = []
for i, instance in enumerate(mc4):
    data.append(instance['text'].strip())
    if i > amount:
        break
print(f'total size of data is {len(data)}')


# splitting dataset into train, validation
split = 0.1
train_dataset, eval_dataset = random_split(data, lengths=[int((1-split)*len(data))+1, int(split*len(data))])

#A dataset wrapper, that tokenizes our data on-the-fly
class TokenizedSentencesDataset:
    def __init__(self, sentences, tokenizer, max_length, cache_tokenization=False):
        self.tokenizer = tokenizer
        self.sentences = sentences
        self.max_length = max_length
        self.cache_tokenization = cache_tokenization

    def __getitem__(self, item):
        # if not self.cache_tokenization:
        if len(self.sentences[item]) > self.max_length:
            text = self.sentences[item][:self.max_length]
        else:
            text = self.sentences[item]
        batch = tokenizer(text, max_length=self.max_length, padding='max_length')
        return batch

    def __len__(self):
        return len(self.sentences)

max_length = 256
train_dataset = TokenizedSentencesDataset(train_dataset, tokenizer, max_length)
dev_dataset = TokenizedSentencesDataset(eval_dataset, tokenizer, max_length)

data_collator = DataCollatorForDenoisingTasks(tokenizer=tokenizer)

# defining training related arguments
args = Seq2SeqTrainingArguments(output_dir=save_dir,
                        do_train=True,
                        do_eval=False,
                        # evaluation_strategy="epoch",
                        per_device_train_batch_size=6,
                        # per_device_eval_batch_size=8,
                        learning_rate=5e-5,
                        num_train_epochs=10,
                        save_strategy="epoch",
                        logging_dir=os.path.join(save_dir,"logs"))


# defining trainer using 🤗
trainer = Seq2SeqTrainer(model=model, 
                args=args, 
                data_collator=data_collator, 
                train_dataset=train_dataset)


# ## Training time
trainer.train()
# It will take hours to train this model on this dataset

# lets save model
# trainer.evaluate(eval_dataset=eval_dataset)
trainer.save_model(save_dir)
model.save_adapter(os.path.join(save_dir,"adapter"), lang)