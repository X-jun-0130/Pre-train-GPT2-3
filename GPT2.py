import torch
from transformers import BertTokenizer, GPT2LMHeadModel, AutoConfig
from transformers import Trainer, TrainingArguments
import json
from torch.utils.data import  random_split
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

tokenizer = BertTokenizer.from_pretrained("./gptbase/")
training_args = TrainingArguments(output_dir='./results', 
                                 overwrite_output_dir=True,
                                 num_train_epochs=3, 
                                 learning_rate=5e-4,
                                 save_strategy='epoch',
                                 evaluation_strategy = 'epoch',
                                 per_device_train_batch_size=4, 
                                 per_device_eval_batch_size=4, 
                                 lr_scheduler_type="cosine",
                                 warmup_steps=1000,
                                 weight_decay=0.01,
                                 fp16=True)


doc_all = json.load(open('./new_data/doc_data.json', 'r', encoding='utf-8'))

data_list = [[l,i] for i,l in enumerate(doc_all)]

data = json.load(open('./new_data/diseasexls_train_data.json', 'r', encoding='utf-8')) + json.load(open('./new_data/drugxls_train_data.json', 'r', encoding='utf-8')) +json.load(open('./new_data/kg_drug_train_data.json', 'r', encoding='utf-8')) 
kg_dataset = [[k['text'] + k['answer'] ,j] for j, k in enumerate(data)]


dataset =[k for k in kg_dataset +data_list if len(k[0])< 510]
train_size = int(0.98 * len(dataset))
train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])

print(len(train_dataset))




config = AutoConfig.from_pretrained("/home/Xuxiangjun/Model_TH/gptbase/config.json")
model = GPT2LMHeadModel(config).cuda()
model_size = sum(t.numel() for t in model.parameters())
print(f"GPT-2 size: {model_size/1000**2:.1f}M parameters")





'''
model training
'''
def the_collate_fn(batch):  
    r = tokenizer([b[0] for b in batch], padding=True )
    input_ids = torch.LongTensor(r['input_ids'])
    attention_mask = torch.LongTensor(r['attention_mask'])
    return {'input_ids':input_ids, 'attention_mask':attention_mask, 'labels':input_ids}


class Mytrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(inputs["input_ids"], attention_mask=inputs["attention_mask"], labels = inputs["labels"])
        loss, logits = outputs[:2]
        return (loss, logits) if return_outputs else loss

trainer = Mytrainer(model=model, args=training_args, train_dataset=train_dataset,eval_dataset=val_dataset, data_collator=the_collate_fn)
trainer.train()
