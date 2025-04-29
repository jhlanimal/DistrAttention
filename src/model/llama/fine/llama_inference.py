from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer,SchedulerType,default_data_collator,get_scheduler
import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from tqdm import tqdm
import time
from datasets import load_dataset
from math_data_utils import make_supervised_data_module,get_local_data
import numpy as np
from torch.optim import AdamW
import torch.nn.utils as utils
import math
from typing import Dict
import torch
from transformers import PreTrainedTokenizer, PreTrainedModel
import transformers
from fine_llama import smart_tokenizer_and_embedding_resize,fine_tune,get_tokenizer

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

device = "cuda:0"
model = AutoModelForCausalLM.from_pretrained("/data/share/llama3-1b",ignore_mismatched_sizes=True).to(device) #模型地址
#tokenizer = AutoTokenizer.from_pretrained("/data/share/llama3-1b")
tokenizer = get_tokenizer("/data/share/llama3-1b")
#tokenizer.model_max_length = 512  # 设置最大长度为 512
#tokenizer.padding_side = "right"  # 设置填充方向，默认是 "right"
#if tokenizer.pad_token is None:
#    tokenizer.pad_token = tokenizer.eos_token  # 如果没有 pad_token, 使用 eos_token 作为 pad_token
#model = model.half()
special_tokens_dict = dict()
if tokenizer.pad_token is None:
    special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
if tokenizer.eos_token is None:
    special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
if tokenizer.bos_token is None:
    special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN    
if tokenizer.unk_token is None:
    special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
smart_tokenizer_and_embedding_resize(
    special_tokens_dict=special_tokens_dict,
    tokenizer=tokenizer,
    model=model,
  )
train_dataset, data_collator, train_dataloader = make_supervised_data_module(tokenizer, 5,"MathInstruct.json",device)
fine_tune(tokenizer,model, train_dataloader, epochs=5, lr=5e-6,device=device)
''' #inference
for data_test in train_dataloader:
    with torch.no_grad():
        start = time.time()
        outputs = model(data_test, labels=data_test,use_cache = False)
        end = time.time()
        time_arr.append(end - start)
'''
'''
optimizer = AdamW(model.parameters(), lr=lr)
for data_test in train_dataloader:
    optimizer.zero()
    outputs = model(data_test, labels=data_test,use_cache = False)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
'''
'''#train
def fine_tune(model, train_dataloader, epochs, lr):
    optimizer = AdamW(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            batch = {key: value.to(device, non_blocking=False) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch, use_cache=False)
            loss = outputs.loss
            loss.backward()
            print(loss)
            total_norm = utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            print(f"Step {step}: Gradient Norm - {total_norm}")
            optimizer.step()
        print(f"Epoch {epoch}: Loss - {loss}")
fine_tune(model, train_dataloader, epochs=5, lr=1e-8)
'''
'''
        neg_log_likelihood = outputs.loss
        contains_nan = torch.isnan(neg_log_likelihood).any()
        if contains_nan:
            pass
        else:
            nlls.append(neg_log_likelihood)
        '''
#ppl = torch.exp(torch.stack(nlls).mean())
'''
for step, batch in enumerate(train_dataloader):
    start = time.time()
    batch = to_device(batch, engine.device)
    outputs = engine(**batch, use_cache=False)
    loss = outputs.loss
    if args.print_loss and (step == 0 or (step+1)%10==0):
        print_rank_0(
            f"Epoch: {epoch}, Step: {step}, Rank: {torch.distributed.get_rank()}, loss = {loss}"
            )
    engine.backward(loss)
    engine.step()
    '''