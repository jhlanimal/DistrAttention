from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer,SchedulerType,default_data_collator,get_scheduler
import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from tqdm import tqdm
import time
from datasets import load_dataset
from math_data_utils import make_supervised_data_module,get_local_data
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW, get_linear_schedule_with_warmup
import numpy as np
from torch.optim import AdamW
import torch.nn.utils as utils
import math
from typing import Dict
import torch
from transformers import PreTrainedTokenizer, PreTrainedModel
import transformers
#from checkpoints_primal512_new.optim import get_optimizer_grouped_parameters
def fine_tune(tokenizer,model, train_dataloader, epochs, lr,device):
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-6, foreach=False)
    num_epochs = 5
    total_steps = len(train_dataloader) * num_epochs
    #optimizer = AdamW(model.parameters(), lr=lr)
    #optimizer_grouped_parameters = get_optimizer_grouped_parameters(
    #    model, 0, 5e-4)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=1000, num_training_steps=total_steps)
    for epoch in range(epochs):
        model.train()
        print(f"start {epoch}")
        for step, batch in enumerate(train_dataloader):
            batch = {key: value.to(device, non_blocking=False) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch, use_cache=False)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            #break
            print(f"Epoch {epoch},Step {step} Loss - {loss}")
        print(f"Epoch {epoch}: Loss - {loss}")
        model.save_pretrained(f"./checkpoints_ourl512_new/epoch_{epoch + 1}")
        tokenizer.save_pretrained(f"./checkpoints_ourl512_new/epoch_{epoch + 1}")
        print(f"Saved checkpoint at epoch {epoch + 1}")
#fine_tune(model, train_dataloader, epochs=5, lr=1e-8)



def smart_tokenizer_and_embedding_resize(
  special_tokens_dict: Dict,
  tokenizer: transformers.PreTrainedTokenizer,
  model: transformers.PreTrainedModel,
):
  """Resize tokenizer and embedding.

  Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
  """
  num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)

  model.config.end_token_id = tokenizer.eos_token_id
  model.config.pad_token_id = model.config.eos_token_id
  model.resize_token_embeddings(int(8 * math.ceil(len(tokenizer) / 8.0)))
  
  # model.resize_token_embeddings(len(tokenizer))

  if num_new_tokens > 0:
    input_embeddings = model.get_input_embeddings().weight.data
    output_embeddings = model.get_output_embeddings().weight.data

    input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
    output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

    input_embeddings[-num_new_tokens:] = input_embeddings_avg
    output_embeddings[-num_new_tokens:] = output_embeddings_avg


def get_tokenizer(model_name_or_path, fast_tokenizer=True, model_max_length=512, padding_side="right"):
    print(model_name_or_path)
    if "llama" not in model_name_or_path and "Llama" not in model_name_or_path:
        from transformers import LlamaTokenizer
        tokenizer = LlamaTokenizer.from_pretrained(
        model_name_or_path, fast_tokenizer=fast_tokenizer, model_max_length=model_max_length, padding_side=padding_side)
        print("===============================",model_name_or_path)
        if tokenizer.pad_token is None:
        # assert tokenizer.eos_token is not None
        # tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            tokenizer.padding_side = 'right'
    elif "Llama" in model_name_or_path:
        print("*******LLLL")
        tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, fast_tokenizer=fast_tokenizer, model_max_length=model_max_length, padding_side=padding_side)
        tokenizer.pad_token = tokenizer.eos_token
        # make sure tokenizer is right pad in our logic
        tokenizer.padding_side = 'right'
    else:
        tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, fast_tokenizer=fast_tokenizer, model_max_length=model_max_length, padding_side=padding_side)
        tokenizer.pad_token = tokenizer.eos_token
        # make sure tokenizer is right pad in our logic
        tokenizer.padding_side = 'right'
    return tokenizer