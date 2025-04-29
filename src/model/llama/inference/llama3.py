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
import warnings
warnings.filterwarnings("ignore", message="Setting `pad_token_id` to `eos_token_id`.*")
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

device = "cuda:0"
base_llama = "/data/share/llama3-1b"
trained_llama = "/home/jhl/llm/llama/checkpoints_new/epoch_5"
model = AutoModelForCausalLM.from_pretrained(trained_llama,ignore_mismatched_sizes=True).to(device) #模型地址
#tokenizer = AutoTokenizer.from_pretrained("/data/share/llama3-1b")
tokenizer = get_tokenizer(trained_llama)
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
if tokenizer.pad_token_id is None:
    pad_token_id = tokenizer.eos_token_id
smart_tokenizer_and_embedding_resize(
    special_tokens_dict=special_tokens_dict,
    tokenizer=tokenizer,
    model=model,
  )

# 加载模型
#model = pipeline("text-generation", model="gpt-3.5-turbo")
#model = AutoModelForCausalLM.from_pretrained("/data/share/llama3-1b")

# 加载 MMLU 数据集
dataset = load_dataset(
    "parquet",
    data_files={
        "test": "/data/jhl/mmlu_1/machine_learning/test-00000-of-00001.parquet",
    }
)

#dataset = load_dataset("/data/jhl/mmlu", "all")
#加载csv数据
'''
dataset = load_dataset(
    "csv",
    data_files={
        "test": "/data/jhl/mmlu_1/machine_learning_test/machine_learning_test.csv",
    }
)'''
# 零样本评估
correct = 0
total = 0

for example in dataset["test"]:
    question = example["question"]
    options = example["choices"]
    answer = example["answer"]
    
    # 构建输入
    input_text = f"Question: {question}\n"
    for i, option in enumerate(options):
        #if i == 0:
        input_text += f"{chr(65 + i)}. {option}\n"
    #for i, option in enumerate(options):
        # elif i < 5:
        #     input_text += f"{chr(65 + i-1)}. {option}\n"
        # else:
        #     answer = option
    inputs = tokenizer(input_text,padding='max_length', max_length=512, return_tensors="pt").to(device)#
    
    #print(total,inputs['input_ids'])
    # if inputs['input_ids'].shape[1]==51:
    #     print(inputs)
    # if inputs['input_ids'].shape[1]==57:
    #     print(inputs)
    #if inputs['input_ids'].shape[1]>256 and inputs['input_ids'].shape[1] <=512:
    
    #if inputs['input_ids'].shape[1] <=128:
    if True:
        start = time.time()
        output = model(**inputs, max_new_tokens=512,use_cache=False)
        end = time.time()
        print(end - start)
        # 模型生成答案
        #output = model(input_text, max_length=10)
        print(output)
        '''
        predicted_text = tokenizer.decode(output[0], skip_special_tokens=True)

            # 提取预测的答案（如 "A"）
        predicted_answer = predicted_text.strip().split()[-1]
        parts = predicted_text.split('Answer:')
        answer1 = 0
        if len(parts) == 1:
            total -= 1
        # 提取 'Answer:' 后的部分
        else:
            answer1 = parts[1].strip()
        if answer1 != 'A' and  answer1 != 'B' and answer1 != 'C' and answer1 != 'D':
            total -= 1
        # 计算准确率
        #print(inputs['input_ids'].shape)
        if answer1 == chr(65 + answer):
            correct += 1
        total += 1
        '''
accuracy = correct / total
print(f"Accuracy: {accuracy:.4f}")
#lm_eval --model hf --model_args pretrained=/home/jhl/llm/llama/checkpoints_new/epoch_5,dtype="float" --tasks mmlu --device cuda:2 --batch_size 1