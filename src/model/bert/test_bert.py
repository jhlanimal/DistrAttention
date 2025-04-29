from transformers import BertTokenizer, BertForTokenClassification
from datasets import load_dataset
import torch
# 加载模型和tokenizer
tokenizer = BertTokenizer.from_pretrained('bert_neo')
model = BertForTokenClassification.from_pretrained('bert_neo')
 
# 加载CoNLL-2003数据集
dataset = load_dataset('conll2003')
 
# 测试集数据
test_dataset = dataset['test']
 
# 预测
def predict(model_input):
    model.eval()
    with torch.no_grad():
        outputs = model(**model_input)
        logits = outputs[0]
        predictions = torch.argmax(logits, dim=2)
        return predictions
 
# 解析测试集并进行预测
def run_ner_prediction(test_dataset):
    tokenized_datasets = test_dataset.map(lambda examples: tokenizer(examples['tokens'], truncation=True), batched=True)
 
    # 加载数据进行预测
    for i, inputs in enumerate(tokenized_datasets):
        model_inputs = {k: v.to(device) for k, v in inputs.data.items()}
        predictions = predict(model_inputs)
        print(predictions)
 
# 运行设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'
 
# 执行预测
run_ner_prediction(test_dataset)