from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig
from peft import LoraConfig, TaskType, get_peft_model
import torch

# 将JSON文件转换为CSV文件
df = pd.read_json('data/huanhuan.json')
ds = Dataset.from_pandas(df)

print(ds[0]["instruction"])

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    "./model",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("./model")

# def process_func(example):
#     MAX_LENGTH = 384    # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
#     input_ids, attention_mask, labels = [], [], []
#     instruction = tokenizer(f"<|im_start|>system\n现在你要扮演皇帝身边的女人--甄嬛<|im_end|>\n<|im_start|>user\n{example['instruction'] + example['input']}<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens
#     response = tokenizer(f"{example['output']}", add_special_tokens=False)
#     input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
#     attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
#     labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
#     if len(input_ids) > MAX_LENGTH:  # 做一个截断
#         input_ids = input_ids[:MAX_LENGTH]
#         attention_mask = attention_mask[:MAX_LENGTH]
#         labels = labels[:MAX_LENGTH]
#     return {
#         "input_ids": input_ids,
#         "attention_mask": attention_mask,
#         "labels": labels
#     }
#
# tokenized_id = ds.map(process_func, remove_columns=ds.column_names)
# print(tokenizer.decode(tokenized_id[0]['input_ids']))
# print(tokenizer.decode(list(filter(lambda x: x != -100, tokenized_id[1]["labels"]))))
#
# model.enable_input_require_grads() # 开启梯度检查点时，要执行该方法
# print(model.dtype)
#
# config = LoraConfig(
#     task_type=TaskType.CAUSAL_LM,
#     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
#     inference_mode=False, # 训练模式
#     r=8, # Lora 秩
#     lora_alpha=32, # Lora alaph，具体作用参见 Lora 原理
#     lora_dropout=0.1# Dropout 比例
# )
# print(config)
#
# model = get_peft_model(model, config)
# model.print_trainable_parameters()
#
# args = TrainingArguments(
#     output_dir="./output/Qwen2.5_instruct_lora",
#     per_device_train_batch_size=4,
#     gradient_accumulation_steps=4,
#     logging_steps=10,
#     num_train_epochs=3,
#     save_steps=100,
#     learning_rate=1e-4,
#     save_on_each_node=True,
#     gradient_checkpointing=True
# )
#
# trainer = Trainer(
#     model=model,
#     args=args,
#     train_dataset=tokenized_id,
#     data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
# )
#
# trainer.train()

from peft import PeftModel
lora_path = './output/Qwen2.5_instruct_lora/checkpoint-699' # 这里改称你的 lora 输出对应 checkpoint 地址

# 加载lora权重
model = PeftModel.from_pretrained(model, model_id=lora_path)

prompt = "你是谁？"
inputs = tokenizer.apply_chat_template([{"role": "user", "content": "假设你是皇帝身边的女人--甄嬛。"},{"role": "user", "content": prompt}],
                                       add_generation_prompt=True,
                                       tokenize=True,
                                       return_tensors="pt",
                                       return_dict=True
                                       ).to('cuda')


gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
with torch.no_grad():
    outputs = model.generate(**inputs, **gen_kwargs)
    outputs = outputs[:, inputs['input_ids'].shape[1]:]
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))