from transformers import Trainer
from transformers import TrainingArguments
from transformers import DataCollatorForLanguageModeling
import transformers
from transformers import AutoTokenizer, GPT2LMHeadModel, AutoConfig
from datasets import load_dataset, DatasetDict
from glob import glob
from transformers import AutoTokenizer, AutoConfig


def tokenize(text, context_length=128):
    outputs = tokenizer(
        text['content'],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length <= context_length:
            input_batch.append(input_ids)
    return {"input_ids": input_batch}


raw_datasets = load_dataset("csv",
                            data_files={
                                'train': "training_data.csv",
                                'valid': "validation_data.csv"},
                            cache_dir="cache_data")


tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
tokenizer.add_special_tokens(special_tokens_dict={'bos_token': '<|endoftext|>',
                                                  'eos_token': '<|endoftext|>',
                                                  'unk_token': '<|endoftext|>'})

tokenized_datasets = raw_datasets.map(
    tokenize, batched=True, remove_columns=raw_datasets["train"].column_names
)

print("Size of training set = ", len(tokenized_datasets["train"]))
print("Size of validation set = ", len(tokenized_datasets["valid"]))
context_length = 128

config = AutoConfig.from_pretrained(
    "yuanzhoulvpi/gpt2_chinese",
    vocab_size=len(tokenizer),
    n_ctx=context_length,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)
print("Load model!")
model = GPT2LMHeadModel(config)
# model = GPT2LMHeadModel.from_pretrained(
#     "emoji reply model", pad_token_id=tokenizer.eos_token_id)
tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

args = TrainingArguments(
    output_dir="emoji reply gpt2",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="steps",
    eval_steps=2_000,
    logging_steps=2_000,
    gradient_accumulation_steps=8,
    num_train_epochs=50,
    weight_decay=0.1,
    warmup_steps=1_000,
    lr_scheduler_type="cosine",
    learning_rate=5e-4,
    save_steps=2_000,
    fp16=True,
    push_to_hub=False,
)

print("Training!")
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["valid"],
)
trainer.train()

output_dir = 'emoji reply model'
model_to_save = model.module if hasattr(model, 'module') else model
model_to_save.save_pretrained(output_dir)
