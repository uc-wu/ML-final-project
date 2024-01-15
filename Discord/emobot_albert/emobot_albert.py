import discord
from discord.ext import commands
import json
import numpy as np
import pandas as pd
import random
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split, RandomSampler, SequentialSampler
from transformers import AdamW, AlbertForSequenceClassification, AlbertConfig, BertTokenizerFast, get_linear_schedule_with_warmup
import re

# load cls dict
jsonFile = open('cls.json', 'r')
f = jsonFile.read()
cls_dict = json.loads(f)

# use pretrained tokenizer_albert
tokenizer = BertTokenizerFast.from_pretrained(
    'bert-base-chinese')
# use fine-tuned model
model = AlbertForSequenceClassification.from_pretrained("emojify model")

# bot_initialize
intents = discord.Intents.all()
intents.message_content = True
intents.members = True
intents.typing = True
intents.presences = True
bot = commands.Bot(command_prefix='~', intents=intents) #The symbol for using commands

#check bot is working
@bot.event
async def on_ready():
    print(">> Bot is online <<")

#Feature 1: Automatic Emoji Responses
@bot.event
async def on_message(message):
    #To prevent the bot's own messages from triggering this event and causing an infinite loop.
    if message.author == bot.user:
        return
    #If the first character of the message is '~', the subsequent operations will not be executed.
    if not message.content.startswith('~'):
        test_sentence = message.content
        encoded_dict = tokenizer.encode_plus(
            test_sentence,
            add_special_tokens=True,
            max_length=64,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_id = torch.tensor(encoded_dict['input_ids'])
        attention_mask = torch.tensor(encoded_dict['attention_mask'])

        with torch.no_grad():
            preds = model(input_id, token_type_ids=None, attention_mask=attention_mask).logits
            pred_flat = str(np.argmax(preds, axis=1).flatten().item())
            await message.add_reaction(cls_dict[pred_flat])
        # Ensure that after processing the message, other events continue to function normally.
    await bot.process_commands(message)

#Feature 2: Copypasta Generation
@bot.command()
async def add_emoji(ctx, *args):
    output_text = ""

    # Split the sentence into words.
    input_text = ' '.join(args)
     # 使用正規表達式以標點符號為基準進行斷句
    sentences = re.split(r'[。！？：、， ]+', input_text)
    # print(sentences)
    # print(sentences)
    for txt in sentences:
        # 將句子分成詞語
        if txt in [',', '。', '!', '?', ':', '：', '、', '，', '']:
            continue
        else:
            output_text += txt
            encoded_dict = tokenizer.encode_plus(
                txt,
                add_special_tokens=True,
                max_length=64,
                pad_to_max_length=True,
                return_attention_mask=True,
                return_tensors='pt',
            )
            input_id = torch.tensor(encoded_dict['input_ids'])
            attention_mask = torch.tensor(encoded_dict['attention_mask'])

            with torch.no_grad():
                preds = model(input_id, token_type_ids=None,
                              attention_mask=attention_mask).logits
                pred_flat = str(np.argmax(preds, axis=1).flatten().item())

                # 在每個斷詞後添加 cls_dict[pred_flat]
                times = random.randint(1, 3)
                output_text += times*cls_dict[pred_flat]

    await ctx.send(output_text)

#To connect to your bot, replace the 'bot_token' with your own token!
bot.run('bot_token')