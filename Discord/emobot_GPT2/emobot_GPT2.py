import discord
from discord.ext import commands
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer
import json

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
tokenizer.add_special_tokens(special_tokens_dict={'bos_token': '<|endoftext|>',
                                                  'eos_token': '<|endoftext|>',
                                                  'unk_token': '<|endoftext|>'})
# add the EOS token as PAD token to avoid warnings
model = GPT2LMHeadModel.from_pretrained(
    "emoji reply model", pad_token_id=tokenizer.eos_token_id)

jsonFile = open('cls.json', 'r')
f = jsonFile.read()
emoji_dict = json.loads(f)

# intents是要求機器人的權限
intents = discord.Intents.all()
intents.message_content = True
intents.members = True
intents.typing = True
intents.presences = True
client = discord.Client(intents = intents)
# command_prefix是前綴符號，可以自由選擇($, #, &...)
bot = commands.Bot(command_prefix = "%", intents = intents)

positive_emojis = {'😊', '😄', '👍', '😁', '😂', '🎉', '😃', '🥳', '❤️', '🌞', '🌟', '😉', '😍','🤩','😊','🤣','😆','🥰','😘'}

negative_emojis = {'😞', '😠', '👎', '😢', '😡', '🙁', '😔', '😕', '😫', '💔', '🥶', '😓', '😤', '🤯', '😵','🤮','🙅‍♂️'}

@bot.event
# 當機器人完成啟動
async def on_ready():
    print(f"目前登入身份 --> {bot.user}")

@bot.event
async def on_reaction_add(reaction, user):
    
    txt = []
    idx = 0
    flag = False
    # 判斷@的人
    for ch in reaction.message.content:
        if ch == '<':
            flag = True
        if ch == '>':
            idx += 1
            break
        if flag:
            idx += 1
    txt = reaction.message.content[2:idx-1]

    print(str(txt) == str(user.id))
    #
    if(str(txt) != str(user.id)):
        return
        
    # 檢查反應是否來自機器人自己
    if user == bot.user:
        return
    idx = 0
    flag = False
    for ch in reaction.message.content:
        if ch == '<':
            flag = True
        if ch == '>':
            idx += 1
            break
        if flag:
            idx += 1
    txt = reaction.message.content[idx:]
    print("message",txt)
    if(str(reaction.emoji) in positive_emojis):
        txt = "「" + txt + "」這句話的正面回覆是："
    if(str(reaction.emoji) in negative_emojis):
        txt = "「" + txt + "」這句話的負面回覆是："

    input_ids = tokenizer.encode(txt, return_tensors='pt')
    beam_output = model.generate(
        input_ids,
        max_length=64,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True,
        temperature=1
    )
    generated_reply = tokenizer.decode(
        beam_output[0], skip_special_tokens=True).replace(" ", "")
    reply = generated_reply[len(txt):]
    # reply = generated_reply[:]
    output_str = ""
    for ch in reply:
        if (ch == "，" or ch == "。") and output_str == "":
            continue
        output_str += ch
        if ch == "。" or ch == "！" or ch == "？":
            break
    # print(output_str, "\n")
    await reaction.message.channel.send(f"{user.mention}表示:{output_str}")
    
@bot.command()
# 輸入%Hello呼叫指令
async def Hello(ctx):
    # 回覆Hello, world!
    await ctx.send("Hello, world!")

#To connect to your bot, replace the 'bot_token' with your own token!
bot.run('bot_token')
