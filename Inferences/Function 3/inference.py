from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
tokenizer.add_special_tokens(special_tokens_dict={'bos_token': '<|endoftext|>',
                                                  'eos_token': '<|endoftext|>',
                                                  'unk_token': '<|endoftext|>'})

# add the EOS token as PAD token to avoid warnings
model = GPT2LMHeadModel.from_pretrained(
    "emoji reply model", pad_token_id=tokenizer.eos_token_id)

while True:
    txt = input()
    txt = "「" + txt[:-2] + "」這句話的" + txt[-2:] + "回覆是："
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
    print(output_str, "\n")
