from transformers import AlbertForSequenceClassification, BertTokenizerFast
import re
import torch
import json
import numpy as np
import random


model = AlbertForSequenceClassification.from_pretrained("emojify model")
tokenizer = BertTokenizerFast.from_pretrained(
    'bert-base-chinese')  # use pretrained tokenizer
jsonFile = open('cls.json', 'r')
f = jsonFile.read()   # 要先使用 read 讀取檔案
cls_dict = json.loads(f)

while True:
    output_text = ""

    # 將句子列表合併成一個字符串
    input_text = input()

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

    print(output_text)
