# EMOBOT: A Daily Chatbot In Life
This repo contains the code for our project
Emobot is a practical chatbot designed for use on Discord, featuring three main functions. 
* The first and second functionalities assist in generating trendy text, utilizing the Albert model for training.
* The third function aids in automatic responses, trained using the GPT-2 model.
### Function 1 2&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;                   Function3
<table>
  <tr>
    <td><img src="https://github.com/uc-wu/ML-final-project/blob/main/截圖%202024-01-15%20晚上7.06.41.png" width="800"/></td>
    <td><img src="https://github.com/uc-wu/ML-final-project/blob/main/截圖%202024-01-15%20晚上7.09.28.png" width="800"/></td>
  </tr>
</table>

## Contents
1. [Datasets](#Datasets)
2. [Installation Instructions](#Installation)
3. [Execution](#Execution)
   - [Model 1](#Model1)
   - [Model 2](#Model2)
   - [Discord](#Discord)
5. [Results](#Results)
## Datasets
* We experiment on datasets we collected and generated
* Please see dictrectory Datasets
## Installation
```
pip install numpy
pip install torch
pip install transformers
pip install matplotlib

```
## Execution
### Model 1
```
# Training
python emojify_model.py
# Inference
python inference.py

```
### Model 2
```
# Training
python emojify_model.py
# Inference
python inference.py
```
### Discord
```
# Create discord bot 
# Load pretrained model
# Please see dictrectorry Discord and run corresponing function file
# Example : function 3
python emobot_GPT2
```
## Results
### Function1&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;                     Function2&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;                    Function3
<table>
  <tr>
    <td><img src="https://github.com/uc-wu/ML-final-project/blob/main/56e1c147-0e7f-4724-a602-e20eaa7dd686.gif" width="200"/></td>
    <td><img src="https://github.com/uc-wu/ML-final-project/blob/main/510c7584-a87f-4be5-8a4a-12ea7f7dcdd6.gif" width="200"/></td>
     <td><img src="https://github.com/uc-wu/ML-final-project/blob/main/282bdd35-680d-44c9-99d2-8cf7d33f3e8e.gif" width="200"/></td>
  </tr>
</table>

