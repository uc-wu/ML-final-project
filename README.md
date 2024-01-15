# EMOBOT: An Emoji Chatbot For Online Chat
This repo contains the code for our project.

Emobot is a useful chatbot designed for use on Discord, featuring three main functions. 
* The first and second functionalities assist in generating trendy text and auto emoji reply, utilizing the Albert model for training.
* The third function aids in automatic responses, trained using the GPT-2 model.
### Function 1 2&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;            Function3
<table>
  <tr>
    <td><img src="https://github.com/uc-wu/ML-final-project/blob/main/Albert.png" width="800"/></td>
    <td><img src="https://github.com/uc-wu/ML-final-project/blob/main/GPT-2.png" width="800"/></td>
  </tr>
</table>

## Contents
1. [Datasets](#Datasets)
2. [Dependency](#Dependency)
3. [Execution](#Execution)
   - [Model 1](#Model-1)
   - [Model 2](#Model-2)
   - [Discord](#Discord)
4. [Results](#Results) 
5. [Demo](#Demo)
## Datasets
* We experiment on datasets we collected and generated
* Please see dictrectory **Datasets**
## Dependency
```
pip install numpy
pip install torch
pip install transformers
pip install matplotlib
pip install pandas
```
## Execution
* Please select the corresponding folder based on the functionality.
* For the complete model information please see [Drive](https://drive.google.com/drive/folders/1Yt165MQ2rKarZ9ROnHOHJ6IaM8IULFT7?usp=share_link)
### Model-1
```
#Function 1 2
# Training
python main.py
# Inference
python inference.py
```
### Model-2

```
#Function 3
# Training
python main.py
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
### The Comparison Between Different Emojify Training Method's Performance On The Testing Set
* Number of epochs to train on the two datasets and accuracy
  
| Dataset 1 | Dataset 2 | Accuracy |
|----------|----------|----------|
| 10   | 0   | 0.75   |
| 4   | 6   | 0.82   |
| 2   | 8   | 0.91   |
| 0   | 10   | 0.85   |
### The Average Losses On Training Set And Testing Set 
* During the training on the Dataset 2 for 8 epochs.
* Blue curve represents the training loss and the orange curve represents the loss on the testing set.
### The Average Accuracy On Training Set And Testing Set 
* During the training on the Dataset 2 for 8 epochs.
* Blue curve represents the training loss and the orange curve represents the loss on the testing set.

## Demo
### Function1&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;                     Function2&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;                    Function3
<table>
  <tr>
    <td><img src="https://github.com/uc-wu/ML-final-project/blob/main/result1.gif" width="200"/></td>
    <td><img src="https://github.com/uc-wu/ML-final-project/blob/main/result2.gif" width="200"/></td>
     <td><img src="https://github.com/uc-wu/ML-final-project/blob/main/result3.gif" width="200"/></td>
  </tr>
</table>

