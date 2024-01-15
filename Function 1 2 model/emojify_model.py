import numpy as np
import time
import datetime
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split, RandomSampler, SequentialSampler
from transformers import AdamW, AlbertForSequenceClassification, AlbertConfig, BertTokenizerFast, get_linear_schedule_with_warmup
import os
import matplotlib.pyplot as plt


class EmojifyModel():
    def __init__(self) -> None:
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.cnt = 0
        self.losses = []
        self.val_losses = []
        self.accur = []
        self.val_accur = []

        pass

    def load_model(self, load_old_model=False):
        if load_old_model == False:
            # use pretrained Albert
            self.model = AlbertForSequenceClassification.from_pretrained(
                "ckiplab/albert-base-chinese",
                num_labels=308,
                output_attentions=False,
                output_hidden_states=False,
            )
        else:
            # use fine-tuned model
            self.model = AlbertForSequenceClassification.from_pretrained(
                "emojify model")
        if torch.cuda.is_available():
            self.model.cuda()

    def save_model(self):
        output_dir = 'emojify model'
        model_to_save = self.model.module if hasattr(
            self.model, 'module') else self.model
        model_to_save.save_pretrained(output_dir)
        return

    def train(self, train_dataloader, validation_dataloader, epochs=4, lr=2e-5, eps=1e-8):
        if train_dataloader == None:
            print("There is no training data for training.")
            return
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.epochs = epochs
        total_steps = len(self.train_dataloader) * self.epochs
        # set up optimizer and scheduler
        self.optimizer = AdamW(self.model.parameters(), lr=lr, eps=eps)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=0, num_training_steps=total_steps)
        # start to run training and validation
        self.__eval_train__()
        if self.validation_dataloader != None:
            self.__eval_validation__()

        for i in range(self.epochs):
            print(
                '\n========== Epoch {:} / {:} =========='.format(i + 1, epochs))
            self.__run_training__()
            self.__eval_train__()
            if self.validation_dataloader != None:
                self.__eval_validation__()

    def __eval_train__(self):
        total_train_loss = 0
        total_eval_accuracy = 0
        self.model.eval()
        for batch in self.train_dataloader:
            # copy the data into GPU
            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)
            b_labels = batch[2].to(self.device)
            # reset the grad
            with torch.no_grad():
                loss, logits = self.model(b_input_ids, token_type_ids=None,
                                          attention_mask=b_input_mask, labels=b_labels)[:2]
            total_train_loss += loss.item()
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            total_eval_accuracy += self.flat_accuracy(logits, label_ids)

        avg_train_loss = total_train_loss / len(self.train_dataloader)
        self.losses.append(avg_train_loss)
        avg_train_accuracy = total_eval_accuracy / \
            len(self.train_dataloader)
        self.accur.append(avg_train_accuracy)
        print("-> Accuracy: {0:.2f}".format(avg_train_accuracy))
        print("-> Average training loss: {0:.2f}".format(avg_train_loss))

    def __eval_validation__(self):
        print("Running Validation...")
        t0 = time.time()
        # change model to examination model
        self.model.eval()

        total_eval_accuracy = 0
        total_eval_loss = 0
        # Evaluate data for one epoch
        for batch in self.validation_dataloader:
            # copy the data into GPU
            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)
            b_labels = batch[2].to(self.device)
            # no need to update model while running validation
            with torch.no_grad():
                loss, logits = self.model(b_input_ids,
                                          token_type_ids=None,
                                          attention_mask=b_input_mask,
                                          labels=b_labels)[:2]

            total_eval_loss += loss.item()

            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            total_eval_accuracy += self.flat_accuracy(logits, label_ids)

        validation_time = self.__format_time__(time.time() - t0)
        print("-> Validation took: {:}".format(validation_time))

        avg_val_accuracy = total_eval_accuracy / \
            len(self.validation_dataloader)
        self.val_accur.append(avg_val_accuracy)
        print("-> Accuracy: {0:.2f}".format(avg_val_accuracy))

        avg_val_loss = total_eval_loss / len(self.validation_dataloader)
        self.val_losses.append(avg_val_loss)
        print("-> Validation Loss: {0:.2f}".format(avg_val_loss))

    def flat_accuracy(self, preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    def __format_time__(self, elapsed):
        elapsed_rounded = int(round((elapsed)))
        return str(datetime.timedelta(seconds=elapsed_rounded))

    def __run_training__(self):
        print("Running Training...")
        t0 = time.time()
        total_train_loss = 0
        total_eval_accuracy = 0
        self.model.train()
        for batch in self.train_dataloader:
            # copy the data into GPU
            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)
            b_labels = batch[2].to(self.device)
            # reset the grad
            self.model.zero_grad()

            loss, logits = self.model(b_input_ids, token_type_ids=None,
                                      attention_mask=b_input_mask, labels=b_labels)[:2]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            # update
            self.optimizer.step()
            self.scheduler.step()

        training_time = self.__format_time__(time.time() - t0)
        print("-> Training epoch took: {:}".format(training_time))
        return

    def showLossFigure(self):
        plt.figure(figsize=(6, 4))
        plt.plot(np.squeeze(self.losses))
        plt.plot(np.squeeze(self.val_losses))
        plt.ylabel('average loss')
        plt.xlabel('epoch')
        plt.show()

    def showAccurFigure(self):
        plt.figure(figsize=(6, 4))
        plt.plot(np.squeeze(self.accur))
        plt.plot(np.squeeze(self.val_accur))
        plt.ylabel('average accuracy')
        plt.xlabel('epoch')
        plt.show()
