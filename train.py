import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from collections import defaultdict
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.optim import AdamW


class AbsGenDataset(Dataset):
    def __init__(self, tokenizer, texts):
        self.tokenizer = tokenizer
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokenized_text = self.tokenizer(self.texts[idx],
                                        truncation=True,
                                        max_length=1024,
                                        padding='max_length',
                                        add_special_tokens=True,
                                        return_tensors='pt'
                                        )
        return {
            'input_ids': tokenized_text['input_ids'].flatten(),
            'attention_mask': tokenized_text['attention_mask'].flatten()
        }


class Trainer:
    def __init__(self):
        self.data = pd.read_csv('arxiv_cs-CL.csv')
        with open('config.json', 'rb') as f:
            self.config = json.load(f)
        self.config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_val_df, self.test_df = train_test_split(self.data,
                                                      test_size=self.config['holdout_test_frac'],
                                                      random_state=42)
        self.train_df, self.val_df = train_test_split(train_val_df,
                                                      train_size=self.config['train_frac'],
                                                      random_state=42)
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2',
                                                       bos_token=self.config['bos'],
                                                       eos_token=self.config['eos'],
                                                       pad_token=self.config['pad'],
                                                       unk_token=self.config['unk'])

    def set_seed(self):
        np.random.seed(self.config['seed'])
        torch.manual_seed(self.config['seed'])
        torch.cuda.manual_seed(self.config['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        os.environ['PYTHONHASHSEED'] = str(self.config['seed'])

    def prepare_data(self):
        self.train_df['title'] = self.train_df['title'].apply(lambda text: ' '.join([x for x in text.split()]))
        self.train_df['abstract'] = self.train_df['abstract'].apply(lambda text: ' '.join([x for x in text.split()]))
        self.train_df['input_text'] = self.config['bos'] + ' ' + self.train_df['title'] + ' ' + '<|SEP|>' + ' ' + \
                                      self.train_df['abstract'] + ' ' + self.config['eos']
        self.val_df['title'] = self.val_df['title'].apply(lambda text: ' '.join([x for x in text.split()]))
        self.val_df['abstract'] = self.val_df['abstract'].apply(lambda text: ' '.join([x for x in text.split()]))
        self.val_df['input_text'] = self.config['bos'] + ' ' + self.val_df['title'] + ' ' + '<|SEP|>' + ' ' + \
                                    self.val_df['abstract'] + ' ' + self.config['eos']
        self.test_df['title'] = self.test_df['title'].apply(lambda text: ' '.join([x for x in text.split()]))
        self.test_df['prompt'] = self.config['bos'] + ' ' + self.test_df['title'] + ' ' + '<|SEP|>'
        return self.train_df.reset_index(), self.val_df.reset_index(), self.test_df.reset_index()

    def create_dataloaders(self):
        train_df, val_df, _ = self.prepare_data()
        train_dataset = AbsGenDataset(self.tokenizer, train_df['input_text'].tolist())
        val_dataset = AbsGenDataset(self.tokenizer, val_df['input_text'].tolist())
        train_dataloader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True,
                                      pin_memory=True)
        val_dataloader = DataLoader(val_dataset, batch_size=self.config['batch_size'], shuffle=False, pin_memory=True)
        return train_dataloader, val_dataloader

    def train_loop(self, model, dataloader, optimizer):
        model.train()
        scaler = torch.cuda.amp.GradScaler()
        batch_losses = []
        for batch_num, batch in tqdm(enumerate(dataloader)):
            input_ids = batch['input_ids'].to(self.config['device'], non_blocking=True)
            attention_mask = batch['attention_mask'].to(self.config['device'], non_blocking=True)
            with torch.cuda.amp.autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids, return_dict=False)
            batch_loss = outputs[0]
            batch_loss = batch_loss / self.config['n_accumulate']
            batch_losses.append(batch_loss.item())
            scaler.scale(batch_loss).backward()
            if (batch_num + 1) % self.config['n_accumulate'] == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        return np.mean(batch_losses)

    @torch.no_grad()
    def val_loop(self, model, dataloader):
        model.eval()
        batch_losses = []
        for batch_num, batch in tqdm(enumerate(dataloader)):
            input_ids = batch['input_ids'].to(self.config['device'], non_blocking=True)
            attention_mask = batch['attention_mask'].to(self.config['device'], non_blocking=True)
            with torch.cuda.amp.autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids, return_dict=False)
            batch_loss = outputs[0]
            batch_loss = batch_loss / self.config['n_accumulate']
            batch_losses.append(batch_loss.item())

        return np.mean(batch_losses)

    def run_training(self):
        train_dataloader, val_dataloader = self.create_dataloaders()
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        model.resize_token_embeddings(len(self.tokenizer))
        model.to(self.config['device'])
        optimizer = AdamW(model.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'])
        history = defaultdict(list)
        earlystop_trigger = 0
        best_val_loss, prev_val_loss = np.inf, np.inf
        for epoch in range(self.config['epochs']):
            print(f'Epoch {epoch + 1} of ' + str(self.config['epochs']))
            train_loss = self.train_loop(model, train_dataloader, optimizer)
            print(f'Train Loss: {train_loss}')
            val_loss = self.val_loop(model, val_dataloader)
            print(f'Val Loss: {val_loss}')
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            if val_loss <= prev_val_loss:
                earlystop_trigger = 0
            else:
                earlystop_trigger += 1
                if earlystop_trigger == self.config['patience']:
                    print('Early Stopping Triggered. Aborting Training.')
                    break
            if val_loss <= best_val_loss:
                best_val_loss = val_loss
                print(f'New Best Validation Loss. Saving model checkpoint at epoch {epoch + 1}.')
                model_to_save = model.module if hasattr(model, 'module') else model
                model_to_save.save_pretrained(self.config['saved_checkpoint'])
                self.tokenizer.save_pretrained(self.config['saved_checkpoint'])
            prev_val_loss = val_loss

        return history


if __name__ == '__main__':
    trainer = Trainer()
    trainer.set_seed()
    history = trainer.run_training()
