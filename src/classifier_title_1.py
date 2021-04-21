import spacy
import pandas 
import os
import sys
import json 
from spacy.tokenizer import Tokenizer
import glob
import multiprocessing as mp
from joblib import Parallel, delayed
import numpy as np
import torch
from transformers import AutoTokenizer
from transformers import AutoModel
from transformers import AdamW
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast, ReformerTokenizer,ReformerTokenizerFast
from transformers import DistilBertForSequenceClassification, BertForSequenceClassification, ReformerForSequenceClassification
from transformers import Trainer, TrainingArguments
from torch.utils.data import DataLoader
from torch.nn import functional as F
import matplotlib.pyplot as plt
from transformers import AlbertTokenizer, AlbertModel
from sklearn.metrics import classification_report
from tqdm import tqdm
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

NLP_object = spacy.load("en_core_web_md")

def obtain_title_cleaned_v1(text):
    global NLP_object
    doc = NLP_object(text)
    punct_removed_title = [ _.lemma_.lower() for _ in doc if not _.is_punct]
    return punct_removed_title

labelled_data_folders = {
    1 : './../../Data_042021/ground_truth/positive/',
    0 : './../../Data_042021/ground_truth/negative/',
}

def get_titles(file):  
    object_dict = {}
    with open(file, 'r') as fh:
        for _line_ in fh.readlines():
            _obj_ = json.loads(_line_)
            _id_ = _obj_['id'] 
            object_dict[_id_] = _obj_ 
            
    text_list = Parallel(n_jobs=mp.cpu_count(), prefer="threads")(delayed(obtain_title_cleaned_v1)(obj['title'], ) for obj in object_dict.values())
    return text_list


# ===========================
# Return titles
# ===========================
def get_labelled_titles():
    labelled_title_dict = {}
    for label, folder in labelled_data_folders.items():
        files = glob.glob(os.path.join(folder, '**.json'))
        title_list = []
        for file in files:
            _results = get_titles(file)
            title_list.extend(_results)
        labelled_title_dict[label] = title_list
    return labelled_title_dict

def get_tokenizer(model_type = 'BERT'):
    if model_type =='distilBERT':
        tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    elif model_type =='BERT':
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    elif model_type =='alBERT':
        tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
    else:
        print('model_type not allowed ', model_type)
    return tokenizer


def get_labelled_data():
    labelled_title_dict = get_labelled_titles()
    X = []
    Y = []
    for _label, _x in labelled_title_dict.items():
        
        if _label == 1 :
            labels = np.ones(len(_x), dtype=int).tolist()
        if _label == 0 :
            labels = np.zeros(len(_x),  dtype=int).tolist()
        X.extend(_x)
        Y.extend(labels)
    return X,Y
        

def get_train_test_data():
    text, labels = get_labelled_data()  
    train_text, test_text, train_label, test_label = train_test_split(text, labels, test_size=0.2)
    return train_text, test_text, train_label, test_label

class titleDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def create_datasets(model_type='BERT'):
    tokenizer = get_tokenizer(model_type)
    train_text, test_text, train_label, test_label = get_train_test_data()
    train_encodings = tokenizer(train_text, is_split_into_words=True, padding=True, truncation=True,  return_tensors='pt')
    test_encodings = tokenizer(test_text, is_split_into_words=True, padding=True, truncation=True,  return_tensors='pt')
    train_dataset = titleDataset(train_encodings, train_label)
    test_dataset = titleDataset(test_encodings, test_label)
    return train_dataset, test_dataset

def run_model(model_type, num_epochs = 100, log_interval=75):
    train_dataset, test_dataset = create_datasets(model_type)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)
    if model_type=='BERT':
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    elif model_type=='distilBERT':
        model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
    elif model_type=='alBERT':
        model = AlbertForSequenceClassification.from_pretrained('albert-base-v2')
    model.train()
    optim = AdamW(model.parameters(), lr=5e-5)
    for epoch in tqdm(range(num_epochs)):
        b_idx = 0
        for batch in train_loader:
            optim.zero_grad()
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            if b_idx % log_interval == 0:
                print('Loss :: {:.4f}'.format(np.mean(loss.cpu().data.numpy())))
            loss.backward()
            optim.step()
            b_idx+=1
            
    model.eval()
    y_pred = []
    y_true = []
    for batch in test_loader:
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)
        outputs = model(input_ids, attention_mask=attention_mask)
        predicted_label = torch.argmax(F.softmax(outputs.logits, dim=-1),dim=-1, keepdims=False)
        _pred_y = predicted_label.cpu().data.numpy().tolist()
        _true_y = labels.cpu().data.numpy().tolist()
        y_pred.extend(_pred_y)
        y_true.extend(_true_y)
    print('MODEL {}'.format(model_type))
    print(classification_report(y_true, y_pred))
    report = classification_report(y_true, y_pred)
    return report



model_type = 'BERT'
report1 = run_model(model_type=model_type, num_epochs = 2)
with open('./results_{}.txt'.format(model_type),'w+') as fh:
    fh.write(report1)

model_type = 'alBERT'
report2 = run_model(model_type=model_type, num_epochs = 100)
with open('./results_{}.txt'.format(model_type),'w+') as fh:
    fh.write(report2)


model_type = 'distilBERT'
report3 = run_model(model_type=model_type, num_epochs = 100)
with open('./results_{}.txt'.format(model_type),'w+') as fh:
    fh.write(report3)







