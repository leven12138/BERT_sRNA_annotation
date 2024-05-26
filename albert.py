import numpy as np
import os
import torch

from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AlbertTokenizer, AlbertForSequenceClassification
from sklearn.model_selection import train_test_split

class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        inputs = self.tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors="pt")
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def get_data(txt_path):
    texts, labels = [], []
    with open(txt_path, 'r') as f:
        for line in f.readlines():
            content = line.rstrip().split(',')
            assert len(content) == 2
            texts.append(content[1])
            if content[0] == 'rsRNA':
                labels.append(0)
            elif content[0] == 'tsRNA':
                labels.append(1)
            else:
                labels.append(2)
    return texts, labels

def out_data(texts, labels, outfile):
    with open(outfile, 'w') as ofile:
        for text, label in zip(texts, labels):
            if label == 0:
                RNA_type = 'rsRNA'
            elif label == 1:
                RNA_type = 'tsRNA'
            else:
                RNA_type = 'other'
            ofile.write(f'{RNA_type},{text}\n')

def split_data(src_data, out_path, rate=0.2):
    texts, labels = get_data(src_data)
    texts_train, texts_test, labels_train, labels_test = train_test_split(texts, labels, test_size=rate, shuffle=True)
    out_data(texts_train, labels_train, os.path.join(out_path, 'train.csv'))
    out_data(texts_test, labels_test, os.path.join(out_path, 'test.csv'))

if __name__ == '__main__':
    #split_data('seq_type.txt', 'data')
    texts_train, labels_train = get_data(os.path.join('data', 'train.csv'))
    texts_test, labels_test = get_data(os.path.join('data', 'test.csv'))

    tokenizer = AlbertTokenizer.from_pretrained("textattack/albert-base-v2-imdb")
    dataset = TextClassificationDataset(texts_train, labels_train, tokenizer)
    dataloader = DataLoader(dataset, batch_size=4)
    device = torch.device('cuda')
    model = AlbertForSequenceClassification.from_pretrained("textattack/albert-base-v2-imdb", num_labels=3, ignore_mismatched_sizes=True)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=1e-5)
    val_dataset = TextClassificationDataset(texts_test, labels_test, tokenizer)
    val_dataloader = DataLoader(val_dataset, batch_size=8)
    val_num = len(texts_test)

    """predicts, truth = np.array([], dtype='int8'), np.array([], dtype='int8')
    for batch in val_dataloader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        outputs = model(batch['input_ids'], attention_mask=batch['attention_mask'])
        predicts = np.append(predicts, np.argmax(outputs.logits.detach().numpy(), axis=1))
        truth = np.append(truth, batch['labels'].numpy())
    print(np.sum(predicts == truth)/val_num)
    print(np.unique(predicts, return_counts=True))"""

    # 训练模型
    model_path = 'albert'
    os.makedirs(model_path, exist_ok=True)
    for epoch in range(10):
        for batch in dataloader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']

            outputs = model(input_ids.to(device), attention_mask=attention_mask.to(device), labels=labels.to(device))
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
        
        torch.save(model, os.path.join(model_path, f'epoch{epoch}.model'))
        """predicts, truth = np.array([], dtype='int8'), np.array([], dtype='int8')
        for batch in val_dataloader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']

            outputs = model(batch['input_ids'], attention_mask=batch['attention_mask'])
            predicts = np.append(predicts, np.argmax(outputs.logits.detach().numpy(), axis=1))
            truth = np.append(truth, batch['labels'].numpy())
        print(np.sum(predicts == truth)/val_num)
        print(np.unique(predicts, return_counts=True))"""

        print(f'Epoch {epoch + 1} completed')
# 假设texts和labels分别是文本和标签的列表
