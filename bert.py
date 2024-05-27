import numpy as np
import os
import torch

from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
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

def test_model(model, dataloader, data_size):
    predicts, truth = np.array([], dtype='int8'), np.array([], dtype='int8')
    with tqdm(range(test_num), desc='Test') as tbar:
        for batch in dataloader:
            outputs = model(batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device))
            predicts = np.append(predicts, np.argmax(outputs.logits.cpu().detach().numpy(), axis=1))
            truth = np.append(truth, batch['labels'].numpy())
            if predicts.shape[0] % 400 == 0:
                tbar.update(400)
    print(np.sum(predicts == truth)/data_size)
    print(np.unique(predicts, return_counts=True))

if __name__ == '__main__':
    #split_data('seq_type.txt', 'data')
    texts_train, labels_train = get_data(os.path.join('data', 'sub_train.csv'))
    texts_test, labels_test = get_data(os.path.join('data', 'sub_test.csv'))
    device = torch.device('cuda:0')

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
    optimizer = AdamW(model.parameters(), lr=1e-5)
    model.to(device)

    torch.cuda.empty_cache()
    
    dataset = TextClassificationDataset(texts_train, labels_train, tokenizer)
    dataloader = DataLoader(dataset, batch_size=4)
    train_num = len(texts_train)
    test_dataset = TextClassificationDataset(texts_test, labels_test, tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=8)
    test_num = len(texts_test)

    test_model(model, test_dataloader, test_num)
    # 训练模型
    model_path = 'model_save1'
    os.makedirs(model_path, exist_ok=True)
    for epoch in range(5):
        for batch in dataloader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']

            outputs = model(input_ids.to(device), attention_mask=attention_mask.to(device), labels=labels.to(device))
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
        print(f'Epoch {epoch + 1} completed')
        torch.save(model, os.path.join(model_path, f'epoch{epoch}.model'))
        test_model(model, test_dataloader, test_num)

