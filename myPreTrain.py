import os
import torch
from torch import nn

from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from module.common.activate import activations
from module.common.nlp_tokenization import MLMTokenizer
from module.BertForMLM import BertForMaskedLM

class PreTrainDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer.encode(text)
        return {
            'input_ids': inputs.input_ids.flatten(),
            'token_type_ids': inputs.token_type_ids.flatten(),
            'attention_mask': inputs.attention_mask.flatten(),
            'masked_lm_labels': inputs.masked_lm_labels.flatten()
        }

def get_data(txt_path):
    texts = []
    with open(txt_path, 'r') as f:
        for line in f.readlines():
            content = line.rstrip().split(',')
            assert len(content) == 2
            texts.append(content[1])
    return texts

def accuracy(mlm_logits, mlm_labels, PAD_IDX):
    mlm_pred = mlm_logits.argmax(axis=2).reshape(-1)
    mlm_true = mlm_labels.reshape(-1)
    mlm_acc = mlm_pred.eq(mlm_true)  # 计算预测值与正确值比较的情况，得到预测正确的个数（此时还包括有mask位置）
    mask = torch.logical_not(mlm_true.eq(PAD_IDX))  # 找到真实标签中，mask位置的信息。 mask位置为FALSE，非mask位置为TRUE
    mlm_acc = mlm_acc.logical_and(mask)  # 去掉mlm_acc中mask的部分
    mlm_correct = mlm_acc.sum().item()
    mlm_total = mask.sum().item()
    mlm_acc = float(mlm_correct) / mlm_total
    return mlm_acc, mlm_correct, mlm_total

if __name__ == '__main__': 
    myMLMbert = BertForMaskedLM("/home/ubuntu/pure_attention/model/config.json")
    mlm_tokenizer = MLMTokenizer("/home/ubuntu/pure_attention/model/vocab.txt")
    device = torch.device('cuda:0')
    
    myMLMbert.to(device)
    optimizer = AdamW(myMLMbert.parameters(), lr=1e-5)
    torch.cuda.empty_cache()

    texts_train = get_data('/home/ubuntu/seq_bert/data/sub_train.csv')
    dataset = PreTrainDataset(texts_train, mlm_tokenizer)
    dataloader = DataLoader(dataset, batch_size=4)
    train_num = len(texts_train)

    # 训练模型
    model_path = 'model_save3'
    os.makedirs(model_path, exist_ok=True)
    epochs = 5
    for epoch in range(epochs):
        for idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            masked_lm_labels = batch['masked_lm_labels'].to(device)

            loss, mlm_logits = myMLMbert(input_ids=input_ids,
                                         token_type_ids=token_type_ids,
                                         attention_mask=attention_mask,
                                         masked_lm_labels=masked_lm_labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            mlm_acc, _, _ = accuracy(mlm_logits, masked_lm_labels, mlm_tokenizer.PAD_IDX)
            torch.cuda.empty_cache()
            if (idx+1) % 200 == 0:
                print(f"Epoch: [{epoch + 1}/{epochs}], Batch[{idx}/{len(dataloader)}], "
                      f"Train loss :{loss.item():.3f}, Train mlm acc: {mlm_acc:.3f}")
        print(f'Epoch {epoch + 1} completed')
        torch.save(myMLMbert, os.path.join(model_path, f'epoch{epoch}.model'))
