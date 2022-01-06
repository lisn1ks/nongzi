import torch
from transformers import BertTokenizer, BertModel, BertConfig
import numpy as np
from torch.utils import data
from sklearn.model_selection import train_test_split
import pandas as pd
import tqdm

# https://www.pythonheidong.com/blog/article/551205/8f04eb6a0d254ce2fad2/

pretrained = 'voidful/albert_chinese_small'  # 使用small版本Albert
tokenizer = BertTokenizer.from_pretrained(pretrained)
model = BertModel.from_pretrained(pretrained)
config = BertConfig.from_pretrained(pretrained)

inputtext = "今天心情情很好啊，买了很多东西，我特别喜欢，终于有了自己喜欢的电子产品，这次总算可以好好学习了"
tokenized_text = tokenizer.encode(inputtext)
input_ids = torch.tensor(tokenized_text).view(-1, len(tokenized_text))
outputs = model(input_ids)


class AlbertClassfier(torch.nn.Module):
    def __init__(self, bert_model, bert_config, num_class):
        super(AlbertClassfier, self).__init__()
        self.bert_model = bert_model
        self.dropout = torch.nn.Dropout(0.4)
        self.fc1 = torch.nn.Linear(bert_config.hidden_size, bert_config.hidden_size)
        self.fc2 = torch.nn.Linear(bert_config.hidden_size, num_class)

    def forward(self, token_ids):
        bert_out = self.bert_model(token_ids)[1]  # 句向量 [batch_size,hidden_size]
        bert_out = self.dropout(bert_out)
        bert_out = self.fc1(bert_out)
        bert_out = self.dropout(bert_out)
        bert_out = self.fc2(bert_out)  # [batch_size,num_class]
        return bert_out


albertBertClassifier = AlbertClassfier(model, config, 2)
device = torch.device("cuda:0") if torch.cuda.is_available() else 'cpu'
albertBertClassifier = albertBertClassifier.to(device)


def get_train_test_data(pos_file_path, neg_file_path, max_length=50, test_size=0.2):
    data = []
    label = []
    pos_df = pd.read_excel(pos_file_path, header=None)
    pos_df.columns = ['a.live_stream_id', 'b.author_id', 'fixed_result', 'koubo.start_time', 'a.key_cnt']
    for index, row in pos_df.iterrows():
        row = row['fixed_result']
        ids = tokenizer.encode(row.strip(), max_length=max_length, padding='max_length', truncation=True)
        data.append(ids)
        label.append(1)

    neg_df = pd.read_excel(neg_file_path, header=None)
    neg_df.columns = ['a.live_stream_id', 'b.author_id', 'fixed_result', 'koubo.start_time', 'a.key_cnt']
    for index, row in neg_df.iterrows():
        row = row['fixed_result']
        ids = tokenizer.encode(row.strip(), max_length=max_length, padding='max_length', truncation=True)
        data.append(ids)
        label.append(0)
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=test_size, shuffle=True)
    return (X_train, y_train), (X_test, y_test)


pos_file_path = "./data/idp_report_pos.xlsx"
neg_file_path = "./data/idp_report_neg.xlsx"
(X_train, y_train), (X_test, y_test) = get_train_test_data(pos_file_path, neg_file_path)
len(X_train), len(X_test), len(y_train), len(y_test), len(X_train[0])


class DataGen(data.Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return np.array(self.data[index]), np.array(self.label[index])


train_dataset = DataGen(X_train, y_train)
test_dataset = DataGen(X_test, y_test)
train_dataloader = data.DataLoader(train_dataset, batch_size=32)
test_dataloader = data.DataLoader(test_dataset, batch_size=32)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(albertBertClassifier.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)

for epoch in range(50):
    print('Epoch: ', epoch)
    loss_sum = 0.0
    accu = 0
    albertBertClassifier.train()
    for step, (token_ids, label) in enumerate(train_dataloader):
        token_ids = token_ids.to(device)
        label = label.to(device)
        out = albertBertClassifier(token_ids)
        loss = criterion(out, label)
        optimizer.zero_grad()
        loss.backward()  # 反向传播
        optimizer.step()  # 梯度更新
        loss_sum += loss.cpu().data.numpy()
        accu += (out.argmax(1) == label).sum().cpu().data.numpy()

    test_loss_sum = 0.0
    test_accu = 0
    albertBertClassifier.eval()
    for step, (token_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.to(device)
        label = label.to(device)
        with torch.no_grad():
            out = albertBertClassifier(token_ids)
            loss = criterion(out, label)
            test_loss_sum += loss.cpu().data.numpy()
            test_accu += (out.argmax(1) == label).sum().cpu().data.numpy()
    print("train loss:%f,train acc:%f,test loss:%f,test acc:%f" % (
        loss_sum / len(train_dataset), accu / len(train_dataset), test_loss_sum / len(test_dataset),
        test_accu / len(test_dataset)))
