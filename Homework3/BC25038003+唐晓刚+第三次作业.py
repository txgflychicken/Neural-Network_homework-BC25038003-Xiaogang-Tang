import os
import json
import re
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from torch.utils.data import DataLoader


################################################
# 在这里改模型
################################################

# model_type = "rnn"
# model_type="lstm"
model_type="transformer"

data_dir = "./data"

epochs = 50

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("device:", device)


################################################
# 数据处理
################################################

def clean_text(paragraphs):

    text = ''.join(paragraphs)

    text = re.sub(
        '[，。！？、；：]',
        '',
        text
    )

    return text


def load_poems(data_dir):

    poems = []

    for file in os.listdir(data_dir):

        if not file.endswith(".json"):
            continue

        print("loading:", file)

        path = os.path.join(
            data_dir,
            file
        )

        data = json.load(
            open(
                path,
                "r",
                encoding="utf8"
            )
        )

        for item in data:

            poem = clean_text(
                item["paragraphs"]
            )

            # 七言绝句
            if len(poem) == 28:

                poems.append(
                    poem
                )

    print("poems:", len(poems))

    return poems


################################################
# Dataset
################################################

class PoemDataset(Dataset):

    def __init__(self, poems):

        self.poems = poems

        chars = sorted(
            list(
                set(
                    ''.join(poems)
                )
            )
        )

        self.char2idx = {
            c:i
            for i,c in enumerate(chars)
        }

        self.idx2char = {
            i:c
            for c,i in self.char2idx.items()
        }

        self.vocab_size = len(chars)

        print(
            "vocab:",
            self.vocab_size
        )


    def encode(self,text):

        return [
            self.char2idx[c]
            for c in text
        ]


    def __len__(self):

        return len(
            self.poems
        )


    def __getitem__(self,idx):

        poem = self.poems[idx]

        ids = self.encode(
            poem
        )

        x = torch.tensor(
            ids[:-1],
            dtype=torch.long
        )

        y = torch.tensor(
            ids[1:],
            dtype=torch.long
        )

        return x,y


################################################
# 模型
################################################

class RNNModel(nn.Module):

    def __init__(self,vocab_size):

        super().__init__()

        self.embed = nn.Embedding(
            vocab_size,
            128
        )

        self.rnn = nn.RNN(
            128,
            256,
            num_layers=2,
            batch_first=True
        )

        self.fc = nn.Linear(
            256,
            vocab_size
        )


    def forward(self,x):

        x = self.embed(x)

        out,_ = self.rnn(x)

        out = self.fc(out)

        return out


class LSTMModel(nn.Module):

    def __init__(self,vocab_size):

        super().__init__()

        self.embed = nn.Embedding(
            vocab_size,
            128
        )

        self.rnn = nn.LSTM(
            128,
            256,
            num_layers=2,
            batch_first=True
        )

        self.fc = nn.Linear(
            256,
            vocab_size
        )


    def forward(self,x):

        x = self.embed(x)

        out,_ = self.rnn(x)

        out = self.fc(out)

        return out


class TransformerModel(nn.Module):

    def __init__(self,vocab_size):

        super().__init__()

        self.embed = nn.Embedding(
            vocab_size,
            256
        )

        layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=8,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=4
        )

        self.fc = nn.Linear(
            256,
            vocab_size
        )


    def forward(self,x):

        x = self.embed(x)

        seq_len = x.shape[1]

        mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                device=x.device
            ),
            diagonal=1
        ).bool()

        x = self.transformer(
            x,
            mask=mask
        )

        x = self.fc(x)

        return x


################################################
# 加载数据
################################################

poems = load_poems(
    data_dir
)

dataset = PoemDataset(
    poems
)

loader = DataLoader(

    dataset,

    batch_size=64,

    shuffle=True
)


################################################
# 选择模型
################################################

if model_type=="rnn":

    model = RNNModel(
        dataset.vocab_size
    )

elif model_type=="lstm":

    model = LSTMModel(
        dataset.vocab_size
    )

else:

    model = TransformerModel(
        dataset.vocab_size
    )


model = model.to(
    device
)


################################################
# 训练
################################################

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)

losses = []

for epoch in range(epochs):

    model.train()

    total_loss = 0

    for x,y in loader:

        x = x.to(device)

        y = y.to(device)

        pred = model(x)

        loss = criterion(

            pred.reshape(
                -1,
                pred.shape[-1]
            ),

            y.reshape(-1)
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()


    total_loss /= len(
        loader
    )

    losses.append(
        total_loss
    )

    print(
        f"epoch {epoch+1}, loss={total_loss:.4f}"
    )


################################################
# loss图
################################################

plt.figure()

plt.plot(
    losses
)

plt.title(
    model_type
)

plt.xlabel(
    "epoch"
)

plt.ylabel(
    "loss"
)

plt.show()


################################################
# 生成古诗
################################################

model.eval()

result = list(
    "明月"
)

with torch.no_grad():

    for _ in range(
        28-len(result)
    ):

        ids = dataset.encode(
            result
        )

        x = torch.tensor(
            ids,
            dtype=torch.long
        ).unsqueeze(0)

        x = x.to(
            device
        )

        pred = model(x)

        next_id = pred[
            0,-1
        ].argmax()

        next_char = (
            dataset.idx2char[
                next_id.item()
            ]
        )

        result.append(
            next_char
        )


poem = ''.join(
    result
)

print("\n生成结果：\n")

print(
    poem[:7]+"，"
)

print(
    poem[7:14]+"。"
)

print(
    poem[14:21]+"，"
)

print(
    poem[21:28]+"。"
)