import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

#from google.colab import drive
#drive.mount('/content/gdrive')

#cd/content/gdrive/My Drive/deeplearningbro/deeplearningbro/pytorch

df = pd.read_csv("./data/kospi.csv") #주식 데이따

df.head()

scaler = MinMaxScaler()
df[['Open', 'High', 'Low', 'Close', 'Volume']] = scaler.fit_transform(df[['Open', 'High', 'Low', 'Close', 'Volume']])

df.info()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'{device} is available')

#dataset
X = df[['Open', 'High', 'Low', 'Volume']].values
y = df['Close'].values

def seq_data(x, y, sequense_length):
  x_seq = []
  y_seq = []
  for i in range(len(x)-sequense_length):
    x_seq.append(x[i:i+sequense_length]) # a[2:6] -> 2,3,4,5
    y_seq.append(y[i+sequense_length])

  return torch.FloatTensor(x_seq).to(device), torch.FloatTensor(y_seq).to(device)

split = 200
sequense_length = 5

x_seq, y_seq = seq_data(X, y, sequense_length)
x_train_seq = x_seq[:split]
y_train_seq = y_seq[:split]
x_test_seq = x_seq[split:]
y_test_seq = y_seq[split:]
print(x_train_seq.size(), y_train_seq.size())
print(x_test_seq.size(), y_test_seq.size())

train = torch.utils.data.TensorDataset(x_train_seq, y_train_seq)
test = torch.utils.data.TensorDataset(x_test_seq, y_test_seq)

batch_size = 20
train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=batch_size, shuffle=False)

##RNN
input_size = x_seq.size(2)
num_layers = 2
hidden_size = 8

class VanillaRNN(nn.Module):
  def __init__(self, input_size, hidden_size, sequense_length, num_layers, device):
    super(VanillaRNN, self).__init__()
    self.device = device
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
    self.fc = nn.Sequential(nn.Linear(hidden_size*sequense_length, 1), nn.Sigmoid())

  def forward(self, x):
    h0 = torch.zeros(self.num_layers, x.size()[0], self.hidden_size).to(self.device)
    out, _ = self.rnn(x, h0) #out: RNN의 마지막 레이어로부터 나온 output feature 변환, hh: hidden state 반환
    out = out.reshape(out.shape[0], -1) #many to many 전략
    out = self.fc(out)
    return out

model = VanillaRNN(input_size = input_size,
                   hidden_size = hidden_size,
                   sequense_length = sequense_length,
                   num_layers = num_layers,
                   device = device).to(device)

criterion = nn.MSELoss() #주가를 예측하기 때문에 연속적인 값 예측 MSE 사용

lr=1e-3
num_epoch = 200
optimizer = optim.Adam(model.parameters(), lr=lr)

loss_graph = []
n = len(train_loader)

for epoch in range(num_epoch):
  running_loss = 0.0

  for data in train_loader:
    seq, target = data
    out = model(seq)
    loss = criterion(out, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    running_loss += loss.item()

  loss_graph.append(running_loss/n)
  if epoch % 100 == 0:
    print('[epoch: %d] loss: %.4f' %(epoch, running_loss/n))

plt.figure(figsize=(20,10))
plt.plot(loss_graph)
plt.show()


def plotting(train_loader, test_loader, actual):
    with torch.no_grad():
        train_pred = []
        test_pred = []

        for data in train_loader:
            seq, target = data
            out = model(seq)
            train_pred += out.cpu().numpy().tolist()

        for data in test_loader:
            seq, target = data
            out = model(seq)
            test_pred += out.cpu().numpy().tolist()

        total = train_pred + test_pred
        plt.figure(figsize=(20, 10))
        plt.plot(np.ones(100) * len(train_pred), np.linspace(0, 1, 100), '--', linewidth=1)
        plt.plot(actual, '--')
        plt.plot(total, 'b', linewidth=0.8)

        plt.legend(['train boundary', 'actual', 'prediction'])
        plt.show()


plotting(train_loader, test_loader, df['Close'][sequense_length:])