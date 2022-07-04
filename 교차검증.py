from google.colab import drive
drive.mount('/content/gdrive')

cd/content/gdrive/My Drive/deeplearningbro/deeplearningbro/pytorch

import pandas as pd #데이터프레임
import numpy as np
from sklearn.model_selection import train_test_split

#ANN
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

#Cross Validation
from sklearn.model_selection import KFold

#Loss
from sklearn.metrics import mean_squared_error

#Plot
import matplotlib.pyplot as plt

df = pd.read_csv('./data/reg.csv', index_col=[0])

X = df.drop('Price', axis=1).to_numpy() #타겟값 제외 넘파이 배열
Y = df['Price'].to_numpy().reshape((-1, 1))

class TensorData(Dataset):
  def __init__(self, x_data, y_data):
    self.x_data = torch.FloatTensor(x_data)
    self.y_data = torch.FloatTensor(y_data)
    self.len = self.y_data.shape[0]

  def __getitem__(self, index):
    return self.x_data[index], self.y_data[index]

  def __len__(self):
    return self.len

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.7)
trainset = TensorData(X_train, Y_train)
testset = TensorData(X_test, Y_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

class Regressor(nn.Module):
  def __init__(self):
    super().__init__() #모델 연산 정의
    self.fc1 = nn.Linear(13, 50, bias=True)
    self.fc2 = nn.Linear(50, 30, bias=True)
    self.fc3 = nn.Linear(30, 1, bias=True)

  def forward(self, x):
    x = self.fc1(x)
    x = self.fc2(x)
    x = self.fc3(x)

    return x

KFold = KFold(n_splits=3, shuffle=True)
criterion = nn.MSELoss()

def evaluation(dataloader):
  predictions = torch.tensor([], dtype=torch.float)
  actual = torch.tensor([], dtype=torch.float)

  with torch.no_grad():
    model.eval() #평가시 .eval() 반드시 사용
    for data in dataloader:
      inputs, values = data
      outputs = model(inputs)

      predictions = torch.cat((predictions, outputs), 0)
      actual = torch.cat((actual, values), 0)

    predictions = predictions.numpy()
    actual = actual.numpy()
    rmse = np.sqrt(mean_squared_error(predictions, actual))
    model.train()
    return rmse

validation_loss = []
for fold, (train_idx, val_idx) in enumerate(KFold.split(trainset)):
  train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
  val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)

  #sampler를 이용한 DataLoader 정의
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, sampler=train_subsampler)
  valloader = torch.utils.data.DataLoader(trainset, batch_size=32, sampler=val_subsampler)

  model = Regressor()
  optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-7)

  for epoch in range(400):
    for data in trainloader: #무작위로 섞인 32개 데이터가 있는 배치가 하나씩
      inputs, values = data
      optimizer.zero_grad() #최적화 초기화

      outputs = model(inputs) #모델에 입력값 대입후 예측값
      loss = criterion(outputs, values) #손실
      loss.backward() #손실함수 기준으로 역전파
      optimizer.step() #역전파 진행후 가중치

    train_rmse = evaluation(trainloader) #학습 데이터의 RMSE
    val_rmse = evaluation(valloader)
    print("k-fold", fold,"Train Loss = %.4f, Validation Loss = %.4f" %(train_rmse, val_rmse))
    validation_loss.append(val_rmse)

validation_loss = np.array(validation_loss)
mean = np.mean(validation_loss)
std = np.std(validation_loss)
print("Validation Score: %.4f, +-%.4f" %(mean, std))

trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=False)
train_rmse = evaluation(trainloader)
test_rmse = evaluation(testloader)

print("Train RMSE: %.4f" %train_rmse)
print("Test RMSE: %.4f" %test_rmse)