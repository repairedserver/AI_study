from sklearn.datasets import load_boston #보스턴 집 값 데이터
import pandas as pd #판다스
import numpy as np #넘파이
from sklearn.model_selection import train_test_split # 학습 데이터와 평가 데이터로 나눔
from sklearn.preprocessing import MinMaxScaler #데이터 내 값을 0이상 1이하로 조정

#인공신경망
import torch
from torch import nn, optim #torch 내 세부적 기능
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

from sklearn.metrics import mean_squared_error #로스

import matplotlib.pyplot as plt #시각화

bos = load_boston() #데이터불러오기
df = pd.DataFrame(bos.data)
df.columns = bos.feature_names
df['Price'] = bos.target

#df.head()는 상위 5줄만 보여줌
df.head(10) #상위 10줄을 보여줌

#데이터를 넘파이 배열로
X = df.drop('Price', axis=1).to_numpy() #타겟값 price 제외 넘파이 배열로
Y = df['Price'].to_numpy().reshape((-1, 1))

#데이터 스케일링
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)

scaler.fit(Y)
Y = scaler.transform(Y)

class TensorData(Dataset):
  def __init__(self, x_data, y_data):
    self.x_data = torch.FloatTensor(x_data)
    self.y_data = torch.FloatTensor(y_data)
    self.len = self.y_data.shape[0]

  def __getitem__(self, index):
    return self.x_data[index], self.y_data[index]

  def __len__(self):
    return self.len

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5)

#학습 데이터, 시험 데이터 배치형태로 구축
trainsets = TensorData(X_train, Y_train)
trainloader = torch.utils.data.DataLoader(trainsets, batch_size=32, shuffle=True)

testsets = TensorData(X_test, Y_test)
testloader = torch.utils.data.DataLoader(testsets, batch_size=32, shuffle=False)

class Regressor(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(13, 50, bias=True) #입력층 _> 은닉층1
    self.fc2 = nn.Linear(50, 30, bias=True) #은닉층1 -> 은닉층2
    self.fc3 = nn.Linear(30, 1, bias=True) # 은닉충2 -> 출력층
    self.dropout = nn.Dropout(0.2)

  def forward(self, x): #모델 연산 순서
    x = F.relu(self.fc1(x)) #Linear 계산 후 함수 ReLU 적용
    x = self.dropout(F.relu(self.fc2(x)))
    x = F.relu(self.fc3(x))

    return x

model = Regressor()
criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-7)

loss_ = []
n = len(trainloader)

for epoch in range(400):  # 400번 학습
    running_loss = 0.0

    for i, data in enumerate(trainloader, 0):
        inputs, values = data

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, values)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    loss_.append(np.sqrt(running_loss) / n)

print('Finished Training')

plt.plot(loss_)
plt.title("Training Loss")
plt.xlabel("epoch")
plt.show()


def evaluation(dataloader):
    predictions = torch.tensor([], dtype=torch.float)  # 예측값 저장
    actual = torch.tensor([], dtype=torch.float)  # 실제값 저장

    with torch.no_grad():
        model.eval()  # 평가시 반드시 .eval() 사용
        for data in dataloader:
            inputs, values = data
            outputs = model(inputs)

            predictions = torch.cat((predictions, outputs), 0)  # cat을 통해 예측값 누적
            actual = torch.cat((actual, values), 0)  # cat을 통해 실제값 누적

        predictions = predictions.numpy()  # 넘파이 배열로 변경
        actual = actual.numpy()
        rmse = np.sqrt(mean_squared_error(predictions, actual))  # sklearn을 이용해 RMSE 계산

        return rmse

<<<<<<< HEAD
train_rmse = evaluation(trainloader) #학습 데이터의 MSE
test_rmse = evaluation(testloader) #시험 데이터의 MSE
=======
train_rmse = evaluation(trainloader)
test_rmse = evaluation(testloader)
>>>>>>> master

print("Train RMSE: ", train_rmse)
print("Test RMSE: ", test_rmse)