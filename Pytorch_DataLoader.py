"""
from.torch.utils.data import Dataset

class MyDataset(Dataset)
  def __init__(self):

  def __getitem__(self, index):

  def __len__(self):

통으로 가지고 다니기

"""
import torch
import torchvision
import torchvision.transforms as tr
from torch.utils.data import DataLoader, Dataset
import numpy as np


transf = tr.Compose([tr.Resize(16), tr.ToTensor()])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transf)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transf)

print(trainset[0][0].size())

trainloader = DataLoader(trainset, batch_size=50, shuffle=True)
testloader = DataLoader(testset, batch_size=50, shuffle=False)

len(trainloader)

dataiter = iter(trainloader)
images, labels = dataiter.next()

print(images.size())

train_images = np.random.randint(256, size=(100, 2, 32, 3))
train_labels = np.random.randint(2, size=(100, 1))
print(train_images.shape, train_labels.shape)

class TensorData(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = torch.FloatTensor(x_data)  # 이미지 데이터를 FloatTensor로 변경
        self.x_data = self.x_data.permute(0, 3, 1, 2)  # 이미지 수 x 너비 x 높이 x 채널 -> 배치크기 x 채널 수 x 너비 x 높이
        self.y_data = torch.LongTensor(y_data)  # 라벨 데이터를 LongTensor로 변형
        self.len = self.y_data.shape[0]  # 클래스 내의 들어온 데이터 개수

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]  # 뽑아낼 데이터를 적어줌

    def __len__(self):
        return self.len  # 클래스 내의 들어온 데이터 개수