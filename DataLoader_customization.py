import torch
import torchvision.transforms as tr #이미지 전처리 기능
from torch.utils.data import DataLoader, Dataset # 데이터를 모델에 사용할수 있도록 정리
import numpy as np

# 32x32 이미지와 100장
train_images = np.random.randint(256, size=(100, 32, 32, 3)) #이미지수x넓이x채널
train_labels = np.random.randint(2, size=(100, 1))

#transform 추가
class MyDataset(Dataset):
  def __init__(self, x_data, y_data, transform=None):
    self.x_data = x_data
    self.y_data = y_data
    self.transform = transform
    self.len = len(y_data)

  def __getitem__(self, index):
    sample = self.x_data[index], self.y_data[index]

    if self.transform:
      sample = self.transform(sample) #self.transform이 None이 아니라면 작업

    return sample

  def __len__(self):
    return self.len

# 텐서로 변환
class ToTensor:
    def __call__(self, sample):
        inputs, labels = sample
        inputs = torch.FloatTensor(inputs)  # 텐서로 변환
        inputs = inputs.permute(2, 0, 1)  # 크기 변환
        return inputs, torch.LongTensor(labels)  # 텐서로 변환

class LinearTensor:
    def __init__(self, slope=1, bias=0):
        self.slope = slope
        self.bias = bias

    def __call__(self, sample):
        inputs, labels = sample
        inputs = self.slope * inputs + self.bias  # ax + b
        return inputs, labels

trans = tr.Compose([ToTensor(), LinearTensor(2, 5)])
dataset1 = MyDataset(train_images, train_labels, transform = trans)
train_loader1 = DataLoader(dataset1, batch_size=10, shuffle=True)