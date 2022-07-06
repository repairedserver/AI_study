import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

#from google.colab import drive
#drive.mount('/content/gdrive')

#cd/content/gdrive/My Drive/deeplearningbro/deeplearningbro/pytorch

#CIFAR10 이미지데이터
# plane car bird cat dog deer frog horse ship truck
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                       download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False)

#CPU GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'{device} is available')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.feature_extraction = nn.Sequential(nn.Conv2d(3, 6, 5),
                                                nn.ReLU(),
                                                nn.MaxPool2d(2, 2),
                                                nn.Conv2d(6, 16, 5),
                                                nn.ReLU(),
                                                nn.MaxPool2d(2, 2))

        self.classifier = nn.Sequential(nn.Linear(16 * 5 * 5, 120),
                                        nn.ReLU(),
                                        nn.Linear(120, 20),
                                        nn.ReLU())
    def forward(self, x):
        x = self.feature_extraction(x)
        x = x.view(-1, 16 * 5 * 5)  # 5x5피쳐맵 16개를 일렬로 만든다
        x = self.classifier(x)

        return x
net = Net().to(device)
print(net)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)

loss_ = []
n = len(trainloader) #배치 갯수

for epoch in range(10):
  running_loss = 0.0
  for i, data in enumerate(trainloader, 0):
    inputs, labels = data[0].to(device), data[1].to(device) #배치 데이터

    optimizer.zero_grad()

    outputs = net(inputs) #예측값 산출
    loss = criterion(outputs, labels) #손실함수 계산
    loss.backward() #손실함수 기준으로 역전파 선언
    optimizer.step() #가중치 최적화

    running_loss += loss.item()

  loss_.append(running_loss / n)
  print('[%d] loss: %.3f'%(epoch+1, running_loss / len(trainloader)))

print("Finished Training")

plt.plot(loss_)
plt.title("Training Loss")
plt.xlabel("epoch")
plt.show()

PATH = './cifar_net.pth' # 모델 저장 경로
torch.save(net.state_dict(), PATH) #모델 저장

#평가 데이터를 이용해 정확도 구하기

correct = 0
total = 0
with torch.nn_grad():
  for data in testloader:
    images, labels = data[0].to(device), data[1].to(device)
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' %(100*correct/total))