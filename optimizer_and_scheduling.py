import torch
import torchvision

model = torchvision.models.resnet18(pretrained=False)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

for epoch in range(400):
  running_loss = 0.0
  for data in trainloader:
    inputs, values = data
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, values)
    loss.backward()
    optimizer.step()

  scheduler.step()