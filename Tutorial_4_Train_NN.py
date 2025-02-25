#1 loading and normalizing CIFAR10
import torch
import torchvision
import torchvision.transforms as transforms

#transform the images of range[0,1] to Tensors of normalized range[-1,1]
transform = transforms.Compose(
    [transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data',train=True,
                                        download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset,batch_size=4,
                                          shuffle=True, num_workers= 2)


testset = torchvision.datasets.CIFAR10(root='./data',train=False,
                                       download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset,batch_size=4,
                                         shuffle=False, num_workers= 2) #original in the tutorial: both num_workers variables are 2

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

print(torch.__version__)

#show images
import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

#
# # get some random training images
# dataiter = iter(trainloader)
# images, labels = dataiter.next()
# #
# # # show images
# imshow(torchvision.utils.make_grid(images))
# # # print labels
# print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
#
# #DEFINE A CNN
# #Neural Network
import torch.nn as nn
import torch.nn.functional as F
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

class Net(nn.Module):

    def __init__(self):
        #constructor
        super(Net, self).__init__()
        #1 input, 6 output, 5x5 square conv
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        #y = Wx+b
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self,x):
        #max pooling over 2x2 window
        x = self.pool(F.relu(self.conv1(x)))
        #if the size is a square it is possible to specify only one number
        x = self.pool(F.relu(self.conv2(x)))
        #view is a resize
        x = x.view(-1,16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

net = Net()
net.to(device)
print(net)

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),lr=0.001,momentum=0.9)

colours = ['blue','green']
#TRAIN A CNN
for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader,0):
        #input
        inputs,labels = data
        inputs, labels = inputs.cuda(), labels.cuda() #to(device), labels.to(device)

        #zero parameters gradient
        optimizer.zero_grad()

        #forward+backward+optimize
        outputs = net(inputs)
        loss = criterion(outputs,labels)
        # plt.scatter(i, loss.item(),c=colours[epoch])
        # plt.pause(0.05)
        loss.backward()
        optimizer.step()


        running_loss += loss.item()

        if i%2000 == 1999:# print every 2000 mini-batches
            print('[%d,%5d] loss: %3f' %
                  (epoch+1,i+1,running_loss/2000))
            running_loss = 0.0

print('finished training')
# plt.show()

# #TEST
dataiter = iter(testloader)
images,labels = dataiter.next()
images, labels = images.to(device), labels.to(device)

outputs = net(images)

_,predicted = torch.max(outputs,1)
print('Predicted:' , ' '.join('%5s' %classes[predicted[j]] for j in range(4)))
#
# #test without distiguishing between classes
#
correct = 0
total = 0

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

with torch.no_grad():
    for data in testloader:
        images,labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _,predicted = torch.max(outputs.data,1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i]/
                                        class_total[i]))

print('accuracy of the network on the 10000 test images: %d %%' %
      (100 * correct/total))


