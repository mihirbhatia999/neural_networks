import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
import pandas as pd 
import torchvision.transforms as transforms 
import torch 
import torchvision
import numpy as np
import cv2
import time

class img2num:
    
    def train(self):
      
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(1, 6, 5)
                self.pool1 = nn.MaxPool2d(2, 2)
                self.conv2 = nn.Conv2d(6, 16, 5)
                self.pool2 = nn.MaxPool2d(2, 2)
                self.bn1 = nn.BatchNorm2d(16)
                self.conv3 = nn.Conv2d(16,120,4)
                self.bn2 = nn.BatchNorm2d(120)
                self.h4 = nn.Linear(120,84)
                self.bn3 = nn.BatchNorm1d(84)
                self.h5 = nn.Linear(84,10)
            
            def forward(self,x):
                #forward
                y1 = self.pool1(F.rrelu(self.conv1(x)))
                #print(y1.shape)
                y2 = self.pool2(self.bn1(F.rrelu(self.conv2(y1))))
                #print(y2.shape)
                y3 = self.bn2(F.rrelu(self.conv3(y2)))
                #print(y3.shape)
                y4 = self.bn3(F.rrelu(self.h4(y3.squeeze(3).squeeze(2))))
                #print(y4.shape)
                output_prob = F.softmax(self.h5(y4))
                return output_prob
        

        
      
        self.model = Net()
        
        #load cifar dataset
        transform = transforms.Compose(
            [transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))
             ])

        trainset = torchvision.datasets.MNIST(root='./data_mnist', train=True,
                                                download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=500,
                                                  shuffle=True, num_workers=1)

        testset = torchvision.datasets.MNIST(root='./data_mnist', train=False,
                                               download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=3000,
                                                 shuffle=False, num_workers=1)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr = 0.01)
        
        #initializing list to store all the parameters 
        self.running_loss_test_list = []
        self.running_loss_list = []
        self.total_list = [] 
        self.total_test_list = []
        self.correct_list = []
        self.correct_test_list = []
        self.training_accuracy_list = []
        self.testing_accuracy_list = [] 
        print('total no of epochs')
        epochs = 20
        print(epochs)
        for epoch in range(epochs):  # loop over the dataset multiple times
            running_loss_test = 0.0
            running_loss = 0.0
            total = 0.0 
            total_test = 0 
            correct = 0
            correct_test = 0 
            for i, data in enumerate(trainloader):
                # get the inputs
                inputs, labels = data
              
                optimizer.zero_grad()

                output = self.model.forward(inputs) 
             
                _ , predicted = torch.max(output.data,1)

                total = total + labels.size(0)
                correct = correct + (predicted == labels).sum().item()
                #print(output)
                #print(labels)
                #print(output.shape)
                #print(labels.shape)
                loss = criterion(output, labels)
                
                #optimizer 
                loss.backward()
                optimizer.step()

                #print statistics
                running_loss += loss.item()
            print('EPOCH NO.', epoch + 1)
            print('training loss  = ',running_loss)
            #print('total train = ',total)
            #print('correct train = ', correct)
            print('train accuracy = ', (correct/total)*100, ' %')
            
            for i, data in enumerate(testloader):
                inputs,labels = data
                output = self.model.forward(inputs)
                _ , predicted = torch.max(output.data,1)
                total_test = total_test + labels.size(0)
                correct_test = correct_test + (predicted == labels).sum().item()
                loss = criterion(output, labels)
                running_loss_test += loss.item()
            print('test loss = ', running_loss_test)
            #print('total test = ',total_test)
            #print('correct test = ', correct_test)
            print('test accuracy = ', (correct_test/total_test)*100, ' %')
            print('--------------------------------------')
            print(' ')
            
            #adding everything to lists 
            self.running_loss_test_list.append(running_loss_test)
            self.running_loss_list.append(running_loss)
            self.total_list.append(total)
            self.total_test_list.append(total_test)
            self.correct_list.append(correct)
            self.correct_test_list.append(correct_test)
            self.training_accuracy_list.append((correct/total)*100)
            self.testing_accuracy_list.append((correct_test/total_test)*100)
        
       
        print('Finished Training')
        
            
    def forward(self,img):
        m = self.model.eval()
        return m.forward(img)


obj = img2num()
obj.train()
        




